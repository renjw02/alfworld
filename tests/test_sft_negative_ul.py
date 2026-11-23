import copy
import unittest
from unittest.mock import patch

from alfworld.continual.config import load_config
from alfworld.continual.methods import Method
from alfworld.continual.model import HFBackend, MockBackend
from alfworld.continual.negative_ul import candidate_loss
from alfworld.continual.sft_negative_ul import build_joint_sample, positive_error, unique_negatives
from tests.test_trajectory_sft import config, episode, backend, Tokenizer


class JointTests(unittest.TestCase):
    def test_filters_masks_and_dedup(self):
        e = episode()
        bad = e.steps[1]
        other = {**bad, 'executed_action': 'inventory'}
        steps = [bad, bad, other, bad, {'executed_action': None, 'format_error': 'bad'},
                 bad, e.steps[2], bad]
        retained, stats = unique_negatives(steps)
        self.assertEqual(retained, [bad, other, bad])
        self.assertEqual(stats['duplicate_negative_count'], 3)
        self.assertIsNone(positive_error({**e.steps[0], 'admissible_commands': [],
                                         'next_observation': e.steps[0]['observation']}))
        c = config(); c.method = 'sft-negative-ul'
        b = backend(c)
        b.tokenizer.decode = lambda ids, **kwargs: ''.join('<eos>' if i == 2 else chr(i - 3) for i in ids)
        for step in e.steps:
            step['input_token_ids'] = [100, 101]
            step['output_token_ids'] = b.tokenizer.encode(step['raw_output']) + [2]
        bad['admissible_commands'] = ['look', 'inventory']
        sample = build_joint_sample(c, e, b._render, b.tokenizer, 8192)
        self.assertEqual(sample['negative_samples'][0]['step'], 2)
        self.assertEqual(sample['trajectory']['retained_steps'], [1, 3])
        self.assertEqual(sample['trainable_negative_count'], 1)
        self.assertIn(bad['raw_output'], sample['trajectory']['text'])
        m = Method(c, MockBackend(c))
        self.assertIsNone(m.observe_step(bad))
        self.assertEqual(m.version, 0)
        self.assertFalse(m.observe_episode(e)['skipped'])
        self.assertEqual(m.version, 1)

    def test_runner_checkpoint_and_resume(self):
        from tests.test_trajectory_sft import TrajectoryTests
        def joint_config():
            c = config(); c.method = "sft-negative-ul"
            return c
        with patch("tests.test_trajectory_sft.config", side_effect=joint_config):
            TrajectoryTests().test_runner_failed_episode_checkpoint_resume_and_old_config()

    def test_config(self):
        c = load_config('configs/continual-heat-place-200-sft-negative-ul.yaml').validate()
        self.assertTrue(c.trains_parameters)
        self.assertEqual(c.stream.split, 'heat_place_200')
        for value in (-1, float('nan'), float('inf'), True):
            c.training.lambda_neg = value
            with self.assertRaises(ValueError): c.validate()

    def test_joint_reference_resume_and_empty_sides(self):
        import torch
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        import inspect
        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
        # Installed CUDA kernels cannot run this CPU numerical reference test.
        for name in ('causal_conv1d_fn', 'causal_conv1d_update',
                     'torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
            self.enterContext(patch.object(modeling, name, inspect.unwrap(getattr(modeling, name))))
        torch.set_num_threads(2); torch.manual_seed(42)
        c = config(); c.method = 'sft-negative-ul'; c.training.lambda_neg = .7
        c.training.learning_rate = 1e-3; c.sft.logits_chunk_tokens = 2
        arch = Qwen3_5TextConfig(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, linear_key_head_dim=8,
            linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=['linear_attention', 'full_attention'], max_position_embeddings=8192,
            eos_token_id=2, pad_token_id=0,
            rope_parameters={'rope_type':'default', 'rope_theta':10000., 'partial_rotary_factor':1., 'mrope_section':[1,1,2]})
        b = HFBackend(c, model=Qwen3_5ForCausalLM(arch), tokenizer=Tokenizer())
        reference = copy.deepcopy(b.trainer)
        sample = {'trajectory': {'input_ids': [3,4,5,6,2,7], 'labels': [-100,-100,5,6,2,-100]},
                  'negative_samples': [
                      {'prefix_token_ids': [3,4], 'candidate_token_ids': [[5,2],[6,7,2]], 'negative_index': 0},
                      {'prefix_token_ids': [3,8], 'candidate_token_ids': [[5,2],[9,2]], 'negative_index': 1}]}
        expected = []
        for _ in range(c.sft.optimizer_steps):
            reference.model.train(); reference.optimizer.zero_grad(set_to_none=True)
            logits = reference.model(input_ids=torch.tensor([sample['trajectory']['input_ids']]), use_cache=False).logits
            pos = torch.nn.functional.cross_entropy(logits[0,:-1].float(), torch.tensor(sample['trajectory']['labels'][1:]))
            neg = 0
            for s in sample['negative_samples']:
                scores = []
                for target in s['candidate_token_ids']:
                    prefix = s['prefix_token_ids']
                    logits = reference.model(input_ids=torch.tensor([prefix + target]), use_cache=False).logits
                    scores.append(-torch.nn.functional.cross_entropy(logits[0,len(prefix)-1:-1].float(),
                                                                     torch.tensor(target), reduction='sum'))
                neg = neg + candidate_loss(torch.stack(scores), s['negative_index']) / 2
            loss = pos + c.training.lambda_neg * neg
            expected.append(loss.item()); loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters, c.training.max_grad_norm)
            reference.optimizer.step()
        actual = b.trainer.train_joint(sample)
        for row, value in zip(actual['records'], expected): self.assertAlmostEqual(row['loss'], value, places=5)
        for p, q in zip(b.trainer.parameters, reference.parameters):
            torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-4)
        saved = copy.deepcopy(b.state_dict())
        b.trainer.train_joint(sample); expected_state = copy.deepcopy(b.state_dict())
        b.load_state_dict(saved); b.trainer.train_joint(sample)
        for k, v in expected_state['adapter'].items():
            torch.testing.assert_close(b.state_dict()['adapter'][k], v, atol=0, rtol=0)
        for no_pos, no_neg in ((True, False), (False, True), (True, True)):
            s = copy.deepcopy(sample)
            if no_pos: s['trajectory']['labels'] = [-100] * 6
            if no_neg: s['negative_samples'] = []
            result = b.trainer.train_joint(s)
            self.assertEqual(result['skipped'], no_pos and no_neg)
        c.training.lambda_neg = 0
        s['negative_samples'] = sample['negative_samples']
        self.assertTrue(b.trainer.train_joint(s)['skipped'])
        self.assertFalse(b.model.training)
        self.assertTrue(all(p.grad is None for p in b.trainer.parameters))
