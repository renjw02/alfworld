"""Interleaved labels, full causal context, and two independent LoRA updates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from alfworld.continual.config import Config, load_config
from alfworld.continual.logging import load_checkpoint
from alfworld.continual.methods import Method
from alfworld.continual.model import HFBackend, MockBackend
from alfworld.continual.prompting import initial_message, observation_message, terminal_message
from alfworld.continual.runner import run
from alfworld.continual.trajectory_sft import build_trajectory, TrajectorySFTTrainer
from alfworld.continual.types import Episode


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0
    all_special_ids = [0, 2]

    def __call__(self, text, **kwargs):
        ids, offsets, i = [], [], 0
        while i < len(text):
            end = i + (5 if text.startswith('<eos>', i) else 1)
            ids.append(2 if end - i == 5 else ord(text[i]) + 3)
            offsets.append((i, end))
            i = end
        return {'input_ids': ids, 'offset_mapping': offsets}

    def encode(self, text, **kwargs):
        return self(text)['input_ids']

    def apply_chat_template(self, messages, **kwargs):
        return ''.join(m['role'] + ':\n' + m['content'] + '<eos>\n' for m in messages)


def config():
    c = Config(method='sft')
    c.model.context_overflow = 'error'
    c.model.device = 'cpu'
    c.sft.max_seq_tokens = 8192
    return c


def episode():
    e = Episode('test', 'pick_heat_then_place_in_recep', 0)
    observation = 'Your task is to: heat a potato.'
    e.messages = [initial_message(e.episode_id, observation, ['look'], 'admissible')]
    for i in range(3):
        response = f'Reasoning: inspect view {i}\nAction: look'
        feedback = 'Nothing happens.' if i == 1 else f'New view {i}'
        e.steps.append(dict(step=i+1, observation=observation, admissible_commands=['look'],
                            raw_output=response, action='look', executed_action='look', format_error=None,
                            generation_truncated=False, next_observation=feedback, feedback=feedback,
                            next_admissible_commands=['look'], success=False))
        e.messages.extend([{'role': 'assistant', 'content': response},
                           observation_message(feedback, ['look'], 'admissible')])
        observation = feedback
    e.outcome = dict(success=False, end_reason='step_limit', score=0, attempts=3, environment_calls=3)
    e.messages.append(terminal_message(e.outcome))
    return e


def backend(c):
    b = HFBackend.__new__(HFBackend)
    b.config, b.tokenizer = c, Tokenizer()
    return b


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.config = config()
        self.backend = backend(self.config)

    def build(self, e):
        return build_trajectory(self.config, e, self.backend._render, self.backend.tokenizer, 8192)

    def test_interleaved_masks_failed_episode_and_eos(self):
        e = episode()
        sample = self.build(e)
        self.assertEqual(sample['retained_steps'], [1, 3])
        self.assertEqual(sample['rejected'], [{'step': 2, 'reason': 'no_effect'}])
        expected = sum((self.backend.tokenizer.encode(e.steps[i]['raw_output']) + [2] for i in (0, 2)), [])
        self.assertEqual([v for v in sample['labels'] if v != -100], expected)
        self.assertEqual(sample['target_tokens'], len(expected))
        self.assertIn(e.steps[1]['raw_output'], sample['text'])
        self.assertIn('END EPISODE', sample['text'])
        for span in sample['target_spans']:
            self.assertEqual(sample['labels'][span['end']-1], 2)
            self.assertEqual(sample['labels'][span['start']-1], -100)
            self.assertEqual(sample['labels'][span['end']], -100)
        # Adding unrelated future feedback cannot change a previous target/prefix.
        changed = copy.deepcopy(e)
        changed.messages[-1]['content'] += '\nFuture outcome extension'
        other = self.build(changed)
        end = sample['target_spans'][0]['end']
        self.assertEqual(sample['input_ids'][:end], other['input_ids'][:end])
        self.assertEqual(sample['labels'][:end], other['labels'][:end])

    def test_strict_filters_and_terminal_success_exception(self):
        for change, reason in [({'generation_truncated': True}, 'format_or_truncated'),
                               ({'executed_action': None}, 'unexecuted_or_inadmissible'),
                               ({'admissible_commands': []}, 'unexecuted_or_inadmissible'),
                               ({'next_observation': 'Your task is to: heat a potato.'}, 'unchanged_observation')]:
            e = episode(); e.steps[0].update(change)
            self.assertEqual(self.build(e)['rejected'][0]['reason'], reason)
        e = episode(); e.steps[1]['success'] = True
        self.assertEqual(self.build(e)['retained_steps'], [1, 2, 3])

    def test_special_tokens_alignment_and_length_fail_closed(self):
        e = episode()
        e.steps[0]['raw_output'] = 'Reasoning: <eos> inspect\nAction: look'
        e.messages[1]['content'] = e.steps[0]['raw_output']
        self.assertEqual(self.build(e)['rejected'][0]['reason'], 'response_control_tokens')
        e = episode(); e.messages[1]['content'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'align'): self.build(e)
        e = episode(); n = self.build(e)['tokens']
        self.config.sft.max_seq_tokens = n - 1
        with self.assertRaisesRegex(ValueError, 'no truncation'): self.build(e)
        self.config.sft.max_seq_tokens = n
        self.assertEqual(self.build(e)['tokens'], n)

    def test_config_and_full_history_without_cross_episode_icl(self):
        c = load_config('configs/continual-sft.yaml').validate()
        self.assertTrue(c.trains_parameters)
        self.assertEqual(c.sft.optimizer_steps, 2)
        m = Method(c, MockBackend(c)); e = episode()
        self.assertEqual(m.build_context(e)[1:], e.messages)
        result = m.observe_episode(e)
        self.assertFalse(result['skipped'])
        self.assertEqual(m.history, [])
        self.assertEqual(m.backend.updates, 2)
        for step in e.steps: step['executed_action'] = None
        self.assertTrue(m.observe_episode(e)['skipped'])
        self.assertEqual(m.version, 1)
        c.model.context_overflow = 'truncate_oldest'
        with self.assertRaisesRegex(ValueError, 'context_overflow=error'): c.validate()
        for field in ('optimizer_steps', 'logits_chunk_tokens'):
            for value in (0, True, 1.5):
                c = config(); setattr(c.sft, field, value)
                with self.assertRaises(ValueError): c.validate()

    def test_runner_failed_episode_checkpoint_resume_and_old_config(self):
        class Env:
            def __init__(self, *args): pass
            def reset(self): return 'Your task is to: heat a potato.', {'admissible_commands': ['look']}
            def step(self, action): return 'New view.', 0, True, {'admissible_commands': ['look'], 'won': False}
            def close(self): pass

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c = config(); c.model.backend = 'mock'; c.max_steps = 1; c.checkpoint_interval = 1
            c.output_dir = str(root/'run'); c.stream.data_dir = str(root/'data')
            p = root/'data/json_2.1.1/valid_seen/heat/trial'; p.mkdir(parents=True)
            (p/'game.tw-pddl').write_text('{"solvable": true}')
            (p/'traj_data.json').write_text('{"task_type": "pick_heat_then_place_in_recep"}')
            self.assertEqual(run(c, env_factory=Env)['episodes'], 1)
            saved, rows = load_checkpoint(root/'run')
            self.assertEqual(saved['backend']['updates'], 2)
            self.assertFalse(rows[0]['success'])
            trace = json.loads((root/'run'/rows[0]['attempt_dir']/'episode.json').read_text())
            self.assertTrue(trace['steps'][0]['input_token_ids'])
            run(c, resume=True, env_factory=Env)
            self.assertEqual(load_checkpoint(root/'run')[0]['backend'], saved['backend'])
            class InterruptedBackend(MockBackend):
                def sft_episode(self, e):
                    super().sft_episode(e)
                    raise RuntimeError('after update, before commit')
            c.output_dir = str(root/'interrupted')
            with self.assertRaisesRegex(RuntimeError, 'before commit'):
                run(c, backend_factory=InterruptedBackend, env_factory=Env)
            uncommitted, rows = load_checkpoint(root/'interrupted')
            self.assertEqual(uncommitted['next_position'], 0)
            self.assertEqual(rows, [])
            run(c, resume=True, env_factory=Env)
            self.assertEqual(load_checkpoint(root/'interrupted')[0]['backend'], saved['backend'])
            # New storage fields must not break existing method checkpoints.
            c.method = 'success-only-sft'; c.output_dir = str(root/'legacy')
            run(c, env_factory=Env)
            path = root/'legacy/config.json'; values = json.loads(path.read_text())
            for key in ('optimizer_steps', 'logits_chunk_tokens'): values['sft'].pop(key)
            path.write_text(json.dumps(values))
            self.assertEqual(run(c, resume=True, env_factory=Env)['episodes'], 1)


class TrajectoryModelTests(unittest.TestCase):
    def test_two_real_updates_match_full_logits_reference_and_resume(self):
        import torch
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        torch.set_num_threads(2); torch.manual_seed(42)
        c = config(); c.training.learning_rate = 1e-3; c.sft.logits_chunk_tokens = 2
        arch = Qwen3_5TextConfig(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, linear_key_head_dim=8,
            linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=['linear_attention', 'full_attention'], max_position_embeddings=8192,
            eos_token_id=2, pad_token_id=0,
            rope_parameters={'rope_type':'default', 'rope_theta':10000., 'partial_rotary_factor':1., 'mrope_section':[1,1,2]})
        b = HFBackend(c, model=Qwen3_5ForCausalLM(arch), tokenizer=Tokenizer())
        reference = TrajectorySFTTrainer(copy.deepcopy(b.model), c.training, c.sft)
        sample = {'input_ids': [3,4,5,6,2,7,8,9,10,2,11],
                  'labels': [-100,-100,5,6,2,-100,-100,-100,10,2,-100]}
        frozen = {n:p.detach().clone() for n,p in b.model.named_parameters() if not p.requires_grad}
        before = copy.deepcopy(b.state_dict())
        expected_losses = []
        # Independent full-vocabulary loss, causal shift and two fresh gradients.
        for _ in range(2):
            reference.model.train(); reference.optimizer.zero_grad(set_to_none=True)
            logits = reference.model(input_ids=torch.tensor([sample['input_ids']]), use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(logits[0, :-1].float(), torch.tensor(sample['labels'][1:]))
            expected_losses.append(loss.item()); loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters, c.training.max_grad_norm)
            reference.optimizer.step()
        with patch('alfworld.continual.training.token_chunks', side_effect=AssertionError('no context chunks')):
            result = b.trainer.train_trajectory(sample)
        self.assertEqual(result['optimizer_steps'], 2)
        self.assertEqual(result['target_tokens'], 5)
        for record, expected in zip(result['records'], expected_losses):
            self.assertAlmostEqual(record['loss'], expected, places=5)
        for p, q in zip(b.trainer.parameters, reference.parameters):
            torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-4)
        self.assertTrue(any(not torch.equal(before['adapter'][k], v) for k,v in b.state_dict()['adapter'].items()))
        for name,p in b.model.named_parameters():
            if name in frozen: torch.testing.assert_close(p, frozen[name], atol=0, rtol=0)
        saved = copy.deepcopy(b.state_dict())
        b.trainer.train_trajectory(sample); expected = copy.deepcopy(b.state_dict()['adapter'])
        b.load_state_dict(saved); b.trainer.train_trajectory(sample)
        for k,v in expected.items(): torch.testing.assert_close(b.state_dict()['adapter'][k], v, atol=0, rtol=0)
        # Masked future text cannot affect earlier predictions.
        b.model.eval()
        changed = list(sample['input_ids']); changed[-1] = 12
        with torch.no_grad():
            a = b.model(input_ids=torch.tensor([sample['input_ids']]), use_cache=False).logits
            z = b.model(input_ids=torch.tensor([changed]), use_cache=False).logits
        torch.testing.assert_close(a[:, :-1], z[:, :-1], atol=0, rtol=0)
        empty = {**sample, 'labels': [-100]*len(sample['labels'])}
        prior = b.trainer.steps
        self.assertTrue(b.trainer.train_trajectory(empty)['skipped'])
        self.assertEqual(b.trainer.steps, prior)
        self.assertTrue(all(p.grad is None for p in b.trainer.parameters))
        # Full failed-episode path also updates the real adapter twice.
        result = Method(c, b).observe_episode(episode())
        self.assertEqual(result['optimizer_steps'], 2)
        self.assertEqual(result['trajectory']['retained_steps'], [1,3])
        self.assertFalse(b.model.training)
        self.assertTrue(b.model.config.use_cache)


if __name__ == '__main__':
    unittest.main()
