import copy
import tempfile
import unittest
from pathlib import Path

from alfworld.continual.config import Config
from alfworld.continual.negative_ul import build_sample, candidate_loss, is_negative


class CharTokenizer:
    eos_token_id = 0
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]
    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids)


def record():
    raw = 'Reasoning: test\nAction: heat apple 1 with microwave 1'
    return dict(raw_output=raw, output_token_ids=[ord(c) for c in raw] + [0],
                input_token_ids=[1, 2], executed_action='heat apple 1 with microwave 1',
                format_error=None, feedback='Nothing happens.',
                admissible_commands=['look', 'look', 'inventory'])


class NegativeULTests(unittest.TestCase):
    def test_filter_and_sample(self):
        step = record()
        for feedback in ('Nothing happens.', ' NOTHING HAPPENS ', 'nothing happens'):
            self.assertTrue(is_negative({**step, 'feedback': feedback}))
        for feedback in ('You see nothing happens here.', 'unchanged', 'Success'):
            self.assertFalse(is_negative({**step, 'feedback': feedback}))
        self.assertFalse(is_negative({**step, 'executed_action': None}))
        self.assertFalse(is_negative({**step, 'format_error': 'bad'}))
        sample, reason = build_sample(step, CharTokenizer(), 1000)
        self.assertIsNone(reason)
        self.assertEqual(sample['candidates'], ['look', 'inventory', step['executed_action']])
        self.assertTrue(sample['negative_added'])
        self.assertEqual(sample['negative_index'], 2)
        self.assertEqual(CharTokenizer().decode(sample['prefix_token_ids'][2:]), 'Reasoning: test\nAction:')
        self.assertEqual(CharTokenizer().decode(sample['candidate_token_ids'][0]), ' look\x00')
        self.assertEqual(build_sample({**step, 'admissible_commands': []}, CharTokenizer(), 1000)[1],
                         'no_alternative_candidates')
        with self.assertRaisesRegex(ValueError, 'context capacity'):
            build_sample(step, CharTokenizer(), 2)
        self.assertEqual(build_sample({**step, 'output_token_ids': [1, 0]}, CharTokenizer(), 1000)[1],
                         'action_token_boundary')

    def test_math_and_two_pass_gradient(self):
        import torch
        torch.manual_seed(4)
        weights = torch.randn(3, 5, requires_grad=True)
        # Complete sequences of different lengths, not average token scores.
        scores = torch.stack([weights[0].log_softmax(0)[:2].sum(),
                              weights[1].log_softmax(0)[:3].sum(),
                              weights[2].log_softmax(0)[:1].sum()])
        loss = candidate_loss(scores, 1)
        torch.testing.assert_close(loss, -torch.log1p(-scores.softmax(0)[1]))
        direct, = torch.autograd.grad(loss, weights, retain_graph=True)
        detached = scores.detach().requires_grad_()
        coefficients, = torch.autograd.grad(candidate_loss(detached, 1), detached)
        for index in range(3):
            (scores[index] * coefficients[index]).backward(retain_graph=True)
        torch.testing.assert_close(direct, weights.grad)
        torch.testing.assert_close(candidate_loss(scores[[2, 0, 1]], 2), loss)
        for values in ([10000., -10000.], [-10000., 10000.], [0., 0.]):
            z = torch.tensor(values, requires_grad=True)
            result = candidate_loss(z, 0)
            result.backward()
            self.assertTrue(torch.isfinite(result) and torch.isfinite(z.grad).all())

    def test_online_policy(self):
        from alfworld.continual.methods import Method
        from alfworld.continual.model import MockBackend
        from alfworld.continual.types import Episode
        cfg = Config(method='negative-ul')
        self.assertTrue(cfg.trains_parameters)
        cfg.validate()
        method = Method(cfg, MockBackend(cfg))
        step = record()
        update = method.observe_step(step)
        self.assertEqual(method.version, 1)
        self.assertEqual(update['version_before'], 0)
        step['negative_ul_update'] = update
        ep = Episode('test', 'heat', 0, steps=[step])
        self.assertEqual(method.observe_episode(ep)['negative_feedback_count'], 1)
        self.assertEqual(method.version, 1)
        self.assertIsNone(method.observe_step({**record(), 'feedback': 'ok'}))
        cfg.training.dropout = .1
        with self.assertRaisesRegex(ValueError, 'dropout'):
            cfg.validate()

    def test_runner_immediate_update_and_recovery(self):
        import json
        from alfworld.continual.runner import run
        from alfworld.continual.model import MockBackend
        from alfworld.continual.logging import load_checkpoint
        from alfworld.agents.environment.alfred_tw_env import TASK_TYPES

        seen = []
        class Backend(MockBackend):
            def generate(self, messages, **kwargs):
                seen.append(self.updates)
                return super().generate(messages, **kwargs)
        class Env:
            fail = False
            def __init__(self, *args):
                self.calls = 0
            def reset(self):
                return 'room', {'admissible_commands': ['look', 'inventory']}
            def step(self, action):
                self.calls += 1
                if self.calls == 2 and self.fail:
                    raise RuntimeError('injected crash')
                return 'Nothing happens.', 0., self.calls == 2, {
                    'admissible_commands': ['look', 'inventory'], 'won': False}
            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = Config(method='negative-ul', max_steps=2, checkpoint_interval=1)
            cfg.model.backend = 'mock'
            cfg.stream.data_dir = str(root / 'data')
            cfg.stream.limit = 2
            for task in TASK_TYPES.values():
                game = root / 'data/json_2.1.1/valid_seen' / task / '0'
                game.mkdir(parents=True)
                (game / 'game.tw-pddl').write_text(json.dumps({'solvable': True}))
                (game / 'traj_data.json').write_text(json.dumps({'task_type': task}))
            cfg.output_dir = str(root / 'continuous')
            run(cfg, backend_factory=Backend, env_factory=Env)
            continuous, completed = load_checkpoint(Path(cfg.output_dir))
            self.assertEqual(seen, [0, 1, 2, 3])
            self.assertEqual(continuous['method']['version'], 4)
            self.assertEqual([r['negative_feedback_count'] for r in completed], [2, 2])
            episode = json.loads((Path(cfg.output_dir) / completed[0]['attempt_dir'] / 'episode.json').read_text())
            self.assertEqual([s['model_version'] for s in episode['steps']], [0, 1])
            self.assertIn('input_token_ids', episode['steps'][0])
            cfg.output_dir = str(root / 'interrupted')
            Env.fail = True
            with self.assertRaisesRegex(RuntimeError, 'injected crash'):
                run(cfg, backend_factory=Backend, env_factory=Env)
            Env.fail = False
            seen.clear()
            run(cfg, resume=True, backend_factory=Backend, env_factory=Env)
            resumed, _ = load_checkpoint(Path(cfg.output_dir))
            self.assertEqual(seen, [0, 1, 2, 3])
            self.assertEqual(resumed['method'], continuous['method'])
            self.assertEqual(resumed['backend'], continuous['backend'])

    def test_tiny_model_update_restore(self):
        import torch
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        from alfworld.continual.model import HFBackend
        torch.set_num_threads(2)
        torch.manual_seed(42)
        architecture = Qwen3_5TextConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=['linear_attention', 'full_attention'],
            max_position_embeddings=4096, eos_token_id=0, pad_token_id=0,
            rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                             'partial_rotary_factor': 1., 'mrope_section': [1, 1, 2]})
        cfg = Config(method='negative-ul')
        cfg.model.device = 'cpu'
        cfg.training.learning_rate = 1e-3
        backend = HFBackend(cfg, Qwen3_5ForCausalLM(architecture), CharTokenizer())
        sample = {'prefix_token_ids': [1, 2, 3], 'candidate_token_ids': [[4, 0], [5, 0]],
                  'negative_index': 0}
        trainer = backend.trainer
        before = copy.deepcopy(backend.state_dict())
        result = trainer.train_sample(sample)
        self.assertEqual(result['optimizer_steps'], 1)
        with torch.no_grad():
            after_scores = torch.stack([trainer.score(sample['prefix_token_ids'], t)
                                        for t in sample['candidate_token_ids']])
        self.assertLess(after_scores.softmax(0)[0].item(), result['negative_probability'])
        first = copy.deepcopy(backend.state_dict())
        backend.load_state_dict(before)
        trainer.logits_chunk_tokens = 1
        repeated = trainer.train_sample(sample)
        self.assertAlmostEqual(repeated['loss'], result['loss'], places=5)
        for key, value in first['adapter'].items():
            torch.testing.assert_close(backend.state_dict()['adapter'][key], value, atol=1e-6, rtol=1e-4)
        backend.load_state_dict(first)
        trainer.train_sample(sample)
        second = copy.deepcopy(backend.state_dict())
        backend.load_state_dict(first)
        trainer.train_sample(sample)
        for key, value in second['adapter'].items():
            torch.testing.assert_close(backend.state_dict()['adapter'][key], value, atol=0, rtol=0)
        self.assertFalse(backend.model.training)
        self.assertTrue(backend.model.config.use_cache)


if __name__ == '__main__':
    unittest.main()
