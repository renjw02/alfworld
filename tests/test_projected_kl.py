import copy
import unittest
import tempfile
from pathlib import Path
from alfworld.continual.config import Config, load_config
from alfworld.continual.projected_kl import projected_target, candidate_loss
from alfworld.continual.negative_ul import candidate_loss as ul_loss
from tests.test_negative_ul import CharTokenizer


class ProjectedKLTests(unittest.TestCase):
    def test_projection_and_gradients(self):
        import torch
        z = torch.tensor([.6, .2, .15, .05], dtype=torch.float64).log().requires_grad_()
        q = projected_target(z, 0)
        torch.testing.assert_close(q, torch.tensor([0., .5, .375, .125], dtype=z.dtype))
        self.assertFalse(q.requires_grad)
        pkl_grad, = torch.autograd.grad(candidate_loss(z, q), z)
        ul_grad, = torch.autograd.grad(ul_loss(z, 0), z)
        torch.testing.assert_close(pkl_grad, ul_grad)
        changed = (z.detach() + torch.tensor([-.2, .3, -.1, .1])).requires_grad_()
        fixed_grad, = torch.autograd.grad(candidate_loss(changed, q), changed)
        moving_grad, = torch.autograd.grad(candidate_loss(changed, projected_target(changed, 0)), changed)
        self.assertFalse(torch.allclose(fixed_grad, moving_grad))
        perm = torch.tensor([2, 0, 3, 1])
        torch.testing.assert_close(candidate_loss(z[perm], q[perm]), candidate_loss(z, q))
        for values in ([10000., -10000., -9999.], [-10000., 10000., 0.]):
            scores = torch.tensor(values, requires_grad=True)
            loss = candidate_loss(scores, projected_target(scores, 0))
            loss.backward()
            self.assertTrue(torch.isfinite(loss) and torch.isfinite(scores.grad).all())
        with self.assertRaises(ValueError):
            projected_target(torch.tensor([1.]), 0)
        with self.assertRaises(RuntimeError):
            projected_target(torch.tensor([float('nan'), 0.]), 0)

    def test_two_pass_gradient(self):
        import torch
        torch.manual_seed(9)
        weights = torch.randn(3, 5, requires_grad=True)
        scores = torch.stack([weights[i].log_softmax(0)[:i+1].sum() for i in range(3)])
        q = projected_target(scores, 1)
        direct, = torch.autograd.grad(candidate_loss(scores, q), weights, retain_graph=True)
        detached = scores.detach().requires_grad_()
        coeff, = torch.autograd.grad(candidate_loss(detached, q), detached)
        for score, c in zip(scores, coeff):
            (score * c).backward(retain_graph=True)
        torch.testing.assert_close(direct, weights.grad)

    def test_config_and_skip(self):
        from alfworld.continual.methods import Method
        from alfworld.continual.model import MockBackend
        from tests.test_negative_ul import record
        cfg = load_config('configs/continual-projected-kl.yaml').validate()
        self.assertTrue(cfg.trains_parameters)
        self.assertEqual(cfg.projected_kl.optimizer_steps, 2)
        method = Method(cfg, MockBackend(cfg))
        self.assertIsNone(method.observe_step({**record(), 'feedback': 'ok',
            'admissible_commands': [record()['executed_action']],
            'observation': 'before', 'next_observation': 'after'}))
        skipped = method.observe_step({**record(), 'admissible_commands': []})
        self.assertTrue(skipped['skipped'])
        self.assertEqual(method.version, 0)
        update = method.observe_step(record())
        self.assertEqual(update['simulated_optimizer_steps'], 2)
        self.assertEqual(update['optimizer_steps'], 0)
        self.assertEqual(method.version, 1)
        for value in (0, -1, True, 1.5):
            cfg.projected_kl.optimizer_steps = value
            with self.assertRaises(ValueError):
                cfg.validate()
        cfg.projected_kl.optimizer_steps = 1
        cfg.validate()
        cfg.training.dropout = .1
        with self.assertRaises(ValueError):
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
            cfg = Config(method='projected-kl', max_steps=2, checkpoint_interval=1)
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
        cfg = Config(method='projected-kl')
        cfg.model.device = 'cpu'
        cfg.training.learning_rate = 1e-3
        backend = HFBackend(cfg, Qwen3_5ForCausalLM(architecture), CharTokenizer())
        sample = {'prefix_token_ids': [1, 2, 3], 'candidate_token_ids': [[4, 0], [5, 0], [6, 7, 0]],
                  'negative_index': 0}
        self.assertTrue(all('.mlp.' in n and n.rsplit('.', 1)[-1] in
                            {'gate_proj', 'up_proj', 'down_proj'} for n in backend.target_modules))
        trainer = backend.trainer
        before = copy.deepcopy(backend.state_dict())
        result = trainer.train_sample(sample)
        self.assertEqual(result['optimizer_steps'], 2)
        with torch.no_grad():
            after_scores = torch.stack([trainer.score(sample['prefix_token_ids'], t)
                                        for t in sample['candidate_token_ids']])
        self.assertLess(after_scores.softmax(0)[0].item(), result['substeps'][0]['negative_probability'])
        self.assertEqual(trainer.steps, 2)
        q = torch.tensor(result['target_probabilities'])
        for substep in result['substeps']:
            actual = candidate_loss(torch.tensor(substep['candidate_log_probs']), q).item()
            self.assertAlmostEqual(actual, substep['loss'], places=5)
        self.assertEqual(q[0].item(), 0.)
        torch.testing.assert_close(q, projected_target(torch.tensor(result['reference_log_probs']), 0))
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
