"""ASCENT protocol and tiny real-model tests; no downloaded weights required."""
import copy
import unittest
from alfworld.continual.ascent import success_record, forward_kl_sum, reverse_kl_sum, current_episode_training_steps
from alfworld.continual.config import Config, load_config
from alfworld.continual.methods import Method
from alfworld.continual.model import HFBackend, MockBackend
from alfworld.continual.types import Episode


def episode():
    e = Episode("test", "pick", 0, outcome={"success": True})
    for i, action in enumerate(("look", "invented command"), 1):
        e.steps.append(dict(step=i, observation="Room description.\nYour task is to: find a mug.",
                            format_error=None, executed_action=action, action=action,
                            admissible_commands=["look"], raw_output=f"Reasoning: inspect\nAction: {action}",
                            input_token_ids=[3, 4, 5], training_input_token_ids=[3, 4, 5],
                            output_token_ids=[6, 7, 2] if i == 1 else [8, 2]))
    return e


class AscentProtocolTests(unittest.TestCase):
    def test_kl_direction_config_and_cli(self):
        from unittest.mock import patch
        from alfworld.continual.__main__ import main
        config = Config(method="ascent")
        self.assertEqual(config.ascent.kl_direction, "forward")
        config.ascent.kl_direction = "reverse"
        config.validate()
        config.ascent.kl_direction = "invalid"
        with self.assertRaisesRegex(ValueError, "kl_direction"):
            config.validate()
        with patch("sys.argv", ["continual", "--config", "configs/continual-heat-place.yaml",
                                "--method", "ascent", "--kl-direction", "reverse"]), \
                patch("alfworld.continual.__main__.run", return_value={}) as run:
            main()
            self.assertEqual(run.call_args.args[0].ascent.kl_direction, "reverse")

    def test_reverse_kl_direction_and_detachment(self):
        import torch
        p = torch.tensor([[1., -1., 0.]], requires_grad=True)
        q = torch.tensor([[-2., 2., 0.]], requires_grad=True)
        loss = reverse_kl_sum(p, q)
        expected = torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=p),
            torch.distributions.Categorical(logits=q.detach())).sum()
        torch.testing.assert_close(loss, expected)
        self.assertNotAlmostEqual(loss.item(), forward_kl_sum(p, q).item())
        loss.backward()
        prob = p.detach().softmax(-1)
        delta = p.detach().log_softmax(-1) - q.detach().log_softmax(-1)
        torch.testing.assert_close(p.grad, prob * (delta - (prob * delta).sum(-1, keepdim=True)))
        self.assertIsNone(q.grad)

    def test_local_prefix_preserves_responses_and_rejects_legacy_steps(self):
        e = episode()
        e.steps[0]["input_token_ids"] = [20, 21, 22, 3, 4, 5]
        original = copy.deepcopy(e.steps)
        selected = current_episode_training_steps(e)
        self.assertEqual(selected[0]["input_token_ids"], [3, 4, 5])
        self.assertEqual(selected[0]["output_token_ids"], original[0]["output_token_ids"])
        self.assertEqual(e.steps, original)
        del e.steps[0]["training_input_token_ids"]
        with self.assertRaisesRegex(ValueError, "restart legacy runs"):
            current_episode_training_steps(e)

    def test_icl_history_success_gate_window_and_scope(self):
        config = load_config("configs/continual-ascent-icl.yaml").validate()
        self.assertTrue(config.is_ascent and config.uses_icl and config.trains_parameters)
        config.window_episodes = 3
        backend = MockBackend(config)
        method = Method(config, backend)
        method.begin_task("pick")
        first = episode()
        first.messages = [{"role": "user", "content": "failed first episode"}]
        first.outcome["success"] = False
        self.assertTrue(method.observe_episode(first)["skipped"])
        self.assertEqual(backend.updates, 0)
        self.assertEqual(method.history, [first.messages])
        second = episode()
        second.messages = [{"role": "user", "content": "successful second episode"}]
        self.assertEqual(method.observe_episode(second)["version_after"], 1)
        self.assertFalse(method.begin_task("clean"))
        current = Episode("current", "clean", 2, [{"role": "user", "content": "current episode"}])
        expected = first.messages + second.messages + current.messages
        self.assertEqual(method.build_context(current)[1:], expected)
        saved = copy.deepcopy(method.state_dict())
        restored = Method(config, backend)
        restored.load_state_dict(saved)
        self.assertEqual(restored.build_context(current), method.build_context(current))
        third = episode()
        third.messages = [{"role": "user", "content": "third episode"}]
        third.outcome["success"] = False
        method.observe_episode(third)
        self.assertEqual(method.history, [second.messages, third.messages])
        third.messages[0]["content"] = "mutated"
        self.assertEqual(method.history[-1][0]["content"], "third episode")
        self.assertEqual(backend.updates, 1)
        config.window_episodes = 1
        method.observe_episode(third)
        self.assertEqual(method.build_context(current)[1:], current.messages)

        config.memory_scope = "task_type"
        config.window_episodes = 3
        backend = MockBackend(config)
        method = Method(config, backend)
        method.begin_task("pick")
        method.observe_episode(second)
        self.assertTrue(method.begin_task("clean"))
        self.assertEqual((method.history, method.version, backend.updates), ([], 0, 0))
        config.ascent.optimizer_steps = 0
        with self.assertRaisesRegex(ValueError, "ascent.optimizer_steps"):
            config.validate()

    def test_config_filter_gate_and_reset(self):
        config = load_config("configs/continual-ascent.yaml").validate()
        self.assertTrue(config.trains_parameters)
        self.assertFalse(config.is_ntp)
        self.assertEqual(config.ascent.learning_rate, 1.5e-4)
        text, kept, removed = success_record(episode())
        self.assertEqual((kept, removed), ([1], [2]))
        self.assertIn("find a mug.", text)
        self.assertNotIn("Room description", text)
        self.assertNotIn("invented command", text)
        e = episode()
        e.steps[1].update(format_error="bad format", executed_action=None)
        self.assertEqual(success_record(e)[2], [2])
        config.memory_scope = "task_type"
        backend = MockBackend(config)
        method = Method(config, backend)
        method.begin_task("pick")
        e.outcome["success"] = False
        before = copy.deepcopy(backend.state_dict())
        self.assertTrue(method.observe_episode(e)["skipped"])
        self.assertEqual(before, backend.state_dict())
        self.assertEqual(method.version, 0)
        e.outcome["success"] = True
        self.assertEqual(method.observe_episode(e)["version_after"], 1)
        self.assertTrue(method.begin_task("clean"))
        self.assertEqual(backend.state_dict(), before)
        self.assertEqual(method.version, 0)

    def test_forward_kl_direction_and_detachment(self):
        import torch
        p = torch.tensor([[1., -1., 0.]], requires_grad=True)
        q = torch.tensor([[-2., 2., 0.]], requires_grad=True)
        loss = forward_kl_sum(p, q)
        expected = torch.nn.functional.kl_div(p.log_softmax(-1), q.softmax(-1), reduction="sum")
        torch.testing.assert_close(loss, expected)
        loss.backward()
        torch.testing.assert_close(p.grad, p.detach().softmax(-1) - q.detach().softmax(-1))
        self.assertIsNone(q.grad)


class AscentModelTests(unittest.TestCase):
    def test_reverse_updates_teacher_freeze_chunking_and_resume(self):
        for method in ("ascent", "ascent-icl"):
            with self.subTest(method=method):
                self.check_model(method, "reverse")

    def test_exact_prefix_updates_teacher_freeze_and_resume(self):
        self.check_model("ascent")

    def test_icl_exact_prefix_updates_teacher_freeze_and_resume(self):
        self.check_model("ascent-icl")

    def check_model(self, method, kl_direction="forward"):
        import torch
        from types import SimpleNamespace
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        torch.set_num_threads(2)
        torch.manual_seed(42)
        config = Config(method=method)
        config.ascent.kl_direction = kl_direction
        config.model.device = "cpu"
        config.ascent.logits_chunk_tokens = 1
        architecture = Qwen3_5TextConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=["linear_attention", "full_attention"],
            max_position_embeddings=4096, eos_token_id=2, pad_token_id=0,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]},
        )
        backend = HFBackend(config, model=Qwen3_5ForCausalLM(architecture),
                            tokenizer=SimpleNamespace(encode=lambda text, **kwargs: [10, 11]))
        self.assertTrue(all("down_proj" in name for name in backend.target_modules))
        trainer = backend.trainer
        # Independently verify the configured objective through the actual head.
        student_hidden = torch.randn(1, 2, architecture.hidden_size, requires_grad=True)
        teacher_hidden = torch.randn_like(student_hidden)
        head = backend.model.get_base_model().get_output_embeddings()
        student_dist = torch.distributions.Categorical(logits=head(student_hidden).float())
        teacher_dist = torch.distributions.Categorical(logits=head(teacher_hidden).detach().float())
        first, second = ((teacher_dist, student_dist) if kl_direction == "forward"
                         else (student_dist, teacher_dist))
        torch.testing.assert_close(trainer._block_loss(student_hidden, teacher_hidden),
                                   torch.distributions.kl_divergence(first, second).sum())
        e, z = episode(), [10, 11]
        if config.uses_icl:
            for step in e.steps:
                step["input_token_ids"] = [20, 21, 22] + step["input_token_ids"]
        training_steps = current_episode_training_steps(e) if config.uses_icl else e.steps
        original = copy.deepcopy(e.steps)
        before = {name: p.detach().clone() for name, p in backend.model.named_parameters()}
        with backend.model.disable_adapter(), torch.no_grad():
            teacher_before = trainer._response_hidden(z + [3, 4, 5, 6, 7], 3).clone()
            full = backend.model(input_ids=torch.tensor([z + [3, 4, 5, 6, 7]]), use_cache=False).logits
            head = backend.model.get_base_model().get_output_embeddings()
            torch.testing.assert_close(head(teacher_before), full[:, -3:])
        initial = copy.deepcopy(backend.state_dict())
        result = backend.distill_episode(e)
        self.assertEqual(result["kl_direction"], kl_direction)
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(result["target_tokens"], 5)  # Invalid turn AND EOS included.
        self.assertEqual(result["training_turns"], 2)
        self.assertEqual(e.steps, original)
        self.assertTrue(any(not torch.equal(before[n], p) for n, p in backend.model.named_parameters()
                            if "lora_" in n))
        for name, param in backend.model.named_parameters():
            if "lora_" not in name:
                torch.testing.assert_close(before[name], param, rtol=0, atol=0)
        with backend.model.disable_adapter(), torch.no_grad():
            teacher_after = trainer._response_hidden(z + [3, 4, 5, 6, 7], 3)
        torch.testing.assert_close(teacher_before, teacher_after, rtol=0, atol=0)
        final = copy.deepcopy(backend.state_dict())
        # Different head chunking must preserve the full objective and update.
        backend.load_state_dict(initial)
        config.ascent.logits_chunk_tokens = 32
        repeated = trainer.train(training_steps, z, 4096)
        for key, value in final["adapter"].items():
            torch.testing.assert_close(value, backend.state_dict()["adapter"][key], rtol=1e-4, atol=1e-6)
        self.assertAlmostEqual(result["loss"], repeated["loss"], places=5)
        # Exact replay from an optimizer checkpoint (beyond initial empty state).
        backend.load_state_dict(final)
        trainer.train(training_steps, z, 4096)
        continued = copy.deepcopy(backend.state_dict())
        backend.load_state_dict(final)
        trainer.train(training_steps, z, 4096)
        for key, value in continued["adapter"].items():
            torch.testing.assert_close(value, backend.state_dict()["adapter"][key], rtol=0, atol=0)
        steps_before = trainer.steps
        with self.assertRaisesRegex(ValueError, "context overflow"):
            trainer.train(training_steps, z, 2)
        self.assertEqual(trainer.steps, steps_before)
        self.assertFalse(backend.model.training)
        self.assertTrue(backend.model.config.use_cache)


if __name__ == "__main__":
    unittest.main()
