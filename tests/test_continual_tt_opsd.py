"""Paper B.2 invariants and full-logit reference checks on a real tiny Qwen3.5."""
import copy
import unittest
from unittest.mock import patch

from alfworld.continual.ascent import forward_kl_sum, success_record
from alfworld.continual.config import Config, load_config
from alfworld.continual.methods import Method
from alfworld.continual.model import HFBackend
from alfworld.continual.tt_opsd import turn_records
from test_continual_ascent import episode


def local_episode():
    e = episode()
    e.steps[0]["feedback"] = "A mug is on the table."
    e.steps[1]["feedback"] = "Nothing happens."
    return e


def tiny_backend():
    import torch
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = Config(method="tt-opsd")
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

    class Tokenizer:
        def encode(self, text, **kwargs):
            # Distinct contents and lengths expose misalignment/broadcast bugs.
            return [3 + ord(c) % 61 for c in text]

    return HFBackend(config, model=Qwen3_5ForCausalLM(architecture), tokenizer=Tokenizer())


class TTOPSDProtocolTests(unittest.TestCase):
    def test_config_is_separate_but_optimization_matches(self):
        config = load_config("configs/continual-tt-opsd.yaml").validate()
        ascent = load_config("configs/continual-ascent.yaml").validate()
        self.assertTrue(config.is_tt_opsd and config.is_distillation and config.trains_parameters)
        self.assertFalse(config.is_ascent or config.is_ntp or config.uses_icl)
        self.assertEqual(config.ascent, ascent.ascent)
        config.model.devices = ["cuda:0", "cuda:1"]
        config.validate()
        config.ascent.dropout = 0.1
        with self.assertRaisesRegex(ValueError, "dropout=0"):
            config.validate()

    def test_local_hindsight_includes_invalid_turn_and_no_future(self):
        e = local_episode()
        records = turn_records(e)
        self.assertEqual([r["turn"] for r in records], [1, 2])
        for i, record in enumerate(records):
            text = record["teacher_record"]
            self.assertIn("find a mug.", text)
            self.assertIn(e.steps[i]["raw_output"], text)
            self.assertIn(e.steps[i]["feedback"], text)
            self.assertNotIn(e.steps[1 - i]["raw_output"], text)
            self.assertNotIn(e.steps[1 - i]["feedback"], text)
            self.assertNotIn("GOAL ACHIEVED", text)
            self.assertNotIn("Room description", text)
        self.assertNotIn("invented command", success_record(e)[0])
        e.steps[1].update(raw_output="future changed", feedback="future observation changed")
        e.messages.append({"role": "user", "content": "terminal success information"})
        self.assertEqual(turn_records(e)[0], records[0])
        # For a parse error, use returned harness feedback, not stale env state.
        e.steps[1].update(format_error="missing action", executed_action=None,
                          feedback="Format error: missing action", next_observation="STALE STATE")
        text = turn_records(e)[1]["teacher_record"]
        self.assertIn("Format error: missing action", text)
        self.assertNotIn("STALE STATE", text)
        del e.steps[1]["feedback"]
        with self.assertRaises(KeyError):
            turn_records(e)  # Never silently replace missing feedback.
        e.outcome["success"] = False
        with self.assertRaisesRegex(ValueError, "verified successful"):
            turn_records(e)


class TTOPSDModelTests(unittest.TestCase):
    def assert_state_equal(self, left, right):
        import torch
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_reference_objective_prefixes_teacher_freeze_and_resume(self):
        import torch
        backend = tiny_backend()
        trainer = backend.trainer
        e = local_episode()
        original = copy.deepcopy(e)
        initial = copy.deepcopy(backend.state_dict())
        base_before = {n: p.detach().clone() for n, p in backend.model.named_parameters()
                       if "lora_" not in n}
        self.assertTrue(all("down_proj" in n for n in backend.target_modules))
        records = turn_records(e)
        privileged = [backend.tokenizer.encode(r["teacher_record"]) for r in records]
        expected_calls, teacher_logits = [], []
        reference = 0.0
        with torch.no_grad():
            for step, z in zip(e.steps, privileged):
                ids = step["input_token_ids"] + step["output_token_ids"][:-1]
                length = len(step["output_token_ids"])
                with backend.model.disable_adapter():
                    q = backend.model(input_ids=torch.tensor([z + ids]), use_cache=False).logits[:, -length:]
                p = backend.model(input_ids=torch.tensor([ids]), use_cache=False).logits[:, -length:]
                teacher_logits.append(q.clone())
                reference += forward_kl_sum(p, q).item() / length / len(e.steps)
                expected_calls.extend([(z + ids, length), (ids, length)])
        with patch.object(trainer, "_response_hidden", wraps=trainer._response_hidden) as hidden:
            result = backend.distill_episode(e)
        self.assertEqual([c.args for c in hidden.call_args_list], expected_calls * 2)
        self.assertAlmostEqual(result["updates"][0]["loss"], reference, places=6)
        self.assertEqual(result["kind"], "tt-opsd")
        self.assertEqual(result["retained_turns"], [1, 2])
        self.assertEqual(result["filtered_turns"], [])
        self.assertEqual(result["target_tokens"], 5)  # Includes invalid turn and EOS.
        self.assertEqual(result["optimizer_steps"], 2)  # Per episode, not per turn.
        self.assertEqual([r["teacher_token_ids"] for r in result["teacher_records"]], privileged)
        self.assertEqual(e, original)
        final = copy.deepcopy(backend.state_dict())
        self.assertTrue(any(not torch.equal(p, final["adapter"][n]) for n, p in initial["adapter"].items()))
        for name, param in backend.model.named_parameters():
            if name in base_before:
                torch.testing.assert_close(param, base_before[name], rtol=0, atol=0)
                self.assertIsNone(param.grad)
        with backend.model.disable_adapter(), torch.no_grad():
            for step, z, q in zip(e.steps, privileged, teacher_logits):
                ids = step["input_token_ids"] + step["output_token_ids"][:-1]
                actual = backend.model(input_ids=torch.tensor([z + ids]), use_cache=False).logits
                torch.testing.assert_close(actual[:, -q.shape[1]:], q, rtol=0, atol=0)
        # Full response-head blocks give the same update as single-token blocks.
        backend.load_state_dict(initial)
        backend.config.ascent.logits_chunk_tokens = 32
        repeated = backend.distill_episode(e)
        self.assertAlmostEqual(result["loss"], repeated["loss"], places=6)
        for name, value in final["adapter"].items():
            torch.testing.assert_close(value, backend.state_dict()["adapter"][name], rtol=1e-4, atol=1e-6)
        # Resume nonempty AdamW state and carry the update across episodes.
        backend.load_state_dict(final)
        backend.distill_episode(e)
        continued = copy.deepcopy(backend.state_dict())
        backend.load_state_dict(final)
        backend.distill_episode(e)
        self.assert_state_equal(continued, backend.state_dict())
        self.assertEqual(trainer.steps, 4)
        self.assertFalse(backend.model.training)
        self.assertTrue(backend.model.config.use_cache)

    def test_failure_empty_response_and_late_overflow_leave_state_unchanged(self):
        backend = tiny_backend()
        method = Method(backend.config, backend)
        e = local_episode()
        method.observe_episode(e)  # Gate must also preserve nonempty optimizer state.
        state = copy.deepcopy(backend.state_dict())
        e.outcome["success"] = False
        result = method.observe_episode(e)
        self.assertEqual((result["kind"], result["version_after"]), ("tt-opsd", 1))
        self.assertTrue(result["skipped"])
        self.assert_state_equal(state, backend.state_dict())
        with self.assertRaisesRegex(ValueError, "verified successful"):
            backend.distill_episode(e)
        e.outcome["success"] = True
        z = [[10], [11] * 100]
        with self.assertRaisesRegex(ValueError, "context overflow at turn 2"):
            backend.trainer.train_turns(e.steps, z, 50)
        self.assert_state_equal(state, backend.state_dict())
        with self.assertRaisesRegex(ValueError, "align"):
            backend.trainer.train_turns(e.steps, z[:1], 4096)
        # Empty turns must not shift the remaining turn's privileged information.
        e.steps[0]["output_token_ids"] = []
        with patch.object(backend.trainer, "_response_hidden", wraps=backend.trainer._response_hidden) as hidden:
            result = backend.distill_episode(e)
        self.assertEqual(result["training_turns"], 1)
        self.assertEqual(hidden.call_args_list[0].args[0][:3],
                         result["teacher_records"][1]["teacher_token_ids"][:3])
        self.assertEqual(hidden.call_args_list[0].args[0],
                         result["teacher_records"][1]["teacher_token_ids"] + [3, 4, 5, 8])
        state = copy.deepcopy(backend.state_dict())
        for step in e.steps:
            step["output_token_ids"] = []
        result = method.observe_episode(e)
        self.assertTrue(result["skipped"])
        self.assertEqual(result["version_after"], 1)
        self.assert_state_equal(state, backend.state_dict())


if __name__ == "__main__":
    unittest.main()
