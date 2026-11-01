"""CPU tests against the actual Qwen3.5 hybrid architecture (random tiny weights)."""
import copy
import unittest


class TinyModelTests(unittest.TestCase):
    def test_lora_ntp_restore_and_generation(self):
        import torch
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        from alfworld.continual.config import Config
        from alfworld.continual.model import HFBackend
        torch.set_num_threads(2)
        torch.manual_seed(42)
        tiny_config = Qwen3_5TextConfig(
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
            eos_token_id = 2
            pad_token_id = 0
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
                return "".join(m["role"] + ":" + m["content"] + "\n" for m in messages) + ("assistant:" if add_generation_prompt else "")
            def encode(self, text, **kwargs):
                return [3 + ord(c) % 61 for c in text]
            def __call__(self, text, **kwargs):
                return {"input_ids": self.encode(text),
                        "offset_mapping": [(i, i + 1) for i in range(len(text))]}
            def decode(self, ids, **kwargs):
                return "".join(chr(i + 60) for i in ids)
        config = Config(method="history-ntp")
        config.model.device = "cpu"
        config.model.max_new_tokens = 4
        config.training.chunk_tokens = 64
        config.training.learning_rate = 1e-3
        model = Qwen3_5ForCausalLM(tiny_config)
        backend = HFBackend(config, model=model, tokenizer=Tokenizer())
        before = {name: p.detach().clone() for name, p in backend.model.named_parameters()}
        messages = [{"role": "user", "content": "Observation: room"},
                    {"role": "assistant", "content": "Reasoning: inspect\nAction: look"},
                    {"role": "user", "content": "Feedback: success"}]
        result = backend.train_episode(messages)
        self.assertEqual(result["target_tokens"], result["tokens"] - 1)
        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))
        self.assertIn("Observation: room", result["training_text"])
        self.assertIn("Reasoning: inspect", result["training_text"])
        self.assertIn("Feedback: success", result["training_text"])
        changed = []
        for name, param in backend.model.named_parameters():
            if not torch.equal(before[name], param):
                changed.append(name)
                self.assertIn("lora_", name)
        self.assertTrue(changed)
        state = copy.deepcopy(backend.state_dict())
        second = backend.train_episode(messages)
        second_adapter = copy.deepcopy(backend.state_dict()["adapter"])
        self.assertEqual(second["total_optimizer_steps"], 2 * result["optimizer_steps"])
        backend.load_state_dict(state)
        self.assertEqual(backend.trainer.steps, result["optimizer_steps"])
        for key, tensor in state["adapter"].items():
            torch.testing.assert_close(backend.state_dict()["adapter"][key], tensor)
        self.assertEqual(backend.trainer.optimizer.state_dict()["state"].keys(), state["trainer"]["optimizer"]["state"].keys())
        repeated = backend.train_episode(messages)
        self.assertEqual(repeated["total_optimizer_steps"], second["total_optimizer_steps"])
        for key, tensor in second_adapter.items():
            torch.testing.assert_close(backend.state_dict()["adapter"][key], tensor, rtol=0, atol=0)
        generated = backend.generate(messages)
        self.assertLessEqual(len(generated.output_ids), 4)
        self.assertFalse(backend.model.training)
        self.assertTrue(backend.model.config.use_cache)
        for method in ("history-ntp-assistant", "history-ntp-assistant-success"):
            config.method = method
            masked_backend = HFBackend(config, model=Qwen3_5ForCausalLM(tiny_config), tokenizer=Tokenizer())
            masked_messages = [
                {"role": "system", "content": "instructions " * 12},
                {"role": "user", "content": "Observation: room"},
                {"role": "assistant", "content": "<think>inspect</think>\nReasoning: inspect\nAction: look"},
                {"role": "user", "content": "Feedback: success"},
                {"role": "assistant", "content": "Reasoning: done\nAction: look"},
            ]
            result = masked_backend.train_episode(masked_messages)
            text = result["training_text"]
            expected = set()
            for message in masked_messages:
                if message["role"] == "assistant":
                    start = text.index(message["content"])
                    expected.update(range(start, start + len(message["content"])))
            actual = {i for i, label in enumerate(result["training_labels"]) if label != -100}
            self.assertEqual(actual, expected)
            self.assertEqual(result["target_tokens"], len(expected))
            self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))
            before = copy.deepcopy(masked_backend.state_dict())
            empty = masked_backend.train_episode(masked_messages[:2])
            self.assertTrue(empty["skipped"])
            self.assertEqual(empty["optimizer_steps"], 0)
            self.assertIsNone(empty["loss"])
            self.assertEqual(masked_backend.trainer.steps, before["trainer"]["steps"])
            for key, tensor in before["adapter"].items():
                torch.testing.assert_close(masked_backend.state_dict()["adapter"][key], tensor, rtol=0, atol=0)


        # DC uses the same inference backend with a separate curator budget, no adapter.
        from unittest.mock import patch
        config.method = "dc-cumulative"
        dc_backend = HFBackend(config, model=Qwen3_5ForCausalLM(tiny_config), tokenizer=Tokenizer())
        self.assertIsNone(dc_backend.trainer)
        self.assertFalse(any(p.requires_grad for p in dc_backend.model.parameters()))
        with patch.object(dc_backend.model, "generate", wraps=dc_backend.model.generate) as generate:
            dc_backend.generate(messages, max_new_tokens=7)
            self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 7)
            dc_backend.generate(messages)
            self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 4)
        self.assertEqual(config.model.max_new_tokens, 4)
        with self.assertRaisesRegex(ValueError, "context length"):
            dc_backend.generate(messages, max_new_tokens=4096)


if __name__ == "__main__":
    unittest.main()
