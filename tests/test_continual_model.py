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


if __name__ == "__main__":
    unittest.main()
