"""Opt-in real CUDA validation: ALFWORLD_TEST_MULTIGPU=1, two or more visible GPUs."""
import copy
import os
from types import SimpleNamespace
import unittest

from alfworld.continual.config import Config
from alfworld.continual.model import HFBackend
from alfworld.continual.runner import rng_state, restore_rng, seed_all
from alfworld.continual.types import Episode


@unittest.skipUnless(os.environ.get("ALFWORLD_TEST_MULTIGPU") == "1", "opt-in multi-GPU test")
class MultiGPUAscentTests(unittest.TestCase):
    def test_generation_gradients_teacher_and_optimizer_resume(self):
        import torch
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        torch.set_num_threads(2)
        torch.manual_seed(42)
        count = min(torch.cuda.device_count(), 6)
        architecture = Qwen3_5TextConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=count * 2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=["linear_attention", "full_attention"] * count,
            max_position_embeddings=4096, eos_token_id=2, pad_token_id=0,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]},
        )
        base = Qwen3_5ForCausalLM(architecture)
        config = Config(method="ascent-icl")
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [10, 11])
        single = HFBackend(config, model=copy.deepcopy(base).to("cuda:0"), tokenizer=tokenizer)
        parallel_config = copy.deepcopy(config)
        parallel_config.model.devices = [f"cuda:{i}" for i in range(count)]
        parallel = HFBackend(parallel_config, model=base, tokenizer=tokenizer)
        parallel.load_state_dict(single.state_dict())
        self.assertEqual({p.device.index for p in parallel.model.parameters()}, set(range(count)))
        self.assertFalse(any(hasattr(m, "_hf_hook") for m in parallel.model.modules()))
        ids = torch.tensor([[3, 4, 5]], device="cuda:0")
        with torch.no_grad():
            a = single.model.generate(input_ids=ids, max_new_tokens=3, do_sample=False)
            b = parallel.model.generate(input_ids=ids, max_new_tokens=3, do_sample=False)
        torch.testing.assert_close(a, b)
        e = Episode("test", "pick", 0, outcome={"success": True})
        for turn, response in enumerate(([6, 7, 2], [8, 2]), 1):
            e.steps.append(dict(step=turn, observation="Your task is to: find a mug.",
                                format_error=None, executed_action="look", action="look",
                                admissible_commands=["look"], raw_output="Reasoning: inspect\nAction: look",
                                input_token_ids=[20, 21, 22, 3, 4, 5],
                                training_input_token_ids=[3, 4, 5], output_token_ids=response))
        before = parallel.state_dict()
        with parallel.model.disable_adapter(), torch.no_grad():
            teacher_before = parallel.trainer._response_hidden([10, 11, 3, 4, 5], 2).clone()
        a = single.distill_episode(e)
        b = parallel.distill_episode(e)
        self.assertAlmostEqual(a["loss"], b["loss"], delta=2e-5)
        self.assertEqual(b["optimizer_steps"], 2)
        checkpoint = single.state_dict()
        sharded_state = parallel.state_dict()
        for key, value in checkpoint["adapter"].items():
            torch.testing.assert_close(value, sharded_state["adapter"][key], rtol=3e-4, atol=2e-6)
        for device in parallel_config.model.devices:
            names = [name for name, p in parallel.model.named_parameters()
                     if str(p.device) == device and "lora_B" in name]
            self.assertTrue(names)
            self.assertTrue(any(torch.count_nonzero(dict(parallel.model.named_parameters())[n]).item()
                                for n in names), f"No adapter update on {device}")
        self.assertTrue(any(not torch.equal(value, sharded_state["adapter"][key])
                            for key, value in before["adapter"].items()))
        with parallel.model.disable_adapter(), torch.no_grad():
            teacher_after = parallel.trainer._response_hidden([10, 11, 3, 4, 5], 2)
        torch.testing.assert_close(teacher_before, teacher_after, rtol=0, atol=0)
        # Move a nonempty single-device AdamW checkpoint to the sharded model.
        parallel.load_state_dict(checkpoint)
        for parameter, state in parallel.trainer.optimizer.state.items():
            self.assertEqual(state["exp_avg"].device, parameter.device)
            self.assertEqual(state["exp_avg_sq"].device, parameter.device)
        a = single.distill_episode(e)
        b = parallel.distill_episode(e)
        self.assertAlmostEqual(a["loss"], b["loss"], delta=2e-5)
        for key, value in single.state_dict()["adapter"].items():
            torch.testing.assert_close(value, parallel.state_dict()["adapter"][key], rtol=3e-4, atol=2e-6)
        # Added devices get a saved RNG state; later snapshots preserve all GPUs.
        seed_all(42, config)
        old_rng = rng_state(config)
        restore_rng(old_rng, parallel_config)
        for device in parallel_config.model.devices:
            torch.testing.assert_close(torch.cuda.get_rng_state(device), old_rng["cuda"])
        new_rng = rng_state(parallel_config)
        self.assertEqual(set(new_rng["cuda_devices"]), set(parallel_config.model.devices))


if __name__ == "__main__":
    unittest.main()
