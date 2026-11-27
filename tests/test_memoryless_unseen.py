import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from alfworld.continual.config import Config
from alfworld.continual.memoryless_unseen import TASKS, prepare
from alfworld.continual.model import MockBackend
from alfworld.continual.stream import build_manifest
from alfworld.continual.transfer import evaluate


class MemorylessTests(unittest.TestCase):
    def test_all_six_tasks_preserve_order_and_never_load_or_retain_source_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            reference.mkdir()
            config = Config(method="ascent")
            config.model.backend = "mock"
            config.model.context_overflow = "error"
            config.stream.split = "valid_unseen"
            config.stream.data_dir = str(root / "data")
            for task in TASKS.values():
                for index in range(2):
                    game = root / "data/json_2.1.1/valid_unseen" / task / str(index) / "game.tw-pddl"
                    game.parent.mkdir(parents=True)
                    game.write_text(json.dumps({"solvable": True}))
                    game.with_name("traj_data.json").write_text(json.dumps({
                        "task_type": task, "scene": {"floor_plan": "FloorPlan10"}}))
            full = build_manifest(config.stream)
            (reference / "config.json").write_text(json.dumps(config.to_dict()))
            (reference / "provenance.json").write_text(json.dumps({
                "source_rooms": ["FloorPlan10"], "source_run": "comparison-only",
                "target_task_type": TASKS["clean_place"]}))
            clean = copy.deepcopy(full)
            clean["episodes"] = [e for e in clean["episodes"] if e["task_type"] == TASKS["clean_place"]]
            for entry in clean["episodes"]:
                entry.update(floor_plan="FloorPlan10", room_seen_in_source=True)
            (reference / "manifest.json").write_text(json.dumps(clean))

            class Env:
                def __init__(self, *args): pass
                def reset(self): return "Find a mug.", {"admissible_commands": ["look"]}
                def step(self, action): return "Found.", 1, True, {"admissible_commands": ["look"], "won": True}
                def close(self): pass

            class Backend(MockBackend):
                def __init__(self, config):
                    super().__init__(config)
                    self.contexts = []
                def generate(self, messages, **kwargs):
                    self.contexts.append(copy.deepcopy(messages))
                    return super().generate(messages, **kwargs)
                def load_state_dict(self, state):
                    raise AssertionError("Memoryless must not load a checkpoint")
                def distill_episode(self, episode):
                    raise AssertionError("Memoryless must not train")

            all_ids = []
            for task, task_type in TASKS.items():
                prepared = prepare(reference, root / task, target_task=task, device="cuda:2")
                cfg, manifest, history, provenance = prepared
                self.assertEqual(cfg.method, "memoryless")
                self.assertFalse(cfg.trains_parameters)
                self.assertEqual(cfg.max_steps, config.max_steps)
                self.assertEqual(cfg.model.max_input_tokens, config.model.max_input_tokens)
                self.assertEqual(cfg.model.context_overflow, "error")
                self.assertEqual(history, [])
                self.assertIsNone(provenance["source_checkpoint"])
                ids = [e["episode_id"] for e in manifest["episodes"]]
                self.assertEqual(ids, [e["episode_id"] for e in full["episodes"] if e["task_type"] == task_type])
                all_ids.extend(ids)
                backend = Backend(cfg)
                with patch("alfworld.continual.transfer.seed_all") as seed:
                    summary = evaluate(*prepared, backend_factory=lambda _: backend, env_factory=Env)
                self.assertEqual([call.args[0] for call in seed.call_args_list], [42, 42, 43])
                self.assertEqual(summary["episodes"], 2)
                self.assertNotIn(ids[0], json.dumps(backend.contexts[1]))
                self.assertIn(ids[1], json.dumps(backend.contexts[1]))
            self.assertEqual(len(all_ids), 12)
            self.assertEqual(len(set(all_ids)), 12)


if __name__ == "__main__":
    unittest.main()
