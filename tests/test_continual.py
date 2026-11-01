import copy
import json
from pathlib import Path
import tempfile
import unittest
from alfworld.continual.config import Config, load_config
from alfworld.continual.methods import Method
from alfworld.continual.model import MockBackend
from alfworld.continual.prompting import parse_react, observation_message
from alfworld.continual.stream import build_manifest, validate_manifest
from alfworld.continual.training import token_chunks
from alfworld.continual.types import Episode, Generation
from alfworld.continual.runner import run
from alfworld.continual.logging import load_checkpoint
from alfworld.agents.environment.alfred_tw_env import TASK_TYPES


class FakeEnv:
    closed = 0
    def __init__(self, *args):
        self.calls = 0
    def reset(self):
        return "Your task is to find a mug.", {"admissible_commands": ["look"], "won": False}
    def step(self, action):
        self.calls += 1
        return "Found it.", 1, True, {"admissible_commands": ["look"], "won": True}
    def close(self):
        type(self).closed += 1


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = Config()
        self.config.model.backend = "mock"
        self.config.max_steps = 2
        self.config.stream.data_dir = str(self.root / "data")
        for task in TASK_TYPES.values():
            for i in range(2):
                path = Path(self.config.stream.data_dir) / "json_2.1.1/valid_seen" / task / str(i)
                path.mkdir(parents=True)
                (path / "game.tw-pddl").write_text(json.dumps({"solvable": True}))
                (path / "traj_data.json").write_text(json.dumps({"task_type": task}))
        self.config.stream.limit = 2
        self.config.output_dir = str(self.root / "run")

    def tearDown(self):
        self.temp.cleanup()

    def test_default_config(self):
        config = load_config("configs/continual.yaml").validate()
        self.assertEqual(config.model.max_new_tokens, 1024)

    def test_react(self):
        parsed = parse_react("Reasoning: First inspect.\nThen take it.\nAction: look\n")
        self.assertIsNone(parsed.error)
        self.assertEqual(parsed.action, "look")
        self.assertIn("Then", parsed.reasoning)
        for text in ("look", "Reasoning: x", "Reasoning: x\nAction: look\nAction: open fridge 1",
                     "Reasoning: x\nAction: look\nextra", "Reasoning: \nAction: look",
                     "Reasoning: <think>x</think>\nAction: look"):
            self.assertIsNotNone(parse_react(text).error, text)
        self.assertIsNotNone(parse_react("Reasoning: x\nAction: look", truncated=True).error)

    def test_windows_and_ntp_scope(self):
        self.config.method = "icl"
        self.config.window_episodes = 2
        method = Method(self.config, MockBackend(self.config))
        first = Episode("one", "task", 0, [{"role": "user", "content": "one feedback"}])
        second = Episode("two", "task", 1, [{"role": "user", "content": "two feedback"}])
        third = Episode("three", "task", 2, [{"role": "user", "content": "current"}])
        method.observe_episode(first)
        method.observe_episode(second)
        self.assertEqual([m["content"] for m in method.build_context(third)[1:]], ["two feedback", "current"])
        self.config.window_episodes = 1
        method.observe_episode(second)
        baseline = copy.deepcopy(self.config)
        baseline.method = "memoryless"
        self.assertEqual(method.build_context(third), Method(baseline, MockBackend(baseline)).build_context(third))
        baseline.method = "history-ntp"
        ntp = Method(baseline, MockBackend(baseline))
        result = ntp.observe_episode(second)
        self.assertNotIn("one feedback", result["training_text"])
        self.assertIn("two feedback", result["training_text"])
        self.assertEqual(ntp.history, [])
        self.assertEqual(ntp.version, 1)

    def test_token_target_coverage(self):
        for length in (2, 7, 8, 9, 20):
            ids = list(range(length))
            chunks = list(token_chunks(ids, 8))
            self.assertEqual([token for chunk in chunks for token in chunk[1:]], ids[1:])
            self.assertTrue(all(2 <= len(chunk) <= 8 for chunk in chunks))

    def test_prompt_modes(self):
        free = observation_message("room", ["look"], "free")
        candidates = observation_message("room", ["look"], "admissible")
        self.assertNotIn("Admissible", free["content"])
        self.assertIn("Admissible commands:\nlook", candidates["content"])

    def test_stream_reproducibility_integrity(self):
        shuffled = build_manifest(self.config.stream)
        self.assertEqual(shuffled, build_manifest(self.config.stream))
        self.config.stream.order = "task_blocked"
        blocked = build_manifest(self.config.stream)
        self.assertEqual(len(blocked["episodes"]), 12)
        self.assertEqual([e["task_type"] for e in blocked["episodes"]],
                         [task for task in TASK_TYPES.values() for _ in range(2)])
        self.assertEqual({e["episode_id"] for e in shuffled["episodes"]},
                         {e["episode_id"] for e in blocked["episodes"]})
        validate_manifest(blocked, self.config.stream)
        duplicate = copy.deepcopy(blocked)
        duplicate["episodes"].append(duplicate["episodes"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_manifest(duplicate, self.config.stream)
        path = Path(self.config.stream.data_dir) / blocked["episodes"][0]["game"]
        path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            validate_manifest(blocked, self.config.stream)

    def test_interrupted_update_resume(self):
        self.config.method = "history-ntp"
        class FailingBackend(MockBackend):
            def train_episode(self, messages):
                result = super().train_episode(messages)
                if self.updates == 2:
                    raise RuntimeError("simulated failure after update")
                return result
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            run(self.config, backend_factory=FailingBackend, env_factory=FakeEnv)
        root = Path(self.config.output_dir)
        state, rows = load_checkpoint(root)
        self.assertEqual(state["next_position"], 1)
        self.assertEqual(state["backend"]["updates"], 1)
        self.assertEqual(len(rows), 1)
        result = run(self.config, resume=True, backend_factory=MockBackend, env_factory=FakeEnv)
        self.assertEqual(result["episodes"], 2)
        state, rows = load_checkpoint(root)
        self.assertEqual(state["backend"]["updates"], 2)
        self.assertEqual([r["model_version_after"] for r in rows], [1, 2])
        self.assertEqual(len(list(root.glob("episodes/0001/*/steps.jsonl"))), 2)
        # Already complete resume never generates or trains again.
        run(self.config, resume=True, backend_factory=MockBackend, env_factory=FakeEnv)
        self.assertEqual(len(list(root.glob("episodes/0001/*/steps.jsonl"))), 2)
        committed_log = root / rows[1]["attempt_dir"] / "steps.jsonl"
        step = [json.loads(line) for line in committed_log.read_text().splitlines() if json.loads(line)["event"] == "step"][0]
        self.assertEqual(bytes(step["input_token_ids"]).decode(), step["prompt"])
        self.assertIn("Reasoning:", step["raw_output"])
        self.assertNotIn("Reasoning:", step["executed_action"])

    def test_malformed_output_counts_without_environment_step(self):
        class MalformedBackend(MockBackend):
            def generate(self, messages):
                return Generation("bad", "prompt", [1], [2], 0)
        run(self.config, backend_factory=MalformedBackend, env_factory=FakeEnv)
        _, rows = load_checkpoint(Path(self.config.output_dir))
        self.assertTrue(all(r["environment_calls"] == 0 for r in rows))
        self.assertTrue(all(r["format_errors"] == 2 for r in rows))
        self.assertTrue(all(r["end_reason"] == "step_limit" for r in rows))


if __name__ == "__main__":
    unittest.main()
