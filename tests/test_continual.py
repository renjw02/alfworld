import copy
import json
from pathlib import Path
import tempfile
import unittest
from alfworld.continual.config import Config, load_config
from alfworld.continual.methods import Method
from alfworld.continual.model import MockBackend, layer_device_map
from alfworld.continual.prompting import parse_react, observation_message, system_message
from alfworld.continual.stream import build_manifest, validate_manifest
from alfworld.continual.training import token_chunks
from alfworld.continual.types import Episode, Generation
from alfworld.continual.runner import run
from alfworld.continual.logging import load_checkpoint, commit_checkpoint
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
    def test_icl_success_only_history_and_resume(self):
        class Env(FakeEnv):
            def step(self, action):
                if action == "invented command":
                    return "Nothing happens.", 0, False, {"admissible_commands": ["look"], "won": False}
                return super().step(action)

        class Backend(MockBackend):
            def generate(self, messages, **kwargs):
                start = max(i for i, m in enumerate(messages)
                            if m["content"].startswith("BEGIN EPISODE"))
                turn = sum(m["role"] == "assistant" for m in messages[start:])
                text = ["malformed response", "Reasoning: try\nAction: invented command",
                        "Reasoning: inspect\nAction: look"][turn]
                prompt = json.dumps(messages)
                return Generation(text, prompt, list(prompt.encode()), list(text.encode()), 0.0)

        for method_name in ("icl", "ascent-icl"):
            with self.subTest(method=method_name):
                self.config.method = method_name
                self.config.output_dir = str(self.root / method_name)
                self.config.icl_success_only = True
                self.config.max_steps = 3
                self.config.window_episodes = 3
                run(self.config, backend_factory=Backend, env_factory=Env)
                state, _ = load_checkpoint(Path(self.config.output_dir))
                self.assertEqual(len(state["method"]["history"]), 2)
                for messages in state["method"]["history"]:
                    self.assertEqual(len(messages), 4)
                    self.assertTrue(messages[0]["content"].startswith("BEGIN EPISODE"))
                    self.assertEqual(messages[1]["content"], "Reasoning: inspect\nAction: look")
                    self.assertIn("Found it.", messages[2]["content"])
                    self.assertIn("Success: True", messages[3]["content"])
                run(self.config, resume=True, backend_factory=Backend, env_factory=Env)
                restored, _ = load_checkpoint(Path(self.config.output_dir))
                self.assertEqual(restored["method"], state["method"])
                self.config.icl_success_only = False
                with self.assertRaisesRegex(ValueError, "Resume configuration differs"):
                    run(self.config, resume=True, backend_factory=Backend, env_factory=Env)

    def test_icl_success_only_window_and_isolation(self):
        config = Config(method="icl", icl_success_only=True, window_episodes=3)
        method = Method(config, MockBackend(config))
        def episode(name, success=True, valid=True):
            return Episode(name, "pick", 0,
                           messages=[{"role": "user", "content": name},
                                     {"role": "assistant", "content": "look"},
                                     {"role": "user", "content": "feedback"},
                                     {"role": "user", "content": "terminal"}],
                           steps=[dict(format_error=None, executed_action="look",
                                       action="look", admissible_commands=["look"] if valid else [])],
                           outcome={"success": success})
        for name in ("first", "second", "third"):
            e = episode(name)
            method.observe_episode(e)
            e.messages[0]["content"] = "mutated"
        self.assertEqual([m[0]["content"] for m in method.history], ["second", "third"])
        saved = copy.deepcopy(method.history)
        method.observe_episode(episode("failed", success=False))
        method.observe_episode(episode("empty", valid=False))
        self.assertEqual(method.history, saved)
        current = episode("current", success=False)
        self.assertEqual(method.build_context(current)[-4:], current.messages)
        config.window_episodes = 1
        method.observe_episode(current)
        self.assertEqual(method.history, [])
        config.icl_success_only = "false"
        with self.assertRaisesRegex(ValueError, "icl_success_only"):
            config.validate()

    def test_layer_sharding_and_inference_resume(self):
        mapping = layer_device_map(32, ["cuda:0", "cuda:1"])
        self.assertEqual(mapping["model.layers.15"], "cuda:0")
        self.assertEqual(mapping["model.layers.16"], "cuda:1")
        self.assertEqual(mapping["lm_head"], mapping["model.embed_tokens"])
        self.assertEqual(len([k for k in mapping if k.startswith("model.layers.")]), 32)
        eight = layer_device_map(32, [f"cuda:{i}" for i in range(8)])
        for index in range(32):
            self.assertEqual(eight[f"model.layers.{index}"], f"cuda:{index // 4}")
        run(self.config, backend_factory=MockBackend, env_factory=FakeEnv)
        self.config.model.devices = ["cuda:0", "cuda:1"]
        self.config.model.gpu_memory_gib = [34, 25]
        result = run(self.config, resume=True, backend_factory=MockBackend, env_factory=FakeEnv)
        self.assertEqual(result["episodes"], 2)
        self.config.method = "history-ntp"
        with self.assertRaisesRegex(ValueError, "inference methods, ASCENT and TT-OPSD only"):
            self.config.validate()

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

    def test_ascent_runner_tokens_checkpoint_and_resume(self):
        self.check_distillation_runner("ascent")

    def test_tt_opsd_runner_tokens_checkpoint_and_resume(self):
        self.check_distillation_runner("tt-opsd")

    def test_tt_opsd_invalid_feedback_and_uncommitted_resume(self):
        self.config.method = "tt-opsd"
        self.config.max_steps = 3
        self.config.stream.limit = 4
        self.config.checkpoint_interval = 2
        self.config.action_mode = "free"
        seen = []

        class Env(FakeEnv):
            def step(self, action):
                if action == "invented command":
                    return "Nothing happens.", 0, False, {"admissible_commands": ["look"], "won": False}
                return super().step(action)

        class Backend(MockBackend):
            def generate(self, messages, **kwargs):
                turn = sum(m["role"] == "assistant" for m in messages)
                text = ["malformed response", "Reasoning: try\nAction: invented command",
                        "Reasoning: inspect\nAction: look"][turn]
                prompt = json.dumps(messages)
                return Generation(text, prompt, list(prompt.encode()), list(text.encode()), 0.0)

            def distill_episode(self, episode):
                seen.append((self.updates, copy.deepcopy(episode.steps)))
                return super().distill_episode(episode)

        class InterruptedBackend(Backend):
            def generate(self, messages, **kwargs):
                if self.updates == 3:
                    raise RuntimeError("interrupt after uncommitted TT-OPSD update")
                return super().generate(messages, **kwargs)

        with self.assertRaisesRegex(RuntimeError, "uncommitted TT-OPSD"):
            run(self.config, backend_factory=InterruptedBackend, env_factory=Env)
        root = Path(self.config.output_dir)
        state, rows = load_checkpoint(root)
        self.assertEqual((state["next_position"], state["backend"]["updates"], len(rows)), (2, 2, 2))
        self.assertEqual(state["method"]["history"], [])
        uncommitted = seen[-1]
        seen.clear()
        run(self.config, resume=True, backend_factory=Backend, env_factory=Env)
        self.assertEqual(seen[0][0], uncommitted[0])
        for old, new in zip(uncommitted[1], seen[0][1]):
            self.assertEqual(old["input_token_ids"], new["input_token_ids"])
            self.assertEqual(old["output_token_ids"], new["output_token_ids"])
        resumed, rows = load_checkpoint(root)
        self.assertEqual(resumed["backend"]["updates"], 4)
        for row in rows:
            folder = root / row["attempt_dir"]
            e = json.loads((folder / "episode.json").read_text())
            update = json.loads((folder / "update.json").read_text())
            self.assertEqual(e["outcome"]["environment_calls"], 2)
            self.assertEqual(update["retained_turns"], [1, 2, 3])
            records = update["teacher_records"]
            self.assertIn("Format error:", records[0]["teacher_record"])
            self.assertNotIn("Nothing happens.", records[0]["teacher_record"])
            self.assertIn("invented command", records[1]["teacher_record"])
            self.assertIn("Nothing happens.", records[1]["teacher_record"])
            self.assertNotIn("Found it.", records[1]["teacher_record"])
            self.assertIn("Found it.", records[2]["teacher_record"])
        self.config.output_dir = str(self.root / "tt-opsd-baseline")
        run(self.config, backend_factory=Backend, env_factory=Env)
        baseline, _ = load_checkpoint(Path(self.config.output_dir))
        self.assertEqual(resumed["method"], baseline["method"])
        self.assertEqual(resumed["backend"], baseline["backend"])

    def check_distillation_runner(self, method):
        self.config.method = method
        self.config.checkpoint_interval = 1
        result = run(self.config, backend_factory=MockBackend, env_factory=FakeEnv)
        self.assertEqual(result["episodes"], 2)
        state, completed = load_checkpoint(Path(self.config.output_dir))
        self.assertEqual(state["backend"]["updates"], 2)
        self.assertEqual(state["method"]["version"], 2)
        for row in completed:
            folder = Path(self.config.output_dir) / row["attempt_dir"]
            ep = json.loads((folder / "episode.json").read_text())
            update = json.loads((folder / "update.json").read_text())
            self.assertTrue(ep["steps"][0]["input_token_ids"])
            self.assertTrue(ep["steps"][0]["output_token_ids"])
            self.assertTrue(update["mock"])
            self.assertEqual(update["kind"], method)
            if method == "tt-opsd":
                self.assertEqual(update["filtered_turns"], [])
                self.assertIn("Found it.", update["teacher_records"][0]["teacher_record"])
                self.assertNotIn("teacher_record", update)
        run(self.config, resume=True, backend_factory=MockBackend, env_factory=FakeEnv)
        resumed, _ = load_checkpoint(Path(self.config.output_dir))
        self.assertEqual(resumed["backend"], state["backend"])

    def test_ascent_resume_changes_only_device_placement(self):
        self.config.method = "ascent-icl"
        run(self.config, backend_factory=MockBackend, env_factory=FakeEnv)
        root = Path(self.config.output_dir)
        before, rows = load_checkpoint(root)
        self.config.model.devices = ["cuda:0", "cuda:1"]
        self.config.model.gpu_memory_gib = [70, 70]
        run(self.config, resume=True, backend_factory=MockBackend, env_factory=FakeEnv)
        after, resumed_rows = load_checkpoint(root)
        self.assertEqual(before["backend"], after["backend"])
        self.assertEqual(before["method"], after["method"])
        self.assertEqual(rows, resumed_rows)
        self.config.window_episodes += 1
        with self.assertRaisesRegex(ValueError, "Resume configuration differs"):
            run(self.config, resume=True, backend_factory=MockBackend, env_factory=FakeEnv)

    def test_sft_runner_checkpoint_and_resume(self):
        self.config.method = "success-only-sft"
        self.config.checkpoint_interval = 1
        result = run(self.config, backend_factory=MockBackend, env_factory=FakeEnv)
        self.assertEqual(result["episodes"], 2)
        root = Path(self.config.output_dir)
        state, completed = load_checkpoint(root)
        self.assertEqual(state["backend"]["updates"], 2)
        self.assertEqual(state["method"]["version"], 2)
        for row in completed:
            update = json.loads((root / row["attempt_dir"] / "update.json").read_text())
            self.assertEqual(update["retained_steps"], [1])
            self.assertEqual(update["kind"], "success-only-sft")
        run(self.config, resume=True, backend_factory=MockBackend, env_factory=FakeEnv)
        resumed, _ = load_checkpoint(root)
        self.assertEqual(resumed["backend"], state["backend"])

        # Legacy SFT config has no batching fields: only explicit 1x1 may resume.
        path = root / "config.json"
        saved = json.loads(path.read_text())
        saved["sft"].pop("batch_size")
        saved["sft"].pop("gradient_accumulation_steps")
        path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            run(self.config, resume=True, backend_factory=MockBackend, env_factory=FakeEnv)
        self.config.sft.batch_size = 1
        self.config.sft.gradient_accumulation_steps = 1
        run(self.config, resume=True, backend_factory=MockBackend, env_factory=FakeEnv)

    def test_default_config(self):
        config = load_config("configs/continual.yaml").validate()
        self.assertEqual(config.model.max_new_tokens, 1024)
        self.assertEqual(config.checkpoint_interval, 10)
        self.assertEqual(config.action_mode, "admissible")
        for invalid in (0, -1, True, 1.5):
            config.checkpoint_interval = invalid
            with self.assertRaisesRegex(ValueError, "checkpoint_interval"):
                config.validate()

    def test_ascent_icl_checkpoint_replays_history_and_updates_together(self):
        self.config.method = "ascent-icl"
        self.config.stream.order = "task_blocked"
        self.config.stream.limit = 4
        self.config.window_episodes = 3
        self.config.checkpoint_interval = 2
        seen = []
        distilled = []
        class Backend(MockBackend):
            def generate(self, messages, **kwargs):
                seen.append((self.updates, copy.deepcopy(messages)))
                return super().generate(messages, **kwargs)
            def distill_episode(self, episode):
                # Retain the exact rollout prefix, including ICL history, for ASCENT.
                prefix = json.loads(bytes(episode.steps[0]["input_token_ids"]).decode())
                self_prefix = seen[-1][1]
                if prefix != self_prefix:
                    raise AssertionError("Distillation lost rollout context")
                distilled.append(copy.deepcopy(prefix))
                return super().distill_episode(episode)
        class FailingBackend(Backend):
            def generate(self, messages, **kwargs):
                if self.updates == 3:
                    raise RuntimeError("interrupt after uncommitted update")
                return super().generate(messages, **kwargs)
        with self.assertRaisesRegex(RuntimeError, "uncommitted update"):
            run(self.config, backend_factory=FailingBackend, env_factory=FakeEnv)
        root = Path(self.config.output_dir)
        state, rows = load_checkpoint(root)
        self.assertEqual((state["next_position"], state["backend"]["updates"], len(rows)), (2, 2, 2))
        self.assertEqual(len(state["method"]["history"]), 2)
        interrupted_prefix = distilled[-1]
        seen.clear()
        run(self.config, resume=True, backend_factory=Backend, env_factory=FakeEnv)
        self.assertEqual(seen[0], (2, interrupted_prefix))
        self.assertEqual([sum(m["content"].startswith("BEGIN EPISODE") for m in messages)
                          for _, messages in seen], [3, 3])
        resumed, rows = load_checkpoint(root)
        self.assertEqual(resumed["backend"]["updates"], 4)
        self.assertEqual(resumed["method"]["version"], 4)
        self.assertEqual(len(resumed["method"]["history"]), 2)
        self.assertEqual([r["memory_reset_before"] for r in rows], [False] * 4)
        self.assertEqual(len(list((root / "checkpoints").glob("*.pkl"))), 1)
        # A clean run must reach the same combined context/parameter checkpoint.
        self.config.output_dir = str(self.root / "baseline")
        run(self.config, backend_factory=Backend, env_factory=FakeEnv)
        baseline, _ = load_checkpoint(Path(self.config.output_dir))
        self.assertEqual(resumed["method"], baseline["method"])
        self.assertEqual(resumed["backend"], baseline["backend"])

    def test_task_scope_context_and_cheatsheet_reset(self):
        for method in ("icl", "dc-cumulative"):
            with self.subTest(method=method):
                self.config.method = method
                self.config.memory_scope = "task_type"
                self.config.stream.order = "task_blocked"
                self.config.stream.limit = 12
                self.config.output_dir = str(self.root / method)
                contexts = []
                class Backend(MockBackend):
                    def generate(self, messages, **kwargs):
                        from alfworld.continual.cheatsheet import CURATOR_SYSTEM
                        if messages[0]["content"] != CURATOR_SYSTEM:
                            contexts.append(copy.deepcopy(messages))
                        return super().generate(messages, **kwargs)
                result = run(self.config, backend_factory=Backend, env_factory=FakeEnv)
                self.assertEqual(len(result["by_task"]), 6)
                self.assertTrue(all(v["episodes"] == 2 for v in result["by_task"].values()))
                for i, messages in enumerate(contexts):
                    if method == "icl":
                        self.assertEqual(sum(m["content"].startswith("BEGIN EPISODE") for m in messages),
                                         1 if i % 2 == 0 else 2)
                    else:
                        self.assertEqual("(empty)" in messages[1]["content"], i % 2 == 0)
                root = Path(self.config.output_dir)
                _, rows = load_checkpoint(root)
                self.assertEqual([r["memory_reset_before"] for r in rows],
                                 [False, False] + [True, False] * 5)
                if method == "dc-cumulative":
                    self.assertEqual([r["cheatsheet_version_after"] for r in rows], [1, 2] * 6)
                import csv
                scored = list(csv.DictReader((root / "episodes.csv").open()))
                self.assertEqual([int(r["task_episode"]) for r in scored], [1, 2] * 6)
                self.assertEqual(len(list(csv.DictReader((root / "task_summary.csv").open()))), 6)

    def test_task_scope_training_boundary_resume(self):
        self.config.method = "history-ntp-assistant"
        self.config.memory_scope = "task_type"
        self.config.stream.order = "task_blocked"
        self.config.stream.limit = 12
        class FailingBackend(MockBackend):
            calls = 0
            def train_episode(self, messages):
                self.calls += 1
                result = super().train_episode(messages)
                if self.calls == 3:
                    raise RuntimeError("new category interrupted")
                return result
        with self.assertRaisesRegex(RuntimeError, "new category"):
            run(self.config, backend_factory=FailingBackend, env_factory=FakeEnv)
        root = Path(self.config.output_dir)
        state, rows = load_checkpoint(root)
        self.assertEqual(state["next_position"], 2)  # category end saved before interval 10
        self.assertEqual(state["backend"]["updates"], 2)
        old_traces = {p: p.read_bytes() for p in root.glob("traces/*/episodes/*/*/steps.jsonl")}
        run(self.config, resume=True, backend_factory=MockBackend, env_factory=FakeEnv)
        state, rows = load_checkpoint(root)
        self.assertEqual(state["backend"]["updates"], 2)  # only the last category's updates
        self.assertEqual([r["model_version_after"] for r in rows], [1, 2] * 6)
        self.assertEqual([r["position"] for r in rows], list(range(12)))
        for path, contents in old_traces.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_task_scope_requires_blocked_order(self):
        self.config.memory_scope = "task_type"
        with self.assertRaisesRegex(ValueError, "task_blocked"):
            self.config.validate()

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

    def test_assistant_episode_filter(self):
        for name in ("history-ntp-assistant", "history-ntp-assistant-success"):
            self.config.method = name
            self.config.validate()
            backend = MockBackend(self.config)
            method = Method(self.config, backend)
            episode = Episode("one", "task", 0, [{"role": "assistant", "content": "Reasoning: x\nAction: look"}])
            episode.outcome = {"success": False}
            result = method.observe_episode(episode)
            expected = int(name == "history-ntp-assistant")
            self.assertEqual(backend.updates, expected)
            self.assertEqual(method.version, expected)
            self.assertEqual(result.get("skipped", False), not bool(expected))
            episode.outcome["success"] = True
            method.observe_episode(episode)
            self.assertEqual(backend.updates, expected + 1)
            self.assertEqual(method.version, expected + 1)
            self.config.output_dir = str(self.root / name)
            run(self.config, env_factory=FakeEnv)
            state, completed = load_checkpoint(Path(self.config.output_dir))
            self.assertEqual(state["backend"]["updates"], 2)
            self.assertEqual(state["method"]["version"], 2)
            self.assertEqual(len(completed), 2)
            run(self.config, resume=True, env_factory=FakeEnv)
            resumed, rows = load_checkpoint(Path(self.config.output_dir))
            self.assertEqual(resumed["method"], state["method"])
            self.assertEqual(rows, completed)

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
        instructions = system_message("admissible")["content"]
        self.assertIn("copying exactly one command", instructions)
        self.assertNotIn("Action syntax examples", instructions)
        self.assertIn("Reasoning:", system_message("free")["content"])

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
        self.config.checkpoint_interval = 1
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
        original_traces = {p: p.read_bytes() for p in root.glob("traces/*/episodes/*/*/steps.jsonl")}
        result = run(self.config, resume=True, backend_factory=MockBackend, env_factory=FakeEnv)
        self.assertEqual(len(list(root.glob("traces/*/invocation.json"))), 2)
        for path, contents in original_traces.items():
            self.assertEqual(path.read_bytes(), contents)
        self.assertEqual(result["episodes"], 2)
        state, rows = load_checkpoint(root)
        self.assertEqual(state["backend"]["updates"], 2)
        self.assertEqual([r["model_version_after"] for r in rows], [1, 2])
        self.assertEqual(len(list(root.glob("traces/*/episodes/0001/*/steps.jsonl"))), 2)
        # Already complete resume never generates or trains again.
        run(self.config, resume=True, backend_factory=MockBackend, env_factory=FakeEnv)
        self.assertEqual(len(list(root.glob("traces/*/episodes/0001/*/steps.jsonl"))), 2)
        committed_log = root / rows[1]["attempt_dir"] / "steps.jsonl"
        step = [json.loads(line) for line in committed_log.read_text().splitlines() if json.loads(line)["event"] == "step"][0]
        self.assertEqual(bytes(step["input_token_ids"]).decode(), step["prompt"])
        self.assertIn("Reasoning:", step["raw_output"])
        self.assertNotIn("Reasoning:", step["executed_action"])

    def test_dc_curator_failure_resume(self):
        from alfworld.continual.cheatsheet import CURATOR_SYSTEM
        self.config.method = "dc-cumulative"
        curator_calls, action_contexts = [], []
        class Backend(MockBackend):
            def state_dict(self):
                raise AssertionError("DC must not snapshot weights")
            def generate(self, messages, *, max_new_tokens=None):
                if messages[0]["content"] == CURATOR_SYSTEM:
                    curator_calls.append(copy.deepcopy(messages))
                    if len(curator_calls) == 2:
                        raise RuntimeError("curator interrupted")
                else:
                    action_contexts.append(copy.deepcopy(messages))
                return super().generate(messages, max_new_tokens=max_new_tokens)
        with self.assertRaisesRegex(RuntimeError, "curator interrupted"):
            run(self.config, backend_factory=Backend, env_factory=FakeEnv)
        root = Path(self.config.output_dir)
        state, rows = load_checkpoint(root)
        self.assertEqual(state["next_position"], 1)
        self.assertEqual(state["method"]["cheatsheet_version"], 1)
        self.assertEqual(state["method"]["version"], 0)
        self.assertNotIn("backend", state)
        self.assertIn(state["method"]["cheatsheet"], action_contexts[1][1]["content"])
        self.assertFalse((root / "checkpoints").exists())
        first_update = root / rows[0]["attempt_dir"] / "update.json"
        original = first_update.read_bytes()
        run(self.config, resume=True, backend_factory=Backend, env_factory=FakeEnv)
        state, rows = load_checkpoint(root)
        self.assertEqual(len(curator_calls), 3)  # failed second call plus one retry on resume
        self.assertEqual(len(rows), 2)
        self.assertEqual(state["method"]["cheatsheet_version"], 2)
        self.assertEqual(first_update.read_bytes(), original)
        self.assertEqual(curator_calls[1], curator_calls[2])
        self.assertEqual([r["model_version_after"] for r in rows], [0, 0])
        self.assertEqual([r["training_seconds"] for r in rows], [0, 0])
        for row in rows:
            update = json.loads((root / row["attempt_dir"] / "update.json").read_text())
            self.assertEqual(bytes(update["input_token_ids"]).decode(), update["prompt"])
            self.assertEqual(row["curator_output_tokens"], len(update["output_token_ids"]))
            self.assertTrue(row["cheatsheet_update_accepted"])
        run(self.config, resume=True, backend_factory=Backend, env_factory=FakeEnv)
        self.assertEqual(len(curator_calls), 3)

    def test_training_checkpoint_interval_and_final_save(self):
        self.config.method = "history-ntp"
        self.config.stream.limit = 12
        snapshots = []
        class Backend(MockBackend):
            def state_dict(self):
                snapshots.append(self.updates)
                return super().state_dict()
            def train_episode(self, messages):
                result = super().train_episode(messages)
                if self.updates == 12:
                    raise RuntimeError("interrupted before final checkpoint")
                return result
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            run(self.config, backend_factory=Backend, env_factory=FakeEnv)
        root = Path(self.config.output_dir)
        state, rows = load_checkpoint(root)
        self.assertEqual(snapshots, [10])
        self.assertEqual(state["next_position"], 10)
        self.assertEqual(state["backend"]["updates"], 10)
        self.assertEqual(len(rows), 10)
        # Episode 11 has traces, but cannot be committed without its trained weights.
        old_traces = {p: p.read_bytes() for p in root.glob("traces/*/episodes/*/*/steps.jsonl")}
        class ResumedBackend(MockBackend):
            def state_dict(self):
                snapshots.append(self.updates)
                return super().state_dict()
        run(self.config, resume=True, backend_factory=ResumedBackend, env_factory=FakeEnv)
        state, rows = load_checkpoint(root)
        self.assertEqual(snapshots, [10, 12])
        self.assertEqual(state["backend"]["updates"], 12)
        self.assertEqual([r["position"] for r in rows], list(range(12)))
        self.assertEqual(len(list(root.glob("checkpoints/*.pkl"))), 1)
        for path, contents in old_traces.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_training_resume_before_first_checkpoint(self):
        self.config.method = "history-ntp"
        class FailingBackend(MockBackend):
            def train_episode(self, messages):
                super().train_episode(messages)
                raise RuntimeError("before checkpoint")
        with self.assertRaisesRegex(RuntimeError, "before checkpoint"):
            run(self.config, backend_factory=FailingBackend, env_factory=FakeEnv)
        root = Path(self.config.output_dir)
        state, rows = load_checkpoint(root)
        self.assertNotIn("backend", state)
        self.assertEqual(rows, [])
        self.assertFalse((root / "checkpoints").exists())
        run(self.config, resume=True, backend_factory=MockBackend, env_factory=FakeEnv)
        state, rows = load_checkpoint(root)
        self.assertEqual(state["backend"]["updates"], 2)
        self.assertEqual(len(rows), 2)

    def test_legacy_config_and_checkpoint_resume(self):
        class FailingBackend(MockBackend):
            def generate(self, messages):
                raise RuntimeError("legacy interruption")
        with self.assertRaisesRegex(RuntimeError, "legacy interruption"):
            run(self.config, backend_factory=FailingBackend, env_factory=FakeEnv)
        root = Path(self.config.output_dir)
        saved = json.loads((root / "config.json").read_text())
        saved.pop("icl_success_only")
        saved.pop("checkpoint_interval")
        saved.pop("cheatsheet_max_new_tokens")
        (root / "config.json").write_text(json.dumps(saved))
        state, rows = load_checkpoint(root)
        state["backend"] = {"updates": 0}
        commit_checkpoint(root, state, rows)
        pointer = json.loads((root / "commit.json").read_text())
        pointer.pop("model_checkpoint")
        (root / "commit.json").write_text(json.dumps(pointer))
        run(self.config, resume=True, backend_factory=MockBackend, env_factory=FakeEnv)
        state, rows = load_checkpoint(root)
        self.assertEqual(len(rows), 2)
        self.assertNotIn("backend", state)
        self.assertEqual(list(root.glob("checkpoints/*.pkl")), [])

    def test_inference_resume_without_model_checkpoint(self):
        for method in ("memoryless", "icl"):
            with self.subTest(method=method):
                self.config.method = method
                self.config.output_dir = str(self.root / method)
                class Backend(MockBackend):
                    def state_dict(self):
                        raise AssertionError("Inference must never snapshot model state")
                class FailingBackend(Backend):
                    calls = 0
                    def generate(self, messages):
                        self.calls += 1
                        if self.calls == 2:
                            raise RuntimeError("interrupted inference")
                        return super().generate(messages)
                with self.assertRaisesRegex(RuntimeError, "interrupted inference"):
                    run(self.config, backend_factory=FailingBackend, env_factory=FakeEnv)
                root = Path(self.config.output_dir)
                state, rows = load_checkpoint(root)
                self.assertEqual(state["next_position"], 1)
                self.assertNotIn("backend", state)
                self.assertEqual(len(state["method"]["history"]), 1 if method == "icl" else 0)
                run(self.config, resume=True, backend_factory=Backend, env_factory=FakeEnv)
                self.assertFalse((root / "checkpoints").exists())
                self.assertEqual(len(load_checkpoint(root)[1]), 2)

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
