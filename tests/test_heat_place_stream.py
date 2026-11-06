import copy
import json
from pathlib import Path
import tempfile
import unittest

from alfworld.continual.config import Config, load_config
from alfworld.continual.logging import load_checkpoint
from alfworld.continual.model import MockBackend
from alfworld.continual.runner import run
from alfworld.continual.stream import HEAT_PLACE_TASK, prepare_manifest, validate_manifest


class HeatEnv:
    def __init__(self, game, *args):
        self.success = game.parent.name == "success"

    def reset(self):
        return "Your task is to: heat a potato and put it on the counter.", {
            "admissible_commands": ["look"], "won": False}

    def step(self, action):
        return "Done." if self.success else "Nothing happens.", int(self.success), True, {
            "admissible_commands": ["look"], "won": self.success}

    def close(self):
        pass


class HeatPlaceStreamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = Config()
        self.config.stream.data_dir = str(self.root)
        self.config.stream.split = "valid_heat_place"
        self.config.model.backend = "mock"
        self.config.max_steps = 1
        for split in ("valid_seen", "valid_unseen"):
            # Identical task/trial names in different splits must remain distinct.
            for trial in ("success", "failure"):
                self.game(split, HEAT_PLACE_TASK, trial)
            self.game(split, "pick_and_place_simple", "other")
            self.game(split, HEAT_PLACE_TASK, "unsolvable", solvable=False)
            self.game(split, HEAT_PLACE_TASK, "AppleSliced")
            self.game(split, HEAT_PLACE_TASK, "missing-traj").with_name("traj_data.json").unlink()

    def game(self, split, task, trial, solvable=True):
        folder = self.root / "json_2.1.1" / split / task / trial
        folder.mkdir(parents=True)
        (folder / "traj_data.json").write_text(json.dumps({"task_type": task}))
        game = folder / "game.tw-pddl"
        game.write_text(json.dumps({"solvable": solvable}))
        return game

    def test_merge_filter_order_and_portable_manifest(self):
        self.config.validate()
        manifest = prepare_manifest(self.config.stream)
        self.assertEqual(manifest, prepare_manifest(self.config.stream))
        entries = manifest["episodes"]
        self.assertEqual(len(entries), 4)
        self.assertEqual(len({e["episode_id"] for e in entries}), 4)
        self.assertEqual({e["task_type"] for e in entries}, {HEAT_PLACE_TASK})
        self.assertEqual({e["source_split"] for e in entries}, {"valid_seen", "valid_unseen"})
        self.assertTrue(all(not Path(e["game"]).is_absolute() for e in entries))
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest))
        self.config.stream.manifest = str(path)
        self.assertEqual(manifest, prepare_manifest(self.config.stream))
        self.config.stream.manifest = None
        self.config.stream.order = "task_blocked"
        blocked = prepare_manifest(self.config.stream)
        self.assertEqual({e["episode_id"] for e in blocked["episodes"]}, {e["episode_id"] for e in entries})

    def test_reject_wrong_source_task_identity_and_content(self):
        manifest = prepare_manifest(self.config.stream)
        for key, value, error in (("source_split", "train", "outside"),
                                  ("task_type", "pick_and_place_simple", "Non-heat"),
                                  ("episode_id", "wrong", "episode_id")):
            with self.subTest(key=key):
                bad = copy.deepcopy(manifest)
                bad["episodes"][0][key] = value
                with self.assertRaisesRegex(ValueError, error):
                    validate_manifest(bad, self.config.stream)
        bad = copy.deepcopy(manifest)
        bad["episodes"] = [e for e in bad["episodes"] if e["source_split"] == "valid_seen"]
        with self.assertRaisesRegex(ValueError, "both"):
            validate_manifest(bad, self.config.stream)
        game = self.root / manifest["episodes"][0]["game"]
        game.write_text("changed")
        with self.assertRaisesRegex(ValueError, "Dataset changed"):
            validate_manifest(manifest, self.config.stream)

    def test_missing_source_fails_instead_of_silently_shrinking(self):
        for path in (self.root / "json_2.1.1/valid_unseen").rglob("game.tw-pddl"):
            path.unlink()
        with self.assertRaisesRegex(ValueError, "No usable games.*valid_unseen"):
            prepare_manifest(self.config.stream)

    def test_three_methods_run_and_resume_same_stream(self):
        manifests = []
        for method in ("icl", "dc-cumulative", "ascent"):
            with self.subTest(method=method):
                self.config.method = method
                self.config.output_dir = str(self.root / method)
                result = run(self.config, backend_factory=MockBackend, env_factory=HeatEnv)
                self.assertEqual(result["episodes"], 4)
                out = Path(self.config.output_dir)
                state, rows = load_checkpoint(out)
                manifests.append(json.loads((out / "manifest.json").read_text()))
                self.assertEqual(sum(row["success"] for row in rows), 2)
                if method == "icl":
                    self.assertEqual(len(state["method"]["history"]), 3)
                    self.assertTrue(any("Nothing happens." in str(e) for e in state["method"]["history"]))
                elif method == "dc-cumulative":
                    self.assertEqual(state["method"]["cheatsheet_version"], 4)
                else:
                    self.assertEqual(state["backend"]["updates"], 2)
                    for row in rows:
                        update = json.loads((out / row["attempt_dir"] / "update.json").read_text())
                        if not row["success"]:
                            self.assertEqual(update["skip_reason"], "unsuccessful_episode")
                metadata = json.loads((out / "metadata.json").read_text())
                self.assertEqual(metadata["protocol"], "valid_heat_place_streaming_adaptation")
                resumed = run(self.config, resume=True, backend_factory=MockBackend, env_factory=HeatEnv)
                self.assertEqual(resumed["episodes"], 4)
                if method == "ascent":
                    # Old runs omitted the direction and must retain forward KL.
                    path = out / "config.json"
                    values = json.loads(path.read_text())
                    values["ascent"].pop("kl_direction")
                    path.write_text(json.dumps(values))
                    self.assertEqual(run(self.config, resume=True, backend_factory=MockBackend,
                                         env_factory=HeatEnv)["episodes"], 4)
                    self.config.ascent.kl_direction = "reverse"
                    with self.assertRaisesRegex(ValueError, "Resume configuration differs"):
                        run(self.config, resume=True, backend_factory=MockBackend, env_factory=HeatEnv)
        self.assertEqual(manifests[0], manifests[1])
        self.assertEqual(manifests[0], manifests[2])

    def test_shared_preset_accepts_all_three_methods(self):
        config = load_config("configs/continual-heat-place.yaml")
        for method in ("icl", "dc-cumulative", "ascent"):
            config.method = method
            config.validate()


if __name__ == "__main__":
    unittest.main()
