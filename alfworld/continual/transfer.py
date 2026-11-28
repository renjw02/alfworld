"""Frozen cross-task evaluation of a completed heat&place200 run.

Usage: python -m alfworld.continual.transfer --source-run RUN --output-dir NEW_RUN
Use --target-task to select an unseen task (default: clean_place), and
--prepare-only to inspect the episodes without a GPU.
--source-position selects a training boundary (default: 200). ICL reconstructs
the last three complete episodes at that boundary (including failures).
ASCENT loads the matching retained checkpoint without source history.
Neither method retains test episodes or updates parameters. Room-level scores
separate rooms present in heat200 from rooms new to heat200.
Source checkpoints are trusted local pickle files. Use a fresh output directory;
partial results are persisted per episode, but resuming is not supported.
"""
import argparse
import copy
import json
import pickle
from pathlib import Path

from .config import Config
from .environment import TextEnvironment
from .logging import AttemptLog, run_lock, write_json
from .methods import Method
from .model import make_backend
from .runner import run_episode, seed_all
from .stream import build_manifest, data_root, digest, validate_manifest


TARGET_TASKS = {
    "clean_place": "pick_clean_then_place_in_recep",
    "pick_place": "pick_and_place_simple",
    "cool_place": "pick_cool_then_place_in_recep",
    "look_at_light": "look_at_obj_in_light",
    "pick_two_place": "pick_two_obj_and_place",
}


class FrozenMethod(Method):
    """Retain source demonstrations, but never learn from evaluation episodes."""

    def observe_step(self, record):
        return None

    def observe_episode(self, episode):
        return None


def prepare(source, output, *, device=None, data_dir=None, target_task="clean_place",
            source_position=200):
    if (isinstance(source_position, bool) or not isinstance(source_position, int)
            or not 1 <= source_position <= 200):
        raise ValueError("source_position must be an integer in 1..200")
    if target_task not in TARGET_TASKS:
        raise ValueError(f"Unknown target task: {target_task}")
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Evaluation output must be separate from the source run")
    config = Config()
    for key, value in json.loads((source / "config.json").read_text()).items():
        if isinstance(value, dict):
            value = type(getattr(config, key))(**value)
        setattr(config, key, value)
    if config.method not in ("icl", "ascent"):
        raise ValueError("Source must be a standard ICL or ASCENT run")
    if config.stream.split != "heat_place_200":
        raise ValueError("Source must be a heat_place_200 run")
    source_commit = "commit.json"
    commit = json.loads((source / source_commit).read_text())
    if config.method == "ascent" and commit["next_position"] != source_position:
        source_commit = f"milestones/episode-{source_position:04d}.json"
        if not (source / source_commit).is_file():
            raise ValueError(f"No retained ASCENT checkpoint at position {source_position}")
        commit = json.loads((source / source_commit).read_text())
    completed = commit["completed"]
    if (len(completed) != commit["next_position"] or len(completed) < source_position
            or (config.method == "ascent" and commit["next_position"] != source_position)):
        raise ValueError("Source does not contain the requested committed position")
    completed = completed[:source_position]
    if any(e["task_type"] != "pick_heat_then_place_in_recep" for e in completed):
        raise ValueError("Source contains non-heat episodes")
    source_manifest = json.loads((source / "manifest.json").read_text())
    if len(source_manifest["episodes"]) < source_position:
        raise ValueError("Source manifest is shorter than the requested position")
    source_rooms = set()
    if data_dir:
        config.stream.data_dir = data_dir
    root = data_root(config.stream)
    for row, entry in zip(completed, source_manifest["episodes"][:source_position]):
        if row["episode_id"] != entry["episode_id"]:
            raise ValueError("Source committed episodes do not match manifest order")
        trajectory = root / Path(entry["game"]).with_name("traj_data.json")
        if digest(trajectory) != entry["trajectory_sha256"]:
            raise ValueError("Source trajectory has changed")
        source_rooms.add(json.loads(trajectory.read_text())["scene"]["floor_plan"])
    history, history_sources = [], []
    if config.method == "icl":
        # Never use an episode beyond the requested boundary.
        for row in completed[-3:]:
            path = source / row["attempt_dir"] / "episode.json"
            episode = json.loads(path.read_text())
            if episode["episode_id"] != row["episode_id"]:
                raise ValueError("Source episode identity mismatch")
            history.append(episode["messages"])
            history_sources.append({"episode_id": row["episode_id"], "sha256": digest(path)})
    config.output_dir = str(output)
    config.checkpoint_keep_positions = []
    config.memory_scope = "stream"
    config.stream.split = "valid_unseen"
    config.stream.manifest = None
    config.stream.limit = None
    config.stream.order = "shuffled"
    config.stream.seed = 42
    # Never silently drop the source demonstrations in a frozen ICL comparison.
    config.model.context_overflow = "error"
    if device:
        config.model.device = device
        config.model.devices = []
        config.model.gpu_memory_gib = []
    config.validate()
    manifest = build_manifest(config.stream)
    manifest["episodes"] = [e for e in manifest["episodes"]
                            if e["task_type"] == TARGET_TASKS[target_task]]
    validate_manifest(manifest, config.stream)
    for entry in manifest["episodes"]:
        trajectory = root / Path(entry["game"]).with_name("traj_data.json")
        room = json.loads(trajectory.read_text())["scene"]["floor_plan"]
        entry.update(floor_plan=room, room_seen_in_source=room in source_rooms)
    provenance = {
        "protocol": ("frozen_heat_prefix_to_task_unseen_v1" if source_position != 200
                     else "frozen_heat200_to_clean_unseen_v1" if target_task == "clean_place"
                     else "frozen_heat200_to_task_unseen_v1"), "source_run": str(source),
        "target_task": target_task, "target_task_type": TARGET_TASKS[target_task],
        "source_method": config.method, "source_position": source_position,
        "source_commit": source_commit, "source_commit_sha256": digest(source / source_commit),
        "source_checkpoint": (None if config.method == "icl" and source_position != commit["next_position"]
                              else commit["checkpoint"]),
        "source_checkpoint_sha256": (None if config.method == "icl" and source_position != commit["next_position"]
                                     else digest(source / commit["checkpoint"])),
        "history_reconstructed_from_traces": config.method == "icl",
        "history_positions": list(range(max(1, source_position - 2), source_position + 1)) if history else [],
        "history_sources": history_sources, "history_episodes": len(history),
        "parameter_updates": False, "test_history_retention": False,
        "source_rooms": sorted(source_rooms), "test_episodes": len(manifest["episodes"]),
    }
    return config, manifest, history, provenance


def summarize(rows, provenance):
    def stats(values):
        wins = sum(e["success"] for e in values)
        return {"episodes": len(values), "successes": wins,
                "success_rate": wins / len(values) if values else None}
    return {**stats(rows), "protocol": provenance["protocol"],
            "source_method": provenance["source_method"],
            "source_position": provenance.get("source_position"),
            "target_task": provenance.get("target_task", "clean_place"),
            "by_room": {room: stats([r for r in rows if r["floor_plan"] == room])
                        for room in sorted({r["floor_plan"] for r in rows})},
            "by_source_room_exposure": {
                label: stats([r for r in rows if r["room_seen_in_source"] == seen])
                for label, seen in (("seen", True), ("unseen", False))}}


def evaluate(config, manifest, history, provenance, *, backend_factory=make_backend,
             env_factory=TextEnvironment):
    root = Path(config.output_dir)
    with run_lock(root):
        if (root / "results.json").exists() or (root / "provenance.json").exists():
            raise ValueError("Use a fresh evaluation output directory")
        write_json(root / "config.json", config.to_dict())
        write_json(root / "manifest.json", manifest)
        write_json(root / "provenance.json", provenance)
        write_json(root / "frozen_history.json", history)
        seed_all(config.stream.seed, config)
        backend = backend_factory(config)
        if config.method == "ascent":
            source = Path(provenance["source_run"])
            # Completed runs are immutable inputs. Lock while checking/loading their pointer.
            with run_lock(source, shared=True):
                pointer = source / provenance.get("source_commit", "commit.json")
                commit = json.loads(pointer.read_text())
                if (commit["checkpoint"] != provenance["source_checkpoint"]
                        or ("source_commit_sha256" in provenance
                            and digest(pointer) != provenance["source_commit_sha256"])
                        or digest(source / commit["checkpoint"]) != provenance["source_checkpoint_sha256"]):
                    raise ValueError("Source checkpoint changed after preparation")
                # Trusted local state: load the selected milestone, never the final pointer.
                with (source / commit["checkpoint"]).open("rb") as stream:
                    state = pickle.load(stream)
                if state["next_position"] != provenance.get("source_position", commit["next_position"]):
                    raise ValueError("Checkpoint state position does not match selection")
            if "backend" not in state:
                raise ValueError("ASCENT source has no model checkpoint")
            backend.load_state_dict(state["backend"])
            del state
        if hasattr(backend, "model"):
            backend.model.requires_grad_(False)
            backend.model.eval()
            # No optimizer is needed during transfer evaluation.
            backend.trainer = None
        write_json(root / "model.json", backend.metadata())
        method = FrozenMethod(config, backend)
        method.history = copy.deepcopy(history)
        rows = []
        for index, entry in enumerate(manifest["episodes"]):
            # Fixed source state; each target episode gets a fresh local conversation.
            seed_all(config.stream.seed + index, config)
            log = AttemptLog(root, index)
            try:
                episode = run_episode(config, entry, index, method, log, env_factory)
                row = {"episode_id": entry["episode_id"], "position": index,
                       "task_type": entry["task_type"],
                       "floor_plan": entry["floor_plan"],
                       "room_seen_in_source": entry["room_seen_in_source"],
                       "attempt_dir": str(log.path.relative_to(root)), **episode.outcome}
                rows.append(row)
                write_json(root / "results.json", rows)
                write_json(root / "summary.json", summarize(rows, provenance))
                print(f"[{index + 1}/{len(manifest['episodes'])}] frozen {config.method}: {episode.outcome}", flush=True)
            finally:
                log.close()
        return summarize(rows, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--source-position", type=int, default=200)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device")
    parser.add_argument("--data-dir")
    parser.add_argument("--target-task", choices=TARGET_TASKS, default="clean_place")
    parser.add_argument("--prepare-only", action="store_true", help="Inspect protocol and data without loading a model")
    args = parser.parse_args()
    prepared = prepare(args.source_run, args.output_dir, device=args.device,
                       data_dir=args.data_dir, target_task=args.target_task,
                       source_position=args.source_position)
    if args.prepare_only:
        config, manifest, history, provenance = prepared
        print(json.dumps(provenance, indent=2))
    else:
        print(json.dumps(evaluate(*prepared), indent=2))


if __name__ == "__main__":
    main()
