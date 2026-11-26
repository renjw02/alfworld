"""Frozen cross-task evaluation of a completed heat&place200 run.

Usage: python -m alfworld.continual.transfer --source-run RUN --output-dir NEW_RUN
Use --prepare-only to inspect the 31 clean&place unseen episodes without a GPU.
ICL uses the source run's last three complete episodes (including failures).
ASCENT uses final adapter weights without source conversation history.
Neither method retains test episodes or updates parameters. Room-level scores
separate FloorPlan10 (present in heat200) from FloorPlan424 (new to heat200).
Source checkpoints are trusted local pickle files. Use a fresh output directory;
partial results are persisted per episode, but resuming is not supported.
"""
import argparse
import copy
import json
from pathlib import Path

from .config import Config
from .environment import TextEnvironment
from .logging import AttemptLog, load_checkpoint, run_lock, write_json
from .methods import Method
from .model import make_backend
from .runner import run_episode, seed_all
from .stream import build_manifest, data_root, digest, validate_manifest


class FrozenMethod(Method):
    """Retain source demonstrations, but never learn from evaluation episodes."""

    def observe_step(self, record):
        return None

    def observe_episode(self, episode):
        return None


def prepare(source, output, *, device=None, data_dir=None):
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
    commit = json.loads((source / "commit.json").read_text())
    completed = commit["completed"]
    if commit["next_position"] != 200 or len(completed) != 200:
        raise ValueError("Source must have all 200 episodes committed")
    if any(e["task_type"] != "pick_heat_then_place_in_recep" for e in completed):
        raise ValueError("Source contains non-heat episodes")
    source_manifest = json.loads((source / "manifest.json").read_text())
    source_rooms = set()
    if data_dir:
        config.stream.data_dir = data_dir
    root = data_root(config.stream)
    for entry in source_manifest["episodes"]:
        trajectory = root / Path(entry["game"]).with_name("traj_data.json")
        if digest(trajectory) != entry["trajectory_sha256"]:
            raise ValueError("Source trajectory has changed")
        source_rooms.add(json.loads(trajectory.read_text())["scene"]["floor_plan"])
    history, history_sources = [], []
    if config.method == "icl":
        # Use actual final episodes, irrespective of the source retention policy.
        for row in completed[-3:]:
            path = source / row["attempt_dir"] / "episode.json"
            episode = json.loads(path.read_text())
            if episode["episode_id"] != row["episode_id"]:
                raise ValueError("Source episode identity mismatch")
            history.append(episode["messages"])
            history_sources.append({"episode_id": row["episode_id"], "sha256": digest(path)})
    config.output_dir = str(output)
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
                            if e["task_type"] == "pick_clean_then_place_in_recep"]
    validate_manifest(manifest, config.stream)
    for entry in manifest["episodes"]:
        trajectory = root / Path(entry["game"]).with_name("traj_data.json")
        room = json.loads(trajectory.read_text())["scene"]["floor_plan"]
        entry.update(floor_plan=room, room_seen_in_source=room in source_rooms)
    provenance = {
        "protocol": "frozen_heat200_to_clean_unseen_v1", "source_run": str(source),
        "source_method": config.method, "source_position": 200,
        "source_checkpoint": commit["checkpoint"],
        "source_checkpoint_sha256": digest(source / commit["checkpoint"]),
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
            with run_lock(source):
                commit = json.loads((source / "commit.json").read_text())
                if (commit["checkpoint"] != provenance["source_checkpoint"]
                        or digest(source / commit["checkpoint"]) != provenance["source_checkpoint_sha256"]):
                    raise ValueError("Source checkpoint changed after preparation")
                state, _ = load_checkpoint(source)
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device")
    parser.add_argument("--data-dir")
    parser.add_argument("--prepare-only", action="store_true", help="Inspect protocol and data without loading a model")
    args = parser.parse_args()
    prepared = prepare(args.source_run, args.output_dir, device=args.device, data_dir=args.data_dir)
    if args.prepare_only:
        config, manifest, history, provenance = prepared
        print(json.dumps(provenance, indent=2))
    else:
        print(json.dumps(evaluate(*prepared), indent=2))


if __name__ == "__main__":
    main()
