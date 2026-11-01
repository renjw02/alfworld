"""An explicit, portable, content-checked stream shared across methods."""
import hashlib
import json
import os
from pathlib import Path
import random
from alfworld.agents.environment.alfred_tw_env import TASK_TYPES


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_root(config):
    return Path(os.path.expandvars(config.data_dir)).expanduser().resolve()


def build_manifest(config):
    root = data_root(config)
    split = root / "json_2.1.1" / config.split
    entries = []
    for game in sorted(split.rglob("game.tw-pddl")):
        relative = game.relative_to(root).as_posix()
        if "movable" in relative or "Sliced" in relative:
            continue
        trajectory = game.with_name("traj_data.json")
        if not trajectory.exists():
            continue
        data = json.loads(game.read_text())
        task = json.loads(trajectory.read_text())["task_type"]
        if not data.get("solvable", False) or task not in TASK_TYPES.values():
            continue
        entries.append({"episode_id": game.parent.relative_to(split).as_posix(),
                        "game": relative, "task_type": task,
                        "sha256": digest(game), "trajectory_sha256": digest(trajectory)})
    if not entries:
        raise ValueError(f"No usable games in {split}; run alfworld-download --text-only")
    rng = random.Random(config.seed)
    if config.order == "shuffled":
        rng.shuffle(entries)
    else:
        grouped = []
        for task_id in config.task_order:
            block = [entry for entry in entries if entry["task_type"] == TASK_TYPES[task_id]]
            rng.shuffle(block)
            grouped.extend(block)
        entries = grouped
    return {"schema_version": 1, "split": config.split, "order": config.order,
            "seed": config.seed, "task_order": config.task_order, "episodes": entries}


def validate_manifest(manifest, config):
    if manifest.get("schema_version") != 1 or manifest.get("split") != config.split:
        raise ValueError("Manifest schema or split mismatch")
    # A supplied manifest is authoritative for ordering, but mismatched config is an error.
    for key in ("order", "seed", "task_order"):
        if manifest[key] != getattr(config, key):
            raise ValueError(f"Manifest/config mismatch: {key}")
    root = data_root(config)
    seen = set()
    if not manifest["episodes"]:
        raise ValueError("Empty manifest")
    for entry in manifest["episodes"]:
        if entry["episode_id"] in seen:
            raise ValueError("Duplicate episode in manifest")
        seen.add(entry["episode_id"])
        game = (root / entry["game"]).resolve()
        if not game.is_relative_to(root / "json_2.1.1" / config.split):
            raise ValueError("Game path outside configured split")
        if digest(game) != entry["sha256"] or digest(game.with_name("traj_data.json")) != entry["trajectory_sha256"]:
            raise ValueError(f"Dataset changed: {entry['game']}")
    return manifest


def prepare_manifest(config):
    manifest = json.loads(Path(config.manifest).read_text()) if config.manifest else build_manifest(config)
    return validate_manifest(manifest, config)
