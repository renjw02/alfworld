"""Memoryless baselines on all six unseen task types, matched to a transfer run.

Omit --target-task to queue all six types sequentially on one device. Each task
uses the full seed-shuffled unseen stream filtered by type, exactly as transfer
evaluation does. The base model is reloaded for each type, without any adapter.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from .config import Config
from .logging import write_json
from .stream import build_manifest, data_root, digest, validate_manifest
from .transfer import TARGET_TASKS, evaluate


TASKS = {**TARGET_TASKS, "heat_place": "pick_heat_then_place_in_recep"}


def prepare(reference, output, *, target_task, device=None, data_dir=None):
    if target_task not in TASKS:
        raise ValueError(f"Unknown target task: {target_task}")
    reference, output = Path(reference).resolve(), Path(output).resolve()
    if reference == output or reference in output.parents or output in reference.parents:
        raise ValueError("Evaluation output must be separate from the reference run")
    config = Config()
    for key, value in json.loads((reference / "config.json").read_text()).items():
        if isinstance(value, dict):
            value = type(getattr(config, key))(**value)
        setattr(config, key, value)
    if config.stream.split != "valid_unseen" or config.stream.limit is not None:
        raise ValueError("Reference must evaluate the full valid_unseen task subset")
    previous = json.loads((reference / "provenance.json").read_text())
    config.method = "memoryless"
    config.output_dir = str(output)
    config.checkpoint_keep_positions = []
    config.stream.manifest = None
    if data_dir:
        config.stream.data_dir = data_dir
    if device:
        config.model.device = device
        config.model.devices = []
        config.model.gpu_memory_gib = []
    config.validate()
    manifest = build_manifest(config.stream)
    manifest["episodes"] = [e for e in manifest["episodes"]
                            if e["task_type"] == TASKS[target_task]]
    validate_manifest(manifest, config.stream)
    # These labels refer to the comparison heat run, not memoryless training.
    source_rooms = set(previous["source_rooms"])
    root = data_root(config.stream)
    for entry in manifest["episodes"]:
        trajectory = root / Path(entry["game"]).with_name("traj_data.json")
        room = json.loads(trajectory.read_text())["scene"]["floor_plan"]
        entry.update(floor_plan=room, room_seen_in_source=room in source_rooms)
    if TASKS[target_task] == previous.get("target_task_type", "pick_clean_then_place_in_recep"):
        if manifest != json.loads((reference / "manifest.json").read_text()):
            raise ValueError("Target manifest differs from the reference evaluation")
    provenance = {
        "protocol": "memoryless_task_unseen_v1",
        "source_method": "memoryless", "source_run": None, "source_position": 0,
        "source_checkpoint": None, "source_checkpoint_sha256": None,
        "target_task": target_task, "target_task_type": TASKS[target_task],
        "history_sources": [], "history_episodes": 0,
        "parameter_updates": False, "test_history_retention": False,
        "adapter_loaded": False,
        "reference_eval": str(reference),
        "reference_config_sha256": digest(reference / "config.json"),
        "reference_provenance_sha256": digest(reference / "provenance.json"),
        "room_exposure_reference_run": previous.get("source_run"),
        "source_rooms": sorted(source_rooms),
        "test_episodes": len(manifest["episodes"]),
        "episode_seed_rule": "stream.seed + index_within_task",
    }
    return config, manifest, [], provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-eval", required=True,
                        help="One previous transfer result, e.g. CLEAN_RUN/ascent")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device")
    parser.add_argument("--data-dir")
    parser.add_argument("--target-task", choices=TASKS)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    options = {"device": args.device, "data_dir": args.data_dir}
    if args.target_task:
        prepared = prepare(args.reference_eval, args.output_dir,
                           target_task=args.target_task, **options)
        print(json.dumps(prepared[3] if args.prepare_only else evaluate(*prepared), indent=2))
        return
    output = Path(args.output_dir).resolve()
    plans = []
    for task in TASKS:
        prepared = prepare(args.reference_eval, output / task, target_task=task, **options)
        command = [sys.executable, "-u", "-m", "alfworld.continual.memoryless_unseen",
                   "--reference-eval", str(Path(args.reference_eval).resolve()),
                   "--output-dir", str(output / task), "--target-task", task]
        for flag, value in (("--device", args.device), ("--data-dir", args.data_dir)):
            if value is not None:
                command.extend([flag, value])
        plans.append({"task": task, "method": "memoryless", "command": command,
                      "provenance": prepared[3], "status": "queued"})
    if args.prepare_only:
        print(json.dumps(plans, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "launch.json", plans)
    failed = False
    for plan in plans:
        task = plan["task"]
        plan.update(status="running", started_at=datetime.now(timezone.utc).isoformat())
        print(f"Starting {task}: {plan['provenance']['test_episodes']} episodes", flush=True)
        with (output / f"{task}.log").open("w") as log:
            process = subprocess.Popen(plan["command"], stdout=log, stderr=subprocess.STDOUT)
            plan["pid"] = process.pid
            write_json(output / "launch.json", plans)
            code = process.wait()
        plan.update(status="completed" if code == 0 else "failed", returncode=code,
                    finished_at=datetime.now(timezone.utc).isoformat())
        if code == 0:
            plan["summary"] = json.loads((output / task / "summary.json").read_text())
        else:
            failed = True
        write_json(output / "launch.json", plans)
        print(f"Finished {task}: {plan['status']}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
