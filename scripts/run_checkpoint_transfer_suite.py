"""Evaluate heat200 source positions with ICL and ASCENT on five unseen types."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading

from alfworld.continual.logging import write_json
from alfworld.continual.transfer import TARGET_TASKS, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icl-source", required=True)
    parser.add_argument("--ascent-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positions", nargs="+", type=int, default=[50, 100, 150])
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.positions)) != len(args.positions) or len(set(args.devices)) != len(args.devices):
        parser.error("Positions and devices must not contain duplicates")
    output = Path(args.output_dir).resolve()
    sources = {"icl": Path(args.icl_source).resolve(), "ascent": Path(args.ascent_source).resolve()}
    plans, manifests = [], {}
    for position in args.positions:
        for method, source in sources.items():
            for task in TARGET_TASKS:
                relative = f"checkpoint-{position:04d}/{method}/{task}"
                config, manifest, history, provenance = prepare(
                    source, output / relative, source_position=position,
                    target_task=task, device=args.devices[0])
                if config.method != method:
                    raise ValueError(f"Wrong method for source: {source}")
                # Room-exposure flags legitimately depend on the source prefix.
                games = [{k: v for k, v in entry.items() if k != "room_seen_in_source"}
                         for entry in manifest["episodes"]]
                if task in manifests and games != manifests[task]:
                    raise ValueError(f"Target episode ordering differs across sources: {task}")
                manifests[task] = games
                command = [sys.executable, "-u", "-m", "alfworld.continual.transfer",
                           "--source-run", str(source), "--source-position", str(position),
                           "--output-dir", str(output / relative), "--target-task", task]
                plans.append({"position": position, "method": method, "task": task,
                              "relative_output": relative, "command": command,
                              "provenance": provenance, "status": "queued"})
    if args.prepare_only:
        print(json.dumps(plans, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "launch.json", plans)
    jobs = queue.Queue()
    for plan in plans:
        jobs.put(plan)
    lock = threading.Lock()

    def worker(device):
        while True:
            try:
                plan = jobs.get_nowait()
            except queue.Empty:
                return
            command = [*plan["command"], "--device", device]
            label = plan["relative_output"]
            log = output / (label.replace("/", "-") + ".log")
            try:
                with log.open("w") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                    with lock:
                        plan.update(status="running", device=device, command=command, pid=process.pid,
                                    started_at=datetime.now(timezone.utc).isoformat())
                        write_json(output / "launch.json", plans)
                        print(f"Starting {label} on {device}", flush=True)
                    code = process.wait()
                with lock:
                    plan.update(status="completed" if code == 0 else "failed", returncode=code,
                                finished_at=datetime.now(timezone.utc).isoformat())
                    if code == 0:
                        plan["summary"] = json.loads((output / label / "summary.json").read_text())
                    write_json(output / "launch.json", plans)
                    print(f"Finished {label}: {plan['status']}", flush=True)
            except Exception as exc:
                with lock:
                    plan.update(status="failed", error=str(exc),
                                finished_at=datetime.now(timezone.utc).isoformat())
                    write_json(output / "launch.json", plans)
                    print(f"Failed {label}: {exc}", flush=True)

    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        futures = [pool.submit(worker, device) for device in args.devices]
        for future in futures:
            future.result()
    sys.exit(0 if all(p["status"] == "completed" for p in plans) else 1)


if __name__ == "__main__":
    main()
