"""Queue the four remaining unseen tasks using a previous clean transfer's sources.

Run with the same Python environment as alfworld.continual.transfer. Each pair
uses the same target manifest and restarts from the original heat200 state.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from alfworld.continual.logging import write_json
from alfworld.continual.transfer import prepare


TASKS = ("pick_place", "cool_place", "look_at_light", "pick_two_place")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-eval", required=True,
                        help="Previous clean evaluation directory containing icl/ and ascent/")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    reference = Path(args.reference_eval).resolve()
    output = Path(args.output_dir).resolve()
    plans = []
    for task in TASKS:
        manifests = []
        for method in ("icl", "ascent"):
            previous = json.loads((reference / method / "provenance.json").read_text())
            target = output / task / method
            config, manifest, history, provenance = prepare(
                previous["source_run"], target, device=args.device, target_task=task)
            if config.method != method:
                raise ValueError(f"Reference source method mismatch: {method}")
            for key in ("source_checkpoint_sha256", "history_sources"):
                if provenance[key] != previous[key]:
                    raise ValueError(f"Source differs from the clean evaluation: {method}/{key}")
            manifests.append(manifest)
            command = [sys.executable, "-u", "-m", "alfworld.continual.transfer",
                       "--source-run", provenance["source_run"], "--output-dir", str(target),
                       "--device", args.device, "--target-task", task]
            plans.append({"task": task, "method": method, "command": command,
                          "provenance": provenance, "status": "queued"})
        if manifests[0] != manifests[1]:
            raise ValueError(f"ICL/ASCENT target manifest mismatch: {task}")
    if args.prepare_only:
        print(json.dumps(plans, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "launch.json", plans)
    failed = False
    for plan in plans:
        label = f"{plan['task']}/{plan['method']}"
        log = output / f"{plan['task']}-{plan['method']}.log"
        plan.update(status="running", started_at=datetime.now(timezone.utc).isoformat())
        print(f"Starting {label}: {plan['provenance']['test_episodes']} episodes", flush=True)
        with log.open("w") as stream:
            process = subprocess.Popen(plan["command"], stdout=stream, stderr=subprocess.STDOUT)
            plan["pid"] = process.pid
            write_json(output / "launch.json", plans)
            code = process.wait()
        plan.update(status="completed" if code == 0 else "failed", returncode=code,
                    finished_at=datetime.now(timezone.utc).isoformat())
        if code == 0:
            plan["summary"] = json.loads((output / label / "summary.json").read_text())
        else:
            failed = True
        write_json(output / "launch.json", plans)
        print(f"Finished {label}: {plan['status']}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
