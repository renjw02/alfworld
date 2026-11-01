"""Export committed episodes only; failed/incomplete attempts stay available on disk."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from .logging import write_json
from alfworld.agents.environment.alfred_tw_env import TASK_TYPES


def summarize(root, rolling_window=20):
    root = Path(root)
    if rolling_window < 1:
        raise ValueError("rolling_window must be positive")
    rows = json.loads((root / "commit.json").read_text())["completed"]
    successes = 0
    groups = defaultdict(list)
    exported = []
    for i, original in enumerate(rows):
        row = dict(original)
        successes += row["success"]
        window = rows[max(0, i + 1 - rolling_window):i + 1]
        row["cumulative_success_rate"] = successes / (i + 1)
        row["rolling_success_rate"] = sum(r["success"] for r in window) / len(window)
        row["format_error_rate"] = row["format_errors"] / row["attempts"]
        row["invalid_action_rate"] = row["invalid_actions"] / max(1, row["environment_calls"])
        row["mean_reasoning_characters"] = row["reasoning_characters"] / row["attempts"]
        row["mean_output_tokens"] = row["output_tokens"] / row["attempts"]
        groups[row["task_type"]].append(row)
        task_rows = groups[row["task_type"]]
        row["task_episode"] = len(task_rows)
        row["task_cumulative_success_rate"] = sum(r["success"] for r in task_rows) / len(task_rows)
        task_window = task_rows[-rolling_window:]
        row["task_rolling_success_rate"] = sum(r["success"] for r in task_window) / len(task_window)
        exported.append(row)
    if exported:
        with (root / "episodes.csv").open("w") as stream:
            fields = list(dict.fromkeys(key for row in exported for key in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(exported)
    by_task = {}
    for task in TASK_TYPES.values():
        values = groups[task]
        wins = sum(v["success"] for v in values)
        by_task[task] = {"episodes": len(values), "successes": wins,
                         "success_rate": wins / len(values) if values else None}
    with (root / "task_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task_type", "episodes", "successes", "success_rate"])
        writer.writeheader()
        writer.writerows({"task_type": task, **values} for task, values in by_task.items())
    result = {"episodes": len(rows), "success_rate": successes / len(rows) if rows else None,
              "rolling_window": rolling_window,
              "by_task": by_task,
              "backend": json.loads((root / "model.json").read_text())["backend"]}
    write_json(root / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--rolling-window", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_dir, args.rolling_window), indent=2))


if __name__ == "__main__":
    main()
