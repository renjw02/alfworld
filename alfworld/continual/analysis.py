"""Export committed episodes only; failed/incomplete attempts stay available on disk."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from .logging import write_json


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
        exported.append(row)
    if exported:
        with (root / "episodes.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
            writer.writeheader()
            writer.writerows(exported)
    result = {"episodes": len(rows), "success_rate": successes / len(rows) if rows else None,
              "rolling_window": rolling_window,
              "by_task": {task: {"episodes": len(values), "success_rate": sum(v["success"] for v in values) / len(values)}
                          for task, values in groups.items()},
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
