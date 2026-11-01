"""Append-only attempts and atomic episode-boundary commits."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import pickle
import uuid


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def run_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is using {root}") from exc
        yield


class AttemptLog:
    def __init__(self, root, position):
        self.path = root / "episodes" / f"{position:04d}" / uuid.uuid4().hex
        self.path.mkdir(parents=True)
        self.stream = (self.path / "steps.jsonl").open("w")

    def append(self, record):
        self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


def commit_checkpoint(root, state, completed):
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    previous = json.loads((root / "commit.json").read_text()) if (root / "commit.json").exists() else None
    checkpoint = checkpoint_dir / (uuid.uuid4().hex + ".pkl")
    with checkpoint.open("wb") as stream:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    # Only this atomic pointer makes the checkpoint and its episode results visible.
    write_json(root / "commit.json", {"schema_version": 1, "next_position": state["next_position"],
                                     "checkpoint": str(checkpoint.relative_to(root)), "completed": completed})
    if previous:
        (root / previous["checkpoint"]).unlink(missing_ok=True)


def load_checkpoint(root):
    committed = json.loads((root / "commit.json").read_text())
    # Checkpoints belong to this local run; never load an untrusted pickle.
    with (root / committed["checkpoint"]).open("rb") as stream:
        state = pickle.load(stream)
    return state, committed["completed"]
