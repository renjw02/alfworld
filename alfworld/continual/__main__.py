"""Unified streaming experiment entry point."""
import argparse
import json
import os
# alfworld.info initializes this variable during imports; only honor the caller's value.
DATA_DIR_ENV = os.environ.get("ALFWORLD_DATA")
from .config import METHODS, load_config
from .runner import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/continual.yaml")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--stream-order", choices=["shuffled", "task_blocked"])
    parser.add_argument("--action-mode", choices=["free", "admissible"])
    parser.add_argument("--output-dir")
    parser.add_argument("--manifest")
    parser.add_argument("--data-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--window-episodes", type=int)
    parser.add_argument("--memory-scope", choices=["stream", "task_type"])
    parser.add_argument("--checkpoint-interval", type=int,
                        help="Save training state every N episodes (default: 10), and at completion")
    parser.add_argument("--cheatsheet-max-new-tokens", type=int,
                        help="DC curator output budget (default: 2048 tokens)")
    parser.add_argument("--device")
    parser.add_argument("--devices", nargs="+", help="Inference or ASCENT layer sharding, e.g. cuda:0 cuda:1")
    parser.add_argument("--gpu-memory-gib", nargs="+", type=float,
                        help="Per-device PyTorch allocator caps for --devices")
    parser.add_argument("--backend", choices=["hf", "mock"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    for name in ("method", "action_mode", "output_dir", "max_steps", "window_episodes", "checkpoint_interval",
                 "cheatsheet_max_new_tokens", "memory_scope"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    for arg, field in (("stream_order", "order"), ("manifest", "manifest"), ("data_dir", "data_dir"),
                       ("seed", "seed"), ("limit", "limit")):
        value = getattr(args, arg)
        if value is not None:
            setattr(config.stream, field, value)
    if args.data_dir is None and DATA_DIR_ENV:
        config.stream.data_dir = DATA_DIR_ENV
    for name in ("device", "backend"):
        if getattr(args, name) is not None:
            setattr(config.model, name, getattr(args, name))
    if args.devices is not None:
        config.model.devices = args.devices
        if args.device is None:
            config.model.device = args.devices[0]
    if args.gpu_memory_gib is not None:
        config.model.gpu_memory_gib = args.gpu_memory_gib
    print(json.dumps(run(config, args.resume), indent=2))


if __name__ == "__main__":
    main()
