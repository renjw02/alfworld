"""Experiment configuration, independent of the legacy training configuration."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import yaml


NTP_METHODS = ("history-ntp", "history-ntp-assistant", "history-ntp-assistant-success")
METHODS = ("memoryless", "icl", "dc-cumulative", *NTP_METHODS)


@dataclass
class StreamConfig:
    order: str = "shuffled"
    seed: int = 42
    task_order: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    data_dir: str = "~/.cache/alfworld"
    split: str = "valid_seen"
    manifest: str | None = None
    limit: int | None = None


@dataclass
class ModelConfig:
    path: str = "/data/jiawei/models/Qwen3.5-9B"
    device: str = "cuda:0"
    max_input_tokens: int = 100000
    max_new_tokens: int = 1024
    # Test backend exercises the real runner/environment without loading weights.
    backend: str = "hf"


@dataclass
class TrainingConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    chunk_tokens: int = 2048


@dataclass
class Config:
    method: str = "memoryless"
    action_mode: str = "admissible"
    window_episodes: int = 4
    max_steps: int = 50
    output_dir: str = "runs/experiment"
    cheatsheet_max_new_tokens: int = 2048
    checkpoint_interval: int = 10  # training episodes per model/optimizer checkpoint
    stream: StreamConfig = field(default_factory=StreamConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def is_ntp(self):
        return self.method in NTP_METHODS

    @property
    def assistant_only(self):
        return self.method in NTP_METHODS[1:]

    def validate(self):
        if self.method not in METHODS:
            raise ValueError("Unknown method")
        if self.action_mode not in ("free", "admissible"):
            raise ValueError("Unknown action_mode")
        if self.stream.order not in ("shuffled", "task_blocked"):
            raise ValueError("Unknown stream order")
        if self.stream.split != "valid_seen":
            raise ValueError("This protocol uses valid_seen")
        if sorted(self.stream.task_order) != list(range(1, 7)):
            raise ValueError("task_order must be a permutation of 1..6")
        if self.model.backend not in ("hf", "mock"):
            raise ValueError("Unknown model backend")
        for name, value in {
            "window_episodes": self.window_episodes, "max_steps": self.max_steps,
            "max_input_tokens": self.model.max_input_tokens,
            "max_new_tokens": self.model.max_new_tokens, "rank": self.training.rank,
            "alpha": self.training.alpha,
            "checkpoint_interval": self.checkpoint_interval,
            "cheatsheet_max_new_tokens": self.cheatsheet_max_new_tokens,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.stream.limit is not None and self.stream.limit < 1:
            raise ValueError("stream.limit must be positive")
        if self.training.chunk_tokens < 2:
            raise ValueError("chunk_tokens must be >= 2")
        if not 0 <= self.training.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.training.learning_rate <= 0 or self.training.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.training.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path):
    values = yaml.safe_load(Path(path).read_text()) or {}
    for name, cls in (("stream", StreamConfig), ("model", ModelConfig), ("training", TrainingConfig)):
        values[name] = cls(**values.get(name, {}))
    return Config(**values)
