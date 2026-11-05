"""Experiment configuration, independent of the legacy training configuration."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import yaml


NTP_METHODS = ("history-ntp", "history-ntp-assistant", "history-ntp-assistant-success")
ASCENT_METHODS = ("ascent", "ascent-icl")
METHODS = ("memoryless", "icl", "dc-cumulative", *NTP_METHODS, *ASCENT_METHODS, "tt-opsd", "success-only-sft")


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
    devices: list[str] = field(default_factory=list)  # inference or distillation layer sharding
    gpu_memory_gib: list[float] = field(default_factory=list)  # allocator caps per device
    max_input_tokens: int = 65536
    context_overflow: str = "truncate_oldest"
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
class AscentConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optimizer_steps: int = 2
    logits_chunk_tokens: int = 32


@dataclass
class SFTConfig:
    max_seq_tokens: int = 2048
    history_interactions: int = 5
    batch_size: int = 2
    gradient_accumulation_steps: int = 4


@dataclass
class Config:
    method: str = "memoryless"
    action_mode: str = "admissible"
    window_episodes: int = 4
    icl_success_only: bool = False  # only valid interactions from successful episodes
    max_steps: int = 50
    output_dir: str = "runs/experiment"
    cheatsheet_max_new_tokens: int = 2048
    checkpoint_interval: int = 10  # training episodes per model/optimizer checkpoint
    memory_scope: str = "stream"  # stream | task_type (reset on task-block boundaries)
    stream: StreamConfig = field(default_factory=StreamConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ascent: AscentConfig = field(default_factory=AscentConfig)  # shared by ASCENT and TT-OPSD
    sft: SFTConfig = field(default_factory=SFTConfig)

    @property
    def trains_parameters(self):
        return self.is_ntp or self.is_distillation or self.method == "success-only-sft"

    @property
    def is_tt_opsd(self):
        return self.method == "tt-opsd"

    @property
    def is_distillation(self):
        return self.is_ascent or self.is_tt_opsd

    @property
    def is_ascent(self):
        return self.method in ASCENT_METHODS

    @property
    def uses_icl(self):
        return self.method in ("icl", "ascent-icl")

    @property
    def is_ntp(self):
        return self.method in NTP_METHODS

    @property
    def assistant_only(self):
        return self.method in NTP_METHODS[1:]

    def validate(self):
        if self.method not in METHODS:
            raise ValueError("Unknown method")
        if not isinstance(self.icl_success_only, bool):
            raise ValueError("icl_success_only must be a boolean")
        if self.action_mode not in ("free", "admissible"):
            raise ValueError("Unknown action_mode")
        if self.stream.order not in ("shuffled", "task_blocked"):
            raise ValueError("Unknown stream order")
        if self.memory_scope not in ("stream", "task_type"):
            raise ValueError("Unknown memory_scope")
        if self.memory_scope == "task_type" and self.stream.order != "task_blocked":
            raise ValueError("task_type memory_scope requires task_blocked order")
        if self.stream.split != "valid_seen":
            raise ValueError("This protocol uses valid_seen")
        if sorted(self.stream.task_order) != list(range(1, 7)):
            raise ValueError("task_order must be a permutation of 1..6")
        if self.model.backend not in ("hf", "mock"):
            raise ValueError("Unknown model backend")
        if self.model.context_overflow not in ("truncate_oldest", "error"):
            raise ValueError("Unknown model.context_overflow policy")
        if self.model.devices:
            if self.trains_parameters and not self.is_distillation:
                raise ValueError("Multi-GPU layer sharding supports inference methods, ASCENT and TT-OPSD only")
            if (len(self.model.devices) < 2 or len(set(self.model.devices)) != len(self.model.devices)
                    or any(not isinstance(d, str) or not d.startswith("cuda:")
                           or not d[5:].isdigit() for d in self.model.devices)):
                raise ValueError("model.devices must contain at least two distinct cuda:N devices")
            if self.model.device != self.model.devices[0]:
                raise ValueError("model.device must be the first entry in model.devices")
        if self.model.gpu_memory_gib:
            import math
            if (len(self.model.gpu_memory_gib) != len(self.model.devices)
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) or v <= 0 for v in self.model.gpu_memory_gib)):
                raise ValueError("gpu_memory_gib must specify one positive finite cap per device")
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
        for name, minimum in (("max_seq_tokens", 2), ("history_interactions", 0),
                              ("batch_size", 1), ("gradient_accumulation_steps", 1)):
            value = getattr(self.sft, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"sft.{name} must be an integer >= {minimum}")
        if self.is_distillation:
            a = self.ascent
            for name in ("rank", "alpha", "optimizer_steps", "logits_chunk_tokens"):
                value = getattr(a, name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"ascent.{name} must be a positive integer")
            import math
            for name in ("learning_rate", "max_grad_norm", "weight_decay", "dropout"):
                if not math.isfinite(getattr(a, name)):
                    raise ValueError(f"ascent.{name} must be finite")
            if a.learning_rate <= 0 or a.max_grad_norm <= 0 or a.weight_decay < 0:
                raise ValueError("Invalid ASCENT/TT-OPSD optimizer configuration")
            if a.dropout != 0:
                raise ValueError("ASCENT/TT-OPSD requires dropout=0 for deterministic teacher/student matching")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path):
    values = yaml.safe_load(Path(path).read_text()) or {}
    for name, cls in (("stream", StreamConfig), ("model", ModelConfig), ("training", TrainingConfig), ("ascent", AscentConfig), ("sft", SFTConfig)):
        values[name] = cls(**values.get(name, {}))
    return Config(**values)
