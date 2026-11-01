"""Serializable records shared by methods, model backends, and the runner."""
from dataclasses import asdict, dataclass, field


@dataclass
class Generation:
    text: str
    prompt: str
    input_ids: list[int]
    output_ids: list[int]
    seconds: float
    truncated: bool = False
    input_truncation: dict | None = None
    input_messages: list[dict] | None = None


@dataclass
class ParsedAction:
    reasoning: str | None
    action: str | None
    error: str | None


@dataclass
class Episode:
    episode_id: str
    task_type: str
    position: int
    messages: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    outcome: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
