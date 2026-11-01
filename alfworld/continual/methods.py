"""Methods only own cross-episode state and adaptation policy."""
import copy
from .prompting import system_message


class Method:
    def __init__(self, config, backend):
        self.config = config
        self.backend = backend
        self.history = []
        self.version = 0

    def build_context(self, episode):
        messages = [system_message(self.config.action_mode)]
        for previous in self.history:
            messages.extend(previous)
        messages.extend(episode.messages)
        return copy.deepcopy(messages)

    def observe_episode(self, episode):
        if self.config.method == "icl":
            keep = self.config.window_episodes - 1
            self.history = (self.history + [copy.deepcopy(episode.messages)])[-keep:] if keep else []
        elif self.config.is_ntp:
            if (self.config.method == "history-ntp-assistant-success"
                    and not episode.outcome["success"]):
                return {"skipped": True, "skip_reason": "unsuccessful_episode",
                        "optimizer_steps": 0, "seconds": 0.0, "loss": None,
                        "version_before": self.version, "version_after": self.version}
            messages = [system_message(self.config.action_mode), *episode.messages]
            result = self.backend.train_episode(messages)
            result["version_before"] = self.version
            if not result.get("skipped", False):
                self.version += 1
            result["version_after"] = self.version
            return result
        return None

    def state_dict(self):
        return {"history": self.history, "version": self.version}

    def load_state_dict(self, state):
        self.history = state["history"]
        self.version = state["version"]
