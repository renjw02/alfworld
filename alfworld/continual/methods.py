"""Methods only own cross-episode state and adaptation policy."""
import copy
from .cheatsheet import curator_messages, extract_cheatsheet, reference_message
from .prompting import system_message


class Method:
    def __init__(self, config, backend):
        self.config = config
        self.backend = backend
        self.history = []
        self.version = 0
        self.cheatsheet = ""
        self.cheatsheet_version = 0

    def build_context(self, episode):
        messages = [system_message(self.config.action_mode)]
        if self.config.method == "dc-cumulative":
            messages.append(reference_message(self.cheatsheet))
        for previous in self.history:
            messages.extend(previous)
        messages.extend(episode.messages)
        return copy.deepcopy(messages)

    def observe_episode(self, episode):
        if self.config.method == "icl":
            keep = self.config.window_episodes - 1
            self.history = (self.history + [copy.deepcopy(episode.messages)])[-keep:] if keep else []
        elif self.config.method == "dc-cumulative":
            before = self.cheatsheet
            version_before = self.cheatsheet_version
            messages = curator_messages(before, episode, self.config.cheatsheet_max_new_tokens)
            generation = self.backend.generate(messages, max_new_tokens=self.config.cheatsheet_max_new_tokens)
            replacement, error = extract_cheatsheet(generation)
            if replacement is not None:
                self.cheatsheet = replacement
                self.cheatsheet_version += 1
            return {"kind": "dc-cumulative", "accepted": error is None, "skip_reason": error,
                    "cheatsheet_before": before, "cheatsheet_after": self.cheatsheet,
                    "cheatsheet_version_before": version_before,
                    "cheatsheet_version_after": self.cheatsheet_version,
                    "messages": messages, "prompt": generation.prompt, "raw_output": generation.text,
                    "input_token_ids": generation.input_ids, "output_token_ids": generation.output_ids,
                    "input_tokens": len(generation.input_ids), "output_tokens": len(generation.output_ids),
                    "generation_seconds": generation.seconds, "generation_truncated": generation.truncated,
                    "optimizer_steps": 0}
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
        state = {"history": self.history, "version": self.version}
        if self.config.method == "dc-cumulative":
            state.update(cheatsheet=self.cheatsheet, cheatsheet_version=self.cheatsheet_version)
        return state

    def load_state_dict(self, state):
        self.history = state["history"]
        self.version = state["version"]
        self.cheatsheet = state.get("cheatsheet", "")
        self.cheatsheet_version = state.get("cheatsheet_version", 0)
