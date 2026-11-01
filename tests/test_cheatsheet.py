import copy
import json
import unittest

from alfworld.continual.cheatsheet import CURATOR_SYSTEM
from alfworld.continual.config import Config
from alfworld.continual.methods import Method
from alfworld.continual.model import MockBackend
from alfworld.continual.types import Episode, Generation


class CuratorBackend(MockBackend):
    def __init__(self, config):
        super().__init__(config)
        self.requests = []
        self.response = "<cheatsheet>Reusable strategy from episode one.</cheatsheet>"
        self.truncated = False

    def generate(self, messages, *, max_new_tokens=None):
        self.requests.append((copy.deepcopy(messages), max_new_tokens))
        return Generation(self.response, json.dumps(messages), [1, 2], [3, 4, 5],
                          0.25, self.truncated)

    def train_episode(self, messages):
        raise AssertionError("DC must not train weights")


class CheatsheetTests(unittest.TestCase):
    def test_cumulative_context_and_update(self):
        config = Config(method="dc-cumulative", cheatsheet_max_new_tokens=512).validate()
        backend = CuratorBackend(config)
        method = Method(config, backend)
        first = Episode("one", "task", 0, [
            {"role": "user", "content": "TASK ONE and initial observation"},
            {"role": "assistant", "content": "Reasoning: raw attempt\nAction: look"},
            {"role": "user", "content": "Feedback: failure"},
        ], outcome={"success": False})
        context = method.build_context(first)
        self.assertIn("(empty)", context[1]["content"])
        result = method.observe_episode(first)
        self.assertEqual(len(backend.requests), 1)
        request, budget = backend.requests[0]
        self.assertEqual(budget, 512)
        self.assertEqual(request[0]["content"], CURATOR_SYSTEM)
        evidence = json.loads(request[1]["content"].split("observations and feedback):\n")[1])
        self.assertEqual(evidence["messages"], first.messages)
        self.assertIn('"success": false', request[1]["content"])
        self.assertTrue(result["accepted"])
        self.assertEqual(result["generation_seconds"], 0.25)
        self.assertEqual(method.version, 0)
        self.assertEqual(method.cheatsheet_version, 1)
        self.assertEqual(method.history, [])
        second = Episode("two", "task", 1, [{"role": "user", "content": "TASK TWO"}],
                         outcome={"success": True})
        context = method.build_context(second)
        self.assertIn(method.cheatsheet, context[1]["content"])
        self.assertNotIn("TASK ONE", str(context))
        self.assertEqual(context[-1], second.messages[-1])
        context[-1]["content"] = "mutated"
        self.assertEqual(second.messages[-1]["content"], "TASK TWO")
        previous = method.cheatsheet
        backend.response = "<cheatsheet>Old and new strategies.</cheatsheet>"
        method.observe_episode(second)
        self.assertEqual(len(backend.requests), 2)
        self.assertIn(previous, backend.requests[1][0][1]["content"])
        self.assertEqual(method.cheatsheet, "Old and new strategies.")
        restored = Method(config, backend)
        restored.load_state_dict(copy.deepcopy(method.state_dict()))
        self.assertEqual(restored.build_context(second), method.build_context(second))
        self.assertEqual(restored.cheatsheet_version, 2)

    def test_invalid_updates_keep_previous_memory_without_retry(self):
        config = Config(method="dc-cumulative")
        backend = CuratorBackend(config)
        method = Method(config, backend)
        method.cheatsheet = "Keep this knowledge"
        method.cheatsheet_version = 3
        episode = Episode("one", "task", 0, outcome={"success": False})
        for text, truncated in [("plain response", False), ("<cheatsheet>partial", False),
                                ("<cheatsheet> </cheatsheet>", False),
                                ("<cheatsheet>one</cheatsheet><cheatsheet>two</cheatsheet>", False),
                                ("<cheatsheet>complete</cheatsheet>", True)]:
            backend.response, backend.truncated = text, truncated
            before = copy.deepcopy(method.state_dict())
            calls = len(backend.requests)
            update = method.observe_episode(episode)
            self.assertFalse(update["accepted"])
            self.assertEqual(method.state_dict(), before)
            self.assertEqual(len(backend.requests), calls + 1)

    def test_curator_budget_validation(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "cheatsheet_max_new_tokens"):
                Config(method="dc-cumulative", cheatsheet_max_new_tokens=value).validate()
