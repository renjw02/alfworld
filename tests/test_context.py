import copy
import unittest
from alfworld.continual.context import fit_context


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


def serialize(messages, generation):
    text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
    if generation:
        text += "<assistant>"
    return text, Tokenizer().encode(text)


class ContextTests(unittest.TestCase):
    def test_drop_oldest_turns_preserve_system_and_latest_commands(self):
        messages = [{"role": "system", "content": "Follow the rules."},
                    {"role": "user", "content": "old observation" * 50},
                    {"role": "assistant", "content": "old reasoning"},
                    {"role": "user", "content": "recent observation"},
                    {"role": "assistant", "content": "recent reasoning"},
                    {"role": "user", "content": "Admissible commands:\nmove mug 1 to shelf 1"}]
        before = copy.deepcopy(messages)
        expected = messages[:1] + messages[3:]
        maximum = len(serialize(expected, True)[1])
        kept, text, ids, report = fit_context(messages, maximum, serialize, Tokenizer())
        self.assertEqual(kept, expected)
        self.assertEqual(messages, before)
        self.assertEqual(len(ids), maximum)
        self.assertEqual(report["dropped_messages"], 2)
        self.assertNotIn("old observation", text)

    def test_trim_oldest_text_of_oversized_last_message(self):
        suffix = "Admissible commands:\nlook"
        messages = [{"role": "system", "content": "Rules"},
                    {"role": "user", "content": "old " * 1000 + suffix}]
        kept, text, ids, report = fit_context(messages, 100, serialize, Tokenizer())
        self.assertEqual(len(ids), 100)
        self.assertEqual(kept[0], messages[0])
        self.assertTrue(kept[-1]["content"].endswith(suffix))
        self.assertGreater(report["truncated_content_tokens"], 0)
        self.assertEqual(Tokenizer().encode(text), ids)

    def test_no_truncation_at_boundary_and_error_policy(self):
        messages = [{"role": "system", "content": "Rules"}, {"role": "user", "content": "Hi"}]
        maximum = len(serialize(messages, True)[1])
        kept, _, ids, report = fit_context(messages, maximum, serialize, Tokenizer())
        self.assertEqual(kept, messages)
        self.assertEqual(report["original_input_tokens"], len(ids))
        self.assertEqual(report["dropped_messages"], 0)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            fit_context(messages, maximum - 1, serialize, Tokenizer(), "error")
        with self.assertRaisesRegex(ValueError, "System prompt"):
            fit_context(messages, 1, serialize, Tokenizer())


if __name__ == "__main__":
    unittest.main()
