import json
from pathlib import Path
import tempfile
import unittest

from alfworld.continual.config import Config, load_config
from alfworld.continual.model import MockBackend
from alfworld.continual.prompting import parse_action_only, system_message
from alfworld.continual.runner import run
from alfworld.continual.types import Generation
from tests.test_continual import FakeEnv


class ActionOnlyTests(unittest.TestCase):
    def test_format_and_preset(self):
        self.assertEqual(parse_action_only('Action: look').action, 'look')
        self.assertIsNone(parse_action_only('Action: look').reasoning)
        for text in ('look', 'Action:', 'Reasoning: inspect\nAction: look',
                     'Action: look\nAction: inventory', 'Action: look\nExtra text'):
            self.assertIsNotNone(parse_action_only(text).error)
        self.assertIsNotNone(parse_action_only('Action: look', truncated=True).error)
        self.assertNotIn('Reasoning:', system_message('admissible', 'action_only')['content'])
        self.assertIn('Reasoning:', system_message('admissible')['content'])
        c = load_config('configs/continual-heat-place-200-icl-action-only.yaml').validate()
        self.assertEqual(c.window_episodes, 4)
        self.assertEqual(c.output_format, 'action_only')

    def test_runner_history_resume_and_format_guard(self):
        observed = []

        class Backend(MockBackend):
            def generate(self, messages, **kwargs):
                observed.append(messages)
                text = 'Action: look'
                prompt = json.dumps(messages)
                return Generation(text, prompt, list(prompt.encode()), list(text.encode()), 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(6):
                path = root / 'json_2.1.1/valid_seen/pick_and_place_simple' / str(i)
                path.mkdir(parents=True)
                (path / 'game.tw-pddl').write_text(json.dumps({'solvable': True}))
                (path / 'traj_data.json').write_text(json.dumps({'task_type': 'pick_and_place_simple'}))
            c = Config(method='icl', output_format='action_only')
            c.model.backend = 'mock'
            c.stream.data_dir = tmp
            c.output_dir = str(root / 'run')
            result = run(c, backend_factory=Backend, env_factory=FakeEnv)
            self.assertEqual(result['episodes'], 6)
            self.assertEqual(result['success_rate'], 1)
            self.assertEqual(sum(m['content'].startswith('BEGIN EPISODE ') for m in observed[-1]), 4)
            self.assertTrue(all(m['content'] == 'Action: look' for m in observed[-1] if m['role'] == 'assistant'))
            rows = json.loads((root / 'run/commit.json').read_text())['completed']
            self.assertTrue(all(r['format_errors'] == 0 and r['reasoning_characters'] == 0 for r in rows))
            run(c, resume=True, backend_factory=Backend, env_factory=FakeEnv)
            c.output_format = 'react'
            with self.assertRaisesRegex(ValueError, 'Resume configuration differs'):
                run(c, resume=True, backend_factory=Backend, env_factory=FakeEnv)
