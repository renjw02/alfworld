import copy
import json
from pathlib import Path
import tempfile
import unittest

from alfworld.continual.config import Config
from alfworld.continual.logging import commit_checkpoint
from alfworld.continual.model import MockBackend
from alfworld.continual.stream import digest
from alfworld.continual.transfer import evaluate


class Env:
    def __init__(self, *args):
        pass

    def reset(self):
        return 'Clean the mug.', {'admissible_commands': ['look']}

    def step(self, action):
        return 'Cleaned.', 1, True, {'admissible_commands': ['look'], 'won': True}

    def close(self):
        pass


class Backend(MockBackend):
    def __init__(self, config):
        super().__init__(config)
        self.contexts = []
        self.loaded = None

    def generate(self, messages, **kwargs):
        self.contexts.append(copy.deepcopy(messages))
        return super().generate(messages, **kwargs)

    def load_state_dict(self, state):
        self.loaded = state

    def distill_episode(self, episode):
        raise AssertionError('Evaluation must not train')


class TransferTests(unittest.TestCase):
    def test_frozen_icl_and_ascent(self):
        for name in ('icl', 'ascent'):
            with self.subTest(method=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = Config(method=name, output_dir=str(root / 'eval'))
                config.model.backend = 'mock'
                config.stream.split = 'valid_unseen'
                backend = Backend(config)
                source = root / 'source'
                source.mkdir()
                commit_checkpoint(source, {'next_position': 200, 'backend': {'adapter': 'final'}}, [])
                pointer = json.loads((source / 'commit.json').read_text())
                checkpoint = source / pointer['checkpoint']
                before = checkpoint.read_bytes()
                provenance = {'protocol': 'frozen_test', 'source_method': name,
                              'source_run': str(source), 'source_checkpoint': pointer['checkpoint'],
                              'source_checkpoint_sha256': digest(checkpoint)}
                history = [[{'role': 'user', 'content': f'heat demo {i}'}] for i in range(3)] if name == 'icl' else []
                original = copy.deepcopy(history)
                entries = [{'episode_id': f'clean-{i}', 'game': f'{i}/game.tw-pddl',
                            'task_type': 'pick_clean_then_place_in_recep',
                            'floor_plan': room, 'room_seen_in_source': i == 0}
                           for i, room in enumerate(('FloorPlan10', 'FloorPlan424'))]
                result = evaluate(config, {'episodes': entries}, history, provenance,
                                  backend_factory=lambda _: backend, env_factory=Env)
                self.assertEqual(result['episodes'], 2)
                self.assertEqual(result['by_source_room_exposure']['unseen']['episodes'], 1)
                self.assertEqual(history, original)
                self.assertEqual(len(backend.contexts), 2)
                second = json.dumps(backend.contexts[1])
                self.assertNotIn('clean-0', second)
                self.assertEqual('heat demo 0' in second, name == 'icl')
                self.assertEqual(backend.loaded, {'adapter': 'final'} if name == 'ascent' else None)
                self.assertEqual(checkpoint.read_bytes(), before)
                with self.assertRaisesRegex(ValueError, 'fresh'):
                    evaluate(config, {'episodes': entries}, history, provenance,
                             backend_factory=lambda _: backend, env_factory=Env)


if __name__ == '__main__':
    unittest.main()
