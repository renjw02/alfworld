import copy
import json
from pathlib import Path
import tempfile
import unittest

from alfworld.continual.config import Config
from alfworld.continual.logging import commit_checkpoint, run_lock
from alfworld.continual.model import MockBackend
from alfworld.continual.stream import digest
from alfworld.continual.transfer import TARGET_TASKS, evaluate, prepare


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
    def test_prepare_targets_are_disjoint_and_preserve_source_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            config = Config(method='icl')
            config.stream.split = 'heat_place_200'
            config.stream.data_dir = str(root / 'data')
            (source / 'config.json').write_text(json.dumps(config.to_dict()))
            heat = 'pick_heat_then_place_in_recep'
            entries = []
            for task in (*TARGET_TASKS.values(), heat):
                game = root / 'data/json_2.1.1/valid_unseen' / task / 'trial/game.tw-pddl'
                game.parent.mkdir(parents=True)
                game.write_text(json.dumps({'solvable': True}))
                trajectory = game.with_name('traj_data.json')
                trajectory.write_text(json.dumps({'task_type': task,
                                                  'scene': {'floor_plan': 'FloorPlan10'}}))
                entries.append({'game': str(game.relative_to(root / 'data')),
                                'trajectory_sha256': digest(trajectory)})
            source_entries = [{**entries[-1], 'episode_id': str(i)} for i in range(200)]
            (source / 'manifest.json').write_text(json.dumps({'episodes': source_entries}))
            completed = []
            for i in range(200):
                completed.append({'episode_id': str(i), 'attempt_dir': f'episodes/{i}',
                                  'task_type': heat})
                episode = source / f'episodes/{i}/episode.json'
                episode.parent.mkdir(parents=True)
                episode.write_text(json.dumps({'episode_id': str(i), 'messages': [
                    {'role': 'user', 'content': f'heat demo {i}'}]}))
            commit_checkpoint(source, {'next_position': 200}, completed, model_checkpoint=False)
            targets = []
            for task, task_type in TARGET_TASKS.items():
                prepared = prepare(source, root / task, target_task=task)
                _, manifest, history, provenance = prepared
                self.assertEqual(len(manifest['episodes']), 1)
                self.assertEqual(manifest['episodes'][0]['task_type'], task_type)
                targets.append(manifest['episodes'][0]['episode_id'])
                self.assertEqual(len(history), 3)
                self.assertEqual(history[0][0]['content'], 'heat demo 197')
                self.assertEqual(provenance['target_task'], task)
                self.assertFalse(provenance['parameter_updates'])
                self.assertFalse(provenance['test_history_retention'])
            self.assertEqual(len(set(targets)), 5)
            default = prepare(source, root / 'default')
            self.assertEqual(default[1], prepare(source, root / 'explicit', target_task='clean_place')[1])
            self.assertEqual(default[3]['protocol'], 'frozen_heat200_to_clean_unseen_v1')
            with self.assertRaisesRegex(ValueError, 'Unknown target task'):
                prepare(source, root / 'invalid', target_task='heat_place')

            # The future source room and future demonstrations must not leak into prefix evaluation.
            late = root / 'data/json_2.1.1/train/late/traj_data.json'
            late.parent.mkdir(parents=True)
            late.write_text(json.dumps({'task_type': heat, 'scene': {'floor_plan': 'FloorPlan99'}}))
            for entry in source_entries[50:]:
                entry.update(game=str(late.with_name('game.tw-pddl').relative_to(root / 'data')),
                             trajectory_sha256=digest(late))
            (source / 'manifest.json').write_text(json.dumps({'episodes': source_entries}))
            for position in (50, 100, 150):
                _, _, history, provenance = prepare(source, root / f'icl-{position}', source_position=position)
                self.assertEqual([h[0]['content'] for h in history],
                                 [f'heat demo {i}' for i in range(position - 3, position)])
                self.assertIsNone(provenance['source_checkpoint'])
                self.assertEqual(provenance['source_position'], position)
                self.assertEqual('FloorPlan99' in provenance['source_rooms'], position > 50)

            config.method = 'ascent'
            (source / 'config.json').write_text(json.dumps(config.to_dict()))
            for position in (50, 100, 150, 200):
                commit_checkpoint(source, {'next_position': position, 'backend': {'adapter': position}},
                                  completed[:position], keep_positions=[50, 100, 150, 200])
            prepared = prepare(source, root / 'ascent-50', source_position=50)
            self.assertEqual(prepared[3]['source_checkpoint'], 'milestones/episode-0050.pkl')
            self.assertEqual(prepared[3]['source_rooms'], ['FloorPlan10'])
            # A newer rolling pointer must not change which retained checkpoint is loaded.
            commit_checkpoint(source, {'next_position': 200, 'backend': {'adapter': 'other'}}, completed)
            backend = Backend(prepared[0])
            with run_lock(source, shared=True):
                result = evaluate(*prepared, backend_factory=lambda _: backend, env_factory=Env)
            self.assertEqual(backend.loaded, {'adapter': 50})
            self.assertEqual(result['source_position'], 50)
            with self.assertRaisesRegex(ValueError, 'No retained ASCENT checkpoint'):
                prepare(source, root / 'missing', source_position=75)
            for position in (0, 201, True):
                with self.assertRaisesRegex(ValueError, 'source_position'):
                    prepare(source, root / 'invalid-position', source_position=position)

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
