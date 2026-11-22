"""Reproduce the 39 validation + 161 room-balanced training heat&place stream."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


def sample(root, base, seed=42):
    entries = list(base['episodes'])
    rooms, objects, targets, combinations = (Counter() for _ in range(4))

    def attributes(game):
        data = json.loads(game.with_name('traj_data.json').read_text())
        params = data['pddl_params']
        return data['scene']['floor_plan'], params['object_target'], params['parent_target']

    def record(attrs):
        room, obj, target = attrs
        rooms[room] += 1
        objects[obj] += 1
        targets[target] += 1
        combinations[attrs] += 1

    for entry in entries:
        record(attributes(root / entry['game']))
    candidates = []
    for game in sorted((root / 'json_2.1.1/train').glob('pick_heat_then_place_in_recep-*/*/game.tw-pddl')):
        if 'Sliced' in str(game) or not game.with_name('traj_data.json').exists():
            continue
        if not json.loads(game.read_text()).get('solvable', False):
            continue
        candidates.append((game, attributes(game)))
    random.Random(seed).shuffle(candidates)
    # Balance final room counts first; favor distinct room/object/target tuples,
    # then underrepresented objects and targets. Seeded order breaks ties.
    while len(entries) < 200:
        if not candidates:
            raise ValueError('Not enough usable training games')
        index = min(range(len(candidates)), key=lambda i: (
            rooms[candidates[i][1][0]], combinations[candidates[i][1]],
            objects[candidates[i][1][1]], targets[candidates[i][1][2]]))
        game, attrs = candidates.pop(index)
        record(attrs)
        entries.append({
            'episode_id': game.parent.relative_to(root / 'json_2.1.1').as_posix(),
            'game': game.relative_to(root).as_posix(),
            'task_type': 'pick_heat_then_place_in_recep',
            'sha256': hashlib.sha256(game.read_bytes()).hexdigest(),
            'trajectory_sha256': hashlib.sha256(game.with_name('traj_data.json').read_bytes()).hexdigest(),
            'source_split': 'train',
        })
    random.Random(seed).shuffle(entries)
    manifest = dict(base, split='heat_place_200', seed=seed, order='shuffled', episodes=entries)
    manifest['sampling'] = {
        'method': 'least-filled room, then room/object/target, object, target; seeded tie breaks',
        'base_episodes': len(base['episodes']), 'added_train_episodes': 200 - len(base['episodes']),
        'room_counts': dict(sorted(rooms.items())),
        'object_counts': dict(sorted(objects.items())),
        'target_counts': dict(sorted(targets.items())),
    }
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path.home() / '.cache/alfworld')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('configs/manifests/heat_place_200.seed42.json'))
    args = parser.parse_args()
    base = json.loads(Path('configs/manifests/valid_heat_place.seed42.json').read_text())
    manifest = sample(args.data_dir.expanduser().resolve(), base, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['sampling'], indent=2))
