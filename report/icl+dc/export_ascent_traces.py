"""Export the two local ASCENT runs selected by commit.json, checking raw traces.

Run from any directory with Python 3; plotting then needs only these snapshots.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUNS = ROOT / 'runs/ascent-task_blocked-seed42-20261103T070526Z'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = []
    provenance = {'selection': 'Only completed attempts referenced by commit.json', 'runs': {}}
    for repeat in (1, 2):
        run = RUNS / f'run{repeat}'
        config = json.loads((run / 'config.json').read_text())
        assert config['method'] == 'ascent' and config['memory_scope'] == 'stream'
        assert config['stream']['seed'] == 42 and config['stream']['order'] == 'task_blocked'
        assert config['stream']['split'] == 'valid_seen'
        assert config['action_mode'] == 'admissible' and config['max_steps'] == 50
        assert Path(config['model']['path']).name == 'Qwen3.5-9B'
        commit = json.loads((run / 'commit.json').read_text())
        completed = sorted(commit['completed'], key=lambda r: r['position'])
        assert commit['next_position'] == 140
        assert [r['position'] for r in completed] == list(range(140))
        assert len({r['episode_id'] for r in completed}) == 140
        counts = Counter()
        successes = Counter()
        hashes = {name: sha256(run / name) for name in ('config.json', 'commit.json', 'summary.json')}
        for row in completed:
            path = run / row['attempt_dir'] / 'episode.json'
            episode = json.loads(path.read_text())
            for key in ('position', 'episode_id', 'task_type'):
                assert episode[key] == row[key], (path, key)
            outcome = episode['outcome']
            assert outcome['success'] == row['success']
            assert outcome['score'] == row['score'] == float(outcome['success'])
            assert row['memory_scope'] == 'stream'
            counts[row['task_type']] += 1
            successes[row['task_type']] += int(outcome['success'])
            records.append(dict(run=f'ascent-stream-{repeat}', method='ascent', seed=42,
                                memory_scope='stream', stream_order='task_blocked',
                                task_index=episode['position'] + 1, episode_id=episode['episode_id'],
                                task_type=episode['task_type'], reward=outcome['score']))
            hashes[str(path.relative_to(run))] = sha256(path)
        summary = json.loads((run / 'summary.json').read_text())
        assert summary['episodes'] == 140
        for task, values in summary['by_task'].items():
            assert counts[task] == values['episodes']
            assert successes[task] == values['successes']
        provenance['runs'][f'ascent-stream-{repeat}'] = {
            'path': str(run.relative_to(ROOT)), 'config': config,
            'episodes': len(completed), 'successes': sum(successes.values()),
            'successes_by_task': dict(successes), 'source_sha256': hashes,
        }
    assert sum(r['reward'] for r in records[:140]) == 89
    assert sum(r['reward'] for r in records[140:]) == 98
    target = OUT / 'ascent_trace_instances.csv'
    with target.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    provenance['snapshot_sha256'] = sha256(target)
    (OUT / 'ascent_trace_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Verified and exported 280 committed ASCENT traces; totals: 89, 98.')


if __name__ == '__main__':
    main()
