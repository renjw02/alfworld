"""Rebuild the report snapshot from committed attempts, never orphaned traces.

Run from the repository root: .venv/bin/python report/analyze_results.py
Requires local runs/ artifacts; no model loading or GPU access.
"""
import collections
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text())

def ratio(a, b):
    return a / b if b else None

def main():
    paths = sorted([*ROOT.glob('runs/memoryless-task-blocked-admissible-seed*'),
                    *ROOT.glob('runs/ntp-*-admissible-seed*')])
    results = []
    episode_rows = {}
    for root in paths:
        config = read(root / 'config.json')
        commit = read(root / 'commit.json')
        rows = commit['completed']
        episode_rows[root.name] = {r['episode_id']: r for r in rows}
        tasks = collections.defaultdict(lambda: {'episodes': 0, 'successes': 0})
        errors = collections.Counter()
        updates = []
        examples = []
        for row in rows:
            tasks[row['task_type']]['episodes'] += 1
            tasks[row['task_type']]['successes'] += int(row['success'])
            attempt = root / row['attempt_dir']
            episode = read(attempt / 'episode.json')
            steps = episode['steps']
            for step in steps:
                if step['format_error'] is not None:
                    errors['truncated' if step['generation_truncated'] else 'other_format'] += 1
                elif step['invalid_action']:
                    errors['outside_admissible'] += 1
                else:
                    errors['in_admissible'] += 1
            actions = collections.Counter(s['action'] for s in steps if s['action'] is not None)
            if actions:
                action, repeats = actions.most_common(1)[0]
                if repeats >= 30:
                    sample = next(s for s in steps if s['action'] == action)
                    examples.append({'episode': row['position'] + 1, 'task_type': row['task_type'],
                        'action': action, 'repeats': repeats, 'success': row['success'],
                        'outside_admissible_at_sample': sample['invalid_action'],
                        'feedback': sample['feedback'][:350], 'raw_output_excerpt': sample['raw_output'][:500],
                        'source': str((attempt / 'episode.json').relative_to(ROOT))})
            if (attempt / 'update.json').exists():
                u = read(attempt / 'update.json')
                updates.append({'episode': row['position'] + 1, 'success': row['success'],
                                'tokens': u['tokens'], 'targets': u['target_tokens'],
                                'loss': u['loss'], 'steps': u['optimizer_steps']})
        attempts = sum(r['attempts'] for r in rows)
        calls = sum(r['environment_calls'] for r in rows)
        formats = sum(r['format_errors'] for r in rows)
        invalids = sum(r['invalid_actions'] for r in rows)
        assert sum(errors.values()) == attempts
        assert errors['truncated'] + errors['other_format'] == formats
        assert errors['outside_admissible'] == invalids
        invocations = [read(p) for p in sorted((root/'traces').glob('*/invocation.json'))]
        prompts = sorted(set(hashlib.sha256(i['system_message']['content'].encode()).hexdigest()
                             for i in invocations))
        result = {'run': root.name, 'method': config['method'], 'order': config['stream']['order'],
            'seed': config['stream']['seed'], 'episodes': len(rows),
            'successes': sum(r['success'] for r in rows), 'complete': len(rows) == 140,
            'attempts': attempts, 'environment_calls': calls, 'format_errors': formats,
            'format_error_rate': ratio(formats, attempts), 'invalid_actions': invalids,
            'outside_admissible_rate': ratio(invalids, calls), 'error_categories': dict(errors),
            'generation_hours': sum(r['generation_seconds'] for r in rows)/3600,
            'training_minutes': sum(r['training_seconds'] for r in rows)/60,
            'by_task': dict(tasks), 'blocks10': [sum(r['success'] for r in rows[i:i+10])
                                              for i in range(0,len(rows),10)],
            'updates': updates, 'repetition_examples': sorted(examples, key=lambda e: -e['repeats'])[:4],
            'prompt_sha256': prompts, 'config': config,
            'git_commit': read(root/'metadata.json')['git_commit']}
        result['training_by_outcome'] = {
            label: {'episodes': len(us), 'tokens': sum(u['tokens'] for u in us),
                    'optimizer_steps': sum(u['steps'] for u in us),
                    'mean_tokens': ratio(sum(u['tokens'] for u in us), len(us))}
            for success, label in [(True,'success'),(False,'failure')]
            if (us := [u for u in updates if u['success'] == success])}
        results.append(result)
    paired = []
    for seed in (42,43):
        baseline = next(r for r in results if r['method']=='memoryless' and r['seed']==seed)
        ntp = next(r for r in results if r['method']=='history-ntp' and r['order']=='task_blocked' and r['seed']==seed)
        a,b=episode_rows[baseline['run']],episode_rows[ntp['run']]
        assert a.keys()==b.keys()
        assert read(ROOT/'runs'/baseline['run']/'manifest.json') == read(ROOT/'runs'/ntp['run']/'manifest.json')
        assert baseline['prompt_sha256']==ntp['prompt_sha256']
        counts=collections.Counter((bool(a[k]['success']),bool(b[k]['success'])) for k in a)
        paired.append({'seed':seed,'both_success':counts[True,True],
                       'only_memoryless':counts[True,False],'only_ntp':counts[False,True],
                       'both_failure':counts[False,False]})
    snapshot = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                'counting_rule': 'commit.json completed rows and their exact attempt_dir only',
                'runs':results,'paired_outcomes':paired}
    (OUT/'results_snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    print('Validated',len(results),'runs; wrote report/results_snapshot.json')
    for r in results:
        print(r['run'],r['episodes'],r['successes'],r['error_categories'],r['training_by_outcome'])
    print('PAIRED',paired)

if __name__ == '__main__':
    main()
