"""Aggregate paired KL measurements with exact action-span token masks."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from transformers import AutoTokenizer
from probe import MODEL_PATH, OUT, labels, samples, summarize

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
source = {s['id']:s for s in samples()}
raw = [json.loads(line) for line in (OUT/'results.jsonl').open()]
assert len({x['id'] for x in raw}) == len(raw), 'Duplicate samples'
for item in raw:
    s = source[item['id']]
    mask = labels(tokenizer, s['response'])
    assert len(mask) == len(item['tokens'])
    for i, row in enumerate(item['tokens']):
        assert row['token_id'] == s['response'][i]
        row['label'] = mask[i]
        row['first_action'] = i == mask.index('action')
        assert row['kl'] > -1e-5, row
    item['summary'] = summarize(item['tokens'])
    item['action_span_definition'] = 'Tokens overlapping action text, including internal whitespace; excludes Action: and EOS'
(OUT/'normalized_results.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in raw))

def aggregate(items):
    rows = [r for x in items for r in x['tokens']]
    out = dict(turns=len(items), episodes=len({(x['group'],x['position']) for x in items}),
               pooled_token_means=summarize(rows))
    for label, stats in out['pooled_token_means'].items():
        per_turn = [x['summary'][label]['kl'] for x in items if label in x['summary']]
        token_values = [r['kl'] for r in rows if label == 'all' or r['label'] == label or
                        (label=='first_action' and r['first_action'])]
        stats['turn_mean_kl'] = statistics.mean(per_turn)
        stats['turn_median_kl'] = statistics.median(per_turn)
        stats['token_median_kl'] = statistics.median(token_values)
        stats['kl_sum'] = sum(token_values)
        stats['flipped_tokens'] = sum(r['top1_flip'] for r in rows
            if label == 'all' or r['label'] == label or (label=='first_action' and r['first_action']))
    return out

groups = defaultdict(list)
tasks = defaultdict(list)
for item in raw:
    # Two additional initial-episode turns are calibration-only.
    if item['panel_sample'] or 'ascent-icl' in item['group']:
        name = 'ascent_icl' if 'ascent-icl' in item['group'] else 'ascent'
        groups[name].append(item)
        tasks[(name,item['task'])].append(item)
result = dict(measured_turns=len(raw), expected_turns=len(source),
              complete=len(raw)==len(source), groups={k:aggregate(v) for k,v in groups.items()},
              by_task={k+'/'+t:aggregate(v) for (k,t),v in tasks.items()})
calibration = [x for x in raw if 'ascent-icl' not in x['group'] and x['position']==0]
result['calibration'] = dict(turns=len(calibration),
    reconstructed_episode_kl=statistics.mean(x['summary']['all']['kl'] for x in calibration),
    logged_episode_kl=statistics.mean(x['logged_initial_turn_kl'] for x in calibration),
    max_turn_absolute_error=max(abs(x['summary']['all']['kl']-x['logged_initial_turn_kl']) for x in calibration))
result['null_control'] = json.loads((OUT/'null_control.json').read_text())
result['bounded_recurrence_validation'] = json.loads((OUT/'recurrence_validation.json').read_text())
validation_path = OUT/'full_model_validation.json'
result['full_model_validation'] = (json.loads(validation_path.read_text()) if validation_path.exists()
                                   else {'status': 'Optional original-versus-bounded comparison not recorded'})
result['peak_cuda_allocated_gib'] = max(x['peak_allocated_gib'] for x in raw)
(OUT/'summary.json').write_text(json.dumps(result,indent=2))
with (OUT/'turn_metrics.csv').open('w') as f:
    names=['group','task','position','turn_index','action','prompt_tokens','privileged_tokens',
           'action_tokens','action_kl','reasoning_kl','first_action_kl','action_top1_flip','action_logp_gain']
    writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
    for x in raw:
        d={k:x[k] for k in names if k in x}; a=x['summary']['action']
        d.update(action_tokens=a['tokens'],action_kl=a['kl'],reasoning_kl=x['summary'].get('reasoning',{}).get('kl'),
                 first_action_kl=x['summary']['first_action']['kl'],action_top1_flip=a['top1_flip'],
                 action_logp_gain=a['logp_gain'])
        writer.writerow(d)
lines=['# ASCENT teacher-to-base paired KL diagnostic','',
       'Frozen Qwen3.5-9B, BF16 forward / FP32 full-vocabulary KL, temperature 1. '
       'Both forwards use identical saved student prefix and response tokens. Only the teacher gets '
       'the saved privileged prefix. No generation, truncation, LoRA, or optimizer updates.', '',
       'Action mask includes the action text and its internal whitespace; excludes the Action: marker and EOS. '
       'All table KL values are pooled means in nats/token. The sample is deterministic and small: '
       'first successful committed episode per task, first/middle/last turns for ASCENT; '
       'first turn only for ASCENT+ICL. Two extra ASCENT turns are used only for calibration.', '',
       '| Group / task | Turns | Action tokens | Action KL | Reasoning KL | First-action KL | Action top-1 flips |',
       '|---|---:|---:|---:|---:|---:|---:|']
for name,v in list(result['groups'].items())+list(result['by_task'].items()):
    z=v['pooled_token_means'];a=z['action']
    lines.append(f"| {name} | {v['turns']} | {a['tokens']} | {a['kl']:.8g} | {z['reasoning']['kl']:.8g} | "
                 f"{z['first_action']['kl']:.8g} | {a['flipped_tokens']}/{a['tokens']} |")
lines += ['', '## Validation', '', f"- Samples: {len(raw)}/{len(source)}.",
          f"- Initial episode reconstructed KL: {result['calibration']['reconstructed_episode_kl']:.12g}; "
          f"logged: {result['calibration']['logged_episode_kl']:.12g}.",
          f"- Maximum per-turn calibration error: {result['calibration']['max_turn_absolute_error']:.4g}.",
          f"- Null control: {result['null_control']}.",
          f"- Memory-bounded recurrence: {result['bounded_recurrence_validation']}.",
          f"- Real-model original-versus-bounded validation: {result['full_model_validation']}.",
          f"- Peak CUDA allocator usage: {result['peak_cuda_allocated_gib']:.3f} GiB (cap 4 GiB).", '',
          '## Limits', '',
          'These are conditional next-token divergences on successful student trajectories. '
          'The exact student reasoning before Action: is held fixed and may already specify the action. '
          'Small action KL therefore does not rule out policy changes mediated through different reasoning. '
          'The privileged record contains the successful answers themselves, so large reasoning KL can reflect copying. '
          'No claim about held-out success improvement, full command-distribution KL, or post-distillation student quality is made.']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
