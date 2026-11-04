"""Audit full five-interaction examples and optionally run a few real SFT steps.

No rollout or existing run is changed. Training starts from the base model and
only writes a pilot report; it does not overwrite experiment checkpoints.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alfworld.continual.config import Config
from alfworld.continual.model import HFBackend
from alfworld.continual.sft import build_samples, rollout_context
from alfworld.continual.types import Episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-steps', type=int, default=0)
    parser.add_argument('--max-seq-tokens', type=int, default=2048)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--memory-gib', type=float, default=30)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    config = Config(method='success-only-sft')
    config.model.device = args.device
    config.sft.max_seq_tokens = args.max_seq_tokens
    tokenizer = AutoTokenizer.from_pretrained(config.model.path, local_files_only=True)
    renderer = HFBackend.__new__(HFBackend)
    renderer.tokenizer = tokenizer
    lengths, pool, reasons = [], [], Counter()
    successful = 0
    # Measure full lengths before applying a budget; no text/token truncation.
    config.sft.max_seq_tokens = 10**9
    for folder in args.runs:
        root = Path(folder)
        for row in json.loads((root/'commit.json').read_text())['completed']:
            if not row['success']:
                continue
            successful += 1
            e = Episode(**json.loads((root/row['attempt_dir']/'episode.json').read_text()))
            samples, rejected = build_samples(config, e, renderer.serialize, tokenizer)
            reasons.update(r['reason'] for r in rejected)
            for sample in samples:
                index = next(i for i,s in enumerate(e.steps) if s['step'] == sample['step'])
                prefix = Episode(e.episode_id, e.task_type, e.position,
                                 messages=e.messages, steps=e.steps[:index])
                assert sample['messages'] == rollout_context(config, prefix), 'Train/rollout prefix mismatch'
                length = len(sample['input_ids'])
                lengths.append(length)
                # Keep the longest fitting examples across all episodes, not just early easy turns.
                if length <= args.max_seq_tokens:
                    pool.append((sample, str(root/row['attempt_dir']/'episode.json')))
                    pool.sort(key=lambda item: len(item[0]['input_ids']))
                    if len(pool) > 200:
                        pool.pop(0)
    lengths.sort()
    if not lengths:
        raise RuntimeError('No eligible samples')
    report = {'source_runs': args.runs, 'successful_episodes': successful,
              'eligible_samples': len(lengths), 'rejected_interactions': dict(reasons),
              'length_percentiles': {str(q): lengths[int((len(lengths)-1)*q/100)] for q in (0,50,90,95,99,100)},
              'budget_overflows': {str(b): sum(n>b for n in lengths) for b in (2048,4096,8192,16384)},
              'policy': 'skip whole overlength sample; keep all five recent interactions and target EOS',
              'prefix_checks': len(lengths), 'trained': False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    if args.train_steps:
        import torch
        if len(pool) < args.train_steps:
            raise RuntimeError('Not enough samples fit the requested budget')
        torch.manual_seed(42)
        torch.cuda.set_device(args.device)
        free, total = torch.cuda.mem_get_info(args.device)
        cap = min(args.memory_gib*1024**3, free-5*1024**3)
        if cap < 24*1024**3:
            raise RuntimeError('Insufficient free GPU memory for an isolated pilot')
        torch.cuda.set_per_process_memory_fraction(cap/total, args.device)
        config.sft.max_seq_tokens = args.max_seq_tokens
        backend = HFBackend(config)
        # Include the longest fitting examples, rather than only easy first turns.
        chosen = sorted(pool, key=lambda item: len(item[0]['input_ids']), reverse=True)[:args.train_steps]
        before = {n: p.detach().cpu().clone() for n,p in backend.model.named_parameters() if p.requires_grad}
        result = backend.trainer.train_samples([s for s,_ in chosen])
        changed = [n for n,p in backend.model.named_parameters() if p.requires_grad and not torch.equal(before[n], p.detach().cpu())]
        if not changed:
            raise RuntimeError('Pilot did not change trainable parameters')
        assert all('lora_' in n for n in changed)
        report.update(trained=True, config=config.to_dict(), training=result,
                      changed_lora_tensors=len(changed), peak_allocated_gib=torch.cuda.max_memory_allocated(args.device)/1024**3,
                      sources=[{'source':src,'step':s['step'],'tokens':len(s['input_ids']),
                                'history_steps':s['history_steps'],'target_tokens':s['target_tokens']} for s,src in chosen])
        output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ('training','sources','peak_allocated_gib')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
