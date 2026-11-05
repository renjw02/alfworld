"""Paired, frozen-base ASCENT diagnostic. No environment or optimizer updates."""
import argparse
import hashlib
import os
import json
import re
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('ALFWORLD_KL_OUTPUT', ROOT / 'runs/reports/teacher-base-kl-20261105'))
MODEL_PATH = Path(os.environ.get('ALFWORLD_MODEL_PATH', ROOT.parent / 'models/Qwen3.5-9B'))

def read(path):
    return json.loads(path.read_text())

def samples():
    main = 'ascent-task_type-task_blocked-seed42-20261104T025254Z/run1'
    icl = 'ascent-icl-task_blocked-seed42-20261104T065406Z/run1'
    result = []
    for group in (main, icl):
        run = ROOT / 'runs' / group
        seen = set()
        for episode in read(run / 'commit.json')['completed']:
            task = episode['task_type']
            if not episode['success'] or task in seen:
                continue
            seen.add(task)
            attempt = run / episode['attempt_dir']
            update = read(attempt / 'update.json')
            steps = [json.loads(line) for line in (attempt / 'steps.jsonl').open()]
            steps = [s for s in steps if 'input_token_ids' in s]
            indices = sorted({0, len(steps)//2, len(steps)-1}) if group == main else [0]
            # Every turn of the initial episode reproduces the logged initial loss.
            if group == main and episode['position'] == 0:
                indices = list(range(len(steps)))
            for index in indices:
                step = steps[index]
                result.append(dict(id=f'{group}:{episode["position"]}:{index}',
                    group=group, task=task, position=episode['position'],
                    turn_index=index, turn_count=len(steps), attempt=str(attempt),
                    panel_sample=index in {0, len(steps)//2, len(steps)-1},
                    prompt=step['input_token_ids'], response=step['output_token_ids'],
                    privileged=update['teacher_token_ids'], action=step['action'],
                    format_error=step['format_error'], raw_output=step['raw_output'],
                    logged_initial_turn_kl=next((x['kl'] for x in update['updates'][0]['turns']
                                                if x['turn'] == step['step']), None),
                    model_version=step['model_version']))
    return result

def labels(tokenizer, ids):
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    encoded = tokenizer(decoded, add_special_tokens=False, return_offsets_mapping=True)
    if encoded['input_ids'] != ids:
        raise ValueError('Tokenizer round trip differs; refusing ambiguous token-span mapping')
    action = re.search(r'^Action:[ \t]*([^\r\n]+)', decoded, re.M)
    reasoning = re.search(r'^Reasoning:[ \t]*(.*?)(?=\nAction:)', decoded, re.S | re.M)
    if action is None:
        raise ValueError('No unambiguous Action field')
    spans = {'action': action.span(1)}
    if reasoning:
        spans['reasoning'] = reasoning.span(1)
    result = []
    for token, (start, end) in zip(ids, encoded['offset_mapping']):
        label = 'eos' if token == tokenizer.eos_token_id else 'format'
        if label != 'eos':
            for name, (a, b) in spans.items():
                lo, hi = max(start, a), min(end, b)
                if hi > lo:
                    label = name
                    break
        result.append(label)
    if 'action' not in result:
        raise ValueError('Empty action token mask')
    return result

def summarize(rows):
    result = {}
    for label in ('all', 'action', 'first_action', 'reasoning', 'format', 'eos'):
        chosen = [x for x in rows if label == 'all' or x['label'] == label or
                  (label == 'first_action' and x['first_action'])]
        if chosen:
            keys = ('kl', 'tv', 'teacher_entropy', 'base_entropy', 'logp_gain', 'top1_flip')
            result[label] = {'tokens': len(chosen), **{
                key: sum(x[key] for x in chosen)/len(chosen) for key in keys}}
    return result

def bounded_delta_rule(original, segment_tokens=2048):
    """Preserve the original 64-token recurrence blocks while bounding temporaries."""
    def forward(query, key, value, g, beta, initial_state=None,
                output_final_state=False, use_qk_l2norm_in_kernel=False,
                chunk_size=64, **kwargs):
        import torch
        if segment_tokens % chunk_size:
            raise ValueError('Segment boundaries must preserve original delta-rule chunks')
        if query.shape[1] <= segment_tokens:
            return original(query, key, value, g=g, beta=beta, initial_state=initial_state,
                output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                chunk_size=chunk_size, **kwargs)
        output = torch.empty_like(value)
        state = initial_state
        for start in range(0, query.shape[1], segment_tokens):
            end = start + segment_tokens
            block, state = original(query[:,start:end], key[:,start:end], value[:,start:end],
                g=g[:,start:end], beta=beta[:,start:end], initial_state=state,
                output_final_state=True, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                chunk_size=chunk_size, **kwargs)
            output[:,start:end] = block
        return output, state if output_final_state else None
    return forward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--recheck', help='Re-evaluate one recorded sample into a separate validation file')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    from accelerate import cpu_offload
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    device = torch.device(args.device)
    free, total = torch.cuda.mem_get_info(device)
    cap = 4 * 1024**3
    if free < cap + 2 * 1024**3:
        raise RuntimeError('Need 4 GiB diagnostic budget plus 2 GiB remaining headroom')
    torch.cuda.set_per_process_memory_fraction(cap / total, device)
    batch = samples()
    manifest = [{k: v for k, v in s.items() if k not in ('prompt','response','privileged','raw_output')}
                | {k+'_length':len(s[k]) for k in ('prompt','response','privileged')}
                | {'prefix_sha256': hashlib.sha256(json.dumps(s['prompt']+s['response'][:-1]).encode()).hexdigest()}
                for s in batch]
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    model_path = MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print('Loading frozen BF16 base into CPU RAM, 4 GiB CUDA allocator cap', flush=True)
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16,
        device_map={'':'cpu'}, output_loading_info=True)
    if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise RuntimeError(str(loading))
    if any(not any(w in k for w in ('visual','vision','mtp')) for k in loading.get('unexpected_keys', [])):
        raise RuntimeError(str(loading))
    model.requires_grad_(False).eval()
    cpu_offload(model, execution_device=device, offload_buffers=False)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    original_delta = modeling.torch_chunk_gated_delta_rule
    # Check recurrence continuation, including a final partial 64-token block.
    torch.manual_seed(42)
    with torch.inference_mode():
        q, k, v = [torch.randn(1, 387, 4, 16, device=device, dtype=torch.bfloat16) for _ in range(3)]
        g = -torch.rand(1,387,4,device=device)
        beta = torch.rand(1,387,4,device=device,dtype=torch.bfloat16)
        reference, ref_state = original_delta(q,k,v,g=g,beta=beta,output_final_state=True,use_qk_l2norm_in_kernel=True)
        bounded, state = bounded_delta_rule(original_delta,128)(q,k,v,g=g,beta=beta,output_final_state=True,use_qk_l2norm_in_kernel=True)
        error = (bounded.float()-reference.float()).abs().max().item()
        state_error = (state-ref_state).abs().max().item()
        torch.testing.assert_close(bounded, reference, rtol=0, atol=3e-5)
        torch.testing.assert_close(state, ref_state, rtol=0, atol=1e-5)
        (OUT/'recurrence_validation.json').write_text(json.dumps(dict(
            output_max_absolute_error=error,state_max_absolute_error=state_error,
            reference_chunk_size=64,test_segment_size=128,inference_segment_size=2048,
            output_absolute_tolerance=3e-5,state_absolute_tolerance=1e-5,passed=True),indent=2))
        del q,k,v,g,beta,reference,ref_state,bounded,state
    modeling.torch_chunk_gated_delta_rule = bounded_delta_rule(original_delta)
    metadata = dict(torch=torch.__version__, transformers=transformers.__version__,
                    device=str(device), dtype='bfloat16', logits_dtype='float32',
                    allocator_cap_gib=4, model=str(model_path), temperature=1,
                    teacher='frozen base + exact saved privileged tokens',
                    base='same frozen base without privileged tokens',
                    selection='first committed successful episode per task; first/middle/last turns; '
                              'all first-episode turns for calibration; ICL first turn per observed task',
                    units='nats/token', truncation=False,
                    linear_attention_memory='Original 64-token delta-rule recurrence, segmented in groups of 2048; no token truncation')
    (OUT/'metadata.json').write_text(json.dumps(metadata, indent=2))
    done = set()
    if (OUT/'results.jsonl').exists():
        done = {json.loads(line)['id'] for line in (OUT/'results.jsonl').open()}
    result_path = OUT/'results.jsonl'
    if args.recheck:
        batch = [s for s in batch if s['id'] == args.recheck]
        if len(batch) != 1:
            raise ValueError('Recheck requires an exact known sample id')
        done = set()
        result_path = OUT/'validation_results.jsonl'

    def hidden(ids, length):
        inputs = torch.tensor([ids], device=device)
        result = model.model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                             use_cache=False, return_dict=True)
        return result.last_hidden_state[:, -length:, :].detach().cpu().clone()

    with torch.inference_mode():
        completed = 0
        for sample in batch:
            if sample['id'] in done:
                continue
            start = time.monotonic()
            print('START', sample['id'], 'tokens', len(sample['prompt']), len(sample['privileged']), flush=True)
            kinds = labels(tokenizer, sample['response'])
            ids = sample['prompt'] + sample['response'][:-1]
            length = len(sample['response'])
            base_hidden = hidden(ids, length)
            if not (OUT/'null_control.json').exists():
                repeated_hidden = hidden(ids, length)
                difference = (base_hidden.float()-repeated_hidden.float()).abs().max().item()
                (OUT/'null_control.json').write_text(json.dumps(dict(
                    sample_id=sample['id'], max_hidden_absolute_difference=difference,
                    identical_input_ids=True, passed=difference == 0), indent=2))
                if difference != 0:
                    raise RuntimeError('Repeated identical baseline forwards differ')
                del repeated_hidden
            teacher_hidden = hidden(sample['privileged'] + ids, length)
            rows = []
            first_action = kinds.index('action')
            for offset in range(0, length, 32):
                end = min(offset+32, length)
                lp = model.lm_head(base_hidden[:, offset:end].to(device)).float().log_softmax(-1)[0]
                lq = model.lm_head(teacher_hidden[:, offset:end].to(device)).float().log_softmax(-1)[0]
                p, q = lp.exp(), lq.exp()
                kl = (q*(lq-lp)).sum(-1).cpu().tolist()
                tv = (.5*(q-p).abs().sum(-1)).cpu().tolist()
                hp = (-(p*lp).sum(-1)).cpu().tolist()
                hq = (-(q*lq).sum(-1)).cpu().tolist()
                bp, tp = lp.argmax(-1).cpu().tolist(), lq.argmax(-1).cpu().tolist()
                targets = torch.tensor(sample['response'][offset:end], device=device)
                gain = (lq-lp).gather(-1, targets[:, None])[:, 0].cpu().tolist()
                for j, token in enumerate(sample['response'][offset:end]):
                    rows.append(dict(index=offset+j, token_id=token,
                        token=tokenizer.decode([token]), label=kinds[offset+j],
                        first_action=offset+j==first_action,
                        kl=kl[j], tv=tv[j], base_entropy=hp[j], teacher_entropy=hq[j],
                        logp_gain=gain[j], top1_flip=int(bp[j]!=tp[j]),
                        base_top1=tokenizer.decode([bp[j]]), teacher_top1=tokenizer.decode([tp[j]])))
                del lp, lq, p, q
            item = {k:v for k,v in sample.items() if k not in ('prompt','response','privileged','raw_output')}
            item.update(summary=summarize(rows), tokens=rows,
                        prompt_tokens=len(sample['prompt']), privileged_tokens=len(sample['privileged']),
                        seconds=time.monotonic()-start,
                        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3)
            with result_path.open('a') as f:
                f.write(json.dumps(item, ensure_ascii=False)+'\n')
                f.flush()
            print('DONE', sample['id'], json.dumps(item['summary']), 'seconds', round(item['seconds'],1), flush=True)
            del base_hidden, teacher_hidden
            torch.cuda.empty_cache()
            completed += 1
            if args.limit and completed >= args.limit:
                break

if __name__ == '__main__':
    main()
