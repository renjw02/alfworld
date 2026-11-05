"""Audit actual committed SFT updates and export a portable plotting snapshot.
Run: python3 report/icl+dc/analyze_sft_sources.py (requires local runs/ artifacts).
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
RUN = REPO/'runs/success-only-sft-seed42-task-blocked-gpu2-20261104T084004Z/experiment'
TASKS = [('pick_and_place_simple','简单放置'), ('look_at_obj_in_light','灯下查看'),
         ('pick_clean_then_place_in_recep','清洗后放置'), ('pick_heat_then_place_in_recep','加热后放置'),
         ('pick_cool_then_place_in_recep','冷却后放置'), ('pick_two_obj_and_place','双物体放置')]


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write_csv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    config=read(RUN/'config.json');rows=read(RUN/'commit.json')['completed']
    assert config['method']=='success-only-sft' and config['memory_scope']=='stream'
    assert config['stream']['seed']==42 and config['stream']['order']=='task_blocked'
    assert len(rows)==140
    groups=defaultdict(Counter);episodes=[];instances=[];hashes={};eos_ids=set();total_steps=0
    for index,row in enumerate(rows,1):
        assert row['position']==index-1 and row['score']==int(row['success'])
        update_file=RUN/row['attempt_dir']/'update.json';u=read(update_file)
        hashes[str(update_file.relative_to(REPO))]=sha(update_file)
        samples=u.get('samples',[]);records=u.get('records',[]);rejected=u.get('rejected',[])
        n=u.get('samples_trained',0);steps=u.get('optimizer_steps',0);targets=u.get('target_tokens',0)
        assert n==len(samples) and steps==len(records)
        assert n==sum(r['samples'] for r in records) and targets==sum(r['target_tokens'] for r in records)
        assert targets==sum(s['target_tokens'] for s in samples)
        if row['success']:
            assert n+len(rejected)==row['attempts']
            effective=config['sft']['batch_size']*config['sft']['gradient_accumulation_steps']
            assert steps==(n+effective-1)//effective
            total_steps+=steps;assert u['total_optimizer_steps']==total_steps
        else:
            assert u['skipped'] and not samples and not records and steps==0
        assert len({s['step'] for s in samples})==n
        for s in samples:
            k=s['prompt_tokens'];labels=s['labels']
            assert labels[:k]==[-100]*k and labels[k:]==s['input_ids'][k:]
            assert len(labels[k:])==s['target_tokens'] and labels[-1]!=-100
            eos_ids.add(labels[-1])
        counts=Counter(x['reason'] for x in rejected);g=groups[row['task_type']]
        stats=dict(episodes=1,successful_episodes=int(row['success']),updated_episodes=int(steps>0),
                   samples=n,target_tokens=targets,optimizer_steps=steps,micro_batches=u.get('micro_batches',0),
                   rejected_invalid=counts['unexecuted_or_inadmissible'],
                   rejected_no_change=counts['unchanged_observation'],rejected_overlength=counts['overlength'])
        g.update(stats)
        episodes.append(dict(task_index=index,episode_id=row['episode_id'],task_type=row['task_type'],
                             success=int(row['success']),**{k:v for k,v in stats.items() if k!='episodes'},
                             update_file=str(update_file.relative_to(REPO))))
        instances.append(dict(method='success-only-sft',seed=42,memory_scope='stream',stream_order='task_blocked',
                              task_index=index,episode_id=row['episode_id'],task_type=row['task_type'],reward=row['score']))
    total=Counter()
    for group in groups.values():total.update(group)
    assert (total['successful_episodes'],total['samples'],total['optimizer_steps'],total['target_tokens'])==(65,987,149,57019)
    aggregate=[]
    for key,label in TASKS:
        g=groups[key]
        aggregate.append(dict(task_type=key,task_label=label,**g,
                              sample_share=g['samples']/total['samples'],
                              target_token_share=g['target_tokens']/total['target_tokens'],
                              optimizer_step_share=g['optimizer_steps']/total['optimizer_steps']))
    write_csv(OUT/'success_only_sft_training_by_task.csv',aggregate)
    write_csv(OUT/'success_only_sft_training_episodes.csv',episodes)
    write_csv(OUT/'success_only_sft_instances.csv',instances)
    snapshot={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'source':str(RUN.relative_to(REPO)),
              'commit_sha256':sha(RUN/'commit.json'),'config_sha256':sha(RUN/'config.json'),'config':config,
              'totals':dict(total),'by_task':aggregate,'observed_terminal_target_ids':sorted(eos_ids),
              'counting_rule':'current-response targets (including EOS) from committed update.json; no prompt/history tokens',
              'update_sha256':hashes}
    (OUT/'success_only_sft_training_snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    text=['# Success-only SFT：实际更新数据来自哪些任务',
          '\n数据：GPU 2 单卡运行，seed 42、task_blocked、stream，140 局全部完成；最终 reward 65。逐条读取 commit.json 指向的 update.json，核对 samples、labels、训练记录与累计 optimizer_steps，不把“成功局数”直接当作训练样本数。',
          '\n## 训练数据构成',
          '\n共 65 个成功 episode 产生 **987 条训练样本、57,019 个目标 token（含 EOS）、149 次 optimizer 更新**。75 个失败 episode 的更新均为 0。65 个成功 episode 都有实际更新，未出现成功但零样本的局。',
          '\n| 任务类型 | 成功并更新的局/全部局 | 训练样本数（占比） | 目标 token（占比） | optimizer steps（占比） | 每成功局平均样本数 |\n|---|---:|---:|---:|---:|---:|']
    for g in aggregate:
        text.append(f"| {g['task_label']} | {g['successful_episodes']}/{g['episodes']} | {g['samples']}（{100*g['sample_share']:.2f}%） | {g['target_tokens']:,}（{100*g['target_token_share']:.2f}%） | {g['optimizer_steps']}（{100*g['optimizer_step_share']:.2f}%） | {g['samples']/g['successful_episodes']:.2f} |")
    top3=[groups[k] for k in ('pick_and_place_simple','pick_two_obj_and_place','pick_heat_then_place_in_recep')]
    topn=sum(g['samples'] for g in top3)
    text += ['\n## 结论',
             '\n1. **简单放置是最大的更新来源**：325 条样本（32.93%）、18,668 个目标 token（32.74%）、52 次更新（34.90%）。虽然它占 30/65 个成功局（46.15%），按真实样本/目标 token 统计的份额更接近三分之一。',
             f'\n2. **其次是双物体放置和加热**，分别贡献 194 条、172 条样本。简单放置、双物体、加热合计 {topn}/987 条样本（{100*topn/987:.2f}%）。加热仅 7 个成功局，但平均每局保留 24.57 条交互，因此它的训练份额不能仅由成功局数判断。',
             '\n3. **冷却数据最少**：25 个任务只有 3 个成功局，最终仅 47 条样本（4.76%）、2,444 个目标 token（4.29%）、7 次更新（4.70%）。清洗也只有 8/27 局提供更新，贡献 119 条样本（12.06%）。这表明成功筛选后的数据在任务类型间明显不均衡。',
             '\n4. 类型顺序与训练时间绑定：第 1–35 个任务的简单放置先提供全部 52 次更新；冷却直到第 92–116 个任务才出现。现有记录支持“简单任务先占据较大训练份额、难任务成功信号少”，但不能单凭这些统计证明它导致了后续任务表现下降。',
             '\n## 统计口径与过滤',
             '\n- 只数实际送入训练的当前 agent 回答及 EOS；system、task、历史和 observation 虽在 input_ids 中，但标签为 -100，不计为目标 token。\n- batch_size=2、gradient_accumulation_steps=4；每个 episode 单独提交尾部窗口，所以总步数是各局 ceil(samples/8) 之和，即 149，而不是 ceil(987/8)=124。\n- 每个回答内部按目标 token 求均值，累积窗口再按样本求均值。目标 token 份额不等同于 loss 权重或参数变化份额；optimizer 步数也不能直接表示有效梯度贡献。\n- 成功局内共排除 97 次非法/未执行动作和 34 次 observation 不变的交互；没有样本因 4096 上限被排除。987 条保留样本与 131 条排除记录合计覆盖成功局的 1118 次尝试。\n- 此处不把失败局中的交互计为“过滤后训练样本”；失败局整体跳过，也没有把这些失败局的文本训练进去。',
             '\n## 文件',
             '\n- [按类型统计 CSV](success_only_sft_training_by_task.csv)\n- [逐局更新 CSV](success_only_sft_training_episodes.csv)\n- [配置、统计与来源哈希](success_only_sft_training_snapshot.json)\n- [用于合图的逐任务 reward](success_only_sft_instances.csv)\n- [审计脚本](analyze_sft_sources.py)：`python3 report/icl+dc/analyze_sft_sources.py`。']
    (OUT/'success_only_sft_training_sources.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'totals':dict(total),'by_task':aggregate},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
