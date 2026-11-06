"""Combine checked-in seed-42, task_blocked, stream snapshots; no raw runs needed.
Run: python3 report/icl+dc/plot_five_methods.py (numpy, matplotlib required).
"""
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, FontProperties
import numpy as np

OUT = Path(__file__).resolve().parent
REPORT = OUT.parent
ORDER = ['icl', 'dc-cumulative', 'memoryless', 'ascent', 'success-only-sft']
NAMES = {'icl':'ICL', 'dc-cumulative':'DC-Cumulative', 'memoryless':'Memoryless', 'ascent':'ASCENT', 'success-only-sft':'Success-only SFT'}
COLORS = {'icl':'#2563eb', 'dc-cumulative':'#ea580c', 'memoryless':'#64748b', 'ascent':'#16a34a', 'success-only-sft':'#9333ea'}


def load_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    paths = [OUT/'cumulative_reward.csv', REPORT/'memoryless_and_ascent/instances.csv',
             REPORT/'memoryless_and_ascent/provenance.json', OUT/'success_only_sft_instances.csv',
             OUT/'success_only_sft_training_snapshot.json', OUT/'ascent_trace_instances.csv',
             OUT/'ascent_trace_provenance.json']
    raw = defaultdict(list)
    for r in load_csv(paths[0]):
        if r['seed']=='42' and r['memory_scope']=='stream':
            raw[r['run']].append(dict(method=r['method'], task_index=int(r['task_index']),
                                      episode_id=r['episode_id'], task_type=r['task_type'], reward=float(r['reward'])))
    for r in load_csv(paths[1]):
        if r['seed']=='42' and r['method']=='memoryless':
            raw[r['method']+'-seed42'].append(dict(method=r['method'],task_index=int(r['instance']),
                                                  episode_id=r['episode_id'],task_type=r['task_type'],reward=float(r['reward'])))
    for r in load_csv(paths[3]):
        assert r['seed']=='42' and r['memory_scope']=='stream' and r['stream_order']=='task_blocked'
        assert r['method']=='success-only-sft'
        raw['success-only-sft-seed42'].append(dict(method=r['method'],task_index=int(r['task_index']),
                                                  episode_id=r['episode_id'],task_type=r['task_type'],reward=float(r['reward'])))
    for r in load_csv(paths[5]):
        assert r['seed']=='42' and r['memory_scope']=='stream' and r['stream_order']=='task_blocked'
        assert r['method']=='ascent'
        raw[r['run']].append(dict(method='ascent',task_index=int(r['task_index']),
                                 episode_id=r['episode_id'],task_type=r['task_type'],reward=float(r['reward'])))
    ascent_source = json.loads(paths[6].read_text())
    assert hashlib.sha256(paths[5].read_bytes()).hexdigest() == ascent_source['snapshot_sha256']
    reference = sorted(raw['memoryless-seed42'],key=lambda r:r['task_index'])
    assert len(reference)==140
    groups=defaultdict(list); records=[]; excluded=[]
    for name,rows in sorted(raw.items()):
        rows.sort(key=lambda r:r['task_index'])
        assert [r['task_index'] for r in rows]==list(range(1,len(rows)+1))
        assert [(r['episode_id'],r['task_type']) for r in rows]==[(r['episode_id'],r['task_type']) for r in reference[:len(rows)]]
        assert all(r['reward'] in (0,1) for r in rows)
        if len(rows)!=140:
            excluded.append({'run':name,'episodes':len(rows),'reason':'incomplete'})
            continue
        method=rows[0]['method']
        cumulative=np.r_[0,np.cumsum([r['reward'] for r in rows])]
        groups[method].append({'run':name,'curve':cumulative})
        records.extend(dict(run=name, seed=42, memory_scope='stream', stream_order='task_blocked',
                            **r,cumulative_reward=float(cumulative[i])) for i,r in enumerate(rows,1))
    stage_lengths = [35,13,27,16,25,24]
    assert [r['curve'][-1] for r in groups['ascent']] == [89,98]
    assert {m:len(groups[m]) for m in ORDER}=={'icl':2,'dc-cumulative':3,'memoryless':1,'ascent':2,'success-only-sft':1}
    with (OUT/'ascent_stage_cumulative.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['task_index','run_1_cumulative','run_2_cumulative','mean_cumulative'])
        for j,x in enumerate(np.r_[0,np.cumsum(stage_lengths)]):
            vals=[r['curve'][x] for r in groups['ascent']]
            w.writerow([x,*vals,float(np.mean(vals))])
    with (OUT/'five_methods_seed42_stream.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
    means={m:np.mean([r['curve'] for r in groups[m]],axis=0) for m in ORDER}
    provenance={'created_at_utc':datetime.now(timezone.utc).isoformat(),
                'seed':42,'memory_scope':'stream','stream_order':'task_blocked',
                'episode_id_and_task_order_match':{'verified_methods':ORDER},'excluded':excluded,
                'ascent_source':ascent_source,
                'snapshot_sha256':{str(p.relative_to(REPORT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                'memoryless_ascent_original_sources':{k:v for k,v in json.loads(paths[2].read_text()).items() if k=='memoryless/42'},
                'groups':{m:{'n':len(groups[m]),'runs':[r['run'] for r in groups[m]],
                             'final_rewards':[float(r['curve'][-1]) for r in groups[m]],
                             'mean_final_reward':float(means[m][-1])} for m in ORDER}}
    (OUT/'five_methods_seed42_stream_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    font=Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    if font.exists():
        fontManager.addfont(str(font));plt.rcParams['font.family']=['DejaVu Sans',FontProperties(fname=font).get_name()]
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'path'})
    fig,ax=plt.subplots(figsize=(12,7))
    boundaries=[0]+[i for i in range(1,140) if reference[i]['task_type']!=reference[i-1]['task_type']]+[140]
    task_labels=['Place one','Look under\nlight','Clean &\nplace','Heat &\nplace','Cool &\nplace','Place two']
    for i,(a,b) in enumerate(zip(boundaries,boundaries[1:])):
        if i%2==0:ax.axvspan(a+.5,b+.5,color='#f1f5f9',zorder=0)
        if b<140:ax.axvline(b+.5,color='#94a3b8',ls=':',lw=.9)
        ax.text((a+b)/2,139,task_labels[i],ha='center',va='top',fontsize=10,color='#64748b')
    for method in ORDER:
        matrix=np.array([r['curve'] for r in groups[method]])
        if len(matrix)>1:
            ax.fill_between(np.arange(141),matrix.min(0),matrix.max(0),color=COLORS[method],alpha=.10,lw=0)
        label=f"{NAMES[method]} (n={len(matrix)})"
        ax.plot(np.arange(141),means[method],label=label,color=COLORS[method],lw=2.5)
        last=means[method][-1]
        ax.scatter([140],[last],s=25,color=COLORS[method],zorder=5)
        dy={'icl':-15,'dc-cumulative':9,'memoryless':9,'ascent':-16,'success-only-sft':-12}[method]
        ax.annotate(f'{last:.2f}' if len(matrix)>1 else f'{last:.0f}',(140,last),xytext=(-5,dy),
                    textcoords='offset points',ha='right',color=COLORS[method],fontweight='bold',fontsize=12)
    ax.set(xlim=(0,143),ylim=(0,143),xlabel='Task index (task-blocked)',ylabel='Cumulative reward (successful tasks)')
    ax.set_xticks([0,20,40,60,80,100,120,140]);ax.set_yticks(range(0,141,20))
    ax.grid(axis='y',color='#e2e8f0',lw=.7);ax.set_axisbelow(True)
    ax.legend(loc='upper left',bbox_to_anchor=(.015,.86),frameon=False)
    ax.set_title('ALFWorld: Cumulative reward across five methods\nSeed 42 | Memory scope: stream | Qwen3.5-9B',pad=19,fontsize=15)
    fig.text(.5,.03,'Complete 140-task runs only. Lines: same-seed means; bands: min-max (not confidence intervals).',ha='center',fontsize=10,color='#475569')
    fig.subplots_adjust(left=.09,right=.98,bottom=.14,top=.84)
    for ext in ('png','pdf','svg'):
        path=OUT/f'five_methods_seed42_stream.{ext}';fig.savefig(path,dpi=200,bbox_inches='tight')
        if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)
    text=['# 五种方法的累计 reward：seed 42 / stream',
          '\n![五种方法趋势](five_methods_seed42_stream.png)',
          '\n补充：[各阶段完成结果柱状图](five_methods_stage_results.md)。',
          '\n比较上下文为 Qwen3.5-9B、valid_seen、seed 42、task_blocked、memory_scope=stream、admissible、最多 50 步。五种方法的 140 个 episode ID、task_type 和任务顺序已逐条核对一致。ASCENT 使用本地两次完整运行的原始 traces，按每个任务位置计算累计成功次数均值与 min–max 阴影。横轴为任务序号，纵轴为已完成任务的 reward 累积；reward 为 0/1，所以也等于累计成功任务数。',
          '\n| 方法 | 完整运行数 | 各次最终 reward | 图中最终值 | 成功率 |\n|---|---:|---|---:|---:|']
    for m in ORDER:
        g=provenance['groups'][m]
        text.append(f"| {NAMES[m]} | {g['n']} | "+', '.join(f'{x:.0f}' for x in g['final_rewards'])+f" | {g['mean_final_reward']:.2f} | {g['mean_final_reward']/140*100:.2f}% |")
    text+=['\n## SFT 更新数据来源',
           '\nSuccess-only SFT 的最终累计 reward 为 65（46.43%）。实际更新的 987 条样本中，简单放置占 32.93%，双物体占 19.66%，加热占 17.43%，三者合计 70.01%；冷却仅占 4.76%。[详细任务类型、目标 token 和 optimizer steps 分析](success_only_sft_training_sources.md)。',
           '\n## 选取口径',
           '\n- ICL、DC 延续前一张图：只取 stream 的完整运行；同 seed 多次重复按每个任务位置求均值。阴影为这些重复的 min–max，不是置信区间。ICL repeat 1 只完成 69 局，被排除。\n- Memoryless 使用 `report/memoryless_and_ascent/instances.csv` 中 seed 42 的 140 条记录；不混入 seed 43/44 或 shuffled 结果。\n- ASCENT 使用 `runs/ascent-task_blocked-seed42-20261103T070526Z/run1` 和 `run2`，只读取 `commit.json` 引用的已完成 attempts；逐局核对 `episode.json` 与 commit 的 ID、类型、位置、success、score，并与 summary 汇总交叉验证。两次均为 seed 42 / stream / task_blocked，成功总数为 89、98，平均 93.5（66.79%）。280 条逐任务 reward 保存在 `ascent_trace_instances.csv`，原始配置与文件哈希保存在 `ascent_trace_provenance.json`。\n- Success-only SFT 使用本机 GPU 2 单卡正式运行的 140 条已提交结果，batch_size=2、梯度累积=4、训练样本长度上限=4096、历史窗口为之前 5 次交互；没有把试跑或双卡诊断加入曲线。\n- Memoryless 的 stream 标签表示同一连续任务流；它本身不保存跨局记忆。\n- Memoryless 的原始运行目录当前不在本机，读取仓库逐实例快照和原始 commit 哈希；对其配置的描述依据已有报告。ASCENT 已核对本地原始配置，其两次运行均值替换此前单次 75/140 的结果。\n- 不同方法来自不同批次，未证明提示词、软件版本及所有其他设置完全相同。该图是现有结果趋势的对照，不是严格控制全部变量的多 seed 因果实验；n 不同也不能据此作显著性声明。',
           '\n## 文件',
           '\n- [PNG](five_methods_seed42_stream.png) / [PDF](five_methods_seed42_stream.pdf) / [SVG](five_methods_seed42_stream.svg)\n- [五种方法的逐任务 CSV](five_methods_seed42_stream.csv) / [ASCENT 阶段末累计数据](ascent_stage_cumulative.csv) / [ASCENT 逐任务快照](ascent_trace_instances.csv) / [ASCENT traces 来源](ascent_trace_provenance.json)\n- [来源与哈希](five_methods_seed42_stream_provenance.json)\n- [生成脚本](plot_five_methods.py)：`python3 report/icl+dc/plot_five_methods.py`，仅依赖仓库内快照，无需原始 runs/。如需从本地 traces 重新导出 ASCENT，先运行 `python3 report/icl+dc/export_ascent_traces.py`。']
    (OUT/'five_methods_seed42_stream.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(provenance['groups'],indent=2));print('Verified 1260 task-level rewards from 9 complete runs, including 280 ASCENT traces.')


if __name__=='__main__':main()
