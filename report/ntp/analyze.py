"""Compare committed NTP runs under task_blocked; seed=42, two memory scopes.
Run: python3 report/ntp/analyze.py (requires local runs, numpy, matplotlib).
"""
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
BASE=ROOT/'runs/continual-matrix-20261101T135608Z'
METHODS=['history-ntp-assistant','history-ntp-assistant-success']
NAMES=dict(zip(METHODS,['NTP-Assistant','NTP-Assistant-Success']))
COLORS=dict(zip(METHODS,['#2563eb','#ea580c']))
SCOPES=['stream','task_type']
TASKS=[('pick_and_place_simple','简单放置','Place one'),('look_at_obj_in_light','灯下查看','Look under\nlight'),
       ('pick_clean_then_place_in_recep','清洗后放置','Clean &\nplace'),('pick_heat_then_place_in_recep','加热后放置','Heat &\nplace'),
       ('pick_cool_then_place_in_recep','冷却后放置','Cool &\nplace'),('pick_two_obj_and_place','双物体放置','Place two')]


def read(path):return json.loads(path.read_text())

def main():
    manifest=read(BASE/'manifest-blocked.json')['episodes'];runs=[];instances=[]
    for job in read(BASE/'plan.json')['jobs']:
        if job['method'] not in METHODS:continue
        p=Path(job['output_dir']);cfg=read(p/'config.json')
        assert cfg['stream']['seed']==42 and cfg['stream']['order']=='task_blocked'
        assert cfg['memory_scope']==job['memory_scope']
        assert read(p/'manifest.json')['episodes']==manifest
        rows=read(p/'commit.json')['completed'] if (p/'commit.json').exists() else []
        stats=Counter();tasks=defaultdict(Counter);reward=0;curve=[0]
        for i,r in enumerate(rows,1):
            assert r['position']==i-1 and r['episode_id']==manifest[i-1]['episode_id']
            assert r['score']==int(r['success'])
            reward+=r['score'];curve.append(reward)
            for k in ['attempts','format_errors','invalid_actions','environment_calls']:stats[k]+=r[k]
            stats['zero_call_episodes']+=r['environment_calls']==0
            tasks[r['task_type']]['episodes']+=1;tasks[r['task_type']]['reward']+=r['score']
            instances.append(dict(run=job['name'],method=job['method'],memory_scope=job['memory_scope'],seed=42,
                                  task_index=i,episode_id=r['episode_id'],task_type=r['task_type'],reward=r['score'],cumulative_reward=reward))
        runs.append(dict(name=job['name'],method=job['method'],scope=job['memory_scope'],repeat=job['repeat'],
                         episodes=len(rows),complete=len(rows)==140,reward=reward,curve=curve,stats=dict(stats),
                         by_task=dict(tasks),source=str(p.relative_to(ROOT))))
    assert len(runs)==12 and sum(r['complete'] for r in runs)==11
    groups={(m,s):[r for r in runs if r['complete'] and r['method']==m and r['scope']==s] for s in SCOPES for m in METHODS}
    with (OUT/'instances.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(instances[0]),lineterminator='\n');w.writeheader();w.writerows(instances)
    (OUT/'snapshot.json').write_text(json.dumps({'created_at_utc':datetime.now(timezone.utc).isoformat(),
                                               'seed':42,'order':'task_blocked','runs':runs},ensure_ascii=False,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'path'})
    fig,axes=plt.subplots(1,2,figsize=(15,6.5),sharey=True)
    ends=np.cumsum([sum(e['task_type']==t for e in manifest) for t,_,_ in TASKS]);starts=np.r_[0,ends[:-1]]
    for ax,scope in zip(axes,SCOPES):
        for k,(a,b) in enumerate(zip(starts,ends)):
            if k%2==0:ax.axvspan(a+.5,b+.5,color='#f1f5f9',zorder=0)
            if b<140:ax.axvline(b+.5,color='#94a3b8',ls=':',lw=.8)
            ax.text((a+b)/2,140,TASKS[k][2],ha='center',va='top',fontsize=8,color='#64748b')
        for method in METHODS:
            g=groups[method,scope];matrix=np.array([r['curve'] for r in g]);mean=matrix.mean(0);x=np.arange(141)
            ax.fill_between(x,matrix.min(0),matrix.max(0),color=COLORS[method],alpha=.12,lw=0)
            for curve in matrix:ax.plot(x,curve,color=COLORS[method],lw=.7,alpha=.22)
            ax.plot(x,mean,color=COLORS[method],lw=2.5,label=f'{NAMES[method]} (n={len(g)})')
            ax.scatter([140],[mean[-1]],color=COLORS[method],s=22)
            ax.annotate(f'{mean[-1]:.2f}',(140,mean[-1]),xytext=(-5,8),textcoords='offset points',ha='right',color=COLORS[method],fontweight='bold')
        ax.set(xlim=(0,143),ylim=(0,143),xlabel='Task index (task-blocked)')
        ax.set_xticks([0,20,40,60,80,100,120,140]);ax.set_yticks(range(0,141,20));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.set_title('stream: retain state across task types' if scope=='stream' else 'task_type: reset state at type boundaries',pad=15)
        ax.legend(loc='upper left',bbox_to_anchor=(.01,.86),frameon=False,fontsize=9)
    axes[0].set_ylabel('Cumulative reward (successful tasks)')
    fig.suptitle('NTP comparison | task-blocked | seed 42 | Qwen3.5-9B',fontsize=16,y=.99)
    fig.text(.5,.035,'Complete 140-task runs only. Lines: same-seed means; bands: min-max (not confidence intervals).',ha='center',fontsize=10,color='#475569')
    fig.subplots_adjust(left=.06,right=.98,bottom=.15,top=.87,wspace=.10)
    for ext in ('png','pdf','svg'):
        p=OUT/f'task_blocked_trends.{ext}';fig.savefig(p,dpi=200,bbox_inches='tight')
        if ext=='svg':p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')
    plt.close(fig)
    lines=['# NTP-Assistant 与 NTP-Assistant-Success：task_blocked 趋势',
           '\n`task_blocked` 是任务顺序，`stream` 是记忆范围，两者不是互斥配置。本报告固定 task_blocked + seed 42，比较 stream（跨类型保留状态）与 task_type（类型边界重置）的两种 NTP 方法。',
           '\n![NTP trends](task_blocked_trends.png)',
           '\n横轴是同一份 manifest 的任务序号 1–140，纵轴为已提交 episode 的终局 score 累加。已核对 score=success，均为 0/1。只取完整运行求均值；同 seed 的重复不是不同 seed。',
           '\n## 完整结果',
           '\n| 方法 | 记忆范围 | 完成次数 | 各次最终 reward | 平均累计 reward | 成功率 |\n|---|---|---:|---|---:|---:|']
    for scope in SCOPES:
        for method in METHODS:
            g=groups[method,scope];avg=np.mean([r['reward'] for r in g])
            lines.append(f'| {NAMES[method]} | {scope} | {len(g)}/3 | '+', '.join(f"r{r['repeat']}={r['reward']:.0f}" for r in g)+f' | {avg:.2f} | {100*avg/140:.2f}% |')
    lines+=['\n## 趋势解读',
            '\n- **task_blocked + stream：** 普通 NTP 在前两个任务块后基本进入平台，最终均值 36.67；success-only 版本后面仍有增益，最终均值 69.00，增加 32.33 个成功任务（23.10 个百分点）。',
            '\n- **task_blocked + task_type：** 重置后普通 NTP 在后续类型上也能继续积累 reward，最终均值 61.00；success-only 版本为 81.00，增加 20 个任务（14.29 个百分点）。',
            '\n- 从 stream 改成 task_type，普通 NTP 均值增加 24.33（17.38 个百分点），success-only 增加 12.00（8.57 个百分点）。但普通 NTP 的 task_type 只有 2 次完成，不能将该差值视为严格的因果效应。',
            '\n- task_type 中双方共同完整运行的 repeat 为 1、3：普通 NTP 均值 61.00，success-only 为 79.50，差 18.50（13.21 个百分点）；不依赖 success-only 的额外 repeat 2，优势方向仍相同。',
            '\n- 固定类型顺序会把任务难度与训练时间混合；累计曲线必然不下降，曲线上升本身不代表迁移能力提升。本批次没有不同 seed 的独立重复。',
            '\n## 各任务块成功率',
            '\n每格合并该组完整运行，分母随完成次数而异。',
            '\n| 任务类型 | NTP / stream | NTP-Success / stream | NTP / task_type | NTP-Success / task_type |\n|---|---:|---:|---:|---:|']
    for key,label,_ in TASKS:
        cells=[]
        for scope in SCOPES:
            for method in METHODS:
                g=groups[method,scope];n=sum(r['by_task'][key]['episodes'] for r in g);w=sum(r['by_task'][key]['reward'] for r in g)
                cells.append(f'{w:.0f}/{n}（{100*w/n:.1f}%）')
        lines.append(f'| {label} | '+' | '.join(cells)+' |')
    lines+=['\n## 失败表现',
            '\n| 方法 / 范围 | 尝试数 | 格式错误（占尝试） | 非法动作（占尝试） | 零环境调用局数 |\n|---|---:|---:|---:|---:|']
    for scope in SCOPES:
        for method in METHODS:
            c=Counter()
            for r in groups[method,scope]:c.update(r['stats'])
            lines.append(f"| {NAMES[method]} / {scope} | {c['attempts']} | {c['format_errors']}（{100*c['format_errors']/c['attempts']:.2f}%） | {c['invalid_actions']}（{100*c['invalid_actions']/c['attempts']:.2f}%） | {c['zero_call_episodes']} |")
    lines+=['\n普通 NTP / stream 的 repeat 1、2 分别有 80、83 局完全没有环境调用；repeat 3 没有这种格式崩溃，但 5484 次尝试中有 4673 次非法动作。因此不同重复的直接失败模式并不完全相同。success-only 的 stream 三次均无零调用局，但仍有大量非法动作，加热和冷却也很弱。',
            '\n这些都是旧 NTP 实现的结果：assistant-only body 标签、不监督模板外 EOS、训练/生成前缀不一致，以及整段 transcript 的 token 分块等问题见[此前审计](../2026-11-04-ntp-error-analysis.md)。success-only 只筛选成功 episode，并不逐步过滤其中的错误输出。不能用这批结果概括所有 NTP 实现；也不能将它与后来手工构造样本、带 EOS 监督的 success-only-sft 混为同一种方法。',
            '\n## 未完成项与文件',
            '\n`history-ntp-assistant-task_type-2` 在 0/140 局时因 cuDNN/CUDA 错误失败，未纳入均值。其余 11 次均已完整完成；先前还在运行的 stream repeat 1、2 的结果已更新到最终 140 局。',
            '\n[PNG](task_blocked_trends.png) / [PDF](task_blocked_trends.pdf) / [SVG](task_blocked_trends.svg) · [逐任务 CSV](instances.csv) · [统计快照](snapshot.json) · [生成脚本](analyze.py)']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print('Validated 11 complete runs, 1540 rewards; one failed startup excluded.')


if __name__=='__main__':main()
