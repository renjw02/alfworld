"""Rebuild report from committed episodes, or from the checked-in CSV snapshot.
Run: .venv/bin/python report/memoryless_and_ascent/analyze.py [--from-csv]
Dependencies: numpy, matplotlib.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SEEDS = [42, 43, 44]
METHODS = ['memoryless', 'ascent']
COLORS = {'memoryless': '#2563eb', 'ascent': '#e65f24'}
NAMES = ['Place', 'Look', 'Clean', 'Heat', 'Cool', 'Place two']
rows, provenance = [], {}
if '--from-csv' in sys.argv:
    with (OUT / 'instances.csv').open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ['seed', 'instance']: r[k] = int(r[k])
        r['reward'] = float(r['reward'])
else:
    for seed in SEEDS:
        paired = []
        for method in METHODS:
            d = ROOT / f'runs/blocked-{method}-3runs-20261102/seed-{seed}'
            commit = d / 'commit.json'
            episodes = json.loads(commit.read_text())['completed']
            assert len(episodes) == 140
            assert [e['position'] for e in episodes] == list(range(140))
            manifest = json.loads((d / 'manifest.json').read_text())
            paired.append(([e['episode_id'] for e in episodes], manifest))
            provenance[f'{method}/{seed}'] = {'source': str(commit.relative_to(ROOT)), 'sha256': hashlib.sha256(commit.read_bytes()).hexdigest()}
            for i, e in enumerate(episodes, 1):
                assert e['score'] == int(e['success'])
                rows.append(dict(method=method, seed=seed, instance=i, episode_id=e['episode_id'], task_type=e['task_type'], reward=e['score'], attempt_dir=e['attempt_dir']))
        assert paired[0] == paired[1], f'Unpaired stream: seed {seed}'
    with (OUT / 'instances.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')

rewards = {}
for method in METHODS:
    rewards[method] = np.array([[r['reward'] for r in rows if r['method'] == method and r['seed'] == s] for s in SEEDS])
    assert rewards[method].shape == (3, 140)
curves = {m: np.column_stack([np.zeros(3), np.cumsum(rewards[m], axis=1)]) for m in METHODS}
x = np.arange(141)
ref = [r for r in rows if r['method'] == 'memoryless' and r['seed'] == 42]
bounds = [0] + [i for i in range(1, 140) if ref[i]['task_type'] != ref[i-1]['task_type']] + [140]
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 180})
def decorate(ax, labels=True):
    for j, (a,b) in enumerate(zip(bounds[:-1], bounds[1:])):
        if j % 2 == 0: ax.axvspan(a,b,color='#64748b',alpha=.06,lw=0)
        if a: ax.axvline(a,color='#94a3b8',ls=':',lw=.8)
        if labels: ax.text((a+b)/2,1.015,NAMES[j],transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
    ax.set_xlim(0,140); ax.set_xticks([0,20,40,60,80,100,120,140]); ax.grid(axis='y',alpha=.18)
    ax.set_xlabel('Number of tasks completed')
def save(fig, name):
    for ext in ['png','pdf','svg']: fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)

fig, axes = plt.subplots(2,2,figsize=(14,9),layout='constrained')
for i, ax in enumerate(axes.flat):
    for m in METHODS:
        y = curves[m][i] if i < 3 else curves[m].mean(axis=0)
        ax.plot(x,y,color=COLORS[m],label=m.capitalize(),lw=1.9)
        if i == 3: ax.fill_between(x,curves[m].min(axis=0),curves[m].max(axis=0),color=COLORS[m],alpha=.12)
    ax.set_title(f'Seed {SEEDS[i]}' if i < 3 else 'Mean across seeds (band: min-max)',pad=25)
    decorate(ax); ax.set_ylim(0,105); ax.set_ylabel('Cumulative reward'); ax.legend(loc='upper left')
fig.suptitle('ALFWorld: instance-wise cumulative reward\nQwen3.5-9B | task-blocked | reward = episode success (0/1)',fontsize=15)
save(fig,'cumulative_reward')

delta = curves['ascent']-curves['memoryless']
fig,ax=plt.subplots(figsize=(12,4.8),layout='constrained')
for i,s in enumerate(SEEDS): ax.plot(x,delta[i],label=f'Seed {s}',alpha=.65,lw=1.3)
ax.plot(x,delta.mean(axis=0),color='#111827',lw=2.5,label='Mean')
ax.axhline(0,color='#475569',lw=1); decorate(ax)
ax.set_ylabel('Cumulative reward: ASCENT - Memoryless'); ax.legend(ncol=4,loc='lower left')
ax.set_title('Paired cumulative advantage (positive favors ASCENT)',pad=27)
save(fig,'cumulative_advantage')

# Local slope diagnostic: no window crosses a task-type boundary.
fig,ax=plt.subplots(figsize=(12,4.8),layout='constrained')
for m in METHODS:
    for j,(a,b) in enumerate(zip(bounds[:-1],bounds[1:])):
        y=[rewards[m][:,max(a,t-9):t+1].mean() for t in range(a,b)]
        ax.plot(np.arange(a+1,b+1),y,color=COLORS[m],lw=2,label=m.capitalize() if j==0 else None)
decorate(ax); ax.set_ylim(0,1.05); ax.set_ylabel('Mean reward (last up to 10 tasks within block)')
ax.set_title('Local trend: trailing reward, reset at each task boundary',pad=27); ax.legend()
save(fig,'local_reward')

lines=['# Memoryless 与 ASCENT：逐实例持续学习趋势','',
'数据：三个 seed（42、43、44），每次 140 个任务；Qwen3.5-9B，valid_seen，task_blocked，admissible，最多 50 步。ASCENT 的参数跨任务类别保留（memory_scope=stream）。本报告仅分析 `blocked-memoryless-3runs-20261102` 与 `blocked-ascent-3runs-20261102`，不混用其他批次或旧报告的数据。','',
'## 统计口径','',
'仅取 `commit.json.completed` 的已提交记录，按 position 排序；恢复产生的废弃尝试不重复统计。脚本验证同 seed 的两种方法具有相同 manifest 和逐实例 episode ID 顺序，且六次运行全部完成。reward 使用记录中的 episode score；本批次所有 score 均为 0/1，且等于成功标记。因此累积 reward 就是截至该位置成功完成的任务数：R(n)=Σᵢ₌₁ⁿ rᵢ。横轴是流中已完成的任务数量，不是训练 step。','',
'## 逐实例累积 reward','',
'![逐实例累积 reward](cumulative_reward.png)','',
'每个点对应一个任务，未经平滑。右下图为三个 seed 的逐位置均值，阴影为 seed 最小值到最大值，并非置信区间。不同 seed 的块内实例顺序不同，均值表示相同流位置的平均结果。','',
'| Seed | Memoryless reward | ASCENT reward | ASCENT − Memoryless |','|---|---:|---:|---:|']
for i,s in enumerate(SEEDS): lines.append(f'| {s} | {curves["memoryless"][i,-1]:.0f} | {curves["ascent"][i,-1]:.0f} | {delta[i,-1]:+.0f} |')
lines += [f'| 平均 | {curves["memoryless"][:,-1].mean():.2f} | {curves["ascent"][:,-1].mean():.2f} | {delta[:,-1].mean():+.2f} |','',
'平均最终成功率为 Memoryless **59.76%**、ASCENT **60.48%**，差 **+0.71 个百分点**（三个流合计多成功 3 局）。Seed 42、43 的 ASCENT 略低，seed 44 更高，因此总体提升不稳定。','',
'## 相对基线的累积优势','', '![累积优势](cumulative_advantage.png)','',
'该图在相同任务位置相减，比单独观察单调上升的累积曲线更容易判断优势是否随经验积累扩大。正值表示 ASCENT 领先。','',
'| 类别（任务位置） | Memoryless 成功率 | ASCENT 成功率 | 块内平均 reward 差 | 块末平均累积差 |','|---|---:|---:|---:|---:|']
zh=['简单放置','灯下查看','清洗','加热','冷却','放置两个物体']
for j,(a,b) in enumerate(zip(bounds[:-1],bounds[1:])):
    mm=rewards['memoryless'][:,a:b]; aa=rewards['ascent'][:,a:b]
    lines.append(f'| {zh[j]}（{a+1}–{b}） | {100*mm.mean():.2f}% | {100*aa.mean():.2f}% | {(aa.sum()-mm.sum())/3:+.2f} | {delta[:,b].mean():+.2f} |')
lines += ['',
'ASCENT 的平均累积优势在灯下查看和清洗阶段扩大，但在加热、冷却阶段回落，最后在放置两个物体阶段重新追回。优势没有随任务数量持续扩大；曲线主要呈现明显的任务类别依赖。','',
'## 局部趋势与解释边界','', '![块内局部 reward](local_reward.png)','',
'辅助图展示最近最多 10 个任务的平均 reward，先在每个 seed 内计算，再跨 seed 平均；窗口在类别边界清空，起始不足 10 个任务时使用已有任务。它近似反映累积曲线的局部斜率，不替代主图的原始逐实例数据。','',
'- 累积 reward 必然不下降，因此曲线上升本身不代表学会了更多；应比较斜率及相对 Memoryless 的差值。','- 类别顺序固定，任务难度分布随位置改变。后半程斜率变化不能单独归因于持续学习或遗忘。','- 当前是在线适应成绩：本局先计分，ASCENT 更新只影响后续局。没有周期性重测早期任务，因此不能直接测量灾难性遗忘；也不能从这三条流证明稳定的正向迁移。','- 同一批 140 个游戏被三个 seed 重排，420 次结果并非 420 个独立任务；未作独立样本显著性声明。','',
'## 复现与文件','',
'安装 numpy、matplotlib 后，在仓库根目录执行：','',
'```bash','.venv/bin/python report/memoryless_and_ascent/analyze.py', '# 无原始 runs 时，使用已保存的逐实例数据：','.venv/bin/python report/memoryless_and_ascent/analyze.py --from-csv','```','',
'`instances.csv` 保存 840 条逐实例记录及 attempt 路径；`provenance.json` 保存原始 commit 文件路径和 SHA256。每张图提供 PNG、PDF、SVG。']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[lines.index('| 类别（任务位置） | Memoryless 成功率 | ASCENT 成功率 | 块内平均 reward 差 | 块末平均累积差 |'):][:8]))
