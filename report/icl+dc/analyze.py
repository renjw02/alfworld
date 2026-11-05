"""Rebuild task_blocked ICL/DC report from committed local experiment artifacts.

Run: python3 report/icl+dc/analyze.py
Requires numpy and matplotlib. No model, GPU, or network access.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BATCH = REPO / 'runs/continual-matrix-20261101T135608Z'
TASKS = [
    ('pick_and_place_simple', '简单放置', '简单\n放置'),
    ('look_at_obj_in_light', '灯下查看', '灯下\n查看'),
    ('pick_clean_then_place_in_recep', '清洗后放置', '清洗后\n放置'),
    ('pick_heat_then_place_in_recep', '加热后放置', '加热后\n放置'),
    ('pick_cool_then_place_in_recep', '冷却后放置', '冷却后\n放置'),
    ('pick_two_obj_and_place', '放置两个物体', '放置\n双物体'),
]
METHODS = ['icl', 'dc-cumulative']
SCOPES = ['stream', 'task_type']
LABEL = {'icl': 'ICL', 'dc-cumulative': 'DC-Cumulative'}
COLOR = {'icl': '#2563eb', 'dc-cumulative': '#ea580c'}


def read(path):
    return json.loads(path.read_text())


def csv_write(path, rows, fields):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def configure_plot():
    # Use DejaVu for Latin/digits and the local CJK font for Chinese glyphs.
    font = Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    if font.exists():
        fontManager.addfont(str(font))
        plt.rcParams['font.family'] = ['DejaVu Sans', FontProperties(fname=font).get_name()]
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.unicode_minus': False, 'pdf.fonttype': 42, 'svg.fonttype': 'path',
                         'axes.labelcolor': '#334155', 'text.color': '#172033'})


def main():
    manifest = read(BATCH / 'manifest-blocked.json')
    entries = manifest['episodes']
    assert len(entries) == 140 and manifest['seed'] == 42 and manifest['order'] == 'task_blocked'
    plan = read(BATCH / 'plan.json')['jobs']
    status = read(BATCH / 'status.json')
    runs, long_rows = [], []
    for job in plan:
        if job['method'] not in METHODS:
            continue
        root = Path(job['output_dir'])
        config = read(root / 'config.json')
        assert config['stream']['order'] == 'task_blocked' and config['stream']['seed'] == 42
        assert read(root / 'manifest.json')['episodes'] == entries
        commit = root / 'commit.json'
        rows = read(commit)['completed'] if commit.exists() else []
        stats, rejected, by_task = Counter(), Counter(), defaultdict(lambda: Counter())
        reward, stale, longest_stale = 0, 0, 0
        for position, row in enumerate(rows):
            assert row['position'] == position and row['episode_id'] == entries[position]['episode_id']
            assert row['task_type'] == entries[position]['task_type']
            assert row['score'] in (0, 1) and row['score'] == int(row['success'])
            reward += row['score']
            by_task[row['task_type']]['episodes'] += 1
            by_task[row['task_type']]['reward'] += row['score']
            for key in ('attempts', 'environment_calls', 'invalid_actions', 'format_errors'):
                stats[key] += row[key]
            stats['failed_episodes'] += not row['success']
            attempt = root / row['attempt_dir']
            episode = read(attempt / 'episode.json')
            if not row['success']:
                actions = Counter(s.get('action') for s in episode['steps'] if s.get('action'))
                stats['failure_repeating_one_action_20_times'] += bool(actions) and actions.most_common(1)[0][1] >= 20
            accepted = ''
            if job['method'] == 'dc-cumulative':
                update = read(attempt / 'update.json')
                accepted = bool(update['accepted'])
                stats['curator_updates'] += 1
                stats['curator_accepted'] += accepted
                if not accepted:
                    rejected[update['skip_reason']] += 1
                stale = 0 if accepted or row.get('memory_reset_before') else stale
                if not accepted:
                    stale += 1
                longest_stale = max(longest_stale, stale)
            long_rows.append(dict(run=job['name'], method=job['method'], memory_scope=job['memory_scope'],
                                  repeat=job['repeat'], seed=42, task_index=position+1,
                                  episode_id=row['episode_id'], task_type=row['task_type'],
                                  reward=row['score'], cumulative_reward=reward,
                                  success=row['success'], curator_accepted=accepted))
        runs.append(dict(name=job['name'], method=job['method'], memory_scope=job['memory_scope'],
                         repeat=job['repeat'], state=status[job['name']]['state'], episodes=len(rows),
                         complete=len(rows)==len(entries), total_reward=reward,
                         success_rate=reward/len(rows) if rows else None,
                         cumulative_reward=[0]+np.cumsum([r['score'] for r in rows]).tolist(),
                         by_task=dict(by_task), stats=dict(stats), curator_rejections=dict(rejected),
                         longest_rejected_curator_streak=longest_stale,
                         output_dir=str(root.relative_to(REPO)),
                         commit_sha256=hashlib.sha256(commit.read_bytes()).hexdigest() if commit.exists() else None))
    assert len(runs) == 12 and sum(r['complete'] for r in runs) == 9
    groups = {(m,s):[r for r in runs if r['complete'] and r['method']==m and r['memory_scope']==s]
              for s in SCOPES for m in METHODS}
    curves = []
    for (method, scope), group in groups.items():
        matrix = np.array([r['cumulative_reward'] for r in group])
        for i in range(141):
            curves.append(dict(method=method, memory_scope=scope, task_index=i, n=len(group),
                               mean=matrix[:,i].mean(), minimum=matrix[:,i].min(), maximum=matrix[:,i].max()))
    csv_write(OUT/'cumulative_reward.csv', long_rows, list(long_rows[0]))
    csv_write(OUT/'mean_curves.csv', curves, list(curves[0]))
    snapshot = {'generated_at_utc':datetime.now(timezone.utc).isoformat(),
                'reward_definition':'terminal episode score; verified binary and equal to success',
                'mean_population':'complete 140-episode runs only; fixed n over the entire curve',
                'seed':42, 'stream_order':'task_blocked', 'runs':runs}
    (OUT/'snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')

    configure_plot()
    ends = np.cumsum([sum(e['task_type']==key for e in entries) for key,_,_ in TASKS])
    starts = np.r_[0, ends[:-1]]
    def decorate(ax):
        ax.set(xlim=(0,143), ylim=(0,142), xlabel='任务序号（task_blocked）')
        ax.set_xticks([0,20,40,60,80,100,120,140])
        ax.set_yticks(range(0,141,20))
        ax.grid(axis='y', color='#e2e8f0', linewidth=.7)
        ax.set_axisbelow(True)
        for k,(start,end) in enumerate(zip(starts,ends)):
            if k%2==0: ax.axvspan(start+.5,end+.5,color='#f1f5f9',zorder=0)
            if end<140: ax.axvline(end+.5,color='#94a3b8',linestyle=':',linewidth=.9)
            ax.text((start+end)/2,139,TASKS[k][2],ha='center',va='top',fontsize=9,color='#64748b')
    fig,axes=plt.subplots(1,2,figsize=(15,6.6),sharey=True)
    for ax,scope in zip(axes,SCOPES):
        decorate(ax)
        ax.set_title('stream：跨类型保留记忆' if scope=='stream' else 'task_type：类型切换时重置记忆',pad=14,fontsize=13)
        for method in METHODS:
            group=groups[method,scope];matrix=np.array([r['cumulative_reward'] for r in group]);mean=matrix.mean(0)
            x=np.arange(141);color=COLOR[method]
            ax.fill_between(x,matrix.min(0),matrix.max(0),color=color,alpha=.12,linewidth=0)
            for row in matrix:ax.plot(x,row,color=color,alpha=.22,linewidth=.8)
            ax.plot(x,mean,color=color,linewidth=2.5,label=f'{LABEL[method]}（n={len(group)}）')
            ax.scatter([140],[mean[-1]],color=color,s=25,zorder=5)
            ax.annotate(f'{mean[-1]:.2f}',(140,mean[-1]),xytext=(-4,8 if mean[-1]==max(np.mean([r['total_reward'] for r in groups[m,scope]]) for m in METHODS) else -15),textcoords='offset points',ha='right',color=color,fontsize=11,fontweight='bold')
        ax.legend(loc='upper left',bbox_to_anchor=(.02,.87),frameon=False,fontsize=10)
    axes[0].set_ylabel('累计 reward（累计成功任务数）')
    fig.suptitle('ALFWorld · ICL 与 DC-Cumulative 的累计 reward',fontsize=17,y=.99)
    fig.text(.5,.045,'Qwen3.5-9B · seed 42 · valid_seen 140 tasks · 实线：完整运行均值；阴影：最小—最大范围（非置信区间）',ha='center',fontsize=10,color='#475569')
    fig.subplots_adjust(left=.065,right=.98,bottom=.15,top=.87,wspace=.09)
    for ext in ('png','pdf','svg'):
        path = OUT/f'cumulative_reward.{ext}'
        fig.savefig(path,dpi=200,bbox_inches='tight')
        if ext == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(14,10),sharex=True,sharey=True)
    for row,scope in enumerate(SCOPES):
        for col,method in enumerate(METHODS):
            ax=axes[row,col];decorate(ax);ax.set_title(f'{LABEL[method]} / {scope}')
            for r in [r for r in runs if r['method']==method and r['memory_scope']==scope]:
                if not r['episodes']:
                    ax.plot([],[],color='#64748b',linestyle='--',label=f"repeat {r['repeat']}：0/140，启动失败")
                    continue
                ax.plot(range(r['episodes']+1),r['cumulative_reward'],linewidth=1.8,
                        color={1:'#2563eb',2:'#ea580c',3:'#16a34a'}[r['repeat']],
                        linestyle='-' if r['complete'] else '--',
                        label=f"repeat {r['repeat']}：{r['total_reward']:.0f}/{r['episodes']}"+('（未完成）' if not r['complete'] else ''))
                if not r['complete']:ax.scatter([r['episodes']],[r['total_reward']],marker='x',s=65,color='#b91c1c',zorder=5)
            ax.legend(loc='upper left',bbox_to_anchor=(.01,.88),frameon=False,fontsize=9)
            if col==0:ax.set_ylabel('累计 reward')
    fig.suptitle('每次运行的累计 reward（未完成曲线止于最后已提交任务）',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/'individual_runs.png',dpi=180,bbox_inches='tight');plt.close(fig)

    lines=['# ICL 与 DC-Cumulative：task_blocked 结果分析',
           '\n最新：[加入 Success-only SFT 的五方法合图（英文图例）](five_methods_seed42_stream.md) · [SFT 更新数据来源](success_only_sft_training_sources.md)。',
           '\n新增：[ICL、DC、Memoryless、ASCENT 四方法合图（seed 42 / stream）](four_methods_seed42_stream.md)。',
           f'\n生成时间：{snapshot["generated_at_utc"]}。数据来自本机已提交实验记录。',
           '\n## 累计 reward 折线图',
           '\n![累计 reward](cumulative_reward.png)',
           '\n横轴是任务执行序号 1–140，按 task_blocked 顺序排列；纵轴为前 t 个任务的终局 reward 之和：`R(t) = sum(score_i, i=1..t)`。已逐条核对日志：score 为 0/1 且等于 success，因此累计 reward 等于累计成功任务数。不是累计成功率，也不是每一步环境 score 的累加。',
           '\n两幅子图分别比较 stream 和 task_type；粗线是完整运行的均值，细线是各次完整运行，阴影为这些重复运行的最小–最大值，不是置信区间。只纳入完整跑完 140 局的运行，因此每条均值曲线的 n 从头到尾不变。未完成运行不会补零、外推或在中途退出均值计算。',
           '\n任务区间：简单放置 1–35；灯下查看 36–48；清洗后放置 49–75；加热后放置 76–91；冷却后放置 92–116；双物体放置 117–140。竖虚线标注类型边界。',
           '\n## 设置与统计范围',
           '\n- 模型 Qwen3.5-9B，valid_seen，admissible 动作模式，max_steps=50。所有运行均为 seed 42；repeat 编号不代表不同 seed。任务 manifest 和逐局任务 ID 已核对一致。\n- ICL 使用前 3 局完整轨迹加当前局（window_episodes=4），不筛选前局是否成功。DC-Cumulative 每局后生成替换整个 cheatsheet 的总结，curator 输出预算 2048 tokens。两者均不更新模型参数。\n- stream 跨任务类型保留历史/cheatsheet；task_type 在类型切换时清空。任务类型与时间位置绑定，曲线斜率变化同时受任务难度和记忆影响，不能单独归因为遗忘。\n- 共 12 次运行：9 次完整完成、3 次因资源/执行错误失败。统计使用 commit.json 中的已提交轨迹，不使用孤立或重试后被覆盖的尝试。',
           '\n## 整体结果',
           '\n| 方法 | 记忆范围 | 完成次数 | 完整运行最终 reward | 平均累计 reward | 平均成功率 |\n|---|---|---:|---|---:|---:|']
    for scope in SCOPES:
        for method in METHODS:
            group=groups[method,scope];avg=statistics.mean(r['total_reward'] for r in group)
            lines.append(f'| {LABEL[method]} | {scope} | {len(group)}/3 | '+', '.join(f"r{r['repeat']}={r['total_reward']:.0f}" for r in group)+f' | {avg:.2f}/140 | {100*avg/140:.2f}% |')
    lines+=['\n**stream：DC 完整运行均值 110.33，高于 ICL 的 104.00，差 6.33 个成功任务（4.52 个百分点）。task_type：ICL 均值 109.00，高于 DC 的 106.00，差 3 个任务（2.14 个百分点）。当前证据不能支持“DC 在所有记忆范围都更好”。**',
            '\nICL / stream 的 repeat 2 和 3 最终 reward 分别为 91、117，差 26；DC / stream 为 98、119、114。同 seed 仍有较大波动，不把这些重复视为独立多 seed 估计，也不进行显著性声明。',
            '\n### 对齐完整重复编号的敏感性检查',
            '\nstream 下双方共同完成的 repeat 为 2、3：ICL 均值 104.00，DC 均值 116.50，差 12.50 个任务（8.93 个百分点）；去掉 DC 独有的完整 repeat 1 后，优势仍存在，但差值明显变化。task_type 下双方共同完成 repeat 1、3，其比较与主图一致。重复编号对齐不等同于多 seed 配对实验。',
            '\n## 各任务类型表现',
            '\n下表合并各组完整运行；每格为成功数/任务数（成功率）。每组重复数与主图一致。',
            '\n| 类型（序号） | ICL / stream | DC / stream | ICL / task_type | DC / task_type |\n|---|---:|---:|---:|---:|']
    for (key,label,_),start,end in zip(TASKS,starts,ends):
        cells=[]
        for scope in SCOPES:
            for method in METHODS:
                group=groups[method,scope];n=sum(r['by_task'][key]['episodes'] for r in group);w=sum(r['by_task'][key]['reward'] for r in group)
                cells.append(f'{w:.0f}/{n}（{100*w/n:.1f}%）')
        lines.append(f'| {label}（{start+1}–{end}） | '+' | '.join(cells)+' |')
    lines+=['\n- stream 下，DC 在灯下查看和双物体任务上明显领先，ICL 则在加热、冷却上更高。因此主图中的差距不是所有任务上均匀累积出来的。\n- task_type 下，DC 的灯下查看更好；ICL 的清洗优势更大，并在加热、冷却略高，最终总 reward 小幅领先。\n- ICL / stream repeat 2 的灯下查看为 0/13、双物体为 0/24；repeat 3 分别为 9/13、23/24。个别任务块整段失败会形成明显平台，也是重复间差异的重要来源。',
            '\n## 错误与记忆更新分析',
            '\n下表仅统计完整运行，非法动作和格式错误以全部尝试数为分母；重复循环按失败局中同一动作至少出现 20 次计数，不要求连续。这些类别可重叠，不能相加视为互斥失败原因。',
            '\n| 方法 / 范围 | 尝试数 | 非法动作 | 格式错误 | 失败局 | 含大量重复的失败局 |\n|---|---:|---:|---:|---:|---:|']
    aggregates={}
    for scope in SCOPES:
        for method in METHODS:
            c=Counter()
            for r in groups[method,scope]:c.update(r['stats'])
            aggregates[method,scope]=c
            lines.append(f"| {LABEL[method]} / {scope} | {c['attempts']} | {c['invalid_actions']}（{100*c['invalid_actions']/c['attempts']:.2f}%） | {c['format_errors']}（{100*c['format_errors']/c['attempts']:.2f}%） | {c['failed_episodes']} | {c['failure_repeating_one_action_20_times']} |")
    lines+=['\nICL 的主要问题是非法动作、合法但无进展的循环以及误判任务已完成，格式错误占比很低。已抽查的 ICL / stream repeat 2 第 37 局，开局历史包含前局 44 次 `examine desklamp 1`，本局又执行了 47 次；失败轨迹继续进入历史，支持错误策略被后续上下文继承的解释，但未通过删除失败历史消融确认因果。',
            '\nDC 在 stream 下的动作非法率较低，task_type 下与 ICL 接近；此外，总结更新有独立瓶颈。下面统计 curator 的更新接受率；总结失败时保留旧 cheatsheet，并不等于本局任务失败：',
            '\n| DC 范围 | 成功替换 cheatsheet | 截断拒绝 | 标签错误拒绝 |\n|---|---:|---:|---:|']
    for scope in SCOPES:
        c=aggregates['dc-cumulative',scope];rej=Counter()
        for r in groups['dc-cumulative',scope]:rej.update(r['curator_rejections'])
        lines.append(f"| {scope} | {c['curator_accepted']}/{c['curator_updates']}（{100*c['curator_accepted']/c['curator_updates']:.2f}%） | {rej['truncated_output']} | {rej['invalid_cheatsheet_tags']} |")
    lines+=['\nstream 中只有约 28% 的总结成功替换，主要被 2048 token 输出上限截断；task_type 的替换接受率约 72%。因此不能假设 DC 每局都成功吸收新经验，也不能仅凭总 reward 优势断言累计知识持续增长。该差异与跨类型累计 cheatsheet 更难压缩的解释一致，但要确认原因需分析长度轨迹或提高总结预算的对照。',
            '\n## 未完成运行及全部重复曲线',
            '\n| 运行 | 已提交进度 | 已获得 reward | 故障 |\n|---|---:|---:|---|',
            '| icl-stream-1 | 69/140 | 55 | CUDA OOM |',
            '| icl-task_type-2 | 0/140 | 0 | 初始化时空闲显存不足 24 GiB |',
            '| dc-cumulative-task_type-2 | 4/140 | 4 | cuDNN SDPA 执行错误 |',
            '\n上述部分分数不能当作完整成功率。主图排除这些运行；下图将部分曲线画到最后已提交任务处，红叉标出停止位置，0 局的运行只在图例说明。',
            '\n![各次运行](individual_runs.png)',
            '\n## 文件与复现',
            '\n- [主图 PNG](cumulative_reward.png) / [PDF](cumulative_reward.pdf) / [SVG](cumulative_reward.svg)。\n- [逐任务 reward 与累积值 CSV](cumulative_reward.csv)。\n- [均值曲线 CSV](mean_curves.csv)：包含固定样本数 n 和 min/max。\n- [统计快照](snapshot.json)：各运行来源、commit 哈希、终局 reward、错误统计。\n- [生成脚本](analyze.py)：在仓库任意目录运行 `python3 /data/jiawei/alfworld/report/icl+dc/analyze.py`；需要本机原始 runs/ 数据、numpy 和 matplotlib。原始 runs/ 不提交到 Git，图表和快照随报告保存。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(f'Validated {len(runs)} runs ({sum(r["complete"] for r in runs)} complete), {len(long_rows)} committed episodes.')
    print('Wrote report, figures, snapshot, and CSV files to',OUT)


if __name__=='__main__':
    main()
