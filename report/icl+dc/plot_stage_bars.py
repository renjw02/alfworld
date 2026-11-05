"""Stage-wise bars for exactly the five-method cumulative-reward cohort.
Run: python3 report/icl+dc/plot_stage_bars.py (numpy and matplotlib).
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
METHODS=['icl','dc-cumulative','memoryless','ascent','success-only-sft']
NAMES=['ICL','DC-Cumulative','Memoryless','ASCENT','Success-only SFT']
COLORS=['#2563eb','#ea580c','#64748b','#16a34a','#9333ea']
TASKS=[('pick_and_place_simple','Place one','简单放置',35),
       ('look_at_obj_in_light','Look under light','灯下查看',13),
       ('pick_clean_then_place_in_recep','Clean & place','清洗后放置',27),
       ('pick_heat_then_place_in_recep','Heat & place','加热后放置',16),
       ('pick_cool_then_place_in_recep','Cool & place','冷却后放置',25),
       ('pick_two_obj_and_place','Place two','双物体放置',24)]


def main():
    source=OUT/'five_methods_seed42_stream.csv'
    with source.open() as f:rows=list(csv.DictReader(f))
    provenance=json.loads((OUT/'five_methods_seed42_stream_provenance.json').read_text())
    by_run=defaultdict(list)
    for row in rows:
        assert row['seed']=='42' and row['memory_scope']=='stream' and row['stream_order']=='task_blocked'
        by_run[row['method'],row['run']].append(row)
    for key,values in by_run.items():
        values.sort(key=lambda r:int(r['task_index']))
        assert [int(r['task_index']) for r in values]==list(range(1,141))
        assert sum(float(r['reward']) for r in values)==float(values[-1]['cumulative_reward'])
    stats=[]
    for method in METHODS:
        runs=[values for (m,_),values in by_run.items() if m==method]
        assert len(runs)==provenance['groups'][method]['n']
        for task,_,_,length in TASKS:
            totals=[]
            for values in runs:
                stage=[r for r in values if r['task_type']==task]
                assert len(stage)==length
                success=sum(float(r['reward']) for r in stage)
                start=int(stage[0]['task_index']);end=int(stage[-1]['task_index'])
                before=float(values[start-2]['cumulative_reward']) if start>1 else 0
                assert success==float(values[end-1]['cumulative_reward'])-before
                totals.append(success)
            stats.append(dict(method=method,task_type=task,runs=len(runs),tasks_per_run=length,
                              mean_completed=float(np.mean(totals)),mean_success_rate=float(np.mean(totals)/length),
                              total_completed=sum(totals),total_tasks=length*len(runs),
                              min_completed=min(totals),max_completed=max(totals)))
        assert np.isclose(sum(x['mean_completed'] for x in stats if x['method']==method),
                          provenance['groups'][method]['mean_final_reward'])
    with (OUT/'five_methods_stage_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(stats[0]),lineterminator='\n');w.writeheader();w.writerows(stats)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'path'})
    x=np.arange(6);width=.155
    for rate in (True,False):
        fig,ax=plt.subplots(figsize=(15,6.6))
        for i,(method,label,color) in enumerate(zip(METHODS,NAMES,COLORS)):
            group=[r for r in stats if r['method']==method]
            y=[r['mean_success_rate']*100 if rate else r['mean_completed'] for r in group]
            bars=ax.bar(x+(i-2)*width,y,width=width*.93,color=color,label=f'{label} (n={group[0]["runs"]})',zorder=3)
            labels=[f'{v:.1f}' if rate or not float(v).is_integer() else f'{v:.0f}' for v in y]
            ax.bar_label(bars,labels=labels,padding=3,fontsize=8.5,color='#334155')
        ax.set_xticks(x,[f'{name}\n({count} tasks)' for _,name,_,count in TASKS])
        ax.set_xlabel('Task stage (task-blocked order)',labelpad=12)
        ax.set_ylabel('Stage success rate (%)' if rate else 'Completed tasks per run (mean)')
        ax.set_ylim(0,112 if rate else 40)
        ax.set_yticks(np.arange(0,101,20) if rate else np.arange(0,36,5))
        ax.grid(axis='y',color='#e2e8f0',lw=.8,zorder=0);ax.set_axisbelow(True)
        for border in np.arange(.5,5.5):ax.axvline(border,color='#e2e8f0',ls=':',lw=.8,zorder=0)
        fig.suptitle(('ALFWorld: Success rate by task stage' if rate else 'ALFWorld: Completed tasks by stage')+
                     '\nSeed 42 | Memory scope: stream | Qwen3.5-9B',fontsize=16,y=.99)
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.89),ncol=5,frameon=False,fontsize=10)
        fig.text(.5,.035,'Same complete runs as the cumulative-reward plot. Bars are same-seed means; n denotes runs.',
                 ha='center',fontsize=10,color='#475569')
        fig.subplots_adjust(left=.065,right=.985,top=.79,bottom=.19)
        stem='five_methods_stage_success_rate' if rate else 'five_methods_stage_completed_tasks'
        for ext in ('png','pdf','svg'):
            path=OUT/f'{stem}.{ext}';fig.savefig(path,dpi=200,bbox_inches='tight')
            if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        plt.close(fig)
    lines=['# 五种方法的阶段完成结果',
           '\n与[五方法累计 reward 图](five_methods_seed42_stream.md)使用完全相同的数据：seed 42、stream、task_blocked，ICL 2 次、DC 3 次，其余各 1 次完整运行。',
           '\n## 阶段成功率',
           '\n![Stage success rate](five_methods_stage_success_rate.png)',
           '\n每组柱对应一个任务阶段；纵轴是该阶段的成功率，不是截至该阶段的累计成功率。阶段长度不同（35/13/27/16/25/24），所以成功率更适合跨阶段比较。图例与标注均为英文，n 为完整运行次数。',
           '\n| 阶段 | ICL | DC-Cumulative | Memoryless | ASCENT | Success-only SFT |\n|---|---:|---:|---:|---:|---:|']
    for task,_,zh,count in TASKS:
        cells=[]
        for method in METHODS:
            r=next(s for s in stats if s['method']==method and s['task_type']==task)
            cells.append(f"{r['mean_completed']:.2f}/{count}（{100*r['mean_success_rate']:.1f}%）")
        lines.append('| '+zh+' | '+' | '.join(cells)+' |')
    lines+=['\n表格中分子为每次运行平均完成任务数，分母为每次运行该阶段的任务数；均值可以是小数。多次重复均为 seed 42，不是多 seed 均值。',
            '\n## 阶段完成数量',
            '\n![Completed tasks](five_methods_stage_completed_tasks.png)',
            '\n这张图直接显示各阶段平均完成了多少任务；各阶段值相加，分别等于原累计曲线终点：ICL 104.00、DC 110.33、Memoryless 78、ASCENT 75、SFT 65。脚本逐运行验证阶段完成数等于累计 reward 在该阶段首尾的差值。',
            '\n## 文件与口径',
            '\n- [成功率 PNG](five_methods_stage_success_rate.png) / [PDF](five_methods_stage_success_rate.pdf) / [SVG](five_methods_stage_success_rate.svg)。\n- [完成数量 PNG](five_methods_stage_completed_tasks.png) / [PDF](five_methods_stage_completed_tasks.pdf) / [SVG](five_methods_stage_completed_tasks.svg)。\n- [阶段统计 CSV](five_methods_stage_results.csv) / [绘图脚本](plot_stage_bars.py)。\n- 运行 `python3 report/icl+dc/plot_stage_bars.py`，只需已提交的合并 CSV 与 provenance，无需原始 runs/。\n- 与原趋势图相同：不同方法来自不同批次、重复次数不同，这些柱状图描述现有结果，不构成控制所有变量的显著性或因果结论。',
            f'\n源 CSV SHA256：`{hashlib.sha256(source.read_bytes()).hexdigest()}`。']
    (OUT/'five_methods_stage_results.md').write_text('\n'.join(lines)+'\n')
    print('Validated 30 method-stage aggregates against all 8 complete runs and cumulative curve endpoints.')


if __name__=='__main__':main()
