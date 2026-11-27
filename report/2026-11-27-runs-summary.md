# runs/ 实验汇总（2026-11-27）

统计时间：2026-11-27 03:25 UTC。仅汇总当前本地 `runs/` 的落盘结果，不将旧报告中已不在该目录的实验并入本次统计。

## 1. 当前状态与主要结论

- **21 个 heat-place 在线实验全部完成 200/200 episode**，共 4,200 次 episode 执行；另有 2 个 frozen transfer 评测，各完成 31/31，共 62 次。所有主实验的 `episodes.csv` 与 `summary.json` 成功率一致，position 连续且无重复。
- 检查时本机进程列表未发现 `alfworld.continual` 或 `run_continual` 进程；结合完整结果，当前目录中的主实验均已完成。这里不推断其他机器是否仍有任务运行。
- **当前最高为 ICL 前4条 / 192K：183/200（91.5%）**；Ascent + ICL 前3条为 181/200（90.5%）；标准 ICL 前3条为 175/200（87.5%）。前两者只差 2 个任务，不能据此宣称统计显著优劣。
- 标准 Ascent 为 68.0%，Memoryless 为 35.0%。当前任务流上，历史上下文带来的收益很大；但 ICL 前1条、前2条分别只有 53.0%、21.5%，说明收益并非随窗口大小平滑变化。
- feedback-v2 / FFN 版本改善了 Negative UL 和 Projected KL；SFT + Negative UL 从初版 13.0% 提升到 v2 的 54.0%，但仍落后于 ICL / Ascent 系列。
- 冻结迁移到 clean valid_unseen：ICL 24/31（77.4%），Ascent 19/31（61.3%）；这不是全部来自未见房间的评测，需分房间解读。

## 2. 实验口径与可比性

主实验均使用 Qwen3.5-9B、HF backend、seed=42、`heat_place_200` manifest、`action_mode=admissible`、每 episode 最多 50 步、最多生成 1,024 tokens。21 个实验的 episode ID **及顺序完全相同**；将 ID 按换行拼接后的 SHA-256 为 `3dedf1874dca3409fb0cb3cf896faf79626d18cda909c9d42e152c792d698b2d`。

这 200 条都是 `pick_heat_then_place_in_recep`，来源为 train 161 条、valid_seen 16 条、valid_unseen 23 条。因此本报告衡量的是**固定混合任务流上的在线适应表现**，不能当成标准 held-out test 成绩。每个方法沿任务流更新历史或参数，后面的 episode 并非独立评测。

`window_episodes` 包含当前 episode：w=2/3/4/5 对应保留前 1/2/3/4 条历史。表中的“前N条”统一使用实际历史数量；依据运行时 `source_snapshot/continual/methods.py` 的 `keep = window_episodes - 1`。ICL 配置均未启用 success-only 历史筛选。

大多数推理上下文上限为 65,536 tokens；ICL 前4条使用 196,608。SFT + Negative UL 两版为 64,512，溢出策略为 error；其他主实验为 truncate_oldest。SFT 的 4K/32K/64K 是训练序列上限，不能与推理上下文上限混为一谈。Ascent 的配置为 forward KL、学习率 1.5e-4、每次 2 optimizer steps；通用训练配置为 LoRA rank=16、alpha=32、学习率 1e-5。

不同批次存在源码、模板、训练目标和上下文预算差异，不是所有行都构成单变量消融。多个运行的 `metadata.json` 标记 `git_dirty=true`，复现时应同时保存/参考 `source_snapshot/` 和 `working-tree.patch`，不能只依赖 git commit。

## 3. 完整主实验结果

每行均为 200/200 已完成。“分段成功数”依次对应 episode 1–50 / 51–100 / 101–150 / 151–200，每段分母为 50。“记录耗时”是 CSV 中 `interaction_seconds + training_seconds` 的总和，不含模型加载等开销，也不是 GPU-hours 或严格可比的速度基准。链接指向原始实验目录。

| ID | 实验 | 成功数 | 成功率 | 分段成功数（每段50） | 记录耗时（小时） |
|---|---|---:|---:|---|---:|
| R01 | [Ascent repeat1](../runs/heat-place-200-accelerated-20261122T033018Z/ascent-seed42-repeat1/) | 136/200 | 68.0% | 26/30/40/40 | 6.30 |
| R02 | [Ascent repeat2](../runs/heat-place-200-accelerated-20261122T033018Z/ascent-seed42-repeat2/) | 136/200 | 68.0% | 26/30/40/40 | 6.36 |
| R03 | [ICL 前3条 repeat1](../runs/heat-place-200-accelerated-20261122T033018Z/icl-seed42-repeat1/) | 175/200 | 87.5% | 38/41/48/48 | 5.69 |
| R04 | [ICL 前3条 repeat2](../runs/heat-place-200-accelerated-20261122T033018Z/icl-seed42-repeat2/) | 175/200 | 87.5% | 38/41/48/48 | 5.48 |
| R05 | [Memoryless repeat1](../runs/heat-place-200-accelerated-20261122T033018Z/memoryless-seed42-repeat1/) | 70/200 | 35.0% | 19/16/16/19 | 11.27 |
| R06 | [Memoryless repeat2](../runs/heat-place-200-accelerated-20261122T033018Z/memoryless-seed42-repeat2/) | 70/200 | 35.0% | 19/16/16/19 | 11.52 |
| R07 | [Ascent + ICL 前3条](../runs/heat-place-200-ascent-icl-w4-seed42-20261125T122221Z/) | 181/200 | 90.5% | 42/47/44/48 | 4.52 |
| R08 | [DC cumulative](../runs/heat-place-200-dc-sft-ul-pkl-4steps-20261123T075718Z/dc-cumulative-seed42/) | 101/200 | 50.5% | 28/5/25/43 | 8.32 |
| R09 | [Negative UL 初版](../runs/heat-place-200-dc-sft-ul-pkl-4steps-20261123T075718Z/negative-ul-seed42/) | 59/200 | 29.5% | 27/14/9/9 | 18.09 |
| R10 | [Projected KL 初版](../runs/heat-place-200-dc-sft-ul-pkl-4steps-20261123T075718Z/projected-kl-seed42/) | 27/200 | 13.5% | 22/5/0/0 | 14.91 |
| R11 | [Success-only SFT（局部历史）](../runs/heat-place-200-dc-sft-ul-pkl-4steps-20261123T075718Z/success-only-sft-seed42/) | 105/200 | 52.5% | 22/25/29/29 | 12.60 |
| R12 | [Negative UL feedback-v2 / FFN](../runs/heat-place-200-feedback-v2-ffn-4steps-20261124T051709Z/negative-ul-seed42/) | 84/200 | 42.0% | 23/19/20/22 | 13.16 |
| R13 | [Projected KL feedback-v2 / FFN](../runs/heat-place-200-feedback-v2-ffn-4steps-20261124T051709Z/projected-kl-seed42/) | 72/200 | 36.0% | 24/17/12/19 | 13.34 |
| R14 | [ICL 前1条](../runs/heat-place-200-fullhistory-sft-icl-prev1-20261125T012832Z/icl-prev1-seed42/) | 106/200 | 53.0% | 25/33/37/11 | 7.41 |
| R15 | [SFT full-history 4K](../runs/heat-place-200-fullhistory-sft-icl-prev1-20261125T012832Z/success-only-sft-fullhistory-seed42/) | 68/200 | 34.0% | 26/10/15/17 | 11.32 |
| R16 | [ICL action-only 前3条](../runs/heat-place-200-icl-action-only-w4-20261125T014901Z/icl-action-only-seed42/) | 5/200 | 2.5% | 5/0/0/0 | 8.51 |
| R17 | [ICL 前2条](../runs/heat-place-200-icl-prev2-20261126T030948Z/icl-prev2-seed42/) | 43/200 | 21.5% | 8/0/0/35 | 13.51 |
| R18 | [ICL 前4条 / 192K](../runs/heat-place-200-icl-prev4-192k-20261126T031634Z/icl-prev4-seed42/) | 183/200 | 91.5% | 44/47/49/43 | 3.53 |
| R19 | [SFT full-history 32K](../runs/heat-place-200-sft-fullhistory-32k-20261126T024506Z/success-only-sft-seed42/) | 96/200 | 48.0% | 31/16/18/31 | 16.19 |
| R20 | [SFT + Negative UL v2](../runs/heat-place-200-sft-negative-ul-ffn-v2-seed42-20261125T012133Z/) | 108/200 | 54.0% | 31/19/31/27 | 7.98 |
| R21 | [SFT + Negative UL 初版](../runs/heat-place-200-sft-negative-ul-seed42/) | 26/200 | 13.0% | 19/7/0/0 | 30.14 |

所有主实验中的失败 `end_reason` 都是 `step_limit`，不是记录中的程序崩溃。Memoryless、ICL、Ascent 的 repeat1/2 成功数及四段统计分别相同；它们使用相同 seed 和同一任务顺序，不能算作两个独立 seed，更不能用来估计跨 seed 方差。

## 4. 各实验方向的观察

### 4.1 ICL 与 Ascent

- 标准 ICL（R03/R04）比 Memoryless（R05/R06）高 **52.5 个百分点**；Ascent（R01/R02）高 **33.0 个百分点**。
- Ascent + ICL（R07）相对标准 ICL 多成功 6 条（+3.0 个百分点），相对 Ascent 多 45 条（+22.5 个百分点）。这是本次组合运行的观察结果，尚需同版本、多 seed 对照确认。
- ICL 前4条 / 192K（R18）比前3条 / 64K 多 8 条（+4.0 个百分点），同时改变了历史长度和上下文预算，不能把提升单独归因于其中一个因素。
- 前1条（R14）最后一段降至 11/50；前2条（R17）中间两段均为 0/50，末段恢复至 35/50。该现象提示失败历史、动作循环及上下文处理值得排查，不能只看总体平均。
- action-only（R16）仅 5/200，后150条全部失败；其格式错误计数为0，但 invalid actions 累计819。说明“格式合法”并不意味着任务策略有效；这一结果仅针对当前 action-only 提示和运行设置，不能泛化为所有不输出推理的策略。

### 4.2 负反馈训练

- Negative UL：初版 R09 为 29.5%，v2 R12 为 42.0%，提升12.5个百分点。
- Projected KL：初版 R10 为13.5%，v2 R13 为36.0%，提升22.5个百分点。初版后100条零成功，v2未出现同样的整段归零，但总体仍接近Memoryless的35.0%。
- SFT + Negative UL：R21 初版13.0%，R20 v2 54.0%，提升41.0个百分点。初版后100条零成功，累计2,827次格式错误；v2累计44次，输出稳定性明显改善，但这些相关变化不足以单独证明根因。
- 4-step Negative UL / Projected KL 对应各自实际训练配置字段；不能因为批次名包含“4steps”，就认为该批次所有方法都用了同一种4步更新。

已有 [feedback-v2 轨迹诊断](../runs/heat-place-200-feedback-v2-ffn-4steps-20261124T051709Z/trajectory-diagnosis-v2.json) 中，Negative UL 的116个失败episode里有110个被计为 `no_negatives`，96个被计为 `no_heat`。这支持一个待验证的解释：许多失败来自未推进关键子目标，而非触发显式负反馈；仅惩罚被检测到的坏动作可能覆盖不了这类失败。它是诊断线索，不是因果结论。

### 4.3 SFT 与 DC

- 局部历史 success-only SFT（R11）为52.5%；full-history 4K（R15）为34.0%；full-history 32K（R19）为48.0%。32K比4K恢复14.0个百分点，但仍未超过局部历史版本。
- full-history 两组的训练 `max_seq_tokens` 分别为4,096和32,768，均配置 `updates_per_batch=4`。方法名相同不代表训练样本构造相同，应结合源码快照核对；结果不能简单概括为“历史越长越好”。
- DC cumulative（R08）为50.5%，第二段仅5/50，末段43/50，阶段波动很大。

## 5. Frozen heat → clean 迁移

来源为 accelerated 批次的 Ascent repeat1 与 ICL repeat1 在完成200条heat任务后的状态，并非当前排名最高的 Ascent+ICL 或 ICL前4条模型。目标为31条clean valid_unseen任务。

依据各自 `provenance.json`，评测期间不更新参数、不保留测试历史；ICL固定使用源流最后3条历史，Ascent使用源checkpoint且不携带历史。

| 方法 | 总体 | 源流见过的房间 FloorPlan10 | 源流未见房间 FloorPlan424 |
|---|---:|---:|---:|
| [Ascent](../runs/frozen-heat200-to-clean-unseen-repeat1-20261126T025854Z/ascent/) | 19/31（61.3%） | 12/21（57.1%） | 7/10（70.0%） |
| [ICL](../runs/frozen-heat200-to-clean-unseen-repeat1-20261126T025854Z/icl/) | 24/31（77.4%） | 18/21（85.7%） | 6/10（60.0%） |

ICL总体多成功5条，但在真正未见房间的10条任务上少成功1条。样本小，且总体差异主要来自已见房间；不能据此宣称ICL跨房间泛化更强。目录名中的unseen指目标数据split，不等于相对源流从未出现过该房间。当前目录无同协议Memoryless迁移对照，不能量化相对原始模型的迁移收益。

## 6. 上下文容量检查与辅助目录

[icl4-context-check-20261126](../runs/icl4-context-check-20261126/) 是容量探测，不是额外的成功率实验。其 `lengths.json` 用既有ICL前3条轨迹重建前4条上下文，200个样本中28个超过65,536 tokens、3个超过131,072；最大163,339，中位数33,070，p95为86,646。这解释了为何另开192K配置，但不是新192K运行的实际输入长度分布。

`memory.json` 记录在总显存79.18 GiB的设备上，131,072输入+1,024输出探测成功，峰值allocated/reserved为37.85/44.64 GiB；196,608输入+1,024输出也成功，为48.41/58.48 GiB。该数据是容量可行性检查，不代表所有轨迹的显存峰值。

`runs/launches/` 保存3个启动记录及stdout，与主实验对应，不重复计作实验；各批次的smoke/tests日志及诊断文件也不计入成功率分母。

## 7. 下一步建议

1. 固定代码版本、上下文上限及任务流协议，优先复跑 ICL前3条、ICL前4条、Ascent+ICL 的多个seed，确认91.5%与90.5%的差异是否稳定。
2. 对ICL前2条的连续失败段、action-only的后150条失败及早期负反馈版本，针对重复动作、关键子目标、上下文截断和实际训练目标做配对轨迹检查。
3. 负反馈方向优先解决“失败但未产生训练信号”的覆盖问题；SFT方向先核对full-history训练/推理前缀和长序列处理，再增加更新量。
4. 迁移评测补齐Memoryless及最佳组合方法，并扩大真正未见房间任务数，单独报告源流见过/未见房间的结果。

## 8. 数据来源与复核方式

主表逐行读取各实验的 `episodes.csv`，以 `success == True` 计数，并与同目录 `summary.json` 校验；分段按CSV顺序每50条计算。配置来自各自 `config.json`，版本信息来自 `metadata.json`，方法语义参考运行目录保存的源码。迁移结果以 `results.json` 与 `summary.json` 交叉核对，协议以 `provenance.json` 为准。

本报告只描述本地结果快照；成功率不是多seed均值，阶段差异也可能包含任务难度差异，不能单凭后段提升就证明学习收益。旧报告保留原有内容，本次未改动训练代码、配置或运行产物。
