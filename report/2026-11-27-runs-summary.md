# runs/ 实验汇总（2026-11-27，更新至2026-11-28）

统计时间：2026-11-28 08:00 UTC。汇总当前本地 `runs/`（含软链接目标）中已完成的实验；未完成、中断或被替代的运行不纳入本报告。保留原有R01–R21编号，新增完整主实验为R22。

## 1. 当前状态与主要结论

- **22 个完整 heat-place 在线实验均完成 200/200 episode**，共4,400次执行；另有10组已完成的 frozen transfer 评测（5类任务×2种方法），共222次；6组已完成的 Memoryless valid_unseen 对照，共134次。合计4,756次已完成实验中的episode执行，重复评测不视为独立任务。所有主实验的 `episodes.csv` 与 `summary.json` 成功率一致，position连续且无重复。
- 本次新增：Ascent milestones完整重跑1组、其余4类任务的冻结迁移8组、Memoryless对照6组；本报告不作全部作业均已结束的判断。
- **当前最高为 ICL 前4条 / 192K：183/200（91.5%）**；Ascent + ICL 前3条为 181/200（90.5%）；标准 ICL 前3条为 175/200（87.5%）。前两者只差 2 个任务，不能据此宣称统计显著优劣。
- 原始两次 Ascent 为68.0%；新增保留里程碑checkpoint的Ascent重跑为 **72.0%（144/200）**。Memoryless为35.0%。当前任务流上，历史上下文带来的收益很大；但 ICL 前1条、前2条分别只有 53.0%、21.5%，说明收益并非随窗口大小平滑变化。
- feedback-v2 / FFN 版本改善了 Negative UL 和 Projected KL；SFT + Negative UL 从初版 13.0% 提升到 v2 的 54.0%，但仍落后于 ICL / Ascent 系列。
- 冻结迁移的5类非heat任务合计111条：ICL **95/111（85.6%）**，Ascent **75/111（67.6%）**，新补齐的Memoryless为 **74/111（66.7%）**。ICL的总体优势主要来自clean和cool；pick-two上两个迁移方法都低于Memoryless，不能概括为所有任务都有迁移收益。

## 2. 实验口径与可比性

主实验均使用 Qwen3.5-9B、HF backend、seed=42、`heat_place_200` manifest、`action_mode=admissible`、每 episode 最多 50 步、最多生成 1,024 tokens。22 个完整实验的 episode ID **及顺序完全相同**；将 ID 按换行拼接后的 SHA-256 为 `3dedf1874dca3409fb0cb3cf896faf79626d18cda909c9d42e152c792d698b2d`。

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
| R22 | [Ascent milestones重跑](../runs/heat-place-200-ascent-milestones-seed42-20261127T084728Z/) | 144/200 | 72.0% | 26/36/39/43 | 5.12 |

所有主实验中的失败 `end_reason` 都是 `step_limit`，不是记录中的程序崩溃。Memoryless、ICL、Ascent 的 repeat1/2 成功数及四段统计分别相同；它们使用相同 seed 和同一任务顺序，不能算作两个独立 seed，更不能用来估计跨 seed 方差。

## 4. 各实验方向的观察

### 4.1 ICL 与 Ascent

- 标准 ICL（R03/R04）比 Memoryless（R05/R06）高 **52.5 个百分点**；Ascent（R01/R02）高 **33.0 个百分点**。
- Ascent + ICL（R07）相对标准 ICL 多成功 6 条（+3.0 个百分点），相对 Ascent 多 45 条（+22.5 个百分点）。这是本次组合运行的观察结果，尚需同版本、多 seed 对照确认。
- ICL 前4条 / 192K（R18）比前3条 / 64K 多 8 条（+4.0 个百分点），同时改变了历史长度和上下文预算，不能把提升单独归因于其中一个因素。
- 前1条（R14）最后一段降至 11/50；前2条（R17）中间两段均为 0/50，末段恢复至 35/50。该现象提示失败历史、动作循环及上下文处理值得排查，不能只看总体平均。
- action-only（R16）仅 5/200，后150条全部失败；其格式错误计数为0，但 invalid actions 累计819。说明“格式合法”并不意味着任务策略有效；这一结果仅针对当前 action-only 提示和运行设置，不能泛化为所有不输出推理的策略。

新增R22保留position 50/100/150/200的checkpoint，最终144/200（72.0%），比R01/R02多8条（+4.0个百分点）。四段为26/36/39/43，后100条82/100。其 [启动记录](../runs/launches/heat-place-200-ascent-milestones-seed42-20261127T084728Z/launch.json) 明确注明历史同seed运行不能逐token复现，因此后续checkpoint比较应使用这次重跑自身的200位置checkpoint作对照。R01/R02的重复一致性不能推广到不同版本/运行环境下的所有seed42运行。

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

## 5. Frozen heat → 五类任务迁移与Memoryless对照

### 5.1 协议与数据范围

冻结迁移来源仍为accelerated批次的Ascent repeat1与ICL repeat1完成200条heat任务后的状态，不是R22重跑或当前最高分的ICL前4条。目标为valid_unseen中的clean、pick、cool、look、pick-two，共111条。新增四类迁移使用 `frozen_heat200_to_task_unseen_v1`，此前clean使用 `frozen_heat200_to_clean_unseen_v1`；两批都固定源position=200，测试期间不更新参数、不保留测试历史。ICL固定使用源流最后3条历史，Ascent使用源checkpoint且不携带历史。

新增Memoryless对照使用 `memoryless_task_unseen_v1`，不加载adapter、不保留历史、不更新参数，模型为同一Qwen3.5-9B，react/admissible、50步、65,536输入上限与迁移评测一致。已核对五类任务的 `results.json`：三种方法的episode ID及顺序逐项相同。各类任务仍只有一个seed；版本与运行环境差异不因此被消除。

### 5.2 按任务比较（全部完成）

| 目标任务 | 条数 | Memoryless | Frozen Ascent | Frozen ICL |
|---|---:|---:|---:|---:|
| clean_place | 31 | [9/31（29.0%）](../runs/memoryless-all-unseen-seed42-20261127T091008Z/clean_place/) | [19/31（61.3%）](../runs/frozen-heat200-to-clean-unseen-repeat1-20261126T025854Z/ascent/) | [24/31（77.4%）](../runs/frozen-heat200-to-clean-unseen-repeat1-20261126T025854Z/icl/) |
| pick_place | 24 | [24/24（100.0%）](../runs/memoryless-all-unseen-seed42-20261127T091008Z/pick_place/) | [24/24（100.0%）](../runs/frozen-heat200-to-remaining-unseen-repeat1-20261127T033536Z/pick_place/ascent/) | [24/24（100.0%）](../runs/frozen-heat200-to-remaining-unseen-repeat1-20261127T033536Z/pick_place/icl/) |
| cool_place | 21 | [9/21（42.9%）](../runs/memoryless-all-unseen-seed42-20261127T091008Z/cool_place/) | [8/21（38.1%）](../runs/frozen-heat200-to-remaining-unseen-repeat1-20261127T033536Z/cool_place/ascent/) | [19/21（90.5%）](../runs/frozen-heat200-to-remaining-unseen-repeat1-20261127T033536Z/cool_place/icl/) |
| look_at_light | 18 | [17/18（94.4%）](../runs/memoryless-all-unseen-seed42-20261127T091008Z/look_at_light/) | [17/18（94.4%）](../runs/frozen-heat200-to-remaining-unseen-repeat1-20261127T033536Z/look_at_light/ascent/) | [16/18（88.9%）](../runs/frozen-heat200-to-remaining-unseen-repeat1-20261127T033536Z/look_at_light/icl/) |
| pick_two_place | 17 | [15/17（88.2%）](../runs/memoryless-all-unseen-seed42-20261127T091008Z/pick_two_place/) | [7/17（41.2%）](../runs/frozen-heat200-to-remaining-unseen-repeat1-20261127T033536Z/pick_two_place/ascent/) | [12/17（70.6%）](../runs/frozen-heat200-to-remaining-unseen-repeat1-20261127T033536Z/pick_two_place/icl/) |
| **五类合计（按episode加权）** | **111** | **74/111（66.7%）** | **75/111（67.6%）** | **95/111（85.6%）** |

ICL相对Memoryless多21条（+18.9个百分点），其中clean多15条、cool多10条，pick相同，look少1条、pick-two少3条。Ascent合计仅多1条（+0.9个百分点）：clean多10条，但cool少1条、pick-two少8条，pick和look持平。**迁移收益高度依赖目标任务，不能用单一总分掩盖负迁移。** 这里的差异是当前配对任务集的观察值，不代表统计显著性结论。

Memoryless另外完成了 [heat_place对照](../runs/memoryless-all-unseen-seed42-20261127T091008Z/heat_place/)，10/23（43.5%）。它属于同任务类型的valid_unseen评测，不并入上述五类跨任务合计。Memoryless六类合计84/134（62.7%）；不能直接与五类迁移的111条分母比较，也不能与混合200条任务流的35.0%直接比较。

### 5.3 按源任务流房间暴露分组

“已见/未见”均相对于原始heat200源流；Memoryless的房间分组也使用同一源流作参照，并不表示它进行过该源流训练。valid_unseen这一split名称不等于相对源流从未出现过该房间。

| 任务 | 源流已见/未见条数 | Memoryless已见 / 未见成功数 | Ascent已见 / 未见成功数 | ICL已见 / 未见成功数 |
|---|---|---|---|---|
| clean_place | 21 / 10 | 6/21 / 3/10 | 12/21 / 7/10 | 18/21 / 6/10 |
| pick_place | 9 / 15 | 9/9 / 15/15 | 9/9 / 15/15 | 9/9 / 15/15 |
| cool_place | 21 / 0 | 9/21 / — | 8/21 / — | 19/21 / — |
| look_at_light | 0 / 18 | — / 17/18 | — / 17/18 | — / 16/18 |
| pick_two_place | 3 / 14 | 2/3 / 13/14 | 0/3 / 7/14 | 2/3 / 10/14 |
| **五类合计** | **54 / 57** | **26/54 / 48/57** | **29/54 / 46/57** | **48/54 / 47/57** |

按房间分组后，ICL的优势集中于源流已见房间（88.9%，Memoryless为48.1%）；在未见房间，ICL为82.5%、Ascent为80.7%、Memoryless为84.2%。任务构成在两个房间组之间不同，样本也较小，因此这些数据不支持“跨房间泛化普遍更强”的结论。cool全部在已见的FloorPlan10，look全部在未见的FloorPlan308；clean仍是FloorPlan10的21条和FloorPlan424的10条。

## 6. 上下文容量检查与辅助目录

[icl4-context-check-20261126](../runs/icl4-context-check-20261126/) 是容量探测，不是额外的成功率实验。其 `lengths.json` 用既有ICL前3条轨迹重建前4条上下文，200个样本中28个超过65,536 tokens、3个超过131,072；最大163,339，中位数33,070，p95为86,646。这解释了为何另开192K配置，但不是新192K运行的实际输入长度分布。

`memory.json` 记录在总显存79.18 GiB的设备上，131,072输入+1,024输出探测成功，峰值allocated/reserved为37.85/44.64 GiB；196,608输入+1,024输出也成功，为48.41/58.48 GiB。该数据是容量可行性检查，不代表所有轨迹的显存峰值。

`runs/launches/` 保存启动记录、supervisor记录及stdout，与结果目录对应，不重复计作实验；各批次的smoke/tests日志及诊断文件也不计入成功率分母。Ascent milestones和Memoryless新结果通过 `runs/` 下的软链接指向 `/home/sigma/alfworld-runs/`，本次扫描已跟随软链接核对，避免漏掉这两批已完成结果。

## 7. 下一步建议

1. 固定代码版本、上下文上限及任务流协议，优先复跑 ICL前3条、ICL前4条、Ascent+ICL 的多个seed，确认91.5%与90.5%的差异是否稳定。
2. 对ICL前2条的连续失败段、action-only的后150条失败及早期负反馈版本，针对重复动作、关键子目标、上下文截断和实际训练目标做配对轨迹检查。
3. 负反馈方向优先解决“失败但未产生训练信号”的覆盖问题；SFT方向先核对full-history训练/推理前缀和长序列处理，再增加更新量。
4. Memoryless对照已补齐；后续补充最佳组合方法、多个seed及更多未见房间，并重点验证pick-two的负迁移。Ascent中间checkpoint对照应使用R22自身的200位置checkpoint，避免混用原始68.0%运行。

## 8. 数据来源与复核方式

主表逐行读取各实验的 `episodes.csv`，以 `success == True` 计数，并与同目录 `summary.json` 校验；分段按CSV顺序每50条计算。配置来自各自 `config.json`，版本信息来自 `metadata.json`，方法语义参考运行目录保存的源码。迁移及Memoryless结果以 `results.json` 与 `summary.json` 交叉核对，协议以 `provenance.json` 为准；三方法比较还校验了同任务的episode ID及顺序。扫描包含 `runs/` 下的软链接目标，共核对38组已完成结果（22组主实验、10组迁移、6组Memoryless）。

本报告只描述本地结果快照；成功率不是多seed均值，阶段差异也可能包含任务难度差异，不能单凭后段提升就证明学习收益。旧报告保留原有内容，本次未改动训练代码、配置或运行产物。
