# ALFWorld 在线适应实验阶段报告

快照时间：2026-11-04T07:08:15+00:00。主批次：`continual-matrix-20261101T135608Z`。

## 结论与统计口径

主实验矩阵 27 次运行：21 次完整完成、2 次仍运行、4 次失败。本文将运行故障与模型行为错误分开统计。汇总成功率仅纳入完成全部 140 局的运行；失败、运行中的局数单独列出，不混入完整结果。

当前完整结果中，DC-Cumulative / stream 平均成功率最高（78.81%）。未经成功筛选的 History-NTP-Assistant / stream 已完成一次的成功率仅 26.43%，另外两次后期出现大量格式错误、零环境调用。此阶段结果不足以支持统计显著性或泛化结论。

## 实验设置与可比性

- 模型：本地 Qwen3.5-9B；ALFWorld 文本环境；`valid_seen` 140 局、6 类任务。采用先交互、后更新的在线适应协议，不是冻结模型的标准验证分数。
- 动作模式：`admissible`，提示提供合法命令，但生成结果仍需通过格式解析。最多 50 次尝试/局，输入上限 65,536 tokens，动作输出上限 1,024 tokens；贪心生成 `do_sample=False`。
- 全部 27 次运行的 `stream.seed=42`。重复编号 1/2/3 没有对应不同 seed。Python、NumPy、PyTorch 初始化使用 42；每局环境 seed 为 `42 + position`。均值是同 seed 重复均值，不是多 seed 均值。
- Memoryless 使用 `shuffled`；其他方法使用 `task_blocked`，任务类型顺序为 1–6。顺序不同还会改变每个任务对应的环境 seed，因此跨方法比较含有顺序/环境随机性混杂。
- `stream` 跨任务类型保留记忆；`task_type` 在类型切换时重置历史、cheatsheet，NTP 同时恢复初始 adapter/optimizer。
- ICL 窗口 4 局（当前局加最多 3 局历史）；DC-Cumulative 累积 cheatsheet。NTP 使用 LoRA rank=16、alpha=32、dropout=0、学习率 1e-5、chunk=2048；success 版本仅从成功局更新。
- 同 seed、贪心生成仍观察到重复间差异，尚未确定来源；不能直接归因为采样随机性。需要核对恢复路径、运行环境与底层数值确定性。

## 完整运行汇总

| 方法 | 记忆范围 | 完成次数/计划 | 每次成功率（完成运行） | 平均成功率 |
|---|---|---:|---|---:|
| memoryless | stream | 3/3 | 60.00%, 55.71%, 56.43% | 57.38% |
| dc-cumulative | stream | 3/3 | 70.00%, 85.00%, 81.43% | 78.81% |
| history-ntp-assistant-success | stream | 3/3 | 47.86%, 53.57%, 46.43% | 49.29% |
| icl | task_type | 2/3 | 83.57%, 72.14% | 77.86% |
| dc-cumulative | task_type | 2/3 | 77.14%, 74.29% | 75.71% |
| history-ntp-assistant | task_type | 2/3 | 47.14%, 40.00% | 43.57% |
| history-ntp-assistant-success | task_type | 3/3 | 56.43%, 60.00%, 57.14% | 57.86% |
| icl | stream | 2/3 | 65.00%, 83.57% | 74.29% |
| history-ntp-assistant | stream | 1/3 | 26.43% | 26.43% |

## 各次运行状态

运行中进度取最新 console 日志，可能先于事务 checkpoint 持久化；成功数也是日志口径。失败项只表示故障前已完成的局数。

| 运行 | 状态 | 局数/140 | 成功局数 |
|---|---|---:|---:|
| memoryless-stream-1 | completed | 140/140 | 84 |
| icl-stream-1 | failed | 69/140 | 55 |
| dc-cumulative-stream-1 | completed | 140/140 | 98 |
| history-ntp-assistant-stream-1 | running | 138/140 | 35 |
| history-ntp-assistant-success-stream-1 | completed | 140/140 | 67 |
| icl-task_type-1 | completed | 140/140 | 117 |
| dc-cumulative-task_type-1 | completed | 140/140 | 108 |
| history-ntp-assistant-task_type-1 | completed | 140/140 | 66 |
| history-ntp-assistant-success-task_type-1 | completed | 140/140 | 79 |
| icl-stream-2 | completed | 140/140 | 91 |
| dc-cumulative-stream-2 | completed | 140/140 | 119 |
| history-ntp-assistant-stream-2 | running | 126/140 | 38 |
| history-ntp-assistant-success-stream-2 | completed | 140/140 | 75 |
| icl-task_type-2 | failed | 0/140 | 0 |
| dc-cumulative-task_type-2 | failed | 4/140 | 4 |
| history-ntp-assistant-task_type-2 | failed | 0/140 | 0 |
| history-ntp-assistant-success-task_type-2 | completed | 140/140 | 84 |
| icl-stream-3 | completed | 140/140 | 117 |
| dc-cumulative-stream-3 | completed | 140/140 | 114 |
| history-ntp-assistant-stream-3 | completed | 140/140 | 37 |
| history-ntp-assistant-success-stream-3 | completed | 140/140 | 65 |
| icl-task_type-3 | completed | 140/140 | 101 |
| dc-cumulative-task_type-3 | completed | 140/140 | 104 |
| history-ntp-assistant-task_type-3 | completed | 140/140 | 56 |
| history-ntp-assistant-success-task_type-3 | completed | 140/140 | 80 |
| memoryless-stream-2 | completed | 140/140 | 78 |
| memoryless-stream-3 | completed | 140/140 | 79 |

当前仍存活的进程：

- `history-ntp-assistant-stream-1`：PID 1181308，GPU 2，日志进度 138/140。

- `history-ntp-assistant-stream-2`：PID 1541858，GPU 1，日志进度 126/140。

## 按任务类型的完整运行成功率

每格为成功局数/总局数（合并该组完整运行）；各组重复数不同。任务类型依次为：简单放置、灯下查看、清洗后放置、加热后放置、冷却后放置、放置两个物体。

| 方法 / 范围 | 简单放置 | 灯下查看 | 清洗 | 加热 | 冷却 | 双物体 |
|---|---:|---:|---:|---:|---:|---:|
| memoryless / stream | 105/105 (100.0%) | 17/39 (43.6%) | 37/81 (45.7%) | 9/48 (18.8%) | 18/75 (24.0%) | 55/72 (76.4%) |
| dc-cumulative / stream | 103/105 (98.1%) | 30/39 (76.9%) | 68/81 (84.0%) | 30/48 (62.5%) | 36/75 (48.0%) | 64/72 (88.9%) |
| history-ntp-assistant-success / stream | 101/105 (96.2%) | 19/39 (48.7%) | 50/81 (61.7%) | 1/48 (2.1%) | 1/75 (1.3%) | 35/72 (48.6%) |
| icl / task_type | 69/70 (98.6%) | 13/26 (50.0%) | 45/54 (83.3%) | 6/32 (18.8%) | 42/50 (84.0%) | 43/48 (89.6%) |
| dc-cumulative / task_type | 68/70 (97.1%) | 25/26 (96.2%) | 31/54 (57.4%) | 4/32 (12.5%) | 40/50 (80.0%) | 44/48 (91.7%) |
| history-ntp-assistant / task_type | 66/70 (94.3%) | 12/26 (46.2%) | 14/54 (25.9%) | 4/32 (12.5%) | 4/50 (8.0%) | 22/48 (45.8%) |
| history-ntp-assistant-success / task_type | 103/105 (98.1%) | 21/39 (53.8%) | 36/81 (44.4%) | 17/48 (35.4%) | 19/75 (25.3%) | 47/72 (65.3%) |
| icl / stream | 69/70 (98.6%) | 9/26 (34.6%) | 43/54 (79.6%) | 27/32 (84.4%) | 37/50 (74.0%) | 23/48 (47.9%) |
| history-ntp-assistant / stream | 32/35 (91.4%) | 5/13 (38.5%) | 0/27 (0.0%) | 0/16 (0.0%) | 0/25 (0.0%) | 0/24 (0.0%) |

## 错误分析

### 运行与资源故障

| 运行 | 故障前进度 | 日志证据 | 解释 |
|---|---:|---|---|
| icl-stream-1 | 69/140 | `torch.OutOfMemoryError`；申请 976 MiB 时仅余 173 MiB | 同 GPU 上另一个实验约占 53.09 GiB；存在资源竞争，不能算作模型任务失败 |
| icl-task_type-2 | 0/140 | `Less than 24 GiB GPU memory is free` | 后端初始化保护直接拒绝启动 |
| dc-cumulative-task_type-2 | 4/140 | `CUDNN_STATUS_EXECUTION_FAILED_CUDA_DRIVER`，CUDA err 999 | SDPA 执行失败；仅凭日志不能断定是 OOM 或驱动问题 |
| history-ntp-assistant-task_type-2 | 0/140 | cuDNN SDPA，CUDA err 2；`CUDA ran out of memory outside PyTorch allocator` | 日志指向 PyTorch allocator 之外的显存不足 |

Memoryless repeat 2/3 曾失败，随后重试已完成；本报告将其计为完成，不重复计数。未在本次检查中重启或修改任何实验。

### 完整运行的行为错误

来源为各运行 `episodes.csv`。格式错误和非法动作均按尝试数统计；零调用局表示整局没有执行环境动作。非法动作与格式错误是不同字段，不应混为一类。

| 方法 / 范围 | 失败局 | 零调用局 | 尝试数 | 格式错误数（占尝试比例） | 非法动作数（占尝试比例） |
|---|---:|---:|---:|---:|---:|
| memoryless / stream | 179 | 0 | 13419 | 142 (1.06%) | 871 (6.49%) |
| dc-cumulative / stream | 89 | 0 | 10259 | 143 (1.39%) | 722 (7.04%) |
| history-ntp-assistant-success / stream | 213 | 0 | 13437 | 300 (2.23%) | 3165 (23.55%) |
| icl / task_type | 62 | 0 | 6274 | 26 (0.41%) | 633 (10.09%) |
| dc-cumulative / task_type | 68 | 0 | 6320 | 106 (1.68%) | 643 (10.17%) |
| history-ntp-assistant / task_type | 158 | 0 | 9735 | 63 (0.65%) | 2216 (22.76%) |
| history-ntp-assistant-success / task_type | 177 | 0 | 12973 | 103 (0.79%) | 890 (6.86%) |
| icl / stream | 72 | 0 | 6359 | 1 (0.02%) | 1154 (18.15%) |
| history-ntp-assistant / stream | 103 | 0 | 5484 | 1 (0.02%) | 4673 (85.21%) |

已完成的 NTP / stream repeat 3 中，4,673/5,484 次尝试（85.21%）的动作不在当前合法命令列表中，而格式错误仅 1 次。因此其低成功率主要伴随动作合法性问题，不能与另外两次的格式退化合并解释。其清洗、加热、冷却和双物体四类成功率均为 0%。成功筛选版本在 stream 下的加热、冷却成功率也只有 2.1%、1.3%，说明筛选成功轨迹仍未解决这些类别的问题；是否由跨类型更新造成还需对照验证。

### NTP 后期格式退化：已观察到的证据

两个仍运行的 NTP / stream 实验最近连续多局 `attempts=50`、`environment_calls=0`、`end_reason=step_limit`，单局约 41–44 分钟。按 `runner.py` 的执行逻辑，解析失败时不会调用 `env.step`，但仍消耗一次尝试。故这些局的直接失败机制是输出格式/截断问题，而非执行动作后环境无法完成任务。

抽查 repeat 1 的 position=137（第 138 局）轨迹：模型重复输出 `Reasoning:` / `Action:` 段落和 `</think>`，达到生成长度上限；最后一步 `generation_truncated=True`、`action=None`，错误为 `Generation reached max_new_tokens before EOS; output not executed.`。这支持“重复生成 → 截断 → 不执行动作 → 步数耗尽”的解释。单条抽查不能证明所有失败局都是同一具体输出模式。

代码在训练轨迹中保留格式错误的 assistant 原始输出；未经成功筛选的 NTP 也会更新失败局。因此错误输出被再次学习、形成自我强化是合理假设，但尚未通过消融验证，不能视为已确定根因。success-only 的结果更高也不足以单独证明这一机制。

### 建议后续排查（尚未执行）

1. 先保留当前输出和 checkpoint，量化每局截断比例、格式错误及模型更新版本，定位格式退化首次出现的位置。
2. 单独验证训练时过滤截断/错误输出、仅成功局更新等消融；区分生成预算、解析协议与模型退化的影响。
3. 排查同 seed 重复差异：对齐软件版本、GPU/算子配置、checkpoint 恢复路径和初始权重，比较首个分歧步骤。
4. 解决 GPU 资源竞争后补齐 4 次失败运行，并为各方法采用一致任务清单/顺序；另设真正的多 seed 实验。
5. 继续记录类别级表现。当前 `task_type` 重置、类型顺序和学习策略同时影响结果，不能将差异全部解释为遗忘或迁移。

## 数据来源与复核

- 主实验计划：[plan.json](../runs/continual-matrix-20261101T135608Z/plan.json)；调度状态：[status.json](../runs/continual-matrix-20261101T135608Z/status.json)。
- 完整结果：计划中每个 `output_dir` 下的 `summary.json`、`episodes.csv`；配置：各目录 `config.json`。
- 运行中进度及故障栈：计划/状态中对应 `console_log`；早期 ICL 的日志位于 `runs/continual-blocked-single-20261101T131509Z/console/icl/`。
- NTP 抽查：`runs/continual-matrix-20261101T135608Z/history-ntp-assistant-stream-1/traces/20261101T135838.250336Z/episodes/0137/2f5cb7d6d21b44e391c5575c90362a94/episode.json`。
- 协议实现：[runner.py](../alfworld/continual/runner.py)、[methods.py](../alfworld/continual/methods.py)、[model.py](../alfworld/continual/model.py)。
- `runs/` 被 Git 忽略，原始日志链接只在保留运行目录的机器上有效；本报告内的表格作为可提交快照。未使用可能滞后的聚合 CSV 作为本次汇总依据。
