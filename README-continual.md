# ALFWorld streaming continual learning

本框架比较跨 episode 更新 **context** 与更新 **LoRA 参数**。数据来自 `valid_seen`，
结果是在线适应（streaming adaptation）结果，不是冻结模型的标准验证成绩。
初始实验使用本地 `/data/jiawei/models/Qwen3.5-9B`，仅运行 ALFWorld 文本环境。

## 安装和最小运行

在仓库根目录，使用 Python 3.11+：

```bash
uv sync --locked --extra continual
uv run --locked --extra continual alfworld-download --text-only
uv run --locked --extra continual python -m alfworld.continual \
  --config configs/continual.yaml \
  --method memoryless --stream-order shuffled --action-mode admissible \
  --device cuda:0 --output-dir runs/memoryless-shuffled-admissible
```

默认 BF16，不量化；选择有足够显存的 GPU。加载前要求至少 24 GiB 空闲显存，
这只是最低检查，不代表长上下文或训练的显存保证；完整实验建议使用空闲 H100 80GB。
输入过长、显存不足、模型加载不完整时直接报错，不静默量化或截断历史。

`continual` extra 独立于 `full/vis`，不需要 AI2-THOR、视觉检测器或训练 BUTLER。
基础 `uv sync --locked` 仍然只安装原来的文本环境依赖。
底层 Qwen3.5 线性注意力在没有可选优化 kernel 时使用正确但较慢的 PyTorch 实现。
当前实现优先验证实验语义，没有自动安装 `flash-linear-attention` 或 `causal-conv1d`。

## 方法和交互协议

| 方法 | 每步输入 | 局后更新 |
| --- | --- | --- |
| `memoryless` | 当前局完整历史 | 无 |
| `icl` | 最近 K−1 局 + 当前局完整历史 | 保存本局；淘汰最旧历史 |
| `history-ntp` | 当前局完整历史 | 用本局完整文本执行一轮 LoRA NTP |
| `history-ntp-assistant` | 当前局完整历史 | 所有 episode，仅训练 assistant reasoning/action |
| `history-ntp-assistant-success` | 当前局完整历史 | 仅成功 episode，训练 assistant reasoning/action |

`window_episodes` 默认 4，**包含当前局**；K=1 的 ICL 与 memoryless 输入一致。
所有历史保留 reasoning、action、观察、分数、格式反馈和终局反馈。
每个方法在独立输出目录从同一初始模型开始。切换任务类型时不清除历史或参数。

模型每步显式输出：

```text
Reasoning: I should inspect the countertop to locate the mug.
Action: go to countertop 1
```

默认 greedy decoding，关闭模型 chat template 的内置 thinking，使用显式 `Reasoning:`。
输出预算 **1024 tokens**，同时包含 reasoning 和 action。默认输入预算 100000 tokens，
额外预留输出空间；超过预算就报错，不偷偷缩短配置指定的 ICL 窗口。

Reasoning 可多行，Action 必须是最后一行且仅出现一次。只执行 action，绝不把 reasoning
交给环境。缺失字段、格式错误或达到输出上限而未输出 EOS 时，不执行动作，返回格式反馈；
默认最多 50 次尝试，格式错误也计入。格式正确的非法动作仍交给环境处理，不替换动作。

动作模式可配置：

- `free`：给出固定动作语法，不展示当前合法动作列表。
- `admissible`（默认）：每步直接展示环境给出的当前合法动作列表，要求原样选择其中一条；不附带通用动作语法示例。

两种模式都将合法动作列表保存到日志以便分析，不提供专家解法或额外评审。

## Stream 顺序及对比实验

默认 seed=42，单遍串行运行全部 140 个可用 `valid_seen` 游戏：

- `shuffled`：先按相对路径排序，再固定种子打乱一次。
- `task_blocked`：按 `task_order` 排列任务块；块内排序后用固定种子打乱。
  默认任务 ID 为 `[1, 2, 3, 4, 5, 6]`，对应放置、灯下查看、清洗、加热、冷却、放置两个物体。

每次运行保存完整 `manifest.json`，包括 episode 顺序和数据 SHA256。不同方法可显式复用：

```bash
uv run --locked --extra continual python -m alfworld.continual \
  --method icl --window-episodes 4 --stream-order shuffled --action-mode admissible \
  --manifest runs/memoryless-shuffled-admissible/manifest.json \
  --output-dir runs/icl-shuffled-admissible

uv run --locked --extra continual python -m alfworld.continual \
  --method history-ntp --stream-order shuffled --action-mode admissible \
  --manifest runs/memoryless-shuffled-admissible/manifest.json \
  --output-dir runs/ntp-shuffled-admissible
```

manifest 中的 seed、顺序类型、任务块顺序必须与配置一致，否则报错。三种方法的实际交互
会随策略变化；共享的是任务顺序、交互协议和初始模型，不要求产生相同轨迹。
通过 `--stream-order task_blocked` 可切换任务块顺序；`--action-mode free` 可显式切换为不提供合法命令列表。

其他常用参数：`--limit 2 --max-steps 3` 用于冒烟验证，`--device cuda:6` 指定设备，
`--data-dir /path/to/alfworld` 指定数据根目录。数据路径优先级：CLI > 调用者设置的
`ALFWORLD_DATA` > YAML。完整参数和训练超参数见 [`configs/continual.yaml`](configs/continual.yaml)。

## NTP 和参数设置

通过 `--method history-ntp-assistant` 或 `--method history-ntp-assistant-success`
选择新增版本。两者保留完整对话作为输入，只对 assistant 原始正文（包括 reasoning、
action 和 `<think>` 内容）计算 loss；system、user、环境反馈及模板标记的 label
设为 -100。仅完整落在 assistant 正文内的 token 参与 loss，跨正文边界的 token 被 mask。
不含有效预测目标的分块跳过更新。`update.json` 保存 `training_labels` 和 `loss_scope`。
成功版本依据环境记录的 `episode.outcome.success` 筛选，失败局记录跳过原因，
不更新参数、优化器或模型版本。两个版本均支持 checkpoint/resume。

以下完整文本训练说明针对原有 `history-ntp`：

默认 LoRA rank=16、alpha=32、dropout=0，覆盖语言模型线性层，排除输出头。
基础模型冻结；不合并 adapter，不回放旧 episode，成功和失败都训练。
AdamW 默认 lr=1e-5、weight_decay=0、梯度裁剪=1.0，优化器状态跨局保留。

每个 episode 结束后，序列化一次完整对话：系统指令、任务和初始观察、各步原始
reasoning/action、反馈、终局反馈。所有非 padding 文本 token 都参与 NTP，**不做
assistant-only mask**。不会把每步重复包含历史的推理 prompt 展开成多份训练样本。
即使模型输出格式错误，原始文本也会保留；chat template 不会删除其中的 `<think>` 内容。

训练按最多 2048 tokens 连续分块，前一块末尾 token 用作下一块的预测上下文，
每个可预测 token 每轮恰好作为目标一次。micro-batch=1，每片段更新一次，每局一遍。
训练关闭 cache、开启 gradient checkpointing；结束后切回 eval 并清除旧推理 cache。
当前局先记分，参数更新仅影响后续局。

## 日志、分析和恢复

每个输出目录包含：

```text
config.json                 # CLI 覆盖后的完整配置
metadata.json               # Git 状态、依赖版本、实验协议
model.json                  # 模型文件标识、LoRA 模块和可训练参数量
manifest.json               # 完整 stream
commit.json                 # 已提交 episode 和最新 checkpoint 指针
checkpoints/*.pkl            # 仅训练方法：最新 LoRA、优化器、方法和 RNG 状态
state/*.pkl                  # 轻量恢复状态，不含模型参数（历史、RNG、进度）
traces/<UTC启动时间戳>/
  invocation.json           # 本次启动的配置、版本和系统 prompt
  episodes/<序号>/<attempt>/
    steps.jsonl               # reset、每步交互和终局事件
    last_request.json         # 推理异常时也能看到最后一次尝试的输入
    last_generation.json      # 环境调用异常时也能看到最后一次模型输出
    episode.json              # 不含重复推理前缀的完整本局对话
    update.json               # NTP 文本、token IDs、loss、梯度、更新步数
    error.json                # 仅异常尝试存在
summary.json                # 已提交结果汇总
episodes.csv                # 逐局指标和累计/滚动成功率
```

每次启动（包括 `--resume`）新建一个 UTC 时间戳 traces 子目录，集中保存该次启动的所有 episode traces；旧目录保留。检查点和提交索引仍在实验根目录，分析按 `attempt_dir` 查找记录，兼容旧目录结构。正在运行的旧进程不会改变日志路径。

逐步日志包含精确 prompt/messages、输入和输出 token IDs、原始输出、reasoning、解析 action、
实际执行动作、前后观察、候选动作、分数、成功/结束状态和耗时。每步 flush 并 fsync。
长 ReAct 历史会产生较大的日志，这是为还原实际模型输入保留的审计数据。

模型 checkpoint 不保存在逐步 traces 中。`checkpoint_interval` 默认 **10**，也可通过
`--checkpoint-interval 10` 指定，必须为正整数，适用于所有 NTP 方法：每完成 10 个 episode
保存一次 LoRA 和优化器状态，结束时即使不足 10 局也保存。不会每个训练 step 保存模型。
`memoryless` 和 `icl` 不保存模型 checkpoint，只在每局结束保存轻量恢复状态；ICL 包含历史对话。

只统计 `commit.json` 中的已提交 episode。训练方法在 checkpoint 成功后一次提交对应的一批结果；
非训练方法逐局提交。中途崩溃的尝试仍保留，但不计入分析。
使用**完全相同的配置和覆盖参数**追加 `--resume`：

```bash
uv run --locked --extra continual python -m alfworld.continual \
  --method history-ntp --stream-order shuffled --action-mode admissible \
  --manifest runs/memoryless-shuffled-admissible/manifest.json \
  --output-dir runs/ntp-shuffled-admissible --resume
```

恢复验证配置、依赖版本、模型标识和数据摘要，从上一已提交边界继续。
训练方法恢复到最近 checkpoint 的权重，重跑之后未提交的 episode（默认最多 9 个已完成局，
以及中断中的当前局）；旧 traces 保留，不会重复统计或在已经更新的权重上重复学习。
尚无训练 checkpoint 时从初始模型重跑。checkpoint 为本项目生成的本地 pickle，只加载可信运行目录。
默认仅保留最新 checkpoint；最终完成后该文件就是最终 LoRA/优化器状态。
旧版每局保存的 checkpoint 仍可恢复；旧配置缺少 `checkpoint_interval` 时按 10 处理。

重新生成分析（不加载模型）：

```bash
uv run --locked python -m alfworld.continual.analysis runs/ntp-shuffled-admissible --rolling-window 20
```

输出包括累计/滚动/按任务类型成功率、格式错误率、非法动作率、平均 reasoning 字符数、
输出 token 数、总输入 token、生成和训练耗时。错误尝试和未提交更新不会污染指标。
其中 `invalid_action_rate` 定义为已执行动作不在当前候选列表中的比例，
用于诊断，不等同于环境解析器确认的语法错误率。

## 验证

```bash
# 协议、流顺序、异常、恢复测试；无需 GPU
uv run --locked python -m unittest discover -s tests -p test_continual.py -v

# 真实 Qwen3.5 架构的小型随机模型：NTP 梯度、LoRA、恢复和生成
uv run --locked --extra continual python -m unittest discover -s tests -p test_continual_model.py -v

# 假模型 + 真实 TextWorld，只验证调度，不产生有效模型实验结果
uv run --locked python -m alfworld.continual --backend mock --method icl \
  --stream-order task_blocked --action-mode admissible \
  --limit 2 --max-steps 2 --output-dir runs/mock-smoke
```

`mock` 结果明确标记为测试，其 token IDs 为 UTF-8 bytes，不是模型 tokenizer IDs，
NTP 更新也是模拟的。真正的参数更新由小型 Qwen3.5 测试验证。
模型完整规模验证应在空闲 GPU 上分别运行三种方法的两局短实验；不得把 mock 或小模型
测试当作本地 Qwen3.5-9B 的实际实验结果。

模块职责：`stream/environment` 管理任务与环境；`prompting/types` 管理 ReAct 和轨迹；
`methods` 管理跨局状态；`model/training` 管理生成和 NTP；`runner` 调度；
`logging/analysis` 负责可恢复日志和离线指标。所有代码在 `alfworld/continual/`。
