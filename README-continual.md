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
默认超长输入按下文规则删除最旧历史并记录截断；显存不足、模型加载不完整时直接报错，不静默量化。

推理方法支持跨 GPU 按连续 decoder 层分配（不是张量并行，也不保证加速）。例如
`--devices cuda:0 cuda:1 --gpu-memory-gib 32 24`，或在 `model.devices` 和
`model.gpu_memory_gib` 中填写对应列表。设备编号相对于 `CUDA_VISIBLE_DEVICES`；
模型的输入设备须为列表第一项。每卡实际分配上限取配置值与当时空闲显存减 3 GiB 的较小值。
不进行 CPU offload 或量化，当前不支持用此模式训练 NTP。
模型层拆分可减少每卡权重和缓存，但单层长输入的临时张量仍可能导致 OOM，
因此不能仅凭卡数保证目标上下文长度可运行。

推理恢复时允许更改上述 GPU 布局，任务、窗口和解码配置仍必须一致。
每次启动的 `invocation.json` 和 `placement.json` 记录有效配置和实际层分布；
旧的单卡 checkpoint 仍可恢复，多卡运行也保存各设备 RNG 状态。

`continual` extra 独立于 `full/vis`，不需要 AI2-THOR、视觉检测器或训练 BUTLER。
基础 `uv sync --locked` 仍然只安装原来的文本环境依赖。
底层 Qwen3.5 线性注意力在没有可选优化 kernel 时使用正确但较慢的 PyTorch 实现。
当前实现优先验证实验语义，没有自动安装 `flash-linear-attention` 或 `causal-conv1d`。

## 方法和交互协议

| 方法 | 每步输入 | 局后更新 |
| --- | --- | --- |
| `memoryless` | 当前局完整历史 | 无 |
| `icl` | 最近 K−1 局 + 当前局完整历史 | 保存本局；淘汰最旧历史 |
| `dc-cumulative` | 累计 cheatsheet + 当前局完整历史 | 额外调用一次同一模型，更新完整 cheatsheet |
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
输出预算 **1024 tokens**，同时包含 reasoning 和 action。默认输入预算 **65536 tokens**，
额外预留输出空间。`model.context_overflow: truncate_oldest` 默认保留系统指令，优先删除最旧的
完整对话轮次，直到输入落入预算；最新单条观察仍过长时，从其正文开头删除 token，保留末尾的
近期反馈和合法命令。截断只改变本次推理输入，不删除原始 episode traces 或 NTP 训练文本。
ICL 的 K 是最多保留的 episode 数，实际输入还受 token 预算约束。
`steps.jsonl` 的 `messages`、`prompt` 和 token IDs 对应实际送入模型的截断后内容；
`input_truncation` 记录原始/实际 token 数、丢弃消息数及正文截断 token 数。
`last_request.json` 保留该步截断前的完整请求。设置 `context_overflow: error` 可恢复超限报错行为。

Reasoning 可多行，Action 必须是最后一行且仅出现一次。只执行 action，绝不把 reasoning
交给环境。缺失字段、格式错误或达到输出上限而未输出 EOS 时，不执行动作，返回格式反馈；
默认最多 50 次尝试，格式错误也计入。格式正确的非法动作仍交给环境处理，不替换动作。

动作模式可配置：

- `free`：给出 ReAct 输出格式，不展示当前合法动作列表。
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

## Dynamic Cheatsheet（DC-Cumulative）

参考 [上游 DC-Cumulative 实现](https://github.com/suzgunmirac/dynamic-cheatsheet/blob/main/dynamic_cheatsheet/language_model.py)
和 [curator prompt](https://github.com/suzgunmirac/dynamic-cheatsheet/blob/main/prompts/curator_prompt_for_dc_cumulative.txt)，
采用每局一次 curator 调用的 ALFWorld 适配版本：

1. 从空 cheatsheet 开始。行动时输入系统指令、当前累计 cheatsheet、当前局完整历史。
2. 每局结束并记分后，把旧 cheatsheet、本局任务、原始 reasoning/action、环境观察和终局结果
   交给同一个冻结模型，额外生成一次总结；成功和失败 episode 都执行。
3. 模型输出一份完整替代旧版的 cheatsheet：保留旧的有效知识，合并新策略、失败教训、
   示例和 episode 引用，并按证据维护成功使用次数。下一局开始使用新版。

不做梯度更新、不检索旧轨迹，也不在当前局重新解题。curator 使用独立提示词，
不受行动阶段 ReAct 输出格式约束。提示词针对 ALFWorld 改写，加入环境反馈，
强调物品 ID、位置和背包状态不能跨局复用；因此这是环境适配版本，并非原论文基准的逐字复现。
提示词和解析逻辑见 `alfworld/continual/cheatsheet.py`。

```bash
uv run --locked python -m alfworld.continual \
  --method dc-cumulative --stream-order shuffled --action-mode admissible \
  --cheatsheet-max-new-tokens 2048 --output-dir runs/dc-cumulative
```

`cheatsheet_max_new_tokens` 默认 2048，仅控制 curator 输出（行动输出仍由
`model.max_new_tokens` 控制）。总结要在预算内包含完整 `<cheatsheet>...</cheatsheet>`；
输出截断、空内容或标签无效时保留旧版，记录原因，不额外重试。curator 输入同样遵循
`model.context_overflow`，并在更新日志记录输入截断。与上游相比，这里的提取器更严格，
拒绝不完整的标签和截断输出。

`update.json` 保存总结的完整 messages/prompt、原始输出、token IDs、耗时、接受状态、
更新前后 cheatsheet 和版本；`curator_request.json` 在调用前写入，便于排查模型调用异常。
`episodes.csv` 的 `curator_input_tokens`、`curator_output_tokens`、`curator_generation_seconds`
单独统计总结开销，不混入行动或训练开销。`model_version_after` 始终为 0，
`cheatsheet_version_after` 记录成功接受的总结次数。
每局原子保存 cheatsheet 和进度；使用相同参数加 `--resume` 恢复。模型调用异常会中止本次运行，
恢复时重跑未提交的 episode，已提交局不会重复总结。

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
state/*.pkl                  # 轻量恢复状态，不含模型参数（历史、cheatsheet、RNG、进度）
traces/<UTC启动时间戳>/
  invocation.json           # 本次启动的配置、版本和系统 prompt
  episodes/<序号>/<attempt>/
    steps.jsonl               # reset、每步交互和终局事件
    last_request.json         # 推理异常时也能看到最后一次尝试的输入
    last_generation.json      # 环境调用异常时也能看到最后一次模型输出
    episode.json              # 不含重复推理前缀的完整本局对话
    curator_request.json      # 仅 DC：总结调用前保存的输入和预算
    update.json               # NTP 训练记录，或 DC 总结和 cheatsheet 更新记录
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
`memoryless`、`icl` 和 `dc-cumulative` 不保存模型 checkpoint，只在每局结束保存轻量恢复状态；
ICL 包含历史对话，DC 包含累计 cheatsheet。

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
