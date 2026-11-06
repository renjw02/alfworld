# ALFWorld streaming continual learning

本框架比较跨 episode 更新 **context** 与更新 **LoRA 参数**。默认数据来自 `valid_seen`，
也支持 `valid_unseen` 和合并 seen/unseen 的 `valid_heat_place` 子集，
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

推理方法及 `ascent`/`ascent-icl`/`tt-opsd` 支持跨 GPU 按连续 decoder 层分配（不是张量并行，也不保证加速）。例如
`--devices cuda:0 cuda:1 --gpu-memory-gib 32 24`，或在 `model.devices` 和
`model.gpu_memory_gib` 中填写对应列表。设备编号相对于 `CUDA_VISIBLE_DEVICES`；
模型的输入设备须为列表第一项。每卡实际分配上限取配置值与当时空闲显存减 3 GiB 的较小值。
不进行 CPU offload 或量化，当前不支持用此模式训练 NTP。
模型层拆分可减少每卡权重和缓存，但单层长输入的临时张量仍可能导致 OOM，
因此不能仅凭卡数保证目标上下文长度可运行。

推理、ASCENT 和 TT-OPSD 恢复时允许更改上述 GPU 布局，任务、窗口、解码和训练配置仍必须一致。
每次启动的 `invocation.json` 和 `placement.json` 记录有效配置和实际层分布；
旧的单卡 checkpoint 仍可恢复，多卡运行也保存各设备 RNG 状态。

`continual` extra 独立于 `full/vis`，不需要 AI2-THOR、视觉检测器或训练 BUTLER。
基础 `uv sync --locked` 仍然只安装原来的文本环境依赖。
Linux x86_64、Python 3.11+ 的 Torch/TorchVision 使用官方 CUDA 12.8 wheel 源，
以兼容本机 570 系列驱动；直接使用 PyPI 的 CUDA 13.x wheel 会导致 CUDA 初始化失败。
该源仅用于这两个包，其他依赖继续使用 PyPI，具体版本由 `uv.lock` 固定。
安装后可运行 `uv run --locked --extra continual python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`
检查，最后一项应为 `True`。运行前还须准备 `model.path` 指定的本地模型权重，
并选择空闲显存满足要求的 GPU；安装依赖不会自动下载模型。
底层 Qwen3.5 线性注意力在没有可选优化 kernel 时使用正确但较慢的 PyTorch 实现。
当前实现优先验证实验语义，没有自动安装 `flash-linear-attention` 或 `causal-conv1d`。

## Heat&place 专用子集（seen + unseen）

`valid_heat_place` 合并 `valid_seen` 的 **16** 个和 `valid_unseen` 的 **23** 个
`pick_heat_then_place_in_recep` 游戏，共 **39** 个。它是逻辑合并，直接加载原始游戏，
无需复制数据或创建软链接；沿用 solvable、非 Sliced/非 movable 等原有筛选规则。
每个 episode ID 带来源 split 前缀，manifest 的 `source_split` 保留 seen/unseen 来源。
先合并再按 seed 打乱，跨来源持续保留历史、cheatsheet 或 LoRA。

统一配置为 `configs/continual-heat-place.yaml`，固定清单为
`configs/manifests/valid_heat_place.seed42.json`。清单包含相对数据路径和 SHA-256，
三种方法使用同样的 39 个任务和 seed=42 顺序。仓库根目录下运行：

```bash
for method in icl dc-cumulative ascent; do
  .venv/bin/python -m alfworld.continual \
    --config configs/continual-heat-place.yaml \
    --method "$method" --device cuda:0 \
    --output-dir "runs/heat-place-${method}-seed42"
done
```

`--data-dir` 或 `ALFWORLD_DATA` 可指定包含 `json_2.1.1/` 的数据目录。
默认每种方法都从基座重新开始；运行中断后用相同命令加 `--resume`。
要换 seed/order，可将配置的 `stream.manifest` 设为 `null` 自动生成新顺序，
或在其他现有配置上添加 `--split valid_heat_place --seed <seed>`。
该别名要求 seen/unseen 两个来源均存在，不会在缺失一个来源时静默运行缩小的数据集。

只检查数据、环境和流程而不加载模型，可在单种方法命令后添加
`--backend mock --max-steps 1`，并使用独立输出目录。Mock 不验证真实模型训练或效果。

**失败轨迹研究的当前边界：** ICL 保留失败局历史，DC 的 curator 也接收失败局及其结果；
当前 Ascent 遇到失败局会记录 `skip_reason: unsuccessful_episode`，只有成功局才更新参数。
因此三者可以直接运行在此子集上，但当前 Ascent 是成功轨迹学习对照，
若要让它从失败轨迹更新参数，仍需单独定义并实现新的学习目标。
合并后的成绩是混合 seen/unseen 的在线适应结果，不是标准 unseen 泛化成绩。

## 方法和交互协议

| 方法 | 每步输入 | 局后更新 |
| --- | --- | --- |
| `memoryless` | 当前局完整历史 | 无 |
| `icl` | 最近 K−1 局 + 当前局完整历史 | 保存本局；淘汰最旧历史 |
| `dc-cumulative` | 累计 cheatsheet + 当前局完整历史 | 额外调用一次同一模型，更新完整 cheatsheet |
| `history-ntp` | 当前局完整历史 | 用本局完整文本执行一轮 LoRA NTP |
| `history-ntp-assistant` | 当前局完整历史 | 所有 episode，仅训练 assistant reasoning/action |
| `history-ntp-assistant-success` | 当前局完整历史 | 仅成功 episode，训练 assistant reasoning/action |
| `ascent` | 当前局完整历史 | 仅成功 episode，冻结基座 teacher 提供 hindsight 分布，更新 LoRA |
| `ascent-icl` | 最近 K−1 局 + 当前局完整历史 | 仅成功 episode 执行 ASCENT 更新；成功和失败局都保留到上下文窗口 |
| `tt-opsd` | 当前局完整历史 | 仅成功 episode，teacher 逐轮接收任务、该轮回复及返回观察，更新 LoRA |
| `sft` | 当前局完整历史 | 成功和失败局均以整条轨迹训练，屏蔽无效回复，默认更新 LoRA 两次 |

`window_episodes` 默认 4，**包含当前局**；K=1 的 ICL 与 memoryless 输入一致。
所有历史保留 reasoning、action、观察、分数、格式反馈和终局反馈。
每个方法在独立输出目录从同一初始模型开始。默认 `memory_scope: stream`，
切换任务类型时不清除历史或参数。

使用 `--stream-order task_blocked --memory-scope task_type` 运行按类别独立积累记忆的对照：
类别内照常学习，进入下一类别前，ICL 清空历史，DC 清空 cheatsheet，NTP 恢复本次运行
初始的 LoRA 参数、空优化器状态和更新计数。冻结的基础模型不变，各类也恢复初始 RNG 状态。
这一模式要求分块顺序，避免在 shuffled 中反复切换类别造成含义不同的重置。
它不改变类别内部的任务顺序，也不需要为 memoryless 额外重复一份对照。
训练方法除每 10 局和最终保存外，还在类别末尾保存 checkpoint；恢复时根据保存的当前类别
判断是否需要重置，避免把上一类别的学习带入新类别。边界事件记录在 `steps.jsonl`。

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

所有方法共用的 system prompt 提醒模型：动作返回 `Nothing happens.` 或类似无变化反馈时，
重新检查最新观察并尝试其他动作，避免重复无效动作序列；相关条件改变后才重试失败动作。
这是提示词引导，不会自动拦截或替换动作，也不保证消除循环。

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

## ASCENT：成功经验的在线自蒸馏

实现依据：[ASCENT 论文](https://arxiv.org/pdf/2610.05303) §3.3、Figure 6 和附录 B.3。
这是当前 continual 交互协议下的核心算法复现，不是论文全部实验的逐项复现。
保留本仓库的 `Reasoning:/Action:`、合法动作列表、greedy decoding、50 步预算和
Qwen3.5-9B；没有引入论文的 action 标签协议或 unseen/WebShop/AppWorld 实验。

使用独立配置（以下仅为运行命令示例）：

```bash
uv run --locked --extra continual python -m alfworld.continual \
  --config configs/continual-ascent.yaml \
  --manifest runs/memoryless-shuffled-admissible/manifest.json \
  --output-dir runs/ascent-shuffled-admissible
```

不指定 manifest 时按配置生成任务流。与 `memoryless`、`history-ntp-assistant-success`
比较时应使用相同任务顺序和交互配置。NTP 保留原训练超参数，因此该比较不能单独归因于 loss。
`--method ascent` 也会采用独立 `ascent` 配置的默认值，不读取 `training` 下的 NTP 超参数。

每个成功 episode 在记分后执行以下流程，失败局不改变参数、优化器或模型版本：

1. 从初始环境观察提取 `Your task is to:` 后的目标，按 Figure 6 构造成功声明与编号后的
   reasoning/action 记录。不加入 observation、分数或终局消息。解析失败和不在当时合法动作
   列表中的轮次不进入此记录；合法但没有状态变化的动作仍然保留。
2. 读取每步生成时的精确 input/output token IDs。student 使用原始前缀；teacher 在其前面
   加成功记录的 token IDs，关闭 LoRA 并禁用梯度。成功记录独立编码后拼接，不重新编码
   student 前缀，不调用模型生成新的监督答案。
3. 在所有生成响应 token（含 EOS、格式错误及非法动作轮次）上计算全词表
   `KL(teacher || student)`。先平均每轮 token，再平均轮次。system/user/模板和成功记录
   本身不作为目标；超长推理已经截断的前缀也按当时实际 token 原样复用。
4. 每个成功局执行两次完整 episode 梯度累积与 AdamW 更新，LoRA 和优化器状态跨局延续，
   不回放历史 episode。日志保留轨迹用于审计，不作为 replay buffer。

论文给出的参数为 `down_proj`、rank=16、scale=32、lr=1.5e-4、每局两次更新。
这里把 scale 实现为 PEFT `lora_alpha=32`，采用默认 alpha/r 缩放。
论文未明确的细节采用 dropout=0、weight decay=0、max grad norm=1、AdamW 默认 betas/eps、
蒸馏温度=1，以及上述 EOS 处理方式。这些属于实现假设，不声称来自作者代码。

teacher/student 共享基座权重；decoder 开启 gradient checkpointing，输出头和 KL 按
`ascent.logits_chunk_tokens`（默认 32）个响应位置进行 checkpointed 分块计算。
每块仍计算完整词表，不使用 top-k 或 2048-token 历史切块。该参数只控制输出头计算的显存，
不能保证长前缀训练不 OOM。可通过 `--devices` 将 decoder 层分配到多张 GPU。
若 teacher 成功记录加完整前缀超过模型位置上限，更新前报错，不静默删减上下文或跳过该局。

ASCENT 多卡训练使用 PyTorch 可求导的设备间拷贝和层输入 hooks，支持梯度 checkpoint 重计算；
不使用 Accelerate 的推理卸载 hooks。权重加载时暂存 CPU，运行时各层常驻对应 GPU。
仅响应位置的 hidden states 传到输出头所在卡计算 KL；梯度裁剪覆盖所有卡上的 LoRA 参数。
单卡 checkpoint 中的 LoRA 和 AdamW 状态可恢复到新布局，新增设备的 RNG 从已存主卡状态初始化；
后续 checkpoint 保存每张卡的 RNG。改变布局可能改变浮点运算结果，不保证与单卡轨迹逐位相同。

ASCENT 复用训练方法的 checkpoint 提交和恢复语义：每 10 局、最终局、按类别重置模式的
类别末尾保存；恢复时回滚至已提交边界。`memory_scope: task_type` 是附加对照，论文式
跨任务学习应使用默认 `stream`。
`update.json` 保存 teacher 记录及 token IDs、保留/过滤轮次、每次更新的逐轮 KL、
梯度范数、目标 token 数、耗时和版本；逐步日志与 `episode.json` 保留精确训练前缀。

新增测试覆盖 KL 方向、teacher 冻结、token 对齐、无效轮次过滤、成功门控、分块等价性、
LoRA 更新、恢复与 runner 集成。2026-11-02 执行以下套件，23 项测试全部通过
（CPU 小型随机 Qwen3.5 + mock 环境调度；未运行完整 9B 或正式实验）。测试命令：

```bash
uv run --locked --extra continual python -m unittest discover -s tests -p 'test_continual*.py' -v
```

## TT-OPSD

依据 [论文 Section 4.1 和 Appendix B.2–B.3](https://arxiv.org/pdf/2610.05303)，
TT-OPSD 是本文对 OPSD 的 test-time 改编。这里单独提供 `--method tt-opsd`：

| 项目 | ASCENT | TT-OPSD |
| --- | --- | --- |
| teacher 的额外信息 | 整条成功轨迹的过滤后 reasoning/action | `z[i,t] = (任务, 本轮原始回复, 本轮返回观察)` |
| 无效动作轮次的额外信息 | 从成功记录中过滤 | 保留本轮回复及失败反馈 |
| 训练位置 | 全部非空回复，包括无效动作轮次 | 相同 |
| 更新时机与目标 | episode 成功后，冻结初始 teacher，forward KL | 相同 |

teacher 仍接收该轮 rollout 的原始 student 前缀（包含当时可见的历史），额外信息不含后续
轮次或整局成功声明。训练先平均每轮的响应 token loss，再平均轮次；每个成功 episode
累积全部轮次的梯度后执行 AdamW，一共两次。失败局不更新；不额外采样、不回放、不逐轮更新。
LoRA 和优化器状态跨 episode 保留。

```bash
uv run --locked --extra continual python -m alfworld.continual \
  --config configs/continual-tt-opsd.yaml \
  --output-dir runs/tt-opsd-shuffled-admissible
```

比较 ASCENT 时可加 `--manifest <相同任务流的manifest.json>`，保持交互配置一致。
TT-OPSD 共用 YAML 的 **`ascent:` 优化器配置块**，包括 `down_proj` LoRA、rank=16、alpha=32、
lr=1.5e-4、`optimizer_steps=2`；`training:` 仍只供 NTP 使用。支持同样的多卡布局和 `--resume`。
`update.json` 用 `kind: tt-opsd` 标识，`teacher_records` 按轮保存文本及其 token IDs，
`retained_turns` 包含所有轮次，`filtered_turns` 为空。student 的精确 token 前缀保存在 `episode.json`。
教师上下文超限会在任何更新前报错，不静默截断或跳过轮次。

论文未给出 TT-OPSD 的具体文字模板，本实现用 Task / Current response / Returned observation
包装上述三元组。沿用本仓库的 ReAct 格式和环境反馈：未解析出动作时环境没有执行，使用
runner 实际返回的格式错误反馈，不把未变化的旧 observation 冒充动作反馈。
其余未明确的超参数和 EOS 处理沿用上面的 ASCENT 实现假设；这不是论文成绩的复现声明。

验证命令：

```bash
uv run --locked --extra continual python -m unittest discover -s tests -p 'test_continual*.py' -v
```

TT-OPSD 测试覆盖逐轮信息隔离、无效及格式错误反馈、原始 token 前缀、全词表 KL 与直接
forward 的一致性、冻结 teacher、LoRA 更新、分块等价性、失败门控、空响应、更新前超限检查、
优化器恢复和中断后重跑未提交 episode。模型计算测试使用 CPU 上的小型随机 Qwen3.5，
runner 测试使用假环境；没有据此声称完成完整 9B benchmark 或多卡实测。

## ASCENT + ICL

`ascent-icl` 在 ASCENT 参数更新之外，保留最近 `window_episodes - 1` 局的原始对话。
默认窗口为 4（最近 3 局加当前局），成功和失败局都进入窗口；只有成功局更新 LoRA。
使用 `memory_scope: stream` 时，LoRA、优化器和上下文窗口都跨任务 block 延续。
窗口仍受输入 token 预算约束，沿用 `truncate_oldest` 规则。

ICL 仅用于推理。每步额外构造训练前缀：system prompt + 当前局截至该步之前的对话，
不包含其他 episode，也不包含当前回复或后续反馈。它使用与普通 ASCENT 推理相同的
chat template、输入预算和截断规则。student 和 teacher 的训练输入都使用这个局内前缀；
teacher 仍为关闭 LoRA 的冻结基座，并额外接收当前成功局的成功记录。
训练目标保持为 ICL 推理实际生成的回复 token（含 EOS），不重新生成或重新编码回复。
loss 仍只覆盖当前局回复，原有成功门控、KL、LoRA 和每局两次更新不变。

`steps.jsonl` 分别保存推理的 `input_token_ids` 和训练的 `training_input_token_ids`、
`training_messages`、`training_prompt`、`training_input_truncation`；`episode.json` 保留两套前缀 IDs。
`update.json` 标记 `training_context_scope: current_episode` 并记录各轮训练前缀长度。
checkpoint 同时保存历史窗口、LoRA 和优化器，恢复时一并回到已提交边界。
新版 metadata 标记 `ascent_icl_training_context: current_episode_v1`。
旧版把 ICL 历史带入训练的运行必须从初始模型重跑；禁止用新版续接旧 checkpoint 混合实验语义。

```bash
uv run --locked --extra continual python -m alfworld.continual \
  --config configs/continual-ascent-icl.yaml \
  --seed 42 --stream-order task_blocked --memory-scope stream \
  --output-dir runs/ascent-icl-task-blocked-admissible
```

该组合支持单卡或多卡分层训练，默认学习率、LoRA 设置、每个成功局的两次更新与 ASCENT 相同。
例如六卡运行时加 `--devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5`。

多卡梯度、teacher 冻结、生成及单卡优化器 checkpoint 迁移验证（至少两张空闲卡）：

```bash
CUDA_VISIBLE_DEVICES=2,3 ALFWORLD_TEST_MULTIGPU=1 \
  uv run --locked --extra continual python -m unittest discover -s tests -p test_continual_multigpu.py -v
```

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
    episode.json              # 本局对话；ASCENT 另保留逐步 token 前缀
    curator_request.json      # 仅 DC：总结调用前保存的输入和预算
    update.json               # NTP/ASCENT 训练记录，或 DC 总结更新记录
    error.json                # 仅异常尝试存在
summary.json                # 已提交结果汇总
episodes.csv                # 逐局指标和累计/滚动成功率
task_summary.csv            # 六类各自的局数、成功局数和成功率
```

每次启动（包括 `--resume`）新建一个 UTC 时间戳 traces 子目录，集中保存该次启动的所有 episode traces；旧目录保留。检查点和提交索引仍在实验根目录，分析按 `attempt_dir` 查找记录，兼容旧目录结构。正在运行的旧进程不会改变日志路径。

逐步日志包含精确 prompt/messages、输入和输出 token IDs、原始输出、reasoning、解析 action、
实际执行动作、前后观察、候选动作、分数、成功/结束状态和耗时。每步 flush 并 fsync。
长 ReAct 历史会产生较大的日志，这是为还原实际模型输入保留的审计数据。

模型 checkpoint 不保存在逐步 traces 中。`checkpoint_interval` 默认 **10**，也可通过
`--checkpoint-interval 10` 指定，必须为正整数，适用于 NTP 和 ASCENT：每完成 10 个 episode
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
`summary.json.by_task` 和 `task_summary.csv` 始终包含六类，未运行类别的成功率为空；
`episodes.csv` 另外提供类别内的 episode 序号、累计成功率和滚动成功率，窗口不跨类别混算。
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

## 整轨迹 SFT：交错 mask，每局更新两次

`sft` 在每个 episode 完成评估后训练，包括成功和失败局。推理使用当前局完整历史，
跨 episode 只保留 LoRA 参数，不加入过去局的 ICL 历史。
整条轨迹通过同一聊天模板序列化一次，系统提示、observation、反馈、终局结果、
消息头和无效回复的 labels 都为 `-100`；只有有效回复的完整 reasoning/action 和结束 EOS
是监督目标。mask 不删除输入，后续有效回复仍能利用无效动作和反馈；因果注意力阻止未来信息泄漏。

有效性沿用 `success-only-sft` 的严格筛选：排除格式错误或截断、未执行或非 admissible 动作；
对未报告成功的步骤，还排除 `Nothing happens.` 和 observation 未变化。
含聊天控制 token 的回复也不作为监督目标。被排除步骤的 reasoning/action 一起 mask。
这里的“有效”只表示通过交互筛选，不保证 reasoning 正确或动作有助于最终成功。

每局对同一序列、同一 labels 执行 `sft.optimizer_steps: 2` 次完整前向、反向和 AdamW 更新。
第二次使用第一次更新后的参数重新计算梯度，不重新采样轨迹，也不复用梯度。
loss 对整条轨迹所有有效目标 token 等权平均，所以较长回复占更大权重。
没有有效目标时不更新参数和模型版本。`update.json` 保存完整训练序列及 labels、
有效步骤、剔除原因、目标 token 区间，以及两次更新各自的 loss 和全局 optimizer step。
模型版本按有更新的 episode 递增一次，optimizer step 默认每局递增两次。

默认 LoRA 为 `all-linear`（排除 `lm_head`），rank=16、alpha=32、dropout=0、学习率 `1e-5`。
`sft.batch_size`、`sft.gradient_accumulation_steps`、`sft.history_interactions` 和
`training.chunk_tokens` 不用于此方法。`sft.logits_chunk_tokens: 32` 只控制词表投影的
显存占用：decoder 每次仍处理完整轨迹，所有有效目标的梯度合并后只执行一次 optimizer step。

专用配置固定 heat&place 的 39 个任务、seed=42、每局最多 50 步，默认单 GPU：

```bash
.venv/bin/python -m alfworld.continual \
  --config configs/continual-sft.yaml \
  --device cuda:0 --output-dir runs/heat-place-sft-seed42
```

完整历史要求 `model.context_overflow: error`；默认输入预算 64512、生成预算 1024，
整条训练序列（包括模板和终局消息）上限 65536，且不超过模型上下文上限。
超限明确报错，不切块、不静默截断、不删除历史。保留失败尝试日志，可在排查后重新运行。
使用相同配置加 `--resume` 从已提交的 episode 边界恢复；默认每 10 局和结束时保存参数状态。
原 `success-only-sft` 的成功门控、五次交互历史、minibatch 语义保持不变。

## Success-only SFT：按交互构造训练样本

`success-only-sft` 仅在整局成功后更新，并逐步排除格式错误、截断、未执行或列表外动作、
`Nothing happens.` 以及非终局的 observation 未变化步骤。ALFWorld 没有逐步进展奖励，
这里的“成功交互”是可核查的有效交互筛选，不保证每步都是最短成功路径必需的动作。
失败/无效步骤仍可出现在之前 5 次交互的历史上下文中，但不会成为监督目标。

每条训练样本单独包含 system prompt、从初始 observation 提取的 task description、
之前最多 5 次交互的 observation / reasoning / action / feedback、当前 observation 和合法命令，
再接本次完整的 reasoning/action 和 EOS。loss 仅覆盖本次回答和 EOS，历史回答与所有用户信息
都 mask 为 `-100`。构造回答前缀直接使用与推理相同的 generation template；新方法推理时也
使用同样的 5 次交互窗口，不向模型提供当前交互的未来反馈或整局成功结果。

不再按 `training.chunk_tokens` 切块。使用 mini-batch 和梯度累积，沿用 `training` 中的 LoRA 和 AdamW 参数。
默认 `sft.batch_size: 2`、`sft.gradient_accumulation_steps: 4`，有效 batch 为 8 条交互样本。
每个成功 episode 的更新次数为 `ceil(有效样本数 / 8)`；最后不足 8 条也更新一次，
按实际样本数归一化，各回答内部按目标 token 取均值。Padding、历史和观察不参与 loss，
梯度裁剪与 optimizer.step 只在累积窗口结束时执行，不跨 episode 保留梯度。
`update.json` 分别记录样本数、micro-batch 数和 optimizer steps。`sft.max_seq_tokens` 限制 prompt + 当前回答 + EOS 的总长：
超限整条跳过并记录长度，不截断 action/EOS，也不删除要求的历史。
原始默认值为 2048；对历史 207 个成功局的 2653 个有效交互进行长度检查后，2048 会排除
755 条（28.5%），4096 只排除 7 条（0.26%），所以方法配置使用 4096。

```bash
uv run --locked --extra continual python -m alfworld.continual \
  --config configs/continual-success-only-sft.yaml \
  --device cuda:0 --output-dir runs/success-only-sft-seed42
```

`update.json` 保留每条完整样本的 messages、prompt、input_ids、labels、历史步骤编号、
训练统计及拒绝原因；零有效样本不更新参数或模型版本。`task_type` 重置及事务 checkpoint
与其他参数训练方法一致。

长度审计和少量真实训练可单独执行，不改变已有实验 checkpoint：

```bash
.venv/bin/python scripts/pilot_success_only_sft.py \
  --runs runs/continual-matrix-20261101T135608Z/history-ntp-assistant-success-stream-1 \
  --output runs/sft-pilot/check.json --max-seq-tokens 4096 \
  --train-samples 9 --batch-size 2 --gradient-accumulation-steps 4 \
  --device cuda:0 --memory-gib 30
```

不传 `--train-samples` 时只加载本地 tokenizer 并统计长度，不加载模型。试跑结果仅验证样本构造
和训练可执行性，不是新方法的 ALFWorld 成功率评测。

旧版每条样本更新一次的 SFT 运行若要继续恢复，需显式设 `batch_size: 1` 和
`gradient_accumulation_steps: 1`；直接改成新默认值会触发 resume 配置不一致保护。
