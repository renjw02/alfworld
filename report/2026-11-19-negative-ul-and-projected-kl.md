# Negative-UL 与 Projected-KL 实现报告

日期：2026-11-19。实现版本：`16d9f7e`（包含 Negative-UL 的 `ac834e0` 实现）。

两种方法都是 ALFWorld continual streaming 中的**在线负反馈适应方法**：执行动作后，如果环境明确返回 `Nothing happens`，就在下一次生成前更新 LoRA。它们使用相同的输入、候选动作、评分函数和优化器；主要差别是 loss 与每次负反馈的优化步数。

**Negative-UL 直接惩罚当前错误动作的候选概率；Projected-KL 将更新前的候选分布投影到剩余动作上，固定投影目标做 forward KL 拟合。当前默认分别更新 1 步和 2 步。第一步的 loss 值和梯度相同，第二步才体现固定目标的作用。**

## 1. 共同的交互与数据协议

### 1.1 什么反馈可以触发训练

实际条件由 [`is_negative`](../alfworld/continual/negative_ul.py) 定义：

```python
step["format_error"] is None
and step["executed_action"] is not None
and step["feedback"].strip().lower().rstrip(".") == "nothing happens"
```

因此，`Nothing happens.`、`nothing happens`、带首尾空白的同类反馈会匹配。仅仅 observation 未变化、任务失败、动作不在 admissible 列表中，都不足以触发训练；是否接受负样本不取决于 episode 最终成败。动作虽然不在列表中，但如果确实执行并返回上述反馈，仍然可作为负样本。

格式错误与环境错误有不同的执行路径。输出必须符合 `Reasoning: ...` 和最后一行 `Action: ...` 的严格格式；生成达到长度上限而未结束、字段不匹配、额外前后文本等会被解析器拒绝。被拒绝的输出不执行环境动作，不触发这两种方法的更新，但仍占用每局的尝试步数，并与格式纠正提示一起进入后续历史。见 [`parse_react`](../alfworld/continual/prompting.py) 和 [`run_episode`](../alfworld/continual/runner.py)。

### 1.2 状态实际包含什么

这里的条件不只是环境 observation。记：

- \(h_t\)：实际生成时的 system prompt、任务、当前局截至本步之前的对话历史；包含此前 observation、动作、反馈及当前 admissible commands。
- \(r_t\)：模型在本步已经生成的 Reasoning。
- \(c_t=(h_t,r_t,\texttt{Action:})\)：候选动作共同的评分前缀。

代码直接复用本步记录的 `input_token_ids`，再拼接实际生成的、截至 `Action:` 的输出 token。不会重新构造一个简化 observation，也不会把**本步返回的负反馈**加入本次评分输入。该反馈只决定是否训练；下一次交互时它才成为历史的一部分。

Reasoning 作为条件，不作为 loss 的预测目标。不过 LoRA 参数是共享的，动作 loss 的梯度会通过上下文计算图传播，更新后未来的 Reasoning 或输出格式仍可能改变；“不监督 Reasoning token”不等于“Reasoning 行为被冻结”。

### 1.3 候选集合

设实际执行的错误动作为 \(a^-\)，使用**执行前**记录的候选列表：

\[
\mathcal C_t=\operatorname{dedup}
\left(\text{admissible\_commands before action}\cup\{a^-\}\right).
\]

实现保留候选的首次出现顺序，删除重复字符串；若错误动作不在列表中，将其补入。不能改用执行后的 `next_admissible_commands`。集合中的其他动作只是待分配概率的候选，**不代表已被验证为正确动作或有进展的动作**。

当前实现一次只处理当前 \(a^-\)，不缓存或合并“同状态的多个错误动作”，也不跨 episode 重放负样本。若去重后只有一个动作，记录 `no_alternative_candidates` 并跳过更新。

### 1.4 在线调度与跨局状态

每个交互按以下顺序执行：

```text
生成 Reasoning + Action → 解析 → env.step(Action) → 记录反馈
    → 若为 Nothing happens：构造候选样本并完成更新
    → 使用更新后的参数生成下一步
```

两种方法都保留当前 episode 的历史。默认 `memory_scope: stream` 下，LoRA 参数、AdamW 状态和累计 optimizer step 跨 39 局延续；不保留跨局 ICL 对话。每个独立实验重新加载基座模型并初始化新的 LoRA/优化器。

即使产生负反馈的是终局步骤，也先更新再结束 episode。episode 结束只汇总，不再训练一次。`model_version` 每次完成一个负反馈样本的训练增加 1，因此 Projected-KL 默认做两次 optimizer step，版本号也只增加 1。调度入口为 [`Method.observe_step`](../alfworld/continual/methods.py)。

## 2. 共同的动作概率定义

### 2.1 从语言模型 token 概率到候选动作概率

候选 \(a\) 的目标 token 序列按以下形式构造：

```python
tokenizer.encode(" " + action, add_special_tokens=False) + [eos_token_id]
```

`Action:` 后的分隔空格属于候选续写，因为 BPE 常将空格与首个动作单词合并。所有候选采用同样的格式，并包含动作结束符 EOS。给定 \(c_t\)，候选的序列得分为：

\[
z_\theta(a;c_t)
=\sum_{j=1}^{|y(a)|}
\log P_\theta\left(y_j(a)\mid c_t,y_{<j}(a)\right).
\]

这里 \(P_\theta\) 是语言模型在整个词表上的 token 分布。代码通过目标 token 的 cross entropy **求和并取负值**计算 \(z\)，不做长度平均；长动作可能因此得到更低的序列得分。

随后在候选集合上做温度为 1 的 softmax：

\[
\boxed{
 p_\theta(a\mid c_t,\mathcal C_t)
 =\frac{\exp z_\theta(a;c_t)}
 {\sum_{b\in\mathcal C_t}\exp z_\theta(b;c_t)}
}
\]

后文的 \(p\) 均指这个**候选集合内的归一化分布**。它不是原始完整动作序列的无条件概率，也不是逐 token 的 UL。改变候选集合会改变 \(p\)，即使语言模型参数不变。

### 2.2 推理仍是自由生成

候选评分仅用于训练。推理依然调用 `model.generate(do_sample=False)` 生成完整的 `Reasoning + Action`，没有改成枚举候选后选最大分数，也没有做语法约束解码。

`action_mode: admissible` 通过提示词要求复制当前候选命令，并不强制输出一定合法。因此仍可能出现列表外动作、格式错误或重复无效行为。

## 3. Negative-UL 的 loss 与更新

令 \(p_-=p_\theta(a^-\mid c_t,\mathcal C_t)\)。当前单负样本目标为：

\[
\boxed{\mathcal L_{\mathrm{UL}}=-\log(1-p_-)}.
\]

记 \(\mathcal G=\mathcal C_t\setminus\{a^-\}\)，实现采用等价的稳定形式：

\[
\mathcal L_{\mathrm{UL}}
=\operatorname{softplus}\left(z_- -\operatorname{LSE}_{a\in\mathcal G}z_a\right),
\qquad
\operatorname{LSE}(z)=\log\sum_a e^{z_a}.
\]

这避免直接计算接近 1 的 \(p_-\) 所导致的 `1 - p_-` 精度损失。函数位置：[`negative_ul.candidate_loss`](../alfworld/continual/negative_ul.py)。

对候选分数的导数是：

\[
\frac{\partial\mathcal L_{\mathrm{UL}}}{\partial z_-}=p_-,
\qquad
\frac{\partial\mathcal L_{\mathrm{UL}}}{\partial z_a}
=-\frac{p_-p_a}{1-p_-},\quad a\in\mathcal G.
\]

所以所有候选都参与反向传播。目标只要求错误动作总概率下降，不指定哪个替代动作正确；概率会随当前非错误候选的分布重新分配。它不包含单独的正样本 CE、SFT 或基座 KL 正则。

`NegativeULTrainer.train_sample` 每个负反馈样本做**一次** AdamW 更新，然后回到生成。代码没有 Negative-UL 的多步配置项。

若扩展到多个已确认错误动作，理论上可以使用 \(-\log(1-\sum_{a\in\mathcal B}p_a)\)，但当前实现及本次实验没有启用该扩展。

## 4. Projected-KL 的目标构造与 loss

### 4.1 从当前策略构造冻结的投影目标

收到负反馈、尚未做本次更新时，先计算当前模型的参考分布 \(p^{(0)}\)。将错误动作概率清零，在其余候选上重新归一化：

\[
\boxed{
q(a)=
\begin{cases}
0,&a=a^-,\\[2pt]
\dfrac{p^{(0)}(a)}{1-p^{(0)}(a^-)},&a\in\mathcal G.
\end{cases}
}
\]

实现把参考候选分数 `detach().clone()`，将错误动作对应的 logit 设为 `-inf`，再 softmax。这样不必显式计算接近零的 \(1-p^{(0)}_-\)。

这个投影保留参考策略在剩余候选之间的概率比例；等价于在约束 \(q(a^-)=0\) 下最小化 \(D_{\mathrm{KL}}(q\|p^{(0)})\)。它不需要正确动作标签，也没有单独加载 teacher 模型。参考策略就是**本次负反馈更新之前的当前 LoRA 策略**，不是永远冻结的原始基座。

### 4.2 固定目标的 forward KL

训练目标为：

\[
\boxed{
\mathcal L_{\mathrm{PKL}}(\theta)
=D_{\mathrm{KL}}(q\|p_\theta)
=\sum_{a:q(a)>0}q(a)
\left[\log q(a)-\log p_\theta(a\mid c_t,\mathcal C_t)\right]
}.
\]

这是 **target-to-student 的 forward KL**，不是 \(D_{\mathrm{KL}}(p_\theta\|q)\)。实现只在 \(q>0\) 的位置计算，避免 `0 * log(0)`。目标 \(q\) 不参与反向传播，梯度为：

\[
\frac{\partial\mathcal L_{\mathrm{PKL}}}{\partial z_a}=p_a-q_a.
\]

每次负反馈默认执行 `projected_kl.optimizer_steps: 2`：

1. 用更新前的模型计算一次 \(p^{(0)}\) 和 \(q\)。
2. 第一次以当前分布拟合 \(q\)，执行一次 AdamW step。
3. 重新评分得到更新后的 \(p^{(1)}\)，**仍然拟合同一个 \(q\)**，再执行一次 AdamW step。
4. 返回环境循环；下次新的负反馈到来时，才构造新的参考分布和投影目标。

函数位置：[`projected_target`、`candidate_loss`、`ProjectedKLTrainer.train_sample`](../alfworld/continual/projected_kl.py)。不能在两次内部更新之间重新投影目标，否则会改变这个实现的含义。

### 4.3 与 Negative-UL 的精确关系

定义当前策略在剩余候选上的条件分布：

\[
r_\theta(a)=\frac{p_\theta(a)}{1-p_\theta(a^-)},\quad a\in\mathcal G.
\]

则有以下分解：

\[
\boxed{
D_{\mathrm{KL}}(q\|p_\theta)
=-\log(1-p_\theta(a^-))
+D_{\mathrm{KL}}(q\|r_\theta)
}.
\]

第一项就是 Negative-UL；第二项约束剩余动作的相对分布不要偏离更新前的投影目标。

在第一次更新开始时，\(r_{\theta_0}=q\)，所以第二项的值和一阶梯度均为零。此时两种 loss 的值、对候选分数的梯度、进而对参数的梯度都相同。若模型、样本、优化器状态和计算路径相同，且只做一步，Projected-KL 与 Negative-UL 理论上给出相同的参数更新（浮点误差除外）。

在第二步，Projected-KL 使用固定 \(q\)，通常不再等于只计算 UL。固定目标约束的是候选分布，不是参数距离，也不保证每次实际更新都精确保留剩余候选的比例。

**当前配置同时改变了目标和更新预算：Negative-UL 每次一步，Projected-KL 每次两步。比较实验成绩时不能把所有差异都归因于额外的 KL 约束。**

## 5. 显存控制与精确两遍求导

两种 trainer 共用 `NegativeULTrainer.score`。对每个候选，都以完整前缀加候选 token 做 causal decoder forward，只取对应动作预测位置的 hidden states，并按 32 个目标 token 一块投影到词表。logits 转为 FP32，log-prob 和 loss 也以 FP32 计算。这里的分块仅用于词表投影，**不切断上下文、不减少候选、不按块分别归一化或更新**。

为避免同时保存所有候选的完整计算图，每个 optimizer step 使用两遍链式求导：

```python
# 第一遍：无梯度计算所有候选的序列分数
z = score_all_candidates(no_grad=True)
z.requires_grad_(True)
loss = UL(z, bad)                  # 或 KL(q_fixed || softmax(z))
coefficients = grad(loss, z)       # 每个候选都有一个 dL/dz

# 第二遍：逐候选重算，有梯度；在一个 optimizer step 前累积完毕
optimizer.zero_grad()
for candidate, coefficient in zip(candidates, coefficients):
    (score(candidate) * coefficient).backward()
clip_grad_norm_(trainable_parameters, 1.0)
optimizer.step()
```

其依据是：

\[
\nabla_\theta\mathcal L
=\sum_a\frac{\partial\mathcal L}{\partial z_a}\nabla_\theta z_a.
\]

第一遍虽然没有保存模型计算图，但导数系数包含全部 softmax 分母项；第二遍重算每一个候选的梯度，所以不是省略分母梯度的近似。两遍之间不能更新参数，并强制 `training.dropout=0`。Projected-KL 第一步复用参考分数；后续子步骤重新评分，但不重算 \(q\)。

训练时开启 gradient checkpointing 和 input-require-grads，关闭 KV cache；最后清空梯度、关闭 checkpointing，恢复 eval 与 cache。下一步交互重新生成，不复用更新前的生成 KV cache。计算量随候选数量与上下文长度增长；Projected-KL 默认两步意味着约两倍的候选评分/反传轮数。

## 6. 超参数与实验设置

下表对应仓库专用 YAML；已保存的 Negative-UL 三次实验与 Projected-KL 三次实验配置在这些算法超参数上相同。运行使用的物理 GPU 和输出目录可以不同。

| 参数 | Negative-UL | Projected-KL |
|---|---|---|
| 方法名 | `negative-ul` | `projected-kl` |
| 基座 | 本地 Qwen3.5-9B | 同左 |
| 权重加载 | BF16，不量化 | 同左 |
| 可训练参数 | LoRA；基座冻结 | 同左 |
| LoRA target | `all-linear`，排除 `lm_head` | 同左 |
| LoRA rank / alpha | 16 / 32，缩放因子为 2 | 同左 |
| LoRA dropout / bias | 0 / `none` | 同左 |
| 优化器 | AdamW | 同左 |
| 学习率 / weight decay | `1e-5` / 0 | 同左 |
| AdamW betas / eps | 沿用默认值 `(0.9, 0.999)` / `1e-8` | 同左 |
| 梯度范数上限 | 1.0 | 同左 |
| 每次负反馈 optimizer steps | 固定 1 | `projected_kl.optimizer_steps=2` |
| loss 权重 | 1，无额外正则项 | 1，无额外正则项 |
| 候选评分 | 动作和 EOS 的 log-prob 求和 | 同左 |
| 候选 softmax 温度 | 1，代码固定 | 同左 |
| logits 投影分块 | 32 个目标 token，trainer 类属性 | 继承同一实现 |
| 更新粒度 | 一个负反馈样本，包含全部候选 | 同左 |
| 训练设备 | 单 GPU；目前不支持多 GPU layer sharding | 同左 |

优化器由 [`NTPTrainer.__init__`](../alfworld/continual/training.py) 创建，但这两种方法不调用 NTP 的分块训练循环。没有学习率调度器、warmup 或跨样本 mini-batch；逐候选反传是在同一个 loss 下累积梯度，不是多个独立训练样本。

| 任务与生成参数 | 当前设置 |
|---|---|
| 任务集 | `valid_heat_place`：16 个 valid_seen + 23 个 valid_unseen，共 39 局 |
| manifest | `configs/manifests/valid_heat_place.seed42.json` |
| 顺序与 seed | shuffled，42 |
| 每局上限 | `max_steps=50`，包括格式错误尝试 |
| 生成策略 | `do_sample=False`，自由生成 ReAct 输出 |
| 单次输出预算 | `max_new_tokens=1024` |
| 输入预算 | `max_input_tokens=65536`，并受模型位置容量减输出预算限制 |
| 输入超长策略 | `context_overflow=truncate_oldest`，沿用实际生成后的输入 token |
| 动作提示模式 | `action_mode=admissible` |
| 跨局状态 | `memory_scope=stream`，保留参数和优化器，不保留跨局 ICL |
| checkpoint | 每 10 局及最终完成时保存匹配的参数、优化器和结果 |

配置中还会出现 `window_episodes=4`、`training.chunk_tokens=2048`。前者不使这两种方法获得跨局 ICL，后者不用于这两种 trainer 的上下文切块。`ascent` 和 `sft` 配置块也不控制这里的优化过程。Projected-KL 的步数只由 `projected_kl.optimizer_steps` 控制，不能用 `ascent.optimizer_steps` 替代。

## 7. 一个具体数值例子

以下是解释 loss 的假设例子，不是某次运行的实测概率。模型执行：

```text
Reasoning: I should heat mug 1 using microwave 1.
Action: heat mug 1 with microwave 1
```

环境返回 `Nothing happens.`。执行前候选去重并补入该动作后，假设共有以下四个动作，按照前述序列打分和 softmax 得到：

| 候选动作 | 更新前候选概率 \(p^{(0)}\) | 投影目标 \(q\) |
|---|---:|---:|
| `heat mug 1 with microwave 1`（本次负动作） | 0.60 | 0 |
| `open microwave 1` | 0.20 | 0.50 |
| `inventory` | 0.15 | 0.375 |
| `look` | 0.05 | 0.125 |

如果负动作原本不在 admissible 列表里，它仍会被补入这个评分集合。候选的评分前缀都包含同一段本步 Reasoning；例如为 `inventory` 评分时，并不会重新生成一段偏好 `inventory` 的 Reasoning。

### 7.1 第一次更新

Negative-UL：

\[
\mathcal L_{\mathrm{UL}}=-\log(1-0.60)=-\log 0.40\approx0.916291.
\]

Projected-KL 先固定 \(q=(0,0.5,0.375,0.125)\)，得到：

\[
\mathcal L_{\mathrm{PKL}}
=0.5\log\frac{0.5}{0.2}
+0.375\log\frac{0.375}{0.15}
+0.125\log\frac{0.125}{0.05}
\approx0.916291.
\]

两种 loss 对四个候选分数的梯度均为：

\[
\nabla_z\mathcal L=(0.60,-0.30,-0.225,-0.075).
\]

梯度下降会降低错误动作的分数，并提升其他候选的分数。这里给出的是对候选分数的梯度；实际更新作用于共享 LoRA 参数，使用 AdamW，因此不能直接把梯度向量当作下一次概率变化量。

### 7.2 更新后为什么出现差别

假设第一次参数更新后的候选概率变为：

\[
p^{(1)}=(0.40,0.40,0.15,0.05).
\]

这只是用于展示差异的假设状态，不是声称上述一步 AdamW 必然产生该分布。

如果此时再次计算 UL：

\[
\mathcal L_{\mathrm{UL}}(p^{(1)})=-\log0.60\approx0.510826,
\]

其候选分数梯度为：

\[
(0.40,-0.266667,-0.10,-0.033333).
\]

而 Projected-KL 的第二步仍使用原来的 \(q\)：

\[
\mathcal L_{\mathrm{PKL}}(p^{(1)})\approx0.569717,
\qquad
\nabla_z\mathcal L_{\mathrm{PKL}}=p^{(1)}-q
=(0.40,-0.10,-0.225,-0.075).
\]

相较于继续做 UL，Projected-KL 在这个状态下对 `inventory` 和 `look` 提供更强的提升方向，以接近冻结目标中的相对比例。两者的 loss 差约为 0.058892，正是 \(D_{\mathrm{KL}}(q\|r_{\theta_1})\)。

实际默认 Negative-UL 在第一步后已经返回环境；这里计算第二次 UL 只是为了比较两个目标函数，并不表示当前 Negative-UL 默认也训练两步。

## 8. 边界行为与日志解释

构造样本时会检查原始输出、解析动作与实际执行动作是否一致，能否在原始生成 token 中找到 `Action:` 边界，是否有结束 EOS，以及去掉 EOS 后的解码是否与 `raw_output` 一致。失败会记录 `action_alignment`、`action_token_boundary`、`missing_action_terminator` 或 `output_token_alignment` 并跳过；单候选则记录 `no_alternative_candidates`。

候选评分复用生成时已裁剪好的前缀，不再单独截断每个候选。若前缀加候选超过模型位置容量，明确报错。非有限分数、loss 或梯度同样导致报错，不静默跳过。checkpoint 以 episode 边界事务提交；中途出错后从最近已提交边界恢复，未提交的 episode 会重新执行，没有局内精确恢复。

| 日志字段 | 含义 |
|---|---|
| `negative_feedback` | 本步是否满足负反馈触发条件；不保证最终能构造有效训练样本 |
| `sample` | 候选列表、共同前缀 token、各候选 token、负动作索引、是否补入负动作 |
| `negative_ul_update` | Negative-UL 的一次逐步更新事件 |
| `projected_kl_update` | Projected-KL 的一次负反馈处理事件，默认含两次子更新 |
| `candidate_log_probs` | 原始序列得分 \(z\)，不是候选归一化概率的 log 值 |
| `negative_probability` | 对应 optimizer step 执行前的负动作候选概率 |
| `reference_log_probs` / `target_probabilities` | Projected-KL 本次更新前的参考分数与固定 \(q\) |
| `substeps` | Projected-KL 每个 optimizer step 的 loss、分数、概率、梯度范数 |
| `loss` | UL 为唯一一步更新前的 loss；Projected-KL 顶层为最后一个子步骤更新前的 loss，不是两步均值，也不是最后更新后的重新评估值 |
| `optimizer_steps` / `model_version` | 前者计优化器更新次数；后者计完成的负反馈样本训练次数 |

不要用 `negative_probability` 直接声称更新后概率降低；它记录的是该子步骤更新前的值。若要确认下降，需要在同一候选集合、同一前缀下进行更新后评分。

## 9. 运行入口与代码索引

在仓库根目录使用独立输出目录启动新的实验：

```bash
.venv/bin/python -m alfworld.continual \
  --config configs/continual-negative-ul.yaml \
  --device cuda:0 --output-dir runs/heat-place-negative-ul-new

.venv/bin/python -m alfworld.continual \
  --config configs/continual-projected-kl.yaml \
  --device cuda:0 --output-dir runs/heat-place-projected-kl-new
```

上面两条命令使用同一张卡，适合依次执行；并行执行时应选择不同的空闲 GPU。独立重复实验使用新的输出目录，不传 `--resume`；只有继续同一实验才使用恢复参数。

| 位置 | 主要职责 |
|---|---|
| [negative_ul.py](../alfworld/continual/negative_ul.py) | 负反馈筛选、候选构造、token 对齐、共享评分、UL loss 与单步训练 |
| [projected_kl.py](../alfworld/continual/projected_kl.py) | 固定投影目标、forward KL 与多步训练 |
| [model.py](../alfworld/continual/model.py) | 基座与 LoRA 初始化、生成、样本构造和 trainer 路由 |
| [methods.py](../alfworld/continual/methods.py) / [runner.py](../alfworld/continual/runner.py) | 即时更新调度、版本、日志、episode 汇总和 checkpoint |
| [config.py](../alfworld/continual/config.py) / [training.py](../alfworld/continual/training.py) | 配置默认值和验证、AdamW 初始化及状态保存 |
| [Negative-UL 配置](../configs/continual-negative-ul.yaml) / [Projected-KL 配置](../configs/continual-projected-kl.yaml) | heat&place 39 局的可执行设置 |
| [test_negative_ul.py](../tests/test_negative_ul.py) / [test_projected_kl.py](../tests/test_projected_kl.py) | 数学公式、两遍梯度、在线调度、恢复、小模型训练及固定目标检查 |
