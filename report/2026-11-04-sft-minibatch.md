# Success-only SFT：mini-batch 与梯度累积

默认配置：`sft.batch_size=2`，`sft.gradient_accumulation_steps=4`，有效 batch 为 8 条完整交互样本。
每个成功 episode 有 N 条筛选后且不超长的样本时，`optimizer_steps=ceil(N/8)`；失败或无有效样本为 0。
每局结束提交不足 8 条的剩余样本，不跨 episode 留存梯度。

## 实现

- 左侧 padding 配合 attention mask 和按实际 token 计数的 position_ids；完整上下文仍送入 decoder。
- 回答末端对齐，仅计算当前回答对应位置的 vocabulary logits。较短回答的多余投影位置标签为 -100。
- 各回答内部先对监督 token（含 EOS）取均值，再按本次累积窗口的实际样本数归一化；因此短 mini-batch 和最后不足 8 条的窗口不会被额外降权。
- 每个 mini-batch backward；每个完整或尾部窗口只执行一次梯度裁剪和 optimizer.step。
- update 日志分别记录 `samples_trained`、`micro_batches`、`optimizer_steps` 和累计 `total_optimizer_steps`。
- 旧版逐样本 SFT 配置缺少新字段时按 1×1 解释；恢复时需显式保持 1×1，直接使用新默认 2×4 会触发配置一致性检查。

## 验证

31 项持续学习测试通过。新增真实小型 Qwen3.5 对照：7 条不同 prompt/回答长度样本，mini-batch=2、累积=2，分别以 4 条和 3 条更新；与不含 padding 的逐样本梯度参考及 batch=4 的更新结果一致。覆盖 padding mask、EOS、尾部不完整 mini-batch、尾部累积窗口、零样本与梯度清理。旧版恢复检查补充后，20 项 runner/protocol 测试单独重跑通过。

真实 Qwen3.5-9B 试跑使用 9 条接近 4096 上限的完整样本，来自历史成功轨迹集合（仅用于独立训练冒烟检查，不是在线跨 episode 累积策略）：

- 样本长度：4085, 4048, 4036, 4013, 3994, 3977, 3921, 3913, 3890 tokens。
- mini-batch：2；累积：4；5 次 forward/backward，2 次 optimizer 更新，窗口大小为 8 和 1。
- 两次窗口平均 loss：0.140147, 0.111536；梯度均有限，LoRA 参数确实更新。
- 训练耗时：43.90 秒；峰值已分配显存：24.98 GiB；单进程显存上限 30 GiB。
- 本次未启动完整在线评测，不据此判断任务成功率。

[原始试跑数据](sft-minibatch-pilot-20261104.json) · [配置](../configs/continual-success-only-sft.yaml) · [实现](../alfworld/continual/sft.py)
