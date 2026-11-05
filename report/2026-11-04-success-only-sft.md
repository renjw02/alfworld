# Success-only SFT 实现与试跑

> 本文记录最初逐样本更新版本。后续 mini-batch/梯度累积实现见 [更新报告](2026-11-04-sft-minibatch.md)。

时间：2026-11-04T07:54:26.615280+00:00。

## 已实现的协议

- 新方法 `success-only-sft` 与旧 NTP 方法独立。仅在最终成功的 episode 中训练；逐步过滤格式错误、生成截断、未执行/非法动作、`Nothing happens.` 以及 observation 不变的非终局步骤。环境没有逐步进展标签，因此“成功交互”采用上述有效交互代理定义，不保证最优路径。
- 一条样本对应一次交互，包含 system、任务描述、之前最多 5 次交互的 observation/完整输出/feedback、当前 observation 和合法命令、本次 reasoning/action 及 EOS。之前的错误交互可作为历史上下文；它们本身不成为目标。
- 标签仅覆盖本次回答和 EOS；历史、system、task、当前 observation 都不求 loss。训练回答前缀与生成使用同一个 generation template。
- 新方法推理也使用同样的历史窗口。没有未来反馈或整局成功信息泄露到 prompt。
- 不调用 token_chunks，不重排/切断回答。每条样本独立完成一次 optimizer step；超长样本整条跳过，保留长度和原因。沿用现有 LoRA/AdamW 超参和事务 checkpoint，支持 resume 与 task_type 重置。

## 长度检查与上限选择

用旧 success-only NTP 的三个 stream 运行中 207 个成功局检查：过滤 118 次非法/未执行动作、13 次格式/截断错误、3 次 observation 不变，得到 2653 条候选样本。全部 2653 条通过新训练样本与新 rollout 上下文一致性断言。

| 总长度上限 | 超限样本 | 比例 |
|---|---:|---:|
| 2048 | 755/2653 | 28.46% |
| 4096 | 7/2653 | 0.26% |
| 8192 | 0/2653 | 0.00% |

样本长度中位数 1471，P95=3016，P99=3702，最大 4653 tokens。保留原始 SFTConfig 默认 2048 供初始审计；新方法 YAML 明确使用 **4096**，兼顾完整历史覆盖率和显存。这里统计的是“若限制长度会超限”的比例，实际实现不会截断，而是跳过这些样本。该比例来自旧轨迹，新方法后续应持续观察 `update.json.rejected`。

## Qwen3.5-9B 真实训练试跑

两组都从本地基础模型重新开始，选择全体候选中接近各自上限的 3 条样本，每条均有完整 5 次历史。使用 GPU 1，单进程 allocator 上限 30 GiB；没有修改或停止其他实验，也没有将试跑参数覆盖到实验 checkpoint。

| 上限 | 实际样本长度 | 每步 loss | 峰值已分配显存 | 训练耗时 |
|---|---|---|---:|---:|
| 2048 | 2048, 2046, 2046 | 0.14995, 0.42317, 0.04114 | 19.11 GiB | 7.27 s |
| 4096 | 4085, 4048, 4036 | 0.03460, 0.01274, 0.02709 | 21.08 GiB | 13.36 s |

两组 loss、gradient norm 均有限，LoRA 权重确实更新。试跑用于验证训练可执行性，不是效果对比；这些样本来自不同交互，loss 也不能直接横向比较。

最初按全序列计算 logits 的 4096 试跑曾触及 30 GiB allocator 上限。最终实现使用 Qwen3.5 的 `logits_to_keep` 仅投影当前回答的预测位置，decoder 仍接收完整 prompt 和回答前缀。通过标准因果移位预测完整回答（含 EOS），不会为无监督 prompt 生成庞大词表 logits，也没有重新引入 token 切块。小型真实 Qwen3.5 测试独立验证该 loss 与全序列 response-masked cross entropy 相等。

## 验证与使用

- 29 项 `test_continual*.py` 测试通过，包括样本组成、精确前缀、5 次历史、无未来信息、目标 mask/EOS、错误筛选、失败局不更新、长度边界、禁止调用旧 chunk 逻辑、真实小模型 LoRA 更新、冻结基础权重、参数/optimizer 恢复、task_type 重置，以及 runner checkpoint/resume。
- CLI 已注册新方法，方法 YAML 经 load_config/validate 验证。
- 尚未启动完整在线评测；没有声称提升 ALFWorld 成功率。

```bash
.venv/bin/python -m alfworld.continual \
  --config configs/continual-success-only-sft.yaml \
  --device cuda:0 --output-dir runs/success-only-sft-seed42
```

- [实现](../alfworld/continual/sft.py)
- [配置](../configs/continual-success-only-sft.yaml)
- [可复现的审计/试跑脚本](../scripts/pilot_success_only_sft.py)
- [试跑数据、来源与源文件哈希](success-only-sft-pilot-20261104.json)
- [测试](../tests/test_continual_sft.py)
