# 新模型结构审计与修复记录（2026-10-03）

本次结论来自实际代码与运行测试，不以外部截图作为依据。

## 已确认的问题

1. Shapley teacher 原先只对历史反馈后的 current tokens 做未加权子集评估，而 `closed_loop` 部署会用 detached deployment weights 缩放 current tokens。两条路径的 token 语义不完全一致。
2. `mpath` 的 memory mask 原先只使用 history/modality/speaker mask，没有乘 history admission gate；被历史弃权门控拒绝的内容可能从 memory path 重新注入。
3. EvidenceRouter 原先只接收 current embeddings、solo logits 和 modality mask，没有显式的 history keep/applied/has-history/abstention 状态。
4. M1 的辅助训练使用未加权 counterfactual token view，与 weighted joint deployment 不一致；现在对 `joint_softgate`/`closed_loop` 明确禁止该组合，避免训练目标和部署路径错位。

## 修复

- `measure_shapley(..., token_weight=...)` 支持复用 detached deployment weights；closed-loop 训练时 teacher 使用同一权重语义。
- mpath memory mask 乘以 history admission；`history_override=0` 会切断 history pool 和 mpath 两条路径。
- EvidenceRouter 新增可选 4 维 history state；旧 checkpoint 默认维度为 0，保持兼容；新 evidence 训练配置使用 4 维状态。
- `M1 + weighted joint deployment` 在配置校验阶段直接拒绝。
- `train_gate.py` 的冻结门控校准缓存并传递相同的 history state。

## 验证

- 服务器端编译通过。
- 12 个单元测试通过，其中 3 个为本次新增审计测试：EvidenceRouter 状态敏感性、mpath history override 隔离、M1 组合拒绝。
- M3ED_textQwen 强文本表示真实 smoke 通过（`SMOKE_PASS`，valid-only，test_read=false）。

正式训练与最终 test 使用服务器 GPU 0 的队列脚本运行；模型选择只读取 valid，训练完成后才由独立 test 队列读取 test.pt。
