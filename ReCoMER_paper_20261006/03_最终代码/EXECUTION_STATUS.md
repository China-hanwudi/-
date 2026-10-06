# 创新点三句级弃权执行状态

更新日期：2026年10月2日。

## 已完成

- 从“创新点3 弃权对照”创建独立实现目录，旧基线未覆盖。
- 为历史准入定义 `none`、`legacy_null`、`sentence` 三种明确模式。
- `none` 路径不再使用空候选；`legacy_null` 保留旧空候选行为；`sentence` 使用24维预反馈统计特征和一个32隐藏单元的句级开关。
- 实现 `history_override=0|1` 诊断接口。硬拒绝将历史反馈残差置零，并从joint head的历史token mask中移除历史。
- 已用合成批次检查：非零history gamma下，强制关闭历史后扰动历史输入，deployed与joint logits不变；句级门可获得有限非零梯度。
- 6项单元测试通过，覆盖硬断路、强制开启、空历史、梯度、none模式和legacy_null参数隔离。
- 新增核心空候选竞争实现：`legacy`（B1）、`utility_softmax`（E5）、`utility_semantic`（E6）和`utility_sparsemax`（E7）。四者均保留真实历史候选与空候选；E5/E6只在归一化前校准空候选logit，E7替换为稀疏归一化，不改历史句子和空候选输入。
- 新增空候选效用校准网络、语义特征开关、sparsemax，以及 `history_override=0` 下历史 token 的完整硬断路；服务器端核心实现路径为`code_core_empty_v2`。
- 与 `00_原始代码` 在同随机初始化和同合成输入上的none模式前向逐元素一致。
- 服务器只读审计确认MELD训练包有9989条样本、3个历史槽及 `diaN_uttN` ID；已按现场资源使用 GPU 0/6/7 的独立队列，未停止任何既有进程。
- 已在服务器新建独立实验根目录并生成MELD训练侧对话划分：931个fit对话、107个inner_dev对话，inner_dev为1003/9989条样本（10.04%）。manifest SHA256为`33b9162ce2139e414cadaed1742da0567b09024900937d23a0292362ee899d32`。

## 当前进行中

- 已将独立句级弃权实现上传至服务器的独立实验目录，未覆盖服务器既有代码；真实 MELD 冒烟训练已通过（`SMOKE_PASS`、`test_read=false`）。
- 核心版 seed=7 的 B1/E5/E6/E7 均已 `TRAIN_COMPLETE`，无 OOM/Traceback；inner-dev Weighted-F1分别为0.5916/0.5923/0.5923/0.5918。seed=13、17 的同矩阵已启动，仍只使用训练侧对话级 fit/inner_dev 划分。
- 代码审计发现并修复了 utility 变体的历史残差旁路：原实现只屏蔽 joint 历史 token，却仍将历史 pooled residual 注入 solo/deployed；现在空候选硬胜出时 `history_gate_applied=0`，两条路径都断开。新增测试覆盖该行为，7项单元测试全部通过。
- 因此前一轮 E5/E6/E7 正式指标作废。修复后的三 seed结果为：B1 0.5967±0.0069，E5 0.5970±0.0090，E6 0.5968±0.0088，E7 0.5957±0.0091（inner-dev Weighted-F1，均值±总体标准差）。当前仍不能宣称任何改进稳定优于B1。
- 已生成冻结教师效用缓存`artifacts/history_utility_cache_v1.json`，8986条，覆盖学生fit且排除inner-dev；缓存监督冒烟训练已通过。正式缓存监督矩阵尚未合并到最终结论。

## 未开始

- 3折冻结教师和历史效用缓存（当前核心版先以端到端任务损失训练校准器，缓存接口待补齐）。
- seed=13/17 完成后，再决定是否扩展10 seed矩阵和M3ED辅助复验。

## 已知限制

- 核心空候选校准器已可端到端训练；冻结教师效用缓存的显式监督接口尚未接入，因此当前结果只作为机制筛选，不作为最终论文结论。
- sentence模式暂不支持mpath，以防留下未受开关控制的历史旁路。
- `utility` 教师目标、正式 10-seed 矩阵、正式 valid 评估与 M3ED 辅助复验仍未开始；test 未读取。
