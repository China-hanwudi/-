# CARMA-Affect / TemporalN3

当前公开模型主线是面向 **M3ED、CMU-MOSEI、CH-SIMS_v2** 的 TemporalN3。活动代码位于 [`模型/n3_affect/`](模型/n3_affect/)，历史数据集入口不再属于当前训练流程。

当前主线与历史文档边界见 [`模型/ACTIVE_DATASETS.md`](模型/ACTIVE_DATASETS.md) 和 [`模型/GITHUB_ACTIVE_MAINLINE.md`](模型/GITHUB_ACTIVE_MAINLINE.md)。

## 模型结构

当前 T/A/V 经过统一投影后，与严格过去的历史候选进入候选级 `3x3` 跨模态关系编码。双向效用头估计加入/删除历史证据的影响，模态门控和候选门控执行风险过滤，`CandidateRiskFallback` 在风险过高时硬回退到独立 current-only 分支。分类和连续回归使用独立任务头。

模型保留：有界历史残差、缺失模态 mask、speaker-conditioned 状态、反事实效用监督、冗余抑制和空历史严格回退。历史由 packed `history_index` 规范为 oldest-to-newest、右对齐 K=3；未来或无效索引被屏蔽。

## 活动任务

| 数据集 | 任务 | 特征维度 | checkpoint 选择 |
|---|---|---|---|
| M3ED | 七分类 | T/A/V = 768/1024/342 | valid Weighted-F1，其次 Macro-F1 |
| CMU-MOSEI | 连续回归 | T/A/V = 768/74/35 | valid MAE |
| CH-SIMS_v2 | 连续回归 | 从 packed manifest 读取 | valid MAE |

数据、预计算特征、密钥和训练产物不提交到仓库。test split 只允许在协议冻结后的最终评估中使用，不用于调参、早停、风险校准或 checkpoint 选择。

## 工程检查

```bash
python -m compileall 模型/n3_affect
python -m pytest temporal_n3/tests -q
```

正式训练入口和 A100 预检见 [`模型/run_a100_training.sh`](模型/run_a100_training.sh)。当前仓库只记录源码和实验合同，不声称已经完成正式性能复现。
