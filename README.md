# CARMA-Affect / TemporalN3

当前公开模型主线是面向 **M3ED、CMU-MOSEI、CH-SIMS_v2** 的 TemporalN3 v10 exploratory model。活动代码位于 [`模型/n3_affect/`](模型/n3_affect/)，冻结快照位于 [`模型/TemporalN3_TopJournal_v10_20260914/`](模型/TemporalN3_TopJournal_v10_20260914/)。MELD、IEMOCAP、EmotionTalk 不属于当前训练、基准或 SOTA 流程。

当前主线与历史文档边界见 [`模型/ACTIVE_DATASETS.md`](模型/ACTIVE_DATASETS.md) 和 [`模型/GITHUB_ACTIVE_MAINLINE.md`](模型/GITHUB_ACTIVE_MAINLINE.md)。

## 模型结构

当前 T/A/V 经过输入归一化和零初始化残差瓶颈适配器后，由带 modality mask 的 Transformer 聚合。当前单模态标签锚定头、候选级 `3x3` 跨模态关系、反事实效用与风险过滤共同工作；`CandidateRiskFallback` 在风险过高时硬回退到独立 current-only 分支。分类和连续回归使用独立任务头。

模型保留：有界历史残差、缺失模态 mask、speaker-conditioned 状态、反事实效用监督、冗余抑制和空历史严格回退。历史由 packed `history_index` 规范为 oldest-to-newest、右对齐 K=3；未来或无效索引被屏蔽。

## 活动任务

| 数据集 | 任务 | 特征维度 | checkpoint 选择 |
|---|---|---|---|
| M3ED | 七分类 | T/A/V = 768/1024/342 | valid Weighted-F1，其次 Macro-F1 |
| CMU-MOSEI | 连续回归 | T/A/V = 768/74/35 | valid MAE |
| CH-SIMS_v2 | 连续回归 | 从 packed manifest 读取 | valid MAE |

数据、预计算特征、密钥和训练产物不提交到仓库。v10 建立于 v9 test 已封存评估之后，因此旧 test 不得再次用于调参、早停、风险校准或 checkpoint 选择；新性能结论需要新的冻结评估协议。当前代码仍使用 packed 特征，不声称已经接入 HuBERT、emotion2vec、CLIP 或 Swin。

## 工程检查

```bash
python -m compileall 模型/n3_affect
python -m pytest temporal_n3/tests -q
```

正式训练入口和 A100 预检见 [`模型/run_a100_training_v10.sh`](模型/run_a100_training_v10.sh)，交接说明见 [`模型/TRAINING_AGENT_GUIDE_v10.md`](模型/TRAINING_AGENT_GUIDE_v10.md)。当前仓库只记录源码和实验合同，不声称已经完成 v10 正式性能复现。
