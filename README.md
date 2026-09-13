# TemporalN3 UnifiedTemporalFinal

当前仓库的模型主线是 `TemporalN3_UnifiedTemporalFinal_20260913`，面向 M3ED 与 CMU-MOSEI。实现位于 [`模型/n3_affect/`](模型/n3_affect/)，不是旧版固定历史 ComposerN3 的说明稿。

## 当前模型

模型将当前 T/A/V 表征、严格过去的历史候选和可用性掩码送入统一的候选级路由：

```text
T/A/V current + strict-past history
        │
        ├─ SixWayEncoders
        ├─ current fusion + Transformer current-only anchor
        ├─ SharedThreeByThree: alignment / complementarity / conflict
        ├─ BidirectionalUtilityHeads + TwoLevelGate
        ├─ speaker-conditioned GRU history state
        ├─ HistoricalEvidenceController with bounded recency decay
        ├─ DynamicEvidenceRouter: candidate-level T/A/V evidence weights
        ├─ conflict_v2 risk features and uncertainty estimates
        └─ CandidateRiskFallback → hard-safe / soft / current-only route
                                      │
                                      └─ classification logits + VAD auxiliary head
```

模型输入维度默认是 text 2048、audio 1536、video 768，内部维度为 128。千问仅作为文本塔使用；千问权重、原始数据、预计算特征和训练 checkpoint 均不提交到 GitHub。

## 目录

| 路径 | 内容 |
|---|---|
| [`模型/n3_affect/`](模型/n3_affect/) | 最新分类、回归、关系、门控、动态路由和数据入口 |
| [`模型/MODEL_MANIFEST_M3ED_MOSEI_v2.json`](模型/MODEL_MANIFEST_M3ED_MOSEI_v2.json) | 当前模型版本、输入数据集和 smoke-test 状态 |
| [`docs/17_ComposerN3当前实现架构_对齐E模型_2026-08-14.md`](docs/17_ComposerN3当前实现架构_对齐E模型_2026-08-14.md) | 最新源码级结构说明 |
| [`assets/TemporalN3_UnifiedTemporalFinal_structure.svg`](assets/TemporalN3_UnifiedTemporalFinal_structure.svg) | 可编辑论文级结构图 |
| [`datasets/`](datasets/) | 数据下载、许可和校验说明；不存放原始数据 |

## M3ED 与 CMU-MOSEI

数据集稍后由用户提供。接入时必须分别建立 train/validation/test manifest，记录标签协议、模态可用性、特征维度、数据清单哈希和 split 哈希。test 只用于最终一次评估，不得用于 checkpoint、阈值或校准选择。

当前 manifest 记录的工程 smoke 状态为：分类/回归合成前向、反向和实际回归特征维度检查通过；正式训练尚未开始，因此仓库不宣称已完成的 M3ED 或 CMU-MOSEI 测试结果。

## 运行前检查

```powershell
$env:PYTHONPATH = (Resolve-Path '模型').Path
python -m compileall 模型/n3_affect
python -m pytest temporal_n3/tests -q
```

正式训练前还需要安装 PyTorch/CUDA、配置数据路径和 manifest、提供预计算三模态特征，并在独立输出目录记录代码哈希、配置哈希和随机种子。
