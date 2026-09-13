# TemporalN3 UnifiedTemporalFinal

本目录是当前 M3ED 与 CMU-MOSEI 适配模型的源码入口。核心分类模型为 `n3_affect/model.py`，连续情感回归适配为 `n3_affect/regression_model.py`。

模型包含六路 T/A/V 编码、当前-only Transformer 锚点、`SharedThreeByThree` 关系网格、双向效用头、两级门控、历史证据控制器、same-speaker GRU 状态、候选级 `DynamicEvidenceRouter` 和 `CandidateRiskFallback`。风险条件不满足时输出 current-only 结果。

默认输入维度为 text=2048、audio=1536、video=768，内部维度为 128。千问只作为文本塔使用；权重、原始数据、预计算特征和 checkpoint 不提交到仓库。

数据集由外部提供后再配置。M3ED 与 CMU-MOSEI 必须分别维护 train/validation/test manifest，严格禁止用 test 选择 checkpoint、阈值或校准参数。

```powershell
$env:PYTHONPATH = (Resolve-Path '.').Path
python -m compileall n3_affect
```

当前工程已完成源码级 smoke 检查，正式数据集训练尚未开始。
