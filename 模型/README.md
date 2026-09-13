# TemporalN3 UnifiedTemporalFinal（M3ED + CMU-MOSEI）

> 2026-09-13 同步：当前实现已升级为候选级动态证据路由与 `CandidateRiskFallback`，并同时保留分类与回归适配入口。正式数据集与千问权重不随仓库提交。

本目录保留早期 ComposerN3 接口和历史脚本，便于审计；最新 v4 研究方向以仓库根目录的 `temporal_n3/` 和本次同步的 `n3_affect/` 代码为准。

This directory preserves the earlier **ComposerN3** implementation and its MELD-oriented scripts so prior experiments, source snapshots, and interfaces remain auditable.

It is **not** the reference implementation for the current Temporal N3 v4 research direction. ComposerN3 pools a fixed strict-past history budget (`K=3`) before its shared `3 x 3` relation and gate stack. Temporal N3 v4, implemented in [`../temporal_n3/`](../temporal_n3/), retains a variable candidate axis, computes `K x 3 x 3` relations in batch, adds a Utility-Risk Bottleneck, and specifies an auditable resampling/fallback protocol.

The feature configuration classes and six-way encoders in `n3_affect/` are currently reused by the v4 development module; that reuse does not make the two architectures or their results interchangeable.

Do not start an old `run_meld_*.sh` script as a v4 run. A v4 experiment requires its own frozen candidate manifest, train/dev partition contract, source/config hashes, and test-closed preflight. See [`../docs/20_temporal_n3_v4.md`](../docs/20_temporal_n3_v4.md).
