# Legacy ComposerN3 package

This directory preserves the earlier **ComposerN3** implementation and its MELD-oriented scripts so prior experiments, source snapshots, and interfaces remain auditable.

It is **not** the reference implementation for the current Temporal N3 v4 research direction. ComposerN3 pools a fixed strict-past history budget (`K=3`) before its shared `3 x 3` relation and gate stack. Temporal N3 v4, implemented in [`../temporal_n3/`](../temporal_n3/), retains a variable candidate axis, computes `K x 3 x 3` relations in batch, adds a Utility-Risk Bottleneck, and specifies an auditable resampling/fallback protocol.

The feature configuration classes and six-way encoders in `n3_affect/` are currently reused by the v4 development module; that reuse does not make the two architectures or their results interchangeable.

Do not start an old `run_meld_*.sh` script as a v4 run. A v4 experiment requires its own frozen candidate manifest, train/dev partition contract, source/config hashes, and test-closed preflight. See [`../docs/20_temporal_n3_v4.md`](../docs/20_temporal_n3_v4.md).
