# MELD Baselines

This directory is reserved for reproducible baseline adapters. Baselines are
comparison methods only; `Temporal N3 v4` remains the proposed model.

Adapters must consume the frozen MELD train/dev feature contract and may not
open test labels or test predictions until the final fixed evaluation. Every
adapter must record its upstream commit, configuration, input hashes, seed,
development checkpoint, and test-isolation receipt.

Current upstream sources:

- MMGCN: `https://github.com/hujingwen6666/MMGCN`
- UniMSE: `https://github.com/LeMei/UniMSE`
- CTNet: public source not yet verified

Do not copy published numbers into strict-result tables. Use them only as
literature references until a method is reproduced under the run contract in
`experiment/MELD_SOTA_RUN_CONTRACT.md`.
