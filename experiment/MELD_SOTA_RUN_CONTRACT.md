# MELD SOTA Run Contract

This contract defines the only comparison that may be reported as a strict
comparison for the proposed `Temporal N3 v4` model.

## Fixed protocol

- Dataset: official MELD split
- Inputs: frozen text/audio/video feature contract used by Temporal N3 v4
- Seeds: `17, 29, 43, 71, 101`
- Primary metric: Macro-F1
- Secondary metrics: Weighted-F1, Accuracy, unweighted NLL
- Checkpoint selection: development split only
- History rule: strict past context only
- Test split: one fixed evaluation after all code, hyperparameters and seeds are frozen

## Proposed model

The proposed model is `Temporal N3 v4`, including the 3x3 cross-modal relation
block, two-level modality/candidate gating, and OOF-frozen candidate history
selection. `current-only` is its fail-closed fallback, not a replacement model.

## Baselines

- MMGCN: https://github.com/hujingwen6666/MMGCN
- UniMSE: https://github.com/LeMei/UniMSE
- CTNet: no verified public implementation has been found; a faithful
  reimplementation must be documented before its result can be called strict.

The original MMGCN and UniMSE runners use incompatible feature formats and/or
inspect the test split during training. They must be adapted to this contract
before execution. Published paper values are reference-only until reproduced.

## Required receipts

Each method must save its source revision, input hashes, seed, config hash,
development checkpoint identity, per-example development predictions, and an
explicit `test_data_accessed=false` training receipt. Any run that violates
these requirements is excluded from the strict comparison table.
