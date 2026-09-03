# MELD SOTA Baseline Implementation

## Scope

`Temporal N3 v4` remains the only proposed model. It uses the 3x3 cross-modal
relation block, two-level modality/candidate gating, and the frozen strict-past
candidate-history selector. Baselines are comparison methods only.

## Sources

- MMGCN: `https://github.com/hujingwen6666/MMGCN` (ACL 2021)
- UniMSE: `https://github.com/LeMei/UniMSE` (EMNLP 2022)
- CTNet: no verified public implementation located yet; use a documented faithful
  reimplementation if a source cannot be verified.

## Anti-leakage requirements

Every method must use the same MELD train/dev split, frozen T/A/V feature contract,
five seeds (17, 29, 43, 71, 101), training budget, checkpoint rule, and scorer.
Model selection must use development data only. The official test split is opened
only once after all choices are frozen and must never affect tuning.

## Current implementation state

- Temporal N3 v4: five-seed train/dev run completed; history enhancement gate did
  not pass, so no SOTA win is claimed.
- MMGCN: source obtained; training loop requires dev-only checkpoint selection
  before execution under this protocol.
- UniMSE: source obtained; original MELD preprocessing reads train/dev/test
  together and requires a train/dev-only loader before execution.
- CTNet: implementation and protocol mapping pending.

Published numbers remain literature references until each method is reproduced
under the fixed protocol.
