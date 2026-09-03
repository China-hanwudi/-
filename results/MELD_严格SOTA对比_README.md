# MELD Strict SOTA Comparison

## Current status

No publishable SOTA win/loss conclusion exists yet. The existing
`MELD_TemporalN3_v4/AGGREGATE_METRICS.json` is a current-only (H0) precheck
with `optimizer_steps=0` and `training_epochs=0`. It is not a trained final
model and must not be compared against SOTA results.

## Controlled rerun protocol

Architecture equality is not required. All evaluated methods must instead use
the same official MELD split, frozen T/A/V features, strict-past history rule,
development-only model selection, seeds (17, 29, 43, 71, 101), training budget,
checkpoint rule, and scorer. Macro-F1 is primary; Weighted-F1, Accuracy, and
NLL are secondary.

Published paper numbers are literature references only. A strict comparison
requires a compatible public implementation or a documented faithful
reimplementation under the fixed protocol. Candidate methods are Temporal N3
v4, MMGCN (ACL 2021, 10.18653/v1/2021.acl-long.440), CTNet (TASLP 2021,
10.1109/TASLP.2021.3049898), and UniMSE (EMNLP 2022,
10.18653/v1/2022.emnlp-main.534).

The official test split was already opened by a previous precheck. Later runs
must freeze this protocol and never use its results for model selection; report
them as fixed official-holdout results, not a first blind test.
