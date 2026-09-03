# IEMOCAP Performance Comparison

`Temporal N3 v4` is the proposed model. Its 3x3 cross-modal relation block,
two-level gating, and strict-past candidate history remain unchanged.

## Reported metrics

- Primary: Macro-F1 and UAR
- Secondary: Accuracy and Weighted-F1
- Report: mean +/- standard deviation over fixed seeds or folds

## Comparison policy

Compare against representative IEMOCAP SOTA methods using the same modality
scope whenever possible. Results reproduced from code and results quoted from
papers must be separate rows. Quoted values are reference evidence when split,
features, or scoring differ; they are not presented as reproduced numbers.

Every reproduced run records the dataset split, modalities, seed/fold, config,
source revision, input identity, checkpoint rule, and final metrics. Test data
must not be used for tuning or checkpoint selection.

## Goal

Establish whether Temporal N3 v4 is competitive or superior on IEMOCAP, with a
transparent comparison table and uncertainty estimates. No claim is made until
the corresponding result files and provenance receipts exist.
