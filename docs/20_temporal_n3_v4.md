# Temporal N3 v4: Candidate-Wise Multimodal History Routing

## Research objective

Dialogue emotion-recognition systems often append recent context, pool all past turns, or use attention weights as a proxy for history value. Temporal N3 v4 treats historical context as a **conditional action**: a candidate may help, be irrelevant, or damage the prediction. The target is therefore to identify useful strictly-past candidates and modalities while retaining an explicit current-only route when the frozen safety conditions are not met.

The intended experimental argument has two linked parts:

1. Matched current-only and history-enabled comparisons produce development-time, out-of-fold utility and risk targets; and
2. candidate-wise multimodal routing uses those targets to decide whether historical information should participate in a prediction.

The first part is a protocol objective. It must not be described as completed until its causal/OOF producer, receipts, and checks are verified on the new version.

## Implemented architecture

For a batch of `B` current utterances and `K` frozen candidate histories, the module receives six streams: current and history text, audio, and visual representations. It contains the following implemented components.

| Component | Function |
|---|---|
| Candidate-wise batched relation | Computes a masked low-rank `B x K x 3 x 3` grid of current-modality to history-modality relations without an outer Python loop over the nine pairs. |
| Bidirectional utility heads | Produces per-candidate add-benefit and deletion-risk evidence for text, audio, visual, and joint history. The intended labels are OOF-only. |
| Utility-Risk Bottleneck | Compresses candidate relation, utility/risk, and temporal features into a bounded routing state and risk score. |
| Two-level routing | Applies modality gates first, then candidate gates conditioned on the bottleneck. |
| Current-only route | Emits current-only logits when `history_authorized` is false. It is a separate classifier head, although the present development code still shares upstream encoders. |

The reference implementation is [`../temporal_n3/`](../temporal_n3/). Its unit tests verify candidate dimensions, masks, gate routing, and deterministic fallback authorization; they do not train on a corpus or prove utility quality.

## Temporal candidate and resampling protocol

The current protocol is frozen in [`../temporal_n3/TEMPORAL_RESAMPLING_PROTOCOL.md`](../temporal_n3/TEMPORAL_RESAMPLING_PROTOCOL.md).

1. Build candidates only from strict-past turns in the same dialogue.
2. Partition history into chronological older, middle, and recent bands, then sample without replacement using a predeclared recent-heavy policy recorded in the candidate manifest.
3. Train a round from fresh initialization and fresh optimizer state. It may consume its own frozen candidate manifest, never a prior checkpoint, optimizer state, prediction, loss, or metric.
4. Retain candidates using development-OOF utility/risk decisions. Refill only from previously unvisited history, allocating empty slots according to first-pass retention by temporal band.
5. Authorize at most two resampling fallbacks using the predeclared joint conditions: Macro-F1 gain, Weighted-F1 gain, NLL improvement, harm reduction, and fraction of improved seeds.
6. If a final safety gate fails, route to `current-only`.

The public policy module currently exposes a bounded development pool (`max_candidates=6`) and a deterministic reference selection routine. A production percentage-based candidate manifest must record the available-history count, band boundaries, random seed, sampled identifiers, and realized quotas; it must be preflighted before training. This prevents a presentation of a design intention as an already-completed production runner.

## Non-leakage and evaluation contract

- Training updates use train only; development is limited to validation, early stopping, and predeclared OOF selection.
- Test labels, test features, and test metrics cannot enter candidate sampling, utility targets, model selection, threshold setting, or fallback authorization.
- Code, data/feature manifests, partitions, selector configuration, thresholds, and receipts freeze before the sole final test.
- Each dataset keeps its own official split or fold contract. Aggregate results from different stages are not directly comparable.
- A safety-gate failure remains a valid outcome. It means use the current-only route rather than relax a threshold or search until history wins.

## Verification status

| Item | Status |
|---|---|
| Variable-candidate `K x 3 x 3` module and masks | Implemented and synthetic-unit-tested |
| Utility-Risk Bottleneck and two-stage candidate routing | Implemented and synthetic-unit-tested |
| Fresh-round and maximum-round API guard | Implemented and synthetic-unit-tested |
| Full causal/OOF utility supervision | Not yet verified end to end |
| Selector, refill, and resampling on official data | Not yet verified end to end |
| MELD v4 official train/dev run | Preparation in progress; test unopened |
| IEMOCAP / EmotionTalk v4 official result | Not available |
| SOTA comparison | Not available until split, inputs, metrics, and evaluation protocol are matched |

## Relationship to legacy material

`模型/n3_affect/` is the earlier ComposerN3 codebase. It uses a fixed `K=3` pooled-history pathway and older experiment scripts. Its contracts, documentation, and published aggregate evidence remain historical records. Temporal N3 v4 reuses feature configuration/encoder interfaces where appropriate, but is an independent architecture and experiment version; old checkpoints, results, authorization, and selectors cannot be reused as v4 evidence.
