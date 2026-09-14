# TemporalN3 v10 model revision

This is a post-v9-test exploratory branch. It does not overwrite v9 and cannot use the old test split for model selection.

- Replaced the plain modality projection with normalized residual bottleneck adapters, zero-initialized on the residual path.
- Added attention and pooling masks to current T/A/V fusion.
- Added training-only current modality dropout while guaranteeing retention of at least one originally observed modality.
- Added task-label-anchored current unimodal heads and learned evidence weights for M3ED and regression.
- Added global unimodal auxiliary objectives for M3ED/MOSEI and retained modality-specific supervision for CH-SIMS_v2.
- Changed regression training to a 0.75 Huber + 0.25 exact-L1 objective while preserving exact MAE reporting.
- Fixed history dropout so masks and feature tensors cannot disagree.
- Fixed robustness evaluation semantics: subset names now mean modalities kept, not modalities removed.
- Changed M3ED class weighting default to square-root inverse frequency because valid diagnostics showed rare-class recall collapse.

Known boundary: inputs remain precomputed packed features. Raw HuBERT/emotion2vec/CLIP/Swin towers are not implemented and must not be claimed. CH-SIMS_v2 uses the local 2143/647 split and is not directly comparable with official-split SOTA.
