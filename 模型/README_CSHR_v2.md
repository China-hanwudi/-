# MATCH-CSHR v2

`MATCH-CSHR v2` is the replacement for the discarded lightweight history model.
It treats causal history routing as a controlled modification of a conversational
multimodal backbone, rather than a late feature-concatenation branch.

## Backbone-preserving integration

1. **Entity-calibrated current utterance encoder.** T/A/V embeddings interact
   through cross-modal attention and are conditioned on modality availability.
2. **Conversational hypergraph fusion.** A selected historical utterance becomes
   a candidate hyperedge; its full 3 x 3 current-to-history modality evidence
   controls the hyperedge message instead of being pooled after classification.
3. **Line-graph emotion pathway.** Candidate hyperedges communicate along a
   directed, strict-past temporal path. Recency and speaker continuity are edge
   features, so the pathway is not an unordered memory set.
4. **Causal Selective History Routing.** Candidates come only from strict past
   turns, stratified into near/middle/far pools. Pair gates precede a candidate
   gate, and a learned null candidate is always available.
5. **Benefit-bounded residual and rollback.** A zero-initialized residual
   adapter updates the current-only logits. A separately calibratable router
   mixes the residual with the anchor; a closed router returns the exact anchor
   prediction.

`train.py` is deliberately development-only: it accepts only train and
validation manifests. `make_oof_utility_targets.py` creates dialogue-disjoint
OOF targets required before final router calibration. Test labels must not be
used for selection or calibration.
