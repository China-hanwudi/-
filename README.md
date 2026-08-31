# Temporal N3: Risk-Aware Multimodal History Routing

This repository documents the current development direction: **Temporal N3 v4**, a candidate-wise multimodal history-routing architecture for dialogue emotion recognition.

The research question is narrower than “does more context help?”: **which strictly-past dialogue turns should be used for the current emotion prediction, through which modalities, and when should the system return to a current-only prediction instead?**

> **Evidence status, 2026-08-31.** Temporal N3 v4 has an independently tested architecture and a frozen development protocol. Its complete causal/OOF utility-supervision, selector, resampling, and official-test chain has not yet been verified end to end. The MELD official-split run is train/dev-only preparation; its test split remains unopened. This repository makes no SOTA or completed-test claim for v4.

## Current architecture

```text
current utterance + strictly-past candidates
        |
        +-- frozen, auditable temporal candidate manifest
        +-- candidate-wise K x 3 x 3 current-history relations (T/A/V)
        +-- bidirectional utility/risk heads
        +-- modality gate -> Utility-Risk Bottleneck -> candidate gate
        +-- history route, or a hard current-only route when unauthorized
        v
emotion prediction
```

Temporal N3 keeps the history-candidate axis until routing is decided. For each candidate it computes nine current-modality to history-modality relations, then uses add-benefit and deletion-risk evidence to control modality and candidate gates. A failed frozen safety condition must emit the current-only prediction rather than a weakened history mixture.

The development protocol specifies auditable temporal candidate sampling, candidate refill from previously unvisited history, fresh parameters and optimizer for a permitted resampling round, and a maximum of two fallback resampling attempts. See [the v4 protocol](docs/20_temporal_n3_v4.md).

## Repository map

| Path | Purpose |
|---|---|
| [`temporal_n3/`](temporal_n3/) | Current v4 implementation: batched `K x 3 x 3`, utility-risk bottleneck, candidate gates, and temporal candidate policy. |
| [`docs/20_temporal_n3_v4.md`](docs/20_temporal_n3_v4.md) | Research objective, protocol, verification status, and non-leakage rules. |
| [`模型/`](模型/) | Legacy ComposerN3 implementation retained for reproducibility of earlier work. |
| [`experiment/`](experiment/) | Historical experiment contracts, analysis utilities, and synthetic contract tests. |
| [`datasets/`](datasets/) | Official-source download and checksum guidance only; no controlled corpus is redistributed. |
| [`DATA_BOUNDARY.md`](DATA_BOUNDARY.md) | Public-release and data-protection boundary. |

## Research claims and boundaries

The intended contribution is **risk-aware candidate-wise history routing**, not a claim that historical context is universally beneficial. The architecture is designed to test separately whether a candidate and its modalities have measurable benefit or risk relative to a matched current-only route, and whether a selector can use that evidence without increasing historical harm or degrading frozen classification criteria.

These are research hypotheses until the complete train-only / development-OOF procedure, frozen selector, and one-time official evaluation have been run. Historical aggregated evidence remains available in older documentation; it must not be re-labeled as a v4 result.

## Reproducibility and evaluation discipline

- Training parameters use training data only. Development data is used only for validation, early stopping, and predeclared selection.
- Candidate utility/risk supervision must be generated from development-time OOF predictions, never test labels or test-derived features.
- The final test is evaluated once only after code, manifests, partitions, selector, thresholds, and evidence receipts are frozen.
- Cross-paper SOTA comparison additionally requires matched official split, modality inputs, metric definition, and pretrained representation.
- A failed history safety gate is a valid result: the prescribed output is `current-only`, not threshold tuning or a retrospective retry.

## Quick architecture checks

The v4 module uses PyTorch and the existing feature configuration interfaces. From the repository root, with dependencies installed and `模型/` on the Python path:

```powershell
$env:PYTHONPATH = (Resolve-Path '模型').Path
python -m pytest temporal_n3/tests -q
```

The tests are synthetic interface checks. They verify variable candidate counts, masked `K x 3 x 3` relations, the hard fallback route, and bounded resampling authorization. They do not establish dataset performance.

## Data and release boundary

This repository excludes raw MELD, IEMOCAP, and EmotionTalk data, per-sample labels/manifests, extracted features, Qwen weights, checkpoints, server logs, credentials, and private absolute paths. Obtain data through [`datasets/README.md`](datasets/README.md), then keep run artifacts outside Git as specified in [`DATA_BOUNDARY.md`](DATA_BOUNDARY.md).

## Legacy material

The existing ComposerN3 package and earlier contracts remain available for auditability. They use a different fixed-history architecture and must not be presented as Temporal N3 v4 results. The migration relationship is recorded in [the v4 documentation](docs/20_temporal_n3_v4.md#relationship-to-legacy-material).
