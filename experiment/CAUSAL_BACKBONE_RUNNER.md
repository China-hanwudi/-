# Causal-backbone open-role runners

## EmotionTalk

This runner trains the sub-2M causal multimodal Transformer directly from
fold-local text SVD features, frozen WavLM mean+standard-deviation embeddings,
and frozen DINOv2 embeddings. It does not consume the earlier linear-model
utility target. It regenerates current-only/all-history and
`S / S+h / T / T-h` probabilities before computing model-relative utilities.

The production command requires four physical `allow_pickle=False` sidecars:
fit features, fit labels, model-selection features, and model-selection labels.
The aggregate manifest hashes all four files. Private outputs must be outside
this repository.

Generate that sidecar once at the trusted, hash-pinned boundary. The private
NPZ files contain only their declared open role; the public manifest contains
only counts, dimensions, and hashes. The trusted generator is the only code
allowed to open the complete source media/transcription/train-label containers.
The model runner never receives those source paths.

```powershell
& $researchPython experiment\scripts\prepare_emotiontalk_open_role_sidecar.py `
  --labels 'C:\restricted-data\EmotionTalk\mm_label.npz' `
  --features 'C:\restricted-data\emotiontalk_media_features_v1.npz' `
  --transcription 'C:\restricted-data\EmotionTalk\transcription.csv' `
  --config experiment\configs\emotiontalk_open_role_sidecar_v2.json `
  --private-output-dir 'C:\private-research-artifacts\emotiontalk_open_roles_v2' `
  --public-manifest results\emotiontalk_open_role_sidecar_v2_manifest.json
```

```powershell
$researchPython = 'C:\path\to\cuda-environment\python.exe'
$privateArtifacts = 'C:\private-research-artifacts\carma-causal-v1'

& $researchPython experiment\scripts\run_emotiontalk_causal_backbone.py `
  --sidecar-dir 'C:\private-research-artifacts\emotiontalk_open_roles_v2' `
  --sidecar-manifest results\emotiontalk_open_role_sidecar_v2_manifest.json `
  --private-output-dir $privateArtifacts `
  --public-output results\emotiontalk_causal_backbone_v1_model_selection.json `
  --device cuda
```

There is no single-pickle, full-feature, overwrite, calibration, holdout, dev,
or test fallback in the production runner. Existing public aggregate reports
are write-once and cause a fail-closed error.

The private directory receives atomic per-seed/per-fold processors and
checkpoints plus a row-level NPZ cache. The repository receives only an
aggregate JSON with seed/fold counts, model size, input and matrix hashes,
classification metrics, and aggregate bidirectional utilities.

## MELD loader contract

`hva_affect.meld_causal_backbone_loader.load_meld_open_role_corpus` accepts a
sidecar directory plus its frozen manifest. The manifest must resolve exactly
four private, non-pickled train-role files:

- fit features and fit labels;
- model-selection features and model-selection labels.

There is intentionally no dev/test argument. A filename or archive role that
claims calibration, holdout, dev, or test is rejected. The current MM-Align
sidecars use audio=64 and video=4096 dimensions; instantiate a separate
`CausalBackboneConfig(text_dim=256, audio_dim=64, video_dim=4096,
num_speakers=512)` (or the smallest frozen vocabulary above the observed
maximum speaker code). With the unchanged 128-wide architecture this remains
strictly below 2M parameters.
The loader returns both `OpenRoleCorpus` and an attested
`VerifiedCorpusProvenance`. `execute_crossfit_backbone` accepts that verifier
result and does not accept caller-supplied strictness flags or source hashes.

### MELD v2 sidecar generation

The directory `D:\HVA-Affect_data\MELD\carma_sidecars_v1` is a historical v1
artifact. It contains an older provenance schema and must remain unchanged. Do
not point the v2 generator at it and do not use it for the formal runner.

Generate v2 into a new private directory at the trusted-custodian boundary.
Only this preparation command may open the registered official train CSV and
MM-Align train pickle. It does not open dev/test or compute a model metric.

```powershell
$researchPython = 'C:\path\to\cuda-environment\python.exe'
$meldV2Sidecars = 'D:\HVA-Affect_data\MELD\carma_sidecars_v2'
$meldV2Manifest = 'results\meld_multimodal_role_sidecars_v2_manifest.json'

& $researchPython experiment\scripts\prepare_meld_multimodal_sidecars.py `
  --train-csv 'D:\HVA-Affect_data\MELD\labels\train_sent_emo.csv' `
  --train-pickle 'D:\HVA-Affect_data\MELD\mmalign_open_roles\train.pkl' `
  --config experiment\configs\meld_multimodal_role_sidecars_v2.json `
  --private-output-dir $meldV2Sidecars `
  --public-manifest $meldV2Manifest
```

The v2 manifest binds the official row order, label order, feature dimensions,
role filenames, file hashes, and public-content audit. The formal model-facing
loader remaps the population-independent speaker tokens using fit rows only;
unseen model-selection speakers receive OOV code 0.

### CPU-only structural preflight

Run this before any CUDA training. It verifies the manifest and hashes,
deserializes fit features/labels and model-selection features, and performs one
untrained four-context forward pass. It hashes but never deserializes
`labels_model_selection.npz`; it opens no calibration, holdout, dev, or test
array and computes no metric or utility.

```powershell
& $researchPython experiment\scripts\preflight_meld_causal_backbone.py `
  --sidecar-dir $meldV2Sidecars `
  --sidecar-manifest $meldV2Manifest `
  --backbone-config experiment\configs\carma_causal_backbone_meld_v1.json `
  --utility-config experiment\configs\bidirectional_emotion_utility_v1.json
```

### Formal open-role producer run

Only after the preflight succeeds may the fit/model-selection producer be
started. Its `current_only` output is an empty-history intervention on the same
trained model, not an independently trained baseline. The aggregate report is
therefore a probability/utility producer artifact and cannot by itself
authorize a performance claim.

```powershell
$meldPrivateRun = 'D:\HVA-Affect_data\MELD\carma_causal_backbone_v1'

& $researchPython experiment\scripts\run_meld_causal_backbone.py `
  --sidecar-dir $meldV2Sidecars `
  --sidecar-manifest $meldV2Manifest `
  --backbone-config experiment\configs\carma_causal_backbone_meld_v1.json `
  --utility-config experiment\configs\bidirectional_emotion_utility_v1.json `
  --confirmatory-config experiment\configs\carma_confirmatory_analysis_v1.json `
  --private-output-dir $meldPrivateRun `
  --public-output results\meld_causal_backbone_v1_model_selection.json `
  --device cuda
```

Both private and public outputs are write-once. The runner has no arbitrary
role-file, overwrite, dev, test, calibration, or holdout escape hatch. MELD
reports identify their split as `official_train_open_roles_only`; this does not
mean that official dev or test was accessed.
