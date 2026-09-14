# v10 A100 Training Guide

Use only the deployed `TemporalN3_TopJournal_v10_20260914` directory. Keep v9 untouched. Set `DATA_ROOT` to the server's private data root. The active datasets are M3ED, CMU-MOSEI, and CH-SIMS_v2; do not add MELD, IEMOCAP, or EmotionTalk.

Before every run, execute both GPU checks below. Select only an A100 with at least 12 GiB free, utilization below 10%, and no compute process on that GPU. Never stop another user's process. If GPU 6 is busy, choose another clearly idle A100 and record it.

```bash
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
```

Run the launcher from the v10 directory. Each seed must have a new `OUT` directory. The launcher compiles the package, runs a M3ED CUDA smoke gate, records `CUDA_VISIBLE_DEVICES`, and refuses a non-idle GPU.

```bash
GPU=2 DATASET=m3ed SEED=17 OUT=runs/v10_m3ed/seed_017_formal_20260914 SMOKE=1 bash run_a100_training_v10.sh
```

Formal order:

```text
M3ED: seeds 17, 29, 43, 71, 101; train/valid only; select valid Weighted-F1, then Macro-F1.
CMU-MOSEI: at least seed 17; train/valid only; select valid MAE.
CH-SIMS_v2: at least one seed; train/valid only; select valid MAE; mark its 2143/647 split as non-official.
```

Do not use v9 test metrics to tune v10. Do not claim a test SOTA result from a valid score. A successful process must produce `best.pt`, `FINAL_RESULT.json`, `history.json`, and `RUN_METADATA.json`; a zero exit without these files is a failed run. Preserve logs and record host, physical GPU, seed, config hash, data hashes, and code fingerprint.

This model consumes precomputed packed T/A/V features. It does not claim HuBERT, emotion2vec, CLIP, or Swin integration. The v10 changes are mask-aware current fusion, residual modality projectors, current unimodal label anchoring, measured counterfactual supervision, bounded history residuals, and regression Huber/L1 calibration.
