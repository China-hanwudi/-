#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/肖田泽最强"
MODEL="$ROOT/模型"
TRAIN_OUT="$ROOT/meld结果/train结果"
VAL_OUT="$ROOT/meld结果/val结果"
TEST_OUT="$ROOT/meld结果/test结果"
QWEN="/data/shared/qwen/Qwen3-Omni-30B-A3B-Instruct"
export PYTHONPATH="$MODEL:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
mkdir -p "$TRAIN_OUT/logs" "$VAL_OUT" "$TEST_OUT"
LOG="$TRAIN_OUT/logs/pipeline.log"

{
echo "=== RESUME PIPELINE $(date -Iseconds) ==="
echo "=== FEATURE EXTRACTION $(date -Iseconds) ==="
python "$MODEL/n3_affect/extract_meld_features.py" \
  --manifests "$TRAIN_OUT/manifests/train.csv" "$TRAIN_OUT/manifests/val.csv" "$TRAIN_OUT/manifests/test.csv" \
  --out-dirs "$TRAIN_OUT/features/train" "$TRAIN_OUT/features/val" "$TRAIN_OUT/features/test" \
  --qwen-path "$QWEN" \
  --text-batch-size 8 \
  --av-workers 8

echo "=== TRAIN $(date -Iseconds) ==="
python "$MODEL/n3_affect/train_meld.py" \
  --config "$MODEL/configs/n3_train_v1.json" \
  --train-manifest "$TRAIN_OUT/manifests/train.csv" \
  --val-manifest "$TRAIN_OUT/manifests/val.csv" \
  --train-features "$TRAIN_OUT/features/train" \
  --val-features "$TRAIN_OUT/features/val" \
  --out-dir "$TRAIN_OUT" \
  --epochs 20 --batch-size 64 --patience 5 --lr 3e-4 --seed 17

CKPT="$TRAIN_OUT/checkpoints/best.pt"

echo "=== EVAL TRAIN $(date -Iseconds) ==="
python "$MODEL/n3_affect/eval_meld.py" \
  --checkpoint "$CKPT" \
  --manifest "$TRAIN_OUT/manifests/train.csv" \
  --features "$TRAIN_OUT/features/train" \
  --split train --out-dir "$TRAIN_OUT"

echo "=== EVAL VAL $(date -Iseconds) ==="
python "$MODEL/n3_affect/eval_meld.py" \
  --checkpoint "$CKPT" \
  --manifest "$TRAIN_OUT/manifests/val.csv" \
  --features "$TRAIN_OUT/features/val" \
  --split val --out-dir "$VAL_OUT"

echo "=== EVAL TEST $(date -Iseconds) ==="
python "$MODEL/n3_affect/eval_meld.py" \
  --checkpoint "$CKPT" \
  --manifest "$TRAIN_OUT/manifests/test.csv" \
  --features "$TRAIN_OUT/features/test" \
  --split test --out-dir "$TEST_OUT"

echo "=== PIPELINE DONE $(date -Iseconds) ==="
} 2>&1 | tee -a "$LOG"
