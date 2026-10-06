#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/emo/肖田泽科研/模型/model6_final_system_20261003
CODE=$ROOT/code
PY=/home/emo/anaconda3/envs/torch2.0.1_cuda11.8/bin/python
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

while [[ ! -f "$ROOT/logs/queue.out" ]] || ! grep -q 'TRAIN_QUEUE_COMPLETE' "$ROOT/logs/queue.out"; do
  sleep 30
done

cd "$CODE"
for ckpt in "$ROOT"/runs/*/best.pt; do
  [[ -f "$ckpt" ]] || continue
  run_dir=$(dirname "$ckpt")
  [[ -f "$run_dir/TEST_RESULT.json" ]] && continue
  name=$(basename "$run_dir")
  case "$name" in
    M3ED_strong__*) data=/data/emo/肖田泽科研/数据/M3ED_textQwen/packed ;;
    MELD__*) data=/data/emo/肖田泽科研/数据/MELD/packed ;;
    IEMOCAP__*) data=/data/emo/肖田泽科研/数据/IEMOCAP/packed ;;
    MOSEI__*) data=/data/emo/肖田泽科研/数据/MOSEI/packed ;;
    *) echo "UNKNOWN_DATASET $name" >&2; continue ;;
  esac
  echo "FINAL_TEST $name"
  "$PY" -m n6.evaluate --data "$data" --ckpt "$ckpt" --split test \
    --device cuda --batch-size 256 --out "$run_dir/TEST_RESULT.json"
done
echo "TEST_QUEUE_COMPLETE $(date -Is)"
