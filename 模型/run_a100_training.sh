#!/usr/bin/env bash
set -euo pipefail

# Explicitly require a genuinely idle A100 before formal training.  This
# script never stops or shares another process and never reads sealed test.
GPU="${GPU:-6}"
FREE_MIB=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -v g="$GPU" '$1==g {print $2}')
UTIL=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | awk -v g="$GPU" '$1==g {print $2}')
USED=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | wc -l)
if [[ -z "$FREE_MIB" || -z "$UTIL" || "$FREE_MIB" -lt 12288 || "$UTIL" -ge 10 || "$USED" -gt 0 ]]; then
  echo "GPU $GPU is not an explicitly idle A100; refusing to start." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU"
ROOT=$(cd "$(dirname "$0")" && pwd)
DATA_ROOT="${DATA_ROOT:-/data/emo/肖田泽科研/数据}"
SEED="${SEED:-17}"
DATASET="${DATASET:-m3ed}"
OUT="${OUT:-$ROOT/runs/$DATASET/seed_$(printf '%03d' "$SEED")}"
if [[ "$DATASET" == "m3ed" ]]; then
  exec python -m n3_affect.train_m3ed --data "$DATA_ROOT/M3ED/packed" --out "$OUT" --seed "$SEED" --epochs "${EPOCHS:-20}" --batch-size "${BATCH_SIZE:-64}"
elif [[ "$DATASET" == "mosei" ]]; then
  exec python -m n3_affect.train_mosei --data "$DATA_ROOT/MOSEI/packed" --out "$OUT" --seed "$SEED" --epochs "${EPOCHS:-20}" --batch-size "${BATCH_SIZE:-64}"
else
  echo "DATASET must be m3ed or mosei" >&2
  exit 2
fi
