#!/usr/bin/env bash
set -euo pipefail

# v10 formal-training launcher. One invocation owns one dataset/seed output.
# It never reads sealed test and never stops or shares a non-idle GPU process.
GPU="${GPU:-6}"
DATASET="${DATASET:-m3ed}"
SEED="${SEED:-17}"
ROOT=$(cd "$(dirname "$0")" && pwd)
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the private packed-data root before training}"
PYTHON_BIN="${PYTHON_BIN:-/home/emo/txcao/anaconda3/envs/zyn/bin/python}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$ROOT/runs/v10_${DATASET}/seed_$(printf '%03d' "$SEED")_${RUN_ID}}"

if ! nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader,nounits; then
  echo "nvidia-smi unavailable; refusing to train" >&2; exit 2
fi
GPU_ROW=$(nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F', ' -v g="$GPU" '$1==g {print}')
if [[ -z "$GPU_ROW" ]]; then echo "GPU $GPU not found" >&2; exit 2; fi
GPU_NAME=$(awk -F', ' '{print $2}' <<<"$GPU_ROW")
FREE_MIB=$(awk -F', ' '{print $3}' <<<"$GPU_ROW")
UTIL=$(awk -F', ' '{print $4}' <<<"$GPU_ROW" | tr -d '%')
GPU_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1==g {print $2}')
GPU_PROCS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | awk -F', ' -v u="$GPU_UUID" '$1==u {print}')
if [[ "$GPU_NAME" != NVIDIA\ A100* || "$FREE_MIB" -lt 12288 || "$UTIL" -ge 10 || -n "$GPU_PROCS" ]]; then
  echo "GPU $GPU is not an explicitly idle A100; refusing to start." >&2
  echo "row=$GPU_ROW processes=${GPU_PROCS:-none}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"
"$PYTHON_BIN" -m compileall -q n3_affect
if [[ "${SMOKE:-1}" == "1" ]]; then
  PREFLIGHT="$ROOT/runs/v10_preflight_${RUN_ID}_gpu${GPU}"
  "$PYTHON_BIN" -m n3_affect.train_m3ed --data "$DATA_ROOT/M3ED/packed" \
    --out "$PREFLIGHT" --seed 17 --smoke 1 --device cuda
  test -f "$PREFLIGHT/FINAL_RESULT.json"
fi

if [[ -e "$OUT" ]]; then
  echo "Output already exists; refusing to overwrite: $OUT" >&2
  exit 2
fi

case "$DATASET" in
  m3ed) "$PYTHON_BIN" -m n3_affect.train_m3ed --data "$DATA_ROOT/M3ED/packed" --out "$OUT" --seed "$SEED" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --device cuda ;;
  mosei) "$PYTHON_BIN" -m n3_affect.train_mosei --data "$DATA_ROOT/MOSEI/packed" --out "$OUT" --seed "$SEED" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --device cuda ;;
  chsims) "$PYTHON_BIN" -m n3_affect.train_chsims --data "$DATA_ROOT/CH-SIMS_v2/v7_packed" --out "$OUT" --seed "$SEED" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --device cuda ;;
  *) echo "DATASET must be m3ed, mosei, or chsims" >&2; exit 2 ;;
esac
test -f "$OUT/best.pt" -a -f "$OUT/FINAL_RESULT.json"
