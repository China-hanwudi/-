#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=/home/emo/anaconda3/envs/torch2.0.1_cuda11.8/bin/python
EXP=/data/emo/肖田泽科研/模型/model6_innovation_validation_20260930/experiments
ROOT=$EXP/gate_feature_screen_20261002
CODE=$EXP/gate_router_20261002/code
mkdir -p "$ROOT/logs"
cd "$CODE"
for dataset in "$@"; do
  case "$dataset" in
    mosei_full) DATA=/data/emo/肖田泽科研/数据/MOSEI_full/packed; TASK=reg ;;
    meld_roberta) DATA=/data/emo/肖田泽科研/数据/MELD_robertaFT/packed; TASK=cls ;;
    m3ed_roberta) DATA=/data/emo/肖田泽科研/数据/M3ED_textRobertaFT/packed; TASK=cls ;;
    mosi) DATA=/data/emo/肖田泽科研/数据/CMU-MOSI/packed; TASK=reg ;;
    chsims) DATA=/data/emo/肖田泽科研/数据/CH-SIMS_v2/full_packed; TASK=reg ;;
    *) printf 'UNKNOWN_DATASET %s\n' "$dataset"; exit 1 ;;
  esac
  for seed in 17 29 43; do
    BASE=$ROOT/runs/$dataset/seed${seed}/uniform
    if [ ! -f "$BASE/FINAL_RESULT.json" ]; then
      CUDA_VISIBLE_DEVICES=0 "$PY" -m n6.train \
        --data "$DATA" --out "$BASE" --task "$TASK" \
        --utility uniform --seed "$seed" --epochs 30 --device cuda \
        >"$ROOT/logs/${dataset}_seed${seed}_base.log" 2>&1
      printf 'BASE_COMPLETE %s seed%s\n' "$dataset" "$seed"
    fi
    remaining=""
    for variant in constant_task mlp_task evidence_task; do
      if [ ! -f "$ROOT/runs/$dataset/seed${seed}/$variant/FINAL_RESULT.json" ]; then
        remaining="${remaining:+$remaining,}$variant"
      fi
    done
    if [ -z "$remaining" ]; then continue; fi
    CUDA_VISIBLE_DEVICES=0 "$PY" -m n6.train_gate \
      --base-ckpt "$BASE/best.pt" --data "$DATA" \
      --out "$ROOT/runs/$dataset/seed${seed}" --seed "$seed" \
      --variants "$remaining" --epochs 20 --device cuda \
      >"$ROOT/logs/${dataset}_seed${seed}_gates.log" 2>&1
    printf 'GATES_COMPLETE %s seed%s\n' "$dataset" "$seed"
  done
done
printf 'FEATURE_SCREEN_WORKER_FINISHED\n'
