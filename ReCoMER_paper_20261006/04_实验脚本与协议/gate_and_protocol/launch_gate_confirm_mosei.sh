#!/usr/bin/env bash
set -euo pipefail
PY=/home/emo/anaconda3/envs/torch2.0.1_cuda11.8/bin/python
EXP=/data/emo/肖田泽科研/模型/model6_innovation_validation_20260930/experiments
SCREEN=$EXP/gate_router_20261002
ROOT=$EXP/gate_router_confirm_mosei_20261002
DATA=/data/emo/肖田泽科研/数据/MOSEI/packed
BASE=$EXP/mosei_loop_20260930/runs/mosei_h
mkdir -p "$ROOT/runs/mosei" "$ROOT/logs"
for seed in 17 29 43; do
  mkdir -p "$ROOT/runs/mosei/seed${seed}"
  for variant in constant_task mlp_task evidence_task; do
    if [ -e "$ROOT/runs/mosei/seed${seed}/$variant" ]; then
      printf 'REFUSING_COPY_OVERWRITE seed%s %s\n' "$seed" "$variant"
      exit 1
    fi
    cp -a "$SCREEN/runs/mosei/seed${seed}/$variant" "$ROOT/runs/mosei/seed${seed}/$variant"
  done
done
cd "$SCREEN/code"
for seed in 7 13 23 37 53 71 101; do
  CUDA_VISIBLE_DEVICES=0 "$PY" -m n6.train_gate \
    --base-ckpt "$BASE/seed${seed}/best.pt" --data "$DATA" \
    --out "$ROOT/runs/mosei/seed${seed}" --seed "$seed" \
    --variants constant_task,mlp_task,evidence_task --epochs 20 --device cuda \
    >"$ROOT/logs/seed${seed}.log" 2>&1
  printf 'COMPLETE mosei seed%s\n' "$seed"
done
printf 'CONFIRM_FINISHED\n'
