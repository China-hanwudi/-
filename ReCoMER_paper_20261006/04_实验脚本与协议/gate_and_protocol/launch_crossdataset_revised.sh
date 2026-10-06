#!/usr/bin/env bash
set -euo pipefail

PY=/home/emo/anaconda3/envs/torch2.0.1_cuda11.8/bin/python
CODE=/data/emo/肖田泽科研/模型/model6_innovation_validation_20260930/code
ROOT=/data/emo/肖田泽科研/模型/model6_innovation_validation_20260930/experiments/revised_crossdataset_20261001
MOSEI=/data/emo/肖田泽科研/数据/MOSEI/packed
MELD=/data/emo/肖田泽科研/数据/MELD/packed

run_mosei() {
  local gpu="$1" seed="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m n6.train \
    --data "$MOSEI" --out "$ROOT/mosei/seed${seed}" --seed "$seed" \
    --task reg --utility shapley --contrib-correct --bounded-lambda 0.3 \
    --tag revised_cross --device cuda \
    >"$ROOT/logs/mosei_seed${seed}.log" 2>&1
}

run_meld() {
  local gpu="$1" seed="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m n6.train \
    --data "$MELD" --out "$ROOT/meld/seed${seed}" --seed "$seed" \
    --task cls --utility shapley --contrib-correct --bounded-lambda 0.3 \
    --tag revised_cross --device cuda \
    >"$ROOT/logs/meld_seed${seed}.log" 2>&1
}

mkdir -p "$ROOT/logs" "$ROOT/mosei" "$ROOT/meld"
cd "$CODE"

worker0() { run_mosei 0 7; run_mosei 0 13; run_mosei 0 17; }
worker1() { run_mosei 1 23; run_mosei 1 29; run_mosei 1 37; }
worker2() { run_mosei 2 43; run_mosei 2 53; run_mosei 2 71; }
worker3() { run_mosei 3 101; run_meld 3 7; }
worker4() { run_meld 4 13; run_meld 4 17; run_meld 4 23; }
worker5() { run_meld 5 29; run_meld 5 37; run_meld 5 43; }
worker6() { run_meld 6 53; run_meld 6 71; run_meld 6 101; }

worker0 & worker1 & worker2 & worker3 & worker4 & worker5 & worker6 &
wait
