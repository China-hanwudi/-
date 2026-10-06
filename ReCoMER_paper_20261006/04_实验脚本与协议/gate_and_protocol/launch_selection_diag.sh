#!/usr/bin/env bash
set -euo pipefail
PY=/home/emo/anaconda3/envs/torch2.0.1_cuda11.8/bin/python
CODE=/data/emo/肖田泽科研/模型/model6_innovation_validation_20260930/code
ROOT=/data/emo/肖田泽科研/模型/model6_innovation_validation_20260930/experiments/revised_crossdataset_20261001
MOSEI=/data/emo/肖田泽科研/数据/MOSEI/packed
MELD=/data/emo/肖田泽科研/数据/MELD/packed
OUT=$ROOT/selection_diagnostics
mkdir -p "$OUT"
run_mosei() {
  local gpu="$1" seed="$2"
  PYTHONPATH="$CODE" CUDA_VISIBLE_DEVICES="$gpu" "$PY" /tmp/selection_diagnostic.py \
    "$ROOT/mosei/seed${seed}/best.pt" "$MOSEI" >"$OUT/mosei_seed${seed}.json" 2>"$OUT/mosei_seed${seed}.log"
}
run_meld() {
  local gpu="$1" seed="$2"
  PYTHONPATH="$CODE" CUDA_VISIBLE_DEVICES="$gpu" "$PY" /tmp/selection_diagnostic.py \
    "$ROOT/meld/seed${seed}/best.pt" "$MELD" >"$OUT/meld_seed${seed}.json" 2>"$OUT/meld_seed${seed}.log"
}
worker0() { run_mosei 0 7; run_mosei 0 13; run_mosei 0 17; }
worker1() { run_mosei 1 23; run_mosei 1 29; run_mosei 1 37; }
worker2() { run_mosei 2 43; run_mosei 2 53; run_mosei 2 71; }
worker3() { run_mosei 3 101; run_meld 3 7; }
worker4() { run_meld 4 13; run_meld 4 17; run_meld 4 23; }
worker5() { run_meld 5 29; run_meld 5 37; run_meld 5 43; }
worker6() { run_meld 6 53; run_meld 6 71; run_meld 6 101; }
worker0 & worker1 & worker2 & worker3 & worker4 & worker5 & worker6 &
wait
