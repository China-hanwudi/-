#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/home/emo/anaconda3/envs/torch2.0.1_cuda11.8/bin/python
EXP=/data/emo/肖田泽科研/模型/model6_innovation_validation_20260930/experiments
SCREEN=$EXP/gate_router_20261002
ROOT=$EXP/gate_router_confirm_crossdataset_20261002
mkdir -p "$ROOT/logs"
cd "$SCREEN/code"
for dataset in meld m3ed; do
  if [ "$dataset" = meld ]; then
    DATA=/data/emo/肖田泽科研/数据/MELD/packed
    BASE=$EXP/meld_mechanism_20260930/runs/uniform_h
  else
    DATA=/data/emo/肖田泽科研/数据/M3ED/packed_audio_e2_zh_hubert_large
    BASE=$EXP/runs/uniform_h
  fi
  for seed in 17 29 43; do
    mkdir -p "$ROOT/runs/$dataset/seed${seed}"
    for variant in constant_task mlp_task evidence_task; do
      dest="$ROOT/runs/$dataset/seed${seed}/$variant"
      if [ ! -e "$dest" ]; then
        cp -a "$SCREEN/runs/$dataset/seed${seed}/$variant" "$dest"
      elif [ ! -f "$dest/FINAL_RESULT.json" ]; then
        printf 'INCOMPLETE_EXISTING_COPY %s\n' "$dest"
        exit 1
      fi
    done
  done
  for seed in 7 13 23 37 53 71 101; do
    remaining=""
    for variant in constant_task mlp_task evidence_task; do
      if [ ! -f "$ROOT/runs/$dataset/seed${seed}/$variant/FINAL_RESULT.json" ]; then
        remaining="${remaining:+$remaining,}$variant"
      fi
    done
    if [ -z "$remaining" ]; then continue; fi
    CUDA_VISIBLE_DEVICES=0 "$PY" -m n6.train_gate \
      --base-ckpt "$BASE/seed${seed}/best.pt" --data "$DATA" \
      --out "$ROOT/runs/$dataset/seed${seed}" --seed "$seed" \
      --variants "$remaining" --epochs 20 --device cuda \
      >"$ROOT/logs/${dataset}_seed${seed}.log" 2>&1
    printf 'COMPLETE %s seed%s\n' "$dataset" "$seed"
  done
done
printf 'CROSSDATASET_CONFIRM_FINISHED\n'
