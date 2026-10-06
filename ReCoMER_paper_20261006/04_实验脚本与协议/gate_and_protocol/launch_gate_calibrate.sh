#!/usr/bin/env bash
set -uo pipefail
PY=/home/emo/anaconda3/envs/torch2.0.1_cuda11.8/bin/python
EXP=/data/emo/肖田泽科研/模型/model6_innovation_validation_20260930/experiments
ROOT=$EXP/gate_router_20261002
mkdir -p "$ROOT/logs_task"
cd "$ROOT/code"
failed=0
for dataset in meld mosei m3ed; do
  if [ "$dataset" = meld ]; then
    DATA=/data/emo/肖田泽科研/数据/MELD/packed
    BASE=$EXP/meld_mechanism_20260930/runs/uniform_h
  elif [ "$dataset" = mosei ]; then
    DATA=/data/emo/肖田泽科研/数据/MOSEI/packed
    BASE=$EXP/mosei_loop_20260930/runs/mosei_h
  else
    DATA=/data/emo/肖田泽科研/数据/M3ED/packed_audio_e2_zh_hubert_large
    BASE=$EXP/runs/uniform_h
  fi
  for seed in 17 29 43; do
    if ! CUDA_VISIBLE_DEVICES=0 "$PY" -m n6.train_gate \
      --base-ckpt "$BASE/seed${seed}/best.pt" --data "$DATA" \
      --out "$ROOT/runs/$dataset/seed${seed}" --seed "$seed" \
      --variants constant_task,mlp_task,evidence_task \
      --epochs 20 --device cuda >"$ROOT/logs_task/${dataset}_seed${seed}.log" 2>&1; then
      failed=$((failed+1))
      printf 'FAILED %s seed%s\n' "$dataset" "$seed"
    else
      printf 'COMPLETE %s seed%s\n' "$dataset" "$seed"
    fi
  done
done
printf 'QUEUE_FINISHED failures=%s\n' "$failed"
exit "$failed"
