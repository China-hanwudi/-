#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/emo/肖田泽科研/模型/model6_final_system_20261003
CODE=$ROOT/code
PY=/home/emo/anaconda3/envs/torch2.0.1_cuda11.8/bin/python
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
mkdir -p "$ROOT/runs" "$ROOT/logs"

run_one() {
  local dataset="$1"; local data="$2"; local task="$3"; local arm="$4"; local seed="$5"
  local out="$ROOT/runs/${dataset}__${arm}__s${seed}"
  if [[ -f "$out/FINAL_RESULT.json" ]]; then
    echo "SKIP $out"
    return
  fi
  mkdir -p "$out"
  echo "START dataset=$dataset arm=$arm seed=$seed"
  local common=(--data "$data" --out "$out" --task "$task" --seed "$seed"
    --epochs 20 --batch-size 128 --swa-window 3 --patience 6
    --device cuda --tag final_audit)
  case "$arm" in
    uniform_nohist)
      "$PY" -m n6.train "${common[@]}" --utility uniform --deploy closed_loop --no-history ;;
    uniform_history)
      "$PY" -m n6.train "${common[@]}" --utility uniform --deploy closed_loop \
        --history-abstain-variant utility_softmax ;;
    evidence_closed)
      "$PY" -m n6.train "${common[@]}" --utility shapley --deploy closed_loop \
        --gate-architecture evidence --gate-detach-inputs --detach-utility-path \
        --bounded-w --history-abstain-variant utility_softmax ;;
    evidence_solo)
      "$PY" -m n6.train "${common[@]}" --utility shapley --deploy solo_weighted \
        --gate-architecture evidence --gate-detach-inputs --detach-utility-path \
        --bounded-w --history-abstain-variant utility_softmax ;;
    evidence_mpath)
      "$PY" -m n6.train "${common[@]}" --utility shapley --deploy closed_loop \
        --gate-architecture evidence --gate-detach-inputs --detach-utility-path \
        --bounded-w --mpath --history-abstain-variant utility_softmax ;;
    *) echo "unknown arm $arm" >&2; return 2 ;;
  esac
}

DATASETS=(
  "M3ED_strong|/data/emo/肖田泽科研/数据/M3ED_textQwen/packed|cls"
  "MELD|/data/emo/肖田泽科研/数据/MELD/packed|cls"
  "IEMOCAP|/data/emo/肖田泽科研/数据/IEMOCAP/packed|cls"
  "MOSEI|/data/emo/肖田泽科研/数据/MOSEI/packed|reg"
)
ARMS=(uniform_nohist uniform_history evidence_closed evidence_solo evidence_mpath)
SEEDS=(7 13 17)

cd "$CODE"
for spec in "${DATASETS[@]}"; do
  IFS='|' read -r name data task <<< "$spec"
  for arm in "${ARMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_one "$name" "$data" "$task" "$arm" "$seed" \
        2>&1 | tee -a "$ROOT/logs/${name}__${arm}.log"
    done
  done
done
echo "TRAIN_QUEUE_COMPLETE $(date -Is)"
