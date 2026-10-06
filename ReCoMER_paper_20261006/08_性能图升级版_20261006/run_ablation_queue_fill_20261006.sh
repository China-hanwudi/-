#!/usr/bin/env bash
# 2026-10-06 补种子队列：为四数据集 × 5 臂 × 缺失 seed(7,13,17,23) 补齐运行。
# 与 run_ablation_queue.sh 完全同配置（common/arms 逐字复制），仅改 SEEDS 与完成标记。
# run_one 自带 FINAL_RESULT.json 跳过逻辑，可安全重跑。
set -u
ROOT=/root/autodl-tmp/paper_experiments_20261004
CODE=/root/autodl-tmp/model6_final_system_20261003/code
PY=/root/miniconda3/bin/python
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
DATASETS=(
  "M3ED_strong|/root/autodl-tmp/data/M3ED_textQwen/packed|cls"
  "MELD|/root/autodl-tmp/data/MELD/packed|cls"
  "IEMOCAP|/root/autodl-tmp/data/IEMOCAP/packed|cls"
  "MOSEI_full|/root/autodl-tmp/data/MOSEI_full/packed|reg"
)
ARMS=(uniform_nohist uniform_history evidence_closed evidence_solo evidence_mpath)
SEEDS=(7 13 17 23)
run_one() {
  local dataset="$1"; local data="$2"; local task="$3"; local arm="$4"; local seed="$5"
  local out="$ROOT/runs/${dataset}__${arm}__s${seed}"
  if [[ -f "$out/FINAL_RESULT.json" ]]; then echo "SKIP $out"; return 0; fi
  mkdir -p "$out"
  echo "START dataset=$dataset arm=$arm seed=$seed $(date -Is)"
  local common=(--data "$data" --out "$out" --task "$task" --seed "$seed" --epochs 20 --batch-size 256 --swa-window 3 --patience 6 --device cuda --tag paper_20261004)
  local rc=0
  case "$arm" in
    uniform_nohist) "$PY" -m n6.train "${common[@]}" --utility uniform --deploy closed_loop --no-history || rc=$? ;;
    uniform_history) "$PY" -m n6.train "${common[@]}" --utility uniform --deploy closed_loop --history-abstain-variant utility_softmax || rc=$? ;;
    evidence_closed) "$PY" -m n6.train "${common[@]}" --utility shapley --deploy closed_loop --gate-architecture evidence --gate-detach-inputs --detach-utility-path --bounded-w --history-abstain-variant utility_softmax || rc=$? ;;
    evidence_solo) "$PY" -m n6.train "${common[@]}" --utility shapley --deploy solo_weighted --gate-architecture evidence --gate-detach-inputs --detach-utility-path --bounded-w --history-abstain-variant utility_softmax || rc=$? ;;
    evidence_mpath) "$PY" -m n6.train "${common[@]}" --utility shapley --deploy closed_loop --gate-architecture evidence --gate-detach-inputs --detach-utility-path --bounded-w --mpath --history-abstain-variant utility_softmax || rc=$? ;;
    *) echo "UNKNOWN_ARM $arm"; rc=2 ;;
  esac
  if [[ $rc -ne 0 ]]; then echo "FAIL dataset=$dataset arm=$arm seed=$seed rc=$rc $(date -Is)"; return $rc; fi
  echo "DONE dataset=$dataset arm=$arm seed=$seed $(date -Is)"
  return 0
}
cd "$CODE" || exit 2
failures=0
for spec in "${DATASETS[@]}"; do
  IFS='|' read -r name data task <<< "$spec"
  for arm in "${ARMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_one "$name" "$data" "$task" "$arm" "$seed" 2>&1 | tee -a "$ROOT/logs/${name}__${arm}.log"
      rc=${PIPESTATUS[0]}
      if [[ $rc -ne 0 ]]; then failures=$((failures+1)); fi
    done
  done
done
echo "FILL_QUEUE_20261006_COMPLETE $(date -Is) failures=$failures"
