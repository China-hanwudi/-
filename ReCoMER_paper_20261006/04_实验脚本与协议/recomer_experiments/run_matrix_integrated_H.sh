#!/usr/bin/env bash
set -u

ROOT="/data/emo/肖田泽科研/模型/model6_integrated_H_20261003"
CODE="$ROOT/code"
PY="/home/emo/anaconda3/envs/torch2.0.1_cuda11.8/bin/python"
RUNS="$ROOT/runs/matrix"
mkdir -p "$RUNS"

run_one() {
  local dataset="$1"; local data="$2"; local task="$3"; local arm="$4"; local seed="$5"; shift 5
  local out="$RUNS/${dataset}_${arm}_s${seed}"
  mkdir -p "$out"
  if [[ -f "$out/FINAL_RESULT.json" ]]; then
    echo "SKIP $dataset $arm seed=$seed (already complete)"
    return 0
  fi
  echo "START $(date -Is) $dataset $arm seed=$seed"
  (cd "$CODE" && CUDA_VISIBLE_DEVICES=0 "$PY" -m n6.train \
    --data "$data" --out "$out" --seed "$seed" --task "$task" \
    --device cuda --tag "${dataset}_${arm}_s${seed}" "$@") \
    >"$out/stdout.log" 2>&1
  local rc=$?
  echo "END $(date -Is) rc=$rc $dataset $arm seed=$seed"
  return $rc
}

M3ED="/data/emo/肖田泽科研/数据/M3ED/packed_audio_e2_zh_hubert_large"
MELD="/data/emo/肖田泽科研/数据/MELD/packed"
MOSEI="/data/emo/肖田泽科研/数据/MOSEI/packed"
SEEDS=(7)

for seed in "${SEEDS[@]}"; do
  run_one M3ED "$M3ED" cls nohist "$seed" \
    --utility uniform --deploy closed_loop --no-history || exit $?
  run_one M3ED "$M3ED" cls uniform_h "$seed" \
    --utility uniform --deploy closed_loop --history-abstain-variant none || exit $?
  run_one M3ED "$M3ED" cls history_softmax "$seed" \
    --utility uniform --deploy closed_loop --history-abstain-variant utility_softmax || exit $?
  run_one M3ED "$M3ED" cls integrated "$seed" \
    --utility shapley --gate-architecture evidence --gate-detach-inputs \
    --detach-utility-path --bounded-w --bounded-lambda 0.3 \
    --history-abstain-variant utility_softmax --deploy closed_loop || exit $?
  run_one M3ED "$M3ED" cls integrated_solo "$seed" \
    --utility shapley --gate-architecture evidence --gate-detach-inputs \
    --detach-utility-path --bounded-w --bounded-lambda 0.3 \
    --history-abstain-variant utility_softmax --deploy solo_weighted || exit $?

  run_one MELD "$MELD" cls nohist "$seed" \
    --utility uniform --deploy closed_loop --no-history || exit $?
  run_one MELD "$MELD" cls uniform_h "$seed" \
    --utility uniform --deploy closed_loop --history-abstain-variant none || exit $?
  run_one MELD "$MELD" cls history_softmax "$seed" \
    --utility uniform --deploy closed_loop --history-abstain-variant utility_softmax || exit $?
  run_one MELD "$MELD" cls integrated "$seed" \
    --utility shapley --gate-architecture evidence --gate-detach-inputs \
    --detach-utility-path --bounded-w --bounded-lambda 0.3 \
    --history-abstain-variant utility_softmax --deploy closed_loop || exit $?
  run_one MELD "$MELD" cls integrated_solo "$seed" \
    --utility shapley --gate-architecture evidence --gate-detach-inputs \
    --detach-utility-path --bounded-w --bounded-lambda 0.3 \
    --history-abstain-variant utility_softmax --deploy solo_weighted || exit $?

  run_one MOSEI "$MOSEI" reg nohist "$seed" \
    --utility uniform --deploy closed_loop --no-history || exit $?
  run_one MOSEI "$MOSEI" reg uniform_h "$seed" \
    --utility uniform --deploy closed_loop --history-abstain-variant none || exit $?
  run_one MOSEI "$MOSEI" reg history_softmax "$seed" \
    --utility uniform --deploy closed_loop --history-abstain-variant utility_softmax || exit $?
  run_one MOSEI "$MOSEI" reg integrated "$seed" \
    --utility shapley --gate-architecture evidence --gate-detach-inputs \
    --detach-utility-path --bounded-w --bounded-lambda 0.3 \
    --history-abstain-variant utility_softmax --deploy closed_loop || exit $?
  run_one MOSEI "$MOSEI" reg integrated_solo "$seed" \
    --utility shapley --gate-architecture evidence --gate-detach-inputs \
    --detach-utility-path --bounded-w --bounded-lambda 0.3 \
    --history-abstain-variant utility_softmax --deploy solo_weighted || exit $?
done

echo "MATRIX_DONE $(date -Is)"
