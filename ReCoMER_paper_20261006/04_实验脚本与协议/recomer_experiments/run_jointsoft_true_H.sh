#!/usr/bin/env bash
set -u
ROOT="/data/emo/肖田泽科研/模型/model6_integrated_H_20261003"
CODE="$ROOT/code"
PY="/home/emo/anaconda3/envs/torch2.0.1_cuda11.8/bin/python"
RUNS="$ROOT/runs/matrix"
run_one() {
  local ds="$1"; local data="$2"; local task="$3"
  local out="$RUNS/${ds}_jointsoft_true_s7"
  mkdir -p "$out"
  if [[ -f "$out/FINAL_RESULT.json" ]]; then echo "SKIP $out"; return 0; fi
  echo "START $(date -Is) $out"
  (cd "$CODE" && CUDA_VISIBLE_DEVICES=0 "$PY" -m n6.train \
    --data "$data" --out "$out" --seed 7 --task "$task" --device cuda \
    --utility shapley --gate-architecture evidence --gate-detach-inputs \
    --detach-utility-path --bounded-w --bounded-lambda 0.3 \
    --history-abstain-variant utility_softmax --deploy joint_softgate \
    --tag "${ds}_jointsoft_true_s7") >"$out/stdout.log" 2>&1
  rc=$?; echo "END $(date -Is) rc=$rc $out"; return $rc
}
run_one M3ED "/data/emo/肖田泽科研/数据/M3ED/packed_audio_e2_zh_hubert_large" cls || exit $?
run_one MELD "/data/emo/肖田泽科研/数据/MELD/packed" cls || exit $?
run_one MOSEI "/data/emo/肖田泽科研/数据/MOSEI/packed" reg || exit $?
echo "JOINTSOFT_TRUE_DONE $(date -Is)"
