#!/usr/bin/env bash
set -u
ROOT="/data/emo/肖田泽科研/模型/model6_integrated_H_20261003"
CODE="$ROOT/code"
PY="/home/emo/anaconda3/envs/torch2.0.1_cuda11.8/bin/python"
RUNS="$ROOT/runs/matrix"
run_one() {
  local ds="$1"; local data="$2"; local task="$3"; local arm="$4"; local variant="$5"
  local out="$RUNS/${ds}_${arm}_s7"
  mkdir -p "$out"
  if [[ -f "$out/FINAL_RESULT.json" ]]; then echo "SKIP $out"; return 0; fi
  echo "START $(date -Is) $out"
  (cd "$CODE" && CUDA_VISIBLE_DEVICES=0 "$PY" -m n6.train \
    --data "$data" --out "$out" --seed 7 --task "$task" --device cuda \
    --utility shapley --gate-architecture evidence --gate-detach-inputs \
    --detach-utility-path --bounded-w --bounded-lambda 0.3 \
    --history-abstain-variant "$variant" --deploy closed_loop \
    --tag "${ds}_${arm}_s7") >"$out/stdout.log" 2>&1
  rc=$?; echo "END $(date -Is) rc=$rc $out"; return $rc
}
M3ED="/data/emo/肖田泽科研/数据/M3ED/packed_audio_e2_zh_hubert_large"
MELD="/data/emo/肖田泽科研/数据/MELD/packed"
MOSEI="/data/emo/肖田泽科研/数据/MOSEI/packed"
for spec in \
  "M3ED|$M3ED|cls|jointsoft|utility_softmax" \
  "MELD|$MELD|cls|jointsoft|utility_softmax" \
  "MOSEI|$MOSEI|reg|jointsoft|utility_softmax" \
  "M3ED|$M3ED|cls|semantic|utility_semantic" \
  "M3ED|$M3ED|cls|sparsemax|utility_sparsemax" \
  "MELD|$MELD|cls|semantic|utility_semantic" \
  "MELD|$MELD|cls|sparsemax|utility_sparsemax" \
  "MOSEI|$MOSEI|reg|semantic|utility_semantic" \
  "MOSEI|$MOSEI|reg|sparsemax|utility_sparsemax"; do
  IFS='|' read -r ds data task arm variant <<<"$spec"
  run_one "$ds" "$data" "$task" "$arm" "$variant" || exit $?
done
echo "TARGETED_DONE $(date -Is)"
