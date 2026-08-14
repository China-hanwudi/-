#!/usr/bin/env bash
set -euo pipefail
MODEL="/root/肖田泽最强/模型"
BASE="/root/肖田泽最强/meld结果"
TRAIN_OUT="$BASE/train结果"
VAL_OUT="$BASE/val结果"
TEST_OUT="$BASE/test结果"
SCRATCH="/tmp/n3_run3"
export PYTHONPATH="$MODEL:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

gate() {
  python - <<'PY'
import json, sys
from pathlib import Path
p = Path("/tmp/n3_run3/train_card.json")
card = json.loads(p.read_text())
b = card.get("best_monitor") or {}
acc = float(b.get("monitor_acc") or 0)
tr = float(b.get("train_acc") or 0)
ep = int(b.get("epoch") if b.get("epoch") is not None else 99)
gap = tr - acc
ok = acc >= 0.55 and gap < 0.12 and ep <= 12
print(f"GATE monitor_acc={acc:.4f} train_acc={tr:.4f} gap={gap:.4f} epoch={ep} ok={ok}")
sys.exit(0 if ok else 1)
PY
}

train_once() {
  local lr="$1"
  rm -rf "$SCRATCH"
  mkdir -p "$SCRATCH/logs" "$SCRATCH/checkpoints"
  python "$MODEL/n3_affect/train_meld.py" \
    --config "$MODEL/configs/n3_train_v1.json" \
    --train-manifest "$TRAIN_OUT/manifests/train.csv" \
    --train-features "$TRAIN_OUT/features/train" \
    --out-dir "$SCRATCH" \
    --epochs 20 --patience 4 --min-epochs 3 --monitor-frac 0.1 \
    --batch-size 64 --lr "$lr" --seed 17
}

echo "=== ATTEMPT 1 lr=3e-4 $(date -Iseconds) ==="
train_once 0.0003
if ! gate; then
  echo "=== ATTEMPT 1 failed gate; retry lr=1.5e-4 ==="
  train_once 0.00015
  if ! gate; then
    echo "GATE_FAILED both attempts; not publishing .3"
    exit 2
  fi
fi

CKPT="$SCRATCH/checkpoints/best.pt"
mkdir -p "$TRAIN_OUT/checkpoints"
cp -a "$CKPT" "$TRAIN_OUT/checkpoints/best.pt.3"
cp -a "$SCRATCH/train_card.json" "$TRAIN_OUT/train_card.json.3"
cp -a "$SCRATCH/config_snapshot.json" "$TRAIN_OUT/config_snapshot.json.3"
cp -a "$SCRATCH/logs/pipeline.log" "$TRAIN_OUT/logs/pipeline.log.3"
rm -rf "$TRAIN_OUT/code.3"
cp -a "$SCRATCH/code" "$TRAIN_OUT/code.3"

python "$MODEL/n3_affect/eval_meld.py" \
  --checkpoint "$TRAIN_OUT/checkpoints/best.pt.3" \
  --manifest "$TRAIN_OUT/manifests/train.csv" \
  --features "$TRAIN_OUT/features/train" \
  --split train --out-dir "$TRAIN_OUT" --run-tag 3

python "$MODEL/n3_affect/eval_meld.py" \
  --checkpoint "$TRAIN_OUT/checkpoints/best.pt.3" \
  --manifest "$TRAIN_OUT/manifests/val.csv" \
  --features "$TRAIN_OUT/features/val" \
  --split val --out-dir "$VAL_OUT" --run-tag 3

python "$MODEL/n3_affect/eval_meld.py" \
  --checkpoint "$TRAIN_OUT/checkpoints/best.pt.3" \
  --manifest "$TRAIN_OUT/manifests/test.csv" \
  --features "$TRAIN_OUT/features/test" \
  --split test --out-dir "$TEST_OUT" --run-tag 3

echo "=== PUBLISHED .3 ==="
echo "--- train ---"; cat "$TRAIN_OUT/train_report.txt.3"
echo "--- val ---"; cat "$VAL_OUT/val_report.txt.3"
echo "--- test ---"; cat "$TEST_OUT/test_report.txt.3"
echo "=== DONE $(date -Iseconds) ==="
