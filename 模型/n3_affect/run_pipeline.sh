#!/bin/bash
set -e

ROOT=/root/肖田泽最强
MODEL=$ROOT/模型
MANIFESTS=$ROOT/meld结果/train结果/manifests
FEAT=$ROOT/meld结果/train结果/features
TRAIN_OUT=$ROOT/meld结果/train结果
VAL_OUT=$ROOT/meld结果/val结果
TEST_OUT=$ROOT/meld结果/test结果
LOG=$TRAIN_OUT/logs/pipeline.log

mkdir -p $TRAIN_OUT/logs $FEAT/train $FEAT/val $FEAT/test $VAL_OUT $TEST_OUT

exec >> $LOG 2>&1

echo "=== PIPELINE START $(date -Iseconds) ==="

echo "=== MANIFESTS ==="
python $MODEL/n3_affect/generate_meld_manifests.py --out-dir $MANIFESTS

echo "=== FEATURE EXTRACTION ALL SPLITS $(date -Iseconds) ==="
python $MODEL/n3_affect/extract_meld_features.py \
    --manifests $MANIFESTS/train.csv $MANIFESTS/val.csv $MANIFESTS/test.csv \
    --out-dirs $FEAT/train $FEAT/val $FEAT/test \
    --text-batch-size 64

echo "=== TRAIN $(date -Iseconds) ==="
python $MODEL/n3_affect/train_meld.py \
    --train-manifest $MANIFESTS/train.csv \
    --val-manifest $MANIFESTS/val.csv \
    --train-features $FEAT/train \
    --val-features $FEAT/val \
    --out-dir $TRAIN_OUT \
    --epochs 20 --batch-size 64 --patience 5 --lr 3e-4 --seed 17

echo "=== EVAL VAL $(date -Iseconds) ==="
python $MODEL/n3_affect/eval_meld.py \
    --checkpoint $TRAIN_OUT/checkpoints/best.pt \
    --manifest $MANIFESTS/val.csv \
    --features $FEAT/val \
    --split val --out-dir $VAL_OUT

echo "=== EVAL TEST $(date -Iseconds) ==="
python $MODEL/n3_affect/eval_meld.py \
    --checkpoint $TRAIN_OUT/checkpoints/best.pt \
    --manifest $MANIFESTS/test.csv \
    --features $FEAT/test \
    --split test --out-dir $TEST_OUT

echo "=== PIPELINE DONE $(date -Iseconds) ==="
