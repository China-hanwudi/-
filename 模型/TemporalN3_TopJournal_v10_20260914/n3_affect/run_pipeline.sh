#!/usr/bin/env bash
set -euo pipefail
# v9 intentionally contains no automatic test evaluation or legacy dataset
# feature extraction. Use run_a100_training.sh with DATASET=m3ed|mosei.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec bash "$ROOT/run_a100_training.sh" "$@"
