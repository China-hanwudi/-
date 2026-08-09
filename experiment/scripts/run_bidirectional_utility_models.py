from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.bidirectional_utility_model import (  # noqa: E402
    DEFAULT_SEEDS,
    PAIRED_BOOTSTRAP_REPLICATES,
    REPORT_SCHEMA_VERSION,
    run_private_cache_model_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit five-seed directional/pseudo/true-bidirectional utility MLPs from the private "
            "train-only OOF cache and emit aggregate v3 model-selection JSON with crossed "
            "and legacy bootstrap sensitivities plus an exact-coverage diagnostic."
        )
    )
    parser.add_argument("--private-cache", type=Path, required=True)
    parser.add_argument("--lineage-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-oof-splits", type=int, default=5)
    parser.add_argument(
        "--paired-bootstrap-replicates",
        type=int,
        default=PAIRED_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "replace only an existing report with the same v3 schema; v1/v2 artifacts "
            "are always protected"
        ),
    )
    args = parser.parse_args()
    report = run_private_cache_model_selection(
        args.private_cache,
        args.output,
        lineage_report_path=args.lineage_report,
        seeds=DEFAULT_SEEDS,
        maximum_oof_splits=args.maximum_oof_splits,
        paired_bootstrap_replicates=args.paired_bootstrap_replicates,
        overwrite=args.overwrite,
    )
    print(
        f"schema={REPORT_SCHEMA_VERSION} selected={report['selected_model']} "
        f"status={report['status']} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
