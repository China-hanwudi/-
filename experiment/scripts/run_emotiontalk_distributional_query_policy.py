from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.emotiontalk_distributional_query_policy_runner import (  # noqa: E402
    TRUE_MODEL_NAME,
    run_open_role_distributional_query_policy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the open-role one-prediction-per-query terminal check for the frozen "
            "distributional sign-by-severity utility repair."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--feature", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--utility-config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--lineage-report", type=Path, required=True)
    parser.add_argument("--repair-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_open_role_distributional_query_policy(
        args.data_dir,
        args.feature,
        args.base_config,
        args.utility_config,
        args.cache,
        args.checkpoint_dir,
        args.lineage_report,
        args.repair_config,
        args.output,
    )
    true_strategy = report["utility_seed_strategy_summaries"][
        "distributional_true_selected_history"
    ]
    print(
        f"model={TRUE_MODEL_NAME} successful_utility_seeds="
        f"{true_strategy['successful_utility_seeds_out_of_five']}/5 "
        f"status={report['status']} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
