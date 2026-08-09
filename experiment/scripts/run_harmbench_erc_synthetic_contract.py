from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.harmbench_erc_contract import load_development_protocol  # noqa: E402
from hva_affect.harmbench_erc_inference import (  # noqa: E402
    bootstrap_cell_metrics,
    bootstrap_paired_strategy_contrast,
    make_shared_cluster_bootstrap_plan,
)
from hva_affect.harmbench_erc_public import (  # noqa: E402
    atomic_write_once,
    build_synthetic_public_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic-only HarmBench-ERC contract")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _probability(labels: np.ndarray, true_probability: np.ndarray) -> np.ndarray:
    classes = 3
    result = np.empty((len(labels), classes), dtype=np.float64)
    for index, label in enumerate(labels):
        other = (1.0 - float(true_probability[index])) / (classes - 1)
        result[index] = other
        result[index, int(label)] = float(true_probability[index])
    return result


def synthetic_bundle() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    labels = np.asarray([0, 1, 2] * 4, dtype=np.int64)
    row_ids = np.arange(len(labels), dtype=np.int64)
    clusters = np.asarray(["d0"] * 3 + ["d1"] * 3 + ["d2"] * 3 + ["d3"] * 3)
    eligible = np.asarray([False, True, True] * 4, dtype=bool)
    current_true = np.asarray([0.62, 0.68, 0.58, 0.66, 0.55, 0.64, 0.61, 0.70, 0.57, 0.63, 0.59, 0.67])
    history_true = np.asarray([0.62, 0.76, 0.43, 0.66, 0.71, 0.49, 0.61, 0.54, 0.73, 0.63, 0.46, 0.78])
    offsets = np.asarray([-0.02, -0.01, 0.0, 0.01, 0.02])
    current = np.stack(
        [_probability(labels, np.clip(current_true + offset, 0.35, 0.85)) for offset in offsets]
    )
    history = np.stack(
        [_probability(labels, np.clip(history_true + 0.5 * offset, 0.35, 0.85)) for offset in offsets]
    )
    return labels, row_ids, clusters, eligible, current, history


def main() -> int:
    args = parse_args()
    protocol = load_development_protocol(args.config)
    labels, row_ids, clusters, eligible, current, history = synthetic_bundle()
    plan = make_shared_cluster_bootstrap_plan(
        "synthetic_dialogues",
        row_ids,
        clusters,
        training_seed_count=5,
        replicates=500,
        random_seed=20260810,
    )
    cell = bootstrap_cell_metrics(
        "synthetic_dialogues",
        row_ids,
        clusters,
        labels,
        current,
        history,
        eligible,
        eligible,
        plan,
    )
    contrast = bootstrap_paired_strategy_contrast(
        "synthetic_dialogues",
        row_ids,
        clusters,
        labels,
        current,
        history,
        eligible,
        current,
        eligible,
        eligible,
        plan,
    )
    report = build_synthetic_public_report(
        protocol_sha256=protocol.canonical_sha256,
        cell=cell,
        contrast=contrast,
    )
    digest = atomic_write_once(report, args.output)
    print(f"harmbench_erc_synthetic_contract_complete sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
