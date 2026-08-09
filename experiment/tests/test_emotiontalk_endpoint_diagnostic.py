from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.emotiontalk_endpoint_diagnostic import (
    _write_private_cache,
    assign_frame_roles,
    objective_reversal_summary,
    summarize_seed_targets,
)


def test_role_assignment_keeps_dialogue_rows_together():
    frame = pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "dialogue": ["1", "1", "2", "2"],
    })
    ranges = {
        "base_and_utility_fit": [0, 64],
        "model_selection": [65, 79],
        "calibration": [80, 89],
        "internal_holdout_sealed": [90, 99],
    }
    groups, roles, buckets = assign_frame_roles(
        frame,
        dataset="emotiontalk",
        role_protocol="scu_set_exploration_v1",
        role_ranges=ranges,
    )
    assert groups[0] == groups[1]
    assert roles[0] == roles[1]
    assert buckets[0] == buckets[1]
    assert groups[2] == groups[3]


def test_objective_reversal_requires_mean_and_harm_tradeoff():
    excess = np.asarray([-8.0, 1.0, 1.0, 1.0, -0.5, -0.5, -0.5, -0.5])
    mean_score = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    harm_score = np.asarray([4, 5, 6, 7, 0, 1, 2, 3], dtype=float)
    gates = {
        "maximum_selected_set_jaccard_for_reversal": 0.80,
        "minimum_mean_regret_advantage_for_reversal": 0.01,
        "minimum_harm_rate_advantage_for_reversal": 0.03,
        "minimum_reversal_coverages": 1,
    }
    result = objective_reversal_summary(excess, mean_score, harm_score, [0.5], gates)
    row = result["coverages"]["0.50"]
    assert row["mean_regret_advantage_of_mean_risk"] > 0
    assert row["harm_rate_advantage_of_harm_probability"] > 0
    assert row["preference_reversal"] is True
    assert result["pass"] is True


def test_seed_target_summary_reports_stability():
    ensemble = np.linspace(-1, 1, 20)
    targets = np.vstack([ensemble, ensemble + 0.01, ensemble - 0.01])
    result = summarize_seed_targets(targets, ensemble)
    assert result["pairwise_spearman_median"] > 0.99
    assert result["majority_sign_agreement_mean"] == 1.0
    assert result["unanimous_sign_fraction"] == 1.0


def test_private_cache_uses_cluster_codes_not_identifiers(tmp_path):
    path = tmp_path / "cache.npz"
    _write_private_cache(
        path,
        fit_x=np.ones((3, 2)),
        fit_target=np.asarray([-1.0, 0.5, 1.0]),
        fit_seed_targets=np.ones((2, 3)),
        selection_x=np.ones((2, 2)),
        selection_target=np.asarray([-0.2, 0.3]),
        selection_seed_targets=np.ones((2, 2)),
        selection_counts=np.asarray([1, 2]),
        selection_clusters=np.asarray(["sensitive/a", "sensitive/b"]),
        feature_names=("a", "b"),
        base_config_sha256="a" * 64,
        diagnostic_config_sha256="b" * 64,
    )
    with np.load(path, allow_pickle=False) as archive:
        assert "selection_clusters" not in archive.files
        assert archive["selection_cluster_codes"].tolist() == [0, 1]
        assert archive["fit_x"].dtype == np.float64
        assert archive["selection_x"].dtype == np.float64
        assert set(archive.files) == {
            "schema_version", "fit_x", "fit_target", "fit_seed_targets",
            "selection_x", "selection_target", "selection_seed_targets",
            "selection_counts", "selection_cluster_codes", "feature_names",
            "base_config_sha256", "diagnostic_config_sha256",
        }
    assert b"sensitive" not in path.read_bytes()
