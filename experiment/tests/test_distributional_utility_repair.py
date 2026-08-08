from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.distributional_utility_repair import (  # noqa: E402
    ComponentSpec,
    DistributionalRepairContractError,
    HeadPrediction,
    cache_from_mapping,
    compose_expected_utility,
    compose_registered_models,
    crossed_seed_shared_cluster_bootstrap,
    fit_distributional_head,
    fit_oof_coverage_threshold,
    load_repair_config,
    policy_metrics,
)


def _head(values: list[float]) -> HeadPrediction:
    expected = np.asarray(values, dtype=float)
    probability = np.full(len(expected), 0.5)
    magnitude = np.ones(len(expected))
    return HeadPrediction(probability, magnitude, magnitude, expected)


def test_compose_expected_utility_uses_sign_and_both_severities():
    probability = np.asarray([0.1, 0.9])
    positive = np.asarray([1.0, 1.0])
    negative = np.asarray([2.0, 2.0])
    result = compose_expected_utility(probability, positive, negative)
    assert np.allclose(result, [-1.7, 0.7])


def test_compose_expected_utility_rejects_invalid_components():
    with np.testing.assert_raises(ValueError):
        compose_expected_utility(
            np.asarray([1.1]), np.asarray([1.0]), np.asarray([1.0])
        )
    with np.testing.assert_raises(ValueError):
        compose_expected_utility(
            np.asarray([0.5]), np.asarray([-1.0]), np.asarray([1.0])
        )


def test_registered_pseudo_and_true_decisions_differ_only_in_second_target():
    forward_first = _head([0.8, 0.1])
    forward_second = _head([0.4, 0.3])
    backward_second = _head([-0.2, 0.5])
    models = compose_registered_models(
        forward_first, forward_second, backward_second
    )
    assert np.allclose(
        models["distributional_pseudo_bidirectional"].decision, [0.4, 0.1]
    )
    assert np.allclose(
        models["distributional_true_bidirectional"].decision, [-0.2, 0.1]
    )
    assert models["distributional_pseudo_bidirectional"].head_targets == (
        "forward",
        "forward",
    )
    assert models["distributional_true_bidirectional"].head_targets == (
        "forward",
        "backward",
    )


def test_fit_threshold_and_policy_metrics_preserve_benefit_positive_polarity():
    assert fit_oof_coverage_threshold(np.asarray([4.0, 3.0, 2.0, 1.0]), 0.25) == 3.5
    strict = np.asarray([1.0, -2.0, 0.5, -0.1])
    decision = np.asarray([2.0, 1.0, -1.0, -2.0])
    clusters = np.asarray([0, 0, 1, 1])
    metrics, cluster_excess = policy_metrics(
        strict, decision, clusters, threshold=0.0
    )
    assert metrics["coverage"] == 0.5
    assert metrics["selected_harm_rate"] == 0.5
    assert metrics["selected_positive_precision"] == 0.5
    assert metrics["absolute_row_mean_excess_nll_vs_fallback"] == 0.25
    assert metrics["absolute_cluster_macro_excess_nll_vs_fallback"] == 0.25
    assert np.allclose(cluster_excess, [0.5, 0.0])


def test_crossed_bootstrap_is_paired_deterministic_and_shared_cluster():
    reference = np.arange(15, dtype=float).reshape(3, 5)
    candidate = reference - 1.0
    first = crossed_seed_shared_cluster_bootstrap(
        candidate, reference, replicates=200, seed=41
    )
    second = crossed_seed_shared_cluster_bootstrap(
        candidate, reference, replicates=200, seed=41
    )
    assert first == second
    assert first["point_difference"] == -1.0
    assert first["ci95_low"] == -1.0
    assert first["ci95_high"] == -1.0
    assert first["bootstrap_probability_difference_below_zero"] == 1.0


def test_private_cache_fails_closed_on_restricted_role_field():
    with np.testing.assert_raises(DistributionalRepairContractError):
        cache_from_mapping({"validation_x": np.zeros((2, 1))})


def test_small_distributional_head_emits_valid_components():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(240, 4))
    utility = 0.4 * x[:, 0] + rng.normal(scale=0.2, size=len(x))
    spec = ComponentSpec(
        max_iter=5,
        max_leaf_nodes=7,
        min_samples_leaf=5,
        early_stopping=False,
        minimum_conditional_rows=20,
    )
    model = fit_distributional_head(x, utility, spec=spec, seed=17)
    prediction = model.predict(x[:12])
    assert prediction.expected_utility.shape == (12,)
    assert np.all((prediction.positive_probability >= 0.0) & (prediction.positive_probability <= 1.0))
    assert np.all(prediction.positive_magnitude >= 0.0)
    assert np.all(prediction.negative_magnitude >= 0.0)


def test_frozen_config_registers_matched_six_component_controls():
    config, spec = load_repair_config(
        ROOT / "configs" / "emotiontalk_distributional_utility_repair_v1.json"
    )
    models = {row["mode"]: row for row in config["registered_models"]}
    assert models["pseudo_bidirectional"]["component_estimators"] == 6
    assert models["true_bidirectional"]["component_estimators"] == 6
    assert spec.max_iter == 60
