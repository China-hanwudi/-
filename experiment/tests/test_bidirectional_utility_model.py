from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.bidirectional_utility_model import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    DEFAULT_SEEDS,
    REPORT_SCHEMA_VERSION,
    BidirectionalUtilityCache,
    UtilityModelContractError,
    UtilityModelSpec,
    UtilityPredictions,
    UtilitySplit,
    assert_aggregate_report,
    benefit_positive_policy_metrics,
    cache_from_mapping,
    default_model_specs,
    exact_rank_coverage_diagnostic,
    exact_rank_coverage_selection,
    fit_oof_coverage_threshold,
    fit_utility_model,
    group_oof_predictions,
    load_private_oof_cache,
    paired_seed_cluster_bootstrap,
    paired_seed_shared_cluster_bootstrap,
    run_five_seed_model_selection,
    trainable_parameter_count,
    validate_oof_lineage_report,
    write_aggregate_report,
    _training_targets,
)


def _split(rows_per_group: int, groups: int, *, seed: int) -> UtilitySplit:
    rng = np.random.default_rng(seed)
    base = np.linspace(-2.0, 2.0, rows_per_group)
    x0 = np.tile(base, groups)
    x1 = rng.normal(scale=0.4, size=len(x0))
    x2 = np.sin(x0) + rng.normal(scale=0.03, size=len(x0))
    x = np.column_stack([x0, x1, x2])
    forward = 1.4 * x0 + 0.08 * x1
    backward = 1.2 * x0 - 0.08 * x1
    cluster_codes = np.repeat(np.arange(groups), rows_per_group)
    return UtilitySplit.validated(
        x, forward, backward, cluster_codes, label="synthetic"
    )


def _cache() -> BidirectionalUtilityCache:
    return BidirectionalUtilityCache(
        fit=_split(8, 9, seed=1),
        selection=_split(8, 5, seed=2),
        feature_names=("x0", "x1", "x2"),
        source_hashes={},
    )


def _cache_mapping() -> dict[str, np.ndarray]:
    cache = _cache()
    return {
        "schema_version": np.asarray([CACHE_SCHEMA_VERSION]),
        "fit_x": cache.fit.x,
        "fit_forward": cache.fit.forward,
        "fit_backward": cache.fit.backward,
        "fit_cluster_codes": cache.fit.cluster_codes.astype(np.int32),
        "selection_x": cache.selection.x,
        "selection_forward": cache.selection.forward,
        "selection_backward": cache.selection.backward,
        "selection_cluster_codes": cache.selection.cluster_codes.astype(np.int32),
        "fit_forward_seed": np.stack([cache.fit.forward, cache.fit.forward]),
        "fit_backward_seed": np.stack([cache.fit.backward, cache.fit.backward]),
        "selection_forward_seed": np.stack(
            [cache.selection.forward, cache.selection.forward]
        ),
        "selection_backward_seed": np.stack(
            [cache.selection.backward, cache.selection.backward]
        ),
        "feature_names": np.asarray(cache.feature_names),
        "base_config_sha256": np.asarray(["a" * 64]),
        "utility_config_sha256": np.asarray(["b" * 64]),
    }


def test_private_cache_loader_accepts_only_fit_and_model_selection(
    tmp_path: Path,
) -> None:
    mapping = _cache_mapping()
    path = tmp_path / "synthetic_private_cache.npz"
    np.savez_compressed(path, **mapping)
    cache = load_private_oof_cache(path)
    assert cache.fit.x.shape == (72, 3)
    assert cache.selection.x.shape == (40, 3)
    assert cache.feature_names == ("x0", "x1", "x2")

    mapping["test_x"] = np.zeros((2, 3))
    with pytest.raises(UtilityModelContractError, match="sealed split fields"):
        cache_from_mapping(mapping)


def test_oof_lineage_report_closes_upstream_role_and_hash_contract(
    tmp_path: Path,
) -> None:
    cache = _cache()
    cache = BidirectionalUtilityCache(
        cache.fit,
        cache.selection,
        cache.feature_names,
        {"base_config_sha256": "a" * 64, "utility_config_sha256": "b" * 64},
    )
    payload = {
        "protocol": "bidirectional_emotion_utility_v1",
        "status": (
            "train_only_different_set_oof_supervision_complete; "
            "utility_model_not_yet_selected"
        ),
        "cache_contract": {
            "schema": CACHE_SCHEMA_VERSION,
            "numeric_dtype": "float64",
            "contains_gold_labels": False,
            "contains_row_identifiers": False,
        },
        "task_counts": {
            "fit_oof": len(cache.fit.x),
            "model_selection": len(cache.selection.x),
            "fit_groups": len(np.unique(cache.fit.cluster_codes)),
            "model_selection_groups": len(np.unique(cache.selection.cluster_codes)),
        },
        "hashes": dict(cache.source_hashes),
        "sealed_audit": {
            "calibration_rows_used_for_training_or_metrics": 0,
            "internal_holdout_rows_used_for_training_or_metrics": 0,
            "row_level_output_emitted": False,
            "test_rows_used": 0,
            "validation_rows_used": 0,
        },
    }
    path = tmp_path / "lineage.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    validated = validate_oof_lineage_report(path, cache)
    assert validated["upstream_role_exclusion_verified"] is True
    assert len(validated["lineage_report_sha256"]) == 64

    payload["sealed_audit"]["validation_rows_used"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UtilityModelContractError, match="restricted-role access"):
        validate_oof_lineage_report(path, cache)


def test_model_contract_has_directional_degenerate_and_true_shared_models() -> None:
    specs = default_model_specs()
    assert [spec.mode for spec in specs] == [
        "forward_only",
        "backward_only",
        "pseudo_bidirectional_shared",
        "bidirectional_shared",
    ]
    pseudo = fit_utility_model(_cache().fit, specs[-2], seed=DEFAULT_SEEDS[0])
    shared = fit_utility_model(_cache().fit, specs[-1], seed=DEFAULT_SEEDS[0])
    assert pseudo.estimator.n_outputs_ == 2
    assert shared.estimator.n_outputs_ == 2
    assert pseudo.parameter_count == shared.parameter_count
    pseudo_targets = _training_targets(_cache().fit, "pseudo_bidirectional_shared")
    assert np.array_equal(pseudo_targets[:, 0], _cache().fit.forward)
    assert np.array_equal(pseudo_targets[:, 1], _cache().fit.forward)
    assert shared.parameter_count == trainable_parameter_count(3, (32, 16), 2)
    assert shared.parameter_count < 2_000_000


def test_group_oof_keeps_every_cluster_in_one_held_fold() -> None:
    split = _cache().fit
    spec = UtilityModelSpec(
        name="small_shared",
        mode="bidirectional_shared",
        hidden_layer_sizes=(6,),
        max_iter=200,
        solver="lbfgs",
        early_stopping=False,
    )
    oof = group_oof_predictions(split, spec, seed=11, maximum_splits=4)
    assert np.isfinite(oof.predictions.forward).all()
    assert np.isfinite(oof.predictions.backward).all()
    assert np.isfinite(oof.predictions.decision_score).all()
    for cluster in np.unique(split.cluster_codes):
        held_folds = np.unique(oof.fold_by_row[split.cluster_codes == cluster])
        assert len(held_folds) == 1


def test_benefit_positive_polarity_selects_helpful_and_falls_back_on_harmful() -> None:
    utility = np.asarray([-1.0, -0.2, 0.0, 0.2, 1.0])
    predictions = UtilityPredictions(utility, utility, utility)
    assert predictions.selected().tolist() == [False, False, False, True, True]
    metrics = benefit_positive_policy_metrics(
        utility, utility, np.asarray([0, 0, 1, 1, 1])
    )
    assert metrics["positive_utility_selection_rate"] == 1.0
    assert metrics["negative_utility_fallback_rate"] == 1.0
    assert metrics["mean_excess_nll_vs_fallback"] == pytest.approx(-0.24)
    assert metrics["mean_oracle_opportunity_regret"] == 0.0
    assert metrics["mean_policy_utility"] == pytest.approx(0.24)


def test_primary_coverage_threshold_is_fit_oof_only_and_deterministic() -> None:
    score = np.arange(100, dtype=float)
    threshold = fit_oof_coverage_threshold(score, 0.25)
    assert threshold == 74.5
    assert np.mean(score > threshold) == 0.25
    with pytest.raises(UtilityModelContractError, match="strictly between"):
        fit_oof_coverage_threshold(score, 1.0)


def test_exact_rank_coverage_is_label_blind_exact_and_tie_deterministic() -> None:
    score = np.asarray([0.1, 0.9, 0.9, 0.4, 0.2, 0.3, 0.0, -0.1])
    selected = exact_rank_coverage_selection(score, 0.25)
    # Stable input order is used only to resolve the tied top score; no label,
    # utility, cluster, or identifier is an input to the selector.
    assert selected.tolist() == [False, True, True, False, False, False, False, False]
    assert selected.mean() == 0.25

    split = UtilitySplit.validated(
        np.column_stack([score, score**2]),
        np.linspace(-1.0, 1.0, len(score)),
        np.linspace(1.0, -1.0, len(score)),
        np.repeat(np.arange(2), 4),
        label="coverage_diagnostic",
    )
    predictions = UtilityPredictions(score, None, score)
    diagnostic = exact_rank_coverage_diagnostic(split, predictions, coverage=0.25)
    assert diagnostic["selection_uses_labels_or_utilities"] is False
    assert diagnostic["selected_rows"] == 2
    assert diagnostic["realized_coverage"] == 0.25
    assert "selected" not in diagnostic
    assert "mask" not in diagnostic

    with pytest.raises(UtilityModelContractError, match="integral selected-row"):
        exact_rank_coverage_selection(np.arange(10, dtype=float), 0.25)


def test_paired_seed_cluster_bootstrap_is_paired_deterministic_and_aggregate() -> None:
    reference = np.arange(30, dtype=float).reshape(5, 6) / 100.0
    candidate = reference - 0.02
    first = paired_seed_cluster_bootstrap(candidate, reference, replicates=500, seed=7)
    second = paired_seed_cluster_bootstrap(candidate, reference, replicates=500, seed=7)
    assert first == second
    assert first["five_seed_cluster_macro_difference"] == pytest.approx(-0.02)
    assert first["ci95_upper_below_zero"] is True
    assert first["training_seed_count"] == 5
    assert first["cluster_count"] == 6
    assert first["inferential_role"] == "legacy_v2_sensitivity_only"
    assert first["bootstrap_design"] == (
        "nested_seed_then_independent_cluster_resampling"
    )


def test_crossed_bootstrap_matches_shared_cluster_reference_distribution() -> None:
    reference = np.arange(30, dtype=float).reshape(5, 6) / 50.0
    candidate = reference + np.asarray(
        [
            [-0.09, -0.03, 0.02, 0.04, 0.08, 0.12],
            [-0.07, -0.04, 0.01, 0.03, 0.07, 0.11],
            [-0.08, -0.02, 0.00, 0.02, 0.06, 0.10],
            [-0.10, -0.05, -0.01, 0.05, 0.09, 0.13],
            [-0.06, -0.01, 0.03, 0.06, 0.10, 0.14],
        ]
    )
    replicates = 500
    seed = 23
    observed = paired_seed_shared_cluster_bootstrap(
        candidate,
        reference,
        replicates=replicates,
        seed=seed,
    )
    difference = candidate - reference
    rng = np.random.default_rng(seed)
    seed_index = rng.integers(0, 5, size=(replicates, 5))
    cluster_index = rng.integers(0, 6, size=(replicates, 6))
    expected = difference[
        seed_index[:, :, None],
        cluster_index[:, None, :],
    ].mean(axis=(1, 2))
    assert observed["ci95_percentile"] == pytest.approx(
        np.quantile(expected, [0.025, 0.975]).tolist()
    )
    assert observed["bootstrap_probability_difference_below_zero"] == pytest.approx(
        np.mean(expected < 0.0)
    )
    assert observed["inferential_role"] == "primary_open_role_sensitivity"
    assert observed["bootstrap_design"] == (
        "crossed_seed_with_shared_cluster_resampling"
    )
    assert observed["cluster_resampling"] == (
        "one_shared_whole_cluster_index_vector_across_all_resampled_seed_slots"
    )


def test_bootstrap_equality_and_lower_is_better_direction_are_explicit() -> None:
    reference = np.arange(30, dtype=float).reshape(5, 6) / 100.0
    equal_crossed = paired_seed_shared_cluster_bootstrap(
        reference,
        reference,
        replicates=500,
        seed=31,
    )
    equal_legacy = paired_seed_cluster_bootstrap(
        reference,
        reference,
        replicates=500,
        seed=31,
    )
    for result in (equal_crossed, equal_legacy):
        assert result["five_seed_cluster_macro_difference"] == 0.0
        assert result["ci95_percentile"] == [0.0, 0.0]
        assert result["ci95_upper_below_zero"] is False
        assert result["direction"] == "lower_is_better"

    better = paired_seed_shared_cluster_bootstrap(
        reference - 0.02,
        reference,
        replicates=500,
        seed=37,
    )
    worse = paired_seed_shared_cluster_bootstrap(
        reference,
        reference - 0.02,
        replicates=500,
        seed=37,
    )
    assert better["five_seed_cluster_macro_difference"] == pytest.approx(-0.02)
    assert better["ci95_upper_below_zero"] is True
    assert worse["five_seed_cluster_macro_difference"] == pytest.approx(0.02)
    assert worse["ci95_upper_below_zero"] is False


def test_paired_bootstrap_matches_nested_seed_then_cluster_reference() -> None:
    reference = np.arange(30, dtype=float).reshape(5, 6) / 50.0
    candidate = reference + np.asarray(
        [
            [-0.04, -0.03, -0.02, -0.01, 0.00, 0.01],
            [-0.02, -0.01, 0.00, 0.01, 0.02, 0.03],
            [-0.06, -0.05, -0.04, -0.03, -0.02, -0.01],
            [0.01, 0.00, -0.01, -0.02, -0.03, -0.04],
            [-0.03, -0.02, -0.01, 0.00, 0.01, 0.02],
        ]
    )
    replicates = 500
    seed = 19
    observed = paired_seed_cluster_bootstrap(
        candidate,
        reference,
        replicates=replicates,
        seed=seed,
    )
    difference = candidate - reference
    rng = np.random.default_rng(seed)
    seed_index = rng.integers(0, 5, size=(replicates, 5))
    cluster_index = rng.integers(0, 6, size=(replicates, 5, 6))
    expected = difference[seed_index[:, :, None], cluster_index].mean(axis=(1, 2))
    assert observed["ci95_percentile"] == pytest.approx(
        np.quantile(expected, [0.025, 0.975]).tolist()
    )
    assert observed["bootstrap_probability_difference_below_zero"] == pytest.approx(
        np.mean(expected < 0.0)
    )


def test_end_to_end_five_seed_selection_is_aggregate_and_polarity_safe(
    tmp_path: Path,
) -> None:
    small_specs = tuple(
        UtilityModelSpec(
            name=spec.name,
            mode=spec.mode,
            hidden_layer_sizes=(6,),
            alpha=1e-3,
            max_iter=220,
            solver="lbfgs",
            early_stopping=False,
        )
        for spec in default_model_specs()
    )
    report = run_five_seed_model_selection(
        _cache(),
        specs=small_specs,
        maximum_oof_splits=3,
        paired_bootstrap_replicates=500,
        enforce_registered_contract=False,
    )
    assert REPORT_SCHEMA_VERSION == "bidirectional_utility_model_report_v3"
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["seed_contract"] == {
        "count": 5,
        "distinct": True,
        "seeds": list(DEFAULT_SEEDS),
    }
    assert {model["mode"] for model in report["models"]} == {
        "forward_only",
        "backward_only",
        "pseudo_bidirectional_shared",
        "bidirectional_shared",
    }
    shared = next(
        model for model in report["models"] if model["mode"] == "bidirectional_shared"
    )
    policy = shared["five_seed_ensemble"]["model_selection"]["policy"]
    fit_policy = shared["five_seed_ensemble"]["fit_oof"]["policy"]
    assert fit_policy["coverage"] <= 0.25
    assert fit_policy["coverage"] >= 0.20
    assert policy["selected_positive_precision"] >= 0.95
    assert policy["negative_utility_fallback_rate"] >= 0.95
    assert report["selection_contract"]["target_fit_oof_history_coverage"] == 0.25
    assert report["selection_contract"]["threshold_source"] == "fit_group_oof_only"
    coverage_contract = report["selection_contract"]["coverage_matched_diagnostic"]
    assert coverage_contract["role"] == (
        "transductive_diagnostic_only_not_deployable_or_used_for_ranking"
    )
    assert coverage_contract["selection_inputs"] == (
        "decision_score_only_no_labels_utilities_clusters_or_identifiers"
    )
    assert report["uncertainty_contract"]["primary_open_role_sensitivity"] == (
        "crossed_seed_with_shared_cluster_resampling"
    )
    assert report["uncertainty_contract"]["legacy_sensitivity_only"] == (
        "nested_seed_then_independent_cluster_resampling"
    )
    assert shared["shared_hidden_representation"] is True
    assert shared["output_heads"] == 2
    assert len(shared["per_seed"]) == 5
    for seed_record in shared["per_seed"]:
        diagnostic = seed_record["model_selection_exact_25pct_transductive_diagnostic"]
        assert diagnostic["selected_rows"] == 10
        assert diagnostic["realized_coverage"] == 0.25
        assert diagnostic["selection_uses_labels_or_utilities"] is False
    ensemble_diagnostic = shared["five_seed_ensemble"][
        "model_selection_exact_25pct_transductive_diagnostic"
    ]
    assert ensemble_diagnostic["selected_rows"] == 10
    assert ensemble_diagnostic["realized_coverage"] == 0.25
    pseudo = next(
        model
        for model in report["models"]
        if model["mode"] == "pseudo_bidirectional_shared"
    )
    assert pseudo["target_semantics"] == "degenerate_same_set_duplicated_forward_target"
    assert (
        pseudo["architecture"]["parameter_count"]
        == shared["architecture"]["parameter_count"]
    )
    assert len(report["paired_model_contrasts"]) == 3
    assert {contrast["reference"] for contrast in report["paired_model_contrasts"]} == {
        "forward_only_mlp",
        "backward_only_mlp",
        "pseudo_bidirectional_same_set_mlp",
    }
    for contrast in report["paired_model_contrasts"]:
        deployment = contrast["deployment_operating_point"]
        matched = contrast["coverage_matched_transductive_diagnostic"]
        assert deployment["role"] == "primary_deployment_style_point_estimate"
        assert matched["role"] == ("diagnostic_only_not_deployable_or_used_for_ranking")
        assert matched["ensemble_realized_coverage"] == {
            "candidate": 0.25,
            "reference": 0.25,
            "candidate_minus_reference": 0.0,
        }
        for section, metric_key in (
            (deployment, "paired_cluster_excess_nll_uncertainty"),
            (matched, "paired_cluster_excess_nll_uncertainty"),
            (
                contrast["strict_utility_regression_diagnostic"],
                "paired_cluster_rmse_uncertainty",
            ),
        ):
            designs = section[metric_key]
            assert set(designs) == {
                "primary_crossed_seed_shared_cluster",
                "legacy_nested_seed_independent_cluster_sensitivity",
            }
            assert (
                designs["primary_crossed_seed_shared_cluster"]["inferential_role"]
                == "primary_open_role_sensitivity"
            )
            assert (
                designs["legacy_nested_seed_independent_cluster_sensitivity"][
                    "inferential_role"
                ]
                == "legacy_v2_sensitivity_only"
            )
    assert_aggregate_report(report)

    disguised = json.loads(json.dumps(report))
    disguised["data_contract"]["scores"] = [0.1, 0.2]
    with pytest.raises(UtilityModelContractError, match="forbidden field"):
        assert_aggregate_report(disguised)

    leaked_mask = json.loads(json.dumps(report))
    leaked_mask["data_contract"]["selection_mask"] = [True, False]
    with pytest.raises(UtilityModelContractError, match="forbidden field"):
        assert_aggregate_report(leaked_mask)

    unknown_top = json.loads(json.dumps(report))
    unknown_top["extra"] = "not registered"
    with pytest.raises(UtilityModelContractError, match="top-level schema"):
        assert_aggregate_report(unknown_top)

    stale_schema = json.loads(json.dumps(report))
    stale_schema["schema_version"] = "bidirectional_utility_model_report_v2"
    with pytest.raises(UtilityModelContractError, match="schema must equal"):
        assert_aggregate_report(stale_schema)

    output = write_aggregate_report(report, tmp_path / "aggregate.json")
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["data_contract"]["row_level_output"] is False
    assert stored["data_contract"]["cluster_identifiers_emitted"] is False
    assert "predictions" not in output.read_text(encoding="utf-8")
    serialized = output.read_text(encoding="utf-8")
    assert "selection_mask" not in serialized
    assert "cluster_codes" not in serialized

    v2_path = tmp_path / "legacy_v2.json"
    v2_path.write_text(
        json.dumps({"schema_version": "bidirectional_utility_model_report_v2"}),
        encoding="utf-8",
    )
    with pytest.raises(UtilityModelContractError, match="different schema version"):
        write_aggregate_report(report, v2_path, overwrite=True)


def test_exactly_five_distinct_seeds_are_required() -> None:
    spec = UtilityModelSpec(
        name="small_forward",
        mode="forward_only",
        hidden_layer_sizes=(4,),
        max_iter=50,
    )
    with pytest.raises(UtilityModelContractError, match="five distinct"):
        run_five_seed_model_selection(_cache(), specs=(spec,), seeds=(1, 2, 3, 4, 4))
    with pytest.raises(
        UtilityModelContractError, match="registered run requires all four"
    ):
        run_five_seed_model_selection(
            _cache(),
            specs=(spec,),
            seeds=(1, 2, 3, 4, 5),
        )
