from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.sampled_context_classification import (  # noqa: E402
    CONTEXTS,
    DiagnosticProvenance,
    FrozenUtilityDecision,
    SampledContextDiagnosticError,
    SampledContextInputs,
    array_order_sha256,
    paired_sampled_context_model_contrast,
    sampled_context_classification_diagnostic,
)


FIT_SCOPE_HASH = "a" * 64


def _p(class_index: int, rows: int) -> np.ndarray:
    probability = np.full((rows, 7), 0.01, dtype=np.float32)
    probability[:, class_index] = 0.94
    return probability


def _inputs(*, seeds: int = 1) -> tuple[SampledContextInputs, str]:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    queries = np.asarray([10, 10, 20, 20], dtype=np.int64)
    clusters = np.asarray([3, 3, 8, 8], dtype=np.int64)
    task_hash = array_order_sha256(labels, queries, clusters)
    contexts = {
        "s": np.vstack([_p(1, 2), _p(0, 2)]),
        "s_plus_candidate": np.vstack([_p(0, 2), _p(1, 2)]),
        "t": np.vstack([_p(0, 2), _p(1, 2)]),
        "t_minus_candidate": np.vstack([_p(1, 2), _p(0, 2)]),
    }
    if seeds > 1:
        contexts = {
            name: np.repeat(value[None, :, :], seeds, axis=0)
            for name, value in contexts.items()
        }
    return (
        SampledContextInputs(
            labels,
            queries,
            clusters,
            contexts,
            DiagnosticProvenance("train_fold_oof", FIT_SCOPE_HASH, task_hash),
        ),
        task_hash,
    )


def _decision(scores: np.ndarray, task_hash: str, threshold: float = 0.0) -> FrozenUtilityDecision:
    return FrozenUtilityDecision(scores, threshold, task_hash)


def test_selected_candidate_uses_s_plus_h_and_retains_t() -> None:
    inputs, task_hash = _inputs()
    report = sampled_context_classification_diagnostic(
        inputs,
        _decision(np.ones(4), task_hash),
    )
    addition = report["addition"]
    deletion = report["deletion"]
    assert addition["policy_metrics"]["accuracy"] == 1.0
    assert addition["policy_metrics"]["macro_f1"] == 1.0
    assert deletion["policy_metrics"]["accuracy"] == 1.0
    assert deletion["policy_metrics"]["macro_f1"] == 1.0
    assert addition["relative_to_fixed_endpoints"]["always_s"]["nll_regret"] < 0
    assert addition["relative_to_fixed_endpoints"]["always_s_plus_candidate"]["nll_regret"] == 0
    assert deletion["relative_to_fixed_endpoints"]["always_t"]["nll_regret"] == 0
    assert deletion["relative_to_fixed_endpoints"]["always_t_minus_candidate"]["nll_harm_rate"] == 0
    assert report["decision"]["query_cluster_macro_selected_rate"] == 1.0


def test_fallback_uses_s_and_t_minus_h_with_strict_threshold() -> None:
    inputs, task_hash = _inputs()
    # Equality is deliberately a fallback, not a selection.
    report = sampled_context_classification_diagnostic(
        inputs,
        _decision(np.zeros(4), task_hash, threshold=0.0),
    )
    assert report["decision"]["selected_tasks"] == 0
    assert report["addition"]["relative_to_fixed_endpoints"]["always_s"]["nll_regret"] == 0
    assert (
        report["deletion"]["relative_to_fixed_endpoints"]["always_t_minus_candidate"][
            "nll_regret"
        ]
        == 0
    )


def test_five_base_seeds_are_ensembled_before_the_same_diagnostic() -> None:
    one_seed, task_hash = _inputs(seeds=1)
    five_seed, _ = _inputs(seeds=5)
    decision = _decision(np.ones(4), task_hash)
    first = sampled_context_classification_diagnostic(one_seed, decision)
    fifth = sampled_context_classification_diagnostic(five_seed, decision)
    assert first["counts"]["base_seeds"] == 1
    assert fifth["counts"]["base_seeds"] == 5
    assert first["addition"]["policy_metrics"] == fifth["addition"]["policy_metrics"]
    assert first["deletion"]["policy_metrics"] == fifth["deletion"]["policy_metrics"]


def _duplicated_query_inputs(repetitions: int) -> tuple[SampledContextInputs, str]:
    labels = np.asarray([0] * repetitions + [1], dtype=np.int64)
    queries = np.asarray([10] * repetitions + [20], dtype=np.int64)
    clusters = np.asarray([3] * repetitions + [8], dtype=np.int64)
    rows = len(labels)
    task_hash = array_order_sha256(labels, queries, clusters)
    correct = np.vstack([_p(0, repetitions), _p(1, 1)])
    wrong = np.vstack([_p(1, repetitions), _p(0, 1)])
    inputs = SampledContextInputs(
        labels,
        queries,
        clusters,
        {
            "s": wrong,
            "s_plus_candidate": correct,
            "t": correct,
            "t_minus_candidate": wrong,
        },
        DiagnosticProvenance("train_fit_only", FIT_SCOPE_HASH, task_hash),
    )
    return inputs, task_hash


def test_repeated_candidate_tasks_do_not_inflate_query_or_cluster_metrics() -> None:
    once, once_hash = _duplicated_query_inputs(1)
    repeated, repeated_hash = _duplicated_query_inputs(50)
    once_report = sampled_context_classification_diagnostic(
        once, _decision(np.ones(2), once_hash)
    )
    repeated_report = sampled_context_classification_diagnostic(
        repeated, _decision(np.ones(51), repeated_hash)
    )
    assert once_report["addition"]["policy_metrics"] == repeated_report["addition"][
        "policy_metrics"
    ]
    assert repeated_report["counts"] == {
        "sampled_tasks": 51,
        "unique_queries": 2,
        "clusters": 2,
        "base_seeds": 1,
        "emotion_classes": 7,
    }


def test_paired_model_contrast_is_in_memory_aggregate_only_and_directional() -> None:
    inputs, task_hash = _inputs()
    better = _decision(np.ones(4), task_hash)
    worse = _decision(-np.ones(4), task_hash)
    contrast = paired_sampled_context_model_contrast(inputs, better, worse)
    assert contrast["addition"]["paired_nll"]["a_minus_b"] < 0
    assert contrast["addition"]["paired_nll"]["a_win_rate_vs_b"] == 1.0
    assert contrast["deletion"]["paired_nll"]["a_minus_b"] < 0
    assert contrast["decision_contrast"]["query_cluster_macro_disagreement_rate"] == 1.0
    assert "sampled contexts only" in contrast["claim_boundary"]

    def assert_no_arrays(value: object) -> None:
        assert not isinstance(value, np.ndarray)
        if isinstance(value, dict):
            for item in value.values():
                assert_no_arrays(item)

    assert_no_arrays(contrast)
    assert "query_codes" not in str(contrast)
    assert "cluster_codes" not in str(contrast)


def test_rejects_bad_shape_nonfinite_label_range_and_query_inconsistency() -> None:
    inputs, task_hash = _inputs()
    contexts = {name: value[0].copy() for name, value in inputs.context_probabilities.items()}
    contexts["s"][0, 0] = np.nan
    with pytest.raises(SampledContextDiagnosticError, match="non-finite"):
        SampledContextInputs(
            inputs.query_labels,
            inputs.query_codes,
            inputs.cluster_codes,
            contexts,
            DiagnosticProvenance("train_fold_oof", FIT_SCOPE_HASH, task_hash),
        )

    bad_labels = inputs.query_labels.copy()
    bad_labels[0] = 7
    with pytest.raises(SampledContextDiagnosticError, match="query_labels"):
        SampledContextInputs(
            bad_labels,
            inputs.query_codes,
            inputs.cluster_codes,
            {name: value[0] for name, value in inputs.context_probabilities.items()},
            inputs.provenance,
        )

    inconsistent = inputs.query_labels.copy()
    inconsistent[1] = 1
    with pytest.raises(SampledContextDiagnosticError, match="multiple labels"):
        SampledContextInputs(
            inconsistent,
            inputs.query_codes,
            inputs.cluster_codes,
            {name: value[0] for name, value in inputs.context_probabilities.items()},
            inputs.provenance,
        )


def test_rejects_sealed_or_unfrozen_provenance_and_task_hash_mismatch() -> None:
    with pytest.raises(SampledContextDiagnosticError, match="sealed-role"):
        DiagnosticProvenance("sealed_test", FIT_SCOPE_HASH, "b" * 64)
    with pytest.raises(SampledContextDiagnosticError, match="threshold_provenance"):
        FrozenUtilityDecision(np.ones(2), 0.0, "b" * 64, "selection_tuned")

    inputs, _ = _inputs()
    with pytest.raises(SampledContextDiagnosticError, match="task-order hashes"):
        sampled_context_classification_diagnostic(
            inputs,
            _decision(np.ones(4), "b" * 64),
        )
    with pytest.raises(SampledContextDiagnosticError, match="not aligned"):
        sampled_context_classification_diagnostic(
            inputs,
            _decision(np.ones(3), inputs.provenance.task_order_sha256),
        )


def test_public_report_states_sampled_context_boundary_and_has_no_row_outputs() -> None:
    inputs, task_hash = _inputs()
    report = sampled_context_classification_diagnostic(
        inputs,
        _decision(np.ones(4), task_hash),
    )
    assert report["status"] == "sampled_context_diagnostic_only"
    assert "not a final query-level history policy" in report["claim_boundary"]
    assert report["provenance"]["contains_row_predictions"] is False
    assert report["provenance"]["contains_row_identifiers"] is False
    assert set(report["addition"]["policy_metrics"]) == {
        "macro_f1",
        "accuracy",
        "nll",
        "brier",
        "cluster_macro_macro_f1",
        "cluster_macro_accuracy",
        "cluster_macro_nll",
        "cluster_macro_brier",
    }
    assert all(context not in report for context in CONTEXTS)
