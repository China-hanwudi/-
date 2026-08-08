"""Fail-closed validation for the CARMA-Affect confirmatory contracts.

This module reads configuration metadata only.  It deliberately has no dataset
adapter and must never inspect labels, media, embeddings, or per-query outputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SPLIT_PROTOCOL_ID = "scu_set_exploration_v1"
EXPECTED_ROLE_RANGES = {
    "base_and_utility_fit": (0, 64),
    "model_selection": (65, 79),
    "calibration": (80, 89),
    "internal_holdout": (90, 99),
}
REQUIRED_DATASETS = ("MELD", "EmotionTalk")
MINIMUM_MACRO_F1_GAIN = 0.005
PRIMARY_HISTORY_COVERAGE = 0.25
SEALED_ROLE_NAMES = ("calibration", "internal_holdout")
ADMISSIBLE_REFERENCE_CANDIDATES = {
    "current_only",
    "all_history",
    "coverage_matched_recency",
    "forward_only_utility",
    "backward_only_utility",
}
HISTORY_HARM_REFERENCE_CANDIDATES = (
    "all_history",
    "coverage_matched_recency",
    "forward_only_utility",
    "backward_only_utility",
)
REFERENCE_SELECTION_RULE = (
    "highest_five_seed_mean_model_selection_macro_f1_then_highest_accuracy_"
    "then_lowest_mean_regret_then_lexicographic_model_id"
)
PER_SEED_SUCCESS_CONDITIONS = (
    "macro_f1_candidate_strictly_greater_than_reference",
    "mean_regret_vs_current_non_positive",
)
BOOTSTRAP_DESIGN = "five_training_seeds_crossed_with_shared_whole_cluster_draw"
ACCURACY_NO_HARM_POINT_MINIMUM = 0.0
ACCURACY_NO_HARM_CI95_LOWER_MINIMUM = -0.005
RANDOMIZATION_REPLICATES = 10_000
RANDOMIZATION_SEED = 20_260_829


class ConfirmatoryContractError(ValueError):
    """Raised when a frozen split or confirmatory-analysis rule is unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfirmatoryContractError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{name} must be an array",
    )
    return value


def _number(value: Any, name: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    return float(value)


def load_json_contract(path: Path) -> dict[str, Any]:
    """Load a JSON object without resolving or reading any referenced data."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfirmatoryContractError(f"cannot load contract {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"contract {path} must contain a JSON object")
    return payload


def _validate_role_partition(roles: Mapping[str, Any]) -> None:
    _require(
        set(roles) == set(EXPECTED_ROLE_RANGES),
        "roles must be exactly fit, model-selection, calibration, and holdout",
    )
    assigned_buckets: list[int] = []
    for role_name, expected_range in EXPECTED_ROLE_RANGES.items():
        role = _mapping(roles[role_name], f"roles.{role_name}")
        bucket_range = _sequence(
            role.get("bucket_range_inclusive"),
            f"roles.{role_name}.bucket_range_inclusive",
        )
        _require(
            len(bucket_range) == 2
            and all(isinstance(item, int) and not isinstance(item, bool) for item in bucket_range),
            f"roles.{role_name}.bucket_range_inclusive must contain two integers",
        )
        actual_range = (int(bucket_range[0]), int(bucket_range[1]))
        _require(
            actual_range == expected_range,
            f"role range drift for {role_name}: expected {expected_range}",
        )
        assigned_buckets.extend(range(actual_range[0], actual_range[1] + 1))

    _require(
        assigned_buckets == list(range(100)),
        "role ranges must cover buckets 0..99 exactly once",
    )

    for role_name in SEALED_ROLE_NAMES:
        label_state = str(_mapping(roles[role_name], role_name).get("label_state", ""))
        _require(
            label_state.startswith("sealed_until_"),
            f"{role_name} must remain sealed until its prespecified predecessor is frozen",
        )


def validate_split_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate immutable role assignment, dataset state, and privacy boundaries."""

    _require(
        manifest.get("manifest_id") == "carma_split_manifest_v1",
        "unexpected split manifest id",
    )
    _require(
        manifest.get("split_protocol_id") == SPLIT_PROTOCOL_ID,
        f"split_protocol_id must be {SPLIT_PROTOCOL_ID}",
    )

    assignment = _mapping(manifest.get("assignment"), "assignment")
    _require(
        list(_sequence(assignment.get("hash_inputs"), "assignment.hash_inputs"))
        == ["dataset_id", "group_id", "split_protocol_id"],
        "role assignment may depend only on dataset_id, group_id, and the frozen split_protocol_id",
    )
    forbidden_hash_inputs = set(
        _sequence(assignment.get("forbidden_hash_inputs"), "assignment.forbidden_hash_inputs")
    )
    _require(
        {"model_protocol_id", "candidate_model_id", "coverage", "threshold", "seed", "labels"}
        <= forbidden_hash_inputs,
        "protocol-, candidate-, seed-, threshold-, and label-dependent role drift must be forbidden",
    )
    _require(
        assignment.get("role_assignment_locked") is True,
        "role assignment must be locked",
    )
    _require(
        assignment.get("group_never_crosses_roles") is True,
        "one group must never cross roles",
    )
    _require(
        assignment.get("missing_group_policy") == "fail_closed",
        "missing group ids must fail closed",
    )

    roles = _mapping(manifest.get("roles"), "roles")
    _validate_role_partition(roles)

    external_test = _mapping(manifest.get("external_test_policy"), "external_test_policy")
    _require(
        str(external_test.get("label_state", "")).startswith("sealed_until_"),
        "external test must remain sealed until the complete bundle is frozen",
    )
    _require(
        external_test.get("model_or_threshold_change_after_access") == "forbidden",
        "test access must not change models or thresholds",
    )
    _require(
        external_test.get("per_query_output_export") == "forbidden",
        "test per-query exports must be forbidden",
    )

    datasets = _mapping(manifest.get("datasets"), "datasets")
    _require(
        set(datasets) == {"MELD", "EmotionTalk", "IEMOCAP"},
        "dataset registry must contain exactly MELD, EmotionTalk, and IEMOCAP",
    )
    for dataset_name, dataset in datasets.items():
        dataset = _mapping(dataset, f"datasets.{dataset_name}")
        _require(
            dataset.get("raw_data_in_repository") is False,
            f"{dataset_name} raw data must stay outside the repository",
        )
    for dataset_name in REQUIRED_DATASETS:
        _require(
            _mapping(datasets[dataset_name], dataset_name).get("confirmatory_status")
            == "sealed_unopened",
            f"{dataset_name} confirmatory split must be sealed and unopened",
        )
    _require(
        str(_mapping(datasets["IEMOCAP"], "IEMOCAP").get("confirmatory_status", ""))
        .startswith("ineligible_until_license_verified"),
        "IEMOCAP must remain ineligible until license and session split are verified",
    )

    privacy = _mapping(manifest.get("privacy_boundary"), "privacy_boundary")
    repository_forbidden = set(
        _sequence(privacy.get("repository_forbidden"), "privacy_boundary.repository_forbidden")
    )
    _require(
        "raw_or_redistributed_text_labels_audio_or_video" in repository_forbidden,
        "raw text, labels, audio, and video must be outside the public repository",
    )
    _require(
        "per_query_predictions_or_utilities" in repository_forbidden,
        "per-query predictions and utilities must be outside the public repository",
    )
    llm_boundary = _mapping(privacy.get("external_llm_api"), "privacy_boundary.external_llm_api")
    _require(
        llm_boundary.get("raw_or_row_level_restricted_dataset_content") == "forbidden",
        "restricted row-level data must not be sent to an external LLM API",
    )

    drift = _mapping(manifest.get("drift_policy"), "drift_policy")
    _require(
        drift.get("role_reassignment_after_freeze") == "forbidden",
        "role reassignment after freeze must be forbidden",
    )
    _require(
        drift.get("new_model_protocol_reuses_same_role_map") is True,
        "new model protocols must reuse the frozen role map",
    )
    _require(
        drift.get("manifest_change_requires_new_split_protocol_id") is True,
        "manifest changes must require a new split_protocol_id",
    )


def _validate_primary_operating_point(analysis: Mapping[str, Any]) -> None:
    operating_point = _mapping(
        analysis.get("primary_operating_point"), "primary_operating_point"
    )
    _require(
        operating_point.get("mode") == "fixed_target_history_coverage",
        "the primary operating point must use one fixed target coverage",
    )
    coverages = _sequence(
        operating_point.get("primary_history_coverages"),
        "primary_operating_point.primary_history_coverages",
    )
    _require(len(coverages) == 1, "exactly one primary history coverage is required")
    primary_coverage = _number(coverages[0], "primary history coverage")
    _require(
        primary_coverage == PRIMARY_HISTORY_COVERAGE,
        f"v1 primary history coverage is frozen at {PRIMARY_HISTORY_COVERAGE:.2f}",
    )
    _require(
        operating_point.get("target_was_fixed_before_calibration_unseal") is True,
        "primary coverage must be frozen before calibration unseal",
    )

    calibration_rule = _mapping(
        operating_point.get("calibration_rule"),
        "primary_operating_point.calibration_rule",
    )
    _require(
        calibration_rule.get("role") == "calibration",
        "threshold selection may use only the calibration role",
    )
    _require(
        calibration_rule.get("selected_threshold_is_hashed_before_holdout") is True,
        "calibrated threshold must be frozen and hashed before holdout",
    )

    secondary = _mapping(
        operating_point.get("secondary_coverage_curve"),
        "primary_operating_point.secondary_coverage_curve",
    )
    _require(
        secondary.get("purpose") == "descriptive_sensitivity_only"
        and secondary.get("may_determine_success") is False,
        "secondary coverages must be descriptive and may not determine success",
    )
    forbidden_success_rules = set(
        _sequence(
            operating_point.get("forbidden_success_rules"),
            "primary_operating_point.forbidden_success_rules",
        )
    )
    _require(
        {"at_least_one_coverage_passes", "best_posthoc_coverage_passes"}
        <= forbidden_success_rules,
        "'at least one coverage passes' and best-coverage success rules must be forbidden",
    )


def _validate_statistics(analysis: Mapping[str, Any]) -> None:
    runs = _mapping(analysis.get("independent_runs"), "independent_runs")
    seeds = list(_sequence(runs.get("seeds"), "independent_runs.seeds"))
    _require(
        len(seeds) == 5
        and len(set(seeds)) == 5
        and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds),
        "exactly five distinct integer seeds are required",
    )
    _require(runs.get("required_seed_count") == 5, "required_seed_count must be five")
    _require(
        runs.get("independence_unit") == "full_training_run",
        "each seed must be an independent full training run",
    )

    bootstrap = _mapping(
        analysis.get("hierarchical_bootstrap"), "hierarchical_bootstrap"
    )
    _require(
        isinstance(bootstrap.get("replicates"), int)
        and bootstrap.get("replicates") >= 2000,
        "hierarchical bootstrap requires at least 2000 replicates",
    )
    _require(bootstrap.get("paired") is True, "bootstrap contrasts must be paired")
    _require(
        bootstrap.get("bootstrap_design") == BOOTSTRAP_DESIGN,
        "bootstrap must cross five training seeds with one shared whole-cluster draw",
    )
    _require(
        list(
            _sequence(
                bootstrap.get("resampling_hierarchy"),
                "hierarchical_bootstrap.resampling_hierarchy",
            )
        )
        == [
            "dataset_specific_independent_cluster_shared_across_training_seeds",
            "training_seed_crossed_not_nested",
        ],
        "bootstrap must share the cluster draw across crossed training seeds",
    )
    _require(
        bootstrap.get("shared_cluster_draw_across_training_seeds") is True
        and bootstrap.get("candidate_and_reference_share_seed_and_cluster_draws")
        is True,
        "candidate/reference and all seeds must share each whole-cluster draw",
    )
    _require(
        bootstrap.get("keep_all_queries_within_resampled_cluster") is True
        and bootstrap.get("independent_query_row_bootstrap") is False,
        "queries within a cluster must stay together",
    )
    cluster_units = _mapping(
        bootstrap.get("cluster_units"), "hierarchical_bootstrap.cluster_units"
    )
    _require(
        {"MELD", "EmotionTalk", "IEMOCAP"} <= set(cluster_units),
        "bootstrap cluster units must be declared for every registered dataset",
    )

    randomization = _mapping(
        analysis.get("hypothesis_testing"), "hypothesis_testing"
    )
    _require(
        randomization.get("method") == "paired_whole_cluster_randomization",
        "Holm raw p-values must use paired whole-cluster randomization",
    )
    _require(
        randomization.get("monte_carlo_assignments") == RANDOMIZATION_REPLICATES
        and randomization.get("randomization_seed") == RANDOMIZATION_SEED
        and randomization.get("exact_enumeration_max_clusters") == 16,
        "randomization count, seed, and exact-enumeration boundary changed",
    )
    _require(
        randomization.get("cluster_swap_shared_across_training_seeds") is True
        and randomization.get("nonlinear_metrics_recomputed_per_assignment") is True,
        "randomization must swap whole clusters jointly across seeds and recompute metrics",
    )
    _require(
        randomization.get("bootstrap_tail_probability_may_be_used_as_p_value")
        is False,
        "an uncentered bootstrap tail probability must not be used as a p-value",
    )

    holm = _mapping(analysis.get("holm_family"), "holm_family")
    _require(holm.get("method") == "holm_bonferroni", "Holm correction is required")
    alpha = _number(holm.get("familywise_alpha"), "holm_family.familywise_alpha")
    _require(0.0 < alpha <= 0.05, "Holm familywise alpha must be in (0, 0.05]")
    hypotheses = list(_sequence(holm.get("hypotheses"), "holm_family.hypotheses"))
    hypothesis_ids = [
        str(_mapping(item, "Holm hypothesis").get("id", "")) for item in hypotheses
    ]
    _require(
        len(hypotheses) >= 2
        and len(set(hypothesis_ids)) == len(hypothesis_ids)
        and all(hypothesis_ids),
        "Holm family must contain unique, named hypotheses",
    )
    hypothesis_metrics = {
        str(_mapping(item, "Holm hypothesis").get("metric")) for item in hypotheses
    }
    _require(
        {"macro_f1", "mean_regret"} <= hypothesis_metrics,
        "Holm family must cover both primary classification and safety metrics",
    )
    hypothesis_by_id = {
        str(_mapping(item, "Holm hypothesis").get("id")): _mapping(
            item, "Holm hypothesis"
        )
        for item in hypotheses
    }
    _require(
        hypothesis_by_id["H1_primary_macro_f1"].get("contrast")
        == "carma_bidirectional_vs_frozen_strongest_admissible_baseline",
        "H1 must compare against the frozen strongest admissible baseline",
    )
    _require(
        hypothesis_by_id["H2_primary_mean_regret"].get("contrast")
        == "carma_bidirectional_full_vs_current_only",
        "H2 mean regret must retain current-only as the absolute safety anchor",
    )
    _require(
        list(
            _sequence(
                holm.get("method_success_requires_adjusted_rejection_of"),
                "holm_family.method_success_requires_adjusted_rejection_of",
            )
        )
        == ["H1_primary_macro_f1", "H2_primary_mean_regret"],
        "method success must require Holm-adjusted primary classification and safety rejections",
    )
    _require(
        holm.get("family_is_frozen_before_holdout_unseal") is True,
        "Holm family must be frozen before holdout unseal",
    )


def _validate_effect_gates(analysis: Mapping[str, Any]) -> None:
    metrics = _mapping(analysis.get("metrics"), "metrics")
    classification = _mapping(
        metrics.get("primary_classification"), "metrics.primary_classification"
    )
    _require(
        classification.get("name") == "macro_f1",
        "macro_f1 must be the primary classification metric",
    )
    safety = _mapping(metrics.get("primary_safety"), "metrics.primary_safety")
    _require(safety.get("name") == "mean_regret", "mean_regret must be primary safety")
    mandatory_secondary = set(
        _sequence(metrics.get("mandatory_secondary"), "metrics.mandatory_secondary")
    )
    _require("accuracy" in mandatory_secondary, "accuracy must be a mandatory metric")

    gates = _mapping(
        analysis.get("effect_and_safety_gates"), "effect_and_safety_gates"
    )
    macro_gain = _number(
        gates.get("minimum_macro_f1_gain_absolute"),
        "effect_and_safety_gates.minimum_macro_f1_gain_absolute",
    )
    _require(
        macro_gain >= MINIMUM_MACRO_F1_GAIN,
        f"macro-F1 gain gate must be at least {MINIMUM_MACRO_F1_GAIN:.3f}",
    )
    _require(
        _number(
            gates.get("absolute_mean_regret_vs_current_ci95_upper_must_not_exceed"),
            "effect_and_safety_gates.absolute_mean_regret_vs_current_ci95_upper_must_not_exceed",
        )
        <= 0.0,
        "absolute mean regret versus current-only must have a non-positive CI upper bound",
    )
    _require(
        _number(
            gates.get(
                "mean_regret_difference_vs_frozen_reference_ci95_upper_must_not_exceed"
            ),
            "effect_and_safety_gates.mean_regret_difference_vs_frozen_reference_ci95_upper_must_not_exceed",
        )
        <= 0.0,
        "mean regret versus the frozen reference must have a non-positive CI upper bound",
    )

    legacy_harm_reduction = _number(
        gates.get("minimum_relative_history_harm_rate_reduction"),
        "effect_and_safety_gates.minimum_relative_history_harm_rate_reduction",
    )
    _require(
        legacy_harm_reduction == 0.05,
        "the relative history-harm reduction gate is frozen at 0.05",
    )
    harm_gate = _mapping(
        gates.get("history_harm_rate_reduction"),
        "effect_and_safety_gates.history_harm_rate_reduction",
    )
    _require(
        set(harm_gate)
        == {
            "candidate",
            "reference",
            "reference_candidates",
            "reference_selection_rule",
            "minimum_relative_reduction",
            "zero_reference_harm_rate_action",
        },
        "history-harm reduction must use the complete frozen gate schema",
    )
    _require(
        harm_gate.get("candidate") == "carma_bidirectional_full"
        and harm_gate.get("reference")
        == "strongest_history_using_admissible_baseline_frozen_on_model_selection",
        "history-harm reduction must compare CARMA against the frozen strongest history-using baseline",
    )
    harm_reference_candidates = list(
        _sequence(
            harm_gate.get("reference_candidates"),
            "effect_and_safety_gates.history_harm_rate_reduction.reference_candidates",
        )
    )
    _require(
        harm_reference_candidates == list(HISTORY_HARM_REFERENCE_CANDIDATES)
        and "current_only" not in harm_reference_candidates,
        "history-harm reference candidates must be exactly the four admissible history-using baselines",
    )
    _require(
        harm_gate.get("reference_selection_rule") == REFERENCE_SELECTION_RULE,
        "history-harm baseline selection must use the frozen deterministic model-selection rule",
    )
    harm_reduction = _number(
        harm_gate.get("minimum_relative_reduction"),
        "effect_and_safety_gates.history_harm_rate_reduction.minimum_relative_reduction",
    )
    _require(
        harm_reduction == 0.05 and harm_reduction == legacy_harm_reduction,
        "history-harm reduction thresholds must agree and remain frozen at 0.05",
    )
    _require(
        harm_gate.get("zero_reference_harm_rate_action")
        == "fail_closed_not_estimable",
        "zero reference history-harm rate must fail closed as not estimable",
    )

    _require(
        gates.get("required_seed_successes_per_dataset") == 4,
        "exactly four of five seeds must succeed per dataset",
    )
    per_seed = _mapping(
        gates.get("per_seed_success"),
        "effect_and_safety_gates.per_seed_success",
    )
    _require(
        set(per_seed)
        == {
            "candidate",
            "reference",
            "seed_count",
            "required_successes",
            "same_seed_for_all_conditions",
            "success_requires_all",
            "thresholds",
        },
        "per-seed success must use the complete frozen predicate schema",
    )
    _require(
        per_seed.get("candidate") == "carma_bidirectional_full"
        and per_seed.get("reference")
        == "strongest_admissible_baseline_frozen_on_model_selection",
        "per-seed success must compare CARMA with the frozen strongest admissible baseline",
    )
    _require(
        per_seed.get("seed_count") == 5
        and per_seed.get("required_successes") == 4
        and per_seed.get("required_successes")
        == gates.get("required_seed_successes_per_dataset"),
        "per-seed success must require exactly four of five seeds to succeed",
    )
    _require(
        per_seed.get("same_seed_for_all_conditions") is True,
        "classification and regret conditions must hold on the same seed",
    )
    _require(
        list(
            _sequence(
                per_seed.get("success_requires_all"),
                "effect_and_safety_gates.per_seed_success.success_requires_all",
            )
        )
        == list(PER_SEED_SUCCESS_CONDITIONS),
        "each successful seed must satisfy both strict Macro-F1 gain and non-positive regret",
    )
    per_seed_thresholds = _mapping(
        per_seed.get("thresholds"),
        "effect_and_safety_gates.per_seed_success.thresholds",
    )
    _require(
        set(per_seed_thresholds)
        == {
            "macro_f1_difference_strictly_greater_than",
            "mean_regret_vs_current_must_not_exceed",
        },
        "per-seed thresholds must contain exactly the frozen classification and regret limits",
    )
    _require(
        _number(
            per_seed_thresholds.get("macro_f1_difference_strictly_greater_than"),
            "per_seed_success.thresholds.macro_f1_difference_strictly_greater_than",
        )
        == 0.0
        and _number(
            per_seed_thresholds.get("mean_regret_vs_current_must_not_exceed"),
            "per_seed_success.thresholds.mean_regret_vs_current_must_not_exceed",
        )
        == 0.0,
        "per-seed success requires Macro-F1 difference > 0 and mean regret versus current <= 0",
    )

    accuracy_gate = _mapping(
        gates.get("accuracy_no_harm"), "effect_and_safety_gates.accuracy_no_harm"
    )
    _require(
        accuracy_gate.get("gate_id") == "carma_confirmatory_accuracy_no_harm_v1",
        "accuracy no-harm gate id changed",
    )
    _require(
        list(
            _sequence(
                accuracy_gate.get("required_references"),
                "effect_and_safety_gates.accuracy_no_harm.required_references",
            )
        )
        == ["current_only", "strongest_admissible_baseline_frozen_on_model_selection"],
        "accuracy no-harm must cover current-only and the frozen strongest reference",
    )
    accuracy_contrasts = list(
        _sequence(
            accuracy_gate.get("contrasts"),
            "effect_and_safety_gates.accuracy_no_harm.contrasts",
        )
    )
    observed_accuracy_contrasts = {
        (
            str(_mapping(row, "accuracy contrast").get("id")),
            str(_mapping(row, "accuracy contrast").get("candidate")),
            str(_mapping(row, "accuracy contrast").get("reference")),
        )
        for row in accuracy_contrasts
    }
    _require(
        observed_accuracy_contrasts
        == {
            (
                "A1_accuracy_vs_current",
                "carma_bidirectional_full",
                "current_only",
            ),
            (
                "A2_accuracy_vs_frozen_reference",
                "carma_bidirectional_full",
                "strongest_admissible_baseline_frozen_on_model_selection",
            ),
        }
        and len(accuracy_contrasts) == 2,
        "accuracy no-harm contrasts must cover current and the frozen strongest reference exactly",
    )
    _require(
        _number(
            accuracy_gate.get("minimum_point_difference"),
            "accuracy_no_harm.minimum_point_difference",
        )
        == ACCURACY_NO_HARM_POINT_MINIMUM,
        "accuracy no-harm point difference is frozen at zero",
    )
    _require(
        _number(
            accuracy_gate.get("minimum_ci95_lower"),
            "accuracy_no_harm.minimum_ci95_lower",
        )
        == ACCURACY_NO_HARM_CI95_LOWER_MINIMUM,
        "accuracy no-harm CI lower bound is frozen at -0.005",
    )
    _require(
        accuracy_gate.get("interpretation")
        == "minus_0.005_is_a_noninferiority_no_harm_margin_not_evidence_of_improvement",
        "accuracy non-inferiority must not be described as improvement",
    )
    required_datasets = list(
        _sequence(
            analysis.get("required_confirmatory_datasets"),
            "required_confirmatory_datasets",
        )
    )
    _require(
        required_datasets == list(REQUIRED_DATASETS),
        "MELD and EmotionTalk must both be required confirmatory datasets",
    )
    _require(
        gates.get("required_datasets_passing") == len(required_datasets),
        "every required confirmatory dataset must pass",
    )
    _require(
        gates.get("success_quantifier")
        == "all_required_datasets_at_the_single_primary_operating_point",
        "success must require all datasets at the single primary operating point",
    )
    _require(
        gates.get("classification_and_safety_must_both_pass") is True,
        "classification and safety gates must both pass",
    )
    _require(
        gates.get("classification_safety_and_accuracy_no_harm_must_all_pass")
        is True,
        "classification, safety, and accuracy no-harm gates must all pass",
    )

    mde = _mapping(analysis.get("mde_and_power"), "mde_and_power")
    _require(mde.get("target_metric") == "macro_f1", "MDE must target macro_f1")
    detectable_gain = _number(
        mde.get("minimum_detectable_gain_absolute"),
        "mde_and_power.minimum_detectable_gain_absolute",
    )
    _require(
        0.0 < detectable_gain <= MINIMUM_MACRO_F1_GAIN,
        f"macro-F1 MDE must be no larger than {MINIMUM_MACRO_F1_GAIN:.3f}",
    )
    _require(
        _number(mde.get("minimum_power"), "mde_and_power.minimum_power") >= 0.8,
        "preflight power must be at least 0.80",
    )
    power_roles = set(
        _sequence(mde.get("power_assessment_roles"), "mde_and_power.power_assessment_roles")
    )
    _require(
        power_roles == {"base_and_utility_fit", "model_selection"},
        "power assessment may use only fit and model-selection roles",
    )
    _require(
        mde.get("calibration_holdout_or_test_rows_for_power") == "forbidden",
        "power assessment must not inspect calibration, holdout, or test rows",
    )


def _validate_sealing(analysis: Mapping[str, Any]) -> None:
    sealing = _mapping(
        analysis.get("sealing_and_stage_order"), "sealing_and_stage_order"
    )
    expected_true = (
        "calibration_sealed_before_model_and_analysis_freeze",
        "internal_holdout_sealed_before_calibration_artifact_freeze",
        "external_test_sealed_before_complete_bundle_freeze",
        "one_shot_outputs_are_write_once",
    )
    for field in expected_true:
        _require(sealing.get(field) is True, f"{field} must be true")
    _require(
        sealing.get("test_labels_may_change_model_threshold_or_claim_family") is False,
        "test labels must never change the model, threshold, or claim family",
    )
    stages = list(_sequence(sealing.get("ordered_stages"), "ordered_stages"))
    _require(
        stages.index("calibration_single_threshold_once")
        < stages.index("internal_holdout_one_shot")
        < stages.index("external_test_one_shot_after_authorized_unseal"),
        "calibration, holdout, and test must occur in that sealed order",
    )


def validate_confirmatory_analysis(
    analysis: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    """Validate the unique analysis path and confirmatory success definition."""

    validate_split_manifest(manifest)
    _require(
        analysis.get("analysis_id") == "carma_confirmatory_analysis_v1",
        "unexpected confirmatory analysis id",
    )
    _require(
        analysis.get("split_protocol_id") == manifest.get("split_protocol_id"),
        "analysis and manifest split_protocol_id values must match",
    )
    _require(
        analysis.get("split_manifest") == "carma_split_manifest_v1.json",
        "analysis must reference the frozen split manifest",
    )
    amendment = _mapping(
        analysis.get("pre_execution_amendment"), "pre_execution_amendment"
    )
    _require(
        amendment.get("results_visible_before_amendment") is False,
        "the inference amendment must precede all sealed-role results",
    )

    contrasts = list(
        _sequence(analysis.get("primary_contrasts"), "primary_contrasts")
    )
    _require(len(contrasts) == 1, "exactly one primary contrast is required")
    primary_contrast = _mapping(contrasts[0], "primary_contrasts[0]")
    _require(
        primary_contrast.get("contrast_id")
        == "carma_bidirectional_vs_frozen_strongest_admissible_baseline"
        and primary_contrast.get("reference")
        == "strongest_admissible_baseline_frozen_on_model_selection",
        "primary contrast must use the strongest admissible frozen baseline",
    )
    _require(
        primary_contrast.get("reference_is_frozen_before_calibration_unseal") is True,
        "primary reference must be frozen before calibration unseal",
    )
    _require(
        set(
            _sequence(
                primary_contrast.get("reference_candidates"),
                "primary contrast reference_candidates",
            )
        )
        == ADMISSIBLE_REFERENCE_CANDIDATES,
        "primary reference must consider current, all-history, recency, and both single directions",
    )

    _validate_primary_operating_point(analysis)
    _validate_effect_gates(analysis)
    _validate_statistics(analysis)
    _validate_sealing(analysis)

    failure = _mapping(analysis.get("failure_criteria"), "failure_criteria")
    failure_rules = set(
        _sequence(
            failure.get("method_success_is_false_if_any"),
            "failure_criteria.method_success_is_false_if_any",
        )
    )
    _require(
        "only_a_secondary_coverage_passes" in failure_rules,
        "a secondary-only coverage pass must explicitly count as method failure",
    )
    _require(
        {
            "absolute_mean_regret_vs_current_ci95_upper_above_zero_on_either_required_dataset",
            "mean_regret_difference_vs_frozen_reference_ci95_upper_above_zero_on_either_required_dataset",
            "accuracy_point_difference_below_zero_vs_current_or_frozen_reference_on_either_required_dataset",
            "accuracy_ci95_lower_below_minus_0.005_vs_current_or_frozen_reference_on_either_required_dataset",
        }
        <= failure_rules,
        "failure rules must include absolute safety and accuracy no-harm",
    )
    _require(
        failure.get("all_prespecified_results_must_be_reported") is True,
        "all prespecified results must be reported",
    )


def validate_contract_files(config_dir: Path) -> dict[str, Any]:
    """Validate both JSON files and return provenance without touching data."""

    manifest_path = config_dir / "carma_split_manifest_v1.json"
    analysis_path = config_dir / "carma_confirmatory_analysis_v1.json"
    manifest = load_json_contract(manifest_path)
    analysis = load_json_contract(analysis_path)
    validate_confirmatory_analysis(analysis, manifest)
    return {
        "status": "PASS",
        "split_protocol_id": SPLIT_PROTOCOL_ID,
        "primary_history_coverage": analysis["primary_operating_point"][
            "primary_history_coverages"
        ][0],
        "seed_count": len(analysis["independent_runs"]["seeds"]),
        "minimum_macro_f1_gain_absolute": analysis["effect_and_safety_gates"][
            "minimum_macro_f1_gain_absolute"
        ],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "analysis_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        "data_files_read": 0,
    }
