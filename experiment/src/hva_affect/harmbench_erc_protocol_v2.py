"""Outcome-free candidate protocol for HarmBench-ERC v2.

This module is deliberately separate from the published ``harmbench_erc_v1``
development contract.  It freezes a complete model/strategy/hypothesis roster
but grants no real-data training, selection-outcome, or official-test
authority.  Runtime callers may select only a registered strategy identifier;
they cannot supply alternative ranking, eligibility, or fallback controls.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping


class HarmBenchProtocolV2Error(ValueError):
    """Raised when the v2 candidate protocol or a strategy rule drifts."""


PROTOCOL_V2_SCHEMA = "harmbench_erc_protocol_candidate_v2"
PROTOCOL_V2_ID = "harmbench_erc_v2_candidate"
PROTOCOL_V2_STATUS = "S2_candidate_no_official_test_authority"
PROTOCOL_V2_CANONICAL_SHA256 = (
    "58630569e7cb518b3b04fc9029bd5c78c56e409fee6ae2f36bc0a90143fc4f9a"
)
STRATEGY_RULE_VERSION = "harmbench_erc_context_strategy_rules_v2"

EXPECTED_TRAINING_SEEDS = (17, 29, 43, 71, 101)
EXPECTED_MODEL_ORDER = (
    "hb_linear_pool_v1",
    "hb_deepsets_pool_v1",
    "hb_causal_gru_v1",
)
EXPECTED_ANCHOR_STRATEGY_ID = "independent_current_only"
EXPECTED_HISTORY_STRATEGY_ORDER = (
    "dialogue_all_past",
    "same_speaker_all_past",
    "recent_k3",
    "similarity_top3",
    "modality_balanced_top3",
)
EXPECTED_CONTEXT_ROSTER_ORDER = (
    EXPECTED_ANCHOR_STRATEGY_ID,
    *EXPECTED_HISTORY_STRATEGY_ORDER,
)
EXPECTED_POLICY_ARTIFACT_ORDER = (
    "learned_selector_v1",
    "coverage_matched_recency_v1",
)
EXPECTED_CONFIRMATORY_DATASETS = (
    "EmotionTalk_official_test",
    "MELD_official_test",
)
EXPECTED_SELECTION_DATASETS = ("EmotionTalk", "MELD")
EXPECTED_PRIMARY_METRIC_ORDER = ("Macro-F1", "mean-regret")

_TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    "created_date",
    "legacy_contract",
    "authorization",
    "confirmatory_datasets",
    "training_contract",
    "context_strategy_contract",
    "eligibility_contract",
    "primary_analysis_contract",
    "evaluator_contract",
    "policy_artifact_contract",
    "runtime_control_contract",
    "privacy_contract",
}
_STRATEGY_RULE_KEYS = {
    "strategy_id",
    "candidate_scope",
    "strict_past",
    "top_k",
    "ranking",
    "ranking_tie",
    "zero_vector",
    "modality_order",
    "duplicate_skip",
    "emission_order",
    "empty_fallback",
}
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


_EXPECTED_STRATEGY_RULES: tuple[dict[str, object], ...] = (
    {
        "strategy_id": "independent_current_only",
        "candidate_scope": "none",
        "strict_past": False,
        "top_k": 0,
        "ranking": "none",
        "ranking_tie": "not_applicable",
        "zero_vector": "not_applicable",
        "modality_order": [],
        "duplicate_skip": "not_applicable_empty_candidate_pool",
        "emission_order": "empty_tuple",
        "empty_fallback": "empty_tuple",
    },
    {
        "strategy_id": "dialogue_all_past",
        "candidate_scope": (
            "same_source_role_and_derived_partition_and_same_independent_group"
        ),
        "strict_past": True,
        "top_k": None,
        "ranking": "ascending_turn_id_then_protocol_row_id",
        "ranking_tie": "ascending_protocol_row_id",
        "zero_vector": "not_applicable",
        "modality_order": [],
        "duplicate_skip": "not_applicable_unique_protocol_row_pool",
        "emission_order": "ascending_turn_id_then_protocol_row_id",
        "empty_fallback": "empty_tuple",
    },
    {
        "strategy_id": "same_speaker_all_past",
        "candidate_scope": (
            "same_source_role_and_derived_partition_and_same_independent_group_"
            "and_exact_speaker_identity"
        ),
        "strict_past": True,
        "top_k": None,
        "ranking": (
            "ascending_turn_id_then_protocol_row_id_after_exact_speaker_filter"
        ),
        "ranking_tie": "ascending_protocol_row_id",
        "zero_vector": "not_applicable",
        "modality_order": [],
        "duplicate_skip": "not_applicable_unique_protocol_row_pool",
        "emission_order": "ascending_turn_id_then_protocol_row_id",
        "empty_fallback": "empty_tuple",
    },
    {
        "strategy_id": "recent_k3",
        "candidate_scope": (
            "same_source_role_and_derived_partition_and_same_independent_group"
        ),
        "strict_past": True,
        "top_k": 3,
        "ranking": "descending_turn_id",
        "ranking_tie": "ascending_protocol_row_id",
        "zero_vector": "not_applicable",
        "modality_order": [],
        "duplicate_skip": "not_applicable_unique_protocol_row_pool",
        "emission_order": "ascending_turn_id_then_protocol_row_id",
        "empty_fallback": "empty_tuple",
    },
    {
        "strategy_id": "similarity_top3",
        "candidate_scope": (
            "same_source_role_and_derived_partition_and_same_independent_group"
        ),
        "strict_past": True,
        "top_k": 3,
        "ranking": "descending_fusion_cosine_similarity",
        "ranking_tie": "ascending_protocol_row_id",
        "zero_vector": (
            "cosine_score_zero_if_query_or_candidate_l2_norm_is_zero"
        ),
        "modality_order": [],
        "duplicate_skip": "not_applicable_unique_protocol_row_pool",
        "emission_order": "ascending_turn_id_then_protocol_row_id",
        "empty_fallback": "empty_tuple",
    },
    {
        "strategy_id": "modality_balanced_top3",
        "candidate_scope": (
            "same_source_role_and_derived_partition_and_same_independent_group"
        ),
        "strict_past": True,
        "top_k": 3,
        "ranking": (
            "per_modality_descending_cosine_then_depth_first_round_robin"
        ),
        "ranking_tie": "ascending_protocol_row_id",
        "zero_vector": (
            "cosine_score_zero_if_query_or_candidate_l2_norm_is_zero"
        ),
        "modality_order": ["text", "audio", "video"],
        "duplicate_skip": (
            "skip_already_selected_protocol_row_keep_first_modality_depth_occurrence"
        ),
        "emission_order": "ascending_turn_id_then_protocol_row_id",
        "empty_fallback": "empty_tuple",
    },
)


def _reject_constant(value: str) -> None:
    raise HarmBenchProtocolV2Error(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HarmBenchProtocolV2Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def decode_protocol_v2_json(encoded: str) -> Mapping[str, object]:
    try:
        payload = json.loads(
            encoded,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except HarmBenchProtocolV2Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise HarmBenchProtocolV2Error(f"invalid protocol JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise HarmBenchProtocolV2Error("protocol root must be a JSON object")
    return payload


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise HarmBenchProtocolV2Error(f"{name} must be an object")
    return value


def _exact_keys(
    value: object,
    expected: set[str],
    *,
    name: str,
) -> Mapping[str, object]:
    mapping = _mapping(value, name=name)
    observed = {str(key) for key in mapping}
    if observed != expected:
        raise HarmBenchProtocolV2Error(
            f"{name} schema changed: missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )
    return mapping


def _validate_json_tree(value: object, *, path: str = "$") -> None:
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HarmBenchProtocolV2Error(f"non-finite value at {path}")
        return
    if isinstance(value, str):
        if _WINDOWS_ABSOLUTE_PATH.match(value) or value.startswith(("/", "file://")):
            raise HarmBenchProtocolV2Error(f"absolute/private path at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise HarmBenchProtocolV2Error(f"non-string key at {path}")
            _validate_json_tree(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_tree(child, path=f"{path}[{index}]")
        return
    raise HarmBenchProtocolV2Error(
        f"unsupported JSON type at {path}: {type(value).__name__}"
    )


def _plain_json_tree(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json_tree(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_tree(child) for child in value]
    return value


def _deep_freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze_json(child) for key, child in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze_json(child) for child in value)
    return value


def canonical_protocol_v2_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        _plain_json_tree(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_protocol_v2_bytes(value)).hexdigest()


def _expected_hypotheses() -> list[dict[str, str]]:
    hypotheses: list[dict[str, str]] = []
    for model_id in EXPECTED_MODEL_ORDER:
        for metric_id in EXPECTED_PRIMARY_METRIC_ORDER:
            if metric_id == "Macro-F1":
                contrast = (
                    "same_speaker_all_past_minus_matching_independent_current_only"
                )
            else:
                contrast = (
                    "same_speaker_all_past_NLL_minus_matching_"
                    "independent_current_only_NLL"
                )
            hypotheses.append(
                {
                    "hypothesis_id": f"H_{model_id}_{metric_id}",
                    "model_id": model_id,
                    "metric_id": metric_id,
                    "contrast": contrast,
                    "population": "E_dialogue",
                    "null": "equal_to_zero",
                    "alternative": "two_sided_not_equal_to_zero",
                }
            )
    return hypotheses


def strategy_rule_sha256(strategy_rule: Mapping[str, object]) -> str:
    """Return the canonical rule digest including the global rule version."""

    descriptor = {
        "strategy_rule_version": STRATEGY_RULE_VERSION,
        "strategy_contract": _plain_json_tree(strategy_rule),
    }
    return _canonical_sha256(descriptor)


_STRATEGY_RULE_BY_ID = MappingProxyType(
    {
        str(rule["strategy_id"]): _deep_freeze_json(rule)
        for rule in _EXPECTED_STRATEGY_RULES
    }
)


def get_context_strategy_contract(strategy_id: str) -> Mapping[str, object]:
    """Resolve one exact v2 rule; aliases and caller-authored rules are invalid."""

    if type(strategy_id) is not str or strategy_id not in _STRATEGY_RULE_BY_ID:
        raise HarmBenchProtocolV2Error(
            "strategy identifier is outside the exact v2 context roster"
        )
    rule = _STRATEGY_RULE_BY_ID[strategy_id]
    if not isinstance(rule, Mapping):  # Defensive against in-process corruption.
        raise HarmBenchProtocolV2Error("registered strategy rule is not an object")
    return rule


@dataclass(frozen=True)
class ProtocolV2Contract:
    payload: Mapping[str, object]
    canonical_sha256: str


def validate_protocol_v2(payload: Mapping[str, object]) -> ProtocolV2Contract:
    """Validate the exact, outcome-free v2 candidate without runtime overrides."""

    root = _exact_keys(payload, _TOP_LEVEL_KEYS, name="protocol_v2")
    _validate_json_tree(root)
    if (
        root["schema_version"] != PROTOCOL_V2_SCHEMA
        or root["protocol_id"] != PROTOCOL_V2_ID
        or root["status"] != PROTOCOL_V2_STATUS
        or root["created_date"] != "2026-08-09"
    ):
        raise HarmBenchProtocolV2Error("v2 protocol identity changed")

    legacy = _exact_keys(
        root["legacy_contract"],
        {
            "protocol_id",
            "status",
            "published_s0_artifacts_may_be_rewritten",
            "v2_can_retroactively_authorize_v1",
        },
        name="legacy_contract",
    )
    if dict(legacy) != {
        "protocol_id": "harmbench_erc_v1",
        "status": "legacy_unchanged",
        "published_s0_artifacts_may_be_rewritten": False,
        "v2_can_retroactively_authorize_v1": False,
    }:
        raise HarmBenchProtocolV2Error("legacy v1/S0 boundary changed")

    authorization = _exact_keys(
        root["authorization"],
        {
            "fit_training_authorized",
            "selection_outcomes_authorized",
            "official_test_features_authorized",
            "official_test_predictions_authorized",
            "official_test_labels_authorized",
            "official_test_evaluation_authorized",
            "prospective_sensitivity",
        },
        name="authorization",
    )
    if any(
        authorization[name] is not False
        for name in authorization
        if name != "prospective_sensitivity"
    ) or authorization["prospective_sensitivity"] != "pending":
        raise HarmBenchProtocolV2Error("v2 candidate authorization became permissive")
    if tuple(root["confirmatory_datasets"]) != EXPECTED_CONFIRMATORY_DATASETS:
        raise HarmBenchProtocolV2Error("confirmatory dataset roster changed")

    training = _exact_keys(
        root["training_contract"],
        {
            "training_seed_order",
            "fold_count",
            "model_order",
            "anchor_strategy_id",
            "outcome_selected_model_roster_permitted",
        },
        name="training_contract",
    )
    if (
        tuple(training["training_seed_order"]) != EXPECTED_TRAINING_SEEDS
        or type(training["fold_count"]) is not int
        or training["fold_count"] != 5
        or tuple(training["model_order"]) != EXPECTED_MODEL_ORDER
        or training["anchor_strategy_id"] != EXPECTED_ANCHOR_STRATEGY_ID
        or training["outcome_selected_model_roster_permitted"] is not False
    ):
        raise HarmBenchProtocolV2Error("exact model/seed/fold roster changed")

    context = _exact_keys(
        root["context_strategy_contract"],
        {
            "strategy_rule_version",
            "anchor_strategy_id",
            "history_strategy_order",
            "context_roster_order",
            "strategy_contracts",
            "learned_or_coverage_policy_ids_permitted",
        },
        name="context_strategy_contract",
    )
    if (
        context["strategy_rule_version"] != STRATEGY_RULE_VERSION
        or context["anchor_strategy_id"] != EXPECTED_ANCHOR_STRATEGY_ID
        or tuple(context["history_strategy_order"])
        != EXPECTED_HISTORY_STRATEGY_ORDER
        or tuple(context["context_roster_order"]) != EXPECTED_CONTEXT_ROSTER_ORDER
        or context["learned_or_coverage_policy_ids_permitted"] is not False
    ):
        raise HarmBenchProtocolV2Error("exact context strategy roster changed")
    strategy_contracts = context["strategy_contracts"]
    if not isinstance(strategy_contracts, list):
        raise HarmBenchProtocolV2Error("strategy_contracts must be an ordered array")
    for index, rule in enumerate(strategy_contracts):
        _exact_keys(rule, _STRATEGY_RULE_KEYS, name=f"strategy_contracts[{index}]")
    if _plain_json_tree(strategy_contracts) != _plain_json_tree(
        _EXPECTED_STRATEGY_RULES
    ):
        raise HarmBenchProtocolV2Error(
            "candidate scope, strict-past, top-k, ranking/tie, zero-vector, "
            "modality order, duplicate skip, emission order, or empty fallback changed"
        )
    strategy_ids = tuple(rule["strategy_id"] for rule in strategy_contracts)
    if len(strategy_ids) != len(set(strategy_ids)):
        raise HarmBenchProtocolV2Error("duplicate context strategy identifier")
    if any(policy_id in strategy_ids for policy_id in EXPECTED_POLICY_ARTIFACT_ORDER):
        raise HarmBenchProtocolV2Error("learned/coverage policy entered context roster")

    eligibility = _exact_keys(
        root["eligibility_contract"],
        {
            "common_eligibility",
            "strategy_context_nonempty",
            "speaker_conditional",
            "outcome_selected_eligibility_permitted",
        },
        name="eligibility_contract",
    )
    common = _mapping(eligibility["common_eligibility"], name="common_eligibility")
    strategy_nonempty = _mapping(
        eligibility["strategy_context_nonempty"],
        name="strategy_context_nonempty",
    )
    speaker = _mapping(eligibility["speaker_conditional"], name="speaker_conditional")
    if (
        common.get("eligibility_id") != "E_dialogue"
        or common.get("candidate_source_role") != "same_as_query"
        or common.get("candidate_partition")
        != "same_live_derived_partition_as_query"
        or common.get("candidate_independent_group")
        != "exactly_equal_to_query_independent_group"
        or common.get("candidate_time_rule")
        != "candidate.turn_id_strictly_less_than_query.turn_id"
        or common.get("speaker_restriction") != "none"
        or common.get("strategy_output_restriction") != "none"
        or common.get("outcomes_predictions_or_losses_used") is not False
        or strategy_nonempty.get("may_replace_E_dialogue") is not False
        or strategy_nonempty.get("outcomes_predictions_or_losses_used") is not False
        or speaker.get("eligibility_id") != "E_speaker"
        or speaker.get("may_replace_primary_E_dialogue") is not False
        or speaker.get("outcomes_predictions_or_losses_used") is not False
        or eligibility["outcome_selected_eligibility_permitted"] is not False
    ):
        raise HarmBenchProtocolV2Error(
            "outcome-free E_dialogue/strategy-nonempty/speaker-conditional contract changed"
        )

    primary = _exact_keys(
        root["primary_analysis_contract"],
        {
            "status",
            "primary_model_scope",
            "primary_contrast",
            "scope_control",
            "secondary_strategy_order",
            "final_primary_holm_family",
        },
        name="primary_analysis_contract",
    )
    primary_contrast = _mapping(primary["primary_contrast"], name="primary_contrast")
    scope_control = _mapping(primary["scope_control"], name="scope_control")
    if (
        primary["primary_model_scope"]
        != "all_three_predeclared_co_primary_no_winner"
        or
        primary_contrast.get("history_strategy_id") != "same_speaker_all_past"
        or primary_contrast.get("anchor_strategy_id")
        != EXPECTED_ANCHOR_STRATEGY_ID
        or primary_contrast.get("matching")
        != "exact_dataset_model_seed_and_query_after_exact_5_fold_probability_mean"
        or primary_contrast.get("population") != "E_dialogue"
        or scope_control.get("strategy_id") != "dialogue_all_past"
        or scope_control.get("role") != "scope_control_not_primary"
        or tuple(primary["secondary_strategy_order"])
        != EXPECTED_HISTORY_STRATEGY_ORDER[2:]
    ):
        raise HarmBenchProtocolV2Error("primary/scope-control/secondary roles changed")
    holm = _mapping(
        primary["final_primary_holm_family"],
        name="final_primary_holm_family",
    )
    hypotheses = holm.get("hypothesis_order")
    if (
        holm.get("correction") != "Holm_step_down_familywise_error_control"
        or type(holm.get("familywise_alpha")) is not float
        or holm.get("familywise_alpha") != 0.05
        or holm.get("any_of_winner_selection_permitted") is not False
        or not isinstance(hypotheses, list)
        or not hypotheses
        or hypotheses != _expected_hypotheses()
    ):
        raise HarmBenchProtocolV2Error(
            "exact nonempty three-model by two-metric Holm family changed"
        )

    evaluator = _exact_keys(
        root["evaluator_contract"],
        {
            "status",
            "calibration",
            "worst_stratum",
            "whole_cluster_paired_randomization",
            "whole_cluster_bootstrap",
            "selection_prediction_roster",
            "empty_context_probability_fallback",
            "selection_probability_evaluation_order",
            "irreversible_evaluation_state_machine",
            "tail_risk",
            "sign_severity",
            "harm_thresholds",
            "minimum_practical_effects",
            "no_harm_gate",
            "selection_result_status",
        },
        name="evaluator_contract",
    )
    if evaluator["status"] != (
        "S2_candidate_implementation_pending_no_test_authority"
    ):
        raise HarmBenchProtocolV2Error("evaluator status became permissive")
    calibration = _mapping(evaluator["calibration"], name="calibration")
    if dict(calibration) != {
        "metric_id": "ECE",
        "bin_count": 15,
        "binning": "for_i_0_to_13_[i/15,(i+1)/15)_and_i_14_[14/15,1]",
        "confidence": "top_label_maximum_class_probability",
        "top_label_tie": "argmax_first_class_in_frozen_class_order",
        "correctness": "top_label_equals_reference_label",
        "empty_bins": "ignored_with_zero_weight",
        "bin_weighting": (
            "nonempty_bin_query_count_divided_by_total_query_count_N"
        ),
        "estimator": (
            "sum_over_nonempty_bins_bin_count_over_N_times_absolute_bin_"
            "accuracy_minus_bin_mean_confidence"
        ),
        "population": "E_dialogue",
        "role": "secondary_metric",
    }:
        raise HarmBenchProtocolV2Error("fixed 15-bin top-label ECE changed")

    worst = _mapping(evaluator["worst_stratum"], name="worst_stratum")
    expected_strata = [
        {
            "stratum_id": "depth_1",
            "minimum_inclusive": 1,
            "maximum_inclusive": 1,
        },
        {
            "stratum_id": "depth_2_3",
            "minimum_inclusive": 2,
            "maximum_inclusive": 3,
        },
        {
            "stratum_id": "depth_4_7",
            "minimum_inclusive": 4,
            "maximum_inclusive": 7,
        },
        {
            "stratum_id": "depth_ge_8",
            "minimum_inclusive": 8,
            "maximum_inclusive": None,
        },
    ]
    if dict(worst) != {
        "axis_id": "E_dialogue_strict_past_depth",
        "axis_is_outcome_free": True,
        "depth_source_strategy_id": "dialogue_all_past",
        "depth_source_field": "context_count_for_the_same_model_family_and_query",
        "depth_definition": (
            "same_family_same_query_dialogue_all_past_context_count_before_any_"
            "speaker_filter"
        ),
        "required_seed_fold_consistency": (
            "all_5_training_seeds_times_5_folds_have_exactly_identical_depth_"
            "for_each_family_and_query"
        ),
        "inconsistency_action": (
            "fail_closed_protocol_error_before_any_outcome_evaluation"
        ),
        "ordered_strata": expected_strata,
        "minimum_independent_clusters_per_stratum": 2,
        "insufficient_stratum": (
            "not_estimable_never_drop_merge_or_select_a_replacement_stratum"
        ),
        "worst_definition": (
            "maximum_mean_regret_over_estimable_predeclared_strata"
        ),
        "post_outcome_axis_or_cutpoint_selection_permitted": False,
    }:
        raise HarmBenchProtocolV2Error(
            "outcome-free dialogue-depth worst-stratum contract changed"
        )

    randomization = _mapping(
        evaluator["whole_cluster_paired_randomization"],
        name="whole_cluster_paired_randomization",
    )
    if dict(randomization) != {
        "cluster_unit": "typed_dataset_id_and_independent_group_tuple",
        "swap_unit": (
            "all_queries_and_training_seeds_for_one_cluster_swap_history_and_"
            "anchor_together"
        ),
        "draws_shared_across_models_metrics_and_strategies": True,
        "test_side": "two_sided",
        "statistic": "absolute_equal_dataset_weight_paired_contrast",
        "exact_enumeration_if_cluster_count_at_most": 20,
        "monte_carlo_replicates_if_above_exact_threshold": 100000,
        "monte_carlo_seed": 20260811,
        "exact_p_value": (
            "count_statistics_at_least_observed_divided_by_two_to_the_cluster_"
            "count"
        ),
        "monte_carlo_p_value": (
            "one_plus_count_statistics_at_least_observed_divided_by_one_plus_"
            "replicates"
        ),
    }:
        raise HarmBenchProtocolV2Error(
            "whole-cluster paired randomization parameters changed"
        )

    bootstrap = _mapping(
        evaluator["whole_cluster_bootstrap"],
        name="whole_cluster_bootstrap",
    )
    if dict(bootstrap) != {
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20260810,
        "training_seed_resampling": (
            "sample_the_5_predeclared_training_seeds_with_replacement_to_size_5"
        ),
        "cluster_resampling": (
            "within_each_dataset_sample_whole_independent_group_clusters_with_"
            "replacement_to_that_datasets_original_cluster_count"
        ),
        "same_replicate_draws_shared_across_all_models_strategies_and_metrics": (
            True
        ),
        "paired_candidate_and_anchor_use_identical_draws": True,
        "joint_two_dataset_estimate": (
            "equal_weight_mean_of_the_two_dataset_point_estimates_and_equal_"
            "weight_mean_of_same_index_dataset_replicates"
        ),
        "confidence_interval": "percentile_[2.5,97.5]",
        "minimum_finite_replicate_fraction": 0.95,
        "per_dataset_cell_no_harm_CI_source": (
            "that_dataset_cells_same_origin_shared_replicate_draws"
        ),
        "caller_override_permitted": False,
        "post_outcome_method_change_permitted": False,
    }:
        raise HarmBenchProtocolV2Error(
            "whole-cluster bootstrap/CI parameters changed"
        )

    selection_roster = _mapping(
        evaluator["selection_prediction_roster"],
        name="selection_prediction_roster",
    )
    if dict(selection_roster) != {
        "dataset_order": list(EXPECTED_SELECTION_DATASETS),
        "model_order": list(EXPECTED_MODEL_ORDER),
        "strategy_order": list(EXPECTED_CONTEXT_ROSTER_ORDER),
        "artifact_identity_key": (
            "typed_dataset_id_model_id_strategy_id_tuple"
        ),
        "exact_total_loaded_artifact_count": 36,
        "exact_loaded_artifact_count_per_dataset": 18,
        "exact_loaded_artifact_count_per_dataset_model": 6,
        "exact_current_anchor_count_per_dataset_model": 1,
        "missing_duplicate_or_extra_artifacts_permitted": False,
        "evaluator_live_reverification_required": True,
        "within_dataset_alignment": (
            "all_18_artifacts_have_exactly_identical_query_group_class_seed_"
            "order_and_E_dialogue_vector"
        ),
        "history_nonempty_relation": (
            "every_history_strategy_context_nonempty_is_a_subset_of_E_dialogue"
        ),
        "current_anchor_context_semantics": (
            "for_all_5_seeds_times_5_folds_context_count_is_0_and_context_"
            "nonempty_is_false_while_carrying_the_true_E_dialogue_vector"
        ),
        "dialogue_all_nonempty_relation": (
            "dialogue_all_past_context_nonempty_exactly_equals_E_dialogue"
        ),
        "cross_model_dialogue_depth_relation": (
            "dialogue_all_past_context_count_is_exactly_identical_across_all_3_"
            "models_for_each_dataset_seed_fold_query"
        ),
    }:
        raise HarmBenchProtocolV2Error(
            "exact 36-artifact selection prediction roster/alignment changed"
        )

    probability_fallback = _mapping(
        evaluator["empty_context_probability_fallback"],
        name="empty_context_probability_fallback",
    )
    if dict(probability_fallback) != {
        "trigger": "history_strategy_context_nonempty_is_false",
        "source_strategy_id": EXPECTED_ANCHOR_STRATEGY_ID,
        "matching_key": (
            "exact_dataset_id_model_id_training_seed_fold_query_id"
        ),
        "operation": (
            "elementwise_copy_the_unique_matching_current_fold_anchor_"
            "probability_vector"
        ),
        "requires_identical_frozen_class_order": True,
        "history_checkpoint_raw_empty_context_probability_permitted": False,
        "missing_or_duplicate_anchor_action": (
            "fail_closed_before_any_outcome_evaluation"
        ),
        "post_fallback_regret": "canonical_exact_0_with_no_tolerance",
    }:
        raise HarmBenchProtocolV2Error(
            "empty-context probability fallback changed"
        )

    evaluation_order = _mapping(
        evaluator["selection_probability_evaluation_order"],
        name="selection_probability_evaluation_order",
    )
    if dict(evaluation_order) != {
        "step_1": (
            "verify_exact_36_artifact_roster_and_all_live_alignment_relations"
        ),
        "step_2": (
            "apply_empty_context_probability_fallback_at_exact_dataset_model_"
            "seed_fold_query"
        ),
        "step_3": (
            "arithmetic_probability_mean_over_exactly_5_folds_for_each_dataset_"
            "model_seed_strategy_query"
        ),
        "step_4": (
            "compute_NLL_regret_argmax_and_other_row_metrics_separately_for_"
            "each_seed_query_from_fold_mean_probabilities"
        ),
        "fold_level_metric_computation_permitted": False,
        "cross_seed_probability_averaging_before_row_metrics_permitted": False,
        "seed_metric_aggregation": (
            "only_after_per_seed_query_row_metrics_exist"
        ),
    }:
        raise HarmBenchProtocolV2Error(
            "fold fallback/probability/row-metric evaluation order changed"
        )

    state_machine = _mapping(
        evaluator["irreversible_evaluation_state_machine"],
        name="irreversible_evaluation_state_machine",
    )
    if dict(state_machine) != {
        "exact_state_order": [
            "analysis_frozen",
            "predictions_verified",
            "prelabel_bundle_write_once_fsync",
            "attempt_marker_write_once_fsync",
            "label_capability_created",
            "each_label_sidecar_single_handle_single_deserialization",
            "in_memory_row_metrics",
            "aggregate_only_publication",
            "terminal_exploratory",
        ],
        "attempt_marker_must_precede": [
            "label_path_resolve",
            "label_path_stat",
            "label_path_hash",
            "label_path_open",
        ],
        "state_skip_repeat_or_reverse_permitted": False,
        "any_crash_or_partial_output_is_terminal": True,
        "crash_or_partial_output_terminal_state": "terminal_exploratory",
        "silent_rerun_permitted": False,
    }:
        raise HarmBenchProtocolV2Error(
            "irreversible one-shot evaluator state sequence changed"
        )

    tail = _mapping(evaluator["tail_risk"], name="tail_risk")
    if dict(tail) != {
        "metric_id": "CVaR-regret",
        "alpha": 0.9,
        "regret_unit": (
            "natural_log_nats_history_NLL_minus_matching_anchor_NLL"
        ),
        "definition": (
            "largest_empirical_10_percent_with_fractional_boundary_mass_never_"
            "values_greater_than_or_equal_to_quantile_averaging"
        ),
        "population": "E_dialogue",
        "role": "secondary_metric",
    }:
        raise HarmBenchProtocolV2Error("fractional-boundary CVaR contract changed")
    sign_severity = _mapping(evaluator["sign_severity"], name="sign_severity")
    if dict(sign_severity) != {
        "metric_id": "sign_x_severity_regret_bins",
        "regret_definition": (
            "natural_log_nats_history_NLL_minus_matching_anchor_NLL"
        ),
        "population": "E_dialogue",
        "zero_comparison": (
            "exact_canonical_regret_float_equality_with_no_tolerance_including_"
            "fallback"
        ),
        "ordered_bins": [
            {
                "bin_id": "substantial_benefit",
                "predicate": "regret < -0.05",
            },
            {
                "bin_id": "small_benefit",
                "predicate": "-0.05 <= regret < 0",
            },
            {
                "bin_id": "exact_zero_including_fallback",
                "predicate": "regret == 0",
            },
            {
                "bin_id": "small_harm",
                "predicate": "0 < regret <= 0.05",
            },
            {
                "bin_id": "substantial_harm",
                "predicate": "regret > 0.05",
            },
        ],
        "aggregate_output": (
            "query_count_and_fraction_for_every_bin_no_bin_selection"
        ),
        "role": "secondary_metric",
    }:
        raise HarmBenchProtocolV2Error("exact sign-by-severity regret bins changed")
    thresholds = _mapping(evaluator["harm_thresholds"], name="harm_thresholds")
    if (
        type(thresholds.get("primary_harm_threshold_nats")) is not float
        or thresholds.get("primary_harm_threshold_nats") != 0.0
        or type(thresholds.get("practical_harm_sensitivity_threshold_nats"))
        is not float
        or thresholds.get("practical_harm_sensitivity_threshold_nats") != 0.05
        or thresholds.get("harm_event_operator")
        != "regret_strictly_greater_than_threshold"
        or len(thresholds) != 3
    ):
        raise HarmBenchProtocolV2Error("harm thresholds changed")
    minimum_effects = _mapping(
        evaluator["minimum_practical_effects"],
        name="minimum_practical_effects",
    )
    if dict(minimum_effects) != {
        "Macro-F1_absolute_difference": 0.005,
        "mean-regret_absolute_nats": 0.01,
        "substantive_claim_gate": (
            "Holm_rejected_and_absolute_point_estimate_at_least_the_metric_"
            "specific_minimum"
        ),
    }:
        raise HarmBenchProtocolV2Error("minimum practical effects changed")
    no_harm = _mapping(evaluator["no_harm_gate"], name="no_harm_gate")
    if dict(no_harm) != {
        "mean_regret_two_sided_95_percent_CI_upper_at_most_nats": 0.01,
        "Macro-F1_two_sided_95_percent_CI_lower_at_least": -0.005,
        "scope": "every_predeclared_model_by_dataset_cell",
        "nonestimable_cell": "gate_failure",
        "any_of_cell_selection_permitted": False,
    }:
        raise HarmBenchProtocolV2Error("no-harm gate changed")
    selection_status = _mapping(
        evaluator["selection_result_status"],
        name="selection_result_status",
    )
    if dict(selection_status) != {
        "observed_selection_outcome_results": "permanently_exploratory",
        "may_enter_final_primary_family": False,
        "may_change_model_or_strategy_rosters": False,
        "may_change_thresholds_eligibility_or_strata": False,
        "may_authorize_selection_rerun": False,
    }:
        raise HarmBenchProtocolV2Error(
            "selection outcome results are no longer permanently exploratory"
        )

    policy = _exact_keys(
        root["policy_artifact_contract"],
        {
            "status",
            "policy_artifact_order",
            "artifacts",
            "context_roster_membership_permitted",
            "may_redefine_E_dialogue",
        },
        name="policy_artifact_contract",
    )
    if (
        tuple(policy["policy_artifact_order"]) != EXPECTED_POLICY_ARTIFACT_ORDER
        or policy["context_roster_membership_permitted"] is not False
        or policy["may_redefine_E_dialogue"] is not False
    ):
        raise HarmBenchProtocolV2Error("separate policy-artifact boundary changed")
    artifacts = policy["artifacts"]
    if not isinstance(artifacts, list) or tuple(
        artifact.get("policy_artifact_id")
        for artifact in artifacts
        if isinstance(artifact, Mapping)
    ) != EXPECTED_POLICY_ARTIFACT_ORDER:
        raise HarmBenchProtocolV2Error("policy artifact order or identifiers changed")
    for artifact in artifacts:
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("context_roster_member") is not False
            or artifact.get("requires_separate_typed_receipt") is not True
            or artifact.get("may_define_confirmatory_eligibility") is not False
        ):
            raise HarmBenchProtocolV2Error("policy artifact entered context/eligibility")

    runtime = _mapping(root["runtime_control_contract"], name="runtime_control_contract")
    if not runtime or any(value is not False for value in runtime.values()):
        raise HarmBenchProtocolV2Error("runtime free controls became permitted")
    privacy = _mapping(root["privacy_contract"], name="privacy_contract")
    if (
        privacy.get("restricted_real_data_to_external_GPT_or_API_permitted")
        is not False
        or privacy.get("real_data_or_labels_read_by_protocol_validation") is not False
        or privacy.get("public_artifacts") != "aggregate_only"
        or privacy.get("synthetic_fixtures_only_before_data_authorization") is not True
    ):
        raise HarmBenchProtocolV2Error("privacy/outcome-free boundary changed")

    canonical_sha256 = _canonical_sha256(root)
    if canonical_sha256 != PROTOCOL_V2_CANONICAL_SHA256:
        raise HarmBenchProtocolV2Error("canonical v2 protocol SHA-256 changed")
    return ProtocolV2Contract(
        payload=_deep_freeze_json(root),
        canonical_sha256=canonical_sha256,
    )


def load_protocol_v2(path: str | Path) -> ProtocolV2Contract:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise HarmBenchProtocolV2Error("protocol path must be a plain existing file")
    payload = decode_protocol_v2_json(source.read_text(encoding="utf-8"))
    return validate_protocol_v2(payload)


__all__ = [
    "EXPECTED_ANCHOR_STRATEGY_ID",
    "EXPECTED_CONFIRMATORY_DATASETS",
    "EXPECTED_CONTEXT_ROSTER_ORDER",
    "EXPECTED_HISTORY_STRATEGY_ORDER",
    "EXPECTED_MODEL_ORDER",
    "EXPECTED_POLICY_ARTIFACT_ORDER",
    "EXPECTED_PRIMARY_METRIC_ORDER",
    "EXPECTED_SELECTION_DATASETS",
    "EXPECTED_TRAINING_SEEDS",
    "HarmBenchProtocolV2Error",
    "PROTOCOL_V2_CANONICAL_SHA256",
    "PROTOCOL_V2_ID",
    "PROTOCOL_V2_SCHEMA",
    "PROTOCOL_V2_STATUS",
    "ProtocolV2Contract",
    "STRATEGY_RULE_VERSION",
    "canonical_protocol_v2_bytes",
    "decode_protocol_v2_json",
    "get_context_strategy_contract",
    "load_protocol_v2",
    "strategy_rule_sha256",
    "validate_protocol_v2",
]
