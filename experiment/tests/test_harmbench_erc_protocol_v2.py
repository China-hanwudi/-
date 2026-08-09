from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.harmbench_erc_protocol_v2 import (  # noqa: E402
    EXPECTED_ANCHOR_STRATEGY_ID,
    EXPECTED_CONTEXT_ROSTER_ORDER,
    EXPECTED_HISTORY_STRATEGY_ORDER,
    EXPECTED_MODEL_ORDER,
    EXPECTED_POLICY_ARTIFACT_ORDER,
    EXPECTED_SELECTION_DATASETS,
    HarmBenchProtocolV2Error,
    PROTOCOL_V2_CANONICAL_SHA256,
    STRATEGY_RULE_VERSION,
    canonical_protocol_v2_bytes,
    decode_protocol_v2_json,
    get_context_strategy_contract,
    load_protocol_v2,
    strategy_rule_sha256,
    validate_protocol_v2,
)


CONFIG = ROOT / "configs" / "harmbench_erc_v2_candidate.json"


def payload() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_repository_v2_candidate_is_exact_hash_stable_and_non_authorizing() -> None:
    first = load_protocol_v2(CONFIG)
    second = load_protocol_v2(CONFIG)
    assert first.canonical_sha256 == PROTOCOL_V2_CANONICAL_SHA256
    assert first.canonical_sha256 == second.canonical_sha256
    assert hashlib.sha256(canonical_protocol_v2_bytes(first.payload)).hexdigest() == (
        PROTOCOL_V2_CANONICAL_SHA256
    )
    assert first.payload["legacy_contract"]["status"] == "legacy_unchanged"
    assert first.payload["authorization"]["prospective_sensitivity"] == "pending"
    assert not any(
        value
        for key, value in first.payload["authorization"].items()
        if key != "prospective_sensitivity"
    )
    with pytest.raises(TypeError):
        first.payload["status"] = "authorized"  # type: ignore[index]


def test_exact_model_anchor_history_and_policy_rosters_are_disjoint() -> None:
    contract = load_protocol_v2(CONFIG).payload
    training = contract["training_contract"]
    context = contract["context_strategy_contract"]
    policy = contract["policy_artifact_contract"]
    assert tuple(training["model_order"]) == EXPECTED_MODEL_ORDER
    assert training["anchor_strategy_id"] == EXPECTED_ANCHOR_STRATEGY_ID
    assert tuple(context["history_strategy_order"]) == EXPECTED_HISTORY_STRATEGY_ORDER
    assert tuple(context["context_roster_order"]) == EXPECTED_CONTEXT_ROSTER_ORDER
    assert tuple(policy["policy_artifact_order"]) == EXPECTED_POLICY_ARTIFACT_ORDER
    assert set(EXPECTED_POLICY_ARTIFACT_ORDER).isdisjoint(EXPECTED_CONTEXT_ROSTER_ORDER)
    assert policy["context_roster_membership_permitted"] is False


def test_each_strategy_rule_is_exact_and_has_a_version_bound_digest() -> None:
    contract = load_protocol_v2(CONFIG).payload
    observed = contract["context_strategy_contract"]["strategy_contracts"]
    assert tuple(row["strategy_id"] for row in observed) == EXPECTED_CONTEXT_ROSTER_ORDER
    required = {
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
    for strategy_id in EXPECTED_CONTEXT_ROSTER_ORDER:
        rule = get_context_strategy_contract(strategy_id)
        assert required.issubset(rule)
        assert len(strategy_rule_sha256(rule)) == 64
    assert STRATEGY_RULE_VERSION == "harmbench_erc_context_strategy_rules_v2"
    with pytest.raises(HarmBenchProtocolV2Error, match="outside the exact"):
        get_context_strategy_contract("all_history")


def test_E_dialogue_strategy_nonempty_and_speaker_conditional_are_separate() -> None:
    eligibility = load_protocol_v2(CONFIG).payload["eligibility_contract"]
    assert eligibility["common_eligibility"]["eligibility_id"] == "E_dialogue"
    assert eligibility["common_eligibility"]["speaker_restriction"] == "none"
    assert eligibility["strategy_context_nonempty"]["may_replace_E_dialogue"] is False
    assert eligibility["speaker_conditional"]["eligibility_id"] == "E_speaker"
    assert (
        eligibility["speaker_conditional"]["may_replace_primary_E_dialogue"]
        is False
    )
    assert eligibility["outcome_selected_eligibility_permitted"] is False


def test_primary_is_fixed_same_speaker_contrast_and_six_hypothesis_holm_family() -> None:
    primary = load_protocol_v2(CONFIG).payload["primary_analysis_contract"]
    assert primary["primary_model_scope"] == (
        "all_three_predeclared_co_primary_no_winner"
    )
    assert primary["primary_contrast"] == {
        "history_strategy_id": "same_speaker_all_past",
        "anchor_strategy_id": "independent_current_only",
        "matching": (
            "exact_dataset_model_seed_and_query_after_exact_5_fold_"
            "probability_mean"
        ),
        "population": "E_dialogue",
    }
    assert primary["scope_control"] == {
        "strategy_id": "dialogue_all_past",
        "role": "scope_control_not_primary",
    }
    family = primary["final_primary_holm_family"]
    hypotheses = tuple(family["hypothesis_order"])
    assert len(hypotheses) == 6
    assert tuple((row["model_id"], row["metric_id"]) for row in hypotheses) == tuple(
        (model_id, metric_id)
        for model_id in EXPECTED_MODEL_ORDER
        for metric_id in ("Macro-F1", "mean-regret")
    )
    assert family["correction"] == "Holm_step_down_familywise_error_control"
    assert family["any_of_winner_selection_permitted"] is False


def test_evaluator_freezes_calibration_strata_randomization_tail_and_claim_gates() -> None:
    evaluator = load_protocol_v2(CONFIG).payload["evaluator_contract"]
    calibration = evaluator["calibration"]
    assert calibration["metric_id"] == "ECE"
    assert calibration["bin_count"] == 15
    assert calibration["binning"] == (
        "for_i_0_to_13_[i/15,(i+1)/15)_and_i_14_[14/15,1]"
    )
    assert calibration["confidence"] == "top_label_maximum_class_probability"
    assert calibration["top_label_tie"] == (
        "argmax_first_class_in_frozen_class_order"
    )
    assert calibration["empty_bins"] == "ignored_with_zero_weight"
    assert calibration["bin_weighting"] == (
        "nonempty_bin_query_count_divided_by_total_query_count_N"
    )

    worst = evaluator["worst_stratum"]
    assert worst["axis_is_outcome_free"] is True
    assert worst["depth_source_strategy_id"] == "dialogue_all_past"
    assert worst["depth_source_field"] == (
        "context_count_for_the_same_model_family_and_query"
    )
    assert worst["required_seed_fold_consistency"].startswith(
        "all_5_training_seeds_times_5_folds"
    )
    assert worst["inconsistency_action"].startswith("fail_closed_protocol_error")
    assert tuple(row["stratum_id"] for row in worst["ordered_strata"]) == (
        "depth_1",
        "depth_2_3",
        "depth_4_7",
        "depth_ge_8",
    )
    assert worst["minimum_independent_clusters_per_stratum"] == 2
    assert worst["insufficient_stratum"].startswith("not_estimable")
    assert worst["post_outcome_axis_or_cutpoint_selection_permitted"] is False

    randomization = evaluator["whole_cluster_paired_randomization"]
    assert randomization["exact_enumeration_if_cluster_count_at_most"] == 20
    assert randomization["monte_carlo_replicates_if_above_exact_threshold"] == 100000
    assert randomization["monte_carlo_seed"] == 20260811
    assert randomization["draws_shared_across_models_metrics_and_strategies"] is True

    assert evaluator["tail_risk"]["alpha"] == 0.9
    assert "fractional_boundary_mass" in evaluator["tail_risk"]["definition"]
    sign_severity = evaluator["sign_severity"]
    assert tuple(row["predicate"] for row in sign_severity["ordered_bins"]) == (
        "regret < -0.05",
        "-0.05 <= regret < 0",
        "regret == 0",
        "0 < regret <= 0.05",
        "regret > 0.05",
    )
    assert sign_severity["ordered_bins"][2]["bin_id"] == (
        "exact_zero_including_fallback"
    )
    assert "no_tolerance_including_fallback" in sign_severity["zero_comparison"]
    assert evaluator["harm_thresholds"]["primary_harm_threshold_nats"] == 0.0
    assert (
        evaluator["harm_thresholds"]["practical_harm_sensitivity_threshold_nats"]
        == 0.05
    )
    assert evaluator["minimum_practical_effects"] == {
        "Macro-F1_absolute_difference": 0.005,
        "mean-regret_absolute_nats": 0.01,
        "substantive_claim_gate": (
            "Holm_rejected_and_absolute_point_estimate_at_least_the_metric_"
            "specific_minimum"
        ),
    }
    assert evaluator["no_harm_gate"]["any_of_cell_selection_permitted"] is False
    assert (
        evaluator["selection_result_status"]["observed_selection_outcome_results"]
        == "permanently_exploratory"
    )


def test_evaluator_freezes_bootstrap_roster_fallback_order_and_irreversible_states() -> None:
    evaluator = load_protocol_v2(CONFIG).payload["evaluator_contract"]
    bootstrap = evaluator["whole_cluster_bootstrap"]
    assert bootstrap["bootstrap_replicates"] == 10000
    assert bootstrap["bootstrap_seed"] == 20260810
    assert "with_replacement_to_size_5" in bootstrap["training_seed_resampling"]
    assert "whole_independent_group_clusters" in bootstrap["cluster_resampling"]
    assert (
        bootstrap[
            "same_replicate_draws_shared_across_all_models_strategies_and_metrics"
        ]
        is True
    )
    assert bootstrap["paired_candidate_and_anchor_use_identical_draws"] is True
    assert bootstrap["confidence_interval"] == "percentile_[2.5,97.5]"
    assert bootstrap["minimum_finite_replicate_fraction"] == 0.95
    assert bootstrap["caller_override_permitted"] is False
    assert bootstrap["post_outcome_method_change_permitted"] is False

    roster = evaluator["selection_prediction_roster"]
    assert tuple(roster["dataset_order"]) == EXPECTED_SELECTION_DATASETS
    assert tuple(roster["dataset_order"]) == ("EmotionTalk", "MELD")
    assert all("official_test" not in value for value in roster["dataset_order"])
    assert roster["exact_total_loaded_artifact_count"] == 36
    assert roster["exact_loaded_artifact_count_per_dataset"] == 18
    assert roster["exact_loaded_artifact_count_per_dataset_model"] == 6
    assert roster["exact_current_anchor_count_per_dataset_model"] == 1
    assert roster["missing_duplicate_or_extra_artifacts_permitted"] is False
    assert roster["evaluator_live_reverification_required"] is True
    assert "context_nonempty_is_a_subset_of_E_dialogue" in (
        roster["history_nonempty_relation"]
    )
    assert "context_count_is_0" in roster["current_anchor_context_semantics"]
    assert "exactly_equals_E_dialogue" in roster["dialogue_all_nonempty_relation"]
    assert "identical_across_all_3_models" in (
        roster["cross_model_dialogue_depth_relation"]
    )

    fallback = evaluator["empty_context_probability_fallback"]
    assert fallback["matching_key"] == (
        "exact_dataset_id_model_id_training_seed_fold_query_id"
    )
    assert fallback["source_strategy_id"] == "independent_current_only"
    assert fallback["history_checkpoint_raw_empty_context_probability_permitted"] is False
    assert fallback["post_fallback_regret"] == "canonical_exact_0_with_no_tolerance"

    order = evaluator["selection_probability_evaluation_order"]
    assert order["step_2"].endswith("dataset_model_seed_fold_query")
    assert "exactly_5_folds" in order["step_3"]
    assert "separately_for_each_seed_query" in order["step_4"]
    assert order["fold_level_metric_computation_permitted"] is False
    assert order["cross_seed_probability_averaging_before_row_metrics_permitted"] is False

    state = evaluator["irreversible_evaluation_state_machine"]
    assert tuple(state["exact_state_order"]) == (
        "analysis_frozen",
        "predictions_verified",
        "prelabel_bundle_write_once_fsync",
        "attempt_marker_write_once_fsync",
        "label_capability_created",
        "each_label_sidecar_single_handle_single_deserialization",
        "in_memory_row_metrics",
        "aggregate_only_publication",
        "terminal_exploratory",
    )
    assert tuple(state["attempt_marker_must_precede"]) == (
        "label_path_resolve",
        "label_path_stat",
        "label_path_hash",
        "label_path_open",
    )
    assert state["any_crash_or_partial_output_is_terminal"] is True
    assert state["silent_rerun_permitted"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["training_contract"]["model_order"].__setitem__(
            0, "linear"
        ),
        lambda value: value["context_strategy_contract"][
            "history_strategy_order"
        ].__setitem__(0, "all_history"),
        lambda value: value["context_strategy_contract"][
            "history_strategy_order"
        ].reverse(),
        lambda value: value["context_strategy_contract"][
            "context_roster_order"
        ].append("similarity_top3"),
        lambda value: value["context_strategy_contract"][
            "strategy_contracts"
        ].pop(),
        lambda value: value["context_strategy_contract"][
            "strategy_contracts"
        ][1].__setitem__("strict_past", False),
        lambda value: value["context_strategy_contract"][
            "strategy_contracts"
        ][1].__setitem__("candidate_scope", "all_dialogues"),
        lambda value: value["context_strategy_contract"][
            "strategy_contracts"
        ][3].__setitem__("top_k", 4),
        lambda value: value["context_strategy_contract"][
            "strategy_contracts"
        ][4].__setitem__("ranking_tie", "physical_row_order"),
        lambda value: value["context_strategy_contract"][
            "strategy_contracts"
        ][5].__setitem__("empty_fallback", "current_only_vector"),
        lambda value: value["eligibility_contract"].__setitem__(
            "outcome_selected_eligibility_permitted", True
        ),
        lambda value: value["eligibility_contract"][
            "common_eligibility"
        ].__setitem__("candidate_independent_group", "any_group"),
        lambda value: value["primary_analysis_contract"][
            "final_primary_holm_family"
        ].__setitem__("hypothesis_order", []),
        lambda value: value["primary_analysis_contract"][
            "final_primary_holm_family"
        ].__setitem__("any_of_winner_selection_permitted", True),
        lambda value: value["primary_analysis_contract"].__setitem__(
            "primary_model_scope", "select_best_model_after_results"
        ),
        lambda value: value["context_strategy_contract"][
            "context_roster_order"
        ].append("learned_selector_v1"),
        lambda value: value["runtime_control_contract"].__setitem__(
            "caller_supplied_top_k_permitted", True
        ),
        lambda value: value["authorization"].__setitem__(
            "official_test_labels_authorized", True
        ),
        lambda value: value["evaluator_contract"]["calibration"].__setitem__(
            "bin_count", 10
        ),
        lambda value: value["evaluator_contract"]["calibration"].__setitem__(
            "top_label_tie", "random"
        ),
        lambda value: value["evaluator_contract"]["worst_stratum"][
            "ordered_strata"
        ].pop(),
        lambda value: value["evaluator_contract"]["worst_stratum"].__setitem__(
            "minimum_independent_clusters_per_stratum", 1
        ),
        lambda value: value["evaluator_contract"]["worst_stratum"].__setitem__(
            "depth_source_strategy_id", "same_speaker_all_past"
        ),
        lambda value: value["evaluator_contract"]["worst_stratum"].__setitem__(
            "required_seed_fold_consistency", "majority_vote"
        ),
        lambda value: value["evaluator_contract"][
            "whole_cluster_paired_randomization"
        ].__setitem__("monte_carlo_seed", 1),
        lambda value: value["evaluator_contract"][
            "whole_cluster_bootstrap"
        ].__setitem__("bootstrap_replicates", 9999),
        lambda value: value["evaluator_contract"][
            "whole_cluster_bootstrap"
        ].__setitem__("bootstrap_seed", 1),
        lambda value: value["evaluator_contract"][
            "whole_cluster_bootstrap"
        ].__setitem__("paired_candidate_and_anchor_use_identical_draws", False),
        lambda value: value["evaluator_contract"][
            "whole_cluster_bootstrap"
        ].__setitem__("minimum_finite_replicate_fraction", 0.9),
        lambda value: value["evaluator_contract"][
            "whole_cluster_bootstrap"
        ].__setitem__("caller_override_permitted", True),
        lambda value: value["evaluator_contract"][
            "selection_prediction_roster"
        ].__setitem__("exact_total_loaded_artifact_count", 35),
        lambda value: value["evaluator_contract"][
            "selection_prediction_roster"
        ]["dataset_order"].__setitem__(0, "EmotionTalk_official_test"),
        lambda value: value["evaluator_contract"][
            "selection_prediction_roster"
        ].__setitem__("missing_duplicate_or_extra_artifacts_permitted", True),
        lambda value: value["evaluator_contract"][
            "selection_prediction_roster"
        ].__setitem__("dialogue_all_nonempty_relation", "subset"),
        lambda value: value["evaluator_contract"][
            "empty_context_probability_fallback"
        ].__setitem__("matching_key", "exact_dataset_model_seed_query"),
        lambda value: value["evaluator_contract"][
            "empty_context_probability_fallback"
        ].__setitem__(
            "history_checkpoint_raw_empty_context_probability_permitted", True
        ),
        lambda value: value["evaluator_contract"][
            "empty_context_probability_fallback"
        ].__setitem__("post_fallback_regret", "approximately_zero"),
        lambda value: value["evaluator_contract"][
            "selection_probability_evaluation_order"
        ].__setitem__("fold_level_metric_computation_permitted", True),
        lambda value: value["evaluator_contract"][
            "selection_probability_evaluation_order"
        ].__setitem__(
            "cross_seed_probability_averaging_before_row_metrics_permitted", True
        ),
        lambda value: value["evaluator_contract"][
            "irreversible_evaluation_state_machine"
        ]["exact_state_order"].reverse(),
        lambda value: value["evaluator_contract"][
            "irreversible_evaluation_state_machine"
        ]["attempt_marker_must_precede"].pop(0),
        lambda value: value["evaluator_contract"][
            "irreversible_evaluation_state_machine"
        ].__setitem__("silent_rerun_permitted", True),
        lambda value: value["evaluator_contract"]["tail_risk"].__setitem__(
            "alpha", 0.95
        ),
        lambda value: value["evaluator_contract"]["sign_severity"][
            "ordered_bins"
        ][2].__setitem__("predicate", "abs(regret) < 1e-8"),
        lambda value: value["evaluator_contract"]["harm_thresholds"].__setitem__(
            "primary_harm_threshold_nats", 0.01
        ),
        lambda value: value["evaluator_contract"][
            "minimum_practical_effects"
        ].__setitem__("Macro-F1_absolute_difference", 0.0),
        lambda value: value["evaluator_contract"]["no_harm_gate"].__setitem__(
            "any_of_cell_selection_permitted", True
        ),
        lambda value: value["evaluator_contract"][
            "selection_result_status"
        ].__setitem__("observed_selection_outcome_results", "confirmatory"),
    ],
)
def test_alias_order_duplicate_future_cross_dialogue_outcome_learned_and_rule_drift_fail(
    mutation,
) -> None:
    changed = copy.deepcopy(payload())
    mutation(changed)
    with pytest.raises(HarmBenchProtocolV2Error):
        validate_protocol_v2(changed)


def test_missing_unknown_keys_and_unregistered_runtime_controls_fail() -> None:
    changed = payload()
    del changed["privacy_contract"]
    with pytest.raises(HarmBenchProtocolV2Error, match="schema changed"):
        validate_protocol_v2(changed)

    changed = payload()
    changed["runtime_top_k"] = 7
    with pytest.raises(HarmBenchProtocolV2Error, match="schema changed"):
        validate_protocol_v2(changed)

    changed = payload()
    changed["runtime_control_contract"]["custom"] = False
    with pytest.raises(HarmBenchProtocolV2Error, match="canonical"):
        validate_protocol_v2(changed)


def test_validator_and_strategy_resolver_offer_no_free_control_parameters() -> None:
    assert set(inspect.signature(validate_protocol_v2).parameters) == {"payload"}
    assert set(inspect.signature(get_context_strategy_contract).parameters) == {
        "strategy_id"
    }


def test_duplicate_json_key_nonfinite_and_v1_alias_are_rejected() -> None:
    with pytest.raises(HarmBenchProtocolV2Error, match="duplicate JSON key"):
        decode_protocol_v2_json('{"protocol_id":"v2","protocol_id":"v1"}')
    with pytest.raises(HarmBenchProtocolV2Error, match="non-finite"):
        decode_protocol_v2_json('{"value":NaN}')
    changed = payload()
    changed["protocol_id"] = "harmbench_erc_v1"
    with pytest.raises(HarmBenchProtocolV2Error, match="identity"):
        validate_protocol_v2(changed)


def test_any_other_semantically_plausible_token_is_blocked_by_canonical_sha() -> None:
    changed = payload()
    changed["primary_analysis_contract"]["status"] = (
        "S2_candidate_metrics_ready_but_not_authorized"
    )
    with pytest.raises(HarmBenchProtocolV2Error, match="canonical"):
        validate_protocol_v2(changed)
