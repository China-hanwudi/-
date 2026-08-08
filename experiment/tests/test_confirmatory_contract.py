from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.confirmatory_contract import (  # noqa: E402
    ConfirmatoryContractError,
    validate_confirmatory_analysis,
    validate_contract_files,
    validate_split_manifest,
)


CONFIG_DIR = ROOT / "configs"


def load(name: str) -> dict:
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


def frozen_contracts() -> tuple[dict, dict]:
    return (
        load("carma_split_manifest_v1.json"),
        load("carma_confirmatory_analysis_v1.json"),
    )


def test_frozen_contract_files_validate_without_reading_data() -> None:
    report = validate_contract_files(CONFIG_DIR)
    assert report["status"] == "PASS"
    assert report["split_protocol_id"] == "scu_set_exploration_v1"
    assert report["primary_history_coverage"] == 0.25
    assert report["seed_count"] == 5
    assert report["minimum_macro_f1_gain_absolute"] >= 0.005
    assert report["data_files_read"] == 0
    assert len(report["manifest_sha256"]) == 64
    assert len(report["analysis_sha256"]) == 64


def test_at_least_one_coverage_success_rule_is_rejected() -> None:
    manifest, analysis = frozen_contracts()
    analysis["primary_operating_point"]["primary_history_coverages"] = [0.10, 0.25]
    with pytest.raises(ConfirmatoryContractError, match="exactly one primary"):
        validate_confirmatory_analysis(analysis, manifest)

    _, analysis = frozen_contracts()
    analysis["primary_operating_point"]["forbidden_success_rules"].remove(
        "at_least_one_coverage_passes"
    )
    with pytest.raises(ConfirmatoryContractError, match="at least one coverage"):
        validate_confirmatory_analysis(analysis, manifest)


def test_model_protocol_cannot_change_role_assignment() -> None:
    manifest, _ = frozen_contracts()
    drifted = copy.deepcopy(manifest)
    drifted["assignment"]["hash_inputs"].append("model_protocol_id")
    with pytest.raises(ConfirmatoryContractError, match="role assignment may depend only"):
        validate_split_manifest(drifted)


def test_analysis_cannot_silently_switch_split_protocol() -> None:
    manifest, analysis = frozen_contracts()
    analysis["split_protocol_id"] = "new_model_specific_split"
    with pytest.raises(ConfirmatoryContractError, match="must match"):
        validate_confirmatory_analysis(analysis, manifest)


def test_macro_f1_gain_gate_cannot_drop_below_half_point() -> None:
    manifest, analysis = frozen_contracts()
    analysis["effect_and_safety_gates"]["minimum_macro_f1_gain_absolute"] = 0.0049
    with pytest.raises(ConfirmatoryContractError, match="at least 0.005"):
        validate_confirmatory_analysis(analysis, manifest)


def test_power_design_must_detect_the_half_point_gain_gate() -> None:
    manifest, analysis = frozen_contracts()
    analysis["mde_and_power"]["minimum_detectable_gain_absolute"] = 0.006
    with pytest.raises(ConfirmatoryContractError, match="MDE must be no larger"):
        validate_confirmatory_analysis(analysis, manifest)


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("calibration_sealed_before_model_and_analysis_freeze", False),
        ("internal_holdout_sealed_before_calibration_artifact_freeze", False),
        ("external_test_sealed_before_complete_bundle_freeze", False),
        ("test_labels_may_change_model_threshold_or_claim_family", True),
    ],
)
def test_calibration_holdout_and_test_seals_fail_closed(
    field: str, unsafe_value: bool
) -> None:
    manifest, analysis = frozen_contracts()
    analysis["sealing_and_stage_order"][field] = unsafe_value
    with pytest.raises(ConfirmatoryContractError):
        validate_confirmatory_analysis(analysis, manifest)


def test_sealed_role_cannot_be_relabelled_open() -> None:
    manifest, _ = frozen_contracts()
    manifest["roles"]["calibration"]["label_state"] = "open_for_model_selection"
    with pytest.raises(ConfirmatoryContractError, match="calibration must remain sealed"):
        validate_split_manifest(manifest)


def test_five_seed_requirement_is_exact_and_independent() -> None:
    manifest, analysis = frozen_contracts()
    analysis["independent_runs"]["seeds"] = [17, 17, 43, 71, 101]
    with pytest.raises(ConfirmatoryContractError, match="five distinct"):
        validate_confirmatory_analysis(analysis, manifest)


def test_external_llm_boundary_rejects_row_level_dataset_uploads() -> None:
    manifest, _ = frozen_contracts()
    manifest["privacy_boundary"]["external_llm_api"][
        "raw_or_row_level_restricted_dataset_content"
    ] = "allowed"
    with pytest.raises(ConfirmatoryContractError, match="external LLM API"):
        validate_split_manifest(manifest)


def test_accuracy_no_harm_is_mandatory_and_cannot_be_relaxed() -> None:
    manifest, analysis = frozen_contracts()
    analysis["effect_and_safety_gates"]["accuracy_no_harm"][
        "minimum_point_difference"
    ] = -0.001
    with pytest.raises(ConfirmatoryContractError, match="point difference"):
        validate_confirmatory_analysis(analysis, manifest)

    _, analysis = frozen_contracts()
    analysis["effect_and_safety_gates"]["accuracy_no_harm"][
        "minimum_ci95_lower"
    ] = -0.006
    with pytest.raises(ConfirmatoryContractError, match="CI lower"):
        validate_confirmatory_analysis(analysis, manifest)

    _, analysis = frozen_contracts()
    analysis["effect_and_safety_gates"][
        "classification_safety_and_accuracy_no_harm_must_all_pass"
    ] = False
    with pytest.raises(ConfirmatoryContractError, match="must all pass"):
        validate_confirmatory_analysis(analysis, manifest)


def test_shared_cluster_crossed_bootstrap_cannot_revert_to_nested_seed_draws() -> None:
    manifest, analysis = frozen_contracts()
    analysis["hierarchical_bootstrap"]["resampling_hierarchy"] = [
        "training_seed",
        "dataset_specific_independent_cluster",
    ]
    with pytest.raises(ConfirmatoryContractError, match="share the cluster draw"):
        validate_confirmatory_analysis(analysis, manifest)

    _, analysis = frozen_contracts()
    analysis["hierarchical_bootstrap"][
        "shared_cluster_draw_across_training_seeds"
    ] = False
    with pytest.raises(ConfirmatoryContractError, match="share each whole-cluster"):
        validate_confirmatory_analysis(analysis, manifest)


def test_holm_p_values_require_whole_cluster_randomization_not_bootstrap_tails() -> None:
    manifest, analysis = frozen_contracts()
    analysis["hypothesis_testing"][
        "bootstrap_tail_probability_may_be_used_as_p_value"
    ] = True
    with pytest.raises(ConfirmatoryContractError, match="bootstrap tail"):
        validate_confirmatory_analysis(analysis, manifest)

    _, analysis = frozen_contracts()
    analysis["hypothesis_testing"]["method"] = "uncentered_bootstrap_tail"
    with pytest.raises(ConfirmatoryContractError, match="whole-cluster randomization"):
        validate_confirmatory_analysis(analysis, manifest)


def test_primary_reference_must_include_all_strong_admissible_baselines() -> None:
    manifest, analysis = frozen_contracts()
    analysis["primary_contrasts"][0]["reference_candidates"].remove(
        "coverage_matched_recency"
    )
    with pytest.raises(ConfirmatoryContractError, match="all-history, recency"):
        validate_confirmatory_analysis(analysis, manifest)

    _, analysis = frozen_contracts()
    analysis["holm_family"]["hypotheses"][0][
        "contrast"
    ] = "carma_bidirectional_vs_frozen_strongest_single_direction"
    with pytest.raises(ConfirmatoryContractError, match="strongest admissible"):
        validate_confirmatory_analysis(analysis, manifest)
