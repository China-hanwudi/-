"""Fail-closed development contract for ``harmbench_erc_v1``.

This draft contract deliberately cannot authorize official-test features,
predictions, labels or outcomes.  A later final protocol requires a new exact
schema, a complete model roster and prospective sensitivity evidence.
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


class HarmBenchContractError(ValueError):
    """Raised when the development protocol changes or becomes permissive."""


PROTOCOL_ID = "harmbench_erc_v1"
DRAFT_SCHEMA = "harmbench_erc_protocol_draft_v1"
EXPECTED_TRAINING_SEEDS = [17, 29, 43, 71, 101]
EXPECTED_CONFIRMATORY_DATASETS = ["EmotionTalk_official_test", "MELD_official_test"]
EXPECTED_STAGE_NAMES = [
    "S0_synthetic_contract",
    "S1_open_role_engineering_and_exploration",
    "S2_model_roster_threshold_power_and_code_freeze",
    "S3_official_test_feature_and_prediction_only",
    "S4_official_test_one_shot_label_evaluation",
]
EXPECTED_PAIRED_ESTIMAND_KEYS = {
    "primary_regret",
    "probability_floor_for_nll",
    "primary_harm_threshold_nats",
    "practical_harm_sensitivity_threshold_nats",
    "secondary_regret",
    "classification_transitions",
    "eligible_population",
    "policy_population_regret",
    "conditional_regret",
}
EXPECTED_METRIC_CONTRACT_KEYS = {
    "classification",
    "history_risk",
    "tail_alpha",
    "tail_definition",
    "selection",
    "official_test_top_k_reselection",
    "frozen_threshold_application_only",
    "descriptive_rank_curve_may_select_top_k_on_observed_open_role",
    "descriptive_rank_curve_can_replace_confirmatory_operating_point",
}
EXPECTED_INFERENCE_CONTRACT_KEYS = {
    "bootstrap",
    "bootstrap_replicates",
    "bootstrap_seed",
    "synthetic_profile",
    "cluster_unit",
    "composite_cluster_encoding",
    "all_models_and_strategies_share_draws",
    "seed_and_cluster_pairing_preserved_for_contrasts",
    "confidence_interval",
    "minimum_finite_bootstrap_fraction",
    "multiple_comparisons",
    "final_primary_family",
    "prospective_sensitivity",
    "official_test_authorized",
}
EXPECTED_PUBLIC_ARTIFACT_POLICY_KEYS = {
    "aggregate_only",
    "contains_labels_predictions_probabilities_or_embeddings",
    "contains_query_row_cluster_seed_or_participant_vectors",
    "contains_private_paths_or_outcome_hashes",
    "json_allow_nan",
    "atomic_write_once",
    "restricted_data_license_unchanged_by_repository_license",
}
EXPECTED_NLL_PROBABILITY_FLOOR = 1e-12
EXPECTED_HARM_THRESHOLDS = (0.0, 0.05)
EXPECTED_BOOTSTRAP_REPLICATES = 10000
EXPECTED_BOOTSTRAP_SEED = 20260810
EXPECTED_CONFIDENCE_INTERVAL = "two-sided percentile 95 percent"
EXPECTED_TAIL_ALPHA = 0.90
EXPECTED_MINIMUM_FINITE_BOOTSTRAP_FRACTION = 0.95
EXPECTED_SYNTHETIC_BOOTSTRAP_REPLICATES = 500
EXPECTED_SYNTHETIC_BOOTSTRAP_SEED = 20260810
EXPECTED_OFFICIAL_TEST_FAIL_CLOSED_KEYS = {
    "attempt_started_receipt_before_label_capability",
    "prelabel_bundle_write_once",
    "label_archive_allow_pickle",
    "label_hash_verified_before_and_after_single_load",
    "crash_after_unseal_can_silently_rerun",
    "public_output",
    "verifier_has_outcome_capability",
}
EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    "created_date",
    "research_question",
    "claim_boundary",
    "data_contract",
    "private_input_axes",
    "model_roster_gate",
    "required_strategy_roster",
    "paired_estimands",
    "metric_contract",
    "inference_contract",
    "provisional_claim_gate",
    "stages",
    "official_test_fail_closed",
    "public_artifact_policy",
    "gpt_policy",
}
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True)
class DevelopmentProtocolContract:
    payload: Mapping[str, object]
    canonical_sha256: str


def _reject_constant(value: str) -> None:
    raise HarmBenchContractError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HarmBenchContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def decode_protocol_json(encoded: str) -> Mapping[str, object]:
    try:
        payload = json.loads(
            encoded,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except HarmBenchContractError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise HarmBenchContractError(f"invalid protocol JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise HarmBenchContractError("protocol root must be a JSON object")
    return payload


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise HarmBenchContractError(f"{name} must be an object")
    return value


def _exact_keys(value: object, expected: set[str], *, name: str) -> Mapping[str, object]:
    mapping = _mapping(value, name=name)
    observed = {str(key) for key in mapping}
    if observed != expected:
        raise HarmBenchContractError(
            f"{name} schema changed: missing={sorted(expected-observed)}, "
            f"unknown={sorted(observed-expected)}"
        )
    return mapping


def _validate_json_tree(value: object, *, path: str = "$") -> None:
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HarmBenchContractError(f"non-finite value at {path}")
        return
    if isinstance(value, str):
        if WINDOWS_ABSOLUTE_PATH.match(value) or value.startswith(("/", "file://")):
            raise HarmBenchContractError(f"absolute/private path at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise HarmBenchContractError(f"non-string key at {path}")
            _validate_json_tree(child, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json_tree(child, path=f"{path}[{index}]")
        return
    raise HarmBenchContractError(f"unsupported JSON type at {path}: {type(value).__name__}")


def _frozen_float(value: object, *, expected: float, name: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value != expected:
        raise HarmBenchContractError(f"{name} changed")


def _frozen_integer(value: object, *, expected: int, name: str) -> None:
    if type(value) is not int or value != expected:
        raise HarmBenchContractError(f"{name} changed")


def _deep_freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze_json(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze_json(child) for child in value)
    return value


def _plain_json_tree(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json_tree(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_tree(child) for child in value]
    return value


def canonical_protocol_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        _plain_json_tree(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def validate_development_protocol(payload: Mapping[str, object]) -> DevelopmentProtocolContract:
    root = _exact_keys(payload, EXPECTED_TOP_LEVEL_KEYS, name="protocol")
    _validate_json_tree(root)
    if root["schema_version"] != DRAFT_SCHEMA or root["protocol_id"] != PROTOCOL_ID:
        raise HarmBenchContractError("protocol identity changed")
    if root["status"] != "development_contract_only_no_sealed_outcome_authority":
        raise HarmBenchContractError("draft status became permissive")

    data = _mapping(root["data_contract"], name="data_contract")
    if data.get("required_confirmatory_datasets") != EXPECTED_CONFIRMATORY_DATASETS:
        raise HarmBenchContractError("confirmatory dataset roster changed")
    if data.get("future_turns_permitted") is not False:
        raise HarmBenchContractError("future turns became permitted")
    if data.get("strict_past") is not True:
        raise HarmBenchContractError("strict-past contract changed")
    if data.get("restricted_data_to_external_gpt_api_permitted") is not False:
        raise HarmBenchContractError("restricted data became exportable to GPT")

    roster = _mapping(root["model_roster_gate"], name="model_roster_gate")
    if roster.get("training_seeds") != EXPECTED_TRAINING_SEEDS:
        raise HarmBenchContractError("training seed roster changed")
    if roster.get("minimum_substantively_distinct_model_families") != 3:
        raise HarmBenchContractError("minimum model-family requirement changed")
    if roster.get("exact_model_identifiers") != []:
        raise HarmBenchContractError("draft cannot register a partial model roster")
    if roster.get("freeze_state") != (
        "BLOCKED_until_three_feasible_model_families_and_checkpoints_are_registered"
    ):
        raise HarmBenchContractError("model-roster blocking state changed")
    if roster.get("official_test_authorized") is not False:
        raise HarmBenchContractError("model roster attempted to authorize official test")

    paired = _exact_keys(
        root["paired_estimands"],
        EXPECTED_PAIRED_ESTIMAND_KEYS,
        name="paired_estimands",
    )
    _frozen_float(
        paired["probability_floor_for_nll"],
        expected=EXPECTED_NLL_PROBABILITY_FLOOR,
        name="NLL probability floor",
    )
    _frozen_float(
        paired["primary_harm_threshold_nats"],
        expected=EXPECTED_HARM_THRESHOLDS[0],
        name="primary harm threshold",
    )
    _frozen_float(
        paired["practical_harm_sensitivity_threshold_nats"],
        expected=EXPECTED_HARM_THRESHOLDS[1],
        name="practical harm threshold",
    )

    metric = _exact_keys(
        root["metric_contract"],
        EXPECTED_METRIC_CONTRACT_KEYS,
        name="metric_contract",
    )
    _frozen_float(
        metric["tail_alpha"],
        expected=EXPECTED_TAIL_ALPHA,
        name="tail alpha",
    )
    if metric.get("tail_definition") != (
        "largest empirical 10 percent with fractional boundary mass; never "
        "values-greater-than-or-equal-to-quantile averaging"
    ):
        raise HarmBenchContractError("exact empirical CVaR definition changed")
    if metric.get("official_test_top_k_reselection") is not False:
        raise HarmBenchContractError("official-test top-k reselection became permitted")
    if metric.get("frozen_threshold_application_only") is not True:
        raise HarmBenchContractError("frozen-threshold contract changed")

    inference = _exact_keys(
        root["inference_contract"],
        EXPECTED_INFERENCE_CONTRACT_KEYS,
        name="inference_contract",
    )
    _frozen_integer(
        inference["bootstrap_replicates"],
        expected=EXPECTED_BOOTSTRAP_REPLICATES,
        name="bootstrap replicate count",
    )
    _frozen_integer(
        inference["bootstrap_seed"],
        expected=EXPECTED_BOOTSTRAP_SEED,
        name="bootstrap seed",
    )
    synthetic_profile = _exact_keys(
        inference["synthetic_profile"],
        {"bootstrap_replicates", "bootstrap_seed"},
        name="inference_contract.synthetic_profile",
    )
    _frozen_integer(
        synthetic_profile["bootstrap_replicates"],
        expected=EXPECTED_SYNTHETIC_BOOTSTRAP_REPLICATES,
        name="synthetic bootstrap replicate count",
    )
    _frozen_integer(
        synthetic_profile["bootstrap_seed"],
        expected=EXPECTED_SYNTHETIC_BOOTSTRAP_SEED,
        name="synthetic bootstrap seed",
    )
    if inference["all_models_and_strategies_share_draws"] is not True:
        raise HarmBenchContractError("shared-draw contract changed")
    if inference["seed_and_cluster_pairing_preserved_for_contrasts"] is not True:
        raise HarmBenchContractError("paired seed/cluster contract changed")
    if inference["confidence_interval"] != EXPECTED_CONFIDENCE_INTERVAL:
        raise HarmBenchContractError("confidence interval changed")
    _frozen_float(
        inference["minimum_finite_bootstrap_fraction"],
        expected=EXPECTED_MINIMUM_FINITE_BOOTSTRAP_FRACTION,
        name="minimum finite bootstrap fraction",
    )
    if inference.get("composite_cluster_encoding") != (
        "typed tuple factorization; delimiter joining forbidden"
    ):
        raise HarmBenchContractError("composite cluster encoding changed")
    if inference.get("final_primary_family") != []:
        raise HarmBenchContractError("draft cannot contain a partially frozen primary family")
    if inference.get("prospective_sensitivity") != "not_yet_computed":
        raise HarmBenchContractError("draft prospective sensitivity state changed")
    if inference.get("official_test_authorized") is not False:
        raise HarmBenchContractError("inference draft attempted to authorize official test")

    stages = root["stages"]
    if not isinstance(stages, list) or [row.get("stage") for row in stages if isinstance(row, Mapping)] != EXPECTED_STAGE_NAMES:
        raise HarmBenchContractError("stage roster or order changed")
    if len(stages) != len(EXPECTED_STAGE_NAMES):
        raise HarmBenchContractError("stage roster length changed")
    for index, stage in enumerate(stages):
        stage_map = _exact_keys(stage, {"stage", "outcomes", "authorized"}, name=f"stage[{index}]")
        expected = index <= 2
        if stage_map["authorized"] is not expected:
            raise HarmBenchContractError(f"stage authorization changed at {stage_map['stage']}")

    provisional = _mapping(root["provisional_claim_gate"], name="provisional_claim_gate")
    if provisional.get("status") != "not_frozen_and_cannot_authorize_test":
        raise HarmBenchContractError("provisional claim gate became permissive")
    official = _exact_keys(
        root["official_test_fail_closed"],
        EXPECTED_OFFICIAL_TEST_FAIL_CLOSED_KEYS,
        name="official_test_fail_closed",
    )
    if dict(official) != {
        "attempt_started_receipt_before_label_capability": True,
        "prelabel_bundle_write_once": True,
        "label_archive_allow_pickle": False,
        "label_hash_verified_before_and_after_single_load": True,
        "crash_after_unseal_can_silently_rerun": False,
        "public_output": "aggregate-only exact schema",
        "verifier_has_outcome_capability": False,
    }:
        raise HarmBenchContractError("official-test fail-closed contract changed")
    public = _exact_keys(
        root["public_artifact_policy"],
        EXPECTED_PUBLIC_ARTIFACT_POLICY_KEYS,
        name="public_artifact_policy",
    )
    if (
        public["aggregate_only"] is not True
        or public["contains_labels_predictions_probabilities_or_embeddings"] is not False
        or public["contains_query_row_cluster_seed_or_participant_vectors"] is not False
        or public["contains_private_paths_or_outcome_hashes"] is not False
        or public["json_allow_nan"] is not False
        or public["atomic_write_once"] is not True
        or public["restricted_data_license_unchanged_by_repository_license"] is not True
    ):
        raise HarmBenchContractError("public artifact safety contract changed")
    gpt = _mapping(root["gpt_policy"], name="gpt_policy")
    if gpt.get("current_real_data_status") != "NO_GO":
        raise HarmBenchContractError("real-data GPT gate changed")

    encoded = canonical_protocol_bytes(root)
    return DevelopmentProtocolContract(
        payload=_deep_freeze_json(root),
        canonical_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def load_development_protocol(path: str | Path) -> DevelopmentProtocolContract:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise HarmBenchContractError("protocol path must be a plain existing file")
    payload = decode_protocol_json(source.read_text(encoding="utf-8"))
    return validate_development_protocol(payload)
