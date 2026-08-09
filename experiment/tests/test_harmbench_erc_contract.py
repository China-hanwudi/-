from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.harmbench_erc_contract import (  # noqa: E402
    HarmBenchContractError,
    canonical_protocol_bytes,
    decode_protocol_json,
    load_development_protocol,
    validate_development_protocol,
)


CONFIG = ROOT / "configs" / "harmbench_erc_v1_draft.json"


def payload() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_repository_draft_is_valid_and_hash_stable() -> None:
    first = load_development_protocol(CONFIG)
    second = load_development_protocol(CONFIG)
    assert first.canonical_sha256 == second.canonical_sha256
    assert len(first.canonical_sha256) == 64


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["model_roster_gate"].__setitem__("official_test_authorized", True),
        lambda value: value["inference_contract"].__setitem__("official_test_authorized", True),
        lambda value: value["stages"][3].__setitem__("authorized", True),
        lambda value: value["stages"][4].__setitem__("authorized", True),
        lambda value: value["data_contract"].__setitem__("future_turns_permitted", True),
        lambda value: value["gpt_policy"].__setitem__("current_real_data_status", "GO"),
    ],
)
def test_any_sealed_or_future_capability_change_fails_closed(mutation) -> None:
    changed = copy.deepcopy(payload())
    mutation(changed)
    with pytest.raises(HarmBenchContractError):
        validate_development_protocol(changed)


def test_partial_model_or_hypothesis_roster_fails_closed() -> None:
    changed = payload()
    changed["model_roster_gate"]["exact_model_identifiers"] = ["one_model_only"]
    with pytest.raises(HarmBenchContractError, match="partial model roster"):
        validate_development_protocol(changed)
    changed = payload()
    changed["inference_contract"]["final_primary_family"] = ["H1"]
    with pytest.raises(HarmBenchContractError, match="partially frozen primary family"):
        validate_development_protocol(changed)


@pytest.mark.parametrize(
    ("section", "key", "changed_value", "message"),
    [
        ("paired_estimands", "probability_floor_for_nll", 1e-9, "NLL probability floor"),
        ("paired_estimands", "primary_harm_threshold_nats", 0.01, "primary harm threshold"),
        (
            "paired_estimands",
            "practical_harm_sensitivity_threshold_nats",
            0.10,
            "practical harm threshold",
        ),
        ("inference_contract", "bootstrap_seed", 7, "bootstrap seed"),
        (
            "inference_contract",
            "synthetic_profile",
            {"bootstrap_replicates": 100, "bootstrap_seed": 11},
            "synthetic bootstrap replicate count",
        ),
        (
            "inference_contract",
            "all_models_and_strategies_share_draws",
            False,
            "shared-draw contract",
        ),
        (
            "inference_contract",
            "confidence_interval",
            "one-sided 95 percent",
            "confidence interval",
        ),
        ("metric_contract", "tail_alpha", 0.8, "tail alpha"),
        (
            "inference_contract",
            "minimum_finite_bootstrap_fraction",
            0.90,
            "minimum finite bootstrap fraction",
        ),
        (
            "public_artifact_policy",
            "contains_private_paths_or_outcome_hashes",
            True,
            "public artifact safety",
        ),
    ],
)
def test_frozen_metric_inference_and_public_values_fail_closed(
    section: str, key: str, changed_value: object, message: str
) -> None:
    changed = payload()
    changed[section][key] = changed_value
    with pytest.raises(HarmBenchContractError, match=message):
        validate_development_protocol(changed)


@pytest.mark.parametrize(
    "section",
    ["paired_estimands", "metric_contract", "inference_contract", "public_artifact_policy"],
)
def test_frozen_nested_sections_reject_unknown_keys(section: str) -> None:
    changed = payload()
    changed[section]["unregistered_flexibility"] = False
    with pytest.raises(HarmBenchContractError, match="schema changed"):
        validate_development_protocol(changed)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("verifier_has_outcome_capability", True),
        ("label_archive_allow_pickle", True),
        ("crash_after_unseal_can_silently_rerun", True),
    ],
)
def test_official_test_fail_closed_values_cannot_be_relaxed(
    key: str, value: object
) -> None:
    changed = payload()
    changed["official_test_fail_closed"][key] = value
    with pytest.raises(HarmBenchContractError, match="official-test fail-closed"):
        validate_development_protocol(changed)


def test_official_test_fail_closed_rejects_unknown_capability() -> None:
    changed = payload()
    changed["official_test_fail_closed"]["official_test_labels_authorized"] = True
    with pytest.raises(HarmBenchContractError, match="schema changed"):
        validate_development_protocol(changed)


def test_validated_contract_is_deeply_immutable_and_hash_cannot_go_stale() -> None:
    contract = load_development_protocol(CONFIG)
    with pytest.raises(TypeError):
        contract.payload["status"] = "changed"
    with pytest.raises(TypeError):
        contract.payload["inference_contract"]["bootstrap_seed"] = 1
    assert canonical_protocol_bytes(contract.payload)
    assert contract.canonical_sha256 == __import__("hashlib").sha256(
        canonical_protocol_bytes(contract.payload)
    ).hexdigest()


def test_duplicate_key_and_nonfinite_json_are_rejected() -> None:
    with pytest.raises(HarmBenchContractError, match="duplicate JSON key"):
        decode_protocol_json('{"protocol_id":"a","protocol_id":"b"}')
    with pytest.raises(HarmBenchContractError, match="non-finite"):
        decode_protocol_json('{"value":NaN}')


def test_absolute_path_in_protocol_is_rejected() -> None:
    changed = payload()
    changed["research_question"] = "C:\\private\\labels.npy"
    with pytest.raises(HarmBenchContractError, match="absolute/private path"):
        validate_development_protocol(changed)
