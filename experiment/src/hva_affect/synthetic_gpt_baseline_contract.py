"""Offline, synthetic-only contract for a future GPT text baseline.

This module intentionally contains no API client, network transport, dataset
adapter, or import from the causal modeling stack.  Version 1 can only validate
an explicitly attested synthetic JSON fixture and cache caller-supplied mock
responses.  It is an interface and privacy-contract test, not a model run.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "carma_synthetic_gpt_text_baseline_config_v1"
PROTOCOL_ID = "carma_synthetic_gpt_text_baseline_v1"
ANALYSIS_ID = "carma_synthetic_gpt_text_baseline_contract_v1"
FIXTURE_SCHEMA = "carma_explicit_synthetic_text_fixture_v1"
PRIVATE_CACHE_SCHEMA = "carma_synthetic_gpt_private_response_cache_v1"
PUBLIC_RECEIPT_SCHEMA = "carma_synthetic_gpt_public_aggregate_receipt_v1"

MODULE_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = MODULE_PATH.parents[3]
EXPECTED_MODULE_PATH = (
    REPOSITORY_ROOT
    / "experiment"
    / "src"
    / "hva_affect"
    / "synthetic_gpt_baseline_contract.py"
).resolve()
if MODULE_PATH != EXPECTED_MODULE_PATH:
    raise RuntimeError("synthetic GPT contract module is outside its fixed repository location")

FROZEN_MODEL = "gpt-5.6-terra"
FROZEN_MODEL_SNAPSHOT = "current_alias_unpinned_no_execution"
FROZEN_ENDPOINT = "/v1/responses"
SYNTHETIC_TEXT_PREFIX = "[SYNTHETIC] "
MAX_RECORDS = 256
MAX_CURRENT_CHARACTERS = 512
MAX_HISTORY_ITEMS = 8
MAX_HISTORY_ITEM_CHARACTERS = 512
MAX_TOTAL_INPUT_CHARACTERS = 4096
MINIMUM_HMAC_KEY_BYTES = 32
MAXIMUM_HMAC_KEY_BYTES = 4096
MAXIMUM_FIXTURE_FILE_BYTES = 16 * 1024
MAXIMUM_CONFIG_FILE_BYTES = 64 * 1024
MAXIMUM_SPLIT_MANIFEST_FILE_BYTES = 64 * 1024

SPLIT_MANIFEST_ID = "carma_split_manifest_v1"
SPLIT_MANIFEST_SCHEMA = "1.0.0"
SPLIT_MANIFEST_STATUS = "frozen_before_confirmatory_runs"
SPLIT_PROTOCOL_ID = "scu_set_exploration_v1"
FROZEN_SPLIT_MANIFEST_SHA256 = "81fc2808b3821486db9dfb3f6fa303fb23e49e1471e927a59ac404f5d9122798"
FROZEN_CONFIG_SHA256 = "68f2ef6f711d0a769d8d77a880f990f99da553ecbcb6d9c2c1b4fabebc3c91ef"

EMOTION_LABELS = (
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
)
HISTORY_EFFECTS = ("beneficial", "harmful", "uncertain")

FROZEN_SYSTEM_PROMPT = (
    "You are a frozen text-only emotion baseline contract. Analyze only the "
    "explicitly synthetic current utterance and the explicitly synthetic, "
    "strictly earlier history supplied in this request. Return exactly the "
    "declared JSON schema. Do not infer identities, use tools, browse, upload "
    "files, or request additional context."
)
FROZEN_INPUT_TEMPLATE = (
    "CURRENT_SYNTHETIC_UTTERANCE:\n{current_text}\n"
    "STRICT_PAST_SYNTHETIC_HISTORY_JSON:\n{history_texts_json}"
)
FROZEN_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "predicted_emotion",
        "probabilities",
        "history_effect",
        "vad",
    ],
    "properties": {
        "predicted_emotion": {"type": "string", "enum": list(EMOTION_LABELS)},
        "probabilities": {
            "type": "array",
            "minItems": len(EMOTION_LABELS),
            "maxItems": len(EMOTION_LABELS),
            "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "history_effect": {"type": "string", "enum": list(HISTORY_EFFECTS)},
        "vad": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        },
    },
}

FORBIDDEN_RECORD_FIELDS = (
    "dataset_id",
    "split",
    "role",
    "group_id",
    "dialogue_id",
    "speaker_id",
    "label",
    "labels",
    "audio",
    "video",
    "embedding",
    "media_path",
    "sidecar_path",
)
FORBIDDEN_DATASET_TOKENS = (
    "meld",
    "emotiontalk",
    "iemocap",
    "cped",
    "m3ed",
)
FORBIDDEN_FIXTURE_FILENAME_TOKENS = FORBIDDEN_DATASET_TOKENS + (
    "sidecar",
    "transcript",
    "label",
    "media",
)
SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"(?:https?://|www\.)", re.IGNORECASE),
    re.compile(r"(?:[A-Za-z]:[\\/]|\\\\[^\\]+\\|(?:^|\s)/(?:Users|home|mnt|tmp|var|data)/)"),
    re.compile(r"\b(?:\+?\d[\s().-]?){7,}\b"),
)


class SyntheticGPTContractError(ValueError):
    """Raised when the synthetic-only boundary is missing or has drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SyntheticGPTContractError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{name} must be an array",
    )
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    _require(set(value) == expected, f"{name} schema changed")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SyntheticGPTContractError(f"cannot hash required file {path.name}: {exc}") from exc
    return digest.hexdigest()


FROZEN_PROMPT_SHA256 = canonical_sha256(
    {
        "system_prompt": FROZEN_SYSTEM_PROMPT,
        "input_template": FROZEN_INPUT_TEMPLATE,
    }
)
FROZEN_RESPONSE_SCHEMA_SHA256 = canonical_sha256(FROZEN_RESPONSE_SCHEMA)

_CANONICAL_SYNTHETIC_FIXTURE: dict[str, Any] = {
    "schema_version": FIXTURE_SCHEMA,
    "fixture_kind": "explicit_synthetic_contract_fixture",
    "synthetic": True,
    "attestation": {
        "source": "handwritten_or_programmatically_generated_synthetic_only",
        "generated_for_contract_testing": True,
        "contains_real_dataset_content": False,
        "contains_personal_data": False,
        "contains_dataset_labels": False,
    },
    "records": [
        {
            "synthetic": True,
            "current_text": "[SYNTHETIC] The invented blue lantern feels cheerful now.",
            "history_texts": [
                "[SYNTHETIC] The invented blue lantern was quiet one imaginary turn earlier."
            ],
            "mock_response": {
                "predicted_emotion": "joy",
                "probabilities": [0.05, 0.05, 0.05, 0.70, 0.05, 0.05, 0.05],
                "history_effect": "beneficial",
                "vad": [0.4, 0.2, 0.1],
            },
        },
        {
            "synthetic": True,
            "current_text": "[SYNTHETIC] The invented amber cloud reports a calm imaginary moment.",
            "history_texts": [],
            "mock_response": {
                "predicted_emotion": "neutral",
                "probabilities": [0.05, 0.05, 0.05, 0.05, 0.70, 0.05, 0.05],
                "history_effect": "uncertain",
                "vad": [0.0, 0.1, 0.0],
            },
        },
    ],
}
CANONICAL_FIXTURE_SHA256 = canonical_sha256(_CANONICAL_SYNTHETIC_FIXTURE)


def canonical_synthetic_fixture() -> dict[str, Any]:
    """Return a fresh copy of the sole input fixture accepted by v1."""

    return json.loads(_canonical_json_bytes(_CANONICAL_SYNTHETIC_FIXTURE).decode("utf-8"))


def load_json_object(path: Path, *, name: str, maximum_bytes: int) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise SyntheticGPTContractError(f"cannot stat {name} {path.name}: {exc}") from exc
    _require(0 < size <= maximum_bytes, f"{name} exceeds its frozen file-size limit")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SyntheticGPTContractError(f"cannot load {name} {path.name}: {exc}") from exc
    _require(isinstance(payload, dict), f"{name} must contain a JSON object")
    return payload


def validate_contract_config(config: Mapping[str, Any]) -> None:
    """Validate every frozen field of the non-executable v1 contract."""

    _exact_keys(
        config,
        {
            "schema_version",
            "protocol_id",
            "analysis_id",
            "status",
            "mode",
            "split_manifest_contract",
            "provider_contract",
            "prompt_contract",
            "context_contract",
            "privacy_contract",
            "production_gate",
        },
        "synthetic GPT config",
    )
    _require(config.get("schema_version") == CONFIG_SCHEMA, "config schema changed")
    _require(config.get("protocol_id") == PROTOCOL_ID, "protocol id changed")
    _require(config.get("analysis_id") == ANALYSIS_ID, "analysis id changed")
    _require(
        config.get("status") == "frozen_synthetic_only_no_model_execution",
        "synthetic-only status changed",
    )
    _require(config.get("mode") == "explicit_synthetic_fixture_only", "mode changed")

    split_contract = _mapping(config.get("split_manifest_contract"), "split_manifest_contract")
    _exact_keys(
        split_contract,
        {"manifest_id", "schema_version", "status", "split_protocol_id", "sha256"},
        "split_manifest_contract",
    )
    _require(split_contract.get("manifest_id") == SPLIT_MANIFEST_ID, "split manifest id changed")
    _require(
        split_contract.get("schema_version") == SPLIT_MANIFEST_SCHEMA,
        "split manifest schema changed",
    )
    _require(split_contract.get("status") == SPLIT_MANIFEST_STATUS, "split manifest status changed")
    _require(
        split_contract.get("split_protocol_id") == SPLIT_PROTOCOL_ID,
        "split protocol id changed",
    )
    _require(
        split_contract.get("sha256") == FROZEN_SPLIT_MANIFEST_SHA256,
        "split manifest hash changed",
    )

    provider = _mapping(config.get("provider_contract"), "provider_contract")
    _exact_keys(
        provider,
        {
            "provider",
            "endpoint",
            "model",
            "model_snapshot",
            "api_transport",
            "api_execution_enabled",
            "store",
            "background",
            "batch_enabled",
            "tools_enabled",
            "tool_definitions",
            "web_search_enabled",
            "file_search_enabled",
            "file_uploads_enabled",
            "reasoning",
        },
        "provider_contract",
    )
    _require(provider.get("provider") == "OpenAI", "provider changed")
    _require(provider.get("endpoint") == FROZEN_ENDPOINT, "endpoint changed")
    _require(provider.get("model") == FROZEN_MODEL, "model changed")
    _require(provider.get("model_snapshot") == FROZEN_MODEL_SNAPSHOT, "snapshot changed")
    _require(provider.get("api_transport") == "not_implemented", "API transport must be absent")
    _require(provider.get("api_execution_enabled") is False, "API execution must be disabled")
    _require(provider.get("store") is False, "store must remain false")
    _require(provider.get("background") is False, "background mode must remain disabled")
    _require(provider.get("batch_enabled") is False, "Batch must remain disabled")
    _require(provider.get("tools_enabled") is False, "tools must remain disabled")
    _require(provider.get("tool_definitions") == [], "tool definitions must remain empty")
    for field in ("web_search_enabled", "file_search_enabled", "file_uploads_enabled"):
        _require(provider.get(field) is False, f"{field} must remain disabled")
    reasoning = _mapping(provider.get("reasoning"), "provider_contract.reasoning")
    _exact_keys(reasoning, {"effort", "context"}, "provider_contract.reasoning")
    _require(reasoning == {"effort": "low", "context": "current_turn"}, "reasoning changed")

    prompt = _mapping(config.get("prompt_contract"), "prompt_contract")
    _exact_keys(
        prompt,
        {
            "system_prompt",
            "input_template",
            "prompt_sha256",
            "response_schema",
            "response_schema_sha256",
        },
        "prompt_contract",
    )
    _require(prompt.get("system_prompt") == FROZEN_SYSTEM_PROMPT, "system prompt changed")
    _require(prompt.get("input_template") == FROZEN_INPUT_TEMPLATE, "input template changed")
    _require(prompt.get("prompt_sha256") == FROZEN_PROMPT_SHA256, "prompt hash changed")
    _require(prompt.get("response_schema") == FROZEN_RESPONSE_SCHEMA, "response schema changed")
    _require(
        prompt.get("response_schema_sha256") == FROZEN_RESPONSE_SCHEMA_SHA256,
        "response schema hash changed",
    )

    context = _mapping(config.get("context_contract"), "context_contract")
    _exact_keys(
        context,
        {
            "accepted_fixture_schema",
            "current_text_required",
            "history_semantics",
            "max_records",
            "max_current_characters",
            "max_history_items",
            "max_history_item_characters",
            "max_total_input_characters",
            "maximum_fixture_file_bytes",
            "required_synthetic_text_prefix",
            "forbidden_record_fields",
            "canonical_fixture_sha256_allowlist",
        },
        "context_contract",
    )
    _require(context.get("accepted_fixture_schema") == FIXTURE_SCHEMA, "fixture schema changed")
    _require(context.get("current_text_required") is True, "current text must be required")
    _require(
        context.get("history_semantics") == "strict_past_only_oldest_to_newest",
        "history semantics changed",
    )
    _require(context.get("max_records") == MAX_RECORDS, "record limit changed")
    _require(
        context.get("max_current_characters") == MAX_CURRENT_CHARACTERS,
        "current-text limit changed",
    )
    _require(context.get("max_history_items") == MAX_HISTORY_ITEMS, "history item limit changed")
    _require(
        context.get("max_history_item_characters") == MAX_HISTORY_ITEM_CHARACTERS,
        "history-text limit changed",
    )
    _require(
        context.get("max_total_input_characters") == MAX_TOTAL_INPUT_CHARACTERS,
        "total input limit changed",
    )
    _require(
        context.get("maximum_fixture_file_bytes") == MAXIMUM_FIXTURE_FILE_BYTES,
        "fixture file-size limit changed",
    )
    _require(
        context.get("required_synthetic_text_prefix") == SYNTHETIC_TEXT_PREFIX,
        "synthetic text prefix changed",
    )
    _require(
        list(_sequence(context.get("forbidden_record_fields"), "forbidden_record_fields"))
        == list(FORBIDDEN_RECORD_FIELDS),
        "forbidden record fields changed",
    )
    _require(
        context.get("canonical_fixture_sha256_allowlist") == [CANONICAL_FIXTURE_SHA256],
        "canonical synthetic fixture allowlist changed",
    )

    privacy = _mapping(config.get("privacy_contract"), "privacy_contract")
    _exact_keys(
        privacy,
        {
            "hmac_algorithm",
            "minimum_hmac_key_bytes",
            "maximum_hmac_key_bytes",
            "minimum_hmac_key_distinct_bytes",
            "private_cache_location",
            "private_cache_write_policy",
            "stored_request_fields",
            "raw_request_text_stored",
            "public_output",
            "public_row_level_hmacs",
            "public_responses",
        },
        "privacy_contract",
    )
    _require(privacy.get("hmac_algorithm") == "HMAC-SHA256", "HMAC algorithm changed")
    _require(
        privacy.get("minimum_hmac_key_bytes") == MINIMUM_HMAC_KEY_BYTES,
        "minimum HMAC key length changed",
    )
    _require(
        privacy.get("maximum_hmac_key_bytes") == MAXIMUM_HMAC_KEY_BYTES,
        "maximum HMAC key length changed",
    )
    _require(
        privacy.get("minimum_hmac_key_distinct_bytes") == 16,
        "minimum HMAC key diversity changed",
    )
    _require(
        privacy.get("private_cache_location") == "outside_repository_required",
        "private cache boundary changed",
    )
    _require(
        privacy.get("private_cache_write_policy") == "create_once_refuse_overwrite",
        "private cache write policy changed",
    )
    _require(
        privacy.get("stored_request_fields")
        == ["hmac_sha256", "input_character_count", "input_utf8_bytes", "synthetic"],
        "stored request fields changed",
    )
    _require(privacy.get("raw_request_text_stored") is False, "raw request text must not be stored")
    _require(
        privacy.get("public_output") == "aggregate_receipt_and_file_hashes_only",
        "public output boundary changed",
    )
    _require(privacy.get("public_row_level_hmacs") is False, "row-level HMACs must stay private")
    _require(privacy.get("public_responses") is False, "responses must stay private")

    production = _mapping(config.get("production_gate"), "production_gate")
    _exact_keys(
        production,
        {
            "production_enabled",
            "external_llm_api_authorization",
            "network_access",
            "real_dataset_content",
            "required_future_manifest_state",
            "authorization_change_requires_new_protocol_version",
        },
        "production_gate",
    )
    _require(production.get("production_enabled") is False, "production must remain disabled")
    _require(
        production.get("external_llm_api_authorization") == "absent",
        "v1 must not claim external LLM authorization",
    )
    _require(production.get("network_access") == "forbidden", "network access must be forbidden")
    _require(
        production.get("real_dataset_content") == "forbidden",
        "real dataset content must be forbidden",
    )
    _require(
        production.get("required_future_manifest_state")
        == "separate_license_privacy_and_data_processing_approval",
        "future authorization gate changed",
    )
    _require(
        production.get("authorization_change_requires_new_protocol_version") is True,
        "authorization must require a new protocol version",
    )


def validate_external_llm_boundary(split_manifest: Mapping[str, Any]) -> None:
    """Strictly validate the identity and privacy section of the frozen manifest."""

    _exact_keys(
        split_manifest,
        {
            "manifest_id",
            "schema_version",
            "status",
            "frozen_on",
            "split_protocol_id",
            "assignment",
            "roles",
            "external_test_policy",
            "datasets",
            "privacy_boundary",
            "drift_policy",
        },
        "split manifest",
    )
    _require(split_manifest.get("manifest_id") == SPLIT_MANIFEST_ID, "split manifest id changed")
    _require(
        split_manifest.get("schema_version") == SPLIT_MANIFEST_SCHEMA,
        "split manifest schema changed",
    )
    _require(split_manifest.get("status") == SPLIT_MANIFEST_STATUS, "split manifest status changed")
    _require(split_manifest.get("frozen_on") == "2026-08-08", "split manifest freeze date changed")
    _require(
        split_manifest.get("split_protocol_id") == SPLIT_PROTOCOL_ID,
        "split protocol id changed",
    )
    privacy = _mapping(split_manifest.get("privacy_boundary"), "privacy_boundary")
    expected_privacy = {
        "repository_allowed": [
            "source_code_and_contract_tests",
            "frozen_configuration",
            "non_reidentifying_aggregate_metrics_and_confidence_intervals",
            "synthetic_test_fixtures",
        ],
        "repository_forbidden": [
            "raw_or_redistributed_text_labels_audio_or_video",
            "speaker_dialogue_or_query_keys",
            "media_paths_or_transcript_tables",
            "per_query_predictions_or_utilities",
            "derived_audio_or_video_embeddings",
            "model_weights_checkpoints_or_training_bundles",
            "license_forms_ethics_material_or_authorization_messages",
            "tokens_passwords_cookies_or_private_environment_files",
        ],
        "external_llm_api": {
            "raw_or_row_level_restricted_dataset_content": "forbidden",
            "allowed_payloads": [
                "public_dataset_schema",
                "non_reidentifying_aggregate_statistics",
                "synthetic_examples",
            ],
            "exception_policy": (
                "requires_separate_license_privacy_and_data_processing_approval_"
                "before_manifest_revision"
            ),
        },
    }
    _require(privacy == expected_privacy, "split manifest privacy boundary changed")


def assert_production_authorized(
    config: Mapping[str, Any], split_manifest: Mapping[str, Any]
) -> None:
    """Fail closed: production cannot be enabled within this protocol version."""

    validate_contract_config(config)
    validate_external_llm_boundary(split_manifest)
    raise SyntheticGPTContractError(
        "production rejected: external_llm_api authorization is absent; "
        "a separately approved, versioned protocol is required"
    )


def _validate_fixture_path(path: Path) -> None:
    name = path.name.lower()
    _require(name.endswith(".synthetic.json"), "fixture filename must end in .synthetic.json")
    _require(
        not any(token in name for token in FORBIDDEN_FIXTURE_FILENAME_TOKENS),
        "fixture filename resembles a dataset, sidecar, transcript, label, or media file",
    )
    _require(path.is_file(), "explicit synthetic fixture does not exist")
    resolved_parts = [part.casefold() for part in path.resolve().parts]
    _require(
        not any(
            token in part
            for part in resolved_parts
            for token in FORBIDDEN_FIXTURE_FILENAME_TOKENS
        ),
        "fixture path ancestry resembles a dataset, sidecar, transcript, label, or media boundary",
    )
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise SyntheticGPTContractError(f"cannot stat explicit synthetic fixture: {exc}") from exc
    _require(0 < size <= MAXIMUM_FIXTURE_FILE_BYTES, "synthetic fixture exceeds its file-size limit")


def _validate_synthetic_text(value: Any, *, name: str, maximum: int) -> str:
    _require(isinstance(value, str), f"{name} must be text")
    _require(value.startswith(SYNTHETIC_TEXT_PREFIX), f"{name} lacks the synthetic prefix")
    _require(0 < len(value) <= maximum, f"{name} exceeds its frozen length limit")
    lowered = value.casefold()
    _require(
        not any(token in lowered for token in FORBIDDEN_DATASET_TOKENS),
        f"{name} names a restricted dataset",
    )
    _require(
        not any(pattern.search(value) for pattern in SENSITIVE_TEXT_PATTERNS),
        f"{name} resembles PII, a URL, or a filesystem path",
    )
    _require("\x00" not in value, f"{name} contains a null character")
    return value


def _finite_number(value: Any, *, name: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    number = float(value)
    _require(math.isfinite(number), f"{name} must be finite")
    return number


def validate_mock_response(value: Any) -> dict[str, Any]:
    response = _mapping(value, "mock_response")
    _exact_keys(
        response,
        {"predicted_emotion", "probabilities", "history_effect", "vad"},
        "mock_response",
    )
    predicted = response.get("predicted_emotion")
    _require(predicted in EMOTION_LABELS, "predicted_emotion is outside the frozen label order")
    history_effect = response.get("history_effect")
    _require(history_effect in HISTORY_EFFECTS, "history_effect is invalid")

    probabilities = [
        _finite_number(item, name="probability")
        for item in _sequence(response.get("probabilities"), "probabilities")
    ]
    _require(len(probabilities) == len(EMOTION_LABELS), "probability vector length changed")
    _require(all(0.0 <= item <= 1.0 for item in probabilities), "probability is outside [0, 1]")
    _require(abs(sum(probabilities) - 1.0) <= 1e-6, "probabilities must sum to one")
    maximum = max(probabilities)
    _require(
        probabilities[EMOTION_LABELS.index(str(predicted))] == maximum,
        "predicted_emotion must attain the maximum probability",
    )

    vad = [
        _finite_number(item, name="VAD value")
        for item in _sequence(response.get("vad"), "vad")
    ]
    _require(len(vad) == 3, "VAD must contain exactly three values")
    _require(all(-1.0 <= item <= 1.0 for item in vad), "VAD value is outside [-1, 1]")
    return {
        "predicted_emotion": str(predicted),
        "probabilities": probabilities,
        "history_effect": str(history_effect),
        "vad": vad,
    }


def validate_synthetic_fixture(fixture: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate explicit, bounded synthetic content and caller-supplied mock outputs."""

    _exact_keys(
        fixture,
        {"schema_version", "fixture_kind", "synthetic", "attestation", "records"},
        "synthetic fixture",
    )
    _require(fixture.get("schema_version") == FIXTURE_SCHEMA, "fixture schema changed")
    _require(
        fixture.get("fixture_kind") == "explicit_synthetic_contract_fixture",
        "fixture kind is not explicitly synthetic",
    )
    _require(fixture.get("synthetic") is True, "fixture lacks the synthetic marker")
    attestation = _mapping(fixture.get("attestation"), "attestation")
    _exact_keys(
        attestation,
        {
            "source",
            "generated_for_contract_testing",
            "contains_real_dataset_content",
            "contains_personal_data",
            "contains_dataset_labels",
        },
        "attestation",
    )
    _require(
        attestation
        == {
            "source": "handwritten_or_programmatically_generated_synthetic_only",
            "generated_for_contract_testing": True,
            "contains_real_dataset_content": False,
            "contains_personal_data": False,
            "contains_dataset_labels": False,
        },
        "synthetic fixture attestation is incomplete",
    )

    records = list(_sequence(fixture.get("records"), "records"))
    _require(1 <= len(records) <= MAX_RECORDS, "fixture record count is outside the frozen bound")
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        record = _mapping(raw, f"records[{index}]")
        _exact_keys(
            record,
            {"synthetic", "current_text", "history_texts", "mock_response"},
            f"records[{index}]",
        )
        _require(record.get("synthetic") is True, f"records[{index}] lacks synthetic=true")
        _require(
            not (set(record) & set(FORBIDDEN_RECORD_FIELDS)),
            f"records[{index}] contains a forbidden dataset field",
        )
        current = _validate_synthetic_text(
            record.get("current_text"),
            name=f"records[{index}].current_text",
            maximum=MAX_CURRENT_CHARACTERS,
        )
        history_raw = list(_sequence(record.get("history_texts"), "history_texts"))
        _require(len(history_raw) <= MAX_HISTORY_ITEMS, "history item count exceeds the frozen limit")
        history = [
            _validate_synthetic_text(
                item,
                name=f"records[{index}].history_texts[{history_index}]",
                maximum=MAX_HISTORY_ITEM_CHARACTERS,
            )
            for history_index, item in enumerate(history_raw)
        ]
        total_characters = len(current) + sum(len(item) for item in history)
        _require(
            total_characters <= MAX_TOTAL_INPUT_CHARACTERS,
            f"records[{index}] total input exceeds the frozen limit",
        )
        validated.append(
            {
                "current_text": current,
                "history_texts": history,
                "mock_response": validate_mock_response(record.get("mock_response")),
            }
        )
    _require(
        fixture == canonical_synthetic_fixture(),
        "fixture differs from the sole canonical synthetic fixture accepted by v1",
    )
    _require(
        canonical_sha256(fixture) == CANONICAL_FIXTURE_SHA256,
        "canonical synthetic fixture hash changed",
    )
    return validated


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _read_hmac_key(path: Path) -> bytes:
    _require(not _path_is_within(path, REPOSITORY_ROOT), "HMAC key must stay outside the repository")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise SyntheticGPTContractError(f"cannot stat HMAC key file: {exc}") from exc
    _require(
        MINIMUM_HMAC_KEY_BYTES <= size <= MAXIMUM_HMAC_KEY_BYTES,
        "HMAC key file size is outside the frozen bound",
    )
    try:
        key = path.read_bytes()
    except OSError as exc:
        raise SyntheticGPTContractError(f"cannot read HMAC key file: {exc}") from exc
    _require(len(key) >= MINIMUM_HMAC_KEY_BYTES, "HMAC key is shorter than 32 bytes")
    _require(len(set(key)) >= 16, "HMAC key has insufficient byte diversity")
    return key


def _request_cache_entry(
    record: Mapping[str, Any],
    *,
    key: bytes,
    config: Mapping[str, Any],
    config_sha256: str,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    user_content = {
        "current_text": record["current_text"],
        "history_texts": list(record["history_texts"]),
    }
    authenticated_request = {
        "analysis_id": ANALYSIS_ID,
        "protocol_id": PROTOCOL_ID,
        "config_sha256": config_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "split_manifest_id": SPLIT_MANIFEST_ID,
        "split_manifest_schema": SPLIT_MANIFEST_SCHEMA,
        "split_manifest_status": SPLIT_MANIFEST_STATUS,
        "split_protocol_id": SPLIT_PROTOCOL_ID,
        "canonical_fixture_sha256": CANONICAL_FIXTURE_SHA256,
        "model": FROZEN_MODEL,
        "model_snapshot": FROZEN_MODEL_SNAPSHOT,
        "endpoint": FROZEN_ENDPOINT,
        "prompt_sha256": FROZEN_PROMPT_SHA256,
        "response_schema_sha256": FROZEN_RESPONSE_SCHEMA_SHA256,
        "reasoning": dict(config["provider_contract"]["reasoning"]),
        "store": False,
        "tools_enabled": False,
        "batch_enabled": False,
        "synthetic": True,
        "user_content": user_content,
    }
    digest = hmac.new(key, _canonical_json_bytes(authenticated_request), hashlib.sha256).hexdigest()
    joined_text = "\N{RECORD SEPARATOR}".join(
        [str(record["current_text"]), *[str(item) for item in record["history_texts"]]]
    )
    return {
        "request": {
            "hmac_sha256": digest,
            "input_character_count": len(joined_text),
            "input_utf8_bytes": len(joined_text.encode("utf-8")),
            "synthetic": True,
        },
        "response": dict(record["mock_response"]),
    }


def _write_json_once(path: Path, payload: Mapping[str, Any], *, label: str) -> None:
    _require(not path.exists(), f"{label} already exists; overwrite is forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise SyntheticGPTContractError(f"{label} already exists; overwrite is forbidden") from exc
    except OSError as exc:
        raise SyntheticGPTContractError(f"cannot create {label}: {exc}") from exc


def _valid_sha256(value: Any, *, name: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return value


def validate_public_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_config_sha256: str,
    expected_split_manifest_sha256: str,
    expected_private_cache_sha256: str,
) -> None:
    """Validate the complete aggregate-only receipt, including every nested field."""

    for name, value in (
        ("expected_config_sha256", expected_config_sha256),
        ("expected_split_manifest_sha256", expected_split_manifest_sha256),
        ("expected_private_cache_sha256", expected_private_cache_sha256),
    ):
        _valid_sha256(value, name=name)
    _require(
        expected_config_sha256 == FROZEN_CONFIG_SHA256,
        "expected config hash differs from the frozen v1 config",
    )
    _require(
        expected_split_manifest_sha256 == FROZEN_SPLIT_MANIFEST_SHA256,
        "expected split manifest hash differs from the frozen v1 manifest",
    )
    _exact_keys(
        receipt,
        {
            "schema_version",
            "protocol_id",
            "analysis_id",
            "status",
            "frozen_contract",
            "artifact_hashes",
            "aggregate",
            "execution_audit",
            "public_content_audit",
        },
        "public receipt",
    )
    _require(receipt.get("schema_version") == PUBLIC_RECEIPT_SCHEMA, "receipt schema changed")
    _require(receipt.get("protocol_id") == PROTOCOL_ID, "receipt protocol changed")
    _require(receipt.get("analysis_id") == ANALYSIS_ID, "receipt analysis id changed")
    _require(
        receipt.get("status") == "synthetic_only_contract_exercised_no_api_calls",
        "receipt status changed",
    )

    frozen = _mapping(receipt.get("frozen_contract"), "frozen_contract")
    _exact_keys(
        frozen,
        {
            "model",
            "model_snapshot",
            "prompt_sha256",
            "response_schema_sha256",
            "config_sha256",
            "split_manifest_sha256",
            "split_manifest_id",
            "split_manifest_schema",
            "split_manifest_status",
            "split_protocol_id",
        },
        "frozen_contract",
    )
    _require(frozen.get("model") == FROZEN_MODEL, "receipt model changed")
    _require(frozen.get("model_snapshot") == FROZEN_MODEL_SNAPSHOT, "receipt snapshot changed")
    _require(frozen.get("prompt_sha256") == FROZEN_PROMPT_SHA256, "receipt prompt hash changed")
    _require(
        frozen.get("response_schema_sha256") == FROZEN_RESPONSE_SCHEMA_SHA256,
        "receipt response schema hash changed",
    )
    _require(frozen.get("config_sha256") == expected_config_sha256, "receipt config hash changed")
    _require(
        frozen.get("split_manifest_sha256") == expected_split_manifest_sha256,
        "receipt split manifest hash changed",
    )
    _require(frozen.get("split_manifest_id") == SPLIT_MANIFEST_ID, "receipt manifest id changed")
    _require(
        frozen.get("split_manifest_schema") == SPLIT_MANIFEST_SCHEMA,
        "receipt manifest schema changed",
    )
    _require(
        frozen.get("split_manifest_status") == SPLIT_MANIFEST_STATUS,
        "receipt manifest status changed",
    )
    _require(
        frozen.get("split_protocol_id") == SPLIT_PROTOCOL_ID,
        "receipt split protocol changed",
    )
    for field in ("prompt_sha256", "response_schema_sha256", "config_sha256", "split_manifest_sha256"):
        _valid_sha256(frozen.get(field), name=f"frozen_contract.{field}")

    artifacts = _mapping(receipt.get("artifact_hashes"), "artifact_hashes")
    _exact_keys(
        artifacts,
        {"canonical_fixture_sha256", "private_cache_sha256"},
        "artifact_hashes",
    )
    _require(
        artifacts.get("canonical_fixture_sha256") == CANONICAL_FIXTURE_SHA256,
        "receipt canonical fixture hash changed",
    )
    _require(
        artifacts.get("private_cache_sha256") == expected_private_cache_sha256,
        "receipt private cache hash changed",
    )
    _valid_sha256(artifacts.get("canonical_fixture_sha256"), name="canonical_fixture_sha256")
    _valid_sha256(artifacts.get("private_cache_sha256"), name="private_cache_sha256")

    canonical_records = canonical_synthetic_fixture()["records"]
    expected_aggregate = {
        "synthetic_record_count": len(canonical_records),
        "synthetic_history_item_count": sum(len(row["history_texts"]) for row in canonical_records),
        "total_input_character_count": sum(
            len("\N{RECORD SEPARATOR}".join([row["current_text"], *row["history_texts"]]))
            for row in canonical_records
        ),
        "total_input_utf8_bytes": sum(
            len(
                "\N{RECORD SEPARATOR}"
                .join([row["current_text"], *row["history_texts"]])
                .encode("utf-8")
            )
            for row in canonical_records
        ),
    }
    aggregate = _mapping(receipt.get("aggregate"), "aggregate")
    _exact_keys(aggregate, set(expected_aggregate), "aggregate")
    _require(
        all(type(aggregate.get(field)) is int for field in expected_aggregate),
        "aggregate values must be integers",
    )
    _require(aggregate == expected_aggregate, "aggregate receipt values changed")

    audit = _mapping(receipt.get("execution_audit"), "execution_audit")
    _exact_keys(
        audit,
        {
            "api_calls",
            "network_calls",
            "openai_sdk_imported",
            "real_sidecars_opened",
            "real_dataset_text_rows_opened",
        },
        "execution_audit",
    )
    _require(
        audit
        == {
            "api_calls": 0,
            "network_calls": 0,
            "openai_sdk_imported": False,
            "real_sidecars_opened": 0,
            "real_dataset_text_rows_opened": 0,
        },
        "execution audit must attest a fully offline synthetic run",
    )
    content = _mapping(receipt.get("public_content_audit"), "public_content_audit")
    _exact_keys(
        content,
        {
            "contains_prompt_or_input_text",
            "contains_row_level_hmacs",
            "contains_mock_responses",
            "contains_row_or_dataset_identifiers",
            "contains_private_paths_or_key_material",
        },
        "public_content_audit",
    )
    _require(
        content
        == {
            "contains_prompt_or_input_text": False,
            "contains_row_level_hmacs": False,
            "contains_mock_responses": False,
            "contains_row_or_dataset_identifiers": False,
            "contains_private_paths_or_key_material": False,
        },
        "public receipt content boundary changed",
    )


def run_synthetic_fixture_contract(
    *,
    config_path: Path,
    split_manifest_path: Path,
    synthetic_fixture_path: Path,
    hmac_key_path: Path,
    private_cache_path: Path,
    public_receipt_path: Path,
    explicit_synthetic_acknowledgement: bool = False,
) -> dict[str, Any]:
    """Exercise the contract with mock responses and zero model/API execution."""

    _require(
        explicit_synthetic_acknowledgement is True,
        "explicit synthetic-only acknowledgement is required before reading a fixture",
    )
    _require(REPOSITORY_ROOT.is_dir(), "physically derived repository root does not exist")
    expected_config_path = (
        REPOSITORY_ROOT / "experiment" / "configs" / "synthetic_gpt_text_baseline_v1.json"
    ).resolve()
    expected_manifest_path = (
        REPOSITORY_ROOT / "experiment" / "configs" / "carma_split_manifest_v1.json"
    ).resolve()
    _require(config_path.resolve() == expected_config_path, "config path is not the frozen repository config")
    _require(
        split_manifest_path.resolve() == expected_manifest_path,
        "split manifest path is not the frozen repository manifest",
    )
    _require(
        not _path_is_within(private_cache_path, REPOSITORY_ROOT),
        "private response cache must stay outside the repository",
    )
    _require(
        private_cache_path.resolve() != public_receipt_path.resolve(),
        "private cache and public receipt paths must differ",
    )
    _require(not private_cache_path.exists(), "private response cache already exists; overwrite is forbidden")
    _require(not public_receipt_path.exists(), "public receipt already exists; overwrite is forbidden")
    _validate_fixture_path(synthetic_fixture_path)

    config_sha256 = sha256_file(config_path)
    split_manifest_sha256 = sha256_file(split_manifest_path)
    _require(config_sha256 == FROZEN_CONFIG_SHA256, "config file hash differs from frozen v1")
    _require(
        split_manifest_sha256 == FROZEN_SPLIT_MANIFEST_SHA256,
        "split manifest file hash differs from frozen v1",
    )
    config = load_json_object(
        config_path,
        name="synthetic GPT config",
        maximum_bytes=MAXIMUM_CONFIG_FILE_BYTES,
    )
    validate_contract_config(config)
    _require(
        split_manifest_sha256 == config["split_manifest_contract"]["sha256"],
        "split manifest file hash differs from the frozen config",
    )
    split_manifest = load_json_object(
        split_manifest_path,
        name="split manifest",
        maximum_bytes=MAXIMUM_SPLIT_MANIFEST_FILE_BYTES,
    )
    validate_external_llm_boundary(split_manifest)
    fixture = load_json_object(
        synthetic_fixture_path,
        name="explicit synthetic fixture",
        maximum_bytes=MAXIMUM_FIXTURE_FILE_BYTES,
    )
    records = validate_synthetic_fixture(fixture)
    _require(
        canonical_sha256(fixture) in config["context_contract"]["canonical_fixture_sha256_allowlist"],
        "fixture hash is not in the frozen config allowlist",
    )
    key = _read_hmac_key(hmac_key_path)

    entries = [
        _request_cache_entry(
            record,
            key=key,
            config=config,
            config_sha256=config_sha256,
            split_manifest_sha256=split_manifest_sha256,
        )
        for record in records
    ]
    hmacs = [str(entry["request"]["hmac_sha256"]) for entry in entries]
    _require(len(set(hmacs)) == len(hmacs), "fixture contains duplicate authenticated requests")

    private_payload: dict[str, Any] = {
        "schema_version": PRIVATE_CACHE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "analysis_id": ANALYSIS_ID,
        "status": "synthetic_mock_responses_cached_write_once",
        "canonical_fixture_sha256": CANONICAL_FIXTURE_SHA256,
        "frozen_contract": {
            "config_sha256": config_sha256,
            "split_manifest_sha256": split_manifest_sha256,
            "split_manifest_id": SPLIT_MANIFEST_ID,
            "split_manifest_schema": SPLIT_MANIFEST_SCHEMA,
            "split_manifest_status": SPLIT_MANIFEST_STATUS,
            "split_protocol_id": SPLIT_PROTOCOL_ID,
            "model": FROZEN_MODEL,
            "model_snapshot": FROZEN_MODEL_SNAPSHOT,
            "endpoint": FROZEN_ENDPOINT,
            "prompt_sha256": FROZEN_PROMPT_SHA256,
            "response_schema_sha256": FROZEN_RESPONSE_SCHEMA_SHA256,
            "reasoning": dict(config["provider_contract"]["reasoning"]),
            "store": False,
            "tools_enabled": False,
            "batch_enabled": False,
        },
        "entries": entries,
        "content_audit": {
            "raw_request_text_stored": False,
            "request_fields_exactly_hmac_lengths_and_synthetic_marker": True,
            "responses_are_caller_supplied_synthetic_mocks": True,
            "contains_real_dataset_content": False,
        },
    }
    _write_json_once(private_cache_path, private_payload, label="private response cache")

    total_characters = sum(int(entry["request"]["input_character_count"]) for entry in entries)
    total_utf8_bytes = sum(int(entry["request"]["input_utf8_bytes"]) for entry in entries)
    total_history_items = sum(len(record["history_texts"]) for record in records)
    receipt: dict[str, Any] = {
        "schema_version": PUBLIC_RECEIPT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "analysis_id": ANALYSIS_ID,
        "status": "synthetic_only_contract_exercised_no_api_calls",
        "frozen_contract": {
            "model": FROZEN_MODEL,
            "model_snapshot": FROZEN_MODEL_SNAPSHOT,
            "prompt_sha256": FROZEN_PROMPT_SHA256,
            "response_schema_sha256": FROZEN_RESPONSE_SCHEMA_SHA256,
            "config_sha256": config_sha256,
            "split_manifest_sha256": split_manifest_sha256,
            "split_manifest_id": SPLIT_MANIFEST_ID,
            "split_manifest_schema": SPLIT_MANIFEST_SCHEMA,
            "split_manifest_status": SPLIT_MANIFEST_STATUS,
            "split_protocol_id": SPLIT_PROTOCOL_ID,
        },
        "artifact_hashes": {
            "canonical_fixture_sha256": CANONICAL_FIXTURE_SHA256,
            "private_cache_sha256": sha256_file(private_cache_path),
        },
        "aggregate": {
            "synthetic_record_count": len(entries),
            "synthetic_history_item_count": total_history_items,
            "total_input_character_count": total_characters,
            "total_input_utf8_bytes": total_utf8_bytes,
        },
        "execution_audit": {
            "api_calls": 0,
            "network_calls": 0,
            "openai_sdk_imported": False,
            "real_sidecars_opened": 0,
            "real_dataset_text_rows_opened": 0,
        },
        "public_content_audit": {
            "contains_prompt_or_input_text": False,
            "contains_row_level_hmacs": False,
            "contains_mock_responses": False,
            "contains_row_or_dataset_identifiers": False,
            "contains_private_paths_or_key_material": False,
        },
    }
    validate_public_receipt(
        receipt,
        expected_config_sha256=config_sha256,
        expected_split_manifest_sha256=split_manifest_sha256,
        expected_private_cache_sha256=sha256_file(private_cache_path),
    )
    _write_json_once(public_receipt_path, receipt, label="public aggregate receipt")
    return receipt
