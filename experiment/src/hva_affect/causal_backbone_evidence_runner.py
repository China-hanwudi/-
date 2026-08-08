"""Leakage-closed staged inputs for causal-backbone evidence production.

Stage A is deliberately a non-performance preflight.  It verifies one of the
two registered v2 role-sidecar manifests, materialises only the fit feature and
label archives, hashes the model-selection feature and label archives as opaque
bytes, and writes an aggregate, write-once receipt.  No model is trained here.

The same receipt is the capability required by Stage B.  Selection features
may be materialised only after every byte/config/code/environment hash in the
receipt is reverified.  Selection labels have no Stage-B API in this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

# Frozen protocol literals are repeated here intentionally: fit-preflight must
# remain importable without torch/sklearn/pandas or any training module.  Exact
# schema tests protect these copies from drifting from the registered v2 files.
FIT_ROLE = "base_and_utility_fit"
SELECTION_ROLE = "model_selection"
PRODUCER_CACHE_SCHEMA = "carma_causal_backbone_open_role_private_v2"
INDEPENDENT_CURRENT_ONLY_PROTOCOL = (
    "independently_trained_same_architecture_history_stripped_all_masks_empty_v1"
)
EXPECTED_SEEDS = (17, 29, 43, 71, 101)
ENDPOINT_CONTEXT_NAMES = ("current_only", "all_history")
UTILITY_CONTEXT_NAMES = ("s", "s_plus_candidate", "t", "t_minus_candidate")

EMOTIONTALK_PROTOCOL = "emotiontalk_role_separated_sidecars_v2"
EMOTIONTALK_FEATURE_SCHEMA = "emotiontalk_role_feature_sidecar_v2"
EMOTIONTALK_LABEL_SCHEMA = "emotiontalk_role_label_sidecar_v2"
EMOTIONTALK_MANIFEST_SCHEMA = "emotiontalk_role_sidecar_manifest_v2"
EMOTIONTALK_LABEL_NAMES = (
    "neutral", "happy", "sad", "angry", "surprised", "disgusted", "fearful"
)
MELD_PROTOCOL = "meld_multimodal_role_sidecars_v2"
MELD_SIDECAR_SCHEMA = "meld_multimodal_role_sidecar_v2"
MELD_MANIFEST_SCHEMA = "meld_multimodal_role_sidecar_manifest_v2"
MELD_LABEL_NAMES = (
    "neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"
)

FIT_PREFLIGHT_RECEIPT_SCHEMA = "carma_causal_evidence_fit_preflight_receipt_v1"
FIT_ONLY_PRODUCER_VIEW_SCHEMA = "carma_causal_evidence_fit_producer_view_v1"
CURRENT_ONLY_FIT_ARTIFACT_SCHEMA = "carma_independent_current_only_fit_private_v1"
UTILITY_OOF_SCORE_SCHEMA = "carma_bidirectional_utility_oof_scores_private_v1"
CHECKPOINT_MANIFEST_SCHEMA = "carma_current_only_checkpoint_manifest_v1"

_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class StageAContractError(ValueError):
    """Raised when Stage-A isolation or lineage cannot be proved."""


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise StageAContractError(f"required file is missing: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    digest = str(value).lower()
    if _SHA256.fullmatch(digest) is None:
        raise StageAContractError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _single_text(value: np.ndarray, field: str) -> str:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in {"U", "S"}:
        raise StageAContractError(f"{field} must contain exactly one string")
    return str(array.reshape(-1)[0])


def _integer_vector(value: np.ndarray, field: str, *, unique: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise StageAContractError(f"{field} must be a one-dimensional integer array")
    result = array.astype(np.int64, copy=True)
    if np.any(result < 0):
        raise StageAContractError(f"{field} contains a negative value")
    if unique and len(set(result.tolist())) != len(result):
        raise StageAContractError(f"{field} must contain unique values")
    return result


def _probability(value: np.ndarray, field: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.floating):
        raise StageAContractError(f"{field} must have shape {shape}")
    result = array.astype(np.float32, copy=True)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise StageAContractError(f"{field} contains an invalid probability")
    if not np.allclose(result.sum(axis=-1), 1.0, rtol=1.0e-5, atol=1.0e-6):
        raise StageAContractError(f"{field} probability rows do not sum to one")
    return result


def _safe_named_paths(values: Mapping[str, str | Path], field: str) -> dict[str, Path]:
    if not values:
        raise StageAContractError(f"{field} must not be empty")
    result: dict[str, Path] = {}
    for raw_name, raw_path in sorted(values.items()):
        name = str(raw_name)
        if _SAFE_NAME.fullmatch(name) is None or name in result:
            raise StageAContractError(f"{field} contains an unsafe or duplicate name")
        path = Path(raw_path)
        if not path.is_file():
            raise StageAContractError(f"{field}.{name} is not a file")
        result[name] = path
    return result


def _named_file_hashes(values: Mapping[str, str | Path], field: str) -> dict[str, str]:
    paths = _safe_named_paths(values, field)
    return {name: _file_sha256(path) for name, path in paths.items()}


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    manifest_schema: str
    protocol: str
    status: str
    manifest_fields: frozenset[str]
    role_record_fields: frozenset[str]
    group_count_field: str
    feature_schema: str
    label_schema: str
    feature_fields: frozenset[str]
    label_fields: frozenset[str]
    label_order: tuple[str, ...]


_EMOTIONTALK_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "status",
        "dataset_id",
        "split_protocol_id",
        "label_order",
        "source_contract",
        "seal_contract",
        "roles",
        "config_sha256",
        "public_content_audit",
    }
)
_MELD_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "status",
        "dataset_id",
        "split_protocol_id",
        "label_order",
        "claim_boundary",
        "source_contract",
        "feature_contract",
        "seal_contract",
        "roles",
        "config_sha256",
        "public_content_audit",
    }
)
_EMOTIONTALK_MANIFEST_ROLE_FIELDS = frozenset(
    {
        "feature_filename", "label_filename", "rows", "groups",
        "history_eligible_rows", "audio_dimension", "video_dimension",
        "feature_sha256", "label_sha256", "row_alignment_sha256",
    }
)
_EMOTIONTALK_FEATURE_FIELDS = frozenset(
    {
        "schema_version", "dataset_id", "role", "split_protocol_id",
        "row_alignment_sha256", "opaque_row_hashes", "opaque_group_hashes",
        "speaker_tokens", "turn_ids", "protocol_row_ids", "role_buckets",
        "texts", "audio_features", "video_features",
        "source_feature_config_sha256",
    }
)
_EMOTIONTALK_LABEL_FIELDS = frozenset(
    {
        "schema_version", "dataset_id", "role", "split_protocol_id",
        "row_alignment_sha256", "labels", "source_label_sha256",
    }
)
_MELD_ROLE_FIELDS = frozenset(
    {
        "feature_filename",
        "label_filename",
        "rows",
        "dialogues",
        "history_eligible_rows",
        "audio_dimension",
        "video_dimension",
        "feature_sha256",
        "label_sha256",
        "row_alignment_sha256",
    }
)
_MELD_FEATURE_FIELDS = frozenset(
    {
        "schema_version",
        "role",
        "row_alignment_sha256",
        "utterances",
        "audio_mean_std",
        "video_mean_std",
        "dialogue_codes",
        "speaker_codes",
        "utterance_order",
        "protocol_row_ids",
    }
)
_MELD_LABEL_FIELDS = frozenset(
    {"schema_version", "role", "row_alignment_sha256", "labels"}
)

_SPECS = {
    "EmotionTalk": DatasetSpec(
        dataset="EmotionTalk",
        manifest_schema=EMOTIONTALK_MANIFEST_SCHEMA,
        protocol=EMOTIONTALK_PROTOCOL,
        status="strict_open_role_feature_and_label_sidecars_created_and_hashed",
        manifest_fields=_EMOTIONTALK_MANIFEST_FIELDS,
        role_record_fields=_EMOTIONTALK_MANIFEST_ROLE_FIELDS,
        group_count_field="groups",
        feature_schema=EMOTIONTALK_FEATURE_SCHEMA,
        label_schema=EMOTIONTALK_LABEL_SCHEMA,
        feature_fields=_EMOTIONTALK_FEATURE_FIELDS,
        label_fields=_EMOTIONTALK_LABEL_FIELDS,
        label_order=tuple(str(value) for value in EMOTIONTALK_LABEL_NAMES),
    ),
    "MELD": DatasetSpec(
        dataset="MELD",
        manifest_schema=MELD_MANIFEST_SCHEMA,
        protocol=MELD_PROTOCOL,
        status="role_separated_train_sidecars_created_and_hashed",
        manifest_fields=_MELD_MANIFEST_FIELDS,
        role_record_fields=_MELD_ROLE_FIELDS,
        group_count_field="dialogues",
        feature_schema=MELD_SIDECAR_SCHEMA,
        label_schema=MELD_SIDECAR_SCHEMA,
        feature_fields=_MELD_FEATURE_FIELDS,
        label_fields=_MELD_LABEL_FIELDS,
        label_order=MELD_LABEL_NAMES,
    ),
}


@dataclass(frozen=True)
class SidecarRecord:
    role: str
    feature_path: Path
    label_path: Path
    feature_sha256: str
    label_sha256: str
    row_alignment_sha256: str
    rows: int
    groups: int
    history_eligible_rows: int
    audio_dimension: int
    video_dimension: int


@dataclass(frozen=True)
class HashedSidecarSet:
    dataset: str
    spec: DatasetSpec
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, object]
    fit: SidecarRecord
    selection: SidecarRecord


def _read_manifest_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageAContractError(f"cannot read sidecar manifest: {error}") from error
    if not isinstance(value, dict):
        raise StageAContractError("sidecar manifest root must be a mapping")
    return value


def _plain_npz_filename(value: object, expected: str, field: str) -> str:
    name = str(value)
    if name != expected or Path(name).name != name or not name.endswith(".npz"):
        raise StageAContractError(f"{field} filename changed")
    return name


def _validate_manifest_contract(manifest: Mapping[str, object], spec: DatasetSpec) -> None:
    if set(manifest) != set(spec.manifest_fields):
        raise StageAContractError("sidecar manifest root schema changed")
    if manifest.get("schema_version") != spec.manifest_schema:
        raise StageAContractError("sidecar manifest schema version changed")
    if manifest.get("protocol") != spec.protocol:
        raise StageAContractError("sidecar manifest protocol changed")
    if manifest.get("status") != spec.status:
        raise StageAContractError("sidecar manifest is not a completed v2 artifact")
    if manifest.get("dataset_id") != spec.dataset:
        raise StageAContractError("sidecar manifest dataset changed")
    if manifest.get("split_protocol_id") != "scu_set_exploration_v1":
        raise StageAContractError("sidecar split protocol changed")
    if tuple(str(value) for value in manifest.get("label_order", ())) != spec.label_order:
        raise StageAContractError("sidecar label order changed")
    _require_sha256(manifest.get("config_sha256"), "manifest.config_sha256")
    roles = manifest.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != {FIT_ROLE, SELECTION_ROLE}:
        raise StageAContractError("manifest must expose exactly the two open roles")

    public_audit = manifest.get("public_content_audit")
    expected_public_fields = (
        {
            "contains_labels_or_class_counts",
            "contains_row_group_or_speaker_identifiers",
            "contains_transcripts_or_embeddings",
            "contains_private_absolute_paths",
        }
        if spec.dataset == "EmotionTalk"
        else {
            "contains_labels_or_class_counts",
            "contains_utterances_or_embeddings",
            "contains_dialogue_speaker_or_row_identifiers",
            "contains_private_absolute_paths",
        }
    )
    if not isinstance(public_audit, Mapping) or set(public_audit) != expected_public_fields:
        raise StageAContractError("manifest public-content audit is missing")
    if any(value is not False for value in public_audit.values()):
        raise StageAContractError("manifest public-content audit exposes private material")
    seal = manifest.get("seal_contract")
    if not isinstance(seal, Mapping):
        raise StageAContractError("manifest seal contract is missing")
    if seal.get("allow_pickle_required_to_load_sidecars") is not False:
        raise StageAContractError("sidecars must be readable with allow_pickle=False")
    if seal.get("open_role_runner_may_load_only") != [FIT_ROLE, SELECTION_ROLE]:
        raise StageAContractError("manifest open-role allowlist changed")
    if spec.dataset == "EmotionTalk":
        expected_seal = {
            "model_runner_opens_upstream_media_npz_or_transcription",
            "open_role_runner_may_load_only",
            "calibration_holdout_validation_test_sidecars_created",
            "allow_pickle_required_to_load_sidecars",
        }
        if set(seal) != expected_seal:
            raise StageAContractError("EmotionTalk seal schema changed")
        if (
            seal.get("model_runner_opens_upstream_media_npz_or_transcription") is not False
            or seal.get("calibration_holdout_validation_test_sidecars_created") is not False
        ):
            raise StageAContractError("EmotionTalk physical seal contract changed")
        source = manifest.get("source_contract")
        expected = {
            "label_archive",
            "media_features",
            "transcription",
            "feature_config_sha256",
            "trusted_source_boundary_only",
            "validation_or_test_label_payload_opened",
        }
        if not isinstance(source, Mapping) or set(source) != expected:
            raise StageAContractError("EmotionTalk source contract changed")
        for field in ("label_archive", "media_features", "transcription", "feature_config_sha256"):
            _require_sha256(source[field], f"source_contract.{field}")
        if source["trusted_source_boundary_only"] is not True or source[
            "validation_or_test_label_payload_opened"
        ] is not False:
            raise StageAContractError("EmotionTalk source boundary changed")
    else:
        expected_seal = {
            "features_and_labels_are_in_separate_archives",
            "each_role_has_a_separate_label_archive",
            "allow_pickle_required_to_load_sidecars",
            "open_role_runner_may_load_only",
            "calibration_and_internal_holdout_remain_unopened_by_model_runners",
        }
        if set(seal) != expected_seal:
            raise StageAContractError("MELD seal schema changed")
        if (
            seal.get("features_and_labels_are_in_separate_archives") is not True
            or seal.get("each_role_has_a_separate_label_archive") is not True
            or seal.get("calibration_and_internal_holdout_remain_unopened_by_model_runners")
            is not True
        ):
            raise StageAContractError("MELD physical seal contract changed")
        source = manifest.get("source_contract")
        expected_source = {
            "train_csv_sha256",
            "train_pickle_sha256",
            "official_csv_is_authoritative_label_source",
            "embedded_pickle_label_used_for_training_or_metrics",
            "embedded_pickle_label_consistency_checked_by_trusted_custodian",
            "embedded_pickle_label_mismatch_statistics_exposed",
            "missing_feature_rows",
            "extra_feature_rows",
        }
        if not isinstance(source, Mapping) or set(source) != expected_source:
            raise StageAContractError("MELD source contract is missing")
        for field in ("train_csv_sha256", "train_pickle_sha256"):
            _require_sha256(source.get(field), f"source_contract.{field}")
        feature = manifest.get("feature_contract")
        expected_feature = {
            "audio_mean_std_columns",
            "video_mean_std_columns",
            "numeric_dtype",
            "strict_same_dialogue_same_speaker_past_history_supported",
            "protocol_row_identity",
        }
        if (
            not isinstance(feature, Mapping)
            or set(feature) != expected_feature
            or feature.get("numeric_dtype") != "float32"
            or feature.get("strict_same_dialogue_same_speaker_past_history_supported")
            is not True
        ):
            raise StageAContractError("MELD feature contract changed")


def hash_open_role_sidecars(
    *,
    dataset: str,
    sidecar_dir: str | Path,
    manifest_path: str | Path,
) -> HashedSidecarSet:
    """Verify all four role files by bytes without deserialising any NPZ."""

    if dataset not in _SPECS:
        raise StageAContractError("dataset must be EmotionTalk or MELD")
    spec = _SPECS[dataset]
    directory = Path(sidecar_dir)
    manifest_file = Path(manifest_path)
    manifest = _read_manifest_json(manifest_file)
    _validate_manifest_contract(manifest, spec)
    roles = manifest["roles"]
    assert isinstance(roles, Mapping)
    records: dict[str, SidecarRecord] = {}
    # Hash the two selection payloads first and keep them opaque throughout
    # Stage A; only after all four byte hashes match may fit materialisation run.
    for role in (SELECTION_ROLE, FIT_ROLE):
        raw = roles[role]
        if not isinstance(raw, Mapping) or set(raw) != set(spec.role_record_fields):
            raise StageAContractError(f"manifest role record changed: {role}")
        feature_name = _plain_npz_filename(
            raw.get("feature_filename"), f"features_{role}.npz", f"{role}.feature"
        )
        label_name = _plain_npz_filename(
            raw.get("label_filename"), f"labels_{role}.npz", f"{role}.label"
        )
        feature_path = directory / feature_name
        label_path = directory / label_name
        observed_feature = _file_sha256(feature_path)
        observed_label = _file_sha256(label_path)
        declared_feature = _require_sha256(raw.get("feature_sha256"), f"{role}.feature_sha256")
        declared_label = _require_sha256(raw.get("label_sha256"), f"{role}.label_sha256")
        if observed_feature != declared_feature or observed_label != declared_label:
            raise StageAContractError(f"{role} sidecar byte hash differs from manifest")
        rows = int(raw.get("rows", -1))
        groups = int(raw.get(spec.group_count_field, -1))
        history_eligible = int(raw.get("history_eligible_rows", -1))
        audio_dim = int(raw.get("audio_dimension", -1))
        video_dim = int(raw.get("video_dimension", -1))
        if (
            rows < 1
            or groups < 1
            or groups > rows
            or history_eligible < 0
            or history_eligible > rows
            or audio_dim < 1
            or video_dim < 1
        ):
            raise StageAContractError(f"{role} manifest counts/dimensions are invalid")
        records[role] = SidecarRecord(
            role=role,
            feature_path=feature_path,
            label_path=label_path,
            feature_sha256=observed_feature,
            label_sha256=observed_label,
            row_alignment_sha256=_require_sha256(
                raw.get("row_alignment_sha256"), f"{role}.row_alignment_sha256"
            ),
            rows=rows,
            groups=groups,
            history_eligible_rows=history_eligible,
            audio_dimension=audio_dim,
            video_dimension=video_dim,
        )
    return HashedSidecarSet(
        dataset=dataset,
        spec=spec,
        manifest_path=manifest_file,
        manifest_sha256=_file_sha256(manifest_file),
        manifest=manifest,
        fit=records[FIT_ROLE],
        selection=records[SELECTION_ROLE],
    )


@dataclass(frozen=True)
class FitRoleView:
    dataset: str
    label_order: tuple[str, ...]
    texts: tuple[str, ...]
    audio: np.ndarray
    video: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    speakers: np.ndarray
    turns: np.ndarray
    protocol_row_ids: np.ndarray
    histories: tuple[tuple[int, ...], ...]
    array_hashes: Mapping[str, str]
    contract_sha256: str

    @property
    def rows(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class SelectionFeatureView:
    dataset: str
    texts: tuple[str, ...]
    audio: np.ndarray
    video: np.ndarray
    groups: np.ndarray
    speakers: np.ndarray
    turns: np.ndarray
    protocol_row_ids: np.ndarray
    labels_materialized: bool = False


def _load_npz_exact(path: Path, fields: frozenset[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(fields):
            raise StageAContractError(f"sidecar field schema changed: {path.name}")
        return {name: np.asarray(archive[name]) for name in archive.files}


def _strict_histories(
    groups: np.ndarray,
    speakers: np.ndarray,
    turns: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    values: list[tuple[int, ...]] = []
    for query in range(len(groups)):
        candidates = np.flatnonzero(
            (groups == groups[query])
            & (speakers == speakers[query])
            & (turns < turns[query])
        )
        values.append(
            tuple(sorted(candidates.tolist(), key=lambda row: (int(turns[row]), row)))
        )
    return tuple(values)


def _validate_normalized_rows(
    *,
    record: SidecarRecord,
    texts: np.ndarray,
    audio: np.ndarray,
    video: np.ndarray,
    groups: np.ndarray,
    speakers: np.ndarray,
    turns: np.ndarray,
    protocol_rows: np.ndarray,
    labels: np.ndarray | None,
) -> None:
    rows = record.rows
    if texts.shape != (rows,) or texts.dtype.kind not in {"U", "S"}:
        raise StageAContractError("text rows are not a one-dimensional string array")
    if audio.dtype != np.float32 or video.dtype != np.float32:
        raise StageAContractError("modality sidecars must preserve float32 dtype")
    if audio.shape != (rows, record.audio_dimension) or video.shape != (
        rows,
        record.video_dimension,
    ):
        raise StageAContractError("modality dimensions differ from manifest")
    if not np.isfinite(audio).all() or not np.isfinite(video).all():
        raise StageAContractError("modality sidecar contains non-finite values")
    for name, value in (
        ("groups", groups),
        ("speakers", speakers),
        ("turns", turns),
        ("protocol_row_ids", protocol_rows),
    ):
        if np.asarray(value).shape != (rows,):
            raise StageAContractError(f"{name} is not row-aligned")
    if not np.issubdtype(turns.dtype, np.integer) or not np.issubdtype(
        protocol_rows.dtype, np.integer
    ):
        raise StageAContractError("turn/protocol row arrays must be integer")
    if np.any(turns < 0) or np.any(protocol_rows < 0):
        raise StageAContractError("turn/protocol row array contains a negative value")
    if len(set(protocol_rows.astype(np.int64).tolist())) != rows:
        raise StageAContractError("protocol row identities must be unique")
    if len(set(groups.astype(str).tolist())) != record.groups:
        raise StageAContractError("group count differs from manifest")
    if labels is not None:
        if labels.shape != (rows,) or not np.issubdtype(labels.dtype, np.integer):
            raise StageAContractError("labels must be a row-aligned integer array")


def _materialize_fit_role(sidecars: HashedSidecarSet) -> FitRoleView:
    """Open exactly the fit feature and fit label payloads."""

    spec = sidecars.spec
    record = sidecars.fit
    feature = _load_npz_exact(record.feature_path, spec.feature_fields)
    label = _load_npz_exact(record.label_path, spec.label_fields)
    if spec.dataset == "EmotionTalk":
        for payload, schema in ((feature, spec.feature_schema), (label, spec.label_schema)):
            if _single_text(payload["schema_version"], "schema_version") != schema:
                raise StageAContractError("EmotionTalk sidecar schema changed")
            if _single_text(payload["dataset_id"], "dataset_id") != spec.dataset:
                raise StageAContractError("EmotionTalk sidecar dataset changed")
            if _single_text(payload["role"], "role") != FIT_ROLE:
                raise StageAContractError("non-fit payload entered Stage A")
            if _single_text(payload["split_protocol_id"], "split_protocol_id") != (
                "scu_set_exploration_v1"
            ):
                raise StageAContractError("EmotionTalk sidecar split changed")
        if _single_text(feature["source_feature_config_sha256"], "source_feature_config") != str(
            sidecars.manifest["source_contract"]["feature_config_sha256"]  # type: ignore[index]
        ):
            raise StageAContractError("EmotionTalk feature source hash changed")
        if _single_text(label["source_label_sha256"], "source_label_sha256") != str(
            sidecars.manifest["source_contract"]["label_archive"]  # type: ignore[index]
        ):
            raise StageAContractError("EmotionTalk label source hash changed")
        texts = np.asarray(feature["texts"])
        audio = np.asarray(feature["audio_features"])
        video = np.asarray(feature["video_features"])
        groups = np.asarray(feature["opaque_group_hashes"]).astype(str)
        speakers = np.asarray(feature["speaker_tokens"]).astype(str)
        turns = np.asarray(feature["turn_ids"])
        protocol_rows = np.asarray(feature["protocol_row_ids"])
        labels = np.asarray(label["labels"])
        row_hashes = np.asarray(feature["opaque_row_hashes"]).astype(str)
        if row_hashes.shape != (record.rows,) or len(set(row_hashes.tolist())) != record.rows:
            raise StageAContractError("EmotionTalk opaque row identities changed")
        buckets = np.asarray(feature["role_buckets"])
        if buckets.shape != (record.rows,) or not np.issubdtype(buckets.dtype, np.integer):
            raise StageAContractError("EmotionTalk fit bucket array changed")
        if np.any((buckets < 0) | (buckets > 64)):
            raise StageAContractError("EmotionTalk non-fit bucket entered Stage A")
    else:
        for payload in (feature, label):
            if _single_text(payload["schema_version"], "schema_version") != spec.feature_schema:
                raise StageAContractError("MELD sidecar schema changed")
            if _single_text(payload["role"], "role") != FIT_ROLE:
                raise StageAContractError("non-fit MELD payload entered Stage A")
        texts = np.asarray(feature["utterances"])
        audio = np.asarray(feature["audio_mean_std"])
        video = np.asarray(feature["video_mean_std"])
        groups = np.asarray(feature["dialogue_codes"])
        speakers = np.asarray(feature["speaker_codes"])
        turns = np.asarray(feature["utterance_order"])
        protocol_rows = np.asarray(feature["protocol_row_ids"])
        labels = np.asarray(label["labels"])

    feature_alignment = _single_text(feature["row_alignment_sha256"], "feature_alignment")
    label_alignment = _single_text(label["row_alignment_sha256"], "label_alignment")
    if feature_alignment != record.row_alignment_sha256 or label_alignment != feature_alignment:
        raise StageAContractError("fit feature/label row alignment changed")
    _validate_normalized_rows(
        record=record,
        texts=texts,
        audio=audio,
        video=video,
        groups=groups,
        speakers=speakers,
        turns=turns,
        protocol_rows=protocol_rows,
        labels=labels,
    )
    labels = labels.astype(np.int64, copy=True)
    if np.any((labels < 0) | (labels >= len(spec.label_order))):
        raise StageAContractError("fit label is outside the registered label order")
    turns = turns.astype(np.int64, copy=True)
    protocol_rows = protocol_rows.astype(np.int64, copy=True)
    histories = _strict_histories(groups, speakers, turns)
    if int(sum(bool(value) for value in histories)) != record.history_eligible_rows:
        raise StageAContractError("fit history-eligible count differs from manifest")
    normalized = {
        "texts": np.asarray(texts).astype(str),
        "audio": np.asarray(audio, dtype=np.float32),
        "video": np.asarray(video, dtype=np.float32),
        "labels": labels,
        "groups": np.asarray(groups).astype(str),
        "speakers": np.asarray(speakers).astype(str),
        "turns": turns,
        "protocol_row_ids": protocol_rows,
    }
    hashes = {name: _array_sha256(value) for name, value in normalized.items()}
    hashes["histories"] = _canonical_sha256([list(value) for value in histories])
    contract_payload = {
        "dataset": spec.dataset,
        "role": FIT_ROLE,
        "label_order": list(spec.label_order),
        "row_alignment_sha256": record.row_alignment_sha256,
        "feature_sha256": record.feature_sha256,
        "label_sha256": record.label_sha256,
        "array_hashes": hashes,
        "protocol_index_mapping_sha256": _canonical_sha256(
            [[int(index), int(value)] for index, value in enumerate(protocol_rows)]
        ),
    }
    return FitRoleView(
        dataset=spec.dataset,
        label_order=spec.label_order,
        texts=tuple(str(value) for value in texts),
        audio=normalized["audio"].copy(),
        video=normalized["video"].copy(),
        labels=labels,
        groups=normalized["groups"].copy(),
        speakers=normalized["speakers"].copy(),
        turns=turns,
        protocol_row_ids=protocol_rows,
        histories=histories,
        array_hashes=hashes,
        contract_sha256=_canonical_sha256(contract_payload),
    )


def capture_runtime_environment() -> dict[str, object]:
    """Return a compact, path-free environment record for receipt binding."""

    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def _access_events(sidecars: HashedSidecarSet) -> list[dict[str, object]]:
    return [
        {
            "sequence": 0,
            "role": SELECTION_ROLE,
            "payload": "feature",
            "access": "byte_sha256_only_no_np_load",
            "sha256": sidecars.selection.feature_sha256,
        },
        {
            "sequence": 1,
            "role": SELECTION_ROLE,
            "payload": "label",
            "access": "byte_sha256_only_no_np_load",
            "sha256": sidecars.selection.label_sha256,
        },
        {
            "sequence": 2,
            "role": FIT_ROLE,
            "payload": "feature",
            "access": "allow_pickle_false_materialized",
            "sha256": sidecars.fit.feature_sha256,
        },
        {
            "sequence": 3,
            "role": FIT_ROLE,
            "payload": "label",
            "access": "allow_pickle_false_materialized",
            "sha256": sidecars.fit.label_sha256,
        },
    ]


_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "stage",
        "dataset",
        "claim_boundary",
        "manifest",
        "sidecars",
        "fit_contract",
        "lineage",
        "access_contract",
        "completion_gate",
        "public_artifact_policy",
    }
)


@dataclass(frozen=True)
class FitPreflightResult:
    fit: FitRoleView
    receipt: Mapping[str, object]
    receipt_path: Path
    receipt_sha256: str


def _assert_aggregate_receipt(payload: Mapping[str, object]) -> None:
    forbidden = {
        "labels",
        "texts",
        "utterances",
        "audio",
        "video",
        "groups",
        "speakers",
        "protocol_row_ids",
        "histories",
        "predictions",
        "probabilities",
        "contexts",
        "paths",
    }

    def visit(value: object, path: tuple[str, ...]) -> None:
        if isinstance(value, np.ndarray):
            raise StageAContractError(f"receipt contains ndarray at {'.'.join(path)}")
        if isinstance(value, Mapping):
            for key, child in value.items():
                name = str(key)
                tokens = set(name.lower().replace("-", "_").split("_"))
                if name.lower() in forbidden or "path" in tokens:
                    raise StageAContractError(
                        f"receipt contains row-level/path field {'.'.join((*path, name))}"
                    )
                visit(child, (*path, name))
            return
        if isinstance(value, (list, tuple)):
            if len(value) > 32:
                raise StageAContractError(f"receipt contains overlong list at {'.'.join(path)}")
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))
            return
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise StageAContractError(f"receipt contains non-JSON value at {'.'.join(path)}")

    visit(payload, ())


def _build_receipt(
    *,
    sidecars: HashedSidecarSet,
    fit: FitRoleView,
    config_hashes: Mapping[str, str],
    code_hashes: Mapping[str, str],
    environment: Mapping[str, object],
) -> dict[str, object]:
    protocol_mapping_sha = _canonical_sha256(
        [[int(index), int(value)] for index, value in enumerate(fit.protocol_row_ids)]
    )
    lineage_payload = {
        "config_sha256": dict(sorted(config_hashes.items())),
        "code_sha256": dict(sorted(code_hashes.items())),
        "runtime_environment_sha256": _canonical_sha256(environment),
    }
    events = _access_events(sidecars)
    receipt: dict[str, object] = {
        "schema_version": FIT_PREFLIGHT_RECEIPT_SCHEMA,
        "status": "fit_preflight_complete_no_training_no_performance_evaluation",
        "stage": "fit_preflight",
        "dataset": sidecars.dataset,
        "claim_boundary": (
            "Hash and structural validation of open-role v2 sidecars only; no model was "
            "trained and no performance metric or evidence claim was computed."
        ),
        "manifest": {
            "schema_version": sidecars.spec.manifest_schema,
            "protocol": sidecars.spec.protocol,
            "sha256": sidecars.manifest_sha256,
        },
        "sidecars": {
            "fit": {
                "feature_sha256": sidecars.fit.feature_sha256,
                "label_sha256": sidecars.fit.label_sha256,
                "row_alignment_sha256": sidecars.fit.row_alignment_sha256,
            },
            "model_selection": {
                "feature_sha256": sidecars.selection.feature_sha256,
                "label_sha256": sidecars.selection.label_sha256,
                "row_alignment_sha256": sidecars.selection.row_alignment_sha256,
                "feature_access": "byte_sha256_only_no_np_load",
                "label_access": "byte_sha256_only_no_np_load",
            },
        },
        "fit_contract": {
            "rows": fit.rows,
            "group_count": len(set(fit.groups.tolist())),
            "history_eligible_rows": int(sum(bool(value) for value in fit.histories)),
            "label_count": len(fit.label_order),
            "fit_array_manifest_sha256": _canonical_sha256(
                dict(sorted(fit.array_hashes.items()))
            ),
            "protocol_index_mapping_sha256": protocol_mapping_sha,
            "fit_arrays_contract_sha256": fit.contract_sha256,
        },
        "lineage": lineage_payload,
        "access_contract": {
            "events": events,
            "fit_feature_deserialized": True,
            "fit_label_deserialized": True,
            "selection_feature_deserialized": False,
            "selection_label_deserialized": False,
            "allow_pickle_used": False,
            "training_run": False,
            "performance_metric_computed": False,
        },
        "completion_gate": {
            "selection_feature_materialization_requires_receipt_reverification": True,
            "selection_label_materialization_allowed_in_completion": False,
            "selection_label_reserved_for_evaluate_stage": True,
            "manifest_sidecar_config_code_environment_hashes_must_all_match": True,
        },
        "public_artifact_policy": {
            "aggregate_only": True,
            "contains_row_level_values": False,
            "contains_private_paths": False,
            "contains_predictions_or_performance": False,
        },
    }
    _assert_aggregate_receipt(receipt)
    return receipt


def validate_fit_receipt(payload: Mapping[str, object]) -> None:
    if set(payload) != set(_RECEIPT_KEYS):
        raise StageAContractError("fit receipt top-level schema changed")
    if payload.get("schema_version") != FIT_PREFLIGHT_RECEIPT_SCHEMA:
        raise StageAContractError("fit receipt schema version changed")
    if payload.get("status") != "fit_preflight_complete_no_training_no_performance_evaluation":
        raise StageAContractError("fit receipt status changed")
    if payload.get("stage") != "fit_preflight" or payload.get("dataset") not in _SPECS:
        raise StageAContractError("fit receipt stage/dataset changed")
    access = payload.get("access_contract")
    expected_access_keys = {
        "events",
        "fit_feature_deserialized",
        "fit_label_deserialized",
        "selection_feature_deserialized",
        "selection_label_deserialized",
        "allow_pickle_used",
        "training_run",
        "performance_metric_computed",
    }
    if not isinstance(access, Mapping) or set(access) != expected_access_keys:
        raise StageAContractError("fit receipt access contract is missing")
    expected_flags = {
        "fit_feature_deserialized": True,
        "fit_label_deserialized": True,
        "selection_feature_deserialized": False,
        "selection_label_deserialized": False,
        "allow_pickle_used": False,
        "training_run": False,
        "performance_metric_computed": False,
    }
    for name, expected in expected_flags.items():
        if access.get(name) is not expected:
            raise StageAContractError(f"fit receipt access flag changed: {name}")
    events = access.get("events")
    if not isinstance(events, list) or len(events) != 4:
        raise StageAContractError("fit receipt access event count changed")
    expected_events = (
        (SELECTION_ROLE, "feature", "byte_sha256_only_no_np_load"),
        (SELECTION_ROLE, "label", "byte_sha256_only_no_np_load"),
        (FIT_ROLE, "feature", "allow_pickle_false_materialized"),
        (FIT_ROLE, "label", "allow_pickle_false_materialized"),
    )
    for index, (event, expected) in enumerate(zip(events, expected_events, strict=True)):
        if (
            not isinstance(event, Mapping)
            or set(event) != {"sequence", "role", "payload", "access", "sha256"}
            or event.get("sequence") != index
            or (event.get("role"), event.get("payload"), event.get("access")) != expected
        ):
            raise StageAContractError("fit receipt access event ordering changed")
        _require_sha256(event.get("sha256"), f"access.events.{index}.sha256")
    selection_events = events[:2]
    if any(event.get("role") != SELECTION_ROLE for event in selection_events):
        raise StageAContractError("selection hash-only events changed")
    if any(event.get("access") != "byte_sha256_only_no_np_load" for event in selection_events):
        raise StageAContractError("selection payload was not hash-only")
    completion = payload.get("completion_gate")
    expected_completion = {
        "selection_feature_materialization_requires_receipt_reverification": True,
        "selection_label_materialization_allowed_in_completion": False,
        "selection_label_reserved_for_evaluate_stage": True,
        "manifest_sidecar_config_code_environment_hashes_must_all_match": True,
    }
    if (
        not isinstance(completion, Mapping)
        or set(completion) != set(expected_completion)
        or any(
        completion.get(name) is not expected
        for name, expected in expected_completion.items()
        )
    ):
        raise StageAContractError("fit receipt completion gate changed")
    manifest = payload.get("manifest")
    sidecars = payload.get("sidecars")
    fit_contract = payload.get("fit_contract")
    lineage = payload.get("lineage")
    if not all(isinstance(value, Mapping) for value in (manifest, sidecars, fit_contract, lineage)):
        raise StageAContractError("fit receipt nested contract is malformed")
    assert isinstance(manifest, Mapping)
    assert isinstance(sidecars, Mapping)
    assert isinstance(fit_contract, Mapping)
    assert isinstance(lineage, Mapping)
    if set(manifest) != {"schema_version", "protocol", "sha256"}:
        raise StageAContractError("fit receipt manifest schema changed")
    if set(sidecars) != {"fit", "model_selection"}:
        raise StageAContractError("fit receipt sidecar schema changed")
    fit_sidecar = sidecars.get("fit")
    selection_sidecar = sidecars.get("model_selection")
    if (
        not isinstance(fit_sidecar, Mapping)
        or set(fit_sidecar)
        != {"feature_sha256", "label_sha256", "row_alignment_sha256"}
        or not isinstance(selection_sidecar, Mapping)
        or set(selection_sidecar)
        != {
            "feature_sha256",
            "label_sha256",
            "row_alignment_sha256",
            "feature_access",
            "label_access",
        }
    ):
        raise StageAContractError("fit receipt role-sidecar schema changed")
    if (
        selection_sidecar.get("feature_access") != "byte_sha256_only_no_np_load"
        or selection_sidecar.get("label_access") != "byte_sha256_only_no_np_load"
    ):
        raise StageAContractError("fit receipt selection access changed")
    expected_fit_contract = {
        "rows",
        "group_count",
        "history_eligible_rows",
        "label_count",
        "fit_array_manifest_sha256",
        "protocol_index_mapping_sha256",
        "fit_arrays_contract_sha256",
    }
    if set(fit_contract) != expected_fit_contract:
        raise StageAContractError("fit receipt aggregate fit schema changed")
    for field in ("rows", "group_count", "label_count"):
        if not isinstance(fit_contract.get(field), int) or int(fit_contract[field]) < 1:
            raise StageAContractError(f"fit receipt count changed: {field}")
    if (
        not isinstance(fit_contract.get("history_eligible_rows"), int)
        or not 0 <= int(fit_contract["history_eligible_rows"]) <= int(fit_contract["rows"])
    ):
        raise StageAContractError("fit receipt history count changed")
    if set(lineage) != {"config_sha256", "code_sha256", "runtime_environment_sha256"}:
        raise StageAContractError("fit receipt lineage schema changed")
    for field in ("config_sha256", "code_sha256"):
        hashes = lineage.get(field)
        if not isinstance(hashes, Mapping) or not hashes:
            raise StageAContractError(f"fit receipt {field} is empty")
        for name, digest in hashes.items():
            if _SAFE_NAME.fullmatch(str(name)) is None:
                raise StageAContractError(f"fit receipt {field} name is unsafe")
            _require_sha256(digest, f"lineage.{field}.{name}")
    public = payload.get("public_artifact_policy")
    expected_public = {
        "aggregate_only": True,
        "contains_row_level_values": False,
        "contains_private_paths": False,
        "contains_predictions_or_performance": False,
    }
    if not isinstance(public, Mapping) or dict(public) != expected_public:
        raise StageAContractError("fit receipt public-artifact policy changed")
    for digest in (
        manifest.get("sha256"),  # type: ignore[union-attr]
        fit_contract.get("protocol_index_mapping_sha256"),  # type: ignore[union-attr]
        fit_contract.get("fit_array_manifest_sha256"),  # type: ignore[union-attr]
        fit_contract.get("fit_arrays_contract_sha256"),  # type: ignore[union-attr]
        lineage.get("runtime_environment_sha256"),  # type: ignore[union-attr]
    ):
        _require_sha256(digest, "receipt lineage digest")
    for role_sidecar in (fit_sidecar, selection_sidecar):
        for field in ("feature_sha256", "label_sha256", "row_alignment_sha256"):
            _require_sha256(role_sidecar[field], f"sidecars.{field}")
    if events[0]["sha256"] != selection_sidecar["feature_sha256"] or events[1][
        "sha256"
    ] != selection_sidecar["label_sha256"]:
        raise StageAContractError("selection access events differ from sidecar hashes")
    if events[2]["sha256"] != fit_sidecar["feature_sha256"] or events[3][
        "sha256"
    ] != fit_sidecar["label_sha256"]:
        raise StageAContractError("fit access events differ from sidecar hashes")
    _assert_aggregate_receipt(payload)


def _write_json_once(payload: Mapping[str, object], path: Path) -> str:
    validate_fit_receipt(payload)
    if path.suffix.lower() != ".json":
        raise StageAContractError("fit receipt must be a JSON file")
    if path.exists():
        raise FileExistsError(f"fit receipt already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return _file_sha256(path)


def run_fit_preflight(
    *,
    dataset: str,
    sidecar_dir: str | Path,
    manifest_path: str | Path,
    receipt_path: str | Path,
    config_paths: Mapping[str, str | Path],
    code_paths: Mapping[str, str | Path],
    environment: Mapping[str, object] | None = None,
) -> FitPreflightResult:
    """Run Stage A without any training or performance computation."""

    sidecars = hash_open_role_sidecars(
        dataset=dataset,
        sidecar_dir=sidecar_dir,
        manifest_path=manifest_path,
    )
    config_hashes = _named_file_hashes(config_paths, "config_paths")
    code_hashes = _named_file_hashes(code_paths, "code_paths")
    runtime = dict(capture_runtime_environment() if environment is None else environment)
    if not runtime:
        raise StageAContractError("runtime environment payload must not be empty")
    # This is the only Stage-A NPZ materialisation boundary.
    fit = _materialize_fit_role(sidecars)
    receipt = _build_receipt(
        sidecars=sidecars,
        fit=fit,
        config_hashes=config_hashes,
        code_hashes=code_hashes,
        environment=runtime,
    )
    destination = Path(receipt_path)
    digest = _write_json_once(receipt, destination)
    return FitPreflightResult(fit, receipt, destination, digest)


def _load_receipt(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageAContractError(f"cannot read fit receipt: {error}") from error
    if not isinstance(payload, dict):
        raise StageAContractError("fit receipt root must be a mapping")
    validate_fit_receipt(payload)
    return payload


def verify_fit_receipt_inputs(
    *,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
    dataset: str,
    sidecar_dir: str | Path,
    manifest_path: str | Path,
    config_paths: Mapping[str, str | Path],
    code_paths: Mapping[str, str | Path],
    environment: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], HashedSidecarSet]:
    """Reverify every Stage-A input before any selection NPZ may be opened."""

    receipt_file = Path(receipt_path)
    if _file_sha256(receipt_file) != _require_sha256(
        expected_receipt_sha256, "expected_receipt_sha256"
    ):
        raise StageAContractError("fit receipt file hash changed")
    receipt = _load_receipt(receipt_file)
    sidecars = hash_open_role_sidecars(
        dataset=dataset,
        sidecar_dir=sidecar_dir,
        manifest_path=manifest_path,
    )
    runtime = dict(capture_runtime_environment() if environment is None else environment)
    expected = {
        "dataset": dataset,
        "manifest_sha256": sidecars.manifest_sha256,
        "fit_feature_sha256": sidecars.fit.feature_sha256,
        "fit_label_sha256": sidecars.fit.label_sha256,
        "selection_feature_sha256": sidecars.selection.feature_sha256,
        "selection_label_sha256": sidecars.selection.label_sha256,
        "config_sha256": _named_file_hashes(config_paths, "config_paths"),
        "code_sha256": _named_file_hashes(code_paths, "code_paths"),
        "runtime_environment_sha256": _canonical_sha256(runtime),
    }
    sidecar_receipt = receipt["sidecars"]
    lineage = receipt["lineage"]
    manifest_receipt = receipt["manifest"]
    assert isinstance(sidecar_receipt, Mapping)
    assert isinstance(lineage, Mapping)
    assert isinstance(manifest_receipt, Mapping)
    fit_receipt = sidecar_receipt.get("fit")
    selection_receipt = sidecar_receipt.get("model_selection")
    if not isinstance(fit_receipt, Mapping) or not isinstance(selection_receipt, Mapping):
        raise StageAContractError("receipt sidecar roles are malformed")
    observed = {
        "dataset": receipt.get("dataset"),
        "manifest_sha256": manifest_receipt.get("sha256"),
        "fit_feature_sha256": fit_receipt.get("feature_sha256"),
        "fit_label_sha256": fit_receipt.get("label_sha256"),
        "selection_feature_sha256": selection_receipt.get("feature_sha256"),
        "selection_label_sha256": selection_receipt.get("label_sha256"),
        "config_sha256": lineage.get("config_sha256"),
        "code_sha256": lineage.get("code_sha256"),
        "runtime_environment_sha256": lineage.get("runtime_environment_sha256"),
    }
    if observed != expected:
        changed = sorted(name for name in expected if observed.get(name) != expected[name])
        raise StageAContractError(f"fit receipt input lineage changed: {changed}")
    return receipt, sidecars


def _materialize_selection_feature(sidecars: HashedSidecarSet) -> SelectionFeatureView:
    spec = sidecars.spec
    record = sidecars.selection
    feature = _load_npz_exact(record.feature_path, spec.feature_fields)
    if spec.dataset == "EmotionTalk":
        if (
            _single_text(feature["schema_version"], "schema_version") != spec.feature_schema
            or _single_text(feature["dataset_id"], "dataset_id") != spec.dataset
            or _single_text(feature["role"], "role") != SELECTION_ROLE
            or _single_text(feature["split_protocol_id"], "split_protocol_id")
            != "scu_set_exploration_v1"
        ):
            raise StageAContractError("EmotionTalk selection feature identity changed")
        texts = np.asarray(feature["texts"])
        audio = np.asarray(feature["audio_features"])
        video = np.asarray(feature["video_features"])
        groups = np.asarray(feature["opaque_group_hashes"]).astype(str)
        speakers = np.asarray(feature["speaker_tokens"]).astype(str)
        turns = np.asarray(feature["turn_ids"])
        protocol_rows = np.asarray(feature["protocol_row_ids"])
        buckets = np.asarray(feature["role_buckets"])
        if buckets.shape != (record.rows,) or np.any((buckets < 65) | (buckets > 79)):
            raise StageAContractError("EmotionTalk selection bucket changed")
    else:
        if (
            _single_text(feature["schema_version"], "schema_version") != spec.feature_schema
            or _single_text(feature["role"], "role") != SELECTION_ROLE
        ):
            raise StageAContractError("MELD selection feature identity changed")
        texts = np.asarray(feature["utterances"])
        audio = np.asarray(feature["audio_mean_std"])
        video = np.asarray(feature["video_mean_std"])
        groups = np.asarray(feature["dialogue_codes"])
        speakers = np.asarray(feature["speaker_codes"])
        turns = np.asarray(feature["utterance_order"])
        protocol_rows = np.asarray(feature["protocol_row_ids"])
    if _single_text(feature["row_alignment_sha256"], "row_alignment") != (
        record.row_alignment_sha256
    ):
        raise StageAContractError("selection feature row alignment changed")
    _validate_normalized_rows(
        record=record,
        texts=texts,
        audio=audio,
        video=video,
        groups=groups,
        speakers=speakers,
        turns=turns,
        protocol_rows=protocol_rows,
        labels=None,
    )
    return SelectionFeatureView(
        dataset=spec.dataset,
        texts=tuple(str(value) for value in texts),
        audio=np.asarray(audio, dtype=np.float32).copy(),
        video=np.asarray(video, dtype=np.float32).copy(),
        groups=np.asarray(groups).astype(str),
        speakers=np.asarray(speakers).astype(str),
        turns=np.asarray(turns, dtype=np.int64).copy(),
        protocol_row_ids=np.asarray(protocol_rows, dtype=np.int64).copy(),
        labels_materialized=False,
    )


def materialize_selection_features_after_receipt(
    *,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
    dataset: str,
    sidecar_dir: str | Path,
    manifest_path: str | Path,
    config_paths: Mapping[str, str | Path],
    code_paths: Mapping[str, str | Path],
    environment: Mapping[str, object] | None = None,
) -> SelectionFeatureView:
    """Stage-B gate: verify receipt first, then open selection features only."""

    _, sidecars = verify_fit_receipt_inputs(
        receipt_path=receipt_path,
        expected_receipt_sha256=expected_receipt_sha256,
        dataset=dataset,
        sidecar_dir=sidecar_dir,
        manifest_path=manifest_path,
        config_paths=config_paths,
        code_paths=code_paths,
        environment=environment,
    )
    return _materialize_selection_feature(sidecars)


# ---------------------------------------------------------------------------
# Stage-A private producer/training-output contracts.  These validators enable
# synthetic contract tests now; the fit-preflight CLI intentionally does not
# train or emit these artifacts.


@dataclass(frozen=True)
class EncodedFitTasks:
    query_indices: np.ndarray
    candidate_indices: np.ndarray
    addition_contexts: tuple[tuple[int, ...], ...]
    deletion_contexts: tuple[tuple[int, ...], ...]
    task_sha256: str

    def __len__(self) -> int:
        return len(self.query_indices)


@dataclass(frozen=True)
class FitOnlyProducerView:
    dataset: str
    label_order: tuple[str, ...]
    seeds: tuple[int, ...]
    protocol_row_ids: np.ndarray
    fit_query_indices: np.ndarray
    fit_cluster_codes: np.ndarray
    fit_tasks: EncodedFitTasks
    fit_utility_probability: np.ndarray
    fit_forward_utility: np.ndarray
    fit_backward_utility: np.ndarray
    producer_file_sha256: str
    source_identity_sha256: str
    checkpoint_manifest_sha256: str
    histories_sha256: str
    source_hashes: Mapping[str, str]


_PRODUCER_FIXED_KEYS = frozenset(
    {
        "schema_version", "dataset", "dataset_label_order", "manifest_schema",
        "manifest_status", "manifest_sha256", "verified_provenance_attestation_sha256",
        "corpus_contract_sha256", "histories_sha256", "speaker_mapping_sha256",
        "runtime_environment_sha256", "source_identity_sha256", "seeds",
        "endpoint_context_names", "utility_context_names", "fit_query_indices",
        "selection_query_indices", "fit_cluster_codes", "selection_cluster_codes",
        "protocol_row_ids", "fit_endpoint_probability_oof",
        "selection_endpoint_probability_fold_ensemble", "fit_utility_probability_oof",
        "selection_utility_probability_fold_ensemble", "fit_forward_utility",
        "fit_backward_utility", "fit_asymmetry", "fit_sign_agreement",
        "selection_forward_utility", "selection_backward_utility", "selection_asymmetry",
        "selection_sign_agreement", "fit_task_sha256", "selection_task_sha256",
        "checkpoint_manifest_sha256", "utility_source",
        "matrix_fit_endpoint_probability_oof_sha256",
        "matrix_selection_endpoint_probability_fold_ensemble_sha256",
        "matrix_fit_utility_probability_oof_sha256",
        "matrix_selection_utility_probability_fold_ensemble_sha256",
        "matrix_fit_forward_utility_sha256", "matrix_fit_backward_utility_sha256",
        "matrix_selection_forward_utility_sha256", "matrix_selection_backward_utility_sha256",
        "fit_task_query_indices", "fit_task_candidate_indices", "fit_task_s_indptr",
        "fit_task_s_indices", "fit_task_t_indptr", "fit_task_t_indices",
        "selection_task_query_indices", "selection_task_candidate_indices",
        "selection_task_s_indptr", "selection_task_s_indices", "selection_task_t_indptr",
        "selection_task_t_indices",
    }
)


def _csr_rows(
    indptr_value: np.ndarray,
    indices_value: np.ndarray,
    *,
    rows: int,
    protocol_rows: int,
    field: str,
) -> tuple[tuple[int, ...], ...]:
    indptr = _integer_vector(indptr_value, f"{field}_indptr")
    indices = _integer_vector(indices_value, f"{field}_indices")
    if indptr.shape != (rows + 1,) or int(indptr[0]) != 0 or int(indptr[-1]) != len(indices):
        raise StageAContractError(f"{field} CSR pointer changed")
    if np.any(np.diff(indptr) < 0) or np.any(indices >= protocol_rows):
        raise StageAContractError(f"{field} CSR index changed")
    result: list[tuple[int, ...]] = []
    for row in range(rows):
        values = tuple(int(value) for value in indices[indptr[row] : indptr[row + 1]])
        if len(values) != len(set(values)):
            raise StageAContractError(f"{field} contains duplicate context rows")
        result.append(values)
    return tuple(result)


def _decode_fit_tasks(
    read: Callable[[str], np.ndarray],
    fit_queries: np.ndarray,
    protocol_rows: int,
) -> EncodedFitTasks:
    query = _integer_vector(read("fit_task_query_indices"), "fit_task_query_indices")
    candidate = _integer_vector(read("fit_task_candidate_indices"), "fit_task_candidate_indices")
    if query.shape != candidate.shape or not len(query):
        raise StageAContractError("fit task query/candidate arrays are empty or misaligned")
    if not set(query.tolist()).issubset(set(fit_queries.tolist())):
        raise StageAContractError("fit task query lies outside fit role")
    if np.any(query >= protocol_rows) or np.any(candidate >= protocol_rows):
        raise StageAContractError("fit task row lies outside producer protocol")
    addition = _csr_rows(
        read("fit_task_s_indptr"), read("fit_task_s_indices"), rows=len(query),
        protocol_rows=protocol_rows, field="fit_task_s",
    )
    deletion = _csr_rows(
        read("fit_task_t_indptr"), read("fit_task_t_indices"), rows=len(query),
        protocol_rows=protocol_rows, field="fit_task_t",
    )
    encoded: list[dict[str, object]] = []
    for q, candidate_value, s_context, t_context in zip(
        query, candidate, addition, deletion, strict=True
    ):
        q_int = int(q)
        c_int = int(candidate_value)
        if q_int == c_int or c_int in s_context or c_int not in t_context:
            raise StageAContractError("fit task violates candidate addition/deletion semantics")
        encoded.append({"query": q_int, "candidate": c_int, "s": list(s_context), "t": list(t_context)})
    declared = _require_sha256(_single_text(read("fit_task_sha256"), "fit_task_sha256"), "fit_task_sha256")
    if _canonical_sha256(encoded) != declared:
        raise StageAContractError("fit task encoding hash differs")
    return EncodedFitTasks(query, candidate, addition, deletion, declared)


def load_fit_only_producer_view(path: str | Path) -> FitOnlyProducerView:
    """Read only fit fields from a full producer NPZ; selection entries stay opaque."""

    producer_path = Path(path)
    if producer_path.suffix.lower() != ".npz":
        raise StageAContractError("producer cache must be an NPZ file")
    producer_sha = _file_sha256(producer_path)
    with np.load(producer_path, allow_pickle=False) as archive:
        keys = set(archive.files)
        missing = sorted(_PRODUCER_FIXED_KEYS - keys)
        unknown = sorted(
            key for key in keys - _PRODUCER_FIXED_KEYS
            if not (key.startswith("source_") and key.endswith("_sha256"))
        )
        if missing or unknown:
            raise StageAContractError(
                f"producer cache schema mismatch: missing={missing}, unknown={unknown}"
            )

        def read(name: str) -> np.ndarray:
            if name.startswith("selection_") or name.startswith("matrix_selection_"):
                raise AssertionError("Stage-A producer view attempted to read selection payload")
            return np.asarray(archive[name])

        if _single_text(read("schema_version"), "schema_version") != PRODUCER_CACHE_SCHEMA:
            raise StageAContractError("producer cache version changed")
        dataset = _single_text(read("dataset"), "dataset")
        if dataset not in _SPECS:
            raise StageAContractError("producer dataset is not registered")
        labels = tuple(str(value) for value in np.asarray(read("dataset_label_order")).reshape(-1))
        if labels != _SPECS[dataset].label_order:
            raise StageAContractError("producer label order differs from dataset contract")
        seeds = tuple(int(value) for value in _integer_vector(read("seeds"), "seeds", unique=True))
        if seeds != EXPECTED_SEEDS:
            raise StageAContractError(f"producer seeds must equal {EXPECTED_SEEDS}")
        if tuple(str(value) for value in np.asarray(read("endpoint_context_names")).reshape(-1)) != ENDPOINT_CONTEXT_NAMES:
            raise StageAContractError("producer endpoint context order changed")
        if tuple(str(value) for value in np.asarray(read("utility_context_names")).reshape(-1)) != UTILITY_CONTEXT_NAMES:
            raise StageAContractError("producer utility context order changed")
        protocol = _integer_vector(read("protocol_row_ids"), "protocol_row_ids", unique=True)
        fit_query = _integer_vector(read("fit_query_indices"), "fit_query_indices", unique=True)
        fit_cluster = _integer_vector(read("fit_cluster_codes"), "fit_cluster_codes")
        if fit_query.shape != fit_cluster.shape or not len(fit_query) or np.any(fit_query >= len(protocol)):
            raise StageAContractError("producer fit query/cluster alignment changed")
        tasks = _decode_fit_tasks(read, fit_query, len(protocol))
        utility_probability = _probability(
            read("fit_utility_probability_oof"), "fit_utility_probability_oof",
            (len(seeds), len(tasks), len(UTILITY_CONTEXT_NAMES), len(labels)),
        )
        endpoint_probability = _probability(
            read("fit_endpoint_probability_oof"), "fit_endpoint_probability_oof",
            (len(seeds), len(fit_query), len(ENDPOINT_CONTEXT_NAMES), len(labels)),
        )
        forward = np.asarray(read("fit_forward_utility"))
        backward = np.asarray(read("fit_backward_utility"))
        for field, value in (("fit_forward_utility", forward), ("fit_backward_utility", backward)):
            if value.shape != (len(seeds), len(tasks)) or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
                raise StageAContractError(f"{field} is not seed/task aligned")
        matrices = {
            "fit_endpoint_probability_oof": endpoint_probability,
            "fit_utility_probability_oof": utility_probability,
            "fit_forward_utility": forward,
            "fit_backward_utility": backward,
        }
        for name, value in matrices.items():
            field = f"matrix_{name}_sha256"
            declared = _require_sha256(_single_text(read(field), field), field)
            if declared != _array_sha256(np.asarray(value)):
                raise StageAContractError(f"producer fit matrix hash differs: {name}")
        source_hashes = {
            key: _require_sha256(_single_text(read(key), key), key)
            for key in sorted(keys)
            if key.startswith("source_")
            and key.endswith("_sha256")
            and key != "source_identity_sha256"
        }
        if not source_hashes:
            raise StageAContractError("producer view lacks source hashes")
        return FitOnlyProducerView(
            dataset=dataset,
            label_order=labels,
            seeds=seeds,
            protocol_row_ids=protocol,
            fit_query_indices=fit_query,
            fit_cluster_codes=fit_cluster,
            fit_tasks=tasks,
            fit_utility_probability=utility_probability,
            fit_forward_utility=forward.astype(np.float32, copy=True),
            fit_backward_utility=backward.astype(np.float32, copy=True),
            producer_file_sha256=producer_sha,
            source_identity_sha256=_require_sha256(
                _single_text(read("source_identity_sha256"), "source_identity_sha256"),
                "source_identity_sha256",
            ),
            checkpoint_manifest_sha256=_require_sha256(
                _single_text(read("checkpoint_manifest_sha256"), "checkpoint_manifest_sha256"),
                "checkpoint_manifest_sha256",
            ),
            histories_sha256=_require_sha256(
                _single_text(read("histories_sha256"), "histories_sha256"),
                "histories_sha256",
            ),
            source_hashes=source_hashes,
        )


@dataclass(frozen=True)
class CheckpointFileRecord:
    seed: int
    fold: int
    kind: str
    relative_name: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class CheckpointManifest:
    schema_version: str
    seeds: tuple[int, ...]
    outer_folds: int
    records: tuple[CheckpointFileRecord, ...]
    manifest_sha256: str


def _checkpoint_names(seeds: Sequence[int], outer_folds: int) -> list[tuple[int, int, str, str]]:
    if tuple(int(value) for value in seeds) != EXPECTED_SEEDS or int(outer_folds) < 2:
        raise StageAContractError("checkpoint manifest seeds/folds changed")
    result: list[tuple[int, int, str, str]] = []
    for seed in EXPECTED_SEEDS:
        for fold in range(int(outer_folds)):
            base = f"seed_{seed:05d}/fold_{fold:02d}"
            result.append((seed, fold, "checkpoint", f"{base}/checkpoint.pt"))
            result.append((seed, fold, "text_processor", f"{base}/text_processor.joblib"))
    return result


def build_checkpoint_manifest(
    checkpoint_root: str | Path,
    *,
    seeds: Sequence[int] = EXPECTED_SEEDS,
    outer_folds: int,
) -> CheckpointManifest:
    """Hash every expected file before any joblib/torch deserialisation."""

    root = Path(checkpoint_root)
    records: list[CheckpointFileRecord] = []
    expected = _checkpoint_names(seeds, outer_folds)
    for seed, fold, kind, relative_name in expected:
        path = root / Path(relative_name)
        digest = _file_sha256(path)
        records.append(
            CheckpointFileRecord(seed, fold, kind, relative_name, path.stat().st_size, digest)
        )
    expected_names = {value[3] for value in expected}
    observed_names = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed_names != expected_names:
        raise StageAContractError("checkpoint root contains a missing or unexpected file")
    rows = [record.__dict__ for record in records]
    payload = {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
        "seeds": list(EXPECTED_SEEDS),
        "outer_folds": int(outer_folds),
        "records": rows,
    }
    return CheckpointManifest(
        CHECKPOINT_MANIFEST_SCHEMA,
        EXPECTED_SEEDS,
        int(outer_folds),
        tuple(records),
        _canonical_sha256(payload),
    )


def verify_checkpoint_manifest(
    checkpoint_root: str | Path,
    manifest: CheckpointManifest,
) -> None:
    observed = build_checkpoint_manifest(
        checkpoint_root, seeds=manifest.seeds, outer_folds=manifest.outer_folds
    )
    if observed != manifest:
        raise StageAContractError("checkpoint manifest differs before deserialization")


def load_verified_fold_artifacts(
    checkpoint_root: str | Path,
    manifest: CheckpointManifest,
    *,
    seed: int,
    fold: int,
    checkpoint_loader: Callable[[Path], object],
    processor_loader: Callable[[Path], object],
) -> tuple[object, object]:
    """Verify the complete manifest, then and only then invoke local loaders."""

    verify_checkpoint_manifest(checkpoint_root, manifest)
    if int(seed) not in manifest.seeds or not 0 <= int(fold) < manifest.outer_folds:
        raise StageAContractError("requested checkpoint seed/fold is outside manifest")
    root = Path(checkpoint_root)
    base = root / f"seed_{int(seed):05d}" / f"fold_{int(fold):02d}"
    checkpoint = checkpoint_loader(base / "checkpoint.pt")
    processor = processor_loader(base / "text_processor.joblib")
    return checkpoint, processor


_CURRENT_ONLY_FIT_KEYS = frozenset(
    {
        "schema_version", "dataset", "dataset_label_order", "seeds", "outer_folds",
        "fit_query_indices", "fit_cluster_codes", "fit_probability_oof",
        "fit_fold_by_seed_query", "producer_source_identity_sha256",
        "current_only_source_identity_sha256", "history_backbone_checkpoint_manifest_sha256",
        "checkpoint_manifest_sha256", "training_protocol", "checkpoint_namespace",
        "history_training_items_consumed", "history_inference_items_consumed",
        "matrix_fit_probability_oof_sha256", "fold_assignment_sha256",
    }
)


def validate_current_only_fit_artifact(
    values: Mapping[str, np.ndarray],
    *,
    producer: FitOnlyProducerView,
    checkpoint_manifest: CheckpointManifest,
) -> None:
    if set(values) != set(_CURRENT_ONLY_FIT_KEYS):
        raise StageAContractError("current-only fit artifact schema changed")
    if _single_text(values["schema_version"], "schema_version") != CURRENT_ONLY_FIT_ARTIFACT_SCHEMA:
        raise StageAContractError("current-only fit artifact version changed")
    if _single_text(values["dataset"], "dataset") != producer.dataset:
        raise StageAContractError("current-only fit dataset changed")
    labels = tuple(str(value) for value in np.asarray(values["dataset_label_order"]).reshape(-1))
    seeds = tuple(int(value) for value in _integer_vector(values["seeds"], "seeds", unique=True))
    if labels != producer.label_order or seeds != producer.seeds:
        raise StageAContractError("current-only fit label/seed contract changed")
    folds = int(np.asarray(values["outer_folds"]).reshape(()))
    if folds != checkpoint_manifest.outer_folds:
        raise StageAContractError("current-only outer-fold count changed")
    query = _integer_vector(values["fit_query_indices"], "fit_query_indices", unique=True)
    cluster = _integer_vector(values["fit_cluster_codes"], "fit_cluster_codes")
    if not np.array_equal(query, producer.fit_query_indices) or not np.array_equal(
        cluster, producer.fit_cluster_codes
    ):
        raise StageAContractError("current-only fit alignment differs from producer")
    probability = _probability(
        values["fit_probability_oof"], "fit_probability_oof",
        (len(seeds), len(query), len(labels)),
    )
    if _require_sha256(_single_text(values["matrix_fit_probability_oof_sha256"], "matrix hash"), "matrix hash") != _array_sha256(probability):
        raise StageAContractError("current-only fit probability hash differs")
    fold_by_query = np.asarray(values["fit_fold_by_seed_query"])
    if fold_by_query.shape != (len(seeds), len(query)) or not np.issubdtype(
        fold_by_query.dtype, np.integer
    ) or np.any((fold_by_query < 0) | (fold_by_query >= folds)):
        raise StageAContractError("current-only fit fold assignment changed")
    if _require_sha256(_single_text(values["fold_assignment_sha256"], "fold_assignment_sha256"), "fold_assignment_sha256") != _array_sha256(fold_by_query):
        raise StageAContractError("current-only fold assignment hash differs")
    producer_identity = _require_sha256(
        _single_text(values["producer_source_identity_sha256"], "producer identity"),
        "producer identity",
    )
    current_identity = _require_sha256(
        _single_text(values["current_only_source_identity_sha256"], "current identity"),
        "current identity",
    )
    history_manifest = _require_sha256(
        _single_text(values["history_backbone_checkpoint_manifest_sha256"], "history manifest"),
        "history manifest",
    )
    current_manifest = _require_sha256(
        _single_text(values["checkpoint_manifest_sha256"], "current manifest"),
        "current manifest",
    )
    if producer_identity != producer.source_identity_sha256 or current_identity == producer_identity:
        raise StageAContractError("current-only training identity is not independent")
    if history_manifest != producer.checkpoint_manifest_sha256:
        raise StageAContractError("history checkpoint lineage differs from producer")
    if current_manifest != checkpoint_manifest.manifest_sha256 or current_manifest == history_manifest:
        raise StageAContractError("current-only checkpoint manifest is not independent")
    if _single_text(values["training_protocol"], "training_protocol") != INDEPENDENT_CURRENT_ONLY_PROTOCOL:
        raise StageAContractError("current-only history-stripped protocol changed")
    if _single_text(values["checkpoint_namespace"], "checkpoint_namespace") != "independent_current_only":
        raise StageAContractError("current-only checkpoint namespace changed")
    if int(np.asarray(values["history_training_items_consumed"]).reshape(())) != 0 or int(
        np.asarray(values["history_inference_items_consumed"]).reshape(())
    ) != 0:
        raise StageAContractError("current-only fit artifact consumed history")


_UTILITY_OOF_KEYS = frozenset(
    {
        "schema_version", "dataset", "seeds", "producer_file_sha256",
        "producer_source_identity_sha256", "fit_task_sha256", "fit_task_query_indices",
        "fit_task_cluster_codes", "utility_oof_folds", "fold_by_seed_task",
        "decision_score_oof_by_seed", "decision_score_oof_ensemble",
        "matrix_decision_score_oof_by_seed_sha256",
        "matrix_decision_score_oof_ensemble_sha256", "fold_assignment_sha256",
        "feature_schema_sha256", "model_spec_sha256", "score_source_identity_sha256",
        "selection_payload_consumed", "labels_or_targets_serialized",
    }
)


def validate_utility_oof_score_artifact(
    values: Mapping[str, np.ndarray],
    *,
    producer: FitOnlyProducerView,
) -> None:
    if set(values) != set(_UTILITY_OOF_KEYS):
        raise StageAContractError("utility OOF score artifact schema changed")
    if _single_text(values["schema_version"], "schema_version") != UTILITY_OOF_SCORE_SCHEMA:
        raise StageAContractError("utility OOF score version changed")
    if _single_text(values["dataset"], "dataset") != producer.dataset:
        raise StageAContractError("utility OOF dataset changed")
    seeds = tuple(int(value) for value in _integer_vector(values["seeds"], "seeds", unique=True))
    if seeds != EXPECTED_SEEDS:
        raise StageAContractError("utility OOF seeds changed")
    if _require_sha256(_single_text(values["producer_file_sha256"], "producer_file_sha256"), "producer_file_sha256") != producer.producer_file_sha256:
        raise StageAContractError("utility OOF producer file lineage changed")
    if _require_sha256(_single_text(values["producer_source_identity_sha256"], "producer identity"), "producer identity") != producer.source_identity_sha256:
        raise StageAContractError("utility OOF producer identity changed")
    if _require_sha256(_single_text(values["fit_task_sha256"], "fit_task_sha256"), "fit_task_sha256") != producer.fit_tasks.task_sha256:
        raise StageAContractError("utility OOF task identity changed")
    queries = _integer_vector(values["fit_task_query_indices"], "fit_task_query_indices")
    if not np.array_equal(queries, producer.fit_tasks.query_indices):
        raise StageAContractError("utility OOF task order changed")
    query_cluster = {
        int(query): int(cluster)
        for query, cluster in zip(
            producer.fit_query_indices, producer.fit_cluster_codes, strict=True
        )
    }
    expected_clusters = np.asarray([query_cluster[int(query)] for query in queries], dtype=np.int64)
    clusters = _integer_vector(values["fit_task_cluster_codes"], "fit_task_cluster_codes")
    if not np.array_equal(clusters, expected_clusters):
        raise StageAContractError("utility OOF task clusters changed")
    folds = int(np.asarray(values["utility_oof_folds"]).reshape(()))
    if folds < 2:
        raise StageAContractError("utility OOF requires at least two folds")
    fold_by_task = np.asarray(values["fold_by_seed_task"])
    if fold_by_task.shape != (len(seeds), len(queries)) or not np.issubdtype(
        fold_by_task.dtype, np.integer
    ) or np.any((fold_by_task < 0) | (fold_by_task >= folds)):
        raise StageAContractError("utility OOF fold assignment changed")
    for seed_index in range(len(seeds)):
        for cluster in np.unique(clusters):
            if len(np.unique(fold_by_task[seed_index, clusters == cluster])) != 1:
                raise StageAContractError("utility OOF split one cluster across folds")
    scores = np.asarray(values["decision_score_oof_by_seed"])
    ensemble = np.asarray(values["decision_score_oof_ensemble"])
    if scores.shape != (len(seeds), len(queries)) or ensemble.shape != (len(queries),):
        raise StageAContractError("utility OOF decision scores are misaligned")
    if not np.issubdtype(scores.dtype, np.floating) or not np.issubdtype(
        ensemble.dtype, np.floating
    ) or not np.isfinite(scores).all() or not np.isfinite(ensemble).all():
        raise StageAContractError("utility OOF decision score is invalid")
    if not np.allclose(ensemble, scores.mean(axis=0), rtol=1.0e-7, atol=1.0e-9):
        raise StageAContractError("utility OOF ensemble is not the five-seed mean")
    hash_pairs = (
        ("matrix_decision_score_oof_by_seed_sha256", scores),
        ("matrix_decision_score_oof_ensemble_sha256", ensemble),
        ("fold_assignment_sha256", fold_by_task),
    )
    for field, array in hash_pairs:
        if _require_sha256(_single_text(values[field], field), field) != _array_sha256(array):
            raise StageAContractError(f"utility OOF hash differs: {field}")
    for field in ("feature_schema_sha256", "model_spec_sha256", "score_source_identity_sha256"):
        _require_sha256(_single_text(values[field], field), field)
    if bool(np.asarray(values["selection_payload_consumed"]).reshape(())) or bool(
        np.asarray(values["labels_or_targets_serialized"]).reshape(())
    ):
        raise StageAContractError("utility OOF artifact consumed selection or serialized targets")
