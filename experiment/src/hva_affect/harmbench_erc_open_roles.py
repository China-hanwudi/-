"""Physically separated open-role inputs for HarmBench-ERC.

The prediction capability deliberately has no label field and its loaders never
open, hash, glob, or otherwise touch the model-selection label archive.  A
separate evaluator may obtain that capability only after prediction artifacts
and their receipts have been frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .data_contract import ContractError, sha256_file
from .emotiontalk_role_sidecar import (
    FEATURE_FIELDS as EMOTIONTALK_FEATURE_FIELDS,
    FEATURE_SCHEMA as EMOTIONTALK_FEATURE_SCHEMA,
    FIT_ROLE,
    FROZEN_ROLE_RANGES,
    LABEL_FIELDS as EMOTIONTALK_LABEL_FIELDS,
    LABEL_SCHEMA as EMOTIONTALK_LABEL_SCHEMA,
    MANIFEST_ROLE_FIELDS as EMOTIONTALK_MANIFEST_ROLE_FIELDS,
    MANIFEST_SCHEMA as EMOTIONTALK_MANIFEST_SCHEMA,
    OPEN_ROLES,
    PROTOCOL as EMOTIONTALK_PROTOCOL,
    SELECTION_ROLE,
)
from .emotiontalk_text_p1 import LABEL_NAMES as EMOTIONTALK_LABEL_NAMES
from .meld_causal_backbone_loader import (
    FEATURE_FIELDS as MELD_FEATURE_FIELDS,
    LABEL_FIELDS as MELD_LABEL_FIELDS,
    ROLE_RECORD_FIELDS as MELD_ROLE_RECORD_FIELDS,
    _read_manifest as _read_meld_manifest,
)
from .meld_multimodal_sidecar import (
    EMOTION_TO_INDEX as MELD_EMOTION_TO_INDEX,
    SIDECAR_SCHEMA_VERSION as MELD_SIDECAR_SCHEMA,
)
from .harmbench_erc_role_manifests import (
    CrossRoleFeatureRosterReceipt,
    SANITIZED_EMOTIONTALK_FIT_LABEL_FIELDS,
    SANITIZED_EMOTIONTALK_FIT_LABEL_SCHEMA,
    exact_artifact_path,
    load_feature_manifest,
    load_fit_training_manifest,
    make_legacy_cross_role_feature_roster_receipt,
    make_synthetic_cross_role_feature_roster_receipt,
    read_verified_npz,
    roster_feature_projection_sha256,
    validate_cross_role_feature_roster_receipt,
)


class HarmBenchOpenRoleError(ContractError):
    """Raised when an open-role capability is incomplete or outcome-bearing."""


SHA256_LENGTH = 64
FIT_CAPABILITY = "fit_features_and_labels"
PREDICTION_CAPABILITY = "selection_features_without_outcomes"


def _valid_sha256(value: object, *, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise HarmBenchOpenRoleError(f"{name} must be a lowercase SHA-256")
    return digest


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HarmBenchOpenRoleError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_unique_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except HarmBenchOpenRoleError:
        raise
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise HarmBenchOpenRoleError(f"cannot read sidecar manifest: {error}") from error
    if not isinstance(payload, dict):
        raise HarmBenchOpenRoleError("sidecar manifest root must be an object")
    return payload


def _scalar_text(payload: Mapping[str, np.ndarray], name: str) -> str:
    value = np.asarray(payload[name])
    if value.size != 1:
        raise HarmBenchOpenRoleError(f"{name} must contain one scalar string")
    return str(value.reshape(-1)[0])


def _load_npz(path: Path, expected_fields: set[str], *, name: str) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise HarmBenchOpenRoleError(f"missing exact {name} archive")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_fields:
            raise HarmBenchOpenRoleError(f"{name} archive schema changed")
        return {field: np.asarray(archive[field]).copy() for field in archive.files}


def _readonly(values: object, *, dtype: object | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _canonical_array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\x00")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\x00")
    if array.dtype.kind in {"U", "S", "O"}:
        for value in array.astype(str).reshape(-1):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    else:
        canonical = np.ascontiguousarray(array)
        if canonical.dtype.byteorder == ">" or (
            canonical.dtype.byteorder == "=" and not np.little_endian
        ):
            canonical = canonical.byteswap().newbyteorder("<")
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _history_indices(
    groups: np.ndarray, speaker_identity: np.ndarray, turns: np.ndarray
) -> tuple[tuple[int, ...], ...]:
    histories: list[tuple[int, ...]] = []
    for query in range(len(groups)):
        candidates = np.flatnonzero(
            (groups == groups[query])
            & (speaker_identity == speaker_identity[query])
            & (turns < turns[query])
        )
        histories.append(
            tuple(
                sorted(
                    candidates.tolist(),
                    key=lambda index: (int(turns[index]), int(index)),
                )
            )
        )
    return tuple(histories)


@dataclass(frozen=True)
class OutcomeFreeRoleFeatures:
    """Private features for one role; labels are structurally impossible."""

    dataset_id: str
    role: str
    keys: np.ndarray
    texts: tuple[str, ...]
    audio: np.ndarray
    video: np.ndarray
    groups: np.ndarray
    speaker_identity: np.ndarray
    turn_ids: np.ndarray
    protocol_row_ids: np.ndarray
    same_speaker_histories: tuple[tuple[int, ...], ...]
    row_alignment_sha256: str
    feature_sha256: str
    content_sha256: str

    @property
    def rows(self) -> int:
        return len(self.protocol_row_ids)

    @property
    def history_eligible(self) -> np.ndarray:
        result = np.asarray(
            [bool(history) for history in self.same_speaker_histories], dtype=bool
        )
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class FitFeatureCapability:
    """Outcome-free fit features for split construction and preprocessing."""

    dataset_id: str
    fit: OutcomeFreeRoleFeatures
    feature_manifest_sha256: str
    cross_role_feature_roster_receipt: CrossRoleFeatureRosterReceipt
    cross_role_feature_roster_sha256: str
    capability_sha256: str
    selection_feature_archive_opened: bool = False
    any_label_archive_opened: bool = False
    any_label_archive_hashed: bool = False


@dataclass(frozen=True)
class LabeledFitRole:
    feature_capability: FitFeatureCapability
    labels: np.ndarray
    label_sha256: str
    label_order: tuple[str, ...]
    capability: str = FIT_CAPABILITY

    @property
    def features(self) -> OutcomeFreeRoleFeatures:
        return self.feature_capability.fit


@dataclass(frozen=True)
class FitRoleCapability:
    """Training capability; no selection feature or outcome surface exists."""

    dataset_id: str
    fit: LabeledFitRole
    fit_manifest_sha256: str
    cross_role_feature_roster_receipt: CrossRoleFeatureRosterReceipt
    cross_role_feature_roster_sha256: str
    capability_sha256: str
    selection_feature_archive_opened: bool = False
    selection_label_archive_opened: bool = False
    selection_label_archive_hashed: bool = False


@dataclass(frozen=True)
class SelectionFeatureCapability:
    """Prediction capability; selection labels are structurally impossible."""

    dataset_id: str
    selection: OutcomeFreeRoleFeatures
    manifest_sha256: str
    cross_role_feature_roster_receipt: CrossRoleFeatureRosterReceipt
    cross_role_feature_roster_sha256: str
    capability_sha256: str
    selection_label_archive_opened: bool = False
    selection_label_archive_hashed: bool = False


@dataclass(frozen=True)
class OpenRoleCapabilities:
    dataset_id: str
    fit: LabeledFitRole
    selection: OutcomeFreeRoleFeatures
    fit_manifest_sha256: str
    selection_manifest_sha256: str
    cross_role_feature_roster_receipt: CrossRoleFeatureRosterReceipt
    cross_role_feature_roster_sha256: str
    capability_sha256: str
    fit_capability: str = FIT_CAPABILITY
    selection_capability: str = PREDICTION_CAPABILITY
    selection_label_archive_opened: bool = False
    selection_label_archive_hashed: bool = False


def make_outcome_free_role_features(
    *,
    dataset_id: str,
    role: str,
    keys: object,
    texts: Sequence[str],
    audio: object,
    video: object,
    groups: object,
    speaker_identity: object,
    turn_ids: object,
    protocol_row_ids: object,
    row_alignment_sha256: str,
    feature_sha256: str,
) -> OutcomeFreeRoleFeatures:
    """Validate, copy and freeze an outcome-free role payload."""

    if dataset_id not in {"EmotionTalk", "MELD", "synthetic"}:
        raise HarmBenchOpenRoleError("unsupported dataset id")
    if role not in OPEN_ROLES:
        raise HarmBenchOpenRoleError("only frozen open roles are permitted")
    key_array = _readonly(keys, dtype=str)
    group_array = _readonly(groups, dtype=str)
    speaker_array = _readonly(speaker_identity, dtype=str)
    turn_array = _readonly(turn_ids, dtype=np.int64)
    protocol_array = _readonly(protocol_row_ids, dtype=np.int64)
    audio_array = _readonly(audio, dtype=np.float32)
    video_array = _readonly(video, dtype=np.float32)
    text_values = tuple(str(value) for value in texts)
    rows = len(key_array)
    if rows == 0:
        raise HarmBenchOpenRoleError("role payload is empty")
    if any(
        len(value) != rows
        for value in (
            text_values,
            group_array,
            speaker_array,
            turn_array,
            protocol_array,
            audio_array,
            video_array,
        )
    ):
        raise HarmBenchOpenRoleError("role arrays are not row-aligned")
    if audio_array.ndim != 2 or video_array.ndim != 2:
        raise HarmBenchOpenRoleError("modality arrays must be two-dimensional")
    if not np.isfinite(audio_array).all() or not np.isfinite(video_array).all():
        raise HarmBenchOpenRoleError("modality arrays contain non-finite values")
    if len(set(key_array.tolist())) != rows:
        raise HarmBenchOpenRoleError("row keys must be unique")
    if len(set(protocol_array.tolist())) != rows or np.any(protocol_array < 0):
        raise HarmBenchOpenRoleError("protocol row ids must be unique non-negative integers")
    if np.any(turn_array < 0):
        raise HarmBenchOpenRoleError("turn ids must be non-negative")
    histories = _history_indices(group_array, speaker_array, turn_array)
    content_descriptor = {
        "dataset_id": dataset_id,
        "role": role,
        "keys_sha256": _canonical_array_sha256(key_array),
        "texts_sha256": _canonical_array_sha256(np.asarray(text_values)),
        "audio_sha256": _canonical_array_sha256(audio_array),
        "video_sha256": _canonical_array_sha256(video_array),
        "groups_sha256": _canonical_array_sha256(group_array),
        "speaker_sha256": _canonical_array_sha256(speaker_array),
        "turn_sha256": _canonical_array_sha256(turn_array),
        "protocol_rows_sha256": _canonical_array_sha256(protocol_array),
        "row_alignment_sha256": _valid_sha256(
            row_alignment_sha256, name="row_alignment_sha256"
        ),
        "feature_sha256": _valid_sha256(feature_sha256, name="feature_sha256"),
    }
    content_sha = hashlib.sha256(
        json.dumps(
            content_descriptor,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return OutcomeFreeRoleFeatures(
        dataset_id=dataset_id,
        role=role,
        keys=key_array,
        texts=text_values,
        audio=audio_array,
        video=video_array,
        groups=group_array,
        speaker_identity=speaker_array,
        turn_ids=turn_array,
        protocol_row_ids=protocol_array,
        same_speaker_histories=histories,
        row_alignment_sha256=content_descriptor["row_alignment_sha256"],
        feature_sha256=content_descriptor["feature_sha256"],
        content_sha256=content_sha,
    )


def validate_outcome_free_role_features(
    value: OutcomeFreeRoleFeatures,
) -> OutcomeFreeRoleFeatures:
    """Recompute every live array/content binding before a capability is consumed."""

    if not isinstance(value, OutcomeFreeRoleFeatures):
        raise HarmBenchOpenRoleError("outcome-free role feature type changed")
    for name in (
        "keys",
        "audio",
        "video",
        "groups",
        "speaker_identity",
        "turn_ids",
        "protocol_row_ids",
    ):
        if np.asarray(getattr(value, name)).flags.writeable:
            raise HarmBenchOpenRoleError(f"outcome-free feature array is writable: {name}")
    rebuilt = make_outcome_free_role_features(
        dataset_id=value.dataset_id,
        role=value.role,
        keys=value.keys,
        texts=value.texts,
        audio=value.audio,
        video=value.video,
        groups=value.groups,
        speaker_identity=value.speaker_identity,
        turn_ids=value.turn_ids,
        protocol_row_ids=value.protocol_row_ids,
        row_alignment_sha256=value.row_alignment_sha256,
        feature_sha256=value.feature_sha256,
    )
    scalar_fields = (
        "dataset_id",
        "role",
        "texts",
        "same_speaker_histories",
        "row_alignment_sha256",
        "feature_sha256",
        "content_sha256",
    )
    if any(getattr(rebuilt, name) != getattr(value, name) for name in scalar_fields):
        raise HarmBenchOpenRoleError("outcome-free feature content binding changed")
    for name in (
        "keys",
        "audio",
        "video",
        "groups",
        "speaker_identity",
        "turn_ids",
        "protocol_row_ids",
    ):
        if not np.array_equal(getattr(rebuilt, name), getattr(value, name)):
            raise HarmBenchOpenRoleError("outcome-free feature array binding changed")
    return value


def make_fit_feature_capability(
    *,
    fit_features: OutcomeFreeRoleFeatures,
    feature_manifest_sha256: str,
    cross_role_feature_roster_receipt: CrossRoleFeatureRosterReceipt,
) -> FitFeatureCapability:
    fit_features = validate_outcome_free_role_features(fit_features)
    if fit_features.role != FIT_ROLE:
        raise HarmBenchOpenRoleError("fit feature role identity changed")
    manifest_sha = _valid_sha256(
        feature_manifest_sha256, name="feature_manifest_sha256"
    )
    try:
        roster_receipt = validate_cross_role_feature_roster_receipt(
            cross_role_feature_roster_receipt,
            expected_roster_sha256=(
                cross_role_feature_roster_receipt.roster_sha256
            ),
            expected_dataset=fit_features.dataset_id,
        )
    except (AttributeError, ValueError) as error:
        raise HarmBenchOpenRoleError(
            f"fit cross-role feature roster receipt changed: {error}"
        ) from error
    roster_sha = roster_receipt.roster_sha256
    descriptor = {
        "dataset_id": fit_features.dataset_id,
        "fit_content_sha256": fit_features.content_sha256,
        "feature_manifest_sha256": manifest_sha,
        "cross_role_feature_roster_sha256": roster_sha,
        "fit_feature_projection_sha256": (
            roster_receipt.fit_feature_projection_sha256
        ),
        "selection_feature_archive_opened": False,
        "any_label_archive_opened": False,
        "any_label_archive_hashed": False,
    }
    return FitFeatureCapability(
        dataset_id=fit_features.dataset_id,
        fit=fit_features,
        feature_manifest_sha256=manifest_sha,
        cross_role_feature_roster_receipt=roster_receipt,
        cross_role_feature_roster_sha256=roster_sha,
        capability_sha256=hashlib.sha256(
            json.dumps(
                descriptor,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    )


def validate_fit_feature_capability(
    value: FitFeatureCapability,
) -> FitFeatureCapability:
    if not isinstance(value, FitFeatureCapability):
        raise HarmBenchOpenRoleError("fit feature capability type changed")
    if (
        value.selection_feature_archive_opened is not False
        or value.any_label_archive_opened is not False
        or value.any_label_archive_hashed is not False
    ):
        raise HarmBenchOpenRoleError("fit feature capability access boundary changed")
    validate_outcome_free_role_features(value.fit)
    rebuilt = make_fit_feature_capability(
        fit_features=value.fit,
        feature_manifest_sha256=value.feature_manifest_sha256,
        cross_role_feature_roster_receipt=(
            value.cross_role_feature_roster_receipt
        ),
    )
    if (
        rebuilt.dataset_id != value.dataset_id
        or rebuilt.feature_manifest_sha256 != value.feature_manifest_sha256
        or rebuilt.cross_role_feature_roster_sha256
        != value.cross_role_feature_roster_sha256
        or rebuilt.cross_role_feature_roster_receipt
        != value.cross_role_feature_roster_receipt
        or rebuilt.capability_sha256 != value.capability_sha256
        or rebuilt.fit.content_sha256 != value.fit.content_sha256
    ):
        raise HarmBenchOpenRoleError("fit feature capability binding changed")
    return value


def make_fit_role_capability(
    *,
    fit_feature_capability: FitFeatureCapability,
    fit_labels: object,
    fit_label_sha256: str,
    label_order: Sequence[str],
    fit_manifest_sha256: str,
) -> FitRoleCapability:
    fit_feature_capability = validate_fit_feature_capability(fit_feature_capability)
    fit_features = fit_feature_capability.fit
    labels_raw = np.asarray(fit_labels)
    if labels_raw.ndim != 1 or len(labels_raw) != fit_features.rows:
        raise HarmBenchOpenRoleError("fit labels are not row-aligned")
    if labels_raw.dtype.kind not in "iu" or labels_raw.dtype.kind == "b":
        raise HarmBenchOpenRoleError("fit labels must use an integer dtype")
    labels = _readonly(labels_raw, dtype=np.int64)
    order = tuple(str(value) for value in label_order)
    if len(order) < 2 or len(set(order)) != len(order):
        raise HarmBenchOpenRoleError("label order must contain unique classes")
    if np.any(labels < 0) or np.any(labels >= len(order)):
        raise HarmBenchOpenRoleError("fit label is outside the label order")
    fit_label_sha = _valid_sha256(fit_label_sha256, name="fit_label_sha256")
    manifest_sha = _valid_sha256(
        fit_manifest_sha256, name="fit_manifest_sha256"
    )
    descriptor = {
        "dataset_id": fit_features.dataset_id,
        "fit_feature_capability_sha256": fit_feature_capability.capability_sha256,
        "fit_label_array_sha256": _canonical_array_sha256(labels),
        "fit_label_file_sha256": fit_label_sha,
        "fit_manifest_sha256": manifest_sha,
        "cross_role_feature_roster_sha256": (
            fit_feature_capability.cross_role_feature_roster_sha256
        ),
        "label_order": list(order),
        "selection_feature_archive_opened": False,
        "selection_label_archive_opened": False,
        "selection_label_archive_hashed": False,
    }
    capability_sha = hashlib.sha256(
        json.dumps(
            descriptor,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return FitRoleCapability(
        dataset_id=fit_features.dataset_id,
        fit=LabeledFitRole(
            feature_capability=fit_feature_capability,
            labels=labels,
            label_sha256=fit_label_sha,
            label_order=order,
        ),
        fit_manifest_sha256=manifest_sha,
        cross_role_feature_roster_receipt=(
            fit_feature_capability.cross_role_feature_roster_receipt
        ),
        cross_role_feature_roster_sha256=(
            fit_feature_capability.cross_role_feature_roster_sha256
        ),
        capability_sha256=capability_sha,
    )


def validate_fit_role_capability(value: FitRoleCapability) -> FitRoleCapability:
    """Validate the live fit feature/label payload and its capability receipt."""

    if not isinstance(value, FitRoleCapability):
        raise HarmBenchOpenRoleError("fit capability type changed")
    if (
        value.selection_feature_archive_opened is not False
        or value.selection_label_archive_opened is not False
        or value.selection_label_archive_hashed is not False
    ):
        raise HarmBenchOpenRoleError("fit capability access boundary changed")
    validate_outcome_free_role_features(value.fit.features)
    if np.asarray(value.fit.labels).flags.writeable:
        raise HarmBenchOpenRoleError("fit labels must be immutable")
    rebuilt = make_fit_role_capability(
        fit_feature_capability=value.fit.feature_capability,
        fit_labels=value.fit.labels,
        fit_label_sha256=value.fit.label_sha256,
        label_order=value.fit.label_order,
        fit_manifest_sha256=value.fit_manifest_sha256,
    )
    if (
        rebuilt.dataset_id != value.dataset_id
        or rebuilt.fit_manifest_sha256 != value.fit_manifest_sha256
        or rebuilt.cross_role_feature_roster_sha256
        != value.cross_role_feature_roster_sha256
        or rebuilt.cross_role_feature_roster_receipt
        != value.cross_role_feature_roster_receipt
        or rebuilt.capability_sha256 != value.capability_sha256
        or rebuilt.fit.label_order != value.fit.label_order
        or not np.array_equal(rebuilt.fit.labels, value.fit.labels)
    ):
        raise HarmBenchOpenRoleError("fit capability binding changed")
    return value


def make_selection_feature_capability(
    *,
    selection_features: OutcomeFreeRoleFeatures,
    manifest_sha256: str,
    cross_role_feature_roster_receipt: CrossRoleFeatureRosterReceipt,
) -> SelectionFeatureCapability:
    selection_features = validate_outcome_free_role_features(selection_features)
    if selection_features.role != SELECTION_ROLE:
        raise HarmBenchOpenRoleError("selection role identity changed")
    manifest_sha = _valid_sha256(manifest_sha256, name="manifest_sha256")
    try:
        roster_receipt = validate_cross_role_feature_roster_receipt(
            cross_role_feature_roster_receipt,
            expected_roster_sha256=(
                cross_role_feature_roster_receipt.roster_sha256
            ),
            expected_dataset=selection_features.dataset_id,
        )
    except (AttributeError, ValueError) as error:
        raise HarmBenchOpenRoleError(
            f"selection cross-role feature roster receipt changed: {error}"
        ) from error
    roster_sha = roster_receipt.roster_sha256
    descriptor = {
        "dataset_id": selection_features.dataset_id,
        "selection_content_sha256": selection_features.content_sha256,
        "manifest_sha256": manifest_sha,
        "cross_role_feature_roster_sha256": roster_sha,
        "selection_feature_projection_sha256": (
            roster_receipt.selection_feature_projection_sha256
        ),
        "selection_label_archive_opened": False,
        "selection_label_archive_hashed": False,
    }
    capability_sha = hashlib.sha256(
        json.dumps(
            descriptor,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return SelectionFeatureCapability(
        dataset_id=selection_features.dataset_id,
        selection=selection_features,
        manifest_sha256=manifest_sha,
        cross_role_feature_roster_receipt=roster_receipt,
        cross_role_feature_roster_sha256=roster_sha,
        capability_sha256=capability_sha,
    )


def validate_selection_feature_capability(
    value: SelectionFeatureCapability,
) -> SelectionFeatureCapability:
    """Validate a live feature-only selection capability without labels."""

    if not isinstance(value, SelectionFeatureCapability):
        raise HarmBenchOpenRoleError("selection feature capability type changed")
    if (
        value.selection_label_archive_opened is not False
        or value.selection_label_archive_hashed is not False
    ):
        raise HarmBenchOpenRoleError("selection feature access boundary changed")
    validate_outcome_free_role_features(value.selection)
    rebuilt = make_selection_feature_capability(
        selection_features=value.selection,
        manifest_sha256=value.manifest_sha256,
        cross_role_feature_roster_receipt=(
            value.cross_role_feature_roster_receipt
        ),
    )
    if (
        rebuilt.dataset_id != value.dataset_id
        or rebuilt.manifest_sha256 != value.manifest_sha256
        or rebuilt.cross_role_feature_roster_sha256
        != value.cross_role_feature_roster_sha256
        or rebuilt.cross_role_feature_roster_receipt
        != value.cross_role_feature_roster_receipt
        or rebuilt.capability_sha256 != value.capability_sha256
    ):
        raise HarmBenchOpenRoleError("selection feature capability binding changed")
    return value


def make_synthetic_fit_feature_capability(
    *,
    fit_features: OutcomeFreeRoleFeatures,
    feature_manifest_sha256: str,
    synthetic_feature_projection_sha256: str,
) -> FitFeatureCapability:
    """Explicit test-only adapter from one synthetic projection seed."""

    if fit_features.dataset_id != "synthetic":
        raise HarmBenchOpenRoleError(
            "synthetic fit capability requires the synthetic dataset"
        )
    receipt = make_synthetic_cross_role_feature_roster_receipt(
        dataset_id="synthetic",
        fit_feature_projection_sha256=synthetic_feature_projection_sha256,
        selection_feature_projection_sha256=synthetic_feature_projection_sha256,
    )
    return make_fit_feature_capability(
        fit_features=fit_features,
        feature_manifest_sha256=feature_manifest_sha256,
        cross_role_feature_roster_receipt=receipt,
    )


def make_synthetic_selection_feature_capability(
    *,
    selection_features: OutcomeFreeRoleFeatures,
    manifest_sha256: str,
    synthetic_feature_projection_sha256: str,
) -> SelectionFeatureCapability:
    """Explicit test-only adapter from one synthetic projection seed."""

    if selection_features.dataset_id != "synthetic":
        raise HarmBenchOpenRoleError(
            "synthetic selection capability requires the synthetic dataset"
        )
    receipt = make_synthetic_cross_role_feature_roster_receipt(
        dataset_id="synthetic",
        fit_feature_projection_sha256=synthetic_feature_projection_sha256,
        selection_feature_projection_sha256=synthetic_feature_projection_sha256,
    )
    return make_selection_feature_capability(
        selection_features=selection_features,
        manifest_sha256=manifest_sha256,
        cross_role_feature_roster_receipt=receipt,
    )


def compose_open_role_capabilities(
    fit: FitRoleCapability,
    selection: SelectionFeatureCapability,
) -> OpenRoleCapabilities:
    if not isinstance(fit, FitRoleCapability) or not isinstance(
        selection, SelectionFeatureCapability
    ):
        raise HarmBenchOpenRoleError("typed fit and selection capabilities are required")
    fit = validate_fit_role_capability(fit)
    selection = validate_selection_feature_capability(selection)
    if fit.dataset_id != selection.dataset_id:
        raise HarmBenchOpenRoleError("fit and selection datasets differ")
    if (
        fit.cross_role_feature_roster_sha256
        != selection.cross_role_feature_roster_sha256
        or fit.cross_role_feature_roster_receipt
        != selection.cross_role_feature_roster_receipt
    ):
        raise HarmBenchOpenRoleError("fit and selection feature rosters differ")
    fit_features = fit.fit.features
    selection_features = selection.selection
    if set(fit_features.keys.tolist()).intersection(selection_features.keys.tolist()):
        raise HarmBenchOpenRoleError("fit and selection share a row key")
    if set(fit_features.protocol_row_ids.tolist()).intersection(
        selection_features.protocol_row_ids.tolist()
    ):
        raise HarmBenchOpenRoleError("fit and selection share a protocol row")
    if set(fit_features.groups.tolist()).intersection(selection_features.groups.tolist()):
        raise HarmBenchOpenRoleError("fit and selection split an independent group")
    descriptor = {
        "dataset_id": fit.dataset_id,
        "fit_capability_sha256": fit.capability_sha256,
        "selection_capability_sha256": selection.capability_sha256,
        "fit_manifest_sha256": fit.fit_manifest_sha256,
        "selection_manifest_sha256": selection.manifest_sha256,
        "cross_role_feature_roster_sha256": (
            fit.cross_role_feature_roster_sha256
        ),
    }
    capability_sha = hashlib.sha256(
        json.dumps(
            descriptor,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return OpenRoleCapabilities(
        dataset_id=fit.dataset_id,
        fit=fit.fit,
        selection=selection.selection,
        fit_manifest_sha256=fit.fit_manifest_sha256,
        selection_manifest_sha256=selection.manifest_sha256,
        cross_role_feature_roster_receipt=(
            fit.cross_role_feature_roster_receipt
        ),
        cross_role_feature_roster_sha256=(
            fit.cross_role_feature_roster_sha256
        ),
        capability_sha256=capability_sha,
    )


def make_open_role_capabilities(
    *,
    fit_features: OutcomeFreeRoleFeatures,
    fit_labels: object,
    fit_label_sha256: str,
    selection_features: OutcomeFreeRoleFeatures,
    label_order: Sequence[str],
    manifest_sha256: str,
) -> OpenRoleCapabilities:
    """Synthetic convenience composition; production uses split loaders."""

    if fit_features.role != FIT_ROLE or selection_features.role != SELECTION_ROLE:
        raise HarmBenchOpenRoleError("open-role role identities changed")

    roster_receipt = make_synthetic_cross_role_feature_roster_receipt(
        dataset_id=fit_features.dataset_id,
        fit_feature_projection_sha256=manifest_sha256,
        selection_feature_projection_sha256=manifest_sha256,
    )
    fit_feature_capability = make_fit_feature_capability(
        fit_features=fit_features,
        feature_manifest_sha256=manifest_sha256,
        cross_role_feature_roster_receipt=roster_receipt,
    )
    fit = make_fit_role_capability(
        fit_feature_capability=fit_feature_capability,
        fit_labels=fit_labels,
        fit_label_sha256=fit_label_sha256,
        label_order=label_order,
        fit_manifest_sha256=manifest_sha256,
    )
    selection = make_selection_feature_capability(
        selection_features=selection_features,
        manifest_sha256=manifest_sha256,
        cross_role_feature_roster_receipt=roster_receipt,
    )
    return compose_open_role_capabilities(fit, selection)


def _artifact_record(manifest: Mapping[str, object]) -> Mapping[str, object]:
    projection = manifest["feature_projection"]
    if not isinstance(projection, Mapping) or not isinstance(
        projection.get("artifact"), Mapping
    ):
        raise HarmBenchOpenRoleError("sanitized feature projection changed")
    return projection["artifact"]


def _features_from_sanitized_emotiontalk(
    *,
    role: str,
    projection: Mapping[str, object],
    payload: Mapping[str, np.ndarray],
    observed_sha256: str,
) -> OutcomeFreeRoleFeatures:
    artifact = projection["artifact"]
    if not isinstance(artifact, Mapping):
        raise HarmBenchOpenRoleError("EmotionTalk feature artifact record changed")
    if projection["artifact_schema_version"] != EMOTIONTALK_FEATURE_SCHEMA:
        raise HarmBenchOpenRoleError("EmotionTalk feature schema changed")
    if (
        _scalar_text(payload, "schema_version") != EMOTIONTALK_FEATURE_SCHEMA
        or _scalar_text(payload, "dataset_id") != "EmotionTalk"
        or _scalar_text(payload, "role") != role
        or _scalar_text(payload, "split_protocol_id") != "scu_set_exploration_v1"
        or _scalar_text(payload, "row_alignment_sha256")
        != str(artifact["row_alignment_sha256"])
    ):
        raise HarmBenchOpenRoleError("EmotionTalk feature identity changed")
    _valid_sha256(
        _scalar_text(payload, "source_feature_config_sha256"),
        name="source_feature_config_sha256",
    )
    rows = int(artifact["rows"])
    audio = np.asarray(payload["audio_features"])
    video = np.asarray(payload["video_features"])
    if audio.shape != (rows, int(artifact["audio_dimension"])):
        raise HarmBenchOpenRoleError("EmotionTalk audio dimensions changed")
    if video.shape != (rows, int(artifact["video_dimension"])):
        raise HarmBenchOpenRoleError("EmotionTalk video dimensions changed")
    buckets = np.asarray(payload["role_buckets"], dtype=np.int64)
    low, high = FROZEN_ROLE_RANGES[role]
    if buckets.shape != (rows,) or np.any((buckets < low) | (buckets > high)):
        raise HarmBenchOpenRoleError("EmotionTalk role bucket changed")
    groups = np.asarray(payload["opaque_group_hashes"], dtype=str)
    speakers = np.asarray(
        [
            hashlib.sha256(f"EmotionTalk-speaker\x1f{value}".encode()).hexdigest()
            for value in np.asarray(payload["speaker_tokens"], dtype=str)
        ]
    )
    result = make_outcome_free_role_features(
        dataset_id="EmotionTalk",
        role=role,
        keys=payload["opaque_row_hashes"],
        texts=np.asarray(payload["texts"], dtype=str).tolist(),
        audio=audio,
        video=video,
        groups=groups,
        speaker_identity=speakers,
        turn_ids=payload["turn_ids"],
        protocol_row_ids=payload["protocol_row_ids"],
        row_alignment_sha256=str(artifact["row_alignment_sha256"]),
        feature_sha256=observed_sha256,
    )
    if result.rows != rows or len(set(groups.tolist())) != int(
        artifact["independent_groups"]
    ):
        raise HarmBenchOpenRoleError("EmotionTalk feature counts changed")
    if int(result.history_eligible.sum()) != int(artifact["history_eligible_rows"]):
        raise HarmBenchOpenRoleError("EmotionTalk history count changed")
    return result


def _features_from_sanitized_meld(
    *,
    role: str,
    projection: Mapping[str, object],
    payload: Mapping[str, np.ndarray],
    observed_sha256: str,
) -> OutcomeFreeRoleFeatures:
    artifact = projection["artifact"]
    if not isinstance(artifact, Mapping):
        raise HarmBenchOpenRoleError("MELD feature artifact record changed")
    if projection["artifact_schema_version"] != MELD_SIDECAR_SCHEMA:
        raise HarmBenchOpenRoleError("MELD feature schema changed")
    if (
        _scalar_text(payload, "schema_version") != MELD_SIDECAR_SCHEMA
        or _scalar_text(payload, "role") != role
        or _scalar_text(payload, "row_alignment_sha256")
        != str(artifact["row_alignment_sha256"])
    ):
        raise HarmBenchOpenRoleError("MELD feature identity changed")
    rows = int(artifact["rows"])
    audio = np.asarray(payload["audio_mean_std"])
    video = np.asarray(payload["video_mean_std"])
    if audio.shape != (rows, int(artifact["audio_dimension"])) or video.shape != (
        rows,
        int(artifact["video_dimension"]),
    ):
        raise HarmBenchOpenRoleError("MELD modality dimensions changed")
    dialogues = np.asarray(payload["dialogue_codes"], dtype=np.int64)
    speakers_raw = np.asarray(payload["speaker_codes"], dtype=np.int64)
    orders = np.asarray(payload["utterance_order"], dtype=np.int64)
    if any(value.shape != (rows,) for value in (dialogues, speakers_raw, orders)):
        raise HarmBenchOpenRoleError("MELD structural rows changed")
    groups = np.asarray([f"MELD/{int(value)}" for value in dialogues])
    speakers = np.asarray(
        [
            hashlib.sha256(f"MELD-speaker\x1f{int(value)}".encode()).hexdigest()
            for value in speakers_raw
        ]
    )
    keys = np.asarray(
        [
            hashlib.sha256(
                f"MELD\x1f{int(dialogue)}\x1f{int(order)}".encode()
            ).hexdigest()
            for dialogue, order in zip(dialogues, orders, strict=True)
        ]
    )
    result = make_outcome_free_role_features(
        dataset_id="MELD",
        role=role,
        keys=keys,
        texts=np.asarray(payload["utterances"], dtype=str).tolist(),
        audio=audio,
        video=video,
        groups=groups,
        speaker_identity=speakers,
        turn_ids=orders,
        protocol_row_ids=payload["protocol_row_ids"],
        row_alignment_sha256=str(artifact["row_alignment_sha256"]),
        feature_sha256=observed_sha256,
    )
    if result.rows != rows or len(set(dialogues.tolist())) != int(
        artifact["independent_groups"]
    ):
        raise HarmBenchOpenRoleError("MELD feature counts changed")
    if int(result.history_eligible.sum()) != int(artifact["history_eligible_rows"]):
        raise HarmBenchOpenRoleError("MELD history count changed")
    return result


def _load_sanitized_feature_capability(
    *,
    dataset_id: str,
    role: str,
    capability_root: Path,
    manifest_path: Path,
    cross_role_feature_roster_receipt: CrossRoleFeatureRosterReceipt,
    expected_cross_role_feature_roster_sha256: str,
) -> FitFeatureCapability | SelectionFeatureCapability:
    """Load exactly one feature manifest and one feature NPZ."""

    try:
        roster_receipt = validate_cross_role_feature_roster_receipt(
            cross_role_feature_roster_receipt,
            expected_roster_sha256=expected_cross_role_feature_roster_sha256,
            expected_dataset=dataset_id,
        )
        expected_projection = roster_feature_projection_sha256(
            roster_receipt,
            role=role,
            expected_roster_sha256=expected_cross_role_feature_roster_sha256,
        )
    except ValueError as error:
        raise HarmBenchOpenRoleError(
            f"cross-role feature roster receipt changed: {error}"
        ) from error
    expected_roster = roster_receipt.roster_sha256
    verified_manifest = load_feature_manifest(
        manifest_path, expected_dataset=dataset_id, expected_role=role
    )
    manifest = verified_manifest.payload
    if manifest["cross_role_feature_roster_sha256"] != expected_roster:
        raise HarmBenchOpenRoleError("cross-role feature roster binding changed")
    if manifest["feature_projection_sha256"] != expected_projection:
        raise HarmBenchOpenRoleError("feature projection differs from roster")
    projection = manifest["feature_projection"]
    if not isinstance(projection, Mapping):
        raise HarmBenchOpenRoleError("feature projection changed")
    artifact = _artifact_record(manifest)
    feature_path = exact_artifact_path(capability_root, artifact["filename"])
    if dataset_id == "EmotionTalk":
        loaded = read_verified_npz(
            feature_path,
            expected_sha256=str(artifact["sha256"]),
            expected_fields=set(EMOTIONTALK_FEATURE_FIELDS),
            name=f"EmotionTalk {role} sanitized feature",
        )
        features = _features_from_sanitized_emotiontalk(
            role=role,
            projection=projection,
            payload=loaded.arrays,
            observed_sha256=loaded.sha256,
        )
    elif dataset_id == "MELD":
        loaded = read_verified_npz(
            feature_path,
            expected_sha256=str(artifact["sha256"]),
            expected_fields=set(MELD_FEATURE_FIELDS),
            name=f"MELD {role} sanitized feature",
        )
        features = _features_from_sanitized_meld(
            role=role,
            projection=projection,
            payload=loaded.arrays,
            observed_sha256=loaded.sha256,
        )
    else:
        raise HarmBenchOpenRoleError("unsupported sanitized dataset")
    if role == FIT_ROLE:
        return make_fit_feature_capability(
            fit_features=features,
            feature_manifest_sha256=verified_manifest.sha256,
            cross_role_feature_roster_receipt=roster_receipt,
        )
    return make_selection_feature_capability(
        selection_features=features,
        manifest_sha256=verified_manifest.sha256,
        cross_role_feature_roster_receipt=roster_receipt,
    )


def load_emotiontalk_fit_feature_capability(
    *,
    capability_root: Path,
    manifest_path: Path,
    cross_role_feature_roster_receipt: CrossRoleFeatureRosterReceipt,
    expected_cross_role_feature_roster_sha256: str,
) -> FitFeatureCapability:
    value = _load_sanitized_feature_capability(
        dataset_id="EmotionTalk",
        role=FIT_ROLE,
        capability_root=capability_root,
        manifest_path=manifest_path,
        cross_role_feature_roster_receipt=(
            cross_role_feature_roster_receipt
        ),
        expected_cross_role_feature_roster_sha256=(
            expected_cross_role_feature_roster_sha256
        ),
    )
    if not isinstance(value, FitFeatureCapability):
        raise HarmBenchOpenRoleError("EmotionTalk fit feature capability type changed")
    return value


def load_meld_fit_feature_capability(
    *,
    capability_root: Path,
    manifest_path: Path,
    cross_role_feature_roster_receipt: CrossRoleFeatureRosterReceipt,
    expected_cross_role_feature_roster_sha256: str,
) -> FitFeatureCapability:
    value = _load_sanitized_feature_capability(
        dataset_id="MELD",
        role=FIT_ROLE,
        capability_root=capability_root,
        manifest_path=manifest_path,
        cross_role_feature_roster_receipt=(
            cross_role_feature_roster_receipt
        ),
        expected_cross_role_feature_roster_sha256=(
            expected_cross_role_feature_roster_sha256
        ),
    )
    if not isinstance(value, FitFeatureCapability):
        raise HarmBenchOpenRoleError("MELD fit feature capability type changed")
    return value


def load_emotiontalk_selection_feature_capability(
    *,
    capability_root: Path,
    manifest_path: Path,
    cross_role_feature_roster_receipt: CrossRoleFeatureRosterReceipt,
    expected_cross_role_feature_roster_sha256: str,
) -> SelectionFeatureCapability:
    value = _load_sanitized_feature_capability(
        dataset_id="EmotionTalk",
        role=SELECTION_ROLE,
        capability_root=capability_root,
        manifest_path=manifest_path,
        cross_role_feature_roster_receipt=(
            cross_role_feature_roster_receipt
        ),
        expected_cross_role_feature_roster_sha256=(
            expected_cross_role_feature_roster_sha256
        ),
    )
    if not isinstance(value, SelectionFeatureCapability):
        raise HarmBenchOpenRoleError("EmotionTalk selection capability type changed")
    return value


def load_meld_selection_feature_capability(
    *,
    capability_root: Path,
    manifest_path: Path,
    cross_role_feature_roster_receipt: CrossRoleFeatureRosterReceipt,
    expected_cross_role_feature_roster_sha256: str,
) -> SelectionFeatureCapability:
    value = _load_sanitized_feature_capability(
        dataset_id="MELD",
        role=SELECTION_ROLE,
        capability_root=capability_root,
        manifest_path=manifest_path,
        cross_role_feature_roster_receipt=(
            cross_role_feature_roster_receipt
        ),
        expected_cross_role_feature_roster_sha256=(
            expected_cross_role_feature_roster_sha256
        ),
    )
    if not isinstance(value, SelectionFeatureCapability):
        raise HarmBenchOpenRoleError("MELD selection capability type changed")
    return value


def _load_fit_training_capability(
    *,
    dataset_id: str,
    fit_feature_capability: FitFeatureCapability,
    capability_root: Path,
    manifest_path: Path,
) -> FitRoleCapability:
    fit_feature_capability = validate_fit_feature_capability(fit_feature_capability)
    if fit_feature_capability.dataset_id != dataset_id:
        raise HarmBenchOpenRoleError("fit feature dataset changed")
    verified_manifest = load_fit_training_manifest(
        manifest_path, expected_dataset=dataset_id
    )
    manifest = verified_manifest.payload
    if (
        manifest["fit_feature_manifest_sha256"]
        != fit_feature_capability.feature_manifest_sha256
        or manifest["cross_role_feature_roster_sha256"]
        != fit_feature_capability.cross_role_feature_roster_sha256
    ):
        raise HarmBenchOpenRoleError("fit training feature binding changed")
    projection = manifest["fit_target_projection"]
    if not isinstance(projection, Mapping) or not isinstance(
        projection.get("artifact"), Mapping
    ):
        raise HarmBenchOpenRoleError("fit target projection changed")
    artifact = projection["artifact"]
    if (
        int(artifact["rows"]) != fit_feature_capability.fit.rows
        or str(artifact["row_alignment_sha256"])
        != fit_feature_capability.fit.row_alignment_sha256
    ):
        raise HarmBenchOpenRoleError("fit target alignment changed")
    target_path = exact_artifact_path(capability_root, artifact["filename"])
    if dataset_id == "EmotionTalk":
        loaded = read_verified_npz(
            target_path,
            expected_sha256=str(artifact["sha256"]),
            expected_fields=set(SANITIZED_EMOTIONTALK_FIT_LABEL_FIELDS),
            name="EmotionTalk sanitized fit target",
        )
        if (
            projection["artifact_schema_version"]
            != SANITIZED_EMOTIONTALK_FIT_LABEL_SCHEMA
            or _scalar_text(loaded.arrays, "schema_version")
            != SANITIZED_EMOTIONTALK_FIT_LABEL_SCHEMA
            or _scalar_text(loaded.arrays, "dataset_id") != "EmotionTalk"
            or _scalar_text(loaded.arrays, "role") != FIT_ROLE
            or _scalar_text(loaded.arrays, "split_protocol_id")
            != "scu_set_exploration_v1"
            or _scalar_text(loaded.arrays, "row_alignment_sha256")
            != fit_feature_capability.fit.row_alignment_sha256
        ):
            raise HarmBenchOpenRoleError("EmotionTalk fit target identity changed")
    elif dataset_id == "MELD":
        loaded = read_verified_npz(
            target_path,
            expected_sha256=str(artifact["sha256"]),
            expected_fields=set(MELD_LABEL_FIELDS),
            name="MELD sanitized fit target",
        )
        if (
            projection["artifact_schema_version"] != MELD_SIDECAR_SCHEMA
            or _scalar_text(loaded.arrays, "schema_version") != MELD_SIDECAR_SCHEMA
            or _scalar_text(loaded.arrays, "role") != FIT_ROLE
            or _scalar_text(loaded.arrays, "row_alignment_sha256")
            != fit_feature_capability.fit.row_alignment_sha256
        ):
            raise HarmBenchOpenRoleError("MELD fit target identity changed")
    else:
        raise HarmBenchOpenRoleError("unsupported fit training dataset")
    return make_fit_role_capability(
        fit_feature_capability=fit_feature_capability,
        fit_labels=loaded.arrays["labels"],
        fit_label_sha256=loaded.sha256,
        label_order=projection["class_names"],
        fit_manifest_sha256=verified_manifest.sha256,
    )


def load_emotiontalk_fit_role_capability(
    *,
    fit_feature_capability: FitFeatureCapability,
    capability_root: Path,
    manifest_path: Path,
) -> FitRoleCapability:
    """Combine EmotionTalk fit features with a fit-only target capability."""

    return _load_fit_training_capability(
        dataset_id="EmotionTalk",
        fit_feature_capability=fit_feature_capability,
        capability_root=capability_root,
        manifest_path=manifest_path,
    )


def load_meld_fit_role_capability(
    *,
    fit_feature_capability: FitFeatureCapability,
    capability_root: Path,
    manifest_path: Path,
) -> FitRoleCapability:
    """Combine MELD fit features with a fit-only target capability."""

    return _load_fit_training_capability(
        dataset_id="MELD",
        fit_feature_capability=fit_feature_capability,
        capability_root=capability_root,
        manifest_path=manifest_path,
    )


def _read_emotiontalk_manifest(path: Path) -> dict[str, object]:
    manifest = _read_unique_json(path)
    expected_root = {
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
    if set(manifest) != expected_root:
        raise HarmBenchOpenRoleError("EmotionTalk manifest schema changed")
    if (
        manifest["schema_version"] != EMOTIONTALK_MANIFEST_SCHEMA
        or manifest["protocol"] != EMOTIONTALK_PROTOCOL
        or manifest["status"]
        != "strict_open_role_feature_and_label_sidecars_created_and_hashed"
        or manifest["dataset_id"] != "EmotionTalk"
        or manifest["split_protocol_id"] != "scu_set_exploration_v1"
        or tuple(manifest["label_order"]) != tuple(EMOTIONTALK_LABEL_NAMES)
    ):
        raise HarmBenchOpenRoleError("EmotionTalk manifest identity changed")
    if not isinstance(manifest["roles"], Mapping) or set(manifest["roles"]) != set(
        OPEN_ROLES
    ):
        raise HarmBenchOpenRoleError("EmotionTalk open-role roster changed")
    source = manifest["source_contract"]
    expected_source = {
        "label_archive",
        "media_features",
        "transcription",
        "feature_config_sha256",
        "trusted_source_boundary_only",
        "validation_or_test_label_payload_opened",
    }
    if not isinstance(source, Mapping) or set(source) != expected_source:
        raise HarmBenchOpenRoleError("EmotionTalk source contract changed")
    for name in ("label_archive", "media_features", "transcription", "feature_config_sha256"):
        _valid_sha256(source[name], name=f"source_contract.{name}")
    if (
        source["trusted_source_boundary_only"] is not True
        or source["validation_or_test_label_payload_opened"] is not False
    ):
        raise HarmBenchOpenRoleError("EmotionTalk source boundary changed")
    seal = manifest["seal_contract"]
    if seal != {
        "model_runner_opens_upstream_media_npz_or_transcription": False,
        "open_role_runner_may_load_only": list(OPEN_ROLES),
        "calibration_holdout_validation_test_sidecars_created": False,
        "allow_pickle_required_to_load_sidecars": False,
    }:
        raise HarmBenchOpenRoleError("EmotionTalk physical seal contract changed")
    if manifest["public_content_audit"] != {
        "contains_labels_or_class_counts": False,
        "contains_row_group_or_speaker_identifiers": False,
        "contains_transcripts_or_embeddings": False,
        "contains_private_absolute_paths": False,
    }:
        raise HarmBenchOpenRoleError("EmotionTalk public-content audit changed")
    _valid_sha256(manifest["config_sha256"], name="config_sha256")
    for role in OPEN_ROLES:
        record = manifest["roles"][role]
        if not isinstance(record, Mapping) or set(record) != EMOTIONTALK_MANIFEST_ROLE_FIELDS:
            raise HarmBenchOpenRoleError("EmotionTalk role record changed")
        if (
            record["feature_filename"] != f"features_{role}.npz"
            or record["label_filename"] != f"labels_{role}.npz"
        ):
            raise HarmBenchOpenRoleError("EmotionTalk sidecar filename changed")
        for name in ("feature_sha256", "label_sha256", "row_alignment_sha256"):
            _valid_sha256(record[name], name=f"roles.{role}.{name}")
    return manifest


def _emotiontalk_features(
    sidecar_dir: Path,
    manifest: Mapping[str, object],
    *,
    role: str,
) -> OutcomeFreeRoleFeatures:
    record = manifest["roles"][role]
    feature_path = sidecar_dir / str(record["feature_filename"])
    observed_sha = sha256_file(feature_path)
    if observed_sha != str(record["feature_sha256"]):
        raise HarmBenchOpenRoleError("EmotionTalk feature SHA differs from manifest")
    payload = _load_npz(
        feature_path, EMOTIONTALK_FEATURE_FIELDS, name=f"EmotionTalk {role} feature"
    )
    if (
        _scalar_text(payload, "schema_version") != EMOTIONTALK_FEATURE_SCHEMA
        or _scalar_text(payload, "dataset_id") != "EmotionTalk"
        or _scalar_text(payload, "role") != role
        or _scalar_text(payload, "split_protocol_id") != "scu_set_exploration_v1"
        or _scalar_text(payload, "row_alignment_sha256")
        != str(record["row_alignment_sha256"])
        or _scalar_text(payload, "source_feature_config_sha256")
        != str(manifest["source_contract"]["feature_config_sha256"])
    ):
        raise HarmBenchOpenRoleError("EmotionTalk feature identity changed")
    rows = int(record["rows"])
    if payload["audio_features"].shape != (rows, int(record["audio_dimension"])):
        raise HarmBenchOpenRoleError("EmotionTalk audio dimensions changed")
    if payload["video_features"].shape != (rows, int(record["video_dimension"])):
        raise HarmBenchOpenRoleError("EmotionTalk video dimensions changed")
    buckets = np.asarray(payload["role_buckets"], dtype=np.int64)
    low, high = FROZEN_ROLE_RANGES[role]
    if buckets.shape != (rows,) or np.any((buckets < low) | (buckets > high)):
        raise HarmBenchOpenRoleError("EmotionTalk role bucket changed")
    groups = np.asarray(payload["opaque_group_hashes"], dtype=str)
    speakers = np.asarray(
        [
            hashlib.sha256(f"EmotionTalk-speaker\x1f{value}".encode()).hexdigest()
            for value in np.asarray(payload["speaker_tokens"], dtype=str)
        ]
    )
    result = make_outcome_free_role_features(
        dataset_id="EmotionTalk",
        role=role,
        keys=payload["opaque_row_hashes"],
        texts=np.asarray(payload["texts"], dtype=str).tolist(),
        audio=payload["audio_features"],
        video=payload["video_features"],
        groups=groups,
        speaker_identity=speakers,
        turn_ids=payload["turn_ids"],
        protocol_row_ids=payload["protocol_row_ids"],
        row_alignment_sha256=str(record["row_alignment_sha256"]),
        feature_sha256=observed_sha,
    )
    if result.rows != rows or len(set(groups.tolist())) != int(record["groups"]):
        raise HarmBenchOpenRoleError("EmotionTalk feature counts differ from manifest")
    if int(result.history_eligible.sum()) != int(record["history_eligible_rows"]):
        raise HarmBenchOpenRoleError("EmotionTalk history count differs from manifest")
    return result


def _emotiontalk_fit_labels(
    sidecar_dir: Path, manifest: Mapping[str, object]
) -> tuple[np.ndarray, str]:
    record = manifest["roles"][FIT_ROLE]
    path = sidecar_dir / str(record["label_filename"])
    observed_sha = sha256_file(path)
    if observed_sha != str(record["label_sha256"]):
        raise HarmBenchOpenRoleError("EmotionTalk fit label SHA differs from manifest")
    payload = _load_npz(
        path, EMOTIONTALK_LABEL_FIELDS, name="EmotionTalk fit label"
    )
    if (
        _scalar_text(payload, "schema_version") != EMOTIONTALK_LABEL_SCHEMA
        or _scalar_text(payload, "dataset_id") != "EmotionTalk"
        or _scalar_text(payload, "role") != FIT_ROLE
        or _scalar_text(payload, "split_protocol_id") != "scu_set_exploration_v1"
        or _scalar_text(payload, "row_alignment_sha256")
        != str(record["row_alignment_sha256"])
        or _scalar_text(payload, "source_label_sha256")
        != str(manifest["source_contract"]["label_archive"])
    ):
        raise HarmBenchOpenRoleError("EmotionTalk fit label identity changed")
    return np.asarray(payload["labels"]), observed_sha


def load_emotiontalk_legacy_fit_role_capability(
    *, sidecar_dir: Path, manifest_path: Path
) -> FitRoleCapability:
    """Legacy aggregate-manifest smoke/migration loader; not for production."""

    manifest = _read_emotiontalk_manifest(manifest_path)
    fit = _emotiontalk_features(sidecar_dir, manifest, role=FIT_ROLE)
    labels, label_sha = _emotiontalk_fit_labels(sidecar_dir, manifest)
    manifest_sha = sha256_file(manifest_path)
    roster_receipt = make_legacy_cross_role_feature_roster_receipt(
        dataset_id="EmotionTalk",
        fit_feature_projection_sha256=manifest_sha,
        selection_feature_projection_sha256=manifest_sha,
    )
    fit_feature_capability = make_fit_feature_capability(
        fit_features=fit,
        feature_manifest_sha256=manifest_sha,
        cross_role_feature_roster_receipt=roster_receipt,
    )
    return make_fit_role_capability(
        fit_feature_capability=fit_feature_capability,
        fit_labels=labels,
        fit_label_sha256=label_sha,
        label_order=manifest["label_order"],
        fit_manifest_sha256=manifest_sha,
    )


def load_emotiontalk_legacy_selection_feature_capability(
    *,
    sidecar_dir: Path,
    manifest_path: Path,
    cross_role_feature_roster_receipt: CrossRoleFeatureRosterReceipt,
) -> SelectionFeatureCapability:
    """Legacy aggregate-manifest smoke loader; not an isolated capability."""

    manifest = _read_emotiontalk_manifest(manifest_path)
    selection = _emotiontalk_features(sidecar_dir, manifest, role=SELECTION_ROLE)
    return make_selection_feature_capability(
        selection_features=selection,
        manifest_sha256=sha256_file(manifest_path),
        cross_role_feature_roster_receipt=cross_role_feature_roster_receipt,
    )


def load_emotiontalk_legacy_open_role_capabilities(
    *, sidecar_dir: Path, manifest_path: Path
) -> OpenRoleCapabilities:
    """Smoke-only convenience composition of the two production capabilities."""

    fit = load_emotiontalk_legacy_fit_role_capability(
        sidecar_dir=sidecar_dir,
        manifest_path=manifest_path,
    )
    selection = load_emotiontalk_legacy_selection_feature_capability(
        sidecar_dir=sidecar_dir,
        manifest_path=manifest_path,
        cross_role_feature_roster_receipt=(
            fit.cross_role_feature_roster_receipt
        ),
    )
    return compose_open_role_capabilities(fit, selection)


def _meld_features(
    sidecar_dir: Path,
    manifest: Mapping[str, object],
    *,
    role: str,
) -> OutcomeFreeRoleFeatures:
    record = manifest["roles"][role]
    if not isinstance(record, Mapping) or set(record) != MELD_ROLE_RECORD_FIELDS:
        raise HarmBenchOpenRoleError("MELD role record changed")
    feature_path = sidecar_dir / str(record["feature_filename"])
    observed_sha = sha256_file(feature_path)
    if observed_sha != str(record["feature_sha256"]):
        raise HarmBenchOpenRoleError("MELD feature SHA differs from manifest")
    payload = _load_npz(feature_path, MELD_FEATURE_FIELDS, name=f"MELD {role} feature")
    if (
        _scalar_text(payload, "schema_version") != MELD_SIDECAR_SCHEMA
        or _scalar_text(payload, "role") != role
        or _scalar_text(payload, "row_alignment_sha256")
        != str(record["row_alignment_sha256"])
    ):
        raise HarmBenchOpenRoleError("MELD feature identity changed")
    rows = int(record["rows"])
    audio = payload["audio_mean_std"]
    video = payload["video_mean_std"]
    if audio.shape != (rows, int(record["audio_dimension"])) or video.shape != (
        rows,
        int(record["video_dimension"]),
    ):
        raise HarmBenchOpenRoleError("MELD modality dimensions changed")
    dialogues = np.asarray(payload["dialogue_codes"], dtype=np.int64)
    speakers_raw = np.asarray(payload["speaker_codes"], dtype=np.int64)
    orders = np.asarray(payload["utterance_order"], dtype=np.int64)
    if any(value.shape != (rows,) for value in (dialogues, speakers_raw, orders)):
        raise HarmBenchOpenRoleError("MELD structural rows changed")
    groups = np.asarray([f"MELD/{int(value)}" for value in dialogues])
    speakers = np.asarray(
        [
            hashlib.sha256(f"MELD-speaker\x1f{int(value)}".encode()).hexdigest()
            for value in speakers_raw
        ]
    )
    keys = np.asarray(
        [
            hashlib.sha256(f"MELD\x1f{int(dialogue)}\x1f{int(order)}".encode()).hexdigest()
            for dialogue, order in zip(dialogues, orders, strict=True)
        ]
    )
    result = make_outcome_free_role_features(
        dataset_id="MELD",
        role=role,
        keys=keys,
        texts=np.asarray(payload["utterances"], dtype=str).tolist(),
        audio=audio,
        video=video,
        groups=groups,
        speaker_identity=speakers,
        turn_ids=orders,
        protocol_row_ids=payload["protocol_row_ids"],
        row_alignment_sha256=str(record["row_alignment_sha256"]),
        feature_sha256=observed_sha,
    )
    if result.rows != rows or len(set(dialogues.tolist())) != int(record["dialogues"]):
        raise HarmBenchOpenRoleError("MELD feature counts differ from manifest")
    if int(result.history_eligible.sum()) != int(record["history_eligible_rows"]):
        raise HarmBenchOpenRoleError("MELD history count differs from manifest")
    return result


def _meld_fit_labels(
    sidecar_dir: Path, manifest: Mapping[str, object]
) -> tuple[np.ndarray, str]:
    record = manifest["roles"][FIT_ROLE]
    path = sidecar_dir / str(record["label_filename"])
    observed_sha = sha256_file(path)
    if observed_sha != str(record["label_sha256"]):
        raise HarmBenchOpenRoleError("MELD fit label SHA differs from manifest")
    payload = _load_npz(path, MELD_LABEL_FIELDS, name="MELD fit label")
    if (
        _scalar_text(payload, "schema_version") != MELD_SIDECAR_SCHEMA
        or _scalar_text(payload, "role") != FIT_ROLE
        or _scalar_text(payload, "row_alignment_sha256")
        != str(record["row_alignment_sha256"])
    ):
        raise HarmBenchOpenRoleError("MELD fit label identity changed")
    return np.asarray(payload["labels"]), observed_sha


def _read_verified_meld_manifest(manifest_path: Path) -> dict[str, object]:
    """Validate MELD manifest metadata without resolving any role archive."""

    # The legacy verifier is exhaustive but uses the standard JSON decoder;
    # pre-read with duplicate rejection before invoking it.
    unique_manifest = _read_unique_json(manifest_path)
    manifest = _read_meld_manifest(manifest_path)
    if unique_manifest != manifest:
        raise HarmBenchOpenRoleError("MELD manifest changed across verification reads")
    return manifest


def load_meld_legacy_fit_role_capability(
    *, sidecar_dir: Path, manifest_path: Path
) -> FitRoleCapability:
    """Legacy aggregate-manifest smoke/migration loader; not for production."""

    manifest = _read_verified_meld_manifest(manifest_path)
    fit = _meld_features(sidecar_dir, manifest, role=FIT_ROLE)
    labels, label_sha = _meld_fit_labels(sidecar_dir, manifest)
    manifest_sha = sha256_file(manifest_path)
    roster_receipt = make_legacy_cross_role_feature_roster_receipt(
        dataset_id="MELD",
        fit_feature_projection_sha256=manifest_sha,
        selection_feature_projection_sha256=manifest_sha,
    )
    fit_feature_capability = make_fit_feature_capability(
        fit_features=fit,
        feature_manifest_sha256=manifest_sha,
        cross_role_feature_roster_receipt=roster_receipt,
    )
    return make_fit_role_capability(
        fit_feature_capability=fit_feature_capability,
        fit_labels=labels,
        fit_label_sha256=label_sha,
        label_order=manifest["label_order"],
        fit_manifest_sha256=manifest_sha,
    )


def load_meld_legacy_selection_feature_capability(
    *,
    sidecar_dir: Path,
    manifest_path: Path,
    cross_role_feature_roster_receipt: CrossRoleFeatureRosterReceipt,
) -> SelectionFeatureCapability:
    """Legacy aggregate-manifest smoke loader; not an isolated capability."""

    manifest = _read_verified_meld_manifest(manifest_path)
    selection = _meld_features(sidecar_dir, manifest, role=SELECTION_ROLE)
    return make_selection_feature_capability(
        selection_features=selection,
        manifest_sha256=sha256_file(manifest_path),
        cross_role_feature_roster_receipt=cross_role_feature_roster_receipt,
    )


def load_meld_legacy_open_role_capabilities(
    *, sidecar_dir: Path, manifest_path: Path
) -> OpenRoleCapabilities:
    """Smoke-only convenience composition; never use this in a training runner."""

    fit = load_meld_legacy_fit_role_capability(
        sidecar_dir=sidecar_dir,
        manifest_path=manifest_path,
    )
    selection = load_meld_legacy_selection_feature_capability(
        sidecar_dir=sidecar_dir,
        manifest_path=manifest_path,
        cross_role_feature_roster_receipt=(
            fit.cross_role_feature_roster_receipt
        ),
    )
    return compose_open_role_capabilities(fit, selection)
