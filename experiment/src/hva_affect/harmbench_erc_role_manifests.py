"""Outcome-isolated HarmBench-ERC role manifests and trusted migration helpers.

This module is intentionally filesystem-capability oriented.  A model runner
receives one canonical role manifest and one exact artifact root; it never
receives the legacy aggregate manifest.  The selection manifest is restricted
to a feature-only vocabulary and cannot carry a target filename, digest, class
roster, or metadata for another role.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Mapping, Sequence

import numpy as np

from .data_contract import ContractError
from .emotiontalk_role_sidecar import (
    FIT_ROLE,
    LABEL_FIELDS as LEGACY_EMOTIONTALK_LABEL_FIELDS,
    LABEL_SCHEMA as LEGACY_EMOTIONTALK_LABEL_SCHEMA,
    SELECTION_ROLE,
)


class HarmBenchRoleManifestError(ContractError):
    """Raised when an isolated manifest or artifact fails closed."""


PROTOCOL_ID = "harmbench_erc_v1"
SPLIT_PROTOCOL_ID = "scu_set_exploration_v1"
FEATURE_PROJECTION_SCHEMA = "harmbench_erc_feature_projection_v1"
FIT_TARGET_PROJECTION_SCHEMA = "harmbench_erc_fit_target_projection_v1"
CROSS_ROLE_FEATURE_ROSTER_SCHEMA = "harmbench_erc_cross_role_feature_roster_v1"
CROSS_ROLE_FEATURE_ROSTER_RECEIPT_SCHEMA = (
    "harmbench_erc_cross_role_feature_roster_receipt_v1"
)
FIT_FEATURE_MANIFEST_SCHEMA = "harmbench_erc_fit_feature_manifest_v1"
FIT_TRAINING_MANIFEST_SCHEMA = "harmbench_erc_fit_training_manifest_v1"
SELECTION_FEATURE_MANIFEST_SCHEMA = "harmbench_erc_selection_feature_manifest_v1"
SANITIZED_EMOTIONTALK_FIT_LABEL_SCHEMA = (
    "harmbench_erc_emotiontalk_fit_label_sidecar_v1"
)
SANITIZED_EMOTIONTALK_FIT_LABEL_FIELDS = {
    "schema_version",
    "dataset_id",
    "role",
    "split_protocol_id",
    "row_alignment_sha256",
    "labels",
}

_HEX = frozenset("0123456789abcdef")
_SELECTION_FORBIDDEN_TOKENS = (
    "label",
    "outcome",
    "calibration",
    "holdout",
    "validation",
    "test",
)


@dataclass(frozen=True)
class VerifiedJson:
    payload: dict[str, object]
    sha256: str


@dataclass(frozen=True)
class VerifiedNpz:
    arrays: dict[str, np.ndarray]
    sha256: str


@dataclass(frozen=True)
class CrossRoleFeatureRosterReceipt:
    """Typed, immutable view of one canonical outcome-free roster file.

    Production callers obtain this value only through
    :func:`load_cross_role_feature_roster`, which requires an external file
    SHA.  The two role projection digests are therefore derived from the exact
    roster bytes instead of being supplied independently to a role loader.
    """

    schema_version: str
    roster_schema_version: str
    dataset_id: str
    protocol_id: str
    split_protocol_id: str
    payload_scope: str
    fit_feature_projection_sha256: str
    selection_feature_projection_sha256: str
    roster_sha256: str

    def __post_init__(self) -> None:
        validate_cross_role_feature_roster_receipt(
            self,
            expected_roster_sha256=self.roster_sha256,
            expected_dataset=self.dataset_id,
        )


def _sha256(value: object, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or set(digest) - _HEX:
        raise HarmBenchRoleManifestError(f"{name} must be a lowercase SHA-256")
    return digest


def _exact_keys(
    value: object, expected: set[str], *, name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise HarmBenchRoleManifestError(f"{name} has an unexpected exact schema")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise HarmBenchRoleManifestError("manifest is not canonical JSON data") from error


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HarmBenchRoleManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _is_reparse(stat_result: os.stat_result) -> bool:
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    return bool(attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)))


def _plain_file(path: Path, *, name: str) -> os.stat_result:
    try:
        observed = path.lstat()
    except OSError as error:
        raise HarmBenchRoleManifestError(f"cannot stat exact {name}") from error
    if stat.S_ISLNK(observed.st_mode) or _is_reparse(observed):
        raise HarmBenchRoleManifestError(f"{name} cannot be a symlink or reparse point")
    if not stat.S_ISREG(observed.st_mode):
        raise HarmBenchRoleManifestError(f"{name} must be a plain file")
    return observed


def _same_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        int(first.st_dev),
        int(first.st_ino),
        int(first.st_size),
    ) == (
        int(second.st_dev),
        int(second.st_ino),
        int(second.st_size),
    )


def _path_still_names_handle(path: Path, handle_stat: os.stat_result, *, name: str) -> None:
    current = _plain_file(path, name=name)
    if not _same_identity(handle_stat, current):
        raise HarmBenchRoleManifestError(f"{name} changed identity during verified read")


def read_canonical_json(path: str | Path, *, name: str = "manifest") -> VerifiedJson:
    """Hash and parse one immutable byte snapshot from one open handle."""

    source = Path(path)
    before_path = _plain_file(source, name=name)
    try:
        with source.open("rb") as handle:
            before_handle = os.fstat(handle.fileno())
            if not _same_identity(before_path, before_handle):
                raise HarmBenchRoleManifestError(f"{name} changed before verified read")
            raw = handle.read()
            handle.seek(0)
            repeated = handle.read()
            after_handle = os.fstat(handle.fileno())
    except HarmBenchRoleManifestError:
        raise
    except OSError as error:
        raise HarmBenchRoleManifestError(f"cannot read exact {name}") from error
    if not _same_identity(before_handle, after_handle):
        raise HarmBenchRoleManifestError(f"{name} changed during verified read")
    if repeated != raw:
        raise HarmBenchRoleManifestError(f"{name} bytes changed during verified read")
    _path_still_names_handle(source, after_handle, name=name)
    try:
        text = raw.decode("utf-8", errors="strict")
        payload = json.loads(text, object_pairs_hook=_unique_object)
    except HarmBenchRoleManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise HarmBenchRoleManifestError(f"cannot decode exact {name}") from error
    if not isinstance(payload, dict):
        raise HarmBenchRoleManifestError(f"{name} root must be an object")
    if raw != _canonical_json_bytes(payload):
        raise HarmBenchRoleManifestError(f"{name} must use exact canonical JSON bytes")
    return VerifiedJson(payload=payload, sha256=hashlib.sha256(raw).hexdigest())


def exact_artifact_path(root: str | Path, filename: object) -> Path:
    """Resolve one manifest basename without globbing or directory enumeration."""

    base = Path(root)
    try:
        root_stat = base.lstat()
    except OSError as error:
        raise HarmBenchRoleManifestError("cannot stat explicit capability root") from error
    if stat.S_ISLNK(root_stat.st_mode) or _is_reparse(root_stat):
        raise HarmBenchRoleManifestError("capability root cannot be a symlink or reparse point")
    if not stat.S_ISDIR(root_stat.st_mode):
        raise HarmBenchRoleManifestError("capability root must be a directory")
    value = str(filename)
    candidate = Path(value)
    if (
        not value
        or candidate.name != value
        or candidate.is_absolute()
        or "/" in value
        or "\\" in value
        or value in {".", ".."}
    ):
        raise HarmBenchRoleManifestError("artifact filename must be one exact basename")
    return base / value


def read_verified_npz(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_fields: set[str],
    name: str,
) -> VerifiedNpz:
    """Hash, seek, deserialize, and re-hash one NPZ through one file handle."""

    source = Path(path)
    expected_sha = _sha256(expected_sha256, name=f"{name} expected SHA-256")
    before_path = _plain_file(source, name=name)
    try:
        with source.open("rb") as handle:
            before_handle = os.fstat(handle.fileno())
            if not _same_identity(before_path, before_handle):
                raise HarmBenchRoleManifestError(f"{name} changed before verified read")
            first = hashlib.sha256()
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                first.update(chunk)
            first_sha = first.hexdigest()
            if first_sha != expected_sha:
                raise HarmBenchRoleManifestError(f"{name} SHA-256 differs from manifest")
            handle.seek(0)
            with np.load(handle, allow_pickle=False) as archive:
                if set(archive.files) != expected_fields:
                    raise HarmBenchRoleManifestError(f"{name} NPZ schema changed")
                arrays = {
                    field: np.asarray(archive[field]).copy() for field in archive.files
                }
            handle.seek(0)
            second = hashlib.sha256()
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                second.update(chunk)
            after_handle = os.fstat(handle.fileno())
    except HarmBenchRoleManifestError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise HarmBenchRoleManifestError(f"cannot load exact {name}") from error
    if second.hexdigest() != first_sha or not _same_identity(before_handle, after_handle):
        raise HarmBenchRoleManifestError(f"{name} changed during verified load")
    _path_still_names_handle(source, after_handle, name=name)
    return VerifiedNpz(arrays=arrays, sha256=first_sha)


def make_feature_projection(
    *,
    dataset_id: str,
    role: str,
    artifact_schema_version: str,
    filename: str,
    sha256: str,
    row_alignment_sha256: str,
    rows: int,
    independent_groups: int,
    history_eligible_rows: int,
    audio_dimension: int,
    video_dimension: int,
) -> dict[str, object]:
    if dataset_id not in {"EmotionTalk", "MELD", "synthetic"}:
        raise HarmBenchRoleManifestError("feature projection dataset changed")
    if role not in {FIT_ROLE, SELECTION_ROLE}:
        raise HarmBenchRoleManifestError("feature projection role changed")
    exact_artifact_path(Path.cwd(), filename)
    if any(type(value) is not int for value in (
        rows,
        independent_groups,
        history_eligible_rows,
        audio_dimension,
        video_dimension,
    )):
        raise HarmBenchRoleManifestError("feature projection counts must be exact integers")
    if (
        rows < 1
        or independent_groups < 1
        or not 0 <= history_eligible_rows <= rows
        or audio_dimension < 1
        or video_dimension < 1
    ):
        raise HarmBenchRoleManifestError("feature projection counts are invalid")
    return {
        "artifact": {
            "audio_dimension": audio_dimension,
            "filename": filename,
            "history_eligible_rows": history_eligible_rows,
            "independent_groups": independent_groups,
            "rows": rows,
            "row_alignment_sha256": _sha256(
                row_alignment_sha256, name="row_alignment_sha256"
            ),
            "sha256": _sha256(sha256, name="feature artifact sha256"),
            "video_dimension": video_dimension,
        },
        "artifact_schema_version": str(artifact_schema_version),
        "dataset_id": dataset_id,
        "role": role,
        "schema_version": FEATURE_PROJECTION_SCHEMA,
        "split_protocol_id": SPLIT_PROTOCOL_ID,
    }


def validate_feature_projection(
    value: object, *, expected_dataset: str, expected_role: str
) -> Mapping[str, object]:
    projection = _exact_keys(
        value,
        {
            "schema_version",
            "artifact_schema_version",
            "dataset_id",
            "split_protocol_id",
            "role",
            "artifact",
        },
        name="feature_projection",
    )
    if (
        projection["schema_version"] != FEATURE_PROJECTION_SCHEMA
        or projection["dataset_id"] != expected_dataset
        or projection["split_protocol_id"] != SPLIT_PROTOCOL_ID
        or projection["role"] != expected_role
        or not isinstance(projection["artifact_schema_version"], str)
        or not projection["artifact_schema_version"]
    ):
        raise HarmBenchRoleManifestError("feature projection identity changed")
    artifact = _exact_keys(
        projection["artifact"],
        {
            "filename",
            "sha256",
            "row_alignment_sha256",
            "rows",
            "independent_groups",
            "history_eligible_rows",
            "audio_dimension",
            "video_dimension",
        },
        name="feature_projection.artifact",
    )
    make_feature_projection(
        dataset_id=expected_dataset,
        role=expected_role,
        artifact_schema_version=str(projection["artifact_schema_version"]),
        filename=str(artifact["filename"]),
        sha256=str(artifact["sha256"]),
        row_alignment_sha256=str(artifact["row_alignment_sha256"]),
        rows=artifact["rows"],  # type: ignore[arg-type]
        independent_groups=artifact["independent_groups"],  # type: ignore[arg-type]
        history_eligible_rows=artifact["history_eligible_rows"],  # type: ignore[arg-type]
        audio_dimension=artifact["audio_dimension"],  # type: ignore[arg-type]
        video_dimension=artifact["video_dimension"],  # type: ignore[arg-type]
    )
    return projection


def make_cross_role_feature_roster(
    *,
    dataset_id: str,
    fit_feature_projection_sha256: str,
    selection_feature_projection_sha256: str,
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "feature_roles": [
            {
                "feature_projection_sha256": _sha256(
                    fit_feature_projection_sha256,
                    name="fit_feature_projection_sha256",
                ),
                "role": FIT_ROLE,
            },
            {
                "feature_projection_sha256": _sha256(
                    selection_feature_projection_sha256,
                    name="selection_feature_projection_sha256",
                ),
                "role": SELECTION_ROLE,
            },
        ],
        "partition_contract": {
            "independent_group_disjoint": True,
            "protocol_row_disjoint": True,
            "row_key_disjoint": True,
            "whole_group_assignment": True,
        },
        "payload_scope": "strictly_feature_only",
        "protocol_id": PROTOCOL_ID,
        "schema_version": CROSS_ROLE_FEATURE_ROSTER_SCHEMA,
        "split_protocol_id": SPLIT_PROTOCOL_ID,
    }


def validate_cross_role_feature_roster(value: object) -> Mapping[str, object]:
    roster = _exact_keys(
        value,
        {
            "schema_version",
            "dataset_id",
            "protocol_id",
            "split_protocol_id",
            "feature_roles",
            "partition_contract",
            "payload_scope",
        },
        name="cross_role_feature_roster",
    )
    if (
        roster["schema_version"] != CROSS_ROLE_FEATURE_ROSTER_SCHEMA
        or roster["protocol_id"] != PROTOCOL_ID
        or roster["split_protocol_id"] != SPLIT_PROTOCOL_ID
        or roster["payload_scope"] != "strictly_feature_only"
        or roster["dataset_id"] not in {"EmotionTalk", "MELD", "synthetic"}
    ):
        raise HarmBenchRoleManifestError("cross-role feature roster identity changed")
    roles = roster["feature_roles"]
    if not isinstance(roles, list) or len(roles) != 2:
        raise HarmBenchRoleManifestError("cross-role feature roster changed")
    for record, expected_role in zip(roles, (FIT_ROLE, SELECTION_ROLE), strict=True):
        item = _exact_keys(
            record, {"role", "feature_projection_sha256"}, name="feature_roles[]"
        )
        if item["role"] != expected_role:
            raise HarmBenchRoleManifestError("cross-role feature order changed")
        _sha256(item["feature_projection_sha256"], name="feature projection sha256")
    partition = _exact_keys(
        roster["partition_contract"],
        {
            "whole_group_assignment",
            "row_key_disjoint",
            "protocol_row_disjoint",
            "independent_group_disjoint",
        },
        name="partition_contract",
    )
    if any(value is not True for value in partition.values()):
        raise HarmBenchRoleManifestError("cross-role partition contract weakened")
    return roster


def _projection_digest_from_roster_payload(
    payload: Mapping[str, object], *, role: str
) -> str:
    validated = validate_cross_role_feature_roster(payload)
    if role not in {FIT_ROLE, SELECTION_ROLE}:
        raise HarmBenchRoleManifestError("feature roster role changed")
    records = validated["feature_roles"]
    if not isinstance(records, list):  # Defensive after exact validation.
        raise HarmBenchRoleManifestError("cross-role feature roster changed")
    for record in records:
        if not isinstance(record, Mapping):
            raise HarmBenchRoleManifestError("cross-role feature roster changed")
        if record["role"] == role:
            return _sha256(
                record["feature_projection_sha256"],
                name=f"{role} feature projection sha256",
            )
    raise HarmBenchRoleManifestError("feature roster role is missing")


def _make_cross_role_feature_roster_receipt(
    payload: Mapping[str, object], *, roster_sha256: str
) -> CrossRoleFeatureRosterReceipt:
    validated = validate_cross_role_feature_roster(payload)
    return CrossRoleFeatureRosterReceipt(
        schema_version=CROSS_ROLE_FEATURE_ROSTER_RECEIPT_SCHEMA,
        roster_schema_version=str(validated["schema_version"]),
        dataset_id=str(validated["dataset_id"]),
        protocol_id=str(validated["protocol_id"]),
        split_protocol_id=str(validated["split_protocol_id"]),
        payload_scope=str(validated["payload_scope"]),
        fit_feature_projection_sha256=_projection_digest_from_roster_payload(
            validated, role=FIT_ROLE
        ),
        selection_feature_projection_sha256=(
            _projection_digest_from_roster_payload(validated, role=SELECTION_ROLE)
        ),
        roster_sha256=_sha256(roster_sha256, name="roster_sha256"),
    )


def validate_cross_role_feature_roster_receipt(
    value: object,
    *,
    expected_roster_sha256: str,
    expected_dataset: str,
) -> CrossRoleFeatureRosterReceipt:
    """Rebuild a typed roster receipt and compare it with external authority."""

    if not isinstance(value, CrossRoleFeatureRosterReceipt):
        raise HarmBenchRoleManifestError(
            "cross-role feature roster receipt type changed"
        )
    if (
        value.schema_version != CROSS_ROLE_FEATURE_ROSTER_RECEIPT_SCHEMA
        or value.roster_schema_version != CROSS_ROLE_FEATURE_ROSTER_SCHEMA
        or value.protocol_id != PROTOCOL_ID
        or value.split_protocol_id != SPLIT_PROTOCOL_ID
        or value.payload_scope != "strictly_feature_only"
        or value.dataset_id != expected_dataset
        or value.dataset_id not in {"EmotionTalk", "MELD", "synthetic"}
    ):
        raise HarmBenchRoleManifestError(
            "cross-role feature roster receipt identity changed"
        )
    fit_projection_sha = _sha256(
        value.fit_feature_projection_sha256,
        name="fit_feature_projection_sha256",
    )
    selection_projection_sha = _sha256(
        value.selection_feature_projection_sha256,
        name="selection_feature_projection_sha256",
    )
    canonical_payload = make_cross_role_feature_roster(
        dataset_id=value.dataset_id,
        fit_feature_projection_sha256=fit_projection_sha,
        selection_feature_projection_sha256=selection_projection_sha,
    )
    canonical_roster_sha = canonical_json_sha256(canonical_payload)
    observed_roster_sha = _sha256(value.roster_sha256, name="roster_sha256")
    if observed_roster_sha != canonical_roster_sha:
        raise HarmBenchRoleManifestError(
            "cross-role feature roster receipt differs from canonical payload"
        )
    if observed_roster_sha != _sha256(
        expected_roster_sha256, name="expected_roster_sha256"
    ):
        raise HarmBenchRoleManifestError(
            "cross-role feature roster differs from external authority"
        )
    return value


def load_cross_role_feature_roster(
    path: str | Path, *, expected_roster_sha256: str
) -> CrossRoleFeatureRosterReceipt:
    """Load one canonical feature-only roster under an external SHA authority."""

    expected_sha = _sha256(
        expected_roster_sha256, name="expected_roster_sha256"
    )
    verified = read_canonical_json(path, name="cross-role feature roster")
    if verified.sha256 != expected_sha:
        raise HarmBenchRoleManifestError(
            "cross-role feature roster differs from external authority"
        )
    receipt = _make_cross_role_feature_roster_receipt(
        verified.payload, roster_sha256=verified.sha256
    )
    return validate_cross_role_feature_roster_receipt(
        receipt,
        expected_roster_sha256=expected_sha,
        expected_dataset=receipt.dataset_id,
    )


def load_cross_role_feature_roster_legacy_unbound(path: str | Path) -> VerifiedJson:
    """Explicit migration-only loader without an external SHA authority."""

    verified = read_canonical_json(path, name="legacy unbound cross-role roster")
    validate_cross_role_feature_roster(verified.payload)
    return verified


def make_synthetic_cross_role_feature_roster_receipt(
    *,
    dataset_id: str,
    fit_feature_projection_sha256: str,
    selection_feature_projection_sha256: str,
) -> CrossRoleFeatureRosterReceipt:
    """Create an in-memory synthetic receipt; never a production file proof."""

    if dataset_id != "synthetic":
        raise HarmBenchRoleManifestError(
            "synthetic roster receipt requires the synthetic dataset"
        )
    payload = make_cross_role_feature_roster(
        dataset_id=dataset_id,
        fit_feature_projection_sha256=fit_feature_projection_sha256,
        selection_feature_projection_sha256=selection_feature_projection_sha256,
    )
    return _make_cross_role_feature_roster_receipt(
        payload, roster_sha256=canonical_json_sha256(payload)
    )


def make_legacy_cross_role_feature_roster_receipt(
    *,
    dataset_id: str,
    fit_feature_projection_sha256: str,
    selection_feature_projection_sha256: str,
) -> CrossRoleFeatureRosterReceipt:
    """Create an unbound migration receipt for an explicit legacy loader."""

    if dataset_id not in {"EmotionTalk", "MELD"}:
        raise HarmBenchRoleManifestError(
            "legacy roster receipt requires a production dataset"
        )
    payload = make_cross_role_feature_roster(
        dataset_id=dataset_id,
        fit_feature_projection_sha256=fit_feature_projection_sha256,
        selection_feature_projection_sha256=selection_feature_projection_sha256,
    )
    return _make_cross_role_feature_roster_receipt(
        payload, roster_sha256=canonical_json_sha256(payload)
    )


def roster_feature_projection_sha256(
    roster: CrossRoleFeatureRosterReceipt,
    *,
    role: str,
    expected_roster_sha256: str,
) -> str:
    validated = validate_cross_role_feature_roster_receipt(
        roster,
        expected_roster_sha256=expected_roster_sha256,
        expected_dataset=roster.dataset_id,
    )
    if role == FIT_ROLE:
        return validated.fit_feature_projection_sha256
    if role == SELECTION_ROLE:
        return validated.selection_feature_projection_sha256
    raise HarmBenchRoleManifestError("feature roster role changed")


def make_feature_manifest(
    *,
    feature_projection: Mapping[str, object],
    cross_role_feature_roster_receipt: CrossRoleFeatureRosterReceipt,
    expected_cross_role_feature_roster_sha256: str,
) -> dict[str, object]:
    dataset_id = str(feature_projection.get("dataset_id"))
    role = str(feature_projection.get("role"))
    validate_feature_projection(
        feature_projection, expected_dataset=dataset_id, expected_role=role
    )
    if role == FIT_ROLE:
        schema = FIT_FEATURE_MANIFEST_SCHEMA
        kind = "fit_features"
    elif role == SELECTION_ROLE:
        schema = SELECTION_FEATURE_MANIFEST_SCHEMA
        kind = "selection_features"
    else:
        raise HarmBenchRoleManifestError("feature manifest role changed")
    roster_receipt = validate_cross_role_feature_roster_receipt(
        cross_role_feature_roster_receipt,
        expected_roster_sha256=expected_cross_role_feature_roster_sha256,
        expected_dataset=dataset_id,
    )
    expected_projection_sha = roster_feature_projection_sha256(
        roster_receipt,
        role=role,
        expected_roster_sha256=expected_cross_role_feature_roster_sha256,
    )
    observed_projection_sha = canonical_json_sha256(feature_projection)
    if observed_projection_sha != expected_projection_sha:
        raise HarmBenchRoleManifestError(
            "feature projection differs from typed cross-role roster receipt"
        )
    return {
        "cross_role_feature_roster_sha256": roster_receipt.roster_sha256,
        "dataset_id": dataset_id,
        "feature_projection": dict(feature_projection),
        "feature_projection_sha256": observed_projection_sha,
        "manifest_kind": kind,
        "payload_scope": "strictly_feature_only",
        "protocol_id": PROTOCOL_ID,
        "role": role,
        "runtime_contract": {
            "allow_pickle": False,
            "directory_enumeration_permitted": False,
            "exact_basename_resolution_only": True,
            "legacy_aggregate_manifest_required_at_runtime": False,
        },
        "schema_version": schema,
        "split_protocol_id": SPLIT_PROTOCOL_ID,
    }


def _reject_selection_target_vocabulary(value: object, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in _SELECTION_FORBIDDEN_TOKENS):
                raise HarmBenchRoleManifestError(
                    f"selection feature manifest contains forbidden vocabulary at {path}"
                )
            _reject_selection_target_vocabulary(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_selection_target_vocabulary(child, path=f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in _SELECTION_FORBIDDEN_TOKENS):
            raise HarmBenchRoleManifestError(
                f"selection feature manifest contains forbidden vocabulary at {path}"
            )


def validate_feature_manifest(
    value: object, *, expected_dataset: str, expected_role: str
) -> Mapping[str, object]:
    manifest = _exact_keys(
        value,
        {
            "schema_version",
            "manifest_kind",
            "dataset_id",
            "protocol_id",
            "split_protocol_id",
            "role",
            "feature_projection",
            "feature_projection_sha256",
            "cross_role_feature_roster_sha256",
            "runtime_contract",
            "payload_scope",
        },
        name="feature_manifest",
    )
    expected_schema = (
        FIT_FEATURE_MANIFEST_SCHEMA
        if expected_role == FIT_ROLE
        else SELECTION_FEATURE_MANIFEST_SCHEMA
    )
    expected_kind = "fit_features" if expected_role == FIT_ROLE else "selection_features"
    if (
        manifest["schema_version"] != expected_schema
        or manifest["manifest_kind"] != expected_kind
        or manifest["dataset_id"] != expected_dataset
        or manifest["protocol_id"] != PROTOCOL_ID
        or manifest["split_protocol_id"] != SPLIT_PROTOCOL_ID
        or manifest["role"] != expected_role
        or manifest["payload_scope"] != "strictly_feature_only"
    ):
        raise HarmBenchRoleManifestError("feature manifest identity changed")
    runtime = _exact_keys(
        manifest["runtime_contract"],
        {
            "allow_pickle",
            "directory_enumeration_permitted",
            "exact_basename_resolution_only",
            "legacy_aggregate_manifest_required_at_runtime",
        },
        name="runtime_contract",
    )
    if runtime != {
        "allow_pickle": False,
        "directory_enumeration_permitted": False,
        "exact_basename_resolution_only": True,
        "legacy_aggregate_manifest_required_at_runtime": False,
    }:
        raise HarmBenchRoleManifestError("feature runtime contract weakened")
    projection = validate_feature_projection(
        manifest["feature_projection"],
        expected_dataset=expected_dataset,
        expected_role=expected_role,
    )
    if canonical_json_sha256(projection) != _sha256(
        manifest["feature_projection_sha256"], name="feature_projection_sha256"
    ):
        raise HarmBenchRoleManifestError("feature projection digest changed")
    _sha256(
        manifest["cross_role_feature_roster_sha256"],
        name="cross_role_feature_roster_sha256",
    )
    if expected_role == SELECTION_ROLE:
        _reject_selection_target_vocabulary(manifest)
    return manifest


def load_feature_manifest(
    path: str | Path, *, expected_dataset: str, expected_role: str
) -> VerifiedJson:
    name = f"{expected_dataset} {expected_role} feature manifest"
    verified = read_canonical_json(path, name=name)
    validate_feature_manifest(
        verified.payload,
        expected_dataset=expected_dataset,
        expected_role=expected_role,
    )
    return verified


def make_fit_training_manifest(
    *,
    dataset_id: str,
    fit_feature_manifest_sha256: str,
    cross_role_feature_roster_sha256: str,
    artifact_schema_version: str,
    filename: str,
    sha256: str,
    row_alignment_sha256: str,
    rows: int,
    class_names: Sequence[str],
) -> dict[str, object]:
    exact_artifact_path(Path.cwd(), filename)
    classes = [str(value) for value in class_names]
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise HarmBenchRoleManifestError("fit classes must be unique")
    if type(rows) is not int or rows < 1:
        raise HarmBenchRoleManifestError("fit target rows are invalid")
    projection = {
        "artifact": {
            "filename": filename,
            "row_alignment_sha256": _sha256(
                row_alignment_sha256, name="fit target row_alignment_sha256"
            ),
            "rows": rows,
            "sha256": _sha256(sha256, name="fit target artifact sha256"),
        },
        "artifact_schema_version": str(artifact_schema_version),
        "class_names": classes,
        "dataset_id": dataset_id,
        "role": FIT_ROLE,
        "schema_version": FIT_TARGET_PROJECTION_SCHEMA,
        "split_protocol_id": SPLIT_PROTOCOL_ID,
    }
    return {
        "cross_role_feature_roster_sha256": _sha256(
            cross_role_feature_roster_sha256,
            name="cross_role_feature_roster_sha256",
        ),
        "dataset_id": dataset_id,
        "fit_feature_manifest_sha256": _sha256(
            fit_feature_manifest_sha256, name="fit_feature_manifest_sha256"
        ),
        "fit_target_projection": projection,
        "fit_target_projection_sha256": canonical_json_sha256(projection),
        "manifest_kind": "fit_training",
        "protocol_id": PROTOCOL_ID,
        "role": FIT_ROLE,
        "runtime_contract": {
            "allow_pickle": False,
            "directory_enumeration_permitted": False,
            "exact_basename_resolution_only": True,
            "selection_role_access_permitted": False,
        },
        "schema_version": FIT_TRAINING_MANIFEST_SCHEMA,
        "split_protocol_id": SPLIT_PROTOCOL_ID,
    }


def validate_fit_training_manifest(
    value: object, *, expected_dataset: str
) -> Mapping[str, object]:
    manifest = _exact_keys(
        value,
        {
            "schema_version",
            "manifest_kind",
            "dataset_id",
            "protocol_id",
            "split_protocol_id",
            "role",
            "fit_feature_manifest_sha256",
            "cross_role_feature_roster_sha256",
            "fit_target_projection",
            "fit_target_projection_sha256",
            "runtime_contract",
        },
        name="fit_training_manifest",
    )
    if (
        manifest["schema_version"] != FIT_TRAINING_MANIFEST_SCHEMA
        or manifest["manifest_kind"] != "fit_training"
        or manifest["dataset_id"] != expected_dataset
        or manifest["protocol_id"] != PROTOCOL_ID
        or manifest["split_protocol_id"] != SPLIT_PROTOCOL_ID
        or manifest["role"] != FIT_ROLE
    ):
        raise HarmBenchRoleManifestError("fit training manifest identity changed")
    _sha256(manifest["fit_feature_manifest_sha256"], name="fit_feature_manifest_sha256")
    _sha256(
        manifest["cross_role_feature_roster_sha256"],
        name="cross_role_feature_roster_sha256",
    )
    runtime = _exact_keys(
        manifest["runtime_contract"],
        {
            "allow_pickle",
            "directory_enumeration_permitted",
            "exact_basename_resolution_only",
            "selection_role_access_permitted",
        },
        name="fit training runtime_contract",
    )
    if runtime != {
        "allow_pickle": False,
        "directory_enumeration_permitted": False,
        "exact_basename_resolution_only": True,
        "selection_role_access_permitted": False,
    }:
        raise HarmBenchRoleManifestError("fit training runtime contract weakened")
    projection = _exact_keys(
        manifest["fit_target_projection"],
        {
            "schema_version",
            "artifact_schema_version",
            "dataset_id",
            "split_protocol_id",
            "role",
            "artifact",
            "class_names",
        },
        name="fit_target_projection",
    )
    if (
        projection["schema_version"] != FIT_TARGET_PROJECTION_SCHEMA
        or projection["dataset_id"] != expected_dataset
        or projection["split_protocol_id"] != SPLIT_PROTOCOL_ID
        or projection["role"] != FIT_ROLE
        or not isinstance(projection["artifact_schema_version"], str)
    ):
        raise HarmBenchRoleManifestError("fit target projection identity changed")
    artifact = _exact_keys(
        projection["artifact"],
        {"filename", "sha256", "row_alignment_sha256", "rows"},
        name="fit_target_projection.artifact",
    )
    exact_artifact_path(Path.cwd(), artifact["filename"])
    _sha256(artifact["sha256"], name="fit target artifact sha256")
    _sha256(artifact["row_alignment_sha256"], name="fit target row alignment")
    if type(artifact["rows"]) is not int or int(artifact["rows"]) < 1:
        raise HarmBenchRoleManifestError("fit target rows are invalid")
    classes = projection["class_names"]
    if (
        not isinstance(classes, list)
        or len(classes) < 2
        or not all(isinstance(value, str) for value in classes)
        or len(set(classes)) != len(classes)
    ):
        raise HarmBenchRoleManifestError("fit class roster changed")
    if canonical_json_sha256(projection) != _sha256(
        manifest["fit_target_projection_sha256"],
        name="fit_target_projection_sha256",
    ):
        raise HarmBenchRoleManifestError("fit target projection digest changed")
    return manifest


def load_fit_training_manifest(
    path: str | Path, *, expected_dataset: str
) -> VerifiedJson:
    verified = read_canonical_json(path, name=f"{expected_dataset} fit training manifest")
    validate_fit_training_manifest(verified.payload, expected_dataset=expected_dataset)
    return verified


def write_canonical_json_once(path: str | Path, payload: object) -> str:
    """Atomically publish canonical JSON without overwriting an existing artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_json_bytes(payload)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise HarmBenchRoleManifestError("write-once destination already exists") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    verified = read_canonical_json(destination)
    if verified.sha256 != hashlib.sha256(raw).hexdigest():
        raise HarmBenchRoleManifestError("published canonical JSON changed")
    return verified.sha256


def write_emotiontalk_sanitized_fit_label_sidecar(
    *,
    source_path: str | Path,
    destination_path: str | Path,
    expected_source_sha256: str,
    expected_row_alignment_sha256: str,
) -> str:
    """Trusted one-way migration removing the all-role source label digest.

    This is the only helper in this module allowed to understand the legacy
    ``source_label_sha256`` field.  The field is validated as a scalar but is
    never copied, returned, logged, or included in the destination digest
    descriptor.
    """

    source = read_verified_npz(
        source_path,
        expected_sha256=expected_source_sha256,
        expected_fields=set(LEGACY_EMOTIONTALK_LABEL_FIELDS),
        name="legacy EmotionTalk fit label sidecar",
    )
    payload = source.arrays
    def scalar(name: str) -> str:
        value = np.asarray(payload[name])
        if value.size != 1:
            raise HarmBenchRoleManifestError(f"legacy fit {name} is not scalar")
        return str(value.reshape(-1)[0])

    alignment = _sha256(
        expected_row_alignment_sha256, name="expected_row_alignment_sha256"
    )
    if (
        scalar("schema_version") != LEGACY_EMOTIONTALK_LABEL_SCHEMA
        or scalar("dataset_id") != "EmotionTalk"
        or scalar("role") != FIT_ROLE
        or scalar("split_protocol_id") != SPLIT_PROTOCOL_ID
        or scalar("row_alignment_sha256") != alignment
    ):
        raise HarmBenchRoleManifestError("legacy EmotionTalk fit label identity changed")
    # Require the legacy all-role digest to be syntactically valid, then discard it.
    _sha256(scalar("source_label_sha256"), name="legacy source digest")
    labels = np.asarray(payload["labels"])
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or labels.dtype.kind == "b":
        raise HarmBenchRoleManifestError("legacy EmotionTalk fit labels changed")

    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    published_sha: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                schema_version=np.asarray(SANITIZED_EMOTIONTALK_FIT_LABEL_SCHEMA),
                dataset_id=np.asarray("EmotionTalk"),
                role=np.asarray(FIT_ROLE),
                split_protocol_id=np.asarray(SPLIT_PROTOCOL_ID),
                row_alignment_sha256=np.asarray(alignment),
                labels=np.asarray(labels, dtype=np.int64),
            )
            handle.flush()
            os.fsync(handle.fileno())
            handle.seek(0)
            digest = hashlib.sha256()
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            published_sha = digest.hexdigest()
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise HarmBenchRoleManifestError("write-once destination already exists") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    if published_sha is None:
        raise HarmBenchRoleManifestError("sanitized fit sidecar was not published")
    verified = read_verified_npz(
        destination,
        expected_sha256=published_sha,
        expected_fields=set(SANITIZED_EMOTIONTALK_FIT_LABEL_FIELDS),
        name="sanitized EmotionTalk fit label sidecar",
    )
    return verified.sha256


__all__ = [
    "CROSS_ROLE_FEATURE_ROSTER_RECEIPT_SCHEMA",
    "CROSS_ROLE_FEATURE_ROSTER_SCHEMA",
    "CrossRoleFeatureRosterReceipt",
    "FEATURE_PROJECTION_SCHEMA",
    "FIT_FEATURE_MANIFEST_SCHEMA",
    "FIT_TARGET_PROJECTION_SCHEMA",
    "FIT_TRAINING_MANIFEST_SCHEMA",
    "HarmBenchRoleManifestError",
    "SANITIZED_EMOTIONTALK_FIT_LABEL_FIELDS",
    "SANITIZED_EMOTIONTALK_FIT_LABEL_SCHEMA",
    "SELECTION_FEATURE_MANIFEST_SCHEMA",
    "VerifiedJson",
    "VerifiedNpz",
    "canonical_json_sha256",
    "exact_artifact_path",
    "load_cross_role_feature_roster",
    "load_cross_role_feature_roster_legacy_unbound",
    "load_feature_manifest",
    "load_fit_training_manifest",
    "make_cross_role_feature_roster",
    "make_feature_manifest",
    "make_feature_projection",
    "make_fit_training_manifest",
    "make_legacy_cross_role_feature_roster_receipt",
    "make_synthetic_cross_role_feature_roster_receipt",
    "read_canonical_json",
    "read_verified_npz",
    "roster_feature_projection_sha256",
    "validate_cross_role_feature_roster",
    "validate_cross_role_feature_roster_receipt",
    "validate_feature_manifest",
    "validate_feature_projection",
    "validate_fit_training_manifest",
    "write_canonical_json_once",
    "write_emotiontalk_sanitized_fit_label_sidecar",
]
