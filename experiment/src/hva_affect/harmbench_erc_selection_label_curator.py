"""One-shot curator for frozen legacy HarmBench-ERC selection labels.

This module is an intentionally isolated data-custodian boundary.  It verifies
one frozen public sidecar manifest, opens only that manifest's
``model_selection`` feature and label archives, and publishes the universal
label-only artifact consumed by the selection-label capability.

The curator has no import path to model, prediction, metric, evaluator, role,
checkpoint, or label-capability code.  Its production constants pin the exact
legacy files.  The caller supplies only the fit-training capability digest;
the checkpoint class-order digest is then mechanically recomputed from the
frozen dataset identity and ordered class tokens.

An fsynced, write-once *curator-ingest* marker is published before the legacy
label path is resolved, stated, hashed, or opened.  This is an independent
data-preparation authority marker, never an evaluator attempt and never an
authorization to compute metrics.  Once claimed, every failure is a terminal
fail-closed state.  In particular, an artifact whose manifest could not be
published is deliberately not removed: the marker prevents a silent rerun
and the absent manifest prevents activation.  The already-observed train
selection roles remain permanently exploratory.  A future untouched test
ingest must instead be run by an independent custodian and must never be
pre-opened by a model-training or evaluation process.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


SELECTION_ROLE = "model_selection"
UNIVERSAL_ARTIFACT_SCHEMA = "harmbench_erc_selection_labels_private_v1"
UNIVERSAL_MANIFEST_SCHEMA = "harmbench_erc_selection_label_manifest_v1"
UNIVERSAL_ARTIFACT_FILENAME = "harmbench_erc_selection_labels.npz"
UNIVERSAL_MANIFEST_FILENAME = "harmbench_erc_selection_labels.manifest.json"
CURATOR_ATTEMPT_SCHEMA = "harmbench_erc_selection_label_curator_attempt_v1"
CURATOR_ATTEMPT_FILENAME = "harmbench_erc_selection_label_curator.attempt.json"
CHECKPOINT_CLASS_ORDER_SCHEMA = "harmbench_erc_checkpoint_class_order_v1"

MAX_EXTERNAL_MANIFEST_BYTES = 256 * 1024
MAX_LEGACY_FEATURE_COMPRESSED_BYTES = 256 * 1024 * 1024
MAX_LEGACY_LABEL_COMPRESSED_BYTES = 16 * 1024 * 1024
MAX_LEGACY_MEMBER_BYTES = 256 * 1024 * 1024
MAX_LEGACY_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_LEGACY_ELEMENTS = 100_000_000
MAX_UNIVERSAL_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_UNIVERSAL_MANIFEST_BYTES = 16 * 1024

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_UNIVERSAL_ARRAY_ORDER = (
    "schema_version",
    "dataset_id",
    "role",
    "rows",
    "ordered_protocol_row_alignment_sha256",
    "class_order_sha256",
    "labels",
    "protocol_row_ids",
    "class_tokens",
)


class HarmBenchSelectionLabelCuratorError(ValueError):
    """Raised when frozen ingest provenance or one-shot I/O changes."""


@dataclass(frozen=True)
class _FrozenDatasetContract:
    dataset_id: str
    external_manifest_filename: str
    external_manifest_sha256: str
    external_manifest_schema: str
    external_manifest_protocol: str
    external_manifest_status: str
    legacy_feature_schema: str
    legacy_label_schema: str
    selection_feature_filename: str
    selection_label_filename: str
    selection_feature_sha256: str
    selection_label_sha256: str
    rows: int
    legacy_row_alignment_sha256: str
    ordered_class_tokens: tuple[str, ...]
    feature_member_order: tuple[str, ...]
    label_member_order: tuple[str, ...]
    scalar_shape: tuple[int, ...]
    feature_has_dataset_id: bool
    feature_has_split_protocol_id: bool


_FROZEN_DATASETS: dict[str, _FrozenDatasetContract] = {
    "EmotionTalk": _FrozenDatasetContract(
        dataset_id="EmotionTalk",
        external_manifest_filename="emotiontalk_open_role_sidecar_v2_manifest.json",
        external_manifest_sha256=(
            "bbd843876fa051c5426d0d56870adc939cdf71e1e8eaf552880ab4f89d47f530"
        ),
        external_manifest_schema="emotiontalk_role_sidecar_manifest_v2",
        external_manifest_protocol="emotiontalk_role_separated_sidecars_v2",
        external_manifest_status=(
            "strict_open_role_feature_and_label_sidecars_created_and_hashed"
        ),
        legacy_feature_schema="emotiontalk_role_feature_sidecar_v2",
        legacy_label_schema="emotiontalk_role_label_sidecar_v2",
        selection_feature_filename="features_model_selection.npz",
        selection_label_filename="labels_model_selection.npz",
        selection_feature_sha256=(
            "91e3756374bde05e3f3094a94a7a82de394fc1df1049d8f4df2fa198a3037373"
        ),
        selection_label_sha256=(
            "20854541d4bba2f4854c43a93cc3ef4c74f3695fd57b49842c94b05bda6f802d"
        ),
        rows=2682,
        legacy_row_alignment_sha256=(
            "197545f4d9882c0e280ba86dabccebf32f645174f9df0caf3a0a1bccf4b68224"
        ),
        ordered_class_tokens=(
            "neutral",
            "happy",
            "sad",
            "angry",
            "surprised",
            "disgusted",
            "fearful",
        ),
        feature_member_order=(
            "schema_version",
            "dataset_id",
            "role",
            "split_protocol_id",
            "row_alignment_sha256",
            "opaque_row_hashes",
            "opaque_group_hashes",
            "speaker_tokens",
            "turn_ids",
            "protocol_row_ids",
            "role_buckets",
            "texts",
            "audio_features",
            "video_features",
            "source_feature_config_sha256",
        ),
        label_member_order=(
            "schema_version",
            "dataset_id",
            "role",
            "split_protocol_id",
            "row_alignment_sha256",
            "labels",
            "source_label_sha256",
        ),
        scalar_shape=(),
        feature_has_dataset_id=True,
        feature_has_split_protocol_id=True,
    ),
    "MELD": _FrozenDatasetContract(
        dataset_id="MELD",
        external_manifest_filename="meld_multimodal_role_sidecars_v2_manifest.json",
        external_manifest_sha256=(
            "7b12632066d20dc252c0d0d58ecc72e2d1ceefe015972ac4d73c1d0570826f99"
        ),
        external_manifest_schema="meld_multimodal_role_sidecar_manifest_v2",
        external_manifest_protocol="meld_multimodal_role_sidecars_v2",
        external_manifest_status="role_separated_train_sidecars_created_and_hashed",
        legacy_feature_schema="meld_multimodal_role_sidecar_v2",
        legacy_label_schema="meld_multimodal_role_sidecar_v2",
        selection_feature_filename="features_model_selection.npz",
        selection_label_filename="labels_model_selection.npz",
        selection_feature_sha256=(
            "e42ecd75f2d14d5412e596c26bd81ab8f907f149a20334fe78bd7546b99d0beb"
        ),
        selection_label_sha256=(
            "50eb1029d703a8da6870f82455c3549fa8d9fab616d87f0a428611e5d93b4c70"
        ),
        rows=1419,
        legacy_row_alignment_sha256=(
            "e943bfa793633afb2e789df79213fe5864fc1f499a68a363218ccce426fcecbd"
        ),
        ordered_class_tokens=(
            "neutral",
            "surprise",
            "fear",
            "sadness",
            "joy",
            "disgust",
            "anger",
        ),
        feature_member_order=(
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
        ),
        label_member_order=(
            "schema_version",
            "role",
            "row_alignment_sha256",
            "labels",
        ),
        scalar_shape=(1,),
        feature_has_dataset_id=False,
        feature_has_split_protocol_id=False,
    ),
}


@dataclass(frozen=True)
class CuratedSelectionLabelReceipt:
    """Label-free data-preparation receipt; never an evaluation authorization.

    The universal manifest named by this receipt is activation-compatible but
    carries no claim that evaluation has started.  Current selection evidence
    is already observed and permanently exploratory.
    """

    dataset_id: str
    role: str
    rows: int
    legacy_external_manifest_file_sha256: str
    legacy_selection_feature_file_sha256: str
    legacy_selection_label_file_sha256: str
    expected_fit_training_capability_sha256: str
    ordered_protocol_row_alignment_sha256: str
    class_order_sha256: str
    artifact_file_sha256: str
    manifest_file_sha256: str
    attempt_marker_file_sha256: str


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise HarmBenchSelectionLabelCuratorError(
            f"{name} must be an exact lowercase SHA-256"
        )
    return value


def _contract(dataset_id: object) -> _FrozenDatasetContract:
    if not isinstance(dataset_id, str) or dataset_id not in _FROZEN_DATASETS:
        raise HarmBenchSelectionLabelCuratorError(
            "dataset_id must be one of the two frozen selection datasets"
        )
    return _FROZEN_DATASETS[dataset_id]


def _canonical_json_payload(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise HarmBenchSelectionLabelCuratorError(
            f"curator metadata is not canonical JSON data: {error}"
        ) from error


def _canonical_json_bytes(value: object) -> bytes:
    return _canonical_json_payload(value) + b"\n"


def _array_sha256(values: np.ndarray) -> str:
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
            canonical = canonical.byteswap().view(
                canonical.dtype.newbyteorder("<")
            )
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def frozen_selection_class_order_sha256(
    *, dataset_id: str, expected_fit_training_capability_sha256: str
) -> str:
    """Derive the checkpoint class-order binding without caller-supplied tokens."""

    contract = _contract(dataset_id)
    fit_sha = _sha256(
        expected_fit_training_capability_sha256,
        name="expected_fit_training_capability_sha256",
    )
    return hashlib.sha256(
        _canonical_json_payload(
            {
                "schema_version": CHECKPOINT_CLASS_ORDER_SCHEMA,
                "dataset_id": contract.dataset_id,
                "fit_training_capability_sha256": fit_sha,
                "ordered_class_tokens": list(contract.ordered_class_tokens),
            }
        )
    ).hexdigest()


def _is_reparse(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    return bool(attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)))


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata)


def _reject_reparse_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    for candidate in (absolute, *absolute.parents):
        if os.path.lexists(candidate) and _is_reparse_or_symlink(candidate):
            raise HarmBenchSelectionLabelCuratorError(
                "curator path contains a symlink or reparse point"
            )


def _plain_file_stat(path: Path, *, name: str) -> os.stat_result:
    _reject_reparse_components(path)
    try:
        observed = path.lstat()
    except OSError as error:
        raise HarmBenchSelectionLabelCuratorError(f"cannot stat exact {name}") from error
    if stat.S_ISLNK(observed.st_mode) or _is_reparse(observed):
        raise HarmBenchSelectionLabelCuratorError(
            f"{name} cannot be a symlink or reparse point"
        )
    if not stat.S_ISREG(observed.st_mode):
        raise HarmBenchSelectionLabelCuratorError(f"{name} must be a plain file")
    return observed


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(getattr(metadata, "st_mtime_ns", 0)),
        int(getattr(metadata, "st_ctime_ns", 0)),
    )


def _same_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return _identity(first) == _identity(second)


def _assert_path_still_names_handle(
    path: Path, handle_stat: os.stat_result, *, name: str
) -> None:
    if not _same_identity(_plain_file_stat(path, name=name), handle_stat):
        raise HarmBenchSelectionLabelCuratorError(
            f"{name} path changed identity during verified read"
        )


def _hash_handle(handle: Any) -> str:
    digest = hashlib.sha256()
    while True:
        block = handle.read(1024 * 1024)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def _directory(path: str | Path, *, name: str) -> Path:
    raw = Path(path)
    if not raw.is_absolute() or raw == Path(raw.anchor):
        raise HarmBenchSelectionLabelCuratorError(
            f"{name} must be an explicit absolute non-root directory"
        )
    _reject_reparse_components(raw)
    try:
        metadata = raw.lstat()
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
            raise HarmBenchSelectionLabelCuratorError(
                f"{name} cannot be a symlink or reparse point"
            )
        resolved = raw.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise HarmBenchSelectionLabelCuratorError(
            f"{name} must already exist"
        ) from error
    if not resolved.is_dir():
        raise HarmBenchSelectionLabelCuratorError(f"{name} must be a directory")
    return resolved


def _plain_directory_stat(path: Path, *, name: str) -> os.stat_result:
    _reject_reparse_components(path)
    try:
        observed = path.lstat()
    except OSError as error:
        raise HarmBenchSelectionLabelCuratorError(
            f"cannot stat exact {name}"
        ) from error
    if stat.S_ISLNK(observed.st_mode) or _is_reparse(observed):
        raise HarmBenchSelectionLabelCuratorError(
            f"{name} cannot be a symlink or reparse point"
        )
    if not stat.S_ISDIR(observed.st_mode):
        raise HarmBenchSelectionLabelCuratorError(f"{name} must be a directory")
    return observed


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    # Directory size and timestamps legitimately change while fixed children
    # are published.  Device/inode/type/reparse attributes identify replacement
    # without mistaking this curator's own writes for an attack.
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(stat.S_IFMT(metadata.st_mode)),
        int(getattr(metadata, "st_file_attributes", 0)),
    )


def _assert_root_identity(
    root: Path, expected: os.stat_result, *, name: str
) -> None:
    if _directory_identity(
        _plain_directory_stat(root, name=name)
    ) != _directory_identity(expected):
        raise HarmBenchSelectionLabelCuratorError(
            f"{name} changed identity during curator operation"
        )


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _home_root() -> Path:
    return Path.home().resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _output_root(path: str | Path) -> Path:
    root = _directory(path, name="private output root")
    if _is_within(root, _repository_root().resolve()) or _is_within(
        root, _home_root().resolve()
    ):
        raise HarmBenchSelectionLabelCuratorError(
            "private output root must be outside repository and user home"
        )
    try:
        existing = tuple(root.iterdir())
    except OSError as error:
        raise HarmBenchSelectionLabelCuratorError(
            "private output root cannot be enumerated"
        ) from error
    if existing:
        raise HarmBenchSelectionLabelCuratorError(
            "private output root is not pristine; prior attempt is terminal"
        )
    return root


def _external_manifest_path(
    path: str | Path, *, contract: _FrozenDatasetContract
) -> tuple[Path, os.stat_result]:
    raw = Path(path)
    if not raw.is_absolute() or raw.name != contract.external_manifest_filename:
        raise HarmBenchSelectionLabelCuratorError(
            "external manifest must use its frozen absolute filename"
        )
    metadata = _plain_file_stat(raw, name="legacy external manifest")
    return raw, metadata


def _selection_child(
    root: Path, filename: str, *, name: str
) -> tuple[Path, os.stat_result]:
    path = root / filename
    if path.parent != root or path.name != filename:
        raise HarmBenchSelectionLabelCuratorError(
            f"{name} is not the frozen direct-child path"
        )
    return path, _plain_file_stat(path, name=name)


def _duplicate_key_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HarmBenchSelectionLabelCuratorError(
                f"external manifest contains duplicate JSON key: {key}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise HarmBenchSelectionLabelCuratorError(
        f"external manifest contains invalid JSON constant: {value}"
    )


def _read_external_manifest(
    path: Path,
    *,
    expected_identity: os.stat_result,
    contract: _FrozenDatasetContract,
) -> dict[str, object]:
    if expected_identity.st_size > MAX_EXTERNAL_MANIFEST_BYTES:
        raise HarmBenchSelectionLabelCuratorError(
            "external manifest exceeds byte budget"
        )
    try:
        with path.open("rb") as handle:
            before_handle = os.fstat(handle.fileno())
            if not _same_identity(expected_identity, before_handle):
                raise HarmBenchSelectionLabelCuratorError(
                    "external manifest changed before verified read"
                )
            first_sha = _hash_handle(handle)
            if first_sha != contract.external_manifest_sha256:
                raise HarmBenchSelectionLabelCuratorError(
                    "external manifest SHA-256 differs from frozen contract"
                )
            handle.seek(0)
            raw = handle.read(MAX_EXTERNAL_MANIFEST_BYTES + 1)
            if len(raw) > MAX_EXTERNAL_MANIFEST_BYTES:
                raise HarmBenchSelectionLabelCuratorError(
                    "external manifest exceeds byte budget"
                )
            try:
                payload = json.loads(
                    raw.decode("utf-8"),
                    object_pairs_hook=_duplicate_key_object,
                    parse_constant=_reject_json_constant,
                )
            except HarmBenchSelectionLabelCuratorError:
                raise
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise HarmBenchSelectionLabelCuratorError(
                    "external manifest is not strict UTF-8 JSON"
                ) from error
            handle.seek(0)
            second_sha = _hash_handle(handle)
            after_handle = os.fstat(handle.fileno())
    except HarmBenchSelectionLabelCuratorError:
        raise
    except OSError as error:
        raise HarmBenchSelectionLabelCuratorError(
            "external manifest cannot be read"
        ) from error
    if first_sha != second_sha or not _same_identity(before_handle, after_handle):
        raise HarmBenchSelectionLabelCuratorError(
            "external manifest changed during verified read"
        )
    _assert_path_still_names_handle(
        path, after_handle, name="legacy external manifest"
    )
    return _validate_external_manifest(payload, contract=contract)


def _exact_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise HarmBenchSelectionLabelCuratorError(f"{name} must be an exact integer")
    return value


def _validate_external_manifest(
    value: object, *, contract: _FrozenDatasetContract
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise HarmBenchSelectionLabelCuratorError(
            "external manifest root must be a mapping"
        )
    manifest = dict(value)
    if (
        manifest.get("schema_version") != contract.external_manifest_schema
        or manifest.get("protocol") != contract.external_manifest_protocol
        or manifest.get("status") != contract.external_manifest_status
        or manifest.get("dataset_id") != contract.dataset_id
        or manifest.get("split_protocol_id") != "scu_set_exploration_v1"
        or tuple(manifest.get("label_order", ())) != contract.ordered_class_tokens
    ):
        raise HarmBenchSelectionLabelCuratorError(
            "external manifest frozen identity or class order changed"
        )
    roles = manifest.get("roles")
    if not isinstance(roles, Mapping):
        raise HarmBenchSelectionLabelCuratorError(
            "external manifest lacks role records"
        )
    record = roles.get(SELECTION_ROLE)
    if not isinstance(record, Mapping):
        raise HarmBenchSelectionLabelCuratorError(
            "external manifest lacks model_selection record"
        )
    expected = {
        "feature_filename": contract.selection_feature_filename,
        "label_filename": contract.selection_label_filename,
        "rows": contract.rows,
        "feature_sha256": contract.selection_feature_sha256,
        "label_sha256": contract.selection_label_sha256,
        "row_alignment_sha256": contract.legacy_row_alignment_sha256,
    }
    for name, expected_value in expected.items():
        observed = record.get(name)
        if name == "rows":
            observed = _exact_int(observed, name="model_selection.rows")
        if observed != expected_value:
            raise HarmBenchSelectionLabelCuratorError(
                f"external manifest model_selection {name} changed"
            )
    return manifest


def _inspect_npz(
    archive: zipfile.ZipFile,
    *,
    expected_members: Sequence[str],
    compressed_budget: int,
    name: str,
) -> None:
    infos = archive.infolist()
    expected_archive_names = tuple(f"{member}.npy" for member in expected_members)
    names = tuple(info.filename for info in infos)
    if names != expected_archive_names or len(names) != len(set(names)):
        raise HarmBenchSelectionLabelCuratorError(
            f"{name} must have the exact ordered NPY member schema"
        )
    if archive.comment:
        raise HarmBenchSelectionLabelCuratorError(f"{name} has a ZIP comment")
    total_uncompressed = 0
    total_array_bytes = 0
    total_elements = 0
    for info in infos:
        if (
            info.is_dir()
            or info.flag_bits & 0x1
            or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            or info.file_size > MAX_LEGACY_MEMBER_BYTES
            or info.compress_size > compressed_budget
        ):
            raise HarmBenchSelectionLabelCuratorError(
                f"{name} member violates byte or ZIP budget"
            )
        total_uncompressed += int(info.file_size)
        if total_uncompressed > MAX_LEGACY_UNCOMPRESSED_BYTES:
            raise HarmBenchSelectionLabelCuratorError(
                f"{name} exceeds uncompressed byte budget"
            )
        with archive.open(info, "r") as member:
            try:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_1_0(
                        member
                    )
                elif version in {(2, 0), (3, 0)}:
                    shape, _fortran, dtype = np.lib.format.read_array_header_2_0(
                        member
                    )
                else:
                    raise HarmBenchSelectionLabelCuratorError(
                        f"{name} NPY member version is unsupported"
                    )
            except HarmBenchSelectionLabelCuratorError:
                raise
            except (EOFError, ValueError, TypeError) as error:
                raise HarmBenchSelectionLabelCuratorError(
                    f"{name} NPY header is invalid"
                ) from error
            if dtype.hasobject or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 0
                for dimension in shape
            ):
                raise HarmBenchSelectionLabelCuratorError(
                    f"{name} contains unsafe dtype or shape"
                )
            elements = int(math.prod(shape)) if shape else 1
            array_bytes = elements * int(dtype.itemsize)
            if (
                elements > MAX_LEGACY_ELEMENTS
                or array_bytes > MAX_LEGACY_MEMBER_BYTES
                or member.tell() + array_bytes != info.file_size
            ):
                raise HarmBenchSelectionLabelCuratorError(
                    f"{name} NPY member violates element or byte budget"
                )
            total_elements += elements
            total_array_bytes += array_bytes
            if (
                total_elements > MAX_LEGACY_ELEMENTS
                or total_array_bytes > MAX_LEGACY_UNCOMPRESSED_BYTES
            ):
                raise HarmBenchSelectionLabelCuratorError(
                    f"{name} violates total element or byte budget"
                )


def _read_verified_npz_selected(
    path: Path,
    *,
    expected_identity: os.stat_result,
    expected_sha256: str,
    expected_members: Sequence[str],
    selected_members: Sequence[str],
    compressed_budget: int,
    name: str,
) -> dict[str, np.ndarray]:
    if expected_identity.st_size > compressed_budget:
        raise HarmBenchSelectionLabelCuratorError(
            f"{name} exceeds compressed byte budget"
        )
    try:
        with path.open("rb") as handle:
            before_handle = os.fstat(handle.fileno())
            if not _same_identity(expected_identity, before_handle):
                raise HarmBenchSelectionLabelCuratorError(
                    f"{name} changed before verified read"
                )
            first_sha = _hash_handle(handle)
            if first_sha != expected_sha256:
                raise HarmBenchSelectionLabelCuratorError(
                    f"{name} SHA-256 differs from frozen contract"
                )
            handle.seek(0)
            with zipfile.ZipFile(handle, "r") as zip_archive:
                _inspect_npz(
                    zip_archive,
                    expected_members=expected_members,
                    compressed_budget=compressed_budget,
                    name=name,
                )
            handle.seek(0)
            with np.load(handle, allow_pickle=False) as archive:
                if tuple(archive.files) != tuple(expected_members):
                    raise HarmBenchSelectionLabelCuratorError(
                        f"{name} member order changed"
                    )
                arrays = {
                    member: np.asarray(archive[member]).copy()
                    for member in selected_members
                }
            handle.seek(0)
            second_sha = _hash_handle(handle)
            after_handle = os.fstat(handle.fileno())
    except HarmBenchSelectionLabelCuratorError:
        raise
    except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile) as error:
        raise HarmBenchSelectionLabelCuratorError(
            f"{name} is unreadable with allow_pickle=False"
        ) from error
    if first_sha != second_sha or not _same_identity(before_handle, after_handle):
        raise HarmBenchSelectionLabelCuratorError(
            f"{name} changed during verified read"
        )
    _assert_path_still_names_handle(path, after_handle, name=name)
    if any(array.dtype.kind == "O" for array in arrays.values()):
        raise HarmBenchSelectionLabelCuratorError(
            f"{name} contains a selected object array"
        )
    return arrays


def _text_scalar(
    value: np.ndarray, *, shape: tuple[int, ...], name: str
) -> str:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in {"U", "S"}:
        raise HarmBenchSelectionLabelCuratorError(
            f"{name} must have its exact legacy scalar string shape"
        )
    return str(array.reshape(-1)[0] if array.shape else array.item())


def _validate_feature_arrays(
    arrays: Mapping[str, np.ndarray], *, contract: _FrozenDatasetContract
) -> np.ndarray:
    shape = contract.scalar_shape
    if _text_scalar(arrays["schema_version"], shape=shape, name="feature schema") != (
        contract.legacy_feature_schema
    ):
        raise HarmBenchSelectionLabelCuratorError("legacy feature schema changed")
    if _text_scalar(arrays["role"], shape=shape, name="feature role") != SELECTION_ROLE:
        raise HarmBenchSelectionLabelCuratorError("legacy feature role changed")
    if _text_scalar(
        arrays["row_alignment_sha256"], shape=shape, name="feature alignment"
    ) != contract.legacy_row_alignment_sha256:
        raise HarmBenchSelectionLabelCuratorError("legacy feature alignment changed")
    if contract.feature_has_dataset_id and _text_scalar(
        arrays["dataset_id"], shape=shape, name="feature dataset"
    ) != contract.dataset_id:
        raise HarmBenchSelectionLabelCuratorError("legacy feature dataset changed")
    if contract.feature_has_split_protocol_id and _text_scalar(
        arrays["split_protocol_id"], shape=shape, name="feature split protocol"
    ) != "scu_set_exploration_v1":
        raise HarmBenchSelectionLabelCuratorError(
            "legacy feature split protocol changed"
        )
    protocol = np.asarray(arrays["protocol_row_ids"])
    if protocol.shape != (contract.rows,) or protocol.dtype != np.dtype("int64"):
        raise HarmBenchSelectionLabelCuratorError(
            "legacy feature protocol rows must be exact int64 and frozen length"
        )
    protocol = protocol.copy()
    if np.any(protocol < 0) or len(set(protocol.tolist())) != contract.rows:
        raise HarmBenchSelectionLabelCuratorError(
            "legacy feature protocol rows must be nonnegative and unique"
        )
    return protocol


def _validate_label_arrays(
    arrays: Mapping[str, np.ndarray], *, contract: _FrozenDatasetContract
) -> np.ndarray:
    shape = contract.scalar_shape
    if _text_scalar(arrays["schema_version"], shape=shape, name="label schema") != (
        contract.legacy_label_schema
    ):
        raise HarmBenchSelectionLabelCuratorError("legacy label schema changed")
    if _text_scalar(arrays["role"], shape=shape, name="label role") != SELECTION_ROLE:
        raise HarmBenchSelectionLabelCuratorError("legacy label role changed")
    if _text_scalar(
        arrays["row_alignment_sha256"], shape=shape, name="label alignment"
    ) != contract.legacy_row_alignment_sha256:
        raise HarmBenchSelectionLabelCuratorError("legacy label alignment changed")
    if contract.feature_has_dataset_id and _text_scalar(
        arrays["dataset_id"], shape=shape, name="label dataset"
    ) != contract.dataset_id:
        raise HarmBenchSelectionLabelCuratorError("legacy label dataset changed")
    if contract.feature_has_split_protocol_id and _text_scalar(
        arrays["split_protocol_id"], shape=shape, name="label split protocol"
    ) != "scu_set_exploration_v1":
        raise HarmBenchSelectionLabelCuratorError(
            "legacy label split protocol changed"
        )
    labels = np.asarray(arrays["labels"])
    if labels.shape != (contract.rows,) or labels.dtype != np.dtype("int64"):
        raise HarmBenchSelectionLabelCuratorError(
            "legacy labels must be exact int64 and frozen length"
        )
    labels = labels.copy()
    if np.any((labels < 0) | (labels >= len(contract.ordered_class_tokens))):
        raise HarmBenchSelectionLabelCuratorError(
            "legacy label lies outside frozen class order"
        )
    return labels


def _temporary_file(root: Path, destination: Path) -> tuple[Any, Path]:
    handle = tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=root,
        delete=False,
    )
    return handle, Path(handle.name)


def _move_file_write_through_windows(source: Path, destination: Path) -> None:
    """Atomically move without replacement and request Windows write-through."""

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    move = kernel32.MoveFileExW
    move.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
    move.restype = wintypes.BOOL
    movefile_write_through = 0x00000008
    if not move(str(source), str(destination), movefile_write_through):
        error = ctypes.get_last_error()
        if error in {80, 183}:  # ERROR_FILE_EXISTS / ERROR_ALREADY_EXISTS
            raise FileExistsError(
                f"write-once destination already exists: {destination.name}"
            )
        raise OSError(
            error, f"write-through publication failed: {destination.name}"
        )


def _publish_once(temporary: Path, destination: Path) -> None:
    try:
        if os.name == "nt":
            _move_file_write_through_windows(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError:
                raise FileExistsError(
                    f"write-once destination already exists: {destination.name}"
                ) from None
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sync_directory(root: Path) -> None:
    """Complete an explicit durability barrier for a published directory entry.

    POSIX provides directory ``fsync``.  On Windows, ``_publish_once`` uses
    ``MoveFileExW(MOVEFILE_WRITE_THROUGH)``; this no-op branch is still called
    after every publication so ordering is explicit and failure-injectable.
    """

    if os.name == "nt":
        return
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    descriptor = os.open(root, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_once(
    root: Path,
    destination: Path,
    raw: bytes,
    *,
    root_identity: os.stat_result,
) -> str:
    _assert_root_identity(root, root_identity, name="private output root")
    handle, temporary = _temporary_file(root, destination)
    temporary_value: Path | None = temporary
    try:
        with handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _assert_root_identity(root, root_identity, name="private output root")
        _publish_once(temporary, destination)
        temporary_value = None
        _assert_root_identity(root, root_identity, name="private output root")
        _sync_directory(root)
        _assert_root_identity(root, root_identity, name="private output root")
    finally:
        if temporary_value is not None:
            try:
                temporary_value.unlink()
            except FileNotFoundError:
                pass
    return hashlib.sha256(raw).hexdigest()


def _claim_attempt(
    root: Path,
    *,
    root_identity: os.stat_result,
    contract: _FrozenDatasetContract,
    fit_sha256: str,
    class_order_sha256: str,
) -> str:
    marker = {
        "schema_version": CURATOR_ATTEMPT_SCHEMA,
        "status": "terminal_once_claimed_no_rerun",
        "authority_scope": "independent_data_preparation_custodian_not_evaluator",
        "evidence_status": "permanently_exploratory_observed_selection",
        "future_untouched_test_policy": (
            "independent_custodian_only_never_model_or_evaluator_preopen"
        ),
        "dataset_id": contract.dataset_id,
        "role": SELECTION_ROLE,
        "rows": contract.rows,
        "legacy_external_manifest_file_sha256": contract.external_manifest_sha256,
        "legacy_selection_feature_file_sha256": contract.selection_feature_sha256,
        "legacy_selection_label_file_sha256": contract.selection_label_sha256,
        "expected_fit_training_capability_sha256": fit_sha256,
        "class_order_sha256": class_order_sha256,
    }
    return _write_bytes_once(
        root,
        root / CURATOR_ATTEMPT_FILENAME,
        _canonical_json_bytes(marker),
        root_identity=root_identity,
    )


def _universal_arrays(
    *,
    contract: _FrozenDatasetContract,
    labels: np.ndarray,
    protocol_row_ids: np.ndarray,
    class_order_sha256: str,
) -> dict[str, np.ndarray]:
    row_sha = _array_sha256(protocol_row_ids)
    return {
        "schema_version": np.asarray(UNIVERSAL_ARTIFACT_SCHEMA, dtype=np.str_),
        "dataset_id": np.asarray(contract.dataset_id, dtype=np.str_),
        "role": np.asarray(SELECTION_ROLE, dtype=np.str_),
        "rows": np.asarray(contract.rows, dtype=np.int64),
        "ordered_protocol_row_alignment_sha256": np.asarray(
            row_sha, dtype=np.str_
        ),
        "class_order_sha256": np.asarray(class_order_sha256, dtype=np.str_),
        "labels": np.asarray(labels, dtype=np.int64),
        "protocol_row_ids": np.asarray(protocol_row_ids, dtype=np.int64),
        "class_tokens": np.asarray(contract.ordered_class_tokens, dtype=np.str_),
    }


def _publish_universal_pair(
    root: Path,
    *,
    root_identity: os.stat_result,
    contract: _FrozenDatasetContract,
    labels: np.ndarray,
    protocol_row_ids: np.ndarray,
    class_order_sha256: str,
) -> tuple[dict[str, object], str]:
    artifact_path = root / UNIVERSAL_ARTIFACT_FILENAME
    manifest_path = root / UNIVERSAL_MANIFEST_FILENAME
    arrays = _universal_arrays(
        contract=contract,
        labels=labels,
        protocol_row_ids=protocol_row_ids,
        class_order_sha256=class_order_sha256,
    )
    if tuple(arrays) != _UNIVERSAL_ARRAY_ORDER:
        raise AssertionError("universal artifact member order changed")
    _assert_root_identity(root, root_identity, name="private output root")
    artifact_handle, artifact_temporary_value = _temporary_file(root, artifact_path)
    artifact_temporary: Path | None = artifact_temporary_value
    manifest_temporary: Path | None = None
    try:
        with artifact_handle:
            np.savez_compressed(artifact_handle, **arrays)
            artifact_handle.flush()
            os.fsync(artifact_handle.fileno())
            if artifact_handle.tell() > MAX_UNIVERSAL_ARTIFACT_BYTES:
                raise HarmBenchSelectionLabelCuratorError(
                    "universal selection-label artifact exceeds byte budget"
                )
            artifact_handle.seek(0)
            artifact_sha = _hash_handle(artifact_handle)
        _assert_root_identity(root, root_identity, name="private output root")
        manifest: dict[str, object] = {
            "schema_version": UNIVERSAL_MANIFEST_SCHEMA,
            "artifact_schema_version": UNIVERSAL_ARTIFACT_SCHEMA,
            "dataset_id": contract.dataset_id,
            "role": SELECTION_ROLE,
            "rows": contract.rows,
            "ordered_protocol_row_alignment_sha256": _array_sha256(
                protocol_row_ids
            ),
            "class_order_sha256": class_order_sha256,
            "artifact_filename": UNIVERSAL_ARTIFACT_FILENAME,
            "artifact_file_sha256": artifact_sha,
        }
        manifest_raw = _canonical_json_bytes(manifest)
        if len(manifest_raw) > MAX_UNIVERSAL_MANIFEST_BYTES:
            raise HarmBenchSelectionLabelCuratorError(
                "universal selection-label manifest exceeds byte budget"
            )
        manifest_handle, manifest_temporary = _temporary_file(root, manifest_path)
        with manifest_handle:
            manifest_handle.write(manifest_raw)
            manifest_handle.flush()
            os.fsync(manifest_handle.fileno())
        _assert_root_identity(root, root_identity, name="private output root")
        assert artifact_temporary is not None
        _publish_once(artifact_temporary, artifact_path)
        artifact_temporary = None
        _assert_root_identity(root, root_identity, name="private output root")
        _sync_directory(root)
        _assert_root_identity(root, root_identity, name="private output root")
        assert manifest_temporary is not None
        # Deliberately do not delete the already-published artifact if this
        # final publication fails.  The durable attempt marker makes the
        # partial state terminal; the missing manifest keeps activation shut.
        _publish_once(manifest_temporary, manifest_path)
        manifest_temporary = None
        _assert_root_identity(root, root_identity, name="private output root")
        _sync_directory(root)
        _assert_root_identity(root, root_identity, name="private output root")
    finally:
        for temporary in (artifact_temporary, manifest_temporary):
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
    return manifest, hashlib.sha256(manifest_raw).hexdigest()


def curate_frozen_legacy_selection_labels(
    *,
    dataset_id: str,
    legacy_sidecar_root: str | Path,
    external_manifest_path: str | Path,
    private_output_root: str | Path,
    expected_fit_training_capability_sha256: str,
) -> CuratedSelectionLabelReceipt:
    """Publish one frozen legacy selection label set in universal form.

    Outcome-free manifest and feature validation occurs first.  The independent
    curator-ingest marker is then durably claimed before the legacy label path
    is touched.  It is not an evaluator attempt.  Every call is write-once,
    including failed or crashed calls after claim.  The function returns only
    label-free preparation metadata: it does not return outcomes, class counts,
    statistics, or an evaluation result.  These observed selection datasets
    remain permanently exploratory.  This entry point must not be invoked by a
    model/evaluator to pre-open a future untouched test set; such a test requires
    a separately operated independent custodian boundary.
    """

    contract = _contract(dataset_id)
    fit_sha = _sha256(
        expected_fit_training_capability_sha256,
        name="expected_fit_training_capability_sha256",
    )
    class_sha = frozen_selection_class_order_sha256(
        dataset_id=contract.dataset_id,
        expected_fit_training_capability_sha256=fit_sha,
    )
    legacy_root = _directory(legacy_sidecar_root, name="legacy sidecar root")
    output_root = _output_root(private_output_root)
    output_root_identity = _plain_directory_stat(
        output_root, name="private output root"
    )
    manifest_path, manifest_identity = _external_manifest_path(
        external_manifest_path, contract=contract
    )
    _read_external_manifest(
        manifest_path, expected_identity=manifest_identity, contract=contract
    )

    feature_path, feature_identity = _selection_child(
        legacy_root,
        contract.selection_feature_filename,
        name="legacy model_selection feature NPZ",
    )
    feature_names = [
        "schema_version",
        "role",
        "row_alignment_sha256",
        "protocol_row_ids",
    ]
    if contract.feature_has_dataset_id:
        feature_names.append("dataset_id")
    if contract.feature_has_split_protocol_id:
        feature_names.append("split_protocol_id")
    feature_arrays = _read_verified_npz_selected(
        feature_path,
        expected_identity=feature_identity,
        expected_sha256=contract.selection_feature_sha256,
        expected_members=contract.feature_member_order,
        selected_members=feature_names,
        compressed_budget=MAX_LEGACY_FEATURE_COMPRESSED_BYTES,
        name="legacy model_selection feature NPZ",
    )
    protocol_row_ids = _validate_feature_arrays(feature_arrays, contract=contract)
    del feature_arrays

    attempt_sha = _claim_attempt(
        output_root,
        root_identity=output_root_identity,
        contract=contract,
        fit_sha256=fit_sha,
        class_order_sha256=class_sha,
    )

    # No operation on this path may move above the attempt claim.
    label_path, label_identity = _selection_child(
        legacy_root,
        contract.selection_label_filename,
        name="legacy model_selection label NPZ",
    )
    label_names = ["schema_version", "role", "row_alignment_sha256", "labels"]
    if contract.feature_has_dataset_id:
        label_names.append("dataset_id")
    if contract.feature_has_split_protocol_id:
        label_names.append("split_protocol_id")
    label_arrays = _read_verified_npz_selected(
        label_path,
        expected_identity=label_identity,
        expected_sha256=contract.selection_label_sha256,
        expected_members=contract.label_member_order,
        selected_members=label_names,
        compressed_budget=MAX_LEGACY_LABEL_COMPRESSED_BYTES,
        name="legacy model_selection label NPZ",
    )
    labels = _validate_label_arrays(label_arrays, contract=contract)
    del label_arrays

    manifest, manifest_sha = _publish_universal_pair(
        output_root,
        root_identity=output_root_identity,
        contract=contract,
        labels=labels,
        protocol_row_ids=protocol_row_ids,
        class_order_sha256=class_sha,
    )
    del labels
    return CuratedSelectionLabelReceipt(
        dataset_id=contract.dataset_id,
        role=SELECTION_ROLE,
        rows=contract.rows,
        legacy_external_manifest_file_sha256=contract.external_manifest_sha256,
        legacy_selection_feature_file_sha256=contract.selection_feature_sha256,
        legacy_selection_label_file_sha256=contract.selection_label_sha256,
        expected_fit_training_capability_sha256=fit_sha,
        ordered_protocol_row_alignment_sha256=str(
            manifest["ordered_protocol_row_alignment_sha256"]
        ),
        class_order_sha256=class_sha,
        artifact_file_sha256=str(manifest["artifact_file_sha256"]),
        manifest_file_sha256=manifest_sha,
        attempt_marker_file_sha256=attempt_sha,
    )


__all__ = [
    "CHECKPOINT_CLASS_ORDER_SCHEMA",
    "CURATOR_ATTEMPT_FILENAME",
    "CURATOR_ATTEMPT_SCHEMA",
    "CuratedSelectionLabelReceipt",
    "HarmBenchSelectionLabelCuratorError",
    "SELECTION_ROLE",
    "UNIVERSAL_ARTIFACT_FILENAME",
    "UNIVERSAL_ARTIFACT_SCHEMA",
    "UNIVERSAL_MANIFEST_FILENAME",
    "UNIVERSAL_MANIFEST_SCHEMA",
    "curate_frozen_legacy_selection_labels",
    "frozen_selection_class_order_sha256",
]
