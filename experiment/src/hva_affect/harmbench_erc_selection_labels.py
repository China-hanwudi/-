"""Fail-closed, selection-label-only artifact capability for HarmBench-ERC.

The public phase in this module reads only a small canonical manifest.  It
does not resolve, stat, hash, or open the label NPZ named by that manifest.
The private activation seam is deliberately separate so that the future
selection evaluator can durably publish its attempt marker before calling it.

No feature, text, audio, video, speaker, group, history, path, prediction, or
outcome-derived stratum metadata is permitted in the manifest.  The private
NPZ contains only the ordered protocol rows, their integer labels, and the
frozen ordered class tokens needed to prove exact evaluator alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
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


SELECTION_LABEL_ROLE = "model_selection"
SELECTION_LABEL_MANIFEST_SCHEMA = "harmbench_erc_selection_label_manifest_v1"
SELECTION_LABEL_ARTIFACT_SCHEMA = "harmbench_erc_selection_labels_private_v1"
SELECTION_LABEL_ARTIFACT_FILENAME = "harmbench_erc_selection_labels.npz"
SELECTION_LABEL_MANIFEST_FILENAME = "harmbench_erc_selection_labels.manifest.json"

MAX_SELECTION_LABEL_ROWS = 1_000_000
MAX_SELECTION_LABEL_CLASSES = 512
MAX_CLASS_TOKEN_CHARACTERS = 256
MAX_SELECTION_LABEL_MANIFEST_BYTES = 16 * 1024
MAX_SELECTION_LABEL_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_SELECTION_LABEL_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_SELECTION_LABEL_MEMBER_BYTES = 32 * 1024 * 1024
MAX_SELECTION_LABEL_ELEMENTS = (
    2 * MAX_SELECTION_LABEL_ROWS + MAX_SELECTION_LABEL_CLASSES + 6
)

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_FIELDS = (
    "schema_version",
    "artifact_schema_version",
    "dataset_id",
    "role",
    "rows",
    "ordered_protocol_row_alignment_sha256",
    "class_order_sha256",
    "artifact_filename",
    "artifact_file_sha256",
)
_NPZ_MEMBER_ORDER = (
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
_NPZ_ARCHIVE_MEMBER_ORDER = tuple(f"{name}.npy" for name in _NPZ_MEMBER_ORDER)
_METADATA_SEAL = object()
_ORIGIN_SEAL = object()
_ACTIVATED_SEAL = object()
_ACTIVATED_ORIGIN_SEAL = object()


class HarmBenchSelectionLabelError(ValueError):
    """Raised when the selection-label contract or private I/O changes."""


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise HarmBenchSelectionLabelError(
            f"{name} must match {_IDENTIFIER_PATTERN.pattern}"
        )
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise HarmBenchSelectionLabelError(f"{name} must be a lowercase SHA-256")
    return value


def _exact_int(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HarmBenchSelectionLabelError(f"{name} must be an exact integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise HarmBenchSelectionLabelError(f"{name} is below its minimum")
    if maximum is not None and result > maximum:
        raise HarmBenchSelectionLabelError(f"{name} exceeds its maximum")
    return result


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
        raise HarmBenchSelectionLabelError(
            f"selection-label manifest is not canonical JSON data: {error}"
        ) from error


def _canonical_json_bytes(value: object) -> bytes:
    return _canonical_json_payload(value) + b"\n"


def _array_sha256(values: object) -> str:
    """Match the canonical ndarray hash used by prediction query rosters."""

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
            canonical = canonical.byteswap().view(canonical.dtype.newbyteorder("<"))
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _protocol_row_ids(values: object, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or not len(raw):
        raise HarmBenchSelectionLabelError(f"{name} must be a non-empty vector")
    if raw.dtype != np.dtype("int64"):
        raise HarmBenchSelectionLabelError(f"{name} must have exact int64 dtype")
    if len(raw) > MAX_SELECTION_LABEL_ROWS:
        raise HarmBenchSelectionLabelError(f"{name} exceeds the row budget")
    result = np.asarray(raw, dtype=np.int64).copy()
    if len(set(result.tolist())) != len(result):
        raise HarmBenchSelectionLabelError(f"{name} must be unique")
    return result


def selection_protocol_row_alignment_sha256(protocol_row_ids: object) -> str:
    """Hash an exact ordered, unique, int64 selection protocol-row vector."""

    return _array_sha256(
        _protocol_row_ids(protocol_row_ids, name="protocol_row_ids")
    )


def _class_tokens(values: object, *, name: str) -> np.ndarray:
    try:
        sequence = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise HarmBenchSelectionLabelError(f"{name} must be a sequence") from error
    if not 2 <= len(sequence) <= MAX_SELECTION_LABEL_CLASSES:
        raise HarmBenchSelectionLabelError(
            f"{name} must contain 2..{MAX_SELECTION_LABEL_CLASSES} classes"
        )
    tokens: list[str] = []
    for value in sequence:
        if (
            not isinstance(value, (str, np.str_))
            or not str(value)
            or len(str(value)) > MAX_CLASS_TOKEN_CHARACTERS
            or any(ord(character) < 32 for character in str(value))
        ):
            raise HarmBenchSelectionLabelError(
                f"{name} contains an invalid class token"
            )
        tokens.append(str(value))
    if len(set(tokens)) != len(tokens):
        raise HarmBenchSelectionLabelError(f"{name} must contain unique tokens")
    return np.asarray(tokens, dtype=np.str_)


def _label_vector(values: object, *, rows: int, classes: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (rows,) or raw.dtype != np.dtype("int64"):
        raise HarmBenchSelectionLabelError(
            "labels must be one exact int64 value per protocol row"
        )
    result = np.asarray(raw, dtype=np.int64).copy()
    if np.any((result < 0) | (result >= classes)):
        raise HarmBenchSelectionLabelError(
            "selection label lies outside the frozen class order"
        )
    return result


def _validate_manifest(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(_MANIFEST_FIELDS):
        raise HarmBenchSelectionLabelError(
            "selection-label manifest must have the exact label-only schema"
        )
    manifest = dict(value)
    if manifest["schema_version"] != SELECTION_LABEL_MANIFEST_SCHEMA:
        raise HarmBenchSelectionLabelError("selection-label manifest schema changed")
    if manifest["artifact_schema_version"] != SELECTION_LABEL_ARTIFACT_SCHEMA:
        raise HarmBenchSelectionLabelError("selection-label artifact schema changed")
    _identifier(manifest["dataset_id"], name="dataset_id")
    if manifest["role"] != SELECTION_LABEL_ROLE:
        raise HarmBenchSelectionLabelError("selection-label role changed")
    _exact_int(
        manifest["rows"],
        name="rows",
        minimum=1,
        maximum=MAX_SELECTION_LABEL_ROWS,
    )
    _sha256(
        manifest["ordered_protocol_row_alignment_sha256"],
        name="ordered_protocol_row_alignment_sha256",
    )
    _sha256(manifest["class_order_sha256"], name="class_order_sha256")
    if manifest["artifact_filename"] != SELECTION_LABEL_ARTIFACT_FILENAME:
        raise HarmBenchSelectionLabelError(
            "selection-label artifact filename is not the fixed relative filename"
        )
    _sha256(manifest["artifact_file_sha256"], name="artifact_file_sha256")
    return manifest


def selection_label_manifest_sha256(manifest: object) -> str:
    """Return the external file binding for an exact canonical manifest."""

    validated = _validate_manifest(manifest)
    encoded = _canonical_json_bytes(validated)
    if len(encoded) > MAX_SELECTION_LABEL_MANIFEST_BYTES:
        raise HarmBenchSelectionLabelError("selection-label manifest exceeds byte budget")
    return hashlib.sha256(encoded).hexdigest()


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
            raise HarmBenchSelectionLabelError(
                "selection-label path contains a symlink or reparse point"
            )


def _plain_file_stat(path: Path, *, name: str) -> os.stat_result:
    try:
        observed = path.lstat()
    except OSError as error:
        raise HarmBenchSelectionLabelError(f"cannot stat exact {name}") from error
    if stat.S_ISLNK(observed.st_mode) or _is_reparse(observed):
        raise HarmBenchSelectionLabelError(
            f"{name} cannot be a symlink or reparse point"
        )
    if not stat.S_ISREG(observed.st_mode):
        raise HarmBenchSelectionLabelError(f"{name} must be a plain file")
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


def _same_file_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return _identity(first) == _identity(second)


def _assert_path_still_names_handle(
    path: Path, handle_stat: os.stat_result, *, name: str
) -> None:
    _reject_reparse_components(path)
    if not _same_file_identity(_plain_file_stat(path, name=name), handle_stat):
        raise HarmBenchSelectionLabelError(
            f"{name} path changed identity during verified read"
        )


def _hash_open_handle(handle: Any) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


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


def _validate_private_root(path: str | Path) -> Path:
    raw = Path(path)
    if not raw.is_absolute() or raw.is_symlink():
        raise HarmBenchSelectionLabelError(
            "private root must be an explicit absolute non-symlink directory"
        )
    _reject_reparse_components(raw)
    try:
        observed = raw.lstat()
        if stat.S_ISLNK(observed.st_mode) or _is_reparse(observed):
            raise HarmBenchSelectionLabelError(
                "private root cannot be a symlink or reparse point"
            )
        root = raw.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise HarmBenchSelectionLabelError(
            "private root must already exist"
        ) from error
    if not root.is_dir() or root == Path(root.anchor):
        raise HarmBenchSelectionLabelError(
            "private root must be a safe non-root directory"
        )
    if _is_within(root, _repository_root().resolve()) or _is_within(
        root, _home_root().resolve()
    ):
        raise HarmBenchSelectionLabelError(
            "private root must be outside both the repository and user home"
        )
    return root


def _canonical_child(root: Path, path: str | Path, *, filename: str, name: str) -> Path:
    raw = Path(path)
    if not raw.is_absolute() or raw.name != filename:
        raise HarmBenchSelectionLabelError(
            f"{name} must use the fixed canonical filename"
        )
    _reject_reparse_components(raw)
    if raw.parent.resolve(strict=True) != root:
        raise HarmBenchSelectionLabelError(
            f"{name} must be a direct child of the explicit private root"
        )
    return raw


def _validated_existing_manifest(
    private_root: str | Path, manifest_path: str | Path
) -> tuple[Path, Path, os.stat_result]:
    root = _validate_private_root(private_root)
    path = _canonical_child(
        root,
        manifest_path,
        filename=SELECTION_LABEL_MANIFEST_FILENAME,
        name="selection-label manifest",
    )
    return root, path, _plain_file_stat(path, name="selection-label manifest")


def _decode_manifest(
    path: Path,
    *,
    expected_manifest_sha256: str,
    expected_identity: os.stat_result,
) -> dict[str, object]:
    expected = _sha256(
        expected_manifest_sha256, name="expected_manifest_sha256"
    )
    before_path = _plain_file_stat(path, name="selection-label manifest")
    if not _same_file_identity(before_path, expected_identity):
        raise HarmBenchSelectionLabelError(
            "selection-label manifest changed after path validation"
        )
    if before_path.st_size > MAX_SELECTION_LABEL_MANIFEST_BYTES:
        raise HarmBenchSelectionLabelError(
            "selection-label manifest exceeds byte budget"
        )

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HarmBenchSelectionLabelError(
                    f"selection-label manifest contains duplicate JSON key: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise HarmBenchSelectionLabelError(
            f"selection-label manifest contains invalid JSON constant: {value}"
        )

    try:
        with path.open("rb") as handle:
            before_handle = os.fstat(handle.fileno())
            if not _same_file_identity(before_path, before_handle):
                raise HarmBenchSelectionLabelError(
                    "selection-label manifest changed before verified read"
                )
            first_sha = _hash_open_handle(handle)
            if first_sha != expected:
                raise HarmBenchSelectionLabelError(
                    "selection-label manifest SHA-256 changed"
                )
            handle.seek(0)
            encoded = handle.read(MAX_SELECTION_LABEL_MANIFEST_BYTES + 1)
            if len(encoded) > MAX_SELECTION_LABEL_MANIFEST_BYTES:
                raise HarmBenchSelectionLabelError(
                    "selection-label manifest exceeds byte budget"
                )
            try:
                payload = json.loads(
                    encoded.decode("utf-8"),
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=reject_constant,
                )
            except HarmBenchSelectionLabelError:
                raise
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise HarmBenchSelectionLabelError(
                    "selection-label manifest is not strict UTF-8 JSON"
                ) from error
            manifest = _validate_manifest(payload)
            if encoded != _canonical_json_bytes(manifest):
                raise HarmBenchSelectionLabelError(
                    "selection-label manifest is not canonical JSON"
                )
            handle.seek(0)
            second_sha = _hash_open_handle(handle)
            after_handle = os.fstat(handle.fileno())
    except HarmBenchSelectionLabelError:
        raise
    except OSError as error:
        raise HarmBenchSelectionLabelError(
            "selection-label manifest cannot be read"
        ) from error
    if first_sha != second_sha or not _same_file_identity(
        before_handle, after_handle
    ):
        raise HarmBenchSelectionLabelError(
            "selection-label manifest changed during verified read"
        )
    _assert_path_still_names_handle(
        path, after_handle, name="selection-label manifest"
    )
    return manifest


@dataclass(frozen=True)
class _ManifestOrigin:
    private_root: Path
    manifest_path: Path
    manifest_file_sha256: str
    descriptor: tuple[object, ...]
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _ORIGIN_SEAL:
            raise HarmBenchSelectionLabelError(
                "selection-label manifest origin can only be minted by its loader"
            )


@dataclass(frozen=True)
class SelectionLabelManifestMetadata:
    """Outcome-safe label metadata; it carries no resolved label path."""

    schema_version: str
    artifact_schema_version: str
    dataset_id: str
    role: str
    rows: int
    ordered_protocol_row_alignment_sha256: str
    class_order_sha256: str
    artifact_filename: str
    artifact_file_sha256: str
    manifest_file_sha256: str
    _origin: _ManifestOrigin = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self._seal is not _METADATA_SEAL
            or type(self._origin) is not _ManifestOrigin
            or self._origin._seal is not _ORIGIN_SEAL
        ):
            raise HarmBenchSelectionLabelError(
                "selection-label metadata can only be created by the manifest loader"
            )


def _metadata_descriptor(metadata: SelectionLabelManifestMetadata) -> tuple[object, ...]:
    return tuple(
        getattr(metadata, item.name)
        for item in fields(SelectionLabelManifestMetadata)
        if not item.name.startswith("_")
    )


def load_selection_label_manifest_metadata(
    *,
    private_root: str | Path,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
) -> SelectionLabelManifestMetadata:
    """Load only the canonical JSON manifest, without touching the label NPZ."""

    root, path, identity = _validated_existing_manifest(private_root, manifest_path)
    manifest = _decode_manifest(
        path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_identity=identity,
    )
    expected = _sha256(
        expected_manifest_sha256, name="expected_manifest_sha256"
    )
    public_values = (
        manifest["schema_version"],
        manifest["artifact_schema_version"],
        manifest["dataset_id"],
        manifest["role"],
        manifest["rows"],
        manifest["ordered_protocol_row_alignment_sha256"],
        manifest["class_order_sha256"],
        manifest["artifact_filename"],
        manifest["artifact_file_sha256"],
        expected,
    )
    origin = _ManifestOrigin(
        private_root=root,
        manifest_path=path,
        manifest_file_sha256=expected,
        descriptor=public_values,
        _seal=_ORIGIN_SEAL,
    )
    return SelectionLabelManifestMetadata(
        schema_version=str(public_values[0]),
        artifact_schema_version=str(public_values[1]),
        dataset_id=str(public_values[2]),
        role=str(public_values[3]),
        rows=int(public_values[4]),
        ordered_protocol_row_alignment_sha256=str(public_values[5]),
        class_order_sha256=str(public_values[6]),
        artifact_filename=str(public_values[7]),
        artifact_file_sha256=str(public_values[8]),
        manifest_file_sha256=str(public_values[9]),
        _origin=origin,
        _seal=_METADATA_SEAL,
    )


def _revalidate_manifest_metadata(
    metadata: object,
) -> SelectionLabelManifestMetadata:
    if (
        type(metadata) is not SelectionLabelManifestMetadata
        or metadata._seal is not _METADATA_SEAL
        or type(metadata._origin) is not _ManifestOrigin
        or metadata._origin._seal is not _ORIGIN_SEAL
        or _metadata_descriptor(metadata) != metadata._origin.descriptor
    ):
        raise HarmBenchSelectionLabelError(
            "loader-minted selection-label manifest metadata is required"
        )
    live = load_selection_label_manifest_metadata(
        private_root=metadata._origin.private_root,
        manifest_path=metadata._origin.manifest_path,
        expected_manifest_sha256=metadata._origin.manifest_file_sha256,
    )
    if _metadata_descriptor(live) != _metadata_descriptor(metadata):
        raise HarmBenchSelectionLabelError(
            "selection-label manifest metadata changed after loading"
        )
    return live


def _validated_existing_label_file(
    root: Path, artifact_filename: str
) -> tuple[Path, os.stat_result]:
    if artifact_filename != SELECTION_LABEL_ARTIFACT_FILENAME:
        raise HarmBenchSelectionLabelError("selection-label artifact filename changed")
    path = root / artifact_filename
    _reject_reparse_components(path)
    if path.parent.resolve(strict=True) != root or path.name != artifact_filename:
        raise HarmBenchSelectionLabelError(
            "selection-label artifact must be the fixed direct child"
        )
    observed = _plain_file_stat(path, name="selection-label NPZ")
    if observed.st_size > MAX_SELECTION_LABEL_ARTIFACT_BYTES:
        raise HarmBenchSelectionLabelError(
            "selection-label NPZ exceeds compressed byte budget"
        )
    return path, observed


def _inspect_npz_budget(archive: zipfile.ZipFile) -> None:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if tuple(names) != _NPZ_ARCHIVE_MEMBER_ORDER or len(names) != len(set(names)):
        raise HarmBenchSelectionLabelError(
            "selection-label NPZ must have the exact ordered member schema"
        )
    if archive.comment:
        raise HarmBenchSelectionLabelError("selection-label NPZ has a ZIP comment")
    total_zip_bytes = 0
    total_array_bytes = 0
    total_elements = 0
    for info in infos:
        if (
            info.is_dir()
            or info.flag_bits & 0x1
            or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            or info.file_size > MAX_SELECTION_LABEL_MEMBER_BYTES
            or info.compress_size > MAX_SELECTION_LABEL_ARTIFACT_BYTES
        ):
            raise HarmBenchSelectionLabelError(
                "selection-label NPZ member violates its byte or ZIP contract"
            )
        total_zip_bytes += int(info.file_size)
        if total_zip_bytes > MAX_SELECTION_LABEL_UNCOMPRESSED_BYTES:
            raise HarmBenchSelectionLabelError(
                "selection-label NPZ exceeds uncompressed byte budget"
            )
        with archive.open(info, "r") as member:
            try:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_1_0(
                        member
                    )
                elif version == (2, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_2_0(
                        member
                    )
                elif version == (3, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_2_0(
                        member
                    )
                else:
                    raise HarmBenchSelectionLabelError(
                        "selection-label NPY member uses an unsupported version"
                    )
            except HarmBenchSelectionLabelError:
                raise
            except (EOFError, ValueError, TypeError) as error:
                raise HarmBenchSelectionLabelError(
                    "selection-label NPY header is invalid"
                ) from error
            if dtype.hasobject or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 0
                for dimension in shape
            ):
                raise HarmBenchSelectionLabelError(
                    "selection-label NPY member has an unsafe dtype or shape"
                )
            elements = int(math.prod(shape)) if shape else 1
            array_bytes = elements * int(dtype.itemsize)
            if (
                elements > MAX_SELECTION_LABEL_ELEMENTS
                or array_bytes > MAX_SELECTION_LABEL_MEMBER_BYTES
                or member.tell() + array_bytes != info.file_size
            ):
                raise HarmBenchSelectionLabelError(
                    "selection-label NPY member violates element or byte budget"
                )
            total_elements += elements
            total_array_bytes += array_bytes
            if (
                total_elements > MAX_SELECTION_LABEL_ELEMENTS
                or total_array_bytes > MAX_SELECTION_LABEL_UNCOMPRESSED_BYTES
            ):
                raise HarmBenchSelectionLabelError(
                    "selection-label NPZ violates total element or byte budget"
                )


def _load_label_npz_once(
    path: Path,
    *,
    expected_identity: os.stat_result,
    expected_artifact_sha256: str,
) -> dict[str, np.ndarray]:
    expected_sha = _sha256(
        expected_artifact_sha256, name="artifact_file_sha256"
    )
    before_path = _plain_file_stat(path, name="selection-label NPZ")
    if not _same_file_identity(before_path, expected_identity):
        raise HarmBenchSelectionLabelError(
            "selection-label NPZ changed after path validation"
        )
    if before_path.st_size > MAX_SELECTION_LABEL_ARTIFACT_BYTES:
        raise HarmBenchSelectionLabelError(
            "selection-label NPZ exceeds compressed byte budget"
        )
    try:
        with path.open("rb") as handle:
            before_handle = os.fstat(handle.fileno())
            if not _same_file_identity(before_path, before_handle):
                raise HarmBenchSelectionLabelError(
                    "selection-label NPZ changed before verified read"
                )
            first_sha = _hash_open_handle(handle)
            if first_sha != expected_sha:
                raise HarmBenchSelectionLabelError(
                    "selection-label NPZ SHA-256 changed"
                )
            handle.seek(0)
            with zipfile.ZipFile(handle, "r") as zip_archive:
                _inspect_npz_budget(zip_archive)
            handle.seek(0)
            with np.load(handle, allow_pickle=False) as archive:
                if tuple(archive.files) != _NPZ_MEMBER_ORDER:
                    raise HarmBenchSelectionLabelError(
                        "selection-label NPZ member order changed"
                    )
                arrays = {
                    name: np.asarray(archive[name]).copy() for name in archive.files
                }
            handle.seek(0)
            second_sha = _hash_open_handle(handle)
            after_handle = os.fstat(handle.fileno())
    except HarmBenchSelectionLabelError:
        raise
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, RuntimeError) as error:
        raise HarmBenchSelectionLabelError(
            "selection-label NPZ is unreadable with allow_pickle=False"
        ) from error
    if first_sha != second_sha or not _same_file_identity(
        before_handle, after_handle
    ):
        raise HarmBenchSelectionLabelError(
            "selection-label NPZ changed during verified load"
        )
    _assert_path_still_names_handle(path, after_handle, name="selection-label NPZ")
    if any(array.dtype.kind == "O" for array in arrays.values()):
        raise HarmBenchSelectionLabelError(
            "selection-label NPZ contains an object array"
        )
    return arrays


def _read_unicode_scalar(value: object, *, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind != "U":
        raise HarmBenchSelectionLabelError(f"{name} must be one Unicode scalar")
    return str(array.item())


def _read_int64_scalar(value: object, *, name: str) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.dtype("int64"):
        raise HarmBenchSelectionLabelError(f"{name} must be one int64 scalar")
    return int(array.item())


def _immutable_copy(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value)
    result = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=contiguous.dtype
    ).reshape(contiguous.shape)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class _ActivatedOrigin:
    manifest_descriptor: tuple[object, ...]
    labels_sha256: str
    protocol_row_ids_sha256: str
    class_tokens_sha256: str
    protocol_canonical_sha256: str
    attempt_marker_file_sha256: str
    prelabel_bundle_file_sha256: str
    prelabel_receipt_file_sha256: str
    ticket_binding_sha256: str
    _ticket: object = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _ACTIVATED_ORIGIN_SEAL:
            raise HarmBenchSelectionLabelError(
                "activated label origin can only be minted by the private loader"
            )


@dataclass(frozen=True)
class ActivatedSelectionLabelCapability:
    """Loader-sealed, exact-alignment labels for the future evaluator."""

    dataset_id: str
    role: str
    rows: int
    ordered_protocol_row_alignment_sha256: str
    class_order_sha256: str
    artifact_file_sha256: str
    manifest_file_sha256: str
    protocol_canonical_sha256: str
    attempt_marker_file_sha256: str
    prelabel_bundle_file_sha256: str
    prelabel_receipt_file_sha256: str
    ticket_binding_sha256: str
    labels: np.ndarray = field(repr=False, compare=False)
    protocol_row_ids: np.ndarray = field(repr=False, compare=False)
    class_tokens: np.ndarray = field(repr=False, compare=False)
    _origin: _ActivatedOrigin = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self._seal is not _ACTIVATED_SEAL
            or type(self._origin) is not _ActivatedOrigin
            or self._origin._seal is not _ACTIVATED_ORIGIN_SEAL
            or self.labels.flags.writeable
            or self.protocol_row_ids.flags.writeable
            or self.class_tokens.flags.writeable
        ):
            raise HarmBenchSelectionLabelError(
                "activated selection labels can only be created by the private loader"
            )
        for name in (
            "protocol_canonical_sha256",
            "attempt_marker_file_sha256",
            "prelabel_bundle_file_sha256",
            "prelabel_receipt_file_sha256",
            "ticket_binding_sha256",
        ):
            _sha256(getattr(self, name), name=name)


def _validate_loaded_arrays(
    arrays: Mapping[str, np.ndarray],
    metadata: SelectionLabelManifestMetadata,
    *,
    expected_protocol_row_ids: np.ndarray,
    expected_class_tokens: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tuple(arrays) != _NPZ_MEMBER_ORDER:
        raise HarmBenchSelectionLabelError(
            "selection-label NPZ must have the exact member schema"
        )
    if (
        _read_unicode_scalar(arrays["schema_version"], name="schema_version")
        != metadata.artifact_schema_version
        or _read_unicode_scalar(arrays["dataset_id"], name="dataset_id")
        != metadata.dataset_id
        or _read_unicode_scalar(arrays["role"], name="role") != metadata.role
        or _read_int64_scalar(arrays["rows"], name="rows") != metadata.rows
        or _read_unicode_scalar(
            arrays["ordered_protocol_row_alignment_sha256"],
            name="ordered_protocol_row_alignment_sha256",
        )
        != metadata.ordered_protocol_row_alignment_sha256
        or _read_unicode_scalar(
            arrays["class_order_sha256"], name="class_order_sha256"
        )
        != metadata.class_order_sha256
    ):
        raise HarmBenchSelectionLabelError(
            "selection-label NPZ scalars differ from its manifest"
        )
    protocol = _protocol_row_ids(arrays["protocol_row_ids"], name="protocol_row_ids")
    if (
        protocol.shape != (metadata.rows,)
        or not np.array_equal(protocol, expected_protocol_row_ids)
        or _array_sha256(protocol)
        != metadata.ordered_protocol_row_alignment_sha256
    ):
        raise HarmBenchSelectionLabelError(
            "selection-label protocol-row alignment changed"
        )
    raw_tokens = np.asarray(arrays["class_tokens"])
    if raw_tokens.ndim != 1 or raw_tokens.dtype.kind != "U":
        raise HarmBenchSelectionLabelError(
            "class_tokens must be one Unicode vector"
        )
    tokens = _class_tokens(raw_tokens.tolist(), name="class_tokens")
    if not np.array_equal(tokens, expected_class_tokens):
        raise HarmBenchSelectionLabelError(
            "selection-label class-token order changed"
        )
    labels = _label_vector(
        arrays["labels"], rows=metadata.rows, classes=len(tokens)
    )
    return (
        _immutable_copy(labels),
        _immutable_copy(protocol),
        _immutable_copy(tokens),
    )


def _activate_selection_labels_from_attempt_ticket(
    ticket: object,
) -> ActivatedSelectionLabelCapability:
    """Consume one attempt-bound ticket, then open its sidecar exactly once."""

    # Function-local import avoids a module-initialisation cycle: prelabel
    # imports the manifest metadata type from this module, while this production
    # seam needs prelabel's private temporal capability.
    from . import harmbench_erc_selection_prelabel as _prelabel

    begun = False
    consumed_ticket: object | None = None
    try:
        consumed_ticket = _prelabel._begin_attempt_ticket_consumption(  # noqa: SLF001
            ticket
        )
        begun = True
        live = _prelabel._revalidate_consuming_attempt_ticket(  # noqa: SLF001
            consumed_ticket
        )
        expected_ids = _protocol_row_ids(
            consumed_ticket.expected_protocol_row_ids,
            name="ticket.expected_protocol_row_ids",
        )
        expected_tokens = _class_tokens(
            consumed_ticket.expected_class_tokens,
            name="ticket.expected_class_tokens",
        )
        if (
            live.dataset_id != consumed_ticket.dataset_id
            or live.role != SELECTION_LABEL_ROLE
            or live.rows != len(expected_ids)
            or live.ordered_protocol_row_alignment_sha256
            != _array_sha256(expected_ids)
            or live.class_order_sha256 != consumed_ticket.class_order_sha256
        ):
            raise HarmBenchSelectionLabelError(
                "ticket/manifest differs from the frozen evaluator alignment"
            )

        # The ticket was irreversibly marked consumed before this first label
        # path formation/stat.  Any failure below is terminal and unretryable.
        artifact_path, identity = _validated_existing_label_file(
            live._origin.private_root, live.artifact_filename
        )
        arrays = _load_label_npz_once(
            artifact_path,
            expected_identity=identity,
            expected_artifact_sha256=live.artifact_file_sha256,
        )
        labels, protocol, tokens = _validate_loaded_arrays(
            arrays,
            live,
            expected_protocol_row_ids=expected_ids,
            expected_class_tokens=expected_tokens,
        )
        origin = _ActivatedOrigin(
            manifest_descriptor=_metadata_descriptor(live),
            labels_sha256=_array_sha256(labels),
            protocol_row_ids_sha256=_array_sha256(protocol),
            class_tokens_sha256=_array_sha256(tokens),
            protocol_canonical_sha256=(
                consumed_ticket.protocol_canonical_sha256
            ),
            attempt_marker_file_sha256=consumed_ticket.marker_file_sha256,
            prelabel_bundle_file_sha256=(
                consumed_ticket.prelabel_bundle_file_sha256
            ),
            prelabel_receipt_file_sha256=(
                consumed_ticket.prelabel_receipt_file_sha256
            ),
            ticket_binding_sha256=consumed_ticket.ticket_binding_sha256,
            _ticket=consumed_ticket,
            _seal=_ACTIVATED_ORIGIN_SEAL,
        )
        capability = ActivatedSelectionLabelCapability(
            dataset_id=live.dataset_id,
            role=live.role,
            rows=live.rows,
            ordered_protocol_row_alignment_sha256=(
                live.ordered_protocol_row_alignment_sha256
            ),
            class_order_sha256=live.class_order_sha256,
            artifact_file_sha256=live.artifact_file_sha256,
            manifest_file_sha256=live.manifest_file_sha256,
            protocol_canonical_sha256=(
                consumed_ticket.protocol_canonical_sha256
            ),
            attempt_marker_file_sha256=consumed_ticket.marker_file_sha256,
            prelabel_bundle_file_sha256=(
                consumed_ticket.prelabel_bundle_file_sha256
            ),
            prelabel_receipt_file_sha256=(
                consumed_ticket.prelabel_receipt_file_sha256
            ),
            ticket_binding_sha256=consumed_ticket.ticket_binding_sha256,
            labels=labels,
            protocol_row_ids=protocol,
            class_tokens=tokens,
            _origin=origin,
            _seal=_ACTIVATED_SEAL,
        )
    except BaseException as error:
        if begun and consumed_ticket is not None:
            _prelabel._finish_attempt_ticket_consumption(  # noqa: SLF001
                consumed_ticket, succeeded=False
            )
        if isinstance(error, HarmBenchSelectionLabelError):
            raise
        if isinstance(error, Exception):
            raise HarmBenchSelectionLabelError(
                "attempt-bound label activation failed terminally"
            ) from error
        raise
    _prelabel._finish_attempt_ticket_consumption(  # noqa: SLF001
        consumed_ticket, succeeded=True
    )
    return capability


def _revalidate_activated_selection_labels(
    capability: object,
) -> ActivatedSelectionLabelCapability:
    descriptor: tuple[object, ...] = ()
    if (
        type(capability) is ActivatedSelectionLabelCapability
        and type(capability._origin) is _ActivatedOrigin
    ):
        descriptor = capability._origin.manifest_descriptor
    if (
        type(capability) is not ActivatedSelectionLabelCapability
        or capability._seal is not _ACTIVATED_SEAL
        or type(capability._origin) is not _ActivatedOrigin
        or capability._origin._seal is not _ACTIVATED_ORIGIN_SEAL
        or len(descriptor) != 10
        or capability.dataset_id != descriptor[2]
        or capability.role != descriptor[3]
        or capability.rows != descriptor[4]
        or capability.ordered_protocol_row_alignment_sha256 != descriptor[5]
        or capability.class_order_sha256 != descriptor[6]
        or capability.artifact_file_sha256 != descriptor[8]
        or capability.manifest_file_sha256 != descriptor[9]
        or capability.labels.flags.writeable
        or capability.protocol_row_ids.flags.writeable
        or capability.class_tokens.flags.writeable
        or _array_sha256(capability.labels) != capability._origin.labels_sha256
        or _array_sha256(capability.protocol_row_ids)
        != capability._origin.protocol_row_ids_sha256
        or _array_sha256(capability.class_tokens)
        != capability._origin.class_tokens_sha256
        or capability.protocol_canonical_sha256
        != capability._origin.protocol_canonical_sha256
        or capability.attempt_marker_file_sha256
        != capability._origin.attempt_marker_file_sha256
        or capability.prelabel_bundle_file_sha256
        != capability._origin.prelabel_bundle_file_sha256
        or capability.prelabel_receipt_file_sha256
        != capability._origin.prelabel_receipt_file_sha256
        or capability.ticket_binding_sha256
        != capability._origin.ticket_binding_sha256
    ):
        raise HarmBenchSelectionLabelError(
            "loader-minted activated selection-label capability is required"
        )
    from . import harmbench_erc_selection_prelabel as _prelabel

    try:
        ticket = _prelabel._validate_consumed_ticket_for_capability(  # noqa: SLF001
            capability._origin._ticket,
            attempt=capability._origin._ticket._attempt,
            dataset_id=capability.dataset_id,
            ticket_binding_sha256=capability.ticket_binding_sha256,
        )
    except (TypeError, ValueError, AttributeError) as error:
        raise HarmBenchSelectionLabelError(
            "activated label lost its consumed attempt-ticket origin"
        ) from error
    if (
        capability.protocol_canonical_sha256
        != ticket.protocol_canonical_sha256
        or capability.attempt_marker_file_sha256 != ticket.marker_file_sha256
        or capability.prelabel_bundle_file_sha256
        != ticket.prelabel_bundle_file_sha256
        or capability.prelabel_receipt_file_sha256
        != ticket.prelabel_receipt_file_sha256
    ):
        raise HarmBenchSelectionLabelError(
            "activated label attempt lineage changed"
        )
    return capability


def _string_scalar(value: str) -> np.ndarray:
    return np.asarray(value, dtype=np.str_)


def _artifact_arrays(
    *,
    dataset_id: str,
    labels: np.ndarray,
    protocol_row_ids: np.ndarray,
    class_tokens: np.ndarray,
    class_order_sha256: str,
) -> dict[str, np.ndarray]:
    row_sha = _array_sha256(protocol_row_ids)
    return {
        "schema_version": _string_scalar(SELECTION_LABEL_ARTIFACT_SCHEMA),
        "dataset_id": _string_scalar(dataset_id),
        "role": _string_scalar(SELECTION_LABEL_ROLE),
        "rows": np.asarray(len(protocol_row_ids), dtype=np.int64),
        "ordered_protocol_row_alignment_sha256": _string_scalar(row_sha),
        "class_order_sha256": _string_scalar(class_order_sha256),
        "labels": np.asarray(labels, dtype=np.int64),
        "protocol_row_ids": np.asarray(protocol_row_ids, dtype=np.int64),
        "class_tokens": np.asarray(class_tokens, dtype=np.str_),
    }


def _temporary_file(root: Path, destination: Path) -> tuple[Any, Path]:
    handle = tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=root,
        delete=False,
    )
    return handle, Path(handle.name)


def _publish_once(temporary: Path, destination: Path) -> None:
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


def _publish_trusted_synthetic_selection_labels(
    *,
    private_root: str | Path,
    dataset_id: str,
    labels: object,
    protocol_row_ids: object,
    class_tokens: Sequence[str],
    class_order_sha256: str,
) -> dict[str, object]:
    """Trusted synthetic-only publisher used by contract tests and dry runs.

    This helper is intentionally private and absent from ``__all__``.  Real
    sidecar publication belongs to a separate, explicitly authorized ingest
    boundary and must not be routed through a function named synthetic.
    """

    root = _validate_private_root(private_root)
    dataset = _identifier(dataset_id, name="dataset_id")
    protocol = _protocol_row_ids(protocol_row_ids, name="protocol_row_ids")
    tokens = _class_tokens(class_tokens, name="class_tokens")
    class_sha = _sha256(class_order_sha256, name="class_order_sha256")
    label_values = _label_vector(labels, rows=len(protocol), classes=len(tokens))
    artifact_path = root / SELECTION_LABEL_ARTIFACT_FILENAME
    manifest_path = root / SELECTION_LABEL_MANIFEST_FILENAME
    for path in (artifact_path, manifest_path):
        _reject_reparse_components(path)
        if os.path.lexists(path):
            raise FileExistsError(
                f"write-once destination already exists: {path.name}"
            )

    arrays = _artifact_arrays(
        dataset_id=dataset,
        labels=label_values,
        protocol_row_ids=protocol,
        class_tokens=tokens,
        class_order_sha256=class_sha,
    )
    artifact_handle, artifact_temporary_value = _temporary_file(root, artifact_path)
    artifact_temporary: Path | None = artifact_temporary_value
    manifest_temporary: Path | None = None
    artifact_published_identity: os.stat_result | None = None
    try:
        with artifact_handle:
            np.savez_compressed(artifact_handle, **arrays)
            artifact_handle.flush()
            os.fsync(artifact_handle.fileno())
            if artifact_handle.tell() > MAX_SELECTION_LABEL_ARTIFACT_BYTES:
                raise HarmBenchSelectionLabelError(
                    "selection-label NPZ exceeds compressed byte budget"
                )
            artifact_handle.seek(0)
            artifact_sha = _hash_open_handle(artifact_handle)
        manifest = _validate_manifest(
            {
                "schema_version": SELECTION_LABEL_MANIFEST_SCHEMA,
                "artifact_schema_version": SELECTION_LABEL_ARTIFACT_SCHEMA,
                "dataset_id": dataset,
                "role": SELECTION_LABEL_ROLE,
                "rows": len(protocol),
                "ordered_protocol_row_alignment_sha256": _array_sha256(protocol),
                "class_order_sha256": class_sha,
                "artifact_filename": SELECTION_LABEL_ARTIFACT_FILENAME,
                "artifact_file_sha256": artifact_sha,
            }
        )
        encoded_manifest = _canonical_json_bytes(manifest)
        if len(encoded_manifest) > MAX_SELECTION_LABEL_MANIFEST_BYTES:
            raise HarmBenchSelectionLabelError(
                "selection-label manifest exceeds byte budget"
            )
        manifest_handle, manifest_temporary = _temporary_file(root, manifest_path)
        with manifest_handle:
            manifest_handle.write(encoded_manifest)
            manifest_handle.flush()
            os.fsync(manifest_handle.fileno())
        assert artifact_temporary is not None
        _publish_once(artifact_temporary, artifact_path)
        artifact_temporary = None
        artifact_published_identity = _plain_file_stat(
            artifact_path, name="selection-label NPZ"
        )
        assert manifest_temporary is not None
        try:
            _publish_once(manifest_temporary, manifest_path)
            manifest_temporary = None
        except BaseException:
            try:
                if _same_file_identity(
                    _plain_file_stat(artifact_path, name="selection-label NPZ"),
                    artifact_published_identity,
                ):
                    artifact_path.unlink()
            except (OSError, HarmBenchSelectionLabelError):
                pass
            raise
    finally:
        for temporary in (artifact_temporary, manifest_temporary):
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
    return dict(manifest)


__all__ = [
    "ActivatedSelectionLabelCapability",
    "HarmBenchSelectionLabelError",
    "MAX_SELECTION_LABEL_ARTIFACT_BYTES",
    "MAX_SELECTION_LABEL_CLASSES",
    "MAX_SELECTION_LABEL_ELEMENTS",
    "MAX_SELECTION_LABEL_MANIFEST_BYTES",
    "MAX_SELECTION_LABEL_ROWS",
    "SELECTION_LABEL_ARTIFACT_FILENAME",
    "SELECTION_LABEL_ARTIFACT_SCHEMA",
    "SELECTION_LABEL_MANIFEST_FILENAME",
    "SELECTION_LABEL_MANIFEST_SCHEMA",
    "SELECTION_LABEL_ROLE",
    "SelectionLabelManifestMetadata",
    "load_selection_label_manifest_metadata",
    "selection_label_manifest_sha256",
    "selection_protocol_row_alignment_sha256",
]
