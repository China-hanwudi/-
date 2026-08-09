"""Write-once, repository-external cache for shared HarmBench processors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Mapping

import joblib

from .harmbench_erc_processors import (
    FROZEN_PROCESSOR_SPEC,
    SharedProcessor,
    validate_shared_processor,
)


class HarmBenchProcessorCacheError(ValueError):
    """Raised when a private processor cache is missing or not hash-bound."""


CACHE_SCHEMA = "harmbench_erc_private_processor_cache_v1"
RECEIPT_FILENAME = "processor_cache_receipt.json"
PAYLOAD_FILENAME = "processor.joblib"
SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
RECEIPT_KEYS = {
    "schema_version",
    "dataset_id",
    "seed",
    "fold",
    "protocol_sha256",
    "source_snapshot_sha256",
    "source_capability_sha256",
    "crossfit_plan_sha256",
    "processor_spec_sha256",
    "processor_receipt_sha256",
    "serialized_payload_sha256",
    "serialization",
    "contains_labels_or_outcomes",
    "contains_private_absolute_paths",
    "write_once",
    "cache_binding_sha256",
}


@dataclass(frozen=True)
class ProcessorCacheReceipt:
    schema_version: str
    dataset_id: str
    seed: int
    fold: int
    protocol_sha256: str
    source_snapshot_sha256: str
    source_capability_sha256: str
    crossfit_plan_sha256: str
    processor_spec_sha256: str
    processor_receipt_sha256: str
    serialized_payload_sha256: str
    serialization: str
    contains_labels_or_outcomes: bool
    contains_private_absolute_paths: bool
    write_once: bool
    cache_binding_sha256: str


def _sha256(value: object, *, name: str) -> str:
    text = str(value)
    if not SHA_PATTERN.fullmatch(text):
        raise HarmBenchProcessorCacheError(f"{name} must be a lowercase SHA-256")
    return text


def _identifier(value: object, *, name: str) -> str:
    text = str(value)
    if not IDENTIFIER_PATTERN.fullmatch(text):
        raise HarmBenchProcessorCacheError(f"{name} must be a short opaque identifier")
    return text


def _exact_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HarmBenchProcessorCacheError(f"{name} must be a non-negative exact integer")
    return int(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_handle_sha256(handle: object) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise HarmBenchProcessorCacheError(
                f"processor cache receipt contains duplicate JSON key: {key}"
            )
        value[key] = item
    return value


def _binding_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _repository_root() -> Path:
    root = Path(__file__).resolve().parents[3]
    if not (root / ".git").exists():
        raise HarmBenchProcessorCacheError(
            "processor cache module is not inside the expected Git repository"
        )
    return root.resolve(strict=True)


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _reject_reparse_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    for candidate in (absolute, *absolute.parents):
        if os.path.lexists(candidate) and _is_reparse_or_symlink(candidate):
            raise HarmBenchProcessorCacheError(
                "processor cache path contains a symlink or reparse point"
            )


def _directory_identity(path: Path) -> tuple[int, int, int]:
    _reject_reparse_components(path)
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise HarmBenchProcessorCacheError(
            f"cannot stat processor cache directory: {error}"
        ) from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise HarmBenchProcessorCacheError("processor cache parent is not a directory")
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(getattr(metadata, "st_file_attributes", 0)),
    )


def _outside_repository(path: Path) -> None:
    _reject_reparse_components(path)
    target = path.resolve()
    repository = _repository_root()
    try:
        common = Path(os.path.commonpath((str(target), str(repository))))
    except ValueError:
        return
    if common == repository:
        raise HarmBenchProcessorCacheError("processor cache must be outside repository")


def _receipt_descriptor(
    *,
    processor: SharedProcessor,
    protocol_sha256: str,
    source_snapshot_sha256: str,
    source_capability_sha256: str,
    payload_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": CACHE_SCHEMA,
        "dataset_id": _identifier(
            processor.receipt.dataset_id, name="dataset_id"
        ),
        "seed": _exact_integer(processor.receipt.seed, name="seed"),
        "fold": _exact_integer(processor.receipt.fold, name="fold"),
        "protocol_sha256": _sha256(protocol_sha256, name="protocol_sha256"),
        "source_snapshot_sha256": _sha256(
            source_snapshot_sha256, name="source_snapshot_sha256"
        ),
        "source_capability_sha256": _sha256(
            source_capability_sha256, name="source_capability_sha256"
        ),
        "crossfit_plan_sha256": _sha256(
            processor.receipt.crossfit_plan_sha256,
            name="crossfit_plan_sha256",
        ),
        "processor_spec_sha256": _sha256(
            processor.spec.canonical_sha256, name="processor_spec_sha256"
        ),
        "processor_receipt_sha256": _sha256(
            processor.receipt.processor_receipt_sha256,
            name="processor_receipt_sha256",
        ),
        "serialized_payload_sha256": _sha256(
            payload_sha256, name="serialized_payload_sha256"
        ),
        "serialization": "joblib_gzip3_protocol5_private_trusted_cache",
        "contains_labels_or_outcomes": False,
        "contains_private_absolute_paths": False,
        "write_once": True,
    }


def _validate_receipt_payload(value: object) -> ProcessorCacheReceipt:
    if not isinstance(value, Mapping) or set(value) != RECEIPT_KEYS:
        raise HarmBenchProcessorCacheError("processor cache receipt schema changed")
    descriptor = {key: value[key] for key in RECEIPT_KEYS - {"cache_binding_sha256"}}
    if value["schema_version"] != CACHE_SCHEMA:
        raise HarmBenchProcessorCacheError("processor cache schema changed")
    if value["serialization"] != "joblib_gzip3_protocol5_private_trusted_cache":
        raise HarmBenchProcessorCacheError("processor cache serialization changed")
    if (
        value["contains_labels_or_outcomes"] is not False
        or value["contains_private_absolute_paths"] is not False
        or value["write_once"] is not True
    ):
        raise HarmBenchProcessorCacheError("processor cache privacy/write-once contract changed")
    normalized = {
        **descriptor,
        "dataset_id": _identifier(value["dataset_id"], name="dataset_id"),
        "seed": _exact_integer(value["seed"], name="seed"),
        "fold": _exact_integer(value["fold"], name="fold"),
    }
    for name in (
        "protocol_sha256",
        "source_snapshot_sha256",
        "source_capability_sha256",
        "crossfit_plan_sha256",
        "processor_spec_sha256",
        "processor_receipt_sha256",
        "serialized_payload_sha256",
    ):
        normalized[name] = _sha256(value[name], name=name)
    binding = _sha256(value["cache_binding_sha256"], name="cache_binding_sha256")
    if binding != _binding_sha256(normalized):
        raise HarmBenchProcessorCacheError("processor cache receipt binding changed")
    return ProcessorCacheReceipt(**normalized, cache_binding_sha256=binding)


def _receipt_file_bytes(receipt: ProcessorCacheReceipt) -> bytes:
    if not isinstance(receipt, ProcessorCacheReceipt):
        raise HarmBenchProcessorCacheError(
            "receipt must be a ProcessorCacheReceipt"
        )
    normalized = _validate_receipt_payload(asdict(receipt))
    return _canonical_json_bytes(asdict(normalized)) + b"\n"


def processor_cache_receipt_file_sha256(
    receipt: ProcessorCacheReceipt,
) -> str:
    """Return the out-of-band SHA-256 of the exact canonical receipt file."""

    return hashlib.sha256(_receipt_file_bytes(receipt)).hexdigest()


def write_shared_processor_cache(
    processor: SharedProcessor,
    *,
    target_directory: Path,
    protocol_sha256: str,
    source_snapshot_sha256: str,
    source_capability_sha256: str,
) -> ProcessorCacheReceipt:
    """Atomically publish one private cache directory; never overwrite it."""

    if not isinstance(processor, SharedProcessor):
        raise HarmBenchProcessorCacheError("processor must be a SharedProcessor")
    if processor.spec != FROZEN_PROCESSOR_SPEC:
        raise HarmBenchProcessorCacheError("processor spec is not frozen")
    expected_source_capability_sha = _sha256(
        source_capability_sha256,
        name="source_capability_sha256",
    )
    try:
        processor = validate_shared_processor(
            processor,
            expected_processor_receipt_sha256=(
                processor.receipt.processor_receipt_sha256
            ),
            expected_fit_feature_capability_sha256=(
                expected_source_capability_sha
            ),
            expected_crossfit_plan_sha256=(
                processor.receipt.crossfit_plan_sha256
            ),
            expected_seed=processor.receipt.seed,
            expected_fold=processor.receipt.fold,
        )
    except ValueError as error:
        raise HarmBenchProcessorCacheError(
            f"processor failed live validation before cache write: {error}"
        ) from error
    target = Path(target_directory)
    _outside_repository(target)
    if os.path.lexists(target):
        raise FileExistsError(f"processor cache target already exists: {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    parent_identity = _directory_identity(target.parent)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        payload_path = temporary / PAYLOAD_FILENAME
        joblib.dump(processor, payload_path, compress=("gzip", 3), protocol=5)
        payload_sha = _file_sha256(payload_path)
        descriptor = _receipt_descriptor(
            processor=processor,
            protocol_sha256=protocol_sha256,
            source_snapshot_sha256=source_snapshot_sha256,
            source_capability_sha256=expected_source_capability_sha,
            payload_sha256=payload_sha,
        )
        receipt_payload = {
            **descriptor,
            "cache_binding_sha256": _binding_sha256(descriptor),
        }
        receipt = _validate_receipt_payload(receipt_payload)
        receipt_path = temporary / RECEIPT_FILENAME
        receipt_path.write_bytes(_receipt_file_bytes(receipt))
        if _directory_identity(target.parent) != parent_identity:
            raise HarmBenchProcessorCacheError(
                "processor cache parent identity changed before publish"
            )
        if os.path.lexists(target):
            raise FileExistsError(
                f"processor cache target already exists: {target.name}"
            )
        os.rename(temporary, target)
        return receipt
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def load_shared_processor_cache(
    *,
    target_directory: Path,
    expected_protocol_sha256: str,
    expected_source_snapshot_sha256: str,
    expected_source_capability_sha256: str,
    expected_crossfit_plan_sha256: str,
    expected_processor_receipt_sha256: str,
    expected_cache_receipt_sha256: str,
    expected_serialized_payload_sha256: str,
) -> tuple[SharedProcessor, ProcessorCacheReceipt]:
    """Verify every external binding before loading the private trusted pickle."""

    expected_receipt_file_sha = _sha256(
        expected_cache_receipt_sha256,
        name="expected_cache_receipt_sha256",
    )
    expected_payload_sha = _sha256(
        expected_serialized_payload_sha256,
        name="expected_serialized_payload_sha256",
    )
    target = Path(target_directory)
    _outside_repository(target)
    target_identity = _directory_identity(target)
    children = tuple(target.iterdir())
    if (
        set(path.name for path in children)
        != {
        PAYLOAD_FILENAME,
        RECEIPT_FILENAME,
        }
        or any(_is_reparse_or_symlink(path) or not path.is_file() for path in children)
    ):
        raise HarmBenchProcessorCacheError("processor cache directory schema changed")
    receipt_path = target / RECEIPT_FILENAME
    try:
        # Bind the bytes before parsing them, then continue through the same
        # open handle so a path replacement cannot change the validated file.
        with receipt_path.open("rb") as handle:
            if _open_handle_sha256(handle) != expected_receipt_file_sha:
                raise HarmBenchProcessorCacheError(
                    "processor cache receipt file SHA changed"
                )
            handle.seek(0)
            receipt_bytes = handle.read()
    except HarmBenchProcessorCacheError:
        raise
    except OSError as error:
        raise HarmBenchProcessorCacheError(
            f"cannot read processor cache receipt: {error}"
        ) from error
    try:
        raw = json.loads(
            receipt_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except HarmBenchProcessorCacheError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HarmBenchProcessorCacheError(
            f"cannot decode processor cache receipt: {error}"
        ) from error
    if not isinstance(raw, Mapping) or receipt_bytes != (
        _canonical_json_bytes(raw) + b"\n"
    ):
        raise HarmBenchProcessorCacheError(
            "processor cache receipt is not canonical JSON plus LF"
        )
    receipt = _validate_receipt_payload(raw)
    if receipt.serialized_payload_sha256 != expected_payload_sha:
        raise HarmBenchProcessorCacheError(
            "processor cache receipt payload SHA differs from external expected SHA"
        )
    expected = {
        "protocol_sha256": _sha256(
            expected_protocol_sha256, name="expected_protocol_sha256"
        ),
        "source_snapshot_sha256": _sha256(
            expected_source_snapshot_sha256,
            name="expected_source_snapshot_sha256",
        ),
        "source_capability_sha256": _sha256(
            expected_source_capability_sha256,
            name="expected_source_capability_sha256",
        ),
        "crossfit_plan_sha256": _sha256(
            expected_crossfit_plan_sha256,
            name="expected_crossfit_plan_sha256",
        ),
        "processor_receipt_sha256": _sha256(
            expected_processor_receipt_sha256,
            name="expected_processor_receipt_sha256",
        ),
    }
    if any(getattr(receipt, name) != value for name, value in expected.items()):
        raise HarmBenchProcessorCacheError("processor cache external binding changed")
    payload_path = target / PAYLOAD_FILENAME
    try:
        # Hash and deserialize through the same open handle.  On POSIX this
        # remains the same inode if the path is replaced; on Windows the open
        # handle also prevents a rename race.
        with payload_path.open("rb") as handle:
            if _open_handle_sha256(handle) != expected_payload_sha:
                raise HarmBenchProcessorCacheError("processor cache payload SHA changed")
            handle.seek(0)
            processor = joblib.load(handle)
    except HarmBenchProcessorCacheError:
        raise
    except Exception as error:
        raise HarmBenchProcessorCacheError(f"trusted processor cache could not load: {error}") from error
    if not isinstance(processor, SharedProcessor):
        raise HarmBenchProcessorCacheError("processor cache payload type changed")
    try:
        processor = validate_shared_processor(
            processor,
            expected_processor_receipt_sha256=(
                receipt.processor_receipt_sha256
            ),
            expected_fit_feature_capability_sha256=(
                receipt.source_capability_sha256
            ),
            expected_crossfit_plan_sha256=receipt.crossfit_plan_sha256,
            expected_seed=receipt.seed,
            expected_fold=receipt.fold,
        )
    except ValueError as error:
        raise HarmBenchProcessorCacheError(
            f"loaded processor failed live validation: {error}"
        ) from error
    if (
        processor.spec.canonical_sha256 != receipt.processor_spec_sha256
        or processor.receipt.dataset_id != receipt.dataset_id
    ):
        raise HarmBenchProcessorCacheError("loaded processor differs from cache receipt")
    if _directory_identity(target) != target_identity:
        raise HarmBenchProcessorCacheError(
            "processor cache directory identity changed during load"
        )
    return processor, receipt


__all__ = [
    "CACHE_SCHEMA",
    "HarmBenchProcessorCacheError",
    "ProcessorCacheReceipt",
    "load_shared_processor_cache",
    "processor_cache_receipt_file_sha256",
    "write_shared_processor_cache",
]
