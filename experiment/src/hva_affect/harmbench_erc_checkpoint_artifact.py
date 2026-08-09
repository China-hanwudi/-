"""Write-once, typed checkpoint artifacts for HarmBench-ERC.

The public writer accepts only a production checkpoint wrapper.  Every lineage
digest is copied from that typed wrapper; there is deliberately no public API
that accepts a checkpoint, processor, context, or class-order digest.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from io import BytesIO
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from types import MappingProxyType
from typing import Mapping
import zipfile

import numpy as np
from sklearn.linear_model import SGDClassifier
import torch

from .harmbench_erc_contract import EXPECTED_TRAINING_SEEDS
from .harmbench_erc_contexts import (
    CURRENT_ONLY_STRATEGY_ID,
    FIT_TRAIN_CONTEXT_ROLE,
    STRICT_PAST_STRATEGY_IDS,
    StrictPastContextRoster,
    validate_strict_past_context_roster,
)
from .harmbench_erc_crossfit import (
    ContextTrainingExamples,
    EXPECTED_OUTER_FOLDS,
    SharedGroupCrossfitPlan,
    resolve_shared_group_crossfit_indices,
    validate_context_training_examples,
    validate_shared_group_crossfit_plan,
)
from .harmbench_erc_models import (
    AUDIO_DIMENSION,
    CAUSAL_GRU_ID,
    CAUSAL_GRU_PARAMETER_LIMIT,
    CURRENT_ONLY_NAMESPACE,
    DEEPSETS_PARAMETER_LIMIT,
    DEEPSETS_POOL_ID,
    FROZEN_MODEL_IDS,
    GRU_HIDDEN_DIMENSION,
    HISTORY_NAMESPACE,
    ITEM_HIDDEN_DIMENSION,
    LINEAR_POOL_ID,
    LINEAR_HISTORY_SUMMARY_DIMENSION,
    PROJECTION_DIMENSION,
    QUERY_DIMENSION,
    TEXT_DIMENSION,
    VIDEO_DIMENSION,
    CausalGRUCurrentOnlyCheckpoint,
    CausalGRUHistoryCheckpoint,
    DeepSetsCurrentOnlyCheckpoint,
    DeepSetsHistoryCheckpoint,
    LinearCurrentOnlyCheckpoint,
    LinearHistoryCheckpoint,
    ProductionCurrentOnlyCheckpoint,
    ProductionHistoryCheckpoint,
    _CausalGRUCurrentOnlyNetwork,
    _CausalGRUHistoryNetwork,
    _DeepSetsCurrentOnlyNetwork,
    _DeepSetsHistoryNetwork,
    _initialized_network,
    aggregate_context_roster_sha256,
)
from .harmbench_erc_open_roles import FitRoleCapability, validate_fit_role_capability
from .harmbench_erc_processors import (
    ProcessedRoleEmbeddings,
    SharedProcessor,
    validate_processed_role_embeddings,
    validate_shared_processor,
)


class HarmBenchCheckpointArtifactError(ValueError):
    """Raised when an artifact is mutable, ambiguous, or not provenance-bound."""


class _PredictionOnlySGDClassifier(SGDClassifier):
    """An SGDClassifier state carrier that cannot be resumed or refitted."""

    def fit(self, *args: object, **kwargs: object) -> object:  # pragma: no cover - guard
        del args, kwargs
        raise HarmBenchCheckpointArtifactError(
            "restored checkpoints are prediction-only and cannot be fitted"
        )

    def partial_fit(
        self, *args: object, **kwargs: object
    ) -> object:  # pragma: no cover - guard
        del args, kwargs
        raise HarmBenchCheckpointArtifactError(
            "restored checkpoints are prediction-only and cannot be resumed"
        )


ARTIFACT_RECEIPT_SCHEMA = "harmbench_erc_checkpoint_artifact_receipt_v1"
ARTIFACT_PAYLOAD_SCHEMA = "harmbench_erc_checkpoint_npz_v1"
SHA256_LENGTH = 64
_ARCHITECTURE_STORAGE_KEY = "__architecture_json_utf8__"
_VERIFICATION_TOKEN = object()

# These are serialized-byte ceilings, not training parameter budgets.  They
# are deliberately comfortably above the one exact frozen architecture in
# each family while still preventing a hostile receipt/NPZ from turning a
# checkpoint verification into an unbounded allocation.
MAX_RECEIPT_BYTES = 1 * 1024 * 1024
MAX_CLASS_COUNT = 64
FAMILY_PAYLOAD_BYTE_LIMITS: Mapping[str, int] = MappingProxyType(
    {
        LINEAR_POOL_ID: 2 * 1024 * 1024,
        DEEPSETS_POOL_ID: 8 * 1024 * 1024,
        CAUSAL_GRU_ID: 16 * 1024 * 1024,
    }
)
FAMILY_TOTAL_ELEMENT_LIMITS: Mapping[str, int] = MappingProxyType(
    {
        LINEAR_POOL_ID: 250_000,
        DEEPSETS_POOL_ID: DEEPSETS_PARAMETER_LIMIT,
        CAUSAL_GRU_ID: CAUSAL_GRU_PARAMETER_LIMIT,
    }
)


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_LENGTH
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HarmBenchCheckpointArtifactError(f"{name} must be a lowercase SHA-256")
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
        raise HarmBenchCheckpointArtifactError(
            f"checkpoint artifact metadata is not canonical JSON: {error}"
        ) from error


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _class_order_sha256(
    class_order: tuple[str, ...], *, dataset_id: str, fit_sha256: str
) -> str:
    return _canonical_json_sha256(
        {
            "schema_version": "harmbench_erc_checkpoint_class_order_v1",
            "dataset_id": dataset_id,
            "fit_training_capability_sha256": fit_sha256,
            "ordered_class_tokens": list(class_order),
        }
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HarmBenchCheckpointArtifactError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise HarmBenchCheckpointArtifactError(f"non-finite JSON constant is forbidden: {value}")


def _parse_canonical_json(raw: bytes) -> object:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except HarmBenchCheckpointArtifactError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise HarmBenchCheckpointArtifactError("cannot decode artifact receipt") from error
    if raw != _canonical_json_bytes(value):
        raise HarmBenchCheckpointArtifactError("artifact receipt must use canonical JSON bytes")
    return value


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse)


def _reject_reparse_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    for candidate in (absolute, *absolute.parents):
        if os.path.lexists(candidate) and _is_reparse_or_symlink(candidate):
            raise HarmBenchCheckpointArtifactError(
                "checkpoint artifact path contains a symlink or reparse point"
            )


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(getattr(metadata, "st_mtime_ns", 0)),
    )


def _directory_identity(path: Path) -> tuple[int, int, int]:
    _reject_reparse_components(path)
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise HarmBenchCheckpointArtifactError("cannot stat private artifact root") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise HarmBenchCheckpointArtifactError("private artifact root is not a directory")
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(getattr(metadata, "st_file_attributes", 0)),
    )


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _private_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.is_absolute():
        raise HarmBenchCheckpointArtifactError("private artifact root must be absolute")
    _reject_reparse_components(root)
    if not root.exists() or not root.is_dir():
        raise HarmBenchCheckpointArtifactError(
            "private artifact root must be an existing directory"
        )
    resolved = root.resolve(strict=True)
    repository = _repository_root()
    if resolved == repository or repository in resolved.parents:
        raise HarmBenchCheckpointArtifactError(
            "private artifact root must be outside the repository"
        )
    _directory_identity(resolved)
    return resolved


def _plain_file(path: Path, *, suffix: str, must_exist: bool) -> Path:
    if not path.is_absolute() or path.suffix.lower() != suffix:
        raise HarmBenchCheckpointArtifactError(f"artifact path must be an absolute {suffix} path")
    _reject_reparse_components(path)
    if must_exist:
        try:
            metadata = os.lstat(path)
        except OSError as error:
            raise HarmBenchCheckpointArtifactError("cannot stat checkpoint artifact") from error
        if not stat.S_ISREG(metadata.st_mode) or _is_reparse_or_symlink(path):
            raise HarmBenchCheckpointArtifactError(
                "checkpoint artifact must be a plain non-reparse file"
            )
    return path


def _read_verified_bytes(
    path: Path,
    *,
    expected_sha256: str,
    max_bytes: int | None = None,
) -> bytes:
    expected = _sha256(expected_sha256, name="expected file SHA-256")
    source = _plain_file(path, suffix=path.suffix.lower(), must_exist=True)
    parent_identity = _directory_identity(source.parent)
    before_path = os.lstat(source)
    if max_bytes is not None and (
        type(max_bytes) is not int
        or max_bytes < 1
        or int(before_path.st_size) > max_bytes
    ):
        raise HarmBenchCheckpointArtifactError(
            "checkpoint artifact exceeds its frozen byte budget"
        )
    try:
        with source.open("rb") as handle:
            before_handle = os.fstat(handle.fileno())
            if _identity(before_path) != _identity(before_handle):
                raise HarmBenchCheckpointArtifactError(
                    "checkpoint artifact changed before verified read"
                )
            if max_bytes is not None and int(before_handle.st_size) > max_bytes:
                raise HarmBenchCheckpointArtifactError(
                    "checkpoint artifact exceeds its frozen byte budget"
                )
            first = handle.read()
            if max_bytes is not None and len(first) > max_bytes:
                raise HarmBenchCheckpointArtifactError(
                    "checkpoint artifact exceeds its frozen byte budget"
                )
            first_sha = hashlib.sha256(first).hexdigest()
            handle.seek(0)
            second = handle.read()
            second_sha = hashlib.sha256(second).hexdigest()
            after_handle = os.fstat(handle.fileno())
    except HarmBenchCheckpointArtifactError:
        raise
    except OSError as error:
        raise HarmBenchCheckpointArtifactError("cannot read checkpoint artifact") from error
    if first_sha != expected or second_sha != expected or first != second:
        raise HarmBenchCheckpointArtifactError(
            "checkpoint artifact file differs from external binding"
        )
    if _identity(before_handle) != _identity(after_handle):
        raise HarmBenchCheckpointArtifactError(
            "checkpoint artifact changed during verified read"
        )
    _reject_reparse_components(source)
    if _identity(os.lstat(source)) != _identity(after_handle):
        raise HarmBenchCheckpointArtifactError(
            "checkpoint artifact path changed during verified read"
        )
    if _directory_identity(source.parent) != parent_identity:
        raise HarmBenchCheckpointArtifactError(
            "checkpoint artifact directory changed during verified read"
        )
    return second


def _write_once(path: Path, raw: bytes) -> str:
    _plain_file(path, suffix=path.suffix.lower(), must_exist=False)
    if os.path.lexists(path):
        raise FileExistsError(f"write-once checkpoint artifact already exists: {path.name}")
    parent_identity = _directory_identity(path.parent)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _reject_reparse_components(temporary)
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(
                f"write-once checkpoint artifact already exists: {path.name}"
            ) from None
        except OSError as error:
            if error.errno == errno.EEXIST or os.path.lexists(path):
                raise FileExistsError(
                    f"write-once checkpoint artifact already exists: {path.name}"
                ) from None
            raise HarmBenchCheckpointArtifactError(
                f"cannot publish checkpoint artifact: {error}"
            ) from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    if _directory_identity(path.parent) != parent_identity:
        raise HarmBenchCheckpointArtifactError(
            "checkpoint artifact directory changed during publication"
        )
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError as error:
        raise HarmBenchCheckpointArtifactError("cannot make checkpoint artifact private") from error
    expected = hashlib.sha256(raw).hexdigest()
    if _read_verified_bytes(path, expected_sha256=expected) != raw:
        raise HarmBenchCheckpointArtifactError("published checkpoint artifact changed")
    return expected


def _array_snapshot(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind not in "biufc":
        raise HarmBenchCheckpointArtifactError(f"{name} is not a pickle-free numeric array")
    if array.dtype.kind in "fc" and not np.isfinite(array).all():
        raise HarmBenchCheckpointArtifactError(f"{name} contains non-finite values")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = BytesIO()
    np.lib.format.write_array(stream, array, allow_pickle=False)
    return stream.getvalue()


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    stream = BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for key in sorted(arrays):
            if not key or "/" in key or "\\" in key or key.endswith(".npy"):
                raise HarmBenchCheckpointArtifactError("unsafe NPZ storage key")
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(arrays[key]))
    return stream.getvalue()


def _parameter_manifest(
    named_arrays: list[tuple[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    stored: dict[str, np.ndarray] = {}
    manifest: list[dict[str, object]] = []
    for index, (name, value) in enumerate(named_arrays):
        storage_key = f"p{index:04d}"
        array = _array_snapshot(value, name=name)
        stored[storage_key] = array
        manifest.append(
            {
                "name": name,
                "storage_key": storage_key,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            }
        )
    return stored, manifest


def _extract_payload(checkpoint: object) -> tuple[dict[str, np.ndarray], str]:
    low_level = checkpoint.checkpoint
    common: dict[str, object] = {
        "schema_version": ARTIFACT_PAYLOAD_SCHEMA,
        "family_id": checkpoint.model_id,
        "model_namespace": checkpoint.model_namespace,
        "num_classes": len(checkpoint.class_order),
        "seed": int(checkpoint.training_seed),
        "input_dimensions": {
            "text": TEXT_DIMENSION,
            "audio": AUDIO_DIMENSION,
            "video": VIDEO_DIMENSION,
        },
    }
    if isinstance(low_level, (LinearHistoryCheckpoint, LinearCurrentOnlyCheckpoint)):
        estimator = low_level._estimator
        arrays, parameters = _parameter_manifest(
            [
                ("coef_", estimator.coef_),
                ("intercept_", estimator.intercept_),
                ("classes_", estimator.classes_),
                ("t_", np.asarray(estimator.t_)),
                ("n_iter_", np.asarray(estimator.n_iter_)),
                ("n_features_in_", np.asarray(estimator.n_features_in_)),
            ]
        )
        architecture = {
            **common,
            "implementation": "sklearn.linear_model.SGDClassifier",
            "frozen_estimator": {
                "loss": "log_loss",
                "penalty": "l2",
                "alpha": 1e-4,
                "fit_intercept": True,
                "shuffle": False,
                "average": False,
            },
            "parameters": parameters,
        }
    elif isinstance(
        low_level,
        (
            DeepSetsHistoryCheckpoint,
            DeepSetsCurrentOnlyCheckpoint,
            CausalGRUHistoryCheckpoint,
            CausalGRUCurrentOnlyCheckpoint,
        ),
    ):
        network = low_level._network
        state = network.state_dict()
        arrays, parameters = _parameter_manifest(
            [
                (name, tensor.detach().cpu().numpy())
                for name, tensor in sorted(state.items())
            ]
        )
        architecture = {
            **common,
            "implementation": type(network).__name__,
            "projection_dimension": PROJECTION_DIMENSION,
            "item_hidden_dimension": (
                ITEM_HIDDEN_DIMENSION if checkpoint.model_id == DEEPSETS_POOL_ID else None
            ),
            "gru_hidden_dimension": (
                GRU_HIDDEN_DIMENSION if checkpoint.model_id == CAUSAL_GRU_ID else None
            ),
            "gru_num_layers": 1 if checkpoint.model_id == CAUSAL_GRU_ID else None,
            "gru_batch_first": True if checkpoint.model_id == CAUSAL_GRU_ID else None,
            "parameters": parameters,
        }
    else:
        raise HarmBenchCheckpointArtifactError("unsupported low-level checkpoint type")
    architecture_json = _canonical_json_bytes(architecture).decode("ascii")
    arrays[_ARCHITECTURE_STORAGE_KEY] = np.frombuffer(
        architecture_json.encode("ascii"), dtype=np.uint8
    ).copy()
    return arrays, architecture_json


_RECEIPT_KEYS = {
    "schema_version",
    "dataset_id",
    "model_id",
    "model_namespace",
    "training_seed",
    "fold",
    "ordered_class_tokens",
    "class_order_sha256",
    "fit_training_capability_sha256",
    "fit_feature_capability_sha256",
    "processor_receipt_sha256",
    "processed_output_receipt_sha256",
    "crossfit_plan_sha256",
    "fit_train_protocol_row_ids_sha256",
    "fit_heldout_protocol_row_ids_sha256",
    "context_roster_manifest_sha256",
    "context_training_examples_sha256",
    "independence_roster_sha256",
    "context_count",
    "history_consumption_count",
    "architecture",
    "payload_filename",
    "payload_sha256",
    "receipt_sha256",
}


@dataclass(frozen=True)
class CheckpointArtifactReceipt:
    schema_version: str
    dataset_id: str
    model_id: str
    model_namespace: str
    training_seed: int
    fold: int
    ordered_class_tokens: tuple[str, ...]
    class_order_sha256: str
    fit_training_capability_sha256: str
    fit_feature_capability_sha256: str
    processor_receipt_sha256: str
    processed_output_receipt_sha256: str
    crossfit_plan_sha256: str
    fit_train_protocol_row_ids_sha256: str
    fit_heldout_protocol_row_ids_sha256: str
    context_roster_manifest_sha256: str | None
    context_training_examples_sha256: str | None
    independence_roster_sha256: str | None
    context_count: int | None
    history_consumption_count: int | None
    architecture_json: str
    payload_filename: str
    payload_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        _validate_receipt(self)


def _receipt_descriptor(receipt: CheckpointArtifactReceipt) -> dict[str, object]:
    return {
        "schema_version": receipt.schema_version,
        "dataset_id": receipt.dataset_id,
        "model_id": receipt.model_id,
        "model_namespace": receipt.model_namespace,
        "training_seed": receipt.training_seed,
        "fold": receipt.fold,
        "ordered_class_tokens": list(receipt.ordered_class_tokens),
        "class_order_sha256": receipt.class_order_sha256,
        "fit_training_capability_sha256": receipt.fit_training_capability_sha256,
        "fit_feature_capability_sha256": receipt.fit_feature_capability_sha256,
        "processor_receipt_sha256": receipt.processor_receipt_sha256,
        "processed_output_receipt_sha256": receipt.processed_output_receipt_sha256,
        "crossfit_plan_sha256": receipt.crossfit_plan_sha256,
        "fit_train_protocol_row_ids_sha256": receipt.fit_train_protocol_row_ids_sha256,
        "fit_heldout_protocol_row_ids_sha256": receipt.fit_heldout_protocol_row_ids_sha256,
        "context_roster_manifest_sha256": receipt.context_roster_manifest_sha256,
        "context_training_examples_sha256": receipt.context_training_examples_sha256,
        "independence_roster_sha256": receipt.independence_roster_sha256,
        "context_count": receipt.context_count,
        "history_consumption_count": receipt.history_consumption_count,
        "architecture": json.loads(receipt.architecture_json),
        "payload_filename": receipt.payload_filename,
        "payload_sha256": receipt.payload_sha256,
    }


def checkpoint_artifact_receipt_payload(
    receipt: CheckpointArtifactReceipt,
) -> dict[str, object]:
    _validate_receipt(receipt)
    return {**_receipt_descriptor(receipt), "receipt_sha256": receipt.receipt_sha256}


def _validate_receipt(receipt: CheckpointArtifactReceipt) -> None:
    if receipt.schema_version != ARTIFACT_RECEIPT_SCHEMA:
        raise HarmBenchCheckpointArtifactError("checkpoint artifact receipt schema changed")
    if (
        not receipt.dataset_id
        or receipt.model_id not in FROZEN_MODEL_IDS
        or receipt.model_namespace not in (
        HISTORY_NAMESPACE,
        CURRENT_ONLY_NAMESPACE,
        )
    ):
        raise HarmBenchCheckpointArtifactError("checkpoint artifact identity changed")
    if (
        type(receipt.training_seed) is not int
        or receipt.training_seed not in EXPECTED_TRAINING_SEEDS
        or type(receipt.fold) is not int
        or receipt.fold not in range(EXPECTED_OUTER_FOLDS)
    ):
        raise HarmBenchCheckpointArtifactError("checkpoint seed/fold is outside the frozen roster")
    classes = receipt.ordered_class_tokens
    if (
        type(classes) is not tuple
        or len(classes) < 2
        or len(classes) > MAX_CLASS_COUNT
        or len(set(classes)) != len(classes)
        or any(not isinstance(value, str) or not value for value in classes)
    ):
        raise HarmBenchCheckpointArtifactError("checkpoint class order is invalid")
    for name in (
        "class_order_sha256",
        "fit_training_capability_sha256",
        "fit_feature_capability_sha256",
        "processor_receipt_sha256",
        "processed_output_receipt_sha256",
        "crossfit_plan_sha256",
        "fit_train_protocol_row_ids_sha256",
        "fit_heldout_protocol_row_ids_sha256",
        "payload_sha256",
        "receipt_sha256",
    ):
        _sha256(getattr(receipt, name), name=name)
    if receipt.class_order_sha256 != _class_order_sha256(
        classes,
        dataset_id=receipt.dataset_id,
        fit_sha256=receipt.fit_training_capability_sha256,
    ):
        raise HarmBenchCheckpointArtifactError("class order differs from its typed lineage")
    try:
        architecture = json.loads(receipt.architecture_json)
    except (TypeError, json.JSONDecodeError) as error:
        raise HarmBenchCheckpointArtifactError("architecture metadata is invalid") from error
    if _canonical_json_bytes(architecture).decode("ascii") != receipt.architecture_json:
        raise HarmBenchCheckpointArtifactError("architecture metadata is not canonical")
    if not isinstance(architecture, dict):
        raise HarmBenchCheckpointArtifactError("architecture metadata must be an object")
    if (
        architecture.get("family_id") != receipt.model_id
        or architecture.get("model_namespace") != receipt.model_namespace
        or architecture.get("seed") != receipt.training_seed
        or architecture.get("num_classes") != len(classes)
    ):
        raise HarmBenchCheckpointArtifactError("architecture differs from receipt identity")
    _validate_architecture(architecture, receipt)
    if Path(receipt.payload_filename).name != receipt.payload_filename or not receipt.payload_filename.endswith(
        ".npz"
    ):
        raise HarmBenchCheckpointArtifactError("payload filename is unsafe")
    if receipt.model_namespace == HISTORY_NAMESPACE:
        if (
            receipt.context_roster_manifest_sha256 is None
            or receipt.context_training_examples_sha256 is None
            or receipt.independence_roster_sha256 is not None
            or receipt.context_count is not None
            or receipt.history_consumption_count is not None
        ):
            raise HarmBenchCheckpointArtifactError("history lineage is incomplete or mixed")
        _sha256(receipt.context_roster_manifest_sha256, name="context_roster_manifest_sha256")
        _sha256(
            receipt.context_training_examples_sha256,
            name="context_training_examples_sha256",
        )
    else:
        if (
            receipt.context_roster_manifest_sha256 is not None
            or receipt.context_training_examples_sha256 is not None
            or receipt.independence_roster_sha256 is None
            or type(receipt.context_count) is not int
            or receipt.context_count != 0
            or type(receipt.history_consumption_count) is not int
            or receipt.history_consumption_count != 0
        ):
            raise HarmBenchCheckpointArtifactError(
                "current-only artifact must prove zero context/history consumption"
            )
        _sha256(receipt.independence_roster_sha256, name="independence_roster_sha256")
    if receipt.receipt_sha256 != _canonical_json_sha256(_receipt_descriptor(receipt)):
        raise HarmBenchCheckpointArtifactError("checkpoint artifact receipt SHA changed")


def _network_type(model_id: str, model_namespace: str) -> type[object]:
    if model_id == DEEPSETS_POOL_ID:
        return (
            _DeepSetsHistoryNetwork
            if model_namespace == HISTORY_NAMESPACE
            else _DeepSetsCurrentOnlyNetwork
        )
    if model_id == CAUSAL_GRU_ID:
        return (
            _CausalGRUHistoryNetwork
            if model_namespace == HISTORY_NAMESPACE
            else _CausalGRUCurrentOnlyNetwork
        )
    raise HarmBenchCheckpointArtifactError("linear checkpoints have no torch network")


def _expected_parameter_specs(
    *,
    model_id: str,
    model_namespace: str,
    num_classes: int,
    seed: int,
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    """Derive the only legal parameter schema from frozen code, not JSON."""

    if model_id == LINEAR_POOL_ID:
        feature_dimension = (
            LINEAR_HISTORY_SUMMARY_DIMENSION
            if model_namespace == HISTORY_NAMESPACE
            else QUERY_DIMENSION
        )
        classifier_rows = 1 if num_classes == 2 else num_classes
        return (
            ("coef_", np.dtype(np.float64).str, (classifier_rows, feature_dimension)),
            ("intercept_", np.dtype(np.float64).str, (classifier_rows,)),
            ("classes_", np.dtype(np.int64).str, (num_classes,)),
            ("t_", np.dtype(np.float64).str, (1,)),
            ("n_iter_", np.dtype(np.int64).str, (1,)),
            ("n_features_in_", np.dtype(np.int64).str, (1,)),
        )
    network_type = _network_type(model_id, model_namespace)
    try:
        network = _initialized_network(
            network_type,  # type: ignore[arg-type]
            num_classes=num_classes,
            seed=seed,
        )
        state = network.state_dict()  # type: ignore[attr-defined]
    except (RuntimeError, TypeError, ValueError) as error:
        raise HarmBenchCheckpointArtifactError(
            f"cannot derive frozen neural parameter schema: {error}"
        ) from error
    return tuple(
        (
            name,
            tensor.detach().cpu().numpy().dtype.str,
            tuple(int(value) for value in tensor.shape),
        )
        for name, tensor in sorted(state.items())
    )


def _expected_parameter_manifest(
    receipt: CheckpointArtifactReceipt,
) -> tuple[list[dict[str, object]], int]:
    specs = _expected_parameter_specs(
        model_id=receipt.model_id,
        model_namespace=receipt.model_namespace,
        num_classes=len(receipt.ordered_class_tokens),
        seed=receipt.training_seed,
    )
    total_elements = int(
        sum(int(np.prod(shape, dtype=np.int64)) for _, _, shape in specs)
    )
    limit = FAMILY_TOTAL_ELEMENT_LIMITS[receipt.model_id]
    if total_elements < 1 or total_elements >= limit:
        raise HarmBenchCheckpointArtifactError(
            "checkpoint parameter count exceeds its frozen family budget"
        )
    return (
        [
            {
                "name": name,
                "storage_key": f"p{index:04d}",
                "dtype": dtype,
                "shape": list(shape),
            }
            for index, (name, dtype, shape) in enumerate(specs)
        ],
        total_elements,
    )


def _validate_architecture(
    architecture: object,
    receipt: CheckpointArtifactReceipt,
) -> None:
    if not isinstance(architecture, dict):
        raise HarmBenchCheckpointArtifactError("architecture metadata must be an object")
    common_keys = {
        "schema_version",
        "family_id",
        "model_namespace",
        "num_classes",
        "seed",
        "input_dimensions",
        "implementation",
        "parameters",
    }
    extra = (
        {"frozen_estimator"}
        if receipt.model_id == LINEAR_POOL_ID
        else {
            "projection_dimension",
            "item_hidden_dimension",
            "gru_hidden_dimension",
            "gru_num_layers",
            "gru_batch_first",
        }
    )
    if set(architecture) != common_keys | extra:
        raise HarmBenchCheckpointArtifactError("architecture metadata keys changed")
    if (
        architecture["schema_version"] != ARTIFACT_PAYLOAD_SCHEMA
        or architecture["input_dimensions"]
        != {"text": TEXT_DIMENSION, "audio": AUDIO_DIMENSION, "video": VIDEO_DIMENSION}
    ):
        raise HarmBenchCheckpointArtifactError("frozen input architecture changed")
    parameters = architecture["parameters"]
    if not isinstance(parameters, list) or not parameters:
        raise HarmBenchCheckpointArtifactError("frozen parameter roster is empty")
    if receipt.model_id == LINEAR_POOL_ID:
        expected_impl = "sklearn.linear_model.SGDClassifier"
        expected_estimator = {
            "loss": "log_loss",
            "penalty": "l2",
            "alpha": 1e-4,
            "fit_intercept": True,
            "shuffle": False,
            "average": False,
        }
        if (
            architecture["implementation"] != expected_impl
            or architecture["frozen_estimator"] != expected_estimator
        ):
            raise HarmBenchCheckpointArtifactError("frozen linear architecture changed")
    elif receipt.model_id == DEEPSETS_POOL_ID:
        expected_impl = (
            "_DeepSetsHistoryNetwork"
            if receipt.model_namespace == HISTORY_NAMESPACE
            else "_DeepSetsCurrentOnlyNetwork"
        )
        if (
            architecture["implementation"] != expected_impl
            or architecture["projection_dimension"] != PROJECTION_DIMENSION
            or architecture["item_hidden_dimension"] != ITEM_HIDDEN_DIMENSION
            or architecture["gru_hidden_dimension"] is not None
            or architecture["gru_num_layers"] is not None
            or architecture["gru_batch_first"] is not None
        ):
            raise HarmBenchCheckpointArtifactError("frozen DeepSets architecture changed")
    else:
        expected_impl = (
            "_CausalGRUHistoryNetwork"
            if receipt.model_namespace == HISTORY_NAMESPACE
            else "_CausalGRUCurrentOnlyNetwork"
        )
        if (
            architecture["implementation"] != expected_impl
            or architecture["projection_dimension"] != PROJECTION_DIMENSION
            or architecture["item_hidden_dimension"] is not None
            or architecture["gru_hidden_dimension"] != GRU_HIDDEN_DIMENSION
            or architecture["gru_num_layers"] != 1
            or architecture["gru_batch_first"] is not True
        ):
            raise HarmBenchCheckpointArtifactError("frozen GRU architecture changed")
    expected_parameters, _ = _expected_parameter_manifest(receipt)
    if parameters != expected_parameters:
        raise HarmBenchCheckpointArtifactError(
            "parameter manifest differs from the exact frozen architecture"
        )


@dataclass(frozen=True)
class VerifiedCheckpointArtifact:
    receipt: CheckpointArtifactReceipt
    receipt_path: Path
    payload_path: Path
    receipt_file_sha256: str
    parameters: Mapping[str, np.ndarray]
    _verification_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._verification_token is not _VERIFICATION_TOKEN:
            raise HarmBenchCheckpointArtifactError(
                "verified artifacts can only be created by the publisher or loader"
            )
        _validate_receipt(self.receipt)
        _sha256(self.receipt_file_sha256, name="receipt_file_sha256")
        if not isinstance(self.parameters, MappingProxyType):
            raise HarmBenchCheckpointArtifactError("verified parameters must be immutable")


def _artifact_stem(checkpoint: object) -> str:
    namespace = checkpoint.model_namespace.replace(".", "_")
    return (
        f"{namespace}__{checkpoint.model_id}__seed-{int(checkpoint.training_seed)}"
        f"__fold-{int(checkpoint.fold)}"
    )


def _validated_production_checkpoint(value: object) -> object:
    if not isinstance(value, (ProductionHistoryCheckpoint, ProductionCurrentOnlyCheckpoint)):
        raise HarmBenchCheckpointArtifactError(
            "writer input must be a typed production checkpoint"
        )
    try:
        rebuilt = type(value)(**{item.name: getattr(value, item.name) for item in fields(type(value))})
    except (TypeError, ValueError) as error:
        raise HarmBenchCheckpointArtifactError(
            f"production checkpoint changed: {error}"
        ) from error
    if rebuilt.model_namespace != rebuilt.checkpoint.model_namespace:
        raise HarmBenchCheckpointArtifactError("checkpoint namespace differs from wrapper")
    if int(rebuilt.training_seed) != int(rebuilt.checkpoint.seed):
        raise HarmBenchCheckpointArtifactError("checkpoint seed differs from wrapper")
    if rebuilt.model_id != rebuilt.checkpoint.family_id:
        raise HarmBenchCheckpointArtifactError("checkpoint family differs from wrapper")
    return rebuilt


def _make_receipt(checkpoint: object, architecture_json: str, payload_name: str, payload_sha: str) -> CheckpointArtifactReceipt:
    history = isinstance(checkpoint, ProductionHistoryCheckpoint)
    values = {
        "schema_version": ARTIFACT_RECEIPT_SCHEMA,
        "dataset_id": checkpoint.dataset_id,
        "model_id": checkpoint.model_id,
        "model_namespace": checkpoint.model_namespace,
        "training_seed": int(checkpoint.training_seed),
        "fold": int(checkpoint.fold),
        "ordered_class_tokens": tuple(checkpoint.class_order),
        "class_order_sha256": checkpoint.class_order_sha256,
        "fit_training_capability_sha256": checkpoint.fit_training_capability_sha256,
        "fit_feature_capability_sha256": checkpoint.fit_feature_capability_sha256,
        "processor_receipt_sha256": checkpoint.processor_receipt_sha256,
        "processed_output_receipt_sha256": checkpoint.processed_output_receipt_sha256,
        "crossfit_plan_sha256": checkpoint.crossfit_plan_sha256,
        "fit_train_protocol_row_ids_sha256": checkpoint.fit_train_protocol_row_ids_sha256,
        "fit_heldout_protocol_row_ids_sha256": checkpoint.fit_heldout_protocol_row_ids_sha256,
        "context_roster_manifest_sha256": (
            checkpoint.context_roster_manifest_sha256 if history else None
        ),
        "context_training_examples_sha256": (
            checkpoint.context_training_examples_sha256 if history else None
        ),
        "independence_roster_sha256": (
            None if history else checkpoint.independence_roster_sha256
        ),
        "context_count": None if history else int(checkpoint.context_count),
        "history_consumption_count": (
            None if history else int(checkpoint.history_consumption_count)
        ),
        "architecture_json": architecture_json,
        "payload_filename": payload_name,
        "payload_sha256": payload_sha,
    }
    temporary = object.__new__(CheckpointArtifactReceipt)
    for name, value in values.items():
        object.__setattr__(temporary, name, value)
    descriptor = _receipt_descriptor(temporary)
    return CheckpointArtifactReceipt(
        **values,
        receipt_sha256=_canonical_json_sha256(descriptor),
    )


def publish_checkpoint_artifact(
    private_root: str | Path,
    production_checkpoint: ProductionHistoryCheckpoint | ProductionCurrentOnlyCheckpoint,
) -> VerifiedCheckpointArtifact:
    """Publish one typed production checkpoint as private write-once NPZ + JSON."""

    root = _private_root(private_root)
    checkpoint = _validated_production_checkpoint(production_checkpoint)
    arrays, architecture_json = _extract_payload(checkpoint)
    payload_raw = _deterministic_npz_bytes(arrays)
    if len(payload_raw) > FAMILY_PAYLOAD_BYTE_LIMITS[checkpoint.model_id]:
        raise HarmBenchCheckpointArtifactError(
            "checkpoint payload exceeds its frozen family byte budget"
        )
    payload_sha = hashlib.sha256(payload_raw).hexdigest()
    stem = _artifact_stem(checkpoint)
    payload_path = root / f"{stem}.npz"
    receipt_path = root / f"{stem}.json"
    if os.path.lexists(payload_path) or os.path.lexists(receipt_path):
        raise FileExistsError(f"write-once checkpoint artifact already exists: {stem}")
    receipt = _make_receipt(
        checkpoint,
        architecture_json,
        payload_path.name,
        payload_sha,
    )
    receipt_raw = _canonical_json_bytes(checkpoint_artifact_receipt_payload(receipt))
    _write_once(payload_path, payload_raw)
    receipt_file_sha = _write_once(receipt_path, receipt_raw)
    return load_checkpoint_artifact(
        receipt_path,
        expected_receipt_file_sha256=receipt_file_sha,
    )


def _receipt_from_payload(value: object) -> CheckpointArtifactReceipt:
    if not isinstance(value, dict) or set(value) != _RECEIPT_KEYS:
        raise HarmBenchCheckpointArtifactError("artifact receipt keys changed")
    try:
        architecture_json = _canonical_json_bytes(value["architecture"]).decode("ascii")
        return CheckpointArtifactReceipt(
            schema_version=value["schema_version"],
            dataset_id=value["dataset_id"],
            model_id=value["model_id"],
            model_namespace=value["model_namespace"],
            training_seed=value["training_seed"],
            fold=value["fold"],
            ordered_class_tokens=tuple(value["ordered_class_tokens"]),
            class_order_sha256=value["class_order_sha256"],
            fit_training_capability_sha256=value["fit_training_capability_sha256"],
            fit_feature_capability_sha256=value["fit_feature_capability_sha256"],
            processor_receipt_sha256=value["processor_receipt_sha256"],
            processed_output_receipt_sha256=value["processed_output_receipt_sha256"],
            crossfit_plan_sha256=value["crossfit_plan_sha256"],
            fit_train_protocol_row_ids_sha256=value["fit_train_protocol_row_ids_sha256"],
            fit_heldout_protocol_row_ids_sha256=value["fit_heldout_protocol_row_ids_sha256"],
            context_roster_manifest_sha256=value["context_roster_manifest_sha256"],
            context_training_examples_sha256=value["context_training_examples_sha256"],
            independence_roster_sha256=value["independence_roster_sha256"],
            context_count=value["context_count"],
            history_consumption_count=value["history_consumption_count"],
            architecture_json=architecture_json,
            payload_filename=value["payload_filename"],
            payload_sha256=value["payload_sha256"],
            receipt_sha256=value["receipt_sha256"],
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, HarmBenchCheckpointArtifactError):
            raise
        raise HarmBenchCheckpointArtifactError("artifact receipt values are invalid") from error


def _npy_member_schema(raw: bytes) -> tuple[np.dtype, tuple[int, ...]]:
    stream = BytesIO(raw)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise HarmBenchCheckpointArtifactError("unsupported NPY member version")
    except HarmBenchCheckpointArtifactError:
        raise
    except (EOFError, OSError, ValueError) as error:
        raise HarmBenchCheckpointArtifactError("cannot decode NPY member header") from error
    dtype = np.dtype(dtype)
    normalized_shape = tuple(int(value) for value in shape)
    if fortran_order or dtype.hasobject or dtype.kind not in "biufc":
        raise HarmBenchCheckpointArtifactError(
            "checkpoint NPY member is not canonical numeric C-order data"
        )
    expected_data_bytes = int(np.prod(normalized_shape, dtype=np.int64)) * int(
        dtype.itemsize
    )
    if expected_data_bytes < 0 or stream.tell() + expected_data_bytes != len(raw):
        raise HarmBenchCheckpointArtifactError(
            "checkpoint NPY member byte count differs from its header"
        )
    return dtype, normalized_shape


def _validate_loaded_parameter_values(
    named: Mapping[str, np.ndarray],
    receipt: CheckpointArtifactReceipt,
) -> None:
    _, expected_total = _expected_parameter_manifest(receipt)
    observed_total = int(sum(array.size for array in named.values()))
    if observed_total != expected_total:
        raise HarmBenchCheckpointArtifactError(
            "checkpoint total parameter elements changed"
        )
    if receipt.model_id != LINEAR_POOL_ID:
        return
    classes = named["classes_"]
    expected_classes = np.arange(
        len(receipt.ordered_class_tokens), dtype=np.int64
    )
    if not np.array_equal(classes, expected_classes):
        raise HarmBenchCheckpointArtifactError(
            "linear estimator classes differ from the frozen class order"
        )
    feature_dimension = (
        LINEAR_HISTORY_SUMMARY_DIMENSION
        if receipt.model_namespace == HISTORY_NAMESPACE
        else QUERY_DIMENSION
    )
    if int(named["n_features_in_"].item()) != feature_dimension:
        raise HarmBenchCheckpointArtifactError(
            "linear estimator feature dimension changed"
        )
    if int(named["n_iter_"].item()) < 1 or float(named["t_"].item()) <= 0.0:
        raise HarmBenchCheckpointArtifactError(
            "linear estimator fitted-state counters are invalid"
        )


def _load_npz(raw: bytes, receipt: CheckpointArtifactReceipt) -> Mapping[str, np.ndarray]:
    byte_limit = FAMILY_PAYLOAD_BYTE_LIMITS[receipt.model_id]
    if len(raw) > byte_limit:
        raise HarmBenchCheckpointArtifactError(
            "checkpoint payload exceeds its frozen family byte budget"
        )
    architecture = json.loads(receipt.architecture_json)
    parameters = architecture.get("parameters")
    if not isinstance(parameters, list):
        raise HarmBenchCheckpointArtifactError("parameter manifest is missing")
    expected_member_schemas = {
        f"{item['storage_key']}.npy": (
            np.dtype(item["dtype"]),
            tuple(int(value) for value in item["shape"]),
        )
        for item in parameters
    }
    metadata_bytes = receipt.architecture_json.encode("ascii")
    expected_member_schemas[f"{_ARCHITECTURE_STORAGE_KEY}.npy"] = (
        np.dtype(np.uint8),
        (len(metadata_bytes),),
    )
    try:
        with zipfile.ZipFile(BytesIO(raw), mode="r") as archive:
            infos = archive.infolist()
            members = tuple(info.filename for info in infos)
            if (
                len(members) != len(set(members))
                or set(members) != set(expected_member_schemas)
                or any(
                    info.compress_type != zipfile.ZIP_STORED
                    or bool(info.flag_bits & 0x1)
                    or info.file_size < 0
                    or info.compress_size != info.file_size
                    or Path(info.filename).name != info.filename
                    for info in infos
                )
                or sum(int(info.file_size) for info in infos) > byte_limit
            ):
                raise HarmBenchCheckpointArtifactError("NPZ member roster is unsafe")
            for info in infos:
                member_raw = archive.read(info)
                dtype, shape = _npy_member_schema(member_raw)
                if (dtype, shape) != expected_member_schemas[info.filename]:
                    raise HarmBenchCheckpointArtifactError(
                        "NPZ member differs from the exact frozen schema"
                    )
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            arrays = {
                key: _array_snapshot(archive[key], name=key)
                for key in archive.files
            }
    except HarmBenchCheckpointArtifactError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise HarmBenchCheckpointArtifactError("cannot decode pickle-free checkpoint NPZ") from error
    if _deterministic_npz_bytes(arrays) != raw:
        raise HarmBenchCheckpointArtifactError("checkpoint NPZ is not canonical")
    metadata_array = arrays.pop(_ARCHITECTURE_STORAGE_KEY, None)
    if metadata_array is None or metadata_array.dtype != np.dtype("uint8"):
        raise HarmBenchCheckpointArtifactError("checkpoint architecture metadata is missing")
    try:
        embedded = metadata_array.tobytes().decode("ascii")
    except UnicodeDecodeError as error:
        raise HarmBenchCheckpointArtifactError("embedded architecture metadata is invalid") from error
    if embedded != receipt.architecture_json:
        raise HarmBenchCheckpointArtifactError("payload architecture differs from receipt")
    expected_keys: set[str] = set()
    named: dict[str, np.ndarray] = {}
    for item in parameters:
        if not isinstance(item, dict) or set(item) != {"name", "storage_key", "dtype", "shape"}:
            raise HarmBenchCheckpointArtifactError("parameter manifest entry changed")
        key = item["storage_key"]
        name = item["name"]
        if not isinstance(key, str) or not isinstance(name, str) or key in expected_keys or name in named:
            raise HarmBenchCheckpointArtifactError("parameter manifest contains duplicates")
        expected_keys.add(key)
        array = arrays.get(key)
        if array is None or array.dtype.str != item["dtype"] or list(array.shape) != item["shape"]:
            raise HarmBenchCheckpointArtifactError("parameter differs from architecture manifest")
        named[name] = array
    if set(arrays) != expected_keys:
        raise HarmBenchCheckpointArtifactError("checkpoint NPZ contains unexpected parameters")
    result = MappingProxyType(named)
    _validate_loaded_parameter_values(result, receipt)
    return result


def load_checkpoint_artifact(
    receipt_path: str | Path,
    *,
    expected_receipt_file_sha256: str,
) -> VerifiedCheckpointArtifact:
    """Load one externally bound receipt and its same-directory NPZ payload."""

    path = Path(receipt_path)
    if not path.is_absolute():
        raise HarmBenchCheckpointArtifactError("receipt path must be absolute")
    root = _private_root(path.parent)
    receipt_file = _plain_file(path, suffix=".json", must_exist=True)
    receipt_raw = _read_verified_bytes(
        receipt_file,
        expected_sha256=expected_receipt_file_sha256,
        max_bytes=MAX_RECEIPT_BYTES,
    )
    receipt = _receipt_from_payload(_parse_canonical_json(receipt_raw))
    expected_stem = (
        f"{receipt.model_namespace.replace('.', '_')}__{receipt.model_id}"
        f"__seed-{receipt.training_seed}__fold-{receipt.fold}"
    )
    if receipt_file.name != f"{expected_stem}.json":
        raise HarmBenchCheckpointArtifactError("receipt path differs from typed identity")
    payload_path = root / receipt.payload_filename
    if payload_path.name != f"{expected_stem}.npz":
        raise HarmBenchCheckpointArtifactError("payload path differs from typed identity")
    payload_raw = _read_verified_bytes(
        payload_path,
        expected_sha256=receipt.payload_sha256,
        max_bytes=FAMILY_PAYLOAD_BYTE_LIMITS[receipt.model_id],
    )
    parameters = _load_npz(payload_raw, receipt)
    return VerifiedCheckpointArtifact(
        receipt=receipt,
        receipt_path=receipt_file,
        payload_path=payload_path,
        receipt_file_sha256=_sha256(
            expected_receipt_file_sha256,
            name="expected_receipt_file_sha256",
        ),
        parameters=parameters,
        _verification_token=_VERIFICATION_TOKEN,
    )


def _validate_verified_checkpoint_artifact(
    artifact: VerifiedCheckpointArtifact,
) -> VerifiedCheckpointArtifact:
    """Re-open and re-verify a capability-bearing artifact at consumption time.

    This is intentionally private.  The manifest builder calls it through a
    function-local module import so the artifact publisher has no dependency
    on the manifest schema.  Merely constructing an object with plausible
    64-character digests is therefore not a production capability.
    """

    if (
        not isinstance(artifact, VerifiedCheckpointArtifact)
        or artifact._verification_token is not _VERIFICATION_TOKEN
    ):
        raise HarmBenchCheckpointArtifactError(
            "checkpoint manifest requires a verified checkpoint artifact"
        )
    _validate_receipt(artifact.receipt)
    _sha256(artifact.receipt_file_sha256, name="receipt_file_sha256")
    if not isinstance(artifact.parameters, MappingProxyType):
        raise HarmBenchCheckpointArtifactError(
            "verified artifact parameters are no longer immutable"
        )
    reloaded = load_checkpoint_artifact(
        artifact.receipt_path,
        expected_receipt_file_sha256=artifact.receipt_file_sha256,
    )
    if (
        reloaded.receipt != artifact.receipt
        or reloaded.receipt_path != artifact.receipt_path
        or reloaded.payload_path != artifact.payload_path
        or reloaded.receipt_file_sha256 != artifact.receipt_file_sha256
        or tuple(reloaded.parameters) != tuple(artifact.parameters)
    ):
        raise HarmBenchCheckpointArtifactError(
            "verified checkpoint artifact differs from its live receipt"
        )
    for name, expected in reloaded.parameters.items():
        observed = artifact.parameters[name]
        if (
            observed.dtype != expected.dtype
            or observed.shape != expected.shape
            or not np.array_equal(observed, expected)
        ):
            raise HarmBenchCheckpointArtifactError(
                "verified checkpoint parameters differ from the live payload"
            )
    return artifact


def _restored_linear_checkpoint(
    artifact: VerifiedCheckpointArtifact,
) -> LinearHistoryCheckpoint | LinearCurrentOnlyCheckpoint:
    receipt = artifact.receipt
    values = artifact.parameters
    estimator = _PredictionOnlySGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        fit_intercept=True,
        max_iter=1,
        tol=None,
        shuffle=False,
        random_state=receipt.training_seed,
        learning_rate="optimal",
        average=False,
    )
    for name in ("coef_", "intercept_", "classes_"):
        parameter = np.array(values[name], copy=True, order="C")
        parameter.setflags(write=False)
        setattr(estimator, name, parameter)
    estimator.t_ = float(values["t_"].item())
    estimator.n_iter_ = int(values["n_iter_"].item())
    estimator.n_features_in_ = int(values["n_features_in_"].item())
    checkpoint_type = (
        LinearHistoryCheckpoint
        if receipt.model_namespace == HISTORY_NAMESPACE
        else LinearCurrentOnlyCheckpoint
    )
    return checkpoint_type(
        len(receipt.ordered_class_tokens), receipt.training_seed, estimator
    )


def _restored_neural_checkpoint(
    artifact: VerifiedCheckpointArtifact,
) -> (
    DeepSetsHistoryCheckpoint
    | DeepSetsCurrentOnlyCheckpoint
    | CausalGRUHistoryCheckpoint
    | CausalGRUCurrentOnlyCheckpoint
):
    receipt = artifact.receipt
    network_type = _network_type(receipt.model_id, receipt.model_namespace)
    try:
        network = _initialized_network(
            network_type,  # type: ignore[arg-type]
            num_classes=len(receipt.ordered_class_tokens),
            seed=receipt.training_seed,
        )
        state = {
            name: torch.from_numpy(np.array(value, copy=True, order="C"))
            for name, value in artifact.parameters.items()
        }
        incompatible = network.load_state_dict(state, strict=True)  # type: ignore[attr-defined]
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise HarmBenchCheckpointArtifactError(
                "restored neural state_dict is not exact"
            )
        network.eval()  # type: ignore[attr-defined]
        network.requires_grad_(False)  # type: ignore[attr-defined]
        if any(
            parameter.requires_grad or parameter.device.type != "cpu"
            for parameter in network.parameters()  # type: ignore[attr-defined]
        ):
            raise HarmBenchCheckpointArtifactError(
                "restored neural checkpoint is not frozen on CPU"
            )
    except HarmBenchCheckpointArtifactError:
        raise
    except (RuntimeError, TypeError, ValueError) as error:
        raise HarmBenchCheckpointArtifactError(
            f"cannot restore strict neural state_dict: {error}"
        ) from error
    checkpoint_types = {
        (DEEPSETS_POOL_ID, HISTORY_NAMESPACE): DeepSetsHistoryCheckpoint,
        (DEEPSETS_POOL_ID, CURRENT_ONLY_NAMESPACE): DeepSetsCurrentOnlyCheckpoint,
        (CAUSAL_GRU_ID, HISTORY_NAMESPACE): CausalGRUHistoryCheckpoint,
        (CAUSAL_GRU_ID, CURRENT_ONLY_NAMESPACE): CausalGRUCurrentOnlyCheckpoint,
    }
    checkpoint_type = checkpoint_types[(receipt.model_id, receipt.model_namespace)]
    return checkpoint_type(
        len(receipt.ordered_class_tokens), receipt.training_seed, network
    )


def _restore_low_level_checkpoint(
    artifact: VerifiedCheckpointArtifact,
) -> object:
    if artifact.receipt.model_id == LINEAR_POOL_ID:
        return _restored_linear_checkpoint(artifact)
    return _restored_neural_checkpoint(artifact)


def _validate_live_restore_fit_lineage(
    artifact: VerifiedCheckpointArtifact,
    fit_capability: FitRoleCapability,
    shared_processor: SharedProcessor,
    processed_fit: ProcessedRoleEmbeddings,
    crossfit_plan: SharedGroupCrossfitPlan,
) -> tuple[
    FitRoleCapability,
    SharedProcessor,
    ProcessedRoleEmbeddings,
    np.ndarray,
    np.ndarray,
]:
    receipt = artifact.receipt
    try:
        fit = validate_fit_role_capability(fit_capability)
        feature_capability = fit.fit.feature_capability
        validate_shared_group_crossfit_plan(crossfit_plan, feature_capability)
        train_indices, heldout_indices = resolve_shared_group_crossfit_indices(
            crossfit_plan,
            feature_capability,
            training_seed=receipt.training_seed,
            fold=receipt.fold,
        )
        processor = validate_shared_processor(
            shared_processor,
            expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
            expected_fit_feature_capability_sha256=(
                receipt.fit_feature_capability_sha256
            ),
            expected_crossfit_plan_sha256=receipt.crossfit_plan_sha256,
            expected_seed=receipt.training_seed,
            expected_fold=receipt.fold,
        )
        processed = validate_processed_role_embeddings(
            processed_fit,
            expected_source_capability_sha256=(
                receipt.fit_feature_capability_sha256
            ),
            expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
            expected_output_receipt_sha256=(
                receipt.processed_output_receipt_sha256
            ),
        )
    except (TypeError, ValueError) as error:
        raise HarmBenchCheckpointArtifactError(
            f"live checkpoint restore lineage changed: {error}"
        ) from error
    source = feature_capability.fit
    exact = {
        "dataset_id": fit.dataset_id,
        "fit_training_capability_sha256": fit.capability_sha256,
        "fit_feature_capability_sha256": feature_capability.capability_sha256,
        "processor_receipt_sha256": processor.receipt.processor_receipt_sha256,
        "processed_output_receipt_sha256": processed.output_receipt_sha256,
        "crossfit_plan_sha256": crossfit_plan.plan_sha256,
    }
    if any(getattr(receipt, name) != value for name, value in exact.items()):
        raise HarmBenchCheckpointArtifactError(
            "artifact receipt differs from the live fit lineage"
        )
    if (
        receipt.ordered_class_tokens != tuple(fit.fit.label_order)
        or processed.dataset_id != source.dataset_id
        or processed.role != source.role
        or processed.source_content_sha256 != source.content_sha256
        or processed.source_row_alignment_sha256 != source.row_alignment_sha256
        or processed.cross_role_feature_roster_sha256
        != feature_capability.cross_role_feature_roster_sha256
        or not np.array_equal(processed.protocol_row_ids, source.protocol_row_ids)
    ):
        raise HarmBenchCheckpointArtifactError(
            "artifact classes or processed fit alignment changed"
        )
    return fit, processor, processed, train_indices, heldout_indices


def restore_history_checkpoint_artifact(
    artifact: VerifiedCheckpointArtifact,
    fit_capability: FitRoleCapability,
    shared_processor: SharedProcessor,
    processed_fit: ProcessedRoleEmbeddings,
    crossfit_plan: SharedGroupCrossfitPlan,
    context_rosters: Mapping[str, StrictPastContextRoster],
    context_training_examples: ContextTrainingExamples,
) -> ProductionHistoryCheckpoint:
    """Restart one history checkpoint for prediction only from live lineage.

    No family, namespace, class count, seed/fold, or digest is caller supplied.
    The artifact is reopened at consumption time and every fit/context receipt
    is re-derived from typed live objects before model bytes are materialized.
    """

    verified = _validate_verified_checkpoint_artifact(artifact)
    receipt = verified.receipt
    if receipt.model_namespace != HISTORY_NAMESPACE:
        raise HarmBenchCheckpointArtifactError(
            "history restore requires a history artifact namespace"
        )
    fit, processor, processed, _, _ = _validate_live_restore_fit_lineage(
        verified,
        fit_capability,
        shared_processor,
        processed_fit,
        crossfit_plan,
    )
    if (
        not isinstance(context_rosters, Mapping)
        or tuple(context_rosters) != STRICT_PAST_STRATEGY_IDS
    ):
        raise HarmBenchCheckpointArtifactError(
            "history restore context strategy roster/order changed"
        )
    roster_shas = {
        strategy: context_rosters[strategy].roster_sha256
        for strategy in STRICT_PAST_STRATEGY_IDS
    }
    try:
        examples = validate_context_training_examples(
            context_training_examples,
            context_rosters,
            fit.fit.feature_capability,
            processed,
            processor.receipt,
            crossfit_plan,
            training_seed=receipt.training_seed,
            fold=receipt.fold,
            expected_fit_feature_capability_sha256=(
                fit.fit.feature_capability.capability_sha256
            ),
            expected_processor_receipt_sha256=(
                processor.receipt.processor_receipt_sha256
            ),
            expected_processed_output_receipt_sha256=(
                processed.output_receipt_sha256
            ),
            expected_crossfit_plan_sha256=crossfit_plan.plan_sha256,
            expected_context_roster_sha256_by_strategy=roster_shas,
            expected_context_training_examples_sha256=(
                receipt.context_training_examples_sha256
            ),
        )
    except (TypeError, ValueError, AttributeError) as error:
        raise HarmBenchCheckpointArtifactError(
            f"history restore context lineage changed: {error}"
        ) from error
    aggregate = aggregate_context_roster_sha256(
        examples.context_roster_sha256_by_strategy
    )
    first = context_rosters[STRICT_PAST_STRATEGY_IDS[0]]
    if (
        aggregate != receipt.context_roster_manifest_sha256
        or examples.example_sha256
        != receipt.context_training_examples_sha256
        or first.fit_train_protocol_row_ids_sha256
        != receipt.fit_train_protocol_row_ids_sha256
        or first.fit_heldout_protocol_row_ids_sha256
        != receipt.fit_heldout_protocol_row_ids_sha256
    ):
        raise HarmBenchCheckpointArtifactError(
            "history artifact differs from live roster/example lineage"
        )
    low_level = _restore_low_level_checkpoint(verified)
    if not isinstance(
        low_level,
        (LinearHistoryCheckpoint, DeepSetsHistoryCheckpoint, CausalGRUHistoryCheckpoint),
    ):
        raise HarmBenchCheckpointArtifactError("restored history family changed")
    return ProductionHistoryCheckpoint(
        dataset_id=receipt.dataset_id,
        model_id=receipt.model_id,
        model_namespace=receipt.model_namespace,
        training_seed=receipt.training_seed,
        fold=receipt.fold,
        class_order=receipt.ordered_class_tokens,
        class_order_sha256=receipt.class_order_sha256,
        fit_training_capability_sha256=receipt.fit_training_capability_sha256,
        fit_feature_capability_sha256=receipt.fit_feature_capability_sha256,
        processor_receipt_sha256=receipt.processor_receipt_sha256,
        processed_output_receipt_sha256=receipt.processed_output_receipt_sha256,
        crossfit_plan_sha256=receipt.crossfit_plan_sha256,
        context_training_examples_sha256=(
            receipt.context_training_examples_sha256
        ),
        context_roster_manifest_sha256=receipt.context_roster_manifest_sha256,
        fit_train_protocol_row_ids_sha256=(
            receipt.fit_train_protocol_row_ids_sha256
        ),
        fit_heldout_protocol_row_ids_sha256=(
            receipt.fit_heldout_protocol_row_ids_sha256
        ),
        checkpoint=low_level,
    )


def restore_current_only_checkpoint_artifact(
    artifact: VerifiedCheckpointArtifact,
    fit_capability: FitRoleCapability,
    shared_processor: SharedProcessor,
    processed_fit: ProcessedRoleEmbeddings,
    crossfit_plan: SharedGroupCrossfitPlan,
    independence_roster: StrictPastContextRoster,
) -> ProductionCurrentOnlyCheckpoint:
    """Restart one current-only checkpoint after a live zero-history proof."""

    verified = _validate_verified_checkpoint_artifact(artifact)
    receipt = verified.receipt
    if receipt.model_namespace != CURRENT_ONLY_NAMESPACE:
        raise HarmBenchCheckpointArtifactError(
            "current-only restore requires a current-only artifact namespace"
        )
    fit, processor, processed, _, _ = _validate_live_restore_fit_lineage(
        verified,
        fit_capability,
        shared_processor,
        processed_fit,
        crossfit_plan,
    )
    try:
        independence = validate_strict_past_context_roster(
            independence_roster,
            fit.fit.feature_capability,
            fit.fit.feature_capability,
            processed,
            processor.receipt,
            crossfit_plan,
            training_seed=receipt.training_seed,
            fold=receipt.fold,
            context_role=FIT_TRAIN_CONTEXT_ROLE,
            strategy_id=CURRENT_ONLY_STRATEGY_ID,
            expected_fit_plan_capability_sha256=(
                fit.fit.feature_capability.capability_sha256
            ),
            expected_source_capability_sha256=(
                fit.fit.feature_capability.capability_sha256
            ),
            expected_processor_receipt_sha256=(
                processor.receipt.processor_receipt_sha256
            ),
            expected_processed_output_receipt_sha256=(
                processed.output_receipt_sha256
            ),
            expected_crossfit_plan_sha256=crossfit_plan.plan_sha256,
            expected_context_roster_sha256=receipt.independence_roster_sha256,
        )
    except (TypeError, ValueError, AttributeError) as error:
        raise HarmBenchCheckpointArtifactError(
            f"current-only restore independence proof changed: {error}"
        ) from error
    if (
        independence.roster_sha256 != receipt.independence_roster_sha256
        or any(independence.context_protocol_row_ids)
        or independence.total_context_count != 0
        or independence.history_consumption_count != 0
        or receipt.context_count != 0
        or receipt.history_consumption_count != 0
        or independence.fit_train_protocol_row_ids_sha256
        != receipt.fit_train_protocol_row_ids_sha256
        or independence.fit_heldout_protocol_row_ids_sha256
        != receipt.fit_heldout_protocol_row_ids_sha256
    ):
        raise HarmBenchCheckpointArtifactError(
            "current-only artifact differs from the live zero-history proof"
        )
    low_level = _restore_low_level_checkpoint(verified)
    if not isinstance(
        low_level,
        (
            LinearCurrentOnlyCheckpoint,
            DeepSetsCurrentOnlyCheckpoint,
            CausalGRUCurrentOnlyCheckpoint,
        ),
    ):
        raise HarmBenchCheckpointArtifactError("restored current-only family changed")
    return ProductionCurrentOnlyCheckpoint(
        dataset_id=receipt.dataset_id,
        model_id=receipt.model_id,
        model_namespace=receipt.model_namespace,
        training_seed=receipt.training_seed,
        fold=receipt.fold,
        class_order=receipt.ordered_class_tokens,
        class_order_sha256=receipt.class_order_sha256,
        fit_training_capability_sha256=receipt.fit_training_capability_sha256,
        fit_feature_capability_sha256=receipt.fit_feature_capability_sha256,
        processor_receipt_sha256=receipt.processor_receipt_sha256,
        processed_output_receipt_sha256=receipt.processed_output_receipt_sha256,
        crossfit_plan_sha256=receipt.crossfit_plan_sha256,
        independence_roster_sha256=receipt.independence_roster_sha256,
        fit_train_protocol_row_ids_sha256=(
            receipt.fit_train_protocol_row_ids_sha256
        ),
        fit_heldout_protocol_row_ids_sha256=(
            receipt.fit_heldout_protocol_row_ids_sha256
        ),
        context_count=receipt.context_count,
        history_consumption_count=receipt.history_consumption_count,
        checkpoint=low_level,
    )


__all__ = [
    "ARTIFACT_PAYLOAD_SCHEMA",
    "ARTIFACT_RECEIPT_SCHEMA",
    "CheckpointArtifactReceipt",
    "HarmBenchCheckpointArtifactError",
    "VerifiedCheckpointArtifact",
    "checkpoint_artifact_receipt_payload",
    "load_checkpoint_artifact",
    "publish_checkpoint_artifact",
    "restore_current_only_checkpoint_artifact",
    "restore_history_checkpoint_artifact",
]
