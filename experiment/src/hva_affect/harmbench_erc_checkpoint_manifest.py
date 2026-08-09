"""Canonical, write-once checkpoint lineage for HarmBench-ERC.

One manifest describes exactly one ``(dataset, model_id, model_namespace)``
combination and exactly the frozen five seeds by five outer folds.  Its
production builder accepts only live-reverified ``VerifiedCheckpointArtifact``
capabilities.  There is deliberately no raw digest binding or legacy adapter.

The module does not import the artifact publisher at import time.  Consumption
uses a function-local import, which keeps artifact publication independent from
the manifest schema and avoids an artifact/manifest import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from .harmbench_erc_contract import EXPECTED_TRAINING_SEEDS
from .harmbench_erc_crossfit import (
    EXPECTED_OUTER_FOLDS,
    SharedGroupCrossfitPlan,
    resolve_shared_group_crossfit_indices,
    validate_shared_group_crossfit_plan,
)
from .harmbench_erc_models import (
    CURRENT_ONLY_NAMESPACE,
    FROZEN_MODEL_IDS,
    HISTORY_NAMESPACE,
)
from .harmbench_erc_open_roles import FitRoleCapability, validate_fit_role_capability

if TYPE_CHECKING:  # pragma: no cover - annotations only; avoids a runtime cycle.
    from .harmbench_erc_checkpoint_artifact import VerifiedCheckpointArtifact


class HarmBenchCheckpointManifestError(ValueError):
    """Raised when checkpoint lineage is incomplete, mutable, or ambiguous."""


CHECKPOINT_MANIFEST_SCHEMA = "harmbench_erc_checkpoint_manifest_v2"
CHECKPOINT_ENTRY_SCHEMA = "harmbench_erc_checkpoint_manifest_entry_v2"
CHECKPOINT_CLASS_ORDER_SCHEMA = "harmbench_erc_checkpoint_class_order_v1"
EXPECTED_CHECKPOINT_ENTRY_COUNT = len(EXPECTED_TRAINING_SEEDS) * EXPECTED_OUTER_FOLDS
ALLOWED_MODEL_NAMESPACES = (HISTORY_NAMESPACE, CURRENT_ONLY_NAMESPACE)
SHA256_LENGTH = 64
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_MANIFEST_VERIFICATION_TOKEN = object()


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_LENGTH
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HarmBenchCheckpointManifestError(f"{name} must be a lowercase SHA-256")
    return value


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise HarmBenchCheckpointManifestError(f"{name} is not a frozen identifier")
    return value


def _exact_integer(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise HarmBenchCheckpointManifestError(f"{name} must be an exact integer")
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
        raise HarmBenchCheckpointManifestError(
            f"checkpoint manifest is not canonical JSON data: {error}"
        ) from error


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_array_sha256(values: object) -> str:
    """Match the frozen protocol-row roster digest used by context receipts."""

    array = np.asarray(values)
    if array.ndim != 1 or not len(array) or array.dtype.kind not in {"i", "u"}:
        raise HarmBenchCheckpointManifestError(
            "derived protocol-row roster must be a non-empty integer vector"
        )
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\x00")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\x00")
    canonical = np.ascontiguousarray(array)
    if canonical.dtype.byteorder == ">" or (
        canonical.dtype.byteorder == "=" and not np.little_endian
    ):
        canonical = canonical.byteswap().view(canonical.dtype.newbyteorder("<"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _expected_seed_fold_pairs() -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(seed), fold)
        for seed in EXPECTED_TRAINING_SEEDS
        for fold in range(EXPECTED_OUTER_FOLDS)
    )


def _model_identity(model_id: object, model_namespace: object) -> tuple[str, str]:
    model = _identifier(model_id, name="model_id")
    namespace = _identifier(model_namespace, name="model_namespace")
    if model not in FROZEN_MODEL_IDS:
        raise HarmBenchCheckpointManifestError("model_id is outside the frozen roster")
    if namespace not in ALLOWED_MODEL_NAMESPACES:
        raise HarmBenchCheckpointManifestError(
            "model_namespace is outside the isolated history/current roster"
        )
    return model, namespace


def _class_order_sha256(
    *,
    dataset_id: str,
    fit_training_capability_sha256: str,
    class_order: Sequence[str],
) -> str:
    return _canonical_json_sha256(
        {
            "schema_version": CHECKPOINT_CLASS_ORDER_SCHEMA,
            "dataset_id": dataset_id,
            "fit_training_capability_sha256": fit_training_capability_sha256,
            "ordered_class_tokens": list(class_order),
        }
    )


@dataclass(frozen=True)
class CheckpointManifestEntry:
    schema_version: str
    dataset_id: str
    model_id: str
    model_namespace: str
    training_seed: int
    fold: int
    checkpoint_payload_sha256: str
    artifact_receipt_sha256: str
    artifact_receipt_file_sha256: str
    crossfit_plan_sha256: str
    fit_training_capability_sha256: str
    fit_feature_capability_sha256: str
    processor_receipt_sha256: str
    processed_output_receipt_sha256: str
    fit_train_protocol_row_ids_sha256: str
    fit_heldout_protocol_row_ids_sha256: str
    class_order_sha256: str
    context_roster_manifest_sha256: str | None
    context_training_examples_sha256: str | None
    independence_roster_sha256: str | None
    context_count: int | None
    history_consumption_count: int | None
    entry_sha256: str

    def __post_init__(self) -> None:
        _validate_entry_shape(self)


@dataclass(frozen=True)
class CheckpointManifest:
    schema_version: str
    dataset_id: str
    model_id: str
    model_namespace: str
    training_seed_ids: tuple[int, ...]
    outer_folds: int
    entry_count: int
    fit_training_capability_sha256: str
    fit_feature_capability_sha256: str
    crossfit_plan_sha256: str
    ordered_class_tokens: tuple[str, ...]
    class_order_sha256: str
    entries: tuple[CheckpointManifestEntry, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        _validate_manifest_shape(self)


@dataclass(frozen=True)
class VerifiedCheckpointManifest:
    """Sealed loader result required by downstream prediction publication."""

    manifest: CheckpointManifest
    manifest_path: Path
    manifest_file_sha256: str
    _verification_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._verification_token is not _MANIFEST_VERIFICATION_TOKEN:
            raise HarmBenchCheckpointManifestError(
                "verified manifests can only be created by the manifest loader"
            )
        if not isinstance(self.manifest, CheckpointManifest):
            raise HarmBenchCheckpointManifestError(
                "verified manifest payload must be a CheckpointManifest"
            )
        _validate_manifest_shape(self.manifest)
        if (
            not isinstance(self.manifest_path, Path)
            or not self.manifest_path.is_absolute()
            or self.manifest_path.suffix.lower() != ".json"
        ):
            raise HarmBenchCheckpointManifestError(
                "verified manifest path must be an absolute .json Path"
            )
        _sha256(self.manifest_file_sha256, name="manifest_file_sha256")


def _entry_descriptor(entry: CheckpointManifestEntry) -> dict[str, object]:
    return {
        "schema_version": entry.schema_version,
        "dataset_id": entry.dataset_id,
        "model_id": entry.model_id,
        "model_namespace": entry.model_namespace,
        "training_seed": entry.training_seed,
        "fold": entry.fold,
        "checkpoint_payload_sha256": entry.checkpoint_payload_sha256,
        "artifact_receipt_sha256": entry.artifact_receipt_sha256,
        "artifact_receipt_file_sha256": entry.artifact_receipt_file_sha256,
        "crossfit_plan_sha256": entry.crossfit_plan_sha256,
        "fit_training_capability_sha256": entry.fit_training_capability_sha256,
        "fit_feature_capability_sha256": entry.fit_feature_capability_sha256,
        "processor_receipt_sha256": entry.processor_receipt_sha256,
        "processed_output_receipt_sha256": entry.processed_output_receipt_sha256,
        "fit_train_protocol_row_ids_sha256": (
            entry.fit_train_protocol_row_ids_sha256
        ),
        "fit_heldout_protocol_row_ids_sha256": (
            entry.fit_heldout_protocol_row_ids_sha256
        ),
        "class_order_sha256": entry.class_order_sha256,
        "context_roster_manifest_sha256": entry.context_roster_manifest_sha256,
        "context_training_examples_sha256": entry.context_training_examples_sha256,
        "independence_roster_sha256": entry.independence_roster_sha256,
        "context_count": entry.context_count,
        "history_consumption_count": entry.history_consumption_count,
    }


def _entry_payload(entry: CheckpointManifestEntry) -> dict[str, object]:
    return {**_entry_descriptor(entry), "entry_sha256": entry.entry_sha256}


def _manifest_descriptor(manifest: CheckpointManifest) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "dataset_id": manifest.dataset_id,
        "model_id": manifest.model_id,
        "model_namespace": manifest.model_namespace,
        "training_seed_ids": list(manifest.training_seed_ids),
        "outer_folds": manifest.outer_folds,
        "entry_count": manifest.entry_count,
        "fit_training_capability_sha256": manifest.fit_training_capability_sha256,
        "fit_feature_capability_sha256": manifest.fit_feature_capability_sha256,
        "crossfit_plan_sha256": manifest.crossfit_plan_sha256,
        "ordered_class_tokens": list(manifest.ordered_class_tokens),
        "class_order_sha256": manifest.class_order_sha256,
        "entries": [_entry_payload(entry) for entry in manifest.entries],
    }


def checkpoint_manifest_payload(manifest: CheckpointManifest) -> dict[str, object]:
    """Return the exact canonical JSON payload after structural validation."""

    _validate_manifest_shape(manifest)
    return {**_manifest_descriptor(manifest), "manifest_sha256": manifest.manifest_sha256}


def _validate_entry_shape(entry: CheckpointManifestEntry) -> None:
    if entry.schema_version != CHECKPOINT_ENTRY_SCHEMA:
        raise HarmBenchCheckpointManifestError("checkpoint entry schema changed")
    _identifier(entry.dataset_id, name="entry dataset_id")
    _, namespace = _model_identity(entry.model_id, entry.model_namespace)
    seed = _exact_integer(entry.training_seed, name="entry training_seed")
    fold = _exact_integer(entry.fold, name="entry fold")
    if seed not in EXPECTED_TRAINING_SEEDS or fold not in range(EXPECTED_OUTER_FOLDS):
        raise HarmBenchCheckpointManifestError("checkpoint entry seed/fold is outside the roster")
    for name in (
        "checkpoint_payload_sha256",
        "artifact_receipt_sha256",
        "artifact_receipt_file_sha256",
        "crossfit_plan_sha256",
        "fit_training_capability_sha256",
        "fit_feature_capability_sha256",
        "processor_receipt_sha256",
        "processed_output_receipt_sha256",
        "fit_train_protocol_row_ids_sha256",
        "fit_heldout_protocol_row_ids_sha256",
        "class_order_sha256",
        "entry_sha256",
    ):
        _sha256(getattr(entry, name), name=f"entry {name}")
    if (
        entry.fit_train_protocol_row_ids_sha256
        == entry.fit_heldout_protocol_row_ids_sha256
    ):
        raise HarmBenchCheckpointManifestError(
            "fit-train and heldout protocol-row roster bindings are identical"
        )
    if namespace == HISTORY_NAMESPACE:
        if (
            entry.context_roster_manifest_sha256 is None
            or entry.context_training_examples_sha256 is None
            or entry.independence_roster_sha256 is not None
            or entry.context_count is not None
            or entry.history_consumption_count is not None
        ):
            raise HarmBenchCheckpointManifestError(
                "history checkpoint lineage is incomplete or mixed with current-only"
            )
        _sha256(
            entry.context_roster_manifest_sha256,
            name="entry context_roster_manifest_sha256",
        )
        _sha256(
            entry.context_training_examples_sha256,
            name="entry context_training_examples_sha256",
        )
    else:
        if (
            entry.context_roster_manifest_sha256 is not None
            or entry.context_training_examples_sha256 is not None
            or entry.independence_roster_sha256 is None
            or type(entry.context_count) is not int
            or entry.context_count != 0
            or type(entry.history_consumption_count) is not int
            or entry.history_consumption_count != 0
        ):
            raise HarmBenchCheckpointManifestError(
                "current-only checkpoint must bind independence and zero context/history consumption"
            )
        _sha256(
            entry.independence_roster_sha256,
            name="entry independence_roster_sha256",
        )
    if _canonical_json_sha256(_entry_descriptor(entry)) != entry.entry_sha256:
        raise HarmBenchCheckpointManifestError("checkpoint entry receipt changed")


def _validate_manifest_shape(manifest: CheckpointManifest) -> None:
    if manifest.schema_version != CHECKPOINT_MANIFEST_SCHEMA:
        raise HarmBenchCheckpointManifestError("checkpoint manifest schema changed")
    _identifier(manifest.dataset_id, name="manifest dataset_id")
    _model_identity(manifest.model_id, manifest.model_namespace)
    if type(manifest.training_seed_ids) is not tuple or any(
        type(value) is not int for value in manifest.training_seed_ids
    ):
        raise HarmBenchCheckpointManifestError("training seed roster must be an immutable tuple")
    if manifest.training_seed_ids != tuple(EXPECTED_TRAINING_SEEDS):
        raise HarmBenchCheckpointManifestError("training seed roster/order changed")
    if _exact_integer(manifest.outer_folds, name="outer_folds") != EXPECTED_OUTER_FOLDS:
        raise HarmBenchCheckpointManifestError("outer fold count changed")
    if (
        _exact_integer(manifest.entry_count, name="entry_count")
        != EXPECTED_CHECKPOINT_ENTRY_COUNT
    ):
        raise HarmBenchCheckpointManifestError("checkpoint manifest must contain exactly 25 entries")
    if type(manifest.ordered_class_tokens) is not tuple:
        raise HarmBenchCheckpointManifestError("ordered class tokens must be an immutable tuple")
    classes = manifest.ordered_class_tokens
    if (
        len(classes) < 2
        or any(not isinstance(value, str) or not value for value in classes)
        or len(set(classes)) != len(classes)
    ):
        raise HarmBenchCheckpointManifestError("ordered class token roster is invalid")
    fit_training_sha = _sha256(
        manifest.fit_training_capability_sha256,
        name="fit_training_capability_sha256",
    )
    fit_feature_sha = _sha256(
        manifest.fit_feature_capability_sha256,
        name="fit_feature_capability_sha256",
    )
    plan_sha = _sha256(manifest.crossfit_plan_sha256, name="crossfit_plan_sha256")
    class_sha = _sha256(manifest.class_order_sha256, name="class_order_sha256")
    if class_sha != _class_order_sha256(
        dataset_id=manifest.dataset_id,
        fit_training_capability_sha256=fit_training_sha,
        class_order=classes,
    ):
        raise HarmBenchCheckpointManifestError("class order digest changed")
    if type(manifest.entries) is not tuple or not all(
        isinstance(entry, CheckpointManifestEntry) for entry in manifest.entries
    ):
        raise HarmBenchCheckpointManifestError("checkpoint entries must be an immutable typed tuple")
    if len(manifest.entries) != EXPECTED_CHECKPOINT_ENTRY_COUNT:
        raise HarmBenchCheckpointManifestError("checkpoint manifest must contain exactly 25 entries")
    if tuple((entry.training_seed, entry.fold) for entry in manifest.entries) != (
        _expected_seed_fold_pairs()
    ):
        raise HarmBenchCheckpointManifestError(
            "checkpoint entries must be exact seed-major/fold-minor order"
        )
    receipt_shas: set[str] = set()
    receipt_file_shas: set[str] = set()
    for entry in manifest.entries:
        _validate_entry_shape(entry)
        if (
            entry.dataset_id != manifest.dataset_id
            or entry.model_id != manifest.model_id
            or entry.model_namespace != manifest.model_namespace
            or entry.crossfit_plan_sha256 != plan_sha
            or entry.fit_training_capability_sha256 != fit_training_sha
            or entry.fit_feature_capability_sha256 != fit_feature_sha
            or entry.class_order_sha256 != class_sha
        ):
            raise HarmBenchCheckpointManifestError(
                "checkpoint entry identity/namespace differs from its manifest"
            )
        if entry.artifact_receipt_sha256 in receipt_shas:
            raise HarmBenchCheckpointManifestError("artifact receipt is reused across folds")
        if entry.artifact_receipt_file_sha256 in receipt_file_shas:
            raise HarmBenchCheckpointManifestError("artifact receipt file is reused across folds")
        receipt_shas.add(entry.artifact_receipt_sha256)
        receipt_file_shas.add(entry.artifact_receipt_file_sha256)
    _sha256(manifest.manifest_sha256, name="manifest_sha256")
    if _canonical_json_sha256(_manifest_descriptor(manifest)) != manifest.manifest_sha256:
        raise HarmBenchCheckpointManifestError("checkpoint manifest receipt changed")


def _live_sources(
    fit_training_capability: FitRoleCapability,
    crossfit_plan: SharedGroupCrossfitPlan,
    *,
    expected_fit_training_capability_sha256: str,
    expected_crossfit_plan_sha256: str,
) -> tuple[FitRoleCapability, SharedGroupCrossfitPlan]:
    try:
        fit = validate_fit_role_capability(fit_training_capability)
    except ValueError as error:
        raise HarmBenchCheckpointManifestError(
            f"fit training capability changed: {error}"
        ) from error
    if fit.capability_sha256 != _sha256(
        expected_fit_training_capability_sha256,
        name="expected_fit_training_capability_sha256",
    ):
        raise HarmBenchCheckpointManifestError(
            "fit training capability differs from external binding"
        )
    if not isinstance(crossfit_plan, SharedGroupCrossfitPlan):
        raise HarmBenchCheckpointManifestError(
            "crossfit_plan must be a SharedGroupCrossfitPlan"
        )
    try:
        validate_shared_group_crossfit_plan(crossfit_plan, fit.fit.feature_capability)
    except ValueError as error:
        raise HarmBenchCheckpointManifestError(f"crossfit plan changed: {error}") from error
    if crossfit_plan.plan_sha256 != _sha256(
        expected_crossfit_plan_sha256,
        name="expected_crossfit_plan_sha256",
    ):
        raise HarmBenchCheckpointManifestError(
            "crossfit plan differs from external binding"
        )
    return fit, crossfit_plan


def _derived_protocol_row_hashes(
    fit: FitRoleCapability,
    plan: SharedGroupCrossfitPlan,
    *,
    training_seed: int,
    fold: int,
) -> tuple[str, str]:
    try:
        train, heldout = resolve_shared_group_crossfit_indices(
            plan,
            fit.fit.feature_capability,
            training_seed=training_seed,
            fold=fold,
        )
    except ValueError as error:
        raise HarmBenchCheckpointManifestError(
            f"cannot derive checkpoint fold protocol rows: {error}"
        ) from error
    protocol_ids = np.asarray(fit.fit.features.protocol_row_ids)
    train_ids = protocol_ids[train]
    heldout_ids = protocol_ids[heldout]
    if set(train_ids.tolist()).intersection(heldout_ids.tolist()):
        raise HarmBenchCheckpointManifestError(
            "derived fit-train and heldout protocol-row rosters overlap"
        )
    return _canonical_array_sha256(train_ids), _canonical_array_sha256(heldout_ids)


def _verified_artifact_sequence(
    artifacts: Sequence[VerifiedCheckpointArtifact],
) -> tuple[VerifiedCheckpointArtifact, ...]:
    """Consume the artifact module's private verification capability live."""

    try:
        supplied = tuple(artifacts)
    except TypeError as error:
        raise HarmBenchCheckpointManifestError(
            "checkpoint artifacts must be an ordered sequence"
        ) from error
    if len(supplied) != EXPECTED_CHECKPOINT_ENTRY_COUNT:
        raise HarmBenchCheckpointManifestError(
            "checkpoint artifacts must be exact seed-major/fold-minor 5x5 order"
        )
    # Function-local by design: artifact publication does not import this module.
    from . import harmbench_erc_checkpoint_artifact as artifact_module

    verified: list[VerifiedCheckpointArtifact] = []
    for index, artifact in enumerate(supplied):
        try:
            verified.append(
                artifact_module._validate_verified_checkpoint_artifact(artifact)
            )
        except (TypeError, ValueError, OSError) as error:
            raise HarmBenchCheckpointManifestError(
                f"checkpoint artifact {index} failed live verification: {error}"
            ) from error
    return tuple(verified)


def build_checkpoint_manifest(
    fit_training_capability: FitRoleCapability,
    crossfit_plan: SharedGroupCrossfitPlan,
    artifacts: Sequence[VerifiedCheckpointArtifact],
    *,
    model_id: str,
    model_namespace: str,
    expected_fit_training_capability_sha256: str,
    expected_crossfit_plan_sha256: str,
) -> CheckpointManifest:
    """Build an exact 25-entry manifest only from live verified artifacts."""

    fit, plan = _live_sources(
        fit_training_capability,
        crossfit_plan,
        expected_fit_training_capability_sha256=(
            expected_fit_training_capability_sha256
        ),
        expected_crossfit_plan_sha256=expected_crossfit_plan_sha256,
    )
    model, namespace = _model_identity(model_id, model_namespace)
    supplied = _verified_artifact_sequence(artifacts)
    observed_pairs = tuple(
        (artifact.receipt.training_seed, artifact.receipt.fold)
        for artifact in supplied
    )
    if observed_pairs != _expected_seed_fold_pairs():
        raise HarmBenchCheckpointManifestError(
            "checkpoint artifacts must be exact seed-major/fold-minor 5x5 order"
        )
    classes = tuple(fit.fit.label_order)
    if (
        len(classes) < 2
        or any(not isinstance(value, str) or not value for value in classes)
        or len(set(classes)) != len(classes)
    ):
        raise HarmBenchCheckpointManifestError(
            "class order cannot be derived as unique non-empty tokens"
        )
    fit_feature_sha = fit.fit.feature_capability.capability_sha256
    class_sha = _class_order_sha256(
        dataset_id=fit.dataset_id,
        fit_training_capability_sha256=fit.capability_sha256,
        class_order=classes,
    )
    entries: list[CheckpointManifestEntry] = []
    for artifact in supplied:
        receipt = artifact.receipt
        if (
            receipt.dataset_id != fit.dataset_id
            or receipt.model_id != model
            or receipt.model_namespace != namespace
        ):
            raise HarmBenchCheckpointManifestError(
                "artifact dataset/model/namespace differs from the manifest identity"
            )
        if (
            receipt.fit_training_capability_sha256 != fit.capability_sha256
            or receipt.fit_feature_capability_sha256 != fit_feature_sha
            or receipt.crossfit_plan_sha256 != plan.plan_sha256
        ):
            raise HarmBenchCheckpointManifestError(
                "artifact fit/feature/plan lineage differs from live sources"
            )
        if (
            receipt.ordered_class_tokens != classes
            or receipt.class_order_sha256 != class_sha
        ):
            raise HarmBenchCheckpointManifestError(
                "artifact class order differs from the live fit capability"
            )
        train_sha, heldout_sha = _derived_protocol_row_hashes(
            fit,
            plan,
            training_seed=receipt.training_seed,
            fold=receipt.fold,
        )
        if (
            receipt.fit_train_protocol_row_ids_sha256 != train_sha
            or receipt.fit_heldout_protocol_row_ids_sha256 != heldout_sha
        ):
            raise HarmBenchCheckpointManifestError(
                "artifact protocol-row roster differs from the live crossfit partition"
            )
        values = {
            "schema_version": CHECKPOINT_ENTRY_SCHEMA,
            "dataset_id": fit.dataset_id,
            "model_id": model,
            "model_namespace": namespace,
            "training_seed": receipt.training_seed,
            "fold": receipt.fold,
            "checkpoint_payload_sha256": receipt.payload_sha256,
            "artifact_receipt_sha256": receipt.receipt_sha256,
            "artifact_receipt_file_sha256": artifact.receipt_file_sha256,
            "crossfit_plan_sha256": plan.plan_sha256,
            "fit_training_capability_sha256": fit.capability_sha256,
            "fit_feature_capability_sha256": fit_feature_sha,
            "processor_receipt_sha256": receipt.processor_receipt_sha256,
            "processed_output_receipt_sha256": (
                receipt.processed_output_receipt_sha256
            ),
            "fit_train_protocol_row_ids_sha256": train_sha,
            "fit_heldout_protocol_row_ids_sha256": heldout_sha,
            "class_order_sha256": class_sha,
            "context_roster_manifest_sha256": (
                receipt.context_roster_manifest_sha256
            ),
            "context_training_examples_sha256": (
                receipt.context_training_examples_sha256
            ),
            "independence_roster_sha256": receipt.independence_roster_sha256,
            "context_count": receipt.context_count,
            "history_consumption_count": receipt.history_consumption_count,
        }
        entries.append(
            CheckpointManifestEntry(
                **values,
                entry_sha256=_canonical_json_sha256(values),
            )
        )
    descriptor = {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
        "dataset_id": fit.dataset_id,
        "model_id": model,
        "model_namespace": namespace,
        "training_seed_ids": list(EXPECTED_TRAINING_SEEDS),
        "outer_folds": EXPECTED_OUTER_FOLDS,
        "entry_count": EXPECTED_CHECKPOINT_ENTRY_COUNT,
        "fit_training_capability_sha256": fit.capability_sha256,
        "fit_feature_capability_sha256": fit_feature_sha,
        "crossfit_plan_sha256": plan.plan_sha256,
        "ordered_class_tokens": list(classes),
        "class_order_sha256": class_sha,
        "entries": [_entry_payload(entry) for entry in entries],
    }
    return CheckpointManifest(
        schema_version=CHECKPOINT_MANIFEST_SCHEMA,
        dataset_id=fit.dataset_id,
        model_id=model,
        model_namespace=namespace,
        training_seed_ids=tuple(EXPECTED_TRAINING_SEEDS),
        outer_folds=EXPECTED_OUTER_FOLDS,
        entry_count=EXPECTED_CHECKPOINT_ENTRY_COUNT,
        fit_training_capability_sha256=fit.capability_sha256,
        fit_feature_capability_sha256=fit_feature_sha,
        crossfit_plan_sha256=plan.plan_sha256,
        ordered_class_tokens=classes,
        class_order_sha256=class_sha,
        entries=tuple(entries),
        manifest_sha256=_canonical_json_sha256(descriptor),
    )


def validate_checkpoint_manifest(
    manifest: CheckpointManifest,
    fit_training_capability: FitRoleCapability,
    crossfit_plan: SharedGroupCrossfitPlan,
    artifacts: Sequence[VerifiedCheckpointArtifact],
    *,
    expected_manifest_sha256: str,
    expected_fit_training_capability_sha256: str,
    expected_crossfit_plan_sha256: str,
) -> CheckpointManifest:
    """Live-rederive the manifest from 25 verified artifacts and external SHAs."""

    if not isinstance(manifest, CheckpointManifest):
        raise HarmBenchCheckpointManifestError("manifest must be a CheckpointManifest")
    _validate_manifest_shape(manifest)
    if manifest.manifest_sha256 != _sha256(
        expected_manifest_sha256, name="expected_manifest_sha256"
    ):
        raise HarmBenchCheckpointManifestError(
            "checkpoint manifest differs from external binding"
        )
    rebuilt = build_checkpoint_manifest(
        fit_training_capability,
        crossfit_plan,
        artifacts,
        model_id=manifest.model_id,
        model_namespace=manifest.model_namespace,
        expected_fit_training_capability_sha256=(
            expected_fit_training_capability_sha256
        ),
        expected_crossfit_plan_sha256=expected_crossfit_plan_sha256,
    )
    for item in fields(CheckpointManifest):
        if getattr(manifest, item.name) != getattr(rebuilt, item.name):
            raise HarmBenchCheckpointManifestError(
                f"checkpoint manifest differs from live artifact derivation: {item.name}"
            )
    return manifest


_TOP_LEVEL_KEYS = {
    "schema_version",
    "dataset_id",
    "model_id",
    "model_namespace",
    "training_seed_ids",
    "outer_folds",
    "entry_count",
    "fit_training_capability_sha256",
    "fit_feature_capability_sha256",
    "crossfit_plan_sha256",
    "ordered_class_tokens",
    "class_order_sha256",
    "entries",
    "manifest_sha256",
}
_ENTRY_KEYS = {
    "schema_version",
    "dataset_id",
    "model_id",
    "model_namespace",
    "training_seed",
    "fold",
    "checkpoint_payload_sha256",
    "artifact_receipt_sha256",
    "artifact_receipt_file_sha256",
    "crossfit_plan_sha256",
    "fit_training_capability_sha256",
    "fit_feature_capability_sha256",
    "processor_receipt_sha256",
    "processed_output_receipt_sha256",
    "fit_train_protocol_row_ids_sha256",
    "fit_heldout_protocol_row_ids_sha256",
    "class_order_sha256",
    "context_roster_manifest_sha256",
    "context_training_examples_sha256",
    "independence_roster_sha256",
    "context_count",
    "history_consumption_count",
    "entry_sha256",
}


def _exact_keys(
    value: object, expected: set[str], *, name: str
) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise HarmBenchCheckpointManifestError(f"{name} keys changed")
    return value


def _manifest_from_payload(payload: object) -> CheckpointManifest:
    top = _exact_keys(payload, _TOP_LEVEL_KEYS, name="checkpoint manifest")
    raw_entries = top["entries"]
    if not isinstance(raw_entries, list):
        raise HarmBenchCheckpointManifestError("checkpoint entries must be a JSON list")
    entries: list[CheckpointManifestEntry] = []
    for index, raw_entry in enumerate(raw_entries):
        entry = _exact_keys(raw_entry, _ENTRY_KEYS, name=f"checkpoint entry {index}")
        entries.append(CheckpointManifestEntry(**entry))
    if not isinstance(top["training_seed_ids"], list):
        raise HarmBenchCheckpointManifestError("training_seed_ids must be a JSON list")
    if not isinstance(top["ordered_class_tokens"], list):
        raise HarmBenchCheckpointManifestError("ordered_class_tokens must be a JSON list")
    values = dict(top)
    values["training_seed_ids"] = tuple(top["training_seed_ids"])
    values["ordered_class_tokens"] = tuple(top["ordered_class_tokens"])
    values["entries"] = tuple(entries)
    return CheckpointManifest(**values)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HarmBenchCheckpointManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise HarmBenchCheckpointManifestError(f"non-finite JSON constant is forbidden: {value}")


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
            raise HarmBenchCheckpointManifestError(
                "checkpoint manifest path contains a symlink or reparse point"
            )


def _plain_manifest_path(path: str | Path, *, must_exist: bool) -> Path:
    value = Path(path)
    if not value.is_absolute() or value.suffix.lower() != ".json":
        raise HarmBenchCheckpointManifestError(
            "checkpoint manifest path must be an absolute .json path"
        )
    _reject_reparse_components(value)
    if not value.parent.exists() or not value.parent.is_dir():
        raise HarmBenchCheckpointManifestError(
            "checkpoint manifest parent must be an existing directory"
        )
    if must_exist:
        try:
            metadata = os.lstat(value)
        except OSError as error:
            raise HarmBenchCheckpointManifestError(
                "cannot stat checkpoint manifest"
            ) from error
        if not stat.S_ISREG(metadata.st_mode) or _is_reparse_or_symlink(value):
            raise HarmBenchCheckpointManifestError(
                "checkpoint manifest must be a plain non-reparse file"
            )
    return value


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
        raise HarmBenchCheckpointManifestError(
            "cannot stat checkpoint manifest directory"
        ) from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise HarmBenchCheckpointManifestError(
            "checkpoint manifest parent is not a directory"
        )
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(getattr(metadata, "st_file_attributes", 0)),
    )


def _read_canonical_payload(path: str | Path, *, expected_file_sha256: str) -> object:
    expected_sha = _sha256(expected_file_sha256, name="expected_file_sha256")
    source = _plain_manifest_path(path, must_exist=True)
    parent_identity = _directory_identity(source.parent)
    before_path = os.lstat(source)
    try:
        with source.open("rb") as handle:
            before_handle = os.fstat(handle.fileno())
            if _identity(before_path) != _identity(before_handle):
                raise HarmBenchCheckpointManifestError(
                    "checkpoint manifest changed before verified read"
                )
            raw = handle.read()
            after_handle = os.fstat(handle.fileno())
    except HarmBenchCheckpointManifestError:
        raise
    except OSError as error:
        raise HarmBenchCheckpointManifestError(
            "cannot read checkpoint manifest"
        ) from error
    if _identity(before_handle) != _identity(after_handle):
        raise HarmBenchCheckpointManifestError(
            "checkpoint manifest changed during verified read"
        )
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise HarmBenchCheckpointManifestError(
            "checkpoint manifest file differs from external binding"
        )
    _reject_reparse_components(source)
    if _identity(os.lstat(source)) != _identity(after_handle):
        raise HarmBenchCheckpointManifestError(
            "checkpoint manifest path changed during verified read"
        )
    if _directory_identity(source.parent) != parent_identity:
        raise HarmBenchCheckpointManifestError(
            "checkpoint manifest directory changed during verified read"
        )
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except HarmBenchCheckpointManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise HarmBenchCheckpointManifestError(
            "cannot decode checkpoint manifest"
        ) from error
    if raw != _canonical_json_bytes(payload):
        raise HarmBenchCheckpointManifestError(
            "checkpoint manifest must use exact canonical JSON bytes"
        )
    return payload


def write_checkpoint_manifest_once(
    path: str | Path, manifest: CheckpointManifest
) -> str:
    """Atomically publish one canonical JSON manifest without clobbering."""

    destination = _plain_manifest_path(path, must_exist=False)
    if os.path.lexists(destination):
        raise FileExistsError(
            f"write-once checkpoint manifest already exists: {destination.name}"
        )
    raw = _canonical_json_bytes(checkpoint_manifest_payload(manifest))
    expected_file_sha = hashlib.sha256(raw).hexdigest()
    parent_identity = _directory_identity(destination.parent)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _reject_reparse_components(temporary)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(
                f"write-once checkpoint manifest already exists: {destination.name}"
            ) from None
        except OSError as error:
            if error.errno == errno.EEXIST or os.path.lexists(destination):
                raise FileExistsError(
                    f"write-once checkpoint manifest already exists: {destination.name}"
                ) from None
            raise HarmBenchCheckpointManifestError(
                f"cannot publish checkpoint manifest: {error}"
            ) from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    if _directory_identity(destination.parent) != parent_identity:
        raise HarmBenchCheckpointManifestError(
            "checkpoint manifest directory changed during publication"
        )
    observed = _read_canonical_payload(
        destination, expected_file_sha256=expected_file_sha
    )
    if observed != checkpoint_manifest_payload(manifest):
        raise HarmBenchCheckpointManifestError(
            "published checkpoint manifest payload changed"
        )
    return expected_file_sha


def _validate_verified_checkpoint_manifest(
    value: VerifiedCheckpointManifest,
) -> VerifiedCheckpointManifest:
    """Re-open a sealed manifest and verify its externally bound file live."""

    if (
        not isinstance(value, VerifiedCheckpointManifest)
        or value._verification_token is not _MANIFEST_VERIFICATION_TOKEN
    ):
        raise HarmBenchCheckpointManifestError(
            "downstream consumption requires a verified checkpoint manifest"
        )
    _validate_manifest_shape(value.manifest)
    payload = _read_canonical_payload(
        value.manifest_path,
        expected_file_sha256=value.manifest_file_sha256,
    )
    if _manifest_from_payload(payload) != value.manifest:
        raise HarmBenchCheckpointManifestError(
            "verified checkpoint manifest differs from its live file"
        )
    return value


def load_checkpoint_manifest(
    path: str | Path,
    fit_training_capability: FitRoleCapability,
    crossfit_plan: SharedGroupCrossfitPlan,
    artifacts: Sequence[VerifiedCheckpointArtifact],
    *,
    expected_file_sha256: str,
    expected_manifest_sha256: str,
    expected_fit_training_capability_sha256: str,
    expected_crossfit_plan_sha256: str,
) -> VerifiedCheckpointManifest:
    """Read one canonical snapshot and live-validate all artifact lineage."""

    payload = _read_canonical_payload(path, expected_file_sha256=expected_file_sha256)
    manifest = _manifest_from_payload(payload)
    validated = validate_checkpoint_manifest(
        manifest,
        fit_training_capability,
        crossfit_plan,
        artifacts,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_fit_training_capability_sha256=(
            expected_fit_training_capability_sha256
        ),
        expected_crossfit_plan_sha256=expected_crossfit_plan_sha256,
    )
    return VerifiedCheckpointManifest(
        manifest=validated,
        manifest_path=_plain_manifest_path(path, must_exist=True),
        manifest_file_sha256=_sha256(
            expected_file_sha256,
            name="expected_file_sha256",
        ),
        _verification_token=_MANIFEST_VERIFICATION_TOKEN,
    )


__all__ = [
    "ALLOWED_MODEL_NAMESPACES",
    "CHECKPOINT_CLASS_ORDER_SCHEMA",
    "CHECKPOINT_ENTRY_SCHEMA",
    "CHECKPOINT_MANIFEST_SCHEMA",
    "CheckpointManifest",
    "CheckpointManifestEntry",
    "EXPECTED_CHECKPOINT_ENTRY_COUNT",
    "HarmBenchCheckpointManifestError",
    "VerifiedCheckpointManifest",
    "build_checkpoint_manifest",
    "checkpoint_manifest_payload",
    "load_checkpoint_manifest",
    "validate_checkpoint_manifest",
    "write_checkpoint_manifest_once",
]
