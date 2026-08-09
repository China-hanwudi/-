"""Manifest-bound, prediction-only artifacts for HarmBench-ERC.

There is intentionally no raw prediction-artifact API.  A fold prediction can
only be sealed after the checkpoint manifest file, the source feature
capability, the fold-local processor output and a strict-past context roster
have all been revalidated live.  Each fold separately binds its actual strategy
roster and a ``dialogue_all_past`` roster that defines common history
eligibility.  Aggregate writers consume only a verified 25-entry checkpoint
manifest and a sealed panel.

The private NPZ contains the row-level alignment needed by the later sealed
evaluator.  Its public JSON receipt is aggregate-only and contains no labels,
outcome hashes, paths, query identifiers, groups or row vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .harmbench_erc_checkpoint_manifest import (
    EXPECTED_CHECKPOINT_ENTRY_COUNT,
    CheckpointManifestEntry,
    VerifiedCheckpointManifest,
)
from .harmbench_erc_contexts import (
    CURRENT_ONLY_STRATEGY_ID,
    FIT_HELDOUT_OOF_CONTEXT_ROLE,
    SELECTION_CONTEXT_ROLE,
    STRICT_PAST_STRATEGY_IDS,
    StrictPastContextRoster,
    validate_strict_past_context_roster,
)
from .harmbench_erc_contract import EXPECTED_TRAINING_SEEDS
from .harmbench_erc_crossfit import (
    EXPECTED_OUTER_FOLDS,
    SharedGroupCrossfitPlan,
    validate_shared_group_crossfit_plan,
)
from .harmbench_erc_metrics import HarmBenchMetricError, validated_probability
from .harmbench_erc_models import CURRENT_ONLY_NAMESPACE, HISTORY_NAMESPACE
from .harmbench_erc_open_roles import (
    FitFeatureCapability,
    SelectionFeatureCapability,
    validate_fit_feature_capability,
    validate_selection_feature_capability,
)
from .harmbench_erc_processors import ProcessedRoleEmbeddings, ProcessorReceipt


EXPECTED_TRAINING_SEED_IDS = tuple(int(value) for value in EXPECTED_TRAINING_SEEDS)
EXPECTED_FOLD_COUNT = EXPECTED_OUTER_FOLDS
FIT_ROLE = "fit_oof"
SELECTION_ROLE = "selection_ensemble"
FOLD_PREDICTION_SCHEMA = "harmbench_erc_sealed_fold_prediction_v3"
FIT_PANEL_SCHEMA = "harmbench_erc_sealed_fit_oof_panel_v3"
SELECTION_PANEL_SCHEMA = "harmbench_erc_sealed_selection_panel_v3"
FIT_ARTIFACT_SCHEMA = "harmbench_erc_fit_oof_predictions_private_v3"
SELECTION_ARTIFACT_SCHEMA = "harmbench_erc_selection_ensemble_predictions_private_v3"
PUBLIC_RECEIPT_SCHEMA = "harmbench_erc_prediction_only_public_receipt_v3"
EFFECTIVE_PAIR_SCHEMA = "harmbench_erc_effective_history_current_pair_v2"
DIALOGUE_ALL_PAST_STRATEGY_ID = "dialogue_all_past"
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FOLD_SEAL = object()
_PANEL_SEAL = object()
_LOADED_SEAL = object()
_PAIR_SEAL = object()


class HarmBenchPredictionArtifactError(ValueError):
    """Raised when prediction lineage, semantics or private I/O changes."""


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise HarmBenchPredictionArtifactError(
            f"{name} must be a short opaque identifier"
        )
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise HarmBenchPredictionArtifactError(f"{name} must be a lowercase SHA-256")
    return value


def _exact_int(value: object, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise HarmBenchPredictionArtifactError(f"{name} must be an exact integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise HarmBenchPredictionArtifactError(f"{name} must be at least {minimum}")
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
        raise HarmBenchPredictionArtifactError(
            f"prediction receipt is not canonical JSON data: {error}"
        ) from error


def _canonical_json_bytes(value: object) -> bytes:
    return _canonical_json_payload(value) + b"\n"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_payload(value)).hexdigest()


def _array_sha256(values: object) -> str:
    array = np.asarray(values)
    if array.dtype.kind == "O":
        raise HarmBenchPredictionArtifactError("object arrays cannot be hashed")
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\x00")
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\x00")
    if array.dtype.kind in {"U", "S"}:
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


def _readonly(values: object, *, dtype: object | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _probability_matrix(values: object, *, queries: int, classes: int) -> np.ndarray:
    try:
        probability = validated_probability(values, name="fold_probability")
    except HarmBenchMetricError as error:
        raise HarmBenchPredictionArtifactError(str(error)) from error
    if probability.shape != (queries, classes):
        raise HarmBenchPredictionArtifactError(
            "fold probability must have exact shape [Q_fold, C_manifest]"
        )
    return _readonly(probability, dtype=np.float64)


def _validate_probability_tensor(
    values: object, *, shape: tuple[int, ...], name: str
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape:
        raise HarmBenchPredictionArtifactError(f"{name} shape changed")
    matrices = array.reshape((-1, shape[-2], shape[-1]))
    try:
        for index, matrix in enumerate(matrices):
            validated_probability(matrix, name=f"{name}[{index}]")
    except HarmBenchMetricError as error:
        raise HarmBenchPredictionArtifactError(str(error)) from error
    return array


def _verified_manifest(value: object) -> VerifiedCheckpointManifest:
    """Use the manifest module's private loader capability, not shape duck-typing."""

    from . import harmbench_erc_checkpoint_manifest as manifest_module

    try:
        return manifest_module._validate_verified_checkpoint_manifest(value)
    except (TypeError, ValueError, OSError) as error:
        raise HarmBenchPredictionArtifactError(
            f"checkpoint manifest failed live verification: {error}"
        ) from error


def _manifest_entry(
    manifest: VerifiedCheckpointManifest, *, training_seed: int, fold: int
) -> CheckpointManifestEntry:
    matches = tuple(
        entry
        for entry in manifest.manifest.entries
        if entry.training_seed == training_seed and entry.fold == fold
    )
    if len(matches) != 1:
        raise HarmBenchPredictionArtifactError(
            "fold prediction does not map uniquely to one checkpoint manifest entry"
        )
    return matches[0]


def _expected_pairs() -> tuple[tuple[int, int], ...]:
    return tuple(
        (seed, fold)
        for seed in EXPECTED_TRAINING_SEED_IDS
        for fold in range(EXPECTED_FOLD_COUNT)
    )


@dataclass(frozen=True)
class SealedFoldPrediction:
    """One manifest-entry prediction capability with live source references."""

    schema_version: str
    role: str
    dataset_id: str
    model_id: str
    model_namespace: str
    training_seed: int
    fold: int
    checkpoint_manifest_sha256: str
    checkpoint_manifest_file_sha256: str
    checkpoint_entry_sha256: str
    checkpoint_payload_sha256: str
    checkpoint_artifact_receipt_sha256: str
    checkpoint_artifact_receipt_file_sha256: str
    fit_training_capability_sha256: str
    fit_feature_capability_sha256: str
    crossfit_plan_sha256: str
    source_capability_sha256: str
    cross_role_feature_roster_sha256: str
    source_content_sha256: str
    source_row_alignment_sha256: str
    processor_receipt_sha256: str
    processed_output_receipt_sha256: str
    processed_output_row_alignment_sha256: str
    ordered_class_tokens: tuple[str, ...]
    class_order_sha256: str
    query_protocol_row_ids: tuple[int, ...]
    query_protocol_row_ids_sha256: str
    context_role: str
    strategy_id: str
    strategy_context_roster_sha256: str
    strategy_context_protocol_row_ids_sha256: str
    dialogue_eligibility_roster_sha256: str
    dialogue_eligibility_context_protocol_row_ids_sha256: str
    context_count: tuple[int, ...]
    strategy_context_nonempty: tuple[bool, ...]
    dialogue_history_eligible: tuple[bool, ...]
    probability_sha256: str
    receipt_sha256: str
    probabilities: np.ndarray = field(repr=False, compare=False)
    _fit_plan_capability: FitFeatureCapability = field(repr=False, compare=False)
    _source_capability: FitFeatureCapability | SelectionFeatureCapability = field(
        repr=False, compare=False
    )
    _processed_features: ProcessedRoleEmbeddings = field(repr=False, compare=False)
    _processor_receipt: ProcessorReceipt = field(repr=False, compare=False)
    _crossfit_plan: SharedGroupCrossfitPlan = field(repr=False, compare=False)
    _strategy_context_roster: StrictPastContextRoster = field(
        repr=False, compare=False
    )
    _dialogue_eligibility_roster: StrictPastContextRoster = field(
        repr=False, compare=False
    )
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _FOLD_SEAL:
            raise HarmBenchPredictionArtifactError(
                "fold predictions can only be created by the sealed builder"
            )
        if np.asarray(self.probabilities).flags.writeable:
            raise HarmBenchPredictionArtifactError(
                "sealed fold probability array must be immutable"
            )


def _fold_descriptor(values: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": values["schema_version"],
        "role": values["role"],
        "dataset_id": values["dataset_id"],
        "model_id": values["model_id"],
        "model_namespace": values["model_namespace"],
        "training_seed": values["training_seed"],
        "fold": values["fold"],
        "checkpoint_manifest_sha256": values["checkpoint_manifest_sha256"],
        "checkpoint_manifest_file_sha256": values[
            "checkpoint_manifest_file_sha256"
        ],
        "checkpoint_entry_sha256": values["checkpoint_entry_sha256"],
        "checkpoint_payload_sha256": values["checkpoint_payload_sha256"],
        "checkpoint_artifact_receipt_sha256": values[
            "checkpoint_artifact_receipt_sha256"
        ],
        "checkpoint_artifact_receipt_file_sha256": values[
            "checkpoint_artifact_receipt_file_sha256"
        ],
        "fit_training_capability_sha256": values[
            "fit_training_capability_sha256"
        ],
        "fit_feature_capability_sha256": values["fit_feature_capability_sha256"],
        "crossfit_plan_sha256": values["crossfit_plan_sha256"],
        "source_capability_sha256": values["source_capability_sha256"],
        "cross_role_feature_roster_sha256": values[
            "cross_role_feature_roster_sha256"
        ],
        "source_content_sha256": values["source_content_sha256"],
        "source_row_alignment_sha256": values["source_row_alignment_sha256"],
        "processor_receipt_sha256": values["processor_receipt_sha256"],
        "processed_output_receipt_sha256": values[
            "processed_output_receipt_sha256"
        ],
        "processed_output_row_alignment_sha256": values[
            "processed_output_row_alignment_sha256"
        ],
        "ordered_class_tokens": list(values["ordered_class_tokens"]),
        "class_order_sha256": values["class_order_sha256"],
        "query_protocol_row_ids": list(values["query_protocol_row_ids"]),
        "query_protocol_row_ids_sha256": values[
            "query_protocol_row_ids_sha256"
        ],
        "context_role": values["context_role"],
        "strategy_id": values["strategy_id"],
        "strategy_context_roster_sha256": values[
            "strategy_context_roster_sha256"
        ],
        "strategy_context_protocol_row_ids_sha256": values[
            "strategy_context_protocol_row_ids_sha256"
        ],
        "dialogue_eligibility_roster_sha256": values[
            "dialogue_eligibility_roster_sha256"
        ],
        "dialogue_eligibility_context_protocol_row_ids_sha256": values[
            "dialogue_eligibility_context_protocol_row_ids_sha256"
        ],
        "context_count": list(values["context_count"]),
        "strategy_context_nonempty": list(values["strategy_context_nonempty"]),
        "dialogue_history_eligible": list(values["dialogue_history_eligible"]),
        "probability_sha256": values["probability_sha256"],
    }


def _build_fold_prediction(
    role: str,
    checkpoint_manifest: VerifiedCheckpointManifest,
    fit_plan_capability: FitFeatureCapability,
    source_capability: FitFeatureCapability | SelectionFeatureCapability,
    processed_features: ProcessedRoleEmbeddings,
    processor_receipt: ProcessorReceipt,
    crossfit_plan: SharedGroupCrossfitPlan,
    strategy_context_roster: StrictPastContextRoster,
    dialogue_eligibility_roster: StrictPastContextRoster,
    probabilities: object,
) -> SealedFoldPrediction:
    verified = _verified_manifest(checkpoint_manifest)
    manifest = verified.manifest
    try:
        fit = validate_fit_feature_capability(fit_plan_capability)
        if role == FIT_ROLE:
            if not isinstance(source_capability, FitFeatureCapability):
                raise HarmBenchPredictionArtifactError(
                    "fit OOF prediction requires a fit feature capability"
                )
            source = validate_fit_feature_capability(source_capability)
            if source.capability_sha256 != fit.capability_sha256:
                raise HarmBenchPredictionArtifactError(
                    "fit prediction source differs from the fit-plan capability"
                )
            features = source.fit
            expected_context_role = FIT_HELDOUT_OOF_CONTEXT_ROLE
        elif role == SELECTION_ROLE:
            if not isinstance(source_capability, SelectionFeatureCapability):
                raise HarmBenchPredictionArtifactError(
                    "selection prediction requires a feature-only selection capability"
                )
            source = validate_selection_feature_capability(source_capability)
            features = source.selection
            expected_context_role = SELECTION_CONTEXT_ROLE
        else:
            raise HarmBenchPredictionArtifactError("unsupported fold prediction role")
        validate_shared_group_crossfit_plan(crossfit_plan, fit)
    except HarmBenchPredictionArtifactError:
        raise
    except ValueError as error:
        raise HarmBenchPredictionArtifactError(
            f"fold prediction source failed live verification: {error}"
        ) from error
    if (
        fit.capability_sha256 != manifest.fit_feature_capability_sha256
        or crossfit_plan.plan_sha256 != manifest.crossfit_plan_sha256
        or fit.dataset_id != manifest.dataset_id
        or source.dataset_id != manifest.dataset_id
        or source.cross_role_feature_roster_sha256
        != fit.cross_role_feature_roster_sha256
    ):
        raise HarmBenchPredictionArtifactError(
            "fold prediction source differs from checkpoint manifest lineage"
        )
    if not isinstance(strategy_context_roster, StrictPastContextRoster):
        raise HarmBenchPredictionArtifactError(
            "fold prediction requires a strategy StrictPastContextRoster"
        )
    if not isinstance(dialogue_eligibility_roster, StrictPastContextRoster):
        raise HarmBenchPredictionArtifactError(
            "fold prediction requires a dialogue eligibility StrictPastContextRoster"
        )
    seed = _exact_int(strategy_context_roster.training_seed, name="training_seed")
    fold = _exact_int(strategy_context_roster.fold, name="fold")
    entry = _manifest_entry(verified, training_seed=seed, fold=fold)
    if strategy_context_roster.context_role != expected_context_role:
        raise HarmBenchPredictionArtifactError("context role differs from prediction role")
    try:
        roster = validate_strict_past_context_roster(
            strategy_context_roster,
            fit,
            source,
            processed_features,
            processor_receipt,
            crossfit_plan,
            training_seed=seed,
            fold=fold,
            context_role=expected_context_role,
            strategy_id=strategy_context_roster.strategy_id,
            expected_fit_plan_capability_sha256=fit.capability_sha256,
            expected_source_capability_sha256=source.capability_sha256,
            expected_processor_receipt_sha256=(
                processor_receipt.processor_receipt_sha256
            ),
            expected_processed_output_receipt_sha256=(
                processed_features.output_receipt_sha256
            ),
            expected_crossfit_plan_sha256=crossfit_plan.plan_sha256,
            expected_context_roster_sha256=strategy_context_roster.roster_sha256,
        )
        eligibility_roster = validate_strict_past_context_roster(
            dialogue_eligibility_roster,
            fit,
            source,
            processed_features,
            processor_receipt,
            crossfit_plan,
            training_seed=seed,
            fold=fold,
            context_role=expected_context_role,
            strategy_id=DIALOGUE_ALL_PAST_STRATEGY_ID,
            expected_fit_plan_capability_sha256=fit.capability_sha256,
            expected_source_capability_sha256=source.capability_sha256,
            expected_processor_receipt_sha256=(
                processor_receipt.processor_receipt_sha256
            ),
            expected_processed_output_receipt_sha256=(
                processed_features.output_receipt_sha256
            ),
            expected_crossfit_plan_sha256=crossfit_plan.plan_sha256,
            expected_context_roster_sha256=dialogue_eligibility_roster.roster_sha256,
        )
    except (AttributeError, ValueError) as error:
        raise HarmBenchPredictionArtifactError(
            f"prediction context failed live verification: {error}"
        ) from error
    if (
        entry.processor_receipt_sha256 != roster.processor_receipt_sha256
        or entry.fit_train_protocol_row_ids_sha256
        != roster.fit_train_protocol_row_ids_sha256
        or entry.fit_heldout_protocol_row_ids_sha256
        != roster.fit_heldout_protocol_row_ids_sha256
    ):
        raise HarmBenchPredictionArtifactError(
            "prediction processor/fold roster differs from exact manifest entry"
        )
    eligibility_lineage = (
        "dataset_id",
        "source_role",
        "context_role",
        "training_seed",
        "fold",
        "fit_plan_capability_sha256",
        "source_capability_sha256",
        "cross_role_feature_roster_sha256",
        "source_content_sha256",
        "source_row_alignment_sha256",
        "processor_receipt_sha256",
        "processed_output_receipt_sha256",
        "processed_output_row_alignment_sha256",
        "crossfit_plan_sha256",
        "fit_train_protocol_row_ids_sha256",
        "fit_heldout_protocol_row_ids_sha256",
        "query_protocol_row_ids",
        "query_protocol_row_ids_sha256",
    )
    if any(
        getattr(roster, name) != getattr(eligibility_roster, name)
        for name in eligibility_lineage
    ):
        raise HarmBenchPredictionArtifactError(
            "dialogue eligibility roster differs from the strategy roster lineage"
        )
    if (
        entry.processor_receipt_sha256
        != eligibility_roster.processor_receipt_sha256
        or entry.fit_train_protocol_row_ids_sha256
        != eligibility_roster.fit_train_protocol_row_ids_sha256
        or entry.fit_heldout_protocol_row_ids_sha256
        != eligibility_roster.fit_heldout_protocol_row_ids_sha256
    ):
        raise HarmBenchPredictionArtifactError(
            "dialogue eligibility roster differs from exact manifest entry"
        )
    if (
        role == FIT_ROLE
        and entry.processed_output_receipt_sha256
        != roster.processed_output_receipt_sha256
    ):
        raise HarmBenchPredictionArtifactError(
            "fit prediction output differs from checkpoint processed output"
        )
    if manifest.model_namespace == HISTORY_NAMESPACE:
        if roster.strategy_id not in STRICT_PAST_STRATEGY_IDS:
            raise HarmBenchPredictionArtifactError(
                "history namespace requires one frozen strict-past strategy"
            )
    elif manifest.model_namespace == CURRENT_ONLY_NAMESPACE:
        if (
            roster.strategy_id != CURRENT_ONLY_STRATEGY_ID
            or roster.total_context_count != 0
            or roster.history_consumption_count != 0
            or any(roster.context_protocol_row_ids)
            or any(roster.context_counts)
        ):
            raise HarmBenchPredictionArtifactError(
                "current-only prediction must prove zero context/history consumption"
            )
    else:
        raise HarmBenchPredictionArtifactError("checkpoint model namespace changed")
    query_ids = tuple(int(value) for value in roster.query_protocol_row_ids)
    counts = tuple(int(value) for value in roster.context_counts)
    strategy_nonempty = tuple(value > 0 for value in counts)
    dialogue_counts = tuple(
        int(value) for value in eligibility_roster.context_counts
    )
    dialogue_eligible = tuple(value > 0 for value in dialogue_counts)
    if (
        len(query_ids) != len(counts)
        or len(query_ids) != len(dialogue_eligible)
    ):
        raise HarmBenchPredictionArtifactError(
            "context counts are not aligned to prediction queries"
        )
    if any(
        strategy and not common
        for strategy, common in zip(
            strategy_nonempty, dialogue_eligible, strict=True
        )
    ) or any(
        strategy_count > dialogue_count
        for strategy_count, dialogue_count in zip(
            counts, dialogue_counts, strict=True
        )
    ):
        raise HarmBenchPredictionArtifactError(
            "strategy context coverage exceeds dialogue-all-past eligibility"
        )
    probability = _probability_matrix(
        probabilities,
        queries=len(query_ids),
        classes=len(manifest.ordered_class_tokens),
    )
    values: dict[str, object] = {
        "schema_version": FOLD_PREDICTION_SCHEMA,
        "role": role,
        "dataset_id": manifest.dataset_id,
        "model_id": manifest.model_id,
        "model_namespace": manifest.model_namespace,
        "training_seed": seed,
        "fold": fold,
        "checkpoint_manifest_sha256": manifest.manifest_sha256,
        "checkpoint_manifest_file_sha256": verified.manifest_file_sha256,
        "checkpoint_entry_sha256": entry.entry_sha256,
        "checkpoint_payload_sha256": entry.checkpoint_payload_sha256,
        "checkpoint_artifact_receipt_sha256": entry.artifact_receipt_sha256,
        "checkpoint_artifact_receipt_file_sha256": (
            entry.artifact_receipt_file_sha256
        ),
        "fit_training_capability_sha256": (
            manifest.fit_training_capability_sha256
        ),
        "fit_feature_capability_sha256": manifest.fit_feature_capability_sha256,
        "crossfit_plan_sha256": manifest.crossfit_plan_sha256,
        "source_capability_sha256": source.capability_sha256,
        "cross_role_feature_roster_sha256": (
            source.cross_role_feature_roster_sha256
        ),
        "source_content_sha256": features.content_sha256,
        "source_row_alignment_sha256": features.row_alignment_sha256,
        "processor_receipt_sha256": roster.processor_receipt_sha256,
        "processed_output_receipt_sha256": (
            roster.processed_output_receipt_sha256
        ),
        "processed_output_row_alignment_sha256": (
            roster.processed_output_row_alignment_sha256
        ),
        "ordered_class_tokens": tuple(manifest.ordered_class_tokens),
        "class_order_sha256": manifest.class_order_sha256,
        "query_protocol_row_ids": query_ids,
        "query_protocol_row_ids_sha256": roster.query_protocol_row_ids_sha256,
        "context_role": roster.context_role,
        "strategy_id": roster.strategy_id,
        "strategy_context_roster_sha256": roster.roster_sha256,
        "strategy_context_protocol_row_ids_sha256": (
            roster.context_protocol_row_ids_sha256
        ),
        "dialogue_eligibility_roster_sha256": eligibility_roster.roster_sha256,
        "dialogue_eligibility_context_protocol_row_ids_sha256": (
            eligibility_roster.context_protocol_row_ids_sha256
        ),
        "context_count": counts,
        "strategy_context_nonempty": strategy_nonempty,
        "dialogue_history_eligible": dialogue_eligible,
        "probability_sha256": _array_sha256(probability),
    }
    return SealedFoldPrediction(
        **values,
        receipt_sha256=_canonical_sha256(_fold_descriptor(values)),
        probabilities=probability,
        _fit_plan_capability=fit,
        _source_capability=source,
        _processed_features=processed_features,
        _processor_receipt=processor_receipt,
        _crossfit_plan=crossfit_plan,
        _strategy_context_roster=roster,
        _dialogue_eligibility_roster=eligibility_roster,
        _seal=_FOLD_SEAL,
    )


def build_fit_fold_prediction(
    checkpoint_manifest: VerifiedCheckpointManifest,
    fit_feature_capability: FitFeatureCapability,
    processed_features: ProcessedRoleEmbeddings,
    processor_receipt: ProcessorReceipt,
    crossfit_plan: SharedGroupCrossfitPlan,
    strategy_context_roster: StrictPastContextRoster,
    dialogue_eligibility_roster: StrictPastContextRoster,
    probabilities: object,
) -> SealedFoldPrediction:
    """Seal one heldout fit-fold probability matrix from typed live inputs."""

    return _build_fold_prediction(
        FIT_ROLE,
        checkpoint_manifest,
        fit_feature_capability,
        fit_feature_capability,
        processed_features,
        processor_receipt,
        crossfit_plan,
        strategy_context_roster,
        dialogue_eligibility_roster,
        probabilities,
    )


def build_selection_fold_prediction(
    checkpoint_manifest: VerifiedCheckpointManifest,
    fit_feature_capability: FitFeatureCapability,
    selection_feature_capability: SelectionFeatureCapability,
    processed_features: ProcessedRoleEmbeddings,
    processor_receipt: ProcessorReceipt,
    crossfit_plan: SharedGroupCrossfitPlan,
    strategy_context_roster: StrictPastContextRoster,
    dialogue_eligibility_roster: StrictPastContextRoster,
    probabilities: object,
) -> SealedFoldPrediction:
    """Seal one full-selection fold prediction without any outcome surface."""

    return _build_fold_prediction(
        SELECTION_ROLE,
        checkpoint_manifest,
        fit_feature_capability,
        selection_feature_capability,
        processed_features,
        processor_receipt,
        crossfit_plan,
        strategy_context_roster,
        dialogue_eligibility_roster,
        probabilities,
    )


def _revalidate_fold_prediction(
    checkpoint_manifest: VerifiedCheckpointManifest,
    prediction: object,
    *,
    expected_role: str,
) -> SealedFoldPrediction:
    if (
        not isinstance(prediction, SealedFoldPrediction)
        or prediction._seal is not _FOLD_SEAL
    ):
        raise HarmBenchPredictionArtifactError(
            "panel requires sealed fold prediction capabilities"
        )
    if prediction.role != expected_role:
        raise HarmBenchPredictionArtifactError("fold prediction role changed")
    if np.asarray(prediction.probabilities).flags.writeable:
        raise HarmBenchPredictionArtifactError("fold prediction became writable")
    rebuilt = _build_fold_prediction(
        expected_role,
        checkpoint_manifest,
        prediction._fit_plan_capability,
        prediction._source_capability,
        prediction._processed_features,
        prediction._processor_receipt,
        prediction._crossfit_plan,
        prediction._strategy_context_roster,
        prediction._dialogue_eligibility_roster,
        prediction.probabilities,
    )
    excluded = {
        "probabilities",
        "_fit_plan_capability",
        "_source_capability",
        "_processed_features",
        "_processor_receipt",
        "_crossfit_plan",
        "_strategy_context_roster",
        "_dialogue_eligibility_roster",
        "_seal",
    }
    for item in fields(SealedFoldPrediction):
        if item.name not in excluded and getattr(prediction, item.name) != getattr(
            rebuilt, item.name
        ):
            raise HarmBenchPredictionArtifactError(
                f"sealed fold prediction changed: {item.name}"
            )
    if not np.array_equal(prediction.probabilities, rebuilt.probabilities):
        raise HarmBenchPredictionArtifactError("sealed fold probabilities changed")
    return prediction


def _roster_sha(schema: str, values: Sequence[object]) -> str:
    return _canonical_sha256({"schema_version": schema, "entries": list(values)})


def _panel_descriptor(panel_values: Mapping[str, object]) -> dict[str, object]:
    fields_to_bind = (
        "schema_version",
        "role",
        "dataset_id",
        "model_id",
        "model_namespace",
        "checkpoint_manifest_sha256",
        "checkpoint_manifest_file_sha256",
        "entry_count",
        "fit_training_capability_sha256",
        "fit_feature_capability_sha256",
        "crossfit_plan_sha256",
        "source_capability_sha256",
        "cross_role_feature_roster_sha256",
        "source_content_sha256",
        "source_row_alignment_sha256",
        "class_order_sha256",
        "context_role",
        "strategy_id",
        "query_count",
        "class_count",
        "query_roster_sha256",
        "group_roster_sha256",
        "probability_sha256",
        "per_fold_probability_sha256",
        "fold_assignment_sha256",
        "context_count_sha256",
        "strategy_context_nonempty_sha256",
        "dialogue_history_eligible_sha256",
        "fold_prediction_roster_sha256",
        "checkpoint_entry_roster_sha256",
        "strategy_context_roster_manifest_sha256",
        "dialogue_eligibility_roster_manifest_sha256",
        "source_roster_sha256",
        "processor_roster_sha256",
    )
    return {
        **{name: panel_values[name] for name in fields_to_bind},
        "training_seed_ids": list(panel_values["training_seed_ids"]),
        "fold_count": panel_values["fold_count"],
        "ordered_class_tokens": list(panel_values["ordered_class_tokens"]),
    }


@dataclass(frozen=True)
class SealedFitOOFPredictionPanel:
    schema_version: str
    role: str
    dataset_id: str
    model_id: str
    model_namespace: str
    checkpoint_manifest_sha256: str
    checkpoint_manifest_file_sha256: str
    training_seed_ids: tuple[int, ...]
    fold_count: int
    entry_count: int
    fit_training_capability_sha256: str
    fit_feature_capability_sha256: str
    crossfit_plan_sha256: str
    source_capability_sha256: str
    cross_role_feature_roster_sha256: str
    source_content_sha256: str
    source_row_alignment_sha256: str
    ordered_class_tokens: tuple[str, ...]
    class_order_sha256: str
    context_role: str
    strategy_id: str
    query_count: int
    class_count: int
    query_roster_sha256: str
    group_roster_sha256: str
    probability_sha256: str
    per_fold_probability_sha256: None
    fold_assignment_sha256: str
    context_count_sha256: str
    strategy_context_nonempty_sha256: str
    dialogue_history_eligible_sha256: str
    fold_prediction_roster_sha256: str
    checkpoint_entry_roster_sha256: str
    strategy_context_roster_manifest_sha256: str
    dialogue_eligibility_roster_manifest_sha256: str
    source_roster_sha256: str
    processor_roster_sha256: str
    panel_sha256: str
    fold_prediction_receipt_sha256_by_entry: tuple[str, ...]
    checkpoint_entry_sha256_by_entry: tuple[str, ...]
    strategy_context_roster_sha256_by_entry: tuple[str, ...]
    dialogue_eligibility_roster_sha256_by_entry: tuple[str, ...]
    processor_receipt_sha256_by_entry: tuple[str, ...]
    processed_output_receipt_sha256_by_entry: tuple[str, ...]
    query_protocol_row_ids: np.ndarray = field(repr=False, compare=False)
    group_tokens: np.ndarray = field(repr=False, compare=False)
    probabilities: np.ndarray = field(repr=False, compare=False)
    fold_assignments: np.ndarray = field(repr=False, compare=False)
    context_count: np.ndarray = field(repr=False, compare=False)
    strategy_context_nonempty: np.ndarray = field(repr=False, compare=False)
    dialogue_history_eligible: np.ndarray = field(repr=False, compare=False)
    _fold_predictions: tuple[SealedFoldPrediction, ...] = field(
        repr=False, compare=False
    )
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _PANEL_SEAL:
            raise HarmBenchPredictionArtifactError(
                "fit panels can only be created by the sealed builder"
            )
        for name in (
            "query_protocol_row_ids",
            "group_tokens",
            "probabilities",
            "fold_assignments",
            "context_count",
            "strategy_context_nonempty",
            "dialogue_history_eligible",
        ):
            if np.asarray(getattr(self, name)).flags.writeable:
                raise HarmBenchPredictionArtifactError(
                    f"sealed fit panel array is writable: {name}"
                )


@dataclass(frozen=True)
class SealedSelectionPredictionPanel:
    schema_version: str
    role: str
    dataset_id: str
    model_id: str
    model_namespace: str
    checkpoint_manifest_sha256: str
    checkpoint_manifest_file_sha256: str
    training_seed_ids: tuple[int, ...]
    fold_count: int
    entry_count: int
    fit_training_capability_sha256: str
    fit_feature_capability_sha256: str
    crossfit_plan_sha256: str
    source_capability_sha256: str
    cross_role_feature_roster_sha256: str
    source_content_sha256: str
    source_row_alignment_sha256: str
    ordered_class_tokens: tuple[str, ...]
    class_order_sha256: str
    context_role: str
    strategy_id: str
    query_count: int
    class_count: int
    query_roster_sha256: str
    group_roster_sha256: str
    probability_sha256: str
    per_fold_probability_sha256: str
    fold_assignment_sha256: None
    context_count_sha256: str
    strategy_context_nonempty_sha256: str
    dialogue_history_eligible_sha256: str
    fold_prediction_roster_sha256: str
    checkpoint_entry_roster_sha256: str
    strategy_context_roster_manifest_sha256: str
    dialogue_eligibility_roster_manifest_sha256: str
    source_roster_sha256: str
    processor_roster_sha256: str
    panel_sha256: str
    fold_prediction_receipt_sha256_by_entry: tuple[str, ...]
    checkpoint_entry_sha256_by_entry: tuple[str, ...]
    strategy_context_roster_sha256_by_entry: tuple[str, ...]
    dialogue_eligibility_roster_sha256_by_entry: tuple[str, ...]
    processor_receipt_sha256_by_entry: tuple[str, ...]
    processed_output_receipt_sha256_by_entry: tuple[str, ...]
    query_protocol_row_ids: np.ndarray = field(repr=False, compare=False)
    group_tokens: np.ndarray = field(repr=False, compare=False)
    probabilities: np.ndarray = field(repr=False, compare=False)
    per_fold_probabilities: np.ndarray = field(repr=False, compare=False)
    context_count: np.ndarray = field(repr=False, compare=False)
    strategy_context_nonempty: np.ndarray = field(repr=False, compare=False)
    dialogue_history_eligible: np.ndarray = field(repr=False, compare=False)
    _fold_predictions: tuple[SealedFoldPrediction, ...] = field(
        repr=False, compare=False
    )
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _PANEL_SEAL:
            raise HarmBenchPredictionArtifactError(
                "selection panels can only be created by the sealed builder"
            )
        for name in (
            "query_protocol_row_ids",
            "group_tokens",
            "probabilities",
            "per_fold_probabilities",
            "context_count",
            "strategy_context_nonempty",
            "dialogue_history_eligible",
        ):
            if np.asarray(getattr(self, name)).flags.writeable:
                raise HarmBenchPredictionArtifactError(
                    f"sealed selection panel array is writable: {name}"
                )


def _validated_fold_sequence(
    checkpoint_manifest: VerifiedCheckpointManifest,
    fold_predictions: Sequence[SealedFoldPrediction],
    *,
    role: str,
) -> tuple[SealedFoldPrediction, ...]:
    try:
        supplied = tuple(fold_predictions)
    except TypeError as error:
        raise HarmBenchPredictionArtifactError(
            "fold predictions must be an ordered sequence"
        ) from error
    if len(supplied) != EXPECTED_CHECKPOINT_ENTRY_COUNT:
        raise HarmBenchPredictionArtifactError(
            "panel requires exactly 25 fold predictions"
        )
    validated = tuple(
        _revalidate_fold_prediction(
            checkpoint_manifest, prediction, expected_role=role
        )
        for prediction in supplied
    )
    observed = tuple((item.training_seed, item.fold) for item in validated)
    if observed != _expected_pairs():
        raise HarmBenchPredictionArtifactError(
            "fold predictions must be exact seed-major/fold-minor order"
        )
    receipt_shas = tuple(item.receipt_sha256 for item in validated)
    if len(set(receipt_shas)) != len(receipt_shas):
        raise HarmBenchPredictionArtifactError("fold prediction receipt is duplicated")
    return validated


def _panel_common_values(
    verified: VerifiedCheckpointManifest,
    predictions: tuple[SealedFoldPrediction, ...],
    *,
    role: str,
    query_ids: np.ndarray,
    groups: np.ndarray,
    probabilities: np.ndarray,
    per_fold_probability_sha256: str | None,
    fold_assignment_sha256: str | None,
    context_count: np.ndarray,
    strategy_context_nonempty: np.ndarray,
    dialogue_history_eligible: np.ndarray,
) -> dict[str, object]:
    manifest = verified.manifest
    strategies = {item.strategy_id for item in predictions}
    context_roles = {item.context_role for item in predictions}
    sources = {item.source_capability_sha256 for item in predictions}
    source_contents = {item.source_content_sha256 for item in predictions}
    source_alignments = {item.source_row_alignment_sha256 for item in predictions}
    cross_rosters = {item.cross_role_feature_roster_sha256 for item in predictions}
    if any(len(values) != 1 for values in (
        strategies,
        context_roles,
        sources,
        source_contents,
        source_alignments,
        cross_rosters,
    )):
        raise HarmBenchPredictionArtifactError(
            "source/context/strategy bindings differ across the 25 predictions"
        )
    strategy = next(iter(strategies))
    if manifest.model_namespace == HISTORY_NAMESPACE:
        if strategy not in STRICT_PAST_STRATEGY_IDS:
            raise HarmBenchPredictionArtifactError(
                "history panel requires one consistent strict-past strategy"
            )
    elif manifest.model_namespace == CURRENT_ONLY_NAMESPACE:
        if (
            strategy != CURRENT_ONLY_STRATEGY_ID
            or np.any(context_count != 0)
            or np.any(strategy_context_nonempty)
        ):
            raise HarmBenchPredictionArtifactError(
                "current-only panel must preserve zero context/history consumption"
            )
    if (
        strategy_context_nonempty.shape != context_count.shape
        or not np.array_equal(strategy_context_nonempty, context_count > 0)
        or dialogue_history_eligible.shape != (len(query_ids),)
        or dialogue_history_eligible.dtype != np.dtype("bool")
    ):
        raise HarmBenchPredictionArtifactError(
            "strategy coverage/common dialogue eligibility tensors changed"
        )
    common_broadcast = dialogue_history_eligible.reshape(
        (1,) * (context_count.ndim - 1) + (len(query_ids),)
    )
    if np.any(strategy_context_nonempty & ~common_broadcast):
        raise HarmBenchPredictionArtifactError(
            "strategy context coverage exceeds common dialogue eligibility"
        )
    fold_receipts = tuple(item.receipt_sha256 for item in predictions)
    entry_receipts = tuple(item.checkpoint_entry_sha256 for item in predictions)
    strategy_context_receipts = tuple(
        item.strategy_context_roster_sha256 for item in predictions
    )
    dialogue_eligibility_receipts = tuple(
        item.dialogue_eligibility_roster_sha256 for item in predictions
    )
    processor_receipts = tuple(item.processor_receipt_sha256 for item in predictions)
    output_receipts = tuple(
        item.processed_output_receipt_sha256 for item in predictions
    )
    expected_entries = tuple(entry.entry_sha256 for entry in manifest.entries)
    expected_processors = tuple(
        entry.processor_receipt_sha256 for entry in manifest.entries
    )
    if entry_receipts != expected_entries or processor_receipts != expected_processors:
        raise HarmBenchPredictionArtifactError(
            "panel entries/processors differ from the exact checkpoint manifest"
        )
    query_sha = _array_sha256(query_ids)
    group_sha = _array_sha256(groups)
    values: dict[str, object] = {
        "schema_version": (
            FIT_PANEL_SCHEMA if role == FIT_ROLE else SELECTION_PANEL_SCHEMA
        ),
        "role": role,
        "dataset_id": manifest.dataset_id,
        "model_id": manifest.model_id,
        "model_namespace": manifest.model_namespace,
        "checkpoint_manifest_sha256": manifest.manifest_sha256,
        "checkpoint_manifest_file_sha256": verified.manifest_file_sha256,
        "training_seed_ids": EXPECTED_TRAINING_SEED_IDS,
        "fold_count": EXPECTED_FOLD_COUNT,
        "entry_count": EXPECTED_CHECKPOINT_ENTRY_COUNT,
        "fit_training_capability_sha256": (
            manifest.fit_training_capability_sha256
        ),
        "fit_feature_capability_sha256": manifest.fit_feature_capability_sha256,
        "crossfit_plan_sha256": manifest.crossfit_plan_sha256,
        "source_capability_sha256": next(iter(sources)),
        "cross_role_feature_roster_sha256": next(iter(cross_rosters)),
        "source_content_sha256": next(iter(source_contents)),
        "source_row_alignment_sha256": next(iter(source_alignments)),
        "ordered_class_tokens": tuple(manifest.ordered_class_tokens),
        "class_order_sha256": manifest.class_order_sha256,
        "context_role": next(iter(context_roles)),
        "strategy_id": strategy,
        "query_count": len(query_ids),
        "class_count": len(manifest.ordered_class_tokens),
        "query_roster_sha256": query_sha,
        "group_roster_sha256": group_sha,
        "probability_sha256": _array_sha256(probabilities),
        "per_fold_probability_sha256": per_fold_probability_sha256,
        "fold_assignment_sha256": fold_assignment_sha256,
        "context_count_sha256": _array_sha256(context_count),
        "strategy_context_nonempty_sha256": _array_sha256(
            strategy_context_nonempty
        ),
        "dialogue_history_eligible_sha256": _array_sha256(
            dialogue_history_eligible
        ),
        "fold_prediction_roster_sha256": _roster_sha(
            "harmbench_erc_fold_prediction_roster_v1", fold_receipts
        ),
        "checkpoint_entry_roster_sha256": _roster_sha(
            "harmbench_erc_checkpoint_entry_roster_v1", entry_receipts
        ),
        "strategy_context_roster_manifest_sha256": _roster_sha(
            "harmbench_erc_prediction_strategy_context_roster_v1",
            strategy_context_receipts,
        ),
        "dialogue_eligibility_roster_manifest_sha256": _roster_sha(
            "harmbench_erc_prediction_dialogue_eligibility_roster_v1",
            dialogue_eligibility_receipts,
        ),
        "source_roster_sha256": _canonical_sha256(
            {
                "schema_version": "harmbench_erc_prediction_source_roster_v1",
                "source_capability_sha256": next(iter(sources)),
                "cross_role_feature_roster_sha256": next(iter(cross_rosters)),
                "source_content_sha256": next(iter(source_contents)),
                "source_row_alignment_sha256": next(iter(source_alignments)),
                "query_roster_sha256": query_sha,
                "group_roster_sha256": group_sha,
            }
        ),
        "processor_roster_sha256": _canonical_sha256(
            {
                "schema_version": "harmbench_erc_prediction_processor_roster_v1",
                "entries": [
                    {
                        "training_seed": seed,
                        "fold": fold,
                        "processor_receipt_sha256": processor,
                        "processed_output_receipt_sha256": output,
                    }
                    for (seed, fold), processor, output in zip(
                        _expected_pairs(),
                        processor_receipts,
                        output_receipts,
                        strict=True,
                    )
                ],
            }
        ),
        "fold_prediction_receipt_sha256_by_entry": fold_receipts,
        "checkpoint_entry_sha256_by_entry": entry_receipts,
        "strategy_context_roster_sha256_by_entry": strategy_context_receipts,
        "dialogue_eligibility_roster_sha256_by_entry": (
            dialogue_eligibility_receipts
        ),
        "processor_receipt_sha256_by_entry": processor_receipts,
        "processed_output_receipt_sha256_by_entry": output_receipts,
    }
    values["panel_sha256"] = _canonical_sha256(_panel_descriptor(values))
    return values


def build_fit_oof_prediction_panel(
    checkpoint_manifest: VerifiedCheckpointManifest,
    fold_predictions: Sequence[SealedFoldPrediction],
) -> SealedFitOOFPredictionPanel:
    """Assemble exact 5x5 heldout folds into a source-ordered ``[5,Q,C]`` panel."""

    verified = _verified_manifest(checkpoint_manifest)
    predictions = _validated_fold_sequence(
        verified, fold_predictions, role=FIT_ROLE
    )
    source = predictions[0]._source_capability
    if not isinstance(source, FitFeatureCapability):
        raise HarmBenchPredictionArtifactError("fit panel source capability changed")
    features = source.fit
    query_ids = _readonly(features.protocol_row_ids, dtype=np.int64)
    groups = _readonly(features.groups, dtype=str)
    queries = len(query_ids)
    classes = len(verified.manifest.ordered_class_tokens)
    probabilities = np.empty((5, queries, classes), dtype=np.float64)
    assignments = np.full((5, queries), -1, dtype=np.int64)
    counts = np.zeros((5, queries), dtype=np.int64)
    strategy_nonempty = np.zeros((5, queries), dtype=np.bool_)
    eligibility_by_seed = np.zeros((5, queries), dtype=np.bool_)
    by_protocol = {int(value): index for index, value in enumerate(query_ids)}
    plan = predictions[0]._crossfit_plan
    for seed_index, seed in enumerate(EXPECTED_TRAINING_SEED_IDS):
        covered: set[int] = set()
        for fold in range(EXPECTED_FOLD_COUNT):
            prediction = predictions[seed_index * EXPECTED_FOLD_COUNT + fold]
            if (
                prediction._source_capability.capability_sha256
                != source.capability_sha256
                or prediction._crossfit_plan.plan_sha256 != plan.plan_sha256
            ):
                raise HarmBenchPredictionArtifactError(
                    "fit fold source/plan differs within the panel"
                )
            for local, protocol_id in enumerate(prediction.query_protocol_row_ids):
                if protocol_id not in by_protocol:
                    raise HarmBenchPredictionArtifactError(
                        "fit fold query is outside the source roster"
                    )
                target = by_protocol[protocol_id]
                if target in covered:
                    raise HarmBenchPredictionArtifactError(
                        "fit heldout fold queries overlap within one seed"
                    )
                covered.add(target)
                probabilities[seed_index, target] = prediction.probabilities[local]
                assignments[seed_index, target] = fold
                counts[seed_index, target] = prediction.context_count[local]
                strategy_nonempty[seed_index, target] = (
                    prediction.strategy_context_nonempty[local]
                )
                eligibility_by_seed[seed_index, target] = (
                    prediction.dialogue_history_eligible[local]
                )
        if covered != set(range(queries)):
            raise HarmBenchPredictionArtifactError(
                "five fit heldout rosters do not exactly cover the source"
            )
        for group in set(groups.tolist()):
            if len(set(assignments[seed_index][groups == group].tolist())) != 1:
                raise HarmBenchPredictionArtifactError(
                    "fit independent group crosses heldout folds"
                )
        expected_assignment = np.asarray(plan.fold_assignment[seed_index], dtype=np.int64)
        if not np.array_equal(assignments[seed_index], expected_assignment):
            raise HarmBenchPredictionArtifactError(
                "fit fold assignment differs from the live crossfit plan"
            )
    if not np.all(eligibility_by_seed == eligibility_by_seed[0:1]):
        raise HarmBenchPredictionArtifactError(
            "dialogue-all-past eligibility differs across fit seeds/folds"
        )
    if not np.all(strategy_nonempty == strategy_nonempty[0:1]):
        raise HarmBenchPredictionArtifactError(
            "strategy context coverage differs across fit seeds/folds"
        )
    dialogue_eligible = eligibility_by_seed[0].copy()
    _validate_probability_tensor(
        probabilities, shape=(5, queries, classes), name="fit_oof_probabilities"
    )
    probabilities = _readonly(probabilities, dtype=np.float64)
    assignments = _readonly(assignments, dtype=np.int64)
    counts = _readonly(counts, dtype=np.int64)
    strategy_nonempty = _readonly(strategy_nonempty, dtype=np.bool_)
    dialogue_eligible = _readonly(dialogue_eligible, dtype=np.bool_)
    values = _panel_common_values(
        verified,
        predictions,
        role=FIT_ROLE,
        query_ids=query_ids,
        groups=groups,
        probabilities=probabilities,
        per_fold_probability_sha256=None,
        fold_assignment_sha256=_array_sha256(assignments),
        context_count=counts,
        strategy_context_nonempty=strategy_nonempty,
        dialogue_history_eligible=dialogue_eligible,
    )
    return SealedFitOOFPredictionPanel(
        **values,
        query_protocol_row_ids=query_ids,
        group_tokens=groups,
        probabilities=probabilities,
        fold_assignments=assignments,
        context_count=counts,
        strategy_context_nonempty=strategy_nonempty,
        dialogue_history_eligible=dialogue_eligible,
        _fold_predictions=predictions,
        _seal=_PANEL_SEAL,
    )


def build_selection_prediction_panel(
    checkpoint_manifest: VerifiedCheckpointManifest,
    fold_predictions: Sequence[SealedFoldPrediction],
) -> SealedSelectionPredictionPanel:
    """Assemble exact ``[5,5,Q,C]`` fold predictions and bind their live mean."""

    verified = _verified_manifest(checkpoint_manifest)
    predictions = _validated_fold_sequence(
        verified, fold_predictions, role=SELECTION_ROLE
    )
    source = predictions[0]._source_capability
    if not isinstance(source, SelectionFeatureCapability):
        raise HarmBenchPredictionArtifactError("selection panel source changed")
    features = source.selection
    query_ids = _readonly(features.protocol_row_ids, dtype=np.int64)
    groups = _readonly(features.groups, dtype=str)
    queries = len(query_ids)
    classes = len(verified.manifest.ordered_class_tokens)
    per_fold = np.empty((5, 5, queries, classes), dtype=np.float64)
    counts = np.empty((5, 5, queries), dtype=np.int64)
    strategy_nonempty = np.empty((5, 5, queries), dtype=np.bool_)
    eligibility_by_entry = np.empty((5, 5, queries), dtype=np.bool_)
    by_protocol = {int(value): index for index, value in enumerate(query_ids)}
    complete = set(by_protocol)
    for index, prediction in enumerate(predictions):
        if prediction._source_capability.capability_sha256 != source.capability_sha256:
            raise HarmBenchPredictionArtifactError(
                "selection source differs across fold predictions"
            )
        if set(prediction.query_protocol_row_ids) != complete or len(
            prediction.query_protocol_row_ids
        ) != queries:
            raise HarmBenchPredictionArtifactError(
                "each selection fold must cover the same complete query roster"
            )
        seed_index, fold = divmod(index, EXPECTED_FOLD_COUNT)
        for local, protocol_id in enumerate(prediction.query_protocol_row_ids):
            target = by_protocol[protocol_id]
            per_fold[seed_index, fold, target] = prediction.probabilities[local]
            counts[seed_index, fold, target] = prediction.context_count[local]
            strategy_nonempty[seed_index, fold, target] = (
                prediction.strategy_context_nonempty[local]
            )
            eligibility_by_entry[seed_index, fold, target] = (
                prediction.dialogue_history_eligible[local]
            )
    if not np.all(eligibility_by_entry == eligibility_by_entry[0, 0]):
        raise HarmBenchPredictionArtifactError(
            "dialogue-all-past eligibility differs across selection seed/fold entries"
        )
    if not np.all(strategy_nonempty == strategy_nonempty[0, 0]):
        raise HarmBenchPredictionArtifactError(
            "strategy context coverage differs across all 25 selection entries"
        )
    dialogue_eligible = eligibility_by_entry[0, 0].copy()
    _validate_probability_tensor(
        per_fold,
        shape=(5, 5, queries, classes),
        name="selection_per_fold_probabilities",
    )
    mean = per_fold.mean(axis=1, dtype=np.float64)
    _validate_probability_tensor(
        mean, shape=(5, queries, classes), name="selection_mean_probabilities"
    )
    per_fold = _readonly(per_fold, dtype=np.float64)
    mean = _readonly(mean, dtype=np.float64)
    counts = _readonly(counts, dtype=np.int64)
    strategy_nonempty = _readonly(strategy_nonempty, dtype=np.bool_)
    dialogue_eligible = _readonly(dialogue_eligible, dtype=np.bool_)
    values = _panel_common_values(
        verified,
        predictions,
        role=SELECTION_ROLE,
        query_ids=query_ids,
        groups=groups,
        probabilities=mean,
        per_fold_probability_sha256=_array_sha256(per_fold),
        fold_assignment_sha256=None,
        context_count=counts,
        strategy_context_nonempty=strategy_nonempty,
        dialogue_history_eligible=dialogue_eligible,
    )
    return SealedSelectionPredictionPanel(
        **values,
        query_protocol_row_ids=query_ids,
        group_tokens=groups,
        probabilities=mean,
        per_fold_probabilities=per_fold,
        context_count=counts,
        strategy_context_nonempty=strategy_nonempty,
        dialogue_history_eligible=dialogue_eligible,
        _fold_predictions=predictions,
        _seal=_PANEL_SEAL,
    )


def _revalidate_panel(
    checkpoint_manifest: VerifiedCheckpointManifest,
    panel: object,
    *,
    role: str,
) -> SealedFitOOFPredictionPanel | SealedSelectionPredictionPanel:
    expected_type = (
        SealedFitOOFPredictionPanel if role == FIT_ROLE else SealedSelectionPredictionPanel
    )
    if not isinstance(panel, expected_type) or panel._seal is not _PANEL_SEAL:
        raise HarmBenchPredictionArtifactError(
            "writer requires the matching sealed prediction panel"
        )
    array_names = [
        "query_protocol_row_ids",
        "group_tokens",
        "probabilities",
        "context_count",
        "strategy_context_nonempty",
        "dialogue_history_eligible",
    ]
    array_names.append(
        "fold_assignments" if role == FIT_ROLE else "per_fold_probabilities"
    )
    if any(np.asarray(getattr(panel, name)).flags.writeable for name in array_names):
        raise HarmBenchPredictionArtifactError("sealed panel contains a writable array")
    rebuilt = (
        build_fit_oof_prediction_panel(checkpoint_manifest, panel._fold_predictions)
        if role == FIT_ROLE
        else build_selection_prediction_panel(
            checkpoint_manifest, panel._fold_predictions
        )
    )
    excluded = set(array_names) | {"_fold_predictions", "_seal"}
    for item in fields(expected_type):
        if item.name not in excluded and getattr(panel, item.name) != getattr(
            rebuilt, item.name
        ):
            raise HarmBenchPredictionArtifactError(
                f"sealed prediction panel changed: {item.name}"
            )
    for name in array_names:
        if not np.array_equal(getattr(panel, name), getattr(rebuilt, name)):
            raise HarmBenchPredictionArtifactError(
                f"sealed prediction panel array changed: {name}"
            )
    return panel


_COMMON_NPZ_KEYS = {
    "schema_version",
    "role",
    "dataset_id",
    "model_id",
    "model_namespace",
    "strategy_id",
    "context_role",
    "training_seed_ids",
    "fold_count",
    "entry_count",
    "checkpoint_manifest_sha256",
    "checkpoint_manifest_file_sha256",
    "fit_training_capability_sha256",
    "fit_feature_capability_sha256",
    "crossfit_plan_sha256",
    "source_capability_sha256",
    "cross_role_feature_roster_sha256",
    "source_content_sha256",
    "source_row_alignment_sha256",
    "class_order_sha256",
    "class_tokens",
    "query_protocol_row_ids",
    "group_tokens",
    "probabilities",
    "context_count",
    "strategy_context_nonempty",
    "dialogue_history_eligible",
    "fold_prediction_receipt_sha256_by_entry",
    "checkpoint_entry_sha256_by_entry",
    "strategy_context_roster_sha256_by_entry",
    "dialogue_eligibility_roster_sha256_by_entry",
    "processor_receipt_sha256_by_entry",
    "processed_output_receipt_sha256_by_entry",
    "query_roster_sha256",
    "group_roster_sha256",
    "probability_sha256",
    "context_count_sha256",
    "strategy_context_nonempty_sha256",
    "dialogue_history_eligible_sha256",
    "fold_prediction_roster_sha256",
    "checkpoint_entry_roster_sha256",
    "strategy_context_roster_manifest_sha256",
    "dialogue_eligibility_roster_manifest_sha256",
    "source_roster_sha256",
    "processor_roster_sha256",
    "panel_sha256",
}
_FIT_NPZ_KEYS = _COMMON_NPZ_KEYS | {"fold_assignments", "fold_assignment_sha256"}
_SELECTION_NPZ_KEYS = _COMMON_NPZ_KEYS | {
    "per_fold_probabilities",
    "per_fold_probability_sha256",
}
_PUBLIC_RECEIPT_KEYS = {
    "schema_version",
    "artifact_schema_version",
    "role",
    "prediction_semantics",
    "dataset_id",
    "model_id",
    "model_namespace",
    "strategy_id",
    "context_role",
    "training_seed_ids",
    "training_seed_count",
    "fold_count",
    "entry_count",
    "query_count",
    "class_count",
    "checkpoint_manifest_sha256",
    "checkpoint_manifest_file_sha256",
    "fit_training_capability_sha256",
    "fit_feature_capability_sha256",
    "crossfit_plan_sha256",
    "source_capability_sha256",
    "cross_role_feature_roster_sha256",
    "source_content_sha256",
    "source_row_alignment_sha256",
    "class_order_sha256",
    "query_roster_sha256",
    "group_roster_sha256",
    "probability_sha256",
    "per_fold_probability_sha256",
    "fold_assignment_sha256",
    "context_count_sha256",
    "strategy_context_nonempty_sha256",
    "dialogue_history_eligible_sha256",
    "fold_prediction_roster_sha256",
    "checkpoint_entry_roster_sha256",
    "strategy_context_roster_manifest_sha256",
    "dialogue_eligibility_roster_manifest_sha256",
    "source_roster_sha256",
    "processor_roster_sha256",
    "panel_sha256",
    "private_artifact_file_sha256",
    "context_summary",
    "privacy_contract",
}
_CONTEXT_SUMMARY_KEYS = {
    "strategy_context_tensor_rank",
    "strategy_context_count_total",
    "strategy_context_count_minimum",
    "strategy_context_count_maximum",
    "strategy_context_nonempty_count",
    "dialogue_history_eligible_count",
    "zero_strategy_consumption",
}
_PRIVACY_CONTRACT = {
    "aggregate_only": True,
    "allow_pickle_required": False,
    "contains_labels": False,
    "contains_outcome_hashes": False,
    "contains_paths": False,
    "contains_query_ids": False,
    "contains_group_ids": False,
    "contains_row_vectors": False,
}


def _string_scalar(value: str) -> np.ndarray:
    return np.asarray(value, dtype=f"<U{max(1, len(value))}")


def _panel_arrays(
    panel: SealedFitOOFPredictionPanel | SealedSelectionPredictionPanel,
) -> dict[str, np.ndarray]:
    artifact_schema = (
        FIT_ARTIFACT_SCHEMA if panel.role == FIT_ROLE else SELECTION_ARTIFACT_SCHEMA
    )
    scalar_strings = {
        "schema_version": artifact_schema,
        "role": panel.role,
        "dataset_id": panel.dataset_id,
        "model_id": panel.model_id,
        "model_namespace": panel.model_namespace,
        "strategy_id": panel.strategy_id,
        "context_role": panel.context_role,
        "checkpoint_manifest_sha256": panel.checkpoint_manifest_sha256,
        "checkpoint_manifest_file_sha256": panel.checkpoint_manifest_file_sha256,
        "fit_training_capability_sha256": panel.fit_training_capability_sha256,
        "fit_feature_capability_sha256": panel.fit_feature_capability_sha256,
        "crossfit_plan_sha256": panel.crossfit_plan_sha256,
        "source_capability_sha256": panel.source_capability_sha256,
        "cross_role_feature_roster_sha256": (
            panel.cross_role_feature_roster_sha256
        ),
        "source_content_sha256": panel.source_content_sha256,
        "source_row_alignment_sha256": panel.source_row_alignment_sha256,
        "class_order_sha256": panel.class_order_sha256,
        "query_roster_sha256": panel.query_roster_sha256,
        "group_roster_sha256": panel.group_roster_sha256,
        "probability_sha256": panel.probability_sha256,
        "context_count_sha256": panel.context_count_sha256,
        "strategy_context_nonempty_sha256": (
            panel.strategy_context_nonempty_sha256
        ),
        "dialogue_history_eligible_sha256": (
            panel.dialogue_history_eligible_sha256
        ),
        "fold_prediction_roster_sha256": panel.fold_prediction_roster_sha256,
        "checkpoint_entry_roster_sha256": panel.checkpoint_entry_roster_sha256,
        "strategy_context_roster_manifest_sha256": (
            panel.strategy_context_roster_manifest_sha256
        ),
        "dialogue_eligibility_roster_manifest_sha256": (
            panel.dialogue_eligibility_roster_manifest_sha256
        ),
        "source_roster_sha256": panel.source_roster_sha256,
        "processor_roster_sha256": panel.processor_roster_sha256,
        "panel_sha256": panel.panel_sha256,
    }
    arrays: dict[str, np.ndarray] = {
        name: _string_scalar(value) for name, value in scalar_strings.items()
    }
    arrays.update(
        {
            "training_seed_ids": np.asarray(panel.training_seed_ids, dtype="<i8"),
            "fold_count": np.asarray(panel.fold_count, dtype="<i8"),
            "entry_count": np.asarray(panel.entry_count, dtype="<i8"),
            "class_tokens": np.asarray(panel.ordered_class_tokens, dtype=str),
            "query_protocol_row_ids": np.asarray(
                panel.query_protocol_row_ids, dtype="<i8"
            ),
            "group_tokens": np.asarray(panel.group_tokens, dtype=str),
            "probabilities": np.asarray(panel.probabilities, dtype="<f8"),
            "context_count": np.asarray(panel.context_count, dtype="<i8"),
            "strategy_context_nonempty": np.asarray(
                panel.strategy_context_nonempty, dtype=np.bool_
            ),
            "dialogue_history_eligible": np.asarray(
                panel.dialogue_history_eligible, dtype=np.bool_
            ),
            "fold_prediction_receipt_sha256_by_entry": np.asarray(
                panel.fold_prediction_receipt_sha256_by_entry, dtype="<U64"
            ),
            "checkpoint_entry_sha256_by_entry": np.asarray(
                panel.checkpoint_entry_sha256_by_entry, dtype="<U64"
            ),
            "strategy_context_roster_sha256_by_entry": np.asarray(
                panel.strategy_context_roster_sha256_by_entry, dtype="<U64"
            ),
            "dialogue_eligibility_roster_sha256_by_entry": np.asarray(
                panel.dialogue_eligibility_roster_sha256_by_entry, dtype="<U64"
            ),
            "processor_receipt_sha256_by_entry": np.asarray(
                panel.processor_receipt_sha256_by_entry, dtype="<U64"
            ),
            "processed_output_receipt_sha256_by_entry": np.asarray(
                panel.processed_output_receipt_sha256_by_entry, dtype="<U64"
            ),
        }
    )
    if panel.role == FIT_ROLE:
        assert isinstance(panel, SealedFitOOFPredictionPanel)
        arrays["fold_assignments"] = np.asarray(panel.fold_assignments, dtype="<i8")
        arrays["fold_assignment_sha256"] = _string_scalar(
            panel.fold_assignment_sha256
        )
    else:
        assert isinstance(panel, SealedSelectionPredictionPanel)
        arrays["per_fold_probabilities"] = np.asarray(
            panel.per_fold_probabilities, dtype="<f8"
        )
        arrays["per_fold_probability_sha256"] = _string_scalar(
            panel.per_fold_probability_sha256
        )
    return arrays


def _context_summary(panel: object) -> dict[str, object]:
    counts = np.asarray(panel.context_count)
    strategy_nonempty = np.asarray(panel.strategy_context_nonempty)
    dialogue_eligible = np.asarray(panel.dialogue_history_eligible)
    return {
        "strategy_context_tensor_rank": int(counts.ndim),
        "strategy_context_count_total": int(counts.sum()),
        "strategy_context_count_minimum": int(counts.min()),
        "strategy_context_count_maximum": int(counts.max()),
        "strategy_context_nonempty_count": int(strategy_nonempty.sum()),
        "dialogue_history_eligible_count": int(dialogue_eligible.sum()),
        "zero_strategy_consumption": bool(
            not np.any(counts) and not np.any(strategy_nonempty)
        ),
    }


def _public_receipt(
    panel: SealedFitOOFPredictionPanel | SealedSelectionPredictionPanel,
    *,
    artifact_file_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": PUBLIC_RECEIPT_SCHEMA,
        "artifact_schema_version": (
            FIT_ARTIFACT_SCHEMA
            if panel.role == FIT_ROLE
            else SELECTION_ARTIFACT_SCHEMA
        ),
        "role": panel.role,
        "prediction_semantics": (
            "five_seed_group_disjoint_five_fold_oof"
            if panel.role == FIT_ROLE
            else "five_seed_by_five_fold_predictions_and_live_fold_mean"
        ),
        "dataset_id": panel.dataset_id,
        "model_id": panel.model_id,
        "model_namespace": panel.model_namespace,
        "strategy_id": panel.strategy_id,
        "context_role": panel.context_role,
        "training_seed_ids": list(panel.training_seed_ids),
        "training_seed_count": len(panel.training_seed_ids),
        "fold_count": panel.fold_count,
        "entry_count": panel.entry_count,
        "query_count": panel.query_count,
        "class_count": panel.class_count,
        "checkpoint_manifest_sha256": panel.checkpoint_manifest_sha256,
        "checkpoint_manifest_file_sha256": panel.checkpoint_manifest_file_sha256,
        "fit_training_capability_sha256": panel.fit_training_capability_sha256,
        "fit_feature_capability_sha256": panel.fit_feature_capability_sha256,
        "crossfit_plan_sha256": panel.crossfit_plan_sha256,
        "source_capability_sha256": panel.source_capability_sha256,
        "cross_role_feature_roster_sha256": panel.cross_role_feature_roster_sha256,
        "source_content_sha256": panel.source_content_sha256,
        "source_row_alignment_sha256": panel.source_row_alignment_sha256,
        "class_order_sha256": panel.class_order_sha256,
        "query_roster_sha256": panel.query_roster_sha256,
        "group_roster_sha256": panel.group_roster_sha256,
        "probability_sha256": panel.probability_sha256,
        "per_fold_probability_sha256": panel.per_fold_probability_sha256,
        "fold_assignment_sha256": panel.fold_assignment_sha256,
        "context_count_sha256": panel.context_count_sha256,
        "strategy_context_nonempty_sha256": (
            panel.strategy_context_nonempty_sha256
        ),
        "dialogue_history_eligible_sha256": (
            panel.dialogue_history_eligible_sha256
        ),
        "fold_prediction_roster_sha256": panel.fold_prediction_roster_sha256,
        "checkpoint_entry_roster_sha256": panel.checkpoint_entry_roster_sha256,
        "strategy_context_roster_manifest_sha256": (
            panel.strategy_context_roster_manifest_sha256
        ),
        "dialogue_eligibility_roster_manifest_sha256": (
            panel.dialogue_eligibility_roster_manifest_sha256
        ),
        "source_roster_sha256": panel.source_roster_sha256,
        "processor_roster_sha256": panel.processor_roster_sha256,
        "panel_sha256": panel.panel_sha256,
        "private_artifact_file_sha256": _sha256(
            artifact_file_sha256, name="private_artifact_file_sha256"
        ),
        "context_summary": _context_summary(panel),
        "privacy_contract": dict(_PRIVACY_CONTRACT),
    }


def validate_public_prediction_receipt(receipt: object) -> dict[str, object]:
    if not isinstance(receipt, dict) or set(receipt) != _PUBLIC_RECEIPT_KEYS:
        raise HarmBenchPredictionArtifactError("public prediction receipt schema changed")
    if receipt["schema_version"] != PUBLIC_RECEIPT_SCHEMA:
        raise HarmBenchPredictionArtifactError("public receipt version changed")
    role = receipt["role"]
    if role not in {FIT_ROLE, SELECTION_ROLE}:
        raise HarmBenchPredictionArtifactError("public receipt role changed")
    expected_schema = (
        FIT_ARTIFACT_SCHEMA if role == FIT_ROLE else SELECTION_ARTIFACT_SCHEMA
    )
    expected_semantics = (
        "five_seed_group_disjoint_five_fold_oof"
        if role == FIT_ROLE
        else "five_seed_by_five_fold_predictions_and_live_fold_mean"
    )
    if (
        receipt["artifact_schema_version"] != expected_schema
        or receipt["prediction_semantics"] != expected_semantics
    ):
        raise HarmBenchPredictionArtifactError("prediction semantics changed")
    for name in ("dataset_id", "model_id", "model_namespace", "strategy_id", "context_role"):
        _identifier(receipt[name], name=name)
    seeds = receipt["training_seed_ids"]
    if seeds != list(EXPECTED_TRAINING_SEED_IDS):
        raise HarmBenchPredictionArtifactError("training seed roster/order changed")
    if (
        receipt["training_seed_count"] != 5
        or receipt["fold_count"] != 5
        or receipt["entry_count"] != 25
        or type(receipt["query_count"]) is not int
        or receipt["query_count"] < 1
        or type(receipt["class_count"]) is not int
        or receipt["class_count"] < 2
    ):
        raise HarmBenchPredictionArtifactError("public receipt counts changed")
    sha_fields = _PUBLIC_RECEIPT_KEYS - {
        "schema_version",
        "artifact_schema_version",
        "role",
        "prediction_semantics",
        "dataset_id",
        "model_id",
        "model_namespace",
        "strategy_id",
        "context_role",
        "training_seed_ids",
        "training_seed_count",
        "fold_count",
        "entry_count",
        "query_count",
        "class_count",
        "per_fold_probability_sha256",
        "fold_assignment_sha256",
        "context_summary",
        "privacy_contract",
    }
    for name in sha_fields:
        _sha256(receipt[name], name=name)
    if role == FIT_ROLE:
        _sha256(receipt["fold_assignment_sha256"], name="fold_assignment_sha256")
        if receipt["per_fold_probability_sha256"] is not None:
            raise HarmBenchPredictionArtifactError(
                "fit receipt cannot declare selection per-fold probabilities"
            )
    else:
        _sha256(
            receipt["per_fold_probability_sha256"],
            name="per_fold_probability_sha256",
        )
        if receipt["fold_assignment_sha256"] is not None:
            raise HarmBenchPredictionArtifactError(
                "selection receipt cannot declare fit fold assignments"
            )
    summary = receipt["context_summary"]
    if not isinstance(summary, dict) or set(summary) != _CONTEXT_SUMMARY_KEYS:
        raise HarmBenchPredictionArtifactError("context summary schema changed")
    expected_rank = 2 if role == FIT_ROLE else 3
    if (
        type(summary["strategy_context_tensor_rank"]) is not int
        or summary["strategy_context_tensor_rank"] != expected_rank
        or any(
            type(summary[name]) is not int or summary[name] < 0
            for name in (
                "strategy_context_count_total",
                "strategy_context_count_minimum",
                "strategy_context_count_maximum",
                "strategy_context_nonempty_count",
                "dialogue_history_eligible_count",
            )
        )
        or type(summary["zero_strategy_consumption"]) is not bool
        or summary["strategy_context_count_minimum"]
        > summary["strategy_context_count_maximum"]
        or summary["dialogue_history_eligible_count"] > receipt["query_count"]
    ):
        raise HarmBenchPredictionArtifactError("context summary values changed")
    maximum_strategy_coverage = (
        len(EXPECTED_TRAINING_SEED_IDS)
        * (1 if role == FIT_ROLE else EXPECTED_FOLD_COUNT)
        * receipt["query_count"]
    )
    zero_consumption = (
        summary["strategy_context_count_total"] == 0
        and summary["strategy_context_nonempty_count"] == 0
    )
    if (
        summary["strategy_context_nonempty_count"] > maximum_strategy_coverage
        or summary["strategy_context_count_total"]
        < summary["strategy_context_nonempty_count"]
        or summary["zero_strategy_consumption"] is not zero_consumption
        or (
            zero_consumption
            and (
                summary["strategy_context_count_minimum"] != 0
                or summary["strategy_context_count_maximum"] != 0
            )
        )
    ):
        raise HarmBenchPredictionArtifactError(
            "context summary coverage/consumption semantics changed"
        )
    namespace = receipt["model_namespace"]
    if namespace == CURRENT_ONLY_NAMESPACE:
        if (
            receipt["strategy_id"] != CURRENT_ONLY_STRATEGY_ID
            or summary["zero_strategy_consumption"] is not True
            or summary["strategy_context_count_total"] != 0
            or summary["strategy_context_nonempty_count"] != 0
        ):
            raise HarmBenchPredictionArtifactError(
                "current-only receipt lost its zero-consumption proof"
            )
    elif namespace == HISTORY_NAMESPACE:
        if receipt["strategy_id"] not in STRICT_PAST_STRATEGY_IDS:
            raise HarmBenchPredictionArtifactError(
                "history receipt strategy is not strict-past"
            )
    else:
        raise HarmBenchPredictionArtifactError("model namespace changed")
    if receipt["privacy_contract"] != _PRIVACY_CONTRACT:
        raise HarmBenchPredictionArtifactError("public receipt privacy contract changed")
    return dict(receipt)


def public_prediction_receipt_sha256(receipt: object) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(validate_public_prediction_receipt(receipt))
    ).hexdigest()


def _is_reparse(stat_result: os.stat_result) -> bool:
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
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
            raise HarmBenchPredictionArtifactError(
                "prediction artifact path contains a symlink or reparse point"
            )


def _plain_file_stat(path: Path, *, name: str) -> os.stat_result:
    try:
        observed = path.lstat()
    except OSError as error:
        raise HarmBenchPredictionArtifactError(f"cannot stat exact {name}") from error
    if stat.S_ISLNK(observed.st_mode) or _is_reparse(observed):
        raise HarmBenchPredictionArtifactError(
            f"{name} cannot be a symlink or reparse point"
        )
    if not stat.S_ISREG(observed.st_mode):
        raise HarmBenchPredictionArtifactError(f"{name} must be a plain file")
    return observed


def _same_file_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        int(first.st_dev),
        int(first.st_ino),
        int(first.st_size),
        int(getattr(first, "st_mtime_ns", 0)),
    ) == (
        int(second.st_dev),
        int(second.st_ino),
        int(second.st_size),
        int(getattr(second, "st_mtime_ns", 0)),
    )


def _assert_path_still_names_handle(
    path: Path, handle_stat: os.stat_result, *, name: str
) -> None:
    _reject_reparse_components(path)
    if not _same_file_identity(_plain_file_stat(path, name=name), handle_stat):
        raise HarmBenchPredictionArtifactError(
            f"{name} changed identity during verified read"
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


def validate_private_root(path: str | Path) -> Path:
    raw = Path(path)
    if not raw.is_absolute() or raw.is_symlink():
        raise HarmBenchPredictionArtifactError(
            "private root must be an explicit absolute non-symlink directory"
        )
    _reject_reparse_components(raw)
    try:
        raw_stat = raw.lstat()
        if stat.S_ISLNK(raw_stat.st_mode) or _is_reparse(raw_stat):
            raise HarmBenchPredictionArtifactError(
                "private root cannot be a symlink or reparse point"
            )
        root = raw.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise HarmBenchPredictionArtifactError(
            "private root must already exist"
        ) from error
    if not root.is_dir() or root == Path(root.anchor):
        raise HarmBenchPredictionArtifactError(
            "private root must be a safe non-root directory"
        )
    if _is_within(root, _repository_root().resolve()) or _is_within(
        root, _home_root().resolve()
    ):
        raise HarmBenchPredictionArtifactError(
            "private root must be outside both the repository and user home"
        )
    return root


def _validated_destination(
    private_root: str | Path,
    destination: str | Path,
    *,
    suffix: str,
    name: str,
) -> tuple[Path, Path]:
    root = validate_private_root(private_root)
    raw = Path(destination)
    if not raw.is_absolute():
        raise HarmBenchPredictionArtifactError(f"{name} must be an absolute path")
    _reject_reparse_components(raw)
    resolved = raw.resolve(strict=False)
    if resolved.parent != root:
        raise HarmBenchPredictionArtifactError(
            f"{name} must be a direct child of the explicit private root"
        )
    if resolved.suffix.lower() != suffix:
        raise HarmBenchPredictionArtifactError(f"{name} must use the {suffix} suffix")
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError(f"write-once destination already exists: {resolved.name}")
    return root, resolved


def _validated_existing_private_file(
    private_root: str | Path,
    source: str | Path,
    *,
    suffix: str,
    name: str,
) -> tuple[Path, Path, os.stat_result]:
    root = validate_private_root(private_root)
    raw = Path(source)
    if not raw.is_absolute() or raw.is_symlink():
        raise HarmBenchPredictionArtifactError(f"{name} must be an absolute plain file")
    _reject_reparse_components(raw)
    try:
        parent = raw.parent.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise HarmBenchPredictionArtifactError(f"{name} does not exist") from error
    source_stat = _plain_file_stat(raw, name=name)
    if parent != root or raw.suffix.lower() != suffix:
        raise HarmBenchPredictionArtifactError(
            f"{name} must be a {suffix} file directly under the private root"
        )
    return root, raw, source_stat


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


def _read_string_scalar(value: object, *, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise HarmBenchPredictionArtifactError(f"{name} must be one string scalar")
    return str(array.item())


def _read_int_scalar(value: object, *, name: str) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.dtype("int64"):
        raise HarmBenchPredictionArtifactError(f"{name} must be one int64 scalar")
    return int(array.item())


def _read_sha_vector(value: object, *, name: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if (
        array.shape != (EXPECTED_CHECKPOINT_ENTRY_COUNT,)
        or array.dtype.kind not in {"U", "S"}
    ):
        raise HarmBenchPredictionArtifactError(f"{name} must contain exactly 25 SHA values")
    result = tuple(str(item) for item in array.tolist())
    for item in result:
        _sha256(item, name=name)
    return result


def _decode_receipt(
    path: Path,
    *,
    expected_receipt_sha256: str,
    expected_path_identity: os.stat_result,
) -> dict[str, object]:
    expected = _sha256(expected_receipt_sha256, name="expected_receipt_sha256")
    before_path = _plain_file_stat(path, name="public prediction receipt")
    if not _same_file_identity(before_path, expected_path_identity):
        raise HarmBenchPredictionArtifactError(
            "public receipt changed after path validation"
        )

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HarmBenchPredictionArtifactError(
                    f"public receipt contains duplicate key: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise HarmBenchPredictionArtifactError(
            f"public receipt contains invalid JSON constant: {value}"
        )

    try:
        with path.open("rb") as handle:
            before_handle = os.fstat(handle.fileno())
            if not _same_file_identity(before_path, before_handle):
                raise HarmBenchPredictionArtifactError(
                    "public receipt changed before verified read"
                )
            first_sha = _hash_open_handle(handle)
            if first_sha != expected:
                raise HarmBenchPredictionArtifactError("public receipt SHA-256 changed")
            handle.seek(0)
            encoded = handle.read()
            try:
                payload = json.loads(
                    encoded.decode("utf-8"),
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=reject_constant,
                )
            except HarmBenchPredictionArtifactError:
                raise
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise HarmBenchPredictionArtifactError(
                    "public receipt is not strict UTF-8 JSON"
                ) from error
            validated = validate_public_prediction_receipt(payload)
            if encoded != _canonical_json_bytes(validated):
                raise HarmBenchPredictionArtifactError(
                    "public receipt is not canonical JSON"
                )
            handle.seek(0)
            second_sha = _hash_open_handle(handle)
            after_handle = os.fstat(handle.fileno())
    except HarmBenchPredictionArtifactError:
        raise
    except OSError as error:
        raise HarmBenchPredictionArtifactError("public receipt cannot be read") from error
    if first_sha != second_sha or not _same_file_identity(before_handle, after_handle):
        raise HarmBenchPredictionArtifactError(
            "public receipt changed during verified read"
        )
    _assert_path_still_names_handle(
        path, after_handle, name="public prediction receipt"
    )
    return validated


def _load_npz_arrays(
    path: Path,
    *,
    expected_path_identity: os.stat_result,
    expected_file_sha256: str,
) -> dict[str, np.ndarray]:
    expected_sha = _sha256(expected_file_sha256, name="private_artifact_file_sha256")
    before_path = _plain_file_stat(path, name="private prediction NPZ")
    if not _same_file_identity(before_path, expected_path_identity):
        raise HarmBenchPredictionArtifactError(
            "private prediction NPZ changed after path validation"
        )
    try:
        with path.open("rb") as handle:
            before_handle = os.fstat(handle.fileno())
            if not _same_file_identity(before_path, before_handle):
                raise HarmBenchPredictionArtifactError(
                    "private prediction NPZ changed before verified read"
                )
            first_sha = _hash_open_handle(handle)
            if first_sha != expected_sha:
                raise HarmBenchPredictionArtifactError(
                    "private prediction NPZ file SHA-256 changed"
                )
            handle.seek(0)
            with np.load(handle, allow_pickle=False) as archive:
                names = list(archive.files)
                if len(names) != len(set(names)):
                    raise HarmBenchPredictionArtifactError(
                        "private NPZ contains duplicate members"
                    )
                arrays = {name: np.asarray(archive[name]).copy() for name in names}
            handle.seek(0)
            second_sha = _hash_open_handle(handle)
            after_handle = os.fstat(handle.fileno())
    except HarmBenchPredictionArtifactError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise HarmBenchPredictionArtifactError(
            "private NPZ is unreadable with allow_pickle=False"
        ) from error
    if first_sha != second_sha or not _same_file_identity(before_handle, after_handle):
        raise HarmBenchPredictionArtifactError(
            "private prediction NPZ changed during verified load"
        )
    _assert_path_still_names_handle(path, after_handle, name="private prediction NPZ")
    if any(array.dtype.kind == "O" for array in arrays.values()):
        raise HarmBenchPredictionArtifactError("private NPZ contains an object array")
    return arrays


@dataclass(frozen=True)
class LoadedPredictionArtifact:
    """Loader-only capability for one live-verifiable prediction artifact pair."""

    role: str
    dataset_id: str
    model_id: str
    model_namespace: str
    strategy_id: str
    context_role: str
    training_seed_ids: tuple[int, ...]
    fold_count: int
    entry_count: int
    checkpoint_manifest_sha256: str
    checkpoint_manifest_file_sha256: str
    class_order_sha256: str
    panel_sha256: str
    probabilities: np.ndarray
    per_fold_probabilities: np.ndarray | None
    query_protocol_row_ids: np.ndarray
    group_tokens: np.ndarray
    class_tokens: np.ndarray
    fold_assignments: np.ndarray | None
    context_count: np.ndarray
    strategy_context_nonempty: np.ndarray
    dialogue_history_eligible: np.ndarray
    receipt: Mapping[str, object]
    private_root: Path
    artifact_path: Path
    receipt_path: Path
    artifact_file_sha256: str
    receipt_file_sha256: str
    _checkpoint_manifest: VerifiedCheckpointManifest = field(
        repr=False, compare=False
    )
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _LOADED_SEAL:
            raise HarmBenchPredictionArtifactError(
                "loaded prediction artifacts can only be created by the verified loader"
            )
        for name in ("private_root", "artifact_path", "receipt_path"):
            path = getattr(self, name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise HarmBenchPredictionArtifactError(
                    f"loaded prediction {name} must be an absolute Path"
                )
        _sha256(self.artifact_file_sha256, name="artifact_file_sha256")
        _sha256(self.receipt_file_sha256, name="receipt_file_sha256")
        arrays = (
            self.probabilities,
            self.query_protocol_row_ids,
            self.group_tokens,
            self.class_tokens,
            self.context_count,
            self.strategy_context_nonempty,
            self.dialogue_history_eligible,
        )
        optional = (self.per_fold_probabilities, self.fold_assignments)
        if any(np.asarray(array).flags.writeable for array in arrays) or any(
            array is not None and np.asarray(array).flags.writeable
            for array in optional
        ):
            raise HarmBenchPredictionArtifactError(
                "loaded prediction artifact arrays must be immutable"
            )


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(child) for child in value)
    return value


def _validate_loaded_arrays(
    arrays: Mapping[str, np.ndarray],
    receipt: Mapping[str, object],
    verified: VerifiedCheckpointManifest,
    *,
    private_root: Path,
    artifact_path: Path,
    receipt_path: Path,
    receipt_file_sha256: str,
) -> LoadedPredictionArtifact:
    if "role" not in arrays or "schema_version" not in arrays:
        raise HarmBenchPredictionArtifactError("private NPZ is missing schema controls")
    role = _read_string_scalar(arrays["role"], name="role")
    schema = _read_string_scalar(arrays["schema_version"], name="schema_version")
    expected_keys = _FIT_NPZ_KEYS if role == FIT_ROLE else _SELECTION_NPZ_KEYS
    expected_schema = FIT_ARTIFACT_SCHEMA if role == FIT_ROLE else SELECTION_ARTIFACT_SCHEMA
    if role not in {FIT_ROLE, SELECTION_ROLE} or schema != expected_schema:
        raise HarmBenchPredictionArtifactError("private NPZ role/schema pair changed")
    if set(arrays) != expected_keys:
        raise HarmBenchPredictionArtifactError("private NPZ exact schema changed")
    manifest = verified.manifest
    string_fields = (
        "dataset_id",
        "model_id",
        "model_namespace",
        "strategy_id",
        "context_role",
        "checkpoint_manifest_sha256",
        "checkpoint_manifest_file_sha256",
        "fit_training_capability_sha256",
        "fit_feature_capability_sha256",
        "crossfit_plan_sha256",
        "source_capability_sha256",
        "cross_role_feature_roster_sha256",
        "source_content_sha256",
        "source_row_alignment_sha256",
        "class_order_sha256",
        "query_roster_sha256",
        "group_roster_sha256",
        "probability_sha256",
        "context_count_sha256",
        "strategy_context_nonempty_sha256",
        "dialogue_history_eligible_sha256",
        "fold_prediction_roster_sha256",
        "checkpoint_entry_roster_sha256",
        "strategy_context_roster_manifest_sha256",
        "dialogue_eligibility_roster_manifest_sha256",
        "source_roster_sha256",
        "processor_roster_sha256",
        "panel_sha256",
    )
    strings = {
        name: _read_string_scalar(arrays[name], name=name) for name in string_fields
    }
    for name in string_fields:
        if strings[name] != receipt[name]:
            raise HarmBenchPredictionArtifactError(f"{name} differs from public receipt")
    manifest_bindings = {
        "dataset_id": manifest.dataset_id,
        "model_id": manifest.model_id,
        "model_namespace": manifest.model_namespace,
        "checkpoint_manifest_sha256": manifest.manifest_sha256,
        "checkpoint_manifest_file_sha256": verified.manifest_file_sha256,
        "fit_training_capability_sha256": manifest.fit_training_capability_sha256,
        "fit_feature_capability_sha256": manifest.fit_feature_capability_sha256,
        "crossfit_plan_sha256": manifest.crossfit_plan_sha256,
        "class_order_sha256": manifest.class_order_sha256,
    }
    if any(strings[name] != value for name, value in manifest_bindings.items()):
        raise HarmBenchPredictionArtifactError(
            "private prediction lineage differs from sealed checkpoint manifest"
        )
    seeds = np.asarray(arrays["training_seed_ids"])
    if seeds.dtype != np.dtype("int64") or seeds.tolist() != list(
        EXPECTED_TRAINING_SEED_IDS
    ):
        raise HarmBenchPredictionArtifactError("private training seed roster changed")
    if (
        _read_int_scalar(arrays["fold_count"], name="fold_count") != 5
        or _read_int_scalar(arrays["entry_count"], name="entry_count") != 25
    ):
        raise HarmBenchPredictionArtifactError("private fold/entry count changed")
    classes = np.asarray(arrays["class_tokens"])
    if classes.dtype.kind not in {"U", "S"} or tuple(classes.tolist()) != tuple(
        manifest.ordered_class_tokens
    ):
        raise HarmBenchPredictionArtifactError("private class order changed")
    queries = np.asarray(arrays["query_protocol_row_ids"])
    groups = np.asarray(arrays["group_tokens"])
    if (
        queries.dtype != np.dtype("int64")
        or queries.ndim != 1
        or not len(queries)
        or len(set(queries.tolist())) != len(queries)
        or groups.ndim != 1
        or groups.dtype.kind not in {"U", "S"}
        or len(groups) != len(queries)
    ):
        raise HarmBenchPredictionArtifactError("private query/group alignment changed")
    q, c = len(queries), len(classes)
    probability = _validate_probability_tensor(
        arrays["probabilities"], shape=(5, q, c), name="probabilities"
    )
    counts = np.asarray(arrays["context_count"])
    strategy_nonempty = np.asarray(arrays["strategy_context_nonempty"])
    dialogue_eligible = np.asarray(arrays["dialogue_history_eligible"])
    expected_context_shape = (5, q) if role == FIT_ROLE else (5, 5, q)
    if (
        counts.dtype != np.dtype("int64")
        or counts.shape != expected_context_shape
        or np.any(counts < 0)
        or strategy_nonempty.dtype != np.dtype("bool")
        or strategy_nonempty.shape != expected_context_shape
        or not np.array_equal(strategy_nonempty, counts > 0)
        or dialogue_eligible.dtype != np.dtype("bool")
        or dialogue_eligible.shape != (q,)
    ):
        raise HarmBenchPredictionArtifactError("private context tensors changed")
    common_broadcast = dialogue_eligible.reshape(
        (1,) * (counts.ndim - 1) + (q,)
    )
    if np.any(strategy_nonempty & ~common_broadcast):
        raise HarmBenchPredictionArtifactError(
            "private strategy coverage exceeds dialogue eligibility"
        )
    fold_receipts = _read_sha_vector(
        arrays["fold_prediction_receipt_sha256_by_entry"],
        name="fold_prediction_receipt_sha256_by_entry",
    )
    entry_receipts = _read_sha_vector(
        arrays["checkpoint_entry_sha256_by_entry"],
        name="checkpoint_entry_sha256_by_entry",
    )
    strategy_context_receipts = _read_sha_vector(
        arrays["strategy_context_roster_sha256_by_entry"],
        name="strategy_context_roster_sha256_by_entry",
    )
    dialogue_eligibility_receipts = _read_sha_vector(
        arrays["dialogue_eligibility_roster_sha256_by_entry"],
        name="dialogue_eligibility_roster_sha256_by_entry",
    )
    processor_receipts = _read_sha_vector(
        arrays["processor_receipt_sha256_by_entry"],
        name="processor_receipt_sha256_by_entry",
    )
    output_receipts = _read_sha_vector(
        arrays["processed_output_receipt_sha256_by_entry"],
        name="processed_output_receipt_sha256_by_entry",
    )
    if (
        len(set(fold_receipts)) != 25
        or len(set(strategy_context_receipts)) != 25
        or len(set(dialogue_eligibility_receipts)) != 25
    ):
        raise HarmBenchPredictionArtifactError(
            "private fold/context/eligibility roster is duplicated"
        )
    if entry_receipts != tuple(entry.entry_sha256 for entry in manifest.entries):
        raise HarmBenchPredictionArtifactError("private checkpoint entry roster changed")
    if processor_receipts != tuple(
        entry.processor_receipt_sha256 for entry in manifest.entries
    ):
        raise HarmBenchPredictionArtifactError("private processor roster changed")
    if role == FIT_ROLE and output_receipts != tuple(
        entry.processed_output_receipt_sha256 for entry in manifest.entries
    ):
        raise HarmBenchPredictionArtifactError(
            "fit processed output roster differs from manifest"
        )
    if manifest.model_namespace == CURRENT_ONLY_NAMESPACE:
        if (
            strings["strategy_id"] != CURRENT_ONLY_STRATEGY_ID
            or np.any(counts)
            or np.any(strategy_nonempty)
        ):
            raise HarmBenchPredictionArtifactError(
                "current-only private artifact lost zero-consumption proof"
            )
    elif (
        manifest.model_namespace != HISTORY_NAMESPACE
        or strings["strategy_id"] not in STRICT_PAST_STRATEGY_IDS
    ):
        raise HarmBenchPredictionArtifactError("private namespace/strategy changed")
    per_fold: np.ndarray | None = None
    assignments: np.ndarray | None = None
    if role == FIT_ROLE:
        assignments = np.asarray(arrays["fold_assignments"])
        fold_assignment_sha = _read_string_scalar(
            arrays["fold_assignment_sha256"], name="fold_assignment_sha256"
        )
        if (
            assignments.dtype != np.dtype("int64")
            or assignments.shape != (5, q)
            or np.any(assignments < 0)
            or np.any(assignments >= 5)
            or fold_assignment_sha != receipt["fold_assignment_sha256"]
            or _array_sha256(assignments) != fold_assignment_sha
        ):
            raise HarmBenchPredictionArtifactError("fit fold assignments changed")
        for seed_index in range(5):
            if set(assignments[seed_index].tolist()) != set(range(5)):
                raise HarmBenchPredictionArtifactError(
                    "fit seed no longer covers five folds"
                )
            for group in set(groups.tolist()):
                if len(set(assignments[seed_index][groups == group].tolist())) != 1:
                    raise HarmBenchPredictionArtifactError(
                        "fit group crosses folds in private artifact"
                    )
        per_fold_sha: str | None = None
        fold_assignment_value: str | None = fold_assignment_sha
    else:
        per_fold = _validate_probability_tensor(
            arrays["per_fold_probabilities"],
            shape=(5, 5, q, c),
            name="per_fold_probabilities",
        )
        per_fold_sha = _read_string_scalar(
            arrays["per_fold_probability_sha256"],
            name="per_fold_probability_sha256",
        )
        if (
            per_fold_sha != receipt["per_fold_probability_sha256"]
            or _array_sha256(per_fold) != per_fold_sha
            or not np.array_equal(
                probability, per_fold.mean(axis=1, dtype=np.float64)
            )
        ):
            raise HarmBenchPredictionArtifactError(
                "selection fold mean or per-fold tensor changed"
            )
        fold_assignment_value = None
    derived = {
        "query_roster_sha256": _array_sha256(queries),
        "group_roster_sha256": _array_sha256(groups),
        "probability_sha256": _array_sha256(probability),
        "context_count_sha256": _array_sha256(counts),
        "strategy_context_nonempty_sha256": _array_sha256(strategy_nonempty),
        "dialogue_history_eligible_sha256": _array_sha256(dialogue_eligible),
        "fold_prediction_roster_sha256": _roster_sha(
            "harmbench_erc_fold_prediction_roster_v1", fold_receipts
        ),
        "checkpoint_entry_roster_sha256": _roster_sha(
            "harmbench_erc_checkpoint_entry_roster_v1", entry_receipts
        ),
        "strategy_context_roster_manifest_sha256": _roster_sha(
            "harmbench_erc_prediction_strategy_context_roster_v1",
            strategy_context_receipts,
        ),
        "dialogue_eligibility_roster_manifest_sha256": _roster_sha(
            "harmbench_erc_prediction_dialogue_eligibility_roster_v1",
            dialogue_eligibility_receipts,
        ),
        "source_roster_sha256": _canonical_sha256(
            {
                "schema_version": "harmbench_erc_prediction_source_roster_v1",
                "source_capability_sha256": strings["source_capability_sha256"],
                "cross_role_feature_roster_sha256": strings[
                    "cross_role_feature_roster_sha256"
                ],
                "source_content_sha256": strings["source_content_sha256"],
                "source_row_alignment_sha256": strings[
                    "source_row_alignment_sha256"
                ],
                "query_roster_sha256": _array_sha256(queries),
                "group_roster_sha256": _array_sha256(groups),
            }
        ),
        "processor_roster_sha256": _canonical_sha256(
            {
                "schema_version": "harmbench_erc_prediction_processor_roster_v1",
                "entries": [
                    {
                        "training_seed": seed,
                        "fold": fold,
                        "processor_receipt_sha256": processor,
                        "processed_output_receipt_sha256": output,
                    }
                    for (seed, fold), processor, output in zip(
                        _expected_pairs(),
                        processor_receipts,
                        output_receipts,
                        strict=True,
                    )
                ],
            }
        ),
    }
    if any(strings[name] != value for name, value in derived.items()):
        raise HarmBenchPredictionArtifactError(
            "private prediction aggregate binding changed"
        )
    panel_values: dict[str, object] = {
        "schema_version": FIT_PANEL_SCHEMA if role == FIT_ROLE else SELECTION_PANEL_SCHEMA,
        "role": role,
        "dataset_id": strings["dataset_id"],
        "model_id": strings["model_id"],
        "model_namespace": strings["model_namespace"],
        "checkpoint_manifest_sha256": strings["checkpoint_manifest_sha256"],
        "checkpoint_manifest_file_sha256": strings[
            "checkpoint_manifest_file_sha256"
        ],
        "training_seed_ids": EXPECTED_TRAINING_SEED_IDS,
        "fold_count": 5,
        "entry_count": 25,
        "fit_training_capability_sha256": strings[
            "fit_training_capability_sha256"
        ],
        "fit_feature_capability_sha256": strings["fit_feature_capability_sha256"],
        "crossfit_plan_sha256": strings["crossfit_plan_sha256"],
        "source_capability_sha256": strings["source_capability_sha256"],
        "cross_role_feature_roster_sha256": strings[
            "cross_role_feature_roster_sha256"
        ],
        "source_content_sha256": strings["source_content_sha256"],
        "source_row_alignment_sha256": strings["source_row_alignment_sha256"],
        "ordered_class_tokens": tuple(classes.tolist()),
        "class_order_sha256": strings["class_order_sha256"],
        "context_role": strings["context_role"],
        "strategy_id": strings["strategy_id"],
        "query_count": q,
        "class_count": c,
        "query_roster_sha256": derived["query_roster_sha256"],
        "group_roster_sha256": derived["group_roster_sha256"],
        "probability_sha256": derived["probability_sha256"],
        "per_fold_probability_sha256": per_fold_sha,
        "fold_assignment_sha256": fold_assignment_value,
        "context_count_sha256": derived["context_count_sha256"],
        "strategy_context_nonempty_sha256": derived[
            "strategy_context_nonempty_sha256"
        ],
        "dialogue_history_eligible_sha256": derived[
            "dialogue_history_eligible_sha256"
        ],
        "fold_prediction_roster_sha256": derived[
            "fold_prediction_roster_sha256"
        ],
        "checkpoint_entry_roster_sha256": derived[
            "checkpoint_entry_roster_sha256"
        ],
        "strategy_context_roster_manifest_sha256": derived[
            "strategy_context_roster_manifest_sha256"
        ],
        "dialogue_eligibility_roster_manifest_sha256": derived[
            "dialogue_eligibility_roster_manifest_sha256"
        ],
        "source_roster_sha256": derived["source_roster_sha256"],
        "processor_roster_sha256": derived["processor_roster_sha256"],
    }
    if _canonical_sha256(_panel_descriptor(panel_values)) != strings["panel_sha256"]:
        raise HarmBenchPredictionArtifactError("private panel receipt changed")
    summary = {
        "strategy_context_tensor_rank": int(counts.ndim),
        "strategy_context_count_total": int(counts.sum()),
        "strategy_context_count_minimum": int(counts.min()),
        "strategy_context_count_maximum": int(counts.max()),
        "strategy_context_nonempty_count": int(strategy_nonempty.sum()),
        "dialogue_history_eligible_count": int(dialogue_eligible.sum()),
        "zero_strategy_consumption": bool(
            not np.any(counts) and not np.any(strategy_nonempty)
        ),
    }
    if receipt["context_summary"] != summary:
        raise HarmBenchPredictionArtifactError("public/private context summary changed")
    readonly_probability = _readonly(probability, dtype=np.float64)
    readonly_queries = _readonly(queries, dtype=np.int64)
    readonly_groups = _readonly(groups, dtype=str)
    readonly_classes = _readonly(classes, dtype=str)
    readonly_counts = _readonly(counts, dtype=np.int64)
    readonly_strategy_nonempty = _readonly(strategy_nonempty, dtype=np.bool_)
    readonly_dialogue_eligible = _readonly(dialogue_eligible, dtype=np.bool_)
    return LoadedPredictionArtifact(
        role=role,
        dataset_id=strings["dataset_id"],
        model_id=strings["model_id"],
        model_namespace=strings["model_namespace"],
        strategy_id=strings["strategy_id"],
        context_role=strings["context_role"],
        training_seed_ids=EXPECTED_TRAINING_SEED_IDS,
        fold_count=5,
        entry_count=25,
        checkpoint_manifest_sha256=strings["checkpoint_manifest_sha256"],
        checkpoint_manifest_file_sha256=strings[
            "checkpoint_manifest_file_sha256"
        ],
        class_order_sha256=strings["class_order_sha256"],
        panel_sha256=strings["panel_sha256"],
        probabilities=readonly_probability,
        per_fold_probabilities=(
            None if per_fold is None else _readonly(per_fold, dtype=np.float64)
        ),
        query_protocol_row_ids=readonly_queries,
        group_tokens=readonly_groups,
        class_tokens=readonly_classes,
        fold_assignments=(
            None if assignments is None else _readonly(assignments, dtype=np.int64)
        ),
        context_count=readonly_counts,
        strategy_context_nonempty=readonly_strategy_nonempty,
        dialogue_history_eligible=readonly_dialogue_eligible,
        receipt=_deep_freeze(receipt),
        private_root=private_root,
        artifact_path=artifact_path,
        receipt_path=receipt_path,
        artifact_file_sha256=str(receipt["private_artifact_file_sha256"]),
        receipt_file_sha256=receipt_file_sha256,
        _checkpoint_manifest=verified,
        _seal=_LOADED_SEAL,
    )


def _write_pair_once(
    *,
    private_root: str | Path,
    artifact_path: str | Path,
    receipt_path: str | Path,
    checkpoint_manifest: VerifiedCheckpointManifest,
    panel: SealedFitOOFPredictionPanel | SealedSelectionPredictionPanel,
) -> dict[str, object]:
    root, artifact = _validated_destination(
        private_root, artifact_path, suffix=".npz", name="artifact_path"
    )
    receipt_root, public_receipt = _validated_destination(
        private_root, receipt_path, suffix=".json", name="receipt_path"
    )
    if receipt_root != root or artifact == public_receipt:
        raise HarmBenchPredictionArtifactError(
            "artifact and receipt must be distinct files under one private root"
        )
    arrays = _panel_arrays(panel)
    artifact_handle, artifact_temporary_value = _temporary_file(root, artifact)
    artifact_temporary: Path | None = artifact_temporary_value
    receipt_temporary: Path | None = None
    try:
        with artifact_handle:
            np.savez_compressed(artifact_handle, **arrays)
            artifact_handle.flush()
            os.fsync(artifact_handle.fileno())
            artifact_handle.seek(0)
            artifact_file_sha = _hash_open_handle(artifact_handle)
        receipt = _public_receipt(
            panel, artifact_file_sha256=artifact_file_sha
        )
        validate_public_prediction_receipt(receipt)
        receipt_handle, receipt_temporary = _temporary_file(root, public_receipt)
        with receipt_handle:
            receipt_handle.write(_canonical_json_bytes(receipt))
            receipt_handle.flush()
            os.fsync(receipt_handle.fileno())
        assert artifact_temporary is not None
        _publish_once(artifact_temporary, artifact)
        artifact_temporary = None
        assert receipt_temporary is not None
        _publish_once(receipt_temporary, public_receipt)
        receipt_temporary = None
    finally:
        for temporary in (artifact_temporary, receipt_temporary):
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
    loaded = load_prediction_artifact(
        private_root=root,
        artifact_path=artifact,
        receipt_path=public_receipt,
        checkpoint_manifest=checkpoint_manifest,
        expected_receipt_sha256=public_prediction_receipt_sha256(receipt),
    )
    if loaded.panel_sha256 != panel.panel_sha256:
        raise HarmBenchPredictionArtifactError(
            "published prediction differs from sealed panel"
        )
    return dict(receipt)


def write_fit_oof_prediction_artifact(
    *,
    private_root: str | Path,
    artifact_path: str | Path,
    receipt_path: str | Path,
    checkpoint_manifest: VerifiedCheckpointManifest,
    panel: SealedFitOOFPredictionPanel,
) -> dict[str, object]:
    """Publish a live-revalidated manifest-bound fit OOF panel once."""

    verified = _verified_manifest(checkpoint_manifest)
    sealed = _revalidate_panel(verified, panel, role=FIT_ROLE)
    assert isinstance(sealed, SealedFitOOFPredictionPanel)
    return _write_pair_once(
        private_root=private_root,
        artifact_path=artifact_path,
        receipt_path=receipt_path,
        checkpoint_manifest=verified,
        panel=sealed,
    )


def write_selection_prediction_artifact(
    *,
    private_root: str | Path,
    artifact_path: str | Path,
    receipt_path: str | Path,
    checkpoint_manifest: VerifiedCheckpointManifest,
    panel: SealedSelectionPredictionPanel,
) -> dict[str, object]:
    """Publish all 25 selection folds and their live five-fold means once."""

    verified = _verified_manifest(checkpoint_manifest)
    sealed = _revalidate_panel(verified, panel, role=SELECTION_ROLE)
    assert isinstance(sealed, SealedSelectionPredictionPanel)
    return _write_pair_once(
        private_root=private_root,
        artifact_path=artifact_path,
        receipt_path=receipt_path,
        checkpoint_manifest=verified,
        panel=sealed,
    )


def load_prediction_artifact(
    *,
    private_root: str | Path,
    artifact_path: str | Path,
    receipt_path: str | Path,
    checkpoint_manifest: VerifiedCheckpointManifest,
    expected_receipt_sha256: str,
) -> LoadedPredictionArtifact:
    """Same-handle load and semantic revalidation against a sealed manifest."""

    verified = _verified_manifest(checkpoint_manifest)
    root, artifact, artifact_identity = _validated_existing_private_file(
        private_root, artifact_path, suffix=".npz", name="artifact_path"
    )
    receipt_root, receipt_file, receipt_identity = _validated_existing_private_file(
        private_root, receipt_path, suffix=".json", name="receipt_path"
    )
    if receipt_root != root:
        raise HarmBenchPredictionArtifactError(
            "artifact and receipt do not share the explicit private root"
        )
    receipt = _decode_receipt(
        receipt_file,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_path_identity=receipt_identity,
    )
    manifest = verified.manifest
    exact_manifest = {
        "dataset_id": manifest.dataset_id,
        "model_id": manifest.model_id,
        "model_namespace": manifest.model_namespace,
        "checkpoint_manifest_sha256": manifest.manifest_sha256,
        "checkpoint_manifest_file_sha256": verified.manifest_file_sha256,
        "fit_training_capability_sha256": manifest.fit_training_capability_sha256,
        "fit_feature_capability_sha256": manifest.fit_feature_capability_sha256,
        "crossfit_plan_sha256": manifest.crossfit_plan_sha256,
        "class_order_sha256": manifest.class_order_sha256,
        "entry_count": EXPECTED_CHECKPOINT_ENTRY_COUNT,
    }
    if any(receipt[name] != value for name, value in exact_manifest.items()):
        raise HarmBenchPredictionArtifactError(
            "public receipt differs from sealed checkpoint manifest"
        )
    arrays = _load_npz_arrays(
        artifact,
        expected_path_identity=artifact_identity,
        expected_file_sha256=receipt["private_artifact_file_sha256"],
    )
    return _validate_loaded_arrays(
        arrays,
        receipt,
        verified,
        private_root=root,
        artifact_path=artifact.resolve(strict=True),
        receipt_path=receipt_file.resolve(strict=True),
        receipt_file_sha256=_sha256(
            expected_receipt_sha256, name="expected_receipt_sha256"
        ),
    )


def load_fit_oof_prediction_artifact(
    *,
    private_root: str | Path,
    artifact_path: str | Path,
    receipt_path: str | Path,
    checkpoint_manifest: VerifiedCheckpointManifest,
    expected_receipt_sha256: str,
) -> LoadedPredictionArtifact:
    loaded = load_prediction_artifact(
        private_root=private_root,
        artifact_path=artifact_path,
        receipt_path=receipt_path,
        checkpoint_manifest=checkpoint_manifest,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    if loaded.role != FIT_ROLE:
        raise HarmBenchPredictionArtifactError("artifact is not a fit OOF panel")
    return loaded


def load_selection_prediction_artifact(
    *,
    private_root: str | Path,
    artifact_path: str | Path,
    receipt_path: str | Path,
    checkpoint_manifest: VerifiedCheckpointManifest,
    expected_receipt_sha256: str,
) -> LoadedPredictionArtifact:
    loaded = load_prediction_artifact(
        private_root=private_root,
        artifact_path=artifact_path,
        receipt_path=receipt_path,
        checkpoint_manifest=checkpoint_manifest,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    if loaded.role != SELECTION_ROLE:
        raise HarmBenchPredictionArtifactError(
            "artifact is not a selection ensemble panel"
        )
    return loaded


def _revalidate_loaded_prediction_artifact(
    artifact: object,
    *,
    expected_role: str | None = None,
) -> LoadedPredictionArtifact:
    """Reload a loader capability from its exact files for sealed evaluators."""

    if (
        not isinstance(artifact, LoadedPredictionArtifact)
        or artifact._seal is not _LOADED_SEAL
    ):
        raise HarmBenchPredictionArtifactError(
            "evaluator requires a loader-issued prediction artifact capability"
        )
    if expected_role is not None and artifact.role != expected_role:
        raise HarmBenchPredictionArtifactError("loaded prediction role changed")
    rebuilt = load_prediction_artifact(
        private_root=artifact.private_root,
        artifact_path=artifact.artifact_path,
        receipt_path=artifact.receipt_path,
        checkpoint_manifest=artifact._checkpoint_manifest,
        expected_receipt_sha256=artifact.receipt_file_sha256,
    )
    array_names = (
        "probabilities",
        "per_fold_probabilities",
        "query_protocol_row_ids",
        "group_tokens",
        "class_tokens",
        "fold_assignments",
        "context_count",
        "strategy_context_nonempty",
        "dialogue_history_eligible",
    )
    excluded = {*array_names, "_checkpoint_manifest", "_seal"}
    for item in fields(LoadedPredictionArtifact):
        if item.name in excluded:
            continue
        if getattr(artifact, item.name) != getattr(rebuilt, item.name):
            raise HarmBenchPredictionArtifactError(
                f"loaded prediction capability changed: {item.name}"
            )
    for name in array_names:
        observed = getattr(artifact, name)
        expected = getattr(rebuilt, name)
        if (observed is None) != (expected is None) or (
            observed is not None and not np.array_equal(observed, expected)
        ):
            raise HarmBenchPredictionArtifactError(
                f"loaded prediction array changed: {name}"
            )
    return artifact


@dataclass(frozen=True)
class EffectiveHistoryCurrentPair:
    """Selection probabilities with a current-only anchor at uncovered rows."""

    schema_version: str
    dataset_id: str
    model_id: str
    history_model_namespace: str
    current_model_namespace: str
    history_strategy_id: str
    current_strategy_id: str
    effective_semantics: str
    training_seed_ids: tuple[int, ...]
    query_count: int
    class_count: int
    query_roster_sha256: str
    group_roster_sha256: str
    class_order_sha256: str
    dialogue_history_eligible_sha256: str
    strategy_context_nonempty_sha256: str
    use_history_mask_sha256: str
    history_panel_sha256: str
    current_panel_sha256: str
    history_artifact_file_sha256: str
    history_receipt_file_sha256: str
    current_artifact_file_sha256: str
    current_receipt_file_sha256: str
    per_fold_effective_probability_sha256: str
    effective_probability_sha256: str
    pair_receipt_sha256: str
    receipt: Mapping[str, object]
    probabilities: np.ndarray = field(repr=False, compare=False)
    use_history_mask: np.ndarray = field(repr=False, compare=False)
    dialogue_history_eligible: np.ndarray = field(repr=False, compare=False)
    _history_artifact: LoadedPredictionArtifact = field(
        repr=False, compare=False
    )
    _current_artifact: LoadedPredictionArtifact = field(
        repr=False, compare=False
    )
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _PAIR_SEAL:
            raise HarmBenchPredictionArtifactError(
                "effective pairs can only be created by the sealed pair builder"
            )
        for name in (
            "probabilities",
            "use_history_mask",
            "dialogue_history_eligible",
        ):
            if np.asarray(getattr(self, name)).flags.writeable:
                raise HarmBenchPredictionArtifactError(
                    f"effective pair array is writable: {name}"
                )


def _pair_receipt(values: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": values["schema_version"],
        "dataset_id": values["dataset_id"],
        "model_id": values["model_id"],
        "history_model_namespace": values["history_model_namespace"],
        "current_model_namespace": values["current_model_namespace"],
        "history_strategy_id": values["history_strategy_id"],
        "current_strategy_id": values["current_strategy_id"],
        "effective_semantics": values["effective_semantics"],
        "training_seed_ids": list(values["training_seed_ids"]),
        "query_count": values["query_count"],
        "class_count": values["class_count"],
        "query_roster_sha256": values["query_roster_sha256"],
        "group_roster_sha256": values["group_roster_sha256"],
        "class_order_sha256": values["class_order_sha256"],
        "dialogue_history_eligible_sha256": values[
            "dialogue_history_eligible_sha256"
        ],
        "strategy_context_nonempty_sha256": values[
            "strategy_context_nonempty_sha256"
        ],
        "use_history_mask_sha256": values["use_history_mask_sha256"],
        "history_panel_sha256": values["history_panel_sha256"],
        "current_panel_sha256": values["current_panel_sha256"],
        "history_artifact_file_sha256": values["history_artifact_file_sha256"],
        "history_receipt_file_sha256": values["history_receipt_file_sha256"],
        "current_artifact_file_sha256": values["current_artifact_file_sha256"],
        "current_receipt_file_sha256": values["current_receipt_file_sha256"],
        "per_fold_effective_probability_sha256": values[
            "per_fold_effective_probability_sha256"
        ],
        "effective_probability_sha256": values[
            "effective_probability_sha256"
        ],
        "dialogue_history_eligible_count": values[
            "dialogue_history_eligible_count"
        ],
        "history_used_seed_query_count": values[
            "history_used_seed_query_count"
        ],
        "current_fallback_seed_query_count": values[
            "current_fallback_seed_query_count"
        ],
        "privacy_contract": dict(_PRIVACY_CONTRACT),
    }


def build_effective_history_current_pair(
    history_artifact: LoadedPredictionArtifact,
    current_artifact: LoadedPredictionArtifact,
) -> EffectiveHistoryCurrentPair:
    """Pair two live selection artifacts and mask empty history with current-only.

    The history fold means are usable only when every contributing fold reports
    non-empty strategy context for that seed/query.  Empty or dialogue-ineligible
    positions use the exact current-only probability row.
    """

    history = _revalidate_loaded_prediction_artifact(
        history_artifact, expected_role=SELECTION_ROLE
    )
    current = _revalidate_loaded_prediction_artifact(
        current_artifact, expected_role=SELECTION_ROLE
    )
    if (
        history.model_namespace != HISTORY_NAMESPACE
        or history.strategy_id not in STRICT_PAST_STRATEGY_IDS
        or history.context_role != SELECTION_CONTEXT_ROLE
    ):
        raise HarmBenchPredictionArtifactError(
            "effective pair history input must use a strict-past history namespace"
        )
    if (
        current.model_namespace != CURRENT_ONLY_NAMESPACE
        or current.strategy_id != CURRENT_ONLY_STRATEGY_ID
        or current.context_role != SELECTION_CONTEXT_ROLE
        or np.any(current.context_count)
        or np.any(current.strategy_context_nonempty)
    ):
        raise HarmBenchPredictionArtifactError(
            "effective pair current anchor must prove independent zero consumption"
        )
    scalar_matches = (
        "dataset_id",
        "model_id",
        "training_seed_ids",
        "fold_count",
        "entry_count",
        "class_order_sha256",
        "context_role",
    )
    if any(getattr(history, name) != getattr(current, name) for name in scalar_matches):
        raise HarmBenchPredictionArtifactError(
            "history/current pair differs in dataset/model/class/seed semantics"
        )
    receipt_lineage = (
        "fit_training_capability_sha256",
        "fit_feature_capability_sha256",
        "crossfit_plan_sha256",
        "source_capability_sha256",
        "cross_role_feature_roster_sha256",
        "source_content_sha256",
        "source_row_alignment_sha256",
        "query_roster_sha256",
        "group_roster_sha256",
    )
    if any(
        history.receipt[name] != current.receipt[name]
        for name in receipt_lineage
    ):
        raise HarmBenchPredictionArtifactError(
            "history/current pair differs in source or crossfit lineage"
        )
    array_matches = (
        "query_protocol_row_ids",
        "group_tokens",
        "class_tokens",
        "dialogue_history_eligible",
    )
    if any(
        not np.array_equal(getattr(history, name), getattr(current, name))
        for name in array_matches
    ):
        raise HarmBenchPredictionArtifactError(
            "history/current pair query/group/class/common eligibility changed"
        )
    q = len(history.query_protocol_row_ids)
    c = len(history.class_tokens)
    expected_context_shape = (5, 5, q)
    if history.strategy_context_nonempty.shape != expected_context_shape:
        raise HarmBenchPredictionArtifactError(
            "history strategy coverage tensor shape changed"
        )
    if not np.all(
        history.strategy_context_nonempty
        == history.strategy_context_nonempty[0, 0]
    ):
        raise HarmBenchPredictionArtifactError(
            "history strategy coverage differs across the 25 seed/fold entries"
        )
    strategy_nonempty = np.asarray(
        history.strategy_context_nonempty[0, 0], dtype=np.bool_
    )
    common = np.asarray(history.dialogue_history_eligible, dtype=np.bool_)
    if np.any(strategy_nonempty & ~common):
        raise HarmBenchPredictionArtifactError(
            "history strategy coverage exceeds common dialogue eligibility"
        )
    common_use_history = strategy_nonempty & common
    use_history = np.broadcast_to(
        common_use_history[None, :],
        (len(EXPECTED_TRAINING_SEED_IDS), q),
    )
    assert history.per_fold_probabilities is not None
    assert current.per_fold_probabilities is not None
    per_fold_effective = np.where(
        common_use_history[None, None, :, None],
        history.per_fold_probabilities,
        current.per_fold_probabilities,
    )
    fold_first_mean = per_fold_effective.mean(axis=1, dtype=np.float64)
    mean_first_effective = np.where(
        use_history[..., None], history.probabilities, current.probabilities
    )
    if not np.array_equal(fold_first_mean, mean_first_effective):
        raise HarmBenchPredictionArtifactError(
            "per-fold fallback then mean differs from fallback between fold means"
        )
    effective = fold_first_mean
    _validate_probability_tensor(
        effective,
        shape=(len(EXPECTED_TRAINING_SEED_IDS), q, c),
        name="effective_history_current_probabilities",
    )
    if (
        not np.array_equal(effective[~use_history], current.probabilities[~use_history])
        or not np.array_equal(effective[use_history], history.probabilities[use_history])
    ):
        raise HarmBenchPredictionArtifactError(
            "effective history/current fallback is not exact"
        )
    readonly_effective = _readonly(effective, dtype=np.float64)
    readonly_use_history = _readonly(use_history, dtype=np.bool_)
    readonly_common = _readonly(common, dtype=np.bool_)
    values: dict[str, object] = {
        "schema_version": EFFECTIVE_PAIR_SCHEMA,
        "dataset_id": history.dataset_id,
        "model_id": history.model_id,
        "history_model_namespace": history.model_namespace,
        "current_model_namespace": current.model_namespace,
        "history_strategy_id": history.strategy_id,
        "current_strategy_id": current.strategy_id,
        "effective_semantics": "per_fold_fallback_then_five_fold_mean",
        "training_seed_ids": history.training_seed_ids,
        "query_count": q,
        "class_count": c,
        "query_roster_sha256": str(history.receipt["query_roster_sha256"]),
        "group_roster_sha256": str(history.receipt["group_roster_sha256"]),
        "class_order_sha256": history.class_order_sha256,
        "dialogue_history_eligible_sha256": _array_sha256(readonly_common),
        "strategy_context_nonempty_sha256": _array_sha256(strategy_nonempty),
        "use_history_mask_sha256": _array_sha256(readonly_use_history),
        "history_panel_sha256": history.panel_sha256,
        "current_panel_sha256": current.panel_sha256,
        "history_artifact_file_sha256": history.artifact_file_sha256,
        "history_receipt_file_sha256": history.receipt_file_sha256,
        "current_artifact_file_sha256": current.artifact_file_sha256,
        "current_receipt_file_sha256": current.receipt_file_sha256,
        "per_fold_effective_probability_sha256": _array_sha256(
            per_fold_effective
        ),
        "effective_probability_sha256": _array_sha256(readonly_effective),
        "dialogue_history_eligible_count": int(common.sum()),
        "history_used_seed_query_count": int(use_history.sum()),
        "current_fallback_seed_query_count": int(use_history.size - use_history.sum()),
    }
    receipt = _pair_receipt(values)
    pair_sha = _canonical_sha256(receipt)
    return EffectiveHistoryCurrentPair(
        **{
            name: value
            for name, value in values.items()
            if name
            not in {
                "dialogue_history_eligible_count",
                "history_used_seed_query_count",
                "current_fallback_seed_query_count",
            }
        },
        pair_receipt_sha256=pair_sha,
        receipt=_deep_freeze(receipt),
        probabilities=readonly_effective,
        use_history_mask=readonly_use_history,
        dialogue_history_eligible=readonly_common,
        _history_artifact=history,
        _current_artifact=current,
        _seal=_PAIR_SEAL,
    )


def _revalidate_effective_history_current_pair(
    pair: object,
) -> EffectiveHistoryCurrentPair:
    """Live-rebuild an effective pair before any later sealed evaluation."""

    if (
        not isinstance(pair, EffectiveHistoryCurrentPair)
        or pair._seal is not _PAIR_SEAL
    ):
        raise HarmBenchPredictionArtifactError(
            "evaluator requires a sealed effective history/current pair"
        )
    rebuilt = build_effective_history_current_pair(
        pair._history_artifact, pair._current_artifact
    )
    array_names = (
        "probabilities",
        "use_history_mask",
        "dialogue_history_eligible",
    )
    excluded = {*array_names, "_history_artifact", "_current_artifact", "_seal"}
    for item in fields(EffectiveHistoryCurrentPair):
        if item.name not in excluded and getattr(pair, item.name) != getattr(
            rebuilt, item.name
        ):
            raise HarmBenchPredictionArtifactError(
                f"effective history/current pair changed: {item.name}"
            )
    for name in array_names:
        if not np.array_equal(getattr(pair, name), getattr(rebuilt, name)):
            raise HarmBenchPredictionArtifactError(
                f"effective history/current pair array changed: {name}"
            )
    return pair


__all__ = [
    "EXPECTED_TRAINING_SEED_IDS",
    "EXPECTED_FOLD_COUNT",
    "FIT_ROLE",
    "SELECTION_ROLE",
    "FOLD_PREDICTION_SCHEMA",
    "FIT_PANEL_SCHEMA",
    "SELECTION_PANEL_SCHEMA",
    "FIT_ARTIFACT_SCHEMA",
    "SELECTION_ARTIFACT_SCHEMA",
    "PUBLIC_RECEIPT_SCHEMA",
    "EFFECTIVE_PAIR_SCHEMA",
    "DIALOGUE_ALL_PAST_STRATEGY_ID",
    "HarmBenchPredictionArtifactError",
    "SealedFoldPrediction",
    "SealedFitOOFPredictionPanel",
    "SealedSelectionPredictionPanel",
    "LoadedPredictionArtifact",
    "EffectiveHistoryCurrentPair",
    "build_fit_fold_prediction",
    "build_selection_fold_prediction",
    "build_fit_oof_prediction_panel",
    "build_selection_prediction_panel",
    "validate_private_root",
    "validate_public_prediction_receipt",
    "public_prediction_receipt_sha256",
    "write_fit_oof_prediction_artifact",
    "write_selection_prediction_artifact",
    "load_prediction_artifact",
    "load_fit_oof_prediction_artifact",
    "load_selection_prediction_artifact",
    "build_effective_history_current_pair",
]
