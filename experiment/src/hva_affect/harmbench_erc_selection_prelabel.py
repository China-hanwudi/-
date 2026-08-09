"""Outcome-free prelabel gate for the HarmBench-ERC selection evaluator.

This module stops immediately before label activation.  It live-reloads the
exact 36 private selection-prediction artifacts, builds the 30 dataset-level
history/current effective pairs that implement the frozen 15 hypotheses, and
live-revalidates exactly two label-only *manifests*.  It never resolves, stats,
hashes, or opens either label NPZ.

The validated row-level alignment is written once to a repository-external
private bundle.  Only after that file and its receipt have crossed a durability
barrier may a write-once attempt marker be published.  A marker cannot be
loaded into a new process: if execution dies after publication, its existence
is an irreversible terminal exploratory state rather than a resume token.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
import threading
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from . import harmbench_erc_prediction_artifact as _prediction
from . import harmbench_erc_selection_labels as _labels
from .harmbench_erc_contexts import (
    CURRENT_ONLY_STRATEGY_ID,
    SELECTION_CONTEXT_ROLE,
)
from .harmbench_erc_models import CURRENT_ONLY_NAMESPACE, HISTORY_NAMESPACE
from .harmbench_erc_prediction_artifact import (
    DIALOGUE_ALL_PAST_STRATEGY_ID,
    EffectiveHistoryCurrentPair,
    LoadedPredictionArtifact,
    SELECTION_ROLE,
)
from .harmbench_erc_protocol_v2 import (
    EXPECTED_ANCHOR_STRATEGY_ID,
    EXPECTED_CONTEXT_ROSTER_ORDER,
    EXPECTED_HISTORY_STRATEGY_ORDER,
    EXPECTED_MODEL_ORDER,
    EXPECTED_SELECTION_DATASETS,
    EXPECTED_TRAINING_SEEDS,
    PROTOCOL_V2_ID,
    PROTOCOL_V2_SCHEMA,
    PROTOCOL_V2_STATUS,
    ProtocolV2Contract,
    validate_protocol_v2,
)
from .harmbench_erc_selection_labels import (
    SELECTION_LABEL_ROLE,
    SelectionLabelManifestMetadata,
)


PRELABEL_BUNDLE_SCHEMA = "harmbench_erc_selection_prelabel_bundle_v1"
PRELABEL_RECEIPT_SCHEMA = "harmbench_erc_selection_prelabel_receipt_v1"
ATTEMPT_MARKER_SCHEMA = "harmbench_erc_selection_attempt_started_v1"
PRELABEL_BUNDLE_FILENAME = "harmbench_erc_selection_prelabel_bundle.json"
PRELABEL_RECEIPT_FILENAME = "harmbench_erc_selection_prelabel_bundle.receipt.json"
ATTEMPT_MARKER_FILENAME = "harmbench_erc_selection_attempt.started.json"
FINAL_OUTPUT_FILENAME = "harmbench_erc_selection_evaluation.final.json"

EXPLORATORY_STATUS = "previously_observed_selection_exploratory"
EXPECTED_ARTIFACT_COUNT = 36
EXPECTED_ARTIFACTS_PER_DATASET = 18
EXPECTED_EFFECTIVE_HYPOTHESIS_COUNT = 15
EXPECTED_EFFECTIVE_DATASET_PAIR_COUNT = 30
MAX_PRELABEL_BUNDLE_BYTES = 128 * 1024 * 1024
MAX_PRELABEL_RECEIPT_BYTES = 256 * 1024
MAX_ATTEMPT_MARKER_BYTES = 32 * 1024

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_BUNDLE_SEAL = object()
_ATTEMPT_SEAL = object()
_TICKET_SEAL = object()


class HarmBenchSelectionPrelabelError(ValueError):
    """Raised when the outcome-free evaluator state is not exact and live."""


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise HarmBenchSelectionPrelabelError(
            f"{name} must be one lowercase SHA-256"
        )
    return value


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
        raise HarmBenchSelectionPrelabelError(
            f"prelabel state is not canonical JSON data: {error}"
        ) from error


def _canonical_json_bytes(value: object) -> bytes:
    return _canonical_json_payload(value) + b"\n"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_payload(value)).hexdigest()


def _array_sha256(values: object) -> str:
    """Match the canonical ndarray hash used by prediction/label artifacts."""

    array = np.asarray(values)
    if array.dtype.kind == "O":
        raise HarmBenchSelectionPrelabelError("object arrays cannot enter prelabel")
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
            canonical = canonical.byteswap().view(
                canonical.dtype.newbyteorder("<")
            )
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise HarmBenchSelectionPrelabelError(
        f"unsupported prelabel JSON value: {type(value).__name__}"
    )


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(child) for child in value)
    return value


def _validated_protocol(value: object) -> ProtocolV2Contract:
    if type(value) is not ProtocolV2Contract:
        raise HarmBenchSelectionPrelabelError(
            "the final validated ProtocolV2Contract is required"
        )
    try:
        # ``ProtocolV2Contract`` deliberately freezes JSON arrays as tuples,
        # while the validator quite correctly insists that a fresh JSON input
        # use lists.  Reconstruct the plain JSON tree before live validation.
        plain = _plain_json(value.payload)
        if not isinstance(plain, Mapping):
            raise HarmBenchSelectionPrelabelError(
                "the v2 protocol payload is not a JSON object"
            )
        live = validate_protocol_v2(plain)
    except (TypeError, ValueError) as error:
        raise HarmBenchSelectionPrelabelError(
            "the v2 protocol failed live validation"
        ) from error
    if (
        value.canonical_sha256 != live.canonical_sha256
        or _plain_json(value.payload) != _plain_json(live.payload)
    ):
        raise HarmBenchSelectionPrelabelError(
            "the supplied v2 protocol capability changed after validation"
        )
    return live


def _evaluation_status(*, attempt_started: bool) -> dict[str, object]:
    return {
        "status": EXPLORATORY_STATUS,
        "confirmatory_claim": False,
        "calibration": False,
        "internal_holdout": False,
        "validation": False,
        "official_test": False,
        "selection_labels_opened": False,
        "row_metrics_computed": False,
        "attempt_started": attempt_started,
    }


def _metadata_record(metadata: SelectionLabelManifestMetadata) -> dict[str, object]:
    return {
        "schema_version": metadata.schema_version,
        "artifact_schema_version": metadata.artifact_schema_version,
        "dataset_id": metadata.dataset_id,
        "role": metadata.role,
        "rows": metadata.rows,
        "ordered_protocol_row_alignment_sha256": (
            metadata.ordered_protocol_row_alignment_sha256
        ),
        "class_order_sha256": metadata.class_order_sha256,
        "artifact_filename": metadata.artifact_filename,
        "artifact_file_sha256": metadata.artifact_file_sha256,
        "manifest_file_sha256": metadata.manifest_file_sha256,
    }


def _artifact_binding(artifact: LoadedPredictionArtifact) -> dict[str, object]:
    return {
        "dataset_id": artifact.dataset_id,
        "model_id": artifact.model_id,
        "model_namespace": artifact.model_namespace,
        "strategy_id": artifact.strategy_id,
        "context_role": artifact.context_role,
        "training_seed_ids": list(artifact.training_seed_ids),
        "fold_count": artifact.fold_count,
        "entry_count": artifact.entry_count,
        "checkpoint_manifest_sha256": artifact.checkpoint_manifest_sha256,
        "checkpoint_manifest_file_sha256": (
            artifact.checkpoint_manifest_file_sha256
        ),
        "panel_sha256": artifact.panel_sha256,
        "artifact_file_sha256": artifact.artifact_file_sha256,
        "receipt_file_sha256": artifact.receipt_file_sha256,
        "receipt_payload_sha256": (
            _prediction.public_prediction_receipt_sha256(artifact.receipt)
        ),
    }


def _history_depth_record(
    *, strategy_id: str, depth: np.ndarray
) -> dict[str, object]:
    vector = np.asarray(depth, dtype=np.int64)
    return {
        "strategy_id": strategy_id,
        "depth_by_query": vector.tolist(),
        "depth_roster_sha256": _array_sha256(vector),
        "nonempty_by_query_sha256": _array_sha256(vector > 0),
    }


def _expected_artifact_keys() -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (dataset_id, model_id, strategy_id)
        for dataset_id in EXPECTED_SELECTION_DATASETS
        for model_id in EXPECTED_MODEL_ORDER
        for strategy_id in EXPECTED_CONTEXT_ROSTER_ORDER
    )


def _prediction_roster(
    artifacts: Sequence[LoadedPredictionArtifact],
) -> tuple[
    tuple[LoadedPredictionArtifact, ...],
    dict[tuple[str, str, str], LoadedPredictionArtifact],
]:
    if isinstance(artifacts, (str, bytes)) or not isinstance(artifacts, Sequence):
        raise HarmBenchSelectionPrelabelError(
            "prediction_artifacts must be the exact typed sequence"
        )
    if len(artifacts) != EXPECTED_ARTIFACT_COUNT:
        raise HarmBenchSelectionPrelabelError(
            "selection prediction roster must contain exactly 36 artifacts"
        )
    live_values: list[LoadedPredictionArtifact] = []
    keyed: dict[tuple[str, str, str], LoadedPredictionArtifact] = {}
    for raw in artifacts:
        try:
            live = _prediction._revalidate_loaded_prediction_artifact(  # noqa: SLF001
                raw, expected_role=SELECTION_ROLE
            )
        except (TypeError, ValueError) as error:
            raise HarmBenchSelectionPrelabelError(
                "all prediction artifacts must be loader-sealed and live"
            ) from error
        key = (live.dataset_id, live.model_id, live.strategy_id)
        if key in keyed:
            raise HarmBenchSelectionPrelabelError(
                "duplicate typed dataset/model/strategy prediction artifact"
            )
        keyed[key] = live
        live_values.append(live)
    expected = _expected_artifact_keys()
    if set(keyed) != set(expected):
        missing = sorted(set(expected) - set(keyed))
        extra = sorted(set(keyed) - set(expected))
        raise HarmBenchSelectionPrelabelError(
            f"exact prediction roster changed; missing={missing!r}, extra={extra!r}"
        )
    ordered = tuple(keyed[key] for key in expected)

    path_pairs = {(item.artifact_path, item.receipt_path) for item in ordered}
    artifact_shas = {item.artifact_file_sha256 for item in ordered}
    receipt_shas = {item.receipt_file_sha256 for item in ordered}
    panel_shas = {item.panel_sha256 for item in ordered}
    if (
        len(path_pairs) != EXPECTED_ARTIFACT_COUNT
        or len(artifact_shas) != EXPECTED_ARTIFACT_COUNT
        or len(receipt_shas) != EXPECTED_ARTIFACT_COUNT
        or len(panel_shas) != EXPECTED_ARTIFACT_COUNT
    ):
        raise HarmBenchSelectionPrelabelError(
            "prediction files, receipts, and panels must be unique per roster cell"
        )
    return ordered, keyed


def _label_metadata_roster(
    metadata_values: Sequence[SelectionLabelManifestMetadata],
) -> tuple[
    tuple[SelectionLabelManifestMetadata, ...],
    dict[str, SelectionLabelManifestMetadata],
]:
    if (
        isinstance(metadata_values, (str, bytes))
        or not isinstance(metadata_values, Sequence)
        or len(metadata_values) != len(EXPECTED_SELECTION_DATASETS)
    ):
        raise HarmBenchSelectionPrelabelError(
            "exactly two selection-label manifest metadata capabilities are required"
        )
    keyed: dict[str, SelectionLabelManifestMetadata] = {}
    for raw in metadata_values:
        try:
            live = _labels._revalidate_manifest_metadata(raw)  # noqa: SLF001
        except (TypeError, ValueError) as error:
            raise HarmBenchSelectionPrelabelError(
                "selection-label metadata must be loader-sealed and live"
            ) from error
        if live.dataset_id in keyed:
            raise HarmBenchSelectionPrelabelError(
                "duplicate selection-label dataset manifest"
            )
        keyed[live.dataset_id] = live
    if set(keyed) != set(EXPECTED_SELECTION_DATASETS):
        raise HarmBenchSelectionPrelabelError(
            "selection-label manifest dataset roster changed"
        )
    ordered = tuple(keyed[dataset] for dataset in EXPECTED_SELECTION_DATASETS)
    if len({item.manifest_file_sha256 for item in ordered}) != len(ordered):
        raise HarmBenchSelectionPrelabelError(
            "the two label manifests must have distinct file SHA-256 values"
        )
    return ordered, keyed


def _check_dataset_alignment(
    dataset_id: str,
    keyed: Mapping[tuple[str, str, str], LoadedPredictionArtifact],
    metadata: SelectionLabelManifestMetadata,
) -> tuple[dict[str, object], dict[tuple[str, str], np.ndarray]]:
    reference = keyed[
        (dataset_id, EXPECTED_MODEL_ORDER[0], EXPECTED_ANCHOR_STRATEGY_ID)
    ]
    queries = np.asarray(reference.query_protocol_row_ids)
    groups = np.asarray(reference.group_tokens)
    classes = np.asarray(reference.class_tokens)
    common = np.asarray(reference.dialogue_history_eligible, dtype=np.bool_)
    expected_seeds = tuple(int(value) for value in EXPECTED_TRAINING_SEEDS)
    if (
        reference.training_seed_ids != expected_seeds
        or reference.fold_count != 5
        or reference.entry_count != 25
    ):
        raise HarmBenchSelectionPrelabelError(
            "selection seed/fold/entry roster differs from protocol v2"
        )
    if (
        metadata.dataset_id != dataset_id
        or metadata.role != SELECTION_LABEL_ROLE
        or metadata.rows != len(queries)
        or metadata.ordered_protocol_row_alignment_sha256 != _array_sha256(queries)
        or metadata.class_order_sha256 != reference.class_order_sha256
    ):
        raise HarmBenchSelectionPrelabelError(
            "label manifest differs from prediction row/class alignment"
        )

    depth_by_key: dict[tuple[str, str], np.ndarray] = {}
    artifact_records: list[dict[str, object]] = []
    for model_id in EXPECTED_MODEL_ORDER:
        for strategy_id in EXPECTED_CONTEXT_ROSTER_ORDER:
            artifact = keyed[(dataset_id, model_id, strategy_id)]
            expected_namespace = (
                CURRENT_ONLY_NAMESPACE
                if strategy_id == EXPECTED_ANCHOR_STRATEGY_ID
                else HISTORY_NAMESPACE
            )
            if (
                artifact.role != SELECTION_ROLE
                or artifact.context_role != SELECTION_CONTEXT_ROLE
                or artifact.model_namespace != expected_namespace
                or artifact.training_seed_ids != expected_seeds
                or artifact.fold_count != 5
                or artifact.entry_count != 25
                or artifact.class_order_sha256 != reference.class_order_sha256
            ):
                raise HarmBenchSelectionPrelabelError(
                    "prediction role/namespace/seed/fold/class semantics changed"
                )
            for name, expected_array in (
                ("query_protocol_row_ids", queries),
                ("group_tokens", groups),
                ("class_tokens", classes),
                ("dialogue_history_eligible", common),
            ):
                if not np.array_equal(getattr(artifact, name), expected_array):
                    raise HarmBenchSelectionPrelabelError(
                        f"within-dataset {name} alignment changed"
                    )
            counts = np.asarray(artifact.context_count)
            nonempty = np.asarray(artifact.strategy_context_nonempty)
            if (
                counts.dtype != np.dtype("int64")
                or counts.shape != (5, 5, len(queries))
                or nonempty.dtype != np.dtype("bool")
                or nonempty.shape != counts.shape
                or not np.array_equal(nonempty, counts > 0)
                or np.any(nonempty & ~common[None, None, :])
            ):
                raise HarmBenchSelectionPrelabelError(
                    "history coverage/depth tensors violate E_dialogue"
                )
            if not np.all(counts == counts[0, 0, :][None, None, :]):
                raise HarmBenchSelectionPrelabelError(
                    "context depth must be identical across all 25 seed/fold entries"
                )
            if not np.all(nonempty == nonempty[0, 0, :][None, None, :]):
                raise HarmBenchSelectionPrelabelError(
                    "context nonempty must be identical across all 25 entries"
                )
            if strategy_id == EXPECTED_ANCHOR_STRATEGY_ID:
                if (
                    artifact.strategy_id != CURRENT_ONLY_STRATEGY_ID
                    or np.any(counts)
                    or np.any(nonempty)
                ):
                    raise HarmBenchSelectionPrelabelError(
                        "current-only anchor lost its 25-fold zero-consumption proof"
                    )
            elif strategy_id == DIALOGUE_ALL_PAST_STRATEGY_ID:
                if not np.array_equal(nonempty, np.broadcast_to(common, counts.shape)):
                    raise HarmBenchSelectionPrelabelError(
                        "dialogue_all_past nonempty must exactly equal E_dialogue"
                    )
            depth_by_key[(model_id, strategy_id)] = np.asarray(
                counts[0, 0, :], dtype=np.int64
            ).copy()
            artifact_records.append(_artifact_binding(artifact))

    # Context selection is outcome-free and model-independent.  Enforce the
    # protocol-required dialogue depth relation and the stronger exact relation
    # for every fixed strategy, preventing a model-specific context roster.
    for strategy_id in EXPECTED_CONTEXT_ROSTER_ORDER:
        baseline = depth_by_key[(EXPECTED_MODEL_ORDER[0], strategy_id)]
        for model_id in EXPECTED_MODEL_ORDER[1:]:
            if not np.array_equal(depth_by_key[(model_id, strategy_id)], baseline):
                raise HarmBenchSelectionPrelabelError(
                    f"cross-model context depth drift for {strategy_id}"
                )

    depth_records = [
        _history_depth_record(
            strategy_id=strategy_id,
            depth=depth_by_key[(EXPECTED_MODEL_ORDER[0], strategy_id)],
        )
        for strategy_id in EXPECTED_CONTEXT_ROSTER_ORDER
    ]
    dataset_record = {
        "dataset_id": dataset_id,
        "query_count": len(queries),
        "class_count": len(classes),
        "training_seed_ids": list(expected_seeds),
        "query_protocol_row_ids": queries.astype(np.int64).tolist(),
        "query_roster_sha256": _array_sha256(queries),
        "group_tokens": groups.astype(str).tolist(),
        "group_roster_sha256": _array_sha256(groups),
        "class_tokens": classes.astype(str).tolist(),
        "class_order_sha256": reference.class_order_sha256,
        "dialogue_history_eligible": common.tolist(),
        "dialogue_history_eligible_sha256": _array_sha256(common),
        "depth_rosters": depth_records,
        "depth_roster_sha256": _canonical_sha256(depth_records),
        "label_manifest_metadata": _metadata_record(metadata),
        "prediction_artifacts": artifact_records,
        "prediction_artifact_roster_sha256": _canonical_sha256(artifact_records),
    }
    return dataset_record, depth_by_key


def _effective_pairs(
    keyed: Mapping[tuple[str, str, str], LoadedPredictionArtifact],
) -> tuple[tuple[EffectiveHistoryCurrentPair, ...], list[dict[str, object]]]:
    pairs: list[EffectiveHistoryCurrentPair] = []
    hypothesis_records: list[dict[str, object]] = []
    for model_id in EXPECTED_MODEL_ORDER:
        for history_strategy_id in EXPECTED_HISTORY_STRATEGY_ORDER:
            dataset_records: list[dict[str, object]] = []
            for dataset_id in EXPECTED_SELECTION_DATASETS:
                try:
                    pair = _prediction.build_effective_history_current_pair(
                        keyed[(dataset_id, model_id, history_strategy_id)],
                        keyed[
                            (
                                dataset_id,
                                model_id,
                                EXPECTED_ANCHOR_STRATEGY_ID,
                            )
                        ],
                    )
                except (TypeError, ValueError) as error:
                    raise HarmBenchSelectionPrelabelError(
                        "sealed history/current effective-pair construction failed"
                    ) from error
                pairs.append(pair)
                dataset_records.append(
                    {
                        "dataset_id": dataset_id,
                        "pair_receipt_sha256": pair.pair_receipt_sha256,
                        "pair_receipt": _plain_json(pair.receipt),
                    }
                )
            hypothesis_records.append(
                {
                    "model_id": model_id,
                    "history_strategy_id": history_strategy_id,
                    "current_strategy_id": EXPECTED_ANCHOR_STRATEGY_ID,
                    "dataset_pair_receipts": dataset_records,
                    "dataset_pair_receipt_roster_sha256": _canonical_sha256(
                        dataset_records
                    ),
                }
            )
    if (
        len(pairs) != EXPECTED_EFFECTIVE_DATASET_PAIR_COUNT
        or len(hypothesis_records) != EXPECTED_EFFECTIVE_HYPOTHESIS_COUNT
        or len({pair.pair_receipt_sha256 for pair in pairs}) != len(pairs)
    ):
        raise HarmBenchSelectionPrelabelError(
            "exact 15-hypothesis/30-dataset effective-pair roster changed"
        )
    return tuple(pairs), hypothesis_records


def _build_bundle_state(
    protocol: ProtocolV2Contract,
    prediction_artifacts: Sequence[LoadedPredictionArtifact],
    label_manifests: Sequence[SelectionLabelManifestMetadata],
) -> tuple[
    dict[str, object],
    tuple[LoadedPredictionArtifact, ...],
    tuple[SelectionLabelManifestMetadata, ...],
    tuple[EffectiveHistoryCurrentPair, ...],
]:
    live_protocol = _validated_protocol(protocol)
    ordered_artifacts, keyed_artifacts = _prediction_roster(prediction_artifacts)
    ordered_metadata, keyed_metadata = _label_metadata_roster(label_manifests)
    datasets: list[dict[str, object]] = []
    for dataset_id in EXPECTED_SELECTION_DATASETS:
        record, _ = _check_dataset_alignment(
            dataset_id, keyed_artifacts, keyed_metadata[dataset_id]
        )
        datasets.append(record)
    effective_pairs, effective_records = _effective_pairs(keyed_artifacts)
    label_manifest_file_sha256_by_dataset = {
        item.dataset_id: item.manifest_file_sha256 for item in ordered_metadata
    }
    bundle = {
        "schema_version": PRELABEL_BUNDLE_SCHEMA,
        "protocol": {
            "schema_version": PROTOCOL_V2_SCHEMA,
            "protocol_id": PROTOCOL_V2_ID,
            "status": PROTOCOL_V2_STATUS,
            "canonical_sha256": live_protocol.canonical_sha256,
        },
        "evaluation_status": _evaluation_status(attempt_started=False),
        "exact_roster": {
            "dataset_order": list(EXPECTED_SELECTION_DATASETS),
            "model_order": list(EXPECTED_MODEL_ORDER),
            "strategy_order": list(EXPECTED_CONTEXT_ROSTER_ORDER),
            "history_strategy_order": list(EXPECTED_HISTORY_STRATEGY_ORDER),
            "training_seed_order": list(EXPECTED_TRAINING_SEEDS),
            "fold_count": 5,
            "checkpoint_entry_count_per_artifact": 25,
            "prediction_artifact_count": EXPECTED_ARTIFACT_COUNT,
            "prediction_artifact_count_per_dataset": (
                EXPECTED_ARTIFACTS_PER_DATASET
            ),
            "effective_hypothesis_count": EXPECTED_EFFECTIVE_HYPOTHESIS_COUNT,
            "effective_dataset_pair_count": (
                EXPECTED_EFFECTIVE_DATASET_PAIR_COUNT
            ),
            "label_manifest_count": len(EXPECTED_SELECTION_DATASETS),
        },
        "datasets": datasets,
        "dataset_roster_sha256": _canonical_sha256(datasets),
        "effective_pair_bindings": effective_records,
        "effective_pair_binding_roster_sha256": _canonical_sha256(
            effective_records
        ),
        "label_manifest_file_sha256_by_dataset": (
            label_manifest_file_sha256_by_dataset
        ),
        "label_manifest_file_sha256_roster_sha256": _canonical_sha256(
            label_manifest_file_sha256_by_dataset
        ),
        "privacy_contract": {
            "restricted_real_data_sent_to_external_GPT_or_API": False,
            "label_npz_resolved_statted_hashed_or_opened": False,
            "contains_label_values": False,
            "contains_prediction_probabilities": False,
            "repository_external_private_bundle": True,
        },
    }
    encoded = _canonical_json_bytes(bundle)
    if len(encoded) > MAX_PRELABEL_BUNDLE_BYTES:
        raise HarmBenchSelectionPrelabelError(
            "prelabel bundle exceeds its fixed byte budget"
        )
    return bundle, ordered_artifacts, ordered_metadata, effective_pairs


def _receipt_for_bundle(bundle: Mapping[str, object], bundle_file_sha256: str) -> dict[str, object]:
    bundle_sha = _sha256(bundle_file_sha256, name="bundle_file_sha256")
    datasets = bundle["datasets"]
    assert isinstance(datasets, list)
    dataset_bindings = [
        {
            "dataset_id": row["dataset_id"],
            "query_count": row["query_count"],
            "class_count": row["class_count"],
            "query_roster_sha256": row["query_roster_sha256"],
            "group_roster_sha256": row["group_roster_sha256"],
            "class_order_sha256": row["class_order_sha256"],
            "dialogue_history_eligible_sha256": row[
                "dialogue_history_eligible_sha256"
            ],
            "depth_roster_sha256": row["depth_roster_sha256"],
            "prediction_artifact_roster_sha256": row[
                "prediction_artifact_roster_sha256"
            ],
            "label_manifest_file_sha256": row["label_manifest_metadata"][
                "manifest_file_sha256"
            ],
            "label_artifact_declared_sha256": row["label_manifest_metadata"][
                "artifact_file_sha256"
            ],
        }
        for row in datasets
    ]
    receipt = {
        "schema_version": PRELABEL_RECEIPT_SCHEMA,
        "bundle_schema_version": PRELABEL_BUNDLE_SCHEMA,
        "bundle_filename": PRELABEL_BUNDLE_FILENAME,
        "bundle_file_sha256": bundle_sha,
        "protocol_canonical_sha256": bundle["protocol"]["canonical_sha256"],
        "evaluation_status": _evaluation_status(attempt_started=False),
        "exact_roster": bundle["exact_roster"],
        "dataset_bindings": dataset_bindings,
        "dataset_binding_roster_sha256": _canonical_sha256(dataset_bindings),
        "effective_pair_binding_roster_sha256": bundle[
            "effective_pair_binding_roster_sha256"
        ],
        "label_manifest_file_sha256_roster_sha256": bundle[
            "label_manifest_file_sha256_roster_sha256"
        ],
        "public_safety": {
            "contains_row_identifiers": False,
            "contains_groups": False,
            "contains_labels": False,
            "contains_probabilities": False,
            "contains_private_paths": False,
            "confirmatory_claim": False,
        },
    }
    if len(_canonical_json_bytes(receipt)) > MAX_PRELABEL_RECEIPT_BYTES:
        raise HarmBenchSelectionPrelabelError(
            "prelabel receipt exceeds its fixed byte budget"
        )
    return receipt


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
            raise HarmBenchSelectionPrelabelError(
                "prelabel output path contains a symlink or reparse point"
            )


def _plain_file_stat(path: Path, *, name: str) -> os.stat_result:
    try:
        observed = path.lstat()
    except OSError as error:
        raise HarmBenchSelectionPrelabelError(f"cannot stat exact {name}") from error
    if stat.S_ISLNK(observed.st_mode) or _is_reparse(observed):
        raise HarmBenchSelectionPrelabelError(
            f"{name} cannot be a symlink or reparse point"
        )
    if not stat.S_ISREG(observed.st_mode):
        raise HarmBenchSelectionPrelabelError(f"{name} must be a plain file")
    return observed


def _plain_directory_stat(path: Path, *, name: str) -> os.stat_result:
    try:
        observed = path.lstat()
    except OSError as error:
        raise HarmBenchSelectionPrelabelError(f"cannot stat exact {name}") from error
    if stat.S_ISLNK(observed.st_mode) or _is_reparse(observed):
        raise HarmBenchSelectionPrelabelError(
            f"{name} cannot be a symlink or reparse point"
        )
    if not stat.S_ISDIR(observed.st_mode):
        raise HarmBenchSelectionPrelabelError(f"{name} must be a plain directory")
    return observed


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
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
    return _file_identity(first) == _file_identity(second)


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    # Directory timestamps/size legitimately change as fixed children are
    # published.  Device+inode+type/reparse attributes identify the directory
    # itself without treating our own writes as a replacement attack.
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(stat.S_IFMT(metadata.st_mode)),
        int(getattr(metadata, "st_file_attributes", 0)),
    )


def _assert_root_identity(root: Path, expected: os.stat_result) -> None:
    _reject_reparse_components(root)
    if _directory_identity(
        _plain_directory_stat(root, name="prelabel root")
    ) != _directory_identity(expected):
        raise HarmBenchSelectionPrelabelError(
            "prelabel root changed identity during the state transition"
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


def _validate_output_root(path: str | Path) -> Path:
    raw = Path(path)
    if not raw.is_absolute() or raw.is_symlink():
        raise HarmBenchSelectionPrelabelError(
            "prelabel root must be an explicit absolute non-symlink directory"
        )
    _reject_reparse_components(raw)
    try:
        metadata = raw.lstat()
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
            raise HarmBenchSelectionPrelabelError(
                "prelabel root cannot be a symlink or reparse point"
            )
        root = raw.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise HarmBenchSelectionPrelabelError(
            "prelabel root must already exist"
        ) from error
    if not root.is_dir() or root == Path(root.anchor):
        raise HarmBenchSelectionPrelabelError(
            "prelabel root must be a safe plain non-root directory"
        )
    if _is_within(root, _repository_root().resolve()) or _is_within(
        root, _home_root().resolve()
    ):
        raise HarmBenchSelectionPrelabelError(
            "prelabel root must be outside both repository and user home"
        )
    return root


def _fixed_path(root: Path, filename: str) -> Path:
    result = root / filename
    _reject_reparse_components(result)
    return result


def _terminal_marker_exists(root: Path) -> bool:
    return os.path.lexists(_fixed_path(root, ATTEMPT_MARKER_FILENAME))


def _reject_existing_terminal_state(root: Path) -> None:
    if _terminal_marker_exists(root):
        final_exists = os.path.lexists(_fixed_path(root, FINAL_OUTPUT_FILENAME))
        if final_exists:
            raise HarmBenchSelectionPrelabelError(
                "selection attempt is already terminal; rerun/resume is forbidden"
            )
        raise HarmBenchSelectionPrelabelError(
            "attempt marker exists without final output: terminal crash-replay state"
        )
    if os.path.lexists(_fixed_path(root, FINAL_OUTPUT_FILENAME)):
        raise HarmBenchSelectionPrelabelError(
            "orphan final output is a terminal invalid state"
        )


def _hash_open_handle(handle: Any) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _decode_canonical_file(
    path: Path,
    *,
    expected_sha256: str,
    maximum_bytes: int,
    name: str,
) -> dict[str, object]:
    expected = _sha256(expected_sha256, name=f"expected_{name}_sha256")
    _reject_reparse_components(path)
    before_path = _plain_file_stat(path, name=name)
    if before_path.st_size > maximum_bytes:
        raise HarmBenchSelectionPrelabelError(f"{name} exceeds its byte budget")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HarmBenchSelectionPrelabelError(
                    f"{name} contains duplicate JSON key: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise HarmBenchSelectionPrelabelError(
            f"{name} contains invalid JSON constant: {value}"
        )

    try:
        with path.open("rb") as handle:
            before_handle = os.fstat(handle.fileno())
            if not _same_file_identity(before_path, before_handle):
                raise HarmBenchSelectionPrelabelError(
                    f"{name} changed before verified read"
                )
            first_sha = _hash_open_handle(handle)
            if first_sha != expected:
                raise HarmBenchSelectionPrelabelError(f"{name} SHA-256 changed")
            handle.seek(0)
            encoded = handle.read(maximum_bytes + 1)
            if len(encoded) > maximum_bytes:
                raise HarmBenchSelectionPrelabelError(
                    f"{name} exceeds its byte budget"
                )
            try:
                payload = json.loads(
                    encoded.decode("utf-8"),
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=reject_constant,
                )
            except HarmBenchSelectionPrelabelError:
                raise
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise HarmBenchSelectionPrelabelError(
                    f"{name} is not strict UTF-8 JSON"
                ) from error
            if not isinstance(payload, dict):
                raise HarmBenchSelectionPrelabelError(f"{name} root must be an object")
            if encoded != _canonical_json_bytes(payload):
                raise HarmBenchSelectionPrelabelError(
                    f"{name} is not canonical JSON"
                )
            handle.seek(0)
            second_sha = _hash_open_handle(handle)
            after_handle = os.fstat(handle.fileno())
    except HarmBenchSelectionPrelabelError:
        raise
    except OSError as error:
        raise HarmBenchSelectionPrelabelError(f"cannot read exact {name}") from error
    if first_sha != second_sha or not _same_file_identity(
        before_handle, after_handle
    ):
        raise HarmBenchSelectionPrelabelError(f"{name} changed during verified read")
    _reject_reparse_components(path)
    if not _same_file_identity(_plain_file_stat(path, name=name), after_handle):
        raise HarmBenchSelectionPrelabelError(
            f"{name} path changed identity during verified read"
        )
    return payload


def _temporary_bytes(root: Path, destination: Path, encoded: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=root, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return temporary


def _move_file_write_through_windows(source: Path, destination: Path) -> None:
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
        raise OSError(error, f"write-through publication failed: {destination.name}")


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
    """Durability barrier for directory entries after write-once publication.

    POSIX exposes directory fsync directly.  Windows does not permit Python to
    fsync a directory handle; `_publish_once` therefore uses MoveFileExW with
    MOVEFILE_WRITE_THROUGH, whose documented contract waits for the move to be
    flushed.  Calling this helper in both branches keeps the state ordering
    explicit and testable.
    """

    if os.name == "nt":
        return
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    descriptor = os.open(root, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_pair_once(
    root: Path,
    bundle: Mapping[str, object],
    receipt: Mapping[str, object],
) -> tuple[str, str]:
    root_identity = _plain_directory_stat(root, name="prelabel root")
    bundle_path = _fixed_path(root, PRELABEL_BUNDLE_FILENAME)
    receipt_path = _fixed_path(root, PRELABEL_RECEIPT_FILENAME)
    if os.path.lexists(bundle_path) or os.path.lexists(receipt_path):
        raise FileExistsError(
            "write-once prelabel bundle or receipt already exists"
        )
    bundle_bytes = _canonical_json_bytes(bundle)
    receipt_bytes = _canonical_json_bytes(receipt)
    bundle_temporary: Path | None = None
    receipt_temporary: Path | None = None
    try:
        bundle_temporary = _temporary_bytes(root, bundle_path, bundle_bytes)
        _assert_root_identity(root, root_identity)
        receipt_temporary = _temporary_bytes(root, receipt_path, receipt_bytes)
        _assert_root_identity(root, root_identity)
        _publish_once(bundle_temporary, bundle_path)
        bundle_temporary = None
        _assert_root_identity(root, root_identity)
        _publish_once(receipt_temporary, receipt_path)
        receipt_temporary = None
        _assert_root_identity(root, root_identity)
        _sync_directory(root)
        _assert_root_identity(root, root_identity)
    finally:
        for temporary in (bundle_temporary, receipt_temporary):
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
    return (
        hashlib.sha256(bundle_bytes).hexdigest(),
        hashlib.sha256(receipt_bytes).hexdigest(),
    )


@dataclass(frozen=True)
class LoadedSelectionPrelabelBundle:
    """Loader-only live capability for the outcome-free prelabel state."""

    protocol_canonical_sha256: str
    bundle_file_sha256: str
    receipt_file_sha256: str
    private_root: Path
    bundle_path: Path
    receipt_path: Path
    bundle: Mapping[str, object] = field(repr=False, compare=False)
    receipt: Mapping[str, object] = field(repr=False, compare=False)
    _protocol: ProtocolV2Contract = field(repr=False, compare=False)
    _prediction_artifacts: tuple[LoadedPredictionArtifact, ...] = field(
        repr=False, compare=False
    )
    _label_manifests: tuple[SelectionLabelManifestMetadata, ...] = field(
        repr=False, compare=False
    )
    _effective_pairs: tuple[EffectiveHistoryCurrentPair, ...] = field(
        repr=False, compare=False
    )
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _BUNDLE_SEAL:
            raise HarmBenchSelectionPrelabelError(
                "prelabel bundles can only be created by the verified loader"
            )
        _sha256(self.protocol_canonical_sha256, name="protocol_canonical_sha256")
        _sha256(self.bundle_file_sha256, name="bundle_file_sha256")
        _sha256(self.receipt_file_sha256, name="receipt_file_sha256")
        for name in ("private_root", "bundle_path", "receipt_path"):
            value = getattr(self, name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise HarmBenchSelectionPrelabelError(
                    f"loaded prelabel {name} must be an absolute Path"
                )


def _load_prelabel_bundle_internal(
    *,
    private_root: str | Path,
    protocol: ProtocolV2Contract,
    prediction_artifacts: Sequence[LoadedPredictionArtifact],
    label_manifests: Sequence[SelectionLabelManifestMetadata],
    expected_bundle_file_sha256: str,
    expected_receipt_file_sha256: str,
    marker_permitted: bool,
) -> LoadedSelectionPrelabelBundle:
    root = _validate_output_root(private_root)
    root_identity = _plain_directory_stat(root, name="prelabel root")
    if not marker_permitted:
        _reject_existing_terminal_state(root)
    elif os.path.lexists(_fixed_path(root, FINAL_OUTPUT_FILENAME)):
        raise HarmBenchSelectionPrelabelError(
            "final output appeared before this module implements publication"
        )
    expected_bundle, ordered_artifacts, ordered_metadata, effective_pairs = (
        _build_bundle_state(protocol, prediction_artifacts, label_manifests)
    )
    _assert_root_identity(root, root_identity)
    expected_bundle_sha = _sha256(
        expected_bundle_file_sha256, name="expected_bundle_file_sha256"
    )
    expected_receipt_sha = _sha256(
        expected_receipt_file_sha256, name="expected_receipt_file_sha256"
    )
    bundle_path = _fixed_path(root, PRELABEL_BUNDLE_FILENAME)
    receipt_path = _fixed_path(root, PRELABEL_RECEIPT_FILENAME)
    observed_bundle = _decode_canonical_file(
        bundle_path,
        expected_sha256=expected_bundle_sha,
        maximum_bytes=MAX_PRELABEL_BUNDLE_BYTES,
        name="prelabel bundle",
    )
    _assert_root_identity(root, root_identity)
    if observed_bundle != expected_bundle:
        raise HarmBenchSelectionPrelabelError(
            "prelabel bundle differs from live typed inputs"
        )
    expected_receipt = _receipt_for_bundle(expected_bundle, expected_bundle_sha)
    observed_receipt = _decode_canonical_file(
        receipt_path,
        expected_sha256=expected_receipt_sha,
        maximum_bytes=MAX_PRELABEL_RECEIPT_BYTES,
        name="prelabel receipt",
    )
    _assert_root_identity(root, root_identity)
    if observed_receipt != expected_receipt:
        raise HarmBenchSelectionPrelabelError(
            "prelabel receipt differs from its live bundle"
        )
    return LoadedSelectionPrelabelBundle(
        protocol_canonical_sha256=protocol.canonical_sha256,
        bundle_file_sha256=expected_bundle_sha,
        receipt_file_sha256=expected_receipt_sha,
        private_root=root,
        bundle_path=bundle_path,
        receipt_path=receipt_path,
        bundle=_deep_freeze(observed_bundle),
        receipt=_deep_freeze(observed_receipt),
        _protocol=protocol,
        _prediction_artifacts=ordered_artifacts,
        _label_manifests=ordered_metadata,
        _effective_pairs=effective_pairs,
        _seal=_BUNDLE_SEAL,
    )


def write_selection_prelabel_bundle_once(
    *,
    private_root: str | Path,
    protocol: ProtocolV2Contract,
    prediction_artifacts: Sequence[LoadedPredictionArtifact],
    label_manifests: Sequence[SelectionLabelManifestMetadata],
) -> LoadedSelectionPrelabelBundle:
    """Validate all outcome-free inputs and durably publish the prelabel pair."""

    root = _validate_output_root(private_root)
    _reject_existing_terminal_state(root)
    bundle, _, _, _ = _build_bundle_state(
        protocol, prediction_artifacts, label_manifests
    )
    bundle_bytes = _canonical_json_bytes(bundle)
    bundle_sha = hashlib.sha256(bundle_bytes).hexdigest()
    receipt = _receipt_for_bundle(bundle, bundle_sha)
    observed_bundle_sha, observed_receipt_sha = _write_pair_once(
        root, bundle, receipt
    )
    if observed_bundle_sha != bundle_sha:
        raise HarmBenchSelectionPrelabelError(
            "prelabel bundle hash changed during publication"
        )
    return _load_prelabel_bundle_internal(
        private_root=root,
        protocol=protocol,
        prediction_artifacts=prediction_artifacts,
        label_manifests=label_manifests,
        expected_bundle_file_sha256=observed_bundle_sha,
        expected_receipt_file_sha256=observed_receipt_sha,
        marker_permitted=False,
    )


def load_selection_prelabel_bundle(
    *,
    private_root: str | Path,
    protocol: ProtocolV2Contract,
    prediction_artifacts: Sequence[LoadedPredictionArtifact],
    label_manifests: Sequence[SelectionLabelManifestMetadata],
    expected_bundle_file_sha256: str,
    expected_receipt_file_sha256: str,
) -> LoadedSelectionPrelabelBundle:
    """Live-load a completed prelabel pair only while no attempt exists."""

    return _load_prelabel_bundle_internal(
        private_root=private_root,
        protocol=protocol,
        prediction_artifacts=prediction_artifacts,
        label_manifests=label_manifests,
        expected_bundle_file_sha256=expected_bundle_file_sha256,
        expected_receipt_file_sha256=expected_receipt_file_sha256,
        marker_permitted=False,
    )


def _revalidate_prelabel_bundle(
    capability: object, *, marker_permitted: bool
) -> LoadedSelectionPrelabelBundle:
    if (
        type(capability) is not LoadedSelectionPrelabelBundle
        or capability._seal is not _BUNDLE_SEAL
    ):
        raise HarmBenchSelectionPrelabelError(
            "a loader-issued prelabel bundle capability is required"
        )
    rebuilt = _load_prelabel_bundle_internal(
        private_root=capability.private_root,
        protocol=capability._protocol,
        prediction_artifacts=capability._prediction_artifacts,
        label_manifests=capability._label_manifests,
        expected_bundle_file_sha256=capability.bundle_file_sha256,
        expected_receipt_file_sha256=capability.receipt_file_sha256,
        marker_permitted=marker_permitted,
    )
    excluded = {
        "bundle",
        "receipt",
        "_protocol",
        "_prediction_artifacts",
        "_label_manifests",
        "_effective_pairs",
        "_seal",
    }
    for item in fields(LoadedSelectionPrelabelBundle):
        if item.name not in excluded and getattr(capability, item.name) != getattr(
            rebuilt, item.name
        ):
            raise HarmBenchSelectionPrelabelError(
                f"loaded prelabel capability changed: {item.name}"
            )
    if (
        _plain_json(capability.bundle) != _plain_json(rebuilt.bundle)
        or _plain_json(capability.receipt) != _plain_json(rebuilt.receipt)
    ):
        raise HarmBenchSelectionPrelabelError(
            "loaded prelabel capability payload changed"
        )
    return capability


class _AttemptRuntime:
    """Process-local irreversible state; deliberately absent from artifacts."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ticket_issuance_state = "pending"
        self.terminal_failure = False
        self.failure_stage: str | None = None
        self.tickets: tuple[object, ...] = ()


def _attempt_marker_payload(
    prelabel: LoadedSelectionPrelabelBundle,
    *,
    attempt_nonce: str,
) -> dict[str, object]:
    if _SHA256_PATTERN.fullmatch(attempt_nonce) is None:
        raise HarmBenchSelectionPrelabelError("attempt nonce must be 256 random bits")
    return {
        "schema_version": ATTEMPT_MARKER_SCHEMA,
        "state": "attempt_marker_write_once_fsync_complete",
        "irreversible": True,
        "resume_or_rerun_permitted": False,
        "crash_before_final_is_terminal": True,
        "protocol_canonical_sha256": prelabel.protocol_canonical_sha256,
        "prelabel_bundle_filename": PRELABEL_BUNDLE_FILENAME,
        "prelabel_bundle_file_sha256": prelabel.bundle_file_sha256,
        "prelabel_receipt_filename": PRELABEL_RECEIPT_FILENAME,
        "prelabel_receipt_file_sha256": prelabel.receipt_file_sha256,
        "attempt_nonce": attempt_nonce,
        "evaluation_status": _evaluation_status(attempt_started=True),
        "next_permitted_operation": (
            "private_label_capability_activation_in_future_evaluator_only"
        ),
        "label_npz_access_occurred": False,
    }


@dataclass(frozen=True)
class AttemptStartedCapability:
    """In-process proof that the irreversible marker crossed its barrier."""

    protocol_canonical_sha256: str
    prelabel_bundle_file_sha256: str
    prelabel_receipt_file_sha256: str
    marker_file_sha256: str
    private_root: Path
    marker_path: Path
    marker: Mapping[str, object] = field(repr=False, compare=False)
    _prelabel: LoadedSelectionPrelabelBundle = field(repr=False, compare=False)
    _runtime: _AttemptRuntime = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self._seal is not _ATTEMPT_SEAL
            or type(self._runtime) is not _AttemptRuntime
        ):
            raise HarmBenchSelectionPrelabelError(
                "attempt capabilities can only be created after marker fsync"
            )
        for name in (
            "protocol_canonical_sha256",
            "prelabel_bundle_file_sha256",
            "prelabel_receipt_file_sha256",
            "marker_file_sha256",
        ):
            _sha256(getattr(self, name), name=name)
        if (
            not isinstance(self.private_root, Path)
            or not self.private_root.is_absolute()
            or not isinstance(self.marker_path, Path)
            or not self.marker_path.is_absolute()
        ):
            raise HarmBenchSelectionPrelabelError(
                "attempt capability paths must be absolute"
            )


def start_selection_evaluation_attempt(
    prelabel: LoadedSelectionPrelabelBundle,
) -> AttemptStartedCapability:
    """Durably publish the one-shot marker; this function never opens labels."""

    if (
        type(prelabel) is not LoadedSelectionPrelabelBundle
        or prelabel._seal is not _BUNDLE_SEAL
    ):
        raise HarmBenchSelectionPrelabelError(
            "a loader-issued prelabel capability is required before attempt start"
        )
    root = _validate_output_root(prelabel.private_root)
    root_identity = _plain_directory_stat(root, name="prelabel root")
    # This check intentionally precedes all live input work.  A marker from a
    # dead process is not a resume invitation and cannot be cleaned up here.
    _reject_existing_terminal_state(root)
    live = _revalidate_prelabel_bundle(prelabel, marker_permitted=False)
    _assert_root_identity(root, root_identity)
    marker_path = _fixed_path(root, ATTEMPT_MARKER_FILENAME)
    marker = _attempt_marker_payload(live, attempt_nonce=secrets.token_hex(32))
    encoded = _canonical_json_bytes(marker)
    if len(encoded) > MAX_ATTEMPT_MARKER_BYTES:
        raise HarmBenchSelectionPrelabelError("attempt marker exceeds byte budget")
    temporary = _temporary_bytes(root, marker_path, encoded)
    _assert_root_identity(root, root_identity)
    _publish_once(temporary, marker_path)
    _assert_root_identity(root, root_identity)
    _sync_directory(root)
    _assert_root_identity(root, root_identity)
    marker_sha = hashlib.sha256(encoded).hexdigest()
    observed = _decode_canonical_file(
        marker_path,
        expected_sha256=marker_sha,
        maximum_bytes=MAX_ATTEMPT_MARKER_BYTES,
        name="attempt marker",
    )
    _assert_root_identity(root, root_identity)
    if observed != marker:
        raise HarmBenchSelectionPrelabelError(
            "attempt marker differs from the fsynced prelabel binding"
        )
    return AttemptStartedCapability(
        protocol_canonical_sha256=live.protocol_canonical_sha256,
        prelabel_bundle_file_sha256=live.bundle_file_sha256,
        prelabel_receipt_file_sha256=live.receipt_file_sha256,
        marker_file_sha256=marker_sha,
        private_root=root,
        marker_path=marker_path,
        marker=_deep_freeze(observed),
        _prelabel=live,
        _runtime=_AttemptRuntime(),
        _seal=_ATTEMPT_SEAL,
    )


def _revalidate_attempt_started_capability(
    capability: object,
) -> AttemptStartedCapability:
    """Live-reverify marker, bundle, artifacts, pairs and label manifests."""

    if (
        type(capability) is not AttemptStartedCapability
        or capability._seal is not _ATTEMPT_SEAL
        or type(capability._runtime) is not _AttemptRuntime
    ):
        raise HarmBenchSelectionPrelabelError(
            "an in-process loader-sealed attempt capability is required"
        )
    with capability._runtime.lock:
        if capability._runtime.terminal_failure:
            raise HarmBenchSelectionPrelabelError(
                "attempt is terminal after an earlier post-marker failure"
            )
    try:
        live_prelabel = _revalidate_prelabel_bundle(
            capability._prelabel, marker_permitted=True
        )
        marker_plain = _plain_json(capability.marker)
        if not isinstance(marker_plain, Mapping):
            raise HarmBenchSelectionPrelabelError(
                "attempt marker capability is not a JSON object"
            )
        attempt_nonce = marker_plain.get("attempt_nonce")
        if (
            not isinstance(attempt_nonce, str)
            or _SHA256_PATTERN.fullmatch(attempt_nonce) is None
        ):
            raise HarmBenchSelectionPrelabelError("attempt nonce changed")
        expected = _attempt_marker_payload(
            live_prelabel, attempt_nonce=attempt_nonce
        )
        observed = _decode_canonical_file(
            capability.marker_path,
            expected_sha256=capability.marker_file_sha256,
            maximum_bytes=MAX_ATTEMPT_MARKER_BYTES,
            name="attempt marker",
        )
        if observed != expected or marker_plain != observed:
            raise HarmBenchSelectionPrelabelError(
                "attempt marker/capability binding changed"
            )
        scalar_matches = {
            "protocol_canonical_sha256": live_prelabel.protocol_canonical_sha256,
            "prelabel_bundle_file_sha256": live_prelabel.bundle_file_sha256,
            "prelabel_receipt_file_sha256": live_prelabel.receipt_file_sha256,
            "private_root": live_prelabel.private_root,
            "marker_path": _fixed_path(
                live_prelabel.private_root, ATTEMPT_MARKER_FILENAME
            ),
        }
        if any(
            getattr(capability, name) != expected_value
            for name, expected_value in scalar_matches.items()
        ):
            raise HarmBenchSelectionPrelabelError(
                "attempt capability changed after marker publication"
            )
        return capability
    except BaseException:
        _mark_attempt_terminal_failure(capability, "live_revalidation")
        raise


def _mark_attempt_terminal_failure(
    attempt: AttemptStartedCapability, stage: str
) -> None:
    if (
        type(attempt) is not AttemptStartedCapability
        or type(attempt._runtime) is not _AttemptRuntime
    ):
        return
    with attempt._runtime.lock:
        attempt._runtime.terminal_failure = True
        attempt._runtime.failure_stage = stage
        if attempt._runtime.ticket_issuance_state != "issued":
            attempt._runtime.ticket_issuance_state = "failed"


def _immutable_int64(values: object) -> np.ndarray:
    contiguous = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.int64).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


class _TicketRuntime:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state = "pending"
        self.consumed_irreversibly = False


@dataclass(frozen=True)
class _AttemptBoundLabelAccessTicket:
    """Private one-shot authorization minted only from one live attempt."""

    dataset_id: str
    rows: int
    ordered_protocol_row_alignment_sha256: str
    class_order_sha256: str
    manifest_file_sha256: str
    artifact_file_sha256: str
    protocol_canonical_sha256: str
    marker_file_sha256: str
    prelabel_bundle_file_sha256: str
    prelabel_receipt_file_sha256: str
    ticket_binding_sha256: str
    expected_protocol_row_ids: np.ndarray = field(repr=False, compare=False)
    expected_class_tokens: tuple[str, ...] = field(repr=False, compare=False)
    _metadata: SelectionLabelManifestMetadata = field(repr=False, compare=False)
    _attempt: AttemptStartedCapability = field(repr=False, compare=False)
    _runtime: _TicketRuntime = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self._seal is not _TICKET_SEAL
            or type(self._attempt) is not AttemptStartedCapability
            or self._attempt._seal is not _ATTEMPT_SEAL
            or type(self._runtime) is not _TicketRuntime
            or self.expected_protocol_row_ids.flags.writeable
        ):
            raise HarmBenchSelectionPrelabelError(
                "label-access tickets can only be minted from a live attempt"
            )
        for name in (
            "ordered_protocol_row_alignment_sha256",
            "class_order_sha256",
            "manifest_file_sha256",
            "artifact_file_sha256",
            "protocol_canonical_sha256",
            "marker_file_sha256",
            "prelabel_bundle_file_sha256",
            "prelabel_receipt_file_sha256",
            "ticket_binding_sha256",
        ):
            _sha256(getattr(self, name), name=name)


def _ticket_binding_payload(
    *,
    attempt: AttemptStartedCapability,
    metadata: SelectionLabelManifestMetadata,
    dataset_id: str,
    protocol_row_ids: np.ndarray,
    class_tokens: tuple[str, ...],
    class_order_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": "harmbench_erc_attempt_bound_label_access_ticket_v1",
        "dataset_id": dataset_id,
        "rows": len(protocol_row_ids),
        "ordered_protocol_row_alignment_sha256": _array_sha256(protocol_row_ids),
        "class_tokens_sha256": _array_sha256(
            np.asarray(class_tokens, dtype=np.str_)
        ),
        "class_order_sha256": class_order_sha256,
        "manifest_metadata": _metadata_record(metadata),
        "protocol_canonical_sha256": attempt.protocol_canonical_sha256,
        "marker_file_sha256": attempt.marker_file_sha256,
        "prelabel_bundle_file_sha256": attempt.prelabel_bundle_file_sha256,
        "prelabel_receipt_file_sha256": attempt.prelabel_receipt_file_sha256,
        "attempt_nonce": str(attempt.marker["attempt_nonce"]),
    }


def _issue_attempt_bound_label_access_tickets(
    attempt: AttemptStartedCapability,
) -> tuple[_AttemptBoundLabelAccessTicket, ...]:
    """Consume the exact attempt's whole two-ticket issuance right once."""

    if (
        type(attempt) is not AttemptStartedCapability
        or attempt._seal is not _ATTEMPT_SEAL
        or type(attempt._runtime) is not _AttemptRuntime
    ):
        raise HarmBenchSelectionPrelabelError(
            "a loader-sealed attempt is required for ticket issuance"
        )
    with attempt._runtime.lock:
        if (
            attempt._runtime.terminal_failure
            or attempt._runtime.ticket_issuance_state != "pending"
        ):
            attempt._runtime.terminal_failure = True
            attempt._runtime.failure_stage = "duplicate_ticket_issuance"
            attempt._runtime.ticket_issuance_state = "failed"
            raise HarmBenchSelectionPrelabelError(
                "the attempt's exact ticket suite may be issued only once"
            )
        attempt._runtime.ticket_issuance_state = "issuing"
    try:
        live = _revalidate_attempt_started_capability(attempt)
        metadata_by_dataset = {
            item.dataset_id: item for item in live._prelabel._label_manifests
        }
        tickets: list[_AttemptBoundLabelAccessTicket] = []
        for raw_row in live._prelabel.bundle["datasets"]:
            if not isinstance(raw_row, Mapping):
                raise HarmBenchSelectionPrelabelError(
                    "prelabel dataset roster changed before ticket issuance"
                )
            dataset_id = str(raw_row["dataset_id"])
            if dataset_id not in metadata_by_dataset:
                raise HarmBenchSelectionPrelabelError(
                    "ticket dataset lacks its exact label manifest"
                )
            metadata = metadata_by_dataset[dataset_id]
            protocol_ids = _immutable_int64(raw_row["query_protocol_row_ids"])
            class_tokens = tuple(str(value) for value in raw_row["class_tokens"])
            class_sha = str(raw_row["class_order_sha256"])
            payload = _ticket_binding_payload(
                attempt=live,
                metadata=metadata,
                dataset_id=dataset_id,
                protocol_row_ids=protocol_ids,
                class_tokens=class_tokens,
                class_order_sha256=class_sha,
            )
            tickets.append(
                _AttemptBoundLabelAccessTicket(
                    dataset_id=dataset_id,
                    rows=len(protocol_ids),
                    ordered_protocol_row_alignment_sha256=_array_sha256(
                        protocol_ids
                    ),
                    class_order_sha256=class_sha,
                    manifest_file_sha256=metadata.manifest_file_sha256,
                    artifact_file_sha256=metadata.artifact_file_sha256,
                    protocol_canonical_sha256=live.protocol_canonical_sha256,
                    marker_file_sha256=live.marker_file_sha256,
                    prelabel_bundle_file_sha256=(
                        live.prelabel_bundle_file_sha256
                    ),
                    prelabel_receipt_file_sha256=(
                        live.prelabel_receipt_file_sha256
                    ),
                    ticket_binding_sha256=_canonical_sha256(payload),
                    expected_protocol_row_ids=protocol_ids,
                    expected_class_tokens=class_tokens,
                    _metadata=metadata,
                    _attempt=live,
                    _runtime=_TicketRuntime(),
                    _seal=_TICKET_SEAL,
                )
            )
        result = tuple(tickets)
        if (
            tuple(item.dataset_id for item in result)
            != tuple(EXPECTED_SELECTION_DATASETS)
            or len({item.ticket_binding_sha256 for item in result}) != 2
        ):
            raise HarmBenchSelectionPrelabelError(
                "exact ordered two-ticket suite changed"
            )
        with attempt._runtime.lock:
            if (
                attempt._runtime.terminal_failure
                or attempt._runtime.ticket_issuance_state != "issuing"
            ):
                raise HarmBenchSelectionPrelabelError(
                    "ticket issuance collided with a terminal attempt failure"
                )
            attempt._runtime.tickets = result
            attempt._runtime.ticket_issuance_state = "issued"
        return result
    except BaseException:
        _mark_attempt_terminal_failure(attempt, "ticket_issuance")
        raise


def _begin_attempt_ticket_consumption(
    ticket: object,
) -> _AttemptBoundLabelAccessTicket:
    """Irreversibly consume a ticket before any label path operation."""

    if (
        type(ticket) is not _AttemptBoundLabelAccessTicket
        or ticket._seal is not _TICKET_SEAL
        or type(ticket._runtime) is not _TicketRuntime
    ):
        raise HarmBenchSelectionPrelabelError(
            "an attempt-bound loader-sealed label ticket is required"
        )
    ticket._runtime.lock.acquire()
    try:
        if ticket._runtime.state != "pending":
            _mark_attempt_terminal_failure(
                ticket._attempt, "duplicate_ticket_consumption"
            )
            ticket._runtime.state = "failed"
            raise HarmBenchSelectionPrelabelError(
                "label-access ticket has already been irreversibly consumed"
            )
        ticket._runtime.consumed_irreversibly = True
        ticket._runtime.state = "consuming"
        with ticket._attempt._runtime.lock:
            valid_suite = (
                not ticket._attempt._runtime.terminal_failure
                and ticket._attempt._runtime.ticket_issuance_state == "issued"
                and any(
                    ticket is issued
                    for issued in ticket._attempt._runtime.tickets
                )
            )
        if not valid_suite:
            ticket._runtime.state = "failed"
            _mark_attempt_terminal_failure(ticket._attempt, "ticket_consumption")
            raise HarmBenchSelectionPrelabelError(
                "ticket is not in this attempt's exact issued suite"
            )
        return ticket
    except BaseException:
        ticket._runtime.lock.release()
        raise


def _revalidate_consuming_attempt_ticket(
    ticket: _AttemptBoundLabelAccessTicket,
) -> SelectionLabelManifestMetadata:
    if (
        type(ticket) is not _AttemptBoundLabelAccessTicket
        or ticket._seal is not _TICKET_SEAL
        or ticket._runtime.state != "consuming"
        or not ticket._runtime.consumed_irreversibly
    ):
        raise HarmBenchSelectionPrelabelError(
            "ticket must be irreversibly consumed before label access"
        )
    live_attempt = _revalidate_attempt_started_capability(ticket._attempt)
    live_metadata = _labels._revalidate_manifest_metadata(  # noqa: SLF001
        ticket._metadata
    )
    payload = _ticket_binding_payload(
        attempt=live_attempt,
        metadata=live_metadata,
        dataset_id=ticket.dataset_id,
        protocol_row_ids=ticket.expected_protocol_row_ids,
        class_tokens=ticket.expected_class_tokens,
        class_order_sha256=ticket.class_order_sha256,
    )
    if (
        ticket.rows != len(ticket.expected_protocol_row_ids)
        or ticket.ordered_protocol_row_alignment_sha256
        != _array_sha256(ticket.expected_protocol_row_ids)
        or ticket.manifest_file_sha256 != live_metadata.manifest_file_sha256
        or ticket.artifact_file_sha256 != live_metadata.artifact_file_sha256
        or ticket.protocol_canonical_sha256
        != live_attempt.protocol_canonical_sha256
        or ticket.marker_file_sha256 != live_attempt.marker_file_sha256
        or ticket.prelabel_bundle_file_sha256
        != live_attempt.prelabel_bundle_file_sha256
        or ticket.prelabel_receipt_file_sha256
        != live_attempt.prelabel_receipt_file_sha256
        or ticket.ticket_binding_sha256 != _canonical_sha256(payload)
        or live_metadata.dataset_id != ticket.dataset_id
        or live_metadata.rows != ticket.rows
        or live_metadata.ordered_protocol_row_alignment_sha256
        != ticket.ordered_protocol_row_alignment_sha256
        or live_metadata.class_order_sha256 != ticket.class_order_sha256
    ):
        raise HarmBenchSelectionPrelabelError(
            "attempt-bound ticket changed before label access"
        )
    return live_metadata


def _finish_attempt_ticket_consumption(
    ticket: _AttemptBoundLabelAccessTicket, *, succeeded: bool
) -> None:
    try:
        if succeeded:
            if ticket._runtime.state != "consuming":
                _mark_attempt_terminal_failure(
                    ticket._attempt, "ticket_consumption_state"
                )
                ticket._runtime.state = "failed"
                raise HarmBenchSelectionPrelabelError(
                    "ticket consumption state changed"
                )
            ticket._runtime.state = "consumed"
        else:
            ticket._runtime.state = "failed"
            _mark_attempt_terminal_failure(ticket._attempt, "label_activation")
    finally:
        ticket._runtime.lock.release()


def _validate_consumed_ticket_for_capability(
    ticket: object,
    *,
    attempt: AttemptStartedCapability,
    dataset_id: str,
    ticket_binding_sha256: str,
) -> _AttemptBoundLabelAccessTicket:
    if (
        type(ticket) is not _AttemptBoundLabelAccessTicket
        or ticket._seal is not _TICKET_SEAL
        or ticket._attempt is not attempt
    ):
        raise HarmBenchSelectionPrelabelError(
            "activated label does not originate from this exact attempt"
        )
    with ticket._runtime.lock:
        if (
            ticket._runtime.state != "consumed"
            or not ticket._runtime.consumed_irreversibly
            or ticket.dataset_id != dataset_id
            or ticket.ticket_binding_sha256 != ticket_binding_sha256
        ):
            raise HarmBenchSelectionPrelabelError(
                "activated label ticket is not successfully consumed"
            )
    return ticket


def _validate_activated_label_suite_for_attempt(
    attempt: AttemptStartedCapability,
    capabilities: Sequence[object],
) -> None:
    live = _revalidate_attempt_started_capability(attempt)
    with live._runtime.lock:
        if (
            live._runtime.terminal_failure
            or live._runtime.ticket_issuance_state != "issued"
            or len(live._runtime.tickets) != 2
        ):
            raise HarmBenchSelectionPrelabelError(
                "attempt does not have one successful exact ticket suite"
            )
        issued = tuple(live._runtime.tickets)
    if len(capabilities) != 2:
        raise HarmBenchSelectionPrelabelError(
            "exactly two activated labels are required"
        )
    by_dataset: dict[str, object] = {}
    for capability in capabilities:
        if type(capability) is not _labels.ActivatedSelectionLabelCapability:
            raise HarmBenchSelectionPrelabelError(
                "activated label capability type changed"
            )
        if capability.dataset_id in by_dataset:
            raise HarmBenchSelectionPrelabelError(
                "duplicate activated label dataset"
            )
        ticket = capability._origin._ticket  # noqa: SLF001
        _validate_consumed_ticket_for_capability(
            ticket,
            attempt=live,
            dataset_id=capability.dataset_id,
            ticket_binding_sha256=capability.ticket_binding_sha256,
        )
        by_dataset[capability.dataset_id] = ticket
    if tuple(by_dataset) != tuple(EXPECTED_SELECTION_DATASETS) or any(
        by_dataset[dataset_id] is not issued[index]
        for index, dataset_id in enumerate(EXPECTED_SELECTION_DATASETS)
    ):
        raise HarmBenchSelectionPrelabelError(
            "activated labels do not match the exact issued ticket suite"
        )


__all__ = [
    "ATTEMPT_MARKER_FILENAME",
    "ATTEMPT_MARKER_SCHEMA",
    "AttemptStartedCapability",
    "EXPLORATORY_STATUS",
    "FINAL_OUTPUT_FILENAME",
    "HarmBenchSelectionPrelabelError",
    "LoadedSelectionPrelabelBundle",
    "PRELABEL_BUNDLE_FILENAME",
    "PRELABEL_BUNDLE_SCHEMA",
    "PRELABEL_RECEIPT_FILENAME",
    "PRELABEL_RECEIPT_SCHEMA",
    "load_selection_prelabel_bundle",
    "start_selection_evaluation_attempt",
    "write_selection_prelabel_bundle_once",
]
