"""Stage-B Part 1 producers for the CARMA causal evidence chain.

This module persists the private fit protocol mapping, orchestrates an
independent-current-only fit OOF producer through an explicit fold callback,
and orchestrates fit-only utility OOF scores through an explicit seed callback.
Callbacks make the contract testable without starting real training.  The
production adapters added in the next stage must call the already audited
history-stripped trainer and must satisfy the same outputs.

No function in this module accepts model-selection labels or computes a
performance metric.  Public receipts are aggregate-only; row-level mappings,
probabilities and scores stay in private NPZ artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .causal_backbone_evidence_runner import (
    CURRENT_ONLY_FIT_ARTIFACT_SCHEMA,
    EXPECTED_SEEDS,
    FIT_ROLE,
    INDEPENDENT_CURRENT_ONLY_PROTOCOL,
    UTILITY_OOF_SCORE_SCHEMA,
    CheckpointManifest,
    FitOnlyProducerView,
    FitRoleView,
    StageAContractError,
    _array_sha256,
    _canonical_sha256,
    _file_sha256,
    _load_receipt,
    _materialize_fit_role,
    _require_sha256,
    _single_text,
    build_checkpoint_manifest,
    load_fit_only_producer_view,
    validate_current_only_fit_artifact,
    validate_utility_oof_score_artifact,
    verify_fit_receipt_inputs,
)


FIT_PROTOCOL_MAP_SCHEMA = "carma_fit_protocol_map_private_v1"
CURRENT_ONLY_FIT_PRODUCER_RECEIPT_SCHEMA = (
    "carma_independent_current_only_fit_producer_receipt_v1"
)
UTILITY_OOF_PRODUCER_RECEIPT_SCHEMA = "carma_utility_oof_producer_receipt_v1"


class StageBContractError(StageAContractError):
    """Raised when a Stage-B producer crosses a role or lineage boundary."""


def _atomic_savez_once(path: Path, values: Mapping[str, np.ndarray]) -> str:
    if path.suffix.lower() != ".npz":
        raise StageBContractError("private artifact must be an NPZ file")
    if path.exists():
        raise FileExistsError(f"private artifact already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)
    return _file_sha256(path)


def _atomic_json_once(path: Path, payload: Mapping[str, object]) -> str:
    if path.suffix.lower() != ".json":
        raise StageBContractError("producer receipt must be JSON")
    if path.exists():
        raise FileExistsError(f"producer receipt already exists: {path.name}")
    _validate_aggregate_producer_receipt(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return _file_sha256(path)


def _validate_aggregate_producer_receipt(payload: Mapping[str, object]) -> None:
    forbidden = {
        "groups",
        "group_ids",
        "row_ids",
        "protocol_row_ids",
        "labels",
        "probabilities",
        "scores",
        "contexts",
        "paths",
        "predictions",
    }

    def visit(value: object, location: tuple[str, ...]) -> None:
        if isinstance(value, np.ndarray):
            raise StageBContractError(
                f"aggregate producer receipt contains ndarray at {'.'.join(location)}"
            )
        if isinstance(value, Mapping):
            for key, child in value.items():
                name = str(key)
                tokens = set(name.lower().replace("-", "_").split("_"))
                if name.lower() in forbidden or "path" in tokens:
                    raise StageBContractError(
                        f"aggregate producer receipt contains private field "
                        f"{'.'.join((*location, name))}"
                    )
                visit(child, (*location, name))
            return
        if isinstance(value, (list, tuple)):
            if len(value) > 32:
                raise StageBContractError(
                    f"aggregate producer receipt contains overlong list at "
                    f"{'.'.join(location)}"
                )
            for index, child in enumerate(value):
                visit(child, (*location, str(index)))
            return
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise StageBContractError(
                f"aggregate producer receipt contains non-JSON value at "
                f"{'.'.join(location)}"
            )

    visit(payload, ())


def _integer_vector(value: np.ndarray, field: str, *, unique: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise StageBContractError(f"{field} must be a one-dimensional integer array")
    result = array.astype(np.int64, copy=True)
    if np.any(result < 0):
        raise StageBContractError(f"{field} contains a negative value")
    if unique and len(set(result.tolist())) != len(result):
        raise StageBContractError(f"{field} must contain unique values")
    return result


def _single_int(value: np.ndarray, field: str) -> int:
    array = np.asarray(value)
    if array.size != 1 or not np.issubdtype(array.dtype, np.integer):
        raise StageBContractError(f"{field} must contain one integer")
    return int(array.reshape(()))


def _single_bool(value: np.ndarray, field: str) -> bool:
    array = np.asarray(value)
    if array.size != 1 or array.dtype != np.bool_:
        raise StageBContractError(f"{field} must contain one boolean")
    return bool(array.reshape(()))


@dataclass(frozen=True)
class FitProtocolMap:
    artifact_path: Path
    dataset: str
    receipt_sha256: str
    manifest_sha256: str
    fit_arrays_contract_sha256: str
    protocol_row_ids: np.ndarray
    protocol_index_mapping_sha256: str
    artifact_attestation_sha256: str
    artifact_sha256: str


_FIT_MAP_KEYS = frozenset(
    {
        "schema_version",
        "dataset",
        "fit_preflight_receipt_sha256",
        "manifest_sha256",
        "fit_arrays_contract_sha256",
        "fit_protocol_row_ids",
        "matrix_fit_protocol_row_ids_sha256",
        "protocol_index_mapping_sha256",
        "contains_labels_groups_speakers_or_text",
        "artifact_attestation_sha256",
    }
)


def _fit_map_attestation(values: Mapping[str, np.ndarray]) -> str:
    return _canonical_sha256(
        {
            "schema_version": _single_text(values["schema_version"], "schema_version"),
            "dataset": _single_text(values["dataset"], "dataset"),
            "fit_preflight_receipt_sha256": _single_text(
                values["fit_preflight_receipt_sha256"], "fit_preflight_receipt_sha256"
            ),
            "manifest_sha256": _single_text(values["manifest_sha256"], "manifest_sha256"),
            "fit_arrays_contract_sha256": _single_text(
                values["fit_arrays_contract_sha256"], "fit_arrays_contract_sha256"
            ),
            "fit_protocol_row_ids_sha256": _array_sha256(
                np.asarray(values["fit_protocol_row_ids"])
            ),
            "protocol_index_mapping_sha256": _single_text(
                values["protocol_index_mapping_sha256"],
                "protocol_index_mapping_sha256",
            ),
            "contains_labels_groups_speakers_or_text": _single_bool(
                values["contains_labels_groups_speakers_or_text"],
                "contains_labels_groups_speakers_or_text",
            ),
        }
    )


def fit_protocol_map_values(
    fit: FitRoleView,
    *,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
) -> dict[str, np.ndarray]:
    """Build a private map bound to the write-once aggregate Stage-A receipt."""

    receipt_file = Path(receipt_path)
    receipt_sha = _require_sha256(expected_receipt_sha256, "expected_receipt_sha256")
    if _file_sha256(receipt_file) != receipt_sha:
        raise StageBContractError("fit preflight receipt file hash changed")
    receipt = _load_receipt(receipt_file)
    if receipt.get("dataset") != fit.dataset:
        raise StageBContractError("fit map dataset differs from receipt")
    manifest = receipt.get("manifest")
    contract = receipt.get("fit_contract")
    if not isinstance(manifest, Mapping) or not isinstance(contract, Mapping):
        raise StageBContractError("fit receipt lacks manifest/fit contract")
    mapping_sha = _canonical_sha256(
        [[int(index), int(value)] for index, value in enumerate(fit.protocol_row_ids)]
    )
    if contract.get("fit_arrays_contract_sha256") != fit.contract_sha256:
        raise StageBContractError("fit arrays differ from preflight receipt")
    if contract.get("protocol_index_mapping_sha256") != mapping_sha:
        raise StageBContractError("fit protocol mapping differs from preflight receipt")
    values: dict[str, np.ndarray] = {
        "schema_version": np.asarray(FIT_PROTOCOL_MAP_SCHEMA),
        "dataset": np.asarray(fit.dataset),
        "fit_preflight_receipt_sha256": np.asarray(receipt_sha),
        "manifest_sha256": np.asarray(
            _require_sha256(manifest.get("sha256"), "manifest.sha256")
        ),
        "fit_arrays_contract_sha256": np.asarray(fit.contract_sha256),
        "fit_protocol_row_ids": np.asarray(fit.protocol_row_ids, dtype=np.int64),
        "matrix_fit_protocol_row_ids_sha256": np.asarray(
            _array_sha256(np.asarray(fit.protocol_row_ids, dtype=np.int64))
        ),
        "protocol_index_mapping_sha256": np.asarray(mapping_sha),
        "contains_labels_groups_speakers_or_text": np.asarray(False),
    }
    values["artifact_attestation_sha256"] = np.asarray(_fit_map_attestation(values))
    return values


def validate_fit_protocol_map_values(
    values: Mapping[str, np.ndarray],
    *,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
) -> None:
    if set(values) != set(_FIT_MAP_KEYS):
        raise StageBContractError("fit protocol-map schema changed")
    if _single_text(values["schema_version"], "schema_version") != FIT_PROTOCOL_MAP_SCHEMA:
        raise StageBContractError("fit protocol-map version changed")
    receipt_sha = _require_sha256(expected_receipt_sha256, "expected_receipt_sha256")
    if _file_sha256(Path(receipt_path)) != receipt_sha:
        raise StageBContractError("fit preflight receipt file hash changed")
    receipt = _load_receipt(Path(receipt_path))
    if _single_text(values["fit_preflight_receipt_sha256"], "receipt sha") != receipt_sha:
        raise StageBContractError("fit protocol map is bound to another receipt")
    if _single_text(values["dataset"], "dataset") != receipt.get("dataset"):
        raise StageBContractError("fit protocol-map dataset differs from receipt")
    manifest = receipt.get("manifest")
    contract = receipt.get("fit_contract")
    assert isinstance(manifest, Mapping)
    assert isinstance(contract, Mapping)
    if _single_text(values["manifest_sha256"], "manifest_sha256") != manifest.get("sha256"):
        raise StageBContractError("fit protocol-map manifest differs from receipt")
    if _single_text(
        values["fit_arrays_contract_sha256"], "fit_arrays_contract_sha256"
    ) != contract.get("fit_arrays_contract_sha256"):
        raise StageBContractError("fit protocol-map arrays differ from receipt")
    protocol = _integer_vector(
        values["fit_protocol_row_ids"], "fit_protocol_row_ids", unique=True
    )
    if not len(protocol) or len(protocol) != int(contract.get("rows", -1)):
        raise StageBContractError("fit protocol-map row count changed")
    matrix_hash = _require_sha256(
        _single_text(
            values["matrix_fit_protocol_row_ids_sha256"],
            "matrix_fit_protocol_row_ids_sha256",
        ),
        "matrix_fit_protocol_row_ids_sha256",
    )
    if matrix_hash != _array_sha256(protocol):
        raise StageBContractError("fit protocol row matrix hash differs")
    mapping_sha = _canonical_sha256(
        [[int(index), int(value)] for index, value in enumerate(protocol)]
    )
    if (
        _single_text(values["protocol_index_mapping_sha256"], "mapping sha")
        != mapping_sha
        or mapping_sha != contract.get("protocol_index_mapping_sha256")
    ):
        raise StageBContractError("fit protocol index mapping differs")
    if _single_bool(
        values["contains_labels_groups_speakers_or_text"],
        "contains_labels_groups_speakers_or_text",
    ):
        raise StageBContractError("fit protocol map contains forbidden row material")
    observed_attestation = _require_sha256(
        _single_text(values["artifact_attestation_sha256"], "artifact attestation"),
        "artifact_attestation_sha256",
    )
    if observed_attestation != _fit_map_attestation(values):
        raise StageBContractError("fit protocol-map attestation differs")


def write_fit_protocol_map(
    fit: FitRoleView,
    *,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
    output_path: str | Path,
) -> FitProtocolMap:
    values = fit_protocol_map_values(
        fit,
        receipt_path=receipt_path,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    validate_fit_protocol_map_values(
        values,
        receipt_path=receipt_path,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    destination = Path(output_path)
    artifact_sha = _atomic_savez_once(destination, values)
    return FitProtocolMap(
        artifact_path=destination.resolve(),
        dataset=fit.dataset,
        receipt_sha256=_single_text(
            values["fit_preflight_receipt_sha256"], "fit_preflight_receipt_sha256"
        ),
        manifest_sha256=_single_text(values["manifest_sha256"], "manifest_sha256"),
        fit_arrays_contract_sha256=fit.contract_sha256,
        protocol_row_ids=np.asarray(fit.protocol_row_ids, dtype=np.int64).copy(),
        protocol_index_mapping_sha256=_single_text(
            values["protocol_index_mapping_sha256"], "protocol_index_mapping_sha256"
        ),
        artifact_attestation_sha256=_single_text(
            values["artifact_attestation_sha256"], "artifact_attestation_sha256"
        ),
        artifact_sha256=artifact_sha,
    )


def load_fit_protocol_map(
    path: str | Path,
    *,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
) -> FitProtocolMap:
    artifact = Path(path)
    if artifact.suffix.lower() != ".npz":
        raise StageBContractError("fit protocol map must be NPZ")
    with np.load(artifact, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    validate_fit_protocol_map_values(
        values,
        receipt_path=receipt_path,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    return FitProtocolMap(
        artifact_path=artifact.resolve(),
        dataset=_single_text(values["dataset"], "dataset"),
        receipt_sha256=_single_text(
            values["fit_preflight_receipt_sha256"], "fit_preflight_receipt_sha256"
        ),
        manifest_sha256=_single_text(values["manifest_sha256"], "manifest_sha256"),
        fit_arrays_contract_sha256=_single_text(
            values["fit_arrays_contract_sha256"], "fit_arrays_contract_sha256"
        ),
        protocol_row_ids=np.asarray(values["fit_protocol_row_ids"], dtype=np.int64),
        protocol_index_mapping_sha256=_single_text(
            values["protocol_index_mapping_sha256"], "protocol_index_mapping_sha256"
        ),
        artifact_attestation_sha256=_single_text(
            values["artifact_attestation_sha256"], "artifact_attestation_sha256"
        ),
        artifact_sha256=_file_sha256(artifact),
    )


def _verify_fit_protocol_map_file(
    fit_map: FitProtocolMap,
    *,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
) -> None:
    """Re-open the private map and reject stale/in-memory-only lineage claims."""

    if _file_sha256(fit_map.artifact_path) != fit_map.artifact_sha256:
        raise StageBContractError("private fit protocol-map file hash changed")
    observed = load_fit_protocol_map(
        fit_map.artifact_path,
        receipt_path=receipt_path,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    scalar_pairs = (
        (observed.dataset, fit_map.dataset),
        (observed.receipt_sha256, fit_map.receipt_sha256),
        (observed.manifest_sha256, fit_map.manifest_sha256),
        (observed.fit_arrays_contract_sha256, fit_map.fit_arrays_contract_sha256),
        (
            observed.protocol_index_mapping_sha256,
            fit_map.protocol_index_mapping_sha256,
        ),
        (observed.artifact_attestation_sha256, fit_map.artifact_attestation_sha256),
        (observed.artifact_sha256, fit_map.artifact_sha256),
    )
    if any(left != right for left, right in scalar_pairs) or not np.array_equal(
        observed.protocol_row_ids, fit_map.protocol_row_ids
    ):
        raise StageBContractError("fit protocol-map object differs from its private file")


def _assert_producer_sidecar_lineage(
    producer: FitOnlyProducerView,
    receipt: Mapping[str, object],
) -> None:
    if producer.dataset != receipt.get("dataset"):
        raise StageBContractError("producer dataset differs from fit receipt")
    manifest = receipt.get("manifest")
    sidecars = receipt.get("sidecars")
    assert isinstance(manifest, Mapping)
    assert isinstance(sidecars, Mapping)
    fit = sidecars.get("fit")
    selection = sidecars.get("model_selection")
    assert isinstance(fit, Mapping)
    assert isinstance(selection, Mapping)
    required = {
        "source_sidecar_manifest_sha256": manifest.get("sha256"),
        f"source_{FIT_ROLE}_features_sha256": fit.get("feature_sha256"),
        f"source_{FIT_ROLE}_labels_sha256": fit.get("label_sha256"),
        "source_model_selection_features_sha256": selection.get("feature_sha256"),
        "source_model_selection_labels_sha256": selection.get("label_sha256"),
    }
    missing = sorted(name for name in required if name not in producer.source_hashes)
    changed = sorted(
        name
        for name, digest in required.items()
        if name in producer.source_hashes and producer.source_hashes[name] != digest
    )
    if missing or changed:
        raise StageBContractError(
            f"producer/receipt sidecar lineage mismatch: missing={missing}, changed={changed}"
        )


def align_fit_protocol_to_producer(
    fit_map: FitProtocolMap,
    producer: FitOnlyProducerView,
) -> tuple[np.ndarray, np.ndarray]:
    """Return role-local -> producer-row and role-local -> fit-position maps."""

    if fit_map.dataset != producer.dataset:
        raise StageBContractError("fit map and producer dataset differ")
    producer_by_protocol = {
        int(protocol): int(index)
        for index, protocol in enumerate(producer.protocol_row_ids)
    }
    try:
        combined = np.asarray(
            [producer_by_protocol[int(value)] for value in fit_map.protocol_row_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise StageBContractError("fit protocol row is absent from producer") from error
    fit_position = {
        int(query): int(index) for index, query in enumerate(producer.fit_query_indices)
    }
    if set(combined.tolist()) != set(producer.fit_query_indices.tolist()):
        raise StageBContractError("fit protocol map does not equal producer fit query set")
    return combined, np.asarray([fit_position[int(value)] for value in combined], dtype=np.int64)


@dataclass(frozen=True)
class HistoryFreeFoldRequest:
    dataset: str
    seed: int
    fold: int
    train_indices: np.ndarray
    heldout_indices: np.ndarray
    train_texts: tuple[str, ...]
    train_audio: np.ndarray
    train_video: np.ndarray
    train_labels: np.ndarray
    train_group_tokens: np.ndarray
    train_speaker_tokens: np.ndarray
    train_turns: np.ndarray
    train_histories: tuple[tuple[int, ...], ...]
    heldout_texts: tuple[str, ...]
    heldout_audio: np.ndarray
    heldout_video: np.ndarray
    heldout_group_tokens: np.ndarray
    heldout_speaker_tokens: np.ndarray
    heldout_turns: np.ndarray
    heldout_histories: tuple[tuple[int, ...], ...]
    heldout_labels_materialized: bool
    producer_source_identity_sha256: str
    checkpoint_root: Path
    model_config_sha256: str
    run_config_sha256: str


@dataclass(frozen=True)
class CurrentOnlyFoldOutput:
    probability: np.ndarray
    current_only_source_identity_sha256: str


CurrentOnlyFoldCallback = Callable[[HistoryFreeFoldRequest], CurrentOnlyFoldOutput]


@dataclass(frozen=True)
class CurrentOnlyFitProduction:
    artifact_path: Path
    artifact_sha256: str
    receipt_path: Path
    receipt_sha256: str
    checkpoint_manifest: CheckpointManifest
    current_only_source_identity_sha256: str


def _validated_fold_assignment(
    fit: FitRoleView,
    fold_by_seed_row: np.ndarray,
    *,
    outer_folds: int,
) -> np.ndarray:
    values = np.asarray(fold_by_seed_row)
    if values.shape != (len(EXPECTED_SEEDS), fit.rows) or not np.issubdtype(
        values.dtype, np.integer
    ):
        raise StageBContractError("current-only fold assignment must be seed/fit-row aligned")
    result = values.astype(np.int32, copy=True)
    if int(outer_folds) < 2 or np.any((result < 0) | (result >= int(outer_folds))):
        raise StageBContractError("current-only fold assignment is outside outer folds")
    group_tokens = np.asarray(fit.groups).astype(str)
    for seed_index in range(len(EXPECTED_SEEDS)):
        if set(np.unique(result[seed_index]).tolist()) != set(range(int(outer_folds))):
            raise StageBContractError("each current-only seed must cover every outer fold")
        for group in np.unique(group_tokens):
            if len(np.unique(result[seed_index, group_tokens == group])) != 1:
                raise StageBContractError("current-only fold assignment splits a whole group")
    return result


def _current_fit_receipt(
    *,
    dataset: str,
    fit_preflight_receipt_sha256: str,
    fit_map: FitProtocolMap,
    producer: FitOnlyProducerView,
    model_config_sha256: str,
    run_config_sha256: str,
    source_code_sha256: str,
    runtime_environment_sha256: str,
    fold_assignment_sha256: str,
    current_identity: str,
    checkpoint_manifest: CheckpointManifest,
    artifact_sha256: str,
) -> dict[str, object]:
    receipt = {
        "schema_version": CURRENT_ONLY_FIT_PRODUCER_RECEIPT_SCHEMA,
        "status": "independent_current_only_fit_oof_complete_not_performance_evidence",
        "dataset": dataset,
        "claim_boundary": (
            "History-stripped fit-role OOF artifact production only; no model-selection "
            "payload and no performance metric were consumed."
        ),
        "lineage": {
            "fit_preflight_receipt_sha256": fit_preflight_receipt_sha256,
            "fit_protocol_map_sha256": fit_map.artifact_sha256,
            "producer_file_sha256": producer.producer_file_sha256,
            "producer_source_identity_sha256": producer.source_identity_sha256,
            "model_config_sha256": model_config_sha256,
            "run_config_sha256": run_config_sha256,
            "source_code_sha256": source_code_sha256,
            "runtime_environment_sha256": runtime_environment_sha256,
            "fold_assignment_sha256": fold_assignment_sha256,
            "current_only_source_identity_sha256": current_identity,
            "checkpoint_manifest_sha256": checkpoint_manifest.manifest_sha256,
            "private_fit_artifact_sha256": artifact_sha256,
        },
        "training_contract": {
            "seeds": list(EXPECTED_SEEDS),
            "outer_folds": checkpoint_manifest.outer_folds,
            "history_training_items_consumed": 0,
            "history_inference_items_consumed": 0,
            "checkpoint_file_count": len(checkpoint_manifest.records),
            "fit_query_count": len(producer.fit_query_indices),
            "one_oof_probability_per_seed_and_fit_query": True,
            "selection_payload_consumed": False,
        },
        "public_artifact_policy": {
            "aggregate_only": True,
            "contains_row_level_values": False,
            "contains_private_paths": False,
            "contains_performance_metrics": False,
        },
    }
    _validate_aggregate_producer_receipt(receipt)
    return receipt


def produce_independent_current_only_fit_oof(
    *,
    fit: FitRoleView,
    fit_map: FitProtocolMap,
    producer: FitOnlyProducerView,
    fit_preflight_receipt_path: str | Path,
    expected_fit_preflight_receipt_sha256: str,
    fold_by_seed_row: np.ndarray,
    outer_folds: int,
    checkpoint_root: str | Path,
    artifact_path: str | Path,
    producer_receipt_path: str | Path,
    model_config_sha256: str,
    run_config_sha256: str,
    source_code_sha256: str,
    runtime_environment_sha256: str,
    fold_callback: CurrentOnlyFoldCallback,
) -> CurrentOnlyFitProduction:
    """Produce fit OOF probabilities without accepting any selection payload."""

    preflight_sha = _require_sha256(
        expected_fit_preflight_receipt_sha256,
        "expected_fit_preflight_receipt_sha256",
    )
    if _file_sha256(Path(fit_preflight_receipt_path)) != preflight_sha:
        raise StageBContractError("fit preflight receipt file hash changed")
    receipt = _load_receipt(Path(fit_preflight_receipt_path))
    _assert_producer_sidecar_lineage(producer, receipt)
    _verify_fit_protocol_map_file(
        fit_map,
        receipt_path=fit_preflight_receipt_path,
        expected_receipt_sha256=preflight_sha,
    )
    if fit_map.receipt_sha256 != preflight_sha or fit_map.dataset != fit.dataset:
        raise StageBContractError("fit protocol map differs from fit preflight")
    if fit_map.fit_arrays_contract_sha256 != fit.contract_sha256:
        raise StageBContractError("fit materialisation differs from private fit map")
    if not np.array_equal(fit_map.protocol_row_ids, fit.protocol_row_ids):
        raise StageBContractError("fit protocol rows differ from private fit map")
    _, fit_positions = align_fit_protocol_to_producer(fit_map, producer)
    folds = _validated_fold_assignment(
        fit, fold_by_seed_row, outer_folds=int(outer_folds)
    )
    for digest, field in (
        (model_config_sha256, "model_config_sha256"),
        (run_config_sha256, "run_config_sha256"),
        (source_code_sha256, "source_code_sha256"),
        (runtime_environment_sha256, "runtime_environment_sha256"),
    ):
        _require_sha256(digest, field)

    classes = len(fit.label_order)
    probability = np.full(
        (len(EXPECTED_SEEDS), len(producer.fit_query_indices), classes),
        np.nan,
        dtype=np.float32,
    )
    fold_by_query = np.empty(
        (len(EXPECTED_SEEDS), len(producer.fit_query_indices)), dtype=np.int32
    )
    empty_histories = tuple(() for _ in range(fit.rows))
    current_identities: set[str] = set()
    root = Path(checkpoint_root)
    artifact_destination = Path(artifact_path)
    producer_receipt_destination = Path(producer_receipt_path)
    if root.exists():
        raise FileExistsError("current-only checkpoint root must not exist before training")
    if artifact_destination.exists() or producer_receipt_destination.exists():
        raise FileExistsError("current-only output artifact already exists")
    for seed_index, seed in enumerate(EXPECTED_SEEDS):
        for fold in range(int(outer_folds)):
            held = np.flatnonzero(folds[seed_index] == fold).astype(np.int64)
            train = np.flatnonzero(folds[seed_index] != fold).astype(np.int64)
            if not len(held) or not len(train):
                raise StageBContractError("current-only fold has empty train/heldout rows")
            request = HistoryFreeFoldRequest(
                dataset=fit.dataset,
                seed=seed,
                fold=fold,
                train_indices=train,
                heldout_indices=held,
                train_texts=tuple(fit.texts[int(index)] for index in train),
                train_audio=np.asarray(fit.audio[train]).copy(),
                train_video=np.asarray(fit.video[train]).copy(),
                train_labels=np.asarray(fit.labels[train]).copy(),
                train_group_tokens=np.asarray(fit.groups[train]).copy(),
                train_speaker_tokens=np.asarray(fit.speakers[train]).copy(),
                train_turns=np.asarray(fit.turns[train]).copy(),
                train_histories=tuple(empty_histories[int(index)] for index in train),
                heldout_texts=tuple(fit.texts[int(index)] for index in held),
                heldout_audio=np.asarray(fit.audio[held]).copy(),
                heldout_video=np.asarray(fit.video[held]).copy(),
                heldout_group_tokens=np.asarray(fit.groups[held]).copy(),
                heldout_speaker_tokens=np.asarray(fit.speakers[held]).copy(),
                heldout_turns=np.asarray(fit.turns[held]).copy(),
                heldout_histories=tuple(empty_histories[int(index)] for index in held),
                heldout_labels_materialized=False,
                producer_source_identity_sha256=producer.source_identity_sha256,
                checkpoint_root=root,
                model_config_sha256=model_config_sha256,
                run_config_sha256=run_config_sha256,
            )
            if (
                any(request.train_histories)
                or any(request.heldout_histories)
                or request.heldout_labels_materialized
            ):
                raise AssertionError("current-only producer constructed non-empty history")
            output = fold_callback(request)
            identity = _require_sha256(
                output.current_only_source_identity_sha256,
                "current_only_source_identity_sha256",
            )
            if identity == producer.source_identity_sha256:
                raise StageBContractError("current-only fold reused history source identity")
            current_identities.add(identity)
            held_probability = np.asarray(output.probability)
            if held_probability.shape != (len(held), classes) or not np.issubdtype(
                held_probability.dtype, np.floating
            ):
                raise StageBContractError("current-only fold probability shape changed")
            if (
                not np.isfinite(held_probability).all()
                or np.any(held_probability < 0.0)
                or not np.allclose(
                    held_probability.sum(axis=1), 1.0, rtol=1.0e-5, atol=1.0e-6
                )
            ):
                raise StageBContractError("current-only fold emitted invalid probability")
            target_positions = fit_positions[held]
            if np.isfinite(probability[seed_index, target_positions]).any():
                raise StageBContractError("current-only fold predicted a fit query twice")
            probability[seed_index, target_positions] = held_probability.astype(np.float32)
            fold_by_query[seed_index, target_positions] = int(fold)
    if len(current_identities) != 1 or not np.isfinite(probability).all():
        raise StageBContractError("current-only identity/OOF coverage is incomplete")
    current_identity = next(iter(current_identities))
    checkpoint_manifest = build_checkpoint_manifest(
        root, seeds=EXPECTED_SEEDS, outer_folds=int(outer_folds)
    )
    if checkpoint_manifest.manifest_sha256 == producer.checkpoint_manifest_sha256:
        raise StageBContractError("current-only checkpoints reused history checkpoint manifest")
    artifact_values: dict[str, np.ndarray] = {
        "schema_version": np.asarray(CURRENT_ONLY_FIT_ARTIFACT_SCHEMA),
        "dataset": np.asarray(producer.dataset),
        "dataset_label_order": np.asarray(producer.label_order),
        "seeds": np.asarray(EXPECTED_SEEDS, dtype=np.int64),
        "outer_folds": np.asarray(int(outer_folds), dtype=np.int64),
        "fit_query_indices": producer.fit_query_indices.copy(),
        "fit_cluster_codes": producer.fit_cluster_codes.copy(),
        "fit_probability_oof": probability,
        "fit_fold_by_seed_query": fold_by_query,
        "producer_source_identity_sha256": np.asarray(producer.source_identity_sha256),
        "current_only_source_identity_sha256": np.asarray(current_identity),
        "history_backbone_checkpoint_manifest_sha256": np.asarray(
            producer.checkpoint_manifest_sha256
        ),
        "checkpoint_manifest_sha256": np.asarray(checkpoint_manifest.manifest_sha256),
        "training_protocol": np.asarray(INDEPENDENT_CURRENT_ONLY_PROTOCOL),
        "checkpoint_namespace": np.asarray("independent_current_only"),
        "history_training_items_consumed": np.asarray(0, dtype=np.int64),
        "history_inference_items_consumed": np.asarray(0, dtype=np.int64),
        "matrix_fit_probability_oof_sha256": np.asarray(_array_sha256(probability)),
        "fold_assignment_sha256": np.asarray(_array_sha256(fold_by_query)),
    }
    validate_current_only_fit_artifact(
        artifact_values,
        producer=producer,
        checkpoint_manifest=checkpoint_manifest,
    )
    artifact_sha = _atomic_savez_once(artifact_destination, artifact_values)
    producer_receipt = _current_fit_receipt(
        dataset=producer.dataset,
        fit_preflight_receipt_sha256=preflight_sha,
        fit_map=fit_map,
        producer=producer,
        model_config_sha256=model_config_sha256,
        run_config_sha256=run_config_sha256,
        source_code_sha256=source_code_sha256,
        runtime_environment_sha256=runtime_environment_sha256,
        fold_assignment_sha256=_array_sha256(fold_by_query),
        current_identity=current_identity,
        checkpoint_manifest=checkpoint_manifest,
        artifact_sha256=artifact_sha,
    )
    producer_receipt_sha = _atomic_json_once(
        producer_receipt_destination, producer_receipt
    )
    return CurrentOnlyFitProduction(
        artifact_destination,
        artifact_sha,
        producer_receipt_destination,
        producer_receipt_sha,
        checkpoint_manifest,
        current_identity,
    )


@dataclass(frozen=True)
class UtilityOOFSeedRequest:
    dataset: str
    seed: int
    fold: int
    train_task_indices: np.ndarray
    heldout_task_indices: np.ndarray
    train_query_indices: np.ndarray
    train_candidate_indices: np.ndarray
    train_cluster_codes: np.ndarray
    train_utility_probability: np.ndarray
    train_forward_targets: np.ndarray
    train_backward_targets: np.ndarray
    heldout_query_indices: np.ndarray
    heldout_candidate_indices: np.ndarray
    heldout_cluster_codes: np.ndarray
    heldout_utility_probability: np.ndarray
    heldout_targets_materialized: bool = False
    selection_payload_consumed: bool = False


@dataclass(frozen=True)
class UtilityOOFSeedOutput:
    decision_scores: np.ndarray


UtilityOOFSeedCallback = Callable[[UtilityOOFSeedRequest], UtilityOOFSeedOutput]


@dataclass(frozen=True)
class UtilityOOFProduction:
    artifact_path: Path
    artifact_sha256: str
    receipt_path: Path
    receipt_sha256: str
    score_source_identity_sha256: str


def _utility_score_source_identity(
    *,
    producer: FitOnlyProducerView,
    feature_schema_sha256: str,
    model_spec_sha256: str,
    folds: int,
    fold_assignment_sha256: str,
    score_matrix_sha256: str,
    ensemble_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "protocol": UTILITY_OOF_SCORE_SCHEMA,
            "dataset": producer.dataset,
            "producer_file_sha256": producer.producer_file_sha256,
            "producer_source_identity_sha256": producer.source_identity_sha256,
            "fit_task_sha256": producer.fit_tasks.task_sha256,
            "seeds": list(EXPECTED_SEEDS),
            "utility_oof_folds": int(folds),
            "feature_schema_sha256": feature_schema_sha256,
            "model_spec_sha256": model_spec_sha256,
            "fold_assignment_sha256": fold_assignment_sha256,
            "decision_score_oof_by_seed_sha256": score_matrix_sha256,
            "decision_score_oof_ensemble_sha256": ensemble_sha256,
            "selection_payload_consumed": False,
        }
    )


def _utility_receipt(
    *,
    producer: FitOnlyProducerView,
    fit_map: FitProtocolMap,
    preflight_receipt_sha256: str,
    feature_schema_sha256: str,
    model_spec_sha256: str,
    folds: int,
    fold_assignment_sha256: str,
    score_source_identity_sha256: str,
    artifact_sha256: str,
) -> dict[str, object]:
    receipt = {
        "schema_version": UTILITY_OOF_PRODUCER_RECEIPT_SCHEMA,
        "status": "fit_only_utility_oof_scores_complete_not_performance_evidence",
        "dataset": producer.dataset,
        "claim_boundary": (
            "Fit-role group-OOF decision-score production only; no selection payload, "
            "row score, target, or performance metric is public."
        ),
        "lineage": {
            "fit_preflight_receipt_sha256": preflight_receipt_sha256,
            "fit_protocol_map_sha256": fit_map.artifact_sha256,
            "producer_file_sha256": producer.producer_file_sha256,
            "producer_source_identity_sha256": producer.source_identity_sha256,
            "fit_task_sha256": producer.fit_tasks.task_sha256,
            "feature_schema_sha256": feature_schema_sha256,
            "model_spec_sha256": model_spec_sha256,
            "fold_assignment_sha256": fold_assignment_sha256,
            "score_source_identity_sha256": score_source_identity_sha256,
            "private_score_artifact_sha256": artifact_sha256,
        },
        "oof_contract": {
            "seeds": list(EXPECTED_SEEDS),
            "fold_count": int(folds),
            "fit_task_count": len(producer.fit_tasks),
            "whole_cluster_oof": True,
            "selection_payload_consumed": False,
            "labels_or_targets_serialized": False,
            "performance_metric_computed": False,
        },
        "public_artifact_policy": {
            "aggregate_only": True,
            "contains_row_level_values": False,
            "contains_private_paths": False,
            "contains_performance_metrics": False,
        },
    }
    _validate_aggregate_producer_receipt(receipt)
    return receipt


def produce_utility_oof_scores(
    *,
    producer: FitOnlyProducerView,
    fit_map: FitProtocolMap,
    fit_preflight_receipt_path: str | Path,
    expected_fit_preflight_receipt_sha256: str,
    feature_schema_sha256: str,
    model_spec_sha256: str,
    utility_oof_folds: int,
    fold_by_seed_task: np.ndarray,
    artifact_path: str | Path,
    producer_receipt_path: str | Path,
    seed_callback: UtilityOOFSeedCallback,
) -> UtilityOOFProduction:
    """Produce fit-only utility OOF scores; selection has no API parameter."""

    preflight_sha = _require_sha256(
        expected_fit_preflight_receipt_sha256,
        "expected_fit_preflight_receipt_sha256",
    )
    if _file_sha256(Path(fit_preflight_receipt_path)) != preflight_sha:
        raise StageBContractError("fit preflight receipt file hash changed")
    receipt = _load_receipt(Path(fit_preflight_receipt_path))
    _assert_producer_sidecar_lineage(producer, receipt)
    _verify_fit_protocol_map_file(
        fit_map,
        receipt_path=fit_preflight_receipt_path,
        expected_receipt_sha256=preflight_sha,
    )
    if fit_map.receipt_sha256 != preflight_sha or fit_map.dataset != producer.dataset:
        raise StageBContractError("fit protocol map differs from utility producer")
    align_fit_protocol_to_producer(fit_map, producer)
    _require_sha256(feature_schema_sha256, "feature_schema_sha256")
    _require_sha256(model_spec_sha256, "model_spec_sha256")
    folds = int(utility_oof_folds)
    if folds < 2:
        raise StageBContractError("utility OOF requires at least two folds")
    task_queries = producer.fit_tasks.query_indices
    query_cluster = {
        int(query): int(cluster)
        for query, cluster in zip(
            producer.fit_query_indices, producer.fit_cluster_codes, strict=True
        )
    }
    task_clusters = np.asarray(
        [query_cluster[int(query)] for query in task_queries], dtype=np.int64
    )
    fold_values = np.asarray(fold_by_seed_task)
    if fold_values.shape != (len(EXPECTED_SEEDS), len(task_queries)) or not np.issubdtype(
        fold_values.dtype, np.integer
    ):
        raise StageBContractError("utility fold assignment must be seed/task aligned")
    fold_by_task = fold_values.astype(np.int32, copy=True)
    if np.any((fold_by_task < 0) | (fold_by_task >= folds)):
        raise StageBContractError("utility fold assignment is outside registered folds")
    for seed_index in range(len(EXPECTED_SEEDS)):
        if set(np.unique(fold_by_task[seed_index]).tolist()) != set(range(folds)):
            raise StageBContractError("each utility seed must cover every OOF fold")
        for cluster in np.unique(task_clusters):
            if len(np.unique(fold_by_task[seed_index, task_clusters == cluster])) != 1:
                raise StageBContractError("utility fold assignment split one cluster")
    artifact_destination = Path(artifact_path)
    receipt_destination = Path(producer_receipt_path)
    if artifact_destination.exists() or receipt_destination.exists():
        raise FileExistsError("utility OOF output artifact already exists")
    scores = np.full(
        (len(EXPECTED_SEEDS), len(task_queries)), np.nan, dtype=np.float64
    )
    for seed_index, seed in enumerate(EXPECTED_SEEDS):
        for fold in range(folds):
            held = np.flatnonzero(fold_by_task[seed_index] == fold).astype(np.int64)
            train = np.flatnonzero(fold_by_task[seed_index] != fold).astype(np.int64)
            if not len(held) or not len(train):
                raise StageBContractError("utility OOF fold has empty train/heldout tasks")
            request = UtilityOOFSeedRequest(
                dataset=producer.dataset,
                seed=seed,
                fold=fold,
                train_task_indices=train,
                heldout_task_indices=held,
                train_query_indices=task_queries[train].copy(),
                train_candidate_indices=producer.fit_tasks.candidate_indices[train].copy(),
                train_cluster_codes=task_clusters[train].copy(),
                train_utility_probability=producer.fit_utility_probability[
                    seed_index, train
                ].copy(),
                train_forward_targets=producer.fit_forward_utility[
                    seed_index, train
                ].copy(),
                train_backward_targets=producer.fit_backward_utility[
                    seed_index, train
                ].copy(),
                heldout_query_indices=task_queries[held].copy(),
                heldout_candidate_indices=producer.fit_tasks.candidate_indices[
                    held
                ].copy(),
                heldout_cluster_codes=task_clusters[held].copy(),
                heldout_utility_probability=producer.fit_utility_probability[
                    seed_index, held
                ].copy(),
                heldout_targets_materialized=False,
                selection_payload_consumed=False,
            )
            output = seed_callback(request)
            held_scores = np.asarray(output.decision_scores)
            if held_scores.shape != (len(held),) or not np.issubdtype(
                held_scores.dtype, np.floating
            ) or not np.isfinite(held_scores).all():
                raise StageBContractError("utility callback score is not heldout-task aligned")
            if np.isfinite(scores[seed_index, held]).any():
                raise StageBContractError("utility callback scored one task twice")
            scores[seed_index, held] = held_scores.astype(np.float64)
    if not np.isfinite(scores).all():
        raise StageBContractError("utility OOF score coverage is incomplete")
    ensemble = scores.mean(axis=0)
    score_hash = _array_sha256(scores)
    ensemble_hash = _array_sha256(ensemble)
    fold_hash = _array_sha256(fold_by_task)
    score_identity = _utility_score_source_identity(
        producer=producer,
        feature_schema_sha256=feature_schema_sha256,
        model_spec_sha256=model_spec_sha256,
        folds=folds,
        fold_assignment_sha256=fold_hash,
        score_matrix_sha256=score_hash,
        ensemble_sha256=ensemble_hash,
    )
    values: dict[str, np.ndarray] = {
        "schema_version": np.asarray(UTILITY_OOF_SCORE_SCHEMA),
        "dataset": np.asarray(producer.dataset),
        "seeds": np.asarray(EXPECTED_SEEDS, dtype=np.int64),
        "producer_file_sha256": np.asarray(producer.producer_file_sha256),
        "producer_source_identity_sha256": np.asarray(producer.source_identity_sha256),
        "fit_task_sha256": np.asarray(producer.fit_tasks.task_sha256),
        "fit_task_query_indices": task_queries.copy(),
        "fit_task_cluster_codes": task_clusters,
        "utility_oof_folds": np.asarray(folds, dtype=np.int64),
        "fold_by_seed_task": fold_by_task,
        "decision_score_oof_by_seed": scores,
        "decision_score_oof_ensemble": ensemble,
        "matrix_decision_score_oof_by_seed_sha256": np.asarray(score_hash),
        "matrix_decision_score_oof_ensemble_sha256": np.asarray(ensemble_hash),
        "fold_assignment_sha256": np.asarray(fold_hash),
        "feature_schema_sha256": np.asarray(feature_schema_sha256),
        "model_spec_sha256": np.asarray(model_spec_sha256),
        "score_source_identity_sha256": np.asarray(score_identity),
        "selection_payload_consumed": np.asarray(False),
        "labels_or_targets_serialized": np.asarray(False),
    }
    validate_utility_oof_score_artifact(values, producer=producer)
    artifact_sha = _atomic_savez_once(artifact_destination, values)
    receipt_payload = _utility_receipt(
        producer=producer,
        fit_map=fit_map,
        preflight_receipt_sha256=preflight_sha,
        feature_schema_sha256=feature_schema_sha256,
        model_spec_sha256=model_spec_sha256,
        folds=folds,
        fold_assignment_sha256=fold_hash,
        score_source_identity_sha256=score_identity,
        artifact_sha256=artifact_sha,
    )
    receipt_sha = _atomic_json_once(receipt_destination, receipt_payload)
    return UtilityOOFProduction(
        artifact_destination,
        artifact_sha,
        receipt_destination,
        receipt_sha,
        score_identity,
    )


def materialize_verified_fit_for_stage_b(
    *,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
    dataset: str,
    sidecar_dir: str | Path,
    manifest_path: str | Path,
    config_paths: Mapping[str, str | Path],
    code_paths: Mapping[str, str | Path],
    environment: Mapping[str, object] | None = None,
) -> FitRoleView:
    """Cross-process Stage-B fit loader after complete receipt reverification."""

    receipt, sidecars = verify_fit_receipt_inputs(
        receipt_path=receipt_path,
        expected_receipt_sha256=expected_receipt_sha256,
        dataset=dataset,
        sidecar_dir=sidecar_dir,
        manifest_path=manifest_path,
        config_paths=config_paths,
        code_paths=code_paths,
        environment=environment,
    )
    fit = _materialize_fit_role(sidecars)
    contract = receipt.get("fit_contract")
    assert isinstance(contract, Mapping)
    if contract.get("fit_arrays_contract_sha256") != fit.contract_sha256:
        raise StageBContractError("Stage-B fit materialisation differs from receipt")
    return fit


def load_private_npz_mapping(path: str | Path) -> dict[str, np.ndarray]:
    artifact = Path(path)
    if artifact.suffix.lower() != ".npz":
        raise StageBContractError("private Stage-B artifact must be NPZ")
    with np.load(artifact, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def validate_current_only_fit_files(
    *,
    artifact_path: str | Path,
    producer_path: str | Path,
    checkpoint_root: str | Path,
    outer_folds: int,
) -> dict[str, object]:
    """CLI-facing aggregate validator for a current-only fit artifact."""

    producer = load_fit_only_producer_view(producer_path)
    manifest = build_checkpoint_manifest(
        checkpoint_root, seeds=EXPECTED_SEEDS, outer_folds=int(outer_folds)
    )
    values = load_private_npz_mapping(artifact_path)
    validate_current_only_fit_artifact(
        values, producer=producer, checkpoint_manifest=manifest
    )
    return {
        "schema_version": CURRENT_ONLY_FIT_ARTIFACT_SCHEMA,
        "status": "valid_private_fit_artifact_not_performance_evidence",
        "dataset": producer.dataset,
        "seed_count": len(EXPECTED_SEEDS),
        "outer_folds": int(outer_folds),
        "fit_query_count": len(producer.fit_query_indices),
        "checkpoint_file_count": len(manifest.records),
        "artifact_sha256": _file_sha256(Path(artifact_path)),
        "checkpoint_manifest_sha256": manifest.manifest_sha256,
        "contains_performance_metrics": False,
    }


def validate_utility_oof_files(
    *,
    artifact_path: str | Path,
    producer_path: str | Path,
) -> dict[str, object]:
    """CLI-facing aggregate validator for a fit-only utility OOF artifact."""

    producer = load_fit_only_producer_view(producer_path)
    values = load_private_npz_mapping(artifact_path)
    validate_utility_oof_score_artifact(values, producer=producer)
    return {
        "schema_version": UTILITY_OOF_SCORE_SCHEMA,
        "status": "valid_private_fit_scores_not_performance_evidence",
        "dataset": producer.dataset,
        "seed_count": len(EXPECTED_SEEDS),
        "fit_task_count": len(producer.fit_tasks),
        "artifact_sha256": _file_sha256(Path(artifact_path)),
        "contains_row_scores_or_targets": False,
        "contains_performance_metrics": False,
    }
