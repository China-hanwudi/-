"""Fail-closed performance evidence contracts for the CARMA causal backbone.

This module sits between the open-role probability producer and any public
performance statement.  It deliberately does not open sidecars or sealed
roles.  The caller must provide already verified, row-aligned arrays.

The central scientific distinction is enforced in code: the producer's
``current_only`` endpoint is an empty-history intervention on a model trained
with history and is therefore *not* an independently trained baseline.  A
separate current-only artifact, checkpoint namespace, source identity, and
probability matrix are required before an evidence bundle can be evaluated.

Only aggregate dictionaries leave this module.  Query identifiers, cluster
codes, labels, probabilities, contexts, and private paths are forbidden from
the public report contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .causal_multimodal_backbone import CausalBackboneConfig
from .emotiontalk_causal_backbone_runner import (
    BackboneRunConfig,
    CrossfitSplit,
    OpenRoleCorpus,
    TrainedFold,
    predict_one_probability_per_query,
    train_one_fold_seed,
)
from .emotiontalk_query_policy_runner import (
    coverage_matched_recency_contexts,
    validate_strict_past_contexts,
)


PRODUCER_CACHE_SCHEMA = "carma_causal_backbone_open_role_private_v2"
CURRENT_ONLY_CACHE_SCHEMA = "carma_independent_current_only_private_v1"
PUBLIC_EVIDENCE_SCHEMA = "carma_causal_backbone_evidence_public_v1"
CROSS_DATASET_INDEX_SCHEMA = "carma_required_dataset_evidence_index_v1"
INDEPENDENT_CURRENT_ONLY_PROTOCOL = (
    "independently_trained_same_architecture_history_stripped_all_masks_empty_v1"
)
EXPECTED_SEEDS = (17, 29, 43, 71, 101)
PRIMARY_TARGET_COVERAGE = 0.25
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_808
DEFAULT_RANDOMIZATION_REPLICATES = 10_000
DEFAULT_RANDOMIZATION_SEED = 20_260_829
MAX_EXACT_RANDOMIZATION_CLUSTERS = 16
SUPPORTED_DATASETS = frozenset({"EmotionTalk", "MELD"})
ENDPOINT_CONTEXT_NAMES = ("current_only", "all_history")
UTILITY_CONTEXT_NAMES = ("s", "s_plus_candidate", "t", "t_minus_candidate")

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SEALED_KEY_TOKENS = frozenset({"calibration", "holdout", "validation", "test", "sealed"})
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "query_indices",
        "row_ids",
        "protocol_row_ids",
        "group_ids",
        "cluster_codes",
        "speaker_ids",
        "labels",
        "predictions",
        "probabilities",
        "embeddings",
        "contexts",
        "histories",
        "private_cache_path",
        "checkpoint_path",
    }
)
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SINGLE_DATASET_STATUS = "open_role_evidence_complete_not_confirmatory_or_publication_ready"
SINGLE_DATASET_CLAIM_BOUNDARY = (
    "single_dataset_not_publishable; this report cannot trigger method_success; "
    "EmotionTalk and MELD must both enter the hash-bound cross-dataset gate"
)
_REMAINING_REQUIREMENTS = (
    "analysis_code_and_environment_hash_freeze",
    "single_calibration_threshold_freeze",
    "one_shot_internal_holdout",
    "external_test_after_authorized_unseal",
    "independent_read_only_audit",
)


class EvidenceContractError(ValueError):
    """Raised when an input could support a misleading performance claim."""


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    digest = str(value).lower()
    if not _HEX_64.fullmatch(digest):
        raise EvidenceContractError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _single_string(value: np.ndarray, field: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise EvidenceContractError(f"{field} must contain exactly one string")
    return str(array.reshape(-1)[0])


def _single_integer(value: np.ndarray, field: str) -> int:
    array = np.asarray(value)
    if array.size != 1 or not np.issubdtype(array.dtype, np.integer):
        raise EvidenceContractError(f"{field} must contain exactly one integer")
    return int(array.reshape(-1)[0])


def _contains_sealed_token(key: str) -> bool:
    tokens = str(key).lower().replace("-", "_").split("_")
    return any(token in _SEALED_KEY_TOKENS for token in tokens)


def _validated_integer_vector(
    value: np.ndarray,
    *,
    field: str,
    unique: bool = False,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise EvidenceContractError(f"{field} must be a one-dimensional integer array")
    result = array.astype(np.int64, copy=False)
    if np.any(result < 0):
        raise EvidenceContractError(f"{field} contains a negative value")
    if unique and len(set(result.tolist())) != len(result):
        raise EvidenceContractError(f"{field} must contain unique values")
    return result


def _validated_probability(
    value: np.ndarray,
    *,
    shape: tuple[int, ...],
    field: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.floating):
        raise EvidenceContractError(f"{field} must have floating shape {shape}")
    result = array.astype(np.float64, copy=False)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise EvidenceContractError(f"{field} contains an invalid probability")
    if not np.allclose(result.sum(axis=-1), 1.0, rtol=1.0e-5, atol=1.0e-6):
        raise EvidenceContractError(f"{field} rows do not sum to one")
    return result


@dataclass(frozen=True)
class EncodedCandidateTasks:
    """Identifier-bearing task rows retained only inside the private process."""

    query_indices: np.ndarray
    candidate_indices: np.ndarray
    addition_contexts: tuple[tuple[int, ...], ...]
    deletion_contexts: tuple[tuple[int, ...], ...]
    task_sha256: str

    def __len__(self) -> int:
        return len(self.query_indices)


def _decode_csr_rows(
    indptr_raw: np.ndarray,
    indices_raw: np.ndarray,
    *,
    rows: int,
    field: str,
    protocol_rows: int,
) -> tuple[tuple[int, ...], ...]:
    indptr = _validated_integer_vector(indptr_raw, field=f"{field}_indptr")
    indices = _validated_integer_vector(indices_raw, field=f"{field}_indices")
    if (
        len(indptr) != rows + 1
        or int(indptr[0]) != 0
        or np.any(np.diff(indptr) < 0)
        or int(indptr[-1]) != len(indices)
    ):
        raise EvidenceContractError(f"{field} CSR encoding is malformed")
    if np.any(indices >= protocol_rows):
        raise EvidenceContractError(f"{field} contains a row outside the protocol")
    decoded: list[tuple[int, ...]] = []
    for row in range(rows):
        values = tuple(int(value) for value in indices[indptr[row] : indptr[row + 1]])
        if len(values) != len(set(values)):
            raise EvidenceContractError(f"{field} contains a duplicate context row")
        decoded.append(values)
    return tuple(decoded)


def _decode_tasks(
    values: Mapping[str, np.ndarray],
    *,
    prefix: str,
    allowed_queries: np.ndarray,
    protocol_rows: int,
) -> EncodedCandidateTasks:
    query = _validated_integer_vector(
        values[f"{prefix}_task_query_indices"],
        field=f"{prefix}_task_query_indices",
    )
    candidate = _validated_integer_vector(
        values[f"{prefix}_task_candidate_indices"],
        field=f"{prefix}_task_candidate_indices",
    )
    if len(query) == 0 or len(candidate) != len(query):
        raise EvidenceContractError(f"{prefix} tasks are empty or misaligned")
    if np.any(query >= protocol_rows) or np.any(candidate >= protocol_rows):
        raise EvidenceContractError(f"{prefix} task row is outside the protocol")
    if not set(query.tolist()).issubset(set(allowed_queries.tolist())):
        raise EvidenceContractError(f"{prefix} task query is outside its open role")
    addition = _decode_csr_rows(
        values[f"{prefix}_task_s_indptr"],
        values[f"{prefix}_task_s_indices"],
        rows=len(query),
        field=f"{prefix}_task_s",
        protocol_rows=protocol_rows,
    )
    deletion = _decode_csr_rows(
        values[f"{prefix}_task_t_indptr"],
        values[f"{prefix}_task_t_indices"],
        rows=len(query),
        field=f"{prefix}_task_t",
        protocol_rows=protocol_rows,
    )
    for index, (q, c, s_context, t_context) in enumerate(
        zip(query, candidate, addition, deletion, strict=True)
    ):
        if int(q) == int(c) or int(c) in s_context or int(c) not in t_context:
            raise EvidenceContractError(
                f"{prefix} task {index} violates addition/deletion candidate semantics"
            )
    encoded_payload = [
        {
            "query": int(q),
            "candidate": int(c),
            "s": list(s_context),
            "t": list(t_context),
        }
        for q, c, s_context, t_context in zip(
            query, candidate, addition, deletion, strict=True
        )
    ]
    observed_hash = _canonical_sha256(encoded_payload)
    declared_hash = _require_sha256(
        _single_string(values[f"{prefix}_task_sha256"], f"{prefix}_task_sha256"),
        f"{prefix}_task_sha256",
    )
    if observed_hash != declared_hash:
        raise EvidenceContractError(f"{prefix} task encoding hash differs")
    return EncodedCandidateTasks(query, candidate, addition, deletion, declared_hash)


_PRODUCER_FIXED_KEYS = frozenset(
    {
        "schema_version",
        "dataset",
        "dataset_label_order",
        "manifest_schema",
        "manifest_status",
        "manifest_sha256",
        "verified_provenance_attestation_sha256",
        "corpus_contract_sha256",
        "histories_sha256",
        "speaker_mapping_sha256",
        "runtime_environment_sha256",
        "source_identity_sha256",
        "seeds",
        "endpoint_context_names",
        "utility_context_names",
        "fit_query_indices",
        "selection_query_indices",
        "fit_cluster_codes",
        "selection_cluster_codes",
        "protocol_row_ids",
        "fit_endpoint_probability_oof",
        "selection_endpoint_probability_fold_ensemble",
        "fit_utility_probability_oof",
        "selection_utility_probability_fold_ensemble",
        "fit_forward_utility",
        "fit_backward_utility",
        "fit_asymmetry",
        "fit_sign_agreement",
        "selection_forward_utility",
        "selection_backward_utility",
        "selection_asymmetry",
        "selection_sign_agreement",
        "fit_task_sha256",
        "selection_task_sha256",
        "checkpoint_manifest_sha256",
        "utility_source",
        "matrix_fit_endpoint_probability_oof_sha256",
        "matrix_selection_endpoint_probability_fold_ensemble_sha256",
        "matrix_fit_utility_probability_oof_sha256",
        "matrix_selection_utility_probability_fold_ensemble_sha256",
        "matrix_fit_forward_utility_sha256",
        "matrix_fit_backward_utility_sha256",
        "matrix_selection_forward_utility_sha256",
        "matrix_selection_backward_utility_sha256",
        "fit_task_query_indices",
        "fit_task_candidate_indices",
        "fit_task_s_indptr",
        "fit_task_s_indices",
        "fit_task_t_indptr",
        "fit_task_t_indices",
        "selection_task_query_indices",
        "selection_task_candidate_indices",
        "selection_task_s_indptr",
        "selection_task_s_indices",
        "selection_task_t_indptr",
        "selection_task_t_indices",
    }
)


@dataclass(frozen=True)
class CausalProducerCache:
    """Validated EmotionTalk/MELD causal-backbone producer cache."""

    dataset: str
    label_order: tuple[str, ...]
    seeds: tuple[int, ...]
    fit_query_indices: np.ndarray
    selection_query_indices: np.ndarray
    fit_cluster_codes: np.ndarray
    selection_cluster_codes: np.ndarray
    fit_endpoint_probability: np.ndarray
    selection_endpoint_probability: np.ndarray
    fit_tasks: EncodedCandidateTasks
    selection_tasks: EncodedCandidateTasks
    source_identity_sha256: str
    checkpoint_manifest_sha256: str
    manifest_sha256: str
    histories_sha256: str
    source_hashes: Mapping[str, str]

    def split_arrays(
        self, role: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, EncodedCandidateTasks]:
        if role == "fit":
            return (
                self.fit_query_indices,
                self.fit_cluster_codes,
                self.fit_endpoint_probability,
                self.fit_tasks,
            )
        if role == "model_selection":
            return (
                self.selection_query_indices,
                self.selection_cluster_codes,
                self.selection_endpoint_probability,
                self.selection_tasks,
            )
        raise EvidenceContractError("role must be fit or model_selection")


def producer_cache_from_mapping(
    values: Mapping[str, np.ndarray],
    *,
    require_five_registered_seeds: bool = True,
) -> CausalProducerCache:
    """Validate a producer mapping without reading raw data or a sealed role."""

    keys = set(values)
    sealed = sorted(key for key in keys if _contains_sealed_token(key))
    if sealed:
        raise EvidenceContractError(f"sealed-role cache fields are forbidden: {sealed}")
    unknown = sorted(
        key
        for key in keys - _PRODUCER_FIXED_KEYS
        if not (key.startswith("source_") and key.endswith("_sha256"))
    )
    missing = sorted(_PRODUCER_FIXED_KEYS - keys)
    if missing or unknown:
        raise EvidenceContractError(
            f"producer cache schema mismatch: missing={missing}, unknown={unknown}"
        )
    schema = _single_string(values["schema_version"], "schema_version")
    if schema != PRODUCER_CACHE_SCHEMA:
        raise EvidenceContractError("producer cache schema version changed")
    dataset = _single_string(values["dataset"], "dataset")
    if dataset not in SUPPORTED_DATASETS:
        raise EvidenceContractError(
            f"producer dataset must be one of {sorted(SUPPORTED_DATASETS)}"
        )
    label_order = tuple(str(value) for value in np.asarray(values["dataset_label_order"]).reshape(-1))
    if len(label_order) < 2 or len(set(label_order)) != len(label_order):
        raise EvidenceContractError("dataset label order is empty or ambiguous")
    seeds_array = _validated_integer_vector(values["seeds"], field="seeds", unique=True)
    seeds = tuple(int(value) for value in seeds_array)
    if require_five_registered_seeds and seeds != EXPECTED_SEEDS:
        raise EvidenceContractError(
            f"performance evidence requires registered seeds {EXPECTED_SEEDS}"
        )
    endpoint_names = tuple(
        str(value) for value in np.asarray(values["endpoint_context_names"]).reshape(-1)
    )
    utility_names = tuple(
        str(value) for value in np.asarray(values["utility_context_names"]).reshape(-1)
    )
    if endpoint_names != ENDPOINT_CONTEXT_NAMES or utility_names != UTILITY_CONTEXT_NAMES:
        raise EvidenceContractError("producer context order changed")

    protocol_row_ids = _validated_integer_vector(
        values["protocol_row_ids"], field="protocol_row_ids", unique=True
    )
    protocol_rows = len(protocol_row_ids)
    fit_query = _validated_integer_vector(
        values["fit_query_indices"], field="fit_query_indices", unique=True
    )
    selection_query = _validated_integer_vector(
        values["selection_query_indices"], field="selection_query_indices", unique=True
    )
    if not len(fit_query) or not len(selection_query):
        raise EvidenceContractError("both open roles require query rows")
    if set(fit_query.tolist()) & set(selection_query.tolist()):
        raise EvidenceContractError("fit and model-selection query rows overlap")
    if np.any(fit_query >= protocol_rows) or np.any(selection_query >= protocol_rows):
        raise EvidenceContractError("query index lies outside protocol rows")
    fit_cluster = _validated_integer_vector(
        values["fit_cluster_codes"], field="fit_cluster_codes"
    )
    selection_cluster = _validated_integer_vector(
        values["selection_cluster_codes"], field="selection_cluster_codes"
    )
    if fit_cluster.shape != fit_query.shape or selection_cluster.shape != selection_query.shape:
        raise EvidenceContractError("query and cluster arrays are misaligned")
    if len(np.unique(fit_cluster)) < 2 or len(np.unique(selection_cluster)) < 2:
        raise EvidenceContractError("each open role requires at least two clusters")

    seed_count = len(seeds)
    classes = len(label_order)
    fit_endpoint = _validated_probability(
        values["fit_endpoint_probability_oof"],
        shape=(seed_count, len(fit_query), len(ENDPOINT_CONTEXT_NAMES), classes),
        field="fit_endpoint_probability_oof",
    )
    selection_endpoint = _validated_probability(
        values["selection_endpoint_probability_fold_ensemble"],
        shape=(seed_count, len(selection_query), len(ENDPOINT_CONTEXT_NAMES), classes),
        field="selection_endpoint_probability_fold_ensemble",
    )
    fit_tasks = _decode_tasks(
        values,
        prefix="fit",
        allowed_queries=fit_query,
        protocol_rows=protocol_rows,
    )
    selection_tasks = _decode_tasks(
        values,
        prefix="selection",
        allowed_queries=selection_query,
        protocol_rows=protocol_rows,
    )
    fit_utility_probability = _validated_probability(
        values["fit_utility_probability_oof"],
        shape=(seed_count, len(fit_tasks), len(UTILITY_CONTEXT_NAMES), classes),
        field="fit_utility_probability_oof",
    )
    selection_utility_probability = _validated_probability(
        values["selection_utility_probability_fold_ensemble"],
        shape=(seed_count, len(selection_tasks), len(UTILITY_CONTEXT_NAMES), classes),
        field="selection_utility_probability_fold_ensemble",
    )
    matrix_values: dict[str, np.ndarray] = {
        "fit_endpoint_probability_oof": np.asarray(values["fit_endpoint_probability_oof"]),
        "selection_endpoint_probability_fold_ensemble": np.asarray(
            values["selection_endpoint_probability_fold_ensemble"]
        ),
        "fit_utility_probability_oof": np.asarray(values["fit_utility_probability_oof"]),
        "selection_utility_probability_fold_ensemble": np.asarray(
            values["selection_utility_probability_fold_ensemble"]
        ),
    }
    for prefix, task_count in (("fit", len(fit_tasks)), ("selection", len(selection_tasks))):
        for name in ("forward_utility", "backward_utility", "asymmetry"):
            key = f"{prefix}_{name}"
            array = np.asarray(values[key])
            if array.shape != (seed_count, task_count) or not np.issubdtype(
                array.dtype, np.floating
            ) or not np.isfinite(array).all():
                raise EvidenceContractError(f"{key} is not seed/task aligned")
            if name != "asymmetry":
                matrix_values[key] = array
        agreement = np.asarray(values[f"{prefix}_sign_agreement"])
        if agreement.shape != (seed_count, task_count) or agreement.dtype != np.bool_:
            raise EvidenceContractError(f"{prefix}_sign_agreement must be boolean")
    for name, array in matrix_values.items():
        declared = _require_sha256(
            _single_string(values[f"matrix_{name}_sha256"], f"matrix_{name}_sha256"),
            f"matrix_{name}_sha256",
        )
        if _array_sha256(array) != declared:
            raise EvidenceContractError(f"matrix hash differs for {name}")

    hash_fields = (
        "manifest_sha256",
        "verified_provenance_attestation_sha256",
        "corpus_contract_sha256",
        "histories_sha256",
        "speaker_mapping_sha256",
        "runtime_environment_sha256",
        "source_identity_sha256",
        "checkpoint_manifest_sha256",
    )
    hashes = {
        field: _require_sha256(_single_string(values[field], field), field)
        for field in hash_fields
    }
    source_hashes = {
        key: _require_sha256(_single_string(values[key], key), key)
        for key in sorted(keys)
        if key.startswith("source_")
        and key.endswith("_sha256")
        and key != "source_identity_sha256"
    }
    if not source_hashes:
        raise EvidenceContractError("producer cache requires manifest-bound source hashes")
    if _single_string(values["utility_source"], "utility_source") != (
        "recomputed_from_causal_backbone_probabilities_and_open_role_labels"
    ):
        raise EvidenceContractError("producer utility provenance changed")
    return CausalProducerCache(
        dataset=dataset,
        label_order=label_order,
        seeds=seeds,
        fit_query_indices=fit_query,
        selection_query_indices=selection_query,
        fit_cluster_codes=fit_cluster,
        selection_cluster_codes=selection_cluster,
        fit_endpoint_probability=fit_endpoint,
        selection_endpoint_probability=selection_endpoint,
        fit_tasks=fit_tasks,
        selection_tasks=selection_tasks,
        source_identity_sha256=hashes["source_identity_sha256"],
        checkpoint_manifest_sha256=hashes["checkpoint_manifest_sha256"],
        manifest_sha256=hashes["manifest_sha256"],
        histories_sha256=hashes["histories_sha256"],
        source_hashes=source_hashes,
    )


def load_producer_cache(
    path: str | Path,
    *,
    require_five_registered_seeds: bool = True,
) -> CausalProducerCache:
    """Load the strict producer whitelist with pickle disabled."""

    cache_path = Path(path)
    if cache_path.suffix.lower() != ".npz":
        raise EvidenceContractError("producer cache must be an .npz archive")
    with np.load(cache_path, allow_pickle=False) as archive:
        keys = set(archive.files)
        sealed = sorted(key for key in keys if _contains_sealed_token(key))
        if sealed:
            raise EvidenceContractError(f"sealed-role cache fields are forbidden: {sealed}")
        values = {key: np.asarray(archive[key]) for key in keys}
    return producer_cache_from_mapping(
        values,
        require_five_registered_seeds=require_five_registered_seeds,
    )


def strip_history_for_independent_current_only(corpus: OpenRoleCorpus) -> OpenRoleCorpus:
    """Return an aligned view whose training and inference histories are empty."""

    return replace(corpus, histories=tuple(() for _ in corpus.histories))


def independent_current_only_source_identity(
    *,
    producer_source_identity_sha256: str,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    rows: int,
    data_contract_sha256: str | None = None,
) -> str:
    """Derive a namespace that cannot resume a history-trained checkpoint."""

    producer_identity = _require_sha256(
        producer_source_identity_sha256, "producer_source_identity_sha256"
    )
    baseline_config = replace(run_config, subset_dropout_probability=0.0)
    payload: dict[str, object] = {
        "training_protocol": INDEPENDENT_CURRENT_ONLY_PROTOCOL,
        "producer_source_identity_sha256": producer_identity,
        "model_config": asdict(model_config),
        "run_config": asdict(baseline_config),
        "row_count": int(rows),
        "history_training_items_consumed": 0,
        "history_inference_items_consumed": 0,
        "empty_history_contract_sha256": _canonical_sha256([[] for _ in range(int(rows))]),
    }
    if data_contract_sha256 is not None:
        payload["data_contract_sha256"] = _require_sha256(
            data_contract_sha256, "data_contract_sha256"
        )
    return _canonical_sha256(payload)


@dataclass(frozen=True)
class IndependentCurrentOnlyFold:
    trained: TrainedFold
    source_identity_sha256: str
    training_protocol: str = INDEPENDENT_CURRENT_ONLY_PROTOCOL


def train_independent_current_only_fold_seed(
    corpus: OpenRoleCorpus,
    split: CrossfitSplit,
    *,
    producer_source_identity_sha256: str,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    seed: int,
    checkpoint_root: Path,
    device: torch.device,
    test_interrupt_after_epoch: int | None = None,
    data_contract_sha256: str | None = None,
    require_complete_checkpoint: bool = False,
) -> IndependentCurrentOnlyFold:
    """Train a new checkpoint after physically stripping every history list."""

    stripped = strip_history_for_independent_current_only(corpus)
    stripped.validate(model_config)
    baseline_config = replace(run_config, subset_dropout_probability=0.0)
    identity = independent_current_only_source_identity(
        producer_source_identity_sha256=producer_source_identity_sha256,
        model_config=model_config,
        run_config=baseline_config,
        rows=len(stripped.keys),
        data_contract_sha256=data_contract_sha256,
    )
    if identity == str(producer_source_identity_sha256).lower():
        raise EvidenceContractError("current-only and history source identities collided")
    trained = train_one_fold_seed(
        stripped,
        split,
        model_config=model_config,
        run_config=baseline_config,
        seed=int(seed),
        source_identity=identity,
        checkpoint_root=checkpoint_root,
        device=device,
        test_interrupt_after_epoch=test_interrupt_after_epoch,
        require_complete_checkpoint=bool(require_complete_checkpoint),
    )
    return IndependentCurrentOnlyFold(trained=trained, source_identity_sha256=identity)


def predict_independent_current_only_probability(
    fold: IndependentCurrentOnlyFold,
    corpus: OpenRoleCorpus,
    query_indices: Sequence[int],
    *,
    device: torch.device,
    batch_size: int,
    max_history_items: int,
) -> np.ndarray:
    """Predict once per query with an independently trained, history-free fold."""

    stripped = strip_history_for_independent_current_only(corpus)
    queries = tuple(int(value) for value in query_indices)
    if len(queries) != len(set(queries)):
        raise EvidenceContractError("current-only inference requires unique query rows")
    return predict_one_probability_per_query(
        fold.trained.model,
        stripped,
        fold.trained.text_features,
        queries,
        tuple(() for _ in queries),
        device=device,
        batch_size=int(batch_size),
        max_history_items=int(max_history_items),
    )


_CURRENT_ONLY_KEYS = frozenset(
    {
        "schema_version",
        "dataset",
        "dataset_label_order",
        "seeds",
        "fit_query_indices",
        "selection_query_indices",
        "fit_cluster_codes",
        "selection_cluster_codes",
        "fit_probability_oof",
        "selection_probability_fold_ensemble",
        "producer_source_identity_sha256",
        "current_only_source_identity_sha256",
        "history_backbone_checkpoint_manifest_sha256",
        "checkpoint_manifest_sha256",
        "training_protocol",
        "checkpoint_namespace",
        "history_training_items_consumed",
        "history_inference_items_consumed",
        "matrix_fit_probability_oof_sha256",
        "matrix_selection_probability_fold_ensemble_sha256",
        "independence_attestation_sha256",
    }
)


def current_only_independence_attestation_payload(
    values: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Return the exact metadata payload bound by an independence attestation."""

    fields = (
        "dataset",
        "producer_source_identity_sha256",
        "current_only_source_identity_sha256",
        "history_backbone_checkpoint_manifest_sha256",
        "checkpoint_manifest_sha256",
        "training_protocol",
        "checkpoint_namespace",
    )
    payload: dict[str, object] = {
        field: _single_string(values[field], field) for field in fields
    }
    payload["history_training_items_consumed"] = _single_integer(
        values["history_training_items_consumed"], "history_training_items_consumed"
    )
    payload["history_inference_items_consumed"] = _single_integer(
        values["history_inference_items_consumed"], "history_inference_items_consumed"
    )
    payload["fit_probability_oof_sha256"] = _array_sha256(
        np.asarray(values["fit_probability_oof"])
    )
    payload["selection_probability_fold_ensemble_sha256"] = _array_sha256(
        np.asarray(values["selection_probability_fold_ensemble"])
    )
    return payload


@dataclass(frozen=True)
class IndependentCurrentOnlyArtifact:
    dataset: str
    label_order: tuple[str, ...]
    seeds: tuple[int, ...]
    fit_query_indices: np.ndarray
    selection_query_indices: np.ndarray
    fit_cluster_codes: np.ndarray
    selection_cluster_codes: np.ndarray
    fit_probability: np.ndarray
    selection_probability: np.ndarray
    producer_source_identity_sha256: str
    source_identity_sha256: str
    checkpoint_manifest_sha256: str
    independence_attestation_sha256: str

    def split_probability(self, role: str) -> np.ndarray:
        if role == "fit":
            return self.fit_probability
        if role == "model_selection":
            return self.selection_probability
        raise EvidenceContractError("role must be fit or model_selection")


def current_only_artifact_from_mapping(
    values: Mapping[str, np.ndarray],
    producer: CausalProducerCache,
) -> IndependentCurrentOnlyArtifact:
    """Validate independent training identity and row-aligned probabilities."""

    keys = set(values)
    sealed = sorted(key for key in keys if _contains_sealed_token(key))
    missing = sorted(_CURRENT_ONLY_KEYS - keys)
    unknown = sorted(keys - _CURRENT_ONLY_KEYS)
    if sealed:
        raise EvidenceContractError(f"sealed-role current-only fields are forbidden: {sealed}")
    if missing or unknown:
        raise EvidenceContractError(
            f"current-only cache schema mismatch: missing={missing}, unknown={unknown}"
        )
    if _single_string(values["schema_version"], "schema_version") != CURRENT_ONLY_CACHE_SCHEMA:
        raise EvidenceContractError("current-only cache schema version changed")
    dataset = _single_string(values["dataset"], "dataset")
    label_order = tuple(str(value) for value in np.asarray(values["dataset_label_order"]).reshape(-1))
    seeds = tuple(
        int(value)
        for value in _validated_integer_vector(values["seeds"], field="seeds", unique=True)
    )
    if dataset != producer.dataset or label_order != producer.label_order or seeds != producer.seeds:
        raise EvidenceContractError("current-only artifact dataset/labels/seeds differ from producer")
    alignments = (
        ("fit_query_indices", producer.fit_query_indices),
        ("selection_query_indices", producer.selection_query_indices),
        ("fit_cluster_codes", producer.fit_cluster_codes),
        ("selection_cluster_codes", producer.selection_cluster_codes),
    )
    aligned: dict[str, np.ndarray] = {}
    for field, expected in alignments:
        observed = _validated_integer_vector(values[field], field=field)
        if not np.array_equal(observed, expected):
            raise EvidenceContractError(f"current-only {field} differs from producer")
        aligned[field] = observed
    classes = len(label_order)
    fit_probability = _validated_probability(
        values["fit_probability_oof"],
        shape=(len(seeds), len(producer.fit_query_indices), classes),
        field="fit_probability_oof",
    )
    selection_probability = _validated_probability(
        values["selection_probability_fold_ensemble"],
        shape=(len(seeds), len(producer.selection_query_indices), classes),
        field="selection_probability_fold_ensemble",
    )
    for field, array in (
        ("matrix_fit_probability_oof_sha256", np.asarray(values["fit_probability_oof"])),
        (
            "matrix_selection_probability_fold_ensemble_sha256",
            np.asarray(values["selection_probability_fold_ensemble"]),
        ),
    ):
        declared = _require_sha256(_single_string(values[field], field), field)
        if declared != _array_sha256(array):
            raise EvidenceContractError(f"{field} differs from its matrix")
    producer_identity = _require_sha256(
        _single_string(values["producer_source_identity_sha256"], "producer_source_identity_sha256"),
        "producer_source_identity_sha256",
    )
    current_identity = _require_sha256(
        _single_string(values["current_only_source_identity_sha256"], "current_only_source_identity_sha256"),
        "current_only_source_identity_sha256",
    )
    history_checkpoint = _require_sha256(
        _single_string(
            values["history_backbone_checkpoint_manifest_sha256"],
            "history_backbone_checkpoint_manifest_sha256",
        ),
        "history_backbone_checkpoint_manifest_sha256",
    )
    current_checkpoint = _require_sha256(
        _single_string(values["checkpoint_manifest_sha256"], "checkpoint_manifest_sha256"),
        "checkpoint_manifest_sha256",
    )
    if producer_identity != producer.source_identity_sha256:
        raise EvidenceContractError("current-only artifact is bound to a different producer")
    if history_checkpoint != producer.checkpoint_manifest_sha256:
        raise EvidenceContractError("history checkpoint lineage differs from producer")
    if current_identity == producer_identity or current_checkpoint == history_checkpoint:
        raise EvidenceContractError("current-only baseline reused history-trained identity/checkpoints")
    if _single_string(values["training_protocol"], "training_protocol") != INDEPENDENT_CURRENT_ONLY_PROTOCOL:
        raise EvidenceContractError("current-only training protocol is not history-stripped")
    if _single_string(values["checkpoint_namespace"], "checkpoint_namespace") != (
        "independent_current_only"
    ):
        raise EvidenceContractError("current-only checkpoint namespace is not independent")
    if (
        _single_integer(values["history_training_items_consumed"], "history_training_items_consumed")
        != 0
        or _single_integer(values["history_inference_items_consumed"], "history_inference_items_consumed")
        != 0
    ):
        raise EvidenceContractError("current-only baseline consumed history")
    expected_attestation = _canonical_sha256(current_only_independence_attestation_payload(values))
    observed_attestation = _require_sha256(
        _single_string(values["independence_attestation_sha256"], "independence_attestation_sha256"),
        "independence_attestation_sha256",
    )
    if expected_attestation != observed_attestation:
        raise EvidenceContractError("current-only independence attestation differs")
    return IndependentCurrentOnlyArtifact(
        dataset=dataset,
        label_order=label_order,
        seeds=seeds,
        fit_query_indices=aligned["fit_query_indices"],
        selection_query_indices=aligned["selection_query_indices"],
        fit_cluster_codes=aligned["fit_cluster_codes"],
        selection_cluster_codes=aligned["selection_cluster_codes"],
        fit_probability=fit_probability,
        selection_probability=selection_probability,
        producer_source_identity_sha256=producer_identity,
        source_identity_sha256=current_identity,
        checkpoint_manifest_sha256=current_checkpoint,
        independence_attestation_sha256=observed_attestation,
    )


def build_current_only_artifact_mapping(
    producer: CausalProducerCache,
    *,
    fit_probability_oof: np.ndarray,
    selection_probability_fold_ensemble: np.ndarray,
    current_only_source_identity_sha256: str,
    checkpoint_manifest_sha256: str,
) -> dict[str, np.ndarray]:
    """Build and self-validate the exact independent-current-only cache schema.

    This builder accepts probabilities only after independent training has
    completed.  It copies the producer's row/cluster alignment, fixes both
    history-consumption counters to zero, binds the history and current-only
    checkpoint manifests, and computes the independence attestation.  The
    returned mapping is suitable for an ``allow_pickle=False`` private NPZ.
    """

    current_identity = _require_sha256(
        current_only_source_identity_sha256,
        "current_only_source_identity_sha256",
    )
    current_checkpoint = _require_sha256(
        checkpoint_manifest_sha256, "checkpoint_manifest_sha256"
    )
    if current_identity == producer.source_identity_sha256:
        raise EvidenceContractError(
            "current-only builder cannot reuse the history source identity"
        )
    if current_checkpoint == producer.checkpoint_manifest_sha256:
        raise EvidenceContractError(
            "current-only builder cannot reuse the history checkpoint manifest"
        )
    fit_probability = _validated_probability(
        fit_probability_oof,
        shape=(
            len(producer.seeds),
            len(producer.fit_query_indices),
            len(producer.label_order),
        ),
        field="fit_probability_oof",
    )
    selection_probability = _validated_probability(
        selection_probability_fold_ensemble,
        shape=(
            len(producer.seeds),
            len(producer.selection_query_indices),
            len(producer.label_order),
        ),
        field="selection_probability_fold_ensemble",
    )
    values: dict[str, np.ndarray] = {
        "schema_version": np.asarray(CURRENT_ONLY_CACHE_SCHEMA),
        "dataset": np.asarray(producer.dataset),
        "dataset_label_order": np.asarray(producer.label_order),
        "seeds": np.asarray(producer.seeds, dtype=np.int64),
        "fit_query_indices": np.asarray(producer.fit_query_indices, dtype=np.int64),
        "selection_query_indices": np.asarray(
            producer.selection_query_indices, dtype=np.int64
        ),
        "fit_cluster_codes": np.asarray(producer.fit_cluster_codes, dtype=np.int64),
        "selection_cluster_codes": np.asarray(
            producer.selection_cluster_codes, dtype=np.int64
        ),
        "fit_probability_oof": np.asarray(fit_probability, dtype=np.float32),
        "selection_probability_fold_ensemble": np.asarray(
            selection_probability, dtype=np.float32
        ),
        "producer_source_identity_sha256": np.asarray(
            producer.source_identity_sha256
        ),
        "current_only_source_identity_sha256": np.asarray(current_identity),
        "history_backbone_checkpoint_manifest_sha256": np.asarray(
            producer.checkpoint_manifest_sha256
        ),
        "checkpoint_manifest_sha256": np.asarray(current_checkpoint),
        "training_protocol": np.asarray(INDEPENDENT_CURRENT_ONLY_PROTOCOL),
        "checkpoint_namespace": np.asarray("independent_current_only"),
        "history_training_items_consumed": np.asarray(0, dtype=np.int64),
        "history_inference_items_consumed": np.asarray(0, dtype=np.int64),
    }
    values["matrix_fit_probability_oof_sha256"] = np.asarray(
        _array_sha256(values["fit_probability_oof"])
    )
    values["matrix_selection_probability_fold_ensemble_sha256"] = np.asarray(
        _array_sha256(values["selection_probability_fold_ensemble"])
    )
    values["independence_attestation_sha256"] = np.asarray(
        _canonical_sha256(current_only_independence_attestation_payload(values))
    )
    current_only_artifact_from_mapping(values, producer)
    return values


def load_current_only_artifact(
    path: str | Path,
    producer: CausalProducerCache,
) -> IndependentCurrentOnlyArtifact:
    cache_path = Path(path)
    if cache_path.suffix.lower() != ".npz":
        raise EvidenceContractError("current-only cache must be an .npz archive")
    with np.load(cache_path, allow_pickle=False) as archive:
        keys = set(archive.files)
        sealed = sorted(key for key in keys if _contains_sealed_token(key))
        if sealed:
            raise EvidenceContractError(f"sealed-role current-only fields are forbidden: {sealed}")
        values = {key: np.asarray(archive[key]) for key in keys}
    return current_only_artifact_from_mapping(values, producer)


@dataclass(frozen=True)
class FrozenCoverageRule:
    """A label-blind fit-OOF candidate operating point."""

    target_coverage: float
    fit_pair_count: int
    fit_selected_count: int
    fit_realized_coverage: float
    threshold: float
    boundary_tie_fraction: float
    tie_salt_sha256: str
    fit_task_sha256: str
    score_source_identity_sha256: str
    rule_sha256: str
    frozen_on_role: str = "fit_oof"
    selection_uses_labels_clusters_or_utilities: bool = False


def _aggregate_pair_scores(
    tasks: EncodedCandidateTasks,
    decision_scores: np.ndarray,
) -> dict[tuple[int, int], float]:
    score = np.asarray(decision_scores, dtype=np.float64)
    if score.shape != (len(tasks),) or not np.isfinite(score).all():
        raise EvidenceContractError("candidate decision scores must be finite and task-aligned")
    buckets: dict[tuple[int, int], list[float]] = {}
    for query, candidate, value in zip(
        tasks.query_indices, tasks.candidate_indices, score, strict=True
    ):
        buckets.setdefault((int(query), int(candidate)), []).append(float(value))
    if not buckets:
        raise EvidenceContractError("candidate operating point requires at least one pair")
    return {
        pair: float(np.mean(values)) for pair, values in sorted(buckets.items())
    }


def _stable_pair_hash(pair: tuple[int, int], salt_sha256: str) -> str:
    return hashlib.sha256(
        f"{salt_sha256}:{int(pair[0])}:{int(pair[1])}".encode("ascii")
    ).hexdigest()


def _select_pairs_with_rule(
    pair_scores: Mapping[tuple[int, int], float],
    rule: FrozenCoverageRule,
) -> set[tuple[int, int]]:
    above = {pair for pair, score in pair_scores.items() if score > rule.threshold}
    tied = [pair for pair, score in pair_scores.items() if score == rule.threshold]
    if not tied or rule.boundary_tie_fraction <= 0.0:
        return above
    tie_count = int(math.floor(rule.boundary_tie_fraction * len(tied) + 0.5))
    tie_count = min(max(tie_count, 0), len(tied))
    ranked = sorted(tied, key=lambda pair: (_stable_pair_hash(pair, rule.tie_salt_sha256), pair))
    return above | set(ranked[:tie_count])


def freeze_fit_oof_operating_point(
    tasks: EncodedCandidateTasks,
    decision_scores: np.ndarray,
    *,
    score_source_identity_sha256: str,
    target_coverage: float = PRIMARY_TARGET_COVERAGE,
) -> FrozenCoverageRule:
    """Freeze exact fit-OOF pair coverage without accepting labels or clusters.

    Exact boundary ties are resolved by a stable SHA-256 ordering.  The public
    rule contains only the salt hash and aggregate boundary fraction, never the
    selected query/candidate pairs.
    """

    source_identity = _require_sha256(
        score_source_identity_sha256, "score_source_identity_sha256"
    )
    coverage = float(target_coverage)
    if not 0.0 < coverage < 1.0 or not math.isclose(
        coverage, PRIMARY_TARGET_COVERAGE, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise EvidenceContractError("primary operating point must be the predeclared 25%")
    pair_scores = _aggregate_pair_scores(tasks, decision_scores)
    pair_count = len(pair_scores)
    target_count = int(math.floor(coverage * pair_count + 0.5))
    target_count = min(max(target_count, 1), pair_count - 1)
    ordered = sorted(pair_scores.values(), reverse=True)
    selected_boundary = float(ordered[target_count - 1])
    excluded_boundary = float(ordered[target_count])
    salt = _canonical_sha256(
        {
            "protocol": "fit_oof_pair_coverage_sha256_tie_rule_v1",
            "fit_task_sha256": tasks.task_sha256,
            "score_source_identity_sha256": source_identity,
            "target_coverage": coverage,
        }
    )
    if selected_boundary > excluded_boundary:
        threshold = selected_boundary + (excluded_boundary - selected_boundary) / 2.0
        tie_fraction = 0.0
    else:
        threshold = selected_boundary
        above = sum(value > threshold for value in pair_scores.values())
        tied = sum(value == threshold for value in pair_scores.values())
        tie_fraction = float((target_count - above) / tied)
    payload = {
        "target_coverage": coverage,
        "fit_pair_count": pair_count,
        "fit_selected_count": target_count,
        "fit_realized_coverage": float(target_count / pair_count),
        "threshold": float(threshold),
        "boundary_tie_fraction": tie_fraction,
        "tie_salt_sha256": salt,
        "fit_task_sha256": tasks.task_sha256,
        "score_source_identity_sha256": source_identity,
        "frozen_on_role": "fit_oof",
        "selection_uses_labels_clusters_or_utilities": False,
    }
    provisional = FrozenCoverageRule(**payload, rule_sha256=_canonical_sha256(payload))
    selected = _select_pairs_with_rule(pair_scores, provisional)
    if len(selected) != target_count:
        raise AssertionError("fit tie rule failed to realize the frozen target count")
    return provisional


@dataclass(frozen=True)
class PreparedPolicyContexts:
    selected_contexts: tuple[tuple[int, ...], ...]
    matched_recency_contexts: tuple[tuple[int, ...], ...]
    selected_pair_count: int
    available_pair_count: int
    realized_pair_coverage: float
    history_using_query_count: int
    query_count: int
    policy_sha256: str
    rule: FrozenCoverageRule
    task_sha256: str


def prepare_policy_contexts(
    *,
    query_indices: Sequence[int],
    histories: Sequence[Sequence[int]],
    tasks: EncodedCandidateTasks,
    decision_scores: np.ndarray,
    score_source_identity_sha256: str,
    rule: FrozenCoverageRule,
) -> PreparedPolicyContexts:
    """Apply a frozen rule and build an exactly cardinality-matched recency arm."""

    source_identity = _require_sha256(
        score_source_identity_sha256, "score_source_identity_sha256"
    )
    if source_identity != rule.score_source_identity_sha256:
        raise EvidenceContractError("deployment scores came from a different utility model")
    queries = tuple(int(value) for value in query_indices)
    if len(queries) != len(set(queries)):
        raise EvidenceContractError("policy evaluation requires one unique row per query")
    if set(tasks.query_indices.tolist()) - set(queries):
        raise EvidenceContractError("candidate task belongs to a different evaluation split")
    pair_scores = _aggregate_pair_scores(tasks, decision_scores)
    selected_pairs = _select_pairs_with_rule(pair_scores, rule)
    score_map: dict[int, dict[int, float]] = {}
    for (query, candidate), value in pair_scores.items():
        score_map.setdefault(query, {})[candidate] = value
    selected_contexts: list[tuple[int, ...]] = []
    for query in queries:
        if query < 0 or query >= len(histories):
            raise EvidenceContractError("query index is outside the history source")
        past = tuple(int(value) for value in histories[query])
        if len(past) != len(set(past)):
            raise EvidenceContractError("strict-past history contains duplicate rows")
        outside = set(score_map.get(query, {})) - set(past)
        if outside:
            raise EvidenceContractError("candidate score refers to a non-past history row")
        selected_contexts.append(
            tuple(candidate for candidate in past if (query, candidate) in selected_pairs)
        )
    selected_tuple = tuple(selected_contexts)
    recency = coverage_matched_recency_contexts(queries, histories, selected_tuple)
    validate_strict_past_contexts(queries, selected_tuple, histories)
    validate_strict_past_contexts(queries, recency, histories)
    if any(len(left) != len(right) for left, right in zip(selected_tuple, recency, strict=True)):
        raise AssertionError("coverage-matched recency cardinality differs")
    available = len(pair_scores)
    selected_count = len(selected_pairs)
    payload = {
        "rule_sha256": rule.rule_sha256,
        "task_sha256": tasks.task_sha256,
        "query_indices_sha256": _array_sha256(np.asarray(queries, dtype=np.int64)),
        "selected_contexts_sha256": _canonical_sha256([list(row) for row in selected_tuple]),
        "matched_recency_contexts_sha256": _canonical_sha256([list(row) for row in recency]),
        "score_source_identity_sha256": source_identity,
    }
    return PreparedPolicyContexts(
        selected_contexts=selected_tuple,
        matched_recency_contexts=recency,
        selected_pair_count=selected_count,
        available_pair_count=available,
        realized_pair_coverage=float(selected_count / available),
        history_using_query_count=int(sum(bool(row) for row in selected_tuple)),
        query_count=len(queries),
        policy_sha256=_canonical_sha256(payload),
        rule=rule,
        task_sha256=tasks.task_sha256,
    )


@dataclass(frozen=True)
class MethodPrediction:
    probability: np.ndarray
    source_identity_sha256: str
    context_role: str


@dataclass(frozen=True)
class EvidenceBundle:
    producer: CausalProducerCache
    independent_current_only: IndependentCurrentOnlyArtifact
    role: str
    labels: np.ndarray
    histories: tuple[tuple[int, ...], ...]
    policy: PreparedPolicyContexts
    methods: Mapping[str, MethodPrediction]
    current_only_method: str = "independent_current_only"
    recency_method: str = "coverage_matched_recency"

    @property
    def query_indices(self) -> np.ndarray:
        return self.producer.split_arrays(self.role)[0]

    @property
    def cluster_codes(self) -> np.ndarray:
        return self.producer.split_arrays(self.role)[1]


def validate_evidence_bundle(bundle: EvidenceBundle) -> EvidenceBundle:
    """Fail closed unless every method emits one prediction per query and seed."""

    query, clusters, _, tasks = bundle.producer.split_arrays(bundle.role)
    if bundle.policy.task_sha256 != tasks.task_sha256:
        raise EvidenceContractError("prepared policy belongs to a different producer split")
    if bundle.independent_current_only.producer_source_identity_sha256 != (
        bundle.producer.source_identity_sha256
    ):
        raise EvidenceContractError("current-only artifact is not bound to this producer")
    labels = np.asarray(bundle.labels)
    if labels.shape != (len(query),) or not np.issubdtype(labels.dtype, np.integer):
        raise EvidenceContractError("evaluation labels must be one integer per query")
    if np.any((labels < 0) | (labels >= len(bundle.producer.label_order))):
        raise EvidenceContractError("evaluation label lies outside dataset label order")
    histories = tuple(tuple(int(value) for value in row) for row in bundle.histories)
    if len(histories) <= int(np.max(query)):
        raise EvidenceContractError("history source does not cover every query")
    try:
        validate_strict_past_contexts(query, bundle.policy.selected_contexts, histories)
        validate_strict_past_contexts(
            query, bundle.policy.matched_recency_contexts, histories
        )
        expected_recency = coverage_matched_recency_contexts(
            query, histories, bundle.policy.selected_contexts
        )
    except (ValueError, IndexError) as error:
        raise EvidenceContractError(f"policy context contract failed: {error}") from error
    if expected_recency != bundle.policy.matched_recency_contexts:
        raise EvidenceContractError("recency arm is not the exact matched-cardinality baseline")
    if not {bundle.current_only_method, bundle.recency_method}.issubset(
        set(bundle.methods)
    ):
        raise EvidenceContractError("evidence bundle lacks current-only or recency method")
    seed_count = len(bundle.producer.seeds)
    classes = len(bundle.producer.label_order)
    current_expected = bundle.independent_current_only.split_probability(bundle.role)
    for name, method in bundle.methods.items():
        if not name or any(character in name for character in "\\/\x00"):
            raise EvidenceContractError("method name is empty or unsafe")
        _require_sha256(method.source_identity_sha256, f"{name}.source_identity_sha256")
        probability = _validated_probability(
            method.probability,
            shape=(seed_count, len(query), classes),
            field=f"{name}.probability",
        )
        if method.context_role not in {
            "independent_current_only",
            "coverage_matched_recency",
            "selected_history",
            "selected_history_ablation",
            "all_history_diagnostic",
        }:
            raise EvidenceContractError(f"{name} has an unknown context role")
        if name == bundle.current_only_method:
            if method.context_role != "independent_current_only":
                raise EvidenceContractError("current-only method has a history context role")
            if method.source_identity_sha256 != bundle.independent_current_only.source_identity_sha256:
                raise EvidenceContractError("current-only method reused the history model identity")
            if not np.array_equal(probability, current_expected):
                raise EvidenceContractError("current-only predictions differ from independent artifact")
        if name == bundle.recency_method:
            if method.context_role != "coverage_matched_recency":
                raise EvidenceContractError("recency method has the wrong context role")
            if method.source_identity_sha256 != bundle.producer.source_identity_sha256:
                raise EvidenceContractError("recency predictions came from a different backbone")
    if bundle.independent_current_only.source_identity_sha256 == bundle.producer.source_identity_sha256:
        raise EvidenceContractError("current-only source identity is not independent")
    if len(np.unique(clusters)) < 2:
        raise EvidenceContractError("evidence evaluation requires at least two clusters")
    return bundle


@dataclass(frozen=True)
class HolmHypothesis:
    hypothesis_id: str
    candidate: str
    reference: str
    metric: str
    alternative: str


@dataclass(frozen=True)
class PredeclaredHolmFamily:
    family_id: str
    alpha: float
    hypotheses: tuple[HolmHypothesis, ...]
    analysis_config_sha256: str
    family_sha256: str


@dataclass(frozen=True)
class AccuracyNoHarmContrast:
    contrast_id: str
    candidate: str
    reference: str


@dataclass(frozen=True)
class PredeclaredAccuracyNoHarmGate:
    """Mandatory accuracy non-inferiority gate, separate from the Holm family."""

    gate_id: str
    contrasts: tuple[AccuracyNoHarmContrast, ...]
    minimum_point_difference: float
    minimum_ci95_lower: float
    analysis_config_sha256: str
    gate_sha256: str


@dataclass(frozen=True)
class CrossDatasetAggregationPlan:
    """Hash-bound requirement that both registered datasets enter one gate."""

    plan_id: str
    required_datasets: tuple[str, ...]
    analysis_config_sha256: str
    plan_sha256: str


def predeclare_holm_family(
    *,
    family_id: str,
    alpha: float,
    hypotheses: Sequence[HolmHypothesis],
    analysis_config_sha256: str,
) -> PredeclaredHolmFamily:
    """Build a hash-bound family before evaluation labels enter the API."""

    config_hash = _require_sha256(analysis_config_sha256, "analysis_config_sha256")
    rows = tuple(hypotheses)
    if not family_id or len(rows) < 2 or len({row.hypothesis_id for row in rows}) != len(rows):
        raise EvidenceContractError("Holm family requires at least two unique hypotheses")
    if not 0.0 < float(alpha) <= 0.05:
        raise EvidenceContractError("Holm family alpha must lie in (0, 0.05]")
    for row in rows:
        if not row.hypothesis_id or not row.candidate or not row.reference:
            raise EvidenceContractError("Holm hypothesis identifiers/methods must be non-empty")
        if row.metric not in {"macro_f1", "mean_regret"}:
            raise EvidenceContractError("Holm metric must be macro_f1 or mean_regret")
        if row.alternative not in {"greater", "less"}:
            raise EvidenceContractError("Holm alternative must be greater or less")
    payload = {
        "family_id": family_id,
        "alpha": float(alpha),
        "hypotheses": [asdict(row) for row in rows],
        "analysis_config_sha256": config_hash,
        "method": "holm_bonferroni",
        "frozen_before_evaluation": True,
    }
    return PredeclaredHolmFamily(
        family_id=family_id,
        alpha=float(alpha),
        hypotheses=rows,
        analysis_config_sha256=config_hash,
        family_sha256=_canonical_sha256(payload),
    )


def predeclare_accuracy_no_harm_gate(
    *,
    gate_id: str,
    contrasts: Sequence[AccuracyNoHarmContrast],
    analysis_config_sha256: str,
    minimum_point_difference: float = 0.0,
    minimum_ci95_lower: float = -0.005,
) -> PredeclaredAccuracyNoHarmGate:
    """Freeze accuracy criteria before evaluation.

    The default requires a non-negative point change and rules out a loss of
    0.5 percentage points or more at the lower 95% confidence bound.  Both
    thresholds are hash-bound and therefore cannot be relaxed after results.
    """

    config_hash = _require_sha256(analysis_config_sha256, "analysis_config_sha256")
    rows = tuple(contrasts)
    if not gate_id or not rows or len({row.contrast_id for row in rows}) != len(rows):
        raise EvidenceContractError("accuracy gate requires unique named contrasts")
    if any(
        not row.contrast_id or not row.candidate or not row.reference
        for row in rows
    ):
        raise EvidenceContractError("accuracy gate contrast fields must be non-empty")
    point_minimum = float(minimum_point_difference)
    ci_minimum = float(minimum_ci95_lower)
    if not math.isclose(point_minimum, 0.0, rel_tol=0.0, abs_tol=1.0e-12) or not math.isclose(
        ci_minimum, -0.005, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise EvidenceContractError(
            "accuracy no-harm margins are frozen at point>=0 and CI lower>=-0.005"
        )
    payload = {
        "gate_id": gate_id,
        "metric": "accuracy",
        "contrasts": [asdict(row) for row in rows],
        "minimum_point_difference": point_minimum,
        "minimum_ci95_lower": ci_minimum,
        "analysis_config_sha256": config_hash,
        "frozen_before_evaluation": True,
        "mandatory_for_method_success": True,
    }
    return PredeclaredAccuracyNoHarmGate(
        gate_id=gate_id,
        contrasts=rows,
        minimum_point_difference=point_minimum,
        minimum_ci95_lower=ci_minimum,
        analysis_config_sha256=config_hash,
        gate_sha256=_canonical_sha256(payload),
    )


def predeclare_cross_dataset_aggregation(
    *,
    plan_id: str,
    analysis_config_sha256: str,
    required_datasets: Sequence[str] = ("EmotionTalk", "MELD"),
) -> CrossDatasetAggregationPlan:
    datasets = tuple(str(value) for value in required_datasets)
    if not plan_id or datasets != ("EmotionTalk", "MELD"):
        raise EvidenceContractError(
            "cross-dataset gate must require EmotionTalk and MELD in frozen order"
        )
    config_hash = _require_sha256(analysis_config_sha256, "analysis_config_sha256")
    payload = {
        "plan_id": plan_id,
        "required_datasets": list(datasets),
        "analysis_config_sha256": config_hash,
        "single_dataset_can_trigger_method_success": False,
        "aggregation_role": "required_dataset_index_before_confirmatory_joint_analysis",
    }
    return CrossDatasetAggregationPlan(
        plan_id=plan_id,
        required_datasets=datasets,
        analysis_config_sha256=config_hash,
        plan_sha256=_canonical_sha256(payload),
    )


def holm_bonferroni(
    raw_p_values: Mapping[str, float],
    *,
    declared_order: Sequence[str],
    alpha: float,
) -> dict[str, dict[str, float | int | bool]]:
    """Return monotone Holm-adjusted p-values in the predeclared order."""

    order = tuple(str(value) for value in declared_order)
    if len(order) < 2 or len(set(order)) != len(order) or set(order) != set(raw_p_values):
        raise EvidenceContractError("raw p-values must match the complete declared Holm family")
    values = {key: float(raw_p_values[key]) for key in order}
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values.values()):
        raise EvidenceContractError("raw p-values must be finite values in [0, 1]")
    indexed = {key: index for index, key in enumerate(order)}
    ranked = sorted(order, key=lambda key: (values[key], indexed[key]))
    count = len(ranked)
    adjusted_ranked: dict[str, float] = {}
    running = 0.0
    still_rejecting = True
    rejection: dict[str, bool] = {}
    rank_by_id: dict[str, int] = {}
    for zero_rank, key in enumerate(ranked):
        rank = zero_rank + 1
        running = max(running, min(1.0, (count - zero_rank) * values[key]))
        adjusted_ranked[key] = running
        threshold = float(alpha) / (count - zero_rank)
        reject = bool(still_rejecting and values[key] <= threshold)
        rejection[key] = reject
        if not reject:
            still_rejecting = False
        rank_by_id[key] = rank
    return {
        key: {
            "raw_p_value": values[key],
            "holm_adjusted_p_value": adjusted_ranked[key],
            "holm_rank": rank_by_id[key],
            "rejected_at_familywise_alpha": rejection[key],
        }
        for key in order
    }


def _true_class_loss(labels: np.ndarray, probability: np.ndarray) -> np.ndarray:
    values = probability[np.arange(len(labels)), labels]
    return -np.log(np.clip(values, np.finfo(np.float64).tiny, 1.0))


def _multiclass_ece(labels: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    confidence = probability.max(axis=1)
    correct = probability.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    result = 0.0
    for index in range(int(bins)):
        if index + 1 == int(bins):
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if np.any(mask):
            result += float(np.mean(mask)) * abs(
                float(np.mean(correct[mask])) - float(np.mean(confidence[mask]))
            )
    return float(result)


def _method_seed_metrics(
    labels: np.ndarray,
    probability: np.ndarray,
    current_probability: np.ndarray,
    eligible: np.ndarray,
) -> list[dict[str, float]]:
    classes = probability.shape[-1]
    one_hot = np.eye(classes, dtype=np.float64)[labels]
    records: list[dict[str, float]] = []
    for seed_index in range(probability.shape[0]):
        values = probability[seed_index]
        current = current_probability[seed_index]
        predicted = values.argmax(axis=1)
        excess = _true_class_loss(labels, values) - _true_class_loss(labels, current)
        eligible_excess = excess[eligible]
        quantile = float(np.quantile(eligible_excess, 0.9))
        tail = eligible_excess[eligible_excess >= quantile]
        records.append(
            {
                "macro_f1": float(
                    f1_score(
                        labels,
                        predicted,
                        labels=np.arange(classes),
                        average="macro",
                        zero_division=0,
                    )
                ),
                "weighted_f1": float(
                    f1_score(
                        labels,
                        predicted,
                        labels=np.arange(classes),
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "accuracy": float(accuracy_score(labels, predicted)),
                "nll": float(np.mean(_true_class_loss(labels, values))),
                "brier": float(np.mean(np.sum(np.square(values - one_hot), axis=1))),
                "ece": _multiclass_ece(labels, values),
                "mean_regret": float(np.mean(eligible_excess)),
                "history_harm_rate": float(np.mean(eligible_excess > 0.0)),
                "p90_regret": quantile,
                "cvar90_regret": float(np.mean(tail)),
            }
        )
    return records


def _aggregate_method_metrics(records: Sequence[Mapping[str, float]]) -> dict[str, object]:
    rows = tuple(records)
    if len(rows) != 5:
        raise EvidenceContractError("method aggregate requires exactly five training seeds")
    return {
        key: {
            "five_seed_mean": float(np.mean([float(row[key]) for row in rows])),
            "five_seed_sample_sd": float(np.std([float(row[key]) for row in rows], ddof=1)),
        }
        for key in rows[0]
    }


def _confusion_by_seed_cluster(
    labels: np.ndarray,
    probability: np.ndarray,
    clusters: np.ndarray,
    cluster_values: np.ndarray,
) -> np.ndarray:
    seed_count, _, classes = probability.shape
    result = np.zeros(
        (seed_count, len(cluster_values), classes, classes), dtype=np.int64
    )
    cluster_position = {int(value): index for index, value in enumerate(cluster_values)}
    for seed_index in range(seed_count):
        predicted = probability[seed_index].argmax(axis=1)
        for label, prediction, cluster in zip(labels, predicted, clusters, strict=True):
            result[
                seed_index,
                cluster_position[int(cluster)],
                int(label),
                int(prediction),
            ] += 1
    return result


def _macro_f1_from_confusion(confusion: np.ndarray) -> float:
    diagonal = np.diag(confusion).astype(np.float64)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    per_class = np.divide(
        2.0 * diagonal,
        denominator,
        out=np.zeros_like(diagonal),
        where=denominator > 0,
    )
    return float(np.mean(per_class))


def _classification_score_from_confusion(
    confusion: np.ndarray,
    metric: str,
) -> float:
    if metric == "macro_f1":
        return _macro_f1_from_confusion(confusion)
    if metric == "accuracy":
        total = int(confusion.sum())
        if total <= 0:
            raise EvidenceContractError("accuracy confusion matrix is empty")
        return float(np.trace(confusion) / total)
    raise EvidenceContractError("classification metric is unsupported")


def _draw_crossed_seed_shared_clusters(
    rng: np.random.Generator,
    *,
    seed_count: int,
    cluster_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one seed vector and exactly one cluster vector per replicate.

    The cluster vector intentionally has no seed axis.  Both methods and every
    resampled seed slot consume this same vector, which makes accidental
    seed-specific cluster resampling structurally impossible.
    """

    if int(seed_count) < 2 or int(cluster_count) < 2:
        raise EvidenceContractError("crossed bootstrap needs two seeds and clusters")
    return (
        rng.integers(0, int(seed_count), size=int(seed_count)),
        rng.integers(0, int(cluster_count), size=int(cluster_count)),
    )


def _bootstrap_classification_difference(
    *,
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    clusters: np.ndarray,
    replicates: int,
    seed: int,
    metric: str,
) -> tuple[float, np.ndarray, int]:
    cluster_values = np.unique(clusters)
    candidate_confusion = _confusion_by_seed_cluster(
        labels, candidate, clusters, cluster_values
    )
    reference_confusion = _confusion_by_seed_cluster(
        labels, reference, clusters, cluster_values
    )
    def score(confusion: np.ndarray) -> float:
        return _classification_score_from_confusion(confusion, metric)
    point = float(
        np.mean(
            [
                score(candidate_confusion[index].sum(axis=0))
                - score(reference_confusion[index].sum(axis=0))
                for index in range(candidate.shape[0])
            ]
        )
    )
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        seed_draw, cluster_draw = _draw_crossed_seed_shared_clusters(
            rng,
            seed_count=candidate.shape[0],
            cluster_count=len(cluster_values),
        )
        differences = []
        for sampled_seed in seed_draw:
            candidate_matrix = candidate_confusion[int(sampled_seed), cluster_draw].sum(axis=0)
            reference_matrix = reference_confusion[int(sampled_seed), cluster_draw].sum(axis=0)
            differences.append(
                score(candidate_matrix) - score(reference_matrix)
            )
        bootstrap[replicate] = float(np.mean(differences))
    return point, bootstrap, len(cluster_values)


def _cluster_regret_sums(
    labels: np.ndarray,
    probability: np.ndarray,
    current: np.ndarray,
    clusters: np.ndarray,
    eligible: np.ndarray,
    cluster_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((probability.shape[0], len(cluster_values)), dtype=np.float64)
    counts = np.zeros(len(cluster_values), dtype=np.int64)
    positions = {int(value): index for index, value in enumerate(cluster_values)}
    for cluster in cluster_values:
        mask = (clusters == cluster) & eligible
        position = positions[int(cluster)]
        counts[position] = int(np.sum(mask))
        for seed_index in range(probability.shape[0]):
            excess = _true_class_loss(labels[mask], probability[seed_index, mask]) - (
                _true_class_loss(labels[mask], current[seed_index, mask])
            )
            sums[seed_index, position] = float(np.sum(excess))
    return sums, counts


def _bootstrap_mean_regret_difference(
    *,
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    current: np.ndarray,
    clusters: np.ndarray,
    eligible: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, np.ndarray, int]:
    cluster_values = np.unique(clusters[eligible])
    if len(cluster_values) < 2:
        raise EvidenceContractError("mean-regret bootstrap needs two history-eligible clusters")
    candidate_sum, counts = _cluster_regret_sums(
        labels, candidate, current, clusters, eligible, cluster_values
    )
    reference_sum, reference_counts = _cluster_regret_sums(
        labels, reference, current, clusters, eligible, cluster_values
    )
    if not np.array_equal(counts, reference_counts) or np.any(counts <= 0):
        raise AssertionError("paired regret cluster counts differ")
    point = float(
        np.mean(candidate_sum.sum(axis=1) / counts.sum())
        - np.mean(reference_sum.sum(axis=1) / counts.sum())
    )
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        seed_draw, cluster_draw = _draw_crossed_seed_shared_clusters(
            rng,
            seed_count=candidate.shape[0],
            cluster_count=len(cluster_values),
        )
        denominator = int(np.sum(counts[cluster_draw]))
        candidate_value = np.mean(
            [candidate_sum[int(sampled_seed), cluster_draw].sum() / denominator for sampled_seed in seed_draw]
        )
        reference_value = np.mean(
            [reference_sum[int(sampled_seed), cluster_draw].sum() / denominator for sampled_seed in seed_draw]
        )
        bootstrap[replicate] = float(candidate_value - reference_value)
    return point, bootstrap, len(cluster_values)


def _randomization_assignments(
    *,
    cluster_count: int,
    replicates: int,
    seed: int,
) -> tuple[Sequence[np.ndarray], bool, int]:
    """Return exact or Monte Carlo whole-cluster swap assignments."""

    if int(cluster_count) < 2:
        raise EvidenceContractError("randomization test needs at least two clusters")
    if int(cluster_count) <= MAX_EXACT_RANDOMIZATION_CLUSTERS:
        count = 1 << int(cluster_count)
        bit_positions = np.arange(int(cluster_count), dtype=np.uint64)
        assignments = tuple(
            ((np.uint64(value) >> bit_positions) & np.uint64(1)).astype(bool)
            for value in range(count)
        )
        return assignments, True, count
    if int(replicates) < 1_000:
        raise EvidenceContractError(
            "Monte Carlo paired randomization requires at least 1000 assignments"
        )
    rng = np.random.default_rng(int(seed))
    assignments = tuple(
        rng.integers(0, 2, size=int(cluster_count), dtype=np.int8).astype(bool)
        for _ in range(int(replicates))
    )
    return assignments, False, int(replicates)


def _paired_whole_cluster_randomization_arrays(
    *,
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    current: np.ndarray,
    clusters: np.ndarray,
    eligible: np.ndarray,
    metric: str,
    alternative: str,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Test a sharp paired null by swapping complete clusters across methods.

    One Boolean swap is drawn per independent cluster and is shared across all
    five training seeds and all queries in that cluster.  Seeds are not
    resampled for this hypothesis test.  Nonlinear classification metrics are
    recomputed after every assignment.
    """

    if candidate.shape != reference.shape or candidate.shape != current.shape:
        raise EvidenceContractError("randomization probabilities are not aligned")
    if candidate.ndim != 3 or candidate.shape[0] != 5:
        raise EvidenceContractError("randomization requires five seed probability arrays")
    if alternative not in {"greater", "less"}:
        raise EvidenceContractError("randomization alternative must be greater or less")
    if metric in {"macro_f1", "accuracy"}:
        cluster_values = np.unique(clusters)
        candidate_confusion = _confusion_by_seed_cluster(
            labels, candidate, clusters, cluster_values
        )
        reference_confusion = _confusion_by_seed_cluster(
            labels, reference, clusters, cluster_values
        )

        def statistic(swap: np.ndarray) -> float:
            candidate_matrix = np.where(
                swap[None, :, None, None],
                reference_confusion,
                candidate_confusion,
            ).sum(axis=1)
            reference_matrix = np.where(
                swap[None, :, None, None],
                candidate_confusion,
                reference_confusion,
            ).sum(axis=1)
            return float(
                np.mean(
                    [
                        _classification_score_from_confusion(
                            candidate_matrix[seed_index], metric
                        )
                        - _classification_score_from_confusion(
                            reference_matrix[seed_index], metric
                        )
                        for seed_index in range(candidate.shape[0])
                    ]
                )
            )

    elif metric == "mean_regret":
        cluster_values = np.unique(clusters[eligible])
        if len(cluster_values) < 2:
            raise EvidenceContractError(
                "mean-regret randomization needs two history-eligible clusters"
            )
        candidate_sum, counts = _cluster_regret_sums(
            labels, candidate, current, clusters, eligible, cluster_values
        )
        reference_sum, reference_counts = _cluster_regret_sums(
            labels, reference, current, clusters, eligible, cluster_values
        )
        if not np.array_equal(counts, reference_counts) or np.any(counts <= 0):
            raise AssertionError("randomization regret cluster counts differ")
        denominator = int(np.sum(counts))

        def statistic(swap: np.ndarray) -> float:
            candidate_value = np.where(
                swap[None, :], reference_sum, candidate_sum
            ).sum(axis=1) / denominator
            reference_value = np.where(
                swap[None, :], candidate_sum, reference_sum
            ).sum(axis=1) / denominator
            return float(np.mean(candidate_value - reference_value))

    else:
        raise EvidenceContractError("randomization metric is unsupported")
    observed_swap = np.zeros(len(cluster_values), dtype=bool)
    observed_raw = statistic(observed_swap)
    observed_favorable = observed_raw if alternative == "greater" else -observed_raw
    assignments, exact, assignment_count = _randomization_assignments(
        cluster_count=len(cluster_values),
        replicates=int(replicates),
        seed=int(seed),
    )
    permutation = np.asarray(
        [statistic(swap) for swap in assignments], dtype=np.float64
    )
    favorable = permutation if alternative == "greater" else -permutation
    tolerance = 1.0e-14 * max(1.0, abs(observed_favorable))
    extreme = int(np.sum(favorable >= observed_favorable - tolerance))
    p_value = (
        float(extreme / assignment_count)
        if exact
        else float((extreme + 1) / (assignment_count + 1))
    )
    return {
        "point_difference": observed_raw,
        "favorable_direction_point": observed_favorable,
        "paired_whole_cluster_randomization_p_value": p_value,
        "test_design": "paired_whole_cluster_swap_shared_across_all_five_seeds",
        "sharp_null": "candidate_reference_exchangeable_within_each_independent_cluster",
        "seed_resampling_in_hypothesis_test": False,
        "one_swap_shared_across_five_seeds": True,
        "queries_within_cluster_kept_together": True,
        "nonlinear_metric_recomputed_each_assignment": metric
        in {"macro_f1", "accuracy"},
        "exact_enumeration": exact,
        "assignment_count": assignment_count,
        "randomization_seed": None if exact else int(seed),
        "cluster_count": int(len(cluster_values)),
    }


def paired_whole_cluster_randomization_test(
    bundle: EvidenceBundle,
    hypothesis: HolmHypothesis,
    *,
    replicates: int = DEFAULT_RANDOMIZATION_REPLICATES,
    randomization_seed: int = DEFAULT_RANDOMIZATION_SEED,
) -> dict[str, object]:
    """Run the predeclared paired whole-cluster hypothesis test."""

    validate_evidence_bundle(bundle)
    if hypothesis.candidate not in bundle.methods or hypothesis.reference not in bundle.methods:
        raise EvidenceContractError("randomization contrast names an absent method")
    labels = np.asarray(bundle.labels, dtype=np.int64)
    candidate = np.asarray(bundle.methods[hypothesis.candidate].probability, dtype=np.float64)
    reference = np.asarray(bundle.methods[hypothesis.reference].probability, dtype=np.float64)
    current = np.asarray(bundle.methods[bundle.current_only_method].probability, dtype=np.float64)
    clusters = np.asarray(bundle.cluster_codes, dtype=np.int64)
    eligible = np.asarray(
        [bool(bundle.histories[int(query)]) for query in bundle.query_indices], dtype=bool
    )
    return _paired_whole_cluster_randomization_arrays(
        labels=labels,
        candidate=candidate,
        reference=reference,
        current=current,
        clusters=clusters,
        eligible=eligible,
        metric=hypothesis.metric,
        alternative=hypothesis.alternative,
        replicates=int(replicates),
        seed=int(randomization_seed),
    )


def paired_seed_shared_cluster_contrast(
    bundle: EvidenceBundle,
    hypothesis: HolmHypothesis,
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    randomization_replicates: int = DEFAULT_RANDOMIZATION_REPLICATES,
    randomization_seed: int = DEFAULT_RANDOMIZATION_SEED,
) -> dict[str, object]:
    """Paired five-seed bootstrap with one shared whole-cluster draw."""

    validate_evidence_bundle(bundle)
    if int(replicates) < 100:
        raise EvidenceContractError("paired bootstrap requires at least 100 replicates")
    if hypothesis.candidate not in bundle.methods or hypothesis.reference not in bundle.methods:
        raise EvidenceContractError("Holm contrast names a method absent from the bundle")
    candidate = np.asarray(bundle.methods[hypothesis.candidate].probability, dtype=np.float64)
    reference = np.asarray(bundle.methods[hypothesis.reference].probability, dtype=np.float64)
    current = np.asarray(bundle.methods[bundle.current_only_method].probability, dtype=np.float64)
    labels = np.asarray(bundle.labels, dtype=np.int64)
    clusters = np.asarray(bundle.cluster_codes, dtype=np.int64)
    eligible = np.asarray([bool(bundle.histories[int(query)]) for query in bundle.query_indices])
    if not np.any(eligible):
        raise EvidenceContractError("performance evidence has no history-eligible query")
    if hypothesis.metric in {"macro_f1", "accuracy"}:
        point, bootstrap, cluster_count = _bootstrap_classification_difference(
            labels=labels,
            candidate=candidate,
            reference=reference,
            clusters=clusters,
            replicates=int(replicates),
            seed=int(bootstrap_seed),
            metric=hypothesis.metric,
        )
    elif hypothesis.metric == "mean_regret":
        point, bootstrap, cluster_count = _bootstrap_mean_regret_difference(
            labels=labels,
            candidate=candidate,
            reference=reference,
            current=current,
            clusters=clusters,
            eligible=eligible,
            replicates=int(replicates),
            seed=int(bootstrap_seed),
        )
    else:
        raise EvidenceContractError("unsupported bootstrap metric")
    favorable_point = point if hypothesis.alternative == "greater" else -point
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    randomization = paired_whole_cluster_randomization_test(
        bundle,
        hypothesis,
        replicates=int(randomization_replicates),
        randomization_seed=int(randomization_seed),
    )
    if not math.isclose(
        point,
        float(randomization["point_difference"]),
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    ):
        raise AssertionError("bootstrap and randomization point estimands differ")
    return {
        "metric": hypothesis.metric,
        "alternative": hypothesis.alternative,
        "difference_definition": "candidate_minus_reference",
        "point_difference": point,
        "favorable_direction_point": favorable_point,
        "ci95_percentile": [float(low), float(high)],
        "bootstrap_design": "five_training_seeds_crossed_with_shared_whole_cluster_draw",
        "pairing": "same_seed_same_cluster_same_query_candidate_and_reference",
        "training_seed_count": 5,
        "cluster_count": int(cluster_count),
        "replicates": int(replicates),
        "bootstrap_seed": int(bootstrap_seed),
        "hypothesis_test": randomization,
        "queries_within_cluster_kept_together": True,
        "independent_query_resampling": False,
    }


def _public_method_metrics(bundle: EvidenceBundle) -> dict[str, object]:
    labels = np.asarray(bundle.labels, dtype=np.int64)
    current = np.asarray(bundle.methods[bundle.current_only_method].probability, dtype=np.float64)
    eligible = np.asarray([bool(bundle.histories[int(query)]) for query in bundle.query_indices])
    return {
        name: _aggregate_method_metrics(
            _method_seed_metrics(
                labels,
                np.asarray(method.probability, dtype=np.float64),
                current,
                eligible,
            )
        )
        for name, method in sorted(bundle.methods.items())
    }


def _validate_json_safe_no_paths(value: object) -> None:
    """Generic second-line privacy check used after schema validation."""

    def visit(child: object, trail: tuple[str, ...]) -> None:
        if isinstance(child, np.ndarray):
            raise EvidenceContractError(
                f"public aggregate contains an ndarray at {'.'.join(trail)}"
            )
        if isinstance(child, Mapping):
            for raw_key, nested in child.items():
                key = str(raw_key)
                if key in _FORBIDDEN_PUBLIC_KEYS:
                    raise EvidenceContractError(f"public aggregate exposes forbidden field {key}")
                visit(nested, (*trail, key))
            return
        if isinstance(child, (list, tuple)):
            for index, nested in enumerate(child):
                visit(nested, (*trail, str(index)))
            return
        if isinstance(child, str):
            if _WINDOWS_ABSOLUTE_PATH.match(child) or child.startswith(("/", "file://")):
                raise EvidenceContractError(
                    f"public aggregate contains a private/local path at {'.'.join(trail)}"
                )
            return
        if child is not None and not isinstance(child, (bool, int, float)):
            raise EvidenceContractError(
                f"public aggregate contains a non-JSON value at {'.'.join(trail)}"
            )
        if isinstance(child, float) and not math.isfinite(child):
            raise EvidenceContractError(
                f"public aggregate contains a non-finite value at {'.'.join(trail)}"
            )

    visit(value, ("root",))


def _public_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EvidenceContractError(f"public {label} must be a mapping")
    return value


def _public_exact_keys(
    value: object,
    expected: set[str],
    label: str,
) -> Mapping[str, object]:
    mapping = _public_mapping(value, label)
    observed = {str(key) for key in mapping}
    if observed != expected:
        raise EvidenceContractError(
            f"public {label} schema mismatch: missing={sorted(expected-observed)}, "
            f"unknown={sorted(observed-expected)}"
        )
    return mapping


def _public_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceContractError(f"public {label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceContractError(f"public {label} must be finite")
    return result


def _validate_public_contrast(
    value: object,
    *,
    label: str,
    kind: str,
) -> None:
    base_keys = {
        "metric",
        "alternative",
        "difference_definition",
        "point_difference",
        "favorable_direction_point",
        "ci95_percentile",
        "bootstrap_design",
        "pairing",
        "training_seed_count",
        "cluster_count",
        "replicates",
        "bootstrap_seed",
        "hypothesis_test",
        "queries_within_cluster_kept_together",
        "independent_query_resampling",
    }
    if kind == "holm":
        expected = base_keys | {"multiplicity"}
    elif kind == "accuracy":
        expected = base_keys | {
            "minimum_point_difference",
            "minimum_ci95_lower",
            "passed",
            "accuracy_improvement_supported",
            "noninferiority_is_not_improvement_evidence",
        }
    else:
        raise AssertionError("unknown public contrast kind")
    contrast = _public_exact_keys(value, expected, label)
    if contrast["metric"] not in {"macro_f1", "mean_regret", "accuracy"}:
        raise EvidenceContractError(f"public {label} metric is invalid")
    if contrast["alternative"] not in {"greater", "less"}:
        raise EvidenceContractError(f"public {label} alternative is invalid")
    if contrast["difference_definition"] != "candidate_minus_reference":
        raise EvidenceContractError(f"public {label} difference definition changed")
    if contrast["bootstrap_design"] != (
        "five_training_seeds_crossed_with_shared_whole_cluster_draw"
    ) or contrast["pairing"] != (
        "same_seed_same_cluster_same_query_candidate_and_reference"
    ):
        raise EvidenceContractError(f"public {label} bootstrap pairing changed")
    for field in (
        "point_difference",
        "favorable_direction_point",
        "training_seed_count",
        "cluster_count",
        "replicates",
        "bootstrap_seed",
    ):
        _public_number(contrast[field], f"{label}.{field}")
    ci = contrast["ci95_percentile"]
    if not isinstance(ci, list) or len(ci) != 2:
        raise EvidenceContractError(f"public {label} CI must contain two aggregate bounds")
    _public_number(ci[0], f"{label}.ci_low")
    _public_number(ci[1], f"{label}.ci_high")
    if (
        contrast["training_seed_count"] != 5
        or contrast["queries_within_cluster_kept_together"] is not True
        or contrast["independent_query_resampling"] is not False
    ):
        raise EvidenceContractError(f"public {label} bootstrap axis contract changed")
    test = _public_exact_keys(
        contrast["hypothesis_test"],
        {
            "point_difference",
            "favorable_direction_point",
            "paired_whole_cluster_randomization_p_value",
            "test_design",
            "sharp_null",
            "seed_resampling_in_hypothesis_test",
            "one_swap_shared_across_five_seeds",
            "queries_within_cluster_kept_together",
            "nonlinear_metric_recomputed_each_assignment",
            "exact_enumeration",
            "assignment_count",
            "randomization_seed",
            "cluster_count",
        },
        f"{label}.hypothesis_test",
    )
    for field in (
        "point_difference",
        "favorable_direction_point",
        "paired_whole_cluster_randomization_p_value",
        "assignment_count",
        "cluster_count",
    ):
        _public_number(test[field], f"{label}.hypothesis_test.{field}")
    if test["randomization_seed"] is not None:
        _public_number(test["randomization_seed"], f"{label}.hypothesis_test.randomization_seed")
    p_value = float(test["paired_whole_cluster_randomization_p_value"])
    if not 0.0 <= p_value <= 1.0:
        raise EvidenceContractError(f"public {label} randomization p-value is invalid")
    if (
        test["test_design"]
        != "paired_whole_cluster_swap_shared_across_all_five_seeds"
        or test["sharp_null"]
        != "candidate_reference_exchangeable_within_each_independent_cluster"
        or test["seed_resampling_in_hypothesis_test"] is not False
        or test["one_swap_shared_across_five_seeds"] is not True
        or test["queries_within_cluster_kept_together"] is not True
        or not isinstance(test["nonlinear_metric_recomputed_each_assignment"], bool)
        or not isinstance(test["exact_enumeration"], bool)
    ):
        raise EvidenceContractError(f"public {label} randomization contract changed")
    if kind == "holm":
        multiplicity = _public_exact_keys(
            contrast["multiplicity"],
            {
                "raw_p_value",
                "holm_adjusted_p_value",
                "holm_rank",
                "rejected_at_familywise_alpha",
            },
            f"{label}.multiplicity",
        )
        for field in ("raw_p_value", "holm_adjusted_p_value", "holm_rank"):
            _public_number(multiplicity[field], f"{label}.multiplicity.{field}")
        if not math.isclose(
            float(multiplicity["raw_p_value"]),
            p_value,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ) or not 0.0 <= float(multiplicity["holm_adjusted_p_value"]) <= 1.0:
            raise EvidenceContractError(f"public {label} Holm p-value source differs")
        if not isinstance(multiplicity["rejected_at_familywise_alpha"], bool):
            raise EvidenceContractError(f"public {label} Holm rejection is not boolean")
    else:
        _public_number(contrast["minimum_point_difference"], f"{label}.minimum_point_difference")
        _public_number(contrast["minimum_ci95_lower"], f"{label}.minimum_ci95_lower")
        if (
            not isinstance(contrast["passed"], bool)
            or not isinstance(contrast["accuracy_improvement_supported"], bool)
            or contrast["noninferiority_is_not_improvement_evidence"] is not True
        ):
            raise EvidenceContractError(f"public {label} accuracy interpretation changed")


def validate_aggregate_public_output(payload: Mapping[str, object]) -> None:
    """Validate the exact controlled single-dataset public report schema."""

    # Run the generic screen first so a forbidden field gets an explicit error
    # even when it also violates the exact whitelist.
    _validate_json_safe_no_paths(payload)
    root = _public_exact_keys(
        payload,
        {
            "schema_version",
            "status",
            "claim_boundary",
            "performance_claim_gate",
            "dataset",
            "role",
            "cross_dataset_claim_gate",
            "counts",
            "independent_current_only_contract",
            "operating_point",
            "coverage_matched_recency_contract",
            "aggregate_method_metrics",
            "predeclared_holm_family",
            "mandatory_accuracy_no_harm_gate",
            "contrasts",
            "public_artifact_policy",
        },
        "root",
    )
    if root["schema_version"] != PUBLIC_EVIDENCE_SCHEMA:
        raise EvidenceContractError("public evidence schema version changed")
    if root["status"] != SINGLE_DATASET_STATUS:
        raise EvidenceContractError("public evidence status is not the controlled builder status")
    if root["claim_boundary"] != SINGLE_DATASET_CLAIM_BOUNDARY:
        raise EvidenceContractError("public single-dataset claim boundary changed")
    if root["dataset"] not in SUPPORTED_DATASETS or root["role"] not in {
        "fit",
        "model_selection",
    }:
        raise EvidenceContractError("public dataset or role is invalid")
    performance = _public_exact_keys(
        root["performance_claim_gate"],
        {
            "authorized",
            "reason",
            "accuracy_no_harm_gate_passed",
            "macro_f1_success_cannot_override_accuracy_harm",
            "remaining_requirements",
        },
        "performance_claim_gate",
    )
    if performance["authorized"] is not False or performance["reason"] != (
        "open_role_fit_or_model_selection_evidence_only"
    ):
        raise EvidenceContractError("single-dataset performance claim was authorized")
    if (
        not isinstance(performance["accuracy_no_harm_gate_passed"], bool)
        or performance["macro_f1_success_cannot_override_accuracy_harm"] is not True
    ):
        raise EvidenceContractError("public accuracy gating declaration changed")
    if tuple(performance["remaining_requirements"]) != _REMAINING_REQUIREMENTS:
        raise EvidenceContractError("public remaining-requirement list changed")
    cross = _public_exact_keys(
        root["cross_dataset_claim_gate"],
        {
            "method_success_authorized",
            "single_dataset_can_trigger_method_success",
            "required_datasets",
            "present_dataset",
            "required_aggregator_schema",
        },
        "cross_dataset_claim_gate",
    )
    if (
        cross["method_success_authorized"] is not False
        or cross["single_dataset_can_trigger_method_success"] is not False
        or cross["required_datasets"] != ["EmotionTalk", "MELD"]
        or cross["present_dataset"] != root["dataset"]
        or cross["required_aggregator_schema"] != CROSS_DATASET_INDEX_SCHEMA
    ):
        raise EvidenceContractError("public cross-dataset gate changed")
    counts = _public_exact_keys(
        root["counts"],
        {"queries", "clusters", "history_eligible_queries", "training_seeds"},
        "counts",
    )
    for field in counts:
        _public_number(counts[field], f"counts.{field}")
    if counts["training_seeds"] != 5:
        raise EvidenceContractError("public report does not contain five training seeds")
    independent = _public_exact_keys(
        root["independent_current_only_contract"],
        {
            "training_protocol",
            "history_training_items_consumed",
            "history_inference_items_consumed",
            "source_identity_differs_from_history_backbone",
            "checkpoint_manifest_differs_from_history_backbone",
            "independence_attestation_sha256",
        },
        "independent_current_only_contract",
    )
    if (
        independent["training_protocol"] != INDEPENDENT_CURRENT_ONLY_PROTOCOL
        or independent["history_training_items_consumed"] != 0
        or independent["history_inference_items_consumed"] != 0
        or independent["source_identity_differs_from_history_backbone"] is not True
        or independent["checkpoint_manifest_differs_from_history_backbone"] is not True
    ):
        raise EvidenceContractError("public independent current-only contract changed")
    _require_sha256(
        independent["independence_attestation_sha256"],
        "independence_attestation_sha256",
    )
    operating = _public_exact_keys(
        root["operating_point"],
        {
            "frozen_on_role",
            "target_candidate_pair_coverage",
            "fit_pair_count",
            "fit_selected_pair_count",
            "fit_realized_pair_coverage",
            "evaluation_pair_count",
            "evaluation_selected_pair_count",
            "evaluation_realized_pair_coverage",
            "boundary_tie_rule",
            "boundary_tie_fraction",
            "selection_uses_labels_clusters_or_utilities",
            "rule_sha256",
            "policy_sha256",
        },
        "operating_point",
    )
    for field in (
        "target_candidate_pair_coverage",
        "fit_pair_count",
        "fit_selected_pair_count",
        "fit_realized_pair_coverage",
        "evaluation_pair_count",
        "evaluation_selected_pair_count",
        "evaluation_realized_pair_coverage",
        "boundary_tie_fraction",
    ):
        _public_number(operating[field], f"operating_point.{field}")
    if operating["target_candidate_pair_coverage"] != PRIMARY_TARGET_COVERAGE:
        raise EvidenceContractError("public operating point is not frozen at 25%")
    _require_sha256(operating["rule_sha256"], "rule_sha256")
    _require_sha256(operating["policy_sha256"], "policy_sha256")
    if (
        operating["frozen_on_role"] != "fit_oof"
        or operating["boundary_tie_rule"]
        != "stable_sha256_rank_with_frozen_fraction"
        or operating["selection_uses_labels_clusters_or_utilities"] is not False
    ):
        raise EvidenceContractError("public operating-point selection contract changed")
    recency = _public_exact_keys(
        root["coverage_matched_recency_contract"],
        {
            "same_selected_history_count_for_every_query",
            "recency_uses_most_recent_strict_past_items",
        },
        "coverage_matched_recency_contract",
    )
    if any(value is not True for value in recency.values()):
        raise EvidenceContractError("public matched-recency contract changed")
    metrics = _public_mapping(root["aggregate_method_metrics"], "aggregate_method_metrics")
    metric_names = {
        "macro_f1",
        "weighted_f1",
        "accuracy",
        "nll",
        "brier",
        "ece",
        "mean_regret",
        "history_harm_rate",
        "p90_regret",
        "cvar90_regret",
    }
    if not metrics:
        raise EvidenceContractError("public method metrics are empty")
    for method_name, method_value in metrics.items():
        if not _PUBLIC_ID.fullmatch(str(method_name)):
            raise EvidenceContractError("public method id is unsafe")
        method = _public_exact_keys(
            method_value, metric_names, f"aggregate_method_metrics.{method_name}"
        )
        for metric_name, summary_value in method.items():
            summary = _public_exact_keys(
                summary_value,
                {"five_seed_mean", "five_seed_sample_sd"},
                f"aggregate_method_metrics.{method_name}.{metric_name}",
            )
            _public_number(summary["five_seed_mean"], f"{method_name}.{metric_name}.mean")
            _public_number(summary["five_seed_sample_sd"], f"{method_name}.{metric_name}.sd")
    holm = _public_exact_keys(
        root["predeclared_holm_family"],
        {
            "family_id",
            "method",
            "familywise_alpha",
            "family_sha256",
            "analysis_config_sha256",
            "complete_family_evaluated",
        },
        "predeclared_holm_family",
    )
    if holm["method"] != "holm_bonferroni" or holm["complete_family_evaluated"] is not True:
        raise EvidenceContractError("public Holm family is incomplete")
    if not _PUBLIC_ID.fullmatch(str(holm["family_id"])):
        raise EvidenceContractError("public Holm family id is unsafe")
    _public_number(holm["familywise_alpha"], "holm.familywise_alpha")
    _require_sha256(holm["family_sha256"], "family_sha256")
    _require_sha256(holm["analysis_config_sha256"], "analysis_config_sha256")
    contrasts = _public_mapping(root["contrasts"], "contrasts")
    if not contrasts:
        raise EvidenceContractError("public Holm contrasts are empty")
    for contrast_id, contrast in contrasts.items():
        if not _PUBLIC_ID.fullmatch(str(contrast_id)):
            raise EvidenceContractError("public contrast id is unsafe")
        _validate_public_contrast(contrast, label=f"contrasts.{contrast_id}", kind="holm")
    accuracy = _public_exact_keys(
        root["mandatory_accuracy_no_harm_gate"],
        {
            "gate_id",
            "metric",
            "mandatory_for_method_success",
            "minimum_point_difference",
            "minimum_ci95_lower",
            "ci_lower_margin_interpretation",
            "gate_sha256",
            "analysis_config_sha256",
            "all_predeclared_contrasts_passed",
            "macro_f1_success_cannot_override_failure",
            "contrasts",
        },
        "mandatory_accuracy_no_harm_gate",
    )
    if accuracy["metric"] != "accuracy" or accuracy["mandatory_for_method_success"] is not True:
        raise EvidenceContractError("public accuracy no-harm gate changed")
    if (
        not _PUBLIC_ID.fullmatch(str(accuracy["gate_id"]))
        or not isinstance(accuracy["all_predeclared_contrasts_passed"], bool)
        or accuracy["macro_f1_success_cannot_override_failure"] is not True
        or accuracy["ci_lower_margin_interpretation"]
        != "minus_0.005_is_a_noninferiority_no_harm_margin_not_evidence_of_improvement"
    ):
        raise EvidenceContractError("public accuracy gate interpretation changed")
    _public_number(accuracy["minimum_point_difference"], "accuracy.minimum_point_difference")
    _public_number(accuracy["minimum_ci95_lower"], "accuracy.minimum_ci95_lower")
    if accuracy["minimum_point_difference"] != 0.0 or accuracy["minimum_ci95_lower"] != -0.005:
        raise EvidenceContractError("public accuracy no-harm margins changed")
    _require_sha256(accuracy["gate_sha256"], "accuracy.gate_sha256")
    _require_sha256(accuracy["analysis_config_sha256"], "accuracy.analysis_config_sha256")
    accuracy_contrasts = _public_mapping(accuracy["contrasts"], "accuracy.contrasts")
    if not accuracy_contrasts:
        raise EvidenceContractError("public accuracy contrasts are empty")
    for contrast_id, contrast in accuracy_contrasts.items():
        if not _PUBLIC_ID.fullmatch(str(contrast_id)):
            raise EvidenceContractError("public accuracy contrast id is unsafe")
        _validate_public_contrast(
            contrast,
            label=f"mandatory_accuracy_no_harm_gate.contrasts.{contrast_id}",
            kind="accuracy",
        )
    public_policy = _public_exact_keys(
        root["public_artifact_policy"],
        {
            "aggregate_only",
            "contains_query_row_group_or_speaker_identifiers",
            "contains_predictions_labels_embeddings_or_contexts",
            "contains_private_paths",
        },
        "public_artifact_policy",
    )
    if (
        public_policy["aggregate_only"] is not True
        or any(
            public_policy[field] is not False
            for field in (
                "contains_query_row_group_or_speaker_identifiers",
                "contains_predictions_labels_embeddings_or_contexts",
                "contains_private_paths",
            )
        )
    ):
        raise EvidenceContractError("public artifact privacy declaration changed")


def write_aggregate_public_report(
    payload: Mapping[str, object],
    path: str | Path,
) -> None:
    """Atomically write a validated, write-once aggregate JSON artifact."""

    validate_aggregate_public_output(payload)
    output = Path(path)
    if output.suffix.lower() != ".json":
        raise EvidenceContractError("public evidence artifact must be JSON")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A same-directory hard link gives destination O_EXCL semantics on
            # NTFS and POSIX; unlike exists()+os.replace(), concurrent writers
            # can never overwrite the winner.
            os.link(temporary, output)
        except FileExistsError:
            raise FileExistsError(
                f"public evidence artifact already exists: {output.name}"
            ) from None
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def evaluate_open_role_evidence(
    bundle: EvidenceBundle,
    family: PredeclaredHolmFamily,
    accuracy_gate: PredeclaredAccuracyNoHarmGate,
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    randomization_replicates: int = DEFAULT_RANDOMIZATION_REPLICATES,
    randomization_seed: int = DEFAULT_RANDOMIZATION_SEED,
) -> dict[str, object]:
    """Evaluate a complete frozen family and return aggregate-only output."""

    validate_evidence_bundle(bundle)
    method_names = set(bundle.methods)
    for hypothesis in family.hypotheses:
        if hypothesis.candidate not in method_names or hypothesis.reference not in method_names:
            raise EvidenceContractError("predeclared Holm family is not fully represented")
    contrasts: dict[str, object] = {}
    raw_p_values: dict[str, float] = {}
    for hypothesis in family.hypotheses:
        result = paired_seed_shared_cluster_contrast(
            bundle,
            hypothesis,
            replicates=int(replicates),
            bootstrap_seed=int(bootstrap_seed),
            randomization_replicates=int(randomization_replicates),
            randomization_seed=int(randomization_seed),
        )
        contrasts[hypothesis.hypothesis_id] = result
        raw_p_values[hypothesis.hypothesis_id] = float(
            result["hypothesis_test"]["paired_whole_cluster_randomization_p_value"]
        )
    adjusted = holm_bonferroni(
        raw_p_values,
        declared_order=[row.hypothesis_id for row in family.hypotheses],
        alpha=family.alpha,
    )
    for hypothesis_id, result in contrasts.items():
        assert isinstance(result, dict)
        result["multiplicity"] = adjusted[hypothesis_id]
    accuracy_results: dict[str, object] = {}
    for contrast in accuracy_gate.contrasts:
        if contrast.candidate not in method_names or contrast.reference not in method_names:
            raise EvidenceContractError("predeclared accuracy contrast is not fully represented")
        result = paired_seed_shared_cluster_contrast(
            bundle,
            HolmHypothesis(
                hypothesis_id=contrast.contrast_id,
                candidate=contrast.candidate,
                reference=contrast.reference,
                metric="accuracy",
                alternative="greater",
            ),
            replicates=int(replicates),
            bootstrap_seed=int(bootstrap_seed),
            randomization_replicates=int(randomization_replicates),
            randomization_seed=int(randomization_seed),
        )
        ci_low = float(result["ci95_percentile"][0])
        point = float(result["point_difference"])
        result["minimum_point_difference"] = accuracy_gate.minimum_point_difference
        result["minimum_ci95_lower"] = accuracy_gate.minimum_ci95_lower
        result["passed"] = bool(
            point >= accuracy_gate.minimum_point_difference
            and ci_low >= accuracy_gate.minimum_ci95_lower
        )
        result["accuracy_improvement_supported"] = bool(point > 0.0 and ci_low > 0.0)
        result["noninferiority_is_not_improvement_evidence"] = True
        accuracy_results[contrast.contrast_id] = result
    accuracy_gate_passed = bool(
        accuracy_results
        and all(bool(result["passed"]) for result in accuracy_results.values())
    )
    current = bundle.independent_current_only
    report: dict[str, object] = {
        "schema_version": PUBLIC_EVIDENCE_SCHEMA,
        "status": SINGLE_DATASET_STATUS,
        "claim_boundary": SINGLE_DATASET_CLAIM_BOUNDARY,
        "performance_claim_gate": {
            "authorized": False,
            "reason": "open_role_fit_or_model_selection_evidence_only",
            "accuracy_no_harm_gate_passed": accuracy_gate_passed,
            "macro_f1_success_cannot_override_accuracy_harm": True,
            "remaining_requirements": list(_REMAINING_REQUIREMENTS),
        },
        "dataset": bundle.producer.dataset,
        "role": bundle.role,
        "cross_dataset_claim_gate": {
            "method_success_authorized": False,
            "single_dataset_can_trigger_method_success": False,
            "required_datasets": ["EmotionTalk", "MELD"],
            "present_dataset": bundle.producer.dataset,
            "required_aggregator_schema": CROSS_DATASET_INDEX_SCHEMA,
        },
        "counts": {
            "queries": int(len(bundle.query_indices)),
            "clusters": int(len(np.unique(bundle.cluster_codes))),
            "history_eligible_queries": int(
                sum(bool(bundle.histories[int(query)]) for query in bundle.query_indices)
            ),
            "training_seeds": 5,
        },
        "independent_current_only_contract": {
            "training_protocol": INDEPENDENT_CURRENT_ONLY_PROTOCOL,
            "history_training_items_consumed": 0,
            "history_inference_items_consumed": 0,
            "source_identity_differs_from_history_backbone": (
                current.source_identity_sha256 != bundle.producer.source_identity_sha256
            ),
            "checkpoint_manifest_differs_from_history_backbone": (
                current.checkpoint_manifest_sha256
                != bundle.producer.checkpoint_manifest_sha256
            ),
            "independence_attestation_sha256": current.independence_attestation_sha256,
        },
        "operating_point": {
            "frozen_on_role": bundle.policy.rule.frozen_on_role,
            "target_candidate_pair_coverage": bundle.policy.rule.target_coverage,
            "fit_pair_count": bundle.policy.rule.fit_pair_count,
            "fit_selected_pair_count": bundle.policy.rule.fit_selected_count,
            "fit_realized_pair_coverage": bundle.policy.rule.fit_realized_coverage,
            "evaluation_pair_count": bundle.policy.available_pair_count,
            "evaluation_selected_pair_count": bundle.policy.selected_pair_count,
            "evaluation_realized_pair_coverage": bundle.policy.realized_pair_coverage,
            "boundary_tie_rule": "stable_sha256_rank_with_frozen_fraction",
            "boundary_tie_fraction": bundle.policy.rule.boundary_tie_fraction,
            "selection_uses_labels_clusters_or_utilities": False,
            "rule_sha256": bundle.policy.rule.rule_sha256,
            "policy_sha256": bundle.policy.policy_sha256,
        },
        "coverage_matched_recency_contract": {
            "same_selected_history_count_for_every_query": True,
            "recency_uses_most_recent_strict_past_items": True,
        },
        "aggregate_method_metrics": _public_method_metrics(bundle),
        "predeclared_holm_family": {
            "family_id": family.family_id,
            "method": "holm_bonferroni",
            "familywise_alpha": family.alpha,
            "family_sha256": family.family_sha256,
            "analysis_config_sha256": family.analysis_config_sha256,
            "complete_family_evaluated": True,
        },
        "mandatory_accuracy_no_harm_gate": {
            "gate_id": accuracy_gate.gate_id,
            "metric": "accuracy",
            "mandatory_for_method_success": True,
            "minimum_point_difference": accuracy_gate.minimum_point_difference,
            "minimum_ci95_lower": accuracy_gate.minimum_ci95_lower,
            "ci_lower_margin_interpretation": (
                "minus_0.005_is_a_noninferiority_no_harm_margin_not_evidence_of_improvement"
            ),
            "gate_sha256": accuracy_gate.gate_sha256,
            "analysis_config_sha256": accuracy_gate.analysis_config_sha256,
            "all_predeclared_contrasts_passed": accuracy_gate_passed,
            "macro_f1_success_cannot_override_failure": True,
            "contrasts": accuracy_results,
        },
        "contrasts": contrasts,
        "public_artifact_policy": {
            "aggregate_only": True,
            "contains_query_row_group_or_speaker_identifiers": False,
            "contains_predictions_labels_embeddings_or_contexts": False,
            "contains_private_paths": False,
        },
    }
    validate_aggregate_public_output(report)
    return report


def aggregate_required_dataset_reports(
    reports: Mapping[str, Mapping[str, object]],
    plan: CrossDatasetAggregationPlan,
) -> dict[str, object]:
    """Link both single-dataset reports without authorizing a paper claim.

    This is intentionally an index, not a meta-analytic CI.  Joint
    confirmatory inference must operate on the private paired replicate state,
    not reconstruct uncertainty from two published percentile intervals.
    """

    if tuple(plan.required_datasets) != ("EmotionTalk", "MELD"):
        raise EvidenceContractError("cross-dataset aggregation plan changed")
    if set(reports) != set(plan.required_datasets):
        raise EvidenceContractError(
            "both required dataset reports must be present; one dataset cannot trigger success"
        )
    report_hashes: dict[str, str] = {}
    accuracy_passes: dict[str, bool] = {}
    for dataset in plan.required_datasets:
        report = reports[dataset]
        validate_aggregate_public_output(report)
        if report.get("dataset") != dataset:
            raise EvidenceContractError("dataset report key and payload differ")
        report_hashes[dataset] = _canonical_sha256(report)
        accuracy = report.get("mandatory_accuracy_no_harm_gate")
        if not isinstance(accuracy, Mapping):
            raise EvidenceContractError("dataset report lacks the mandatory accuracy gate")
        accuracy_passes[dataset] = bool(
            accuracy.get("all_predeclared_contrasts_passed") is True
        )
    output: dict[str, object] = {
        "schema_version": CROSS_DATASET_INDEX_SCHEMA,
        "status": "both_required_open_role_reports_linked_not_confirmatory_or_publishable",
        "claim_boundary": (
            "dual_dataset_open_role_index_only; no joint paired replicate inference; "
            "method_success remains unauthorized"
        ),
        "plan_id": plan.plan_id,
        "plan_sha256": plan.plan_sha256,
        "analysis_config_sha256": plan.analysis_config_sha256,
        "required_datasets": list(plan.required_datasets),
        "both_required_datasets_present": True,
        "single_dataset_can_trigger_method_success": False,
        "dataset_report_sha256": report_hashes,
        "dataset_accuracy_no_harm_passed": accuracy_passes,
        "both_accuracy_no_harm_gates_passed": bool(all(accuracy_passes.values())),
        "method_success_authorized": False,
        "required_next_layer": (
            "private_hash_bound_cross_dataset_paired_replicate_aggregator_after_authorized_unseal"
        ),
    }
    _validate_json_safe_no_paths(output)
    return output


def load_holm_family_from_confirmatory_config(
    path: str | Path,
    *,
    method_bindings: Mapping[str, tuple[str, str]],
    expected_sha256: str,
) -> PredeclaredHolmFamily:
    """Load and hash-bind the predeclared family to an immutable config file.

    ``method_bindings`` maps every hypothesis id to the concrete candidate and
    reference method ids produced by the frozen run.  The set must be exact;
    no result-aware family extension or omission is accepted.
    """

    config_path = Path(path)
    observed_hash = _file_sha256(config_path)
    if observed_hash != _require_sha256(expected_sha256, "expected_sha256"):
        raise EvidenceContractError("confirmatory analysis config hash differs")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceContractError(f"cannot read confirmatory analysis config: {error}") from error
    holm = payload.get("holm_family") if isinstance(payload, Mapping) else None
    if not isinstance(holm, Mapping) or holm.get("method") != "holm_bonferroni":
        raise EvidenceContractError("confirmatory config lacks a Holm family")
    rows = holm.get("hypotheses")
    if not isinstance(rows, list) or not rows:
        raise EvidenceContractError("confirmatory config Holm hypotheses are missing")
    ids = [str(row.get("id", "")) for row in rows if isinstance(row, Mapping)]
    if set(method_bindings) != set(ids) or len(ids) != len(rows):
        raise EvidenceContractError("method bindings must cover the exact frozen Holm family")
    hypotheses = []
    for row in rows:
        assert isinstance(row, Mapping)
        hypothesis_id = str(row["id"])
        candidate, reference = method_bindings[hypothesis_id]
        hypotheses.append(
            HolmHypothesis(
                hypothesis_id=hypothesis_id,
                candidate=str(candidate),
                reference=str(reference),
                metric=str(row.get("metric")),
                alternative=str(row.get("alternative")),
            )
        )
    return predeclare_holm_family(
        family_id=str(holm.get("family_id", "")),
        alpha=float(holm.get("familywise_alpha", math.nan)),
        hypotheses=hypotheses,
        analysis_config_sha256=observed_hash,
    )


def load_accuracy_no_harm_gate_from_confirmatory_config(
    path: str | Path,
    *,
    method_bindings: Mapping[str, tuple[str, str]],
    expected_sha256: str,
) -> PredeclaredAccuracyNoHarmGate:
    """Load the exact accuracy no-harm family from the immutable config.

    ``method_bindings`` maps the two logical contrast ids to the concrete
    candidate/reference method ids produced by a frozen run.  Requiring an
    exact id set prevents a caller from silently dropping the harder strong-
    baseline contrast after observing results.
    """

    config_path = Path(path)
    observed_hash = _file_sha256(config_path)
    if observed_hash != _require_sha256(expected_sha256, "expected_sha256"):
        raise EvidenceContractError("confirmatory analysis config hash differs")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceContractError(
            f"cannot read confirmatory analysis config: {error}"
        ) from error
    gates = payload.get("effect_and_safety_gates") if isinstance(payload, Mapping) else None
    accuracy = gates.get("accuracy_no_harm") if isinstance(gates, Mapping) else None
    if not isinstance(accuracy, Mapping):
        raise EvidenceContractError("confirmatory config lacks accuracy no-harm")
    rows = accuracy.get("contrasts")
    if not isinstance(rows, list) or not rows:
        raise EvidenceContractError("confirmatory accuracy contrasts are missing")
    ids = [str(row.get("id", "")) for row in rows if isinstance(row, Mapping)]
    expected_ids = {
        "A1_accuracy_vs_current",
        "A2_accuracy_vs_frozen_reference",
    }
    if (
        set(ids) != expected_ids
        or set(method_bindings) != expected_ids
        or len(ids) != len(rows)
    ):
        raise EvidenceContractError(
            "method bindings must cover the exact frozen accuracy family"
        )
    contrasts: list[AccuracyNoHarmContrast] = []
    for row in rows:
        assert isinstance(row, Mapping)
        contrast_id = str(row["id"])
        candidate, reference = method_bindings[contrast_id]
        contrasts.append(
            AccuracyNoHarmContrast(
                contrast_id=contrast_id,
                candidate=str(candidate),
                reference=str(reference),
            )
        )
    return predeclare_accuracy_no_harm_gate(
        gate_id=str(accuracy.get("gate_id", "")),
        contrasts=contrasts,
        analysis_config_sha256=observed_hash,
        minimum_point_difference=float(
            accuracy.get("minimum_point_difference", math.nan)
        ),
        minimum_ci95_lower=float(accuracy.get("minimum_ci95_lower", math.nan)),
    )
