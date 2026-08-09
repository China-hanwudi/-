"""Leakage-closed EmotionTalk runner for the causal multimodal backbone.

The runner is deliberately separate from the earlier linear probability cache.
It trains the :class:`~hva_affect.causal_multimodal_backbone.CausalMultimodalBackbone`
from fold-local text representations and frozen WavLM/DINOv2 features, then
recomputes current/all-history and S/S+h/T/T-h probabilities.  Consequently,
all utility targets exported by this module are *backbone-relative*.

Only the frozen ``base_and_utility_fit`` and ``model_selection`` roles are
materialised.  The outer held-out group is never used by the text encoder,
optimizer, or early stopping; a second group-disjoint split inside the outer
training set supplies early stopping.  Model-selection rows are inference-only.

Checkpoints, processors, row-level probabilities, and utilities are private
artifacts.  The public JSON contains aggregate metrics and content hashes only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
from importlib import metadata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import joblib
import numpy as np
import torch
import sklearn
import scipy
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from .bidirectional_emotion_utility import (
    BidirectionalCoalitionTask,
    bidirectional_utility_targets,
    sample_bidirectional_coalition_tasks,
)
from .causal_multimodal_backbone import CausalBackboneConfig, CausalMultimodalBackbone
from .data_contract import ContractError, sha256_file, write_json_atomic
from .emotiontalk_role_sidecar import load_emotiontalk_role_sidecars
from .meld_text_pilot import NLL_PROBABILITY_FLOOR, true_class_loss


FIT_ROLE = "base_and_utility_fit"
SELECTION_ROLE = "model_selection"
OPEN_ROLES = (FIT_ROLE, SELECTION_ROLE)
FROZEN_ROLE_RANGES = {
    FIT_ROLE: [0, 64],
    SELECTION_ROLE: [65, 79],
    "calibration": [80, 89],
    "internal_holdout_sealed": [90, 99],
}
EXPECTED_SEEDS = (17, 29, 43, 71, 101)
UTILITY_CONTEXT_NAMES = ("s", "s_plus_candidate", "t", "t_minus_candidate")
ENDPOINT_CONTEXT_NAMES = ("current_only", "all_history")
PRIVATE_CACHE_SCHEMA = "carma_causal_backbone_open_role_private_v2"
PUBLIC_REPORT_SCHEMA = "carma_causal_backbone_open_role_public_v2"


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


def _indices_sha256(indices: Sequence[int]) -> str:
    return _array_sha256(np.asarray(indices, dtype=np.int64))


def _require_sha256(value: object, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ContractError(f"{field} must be a SHA-256 digest")
    return digest


def _atomic_joblib_dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, path)


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _torch_load_local(path: Path) -> dict:
    # These are locally generated, hash-bound private checkpoints.  Explicitly
    # passing weights_only=False avoids version-dependent defaults in PyTorch.
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ContractError(f"malformed local checkpoint: {path.name}")
    return value


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


@dataclass(frozen=True)
class BackboneRunConfig:
    """Training/runtime settings that are independent of model dimensions."""

    outer_folds: int = 5
    inner_validation_fraction: float = 0.15
    max_epochs: int = 32
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1.0e-4
    batch_size: int = 12
    inference_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-2
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.05
    subset_dropout_probability: float = 0.20
    max_history_items: int = 128
    use_amp: bool = True
    max_cuda_memory_mib: int = 7800
    text_analyzer: str = "char"
    text_ngram_min: int = 2
    text_ngram_max: int = 5
    text_min_df: int = 2
    text_max_df: float = 0.995
    text_max_features: int = 50_000
    text_sublinear_tf: bool = True
    text_svd_n_iter: int = 7

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "BackboneRunConfig":
        raw = payload.get("training_runner", payload)
        if not isinstance(raw, Mapping):
            raise ContractError("training_runner must be a mapping")
        text = raw.get("text_encoder", {})
        if text is not None and not isinstance(text, Mapping):
            raise ContractError("training_runner.text_encoder must be a mapping")
        merged = dict(raw)
        merged.pop("text_encoder", None)
        if isinstance(text, Mapping):
            aliases = {
                "analyzer": "text_analyzer",
                "ngram_min": "text_ngram_min",
                "ngram_max": "text_ngram_max",
                "min_df": "text_min_df",
                "max_df": "text_max_df",
                "max_features": "text_max_features",
                "sublinear_tf": "text_sublinear_tf",
                "svd_n_iter": "text_svd_n_iter",
            }
            for source, target in aliases.items():
                if source in text:
                    merged[target] = text[source]
        allowed = set(cls.__dataclass_fields__)
        result = cls(**{name: value for name, value in merged.items() if name in allowed})
        result.validate()
        return result

    def validate(self) -> None:
        if self.outer_folds < 2:
            raise ContractError("outer_folds must be at least two")
        if not 0.0 < self.inner_validation_fraction < 0.5:
            raise ContractError("inner_validation_fraction must be in (0, 0.5)")
        positive = {
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "batch_size": self.batch_size,
            "inference_batch_size": self.inference_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "gradient_clip_norm": self.gradient_clip_norm,
            "max_history_items": self.max_history_items,
            "max_cuda_memory_mib": self.max_cuda_memory_mib,
            "text_ngram_min": self.text_ngram_min,
            "text_ngram_max": self.text_ngram_max,
            "text_max_features": self.text_max_features,
            "text_svd_n_iter": self.text_svd_n_iter,
        }
        if any(float(value) <= 0 for value in positive.values()):
            raise ContractError("runner positive-valued setting is non-positive")
        if self.text_ngram_min > self.text_ngram_max:
            raise ContractError("text ngram range is reversed")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ContractError("label_smoothing must be in [0, 1)")
        if not 0.0 <= self.subset_dropout_probability <= 1.0:
            raise ContractError("subset_dropout_probability must be in [0, 1]")


@dataclass
class OpenRoleCorpus:
    """Private, row-aligned inputs for the two currently open train roles."""

    keys: np.ndarray
    texts: tuple[str, ...]
    audio: np.ndarray
    video: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    roles: np.ndarray
    buckets: np.ndarray
    speaker_ids: np.ndarray
    turn_ids: np.ndarray
    histories: tuple[tuple[int, ...], ...]
    protocol_row_ids: np.ndarray | None = None
    speaker_identity: np.ndarray | None = None
    speaker_mapping_sha256: str = ""
    label_access_mode: str = "synthetic_or_prevalidated"

    def validate(self, model_config: CausalBackboneConfig) -> None:
        rows = len(self.keys)
        aligned = (
            len(self.texts),
            len(self.audio),
            len(self.video),
            len(self.labels),
            len(self.groups),
            len(self.roles),
            len(self.buckets),
            len(self.speaker_ids),
            len(self.turn_ids),
            len(self.histories),
        )
        if rows == 0 or any(value != rows for value in aligned):
            raise ContractError("open-role corpus arrays are empty or misaligned")
        if len(set(self.keys.astype(str))) != rows:
            raise ContractError("open-role corpus contains duplicate keys")
        if self.audio.shape != (rows, model_config.audio_dim):
            raise ContractError(
                f"audio features must have shape ({rows}, {model_config.audio_dim})"
            )
        if self.video.shape != (rows, model_config.video_dim):
            raise ContractError(
                f"video features must have shape ({rows}, {model_config.video_dim})"
            )
        if not np.isfinite(self.audio).all() or not np.isfinite(self.video).all():
            raise ContractError("open-role media features contain non-finite values")
        if np.any((self.labels < 0) | (self.labels >= model_config.num_classes)):
            raise ContractError("open-role label is outside the configured class range")
        if set(self.roles.astype(str)) - set(OPEN_ROLES):
            raise ContractError("sealed role entered the materialised corpus")
        if np.any((self.buckets < 0) | (self.buckets > 79)):
            raise ContractError("sealed role bucket entered the materialised corpus")
        if np.any((self.speaker_ids < 0) | (self.speaker_ids >= model_config.num_speakers)):
            raise ContractError("speaker id is outside backbone vocabulary")
        if np.any((self.turn_ids < 0) | (self.turn_ids >= model_config.max_turns)):
            raise ContractError("turn id is outside backbone vocabulary")
        speaker_identity = (
            np.asarray(self.speaker_ids, dtype=str)
            if self.speaker_identity is None
            else np.asarray(self.speaker_identity, dtype=str)
        )
        if speaker_identity.shape != (rows,):
            raise ContractError("speaker identity must be row-aligned")
        if self.speaker_mapping_sha256:
            _require_sha256(self.speaker_mapping_sha256, "speaker_mapping_sha256")
        protocol_rows = (
            np.arange(rows, dtype=np.int64)
            if self.protocol_row_ids is None
            else np.asarray(self.protocol_row_ids, dtype=np.int64)
        )
        if protocol_rows.shape != (rows,) or len(set(protocol_rows.tolist())) != rows:
            raise ContractError("protocol row ids must be unique and row-aligned")
        if np.any(protocol_rows < 0):
            raise ContractError("protocol row ids must be non-negative")
        for query, history_raw in enumerate(self.histories):
            history = tuple(int(value) for value in history_raw)
            if len(history) != len(set(history)):
                raise ContractError("history contains duplicate row indices")
            for candidate in history:
                if not 0 <= candidate < rows:
                    raise ContractError("history row index is outside the corpus")
                if str(self.groups[candidate]) != str(self.groups[query]):
                    raise ContractError("history crosses a group boundary")
                if str(self.roles[candidate]) != str(self.roles[query]):
                    raise ContractError("history crosses a frozen role boundary")
                if str(speaker_identity[candidate]) != str(speaker_identity[query]):
                    raise ContractError("history crosses a speaker boundary")
                if int(self.turn_ids[candidate]) >= int(self.turn_ids[query]):
                    raise ContractError("history contains a current or future turn")

    def role_indices(self, role: str) -> np.ndarray:
        if role not in OPEN_ROLES:
            raise ContractError(f"role is not open: {role}")
        return np.flatnonzero(self.roles.astype(str) == role).astype(np.int64)


@dataclass(frozen=True)
class CrossfitSplit:
    fold: int
    inner_train_indices: np.ndarray
    inner_validation_indices: np.ndarray
    outer_heldout_indices: np.ndarray


@dataclass
class FoldTextProcessor:
    """Fold-local TF-IDF + SVD with deterministic zero-padding at tiny rank."""

    vectorizer: TfidfVectorizer
    svd: TruncatedSVD | None
    output_dim: int
    effective_dim: int
    fit_indices_sha256: str

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(tuple(str(value) for value in texts))
        if self.svd is None:
            reduced = matrix.toarray().astype(np.float32, copy=False)
        else:
            reduced = self.svd.transform(matrix).astype(np.float32, copy=False)
        if reduced.shape[1] > self.output_dim:
            raise ContractError("fold-local text projection exceeded its frozen dimension")
        result = np.zeros((len(texts), self.output_dim), dtype=np.float32)
        result[:, : reduced.shape[1]] = reduced
        if not np.isfinite(result).all():
            raise ContractError("fold-local text projection contains non-finite values")
        return result


def fit_fold_text_processor(
    texts: Sequence[str],
    fit_indices: Sequence[int],
    *,
    output_dim: int,
    config: BackboneRunConfig,
    seed: int,
) -> FoldTextProcessor:
    fit = np.asarray(fit_indices, dtype=np.int64)
    if fit.ndim != 1 or len(fit) < 2 or len(set(fit.tolist())) != len(fit):
        raise ContractError("fold-local text processor requires unique training rows")
    if np.any((fit < 0) | (fit >= len(texts))):
        raise ContractError("fold-local text fit row is outside the corpus")
    vectorizer = TfidfVectorizer(
        analyzer=config.text_analyzer,
        ngram_range=(config.text_ngram_min, config.text_ngram_max),
        min_df=config.text_min_df,
        max_df=config.text_max_df,
        max_features=config.text_max_features,
        sublinear_tf=config.text_sublinear_tf,
        dtype=np.float32,
    )
    try:
        matrix = vectorizer.fit_transform([str(texts[index]) for index in fit])
    except ValueError as error:
        raise ContractError(f"fold-local text vocabulary could not be fit: {error}") from error
    if matrix.shape[1] < 1:
        raise ContractError("fold-local text vocabulary is empty")
    # TruncatedSVD requires at least two columns.  Production data use all 256
    # components; deterministic zero-padding only exists for synthetic/small
    # folds whose algebraic rank is necessarily lower.
    effective = min(int(output_dim), max(1, int(matrix.shape[1]) - 1))
    if matrix.shape[1] == 1:
        svd = None
        effective = 1
    else:
        svd = TruncatedSVD(
            n_components=effective,
            n_iter=config.text_svd_n_iter,
            algorithm="randomized",
            random_state=int(seed),
        )
        svd.fit(matrix)
    return FoldTextProcessor(
        vectorizer=vectorizer,
        svd=svd,
        output_dim=int(output_dim),
        effective_dim=int(effective),
        fit_indices_sha256=_indices_sha256(fit),
    )


def make_crossfit_splits(
    corpus: OpenRoleCorpus,
    *,
    outer_folds: int,
    validation_fraction: float,
    seed: int,
) -> tuple[CrossfitSplit, ...]:
    """Create label-free outer and inner group-disjoint splits."""

    fit_indices = corpus.role_indices(FIT_ROLE)
    fit_groups = corpus.groups[fit_indices].astype(str)
    unique_groups = sorted(set(fit_groups))
    if len(unique_groups) < outer_folds:
        raise ContractError("too few fit-role groups for requested outer cross-fitting")
    splitter = GroupKFold(n_splits=int(outer_folds))
    result: list[CrossfitSplit] = []
    for fold, (outer_train_local, held_local) in enumerate(
        splitter.split(fit_indices, groups=fit_groups)
    ):
        outer_train = fit_indices[np.asarray(outer_train_local, dtype=np.int64)]
        heldout = fit_indices[np.asarray(held_local, dtype=np.int64)]
        outer_groups = sorted(set(corpus.groups[outer_train].astype(str)))
        if len(outer_groups) < 2:
            raise ContractError("outer training fold has too few groups for early stopping")
        ordered = sorted(
            outer_groups,
            key=lambda group: hashlib.sha256(
                f"inner\x1f{seed}\x1f{fold}\x1f{group}".encode("utf-8")
            ).digest(),
        )
        validation_count = min(
            len(ordered) - 1,
            max(1, int(round(validation_fraction * len(ordered)))),
        )
        validation_groups = set(ordered[:validation_count])
        inner_validation = outer_train[
            np.asarray(
                [str(value) in validation_groups for value in corpus.groups[outer_train]],
                dtype=bool,
            )
        ]
        inner_train = outer_train[
            np.asarray(
                [str(value) not in validation_groups for value in corpus.groups[outer_train]],
                dtype=bool,
            )
        ]
        partitions = (inner_train, inner_validation, heldout)
        group_sets = [set(corpus.groups[value].astype(str)) for value in partitions]
        if any(group_sets[left] & group_sets[right] for left in range(3) for right in range(left + 1, 3)):
            raise ContractError("cross-fit partitions share a group")
        if any(set(corpus.roles[value].astype(str)) != {FIT_ROLE} for value in partitions):
            raise ContractError("non-fit role entered cross-fit training or heldout partitions")
        result.append(CrossfitSplit(fold, inner_train, inner_validation, heldout))
    covered = np.concatenate([value.outer_heldout_indices for value in result])
    if set(covered.tolist()) != set(fit_indices.tolist()) or len(covered) != len(fit_indices):
        raise ContractError("outer cross-fitting did not cover every fit row exactly once")
    return tuple(result)


def _validate_context(corpus: OpenRoleCorpus, query: int, context: Sequence[int]) -> tuple[int, ...]:
    query = int(query)
    if not 0 <= query < len(corpus.keys):
        raise ContractError("query row is outside the open corpus")
    selected = tuple(int(value) for value in context)
    if len(selected) != len(set(selected)):
        raise ContractError("selected context contains duplicate rows")
    allowed = set(corpus.histories[query])
    if not set(selected).issubset(allowed):
        raise ContractError("selected context contains non-history/current/future/cross-role rows")
    return tuple(sorted(selected, key=lambda value: (int(corpus.turn_ids[value]), value)))


def pack_query_contexts(
    corpus: OpenRoleCorpus,
    text_features: np.ndarray,
    query_indices: Sequence[int],
    contexts: Sequence[Sequence[int]],
    *,
    max_history_items: int,
) -> dict[str, torch.Tensor]:
    """Pack exactly one selected history set for each query.

    Returning one row per query-context pair is the canonical one-query,
    one-prediction interface.  Current-only, all-history, and selected-policy
    predictions are all special cases of this function.
    """

    queries = np.asarray(query_indices, dtype=np.int64)
    if queries.ndim != 1 or len(queries) != len(contexts) or len(queries) == 0:
        raise ContractError("query indices and contexts must be non-empty and aligned")
    if text_features.shape[0] != len(corpus.keys):
        raise ContractError("text features are not row-aligned with the corpus")
    selected_contexts = [
        _validate_context(corpus, int(query), context)
        for query, context in zip(queries, contexts, strict=True)
    ]
    if any(len(context) > max_history_items for context in selected_contexts):
        raise ContractError(
            "history exceeds max_history_items; overflow is fail-closed, not silently truncated"
        )
    lengths = [len(context) + 1 for context in selected_contexts]
    length = max(lengths)
    batch = len(queries)
    text = np.zeros((batch, length, text_features.shape[1]), dtype=np.float32)
    audio = np.zeros((batch, length, corpus.audio.shape[1]), dtype=np.float32)
    video = np.zeros((batch, length, corpus.video.shape[1]), dtype=np.float32)
    speakers = np.zeros((batch, length), dtype=np.int64)
    turns = np.zeros((batch, length), dtype=np.int64)
    valid = np.zeros((batch, length), dtype=bool)
    history_mask = np.zeros((batch, length), dtype=bool)
    packed_query = np.empty(batch, dtype=np.int64)
    for row, (query, context) in enumerate(zip(queries, selected_contexts, strict=True)):
        source = np.asarray((*context, int(query)), dtype=np.int64)
        size = len(source)
        text[row, :size] = text_features[source]
        audio[row, :size] = corpus.audio[source]
        video[row, :size] = corpus.video[source]
        speakers[row, :size] = corpus.speaker_ids[source]
        turns[row, :size] = corpus.turn_ids[source]
        valid[row, :size] = True
        history_mask[row, : size - 1] = True
        packed_query[row] = size - 1
    return {
        "text_features": torch.from_numpy(text),
        "audio_features": torch.from_numpy(audio),
        "video_features": torch.from_numpy(video),
        "speaker_ids": torch.from_numpy(speakers),
        "turn_ids": torch.from_numpy(turns),
        "valid_mask": torch.from_numpy(valid),
        "history_mask": torch.from_numpy(history_mask),
        "query_indices": torch.from_numpy(packed_query),
    }


def _histories_sha256(histories: Sequence[Sequence[int]]) -> str:
    return _canonical_sha256([list(map(int, values)) for values in histories])


def _corpus_contract_sha256(corpus: OpenRoleCorpus) -> str:
    return _canonical_sha256(
        {
            "keys": _array_sha256(corpus.keys.astype(str)),
            "texts": _canonical_sha256(list(corpus.texts)),
            "labels": _array_sha256(corpus.labels),
            "groups": _array_sha256(corpus.groups.astype(str)),
            "roles": _array_sha256(corpus.roles.astype(str)),
            "buckets": _array_sha256(corpus.buckets),
            "speaker_ids": _array_sha256(corpus.speaker_ids),
            "speaker_identity": _array_sha256(
                np.asarray(corpus.speaker_identity if corpus.speaker_identity is not None else corpus.speaker_ids).astype(str)
            ),
            "speaker_mapping_sha256": corpus.speaker_mapping_sha256,
            "turn_ids": _array_sha256(corpus.turn_ids),
            "protocol_row_ids": _array_sha256(
                np.arange(len(corpus.keys), dtype=np.int64)
                if corpus.protocol_row_ids is None
                else corpus.protocol_row_ids
            ),
            "histories": _histories_sha256(corpus.histories),
            "audio": _array_sha256(corpus.audio),
            "video": _array_sha256(corpus.video),
        }
    )


def _role_assignment_sha256(corpus: OpenRoleCorpus) -> str:
    return _canonical_sha256(
        [
            [str(key), str(role), int(bucket)]
            for key, role, bucket in zip(
                corpus.keys, corpus.roles, corpus.buckets, strict=True
            )
        ]
    )


def _provenance_attestation_payload(
    provenance: "VerifiedCorpusProvenance",
) -> dict[str, object]:
    return {
        "dataset_id": provenance.dataset_id,
        "manifest_schema": provenance.manifest_schema,
        "manifest_status": provenance.manifest_status,
        "manifest_sha256": provenance.manifest_sha256,
        "source_hashes": dict(sorted(provenance.source_hashes.items())),
        "label_order": list(provenance.label_order),
        "role_rows": dict(sorted(provenance.role_rows.items())),
        "audio_dim": provenance.audio_dim,
        "video_dim": provenance.video_dim,
        "role_assignment_sha256": provenance.role_assignment_sha256,
        "speaker_mapping_sha256": provenance.speaker_mapping_sha256,
        "corpus_contract_sha256": provenance.corpus_contract_sha256,
        "verification_origin": provenance.verification_origin,
        "strict_role_feature_sidecars": provenance.strict_role_feature_sidecars,
        "strict_role_label_sidecars": provenance.strict_role_label_sidecars,
        "sealed_role_arrays_opened": provenance.sealed_role_arrays_opened,
        "validation_or_test_opened": provenance.validation_or_test_opened,
    }


@dataclass(frozen=True)
class VerifiedCorpusProvenance:
    dataset_id: str
    manifest_schema: str
    manifest_status: str
    manifest_sha256: str
    source_hashes: Mapping[str, str]
    label_order: tuple[str, ...]
    role_rows: Mapping[str, int]
    audio_dim: int
    video_dim: int
    role_assignment_sha256: str
    speaker_mapping_sha256: str
    corpus_contract_sha256: str
    verification_origin: str
    verifier_attestation_sha256: str
    strict_role_feature_sidecars: bool = True
    strict_role_label_sidecars: bool = True
    sealed_role_arrays_opened: bool = False
    validation_or_test_opened: bool = False

    def validate(self, corpus: OpenRoleCorpus, model_config: CausalBackboneConfig) -> None:
        if self.verification_origin not in {
            "emotiontalk_manifest_v2",
            "meld_manifest_v2",
            "synthetic_contract_test",
        }:
            raise ContractError("provenance was not produced by a recognised verifier")
        if not self.strict_role_feature_sidecars or not self.strict_role_label_sidecars:
            raise ContractError("strict physical role sidecars are required")
        if self.sealed_role_arrays_opened or self.validation_or_test_opened:
            raise ContractError("provenance attests sealed/dev/test access")
        if not self.dataset_id or any(character in self.dataset_id for character in "\\/\x00"):
            raise ContractError("provenance dataset id is empty or unsafe")
        if not self.manifest_schema or not self.manifest_status:
            raise ContractError("provenance manifest identity is incomplete")
        _require_sha256(self.manifest_sha256, "manifest_sha256")
        _require_sha256(self.role_assignment_sha256, "role_assignment_sha256")
        _require_sha256(self.speaker_mapping_sha256, "speaker_mapping_sha256")
        _require_sha256(self.corpus_contract_sha256, "corpus_contract_sha256")
        _require_sha256(self.verifier_attestation_sha256, "verifier_attestation_sha256")
        if not self.source_hashes:
            raise ContractError("provenance source hashes are empty")
        for name, value in self.source_hashes.items():
            if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in name):
                raise ContractError("provenance source hash name is unsafe")
            _require_sha256(value, f"source_hashes.{name}")
        if self.source_hashes.get("sidecar_manifest") != self.manifest_sha256:
            raise ContractError("provenance manifest hash is not bound to source hashes")
        if tuple(self.label_order) == () or len(self.label_order) != model_config.num_classes:
            raise ContractError("dataset label order does not match model classes")
        if len(set(self.label_order)) != len(self.label_order):
            raise ContractError("dataset label order contains duplicate classes")
        if (
            model_config.auxiliary_vad_weight > 0.0
            and tuple(model_config.emotion_label_order) != tuple(self.label_order)
        ):
            raise ContractError(
                "VAD supervision label order differs from the verified dataset order"
            )
        observed_rows = {role: int(np.sum(corpus.roles.astype(str) == role)) for role in OPEN_ROLES}
        if dict(self.role_rows) != observed_rows:
            raise ContractError("provenance role rows differ from materialised corpus")
        if (self.audio_dim, self.video_dim) != (model_config.audio_dim, model_config.video_dim):
            raise ContractError("provenance modality dimensions differ from model")
        if self.speaker_mapping_sha256 != corpus.speaker_mapping_sha256:
            raise ContractError("provenance speaker mapping differs from corpus")
        if self.role_assignment_sha256 != _role_assignment_sha256(corpus):
            raise ContractError("provenance role assignment differs from corpus")
        if self.corpus_contract_sha256 != _corpus_contract_sha256(corpus):
            raise ContractError("verified corpus contract hash changed")
        expected_attestation = _canonical_sha256(_provenance_attestation_payload(self))
        if self.verifier_attestation_sha256 != expected_attestation:
            raise ContractError("verified provenance fields were changed after verification")


def create_verified_corpus_provenance(
    *,
    dataset_id: str,
    manifest_schema: str,
    manifest_status: str,
    manifest_sha256: str,
    source_hashes: Mapping[str, str],
    label_order: Sequence[str],
    role_rows: Mapping[str, int],
    audio_dim: int,
    video_dim: int,
    role_assignment_sha256: str,
    speaker_mapping_sha256: str,
    corpus_contract_sha256: str,
    verification_origin: str,
) -> VerifiedCorpusProvenance:
    """Create a tamper-evident verifier result for a manifest-loaded corpus."""

    provisional = VerifiedCorpusProvenance(
        dataset_id=str(dataset_id),
        manifest_schema=str(manifest_schema),
        manifest_status=str(manifest_status),
        manifest_sha256=str(manifest_sha256),
        source_hashes={str(name): str(value) for name, value in source_hashes.items()},
        label_order=tuple(str(value) for value in label_order),
        role_rows={str(name): int(value) for name, value in role_rows.items()},
        audio_dim=int(audio_dim),
        video_dim=int(video_dim),
        role_assignment_sha256=str(role_assignment_sha256),
        speaker_mapping_sha256=str(speaker_mapping_sha256),
        corpus_contract_sha256=str(corpus_contract_sha256),
        verification_origin=str(verification_origin),
        verifier_attestation_sha256="0" * 64,
    )
    return VerifiedCorpusProvenance(
        **{
            **provisional.__dict__,
            "verifier_attestation_sha256": _canonical_sha256(
                _provenance_attestation_payload(provisional)
            ),
        }
    )


def _history_indices(
    groups: np.ndarray,
    speaker_identity: np.ndarray,
    turns: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    histories: list[tuple[int, ...]] = []
    for query in range(len(groups)):
        values = np.flatnonzero(
            (groups == groups[query])
            & (speaker_identity == speaker_identity[query])
            & (turns < turns[query])
        )
        histories.append(tuple(sorted(values.tolist(), key=lambda value: (int(turns[value]), value))))
    return tuple(histories)


def load_emotiontalk_open_role_corpus(
    *,
    sidecar_dir: Path,
    manifest_path: Path,
    model_config: CausalBackboneConfig,
) -> tuple[OpenRoleCorpus, VerifiedCorpusProvenance]:
    """Load only physical fit/selection sidecars verified by the manifest."""

    role_arrays, manifest = load_emotiontalk_role_sidecars(
        sidecar_dir=sidecar_dir,
        manifest_path=manifest_path,
    )
    fit_speakers = sorted(set(role_arrays[FIT_ROLE].speaker_tokens.astype(str)))
    # 0 is a frozen OOV id.  Selection tokens never influence this mapping.
    speaker_mapping = {value: index + 1 for index, value in enumerate(fit_speakers)}
    if len(speaker_mapping) + 1 > model_config.num_speakers:
        raise ContractError("fit-only EmotionTalk speaker vocabulary exceeds model config")
    speaker_mapping_sha = _canonical_sha256(
        {"oov": 0, "fit_mapping": [[value, speaker_mapping[value]] for value in fit_speakers]}
    )
    values = [role_arrays[role] for role in OPEN_ROLES]
    keys = np.concatenate([value.row_hashes for value in values])
    groups = np.concatenate([value.group_hashes for value in values])
    speaker_tokens = np.concatenate([value.speaker_tokens for value in values]).astype(str)
    roles = np.concatenate(
        [np.asarray([value.role] * len(value.labels)) for value in values]
    )
    ordering = np.argsort(
        np.concatenate([value.protocol_row_ids for value in values]), kind="stable"
    )
    def combined(name: str) -> np.ndarray:
        return np.concatenate([np.asarray(getattr(value, name)) for value in values], axis=0)[ordering]

    keys = keys[ordering]
    groups = groups[ordering]
    speaker_tokens = speaker_tokens[ordering]
    roles = roles[ordering]
    labels = combined("labels").astype(np.int64)
    buckets = combined("role_buckets").astype(np.int16)
    turns = combined("turn_ids").astype(np.int64)
    protocol_rows = combined("protocol_row_ids").astype(np.int64)
    texts = combined("texts").astype(str)
    audio = combined("audio").astype(np.float32)
    video = combined("video").astype(np.float32)
    speaker_ids = np.asarray([speaker_mapping.get(value, 0) for value in speaker_tokens], dtype=np.int64)
    speaker_identity = np.asarray(
        [hashlib.sha256(f"speaker\x1f{value}".encode()).hexdigest() for value in speaker_tokens]
    )
    histories = _history_indices(groups, speaker_identity, turns)
    corpus = OpenRoleCorpus(
        keys=keys,
        texts=tuple(texts),
        audio=audio,
        video=video,
        labels=labels,
        groups=groups,
        roles=roles,
        buckets=buckets,
        speaker_ids=speaker_ids,
        turn_ids=turns,
        histories=histories,
        protocol_row_ids=protocol_rows,
        speaker_identity=speaker_identity,
        speaker_mapping_sha256=speaker_mapping_sha,
        label_access_mode="strict_physical_emotiontalk_fit_selection_feature_label_sidecars",
    )
    corpus.validate(model_config)
    role_assignment_sha = _role_assignment_sha256(corpus)
    source_hashes = {
        "sidecar_manifest": sha256_file(manifest_path),
        "trusted_source_label_archive": manifest["source_contract"]["label_archive"],
        "trusted_source_media_features": manifest["source_contract"]["media_features"],
        "trusted_source_transcription": manifest["source_contract"]["transcription"],
    }
    for role, value in role_arrays.items():
        source_hashes[f"{role}_features"] = value.feature_sha256
        source_hashes[f"{role}_labels"] = value.label_sha256
    provenance = create_verified_corpus_provenance(
        dataset_id="EmotionTalk",
        manifest_schema=str(manifest["schema_version"]),
        manifest_status=str(manifest["status"]),
        manifest_sha256=sha256_file(manifest_path),
        source_hashes=source_hashes,
        label_order=tuple(str(value) for value in manifest["label_order"]),
        role_rows={role: int(manifest["roles"][role]["rows"]) for role in OPEN_ROLES},
        audio_dim=int(manifest["roles"][FIT_ROLE]["audio_dimension"]),
        video_dim=int(manifest["roles"][FIT_ROLE]["video_dimension"]),
        role_assignment_sha256=role_assignment_sha,
        speaker_mapping_sha256=speaker_mapping_sha,
        corpus_contract_sha256=_corpus_contract_sha256(corpus),
        verification_origin="emotiontalk_manifest_v2",
    )
    provenance.validate(corpus, model_config)
    return corpus, provenance


def _to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device=device, non_blocking=device.type == "cuda")
        for name, value in batch.items()
    }


def _model_state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(device)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, object]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise ContractError("checkpoint RNG state schema changed")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = list(state["torch_cuda"])
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ContractError("checkpoint CUDA RNG state cannot be restored on this runtime")
        torch.cuda.set_rng_state_all(cuda_states)


def _internal_code_hashes() -> dict[str, str]:
    names = (
        "emotiontalk_causal_backbone_runner.py",
        "causal_multimodal_backbone.py",
        "causal_affect_relation.py",
        "bidirectional_emotion_utility.py",
        "meld_text_pilot.py",
        "emotiontalk_role_sidecar.py",
        "meld_causal_backbone_loader.py",
    )
    base = Path(__file__).resolve().parent
    result: dict[str, str] = {}
    for name in names:
        path = base / name
        if not path.is_file():
            raise ContractError(f"required code module is missing: {name}")
        result[name] = sha256_file(path)
    return result


def _runtime_environment(device: torch.device) -> dict[str, object]:
    packages = sorted(
        f"{distribution.metadata.get('Name', '<unnamed>')}=={distribution.version}"
        for distribution in metadata.distributions()
    )
    device_record: dict[str, object] = {"type": device.type, "index": device.index}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        device_record.update(
            {
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": [int(properties.major), int(properties.minor)],
            }
        )
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "determinism": {
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "deterministic_algorithms_warn_only": bool(
                getattr(torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False)()
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "device": device_record,
        "installed_packages_sha256": _canonical_sha256(packages),
        "installed_package_count": len(packages),
        "installed_packages": packages,
    }


class PlannedCheckpointInterruption(RuntimeError):
    """Test-only interruption raised immediately after an atomic partial save."""


def _class_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(float)
    weights = np.zeros(num_classes, dtype=np.float32)
    observed = counts > 0
    weights[observed] = 1.0 / np.sqrt(counts[observed])
    if observed.any():
        weights[observed] /= weights[observed].mean()
    return torch.from_numpy(weights).to(device)


def _nll(labels: np.ndarray, probability: np.ndarray) -> float:
    probability = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probability.ndim != 2 or probability.shape[0] != len(labels):
        raise ContractError("NLL labels and probability rows are misaligned")
    if not np.isfinite(probability).all() or np.any(probability < 0):
        raise ContractError("NLL probability is invalid")
    if np.any((labels < 0) | (labels >= probability.shape[1])):
        raise ContractError("NLL label is outside the probability label order")
    if not np.allclose(probability.sum(axis=1), 1.0, rtol=1.0e-5, atol=1.0e-6):
        raise ContractError("NLL probability rows do not sum to one")
    return float(true_class_loss(labels, probability).mean())


def _brier(labels: np.ndarray, probability: np.ndarray) -> float:
    one_hot = np.eye(probability.shape[1], dtype=np.float64)[np.asarray(labels, dtype=int)]
    return float(np.mean(np.sum((np.asarray(probability) - one_hot) ** 2, axis=1)))


def classification_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probability.ndim != 2 or probability.shape[0] != len(labels) or probability.shape[1] < 2:
        raise ContractError("classification probability shape changed")
    num_classes = int(probability.shape[1])
    if np.any((labels < 0) | (labels >= num_classes)):
        raise ContractError("classification label is outside the probability label order")
    if not np.isfinite(probability).all() or np.any(probability < 0):
        raise ContractError("classification probability is invalid")
    totals = probability.sum(axis=1)
    if not np.allclose(totals, 1.0, rtol=1.0e-5, atol=1.0e-6):
        raise ContractError("classification probability rows do not sum to one")
    predicted = probability.argmax(axis=1)
    return {
        "macro_f1": float(
            f1_score(
                labels,
                predicted,
                labels=np.arange(num_classes),
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(labels, predicted)),
        "nll": _nll(labels, probability),
        "brier": _brier(labels, probability),
    }


@torch.no_grad()
def predict_one_probability_per_query(
    model: CausalMultimodalBackbone,
    corpus: OpenRoleCorpus,
    text_features: np.ndarray,
    query_indices: Sequence[int],
    selected_contexts: Sequence[Sequence[int]],
    *,
    device: torch.device,
    batch_size: int,
    max_history_items: int,
) -> np.ndarray:
    """Return exactly one seven-class prediction per query/context pair."""

    queries = np.asarray(query_indices, dtype=np.int64)
    if len(queries) != len(selected_contexts):
        raise ContractError("selected contexts are not aligned to queries")
    result = np.empty((len(queries), model.config.num_classes), dtype=np.float32)
    model.eval()
    for start in range(0, len(queries), int(batch_size)):
        stop = min(len(queries), start + int(batch_size))
        packed = pack_query_contexts(
            corpus,
            text_features,
            queries[start:stop],
            selected_contexts[start:stop],
            max_history_items=max_history_items,
        )
        batch = _to_device(packed, device)
        output = model(
            text_features=batch["text_features"],
            audio_features=batch["audio_features"],
            video_features=batch["video_features"],
            speaker_ids=batch["speaker_ids"],
            turn_ids=batch["turn_ids"],
            valid_mask=batch["valid_mask"],
            history_mask=batch["history_mask"],
            query_indices=batch["query_indices"],
        )
        result[start:stop] = output.probabilities.detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("backbone inference produced non-finite probability")
    return result


def predict_current_and_all_history(
    model: CausalMultimodalBackbone,
    corpus: OpenRoleCorpus,
    text_features: np.ndarray,
    query_indices: Sequence[int],
    *,
    device: torch.device,
    batch_size: int,
    max_history_items: int,
) -> np.ndarray:
    queries = np.asarray(query_indices, dtype=np.int64)
    flat_queries = np.repeat(queries, len(ENDPOINT_CONTEXT_NAMES))
    contexts: list[tuple[int, ...]] = []
    for query in queries:
        contexts.extend([tuple(), tuple(corpus.histories[int(query)])])
    probability = predict_one_probability_per_query(
        model,
        corpus,
        text_features,
        flat_queries,
        contexts,
        device=device,
        batch_size=batch_size,
        max_history_items=max_history_items,
    )
    return probability.reshape(len(queries), len(ENDPOINT_CONTEXT_NAMES), model.config.num_classes)


def _task_contexts(task: BidirectionalCoalitionTask) -> tuple[tuple[int, ...], ...]:
    candidate = int(task.candidate_index)
    return (
        tuple(task.addition_context),
        tuple(sorted((*task.addition_context, candidate))),
        tuple(task.deletion_context),
        tuple(value for value in task.deletion_context if int(value) != candidate),
    )


def predict_utility_contexts(
    model: CausalMultimodalBackbone,
    corpus: OpenRoleCorpus,
    text_features: np.ndarray,
    tasks: Sequence[BidirectionalCoalitionTask],
    *,
    device: torch.device,
    batch_size: int,
    max_history_items: int,
) -> np.ndarray:
    if not tasks:
        return np.empty((0, len(UTILITY_CONTEXT_NAMES), model.config.num_classes), dtype=np.float32)
    queries: list[int] = []
    contexts: list[tuple[int, ...]] = []
    for task in tasks:
        queries.extend([int(task.query_index)] * len(UTILITY_CONTEXT_NAMES))
        contexts.extend(_task_contexts(task))
    probability = predict_one_probability_per_query(
        model,
        corpus,
        text_features,
        queries,
        contexts,
        device=device,
        batch_size=batch_size,
        max_history_items=max_history_items,
    )
    return probability.reshape(len(tasks), len(UTILITY_CONTEXT_NAMES), model.config.num_classes)


def _evaluate_early_stopping_nll(
    model: CausalMultimodalBackbone,
    corpus: OpenRoleCorpus,
    text_features: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device,
    config: BackboneRunConfig,
) -> float:
    probability = predict_current_and_all_history(
        model,
        corpus,
        text_features,
        indices,
        device=device,
        batch_size=config.inference_batch_size,
        max_history_items=config.max_history_items,
    )
    return float(
        np.mean(
            [
                _nll(corpus.labels[indices], probability[:, context])
                for context in range(len(ENDPOINT_CONTEXT_NAMES))
            ]
        )
    )


def _processor_payload(
    processor: FoldTextProcessor,
    *,
    identity: str,
) -> dict:
    return {
        "schema_version": "fold_local_text_svd_v1",
        "identity_sha256": identity,
        "processor": processor,
    }


def _load_or_fit_processor(
    path: Path,
    corpus: OpenRoleCorpus,
    split: CrossfitSplit,
    *,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    seed: int,
    source_identity: str,
) -> tuple[FoldTextProcessor, str, bool]:
    identity = _canonical_sha256(
        {
            "source_identity": source_identity,
            "fold": split.fold,
            "seed": int(seed),
            "inner_train_indices_sha256": _indices_sha256(split.inner_train_indices),
            "text_dim": model_config.text_dim,
            "text_settings": {
                name: value
                for name, value in asdict(run_config).items()
                if name.startswith("text_")
            },
        }
    )
    if path.is_file():
        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get("identity_sha256") != identity:
            raise ContractError("existing fold-local processor identity mismatch")
        processor = payload.get("processor")
        if not isinstance(processor, FoldTextProcessor):
            raise ContractError("existing fold-local processor is malformed")
        if processor.fit_indices_sha256 != _indices_sha256(split.inner_train_indices):
            raise ContractError("existing text processor was fit on different rows")
        return processor, sha256_file(path), True
    processor = fit_fold_text_processor(
        corpus.texts,
        split.inner_train_indices,
        output_dim=model_config.text_dim,
        config=run_config,
        seed=int(seed),
    )
    _atomic_joblib_dump(_processor_payload(processor, identity=identity), path)
    return processor, sha256_file(path), False


@dataclass
class TrainedFold:
    model: CausalMultimodalBackbone
    processor: FoldTextProcessor
    text_features: np.ndarray
    checkpoint_path: Path
    processor_path: Path
    summary: dict[str, object]


def train_one_fold_seed(
    corpus: OpenRoleCorpus,
    split: CrossfitSplit,
    *,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    seed: int,
    source_identity: str,
    checkpoint_root: Path,
    device: torch.device,
    test_interrupt_after_epoch: int | None = None,
    require_complete_checkpoint: bool = False,
) -> TrainedFold:
    """Train or atomically resume one independent seed/outer-fold model."""

    _set_seed(int(seed) + 1009 * int(split.fold))
    run_dir = checkpoint_root / f"seed_{int(seed):05d}" / f"fold_{split.fold:02d}"
    processor_path = run_dir / "text_processor.joblib"
    checkpoint_path = run_dir / "checkpoint.pt"
    if require_complete_checkpoint and (
        not processor_path.is_file() or not checkpoint_path.is_file()
    ):
        raise ContractError(
            "complete-checkpoint-only inference cannot create a processor or train a fold"
        )
    processor, processor_sha, processor_resumed = _load_or_fit_processor(
        processor_path,
        corpus,
        split,
        model_config=model_config,
        run_config=run_config,
        seed=int(seed),
        source_identity=source_identity,
    )
    text_features = processor.transform(corpus.texts)
    checkpoint_identity = _canonical_sha256(
        {
            "source_identity": source_identity,
            "seed": int(seed),
            "fold": int(split.fold),
            "inner_train": _indices_sha256(split.inner_train_indices),
            "inner_validation": _indices_sha256(split.inner_validation_indices),
            "outer_heldout": _indices_sha256(split.outer_heldout_indices),
            "processor_sha256": processor_sha,
            "model_config": asdict(model_config),
            "run_config": asdict(run_config),
        }
    )
    model = CausalMultimodalBackbone(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=run_config.learning_rate,
        weight_decay=run_config.weight_decay,
    )
    amp_enabled = bool(run_config.use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 0
    best_epoch = -1
    best_nll = math.inf
    bad_epochs = 0
    best_state = _model_state_cpu(model)
    resumed_partial = False
    if checkpoint_path.is_file():
        checkpoint = _torch_load_local(checkpoint_path)
        if checkpoint.get("schema_version") != "causal_backbone_atomic_checkpoint_v2":
            raise ContractError("checkpoint schema version changed")
        if checkpoint.get("identity_sha256") != checkpoint_identity:
            raise ContractError("checkpoint identity differs from current fold/data/config")
        best_state = checkpoint["best_model_state"]
        best_epoch = int(checkpoint["best_epoch"])
        best_nll = float(checkpoint["best_validation_nll"])
        if checkpoint.get("status") == "complete":
            rng_state = checkpoint.get("rng_state")
            if not isinstance(rng_state, Mapping):
                raise ContractError("complete checkpoint lacks complete RNG state")
            _restore_rng_state(rng_state)
            model.load_state_dict(best_state)
            print(
                json.dumps(
                    {
                        "event": "causal_backbone_fold_resume_complete",
                        "seed": int(seed),
                        "fold": int(split.fold),
                        "best_epoch": best_epoch,
                        "best_validation_nll": best_nll,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return TrainedFold(
                model=model,
                processor=processor,
                text_features=text_features,
                checkpoint_path=checkpoint_path,
                processor_path=processor_path,
                summary={
                    "seed": int(seed),
                    "fold": int(split.fold),
                    "best_epoch": best_epoch,
                    "best_validation_nll": best_nll,
                    "epochs_completed": int(checkpoint["epoch"]) + 1,
                    "early_stopped": bool(checkpoint.get("early_stopped", False)),
                    "resumed_complete_checkpoint": True,
                    "resumed_partial_checkpoint": False,
                    "processor_resumed": processor_resumed,
                    "effective_text_svd_dim": processor.effective_dim,
                    "peak_cuda_mib": float(checkpoint.get("peak_cuda_mib", 0.0)),
                },
            )
        if require_complete_checkpoint:
            raise ContractError(
                "complete-checkpoint-only inference refuses a partial training checkpoint"
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        _move_optimizer_state(optimizer, device)
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        bad_epochs = int(checkpoint["bad_epochs"])
        rng_state = checkpoint.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise ContractError("partial checkpoint lacks complete RNG state")
        _restore_rng_state(rng_state)
        resumed_partial = True

    if require_complete_checkpoint:
        raise AssertionError("complete-checkpoint-only inference reached the training path")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    weights = _class_weights(
        corpus.labels[split.inner_train_indices], model_config.num_classes, device
    )
    loss_function = torch.nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=run_config.label_smoothing,
    )
    epoch_completed = start_epoch - 1
    early_stopped = False
    for epoch in range(start_epoch, run_config.max_epochs):
        model.train()
        order_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(split.fold), int(epoch), 811])
        )
        order = np.asarray(split.inner_train_indices, dtype=np.int64).copy()
        order_rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        batch_count = int(math.ceil(len(order) / run_config.batch_size))
        for batch_number, start in enumerate(range(0, len(order), run_config.batch_size)):
            query = order[start : start + run_config.batch_size]
            contexts = [tuple(corpus.histories[int(value)]) for value in query]
            packed = _to_device(
                pack_query_contexts(
                    corpus,
                    text_features,
                    query,
                    contexts,
                    max_history_items=run_config.max_history_items,
                ),
                device,
            )
            empty = torch.zeros_like(packed["history_mask"])
            context_masks = torch.stack((empty, packed["history_mask"]), dim=1)
            subset_generator = torch.Generator(device=device)
            subset_generator.manual_seed(
                int(seed) * 10_000_019
                + int(split.fold) * 100_003
                + int(epoch) * 1_009
                + int(batch_number)
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = model.forward_contexts(
                    text_features=packed["text_features"],
                    audio_features=packed["audio_features"],
                    video_features=packed["video_features"],
                    speaker_ids=packed["speaker_ids"],
                    turn_ids=packed["turn_ids"],
                    valid_mask=packed["valid_mask"],
                    context_masks=context_masks,
                    query_indices=packed["query_indices"],
                    subset_dropout_p=run_config.subset_dropout_probability,
                    subset_generator=subset_generator,
                )
                target = torch.from_numpy(corpus.labels[query]).to(device)
                repeated_target = target[:, None].expand(-1, len(ENDPOINT_CONTEXT_NAMES)).reshape(-1)
                classification_loss = loss_function(
                    output.logits.reshape(-1, model_config.num_classes), repeated_target
                )
                if model_config.auxiliary_vad_weight > 0.0:
                    if output.query_vad is None or model.vad_label_table.shape != (
                        model_config.num_classes,
                        3,
                    ):
                        raise ContractError(
                            "enabled affect relation did not expose its frozen VAD contract"
                        )
                    repeated_vad_target = model.vad_label_table[target][:, None, :].expand(
                        -1, len(ENDPOINT_CONTEXT_NAMES), -1
                    )
                    vad_loss = torch.nn.functional.mse_loss(
                        output.query_vad,
                        repeated_vad_target.to(dtype=output.query_vad.dtype),
                    )
                else:
                    if output.query_vad is not None:
                        raise ContractError(
                            "disabled VAD supervision unexpectedly emitted a VAD state"
                        )
                    vad_loss = classification_loss.new_zeros(())
                loss = (
                    classification_loss
                    + model_config.auxiliary_vad_weight * vad_loss
                ) / run_config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            should_step = (
                (batch_number + 1) % run_config.gradient_accumulation_steps == 0
                or batch_number + 1 == batch_count
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), run_config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        validation_nll = _evaluate_early_stopping_nll(
            model,
            corpus,
            text_features,
            split.inner_validation_indices,
            device=device,
            config=run_config,
        )
        if not math.isfinite(validation_nll):
            raise RuntimeError("non-finite inner-validation NLL")
        if validation_nll < best_nll - run_config.early_stopping_min_delta:
            best_nll = validation_nll
            best_epoch = int(epoch)
            best_state = _model_state_cpu(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
        epoch_completed = int(epoch)
        peak_mib = (
            float(torch.cuda.max_memory_allocated(device) / 2**20)
            if device.type == "cuda"
            else 0.0
        )
        if peak_mib > run_config.max_cuda_memory_mib:
            raise RuntimeError(
                f"CUDA allocation {peak_mib:.1f} MiB exceeded the frozen 8GB budget"
            )
        payload = {
            "schema_version": "causal_backbone_atomic_checkpoint_v2",
            "status": "partial",
            "identity_sha256": checkpoint_identity,
            "epoch": epoch_completed,
            "model_state": _model_state_cpu(model),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_model_state": best_state,
            "best_epoch": best_epoch,
            "best_validation_nll": best_nll,
            "bad_epochs": bad_epochs,
            "early_stopped": False,
            "peak_cuda_mib": peak_mib,
            "rng_state": _capture_rng_state(),
        }
        _atomic_torch_save(payload, checkpoint_path)
        print(
            json.dumps(
                {
                    "event": "causal_backbone_epoch",
                    "seed": int(seed),
                    "fold": int(split.fold),
                    "epoch": int(epoch),
                    "validation_nll": validation_nll,
                    "best_validation_nll": best_nll,
                    "bad_epochs": bad_epochs,
                    "peak_cuda_mib": peak_mib,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if test_interrupt_after_epoch is not None and int(epoch) == int(test_interrupt_after_epoch):
            raise PlannedCheckpointInterruption(
                f"planned interruption after epoch {epoch} atomic checkpoint"
            )
        if bad_epochs >= run_config.early_stopping_patience:
            early_stopped = True
            break

    if best_epoch < 0:
        raise RuntimeError("training completed without a finite early-stopping checkpoint")
    model.load_state_dict(best_state)
    peak_mib = (
        float(torch.cuda.max_memory_allocated(device) / 2**20)
        if device.type == "cuda"
        else 0.0
    )
    complete = {
        "schema_version": "causal_backbone_atomic_checkpoint_v2",
        "status": "complete",
        "identity_sha256": checkpoint_identity,
        "epoch": epoch_completed,
        "model_state": best_state,
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_model_state": best_state,
        "best_epoch": best_epoch,
        "best_validation_nll": best_nll,
        "bad_epochs": bad_epochs,
        "early_stopped": early_stopped,
        "peak_cuda_mib": peak_mib,
        "rng_state": _capture_rng_state(),
    }
    _atomic_torch_save(complete, checkpoint_path)
    return TrainedFold(
        model=model,
        processor=processor,
        text_features=text_features,
        checkpoint_path=checkpoint_path,
        processor_path=processor_path,
        summary={
            "seed": int(seed),
            "fold": int(split.fold),
            "best_epoch": best_epoch,
            "best_validation_nll": best_nll,
            "epochs_completed": epoch_completed + 1,
            "early_stopped": early_stopped,
            "resumed_complete_checkpoint": False,
            "resumed_partial_checkpoint": resumed_partial,
            "processor_resumed": processor_resumed,
            "effective_text_svd_dim": processor.effective_dim,
            "peak_cuda_mib": peak_mib,
        },
    )


@dataclass(frozen=True)
class UtilitySamplingConfig:
    draws_per_query: int
    maximum_candidates: int
    seed: int
    match_context_cardinality: bool

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "UtilitySamplingConfig":
        raw = payload.get("counterfactual_sampling")
        if not isinstance(raw, Mapping):
            raise ContractError("utility config lacks counterfactual_sampling")
        result = cls(
            draws_per_query=int(raw.get("draws_per_query", 0)),
            maximum_candidates=int(raw.get("maximum_candidates_per_query", 0)),
            seed=int(raw.get("seed", 0)),
            match_context_cardinality=bool(raw.get("match_context_cardinality", False)),
        )
        if result.draws_per_query < 1 or result.maximum_candidates < 2:
            raise ContractError("invalid bidirectional utility sampling settings")
        if not result.match_context_cardinality:
            raise ContractError("causal backbone run requires size-matched different-set contexts")
        return result


def sample_corpus_bidirectional_tasks(
    corpus: OpenRoleCorpus,
    config: UtilitySamplingConfig,
) -> list[BidirectionalCoalitionTask]:
    """Sample with the original train-row id as the per-query RNG identity.

    Filtering away sealed groups renumbers the in-memory open corpus.  The
    frozen sampler, however, seeded each query by its original train row.  This
    adapter expands only empty placeholders (never features or labels) for
    non-open rows, calls the frozen sampler, and maps its open tasks back.
    """

    protocol_rows = (
        np.arange(len(corpus.keys), dtype=np.int64)
        if corpus.protocol_row_ids is None
        else np.asarray(corpus.protocol_row_ids, dtype=np.int64)
    )
    local_by_protocol = {int(value): index for index, value in enumerate(protocol_rows)}
    expanded: list[tuple[int, ...]] = [tuple() for _ in range(int(protocol_rows.max()) + 1)]
    for local_query, protocol_query in enumerate(protocol_rows):
        expanded[int(protocol_query)] = tuple(
            int(protocol_rows[int(local_history)])
            for local_history in corpus.histories[local_query]
        )
    sampled = sample_bidirectional_coalition_tasks(
        expanded,
        draws_per_query=config.draws_per_query,
        maximum_candidates=config.maximum_candidates,
        seed=config.seed,
        match_context_cardinality=config.match_context_cardinality,
    )
    result: list[BidirectionalCoalitionTask] = []
    for task in sampled:
        if int(task.query_index) not in local_by_protocol:
            continue
        try:
            result.append(
                BidirectionalCoalitionTask(
                    query_index=local_by_protocol[int(task.query_index)],
                    addition_context=tuple(
                        local_by_protocol[int(value)] for value in task.addition_context
                    ),
                    deletion_context=tuple(
                        local_by_protocol[int(value)] for value in task.deletion_context
                    ),
                    candidate_index=local_by_protocol[int(task.candidate_index)],
                )
            )
        except KeyError as error:
            raise ContractError("an open query history unexpectedly references a sealed row") from error
    return result


def _encode_task_contexts(
    tasks: Sequence[BidirectionalCoalitionTask],
) -> dict[str, np.ndarray]:
    query = np.asarray([int(task.query_index) for task in tasks], dtype=np.int64)
    candidate = np.asarray([int(task.candidate_index) for task in tasks], dtype=np.int64)

    def csr(values: Iterable[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
        indices: list[int] = []
        indptr = [0]
        for row in values:
            indices.extend(int(value) for value in row)
            indptr.append(len(indices))
        return np.asarray(indptr, dtype=np.int64), np.asarray(indices, dtype=np.int64)

    s_indptr, s_indices = csr(task.addition_context for task in tasks)
    t_indptr, t_indices = csr(task.deletion_context for task in tasks)
    return {
        "query_indices": query,
        "candidate_indices": candidate,
        "s_indptr": s_indptr,
        "s_indices": s_indices,
        "t_indptr": t_indptr,
        "t_indices": t_indices,
    }


def _task_sha256(tasks: Sequence[BidirectionalCoalitionTask]) -> str:
    return _canonical_sha256(
        [
            {
                "query": int(task.query_index),
                "candidate": int(task.candidate_index),
                "s": list(task.addition_context),
                "t": list(task.deletion_context),
            }
            for task in tasks
        ]
    )


def _utility_arrays(
    corpus: OpenRoleCorpus,
    tasks: Sequence[BidirectionalCoalitionTask],
    probability: np.ndarray,
) -> dict[str, np.ndarray]:
    probability = np.asarray(probability)
    if (
        probability.ndim != 4
        or probability.shape[1] != len(tasks)
        or probability.shape[2] != len(UTILITY_CONTEXT_NAMES)
        or probability.shape[3] < 2
    ):
        raise ContractError("utility probability array has the wrong shape")
    labels = corpus.labels[
        np.asarray([int(task.query_index) for task in tasks], dtype=np.int64)
    ]
    if np.any((labels < 0) | (labels >= probability.shape[3])):
        raise ContractError("utility label is outside the dataset label order")
    forward = np.empty((probability.shape[0], len(tasks)), dtype=np.float32)
    backward = np.empty_like(forward)
    asymmetry = np.empty_like(forward)
    agreement = np.empty((probability.shape[0], len(tasks)), dtype=bool)
    for seed_index in range(probability.shape[0]):
        targets = bidirectional_utility_targets(
            labels,
            probability[seed_index, :, 0],
            probability[seed_index, :, 1],
            probability[seed_index, :, 2],
            probability[seed_index, :, 3],
        )
        forward[seed_index] = targets.forward_addition.astype(np.float32)
        backward[seed_index] = targets.backward_deletion.astype(np.float32)
        asymmetry[seed_index] = targets.asymmetry.astype(np.float32)
        agreement[seed_index] = targets.sign_agreement
    return {
        "forward": forward,
        "backward": backward,
        "asymmetry": asymmetry,
        "sign_agreement": agreement,
    }


def _utility_summary(values: Mapping[str, np.ndarray]) -> dict[str, object]:
    rows: list[dict[str, float | int]] = []
    for seed_index in range(values["forward"].shape[0]):
        forward = values["forward"][seed_index]
        backward = values["backward"][seed_index]
        rows.append(
            {
                "seed_index": int(seed_index),
                "mean_forward_addition": float(np.mean(forward)) if len(forward) else math.nan,
                "mean_backward_deletion": float(np.mean(backward)) if len(backward) else math.nan,
                "mean_asymmetry": float(np.mean(values["asymmetry"][seed_index])) if len(forward) else math.nan,
                "sign_agreement_rate": float(np.mean(values["sign_agreement"][seed_index])) if len(forward) else math.nan,
            }
        )
    finite_rows = [row for row in rows if math.isfinite(float(row["mean_forward_addition"]))]
    aggregate: dict[str, float] = {}
    for field in (
        "mean_forward_addition",
        "mean_backward_deletion",
        "mean_asymmetry",
        "sign_agreement_rate",
    ):
        aggregate[field] = (
            float(np.mean([float(row[field]) for row in finite_rows]))
            if finite_rows
            else math.nan
        )
    return {"per_seed": rows, "five_seed_mean": aggregate}


def _endpoint_summary(
    corpus: OpenRoleCorpus,
    query_indices: np.ndarray,
    probability: np.ndarray,
    seeds: Sequence[int],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    labels = corpus.labels[query_indices]
    for seed_index, seed in enumerate(seeds):
        current = classification_metrics(labels, probability[seed_index, :, 0])
        history = classification_metrics(labels, probability[seed_index, :, 1])
        rows.append(
            {
                "seed": int(seed),
                "current_only": current,
                "all_history": history,
                "all_minus_current": {
                    name: float(history[name] - current[name])
                    for name in ("macro_f1", "accuracy", "nll", "brier")
                },
            }
        )
    mean: dict[str, dict[str, float]] = {}
    for context in ENDPOINT_CONTEXT_NAMES:
        mean[context] = {
            metric: float(np.mean([row[context][metric] for row in rows]))
            for metric in ("macro_f1", "accuracy", "nll", "brier")
        }
    mean["all_minus_current"] = {
        metric: float(np.mean([row["all_minus_current"][metric] for row in rows]))
        for metric in ("macro_f1", "accuracy", "nll", "brier")
    }
    return {"queries": int(len(query_indices)), "per_seed": rows, "five_seed_mean": mean}


def _assert_private_root(private_root: Path, repository_root: Path | None) -> None:
    if repository_root is None:
        return
    private = private_root.resolve()
    repository = repository_root.resolve()
    try:
        common = Path(os.path.commonpath((str(private), str(repository))))
    except ValueError:
        return  # Different Windows drives cannot overlap.
    if common == repository:
        raise ContractError("private cache/checkpoints must be outside the public repository")


def _checkpoint_manifest(paths: Sequence[Path], root: Path) -> tuple[list[dict[str, object]], str]:
    records = [
        {
            "relative_name": path.resolve().relative_to(root.resolve()).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(paths, key=lambda value: value.as_posix())
    ]
    return records, _canonical_sha256(records)


def execute_crossfit_backbone(
    corpus: OpenRoleCorpus,
    *,
    provenance: VerifiedCorpusProvenance,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    sampling_config: UtilitySamplingConfig,
    seeds: Sequence[int],
    private_output_dir: Path,
    public_output_path: Path,
    repository_root: Path | None,
    device: torch.device,
    private_cache_filename: str = "emotiontalk_causal_backbone_oof_v1.npz",
) -> dict:
    """Train all folds/seeds, write a private cache, and publish aggregates."""

    run_config.validate()
    corpus.validate(model_config)
    provenance.validate(corpus, model_config)
    if not seeds or len(set(int(value) for value in seeds)) != len(seeds):
        raise ContractError("training seeds must be a non-empty unique sequence")
    dataset_id = provenance.dataset_id
    if not dataset_id or any(character in dataset_id for character in "\\/\x00"):
        raise ContractError("dataset_id is empty or unsafe")
    if Path(private_cache_filename).name != private_cache_filename or not private_cache_filename.endswith(".npz"):
        raise ContractError("private cache filename must be a plain .npz filename")
    if public_output_path.exists():
        raise FileExistsError(f"public aggregate already exists: {public_output_path}")
    _assert_private_root(private_output_dir, repository_root)
    private_output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ContractError("CUDA was requested but is unavailable")
    code_hashes_start = _internal_code_hashes()
    runtime_environment = _runtime_environment(device)
    runtime_environment_sha256 = _canonical_sha256(runtime_environment)
    source_hashes = dict(sorted(provenance.source_hashes.items()))

    fit_queries = corpus.role_indices(FIT_ROLE)
    selection_queries = corpus.role_indices(SELECTION_ROLE)
    if len(fit_queries) == 0 or len(selection_queries) == 0:
        raise ContractError("both open roles require rows")
    sampled = sample_corpus_bidirectional_tasks(corpus, sampling_config)
    fit_tasks = [task for task in sampled if str(corpus.roles[task.query_index]) == FIT_ROLE]
    selection_tasks = [
        task for task in sampled if str(corpus.roles[task.query_index]) == SELECTION_ROLE
    ]
    if not fit_tasks or not selection_tasks:
        raise ContractError("bidirectional sampling produced no task for an open role")
    for task in (*fit_tasks, *selection_tasks):
        for context in _task_contexts(task):
            _validate_context(corpus, task.query_index, context)

    seeds = tuple(int(value) for value in seeds)
    classes = model_config.num_classes
    fit_endpoint = np.full(
        (len(seeds), len(fit_queries), len(ENDPOINT_CONTEXT_NAMES), classes),
        np.nan,
        dtype=np.float32,
    )
    selection_endpoint = np.zeros(
        (len(seeds), len(selection_queries), len(ENDPOINT_CONTEXT_NAMES), classes),
        dtype=np.float64,
    )
    fit_utility_probability = np.full(
        (len(seeds), len(fit_tasks), len(UTILITY_CONTEXT_NAMES), classes),
        np.nan,
        dtype=np.float32,
    )
    selection_utility_probability = np.zeros(
        (len(seeds), len(selection_tasks), len(UTILITY_CONTEXT_NAMES), classes),
        dtype=np.float64,
    )
    fit_position = {int(value): index for index, value in enumerate(fit_queries)}
    fit_task_queries = np.asarray([task.query_index for task in fit_tasks], dtype=np.int64)
    selection_task_queries = np.asarray(
        [task.query_index for task in selection_tasks], dtype=np.int64
    )
    source_identity = _canonical_sha256(
        {
            "verified_provenance": {
                **_provenance_attestation_payload(provenance),
                "verifier_attestation_sha256": provenance.verifier_attestation_sha256,
            },
            "dataset_id": dataset_id,
            "corpus_contract_sha256": provenance.corpus_contract_sha256,
            "histories_sha256": _histories_sha256(corpus.histories),
            "speaker_mapping_sha256": provenance.speaker_mapping_sha256,
            "internal_code_hashes": code_hashes_start,
            "runtime_environment": runtime_environment,
            "model_config": asdict(model_config),
            "run_config": asdict(run_config),
            "sampling_config": asdict(sampling_config),
            "seeds": seeds,
        }
    )
    training_summaries: list[dict[str, object]] = []
    fold_count_summaries: list[dict[str, int]] = []
    artifact_paths: list[Path] = []
    checkpoint_root = private_output_dir / "checkpoints"

    for seed_index, seed in enumerate(seeds):
        splits = make_crossfit_splits(
            corpus,
            outer_folds=run_config.outer_folds,
            validation_fraction=run_config.inner_validation_fraction,
            seed=seed,
        )
        for split in splits:
            fold_count_summaries.append(
                {
                    "seed": int(seed),
                    "fold": int(split.fold),
                    "inner_train_rows": int(len(split.inner_train_indices)),
                    "inner_validation_rows": int(len(split.inner_validation_indices)),
                    "outer_heldout_rows": int(len(split.outer_heldout_indices)),
                    "inner_train_groups": int(len(set(corpus.groups[split.inner_train_indices]))),
                    "inner_validation_groups": int(len(set(corpus.groups[split.inner_validation_indices]))),
                    "outer_heldout_groups": int(len(set(corpus.groups[split.outer_heldout_indices]))),
                }
            )
            trained = train_one_fold_seed(
                corpus,
                split,
                model_config=model_config,
                run_config=run_config,
                seed=seed,
                source_identity=source_identity,
                checkpoint_root=checkpoint_root,
                device=device,
            )
            training_summaries.append(trained.summary)
            artifact_paths.extend((trained.checkpoint_path, trained.processor_path))
            held_probability = predict_current_and_all_history(
                trained.model,
                corpus,
                trained.text_features,
                split.outer_heldout_indices,
                device=device,
                batch_size=run_config.inference_batch_size,
                max_history_items=run_config.max_history_items,
            )
            held_positions = np.asarray(
                [fit_position[int(value)] for value in split.outer_heldout_indices],
                dtype=np.int64,
            )
            fit_endpoint[seed_index, held_positions] = held_probability
            held_task_indices = np.flatnonzero(
                np.isin(fit_task_queries, split.outer_heldout_indices)
            )
            fit_utility_probability[seed_index, held_task_indices] = predict_utility_contexts(
                trained.model,
                corpus,
                trained.text_features,
                [fit_tasks[index] for index in held_task_indices],
                device=device,
                batch_size=run_config.inference_batch_size,
                max_history_items=run_config.max_history_items,
            )
            # Model-selection labels are not passed to either of these calls.
            selection_endpoint[seed_index] += predict_current_and_all_history(
                trained.model,
                corpus,
                trained.text_features,
                selection_queries,
                device=device,
                batch_size=run_config.inference_batch_size,
                max_history_items=run_config.max_history_items,
            )
            selection_utility_probability[seed_index] += predict_utility_contexts(
                trained.model,
                corpus,
                trained.text_features,
                selection_tasks,
                device=device,
                batch_size=run_config.inference_batch_size,
                max_history_items=run_config.max_history_items,
            )
            del trained.model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        selection_endpoint[seed_index] /= len(splits)
        selection_utility_probability[seed_index] /= len(splits)

    arrays_to_check = (
        fit_endpoint,
        selection_endpoint,
        fit_utility_probability,
        selection_utility_probability,
    )
    if any(not np.isfinite(value).all() for value in arrays_to_check):
        raise RuntimeError("OOF/model-selection probability generation is incomplete")
    selection_endpoint = selection_endpoint.astype(np.float32)
    selection_utility_probability = selection_utility_probability.astype(np.float32)
    fit_utility = _utility_arrays(corpus, fit_tasks, fit_utility_probability)
    selection_utility = _utility_arrays(
        corpus, selection_tasks, selection_utility_probability
    )
    code_hashes_end = _internal_code_hashes()
    if code_hashes_end != code_hashes_start:
        raise ContractError("causal runner code changed while the training run was active")
    checkpoint_records, checkpoint_manifest_sha = _checkpoint_manifest(
        artifact_paths, private_output_dir
    )
    fit_task_encoding = _encode_task_contexts(fit_tasks)
    selection_task_encoding = _encode_task_contexts(selection_tasks)
    matrix_hashes = {
        "fit_endpoint_probability_oof": _array_sha256(fit_endpoint),
        "selection_endpoint_probability_fold_ensemble": _array_sha256(
            selection_endpoint
        ),
        "fit_utility_probability_oof": _array_sha256(fit_utility_probability),
        "selection_utility_probability_fold_ensemble": _array_sha256(
            selection_utility_probability
        ),
        "fit_forward_utility": _array_sha256(fit_utility["forward"]),
        "fit_backward_utility": _array_sha256(fit_utility["backward"]),
        "selection_forward_utility": _array_sha256(selection_utility["forward"]),
        "selection_backward_utility": _array_sha256(selection_utility["backward"]),
    }
    cluster_values = sorted(set(corpus.groups.astype(str)))
    cluster_map = {value: index for index, value in enumerate(cluster_values)}
    cluster_codes = np.asarray(
        [cluster_map[str(value)] for value in corpus.groups], dtype=np.int32
    )
    private_cache_path = private_output_dir / private_cache_filename
    cache: dict[str, np.ndarray] = {
        "schema_version": np.asarray(PRIVATE_CACHE_SCHEMA),
        "dataset": np.asarray(dataset_id),
        "dataset_label_order": np.asarray(provenance.label_order),
        "manifest_schema": np.asarray(provenance.manifest_schema),
        "manifest_status": np.asarray(provenance.manifest_status),
        "manifest_sha256": np.asarray(provenance.manifest_sha256),
        "verified_provenance_attestation_sha256": np.asarray(
            provenance.verifier_attestation_sha256
        ),
        "corpus_contract_sha256": np.asarray(provenance.corpus_contract_sha256),
        "histories_sha256": np.asarray(_histories_sha256(corpus.histories)),
        "speaker_mapping_sha256": np.asarray(provenance.speaker_mapping_sha256),
        "runtime_environment_sha256": np.asarray(runtime_environment_sha256),
        "source_identity_sha256": np.asarray(source_identity),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "endpoint_context_names": np.asarray(ENDPOINT_CONTEXT_NAMES),
        "utility_context_names": np.asarray(UTILITY_CONTEXT_NAMES),
        "fit_query_indices": fit_queries,
        "selection_query_indices": selection_queries,
        "fit_cluster_codes": cluster_codes[fit_queries],
        "selection_cluster_codes": cluster_codes[selection_queries],
        "protocol_row_ids": (
            np.arange(len(corpus.keys), dtype=np.int64)
            if corpus.protocol_row_ids is None
            else np.asarray(corpus.protocol_row_ids, dtype=np.int64)
        ),
        "fit_endpoint_probability_oof": fit_endpoint,
        "selection_endpoint_probability_fold_ensemble": selection_endpoint,
        "fit_utility_probability_oof": fit_utility_probability,
        "selection_utility_probability_fold_ensemble": selection_utility_probability,
        "fit_forward_utility": fit_utility["forward"],
        "fit_backward_utility": fit_utility["backward"],
        "fit_asymmetry": fit_utility["asymmetry"],
        "fit_sign_agreement": fit_utility["sign_agreement"],
        "selection_forward_utility": selection_utility["forward"],
        "selection_backward_utility": selection_utility["backward"],
        "selection_asymmetry": selection_utility["asymmetry"],
        "selection_sign_agreement": selection_utility["sign_agreement"],
        "fit_task_sha256": np.asarray(_task_sha256(fit_tasks)),
        "selection_task_sha256": np.asarray(_task_sha256(selection_tasks)),
        "checkpoint_manifest_sha256": np.asarray(checkpoint_manifest_sha),
        "utility_source": np.asarray(
            "recomputed_from_causal_backbone_probabilities_and_open_role_labels"
        ),
    }
    for name, value in matrix_hashes.items():
        cache[f"matrix_{name}_sha256"] = np.asarray(value)
    for prefix, encoding in (
        ("fit_task", fit_task_encoding),
        ("selection_task", selection_task_encoding),
    ):
        for name, value in encoding.items():
            cache[f"{prefix}_{name}"] = value
    for name, value in sorted(source_hashes.items()):
        cache[f"source_{name}_sha256"] = np.asarray(str(value))
    _atomic_savez(private_cache_path, **cache)
    private_cache_sha = sha256_file(private_cache_path)

    fit_endpoint_summary = _endpoint_summary(corpus, fit_queries, fit_endpoint, seeds)
    selection_endpoint_summary = _endpoint_summary(
        corpus, selection_queries, selection_endpoint, seeds
    )
    fit_history_eligible = int(
        sum(bool(corpus.histories[int(index)]) for index in fit_queries)
    )
    selection_history_eligible = int(
        sum(bool(corpus.histories[int(index)]) for index in selection_queries)
    )
    public_split = (
        "official_train_open_roles_only"
        if dataset_id == "MELD"
        else "train_corpus_open_roles_only"
    )
    public_report = {
        "schema_version": PUBLIC_REPORT_SCHEMA,
        "status": "open_role_crossfit_complete_not_confirmatory_evidence",
        "artifact_role": "probability_and_utility_producer_only_not_an_evidence_gate",
        "claim_boundary": (
            "Backbone-relative open-role OOF/model-selection evidence only; "
            "not a psychological causal effect and not calibration/holdout/test evidence."
        ),
        "performance_claim_gate": {
            "authorized": False,
            "reason": "open_role_backbone_artifact_generation_only",
            "required_before_any_performance_claim": [
                "independently_trained_current_only_baseline",
                "coverage_matched_recency_baseline",
                "frozen_query_policy_and_operating_point",
                "paired_seed_by_cluster_confidence_intervals",
                "prespecified_multiple_comparison_correction",
            ],
        },
        "dataset": dataset_id,
        "dataset_label_order": list(provenance.label_order),
        "split": public_split,
        "verified_manifest": {
            "schema_version": provenance.manifest_schema,
            "status": provenance.manifest_status,
            "sha256": provenance.manifest_sha256,
            "verification_origin": provenance.verification_origin,
        },
        "data_boundary": {
            "roles_materialised": list(OPEN_ROLES),
            "buckets_materialised": [0, 79],
            "calibration_buckets_80_89_opened": False,
            "internal_holdout_buckets_90_99_opened": False,
            "validation_rows_materialised": False,
            "validation_labels_opened": False,
            "test_rows_materialised": False,
            "test_labels_opened": False,
            "strict_role_feature_sidecars": provenance.strict_role_feature_sidecars,
            "strict_role_label_sidecars": provenance.strict_role_label_sidecars,
            "sealed_role_arrays_opened": provenance.sealed_role_arrays_opened,
            "validation_or_test_opened": provenance.validation_or_test_opened,
            "label_access_mode": corpus.label_access_mode,
        },
        "rows_and_groups": {
            "fit_rows": int(len(fit_queries)),
            "fit_groups": int(len(set(corpus.groups[fit_queries]))),
            "fit_history_eligible_rows": fit_history_eligible,
            "model_selection_rows": int(len(selection_queries)),
            "model_selection_groups": int(len(set(corpus.groups[selection_queries]))),
            "model_selection_history_eligible_rows": selection_history_eligible,
            "fit_utility_tasks": int(len(fit_tasks)),
            "model_selection_utility_tasks": int(len(selection_tasks)),
        },
        "training": {
            "seeds": list(seeds),
            "seed_count": len(seeds),
            "outer_folds": run_config.outer_folds,
            "fold_partitions": fold_count_summaries,
            "early_stopping": {
                "uses_outer_heldout_labels": False,
                "uses_model_selection_labels": False,
                "inner_group_disjoint": True,
                "patience": run_config.early_stopping_patience,
                "minimum_delta": run_config.early_stopping_min_delta,
            },
            "amp_requested": run_config.use_amp,
            "amp_used": bool(run_config.use_amp and device.type == "cuda"),
            "device_type": device.type,
            "max_cuda_memory_mib": run_config.max_cuda_memory_mib,
            "parameter_count": CausalMultimodalBackbone(model_config).parameter_count(),
            "parameter_limit_strictly_less_than": model_config.parameter_limit,
            "fold_runs": training_summaries,
        },
        "feature_contract": {
            "text": f"fold-local TF-IDF then SVD{model_config.text_dim}; fit on inner train groups only",
            "audio": f"manifest-verified frozen sidecar vector {model_config.audio_dim}D",
            "video": f"manifest-verified frozen sidecar vector {model_config.video_dim}D",
            "actual_dimensions": {
                "text": model_config.text_dim,
                "audio": model_config.audio_dim,
                "video": model_config.video_dim,
            },
            "history_overflow_policy": "fail_closed_no_silent_truncation",
            "maximum_history_items": run_config.max_history_items,
            "strict_past_and_arbitrary_subset_mask": True,
        },
        "probability_protocol": {
            "fit": "one outer-heldout prediction per query/seed",
            "model_selection": "inference-only mean probability over outer-fold models within seed",
            "current_only_semantics": (
                "same_trained_model_empty_history_intervention_"
                "not_independently_trained_baseline"
            ),
            "endpoints": list(ENDPOINT_CONTEXT_NAMES),
            "utility_contexts": list(UTILITY_CONTEXT_NAMES),
            "utility_source": (
                "fresh causal-backbone probabilities; no old linear utility target is consumed"
            ),
            "nll_probability_floor": NLL_PROBABILITY_FLOOR,
            "nll_definition": "mean(-log(clip(p_true, 1e-12, 1.0)))",
        },
        "aggregate_results": {
            "fit_oof_endpoints": fit_endpoint_summary,
            "model_selection_endpoints": selection_endpoint_summary,
            "fit_oof_bidirectional_utility": _utility_summary(fit_utility),
            "model_selection_bidirectional_utility": _utility_summary(selection_utility),
        },
        "provenance": {
            "input_hashes": source_hashes,
            "manifest_sha256": provenance.manifest_sha256,
            "verified_provenance_attestation_sha256": (
                provenance.verifier_attestation_sha256
            ),
            "corpus_contract_sha256": provenance.corpus_contract_sha256,
            "histories_sha256": _histories_sha256(corpus.histories),
            "speaker_mapping_sha256": provenance.speaker_mapping_sha256,
            "internal_code_hashes_start": code_hashes_start,
            "internal_code_hashes_end": code_hashes_end,
            "runtime_environment_sha256": runtime_environment_sha256,
            "source_identity_sha256": source_identity,
            "role_assignment_sha256": provenance.role_assignment_sha256,
            "private_cache_sha256": private_cache_sha,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha,
            "checkpoint_file_count": len(checkpoint_records),
            "private_matrix_hashes": matrix_hashes,
            "private_paths_disclosed": False,
        },
        "runtime_environment": runtime_environment,
        "public_artifact_policy": {
            "contains_row_level_keys_predictions_utilities_or_embeddings": False,
            "private_cache_and_checkpoints_outside_repository": repository_root is not None,
        },
    }
    if public_output_path.exists():
        raise FileExistsError(f"public aggregate already exists: {public_output_path}")
    write_json_atomic(public_report, public_output_path.resolve())
    return public_report


def _read_json_mapping(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be a mapping: {path.name}")
    return value


def validate_open_role_backbone_payload(payload: Mapping[str, object]) -> None:
    if payload.get("status") != "frozen_open_role_production_contract_not_performance_evidence":
        raise ContractError("backbone configuration is not a frozen open-role production contract")
    runtime = payload.get("runtime_contract")
    if not isinstance(runtime, Mapping):
        raise ContractError("backbone configuration lacks runtime_contract")
    if runtime.get("staged_execution_required") is True:
        raise ContractError(
            "staged-only backbone configuration cannot use the monolithic runner"
        )
    if runtime.get("sealed_test_labels_must_remain_unopened") is not True:
        raise ContractError("backbone configuration does not require sealed test labels to remain unopened")
    if "sealed_test_labels_required" in runtime:
        raise ContractError("ambiguous legacy sealed-test setting is forbidden")


def run_emotiontalk_causal_backbone(
    *,
    sidecar_dir: Path,
    sidecar_manifest_path: Path,
    backbone_config_path: Path,
    utility_config_path: Path,
    confirmatory_config_path: Path,
    private_output_dir: Path,
    public_output_path: Path,
    repository_root: Path,
    device_name: str = "auto",
) -> dict:
    """Production entry point pinned to the frozen five-seed open-role contract."""

    backbone_payload = _read_json_mapping(backbone_config_path)
    utility_payload = _read_json_mapping(utility_config_path)
    confirmatory_payload = _read_json_mapping(confirmatory_config_path)
    validate_open_role_backbone_payload(backbone_payload)
    model_config = CausalBackboneConfig.from_mapping(backbone_payload)
    run_config = BackboneRunConfig.from_mapping(backbone_payload)
    sampling_config = UtilitySamplingConfig.from_mapping(utility_payload)
    roles = utility_payload.get("data_roles")
    if not isinstance(roles, Mapping):
        raise ContractError("utility configuration lacks frozen data roles")
    role_ranges = {name: roles.get(name) for name in FROZEN_ROLE_RANGES}
    if role_ranges != FROZEN_ROLE_RANGES:
        raise ContractError("utility role ranges changed")
    if str(roles.get("split_protocol_id", "")) != "scu_set_exploration_v1":
        raise ContractError("utility split protocol changed")
    independent = confirmatory_payload.get("independent_runs")
    if not isinstance(independent, Mapping):
        raise ContractError("confirmatory configuration lacks independent_runs")
    seeds = tuple(int(value) for value in independent.get("seeds", ()))
    if seeds != EXPECTED_SEEDS or int(independent.get("required_seed_count", 0)) != 5:
        raise ContractError("production causal backbone requires frozen seeds 17/29/43/71/101")
    if (model_config.text_dim, model_config.audio_dim, model_config.video_dim) != (
        256,
        1536,
        768,
    ):
        raise ContractError("production feature dimensions must be SVD256/WavLM1536/DINO768")
    if CausalMultimodalBackbone(model_config).parameter_count() >= 2_000_000:
        raise ContractError("production causal backbone is not strictly under 2M parameters")
    corpus, provenance = load_emotiontalk_open_role_corpus(
        sidecar_dir=sidecar_dir,
        manifest_path=sidecar_manifest_path,
        model_config=model_config,
    )
    provenance = create_verified_corpus_provenance(
        dataset_id=provenance.dataset_id,
        manifest_schema=provenance.manifest_schema,
        manifest_status=provenance.manifest_status,
        manifest_sha256=provenance.manifest_sha256,
        source_hashes={
            **provenance.source_hashes,
            "backbone_config": sha256_file(backbone_config_path),
            "utility_config": sha256_file(utility_config_path),
            "confirmatory_config": sha256_file(confirmatory_config_path),
        },
        label_order=provenance.label_order,
        role_rows=provenance.role_rows,
        audio_dim=provenance.audio_dim,
        video_dim=provenance.video_dim,
        role_assignment_sha256=provenance.role_assignment_sha256,
        speaker_mapping_sha256=provenance.speaker_mapping_sha256,
        corpus_contract_sha256=provenance.corpus_contract_sha256,
        verification_origin=provenance.verification_origin,
    )
    provenance.validate(corpus, model_config)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    return execute_crossfit_backbone(
        corpus,
        provenance=provenance,
        model_config=model_config,
        run_config=run_config,
        sampling_config=sampling_config,
        seeds=seeds,
        private_output_dir=private_output_dir,
        public_output_path=public_output_path,
        repository_root=repository_root,
        device=device,
    )
