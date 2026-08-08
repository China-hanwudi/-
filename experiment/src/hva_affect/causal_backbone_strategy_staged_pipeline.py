"""Outcome-free completion of the frozen CARMA history-selection strategy.

This module is wired into the main command line only through the typed,
outcome-free ``strategy-complete-selection`` path.  It is the last private
producer before the evidence evaluator and has only one production capability:

* verify the completed history-aware and independent current-only producers;
* fit the registered shared two-head utility scorer on fit-only supervision;
* freeze the exact 25% fit-OOF operating point and apply it without outcomes;
* restore complete history checkpoints and infer selected-history and exactly
  cardinality-matched recency probabilities; and
* publish a write-once private cache plus an aggregate-only receipt.

The public APIs in this file accept feature files and receipts, never an outcome
array or an outcome-file path.  Fit-only utility supervision is recovered from
the canonical, already attested history private root.  Model-selection outcomes
are neither resolved nor deserialised here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast

import numpy as np
import torch

from .bidirectional_emotion_utility import BidirectionalCoalitionTask
from .bidirectional_utility_model import (
    DEFAULT_SEEDS as UTILITY_SEEDS,
    PRIMARY_HISTORY_COVERAGE,
    UtilityModelSpec,
    UtilitySplit,
    default_model_specs,
    fit_utility_model,
    group_oof_predictions,
)
from .causal_backbone_current_only_pipeline import (
    CURRENT_ONLY_COMPLETION_PROTOCOL,
    CURRENT_ONLY_COMPLETION_RECEIPT_SCHEMA,
    CURRENT_ONLY_PRIVATE_CLAIM_SCHEMA,
    CURRENT_ONLY_PRIVATE_CLAIM_NAME,
    CurrentOnlyProducerAlignmentView,
    current_only_private_paths,
)
from .causal_backbone_evidence import (
    CURRENT_ONLY_CACHE_SCHEMA,
    INDEPENDENT_CURRENT_ONLY_PROTOCOL,
    EncodedCandidateTasks,
    FrozenCoverageRule,
    IndependentCurrentOnlyArtifact,
    PreparedPolicyContexts,
    current_only_independence_attestation_payload,
    freeze_fit_oof_operating_point,
    prepare_policy_contexts,
)
from .causal_backbone_evidence_runner import (
    ENDPOINT_CONTEXT_NAMES,
    EXPECTED_SEEDS,
    FIT_ROLE,
    SELECTION_ROLE,
    UTILITY_CONTEXT_NAMES,
    FitRoleView,
    SelectionFeatureRecord,
    SelectionFeatureView,
    _SPECS,
    _array_sha256,
    _canonical_production_source_key,
    _canonical_sha256,
    _file_sha256,
    _load_receipt,
    _read_manifest_json,
    _require_sha256,
    _single_text,
    _validate_manifest_contract,
    _validate_normalized_rows,
    build_checkpoint_manifest,
    validate_fit_receipt,
    verify_checkpoint_manifest,
)
from .causal_backbone_evidence_stage_b import (
    CURRENT_ONLY_FIT_BOOTSTRAP_ARTIFACT_SCHEMA,
    CURRENT_ONLY_FIT_PRODUCER_RECEIPT_SCHEMA,
    CURRENT_ONLY_PRODUCTION_EXECUTION_MODE,
    _validate_aggregate_producer_receipt,
)
from .causal_backbone_history_staged_pipeline import (
    HISTORY_COMPLETE_OUTCOME_SCHEMA,
    HISTORY_STAGED_PROTOCOL,
    PRODUCTION_TRAINER_MODE,
    HistoryFitTargetsView,
    HistoryOutcomeFreeView,
    VerifiedHistoryCompletionAttestation,
    VerifiedHistoryFitState,
    _COMPLETE_OUTCOME_KEYS,
    _cluster_codes,
    _config_sha256,
    _fit_corpus_from_view,
    _outcome_free_view_from_values,
    _production_private_paths,
    _selection_corpus_from_view,
    _split_from_outer_partition,
    _strict_histories,
    _write_json_once,
    _write_npz_once,
    load_history_fit_outcome_view,
    load_history_fit_targets_view,
    validate_strict_past_histories,
    verify_complete_history_checkpoint_payloads,
    verify_history_completion_production_attestation,
)
from .causal_multimodal_backbone import CausalBackboneConfig
from .emotiontalk_bidirectional_oof import probability_task_features
from .emotiontalk_causal_backbone_runner import (
    BackboneRunConfig,
    OpenRoleCorpus,
    predict_one_probability_per_query,
    train_one_fold_seed,
)


STRATEGY_PRIVATE_SCHEMA = "carma_causal_backbone_strategy_complete_private_v1"
STRATEGY_COMPLETION_RECEIPT_SCHEMA = (
    "carma_causal_backbone_strategy_complete_receipt_v1"
)
STRATEGY_PROTOCOL = (
    "fit_group_oof_bidirectional_shared_mlp_exact25_then_complete_checkpoint_v1"
)
STRATEGY_COMPLETION_STATUS = (
    "outcome_free_strategy_probability_cache_complete_not_performance_evidence"
)
STRATEGY_PRIVATE_CLAIM_SCHEMA = "carma_strategy_private_run_claim_v1"
STRATEGY_PRIVATE_CLAIM_NAME = "strategy-run-claim.json"
STRATEGY_PRIVATE_ARTIFACT_NAME = "strategy-complete.npz"
STRATEGY_PRIVATE_RECEIPT_NAME = "strategy-complete-receipt.json"
REGISTERED_UTILITY_SCORER = "bidirectional_shared_mlp"
METHOD_ROSTER = (
    "bidirectional_selected_history",
    "forward_only_selected_history",
    "backward_only_selected_history",
    "coverage_matched_recency",
    "all_history_diagnostic",
)
JOINT_EVALUATION_ROSTER = ("independent_current_only", *METHOD_ROSTER)
REGISTERED_VARIANTS = (
    "full",
    "no_vad",
    "no_history_3x3",
    "capacity_control",
)

_REGISTERED_VARIANT_BY_MODEL_CONTRACT = {
    ("primary_history_relation", True): "full",
    ("primary_history_relation", False): "no_vad",
    ("vad_history_only_no_history_3x3", True): "no_history_3x3",
    ("history_presence_capacity_control", True): "capacity_control",
}


class StrategyStagedPipelineError(ValueError):
    """Raised when the strategy producer crosses its outcome-free contract."""


def derive_registered_variant(model_config: CausalBackboneConfig) -> str:
    """Derive the sole preregistered variant allowed by a model contract.

    Variant names are evidence-bearing protocol fields, not caller-controlled
    display labels.  Deriving them from the validated relation mode and VAD
    switch prevents a no-VAD or ablation checkpoint tree from being relabelled
    as the full model before any strategy output is created.
    """

    if not isinstance(model_config, CausalBackboneConfig):
        raise StrategyStagedPipelineError(
            "strategy variant derivation requires CausalBackboneConfig"
        )
    try:
        model_config.validate()
    except (TypeError, ValueError) as error:
        raise StrategyStagedPipelineError(
            f"invalid strategy model contract: {error}"
        ) from error
    key = (
        str(model_config.affect_relation_mode),
        bool(model_config.affect_relation_use_vad_features),
    )
    derived = _REGISTERED_VARIANT_BY_MODEL_CONTRACT.get(key)
    if derived is None:
        raise StrategyStagedPipelineError(
            "strategy model contract is not one of the four preregistered variants"
        )
    return derived


def _require_registered_variant_matches_model_config(
    registered_variant: str,
    model_config: CausalBackboneConfig,
) -> str:
    if registered_variant not in REGISTERED_VARIANTS:
        raise StrategyStagedPipelineError("strategy variant is not preregistered")
    derived = derive_registered_variant(model_config)
    if registered_variant != derived:
        raise StrategyStagedPipelineError(
            "strategy variant label differs from the validated model contract"
        )
    return derived


@dataclass(frozen=True)
class OutcomeFreeRoleFeatureView:
    """One byte-verified role feature sidecar with no outcome capability."""

    role: str
    dataset: str
    texts: tuple[str, ...]
    audio: np.ndarray
    video: np.ndarray
    groups: np.ndarray
    speakers: np.ndarray
    turns: np.ndarray
    protocol_row_ids: np.ndarray
    histories: tuple[tuple[int, ...], ...]
    feature_path: Path
    feature_file_sha256: str
    row_alignment_sha256: str
    manifest_path: Path
    manifest_sha256: str
    preflight_receipt_path: Path
    preflight_receipt_sha256: str
    feature_identity_sha256: str

    @property
    def rows(self) -> int:
        return len(self.texts)


@dataclass(frozen=True)
class VerifiedCurrentCompletionAttestation:
    """Canonical current-only completion and its still-live private lineage."""

    dataset: str
    artifact: IndependentCurrentOnlyArtifact
    artifact_path: Path
    artifact_sha256: str
    completion_receipt_path: Path
    completion_receipt_sha256: str
    fit_artifact_path: Path
    fit_artifact_sha256: str
    fit_producer_receipt_path: Path
    fit_producer_receipt_sha256: str
    producer_file_sha256: str
    producer_source_identity_sha256: str
    history_checkpoint_manifest_sha256: str
    current_only_source_identity_sha256: str
    current_checkpoint_manifest_sha256: str
    production_run_claim_sha256: str
    model_config_sha256: str
    run_config_sha256: str
    model_config_semantic_sha256: str
    run_config_semantic_sha256: str
    source_code_sha256: str
    runtime_environment_sha256: str
    selection_feature_sha256: str
    outer_folds: int
    fit_query_count: int
    selection_query_count: int
    anchor_history_artifact_sha256: str
    anchor_history_completion_receipt_sha256: str
    anchor_history_production_claim_sha256: str
    producer_alignment: CurrentOnlyProducerAlignmentView
    protocol_row_ids: np.ndarray
    fit_protocol_row_ids: np.ndarray
    selection_protocol_row_ids: np.ndarray


@dataclass(frozen=True)
class VerifiedHistoryStrategyCache:
    """History completion plus the fit-only supervision needed by the scorer."""

    attestation: VerifiedHistoryCompletionAttestation
    outcome: HistoryOutcomeFreeView
    supervision: HistoryFitTargetsView
    fit_outcome: object
    fit_state: VerifiedHistoryFitState
    checkpoint_root: Path
    checkpoint_manifest: object
    fit_features: OutcomeFreeRoleFeatureView
    selection_features: OutcomeFreeRoleFeatureView
    model_config_sha256: str
    run_config_sha256: str
    utility_config_sha256: str
    config_sha256: Mapping[str, str]
    code_sha256: Mapping[str, str]
    execution_environment_sha256: str


@dataclass(frozen=True)
class VerifiedStrategyUpstreamState:
    """Only object accepted by the production strategy completion entry point."""

    history: VerifiedHistoryStrategyCache
    current: VerifiedCurrentCompletionAttestation
    full_history_anchor: VerifiedHistoryCompletionAttestation
    full_history_anchor_model_config: CausalBackboneConfig
    upstream_identity_sha256: str


@dataclass(frozen=True)
class OutcomeFreeStrategyPlan:
    method_roster: tuple[str, ...]
    feature_names: tuple[str, ...]
    model_spec_sha256: str
    forward_model_spec_sha256: str
    backward_model_spec_sha256: str
    fit_feature_sha256: str
    selection_feature_sha256: str
    fit_supervision_sha256: str
    fit_cluster_sha256: str
    fit_oof_fold_sha256: str
    score_source_identity_sha256: str
    fit_forward_by_seed: np.ndarray
    fit_backward_by_seed: np.ndarray
    fit_decision_by_seed: np.ndarray
    selection_forward_by_seed: np.ndarray
    selection_backward_by_seed: np.ndarray
    selection_decision_by_seed: np.ndarray
    fit_forward_ensemble: np.ndarray
    fit_backward_ensemble: np.ndarray
    fit_decision_ensemble: np.ndarray
    selection_forward_ensemble: np.ndarray
    selection_backward_ensemble: np.ndarray
    selection_decision_ensemble: np.ndarray
    rule: FrozenCoverageRule
    policy: PreparedPolicyContexts
    forward_score_source_identity_sha256: str
    backward_score_source_identity_sha256: str
    forward_fit_decision_by_seed: np.ndarray
    backward_fit_decision_by_seed: np.ndarray
    forward_selection_decision_by_seed: np.ndarray
    backward_selection_decision_by_seed: np.ndarray
    forward_fit_decision_ensemble: np.ndarray
    backward_fit_decision_ensemble: np.ndarray
    forward_selection_decision_ensemble: np.ndarray
    backward_selection_decision_ensemble: np.ndarray
    forward_rule: FrozenCoverageRule
    backward_rule: FrozenCoverageRule
    forward_policy: PreparedPolicyContexts
    backward_policy: PreparedPolicyContexts
    all_history_contexts: tuple[tuple[int, ...], ...]
    all_history_context_sha256: str


@dataclass(frozen=True)
class CompletedStrategyProduction:
    artifact_path: Path
    artifact_sha256: str
    receipt_path: Path
    receipt_sha256: str
    production_run_claim_sha256: str
    policy_sha256: str


@dataclass(frozen=True)
class VerifiedStrategyCompletionAttestation:
    dataset: str
    registered_variant: str
    artifact_path: Path
    artifact_sha256: str
    receipt_path: Path
    receipt_sha256: str
    production_run_claim_sha256: str
    cross_variant_alignment_sha256: str
    variant_history_artifact_sha256: str
    full_current_anchor_history_artifact_sha256: str
    current_artifact_sha256: str
    method_roster: tuple[str, ...]
    joint_evaluation_roster: tuple[str, ...]
    base_seeds: tuple[int, ...]
    utility_seeds: tuple[int, ...]
    fit_query_count: int
    selection_query_count: int
    fit_task_count: int
    selection_task_count: int


def _single_int(value: np.ndarray, field: str) -> int:
    array = np.asarray(value)
    if array.size != 1 or not np.issubdtype(array.dtype, np.integer):
        raise StrategyStagedPipelineError(f"{field} must contain one integer")
    return int(array.reshape(()))


def _single_bool(value: np.ndarray, field: str) -> bool:
    array = np.asarray(value)
    if array.size != 1 or array.dtype != np.bool_:
        raise StrategyStagedPipelineError(f"{field} must contain one boolean")
    return bool(array.reshape(()))


def _integer_vector(
    value: np.ndarray, field: str, *, unique: bool = False
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise StrategyStagedPipelineError(f"{field} must be a one-dimensional integer array")
    result = array.astype(np.int64, copy=True)
    if np.any(result < 0) or (unique and len(np.unique(result)) != len(result)):
        raise StrategyStagedPipelineError(f"{field} contains invalid row values")
    return result


def _probability(value: np.ndarray, shape: tuple[int, ...], field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.floating):
        raise StrategyStagedPipelineError(f"{field} must have floating shape {shape}")
    result = array.astype(np.float32, copy=True)
    if (
        not np.isfinite(result).all()
        or np.any(result < 0.0)
        or not np.allclose(result.sum(axis=-1), 1.0, rtol=1.0e-5, atol=1.0e-6)
    ):
        raise StrategyStagedPipelineError(f"{field} contains invalid probabilities")
    return result


def _load_npz_exact(path: Path, fields: Sequence[str]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(fields):
                raise StrategyStagedPipelineError(
                    f"private artifact schema changed: {path.name}"
                )
            return {name: np.asarray(archive[name]) for name in archive.files}
    except StrategyStagedPipelineError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise StrategyStagedPipelineError(f"cannot read {path.name}: {error}") from error


def _feature_identity(
    *,
    role: str,
    dataset: str,
    texts: Sequence[str],
    audio: np.ndarray,
    video: np.ndarray,
    groups: np.ndarray,
    speakers: np.ndarray,
    turns: np.ndarray,
    protocol_row_ids: np.ndarray,
    histories: Sequence[Sequence[int]],
    feature_file_sha256: str,
    row_alignment_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "role": role,
            "dataset": dataset,
            "texts_sha256": _array_sha256(np.asarray(texts).astype(str)),
            "audio_sha256": _array_sha256(np.asarray(audio, dtype=np.float32)),
            "video_sha256": _array_sha256(np.asarray(video, dtype=np.float32)),
            "groups_sha256": _array_sha256(np.asarray(groups).astype(str)),
            "speakers_sha256": _array_sha256(np.asarray(speakers).astype(str)),
            "turns_sha256": _array_sha256(np.asarray(turns, dtype=np.int64)),
            "protocol_rows_sha256": _array_sha256(
                np.asarray(protocol_row_ids, dtype=np.int64)
            ),
            "histories_sha256": _canonical_sha256(
                [[int(index) for index in row] for row in histories]
            ),
            "feature_file_sha256": _require_sha256(
                feature_file_sha256, "feature_file_sha256"
            ),
            "row_alignment_sha256": _require_sha256(
                row_alignment_sha256, "row_alignment_sha256"
            ),
        }
    )


def load_outcome_free_role_features(
    *,
    role: str,
    dataset: str,
    feature_path: str | Path,
    manifest_path: str | Path,
    fit_preflight_receipt_path: str | Path,
    expected_fit_preflight_receipt_sha256: str,
) -> OutcomeFreeRoleFeatureView:
    """Materialise exactly one feature sidecar without resolving any outcome file."""

    if role not in {FIT_ROLE, SELECTION_ROLE} or dataset not in _SPECS:
        raise StrategyStagedPipelineError("role/dataset is not an open-role feature source")
    receipt_path = Path(fit_preflight_receipt_path).resolve()
    expected_receipt = _require_sha256(
        expected_fit_preflight_receipt_sha256,
        "expected_fit_preflight_receipt_sha256",
    )
    if _file_sha256(receipt_path) != expected_receipt:
        raise StrategyStagedPipelineError("fit preflight receipt file hash changed")
    receipt = _load_receipt(receipt_path)
    validate_fit_receipt(receipt)
    if receipt.get("dataset") != dataset:
        raise StrategyStagedPipelineError("feature dataset differs from preflight")

    spec = _SPECS[dataset]
    manifest_file = Path(manifest_path).resolve()
    manifest_sha = _file_sha256(manifest_file)
    manifest = _read_manifest_json(manifest_file)
    _validate_manifest_contract(manifest, spec)
    receipt_manifest = cast(Mapping[str, object], receipt["manifest"])
    if (
        manifest_sha != receipt_manifest.get("sha256")
        or manifest_sha != _file_sha256(manifest_file)
    ):
        raise StrategyStagedPipelineError("feature manifest differs from preflight")
    roles = cast(Mapping[str, object], manifest["roles"])
    record_raw = roles[role]
    if not isinstance(record_raw, Mapping):
        raise StrategyStagedPipelineError("feature manifest role record changed")
    feature = Path(feature_path).resolve()
    expected_name = f"features_{role}.npz"
    if feature.name != expected_name or record_raw.get("feature_filename") != expected_name:
        raise StrategyStagedPipelineError("feature path is not the canonical role file")
    feature_sha = _file_sha256(feature)
    # The manifest uses the protocol role name ``base_and_utility_fit`` while
    # the aggregate preflight receipt deliberately exposes that record as
    # ``fit``.  Keep this translation explicit so an E2E verifier cannot drift
    # into indexing a non-existent receipt key.
    receipt_role = "fit" if role == FIT_ROLE else SELECTION_ROLE
    receipt_sidecar = cast(Mapping[str, object], receipt["sidecars"])[receipt_role]
    if (
        not isinstance(receipt_sidecar, Mapping)
        or feature_sha != record_raw.get("feature_sha256")
        or feature_sha != receipt_sidecar.get("feature_sha256")
    ):
        raise StrategyStagedPipelineError("feature file hash differs from frozen lineage")
    row_sha = _require_sha256(
        record_raw.get("row_alignment_sha256"), f"{role}.row_alignment_sha256"
    )
    sidecar_row_sha = _require_sha256(
        receipt_sidecar.get("row_alignment_sha256"),
        f"receipt.{role}.row_alignment_sha256",
    )
    if row_sha != sidecar_row_sha:
        raise StrategyStagedPipelineError("feature row alignment differs from preflight")

    values = _load_npz_exact(feature, spec.feature_fields)
    if dataset == "EmotionTalk":
        if (
            _single_text(values["schema_version"], "schema_version")
            != spec.feature_schema
            or _single_text(values["dataset_id"], "dataset_id") != dataset
            or _single_text(values["role"], "role") != role
            or _single_text(values["split_protocol_id"], "split_protocol_id")
            != "scu_set_exploration_v1"
        ):
            raise StrategyStagedPipelineError("EmotionTalk feature identity changed")
        texts = np.asarray(values["texts"])
        audio = np.asarray(values["audio_features"])
        video = np.asarray(values["video_features"])
        groups = np.asarray(values["opaque_group_hashes"]).astype(str)
        speakers = np.asarray(values["speaker_tokens"]).astype(str)
        turns = np.asarray(values["turn_ids"])
        protocol = np.asarray(values["protocol_row_ids"])
        buckets = np.asarray(values["role_buckets"])
        expected_bucket = (0, 64) if role == FIT_ROLE else (65, 79)
        if (
            buckets.shape != (int(record_raw["rows"]),)
            or np.any((buckets < expected_bucket[0]) | (buckets > expected_bucket[1]))
        ):
            raise StrategyStagedPipelineError("EmotionTalk role bucket changed")
    else:
        if (
            _single_text(values["schema_version"], "schema_version")
            != spec.feature_schema
            or _single_text(values["role"], "role") != role
        ):
            raise StrategyStagedPipelineError("MELD feature identity changed")
        texts = np.asarray(values["utterances"])
        audio = np.asarray(values["audio_mean_std"])
        video = np.asarray(values["video_mean_std"])
        groups = np.asarray(values["dialogue_codes"])
        speakers = np.asarray(values["speaker_codes"])
        turns = np.asarray(values["utterance_order"])
        protocol = np.asarray(values["protocol_row_ids"])
    if _single_text(values["row_alignment_sha256"], "row_alignment_sha256") != row_sha:
        raise StrategyStagedPipelineError("feature payload row alignment changed")
    normalized_record = SelectionFeatureRecord(
        role=role,
        feature_path=feature,
        feature_sha256=feature_sha,
        row_alignment_sha256=row_sha,
        rows=int(record_raw["rows"]),
        groups=int(record_raw[spec.group_count_field]),
        history_eligible_rows=int(record_raw["history_eligible_rows"]),
        audio_dimension=int(record_raw["audio_dimension"]),
        video_dimension=int(record_raw["video_dimension"]),
    )
    _validate_normalized_rows(
        record=normalized_record,  # type: ignore[arg-type]
        texts=texts,
        audio=audio,
        video=video,
        groups=groups,
        speakers=speakers,
        turns=turns,
        protocol_rows=protocol,
        labels=None,
    )
    histories = _strict_histories(groups, speakers, turns)
    identity = _feature_identity(
        role=role,
        dataset=dataset,
        texts=tuple(str(value) for value in texts),
        audio=np.asarray(audio, dtype=np.float32),
        video=np.asarray(video, dtype=np.float32),
        groups=np.asarray(groups).astype(str),
        speakers=np.asarray(speakers).astype(str),
        turns=np.asarray(turns, dtype=np.int64),
        protocol_row_ids=np.asarray(protocol, dtype=np.int64),
        histories=histories,
        feature_file_sha256=feature_sha,
        row_alignment_sha256=row_sha,
    )
    if (
        _file_sha256(receipt_path) != expected_receipt
        or _file_sha256(manifest_file) != manifest_sha
        or _file_sha256(feature) != feature_sha
    ):
        raise StrategyStagedPipelineError("feature lineage changed while materialising")
    return OutcomeFreeRoleFeatureView(
        role=role,
        dataset=dataset,
        texts=tuple(str(value) for value in texts),
        audio=np.asarray(audio, dtype=np.float32).copy(),
        video=np.asarray(video, dtype=np.float32).copy(),
        groups=np.asarray(groups).astype(str),
        speakers=np.asarray(speakers).astype(str),
        turns=np.asarray(turns, dtype=np.int64).copy(),
        protocol_row_ids=np.asarray(protocol, dtype=np.int64).copy(),
        histories=histories,
        feature_path=feature,
        feature_file_sha256=feature_sha,
        row_alignment_sha256=row_sha,
        manifest_path=manifest_file,
        manifest_sha256=manifest_sha,
        preflight_receipt_path=receipt_path,
        preflight_receipt_sha256=expected_receipt,
        feature_identity_sha256=identity,
    )


def _assert_feature_view_unchanged(view: OutcomeFreeRoleFeatureView) -> None:
    observed = _feature_identity(
        role=view.role,
        dataset=view.dataset,
        texts=view.texts,
        audio=view.audio,
        video=view.video,
        groups=view.groups,
        speakers=view.speakers,
        turns=view.turns,
        protocol_row_ids=view.protocol_row_ids,
        histories=view.histories,
        feature_file_sha256=view.feature_file_sha256,
        row_alignment_sha256=view.row_alignment_sha256,
    )
    if (
        observed != view.feature_identity_sha256
        or _file_sha256(view.feature_path) != view.feature_file_sha256
        or _file_sha256(view.manifest_path) != view.manifest_sha256
        or _file_sha256(view.preflight_receipt_path)
        != view.preflight_receipt_sha256
    ):
        raise StrategyStagedPipelineError("outcome-free feature view changed")


def _dummy_fit_view(
    features: OutcomeFreeRoleFeatureView,
    class_order: Sequence[str],
) -> FitRoleView:
    if features.role != FIT_ROLE:
        raise StrategyStagedPipelineError("fit feature capability has the wrong role")
    return FitRoleView(
        dataset=features.dataset,
        label_order=tuple(str(value) for value in class_order),
        texts=features.texts,
        audio=np.asarray(features.audio, dtype=np.float32).copy(),
        video=np.asarray(features.video, dtype=np.float32).copy(),
        labels=np.zeros(features.rows, dtype=np.int64),
        groups=np.asarray(features.groups).copy(),
        speakers=np.asarray(features.speakers).copy(),
        turns=np.asarray(features.turns, dtype=np.int64).copy(),
        protocol_row_ids=np.asarray(features.protocol_row_ids, dtype=np.int64).copy(),
        histories=features.histories,
        array_hashes={},
        contract_sha256=features.feature_identity_sha256,
    )


def _selection_view(features: OutcomeFreeRoleFeatureView) -> SelectionFeatureView:
    if features.role != SELECTION_ROLE:
        raise StrategyStagedPipelineError("selection feature capability has the wrong role")
    return SelectionFeatureView(
        dataset=features.dataset,
        texts=features.texts,
        audio=np.asarray(features.audio, dtype=np.float32).copy(),
        video=np.asarray(features.video, dtype=np.float32).copy(),
        groups=np.asarray(features.groups).copy(),
        speakers=np.asarray(features.speakers).copy(),
        turns=np.asarray(features.turns, dtype=np.int64).copy(),
        protocol_row_ids=np.asarray(features.protocol_row_ids, dtype=np.int64).copy(),
        labels_materialized=False,
    )


_CURRENT_COMPLETE_LINEAGE_KEYS = frozenset(
    {
        "fit_preflight_receipt_sha256",
        "fit_producer_receipt_sha256",
        "fit_protocol_map_sha256",
        "fit_lineage_file_sha256",
        "fit_lineage_source_identity_sha256",
        "fit_probability_artifact_sha256",
        "producer_file_sha256",
        "producer_source_identity_sha256",
        "history_checkpoint_manifest_sha256",
        "current_checkpoint_manifest_sha256",
        "current_only_source_identity_sha256",
        "fit_array_hash_bundle_sha256",
        "fold_assignment_sha256",
        "model_config_sha256",
        "run_config_sha256",
        "model_config_semantic_sha256",
        "run_config_semantic_sha256",
        "source_code_sha256",
        "runtime_environment_sha256",
        "production_run_claim_sha256",
        "selection_feature_sha256",
        "private_current_only_cache_sha256",
    }
)
_CURRENT_COMPLETE_CONTRACT_KEYS = frozenset(
    {
        "protocol",
        "seeds",
        "outer_folds",
        "fit_query_count",
        "selection_query_count",
        "complete_checkpoint_only",
        "producer_execution_mode",
        "production_trainer_attested",
        "one_selection_probability_per_seed_and_query",
        "history_training_items_consumed",
        "history_inference_items_consumed",
        "selection_feature_materialized",
        "selection_label_materialized",
        "selection_label_deserialized",
        "selection_label_file_accessed",
        "evaluate_stage_run",
        "performance_metric_computed",
    }
)
_CURRENT_CACHE_KEYS = frozenset(
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
_CURRENT_FIT_KEYS = frozenset(
    {
        "schema_version",
        "dataset",
        "dataset_label_order",
        "seeds",
        "outer_folds",
        "fit_protocol_row_ids",
        "fit_probability_oof",
        "fit_fold_by_seed_row",
        "fit_lineage_artifact_sha256",
        "fit_lineage_source_identity_sha256",
        "current_only_source_identity_sha256",
        "checkpoint_manifest_sha256",
        "training_protocol",
        "checkpoint_namespace",
        "history_training_items_consumed",
        "history_inference_items_consumed",
        "selection_payload_consumed",
        "heldout_fit_labels_materialized",
        "matrix_fit_probability_oof_sha256",
        "fold_assignment_sha256",
    }
)
_CURRENT_FIT_LINEAGE_KEYS = frozenset(
    {
        "fit_preflight_receipt_sha256",
        "fit_protocol_map_sha256",
        "fit_lineage_file_sha256",
        "fit_lineage_source_identity_sha256",
        "model_config_sha256",
        "run_config_sha256",
        "model_config_semantic_sha256",
        "run_config_semantic_sha256",
        "source_code_sha256",
        "runtime_environment_sha256",
        "production_run_claim_sha256",
        "fold_assignment_sha256",
        "current_only_source_identity_sha256",
        "checkpoint_manifest_sha256",
        "private_fit_artifact_sha256",
    }
)
_CURRENT_FIT_CONTRACT_KEYS = frozenset(
    {
        "producer_execution_mode",
        "production_trainer_attested",
        "seeds",
        "outer_folds",
        "history_training_items_consumed",
        "history_inference_items_consumed",
        "checkpoint_file_count",
        "fit_query_count",
        "one_oof_probability_per_seed_and_fit_query",
        "selection_payload_consumed",
        "heldout_fit_labels_materialized",
        "history_producer_required",
    }
)


def _read_json_mapping(path: Path, field: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StrategyStagedPipelineError(f"cannot read {field}: {error}") from error
    if not isinstance(value, dict):
        raise StrategyStagedPipelineError(f"{field} root must be a mapping")
    return value


def _validated_class_order(value: np.ndarray, field: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise StrategyStagedPipelineError(f"{field} must be a string vector")
    result = tuple(str(item) for item in array)
    if len(result) < 2 or len(set(result)) != len(result):
        raise StrategyStagedPipelineError(f"{field} is empty or ambiguous")
    return result


def _assert_history_attestation_unchanged(
    value: VerifiedHistoryCompletionAttestation,
) -> VerifiedHistoryCompletionAttestation:
    observed = verify_history_completion_production_attestation(
        value.artifact_path,
        value.completion_receipt_path,
        value.completion_receipt_sha256,
    )
    if observed != value:
        raise StrategyStagedPipelineError("history production attestation changed")
    return observed


def verify_current_only_completion_production_attestation(
    artifact_path: str | Path,
    completion_receipt_path: str | Path,
    expected_completion_receipt_sha256: str,
    *,
    history_attestation: VerifiedHistoryCompletionAttestation,
    producer_alignment: CurrentOnlyProducerAlignmentView,
) -> VerifiedCurrentCompletionAttestation:
    """Verify a canonical production current-only cache without a producer cache.

    The current-only producer cache contains outcome-derived arrays, so this gate
    deliberately does not accept its path.  Instead it verifies the completion
    receipt, canonical fit cache/receipt, private run claim, current checkpoints,
    and the history source/checkpoint identities already proven by the supplied
    history completion attestation.
    """

    if not isinstance(history_attestation, VerifiedHistoryCompletionAttestation):
        raise StrategyStagedPipelineError("current verification requires history attestation")
    history = _assert_history_attestation_unchanged(history_attestation)
    if not isinstance(producer_alignment, CurrentOnlyProducerAlignmentView):
        raise StrategyStagedPipelineError(
            "current verification requires the outcome-free producer alignment view"
        )
    artifact_file = Path(artifact_path).resolve()
    receipt_file = Path(completion_receipt_path).resolve()
    root = artifact_file.parent
    paths = current_only_private_paths(root)
    if (
        artifact_file != paths["complete_artifact"]
        or receipt_file != paths["complete_receipt"]
        or root == Path(root.anchor)
        or not root.is_absolute()
    ):
        raise StrategyStagedPipelineError(
            "current completion must use one canonical private root"
        )
    expected_receipt_sha = _require_sha256(
        expected_completion_receipt_sha256,
        "expected_completion_receipt_sha256",
    )
    if _file_sha256(receipt_file) != expected_receipt_sha:
        raise StrategyStagedPipelineError("current completion receipt file hash changed")
    receipt = _read_json_mapping(receipt_file, "current completion receipt")
    if set(receipt) != {
        "schema_version",
        "status",
        "dataset",
        "claim_boundary",
        "lineage",
        "completion_contract",
        "public_artifact_policy",
    }:
        raise StrategyStagedPipelineError("current completion receipt schema changed")
    if (
        receipt.get("schema_version") != CURRENT_ONLY_COMPLETION_RECEIPT_SCHEMA
        or receipt.get("status")
        != "complete_selection_probability_cache_not_performance_evidence"
        or receipt.get("claim_boundary")
        != (
            "Fit OOF plus model-selection feature-only inference; no model-selection "
            "label and no performance metric were consumed."
        )
        or receipt.get("dataset") not in {"EmotionTalk", "MELD"}
    ):
        raise StrategyStagedPipelineError("current completion is not production evidence")
    lineage = receipt.get("lineage")
    contract = receipt.get("completion_contract")
    if (
        not isinstance(lineage, dict)
        or set(lineage) != set(_CURRENT_COMPLETE_LINEAGE_KEYS)
        or not isinstance(contract, dict)
        or set(contract) != set(_CURRENT_COMPLETE_CONTRACT_KEYS)
    ):
        raise StrategyStagedPipelineError("current completion nested schema changed")
    if receipt.get("public_artifact_policy") != {
        "aggregate_only": True,
        "contains_row_level_values": False,
        "contains_private_paths": False,
        "contains_performance_metrics": False,
    }:
        raise StrategyStagedPipelineError("current public artifact policy changed")
    required_true = {
        "complete_checkpoint_only",
        "production_trainer_attested",
        "one_selection_probability_per_seed_and_query",
        "selection_feature_materialized",
    }
    required_false = {
        "selection_label_materialized",
        "selection_label_deserialized",
        "selection_label_file_accessed",
        "evaluate_stage_run",
        "performance_metric_computed",
    }
    if (
        contract.get("protocol") != CURRENT_ONLY_COMPLETION_PROTOCOL
        or contract.get("seeds") != list(EXPECTED_SEEDS)
        or contract.get("producer_execution_mode")
        != CURRENT_ONLY_PRODUCTION_EXECUTION_MODE
        or contract.get("history_training_items_consumed") != 0
        or contract.get("history_inference_items_consumed") != 0
        or any(contract.get(name) is not True for name in required_true)
        or any(contract.get(name) is not False for name in required_false)
    ):
        raise StrategyStagedPipelineError("current completion isolation contract changed")
    outer_folds = contract.get("outer_folds")
    fit_count = contract.get("fit_query_count")
    selection_count = contract.get("selection_query_count")
    if (
        type(outer_folds) is not int
        or int(outer_folds) < 2
        or type(fit_count) is not int
        or int(fit_count) < 1
        or type(selection_count) is not int
        or int(selection_count) < 1
    ):
        raise StrategyStagedPipelineError("current completion counts changed")
    lineage_sha = {
        name: _require_sha256(lineage[name], name)
        for name in _CURRENT_COMPLETE_LINEAGE_KEYS
    }
    if (
        lineage_sha["producer_source_identity_sha256"]
        != history.source_identity_sha256
        or lineage_sha["history_checkpoint_manifest_sha256"]
        != history.checkpoint_manifest_sha256
    ):
        raise StrategyStagedPipelineError(
            "current completion is bound to another history producer"
        )
    if (
        producer_alignment.dataset != receipt["dataset"]
        or producer_alignment.seeds != EXPECTED_SEEDS
        or producer_alignment.producer_file_sha256
        != lineage_sha["producer_file_sha256"]
        or producer_alignment.source_identity_sha256
        != lineage_sha["producer_source_identity_sha256"]
        or producer_alignment.checkpoint_manifest_sha256
        != lineage_sha["history_checkpoint_manifest_sha256"]
    ):
        raise StrategyStagedPipelineError("current producer alignment lineage changed")
    aligned_protocol = _integer_vector(
        producer_alignment.protocol_row_ids,
        "producer_alignment.protocol_row_ids",
        unique=True,
    )
    aligned_fit_query = _integer_vector(
        producer_alignment.fit_query_indices,
        "producer_alignment.fit_query_indices",
        unique=True,
    )
    aligned_selection_query = _integer_vector(
        producer_alignment.selection_query_indices,
        "producer_alignment.selection_query_indices",
        unique=True,
    )
    if (
        np.any(aligned_fit_query >= len(aligned_protocol))
        or np.any(aligned_selection_query >= len(aligned_protocol))
        or set(aligned_fit_query.tolist()) & set(aligned_selection_query.tolist())
        or set(aligned_fit_query.tolist()) | set(aligned_selection_query.tolist())
        != set(range(len(aligned_protocol)))
    ):
        raise StrategyStagedPipelineError("current producer protocol partition changed")

    artifact_sha = _file_sha256(artifact_file)
    if artifact_sha != lineage_sha["private_current_only_cache_sha256"]:
        raise StrategyStagedPipelineError("current-only artifact hash changed")
    values = _load_npz_exact(artifact_file, _CURRENT_CACHE_KEYS)
    if (
        _single_text(values["schema_version"], "schema_version")
        != CURRENT_ONLY_CACHE_SCHEMA
        or _single_text(values["dataset"], "dataset") != receipt["dataset"]
    ):
        raise StrategyStagedPipelineError("current-only artifact identity changed")
    class_order = _validated_class_order(
        values["dataset_label_order"], "dataset_label_order"
    )
    if producer_alignment.label_order != class_order:
        raise StrategyStagedPipelineError("current producer class order changed")
    seeds = tuple(
        int(value)
        for value in _integer_vector(values["seeds"], "seeds", unique=True)
    )
    if seeds != EXPECTED_SEEDS:
        raise StrategyStagedPipelineError("current-only registered seeds changed")
    fit_query = _integer_vector(
        values["fit_query_indices"], "fit_query_indices", unique=True
    )
    selection_query = _integer_vector(
        values["selection_query_indices"], "selection_query_indices", unique=True
    )
    fit_cluster = _integer_vector(values["fit_cluster_codes"], "fit_cluster_codes")
    selection_cluster = _integer_vector(
        values["selection_cluster_codes"], "selection_cluster_codes"
    )
    if (
        len(fit_query) != int(fit_count)
        or len(selection_query) != int(selection_count)
        or fit_cluster.shape != fit_query.shape
        or selection_cluster.shape != selection_query.shape
        or set(fit_query.tolist()) & set(selection_query.tolist())
        or len(np.unique(fit_cluster)) < 2
        or len(np.unique(selection_cluster)) < 2
        or not np.array_equal(fit_query, producer_alignment.fit_query_indices)
        or not np.array_equal(
            selection_query, producer_alignment.selection_query_indices
        )
        or not np.array_equal(fit_cluster, producer_alignment.fit_cluster_codes)
        or not np.array_equal(
            selection_cluster, producer_alignment.selection_cluster_codes
        )
    ):
        raise StrategyStagedPipelineError("current-only row alignment changed")
    classes = len(class_order)
    fit_probability = _probability(
        values["fit_probability_oof"],
        (len(seeds), len(fit_query), classes),
        "fit_probability_oof",
    )
    selection_probability = _probability(
        values["selection_probability_fold_ensemble"],
        (len(seeds), len(selection_query), classes),
        "selection_probability_fold_ensemble",
    )
    for name, array in (
        ("fit_probability_oof", values["fit_probability_oof"]),
        (
            "selection_probability_fold_ensemble",
            values["selection_probability_fold_ensemble"],
        ),
    ):
        field = f"matrix_{name}_sha256"
        if _require_sha256(_single_text(values[field], field), field) != _array_sha256(
            np.asarray(array)
        ):
            raise StrategyStagedPipelineError(f"current-only matrix hash changed: {name}")
    producer_identity = _require_sha256(
        _single_text(
            values["producer_source_identity_sha256"],
            "producer_source_identity_sha256",
        ),
        "producer_source_identity_sha256",
    )
    current_identity = _require_sha256(
        _single_text(
            values["current_only_source_identity_sha256"],
            "current_only_source_identity_sha256",
        ),
        "current_only_source_identity_sha256",
    )
    history_checkpoint = _require_sha256(
        _single_text(
            values["history_backbone_checkpoint_manifest_sha256"],
            "history_backbone_checkpoint_manifest_sha256",
        ),
        "history_backbone_checkpoint_manifest_sha256",
    )
    current_checkpoint = _require_sha256(
        _single_text(values["checkpoint_manifest_sha256"], "checkpoint_manifest_sha256"),
        "checkpoint_manifest_sha256",
    )
    if (
        producer_identity != lineage_sha["producer_source_identity_sha256"]
        or current_identity != lineage_sha["current_only_source_identity_sha256"]
        or history_checkpoint != lineage_sha["history_checkpoint_manifest_sha256"]
        or current_checkpoint != lineage_sha["current_checkpoint_manifest_sha256"]
        or current_identity == producer_identity
        or current_checkpoint == history_checkpoint
        or _single_text(values["training_protocol"], "training_protocol")
        != INDEPENDENT_CURRENT_ONLY_PROTOCOL
        or _single_text(values["checkpoint_namespace"], "checkpoint_namespace")
        != "independent_current_only"
        or _single_int(
            values["history_training_items_consumed"],
            "history_training_items_consumed",
        )
        != 0
        or _single_int(
            values["history_inference_items_consumed"],
            "history_inference_items_consumed",
        )
        != 0
    ):
        raise StrategyStagedPipelineError("current-only independence contract changed")
    expected_independence = _canonical_sha256(
        current_only_independence_attestation_payload(values)
    )
    observed_independence = _require_sha256(
        _single_text(
            values["independence_attestation_sha256"],
            "independence_attestation_sha256",
        ),
        "independence_attestation_sha256",
    )
    if expected_independence != observed_independence:
        raise StrategyStagedPipelineError("current-only independence attestation changed")
    artifact_view = IndependentCurrentOnlyArtifact(
        dataset=str(receipt["dataset"]),
        label_order=class_order,
        seeds=seeds,
        fit_query_indices=fit_query,
        selection_query_indices=selection_query,
        fit_cluster_codes=fit_cluster,
        selection_cluster_codes=selection_cluster,
        fit_probability=fit_probability.astype(np.float64),
        selection_probability=selection_probability.astype(np.float64),
        producer_source_identity_sha256=producer_identity,
        source_identity_sha256=current_identity,
        checkpoint_manifest_sha256=current_checkpoint,
        independence_attestation_sha256=observed_independence,
    )

    fit_artifact_path = paths["fit_artifact"].resolve()
    fit_receipt_path = paths["fit_receipt"].resolve()
    fit_artifact_sha = _file_sha256(fit_artifact_path)
    fit_receipt_sha = _file_sha256(fit_receipt_path)
    if (
        fit_artifact_sha != lineage_sha["fit_probability_artifact_sha256"]
        or fit_receipt_sha != lineage_sha["fit_producer_receipt_sha256"]
    ):
        raise StrategyStagedPipelineError("current fit artifact/receipt hash changed")
    fit_receipt = _read_json_mapping(fit_receipt_path, "current fit receipt")
    if set(fit_receipt) != {
        "schema_version",
        "status",
        "dataset",
        "claim_boundary",
        "lineage",
        "training_contract",
        "public_artifact_policy",
    }:
        raise StrategyStagedPipelineError("current fit receipt schema changed")
    fit_lineage = fit_receipt.get("lineage")
    fit_contract = fit_receipt.get("training_contract")
    if (
        fit_receipt.get("schema_version")
        != CURRENT_ONLY_FIT_PRODUCER_RECEIPT_SCHEMA
        or fit_receipt.get("status")
        != "production_independent_current_only_fit_oof_complete_not_performance_evidence"
        or fit_receipt.get("dataset") != receipt["dataset"]
        or not isinstance(fit_lineage, dict)
        or set(fit_lineage) != set(_CURRENT_FIT_LINEAGE_KEYS)
        or not isinstance(fit_contract, dict)
        or set(fit_contract) != set(_CURRENT_FIT_CONTRACT_KEYS)
    ):
        raise StrategyStagedPipelineError("current fit receipt is not canonical production")
    fit_lineage_sha = {
        name: _require_sha256(fit_lineage[name], name)
        for name in _CURRENT_FIT_LINEAGE_KEYS
    }
    common = {
        "fit_preflight_receipt_sha256",
        "fit_protocol_map_sha256",
        "fit_lineage_file_sha256",
        "fit_lineage_source_identity_sha256",
        "model_config_sha256",
        "run_config_sha256",
        "model_config_semantic_sha256",
        "run_config_semantic_sha256",
        "source_code_sha256",
        "runtime_environment_sha256",
        "production_run_claim_sha256",
        "fold_assignment_sha256",
        "current_only_source_identity_sha256",
    }
    if (
        any(fit_lineage_sha[name] != lineage_sha[name] for name in common)
        or fit_lineage_sha["checkpoint_manifest_sha256"] != current_checkpoint
        or fit_lineage_sha["private_fit_artifact_sha256"] != fit_artifact_sha
        or fit_contract.get("producer_execution_mode")
        != CURRENT_ONLY_PRODUCTION_EXECUTION_MODE
        or fit_contract.get("production_trainer_attested") is not True
        or fit_contract.get("seeds") != list(EXPECTED_SEEDS)
        or fit_contract.get("outer_folds") != int(outer_folds)
        or fit_contract.get("fit_query_count") != int(fit_count)
        or fit_contract.get("history_training_items_consumed") != 0
        or fit_contract.get("history_inference_items_consumed") != 0
        or fit_contract.get("selection_payload_consumed") is not False
        or fit_contract.get("heldout_fit_labels_materialized") is not False
        or fit_contract.get("history_producer_required") is not False
    ):
        raise StrategyStagedPipelineError("current fit receipt lineage changed")
    if fit_receipt.get("public_artifact_policy") != {
        "aggregate_only": True,
        "contains_row_level_values": False,
        "contains_private_paths": False,
        "contains_performance_metrics": False,
    }:
        raise StrategyStagedPipelineError("current fit public policy changed")

    fit_values = _load_npz_exact(fit_artifact_path, _CURRENT_FIT_KEYS)
    fit_class_order = _validated_class_order(
        fit_values["dataset_label_order"], "fit.dataset_label_order"
    )
    fit_seeds = tuple(
        int(value)
        for value in _integer_vector(fit_values["seeds"], "fit.seeds", unique=True)
    )
    fit_protocol = _integer_vector(
        fit_values["fit_protocol_row_ids"], "fit_protocol_row_ids", unique=True
    )
    fit_fold = np.asarray(fit_values["fit_fold_by_seed_row"])
    fit_probability_local = _probability(
        fit_values["fit_probability_oof"],
        (len(EXPECTED_SEEDS), int(fit_count), len(class_order)),
        "fit bootstrap probability",
    )
    if (
        _single_text(fit_values["schema_version"], "fit.schema_version")
        != CURRENT_ONLY_FIT_BOOTSTRAP_ARTIFACT_SCHEMA
        or _single_text(fit_values["dataset"], "fit.dataset") != receipt["dataset"]
        or fit_class_order != class_order
        or fit_seeds != EXPECTED_SEEDS
        or _single_int(fit_values["outer_folds"], "fit.outer_folds")
        != int(outer_folds)
        or len(fit_protocol) != int(fit_count)
        or fit_fold.shape != (len(EXPECTED_SEEDS), int(fit_count))
        or not np.issubdtype(fit_fold.dtype, np.integer)
        or np.any((fit_fold < 0) | (fit_fold >= int(outer_folds)))
        or _array_sha256(fit_fold)
        != _require_sha256(
            _single_text(fit_values["fold_assignment_sha256"], "fold_assignment"),
            "fold_assignment_sha256",
        )
        or _array_sha256(np.asarray(fit_values["fit_probability_oof"]))
        != _require_sha256(
            _single_text(
                fit_values["matrix_fit_probability_oof_sha256"],
                "matrix_fit_probability_oof_sha256",
            ),
            "matrix_fit_probability_oof_sha256",
        )
        or _single_text(
            fit_values["current_only_source_identity_sha256"],
            "fit.current_only_source_identity_sha256",
        )
        != current_identity
        or _single_text(
            fit_values["checkpoint_manifest_sha256"],
            "fit.checkpoint_manifest_sha256",
        )
        != current_checkpoint
        or _single_text(fit_values["training_protocol"], "fit.training_protocol")
        != INDEPENDENT_CURRENT_ONLY_PROTOCOL
        or _single_text(fit_values["checkpoint_namespace"], "fit.checkpoint_namespace")
        != "independent_current_only"
        or _single_int(
            fit_values["history_training_items_consumed"],
            "fit.history_training_items_consumed",
        )
        != 0
        or _single_int(
            fit_values["history_inference_items_consumed"],
            "fit.history_inference_items_consumed",
        )
        != 0
        or _single_bool(
            fit_values["selection_payload_consumed"],
            "fit.selection_payload_consumed",
        )
        or _single_bool(
            fit_values["heldout_fit_labels_materialized"],
            "fit.heldout_fit_labels_materialized",
        )
    ):
        raise StrategyStagedPipelineError("current fit artifact lineage changed")
    del fit_probability_local

    try:
        manifest = build_checkpoint_manifest(
            paths["checkpoint"], seeds=EXPECTED_SEEDS, outer_folds=int(outer_folds)
        )
        verify_checkpoint_manifest(paths["checkpoint"], manifest)
    except (OSError, ValueError) as error:
        raise StrategyStagedPipelineError(
            f"current checkpoint manifest is invalid: {error}"
        ) from error
    if (
        manifest.manifest_sha256 != current_checkpoint
        or fit_contract.get("checkpoint_file_count") != len(manifest.records)
    ):
        raise StrategyStagedPipelineError("current checkpoint manifest changed")
    claim_sha = lineage_sha["production_run_claim_sha256"]
    claim_path = paths["claim"]
    claim = _read_json_mapping(claim_path, "current private claim")
    if claim != {
        "schema_version": CURRENT_ONLY_PRIVATE_CLAIM_SCHEMA,
        "status": "claimed_for_single_lineage_interruptible_fit",
        "production_claim_sha256": claim_sha,
    }:
        raise StrategyStagedPipelineError("current private run claim changed")
    watched = {
        artifact_file: artifact_sha,
        receipt_file: expected_receipt_sha,
        fit_artifact_path: fit_artifact_sha,
        fit_receipt_path: fit_receipt_sha,
    }
    if any(_file_sha256(path) != digest for path, digest in watched.items()):
        raise StrategyStagedPipelineError("current production changed while attesting")
    _assert_history_attestation_unchanged(history)
    verify_checkpoint_manifest(paths["checkpoint"], manifest)
    return VerifiedCurrentCompletionAttestation(
        dataset=str(receipt["dataset"]),
        artifact=artifact_view,
        artifact_path=artifact_file,
        artifact_sha256=artifact_sha,
        completion_receipt_path=receipt_file,
        completion_receipt_sha256=expected_receipt_sha,
        fit_artifact_path=fit_artifact_path,
        fit_artifact_sha256=fit_artifact_sha,
        fit_producer_receipt_path=fit_receipt_path,
        fit_producer_receipt_sha256=fit_receipt_sha,
        producer_file_sha256=lineage_sha["producer_file_sha256"],
        producer_source_identity_sha256=producer_identity,
        history_checkpoint_manifest_sha256=history_checkpoint,
        current_only_source_identity_sha256=current_identity,
        current_checkpoint_manifest_sha256=current_checkpoint,
        production_run_claim_sha256=claim_sha,
        model_config_sha256=lineage_sha["model_config_sha256"],
        run_config_sha256=lineage_sha["run_config_sha256"],
        model_config_semantic_sha256=lineage_sha["model_config_semantic_sha256"],
        run_config_semantic_sha256=lineage_sha["run_config_semantic_sha256"],
        source_code_sha256=lineage_sha["source_code_sha256"],
        runtime_environment_sha256=lineage_sha["runtime_environment_sha256"],
        selection_feature_sha256=lineage_sha["selection_feature_sha256"],
        outer_folds=int(outer_folds),
        fit_query_count=int(fit_count),
        selection_query_count=int(selection_count),
        anchor_history_artifact_sha256=history.artifact_sha256,
        anchor_history_completion_receipt_sha256=(
            history.completion_receipt_sha256
        ),
        anchor_history_production_claim_sha256=(
            history.production_run_claim_sha256
        ),
        producer_alignment=producer_alignment,
        protocol_row_ids=np.asarray(
            producer_alignment.protocol_row_ids, dtype=np.int64
        ).copy(),
        fit_protocol_row_ids=np.asarray(
            producer_alignment.protocol_row_ids[fit_query], dtype=np.int64
        ).copy(),
        selection_protocol_row_ids=np.asarray(
            producer_alignment.protocol_row_ids[selection_query], dtype=np.int64
        ).copy(),
    )


def _assert_current_attestation_unchanged(
    value: VerifiedCurrentCompletionAttestation,
    history: VerifiedHistoryCompletionAttestation,
) -> VerifiedCurrentCompletionAttestation:
    observed = verify_current_only_completion_production_attestation(
        value.artifact_path,
        value.completion_receipt_path,
        value.completion_receipt_sha256,
        history_attestation=history,
        producer_alignment=value.producer_alignment,
    )
    scalar_fields = (
        "dataset",
        "artifact_path",
        "artifact_sha256",
        "completion_receipt_path",
        "completion_receipt_sha256",
        "fit_artifact_path",
        "fit_artifact_sha256",
        "fit_producer_receipt_path",
        "fit_producer_receipt_sha256",
        "producer_file_sha256",
        "producer_source_identity_sha256",
        "history_checkpoint_manifest_sha256",
        "current_only_source_identity_sha256",
        "current_checkpoint_manifest_sha256",
        "production_run_claim_sha256",
        "model_config_sha256",
        "run_config_sha256",
        "model_config_semantic_sha256",
        "run_config_semantic_sha256",
        "source_code_sha256",
        "runtime_environment_sha256",
        "selection_feature_sha256",
        "outer_folds",
        "fit_query_count",
        "selection_query_count",
        "anchor_history_artifact_sha256",
        "anchor_history_completion_receipt_sha256",
        "anchor_history_production_claim_sha256",
    )
    if any(getattr(observed, name) != getattr(value, name) for name in scalar_fields):
        raise StrategyStagedPipelineError("current production attestation changed")
    for name in (
        "protocol_row_ids",
        "fit_protocol_row_ids",
        "selection_protocol_row_ids",
    ):
        if not np.array_equal(getattr(observed, name), getattr(value, name)):
            raise StrategyStagedPipelineError("current producer row alignment changed")
    return observed


def verify_history_strategy_cache(
    *,
    history_attestation: VerifiedHistoryCompletionAttestation,
    fit_features: OutcomeFreeRoleFeatureView,
    selection_features: OutcomeFreeRoleFeatureView,
) -> VerifiedHistoryStrategyCache:
    """Open only the attested history cache and canonical fit supervision."""

    history = _assert_history_attestation_unchanged(history_attestation)
    _assert_feature_view_unchanged(fit_features)
    _assert_feature_view_unchanged(selection_features)
    if (
        fit_features.role != FIT_ROLE
        or selection_features.role != SELECTION_ROLE
        or fit_features.dataset != history.dataset
        or selection_features.dataset != history.dataset
        or fit_features.preflight_receipt_sha256
        != selection_features.preflight_receipt_sha256
        or fit_features.preflight_receipt_path
        != selection_features.preflight_receipt_path
    ):
        raise StrategyStagedPipelineError("history strategy features cross a role/dataset")
    completion = _read_json_mapping(
        history.completion_receipt_path, "history completion receipt"
    )
    lineage = completion.get("lineage")
    contract = completion.get("completion_contract")
    if not isinstance(lineage, Mapping) or not isinstance(contract, Mapping):
        raise StrategyStagedPipelineError("history completion lacks lineage/contract")
    if (
        lineage.get("fit_preflight_receipt_sha256")
        != fit_features.preflight_receipt_sha256
        or lineage.get("selection_feature_sha256")
        != selection_features.feature_file_sha256
        or contract.get("fit_query_count") != fit_features.rows
        or contract.get("selection_query_count") != selection_features.rows
    ):
        raise StrategyStagedPipelineError("history completion feature lineage changed")
    preflight = _load_receipt(fit_features.preflight_receipt_path)
    validate_fit_receipt(preflight)
    sidecars = preflight.get("sidecars")
    if not isinstance(sidecars, Mapping):
        raise StrategyStagedPipelineError("fit preflight sidecar lineage changed")
    fit_sidecar = sidecars.get("fit")
    selection_sidecar = sidecars.get(SELECTION_ROLE)
    if (
        not isinstance(fit_sidecar, Mapping)
        or not isinstance(selection_sidecar, Mapping)
        or fit_sidecar.get("feature_sha256") != fit_features.feature_file_sha256
        or selection_sidecar.get("feature_sha256")
        != selection_features.feature_file_sha256
    ):
        raise StrategyStagedPipelineError("feature files differ from fit preflight")

    root = history.artifact_path.parent
    paths = _production_private_paths(root)
    outer_folds = contract.get("outer_folds")
    if type(outer_folds) is not int or int(outer_folds) < 2:
        raise StrategyStagedPipelineError("history outer-fold count changed")
    try:
        checkpoint_manifest = build_checkpoint_manifest(
            paths["checkpoint"], seeds=EXPECTED_SEEDS, outer_folds=int(outer_folds)
        )
        verify_checkpoint_manifest(paths["checkpoint"], checkpoint_manifest)
    except (OSError, ValueError) as error:
        raise StrategyStagedPipelineError(
            f"history checkpoint manifest is invalid: {error}"
        ) from error
    if checkpoint_manifest.manifest_sha256 != history.checkpoint_manifest_sha256:
        raise StrategyStagedPipelineError("history checkpoint lineage changed")
    completed_values = _load_npz_exact(history.artifact_path, _COMPLETE_OUTCOME_KEYS)
    if (
        _single_text(completed_values["schema_version"], "schema_version")
        != HISTORY_COMPLETE_OUTCOME_SCHEMA
        or _single_text(completed_values["dataset"], "dataset") != history.dataset
    ):
        raise StrategyStagedPipelineError("history completion cache identity changed")
    class_order = _validated_class_order(
        completed_values["dataset_label_order"], "dataset_label_order"
    )
    fit_view = _dummy_fit_view(fit_features, class_order)
    selection_view = _selection_view(selection_features)
    fit_outcome = load_history_fit_outcome_view(
        paths["fit_outcome"],
        fit=fit_view,
        checkpoint_manifest=checkpoint_manifest,
    )
    supervision = load_history_fit_targets_view(
        paths["fit_targets"],
        expected_fit_outcome_sha256=fit_outcome.artifact_sha256,
        expected_source_identity_sha256=fit_outcome.source_identity_sha256,
        expected_task_sha256=fit_outcome.tasks.task_sha256,
    )
    if (
        supervision.artifact_sha256 != lineage.get("fit_targets_artifact_sha256")
        or fit_outcome.artifact_sha256 != lineage.get("fit_outcome_artifact_sha256")
        or fit_outcome.source_identity_sha256 != history.source_identity_sha256
    ):
        raise StrategyStagedPipelineError("history canonical fit cache lineage changed")
    fit_receipt = _read_json_mapping(
        history.fit_producer_receipt_path, "history fit producer receipt"
    )
    fit_lineage = fit_receipt.get("lineage")
    if not isinstance(fit_lineage, Mapping):
        raise StrategyStagedPipelineError("history fit receipt lacks lineage")
    config_sha = fit_lineage.get("config_sha256")
    code_sha = fit_lineage.get("code_sha256")
    if not isinstance(config_sha, Mapping) or not isinstance(code_sha, Mapping):
        raise StrategyStagedPipelineError("history fit code/config lineage changed")
    normalized_config: dict[str, str] = {}
    normalized_code: dict[str, str] = {}
    config_casefolded: set[str] = set()
    code_casefolded: set[str] = set()
    for raw_name, value in sorted(config_sha.items()):
        name = str(raw_name)
        if (
            not name
            or not name[0].isalnum()
            or len(name) > 128
            or any(
                not (character.isalnum() or character in "_.-")
                for character in name
            )
            or name.casefold() in config_casefolded
        ):
            raise StrategyStagedPipelineError(
                "history fit config lineage contains an unsafe name"
            )
        config_casefolded.add(name.casefold())
        normalized_config[name] = _require_sha256(
            value, f"config_sha256.{name}"
        )
    for raw_name, value in sorted(code_sha.items()):
        try:
            name = _canonical_production_source_key(raw_name)
        except ValueError as error:
            raise StrategyStagedPipelineError(
                "history fit code lineage contains an unsafe source path"
            ) from error
        if name.casefold() in code_casefolded:
            raise StrategyStagedPipelineError(
                "history fit code lineage contains duplicate source paths"
            )
        code_casefolded.add(name.casefold())
        normalized_code[name] = _require_sha256(value, f"code_sha256.{name}")
    execution_environment = _require_sha256(
        fit_lineage.get("execution_environment_sha256"),
        "execution_environment_sha256",
    )
    state = VerifiedHistoryFitState(
        fit_outcome_path=paths["fit_outcome"].resolve(),
        fit_outcome_sha256=fit_outcome.artifact_sha256,
        fit_targets_path=paths["fit_targets"].resolve(),
        fit_targets_sha256=supervision.artifact_sha256,
        fit_receipt_path=history.fit_producer_receipt_path,
        fit_receipt_sha256=history.fit_producer_receipt_sha256,
        fit_preflight_receipt_sha256=fit_features.preflight_receipt_sha256,
        selection_feature_sha256=selection_features.feature_file_sha256,
        checkpoint_manifest=checkpoint_manifest,
        source_identity_sha256=history.source_identity_sha256,
        fit_outcome=fit_outcome,
        production_trainer=True,
        private_output_root=root,
        execution_environment_sha256=execution_environment,
        production_run_claim_sha256=history.production_run_claim_sha256,
    )
    try:
        completed = _outcome_free_view_from_values(
            completed_values,
            fit=fit_view,
            selection=selection_view,
            state=state,
            artifact_sha256=history.artifact_sha256,
        )
    except ValueError as error:
        raise StrategyStagedPipelineError(
            f"history outcome-free cache validation failed: {error}"
        ) from error
    if (
        completed.artifact_sha256 != history.artifact_sha256
        or completed.fit_targets_artifact_sha256 != supervision.artifact_sha256
        or completed.checkpoint_manifest_sha256
        != history.checkpoint_manifest_sha256
        or tuple(completed.seeds) != EXPECTED_SEEDS
    ):
        raise StrategyStagedPipelineError("history outcome-free cache lineage changed")
    _assert_feature_view_unchanged(fit_features)
    _assert_feature_view_unchanged(selection_features)
    _assert_history_attestation_unchanged(history)
    return VerifiedHistoryStrategyCache(
        attestation=history,
        outcome=completed,
        supervision=supervision,
        fit_outcome=fit_outcome,
        fit_state=state,
        checkpoint_root=paths["checkpoint"].resolve(),
        checkpoint_manifest=checkpoint_manifest,
        fit_features=fit_features,
        selection_features=selection_features,
        model_config_sha256=fit_outcome.model_config_sha256,
        run_config_sha256=fit_outcome.run_config_sha256,
        utility_config_sha256=fit_outcome.utility_config_sha256,
        config_sha256=normalized_config,
        code_sha256=normalized_code,
        execution_environment_sha256=execution_environment,
    )


def verify_strategy_upstream_state(
    *,
    history_attestation: VerifiedHistoryCompletionAttestation,
    current_attestation: VerifiedCurrentCompletionAttestation,
    full_history_anchor_attestation: VerifiedHistoryCompletionAttestation,
    full_history_anchor_model_config: CausalBackboneConfig,
    fit_features: OutcomeFreeRoleFeatureView,
    selection_features: OutcomeFreeRoleFeatureView,
) -> VerifiedStrategyUpstreamState:
    """Create the only upstream capability accepted by production completion."""

    history_cache = verify_history_strategy_cache(
        history_attestation=history_attestation,
        fit_features=fit_features,
        selection_features=selection_features,
    )
    full_anchor = _assert_history_attestation_unchanged(
        full_history_anchor_attestation
    )
    _require_registered_variant_matches_model_config(
        "full", full_history_anchor_model_config
    )
    if _config_sha256(full_history_anchor_model_config) != full_anchor.model_config_sha256:
        raise StrategyStagedPipelineError(
            "independent current-only anchor is not the attested full model contract"
        )
    current = _assert_current_attestation_unchanged(
        current_attestation, full_anchor
    )
    outcome = history_cache.outcome
    if (
        current.dataset != outcome.dataset
        or current.artifact.label_order != outcome.label_order
        or current.artifact.seeds != outcome.seeds
        or current.fit_query_count != len(outcome.fit_protocol_row_ids)
        or current.selection_query_count != len(outcome.selection_protocol_row_ids)
        or current.selection_feature_sha256
        != selection_features.feature_file_sha256
        or set(current.fit_protocol_row_ids.tolist())
        != set(fit_features.protocol_row_ids.tolist())
        or set(current.selection_protocol_row_ids.tolist())
        != set(selection_features.protocol_row_ids.tolist())
        or full_anchor.dataset != outcome.dataset
        or current.producer_source_identity_sha256
        != full_anchor.source_identity_sha256
        or current.history_checkpoint_manifest_sha256
        != full_anchor.checkpoint_manifest_sha256
        or current.anchor_history_artifact_sha256 != full_anchor.artifact_sha256
        or current.anchor_history_completion_receipt_sha256
        != full_anchor.completion_receipt_sha256
        or current.anchor_history_production_claim_sha256
        != full_anchor.production_run_claim_sha256
    ):
        raise StrategyStagedPipelineError("history/current completions are not aligned")
    identity = _canonical_sha256(
        {
            "protocol": STRATEGY_PROTOCOL,
            "dataset": outcome.dataset,
            "history_artifact_sha256": history_cache.attestation.artifact_sha256,
            "history_completion_receipt_sha256": (
                history_cache.attestation.completion_receipt_sha256
            ),
            "history_fit_receipt_sha256": (
                history_cache.attestation.fit_producer_receipt_sha256
            ),
            "history_production_claim_sha256": (
                history_cache.attestation.production_run_claim_sha256
            ),
            "current_artifact_sha256": current.artifact_sha256,
            "current_completion_receipt_sha256": current.completion_receipt_sha256,
            "current_fit_receipt_sha256": current.fit_producer_receipt_sha256,
            "current_production_claim_sha256": current.production_run_claim_sha256,
            "full_anchor_history_artifact_sha256": full_anchor.artifact_sha256,
            "full_anchor_history_completion_receipt_sha256": (
                full_anchor.completion_receipt_sha256
            ),
            "full_anchor_history_production_claim_sha256": (
                full_anchor.production_run_claim_sha256
            ),
            "full_anchor_history_model_config_sha256": (
                _config_sha256(full_history_anchor_model_config)
            ),
            "full_anchor_registered_variant": "full",
            "fit_feature_identity_sha256": fit_features.feature_identity_sha256,
            "selection_feature_identity_sha256": (
                selection_features.feature_identity_sha256
            ),
            "fit_task_sha256": outcome.fit_tasks.task_sha256,
            "selection_task_sha256": outcome.selection_tasks.task_sha256,
            "history_checkpoint_manifest_sha256": (
                outcome.checkpoint_manifest_sha256
            ),
            "current_checkpoint_manifest_sha256": (
                current.current_checkpoint_manifest_sha256
            ),
        }
    )
    return VerifiedStrategyUpstreamState(
        history=history_cache,
        current=current,
        full_history_anchor=full_anchor,
        full_history_anchor_model_config=full_history_anchor_model_config,
        upstream_identity_sha256=identity,
    )


def _assert_upstream_unchanged(state: VerifiedStrategyUpstreamState) -> None:
    if not isinstance(state, VerifiedStrategyUpstreamState):
        raise StrategyStagedPipelineError(
            "production completion requires verified upstream state"
        )
    observed = verify_strategy_upstream_state(
        history_attestation=state.history.attestation,
        current_attestation=state.current,
        full_history_anchor_attestation=state.full_history_anchor,
        full_history_anchor_model_config=state.full_history_anchor_model_config,
        fit_features=state.history.fit_features,
        selection_features=state.history.selection_features,
    )
    if observed.upstream_identity_sha256 != state.upstream_identity_sha256:
        raise StrategyStagedPipelineError("strategy upstream identity changed")


@dataclass(frozen=True)
class _ScorerScores:
    spec: UtilityModelSpec
    spec_sha256: str
    source_identity_sha256: str
    fit_forward_by_seed: np.ndarray | None
    fit_backward_by_seed: np.ndarray | None
    fit_decision_by_seed: np.ndarray
    selection_forward_by_seed: np.ndarray | None
    selection_backward_by_seed: np.ndarray | None
    selection_decision_by_seed: np.ndarray
    fit_forward_ensemble: np.ndarray | None
    fit_backward_ensemble: np.ndarray | None
    fit_decision_ensemble: np.ndarray
    selection_forward_ensemble: np.ndarray | None
    selection_backward_ensemble: np.ndarray | None
    selection_decision_ensemble: np.ndarray
    fold_by_seed_task: np.ndarray


def _coalition_tasks(value: object) -> tuple[BidirectionalCoalitionTask, ...]:
    try:
        query = np.asarray(getattr(value, "query_indices"), dtype=np.int64)
        candidate = np.asarray(getattr(value, "candidate_indices"), dtype=np.int64)
        addition = tuple(getattr(value, "addition_contexts"))
        deletion = tuple(getattr(value, "deletion_contexts"))
    except (AttributeError, TypeError, ValueError) as error:
        raise StrategyStagedPipelineError("history task cache is malformed") from error
    if not (len(query) == len(candidate) == len(addition) == len(deletion)) or not len(
        query
    ):
        raise StrategyStagedPipelineError("history task cache is empty or misaligned")
    result: list[BidirectionalCoalitionTask] = []
    try:
        for q, c, s, t in zip(query, candidate, addition, deletion, strict=True):
            result.append(
                BidirectionalCoalitionTask(
                    query_index=int(q),
                    candidate_index=int(c),
                    addition_context=tuple(int(index) for index in s),
                    deletion_context=tuple(int(index) for index in t),
                )
            )
    except (TypeError, ValueError) as error:
        raise StrategyStagedPipelineError("history task semantics changed") from error
    return tuple(result)


def _candidate_tasks(value: object) -> EncodedCandidateTasks:
    return EncodedCandidateTasks(
        query_indices=np.asarray(getattr(value, "query_indices"), dtype=np.int64).copy(),
        candidate_indices=np.asarray(
            getattr(value, "candidate_indices"), dtype=np.int64
        ).copy(),
        addition_contexts=tuple(
            tuple(int(index) for index in row)
            for row in getattr(value, "addition_contexts")
        ),
        deletion_contexts=tuple(
            tuple(int(index) for index in row)
            for row in getattr(value, "deletion_contexts")
        ),
        task_sha256=_require_sha256(
            getattr(value, "task_sha256"), "task_sha256"
        ),
    )


def _registered_specs(
    specs: Sequence[UtilityModelSpec] | None = None,
) -> tuple[UtilityModelSpec, UtilityModelSpec, UtilityModelSpec]:
    values = tuple(default_model_specs() if specs is None else specs)
    by_name = {value.name: value for value in values}
    required = {
        "bidirectional_shared_mlp": "bidirectional_shared",
        "forward_only_mlp": "forward_only",
        "backward_only_mlp": "backward_only",
    }
    if len(by_name) != len(values) or any(
        name not in by_name or by_name[name].mode != mode
        for name, mode in required.items()
    ):
        raise StrategyStagedPipelineError("registered utility scorer roster changed")
    return (
        by_name["bidirectional_shared_mlp"],
        by_name["forward_only_mlp"],
        by_name["backward_only_mlp"],
    )


def _fit_scorer(
    *,
    fit_split: UtilitySplit,
    selection_x: np.ndarray,
    spec: UtilityModelSpec,
    feature_names: Sequence[str],
    feature_hashes: Mapping[str, str],
    supervision_sha256: str,
    fit_task_sha256: str,
    selection_task_sha256: str,
    upstream_identity_sha256: str,
) -> _ScorerScores:
    fit_forward: list[np.ndarray] = []
    fit_backward: list[np.ndarray] = []
    fit_decision: list[np.ndarray] = []
    selection_forward: list[np.ndarray] = []
    selection_backward: list[np.ndarray] = []
    selection_decision: list[np.ndarray] = []
    folds: list[np.ndarray] = []
    for seed in UTILITY_SEEDS:
        oof = group_oof_predictions(fit_split, spec, seed=int(seed))
        fitted = fit_utility_model(fit_split, spec, seed=int(seed))
        deployed = fitted.predict(selection_x)
        folds.append(np.asarray(oof.fold_by_row, dtype=np.int32))
        fit_decision.append(np.asarray(oof.predictions.decision_score, dtype=np.float64))
        selection_decision.append(
            np.asarray(deployed.decision_score, dtype=np.float64)
        )
        if oof.predictions.forward is not None and deployed.forward is not None:
            fit_forward.append(np.asarray(oof.predictions.forward, dtype=np.float64))
            selection_forward.append(np.asarray(deployed.forward, dtype=np.float64))
        if oof.predictions.backward is not None and deployed.backward is not None:
            fit_backward.append(np.asarray(oof.predictions.backward, dtype=np.float64))
            selection_backward.append(np.asarray(deployed.backward, dtype=np.float64))
    fit_decision_array = np.stack(fit_decision)
    selection_decision_array = np.stack(selection_decision)
    fold_array = np.stack(folds).astype(np.int32)
    fit_forward_array = np.stack(fit_forward) if fit_forward else None
    fit_backward_array = np.stack(fit_backward) if fit_backward else None
    selection_forward_array = np.stack(selection_forward) if selection_forward else None
    selection_backward_array = (
        np.stack(selection_backward) if selection_backward else None
    )
    fit_forward_ensemble = (
        None if fit_forward_array is None else fit_forward_array.mean(axis=0)
    )
    fit_backward_ensemble = (
        None if fit_backward_array is None else fit_backward_array.mean(axis=0)
    )
    selection_forward_ensemble = (
        None
        if selection_forward_array is None
        else selection_forward_array.mean(axis=0)
    )
    selection_backward_ensemble = (
        None
        if selection_backward_array is None
        else selection_backward_array.mean(axis=0)
    )
    if fit_forward_ensemble is not None and fit_backward_ensemble is not None:
        # This ordering is part of the registered method: ensemble directional
        # heads first and only then take the strict minimum.
        fit_decision_ensemble = np.minimum(
            fit_forward_ensemble, fit_backward_ensemble
        )
        assert selection_forward_ensemble is not None
        assert selection_backward_ensemble is not None
        selection_decision_ensemble = np.minimum(
            selection_forward_ensemble, selection_backward_ensemble
        )
    elif fit_forward_ensemble is not None:
        fit_decision_ensemble = fit_forward_ensemble.copy()
        assert selection_forward_ensemble is not None
        selection_decision_ensemble = selection_forward_ensemble.copy()
    elif fit_backward_ensemble is not None:
        fit_decision_ensemble = fit_backward_ensemble.copy()
        assert selection_backward_ensemble is not None
        selection_decision_ensemble = selection_backward_ensemble.copy()
    else:
        raise AssertionError("utility scorer emitted no directional head")
    arrays = (
        fit_decision_array,
        selection_decision_array,
        fold_array,
        fit_decision_ensemble,
        selection_decision_ensemble,
    )
    if any(not np.isfinite(value).all() for value in arrays[:2]) or not all(
        np.isfinite(value).all() for value in arrays[3:]
    ):
        raise StrategyStagedPipelineError("utility scorer emitted non-finite values")
    spec_sha = _canonical_sha256(asdict(spec))
    source = _canonical_sha256(
        {
            "protocol": STRATEGY_PROTOCOL,
            "scorer_name": spec.name,
            "scorer_mode": spec.mode,
            "model_spec_sha256": spec_sha,
            "utility_seeds": list(UTILITY_SEEDS),
            "feature_names": list(feature_names),
            "fit_feature_sha256": feature_hashes["fit"],
            "selection_feature_sha256": feature_hashes["selection"],
            "fit_supervision_sha256": supervision_sha256,
            "fit_cluster_sha256": _array_sha256(fit_split.cluster_codes),
            "fit_task_sha256": fit_task_sha256,
            "selection_task_sha256": selection_task_sha256,
            "fold_by_seed_task_sha256": _array_sha256(fold_array),
            "fit_directional_head_sha256": _canonical_sha256(
                {
                    "forward": (
                        None
                        if fit_forward_array is None
                        else _array_sha256(fit_forward_array)
                    ),
                    "backward": (
                        None
                        if fit_backward_array is None
                        else _array_sha256(fit_backward_array)
                    ),
                }
            ),
            "selection_directional_head_sha256": _canonical_sha256(
                {
                    "forward": (
                        None
                        if selection_forward_array is None
                        else _array_sha256(selection_forward_array)
                    ),
                    "backward": (
                        None
                        if selection_backward_array is None
                        else _array_sha256(selection_backward_array)
                    ),
                }
            ),
            "upstream_identity_sha256": _require_sha256(
                upstream_identity_sha256, "upstream_identity_sha256"
            ),
        }
    )
    return _ScorerScores(
        spec=spec,
        spec_sha256=spec_sha,
        source_identity_sha256=source,
        fit_forward_by_seed=fit_forward_array,
        fit_backward_by_seed=fit_backward_array,
        fit_decision_by_seed=fit_decision_array,
        selection_forward_by_seed=selection_forward_array,
        selection_backward_by_seed=selection_backward_array,
        selection_decision_by_seed=selection_decision_array,
        fit_forward_ensemble=fit_forward_ensemble,
        fit_backward_ensemble=fit_backward_ensemble,
        fit_decision_ensemble=fit_decision_ensemble,
        selection_forward_ensemble=selection_forward_ensemble,
        selection_backward_ensemble=selection_backward_ensemble,
        selection_decision_ensemble=selection_decision_ensemble,
        fold_by_seed_task=fold_array,
    )


def _build_outcome_free_strategy_plan(
    *,
    outcome: HistoryOutcomeFreeView,
    supervision: HistoryFitTargetsView,
    fit_histories: Sequence[Sequence[int]],
    selection_histories: Sequence[Sequence[int]],
    upstream_identity_sha256: str,
    model_specs: Sequence[UtilityModelSpec] | None = None,
) -> OutcomeFreeStrategyPlan:
    if (
        outcome.dataset != supervision.dataset
        or outcome.seeds != supervision.seeds
        or outcome.seeds != EXPECTED_SEEDS
        or outcome.fit_tasks.task_sha256 != supervision.task_sha256
        or supervision.forward_utility.shape
        != (len(EXPECTED_SEEDS), len(outcome.fit_tasks))
        or supervision.backward_utility.shape
        != (len(EXPECTED_SEEDS), len(outcome.fit_tasks))
    ):
        raise StrategyStagedPipelineError("history scorer supervision is misaligned")
    fit_history = tuple(tuple(int(index) for index in row) for row in fit_histories)
    selection_history = tuple(
        tuple(int(index) for index in row) for row in selection_histories
    )
    if (
        len(fit_history) != len(outcome.fit_protocol_row_ids)
        or len(selection_history) != len(outcome.selection_protocol_row_ids)
    ):
        raise StrategyStagedPipelineError("strategy histories are not row-aligned")
    fit_tasks = _coalition_tasks(outcome.fit_tasks)
    selection_tasks = _coalition_tasks(outcome.selection_tasks)
    fit_x, feature_names = probability_task_features(
        np.mean(outcome.fit_utility_probability_oof, axis=0),
        fit_tasks,
        fit_history,
    )
    selection_x, selection_names = probability_task_features(
        np.mean(outcome.selection_utility_probability_fold_ensemble, axis=0),
        selection_tasks,
        selection_history,
    )
    if feature_names != selection_names or any(
        token in name.lower() for name in feature_names for token in ("label", "gold")
    ):
        raise StrategyStagedPipelineError("utility feature schema changed or leaked outcomes")
    fit_supervision_sha = _canonical_sha256(
        {
            "forward_by_base_seed_sha256": _array_sha256(
                supervision.forward_utility
            ),
            "backward_by_base_seed_sha256": _array_sha256(
                supervision.backward_utility
            ),
            "source_identity_sha256": supervision.source_identity_sha256,
            "fit_outcome_artifact_sha256": (
                supervision.fit_outcome_artifact_sha256
            ),
            "fit_supervision_artifact_sha256": supervision.artifact_sha256,
        }
    )
    fit_cluster = np.asarray(outcome.fit_cluster_codes, dtype=np.int64)[
        np.asarray(outcome.fit_tasks.query_indices, dtype=np.int64)
    ]
    fit_split = UtilitySplit.validated(
        fit_x,
        np.mean(supervision.forward_utility, axis=0),
        np.mean(supervision.backward_utility, axis=0),
        fit_cluster,
        label="strategy fit",
    )
    feature_hashes = {
        "fit": _array_sha256(np.asarray(fit_x, dtype=np.float64)),
        "selection": _array_sha256(np.asarray(selection_x, dtype=np.float64)),
    }
    bidirectional_spec, forward_spec, backward_spec = _registered_specs(model_specs)
    common = {
        "fit_split": fit_split,
        "selection_x": selection_x,
        "feature_names": feature_names,
        "feature_hashes": feature_hashes,
        "supervision_sha256": fit_supervision_sha,
        "fit_task_sha256": outcome.fit_tasks.task_sha256,
        "selection_task_sha256": outcome.selection_tasks.task_sha256,
        "upstream_identity_sha256": upstream_identity_sha256,
    }
    bidirectional = _fit_scorer(spec=bidirectional_spec, **common)
    forward = _fit_scorer(spec=forward_spec, **common)
    backward = _fit_scorer(spec=backward_spec, **common)
    if (
        bidirectional.fit_forward_by_seed is None
        or bidirectional.fit_backward_by_seed is None
        or bidirectional.selection_forward_by_seed is None
        or bidirectional.selection_backward_by_seed is None
        or bidirectional.fit_forward_ensemble is None
        or bidirectional.fit_backward_ensemble is None
        or bidirectional.selection_forward_ensemble is None
        or bidirectional.selection_backward_ensemble is None
    ):
        raise AssertionError("registered bidirectional scorer lacks two heads")
    encoded_fit = _candidate_tasks(outcome.fit_tasks)
    encoded_selection = _candidate_tasks(outcome.selection_tasks)

    def freeze_and_prepare(scores: _ScorerScores) -> tuple[FrozenCoverageRule, PreparedPolicyContexts]:
        rule = freeze_fit_oof_operating_point(
            encoded_fit,
            scores.fit_decision_ensemble,
            score_source_identity_sha256=scores.source_identity_sha256,
            target_coverage=PRIMARY_HISTORY_COVERAGE,
        )
        policy = prepare_policy_contexts(
            query_indices=np.arange(len(selection_history), dtype=np.int64),
            histories=selection_history,
            tasks=encoded_selection,
            decision_scores=scores.selection_decision_ensemble,
            score_source_identity_sha256=scores.source_identity_sha256,
            rule=rule,
        )
        return rule, policy

    primary_rule, primary_policy = freeze_and_prepare(bidirectional)
    forward_rule, forward_policy = freeze_and_prepare(forward)
    backward_rule, backward_policy = freeze_and_prepare(backward)
    query_indices = tuple(range(len(selection_history)))
    all_history = tuple(selection_history[index] for index in query_indices)
    all_history_sha = _canonical_sha256([list(row) for row in all_history])
    return OutcomeFreeStrategyPlan(
        method_roster=METHOD_ROSTER,
        feature_names=feature_names,
        model_spec_sha256=bidirectional.spec_sha256,
        forward_model_spec_sha256=forward.spec_sha256,
        backward_model_spec_sha256=backward.spec_sha256,
        fit_feature_sha256=feature_hashes["fit"],
        selection_feature_sha256=feature_hashes["selection"],
        fit_supervision_sha256=fit_supervision_sha,
        fit_cluster_sha256=_array_sha256(fit_cluster),
        fit_oof_fold_sha256=_canonical_sha256(
            {
                "bidirectional": _array_sha256(bidirectional.fold_by_seed_task),
                "forward": _array_sha256(forward.fold_by_seed_task),
                "backward": _array_sha256(backward.fold_by_seed_task),
            }
        ),
        score_source_identity_sha256=bidirectional.source_identity_sha256,
        fit_forward_by_seed=bidirectional.fit_forward_by_seed,
        fit_backward_by_seed=bidirectional.fit_backward_by_seed,
        fit_decision_by_seed=bidirectional.fit_decision_by_seed,
        selection_forward_by_seed=bidirectional.selection_forward_by_seed,
        selection_backward_by_seed=bidirectional.selection_backward_by_seed,
        selection_decision_by_seed=bidirectional.selection_decision_by_seed,
        fit_forward_ensemble=bidirectional.fit_forward_ensemble,
        fit_backward_ensemble=bidirectional.fit_backward_ensemble,
        fit_decision_ensemble=bidirectional.fit_decision_ensemble,
        selection_forward_ensemble=bidirectional.selection_forward_ensemble,
        selection_backward_ensemble=bidirectional.selection_backward_ensemble,
        selection_decision_ensemble=bidirectional.selection_decision_ensemble,
        rule=primary_rule,
        policy=primary_policy,
        forward_score_source_identity_sha256=forward.source_identity_sha256,
        backward_score_source_identity_sha256=backward.source_identity_sha256,
        forward_fit_decision_by_seed=forward.fit_decision_by_seed,
        backward_fit_decision_by_seed=backward.fit_decision_by_seed,
        forward_selection_decision_by_seed=forward.selection_decision_by_seed,
        backward_selection_decision_by_seed=backward.selection_decision_by_seed,
        forward_fit_decision_ensemble=forward.fit_decision_ensemble,
        backward_fit_decision_ensemble=backward.fit_decision_ensemble,
        forward_selection_decision_ensemble=forward.selection_decision_ensemble,
        backward_selection_decision_ensemble=backward.selection_decision_ensemble,
        forward_rule=forward_rule,
        backward_rule=backward_rule,
        forward_policy=forward_policy,
        backward_policy=backward_policy,
        all_history_contexts=all_history,
        all_history_context_sha256=all_history_sha,
    )


def build_outcome_free_strategy(
    upstream: VerifiedStrategyUpstreamState,
) -> OutcomeFreeStrategyPlan:
    """Fit the immutable production scorer roster without accepting outcomes."""

    _assert_upstream_unchanged(upstream)
    return _build_outcome_free_strategy_plan(
        outcome=upstream.history.outcome,
        supervision=upstream.history.supervision,
        fit_histories=upstream.history.fit_features.histories,
        selection_histories=upstream.history.selection_features.histories,
        upstream_identity_sha256=upstream.upstream_identity_sha256,
        model_specs=None,
    )


def build_outcome_free_strategy_nonproduction_fixture(
    *,
    outcome: HistoryOutcomeFreeView,
    supervision: HistoryFitTargetsView,
    fit_histories: Sequence[Sequence[int]],
    selection_histories: Sequence[Sequence[int]],
    fixture_identity_sha256: str,
    model_specs: Sequence[UtilityModelSpec] | None = None,
) -> OutcomeFreeStrategyPlan:
    """Synthetic-only test hook; it cannot publish a production receipt."""

    return _build_outcome_free_strategy_plan(
        outcome=outcome,
        supervision=supervision,
        fit_histories=fit_histories,
        selection_histories=selection_histories,
        upstream_identity_sha256=_require_sha256(
            fixture_identity_sha256, "fixture_identity_sha256"
        ),
        model_specs=model_specs,
    )


def _validate_history_model_contract(
    upstream: VerifiedStrategyUpstreamState,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
) -> None:
    model_config.validate()
    run_config.validate()
    class_order = upstream.history.outcome.label_order
    if (
        model_config.num_classes != len(class_order)
        or (
            model_config.auxiliary_vad_weight > 0.0
            and tuple(model_config.emotion_label_order) != tuple(class_order)
        )
        or _config_sha256(model_config) != upstream.history.model_config_sha256
        or _config_sha256(run_config) != upstream.history.run_config_sha256
        or int(run_config.outer_folds)
        != int(upstream.history.checkpoint_manifest.outer_folds)
    ):
        raise StrategyStagedPipelineError(
            "strategy model/run configuration differs from history checkpoints"
        )


def _policy_context_roster(
    plan: OutcomeFreeStrategyPlan,
) -> dict[str, tuple[tuple[int, ...], ...]]:
    result = {
        "bidirectional_selected_history": plan.policy.selected_contexts,
        "forward_only_selected_history": plan.forward_policy.selected_contexts,
        "backward_only_selected_history": plan.backward_policy.selected_contexts,
        "coverage_matched_recency": plan.policy.matched_recency_contexts,
        "all_history_diagnostic": plan.all_history_contexts,
    }
    if tuple(result) != METHOD_ROSTER:
        raise AssertionError("strategy method roster ordering changed")
    query_count = len(plan.policy.selected_contexts)
    if any(len(value) != query_count for value in result.values()):
        raise StrategyStagedPipelineError("method contexts are not query-aligned")
    if any(
        len(left) != len(right)
        for left, right in zip(
            plan.policy.selected_contexts,
            plan.policy.matched_recency_contexts,
            strict=True,
        )
    ):
        raise StrategyStagedPipelineError("recency arm is not exactly coverage matched")
    return result


def _infer_registered_method_probabilities(
    *,
    upstream: VerifiedStrategyUpstreamState,
    plan: OutcomeFreeStrategyPlan,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Restore every complete history fold and infer the fixed method roster."""

    _validate_history_model_contract(upstream, model_config, run_config)
    fit_view = _dummy_fit_view(
        upstream.history.fit_features, upstream.history.outcome.label_order
    )
    selection_view = _selection_view(upstream.history.selection_features)
    fold_by_seed = np.asarray(
        getattr(upstream.history.fit_outcome, "fold_by_seed_query"), dtype=np.int32
    )
    try:
        verify_complete_history_checkpoint_payloads(
            upstream.history.checkpoint_root,
            upstream.history.checkpoint_manifest,
            fit=fit_view,
            model_config=model_config,
            run_config=run_config,
            source_identity_sha256=upstream.history.outcome.source_identity_sha256,
            fold_by_seed_query=fold_by_seed,
        )
    except ValueError as error:
        raise StrategyStagedPipelineError(
            f"history complete-checkpoint gate failed: {error}"
        ) from error
    contexts = _policy_context_roster(plan)
    query_count = upstream.history.selection_features.rows
    class_count = len(upstream.history.outcome.label_order)
    query_indices = np.arange(query_count, dtype=np.int64)
    flat_queries = np.tile(query_indices, len(METHOD_ROSTER))
    flat_contexts = tuple(
        context for method in METHOD_ROSTER for context in contexts[method]
    )
    probability = {
        method: np.zeros(
            (len(EXPECTED_SEEDS), query_count, class_count), dtype=np.float64
        )
        for method in METHOD_ROSTER
    }
    for seed_index, seed in enumerate(EXPECTED_SEEDS):
        for fold in range(int(run_config.outer_folds)):
            held = np.flatnonzero(fold_by_seed[seed_index] == fold).astype(np.int64)
            train = np.flatnonzero(fold_by_seed[seed_index] != fold).astype(np.int64)
            if not len(held) or not len(train):
                raise StrategyStagedPipelineError("history checkpoint fold is empty")
            fit_corpus = _fit_corpus_from_view(
                fit_view,
                model_config=model_config,
                heldout_indices=held,
                speaker_reference_indices=train,
            )
            selection_corpus = _selection_corpus_from_view(
                selection_view,
                fit=fit_view,
                model_config=model_config,
                fit_speaker_reference_indices=train,
            )
            split = _split_from_outer_partition(
                fit_corpus,
                outer_train=train,
                heldout=held,
                validation_fraction=run_config.inner_validation_fraction,
                seed=int(seed),
                fold=int(fold),
            )
            verify_checkpoint_manifest(
                upstream.history.checkpoint_root,
                upstream.history.checkpoint_manifest,
            )
            try:
                trained = train_one_fold_seed(
                    fit_corpus,
                    split,
                    model_config=model_config,
                    run_config=run_config,
                    seed=int(seed),
                    source_identity=upstream.history.outcome.source_identity_sha256,
                    checkpoint_root=upstream.history.checkpoint_root,
                    device=device,
                    require_complete_checkpoint=True,
                )
            except (RuntimeError, ValueError) as error:
                raise StrategyStagedPipelineError(
                    f"complete history checkpoint restore failed: {error}"
                ) from error
            if (
                trained.summary.get("resumed_complete_checkpoint") is not True
                or trained.summary.get("resumed_partial_checkpoint") is not False
            ):
                raise StrategyStagedPipelineError(
                    "strategy inference did not restore a complete checkpoint"
                )
            selection_text = trained.processor.transform(selection_corpus.texts)
            fold_probability = predict_one_probability_per_query(
                trained.model,
                selection_corpus,
                selection_text,
                flat_queries,
                flat_contexts,
                device=device,
                batch_size=run_config.inference_batch_size,
                max_history_items=run_config.max_history_items,
            )
            validated = _probability(
                fold_probability,
                (len(METHOD_ROSTER) * query_count, class_count),
                "strategy fold probability",
            ).reshape(len(METHOD_ROSTER), query_count, class_count)
            for method_index, method in enumerate(METHOD_ROSTER):
                probability[method][seed_index] += validated[method_index]
        for method in METHOD_ROSTER:
            probability[method][seed_index] /= float(run_config.outer_folds)
    verify_checkpoint_manifest(
        upstream.history.checkpoint_root,
        upstream.history.checkpoint_manifest,
    )
    for method in METHOD_ROSTER:
        probability[method] = _probability(
            probability[method],
            (len(EXPECTED_SEEDS), query_count, class_count),
            f"{method} fold ensemble",
        )
    return probability


def _safe_named_file_hashes(
    values: Mapping[str, str | Path], field: str
) -> dict[str, str]:
    if not isinstance(values, Mapping) or not values:
        raise StrategyStagedPipelineError(f"{field} must be a non-empty mapping")
    result: dict[str, str] = {}
    casefolded: set[str] = set()
    source_keys = field == "code_paths"
    for raw_name, raw_path in values.items():
        try:
            name = (
                _canonical_production_source_key(raw_name)
                if source_keys
                else str(raw_name)
            )
        except ValueError as error:
            raise StrategyStagedPipelineError(
                f"{field} contains an unsafe name"
            ) from error
        if (
            (
                not source_keys
                and (
                    not name
                    or not name[0].isalnum()
                    or len(name) > 128
                    or any(
                        not (character.isalnum() or character in "_.-")
                        for character in name
                    )
                )
            )
            or name.casefold() in casefolded
        ):
            raise StrategyStagedPipelineError(f"{field} contains an unsafe name")
        casefolded.add(name.casefold())
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise StrategyStagedPipelineError(f"{field}.{name} is not a file")
        result[name] = _file_sha256(path)
    return dict(sorted(result.items()))


def _live_strategy_lineage(
    *,
    config_paths: Mapping[str, str | Path],
    code_paths: Mapping[str, str | Path],
    environment: Mapping[str, object],
) -> tuple[dict[str, str], dict[str, str], str, str]:
    if not isinstance(environment, Mapping) or not environment:
        raise StrategyStagedPipelineError("strategy runtime environment is empty")
    config = _safe_named_file_hashes(config_paths, "config_paths")
    code = _safe_named_file_hashes(code_paths, "code_paths")
    runtime = _canonical_sha256(dict(environment))
    identity = _canonical_sha256(
        {
            "config_sha256": config,
            "code_sha256": code,
            "runtime_environment_sha256": runtime,
        }
    )
    return config, code, runtime, identity


def _cross_variant_alignment_sha256(
    upstream: VerifiedStrategyUpstreamState,
) -> str:
    outcome = upstream.history.outcome
    fold_by_seed = np.asarray(
        getattr(upstream.history.fit_outcome, "fold_by_seed_query"), dtype=np.int32
    )
    return _canonical_sha256(
        {
            "protocol": "carma_strategy_cross_variant_alignment_v1",
            "dataset": outcome.dataset,
            "class_order": list(outcome.label_order),
            "base_seeds": list(outcome.seeds),
            "utility_seeds": list(UTILITY_SEEDS),
            "fit_protocol_rows_sha256": _array_sha256(
                outcome.fit_protocol_row_ids
            ),
            "selection_protocol_rows_sha256": _array_sha256(
                outcome.selection_protocol_row_ids
            ),
            "full_current_fit_protocol_rows_sha256": _array_sha256(
                upstream.current.fit_protocol_row_ids
            ),
            "full_current_selection_protocol_rows_sha256": _array_sha256(
                upstream.current.selection_protocol_row_ids
            ),
            "fit_cluster_sha256": _array_sha256(outcome.fit_cluster_codes),
            "selection_cluster_sha256": _array_sha256(
                outcome.selection_cluster_codes
            ),
            "fit_histories_sha256": outcome.fit_histories_sha256,
            "selection_histories_sha256": outcome.selection_histories_sha256,
            "fit_task_sha256": outcome.fit_tasks.task_sha256,
            "selection_task_sha256": outcome.selection_tasks.task_sha256,
            "fold_by_base_seed_fit_row_sha256": _array_sha256(fold_by_seed),
            "method_roster": list(METHOD_ROSTER),
        }
    )


def strategy_production_claim_sha256(
    *,
    upstream: VerifiedStrategyUpstreamState,
    plan: OutcomeFreeStrategyPlan,
    registered_variant: str,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    strategy_config_sha256: Mapping[str, str],
    strategy_code_sha256: Mapping[str, str],
    strategy_runtime_environment_sha256: str,
) -> str:
    """Bind one immutable strategy run to all upstream and live lineage."""

    _require_registered_variant_matches_model_config(registered_variant, model_config)
    _validate_history_model_contract(upstream, model_config, run_config)
    return _canonical_sha256(
        {
            "protocol": STRATEGY_PROTOCOL,
            "registered_variant": registered_variant,
            "registered_variants": list(REGISTERED_VARIANTS),
            "method_roster": list(METHOD_ROSTER),
            "upstream_identity_sha256": upstream.upstream_identity_sha256,
            "history_artifact_sha256": upstream.history.attestation.artifact_sha256,
            "history_completion_receipt_sha256": (
                upstream.history.attestation.completion_receipt_sha256
            ),
            "history_fit_receipt_sha256": (
                upstream.history.attestation.fit_producer_receipt_sha256
            ),
            "history_production_claim_sha256": (
                upstream.history.attestation.production_run_claim_sha256
            ),
            "current_artifact_sha256": upstream.current.artifact_sha256,
            "current_completion_receipt_sha256": (
                upstream.current.completion_receipt_sha256
            ),
            "current_fit_receipt_sha256": (
                upstream.current.fit_producer_receipt_sha256
            ),
            "current_production_claim_sha256": (
                upstream.current.production_run_claim_sha256
            ),
            "full_current_anchor_history_artifact_sha256": (
                upstream.full_history_anchor.artifact_sha256
            ),
            "full_current_anchor_history_completion_receipt_sha256": (
                upstream.full_history_anchor.completion_receipt_sha256
            ),
            "full_current_anchor_history_production_claim_sha256": (
                upstream.full_history_anchor.production_run_claim_sha256
            ),
            "history_checkpoint_manifest_sha256": (
                upstream.history.outcome.checkpoint_manifest_sha256
            ),
            "current_checkpoint_manifest_sha256": (
                upstream.current.current_checkpoint_manifest_sha256
            ),
            "history_source_identity_sha256": (
                upstream.history.outcome.source_identity_sha256
            ),
            "current_source_identity_sha256": (
                upstream.current.current_only_source_identity_sha256
            ),
            "fit_feature_identity_sha256": (
                upstream.history.fit_features.feature_identity_sha256
            ),
            "selection_feature_identity_sha256": (
                upstream.history.selection_features.feature_identity_sha256
            ),
            "fit_task_sha256": upstream.history.outcome.fit_tasks.task_sha256,
            "selection_task_sha256": (
                upstream.history.outcome.selection_tasks.task_sha256
            ),
            "fit_supervision_sha256": plan.fit_supervision_sha256,
            "bidirectional_model_spec_sha256": plan.model_spec_sha256,
            "forward_model_spec_sha256": plan.forward_model_spec_sha256,
            "backward_model_spec_sha256": plan.backward_model_spec_sha256,
            "bidirectional_score_source_identity_sha256": (
                plan.score_source_identity_sha256
            ),
            "forward_score_source_identity_sha256": (
                plan.forward_score_source_identity_sha256
            ),
            "backward_score_source_identity_sha256": (
                plan.backward_score_source_identity_sha256
            ),
            "bidirectional_rule_sha256": plan.rule.rule_sha256,
            "forward_rule_sha256": plan.forward_rule.rule_sha256,
            "backward_rule_sha256": plan.backward_rule.rule_sha256,
            "bidirectional_policy_sha256": plan.policy.policy_sha256,
            "forward_policy_sha256": plan.forward_policy.policy_sha256,
            "backward_policy_sha256": plan.backward_policy.policy_sha256,
            "all_history_context_sha256": plan.all_history_context_sha256,
            "cross_variant_alignment_sha256": _cross_variant_alignment_sha256(
                upstream
            ),
            "model_config_sha256": _config_sha256(model_config),
            "run_config_sha256": _config_sha256(run_config),
            "strategy_config_sha256": dict(sorted(strategy_config_sha256.items())),
            "strategy_code_sha256": dict(sorted(strategy_code_sha256.items())),
            "strategy_runtime_environment_sha256": _require_sha256(
                strategy_runtime_environment_sha256,
                "strategy_runtime_environment_sha256",
            ),
        }
    )


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def strategy_private_paths(root: str | Path) -> dict[str, Path]:
    private_root = Path(root).resolve()
    return {
        "claim": private_root / STRATEGY_PRIVATE_CLAIM_NAME,
        "artifact": private_root / STRATEGY_PRIVATE_ARTIFACT_NAME,
        "receipt": private_root / STRATEGY_PRIVATE_RECEIPT_NAME,
    }


def _claim_strategy_private_root(path: str | Path, claim_sha256: str) -> Path:
    root = Path(path).resolve()
    if (
        not root.is_absolute()
        or root == Path(root.anchor)
        or _is_within(root, _repository_root())
        or not root.parent.is_dir()
    ):
        raise StrategyStagedPipelineError(
            "strategy private root must be safe, new, and repository-external"
        )
    try:
        root.mkdir(exist_ok=False)
    except FileExistsError:
        raise FileExistsError("strategy private root must be all-new") from None
    claim = {
        "schema_version": STRATEGY_PRIVATE_CLAIM_SCHEMA,
        "status": "claimed_for_one_outcome_free_strategy_completion",
        "production_claim_sha256": _require_sha256(
            claim_sha256, "production_claim_sha256"
        ),
    }
    try:
        _write_json_once(strategy_private_paths(root)["claim"], claim)
    except ValueError as error:
        raise StrategyStagedPipelineError(f"cannot write strategy claim: {error}") from error
    return root


def _verify_strategy_private_claim(root: Path, claim_sha256: str) -> None:
    observed = _read_json_mapping(
        strategy_private_paths(root)["claim"], "strategy private claim"
    )
    expected = {
        "schema_version": STRATEGY_PRIVATE_CLAIM_SCHEMA,
        "status": "claimed_for_one_outcome_free_strategy_completion",
        "production_claim_sha256": _require_sha256(
            claim_sha256, "production_claim_sha256"
        ),
    }
    if observed != expected:
        raise StrategyStagedPipelineError("strategy private claim changed")


def _encode_contexts(
    contexts: Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, str]:
    indptr = [0]
    indices: list[int] = []
    normalized: list[list[int]] = []
    for row in contexts:
        values = tuple(int(value) for value in row)
        if len(values) != len(set(values)) or any(value < 0 for value in values):
            raise StrategyStagedPipelineError("strategy context contains invalid rows")
        indices.extend(values)
        indptr.append(len(indices))
        normalized.append(list(values))
    return (
        np.asarray(indptr, dtype=np.int64),
        np.asarray(indices, dtype=np.int64),
        _canonical_sha256(normalized),
    )


def _rule_values(prefix: str, rule: FrozenCoverageRule) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_registered_coverage_fraction": np.asarray(
            rule.target_coverage, dtype=np.float64
        ),
        f"{prefix}_fit_pair_count": np.asarray(rule.fit_pair_count, dtype=np.int64),
        f"{prefix}_fit_selected_count": np.asarray(
            rule.fit_selected_count, dtype=np.int64
        ),
        f"{prefix}_fit_realized_coverage": np.asarray(
            rule.fit_realized_coverage, dtype=np.float64
        ),
        f"{prefix}_threshold": np.asarray(rule.threshold, dtype=np.float64),
        f"{prefix}_boundary_tie_fraction": np.asarray(
            rule.boundary_tie_fraction, dtype=np.float64
        ),
        f"{prefix}_tie_salt_sha256": np.asarray(rule.tie_salt_sha256),
        f"{prefix}_rule_sha256": np.asarray(rule.rule_sha256),
    }


def _policy_values(
    prefix: str, policy: PreparedPolicyContexts
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_selection_pair_count": np.asarray(
            policy.selected_pair_count, dtype=np.int64
        ),
        f"{prefix}_available_pair_count": np.asarray(
            policy.available_pair_count, dtype=np.int64
        ),
        f"{prefix}_selection_realized_coverage": np.asarray(
            policy.realized_pair_coverage, dtype=np.float64
        ),
        f"{prefix}_history_using_query_count": np.asarray(
            policy.history_using_query_count, dtype=np.int64
        ),
        f"{prefix}_policy_sha256": np.asarray(policy.policy_sha256),
    }


def _strategy_artifact_mapping(
    *,
    upstream: VerifiedStrategyUpstreamState,
    plan: OutcomeFreeStrategyPlan,
    probabilities: Mapping[str, np.ndarray],
    registered_variant: str,
    production_claim_sha256: str,
    strategy_config_sha256: Mapping[str, str],
    strategy_code_sha256: Mapping[str, str],
    strategy_runtime_environment_sha256: str,
    strategy_live_lineage_sha256: str,
) -> dict[str, np.ndarray]:
    outcome = upstream.history.outcome
    fold_by_seed = np.asarray(
        getattr(upstream.history.fit_outcome, "fold_by_seed_query"), dtype=np.int32
    )
    history_code_bundle = _canonical_sha256(upstream.history.code_sha256)
    values: dict[str, np.ndarray] = {
        "schema_version": np.asarray(STRATEGY_PRIVATE_SCHEMA),
        "dataset": np.asarray(outcome.dataset),
        "dataset_class_order": np.asarray(outcome.label_order),
        "registered_variant": np.asarray(registered_variant),
        "registered_variants": np.asarray(REGISTERED_VARIANTS),
        "method_roster": np.asarray(METHOD_ROSTER),
        "joint_evaluation_roster": np.asarray(JOINT_EVALUATION_ROSTER),
        "base_seeds": np.asarray(EXPECTED_SEEDS, dtype=np.int64),
        "utility_seeds": np.asarray(UTILITY_SEEDS, dtype=np.int64),
        "fit_protocol_rows": np.asarray(outcome.fit_protocol_row_ids, dtype=np.int64),
        "selection_protocol_rows": np.asarray(
            outcome.selection_protocol_row_ids, dtype=np.int64
        ),
        "full_current_fit_protocol_rows": np.asarray(
            upstream.current.fit_protocol_row_ids, dtype=np.int64
        ),
        "full_current_selection_protocol_rows": np.asarray(
            upstream.current.selection_protocol_row_ids, dtype=np.int64
        ),
        "fit_cluster_codes": np.asarray(outcome.fit_cluster_codes, dtype=np.int64),
        "selection_cluster_codes": np.asarray(
            outcome.selection_cluster_codes, dtype=np.int64
        ),
        "fit_protocol_rows_sha256": np.asarray(
            _array_sha256(outcome.fit_protocol_row_ids)
        ),
        "selection_protocol_rows_sha256": np.asarray(
            _array_sha256(outcome.selection_protocol_row_ids)
        ),
        "full_current_fit_protocol_rows_sha256": np.asarray(
            _array_sha256(upstream.current.fit_protocol_row_ids)
        ),
        "full_current_selection_protocol_rows_sha256": np.asarray(
            _array_sha256(upstream.current.selection_protocol_row_ids)
        ),
        "fit_histories_sha256": np.asarray(outcome.fit_histories_sha256),
        "selection_histories_sha256": np.asarray(
            outcome.selection_histories_sha256
        ),
        "fit_task_sha256": np.asarray(outcome.fit_tasks.task_sha256),
        "selection_task_sha256": np.asarray(outcome.selection_tasks.task_sha256),
        "history_fold_assignment_sha256": np.asarray(_array_sha256(fold_by_seed)),
        "cross_variant_alignment_sha256": np.asarray(
            _cross_variant_alignment_sha256(upstream)
        ),
        "fit_feature_identity_sha256": np.asarray(
            upstream.history.fit_features.feature_identity_sha256
        ),
        "selection_feature_identity_sha256": np.asarray(
            upstream.history.selection_features.feature_identity_sha256
        ),
        "fit_feature_file_sha256": np.asarray(
            upstream.history.fit_features.feature_file_sha256
        ),
        "selection_feature_file_sha256": np.asarray(
            upstream.history.selection_features.feature_file_sha256
        ),
        "feature_names": np.asarray(plan.feature_names),
        "fit_feature_matrix_sha256": np.asarray(plan.fit_feature_sha256),
        "selection_feature_matrix_sha256": np.asarray(
            plan.selection_feature_sha256
        ),
        "fit_supervision_sha256": np.asarray(plan.fit_supervision_sha256),
        "fit_supervision_artifact_sha256": np.asarray(
            upstream.history.supervision.artifact_sha256
        ),
        "bidirectional_model_spec_sha256": np.asarray(plan.model_spec_sha256),
        "forward_model_spec_sha256": np.asarray(plan.forward_model_spec_sha256),
        "backward_model_spec_sha256": np.asarray(plan.backward_model_spec_sha256),
        "fit_task_cluster_sha256": np.asarray(plan.fit_cluster_sha256),
        "utility_oof_fold_sha256": np.asarray(plan.fit_oof_fold_sha256),
        "bidirectional_score_source_identity_sha256": np.asarray(
            plan.score_source_identity_sha256
        ),
        "forward_score_source_identity_sha256": np.asarray(
            plan.forward_score_source_identity_sha256
        ),
        "backward_score_source_identity_sha256": np.asarray(
            plan.backward_score_source_identity_sha256
        ),
        "history_artifact_sha256": np.asarray(
            upstream.history.attestation.artifact_sha256
        ),
        "history_completion_receipt_sha256": np.asarray(
            upstream.history.attestation.completion_receipt_sha256
        ),
        "history_fit_receipt_sha256": np.asarray(
            upstream.history.attestation.fit_producer_receipt_sha256
        ),
        "history_production_claim_sha256": np.asarray(
            upstream.history.attestation.production_run_claim_sha256
        ),
        "current_artifact_sha256": np.asarray(upstream.current.artifact_sha256),
        "current_completion_receipt_sha256": np.asarray(
            upstream.current.completion_receipt_sha256
        ),
        "current_fit_receipt_sha256": np.asarray(
            upstream.current.fit_producer_receipt_sha256
        ),
        "current_production_claim_sha256": np.asarray(
            upstream.current.production_run_claim_sha256
        ),
        "full_current_anchor_history_artifact_sha256": np.asarray(
            upstream.full_history_anchor.artifact_sha256
        ),
        "full_current_anchor_history_completion_receipt_sha256": np.asarray(
            upstream.full_history_anchor.completion_receipt_sha256
        ),
        "full_current_anchor_history_production_claim_sha256": np.asarray(
            upstream.full_history_anchor.production_run_claim_sha256
        ),
        "history_source_identity_sha256": np.asarray(outcome.source_identity_sha256),
        "current_source_identity_sha256": np.asarray(
            upstream.current.current_only_source_identity_sha256
        ),
        "history_checkpoint_manifest_sha256": np.asarray(
            outcome.checkpoint_manifest_sha256
        ),
        "current_checkpoint_manifest_sha256": np.asarray(
            upstream.current.current_checkpoint_manifest_sha256
        ),
        "history_model_config_sha256": np.asarray(
            upstream.history.model_config_sha256
        ),
        "history_run_config_sha256": np.asarray(upstream.history.run_config_sha256),
        "history_utility_config_sha256": np.asarray(
            upstream.history.utility_config_sha256
        ),
        "history_code_bundle_sha256": np.asarray(history_code_bundle),
        "history_execution_environment_sha256": np.asarray(
            upstream.history.execution_environment_sha256
        ),
        "current_source_code_sha256": np.asarray(upstream.current.source_code_sha256),
        "current_runtime_environment_sha256": np.asarray(
            upstream.current.runtime_environment_sha256
        ),
        "strategy_config_bundle_sha256": np.asarray(
            _canonical_sha256(dict(sorted(strategy_config_sha256.items())))
        ),
        "strategy_code_bundle_sha256": np.asarray(
            _canonical_sha256(dict(sorted(strategy_code_sha256.items())))
        ),
        "strategy_runtime_environment_sha256": np.asarray(
            strategy_runtime_environment_sha256
        ),
        "strategy_live_lineage_sha256": np.asarray(strategy_live_lineage_sha256),
        "upstream_identity_sha256": np.asarray(upstream.upstream_identity_sha256),
        "production_run_claim_sha256": np.asarray(production_claim_sha256),
        "bidirectional_fit_forward_score_by_utility_seed": np.asarray(
            plan.fit_forward_by_seed, dtype=np.float64
        ),
        "bidirectional_fit_backward_score_by_utility_seed": np.asarray(
            plan.fit_backward_by_seed, dtype=np.float64
        ),
        "bidirectional_fit_decision_score_by_utility_seed": np.asarray(
            plan.fit_decision_by_seed, dtype=np.float64
        ),
        "bidirectional_selection_forward_score_by_utility_seed": np.asarray(
            plan.selection_forward_by_seed, dtype=np.float64
        ),
        "bidirectional_selection_backward_score_by_utility_seed": np.asarray(
            plan.selection_backward_by_seed, dtype=np.float64
        ),
        "bidirectional_selection_decision_score_by_utility_seed": np.asarray(
            plan.selection_decision_by_seed, dtype=np.float64
        ),
        "bidirectional_fit_forward_score_ensemble": np.asarray(
            plan.fit_forward_ensemble, dtype=np.float64
        ),
        "bidirectional_fit_backward_score_ensemble": np.asarray(
            plan.fit_backward_ensemble, dtype=np.float64
        ),
        "bidirectional_fit_decision_score_ensemble": np.asarray(
            plan.fit_decision_ensemble, dtype=np.float64
        ),
        "bidirectional_selection_forward_score_ensemble": np.asarray(
            plan.selection_forward_ensemble, dtype=np.float64
        ),
        "bidirectional_selection_backward_score_ensemble": np.asarray(
            plan.selection_backward_ensemble, dtype=np.float64
        ),
        "bidirectional_selection_decision_score_ensemble": np.asarray(
            plan.selection_decision_ensemble, dtype=np.float64
        ),
        "forward_fit_decision_score_by_utility_seed": np.asarray(
            plan.forward_fit_decision_by_seed, dtype=np.float64
        ),
        "forward_selection_decision_score_by_utility_seed": np.asarray(
            plan.forward_selection_decision_by_seed, dtype=np.float64
        ),
        "forward_fit_decision_score_ensemble": np.asarray(
            plan.forward_fit_decision_ensemble, dtype=np.float64
        ),
        "forward_selection_decision_score_ensemble": np.asarray(
            plan.forward_selection_decision_ensemble, dtype=np.float64
        ),
        "backward_fit_decision_score_by_utility_seed": np.asarray(
            plan.backward_fit_decision_by_seed, dtype=np.float64
        ),
        "backward_selection_decision_score_by_utility_seed": np.asarray(
            plan.backward_selection_decision_by_seed, dtype=np.float64
        ),
        "backward_fit_decision_score_ensemble": np.asarray(
            plan.backward_fit_decision_ensemble, dtype=np.float64
        ),
        "backward_selection_decision_score_ensemble": np.asarray(
            plan.backward_selection_decision_ensemble, dtype=np.float64
        ),
        **_rule_values("bidirectional", plan.rule),
        **_rule_values("forward", plan.forward_rule),
        **_rule_values("backward", plan.backward_rule),
        **_policy_values("bidirectional", plan.policy),
        **_policy_values("forward", plan.forward_policy),
        **_policy_values("backward", plan.backward_policy),
    }
    context_roster = _policy_context_roster(plan)
    for method in METHOD_ROSTER:
        indptr, indices, context_sha = _encode_contexts(context_roster[method])
        values[f"{method}_context_indptr"] = indptr
        values[f"{method}_context_indices"] = indices
        values[f"{method}_context_sha256"] = np.asarray(context_sha)
        method_probability = np.asarray(probabilities[method], dtype=np.float32)
        values[f"{method}_probability_fold_ensemble"] = method_probability
        values[f"matrix_{method}_probability_fold_ensemble_sha256"] = np.asarray(
            _array_sha256(method_probability)
        )
    for name in tuple(values):
        if "score" in name and isinstance(values[name], np.ndarray) and values[name].ndim:
            values[f"matrix_{name}_sha256"] = np.asarray(_array_sha256(values[name]))
    return values


_FORBIDDEN_PRIVATE_KEY_FRAGMENTS = (
    "label",
    "target",
    "gold",
    "accuracy",
    "macro_f1",
    "nll",
    "brier",
    "metric",
)


def _expected_strategy_artifact_keys(
    upstream: VerifiedStrategyUpstreamState,
    plan: OutcomeFreeStrategyPlan,
    registered_variant: str,
    production_claim_sha256: str,
) -> frozenset[str]:
    rows = len(upstream.history.outcome.selection_protocol_row_ids)
    classes = len(upstream.history.outcome.label_order)
    probability = {
        method: np.full(
            (len(EXPECTED_SEEDS), rows, classes),
            1.0 / float(classes),
            dtype=np.float32,
        )
        for method in METHOD_ROSTER
    }
    template = _strategy_artifact_mapping(
        upstream=upstream,
        plan=plan,
        probabilities=probability,
        registered_variant=registered_variant,
        production_claim_sha256=production_claim_sha256,
        strategy_config_sha256={"contract": "0" * 64},
        strategy_code_sha256={"contract": "1" * 64},
        strategy_runtime_environment_sha256="2" * 64,
        strategy_live_lineage_sha256="3" * 64,
    )
    return frozenset(template)


def validate_strategy_artifact_mapping(
    values: Mapping[str, np.ndarray],
    *,
    upstream: VerifiedStrategyUpstreamState,
    plan: OutcomeFreeStrategyPlan,
    registered_variant: str,
    production_claim_sha256: str,
) -> None:
    """Validate the private strategy cache without computing performance."""

    expected_keys = _expected_strategy_artifact_keys(
        upstream, plan, registered_variant, production_claim_sha256
    )
    if set(values) != set(expected_keys):
        missing = sorted(expected_keys - set(values))
        unknown = sorted(set(values) - expected_keys)
        raise StrategyStagedPipelineError(
            f"strategy artifact schema changed: missing={missing}, unknown={unknown}"
        )
    if any(
        fragment in name.lower()
        for name in values
        for fragment in _FORBIDDEN_PRIVATE_KEY_FRAGMENTS
    ):
        raise StrategyStagedPipelineError(
            "strategy artifact contains an outcome or performance field"
        )
    if (
        _single_text(values["schema_version"], "schema_version")
        != STRATEGY_PRIVATE_SCHEMA
        or _single_text(values["dataset"], "dataset")
        != upstream.history.outcome.dataset
        or _single_text(values["registered_variant"], "registered_variant")
        != registered_variant
        or tuple(str(value) for value in np.asarray(values["registered_variants"]))
        != REGISTERED_VARIANTS
        or tuple(str(value) for value in np.asarray(values["method_roster"]))
        != METHOD_ROSTER
        or tuple(
            str(value) for value in np.asarray(values["joint_evaluation_roster"])
        )
        != JOINT_EVALUATION_ROSTER
        or _validated_class_order(values["dataset_class_order"], "class order")
        != upstream.history.outcome.label_order
        or tuple(
            int(value)
            for value in _integer_vector(values["base_seeds"], "base_seeds", unique=True)
        )
        != EXPECTED_SEEDS
        or tuple(
            int(value)
            for value in _integer_vector(
                values["utility_seeds"], "utility_seeds", unique=True
            )
        )
        != tuple(UTILITY_SEEDS)
    ):
        raise StrategyStagedPipelineError("strategy artifact identity changed")
    expected_scalars = {
        "production_run_claim_sha256": production_claim_sha256,
        "upstream_identity_sha256": upstream.upstream_identity_sha256,
        "history_artifact_sha256": upstream.history.attestation.artifact_sha256,
        "current_artifact_sha256": upstream.current.artifact_sha256,
        "full_current_anchor_history_artifact_sha256": (
            upstream.full_history_anchor.artifact_sha256
        ),
        "fit_task_sha256": upstream.history.outcome.fit_tasks.task_sha256,
        "selection_task_sha256": upstream.history.outcome.selection_tasks.task_sha256,
        "fit_supervision_sha256": plan.fit_supervision_sha256,
        "bidirectional_rule_sha256": plan.rule.rule_sha256,
        "forward_rule_sha256": plan.forward_rule.rule_sha256,
        "backward_rule_sha256": plan.backward_rule.rule_sha256,
        "bidirectional_policy_sha256": plan.policy.policy_sha256,
        "forward_policy_sha256": plan.forward_policy.policy_sha256,
        "backward_policy_sha256": plan.backward_policy.policy_sha256,
        "cross_variant_alignment_sha256": _cross_variant_alignment_sha256(upstream),
    }
    for name, expected in expected_scalars.items():
        if _require_sha256(_single_text(values[name], name), name) != expected:
            raise StrategyStagedPipelineError(f"strategy lineage changed: {name}")
    fit_rows = len(upstream.history.outcome.fit_protocol_row_ids)
    selection_rows = len(upstream.history.outcome.selection_protocol_row_ids)
    if (
        not np.array_equal(
            _integer_vector(values["fit_protocol_rows"], "fit_protocol_rows", unique=True),
            upstream.history.outcome.fit_protocol_row_ids,
        )
        or not np.array_equal(
            _integer_vector(
                values["selection_protocol_rows"],
                "selection_protocol_rows",
                unique=True,
            ),
            upstream.history.outcome.selection_protocol_row_ids,
        )
        or not np.array_equal(
            _integer_vector(values["fit_cluster_codes"], "fit_cluster_codes"),
            upstream.history.outcome.fit_cluster_codes,
        )
        or not np.array_equal(
            _integer_vector(
                values["selection_cluster_codes"], "selection_cluster_codes"
            ),
            upstream.history.outcome.selection_cluster_codes,
        )
        or set(
            _integer_vector(
                values["full_current_fit_protocol_rows"],
                "full_current_fit_protocol_rows",
                unique=True,
            ).tolist()
        )
        != set(upstream.history.outcome.fit_protocol_row_ids.tolist())
        or set(
            _integer_vector(
                values["full_current_selection_protocol_rows"],
                "full_current_selection_protocol_rows",
                unique=True,
            ).tolist()
        )
        != set(upstream.history.outcome.selection_protocol_row_ids.tolist())
    ):
        raise StrategyStagedPipelineError("strategy row alignment changed")
    context_roster = _policy_context_roster(plan)
    class_count = len(upstream.history.outcome.label_order)
    for method in METHOD_ROSTER:
        indptr = _integer_vector(
            values[f"{method}_context_indptr"], f"{method}.context_indptr"
        )
        indices = _integer_vector(
            values[f"{method}_context_indices"], f"{method}.context_indices"
        )
        if (
            indptr.shape != (selection_rows + 1,)
            or int(indptr[0]) != 0
            or int(indptr[-1]) != len(indices)
            or np.any(np.diff(indptr) < 0)
            or np.any(indices >= selection_rows)
        ):
            raise StrategyStagedPipelineError(f"{method} context CSR changed")
        decoded = tuple(
            tuple(int(value) for value in indices[indptr[row] : indptr[row + 1]])
            for row in range(selection_rows)
        )
        if decoded != context_roster[method]:
            raise StrategyStagedPipelineError(f"{method} contexts changed")
        context_sha = _canonical_sha256([list(row) for row in decoded])
        if (
            _require_sha256(
                _single_text(values[f"{method}_context_sha256"], "context_sha256"),
                "context_sha256",
            )
            != context_sha
        ):
            raise StrategyStagedPipelineError(f"{method} context hash changed")
        probability_name = f"{method}_probability_fold_ensemble"
        probability = _probability(
            values[probability_name],
            (len(EXPECTED_SEEDS), selection_rows, class_count),
            probability_name,
        )
        matrix_name = f"matrix_{probability_name}_sha256"
        if _require_sha256(
            _single_text(values[matrix_name], matrix_name), matrix_name
        ) != _array_sha256(probability):
            raise StrategyStagedPipelineError(f"{method} probability hash changed")
    score_shapes = {
        "bidirectional_fit_forward_score_by_utility_seed": (
            len(UTILITY_SEEDS),
            len(upstream.history.outcome.fit_tasks),
        ),
        "bidirectional_fit_backward_score_by_utility_seed": (
            len(UTILITY_SEEDS),
            len(upstream.history.outcome.fit_tasks),
        ),
        "bidirectional_fit_decision_score_by_utility_seed": (
            len(UTILITY_SEEDS),
            len(upstream.history.outcome.fit_tasks),
        ),
        "bidirectional_selection_forward_score_by_utility_seed": (
            len(UTILITY_SEEDS),
            len(upstream.history.outcome.selection_tasks),
        ),
        "bidirectional_selection_backward_score_by_utility_seed": (
            len(UTILITY_SEEDS),
            len(upstream.history.outcome.selection_tasks),
        ),
        "bidirectional_selection_decision_score_by_utility_seed": (
            len(UTILITY_SEEDS),
            len(upstream.history.outcome.selection_tasks),
        ),
        "forward_fit_decision_score_by_utility_seed": (
            len(UTILITY_SEEDS),
            len(upstream.history.outcome.fit_tasks),
        ),
        "forward_selection_decision_score_by_utility_seed": (
            len(UTILITY_SEEDS),
            len(upstream.history.outcome.selection_tasks),
        ),
        "backward_fit_decision_score_by_utility_seed": (
            len(UTILITY_SEEDS),
            len(upstream.history.outcome.fit_tasks),
        ),
        "backward_selection_decision_score_by_utility_seed": (
            len(UTILITY_SEEDS),
            len(upstream.history.outcome.selection_tasks),
        ),
    }
    for name, shape in score_shapes.items():
        array = np.asarray(values[name])
        if array.shape != shape or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise StrategyStagedPipelineError(f"strategy score shape changed: {name}")
        matrix_name = f"matrix_{name}_sha256"
        if _require_sha256(
            _single_text(values[matrix_name], matrix_name), matrix_name
        ) != _array_sha256(array):
            raise StrategyStagedPipelineError(f"strategy score hash changed: {name}")
    if fit_rows < 1:
        raise StrategyStagedPipelineError("strategy fit role is empty")


def _strategy_receipt(
    *,
    upstream: VerifiedStrategyUpstreamState,
    plan: OutcomeFreeStrategyPlan,
    registered_variant: str,
    production_claim_sha256: str,
    artifact_sha256: str,
    strategy_config_sha256: Mapping[str, str],
    strategy_code_sha256: Mapping[str, str],
    strategy_runtime_environment_sha256: str,
    strategy_live_lineage_sha256: str,
) -> dict[str, object]:
    outcome = upstream.history.outcome
    receipt: dict[str, object] = {
        "schema_version": STRATEGY_COMPLETION_RECEIPT_SCHEMA,
        "status": STRATEGY_COMPLETION_STATUS,
        "dataset": outcome.dataset,
        "claim_boundary": (
            "Fit-only utility supervision plus completed feature-only caches and "
            "complete-checkpoint inference; no model-selection outcome or performance "
            "evaluation was consumed."
        ),
        "lineage": {
            "history_completion_receipt_sha256": (
                upstream.history.attestation.completion_receipt_sha256
            ),
            "history_fit_receipt_sha256": (
                upstream.history.attestation.fit_producer_receipt_sha256
            ),
            "history_artifact_sha256": upstream.history.attestation.artifact_sha256,
            "history_production_claim_sha256": (
                upstream.history.attestation.production_run_claim_sha256
            ),
            "current_completion_receipt_sha256": (
                upstream.current.completion_receipt_sha256
            ),
            "current_fit_receipt_sha256": (
                upstream.current.fit_producer_receipt_sha256
            ),
            "current_artifact_sha256": upstream.current.artifact_sha256,
            "current_production_claim_sha256": (
                upstream.current.production_run_claim_sha256
            ),
            "full_current_anchor_history_artifact_sha256": (
                upstream.full_history_anchor.artifact_sha256
            ),
            "full_current_anchor_history_completion_receipt_sha256": (
                upstream.full_history_anchor.completion_receipt_sha256
            ),
            "full_current_anchor_history_production_claim_sha256": (
                upstream.full_history_anchor.production_run_claim_sha256
            ),
            "history_source_identity_sha256": outcome.source_identity_sha256,
            "current_source_identity_sha256": (
                upstream.current.current_only_source_identity_sha256
            ),
            "history_checkpoint_manifest_sha256": (
                outcome.checkpoint_manifest_sha256
            ),
            "current_checkpoint_manifest_sha256": (
                upstream.current.current_checkpoint_manifest_sha256
            ),
            "fit_feature_identity_sha256": (
                upstream.history.fit_features.feature_identity_sha256
            ),
            "selection_feature_identity_sha256": (
                upstream.history.selection_features.feature_identity_sha256
            ),
            "fit_task_sha256": outcome.fit_tasks.task_sha256,
            "selection_task_sha256": outcome.selection_tasks.task_sha256,
            "history_fold_assignment_sha256": _array_sha256(
                np.asarray(
                    getattr(upstream.history.fit_outcome, "fold_by_seed_query")
                )
            ),
            "cross_variant_alignment_sha256": _cross_variant_alignment_sha256(
                upstream
            ),
            "fit_supervision_sha256": plan.fit_supervision_sha256,
            "bidirectional_model_spec_sha256": plan.model_spec_sha256,
            "forward_model_spec_sha256": plan.forward_model_spec_sha256,
            "backward_model_spec_sha256": plan.backward_model_spec_sha256,
            "bidirectional_score_source_identity_sha256": (
                plan.score_source_identity_sha256
            ),
            "forward_score_source_identity_sha256": (
                plan.forward_score_source_identity_sha256
            ),
            "backward_score_source_identity_sha256": (
                plan.backward_score_source_identity_sha256
            ),
            "bidirectional_rule_sha256": plan.rule.rule_sha256,
            "forward_rule_sha256": plan.forward_rule.rule_sha256,
            "backward_rule_sha256": plan.backward_rule.rule_sha256,
            "bidirectional_policy_sha256": plan.policy.policy_sha256,
            "forward_policy_sha256": plan.forward_policy.policy_sha256,
            "backward_policy_sha256": plan.backward_policy.policy_sha256,
            "all_history_context_sha256": plan.all_history_context_sha256,
            "history_code_bundle_sha256": _canonical_sha256(
                upstream.history.code_sha256
            ),
            "history_execution_environment_sha256": (
                upstream.history.execution_environment_sha256
            ),
            "current_source_code_sha256": upstream.current.source_code_sha256,
            "current_runtime_environment_sha256": (
                upstream.current.runtime_environment_sha256
            ),
            "strategy_config_bundle_sha256": _canonical_sha256(
                dict(sorted(strategy_config_sha256.items()))
            ),
            "strategy_code_bundle_sha256": _canonical_sha256(
                dict(sorted(strategy_code_sha256.items()))
            ),
            "strategy_runtime_environment_sha256": (
                strategy_runtime_environment_sha256
            ),
            "strategy_live_lineage_sha256": strategy_live_lineage_sha256,
            "upstream_identity_sha256": upstream.upstream_identity_sha256,
            "production_run_claim_sha256": production_claim_sha256,
            "private_strategy_artifact_sha256": artifact_sha256,
        },
        "completion_contract": {
            "protocol": STRATEGY_PROTOCOL,
            "registered_variant": registered_variant,
            "registered_variants": list(REGISTERED_VARIANTS),
            "method_roster": list(METHOD_ROSTER),
            "joint_evaluation_roster": list(JOINT_EVALUATION_ROSTER),
            "base_seeds": list(EXPECTED_SEEDS),
            "utility_seeds": list(UTILITY_SEEDS),
            "outer_folds": upstream.history.checkpoint_manifest.outer_folds,
            "fit_query_count": len(outcome.fit_protocol_row_ids),
            "selection_query_count": len(outcome.selection_protocol_row_ids),
            "fit_task_count": len(outcome.fit_tasks),
            "selection_task_count": len(outcome.selection_tasks),
            "bidirectional_fit_pair_count": plan.rule.fit_pair_count,
            "bidirectional_fit_selected_count": plan.rule.fit_selected_count,
            "bidirectional_selection_pair_count": plan.policy.selected_pair_count,
            "forward_fit_pair_count": plan.forward_rule.fit_pair_count,
            "forward_fit_selected_count": plan.forward_rule.fit_selected_count,
            "forward_selection_pair_count": plan.forward_policy.selected_pair_count,
            "backward_fit_pair_count": plan.backward_rule.fit_pair_count,
            "backward_fit_selected_count": plan.backward_rule.fit_selected_count,
            "backward_selection_pair_count": plan.backward_policy.selected_pair_count,
            "registered_fit_coverage_percent": 25,
            "group_oof_utility_training": True,
            "directional_heads_ensembled_before_strict_minimum": True,
            "recency_cardinality_matched_per_query": True,
            "all_history_diagnostic_included": True,
            "complete_history_checkpoint_only": True,
            "one_probability_per_base_seed_method_and_query": True,
            "fit_supervision_consumed": True,
            "model_selection_outcome_materialized": False,
            "model_selection_outcome_deserialized": False,
            "evaluate_stage_run": False,
            "performance_computed": False,
            "reference_source_contract": {
                "independent_current_only": "single_full_history_anchor_current_artifact",
                "bidirectional_selected_history": "registered_variant_history_checkpoints",
                "forward_only_selected_history": "registered_variant_history_checkpoints",
                "backward_only_selected_history": "registered_variant_history_checkpoints",
                "coverage_matched_recency": "registered_variant_history_checkpoints",
                "all_history_diagnostic": "registered_variant_history_checkpoints",
            },
        },
        "public_artifact_policy": {
            "aggregate_only": True,
            "contains_row_level_values": False,
            "contains_private_paths": False,
            "contains_outcomes_or_performance": False,
        },
    }
    _validate_aggregate_producer_receipt(receipt)
    return receipt


def complete_strategy_selection(
    *,
    upstream: VerifiedStrategyUpstreamState,
    registered_variant: str,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    private_output_root: str | Path,
    config_paths: Mapping[str, str | Path],
    code_paths: Mapping[str, str | Path],
    environment: Mapping[str, object],
    device: torch.device,
) -> CompletedStrategyProduction:
    """Complete the fixed strategy roster without accepting selection outcomes.

    There is deliberately no injectable inference callback on this production
    entry point.  Every probability must come from ``train_one_fold_seed`` with
    ``require_complete_checkpoint=True`` and the exact attested history tree.
    """

    _require_registered_variant_matches_model_config(registered_variant, model_config)
    _assert_upstream_unchanged(upstream)
    _validate_history_model_contract(upstream, model_config, run_config)
    config_sha, code_sha, runtime_sha, live_lineage_sha = _live_strategy_lineage(
        config_paths=config_paths,
        code_paths=code_paths,
        environment=environment,
    )
    plan = build_outcome_free_strategy(upstream)
    claim_sha = strategy_production_claim_sha256(
        upstream=upstream,
        plan=plan,
        registered_variant=registered_variant,
        model_config=model_config,
        run_config=run_config,
        strategy_config_sha256=config_sha,
        strategy_code_sha256=code_sha,
        strategy_runtime_environment_sha256=runtime_sha,
    )
    root = _claim_strategy_private_root(private_output_root, claim_sha)
    paths = strategy_private_paths(root)
    _verify_strategy_private_claim(root, claim_sha)
    probabilities = _infer_registered_method_probabilities(
        upstream=upstream,
        plan=plan,
        model_config=model_config,
        run_config=run_config,
        device=device,
    )
    # Revalidate every mutable input after the long checkpoint pass and before
    # serialising a byte of strategy output.
    _assert_upstream_unchanged(upstream)
    observed_config, observed_code, observed_runtime, observed_lineage = (
        _live_strategy_lineage(
            config_paths=config_paths,
            code_paths=code_paths,
            environment=environment,
        )
    )
    if (
        observed_config != config_sha
        or observed_code != code_sha
        or observed_runtime != runtime_sha
        or observed_lineage != live_lineage_sha
    ):
        raise StrategyStagedPipelineError(
            "strategy config/code/runtime lineage changed during inference"
        )
    _verify_strategy_private_claim(root, claim_sha)
    values = _strategy_artifact_mapping(
        upstream=upstream,
        plan=plan,
        probabilities=probabilities,
        registered_variant=registered_variant,
        production_claim_sha256=claim_sha,
        strategy_config_sha256=config_sha,
        strategy_code_sha256=code_sha,
        strategy_runtime_environment_sha256=runtime_sha,
        strategy_live_lineage_sha256=live_lineage_sha,
    )
    validate_strategy_artifact_mapping(
        values,
        upstream=upstream,
        plan=plan,
        registered_variant=registered_variant,
        production_claim_sha256=claim_sha,
    )
    try:
        artifact_sha = _write_npz_once(paths["artifact"], values)
    except ValueError as error:
        raise StrategyStagedPipelineError(
            f"cannot publish private strategy artifact: {error}"
        ) from error
    receipt = _strategy_receipt(
        upstream=upstream,
        plan=plan,
        registered_variant=registered_variant,
        production_claim_sha256=claim_sha,
        artifact_sha256=artifact_sha,
        strategy_config_sha256=config_sha,
        strategy_code_sha256=code_sha,
        strategy_runtime_environment_sha256=runtime_sha,
        strategy_live_lineage_sha256=live_lineage_sha,
    )
    try:
        receipt_sha = _write_json_once(paths["receipt"], receipt)
    except ValueError as error:
        raise StrategyStagedPipelineError(
            f"cannot publish strategy receipt: {error}"
        ) from error
    if (
        _file_sha256(paths["artifact"]) != artifact_sha
        or _file_sha256(paths["receipt"]) != receipt_sha
    ):
        raise StrategyStagedPipelineError("strategy output changed after publication")
    _verify_strategy_private_claim(root, claim_sha)
    _assert_upstream_unchanged(upstream)
    return CompletedStrategyProduction(
        artifact_path=paths["artifact"],
        artifact_sha256=artifact_sha,
        receipt_path=paths["receipt"],
        receipt_sha256=receipt_sha,
        production_run_claim_sha256=claim_sha,
        policy_sha256=plan.policy.policy_sha256,
    )


def load_completed_strategy_artifact(
    *,
    artifact_path: str | Path,
    expected_artifact_sha256: str,
    upstream: VerifiedStrategyUpstreamState,
    plan: OutcomeFreeStrategyPlan,
    registered_variant: str,
    production_run_claim_sha256: str,
) -> dict[str, np.ndarray]:
    """Reload and validate a private cache; this function never evaluates it."""

    path = Path(artifact_path).resolve()
    expected = _require_sha256(expected_artifact_sha256, "expected_artifact_sha256")
    if _file_sha256(path) != expected:
        raise StrategyStagedPipelineError("strategy artifact file hash changed")
    try:
        with np.load(path, allow_pickle=False) as archive:
            values = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise StrategyStagedPipelineError(
            f"cannot read completed strategy artifact: {error}"
        ) from error
    validate_strategy_artifact_mapping(
        values,
        upstream=upstream,
        plan=plan,
        registered_variant=registered_variant,
        production_claim_sha256=production_run_claim_sha256,
    )
    if _file_sha256(path) != expected:
        raise StrategyStagedPipelineError("strategy artifact changed while validating")
    return values


def verify_strategy_completion_production_attestation(
    artifact_path: str | Path,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
    *,
    upstream: VerifiedStrategyUpstreamState,
) -> VerifiedStrategyCompletionAttestation:
    """Verify a production strategy artifact for the joint outcome evaluator."""

    _assert_upstream_unchanged(upstream)
    artifact_file = Path(artifact_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    root = artifact_file.parent
    paths = strategy_private_paths(root)
    if (
        artifact_file != paths["artifact"]
        or receipt_file != paths["receipt"]
        or not root.is_absolute()
        or root == Path(root.anchor)
        or _is_within(root, _repository_root())
    ):
        raise StrategyStagedPipelineError(
            "strategy completion must use one canonical external private root"
        )
    expected_receipt = _require_sha256(
        expected_receipt_sha256, "expected_receipt_sha256"
    )
    if _file_sha256(receipt_file) != expected_receipt:
        raise StrategyStagedPipelineError("strategy receipt file hash changed")
    receipt = _read_json_mapping(receipt_file, "strategy completion receipt")
    if set(receipt) != {
        "schema_version",
        "status",
        "dataset",
        "claim_boundary",
        "lineage",
        "completion_contract",
        "public_artifact_policy",
    }:
        raise StrategyStagedPipelineError("strategy receipt schema changed")
    if (
        receipt.get("schema_version") != STRATEGY_COMPLETION_RECEIPT_SCHEMA
        or receipt.get("status") != STRATEGY_COMPLETION_STATUS
        or receipt.get("dataset") != upstream.history.outcome.dataset
        or receipt.get("claim_boundary")
        != (
            "Fit-only utility supervision plus completed feature-only caches and "
            "complete-checkpoint inference; no model-selection outcome or performance "
            "evaluation was consumed."
        )
    ):
        raise StrategyStagedPipelineError("strategy receipt is not production evidence")
    lineage = receipt.get("lineage")
    contract = receipt.get("completion_contract")
    if not isinstance(lineage, dict) or not isinstance(contract, dict):
        raise StrategyStagedPipelineError("strategy receipt lacks lineage/contract")
    variant = contract.get("registered_variant")
    if variant not in REGISTERED_VARIANTS:
        raise StrategyStagedPipelineError("strategy receipt variant changed")
    claim_sha = _require_sha256(
        lineage.get("production_run_claim_sha256"),
        "production_run_claim_sha256",
    )
    artifact_sha = _require_sha256(
        lineage.get("private_strategy_artifact_sha256"),
        "private_strategy_artifact_sha256",
    )
    if _file_sha256(artifact_file) != artifact_sha:
        raise StrategyStagedPipelineError("strategy artifact hash changed")
    plan = build_outcome_free_strategy(upstream)
    template = _strategy_receipt(
        upstream=upstream,
        plan=plan,
        registered_variant=str(variant),
        production_claim_sha256=claim_sha,
        artifact_sha256=artifact_sha,
        strategy_config_sha256={"contract": "0" * 64},
        strategy_code_sha256={"contract": "1" * 64},
        strategy_runtime_environment_sha256="2" * 64,
        strategy_live_lineage_sha256="3" * 64,
    )
    template_lineage = cast(Mapping[str, object], template["lineage"])
    if set(lineage) != set(template_lineage):
        raise StrategyStagedPipelineError("strategy receipt lineage schema changed")
    if contract != template["completion_contract"]:
        raise StrategyStagedPipelineError("strategy completion contract changed")
    if receipt.get("public_artifact_policy") != template["public_artifact_policy"]:
        raise StrategyStagedPipelineError("strategy public artifact policy changed")
    dynamic = {
        "strategy_config_bundle_sha256",
        "strategy_code_bundle_sha256",
        "strategy_runtime_environment_sha256",
        "strategy_live_lineage_sha256",
    }
    for name in template_lineage:
        observed = _require_sha256(lineage.get(name), name)
        if name not in dynamic and observed != template_lineage[name]:
            raise StrategyStagedPipelineError(f"strategy receipt lineage changed: {name}")
    values = load_completed_strategy_artifact(
        artifact_path=artifact_file,
        expected_artifact_sha256=artifact_sha,
        upstream=upstream,
        plan=plan,
        registered_variant=str(variant),
        production_run_claim_sha256=claim_sha,
    )
    artifact_dynamic_fields = {
        "strategy_config_bundle_sha256": "strategy_config_bundle_sha256",
        "strategy_code_bundle_sha256": "strategy_code_bundle_sha256",
        "strategy_runtime_environment_sha256": (
            "strategy_runtime_environment_sha256"
        ),
        "strategy_live_lineage_sha256": "strategy_live_lineage_sha256",
    }
    for receipt_name, artifact_name in artifact_dynamic_fields.items():
        artifact_value = _require_sha256(
            _single_text(values[artifact_name], artifact_name), artifact_name
        )
        if _require_sha256(lineage[receipt_name], receipt_name) != artifact_value:
            raise StrategyStagedPipelineError(
                f"strategy artifact/receipt lineage differs: {receipt_name}"
            )
    _verify_strategy_private_claim(root, claim_sha)
    if (
        _file_sha256(receipt_file) != expected_receipt
        or _file_sha256(artifact_file) != artifact_sha
    ):
        raise StrategyStagedPipelineError(
            "strategy production changed while attesting completion"
        )
    _assert_upstream_unchanged(upstream)
    return VerifiedStrategyCompletionAttestation(
        dataset=upstream.history.outcome.dataset,
        registered_variant=str(variant),
        artifact_path=artifact_file,
        artifact_sha256=artifact_sha,
        receipt_path=receipt_file,
        receipt_sha256=expected_receipt,
        production_run_claim_sha256=claim_sha,
        cross_variant_alignment_sha256=_require_sha256(
            lineage["cross_variant_alignment_sha256"],
            "cross_variant_alignment_sha256",
        ),
        variant_history_artifact_sha256=_require_sha256(
            lineage["history_artifact_sha256"], "history_artifact_sha256"
        ),
        full_current_anchor_history_artifact_sha256=_require_sha256(
            lineage["full_current_anchor_history_artifact_sha256"],
            "full_current_anchor_history_artifact_sha256",
        ),
        current_artifact_sha256=_require_sha256(
            lineage["current_artifact_sha256"], "current_artifact_sha256"
        ),
        method_roster=METHOD_ROSTER,
        joint_evaluation_roster=JOINT_EVALUATION_ROSTER,
        base_seeds=EXPECTED_SEEDS,
        utility_seeds=tuple(UTILITY_SEEDS),
        fit_query_count=int(contract["fit_query_count"]),
        selection_query_count=int(contract["selection_query_count"]),
        fit_task_count=int(contract["fit_task_count"]),
        selection_task_count=int(contract["selection_task_count"]),
    )
