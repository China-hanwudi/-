"""Production bridge for the independently trained current-only baseline.

The pipeline has three deliberately separate capabilities:

``fit``
    May see fit features and fit labels.  It trains history-stripped outer-fold
    models and writes fit OOF probabilities plus an aggregate receipt.
``complete-selection``
    May see fit material again and model-selection *features only*.  It first
    verifies the complete checkpoint manifest, then loads completed checkpoints
    in inference-only mode and writes the full current-only probability cache.
``evaluate``
    Is intentionally absent here.  Model-selection labels remain reserved for
    the downstream evidence evaluator.

The full private cache uses the exact schema consumed by
``causal_backbone_evidence.load_current_only_artifact``.  Public receipts never
contain row identities, probabilities, labels, paths, or performance metrics.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast

import numpy as np
import torch
import joblib

from .causal_backbone_evidence import (
    IndependentCurrentOnlyFold,
    build_current_only_artifact_mapping,
    independent_current_only_source_identity,
    predict_independent_current_only_probability,
    train_independent_current_only_fold_seed,
)
from .causal_backbone_evidence_runner import (
    EXPECTED_SEEDS,
    FIT_ROLE,
    PRODUCER_CACHE_SCHEMA,
    SELECTION_ROLE,
    CheckpointManifest,
    FitOnlyProducerView,
    FitRoleView,
    SelectionFeatureView,
    _array_sha256,
    _canonical_sha256,
    _file_sha256,
    _integer_vector,
    _require_sha256,
    _single_text,
    build_checkpoint_manifest,
    verify_checkpoint_manifest,
)
from .causal_backbone_evidence_stage_b import (
    CURRENT_ONLY_FIT_PRODUCER_RECEIPT_SCHEMA,
    CURRENT_ONLY_PRODUCTION_EXECUTION_MODE,
    CurrentOnlyFitProduction,
    CurrentOnlyFoldOutput,
    FitOnlyLineage,
    FitProtocolMap,
    HistoryFreeFoldRequest,
    StageBContractError,
    _assert_producer_sidecar_lineage,
    _atomic_json_once,
    _atomic_savez_once,
    _attest_production_current_only_fold_callback,
    _load_receipt,
    _validate_aggregate_producer_receipt,
    _verify_fit_only_lineage_file,
    _verify_fit_protocol_map_file,
    align_fit_protocol_to_producer,
    load_private_npz_mapping,
    produce_independent_current_only_fit_oof,
    validate_current_only_fit_bootstrap_artifact,
    verify_live_fit_arrays,
)
from .causal_backbone_history_staged_pipeline import (
    HISTORY_COMPLETE_OUTCOME_SCHEMA,
    VerifiedHistoryCompletionAttestation,
)
from .causal_multimodal_backbone import CausalBackboneConfig, CausalMultimodalBackbone
from .emotiontalk_causal_backbone_runner import (
    BackboneRunConfig,
    CrossfitSplit,
    FoldTextProcessor,
    OpenRoleCorpus,
    _capture_rng_state,
    _indices_sha256,
    _restore_rng_state,
    _torch_load_local,
    make_crossfit_splits,
    predict_one_probability_per_query,
)


CURRENT_ONLY_COMPLETION_RECEIPT_SCHEMA = (
    "carma_independent_current_only_complete_selection_receipt_v1"
)
CURRENT_ONLY_COMPLETION_PROTOCOL = (
    "fit_oof_plus_selection_feature_only_complete_checkpoint_fold_ensemble_v1"
)
CURRENT_ONLY_PRIVATE_CLAIM_SCHEMA = "carma_current_only_private_run_claim_v1"
CURRENT_ONLY_PRIVATE_CLAIM_NAME = "current-only-run-claim.json"
CURRENT_ONLY_PRIVATE_LOCK_NAME = ".current-only-fit.lock"
CURRENT_ONLY_PRIVATE_CHECKPOINT_NAME = "checkpoints"
CURRENT_ONLY_PRIVATE_FIT_ARTIFACT_NAME = "current-only-fit.npz"
CURRENT_ONLY_PRIVATE_FIT_RECEIPT_NAME = "current-only-fit-receipt.json"
CURRENT_ONLY_PRIVATE_COMPLETE_ARTIFACT_NAME = "current-only-complete.npz"
CURRENT_ONLY_PRIVATE_COMPLETE_RECEIPT_NAME = "current-only-complete-receipt.json"


class CurrentOnlyPipelineError(StageBContractError):
    """Raised when the fit/completion/evaluate capability boundary is crossed."""


@dataclass(frozen=True)
class CurrentOnlyProducerAlignmentView:
    """Outcome-free producer fields allowed during complete-selection."""

    dataset: str
    label_order: tuple[str, ...]
    seeds: tuple[int, ...]
    protocol_row_ids: np.ndarray
    fit_query_indices: np.ndarray
    selection_query_indices: np.ndarray
    fit_cluster_codes: np.ndarray
    selection_cluster_codes: np.ndarray
    producer_file_sha256: str
    source_identity_sha256: str
    checkpoint_manifest_sha256: str
    source_hashes: Mapping[str, str]


@dataclass(frozen=True)
class AttestedHistoryFitAlignmentView:
    """Fit-only alignment read from one production-attested history cache."""

    dataset: str
    label_order: tuple[str, ...]
    seeds: tuple[int, ...]
    protocol_row_ids: np.ndarray
    fit_query_indices: np.ndarray
    fit_cluster_codes: np.ndarray
    producer_file_sha256: str
    source_identity_sha256: str
    checkpoint_manifest_sha256: str
    source_hashes: Mapping[str, str]


@dataclass(frozen=True)
class VerifiedCurrentOnlyFitState:
    artifact_path: Path
    artifact_sha256: str
    producer_receipt_path: Path
    producer_receipt_sha256: str
    fit_preflight_receipt_sha256: str
    fit_lineage_artifact_sha256: str
    fit_lineage_source_identity_sha256: str
    history_producer_file_sha256: str
    history_producer_source_identity_sha256: str
    history_checkpoint_manifest_sha256: str
    selection_feature_sha256: str
    checkpoint_manifest: CheckpointManifest
    current_only_source_identity_sha256: str
    fit_array_hash_bundle_sha256: str
    fold_assignment_sha256: str
    producer_execution_mode: str
    model_config_sha256: str
    run_config_sha256: str
    model_config_semantic_sha256: str
    run_config_semantic_sha256: str
    source_code_sha256: str
    runtime_environment_sha256: str
    production_run_claim_sha256: str


@dataclass(frozen=True)
class CompletedCurrentOnlyProduction:
    artifact_path: Path
    artifact_sha256: str
    receipt_path: Path
    receipt_sha256: str
    checkpoint_manifest_sha256: str


def _probability(value: np.ndarray, shape: tuple[int, ...], field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.floating):
        raise CurrentOnlyPipelineError(f"{field} is not probability-shaped")
    if (
        not np.isfinite(array).all()
        or np.any(array < 0.0)
        or not np.allclose(array.sum(axis=-1), 1.0, rtol=1.0e-5, atol=1.0e-6)
    ):
        raise CurrentOnlyPipelineError(f"{field} contains invalid probabilities")
    return array.astype(np.float32, copy=True)


def _validate_current_only_model_label_contract(
    model_config: CausalBackboneConfig,
    label_order: Sequence[str],
) -> None:
    """Bind optional VAD targets to the dataset's registered class order."""

    model_config.validate()
    labels = tuple(str(value) for value in label_order)
    if model_config.num_classes != len(labels):
        raise CurrentOnlyPipelineError("model class count differs from dataset label order")
    if (
        model_config.auxiliary_vad_weight > 0.0
        and tuple(model_config.emotion_label_order) != labels
    ):
        raise CurrentOnlyPipelineError(
            "VAD supervision label order differs from the fit dataset"
        )


def _model_config_semantic_sha256(model_config: CausalBackboneConfig) -> str:
    model_config.validate()
    return _canonical_sha256(asdict(model_config))


def _run_config_semantic_sha256(run_config: BackboneRunConfig) -> str:
    baseline = replace(run_config, subset_dropout_probability=0.0)
    baseline.validate()
    return _canonical_sha256(asdict(baseline))


def _verified_live_fit_bundle(fit: FitRoleView) -> str:
    try:
        return verify_live_fit_arrays(fit)
    except StageBContractError as error:
        raise CurrentOnlyPipelineError(str(error)) from error


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def current_only_private_paths(root: str | Path) -> dict[str, Path]:
    """Return the only allowed production paths under one private run root."""

    private_root = Path(root).resolve()
    return {
        "claim": private_root / CURRENT_ONLY_PRIVATE_CLAIM_NAME,
        "lock": private_root / CURRENT_ONLY_PRIVATE_LOCK_NAME,
        "checkpoint": private_root / CURRENT_ONLY_PRIVATE_CHECKPOINT_NAME,
        "fit_artifact": private_root / CURRENT_ONLY_PRIVATE_FIT_ARTIFACT_NAME,
        "fit_receipt": private_root / CURRENT_ONLY_PRIVATE_FIT_RECEIPT_NAME,
        "complete_artifact": (
            private_root / CURRENT_ONLY_PRIVATE_COMPLETE_ARTIFACT_NAME
        ),
        "complete_receipt": (
            private_root / CURRENT_ONLY_PRIVATE_COMPLETE_RECEIPT_NAME
        ),
    }


def _validate_private_root_location(root: Path) -> None:
    if (
        not root.is_absolute()
        or root == Path(root.anchor)
        or _is_within(root, _repository_root())
    ):
        raise CurrentOnlyPipelineError(
            "current-only private root must be a safe repository-external path"
        )


def current_only_production_claim_sha256(
    *,
    fit: FitRoleView,
    fit_map: FitProtocolMap,
    lineage: FitOnlyLineage,
    fit_preflight_receipt_sha256: str,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    model_config_sha256: str,
    run_config_sha256: str,
    source_code_sha256: str,
    runtime_environment_sha256: str,
) -> str:
    """Bind an interruptible current-only run to one exact live lineage."""

    fit_bundle = _verified_live_fit_bundle(fit)
    if fit_bundle != lineage.fit_array_hash_bundle_sha256:
        raise CurrentOnlyPipelineError("current-only claim fit arrays differ from lineage")
    return _canonical_sha256(
        {
            "protocol": CURRENT_ONLY_COMPLETION_PROTOCOL,
            "producer_execution_mode": CURRENT_ONLY_PRODUCTION_EXECUTION_MODE,
            "dataset": fit.dataset,
            "fit_contract_sha256": fit.contract_sha256,
            "fit_array_hash_bundle_sha256": fit_bundle,
            "fit_protocol_map_sha256": fit_map.artifact_sha256,
            "fit_lineage_artifact_sha256": lineage.artifact_sha256,
            "fit_lineage_source_identity_sha256": lineage.source_identity_sha256,
            "fit_preflight_receipt_sha256": _require_sha256(
                fit_preflight_receipt_sha256,
                "fit_preflight_receipt_sha256",
            ),
            "model_config_sha256": _require_sha256(
                model_config_sha256, "model_config_sha256"
            ),
            "run_config_sha256": _require_sha256(
                run_config_sha256, "run_config_sha256"
            ),
            "model_config_semantic_sha256": _model_config_semantic_sha256(
                model_config
            ),
            "run_config_semantic_sha256": _run_config_semantic_sha256(run_config),
            "source_code_sha256": _require_sha256(
                source_code_sha256, "source_code_sha256"
            ),
            "runtime_environment_sha256": _require_sha256(
                runtime_environment_sha256, "runtime_environment_sha256"
            ),
        }
    )


def _claim_payload(claim_sha256: str) -> dict[str, object]:
    return {
        "schema_version": CURRENT_ONLY_PRIVATE_CLAIM_SCHEMA,
        "status": "claimed_for_single_lineage_interruptible_fit",
        "production_claim_sha256": _require_sha256(
            claim_sha256, "production_claim_sha256"
        ),
    }


def claim_or_resume_current_only_private_root(
    path: str | Path,
    *,
    production_claim_sha256: str,
    allow_resume: bool,
) -> Path:
    """Atomically claim a new private root or validate an unfinished resume."""

    root = Path(path).resolve()
    _validate_private_root_location(root)
    if not root.exists():
        if not root.parent.is_dir():
            raise CurrentOnlyPipelineError(
                "current-only private root parent must already exist"
            )
        try:
            root.mkdir(exist_ok=False)
        except FileExistsError:
            raise FileExistsError(
                "current-only private root was concurrently claimed"
            ) from None
        _atomic_json_once(
            current_only_private_paths(root)["claim"],
            _claim_payload(production_claim_sha256),
        )
        return root
    if not allow_resume:
        raise FileExistsError("current-only private root must be all-new")
    if not root.is_dir():
        raise CurrentOnlyPipelineError("current-only resume root is not a directory")
    _verify_current_only_private_claim(root, production_claim_sha256)
    allowed = {
        CURRENT_ONLY_PRIVATE_CLAIM_NAME,
        CURRENT_ONLY_PRIVATE_LOCK_NAME,
        CURRENT_ONLY_PRIVATE_CHECKPOINT_NAME,
    }
    unexpected = sorted(entry.name for entry in root.iterdir() if entry.name not in allowed)
    if unexpected:
        raise CurrentOnlyPipelineError(
            f"current-only resume root contains finalized/unexpected artifacts: {unexpected}"
        )
    checkpoint = current_only_private_paths(root)["checkpoint"]
    if checkpoint.exists() and not checkpoint.is_dir():
        raise CurrentOnlyPipelineError("current-only resume checkpoint path is not a directory")
    return root


def _verify_current_only_private_claim(root: Path, expected_sha256: str) -> None:
    _validate_private_root_location(root)
    marker = current_only_private_paths(root)["claim"]
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CurrentOnlyPipelineError(
            "current-only private claim is missing or corrupt"
        ) from error
    if payload != _claim_payload(expected_sha256):
        raise CurrentOnlyPipelineError("current-only private claim lineage changed")


def _validate_current_only_private_layout(
    *,
    checkpoint_root: str | Path,
    fit_artifact_path: str | Path,
    fit_receipt_path: str | Path,
    complete_artifact_path: str | Path | None = None,
    complete_receipt_path: str | Path | None = None,
) -> Path:
    checkpoint = Path(checkpoint_root).resolve()
    root = checkpoint.parent
    _validate_private_root_location(root)
    expected = current_only_private_paths(root)
    observed = {
        "checkpoint": checkpoint,
        "fit_artifact": Path(fit_artifact_path).resolve(),
        "fit_receipt": Path(fit_receipt_path).resolve(),
    }
    if complete_artifact_path is not None:
        observed["complete_artifact"] = Path(complete_artifact_path).resolve()
    if complete_receipt_path is not None:
        observed["complete_receipt"] = Path(complete_receipt_path).resolve()
    if observed != {name: expected[name] for name in observed}:
        raise CurrentOnlyPipelineError(
            "current-only production artifacts must share one canonical private root"
        )
    return root


class _ExclusiveCurrentOnlyFitLock:
    """Crash-released, non-blocking OS advisory lock for one production fit."""

    def __init__(self, root: Path) -> None:
        self.path = current_only_private_paths(root)["lock"]
        self.handle = None
        self._windows = os.name == "nt"

    def __enter__(self) -> "_ExclusiveCurrentOnlyFitLock":
        self.handle = self.path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"\0")
            self.handle.flush()
            os.fsync(self.handle.fileno())
        self.handle.seek(0)
        try:
            if self._windows:
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(
                    self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                )
        except (OSError, BlockingIOError):
            self.handle.close()
            self.handle = None
            raise CurrentOnlyPipelineError(
                "another current-only fit process holds the private-root lock"
            ) from None
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.handle is None:
            return
        try:
            self.handle.seek(0)
            if self._windows:
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


def _verify_complete_checkpoint_payloads(
    checkpoint_root: str | Path,
    manifest: CheckpointManifest,
    *,
    fit: FitRoleView,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    source_identity_sha256: str,
    fold_by_seed_row: np.ndarray,
    device: torch.device,
) -> None:
    """Strictly restore every processor/checkpoint before selection opens."""

    root = Path(checkpoint_root)
    verify_checkpoint_manifest(root, manifest)
    source_identity = _require_sha256(
        source_identity_sha256, "current_only_source_identity_sha256"
    )
    baseline_run_config = replace(run_config, subset_dropout_probability=0.0)
    baseline_run_config.validate()
    _validate_current_only_model_label_contract(model_config, fit.label_order)
    folds = np.asarray(fold_by_seed_row)
    if (
        folds.shape != (len(manifest.seeds), fit.rows)
        or not np.issubdtype(folds.dtype, np.integer)
        or np.any((folds < 0) | (folds >= manifest.outer_folds))
    ):
        raise CurrentOnlyPipelineError("checkpoint fold assignment shape changed")
    root = Path(checkpoint_root)
    checkpoint_records = [
        record for record in manifest.records if record.kind == "checkpoint"
    ]
    if len(checkpoint_records) != len(manifest.seeds) * manifest.outer_folds:
        raise CurrentOnlyPipelineError("checkpoint manifest has incomplete fold coverage")
    by_key = {
        (record.seed, record.fold, record.kind): record
        for record in manifest.records
    }
    for record in checkpoint_records:
        seed_index = manifest.seeds.index(int(record.seed))
        held = np.flatnonzero(folds[seed_index] == int(record.fold)).astype(np.int64)
        train = np.flatnonzero(folds[seed_index] != int(record.fold)).astype(np.int64)
        if not len(held) or not len(train):
            raise CurrentOnlyPipelineError("checkpoint fold partition is empty")
        corpus = _fit_corpus_from_view(
            fit,
            model_config=model_config,
            heldout_indices=held,
            speaker_reference_indices=train,
        )
        split = _split_from_outer_partition(
            corpus,
            outer_train=train,
            heldout=held,
            validation_fraction=baseline_run_config.inner_validation_fraction,
            seed=int(record.seed),
            fold=int(record.fold),
        )
        processor_record = by_key.get((record.seed, record.fold, "text_processor"))
        if processor_record is None:
            raise CurrentOnlyPipelineError("checkpoint manifest lacks a fold processor")
        processor_path = root / Path(processor_record.relative_name)
        if _file_sha256(processor_path) != processor_record.sha256:
            raise CurrentOnlyPipelineError("fold text processor changed before loading")
        try:
            processor_payload = joblib.load(processor_path)
        except Exception as error:
            raise CurrentOnlyPipelineError(
                f"cannot deserialize fold text processor: {record.relative_name}"
            ) from error
        expected_processor_identity = _canonical_sha256(
            {
                "source_identity": source_identity,
                "fold": int(record.fold),
                "seed": int(record.seed),
                "inner_train_indices_sha256": _indices_sha256(
                    split.inner_train_indices
                ),
                "text_dim": model_config.text_dim,
                "text_settings": {
                    name: value
                    for name, value in asdict(baseline_run_config).items()
                    if name.startswith("text_")
                },
            }
        )
        processor = (
            processor_payload.get("processor")
            if isinstance(processor_payload, Mapping)
            else None
        )
        if (
            not isinstance(processor_payload, Mapping)
            or set(processor_payload)
            != {"schema_version", "identity_sha256", "processor"}
            or processor_payload.get("schema_version") != "fold_local_text_svd_v1"
            or processor_payload.get("identity_sha256")
            != expected_processor_identity
            or not isinstance(processor, FoldTextProcessor)
            or processor.fit_indices_sha256
            != _indices_sha256(split.inner_train_indices)
            or int(processor.output_dim) != int(model_config.text_dim)
            or not 1 <= int(processor.effective_dim) <= int(processor.output_dim)
        ):
            raise CurrentOnlyPipelineError("fold text processor identity is invalid")
        vectorizer_params = processor.vectorizer.get_params(deep=False)
        expected_vectorizer = {
            "analyzer": baseline_run_config.text_analyzer,
            "ngram_range": (
                baseline_run_config.text_ngram_min,
                baseline_run_config.text_ngram_max,
            ),
            "min_df": baseline_run_config.text_min_df,
            "max_df": baseline_run_config.text_max_df,
            "max_features": baseline_run_config.text_max_features,
            "sublinear_tf": baseline_run_config.text_sublinear_tf,
            "dtype": np.float32,
        }
        if any(
            vectorizer_params.get(name) != value
            for name, value in expected_vectorizer.items()
        ):
            raise CurrentOnlyPipelineError(
                "fold text processor vectorizer configuration changed"
            )
        vocabulary_size = len(getattr(processor.vectorizer, "vocabulary_", {}))
        expected_effective = min(
            int(model_config.text_dim), max(1, int(vocabulary_size) - 1)
        )
        if (
            vocabulary_size < 1
            or int(processor.effective_dim) != expected_effective
            or (
                vocabulary_size == 1
                and processor.svd is not None
            )
            or (
                vocabulary_size > 1
                and (
                    processor.svd is None
                    or int(processor.svd.n_components) != expected_effective
                    or int(processor.svd.n_iter)
                    != int(baseline_run_config.text_svd_n_iter)
                    or int(processor.svd.random_state) != int(record.seed)
                )
            )
        ):
            raise CurrentOnlyPipelineError(
                "fold text processor projection configuration changed"
            )
        try:
            probe = processor.transform(corpus.texts[:1])
        except Exception as error:
            raise CurrentOnlyPipelineError(
                "fold text processor cannot transform its bound corpus"
            ) from error
        if probe.shape != (1, model_config.text_dim) or not np.isfinite(probe).all():
            raise CurrentOnlyPipelineError("fold text processor output is invalid")
        processor_sha = _file_sha256(processor_path)
        if processor_sha != processor_record.sha256:
            raise CurrentOnlyPipelineError("fold text processor changed while loading")
        expected_checkpoint_identity = _canonical_sha256(
            {
                "source_identity": source_identity,
                "seed": int(record.seed),
                "fold": int(record.fold),
                "inner_train": _indices_sha256(split.inner_train_indices),
                "inner_validation": _indices_sha256(
                    split.inner_validation_indices
                ),
                "outer_heldout": _indices_sha256(split.outer_heldout_indices),
                "processor_sha256": processor_sha,
                "model_config": asdict(model_config),
                "run_config": asdict(baseline_run_config),
            }
        )
        checkpoint_path = root / Path(record.relative_name)
        if _file_sha256(checkpoint_path) != record.sha256:
            raise CurrentOnlyPipelineError(
                f"checkpoint changed before semantic validation: {record.relative_name}"
            )
        try:
            payload = _torch_load_local(checkpoint_path)
        except Exception as error:
            raise CurrentOnlyPipelineError(
                f"cannot validate complete checkpoint {record.relative_name}: {error}"
            ) from error
        if _file_sha256(checkpoint_path) != record.sha256:
            raise CurrentOnlyPipelineError(
                f"checkpoint changed during semantic validation: {record.relative_name}"
            )
        if (
            payload.get("schema_version") != "causal_backbone_atomic_checkpoint_v2"
            or payload.get("status") != "complete"
        ):
            raise CurrentOnlyPipelineError(
                f"checkpoint is not semantically complete: {record.relative_name}"
            )
        if payload.get("identity_sha256") != expected_checkpoint_identity:
            raise CurrentOnlyPipelineError(
                "checkpoint identity differs from source/fold/config/split"
            )
        epoch = payload.get("epoch")
        best_epoch = payload.get("best_epoch")
        best_nll = payload.get("best_validation_nll")
        if (
            not isinstance(epoch, int)
            or isinstance(epoch, bool)
            or epoch < 0
            or not isinstance(best_epoch, int)
            or isinstance(best_epoch, bool)
            or not 0 <= best_epoch <= epoch
            or not isinstance(best_nll, (int, float))
            or isinstance(best_nll, bool)
            or not np.isfinite(float(best_nll))
        ):
            raise CurrentOnlyPipelineError(
                f"checkpoint epoch/validation summary is malformed: {record.relative_name}"
            )
        model_state = payload.get("model_state")
        best_state = payload.get("best_model_state")
        rng_state = payload.get("rng_state")
        optimizer_state = payload.get("optimizer_state")
        scaler_state = payload.get("scaler_state")
        bad_epochs = payload.get("bad_epochs")
        peak_mib = payload.get("peak_cuda_mib")
        if (
            not isinstance(model_state, Mapping)
            or not model_state
            or not isinstance(best_state, Mapping)
            or not best_state
            or set(model_state) != set(best_state)
            or not isinstance(optimizer_state, Mapping)
            or not isinstance(scaler_state, Mapping)
            or not isinstance(rng_state, Mapping)
            or set(rng_state) != {"python", "numpy", "torch_cpu", "torch_cuda"}
            or not isinstance(bad_epochs, int)
            or isinstance(bad_epochs, bool)
            or bad_epochs < 0
            or not isinstance(payload.get("early_stopped"), bool)
            or not isinstance(peak_mib, (int, float))
            or isinstance(peak_mib, bool)
            or not np.isfinite(float(peak_mib))
            or float(peak_mib) < 0.0
        ):
            raise CurrentOnlyPipelineError(
                f"checkpoint state payload is incomplete: {record.relative_name}"
            )
        try:
            model = CausalMultimodalBackbone(model_config)
            model.load_state_dict(model_state, strict=True)
            model.load_state_dict(best_state, strict=True)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=baseline_run_config.learning_rate,
                weight_decay=baseline_run_config.weight_decay,
            )
            optimizer.load_state_dict(dict(optimizer_state))
            scaler = torch.cuda.amp.GradScaler(
                enabled=bool(baseline_run_config.use_amp and device.type == "cuda")
            )
            scaler.load_state_dict(dict(scaler_state))
            saved_rng = _capture_rng_state()
            try:
                _restore_rng_state(rng_state)
            finally:
                _restore_rng_state(saved_rng)
        except Exception as error:
            raise CurrentOnlyPipelineError(
                "complete checkpoint cannot be strictly restored"
            ) from error
        if _file_sha256(checkpoint_path) != record.sha256:
            raise CurrentOnlyPipelineError("checkpoint changed while strictly restoring")
        verify_checkpoint_manifest(root, manifest)


def _attested_history_source_hashes(
    attestation: VerifiedHistoryCompletionAttestation,
    *,
    fit_preflight_receipt_path: str | Path,
    expected_fit_preflight_receipt_sha256: str,
) -> dict[str, str]:
    preflight_sha = _require_sha256(
        expected_fit_preflight_receipt_sha256,
        "expected_fit_preflight_receipt_sha256",
    )
    if preflight_sha != attestation.fit_preflight_receipt_sha256:
        raise CurrentOnlyPipelineError(
            "history attestation uses another fit preflight receipt"
        )
    receipt_path = Path(fit_preflight_receipt_path)
    if _file_sha256(receipt_path) != preflight_sha:
        raise CurrentOnlyPipelineError("fit preflight receipt file hash changed")
    receipt = _load_receipt(receipt_path)
    manifest = receipt.get("manifest")
    sidecars = receipt.get("sidecars")
    if (
        receipt.get("dataset") != attestation.dataset
        or not isinstance(manifest, Mapping)
        or not isinstance(sidecars, Mapping)
    ):
        raise CurrentOnlyPipelineError(
            "history attestation and fit preflight receipt differ"
        )
    fit_sidecar = sidecars.get("fit")
    selection_sidecar = sidecars.get(SELECTION_ROLE)
    if not isinstance(fit_sidecar, Mapping) or not isinstance(
        selection_sidecar, Mapping
    ):
        raise CurrentOnlyPipelineError("fit preflight receipt lacks role lineage")
    source_hashes = {
        "source_sidecar_manifest_sha256": manifest.get("sha256"),
        f"source_{FIT_ROLE}_features_sha256": fit_sidecar.get("feature_sha256"),
        f"source_{FIT_ROLE}_labels_sha256": fit_sidecar.get("label_sha256"),
        "source_model_selection_features_sha256": selection_sidecar.get(
            "feature_sha256"
        ),
        "source_model_selection_labels_sha256": selection_sidecar.get(
            "label_sha256"
        ),
    }
    result = {
        name: _require_sha256(value, name)
        for name, value in source_hashes.items()
    }
    if _file_sha256(receipt_path) != preflight_sha:
        raise CurrentOnlyPipelineError(
            "fit preflight receipt changed while binding history attestation"
        )
    return result


def load_attested_history_fit_alignment_view(
    attestation: VerifiedHistoryCompletionAttestation,
    *,
    fit_preflight_receipt_path: str | Path,
    expected_fit_preflight_receipt_sha256: str,
) -> AttestedHistoryFitAlignmentView:
    """Read no selection arrays before the current-only fit gate."""

    if not isinstance(attestation, VerifiedHistoryCompletionAttestation):
        raise CurrentOnlyPipelineError(
            "history fit alignment requires a verified production attestation"
        )
    expected_sha = _require_sha256(
        attestation.artifact_sha256, "history_completion_artifact_sha256"
    )
    artifact = attestation.artifact_path.resolve()
    if _file_sha256(artifact) != expected_sha:
        raise CurrentOnlyPipelineError(
            "attested history completion artifact changed before fit alignment"
        )
    source_hashes = _attested_history_source_hashes(
        attestation,
        fit_preflight_receipt_path=fit_preflight_receipt_path,
        expected_fit_preflight_receipt_sha256=(
            expected_fit_preflight_receipt_sha256
        ),
    )
    try:
        with np.load(artifact, allow_pickle=False) as archive:
            keys = set(archive.files)

            def read(name: str) -> np.ndarray:
                if name.startswith("selection_") or name.startswith(
                    "matrix_selection_"
                ):
                    raise AssertionError(
                        "fit alignment attempted to read a selection array"
                    )
                if name not in keys:
                    raise CurrentOnlyPipelineError(
                        f"attested history fit field is missing: {name}"
                    )
                return np.asarray(archive[name])

            schema = _single_text(read("schema_version"), "schema_version")
            dataset = _single_text(read("dataset"), "dataset")
            label_order = tuple(
                str(value)
                for value in np.asarray(read("dataset_label_order")).reshape(-1)
            )
            seeds = tuple(
                int(value)
                for value in _integer_vector(read("seeds"), "seeds", unique=True)
            )
            fit_protocol = _integer_vector(
                read("fit_protocol_row_ids"),
                "fit_protocol_row_ids",
                unique=True,
            )
            fit_cluster = _integer_vector(
                read("fit_cluster_codes"), "fit_cluster_codes"
            )
            source_identity = _require_sha256(
                _single_text(
                    read("source_identity_sha256"), "source_identity_sha256"
                ),
                "source_identity_sha256",
            )
            checkpoint_manifest = _require_sha256(
                _single_text(
                    read("checkpoint_manifest_sha256"),
                    "checkpoint_manifest_sha256",
                ),
                "checkpoint_manifest_sha256",
            )
    except (OSError, ValueError, KeyError) as error:
        raise CurrentOnlyPipelineError(
            f"cannot read attested history fit alignment: {error}"
        ) from error
    if (
        schema != HISTORY_COMPLETE_OUTCOME_SCHEMA
        or dataset != attestation.dataset
        or seeds != EXPECTED_SEEDS
        or len(label_order) < 2
        or not len(fit_protocol)
        or fit_protocol.shape != fit_cluster.shape
        or source_identity != attestation.source_identity_sha256
        or checkpoint_manifest != attestation.checkpoint_manifest_sha256
    ):
        raise CurrentOnlyPipelineError(
            "attested history fit alignment identity changed"
        )
    if _file_sha256(artifact) != expected_sha:
        raise CurrentOnlyPipelineError(
            "attested history artifact changed while reading fit alignment"
        )
    return AttestedHistoryFitAlignmentView(
        dataset=dataset,
        label_order=label_order,
        seeds=seeds,
        protocol_row_ids=fit_protocol,
        fit_query_indices=np.arange(len(fit_protocol), dtype=np.int64),
        fit_cluster_codes=fit_cluster,
        producer_file_sha256=expected_sha,
        source_identity_sha256=source_identity,
        checkpoint_manifest_sha256=checkpoint_manifest,
        source_hashes=source_hashes,
    )


def load_attested_history_producer_alignment_view(
    attestation: VerifiedHistoryCompletionAttestation,
    *,
    fit_producer: AttestedHistoryFitAlignmentView,
) -> CurrentOnlyProducerAlignmentView:
    """Open selection alignment only from the already-attested history cache."""

    if not isinstance(attestation, VerifiedHistoryCompletionAttestation) or not isinstance(
        fit_producer, AttestedHistoryFitAlignmentView
    ):
        raise CurrentOnlyPipelineError(
            "history producer alignment requires attested fit capability"
        )
    expected_sha = _require_sha256(
        attestation.artifact_sha256, "history_completion_artifact_sha256"
    )
    artifact = attestation.artifact_path.resolve()
    if (
        fit_producer.producer_file_sha256 != expected_sha
        or _file_sha256(artifact) != expected_sha
    ):
        raise CurrentOnlyPipelineError(
            "attested history artifact changed before selection alignment"
        )
    try:
        with np.load(artifact, allow_pickle=False) as archive:
            keys = set(archive.files)

            def read(name: str) -> np.ndarray:
                if name not in keys:
                    raise CurrentOnlyPipelineError(
                        f"attested history alignment field is missing: {name}"
                    )
                return np.asarray(archive[name])

            schema = _single_text(read("schema_version"), "schema_version")
            dataset = _single_text(read("dataset"), "dataset")
            label_order = tuple(
                str(value)
                for value in np.asarray(read("dataset_label_order")).reshape(-1)
            )
            seeds = tuple(
                int(value)
                for value in _integer_vector(read("seeds"), "seeds", unique=True)
            )
            fit_protocol = _integer_vector(
                read("fit_protocol_row_ids"),
                "fit_protocol_row_ids",
                unique=True,
            )
            selection_protocol = _integer_vector(
                read("selection_protocol_row_ids"),
                "selection_protocol_row_ids",
                unique=True,
            )
            fit_cluster = _integer_vector(
                read("fit_cluster_codes"), "fit_cluster_codes"
            )
            selection_cluster = _integer_vector(
                read("selection_cluster_codes"), "selection_cluster_codes"
            )
            source_identity = _require_sha256(
                _single_text(
                    read("source_identity_sha256"), "source_identity_sha256"
                ),
                "source_identity_sha256",
            )
            checkpoint_manifest = _require_sha256(
                _single_text(
                    read("checkpoint_manifest_sha256"),
                    "checkpoint_manifest_sha256",
                ),
                "checkpoint_manifest_sha256",
            )
    except (OSError, ValueError, KeyError) as error:
        raise CurrentOnlyPipelineError(
            f"cannot read attested history producer alignment: {error}"
        ) from error
    fit_queries = np.arange(len(fit_protocol), dtype=np.int64)
    selection_queries = np.arange(
        len(fit_protocol),
        len(fit_protocol) + len(selection_protocol),
        dtype=np.int64,
    )
    if (
        schema != HISTORY_COMPLETE_OUTCOME_SCHEMA
        or dataset != attestation.dataset
        or dataset != fit_producer.dataset
        or label_order != fit_producer.label_order
        or seeds != EXPECTED_SEEDS
        or seeds != fit_producer.seeds
        or not np.array_equal(fit_protocol, fit_producer.protocol_row_ids)
        or not np.array_equal(fit_cluster, fit_producer.fit_cluster_codes)
        or fit_protocol.shape != fit_cluster.shape
        or selection_protocol.shape != selection_cluster.shape
        or not len(selection_protocol)
        or set(fit_protocol.tolist()) & set(selection_protocol.tolist())
        or len(np.unique(selection_cluster)) < 2
        or source_identity != attestation.source_identity_sha256
        or source_identity != fit_producer.source_identity_sha256
        or checkpoint_manifest != attestation.checkpoint_manifest_sha256
        or checkpoint_manifest != fit_producer.checkpoint_manifest_sha256
    ):
        raise CurrentOnlyPipelineError(
            "attested history producer alignment identity changed"
        )
    if _file_sha256(artifact) != expected_sha:
        raise CurrentOnlyPipelineError(
            "attested history artifact changed while reading selection alignment"
        )
    return CurrentOnlyProducerAlignmentView(
        dataset=dataset,
        label_order=label_order,
        seeds=seeds,
        protocol_row_ids=np.concatenate((fit_protocol, selection_protocol)),
        fit_query_indices=fit_queries,
        selection_query_indices=selection_queries,
        fit_cluster_codes=fit_cluster,
        selection_cluster_codes=selection_cluster,
        producer_file_sha256=expected_sha,
        source_identity_sha256=source_identity,
        checkpoint_manifest_sha256=checkpoint_manifest,
        source_hashes=dict(fit_producer.source_hashes),
    )


def load_current_only_producer_alignment_view(
    path: str | Path,
    *,
    fit_producer: FitOnlyProducerView,
) -> CurrentOnlyProducerAlignmentView:
    """Read only role alignment and lineage; outcome-derived selection arrays stay opaque."""

    producer_path = Path(path)
    producer_sha = _file_sha256(producer_path)
    if producer_sha != fit_producer.producer_file_sha256:
        raise CurrentOnlyPipelineError("alignment view points to a different producer file")
    with np.load(producer_path, allow_pickle=False) as archive:
        keys = set(archive.files)

        def read(name: str) -> np.ndarray:
            if name not in keys:
                raise CurrentOnlyPipelineError(f"producer alignment field is missing: {name}")
            return np.asarray(archive[name])

        if _single_text(read("schema_version"), "schema_version") != PRODUCER_CACHE_SCHEMA:
            raise CurrentOnlyPipelineError("producer alignment schema changed")
        dataset = _single_text(read("dataset"), "dataset")
        labels = tuple(str(value) for value in np.asarray(read("dataset_label_order")).reshape(-1))
        seeds = tuple(
            int(value)
            for value in _integer_vector(read("seeds"), "seeds", unique=True)
        )
        protocol = _integer_vector(
            read("protocol_row_ids"), "protocol_row_ids", unique=True
        )
        fit_query = _integer_vector(
            read("fit_query_indices"), "fit_query_indices", unique=True
        )
        selection_query = _integer_vector(
            read("selection_query_indices"), "selection_query_indices", unique=True
        )
        fit_cluster = _integer_vector(read("fit_cluster_codes"), "fit_cluster_codes")
        selection_cluster = _integer_vector(
            read("selection_cluster_codes"), "selection_cluster_codes"
        )
        source_identity = _require_sha256(
            _single_text(read("source_identity_sha256"), "source_identity_sha256"),
            "source_identity_sha256",
        )
        checkpoint_manifest = _require_sha256(
            _single_text(read("checkpoint_manifest_sha256"), "checkpoint_manifest_sha256"),
            "checkpoint_manifest_sha256",
        )
        source_hashes = {
            name: _require_sha256(_single_text(read(name), name), name)
            for name in sorted(keys)
            if name.startswith("source_")
            and name.endswith("_sha256")
            and name != "source_identity_sha256"
        }

    shared = (
        (dataset, fit_producer.dataset),
        (labels, fit_producer.label_order),
        (seeds, fit_producer.seeds),
        (source_identity, fit_producer.source_identity_sha256),
        (checkpoint_manifest, fit_producer.checkpoint_manifest_sha256),
        (source_hashes, fit_producer.source_hashes),
    )
    if any(left != right for left, right in shared):
        raise CurrentOnlyPipelineError("fit and completion producer views differ")
    array_shared = (
        (protocol, fit_producer.protocol_row_ids),
        (fit_query, fit_producer.fit_query_indices),
        (fit_cluster, fit_producer.fit_cluster_codes),
    )
    if any(not np.array_equal(left, right) for left, right in array_shared):
        raise CurrentOnlyPipelineError("fit and completion producer alignment differs")
    if (
        fit_query.shape != fit_cluster.shape
        or selection_query.shape != selection_cluster.shape
        or not len(selection_query)
        or np.any(fit_query >= len(protocol))
        or np.any(selection_query >= len(protocol))
        or set(fit_query.tolist()) & set(selection_query.tolist())
        or set(fit_query.tolist()) | set(selection_query.tolist()) != set(range(len(protocol)))
    ):
        raise CurrentOnlyPipelineError("producer open-role row partition is incomplete")
    if len(np.unique(selection_cluster)) < 2:
        raise CurrentOnlyPipelineError("selection alignment requires at least two clusters")
    return CurrentOnlyProducerAlignmentView(
        dataset=dataset,
        label_order=labels,
        seeds=seeds,
        protocol_row_ids=protocol,
        fit_query_indices=fit_query,
        selection_query_indices=selection_query,
        fit_cluster_codes=fit_cluster,
        selection_cluster_codes=selection_cluster,
        producer_file_sha256=producer_sha,
        source_identity_sha256=source_identity,
        checkpoint_manifest_sha256=checkpoint_manifest,
        source_hashes=source_hashes,
    )


def align_selection_protocol_to_producer(
    selection: SelectionFeatureView,
    producer: CurrentOnlyProducerAlignmentView,
) -> tuple[np.ndarray, np.ndarray]:
    """Return selection-local -> combined row and -> selection-position mappings."""

    if selection.dataset != producer.dataset or selection.labels_materialized:
        raise CurrentOnlyPipelineError("selection feature view crossed its capability")
    by_protocol = {
        int(protocol): int(index)
        for index, protocol in enumerate(producer.protocol_row_ids)
    }
    try:
        combined = np.asarray(
            [by_protocol[int(value)] for value in selection.protocol_row_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise CurrentOnlyPipelineError(
            "selection protocol row is absent from producer"
        ) from error
    if set(combined.tolist()) != set(producer.selection_query_indices.tolist()):
        raise CurrentOnlyPipelineError(
            "selection feature rows do not equal producer selection rows"
        )
    position = {
        int(query): int(index)
        for index, query in enumerate(producer.selection_query_indices)
    }
    return combined, np.asarray([position[int(value)] for value in combined], dtype=np.int64)


def _speaker_token(dataset: str, value: object) -> str:
    if dataset == "MELD":
        try:
            return str(int(str(value)))
        except ValueError as error:
            raise CurrentOnlyPipelineError("MELD speaker token is not integral") from error
    return str(value)


def _fit_speaker_mapping(
    dataset: str,
    speakers: Sequence[object],
    *,
    num_speakers: int,
) -> tuple[dict[str, int], str]:
    tokens = {_speaker_token(dataset, value) for value in speakers}
    ordered = sorted(tokens, key=(lambda value: int(value)) if dataset == "MELD" else None)
    mapping = {value: index + 1 for index, value in enumerate(ordered)}
    if len(mapping) + 1 > int(num_speakers):
        raise CurrentOnlyPipelineError("fit-only speaker vocabulary exceeds model config")
    public_values: list[object] = (
        [int(value) for value in ordered] if dataset == "MELD" else list(ordered)
    )
    mapping_sha = _canonical_sha256(
        {
            "oov": 0,
            "fit_mapping": [
                [value, mapping[_speaker_token(dataset, value)]] for value in public_values
            ],
        }
    )
    return mapping, mapping_sha


def _speaker_identity(dataset: str, value: object) -> str:
    token = _speaker_token(dataset, value)
    prefix = "MELD-speaker" if dataset == "MELD" else "speaker"
    return hashlib.sha256(f"{prefix}\x1f{token}".encode("utf-8")).hexdigest()


def _make_corpus(
    *,
    dataset: str,
    texts: Sequence[str],
    audio: np.ndarray,
    video: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    speakers: np.ndarray,
    turns: np.ndarray,
    role: str,
    model_config: CausalBackboneConfig,
    fit_speaker_mapping: Mapping[str, int],
    speaker_mapping_sha256: str,
    label_access_mode: str,
) -> OpenRoleCorpus:
    rows = len(texts)
    speaker_tokens = np.asarray(speakers).astype(str)
    speaker_ids = np.asarray(
        [
            int(fit_speaker_mapping.get(_speaker_token(dataset, value), 0))
            for value in speaker_tokens
        ],
        dtype=np.int64,
    )
    keys = np.asarray(
        [
            hashlib.sha256(
                f"{dataset}\x1findependent-current-only\x1f{index}".encode("utf-8")
            ).hexdigest()
            for index in range(rows)
        ]
    )
    corpus = OpenRoleCorpus(
        keys=keys,
        texts=tuple(str(value) for value in texts),
        audio=np.asarray(audio, dtype=np.float32).copy(),
        video=np.asarray(video, dtype=np.float32).copy(),
        labels=np.asarray(labels, dtype=np.int64).copy(),
        groups=np.asarray(groups).astype(str),
        roles=np.asarray([role] * rows),
        buckets=np.asarray([0 if role == FIT_ROLE else 65] * rows, dtype=np.int16),
        speaker_ids=speaker_ids,
        turn_ids=np.asarray(turns, dtype=np.int64).copy(),
        histories=tuple(() for _ in range(rows)),
        protocol_row_ids=np.arange(rows, dtype=np.int64),
        speaker_identity=np.asarray(
            [_speaker_identity(dataset, value) for value in speaker_tokens]
        ),
        speaker_mapping_sha256=speaker_mapping_sha256,
        label_access_mode=label_access_mode,
    )
    corpus.validate(model_config)
    if any(corpus.histories):
        raise AssertionError("current-only corpus unexpectedly contains history")
    return corpus


def _corpus_from_fold_request(
    request: HistoryFreeFoldRequest,
    *,
    model_config: CausalBackboneConfig,
) -> OpenRoleCorpus:
    all_indices = np.concatenate([request.train_indices, request.heldout_indices])
    rows = len(all_indices)
    if set(all_indices.tolist()) != set(range(rows)) or len(set(all_indices.tolist())) != rows:
        raise CurrentOnlyPipelineError("fold request is not a complete row partition")
    texts = [""] * rows
    audio = np.empty((rows, request.train_audio.shape[1]), dtype=np.float32)
    video = np.empty((rows, request.train_video.shape[1]), dtype=np.float32)
    labels = np.zeros(rows, dtype=np.int64)
    groups = np.empty(rows, dtype=object)
    speakers = np.empty(rows, dtype=object)
    turns = np.empty(rows, dtype=np.int64)
    for indices, source_texts, source_audio, source_video, source_groups, source_speakers, source_turns in (
        (
            request.train_indices,
            request.train_texts,
            request.train_audio,
            request.train_video,
            request.train_group_tokens,
            request.train_speaker_tokens,
            request.train_turns,
        ),
        (
            request.heldout_indices,
            request.heldout_texts,
            request.heldout_audio,
            request.heldout_video,
            request.heldout_group_tokens,
            request.heldout_speaker_tokens,
            request.heldout_turns,
        ),
    ):
        for local, target in enumerate(indices):
            target_int = int(target)
            texts[target_int] = str(source_texts[local])
            audio[target_int] = source_audio[local]
            video[target_int] = source_video[local]
            groups[target_int] = source_groups[local]
            speakers[target_int] = source_speakers[local]
            turns[target_int] = int(source_turns[local])
    labels[request.train_indices] = np.asarray(request.train_labels, dtype=np.int64)
    # The speaker vocabulary is a learned preprocessing object.  Fit it on the
    # outer-training rows only; speakers seen only in the OOF fold must remain
    # OOV, exactly as genuinely unseen model-selection speakers do.
    mapping, mapping_sha = _fit_speaker_mapping(
        request.dataset,
        request.train_speaker_tokens,
        num_speakers=model_config.num_speakers,
    )
    return _make_corpus(
        dataset=request.dataset,
        texts=texts,
        audio=audio,
        video=video,
        labels=labels,
        groups=np.asarray(groups),
        speakers=np.asarray(speakers),
        turns=turns,
        role=FIT_ROLE,
        model_config=model_config,
        fit_speaker_mapping=mapping,
        speaker_mapping_sha256=mapping_sha,
        label_access_mode=(
            "fit_train_labels_only_outer_heldout_labels_physically_absent"
        ),
    )


def _fit_corpus_from_view(
    fit: FitRoleView,
    *,
    model_config: CausalBackboneConfig,
    heldout_indices: np.ndarray | None = None,
    speaker_reference_indices: np.ndarray | None = None,
) -> OpenRoleCorpus:
    labels = np.asarray(fit.labels, dtype=np.int64).copy()
    if heldout_indices is not None:
        labels[np.asarray(heldout_indices, dtype=np.int64)] = 0
    reference = (
        np.arange(fit.rows, dtype=np.int64)
        if speaker_reference_indices is None
        else np.asarray(speaker_reference_indices, dtype=np.int64)
    )
    if (
        reference.ndim != 1
        or not len(reference)
        or np.any((reference < 0) | (reference >= fit.rows))
        or len(np.unique(reference)) != len(reference)
    ):
        raise CurrentOnlyPipelineError("fit speaker reference rows are invalid")
    mapping, mapping_sha = _fit_speaker_mapping(
        fit.dataset,
        np.asarray(fit.speakers)[reference],
        num_speakers=model_config.num_speakers,
    )
    return _make_corpus(
        dataset=fit.dataset,
        texts=fit.texts,
        audio=fit.audio,
        video=fit.video,
        labels=labels,
        groups=fit.groups,
        speakers=fit.speakers,
        turns=fit.turns,
        role=FIT_ROLE,
        model_config=model_config,
        fit_speaker_mapping=mapping,
        speaker_mapping_sha256=mapping_sha,
        label_access_mode="fit_labels_only_completion_checkpoint_identity_revalidation",
    )


def _selection_corpus_from_view(
    selection: SelectionFeatureView,
    *,
    fit: FitRoleView,
    model_config: CausalBackboneConfig,
    fit_speaker_reference_indices: np.ndarray,
) -> OpenRoleCorpus:
    if selection.labels_materialized:
        raise CurrentOnlyPipelineError("selection labels entered complete-selection")
    reference = np.asarray(fit_speaker_reference_indices, dtype=np.int64)
    if (
        reference.ndim != 1
        or not len(reference)
        or np.any((reference < 0) | (reference >= fit.rows))
        or len(np.unique(reference)) != len(reference)
    ):
        raise CurrentOnlyPipelineError("selection speaker reference rows are invalid")
    mapping, mapping_sha = _fit_speaker_mapping(
        fit.dataset,
        np.asarray(fit.speakers)[reference],
        num_speakers=model_config.num_speakers,
    )
    return _make_corpus(
        dataset=selection.dataset,
        texts=selection.texts,
        audio=selection.audio,
        video=selection.video,
        labels=np.zeros(len(selection.texts), dtype=np.int64),
        groups=selection.groups,
        speakers=selection.speakers,
        turns=selection.turns,
        role=SELECTION_ROLE,
        model_config=model_config,
        fit_speaker_mapping=mapping,
        speaker_mapping_sha256=mapping_sha,
        label_access_mode="selection_features_only_zero_placeholder_labels_never_scored",
    )


def _split_from_outer_partition(
    corpus: OpenRoleCorpus,
    *,
    outer_train: np.ndarray,
    heldout: np.ndarray,
    validation_fraction: float,
    seed: int,
    fold: int,
) -> CrossfitSplit:
    train = np.asarray(outer_train, dtype=np.int64)
    held = np.asarray(heldout, dtype=np.int64)
    outer_groups = sorted(set(corpus.groups[train].astype(str)))
    if len(outer_groups) < 2:
        raise CurrentOnlyPipelineError(
            "outer current-only training fold has too few groups for early stopping"
        )
    ordered = sorted(
        outer_groups,
        key=lambda group: hashlib.sha256(
            f"inner\x1f{int(seed)}\x1f{int(fold)}\x1f{group}".encode("utf-8")
        ).digest(),
    )
    validation_count = min(
        len(ordered) - 1,
        max(1, int(round(float(validation_fraction) * len(ordered)))),
    )
    validation_groups = set(ordered[:validation_count])
    validation_mask = np.asarray(
        [str(value) in validation_groups for value in corpus.groups[train]], dtype=bool
    )
    inner_validation = train[validation_mask]
    inner_train = train[~validation_mask]
    partitions = (inner_train, inner_validation, held)
    if any(not len(value) for value in partitions):
        raise CurrentOnlyPipelineError("current-only split contains an empty partition")
    group_sets = [set(corpus.groups[value].astype(str)) for value in partitions]
    if any(
        group_sets[left] & group_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise CurrentOnlyPipelineError("current-only split shares a group")
    return CrossfitSplit(int(fold), inner_train, inner_validation, held)


def build_label_blind_current_only_fold_assignment(
    fit: FitRoleView,
    *,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
) -> np.ndarray:
    """Use the registered label-free group splitter for all five seeds."""

    corpus = _fit_corpus_from_view(fit, model_config=model_config)
    result = np.empty((len(EXPECTED_SEEDS), fit.rows), dtype=np.int32)
    for seed_index, seed in enumerate(EXPECTED_SEEDS):
        splits = make_crossfit_splits(
            corpus,
            outer_folds=int(run_config.outer_folds),
            validation_fraction=float(run_config.inner_validation_fraction),
            seed=int(seed),
        )
        for split in splits:
            result[seed_index, split.outer_heldout_indices] = int(split.fold)
    return result


def make_real_current_only_fold_callback(
    *,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    data_contract_sha256: str,
    device: torch.device,
) -> Callable[[HistoryFreeFoldRequest], CurrentOnlyFoldOutput]:
    """Return the real history-stripped trainer without capturing fit labels."""

    data_contract = _require_sha256(data_contract_sha256, "data_contract_sha256")
    baseline_run_config = replace(run_config, subset_dropout_probability=0.0)

    def callback(request: HistoryFreeFoldRequest) -> CurrentOnlyFoldOutput:
        if request.heldout_labels_materialized:
            raise CurrentOnlyPipelineError("heldout labels entered current-only fitting")
        corpus = _corpus_from_fold_request(request, model_config=model_config)
        split = _split_from_outer_partition(
            corpus,
            outer_train=request.train_indices,
            heldout=request.heldout_indices,
            validation_fraction=baseline_run_config.inner_validation_fraction,
            seed=request.seed,
            fold=request.fold,
        )
        trained = train_independent_current_only_fold_seed(
            corpus,
            split,
            producer_source_identity_sha256=(
                request.fit_lineage_source_identity_sha256
            ),
            model_config=model_config,
            run_config=baseline_run_config,
            seed=request.seed,
            checkpoint_root=request.checkpoint_root,
            device=device,
            data_contract_sha256=data_contract,
            require_complete_checkpoint=False,
        )
        probability = predict_independent_current_only_probability(
            trained,
            corpus,
            request.heldout_indices,
            device=device,
            batch_size=baseline_run_config.inference_batch_size,
            max_history_items=baseline_run_config.max_history_items,
        )
        return CurrentOnlyFoldOutput(
            probability=np.asarray(probability, dtype=np.float32),
            current_only_source_identity_sha256=trained.source_identity_sha256,
        )

    return _attest_production_current_only_fold_callback(
        callback,
        model_config_semantic_sha256=_model_config_semantic_sha256(model_config),
        run_config_semantic_sha256=_run_config_semantic_sha256(run_config),
    )


def produce_current_only_fit_with_real_trainer(
    *,
    fit: FitRoleView,
    fit_map: FitProtocolMap,
    lineage: FitOnlyLineage,
    fit_preflight_receipt_path: str | Path,
    expected_fit_preflight_receipt_sha256: str,
    checkpoint_root: str | Path,
    artifact_path: str | Path,
    producer_receipt_path: str | Path,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    model_config_sha256: str,
    run_config_sha256: str,
    source_code_sha256: str,
    runtime_environment_sha256: str,
    production_run_claim_sha256: str,
    allow_checkpoint_resume: bool,
    device: torch.device,
) -> CurrentOnlyFitProduction:
    """Run the real fit stage; no model-selection payload is an API parameter."""

    _validate_current_only_model_label_contract(model_config, fit.label_order)
    private_root = _validate_current_only_private_layout(
        checkpoint_root=checkpoint_root,
        fit_artifact_path=artifact_path,
        fit_receipt_path=producer_receipt_path,
    )
    expected_claim = current_only_production_claim_sha256(
        fit=fit,
        fit_map=fit_map,
        lineage=lineage,
        fit_preflight_receipt_sha256=expected_fit_preflight_receipt_sha256,
        model_config=model_config,
        run_config=run_config,
        model_config_sha256=model_config_sha256,
        run_config_sha256=run_config_sha256,
        source_code_sha256=source_code_sha256,
        runtime_environment_sha256=runtime_environment_sha256,
    )
    claim_sha = _require_sha256(
        production_run_claim_sha256, "production_run_claim_sha256"
    )
    if claim_sha != expected_claim:
        raise CurrentOnlyPipelineError("current-only production claim lineage changed")
    _verify_current_only_private_claim(private_root, claim_sha)
    with _ExclusiveCurrentOnlyFitLock(private_root):
        _verify_current_only_private_claim(private_root, claim_sha)
        if Path(artifact_path).exists() or Path(producer_receipt_path).exists():
            raise CurrentOnlyPipelineError(
                "current-only fit root already contains a finalized output"
            )
        folds = build_label_blind_current_only_fold_assignment(
            fit, model_config=model_config, run_config=run_config
        )
        callback = make_real_current_only_fold_callback(
            model_config=model_config,
            run_config=run_config,
            data_contract_sha256=fit_map.fit_arrays_contract_sha256,
            device=device,
        )
        produced = produce_independent_current_only_fit_oof(
            fit=fit,
            fit_map=fit_map,
            lineage=lineage,
            fit_preflight_receipt_path=fit_preflight_receipt_path,
            expected_fit_preflight_receipt_sha256=expected_fit_preflight_receipt_sha256,
            fold_by_seed_row=folds,
            outer_folds=run_config.outer_folds,
            checkpoint_root=checkpoint_root,
            artifact_path=artifact_path,
            producer_receipt_path=producer_receipt_path,
            model_config_sha256=model_config_sha256,
            run_config_sha256=run_config_sha256,
            model_config_semantic_sha256=_model_config_semantic_sha256(model_config),
            run_config_semantic_sha256=_run_config_semantic_sha256(run_config),
            source_code_sha256=source_code_sha256,
            runtime_environment_sha256=runtime_environment_sha256,
            fold_callback=callback,
            allow_checkpoint_resume=bool(allow_checkpoint_resume),
            production_run_claim_sha256=claim_sha,
        )
    if produced.producer_execution_mode != CURRENT_ONLY_PRODUCTION_EXECUTION_MODE:
        raise CurrentOnlyPipelineError(
            "real current-only trainer did not emit a production-attested receipt"
        )
    return produced


def _load_fit_producer_receipt(
    path: Path,
    *,
    expected_sha256: str,
    preflight_sha256: str,
    fit_map: FitProtocolMap,
    lineage: FitOnlyLineage,
    fit_artifact_sha256: str,
    checkpoint_manifest: CheckpointManifest,
    current_only_source_identity_sha256: str,
    model_config_sha256: str,
    run_config_sha256: str,
    model_config_semantic_sha256: str,
    run_config_semantic_sha256: str,
    source_code_sha256: str,
    runtime_environment_sha256: str,
    fold_assignment_sha256: str,
    production_run_claim_sha256: str,
) -> Mapping[str, object]:
    expected = _require_sha256(expected_sha256, "fit_producer_receipt_sha256")
    if _file_sha256(path) != expected:
        raise CurrentOnlyPipelineError("fit producer receipt file hash changed")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CurrentOnlyPipelineError(f"cannot read fit producer receipt: {error}") from error
    if not isinstance(payload, Mapping):
        raise CurrentOnlyPipelineError("fit producer receipt root is not a mapping")
    _validate_aggregate_producer_receipt(payload)
    if payload.get("schema_version") != CURRENT_ONLY_FIT_PRODUCER_RECEIPT_SCHEMA:
        raise CurrentOnlyPipelineError("fit producer receipt schema changed")
    receipt_lineage = payload.get("lineage")
    contract = payload.get("training_contract")
    if not isinstance(receipt_lineage, Mapping) or not isinstance(contract, Mapping):
        raise CurrentOnlyPipelineError("fit producer receipt lacks lineage/contract")
    if (
        contract.get("producer_execution_mode")
        != CURRENT_ONLY_PRODUCTION_EXECUTION_MODE
        or contract.get("production_trainer_attested") is not True
        or payload.get("status")
        != "production_independent_current_only_fit_oof_complete_not_performance_evidence"
    ):
        raise CurrentOnlyPipelineError("fit producer receipt is not production evidence")
    required = {
        "fit_preflight_receipt_sha256": preflight_sha256,
        "fit_protocol_map_sha256": fit_map.artifact_sha256,
        "fit_lineage_file_sha256": lineage.artifact_sha256,
        "fit_lineage_source_identity_sha256": lineage.source_identity_sha256,
        "checkpoint_manifest_sha256": checkpoint_manifest.manifest_sha256,
        "private_fit_artifact_sha256": fit_artifact_sha256,
        "current_only_source_identity_sha256": current_only_source_identity_sha256,
        "model_config_sha256": model_config_sha256,
        "run_config_sha256": run_config_sha256,
        "model_config_semantic_sha256": model_config_semantic_sha256,
        "run_config_semantic_sha256": run_config_semantic_sha256,
        "source_code_sha256": source_code_sha256,
        "runtime_environment_sha256": runtime_environment_sha256,
        "fold_assignment_sha256": fold_assignment_sha256,
        "production_run_claim_sha256": production_run_claim_sha256,
    }
    if any(receipt_lineage.get(name) != value for name, value in required.items()):
        raise CurrentOnlyPipelineError("fit producer receipt lineage changed")
    if (
        contract.get("selection_payload_consumed") is not False
        or contract.get("history_training_items_consumed") != 0
        or contract.get("history_inference_items_consumed") != 0
        or contract.get("heldout_fit_labels_materialized") is not False
        or contract.get("history_producer_required") is not False
    ):
        raise CurrentOnlyPipelineError("fit producer receipt crossed isolation contract")
    return payload


def _partition_signature(values: Sequence[object]) -> tuple[int, ...]:
    first: dict[str, int] = {}
    signature: list[int] = []
    for index, value in enumerate(values):
        token = str(value)
        first.setdefault(token, index)
        signature.append(first[token])
    return tuple(signature)


def verify_current_only_fit_for_completion(
    *,
    fit_artifact_path: str | Path,
    fit_producer_receipt_path: str | Path,
    expected_fit_producer_receipt_sha256: str,
    checkpoint_root: str | Path,
    fit: FitRoleView,
    fit_map: FitProtocolMap,
    lineage: FitOnlyLineage,
    producer: FitOnlyProducerView | AttestedHistoryFitAlignmentView,
    fit_preflight_receipt_path: str | Path,
    expected_fit_preflight_receipt_sha256: str,
    outer_folds: int,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    model_config_sha256: str,
    run_config_sha256: str,
    source_code_sha256: str,
    runtime_environment_sha256: str,
    production_run_claim_sha256: str,
    device: torch.device,
) -> VerifiedCurrentOnlyFitState:
    """Hash every checkpoint and validate fit artifacts before selection is opened."""

    preflight_sha = _require_sha256(
        expected_fit_preflight_receipt_sha256,
        "expected_fit_preflight_receipt_sha256",
    )
    fit_array_bundle = _verified_live_fit_bundle(fit)
    if fit_array_bundle != lineage.fit_array_hash_bundle_sha256:
        raise CurrentOnlyPipelineError("live fit arrays differ from fit-only lineage")
    _validate_current_only_model_label_contract(model_config, fit.label_order)
    model_semantic_sha = _model_config_semantic_sha256(model_config)
    run_semantic_sha = _run_config_semantic_sha256(run_config)
    bound_hashes = {
        "model_config_sha256": _require_sha256(
            model_config_sha256, "model_config_sha256"
        ),
        "run_config_sha256": _require_sha256(run_config_sha256, "run_config_sha256"),
        "source_code_sha256": _require_sha256(
            source_code_sha256, "source_code_sha256"
        ),
        "runtime_environment_sha256": _require_sha256(
            runtime_environment_sha256, "runtime_environment_sha256"
        ),
    }
    private_root = _validate_current_only_private_layout(
        checkpoint_root=checkpoint_root,
        fit_artifact_path=fit_artifact_path,
        fit_receipt_path=fit_producer_receipt_path,
    )
    expected_claim = current_only_production_claim_sha256(
        fit=fit,
        fit_map=fit_map,
        lineage=lineage,
        fit_preflight_receipt_sha256=preflight_sha,
        model_config=model_config,
        run_config=run_config,
        model_config_sha256=bound_hashes["model_config_sha256"],
        run_config_sha256=bound_hashes["run_config_sha256"],
        source_code_sha256=bound_hashes["source_code_sha256"],
        runtime_environment_sha256=bound_hashes["runtime_environment_sha256"],
    )
    claim_sha = _require_sha256(
        production_run_claim_sha256, "production_run_claim_sha256"
    )
    if claim_sha != expected_claim:
        raise CurrentOnlyPipelineError("current-only completion claim lineage changed")
    _verify_current_only_private_claim(private_root, claim_sha)
    preflight_path = Path(fit_preflight_receipt_path)
    if _file_sha256(preflight_path) != preflight_sha:
        raise CurrentOnlyPipelineError("fit preflight receipt file hash changed")
    receipt = _load_receipt(preflight_path)
    _assert_producer_sidecar_lineage(producer, receipt)
    _verify_fit_protocol_map_file(
        fit_map,
        receipt_path=fit_preflight_receipt_path,
        expected_receipt_sha256=preflight_sha,
    )
    _verify_fit_only_lineage_file(
        lineage,
        fit=fit,
        fit_map=fit_map,
        receipt_path=fit_preflight_receipt_path,
        expected_receipt_sha256=preflight_sha,
    )
    if (
        producer.dataset != lineage.dataset
        or producer.label_order != lineage.label_order
        or producer.seeds != lineage.seeds
    ):
        raise CurrentOnlyPipelineError(
            "history producer differs from fit-only lineage contract"
        )
    _, fit_positions = align_fit_protocol_to_producer(fit_map, producer)
    aligned_clusters = np.asarray(producer.fit_cluster_codes)[fit_positions]
    if _partition_signature(aligned_clusters) != _partition_signature(fit.groups):
        raise CurrentOnlyPipelineError(
            "history producer fit clusters differ from fit-only groups"
        )
    manifest = build_checkpoint_manifest(
        checkpoint_root, seeds=EXPECTED_SEEDS, outer_folds=int(outer_folds)
    )
    artifact_path = Path(fit_artifact_path).resolve()
    artifact_sha = _file_sha256(artifact_path)
    values = load_private_npz_mapping(artifact_path)
    validate_current_only_fit_bootstrap_artifact(
        values, lineage=lineage, checkpoint_manifest=manifest
    )
    current_identity = _require_sha256(
        _single_text(
            np.asarray(values["current_only_source_identity_sha256"]),
            "current_only_source_identity_sha256",
        ),
        "current_only_source_identity_sha256",
    )
    if current_identity == producer.source_identity_sha256:
        raise CurrentOnlyPipelineError(
            "current-only fit identity collides with history producer"
        )
    expected_current_identity = independent_current_only_source_identity(
        producer_source_identity_sha256=lineage.source_identity_sha256,
        model_config=model_config,
        run_config=run_config,
        rows=fit.rows,
        data_contract_sha256=fit_map.fit_arrays_contract_sha256,
    )
    if current_identity != expected_current_identity:
        raise CurrentOnlyPipelineError(
            "current-only source identity differs from live model/run/data contract"
        )
    fold_by_seed_row = np.asarray(values["fit_fold_by_seed_row"], dtype=np.int32)
    fold_assignment_sha = _require_sha256(
        _single_text(
            np.asarray(values["fold_assignment_sha256"]),
            "fold_assignment_sha256",
        ),
        "fold_assignment_sha256",
    )
    _verify_complete_checkpoint_payloads(
        checkpoint_root,
        manifest,
        fit=fit,
        model_config=model_config,
        run_config=run_config,
        source_identity_sha256=current_identity,
        fold_by_seed_row=fold_by_seed_row,
        device=device,
    )
    if _file_sha256(artifact_path) != artifact_sha:
        raise CurrentOnlyPipelineError("fit probability artifact changed while validating")
    _load_fit_producer_receipt(
        Path(fit_producer_receipt_path),
        expected_sha256=expected_fit_producer_receipt_sha256,
        preflight_sha256=preflight_sha,
        fit_map=fit_map,
        lineage=lineage,
        fit_artifact_sha256=artifact_sha,
        checkpoint_manifest=manifest,
        current_only_source_identity_sha256=current_identity,
        model_config_sha256=bound_hashes["model_config_sha256"],
        run_config_sha256=bound_hashes["run_config_sha256"],
        model_config_semantic_sha256=model_semantic_sha,
        run_config_semantic_sha256=run_semantic_sha,
        source_code_sha256=bound_hashes["source_code_sha256"],
        runtime_environment_sha256=bound_hashes["runtime_environment_sha256"],
        fold_assignment_sha256=fold_assignment_sha,
        production_run_claim_sha256=claim_sha,
    )
    selection_sidecar = cast(Mapping[str, object], receipt["sidecars"])[SELECTION_ROLE]
    if not isinstance(selection_sidecar, Mapping):
        raise CurrentOnlyPipelineError("fit receipt lacks model-selection sidecar lineage")
    selection_feature_sha = _require_sha256(
        selection_sidecar.get("feature_sha256"), "selection_feature_sha256"
    )
    return VerifiedCurrentOnlyFitState(
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha,
        producer_receipt_path=Path(fit_producer_receipt_path).resolve(),
        producer_receipt_sha256=_require_sha256(
            expected_fit_producer_receipt_sha256,
            "expected_fit_producer_receipt_sha256",
        ),
        fit_preflight_receipt_sha256=preflight_sha,
        fit_lineage_artifact_sha256=lineage.artifact_sha256,
        fit_lineage_source_identity_sha256=lineage.source_identity_sha256,
        history_producer_file_sha256=producer.producer_file_sha256,
        history_producer_source_identity_sha256=producer.source_identity_sha256,
        history_checkpoint_manifest_sha256=producer.checkpoint_manifest_sha256,
        selection_feature_sha256=selection_feature_sha,
        checkpoint_manifest=manifest,
        current_only_source_identity_sha256=current_identity,
        fit_array_hash_bundle_sha256=fit_array_bundle,
        fold_assignment_sha256=fold_assignment_sha,
        producer_execution_mode=CURRENT_ONLY_PRODUCTION_EXECUTION_MODE,
        model_config_sha256=bound_hashes["model_config_sha256"],
        run_config_sha256=bound_hashes["run_config_sha256"],
        model_config_semantic_sha256=model_semantic_sha,
        run_config_semantic_sha256=run_semantic_sha,
        source_code_sha256=bound_hashes["source_code_sha256"],
        runtime_environment_sha256=bound_hashes[
            "runtime_environment_sha256"
        ],
        production_run_claim_sha256=claim_sha,
    )


def _completion_receipt(
    *,
    producer: CurrentOnlyProducerAlignmentView,
    fit_map: FitProtocolMap,
    fit_state: VerifiedCurrentOnlyFitState,
    preflight_sha256: str,
    selection_feature_sha256: str,
    artifact_sha256: str,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema_version": CURRENT_ONLY_COMPLETION_RECEIPT_SCHEMA,
        "status": "complete_selection_probability_cache_not_performance_evidence",
        "dataset": producer.dataset,
        "claim_boundary": (
            "Fit OOF plus model-selection feature-only inference; no model-selection "
            "label and no performance metric were consumed."
        ),
        "lineage": {
            "fit_preflight_receipt_sha256": preflight_sha256,
            "fit_producer_receipt_sha256": fit_state.producer_receipt_sha256,
            "fit_protocol_map_sha256": fit_map.artifact_sha256,
            "fit_lineage_file_sha256": fit_state.fit_lineage_artifact_sha256,
            "fit_lineage_source_identity_sha256": (
                fit_state.fit_lineage_source_identity_sha256
            ),
            "fit_probability_artifact_sha256": fit_state.artifact_sha256,
            "producer_file_sha256": producer.producer_file_sha256,
            "producer_source_identity_sha256": producer.source_identity_sha256,
            "history_checkpoint_manifest_sha256": producer.checkpoint_manifest_sha256,
            "current_checkpoint_manifest_sha256": (
                fit_state.checkpoint_manifest.manifest_sha256
            ),
            "current_only_source_identity_sha256": (
                fit_state.current_only_source_identity_sha256
            ),
            "fit_array_hash_bundle_sha256": (
                fit_state.fit_array_hash_bundle_sha256
            ),
            "fold_assignment_sha256": fit_state.fold_assignment_sha256,
            "model_config_sha256": fit_state.model_config_sha256,
            "run_config_sha256": fit_state.run_config_sha256,
            "model_config_semantic_sha256": (
                fit_state.model_config_semantic_sha256
            ),
            "run_config_semantic_sha256": fit_state.run_config_semantic_sha256,
            "source_code_sha256": fit_state.source_code_sha256,
            "runtime_environment_sha256": (
                fit_state.runtime_environment_sha256
            ),
            "production_run_claim_sha256": (
                fit_state.production_run_claim_sha256
            ),
            "selection_feature_sha256": selection_feature_sha256,
            "private_current_only_cache_sha256": artifact_sha256,
        },
        "completion_contract": {
            "protocol": CURRENT_ONLY_COMPLETION_PROTOCOL,
            "seeds": list(EXPECTED_SEEDS),
            "outer_folds": fit_state.checkpoint_manifest.outer_folds,
            "fit_query_count": len(producer.fit_query_indices),
            "selection_query_count": len(producer.selection_query_indices),
            "complete_checkpoint_only": True,
            "producer_execution_mode": fit_state.producer_execution_mode,
            "production_trainer_attested": True,
            "one_selection_probability_per_seed_and_query": True,
            "history_training_items_consumed": 0,
            "history_inference_items_consumed": 0,
            "selection_feature_materialized": True,
            "selection_label_materialized": False,
            "selection_label_deserialized": False,
            "selection_label_file_accessed": False,
            "evaluate_stage_run": False,
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


def complete_current_only_selection_probabilities(
    *,
    fit: FitRoleView,
    selection: SelectionFeatureView,
    fit_map: FitProtocolMap,
    lineage: FitOnlyLineage,
    fit_producer: FitOnlyProducerView | AttestedHistoryFitAlignmentView,
    producer: CurrentOnlyProducerAlignmentView,
    fit_state: VerifiedCurrentOnlyFitState,
    fit_preflight_receipt_path: str | Path,
    expected_fit_preflight_receipt_sha256: str,
    checkpoint_root: str | Path,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    model_config_sha256: str,
    run_config_sha256: str,
    source_code_sha256: str,
    runtime_environment_sha256: str,
    production_run_claim_sha256: str,
    selection_feature_sha256: str,
    artifact_path: str | Path,
    completion_receipt_path: str | Path,
    device: torch.device,
) -> CompletedCurrentOnlyProduction:
    """Complete selection inference and bridge to the real strategy cache schema."""

    if selection.labels_materialized:
        raise CurrentOnlyPipelineError("selection labels entered complete-selection")
    if fit_state.producer_execution_mode != CURRENT_ONLY_PRODUCTION_EXECUTION_MODE:
        raise CurrentOnlyPipelineError("completion refuses a synthetic fit producer")
    live_fit_bundle = _verified_live_fit_bundle(fit)
    if live_fit_bundle != fit_state.fit_array_hash_bundle_sha256:
        raise CurrentOnlyPipelineError("live fit arrays changed after the fit gate")
    _validate_current_only_model_label_contract(model_config, fit.label_order)
    live_bindings = {
        "model_config_sha256": _require_sha256(
            model_config_sha256, "model_config_sha256"
        ),
        "run_config_sha256": _require_sha256(run_config_sha256, "run_config_sha256"),
        "model_config_semantic_sha256": _model_config_semantic_sha256(model_config),
        "run_config_semantic_sha256": _run_config_semantic_sha256(run_config),
        "source_code_sha256": _require_sha256(
            source_code_sha256, "source_code_sha256"
        ),
        "runtime_environment_sha256": _require_sha256(
            runtime_environment_sha256, "runtime_environment_sha256"
        ),
    }
    if any(
        getattr(fit_state, name) != value for name, value in live_bindings.items()
    ):
        changed = sorted(
            name
            for name, value in live_bindings.items()
            if getattr(fit_state, name) != value
        )
        raise CurrentOnlyPipelineError(
            f"completion model/run/code/runtime lineage changed: {changed}"
        )
    claim_sha = _require_sha256(
        production_run_claim_sha256, "production_run_claim_sha256"
    )
    expected_claim = current_only_production_claim_sha256(
        fit=fit,
        fit_map=fit_map,
        lineage=lineage,
        fit_preflight_receipt_sha256=fit_state.fit_preflight_receipt_sha256,
        model_config=model_config,
        run_config=run_config,
        model_config_sha256=live_bindings["model_config_sha256"],
        run_config_sha256=live_bindings["run_config_sha256"],
        source_code_sha256=live_bindings["source_code_sha256"],
        runtime_environment_sha256=live_bindings[
            "runtime_environment_sha256"
        ],
    )
    if (
        claim_sha != expected_claim
        or claim_sha != fit_state.production_run_claim_sha256
    ):
        raise CurrentOnlyPipelineError("completion production claim lineage changed")
    private_root = _validate_current_only_private_layout(
        checkpoint_root=checkpoint_root,
        fit_artifact_path=fit_state.artifact_path,
        fit_receipt_path=fit_state.producer_receipt_path,
        complete_artifact_path=artifact_path,
        complete_receipt_path=completion_receipt_path,
    )
    _verify_current_only_private_claim(private_root, claim_sha)
    preflight_sha = _require_sha256(
        expected_fit_preflight_receipt_sha256,
        "expected_fit_preflight_receipt_sha256",
    )
    if _file_sha256(Path(fit_preflight_receipt_path)) != preflight_sha:
        raise CurrentOnlyPipelineError("fit preflight receipt file hash changed")
    if fit_state.fit_preflight_receipt_sha256 != preflight_sha:
        raise CurrentOnlyPipelineError("completion uses a different fit preflight receipt")
    if (
        _file_sha256(fit_state.artifact_path) != fit_state.artifact_sha256
        or _file_sha256(fit_state.producer_receipt_path)
        != fit_state.producer_receipt_sha256
    ):
        raise CurrentOnlyPipelineError(
            "fit probability artifact/receipt changed after verification"
        )
    _verify_fit_protocol_map_file(
        fit_map,
        receipt_path=fit_preflight_receipt_path,
        expected_receipt_sha256=preflight_sha,
    )
    _verify_fit_only_lineage_file(
        lineage,
        fit=fit,
        fit_map=fit_map,
        receipt_path=fit_preflight_receipt_path,
        expected_receipt_sha256=preflight_sha,
    )
    if (
        fit_state.fit_lineage_artifact_sha256 != lineage.artifact_sha256
        or fit_state.fit_lineage_source_identity_sha256
        != lineage.source_identity_sha256
    ):
        raise CurrentOnlyPipelineError(
            "completion uses a different fit-only lineage"
        )
    if (
        fit_producer.producer_file_sha256
        != fit_state.history_producer_file_sha256
        or producer.producer_file_sha256
        != fit_state.history_producer_file_sha256
        or fit_producer.source_identity_sha256
        != fit_state.history_producer_source_identity_sha256
        or producer.source_identity_sha256
        != fit_state.history_producer_source_identity_sha256
        or fit_producer.checkpoint_manifest_sha256
        != fit_state.history_checkpoint_manifest_sha256
        or producer.checkpoint_manifest_sha256
        != fit_state.history_checkpoint_manifest_sha256
    ):
        raise CurrentOnlyPipelineError(
            "history producer changed after the pre-selection fit gate"
        )
    if fit.contract_sha256 != fit_map.fit_arrays_contract_sha256:
        raise CurrentOnlyPipelineError("completion fit material differs from fit map")
    _, fit_positions = align_fit_protocol_to_producer(fit_map, fit_producer)
    _, selection_positions = align_selection_protocol_to_producer(selection, producer)
    aligned_selection_clusters = np.asarray(producer.selection_cluster_codes)[
        selection_positions
    ]
    if _partition_signature(aligned_selection_clusters) != _partition_signature(
        selection.groups
    ):
        raise CurrentOnlyPipelineError(
            "producer selection clusters differ from selection-feature groups"
        )
    selection_feature_sha = _require_sha256(
        selection_feature_sha256, "selection_feature_sha256"
    )
    if selection_feature_sha != fit_state.selection_feature_sha256:
        raise CurrentOnlyPipelineError("selection feature lineage changed after fit gate")
    receipt = _load_receipt(Path(fit_preflight_receipt_path))
    selection_sidecar = cast(Mapping[str, object], receipt["sidecars"])["model_selection"]
    if not isinstance(selection_sidecar, Mapping) or selection_sidecar.get(
        "feature_sha256"
    ) != selection_feature_sha:
        raise CurrentOnlyPipelineError("selection feature hash differs from preflight")

    destination = Path(artifact_path)
    receipt_destination = Path(completion_receipt_path)
    if destination.exists() or receipt_destination.exists():
        raise FileExistsError("current-only completion output already exists")
    if fit_state.checkpoint_manifest.outer_folds != int(run_config.outer_folds):
        raise CurrentOnlyPipelineError("checkpoint folds differ from run config")
    verify_checkpoint_manifest(checkpoint_root, fit_state.checkpoint_manifest)

    # Re-open and revalidate the exact fit artifact that passed the pre-selection
    # gate.  This prevents a mutable in-memory mapping or a replaced file from
    # silently changing the cache lineage after selection features are opened.
    fit_values = load_private_npz_mapping(fit_state.artifact_path)
    validate_current_only_fit_bootstrap_artifact(
        fit_values,
        lineage=lineage,
        checkpoint_manifest=fit_state.checkpoint_manifest,
    )
    if _file_sha256(fit_state.artifact_path) != fit_state.artifact_sha256:
        raise CurrentOnlyPipelineError("fit probability artifact changed during completion")
    fit_probability_local = _probability(
        np.asarray(fit_values["fit_probability_oof"]),
        (
            len(EXPECTED_SEEDS),
            lineage.rows,
            len(lineage.label_order),
        ),
        "fit_probability_oof",
    )
    fit_probability = np.empty(
        (
            len(EXPECTED_SEEDS),
            len(producer.fit_query_indices),
            len(producer.label_order),
        ),
        dtype=np.float32,
    )
    fit_probability[:, fit_positions] = fit_probability_local
    fold_by_row = np.asarray(fit_values["fit_fold_by_seed_row"], dtype=np.int32)
    current_identity = _require_sha256(
        _single_text(
            np.asarray(fit_values["current_only_source_identity_sha256"]),
            "current_only_source_identity_sha256",
        ),
        "current_only_source_identity_sha256",
    )
    if (
        current_identity != fit_state.current_only_source_identity_sha256
        or _array_sha256(fold_by_row) != fit_state.fold_assignment_sha256
    ):
        raise CurrentOnlyPipelineError(
            "completion fit identity/fold assignment differs from pre-selection gate"
        )
    _verify_complete_checkpoint_payloads(
        checkpoint_root,
        fit_state.checkpoint_manifest,
        fit=fit,
        model_config=model_config,
        run_config=run_config,
        source_identity_sha256=current_identity,
        fold_by_seed_row=fold_by_row,
        device=device,
    )
    selection_probability = np.zeros(
        (
            len(EXPECTED_SEEDS),
            len(producer.selection_query_indices),
            len(producer.label_order),
        ),
        dtype=np.float64,
    )
    baseline_run_config = replace(run_config, subset_dropout_probability=0.0)
    local_selection_queries = np.arange(len(selection.texts), dtype=np.int64)
    for seed_index, seed in enumerate(EXPECTED_SEEDS):
        for fold in range(int(run_config.outer_folds)):
            held = np.flatnonzero(fold_by_row[seed_index] == fold).astype(np.int64)
            train = np.flatnonzero(fold_by_row[seed_index] != fold).astype(np.int64)
            fit_corpus = _fit_corpus_from_view(
                fit,
                model_config=model_config,
                heldout_indices=held,
                speaker_reference_indices=train,
            )
            selection_corpus = _selection_corpus_from_view(
                selection,
                fit=fit,
                model_config=model_config,
                fit_speaker_reference_indices=train,
            )
            split = _split_from_outer_partition(
                fit_corpus,
                outer_train=train,
                heldout=held,
                validation_fraction=baseline_run_config.inner_validation_fraction,
                seed=seed,
                fold=fold,
            )
            verify_checkpoint_manifest(checkpoint_root, fit_state.checkpoint_manifest)
            trained = train_independent_current_only_fold_seed(
                fit_corpus,
                split,
                producer_source_identity_sha256=(
                    fit_state.fit_lineage_source_identity_sha256
                ),
                model_config=model_config,
                run_config=baseline_run_config,
                seed=seed,
                checkpoint_root=Path(checkpoint_root),
                device=device,
                data_contract_sha256=fit_map.fit_arrays_contract_sha256,
                require_complete_checkpoint=True,
            )
            if (
                trained.source_identity_sha256 != current_identity
                or trained.trained.summary.get("resumed_complete_checkpoint") is not True
                or trained.trained.summary.get("resumed_partial_checkpoint") is not False
            ):
                raise CurrentOnlyPipelineError(
                    "selection inference did not load the registered complete checkpoint"
                )
            selection_text = trained.trained.processor.transform(selection_corpus.texts)
            fold_probability = predict_one_probability_per_query(
                trained.trained.model,
                selection_corpus,
                selection_text,
                local_selection_queries,
                tuple(() for _ in local_selection_queries),
                device=device,
                batch_size=baseline_run_config.inference_batch_size,
                max_history_items=baseline_run_config.max_history_items,
            )
            validated = _probability(
                np.asarray(fold_probability),
                (len(selection.texts), len(producer.label_order)),
                "selection_fold_probability",
            )
            selection_probability[seed_index, selection_positions] += validated
        selection_probability[seed_index] /= float(run_config.outer_folds)
    verify_checkpoint_manifest(checkpoint_root, fit_state.checkpoint_manifest)
    selection_probability = _probability(
        selection_probability,
        (
            len(EXPECTED_SEEDS),
            len(producer.selection_query_indices),
            len(producer.label_order),
        ),
        "selection_probability_fold_ensemble",
    )

    # The evidence builder is deliberately duck-typed here: it reads only the
    # outcome-free alignment fields in CurrentOnlyProducerAlignmentView.  The
    # evaluate stage later revalidates the cache against the full producer.
    values = build_current_only_artifact_mapping(  # type: ignore[arg-type]
        producer,
        fit_probability_oof=fit_probability,
        selection_probability_fold_ensemble=selection_probability,
        current_only_source_identity_sha256=current_identity,
        checkpoint_manifest_sha256=fit_state.checkpoint_manifest.manifest_sha256,
    )
    artifact_sha = _atomic_savez_once(destination, values)
    completion_receipt = _completion_receipt(
        producer=producer,
        fit_map=fit_map,
        fit_state=fit_state,
        preflight_sha256=preflight_sha,
        selection_feature_sha256=selection_feature_sha,
        artifact_sha256=artifact_sha,
    )
    receipt_sha = _atomic_json_once(receipt_destination, completion_receipt)
    return CompletedCurrentOnlyProduction(
        artifact_path=destination,
        artifact_sha256=artifact_sha,
        receipt_path=receipt_destination,
        receipt_sha256=receipt_sha,
        checkpoint_manifest_sha256=fit_state.checkpoint_manifest.manifest_sha256,
    )


def validate_completed_current_only_files(
    *,
    artifact_path: str | Path,
    producer_path: str | Path,
) -> dict[str, object]:
    """Validate the strategy-consumable cache without producing performance."""

    from .causal_backbone_evidence import load_current_only_artifact, load_producer_cache

    producer = load_producer_cache(producer_path)
    artifact = load_current_only_artifact(artifact_path, producer)
    return {
        "schema_version": "carma_independent_current_only_private_v1",
        "status": "valid_strategy_consumable_current_only_cache_not_performance_evidence",
        "dataset": artifact.dataset,
        "seed_count": len(artifact.seeds),
        "fit_query_count": len(artifact.fit_query_indices),
        "selection_query_count": len(artifact.selection_query_indices),
        "artifact_sha256": _file_sha256(Path(artifact_path)),
        "contains_row_probabilities_or_labels": False,
        "performance_metric_computed": False,
    }
