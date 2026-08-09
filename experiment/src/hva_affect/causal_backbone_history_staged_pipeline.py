"""Leakage-resistant staged producer for the history-aware backbone.

This module deliberately exposes only two capabilities:

``produce_history_fit_only``
    Sees fit features and fit labels, trains group-cross-fitted history-aware
    models, and writes fit OOF probabilities and fit-only utility targets to
    physically separate private artifacts.

``complete_history_selection_outcomes``
    Sees fit material again and model-selection *features only*.  It verifies
    every fit artifact and every complete checkpoint before inference, restores
    checkpoints with ``require_complete_checkpoint=True``, and writes an
    outcome-free probability cache.  It never computes selection utility
    targets or a performance metric.

There is intentionally no evaluate function here.  Selection labels remain a
separate capability for the later evidence evaluator.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast

import joblib
import numpy as np
import torch

from .bidirectional_emotion_utility import BidirectionalCoalitionTask
from .causal_backbone_evidence_runner import (
    ENDPOINT_CONTEXT_NAMES,
    EXPECTED_SEEDS,
    FIT_ROLE,
    SELECTION_ROLE,
    UTILITY_CONTEXT_NAMES,
    CheckpointManifest,
    FitRoleView,
    HashedSidecarSet,
    SelectionFeatureView,
    SidecarRecord,
    _SPECS,
    _array_sha256,
    _canonical_production_source_key,
    _canonical_sha256,
    _file_sha256,
    _load_receipt,
    _materialize_selection_feature,
    _plain_npz_filename,
    _read_manifest_json,
    _require_sha256,
    _single_text,
    _validate_manifest_contract,
    build_checkpoint_manifest,
    validate_fit_receipt,
    verify_checkpoint_manifest,
    verify_fit_only_receipt_inputs,
    verify_selection_feature_receipt_inputs,
)
from .causal_backbone_evidence_stage_b import (
    FitProtocolMap,
    StageBContractError,
    _validate_aggregate_producer_receipt,
    _verify_fit_protocol_map_file,
)
from .causal_multimodal_backbone import (
    CausalBackboneConfig,
    CausalMultimodalBackbone,
)
from .emotiontalk_causal_backbone_runner import (
    BackboneRunConfig,
    CrossfitSplit,
    FoldTextProcessor,
    OpenRoleCorpus,
    UtilitySamplingConfig,
    _capture_rng_state,
    _encode_task_contexts,
    _indices_sha256,
    _restore_rng_state,
    _task_sha256,
    _torch_load_local,
    _utility_arrays,
    make_crossfit_splits,
    predict_current_and_all_history,
    predict_utility_contexts,
    sample_corpus_bidirectional_tasks,
    train_one_fold_seed,
)


HISTORY_FIT_OUTCOME_SCHEMA = "carma_history_backbone_fit_outcome_private_v1"
HISTORY_FIT_TARGETS_SCHEMA = "carma_history_backbone_fit_targets_private_v1"
HISTORY_FIT_RECEIPT_SCHEMA = "carma_history_backbone_fit_receipt_v1"
HISTORY_COMPLETE_OUTCOME_SCHEMA = "carma_history_backbone_complete_outcome_private_v1"
HISTORY_COMPLETE_RECEIPT_SCHEMA = "carma_history_backbone_complete_receipt_v1"
HISTORY_STAGED_PROTOCOL = (
    "fit_oof_then_selection_feature_only_complete_checkpoint_fold_ensemble_v1"
)
PRODUCTION_TRAINER_MODE = "canonical_real_history_fold_trainer_v1"
SYNTHETIC_TRAINER_MODE = "injected_test_callback_nonproduction_v1"
PRODUCTION_FIT_STATUS = "history_fit_oof_complete_not_performance_evidence"
SYNTHETIC_FIT_STATUS = "history_fit_synthetic_contract_fixture_nonproduction"
PRODUCTION_COMPLETION_STATUS = "selection_outcome_free_probability_cache_complete"
SYNTHETIC_COMPLETION_STATUS = "selection_outcome_free_synthetic_fixture_nonproduction"


class HistoryStagedPipelineError(StageBContractError):
    """Raised when a staged history producer crosses a capability boundary."""


@dataclass(frozen=True)
class EncodedHistoryTasks:
    query_indices: np.ndarray
    candidate_indices: np.ndarray
    addition_contexts: tuple[tuple[int, ...], ...]
    deletion_contexts: tuple[tuple[int, ...], ...]
    task_sha256: str

    def __len__(self) -> int:
        return len(self.query_indices)


@dataclass(frozen=True)
class HistoryFitFoldRequest:
    """A fold request in which outer-heldout labels are physically absent."""

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
    train_protocol_row_ids: np.ndarray
    train_histories: tuple[tuple[int, ...], ...]
    heldout_texts: tuple[str, ...]
    heldout_audio: np.ndarray
    heldout_video: np.ndarray
    heldout_group_tokens: np.ndarray
    heldout_speaker_tokens: np.ndarray
    heldout_turns: np.ndarray
    heldout_protocol_row_ids: np.ndarray
    heldout_histories: tuple[tuple[int, ...], ...]
    heldout_tasks: tuple[BidirectionalCoalitionTask, ...]
    heldout_labels_materialized: bool
    heldout_targets_materialized: bool
    source_identity_sha256: str
    checkpoint_root: Path


@dataclass(frozen=True)
class HistoryFitFoldOutput:
    endpoint_probability: np.ndarray
    utility_probability: np.ndarray
    source_identity_sha256: str


HistoryFitFoldCallback = Callable[[HistoryFitFoldRequest], HistoryFitFoldOutput]


@dataclass(frozen=True)
class HistoryFitProduction:
    outcome_artifact_path: Path
    outcome_artifact_sha256: str
    targets_artifact_path: Path
    targets_artifact_sha256: str
    receipt_path: Path
    receipt_sha256: str
    checkpoint_manifest: CheckpointManifest
    source_identity_sha256: str
    production_trainer: bool


@dataclass(frozen=True)
class _HistoryFitOutcomeView:
    dataset: str
    label_order: tuple[str, ...]
    seeds: tuple[int, ...]
    protocol_row_ids: np.ndarray
    query_indices: np.ndarray
    cluster_codes: np.ndarray
    histories_sha256: str
    tasks: EncodedHistoryTasks
    endpoint_probability_oof: np.ndarray
    utility_probability_oof: np.ndarray
    fold_by_seed_query: np.ndarray
    source_identity_sha256: str
    checkpoint_manifest_sha256: str
    model_config_sha256: str
    run_config_sha256: str
    utility_config_sha256: str
    artifact_sha256: str


@dataclass(frozen=True)
class HistoryFitTargetsView:
    """The label-derived fit utility targets; never accepted by completion."""

    dataset: str
    seeds: tuple[int, ...]
    task_sha256: str
    forward_utility: np.ndarray
    backward_utility: np.ndarray
    asymmetry: np.ndarray
    sign_agreement: np.ndarray
    source_identity_sha256: str
    fit_outcome_artifact_sha256: str
    artifact_sha256: str


@dataclass(frozen=True)
class VerifiedHistoryFitState:
    """Outcome-free state that may be passed into complete-selection."""

    fit_outcome_path: Path
    fit_outcome_sha256: str
    fit_targets_path: Path
    fit_targets_sha256: str
    fit_receipt_path: Path
    fit_receipt_sha256: str
    fit_preflight_receipt_sha256: str
    selection_feature_sha256: str
    checkpoint_manifest: CheckpointManifest
    source_identity_sha256: str
    fit_outcome: _HistoryFitOutcomeView
    production_trainer: bool
    private_output_root: Path | None
    execution_environment_sha256: str
    production_run_claim_sha256: str | None


@dataclass(frozen=True)
class VerifiedHistorySelectionView:
    """Feature-only selection capability opened after the complete-fit gate."""

    view: SelectionFeatureView
    feature_file_sha256: str
    normalized_feature_sha256: str
    fit_gate_sha256: str
    checkpoint_manifest_sha256: str
    config_sha256: Mapping[str, str]
    code_sha256: Mapping[str, str]
    runtime_environment_sha256: str
    live_lineage_sha256: str


@dataclass(frozen=True)
class HistoryOutcomeFreeView:
    """Probabilities and task contexts only; no ground-truth-derived target."""

    dataset: str
    label_order: tuple[str, ...]
    seeds: tuple[int, ...]
    fit_protocol_row_ids: np.ndarray
    selection_protocol_row_ids: np.ndarray
    fit_cluster_codes: np.ndarray
    selection_cluster_codes: np.ndarray
    fit_histories_sha256: str
    selection_histories_sha256: str
    fit_tasks: EncodedHistoryTasks
    selection_tasks: EncodedHistoryTasks
    fit_endpoint_probability_oof: np.ndarray
    selection_endpoint_probability_fold_ensemble: np.ndarray
    fit_utility_probability_oof: np.ndarray
    selection_utility_probability_fold_ensemble: np.ndarray
    source_identity_sha256: str
    checkpoint_manifest_sha256: str
    fit_outcome_artifact_sha256: str
    fit_targets_artifact_sha256: str
    artifact_sha256: str


@dataclass(frozen=True)
class CompletedHistoryProduction:
    artifact_path: Path
    artifact_sha256: str
    receipt_path: Path
    receipt_sha256: str
    checkpoint_manifest_sha256: str


@dataclass(frozen=True)
class VerifiedHistoryCompletionAttestation:
    """Minimal production lineage safe to pass to a downstream fit gate."""

    dataset: str
    artifact_path: Path
    artifact_sha256: str
    completion_receipt_path: Path
    completion_receipt_sha256: str
    fit_producer_receipt_path: Path
    fit_producer_receipt_sha256: str
    source_identity_sha256: str
    checkpoint_manifest_sha256: str
    production_run_claim_sha256: str
    fit_preflight_receipt_sha256: str
    config_sha256: Mapping[str, str]
    code_sha256: Mapping[str, str]
    runtime_environment_sha256: str
    execution_environment_sha256: str
    model_config_sha256: str
    run_config_sha256: str
    utility_config_sha256: str


_PRIVATE_CHECKPOINT_NAME = "checkpoints"
_PRIVATE_FIT_OUTCOME_NAME = "history-fit-outcome.npz"
_PRIVATE_FIT_TARGETS_NAME = "history-fit-targets.npz"
_PRIVATE_FIT_RECEIPT_NAME = "history-fit-receipt.json"
_PRIVATE_COMPLETE_OUTCOME_NAME = "history-complete-outcome.npz"
_PRIVATE_COMPLETE_RECEIPT_NAME = "history-complete-receipt.json"
_PRIVATE_CLAIM_NAME = "history-run-claim.json"
_PRIVATE_LOCK_NAME = "history-run.lock"


def _canonical_repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def claim_new_history_private_root(path: str | Path) -> Path:
    """Atomically claim a new, repository-external production directory."""

    root = Path(path).resolve()
    repository = _canonical_repository_root()
    if _is_within(root, repository):
        raise HistoryStagedPipelineError(
            "history private output root must be outside the repository"
        )
    if not root.is_absolute() or root == Path(root.anchor):
        raise HistoryStagedPipelineError("history private output root is unsafe")
    if not root.parent.is_dir():
        raise HistoryStagedPipelineError(
            "history private output parent must already exist"
        )
    try:
        root.mkdir(exist_ok=False)
    except FileExistsError:
        raise FileExistsError("history private output root must be all-new") from None
    return root


def history_production_claim_sha256(
    *,
    fit: FitRoleView,
    fit_map: FitProtocolMap,
    fit_preflight_receipt_sha256: str,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    utility_config: UtilitySamplingConfig,
    config_sha256: Mapping[str, str],
    code_sha256: Mapping[str, str],
    runtime_environment_sha256: str,
    execution_environment_sha256: str | None = None,
) -> str:
    """Bind a resumable private directory to one exact production lineage."""

    return _canonical_sha256(
        {
            "protocol": HISTORY_STAGED_PROTOCOL,
            "trainer_mode": PRODUCTION_TRAINER_MODE,
            "dataset": fit.dataset,
            "fit_contract_sha256": fit.contract_sha256,
            "fit_protocol_map_sha256": fit_map.artifact_sha256,
            "fit_preflight_receipt_sha256": _require_sha256(
                fit_preflight_receipt_sha256,
                "fit_preflight_receipt_sha256",
            ),
            "model_config_sha256": _config_sha256(model_config),
            "run_config_sha256": _config_sha256(run_config),
            "utility_config_sha256": _config_sha256(utility_config),
            "config_sha256": _normalized_hash_mapping(
                config_sha256, "config_sha256"
            ),
            "code_sha256": _normalized_hash_mapping(code_sha256, "code_sha256"),
            "runtime_environment_sha256": _require_sha256(
                runtime_environment_sha256, "runtime_environment_sha256"
            ),
            "execution_environment_sha256": _require_sha256(
                runtime_environment_sha256
                if execution_environment_sha256 is None
                else execution_environment_sha256,
                "execution_environment_sha256",
            ),
        }
    )


def claim_or_resume_history_private_root(
    path: str | Path,
    *,
    production_claim_sha256: str,
    allow_resume: bool,
) -> Path:
    claim_sha = _require_sha256(
        production_claim_sha256, "production_claim_sha256"
    )
    root = Path(path).resolve()
    marker = root / _PRIVATE_CLAIM_NAME
    if not root.exists():
        claim_new_history_private_root(root)
        payload = {
            "schema_version": "carma_history_private_run_claim_v1",
            "status": "claimed_for_single_lineage_interruptible_fit",
            "production_claim_sha256": claim_sha,
        }
        descriptor = os.open(
            marker,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                (
                    json.dumps(payload, sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
            )
            handle.flush()
            os.fsync(handle.fileno())
        return root
    if not allow_resume:
        raise FileExistsError("history private output root must be all-new")
    if not root.is_dir() or _is_within(root, _canonical_repository_root()):
        raise HistoryStagedPipelineError("history resume root is unsafe")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HistoryStagedPipelineError("history resume claim is missing or corrupt") from error
    expected = {
        "schema_version": "carma_history_private_run_claim_v1",
        "status": "claimed_for_single_lineage_interruptible_fit",
        "production_claim_sha256": claim_sha,
    }
    if payload != expected:
        raise HistoryStagedPipelineError("history resume lineage claim changed")
    allowed_names = {
        _PRIVATE_CLAIM_NAME,
        _PRIVATE_CHECKPOINT_NAME,
        _PRIVATE_LOCK_NAME,
    }
    if any(entry.name not in allowed_names for entry in root.iterdir()):
        raise HistoryStagedPipelineError(
            "history resume root contains a finalized or unexpected artifact"
        )
    return root


def _verify_private_claim(root: Path, expected_sha256: str) -> None:
    claim_sha = _require_sha256(expected_sha256, "production_claim_sha256")
    marker = root / _PRIVATE_CLAIM_NAME
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HistoryStagedPipelineError("production history run lacks a claim") from error
    if payload != {
        "schema_version": "carma_history_private_run_claim_v1",
        "status": "claimed_for_single_lineage_interruptible_fit",
        "production_claim_sha256": claim_sha,
    }:
        raise HistoryStagedPipelineError("production history claim lineage changed")


@contextmanager
def history_private_run_lock(root: str | Path):
    """Hold an OS-released, non-blocking exclusive lock for one production run."""

    directory = Path(root).resolve(strict=True)
    if not directory.is_dir() or _is_within(directory, _canonical_repository_root()):
        raise HistoryStagedPipelineError("history run lock root is unsafe")
    lock_path = directory / _PRIVATE_LOCK_NAME
    handle = lock_path.open("a+b")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            raise HistoryStagedPipelineError(
                "another history-fit process holds the private run lock"
            ) from error
        acquired = True
        yield
    finally:
        if acquired:
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


def _production_private_paths(root: Path) -> dict[str, Path]:
    return {
        "checkpoint": root / _PRIVATE_CHECKPOINT_NAME,
        "fit_outcome": root / _PRIVATE_FIT_OUTCOME_NAME,
        "fit_targets": root / _PRIVATE_FIT_TARGETS_NAME,
        "fit_receipt": root / _PRIVATE_FIT_RECEIPT_NAME,
        "complete_outcome": root / _PRIVATE_COMPLETE_OUTCOME_NAME,
        "complete_receipt": root / _PRIVATE_COMPLETE_RECEIPT_NAME,
    }


def _validate_production_private_layout(
    *,
    checkpoint_root: str | Path,
    fit_outcome: str | Path,
    fit_targets: str | Path,
    fit_receipt: str | Path,
    complete_outcome: str | Path | None = None,
    complete_receipt: str | Path | None = None,
) -> Path:
    checkpoint = Path(checkpoint_root).resolve()
    root = checkpoint.parent
    repository = _canonical_repository_root()
    if _is_within(root, repository):
        raise HistoryStagedPipelineError(
            "history private artifacts must be outside the repository"
        )
    expected = _production_private_paths(root)
    observed = {
        "checkpoint": checkpoint,
        "fit_outcome": Path(fit_outcome).resolve(),
        "fit_targets": Path(fit_targets).resolve(),
        "fit_receipt": Path(fit_receipt).resolve(),
    }
    if complete_outcome is not None:
        observed["complete_outcome"] = Path(complete_outcome).resolve()
    if complete_receipt is not None:
        observed["complete_receipt"] = Path(complete_receipt).resolve()
    if observed != {name: expected[name] for name in observed}:
        raise HistoryStagedPipelineError(
            "history private artifacts must use one canonical private root"
        )
    if len(set(observed.values())) != len(observed):
        raise HistoryStagedPipelineError("history private output paths alias")
    return root


def _publish_temporary_no_clobber(temporary: Path, destination: Path) -> None:
    """Atomically publish a complete file without ever replacing a peer.

    ``os.replace`` is intentionally not used: two producers racing after an
    ``exists`` check could otherwise overwrite one another.  A same-directory
    hard link has O_EXCL-like destination semantics on both NTFS and POSIX.
    """

    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise FileExistsError(
            f"write-once output already exists: {destination.name}"
        ) from None
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _temporary_path(parent: Path, destination_name: str) -> tuple[int, Path]:
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        dir=parent,
        prefix=f".{destination_name}.",
        suffix=".tmp",
    )
    return descriptor, Path(raw)


def _write_npz_once(path: Path, values: Mapping[str, np.ndarray]) -> str:
    if path.suffix.lower() != ".npz":
        raise HistoryStagedPipelineError("private artifact must be an NPZ file")
    descriptor, temporary = _temporary_path(path.parent, path.name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temporary_no_clobber(temporary, path)
    except BaseException:
        # ``os.fdopen`` owns and closes ``descriptor`` even when publication
        # loses a race.  Closing the stale integer again can close an unrelated
        # descriptor that another thread has since opened (observed on Windows
        # while the winning thread hashes the destination).
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return _file_sha256(path)


def _write_json_once(path: Path, payload: Mapping[str, object]) -> str:
    if path.suffix.lower() != ".json":
        raise HistoryStagedPipelineError("public receipt must be JSON")
    _validate_aggregate_producer_receipt(payload)
    serialized = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = _temporary_path(path.parent, path.name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temporary_no_clobber(temporary, path)
    except BaseException:
        # The ``fdopen`` context has already closed the descriptor; never
        # double-close a descriptor number that the OS may have reused.
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return _file_sha256(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() != ".npz":
        raise HistoryStagedPipelineError("private artifact must be NPZ")
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise HistoryStagedPipelineError(
            f"cannot read private history artifact: {error}"
        ) from error


def _integer_vector(
    value: np.ndarray, field: str, *, unique: bool = False
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise HistoryStagedPipelineError(f"{field} must be a one-dimensional integer array")
    result = array.astype(np.int64, copy=True)
    if np.any(result < 0):
        raise HistoryStagedPipelineError(f"{field} contains a negative value")
    if unique and len(set(result.tolist())) != len(result):
        raise HistoryStagedPipelineError(f"{field} must contain unique values")
    return result


def _single_int(value: np.ndarray, field: str) -> int:
    array = np.asarray(value)
    if array.size != 1 or not np.issubdtype(array.dtype, np.integer):
        raise HistoryStagedPipelineError(f"{field} must contain one integer")
    return int(array.reshape(()))


def _probability(value: np.ndarray, shape: tuple[int, ...], field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.floating):
        raise HistoryStagedPipelineError(f"{field} is not probability-shaped")
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise HistoryStagedPipelineError(f"{field} contains invalid probabilities")
    if array.size and not np.allclose(
        array.sum(axis=-1), 1.0, rtol=1.0e-5, atol=1.0e-6
    ):
        raise HistoryStagedPipelineError(f"{field} rows do not sum to one")
    return array.astype(np.float32, copy=True)


def _strict_histories(
    groups: np.ndarray, speakers: np.ndarray, turns: np.ndarray
) -> tuple[tuple[int, ...], ...]:
    groups = np.asarray(groups).astype(str)
    speakers = np.asarray(speakers).astype(str)
    turns = np.asarray(turns)
    rows = len(groups)
    if (
        speakers.shape != (rows,)
        or turns.shape != (rows,)
        or not np.issubdtype(turns.dtype, np.integer)
    ):
        raise HistoryStagedPipelineError("history identity arrays are not row-aligned")
    result: list[tuple[int, ...]] = []
    for query in range(rows):
        candidates = np.flatnonzero(
            (groups == groups[query])
            & (speakers == speakers[query])
            & (turns < turns[query])
        )
        result.append(
            tuple(sorted(candidates.tolist(), key=lambda row: (int(turns[row]), row)))
        )
    return tuple(result)


def validate_strict_past_histories(
    *,
    groups: np.ndarray,
    speakers: np.ndarray,
    turns: np.ndarray,
    histories: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Require the complete same-group, same-speaker, strictly-past history."""

    expected = _strict_histories(groups, speakers, turns)
    try:
        observed = tuple(tuple(int(value) for value in row) for row in histories)
    except (TypeError, ValueError) as error:
        raise HistoryStagedPipelineError("history contains a non-integral row") from error
    if len(observed) != len(expected):
        raise HistoryStagedPipelineError("histories are not row-aligned")
    for query, (left, right) in enumerate(zip(observed, expected, strict=True)):
        if left != right:
            raise HistoryStagedPipelineError(
                f"history for row {query} is not complete same-group/speaker strict past"
            )
    return expected


def _speaker_token(dataset: str, value: object) -> str:
    if dataset == "MELD":
        try:
            return str(int(str(value)))
        except ValueError as error:
            raise HistoryStagedPipelineError("MELD speaker token is not integral") from error
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
        raise HistoryStagedPipelineError("fit-only speaker vocabulary exceeds model config")
    mapping_sha = _canonical_sha256(
        {"oov": 0, "fit_mapping": [[value, mapping[value]] for value in ordered]}
    )
    return mapping, mapping_sha


def _speaker_identity(dataset: str, value: object) -> str:
    token = _speaker_token(dataset, value)
    return hashlib.sha256(f"{dataset}\x1fspeaker\x1f{token}".encode("utf-8")).hexdigest()


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
    protocol_row_ids: np.ndarray,
    histories: Sequence[Sequence[int]],
    role: str,
    model_config: CausalBackboneConfig,
    fit_speaker_mapping: Mapping[str, int],
    speaker_mapping_sha256: str,
    label_access_mode: str,
) -> OpenRoleCorpus:
    rows = len(texts)
    normalized_histories = validate_strict_past_histories(
        groups=np.asarray(groups),
        speakers=np.asarray(speakers),
        turns=np.asarray(turns),
        histories=histories,
    )
    speaker_tokens = np.asarray(speakers).astype(str)
    corpus = OpenRoleCorpus(
        keys=np.asarray(
            [
                hashlib.sha256(
                    f"{dataset}\x1fhistory-staged\x1f{role}\x1f{int(protocol)}".encode(
                        "utf-8"
                    )
                ).hexdigest()
                for protocol in np.asarray(protocol_row_ids)
            ]
        ),
        texts=tuple(str(value) for value in texts),
        audio=np.asarray(audio, dtype=np.float32).copy(),
        video=np.asarray(video, dtype=np.float32).copy(),
        labels=np.asarray(labels, dtype=np.int64).copy(),
        groups=np.asarray(groups).astype(str),
        roles=np.asarray([role] * rows),
        buckets=np.asarray([0 if role == FIT_ROLE else 65] * rows, dtype=np.int16),
        speaker_ids=np.asarray(
            [
                int(fit_speaker_mapping.get(_speaker_token(dataset, value), 0))
                for value in speaker_tokens
            ],
            dtype=np.int64,
        ),
        turn_ids=np.asarray(turns, dtype=np.int64).copy(),
        histories=normalized_histories,
        protocol_row_ids=np.asarray(protocol_row_ids, dtype=np.int64).copy(),
        speaker_identity=np.asarray(
            [_speaker_identity(dataset, value) for value in speaker_tokens]
        ),
        speaker_mapping_sha256=speaker_mapping_sha256,
        label_access_mode=label_access_mode,
    )
    corpus.validate(model_config)
    return corpus


def _fit_corpus_from_view(
    fit: FitRoleView,
    *,
    model_config: CausalBackboneConfig,
    heldout_indices: np.ndarray | None = None,
    speaker_reference_indices: np.ndarray | None = None,
) -> OpenRoleCorpus:
    validate_strict_past_histories(
        groups=fit.groups,
        speakers=fit.speakers,
        turns=fit.turns,
        histories=fit.histories,
    )
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
        raise HistoryStagedPipelineError("fit speaker reference rows are invalid")
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
        protocol_row_ids=fit.protocol_row_ids,
        histories=fit.histories,
        role=FIT_ROLE,
        model_config=model_config,
        fit_speaker_mapping=mapping,
        speaker_mapping_sha256=mapping_sha,
        label_access_mode="fit_train_labels_only_outer_heldout_labels_physically_absent",
    )


def _selection_corpus_from_view(
    selection: SelectionFeatureView,
    *,
    fit: FitRoleView,
    model_config: CausalBackboneConfig,
    fit_speaker_reference_indices: np.ndarray,
) -> OpenRoleCorpus:
    if selection.labels_materialized:
        raise HistoryStagedPipelineError("selection labels entered complete-selection")
    reference = np.asarray(fit_speaker_reference_indices, dtype=np.int64)
    if (
        reference.ndim != 1
        or not len(reference)
        or np.any((reference < 0) | (reference >= fit.rows))
        or len(np.unique(reference)) != len(reference)
    ):
        raise HistoryStagedPipelineError("selection speaker reference rows are invalid")
    mapping, mapping_sha = _fit_speaker_mapping(
        fit.dataset,
        np.asarray(fit.speakers)[reference],
        num_speakers=model_config.num_speakers,
    )
    histories = _strict_histories(selection.groups, selection.speakers, selection.turns)
    return _make_corpus(
        dataset=selection.dataset,
        texts=selection.texts,
        audio=selection.audio,
        video=selection.video,
        labels=np.zeros(len(selection.texts), dtype=np.int64),
        groups=selection.groups,
        speakers=selection.speakers,
        turns=selection.turns,
        protocol_row_ids=selection.protocol_row_ids,
        histories=histories,
        role=SELECTION_ROLE,
        model_config=model_config,
        fit_speaker_mapping=mapping,
        speaker_mapping_sha256=mapping_sha,
        label_access_mode="selection_features_only_zero_placeholder_never_scored",
    )


def _cluster_codes(groups: np.ndarray) -> np.ndarray:
    tokens = np.asarray(groups).astype(str)
    ordered = {value: index for index, value in enumerate(sorted(set(tokens.tolist())))}
    return np.asarray([ordered[value] for value in tokens], dtype=np.int32)


def _config_sha256(value: object) -> str:
    return _canonical_sha256(asdict(value))


def _live_fit_array_hashes(fit: FitRoleView) -> dict[str, str]:
    """Recompute the fit contract from live arrays, never cached hash fields."""

    rows = int(fit.rows)
    arrays: dict[str, np.ndarray] = {
        "texts": np.asarray(fit.texts).astype(str),
        "audio": np.asarray(fit.audio),
        "video": np.asarray(fit.video),
        "labels": np.asarray(fit.labels),
        "groups": np.asarray(fit.groups).astype(str),
        "speakers": np.asarray(fit.speakers).astype(str),
        "turns": np.asarray(fit.turns),
        "protocol_row_ids": np.asarray(fit.protocol_row_ids),
    }
    if arrays["texts"].shape != (rows,):
        raise HistoryStagedPipelineError("live fit text rows changed")
    if (
        arrays["audio"].ndim != 2
        or arrays["video"].ndim != 2
        or arrays["audio"].shape[0] != rows
        or arrays["video"].shape[0] != rows
        or arrays["audio"].dtype != np.float32
        or arrays["video"].dtype != np.float32
        or not np.isfinite(arrays["audio"]).all()
        or not np.isfinite(arrays["video"]).all()
    ):
        raise HistoryStagedPipelineError("live fit modality arrays changed")
    for name in ("labels", "turns", "protocol_row_ids"):
        value = arrays[name]
        if value.shape != (rows,) or not np.issubdtype(value.dtype, np.integer):
            raise HistoryStagedPipelineError(f"live fit {name} array changed")
    for name in ("groups", "speakers"):
        if arrays[name].shape != (rows,):
            raise HistoryStagedPipelineError(f"live fit {name} array changed")
    if (
        np.any(arrays["labels"] < 0)
        or np.any(arrays["labels"] >= len(fit.label_order))
        or np.any(arrays["turns"] < 0)
        or np.any(arrays["protocol_row_ids"] < 0)
        or len(set(arrays["protocol_row_ids"].astype(np.int64).tolist())) != rows
    ):
        raise HistoryStagedPipelineError("live fit integer contract changed")
    histories = validate_strict_past_histories(
        groups=arrays["groups"],
        speakers=arrays["speakers"],
        turns=arrays["turns"],
        histories=fit.histories,
    )
    result = {name: _array_sha256(value) for name, value in arrays.items()}
    result["histories"] = _canonical_sha256([list(value) for value in histories])
    declared = dict(sorted((str(name), str(value)) for name, value in fit.array_hashes.items()))
    if dict(sorted(result.items())) != declared:
        raise HistoryStagedPipelineError("live fit arrays differ from cached contract")
    return dict(sorted(result.items()))


def _named_live_file_hashes(
    values: Mapping[str, str | Path], field: str
) -> dict[str, str]:
    if not isinstance(values, Mapping) or not values:
        raise HistoryStagedPipelineError(f"{field} must be a non-empty mapping")
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
            raise HistoryStagedPipelineError(
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
            raise HistoryStagedPipelineError(f"{field} contains an unsafe name")
        casefolded.add(name.casefold())
        result[name] = _file_sha256(Path(raw_path).resolve(strict=True))
    return dict(sorted(result.items()))


def _live_lineage(
    *,
    config_paths: Mapping[str, str | Path],
    code_paths: Mapping[str, str | Path],
    environment: Mapping[str, object],
) -> tuple[dict[str, str], dict[str, str], str, str]:
    if not isinstance(environment, Mapping) or not environment:
        raise HistoryStagedPipelineError("runtime environment payload must not be empty")
    configs = _named_live_file_hashes(config_paths, "config_paths")
    code = _named_live_file_hashes(code_paths, "code_paths")
    runtime = _canonical_sha256(dict(environment))
    combined = _canonical_sha256(
        {
            "config_sha256": configs,
            "code_sha256": code,
            "runtime_environment_sha256": runtime,
        }
    )
    return configs, code, runtime, combined


def _normalized_hash_mapping(
    values: Mapping[str, str], field: str
) -> dict[str, str]:
    if not isinstance(values, Mapping) or not values:
        raise HistoryStagedPipelineError(f"{field} must be a non-empty mapping")
    result: dict[str, str] = {}
    casefolded: set[str] = set()
    source_keys = field == "code_sha256"
    for name, digest in values.items():
        try:
            token = (
                _canonical_production_source_key(name)
                if source_keys
                else str(name)
            )
        except ValueError as error:
            raise HistoryStagedPipelineError(
                f"{field} contains an unsafe name"
            ) from error
        if (
            (
                not source_keys
                and (
                    not token
                    or not token[0].isalnum()
                    or len(token) > 128
                    or any(
                        not (character.isalnum() or character in "_.-")
                        for character in token
                    )
                )
            )
            or token.casefold() in casefolded
        ):
            raise HistoryStagedPipelineError(f"{field} contains an unsafe name")
        casefolded.add(token.casefold())
        result[token] = _require_sha256(digest, f"{field}.{token}")
    return dict(sorted(result.items()))


def _verify_preflight_and_fit_map(
    *,
    fit: FitRoleView,
    fit_map: FitProtocolMap,
    fit_preflight_receipt_path: str | Path,
    expected_fit_preflight_receipt_sha256: str,
    config_sha256: Mapping[str, str],
    code_sha256: Mapping[str, str],
    runtime_environment_sha256: str,
) -> tuple[dict[str, object], str]:
    preflight_sha = _require_sha256(
        expected_fit_preflight_receipt_sha256,
        "expected_fit_preflight_receipt_sha256",
    )
    receipt_path = Path(fit_preflight_receipt_path)
    if _file_sha256(receipt_path) != preflight_sha:
        raise HistoryStagedPipelineError("fit preflight receipt file hash changed")
    receipt = _load_receipt(receipt_path)
    validate_fit_receipt(receipt)
    if receipt.get("dataset") != fit.dataset:
        raise HistoryStagedPipelineError("fit dataset differs from preflight receipt")
    fit_contract = receipt.get("fit_contract")
    lineage = receipt.get("lineage")
    if not isinstance(fit_contract, Mapping) or not isinstance(lineage, Mapping):
        raise HistoryStagedPipelineError("fit preflight receipt lacks contract lineage")
    live_arrays = _live_fit_array_hashes(fit)
    expected_arrays = _canonical_sha256(live_arrays)
    expected_protocol = _canonical_sha256(
        [[int(index), int(value)] for index, value in enumerate(fit.protocol_row_ids)]
    )
    if (
        fit_contract.get("fit_arrays_contract_sha256") != fit.contract_sha256
        or fit_contract.get("fit_array_manifest_sha256") != expected_arrays
        or fit_contract.get("protocol_index_mapping_sha256") != expected_protocol
        or int(fit_contract.get("rows", -1)) != fit.rows
    ):
        raise HistoryStagedPipelineError("fit material differs from preflight receipt")
    observed_config = _normalized_hash_mapping(config_sha256, "config_sha256")
    observed_code = _normalized_hash_mapping(code_sha256, "code_sha256")
    observed_runtime = _require_sha256(
        runtime_environment_sha256, "runtime_environment_sha256"
    )
    if (
        dict(cast(Mapping[str, str], lineage.get("config_sha256", {})))
        != observed_config
        or dict(cast(Mapping[str, str], lineage.get("code_sha256", {})))
        != observed_code
        or lineage.get("runtime_environment_sha256") != observed_runtime
    ):
        raise HistoryStagedPipelineError("config/code/runtime lineage changed")
    _verify_fit_protocol_map_file(
        fit_map,
        receipt_path=receipt_path,
        expected_receipt_sha256=preflight_sha,
    )
    if (
        fit_map.receipt_sha256 != preflight_sha
        or fit_map.dataset != fit.dataset
        or fit_map.fit_arrays_contract_sha256 != fit.contract_sha256
        or not np.array_equal(fit_map.protocol_row_ids, fit.protocol_row_ids)
    ):
        raise HistoryStagedPipelineError("fit protocol map differs from fit material")
    validate_strict_past_histories(
        groups=fit.groups,
        speakers=fit.speakers,
        turns=fit.turns,
        histories=fit.histories,
    )
    return receipt, preflight_sha


def _fold_assignment(
    fit: FitRoleView,
    *,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
) -> np.ndarray:
    corpus = _fit_corpus_from_view(fit, model_config=model_config)
    result = np.full((len(EXPECTED_SEEDS), fit.rows), -1, dtype=np.int32)
    for seed_index, seed in enumerate(EXPECTED_SEEDS):
        splits = make_crossfit_splits(
            corpus,
            outer_folds=int(run_config.outer_folds),
            validation_fraction=float(run_config.inner_validation_fraction),
            seed=int(seed),
        )
        for split in splits:
            result[seed_index, split.outer_heldout_indices] = int(split.fold)
    if np.any(result < 0):
        raise HistoryStagedPipelineError("history fold assignment is incomplete")
    return result


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
        raise HistoryStagedPipelineError(
            "outer history training fold has too few groups for early stopping"
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
    mask = np.asarray(
        [str(value) in validation_groups for value in corpus.groups[train]], dtype=bool
    )
    inner_validation = train[mask]
    inner_train = train[~mask]
    partitions = (inner_train, inner_validation, held)
    if any(not len(value) for value in partitions):
        raise HistoryStagedPipelineError("history split contains an empty partition")
    group_sets = [set(corpus.groups[value].astype(str)) for value in partitions]
    if any(
        group_sets[left] & group_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise HistoryStagedPipelineError("history split shares a group")
    return CrossfitSplit(int(fold), inner_train, inner_validation, held)


def _validate_task_against_histories(
    task: BidirectionalCoalitionTask,
    histories: Sequence[Sequence[int]],
    *,
    allowed_queries: set[int] | None = None,
) -> None:
    query = int(task.query_index)
    candidate = int(task.candidate_index)
    if not 0 <= query < len(histories):
        raise HistoryStagedPipelineError("utility task query is outside the role")
    if allowed_queries is not None and query not in allowed_queries:
        raise HistoryStagedPipelineError("utility task query entered the wrong fold")
    allowed = set(int(value) for value in histories[query])
    contexts = (
        tuple(int(value) for value in task.addition_context),
        tuple(int(value) for value in task.deletion_context),
    )
    if candidate not in allowed or not set(contexts[0]).issubset(allowed) or not set(
        contexts[1]
    ).issubset(allowed):
        raise HistoryStagedPipelineError("utility task contains a non-history row")
    if candidate in contexts[0] or candidate not in contexts[1]:
        raise HistoryStagedPipelineError("utility task candidate semantics changed")
    if set(contexts[1]) == set(contexts[0]) | {candidate}:
        raise HistoryStagedPipelineError("utility task collapsed to one-way marginal utility")


def _tasks_for_role(
    corpus: OpenRoleCorpus,
    utility_config: UtilitySamplingConfig,
) -> tuple[BidirectionalCoalitionTask, ...]:
    tasks = tuple(sample_corpus_bidirectional_tasks(corpus, utility_config))
    for task in tasks:
        _validate_task_against_histories(task, corpus.histories)
    return tasks


def _corpus_from_fold_request(
    request: HistoryFitFoldRequest,
    *,
    model_config: CausalBackboneConfig,
) -> OpenRoleCorpus:
    if request.heldout_labels_materialized or request.heldout_targets_materialized:
        raise HistoryStagedPipelineError("heldout outcome entered history fitting")
    all_indices = np.concatenate([request.train_indices, request.heldout_indices])
    rows = len(all_indices)
    if len(set(all_indices.tolist())) != rows or set(all_indices.tolist()) != set(
        range(rows)
    ):
        raise HistoryStagedPipelineError("fold request is not a complete row partition")
    texts = [""] * rows
    audio = np.empty((rows, request.train_audio.shape[1]), dtype=np.float32)
    video = np.empty((rows, request.train_video.shape[1]), dtype=np.float32)
    labels = np.zeros(rows, dtype=np.int64)
    groups = np.empty(rows, dtype=object)
    speakers = np.empty(rows, dtype=object)
    turns = np.empty(rows, dtype=np.int64)
    protocol = np.empty(rows, dtype=np.int64)
    histories: list[tuple[int, ...] | None] = [None] * rows
    partitions = (
        (
            request.train_indices,
            request.train_texts,
            request.train_audio,
            request.train_video,
            request.train_group_tokens,
            request.train_speaker_tokens,
            request.train_turns,
            request.train_protocol_row_ids,
            request.train_histories,
        ),
        (
            request.heldout_indices,
            request.heldout_texts,
            request.heldout_audio,
            request.heldout_video,
            request.heldout_group_tokens,
            request.heldout_speaker_tokens,
            request.heldout_turns,
            request.heldout_protocol_row_ids,
            request.heldout_histories,
        ),
    )
    for (
        indices,
        source_texts,
        source_audio,
        source_video,
        source_groups,
        source_speakers,
        source_turns,
        source_protocol,
        source_histories,
    ) in partitions:
        aligned = (
            len(source_texts),
            len(source_audio),
            len(source_video),
            len(source_groups),
            len(source_speakers),
            len(source_turns),
            len(source_protocol),
            len(source_histories),
        )
        if any(value != len(indices) for value in aligned):
            raise HistoryStagedPipelineError("fold request feature arrays are misaligned")
        for local, target in enumerate(indices):
            target_int = int(target)
            texts[target_int] = str(source_texts[local])
            audio[target_int] = source_audio[local]
            video[target_int] = source_video[local]
            groups[target_int] = source_groups[local]
            speakers[target_int] = source_speakers[local]
            turns[target_int] = int(source_turns[local])
            protocol[target_int] = int(source_protocol[local])
            histories[target_int] = tuple(int(value) for value in source_histories[local])
    labels[request.train_indices] = np.asarray(request.train_labels, dtype=np.int64)
    if any(value is None for value in histories):
        raise HistoryStagedPipelineError("fold request histories are incomplete")
    mapping, mapping_sha = _fit_speaker_mapping(
        request.dataset,
        request.train_speaker_tokens,
        num_speakers=model_config.num_speakers,
    )
    corpus = _make_corpus(
        dataset=request.dataset,
        texts=texts,
        audio=audio,
        video=video,
        labels=labels,
        groups=np.asarray(groups),
        speakers=np.asarray(speakers),
        turns=turns,
        protocol_row_ids=protocol,
        histories=cast(Sequence[Sequence[int]], histories),
        role=FIT_ROLE,
        model_config=model_config,
        fit_speaker_mapping=mapping,
        speaker_mapping_sha256=mapping_sha,
        label_access_mode="fit_train_labels_only_outer_heldout_outcomes_absent",
    )
    allowed = set(int(value) for value in request.heldout_indices)
    for task in request.heldout_tasks:
        _validate_task_against_histories(
            task, corpus.histories, allowed_queries=allowed
        )
    return corpus


def make_real_history_fit_fold_callback(
    *,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    device: torch.device,
) -> HistoryFitFoldCallback:
    """Return the real history-aware trainer without capturing heldout labels."""

    def callback(request: HistoryFitFoldRequest) -> HistoryFitFoldOutput:
        corpus = _corpus_from_fold_request(request, model_config=model_config)
        split = _split_from_outer_partition(
            corpus,
            outer_train=request.train_indices,
            heldout=request.heldout_indices,
            validation_fraction=run_config.inner_validation_fraction,
            seed=request.seed,
            fold=request.fold,
        )
        trained = train_one_fold_seed(
            corpus,
            split,
            model_config=model_config,
            run_config=run_config,
            seed=request.seed,
            source_identity=request.source_identity_sha256,
            checkpoint_root=request.checkpoint_root,
            device=device,
            require_complete_checkpoint=False,
        )
        endpoint = predict_current_and_all_history(
            trained.model,
            corpus,
            trained.text_features,
            request.heldout_indices,
            device=device,
            batch_size=run_config.inference_batch_size,
            max_history_items=run_config.max_history_items,
        )
        utility = predict_utility_contexts(
            trained.model,
            corpus,
            trained.text_features,
            request.heldout_tasks,
            device=device,
            batch_size=run_config.inference_batch_size,
            max_history_items=run_config.max_history_items,
        )
        return HistoryFitFoldOutput(
            np.asarray(endpoint, dtype=np.float32),
            np.asarray(utility, dtype=np.float32),
            request.source_identity_sha256,
        )

    return callback


def _task_artifact_values(
    prefix: str, tasks: Sequence[BidirectionalCoalitionTask]
) -> dict[str, np.ndarray]:
    encoded = _encode_task_contexts(tasks)
    return {
        f"{prefix}_task_query_indices": np.asarray(
            encoded["query_indices"], dtype=np.int64
        ),
        f"{prefix}_task_candidate_indices": np.asarray(
            encoded["candidate_indices"], dtype=np.int64
        ),
        f"{prefix}_task_s_indptr": np.asarray(encoded["s_indptr"], dtype=np.int64),
        f"{prefix}_task_s_indices": np.asarray(encoded["s_indices"], dtype=np.int64),
        f"{prefix}_task_t_indptr": np.asarray(encoded["t_indptr"], dtype=np.int64),
        f"{prefix}_task_t_indices": np.asarray(encoded["t_indices"], dtype=np.int64),
        f"{prefix}_task_sha256": np.asarray(_task_sha256(tasks)),
    }


def _csr_contexts(
    indptr_value: np.ndarray,
    indices_value: np.ndarray,
    *,
    rows: int,
    role_rows: int,
    field: str,
) -> tuple[tuple[int, ...], ...]:
    indptr = _integer_vector(indptr_value, f"{field}_indptr")
    indices = _integer_vector(indices_value, f"{field}_indices")
    if (
        indptr.shape != (rows + 1,)
        or int(indptr[0]) != 0
        or int(indptr[-1]) != len(indices)
        or np.any(np.diff(indptr) < 0)
        or np.any(indices >= role_rows)
    ):
        raise HistoryStagedPipelineError(f"{field} CSR encoding changed")
    result: list[tuple[int, ...]] = []
    for row in range(rows):
        values = tuple(int(value) for value in indices[indptr[row] : indptr[row + 1]])
        if len(values) != len(set(values)):
            raise HistoryStagedPipelineError(f"{field} contains duplicate context rows")
        result.append(values)
    return tuple(result)


def _decode_tasks(
    values: Mapping[str, np.ndarray],
    prefix: str,
    *,
    role_rows: int,
    histories: Sequence[Sequence[int]] | None = None,
) -> EncodedHistoryTasks:
    query = _integer_vector(
        values[f"{prefix}_task_query_indices"], f"{prefix}_task_query_indices"
    )
    candidate = _integer_vector(
        values[f"{prefix}_task_candidate_indices"],
        f"{prefix}_task_candidate_indices",
    )
    if query.shape != candidate.shape or np.any(query >= role_rows) or np.any(
        candidate >= role_rows
    ):
        raise HistoryStagedPipelineError(f"{prefix} task alignment changed")
    addition = _csr_contexts(
        values[f"{prefix}_task_s_indptr"],
        values[f"{prefix}_task_s_indices"],
        rows=len(query),
        role_rows=role_rows,
        field=f"{prefix}_task_s",
    )
    deletion = _csr_contexts(
        values[f"{prefix}_task_t_indptr"],
        values[f"{prefix}_task_t_indices"],
        rows=len(query),
        role_rows=role_rows,
        field=f"{prefix}_task_t",
    )
    tasks: list[BidirectionalCoalitionTask] = []
    try:
        for q, c, s, t in zip(query, candidate, addition, deletion, strict=True):
            task = BidirectionalCoalitionTask(
                query_index=int(q),
                candidate_index=int(c),
                addition_context=s,
                deletion_context=t,
            )
            if histories is not None:
                _validate_task_against_histories(task, histories)
            tasks.append(task)
    except ValueError as error:
        raise HistoryStagedPipelineError(f"{prefix} task semantics changed") from error
    declared = _require_sha256(
        _single_text(values[f"{prefix}_task_sha256"], f"{prefix}_task_sha256"),
        f"{prefix}_task_sha256",
    )
    if declared != _task_sha256(tasks):
        raise HistoryStagedPipelineError(f"{prefix} task hash changed")
    return EncodedHistoryTasks(query, candidate, addition, deletion, declared)


_FIT_OUTCOME_KEYS = frozenset(
    {
        "schema_version",
        "dataset",
        "dataset_label_order",
        "seeds",
        "outer_folds",
        "fit_protocol_row_ids",
        "fit_query_indices",
        "fit_cluster_codes",
        "fit_histories_sha256",
        "fit_task_query_indices",
        "fit_task_candidate_indices",
        "fit_task_s_indptr",
        "fit_task_s_indices",
        "fit_task_t_indptr",
        "fit_task_t_indices",
        "fit_task_sha256",
        "fit_endpoint_probability_oof",
        "fit_utility_probability_oof",
        "fit_fold_by_seed_query",
        "source_identity_sha256",
        "checkpoint_manifest_sha256",
        "model_config_sha256",
        "run_config_sha256",
        "utility_config_sha256",
        "matrix_fit_endpoint_probability_oof_sha256",
        "matrix_fit_utility_probability_oof_sha256",
        "fold_assignment_sha256",
    }
)


def _fit_outcome_view_from_values(
    values: Mapping[str, np.ndarray],
    *,
    fit: FitRoleView,
    checkpoint_manifest: CheckpointManifest,
    artifact_sha256: str,
) -> _HistoryFitOutcomeView:
    if set(values) != set(_FIT_OUTCOME_KEYS):
        missing = sorted(_FIT_OUTCOME_KEYS - set(values))
        unknown = sorted(set(values) - _FIT_OUTCOME_KEYS)
        raise HistoryStagedPipelineError(
            f"fit outcome schema changed: missing={missing}, unknown={unknown}"
        )
    if (
        _single_text(values["schema_version"], "schema_version")
        != HISTORY_FIT_OUTCOME_SCHEMA
        or _single_text(values["dataset"], "dataset") != fit.dataset
    ):
        raise HistoryStagedPipelineError("fit outcome identity changed")
    label_order = tuple(
        str(value) for value in np.asarray(values["dataset_label_order"]).reshape(-1)
    )
    seeds = tuple(
        int(value)
        for value in _integer_vector(values["seeds"], "seeds", unique=True)
    )
    if label_order != fit.label_order or seeds != EXPECTED_SEEDS:
        raise HistoryStagedPipelineError("fit outcome label/seed contract changed")
    outer_folds = _single_int(values["outer_folds"], "outer_folds")
    if outer_folds != checkpoint_manifest.outer_folds:
        raise HistoryStagedPipelineError("fit outcome outer-fold count changed")
    protocol = _integer_vector(
        values["fit_protocol_row_ids"], "fit_protocol_row_ids", unique=True
    )
    query = _integer_vector(
        values["fit_query_indices"], "fit_query_indices", unique=True
    )
    cluster = _integer_vector(values["fit_cluster_codes"], "fit_cluster_codes")
    if (
        not np.array_equal(protocol, fit.protocol_row_ids)
        or not np.array_equal(query, np.arange(fit.rows, dtype=np.int64))
        or not np.array_equal(cluster, _cluster_codes(fit.groups))
    ):
        raise HistoryStagedPipelineError("fit outcome row alignment changed")
    histories = validate_strict_past_histories(
        groups=fit.groups,
        speakers=fit.speakers,
        turns=fit.turns,
        histories=fit.histories,
    )
    histories_sha = _require_sha256(
        _single_text(values["fit_histories_sha256"], "fit_histories_sha256"),
        "fit_histories_sha256",
    )
    if histories_sha != _canonical_sha256([list(row) for row in histories]):
        raise HistoryStagedPipelineError("fit histories hash changed")
    tasks = _decode_tasks(values, "fit", role_rows=fit.rows, histories=histories)
    endpoint = _probability(
        values["fit_endpoint_probability_oof"],
        (len(seeds), fit.rows, len(ENDPOINT_CONTEXT_NAMES), len(label_order)),
        "fit_endpoint_probability_oof",
    )
    utility = _probability(
        values["fit_utility_probability_oof"],
        (len(seeds), len(tasks), len(UTILITY_CONTEXT_NAMES), len(label_order)),
        "fit_utility_probability_oof",
    )
    folds = np.asarray(values["fit_fold_by_seed_query"])
    if (
        folds.shape != (len(seeds), fit.rows)
        or not np.issubdtype(folds.dtype, np.integer)
        or np.any((folds < 0) | (folds >= outer_folds))
    ):
        raise HistoryStagedPipelineError("fit fold assignment changed")
    folds = folds.astype(np.int32, copy=True)
    groups = np.asarray(fit.groups).astype(str)
    for seed_index in range(len(seeds)):
        if set(np.unique(folds[seed_index]).tolist()) != set(range(outer_folds)):
            raise HistoryStagedPipelineError("fit seed does not cover every outer fold")
        for group in np.unique(groups):
            if len(np.unique(folds[seed_index, groups == group])) != 1:
                raise HistoryStagedPipelineError("fit fold assignment split a group")
    matrices = {
        "fit_endpoint_probability_oof": endpoint,
        "fit_utility_probability_oof": utility,
        "fold_assignment": folds,
    }
    fields = {
        "fit_endpoint_probability_oof": "matrix_fit_endpoint_probability_oof_sha256",
        "fit_utility_probability_oof": "matrix_fit_utility_probability_oof_sha256",
        "fold_assignment": "fold_assignment_sha256",
    }
    for name, array in matrices.items():
        declared = _require_sha256(
            _single_text(values[fields[name]], fields[name]), fields[name]
        )
        if declared != _array_sha256(np.asarray(array)):
            raise HistoryStagedPipelineError(f"fit matrix hash changed: {name}")
    checkpoint_sha = _require_sha256(
        _single_text(values["checkpoint_manifest_sha256"], "checkpoint manifest"),
        "checkpoint_manifest_sha256",
    )
    if checkpoint_sha != checkpoint_manifest.manifest_sha256:
        raise HistoryStagedPipelineError("fit checkpoint manifest changed")
    return _HistoryFitOutcomeView(
        dataset=fit.dataset,
        label_order=label_order,
        seeds=seeds,
        protocol_row_ids=protocol,
        query_indices=query,
        cluster_codes=cluster.astype(np.int32, copy=True),
        histories_sha256=histories_sha,
        tasks=tasks,
        endpoint_probability_oof=endpoint,
        utility_probability_oof=utility,
        fold_by_seed_query=folds,
        source_identity_sha256=_require_sha256(
            _single_text(values["source_identity_sha256"], "source identity"),
            "source_identity_sha256",
        ),
        checkpoint_manifest_sha256=checkpoint_sha,
        model_config_sha256=_require_sha256(
            _single_text(values["model_config_sha256"], "model config"),
            "model_config_sha256",
        ),
        run_config_sha256=_require_sha256(
            _single_text(values["run_config_sha256"], "run config"),
            "run_config_sha256",
        ),
        utility_config_sha256=_require_sha256(
            _single_text(values["utility_config_sha256"], "utility config"),
            "utility_config_sha256",
        ),
        artifact_sha256=_require_sha256(artifact_sha256, "artifact_sha256"),
    )


def load_history_fit_outcome_view(
    path: str | Path,
    *,
    fit: FitRoleView,
    checkpoint_manifest: CheckpointManifest,
) -> _HistoryFitOutcomeView:
    artifact = Path(path)
    digest = _file_sha256(artifact)
    values = _load_npz(artifact)
    result = _fit_outcome_view_from_values(
        values,
        fit=fit,
        checkpoint_manifest=checkpoint_manifest,
        artifact_sha256=digest,
    )
    if _file_sha256(artifact) != digest:
        raise HistoryStagedPipelineError("fit outcome changed while validating")
    return result


_FIT_TARGET_KEYS = frozenset(
    {
        "schema_version",
        "dataset",
        "seeds",
        "fit_task_sha256",
        "fit_forward_utility",
        "fit_backward_utility",
        "fit_asymmetry",
        "fit_sign_agreement",
        "source_identity_sha256",
        "fit_outcome_artifact_sha256",
        "matrix_fit_forward_utility_sha256",
        "matrix_fit_backward_utility_sha256",
        "matrix_fit_asymmetry_sha256",
        "matrix_fit_sign_agreement_sha256",
    }
)


def load_history_fit_targets_view(
    path: str | Path,
    *,
    expected_fit_outcome_sha256: str | None = None,
    expected_source_identity_sha256: str | None = None,
    expected_task_sha256: str | None = None,
) -> HistoryFitTargetsView:
    artifact = Path(path)
    digest = _file_sha256(artifact)
    values = _load_npz(artifact)
    if set(values) != set(_FIT_TARGET_KEYS):
        raise HistoryStagedPipelineError("fit targets artifact schema changed")
    if _single_text(values["schema_version"], "schema_version") != HISTORY_FIT_TARGETS_SCHEMA:
        raise HistoryStagedPipelineError("fit targets artifact version changed")
    dataset = _single_text(values["dataset"], "dataset")
    seeds = tuple(
        int(value)
        for value in _integer_vector(values["seeds"], "seeds", unique=True)
    )
    if seeds != EXPECTED_SEEDS:
        raise HistoryStagedPipelineError("fit target seeds changed")
    task_sha = _require_sha256(
        _single_text(values["fit_task_sha256"], "fit_task_sha256"),
        "fit_task_sha256",
    )
    forward = np.asarray(values["fit_forward_utility"])
    backward = np.asarray(values["fit_backward_utility"])
    asymmetry = np.asarray(values["fit_asymmetry"])
    agreement = np.asarray(values["fit_sign_agreement"])
    expected_shape = (len(seeds), forward.shape[1] if forward.ndim == 2 else -1)
    if (
        forward.shape != expected_shape
        or backward.shape != expected_shape
        or asymmetry.shape != expected_shape
        or agreement.shape != expected_shape
        or not all(
            np.issubdtype(value.dtype, np.floating)
            for value in (forward, backward, asymmetry)
        )
        or agreement.dtype != np.bool_
        or not all(np.isfinite(value).all() for value in (forward, backward, asymmetry))
    ):
        raise HistoryStagedPipelineError("fit target arrays are not seed/task aligned")
    forward = forward.astype(np.float32, copy=True)
    backward = backward.astype(np.float32, copy=True)
    asymmetry = asymmetry.astype(np.float32, copy=True)
    agreement = agreement.astype(bool, copy=True)
    if not np.allclose(asymmetry, forward - backward, rtol=1.0e-5, atol=1.0e-6):
        raise HistoryStagedPipelineError("fit asymmetry differs from forward minus backward")
    if not np.array_equal(agreement, np.sign(forward) == np.sign(backward)):
        raise HistoryStagedPipelineError("fit sign agreement changed")
    matrices = {
        "fit_forward_utility": forward,
        "fit_backward_utility": backward,
        "fit_asymmetry": asymmetry,
        "fit_sign_agreement": agreement,
    }
    for name, array in matrices.items():
        field = f"matrix_{name}_sha256"
        if _require_sha256(_single_text(values[field], field), field) != _array_sha256(
            np.asarray(array)
        ):
            raise HistoryStagedPipelineError(f"fit target matrix hash changed: {name}")
    source_identity = _require_sha256(
        _single_text(values["source_identity_sha256"], "source identity"),
        "source_identity_sha256",
    )
    outcome_sha = _require_sha256(
        _single_text(values["fit_outcome_artifact_sha256"], "fit outcome hash"),
        "fit_outcome_artifact_sha256",
    )
    if expected_fit_outcome_sha256 is not None and outcome_sha != _require_sha256(
        expected_fit_outcome_sha256, "expected_fit_outcome_sha256"
    ):
        raise HistoryStagedPipelineError("fit targets point to another outcome artifact")
    if expected_source_identity_sha256 is not None and source_identity != _require_sha256(
        expected_source_identity_sha256, "expected_source_identity_sha256"
    ):
        raise HistoryStagedPipelineError("fit target source identity changed")
    if expected_task_sha256 is not None and task_sha != _require_sha256(
        expected_task_sha256, "expected_task_sha256"
    ):
        raise HistoryStagedPipelineError("fit target task identity changed")
    if _file_sha256(artifact) != digest:
        raise HistoryStagedPipelineError("fit targets changed while validating")
    return HistoryFitTargetsView(
        dataset=dataset,
        seeds=seeds,
        task_sha256=task_sha,
        forward_utility=forward,
        backward_utility=backward,
        asymmetry=asymmetry,
        sign_agreement=agreement,
        source_identity_sha256=source_identity,
        fit_outcome_artifact_sha256=outcome_sha,
        artifact_sha256=digest,
    )


def _fit_receipt_payload(
    *,
    dataset: str,
    preflight_sha256: str,
    fit_map: FitProtocolMap,
    config_sha256: Mapping[str, str],
    code_sha256: Mapping[str, str],
    runtime_environment_sha256: str,
    execution_environment_sha256: str,
    model_config_sha256: str,
    run_config_sha256: str,
    utility_config_sha256: str,
    source_identity_sha256: str,
    fold_assignment_sha256: str,
    task_sha256: str,
    checkpoint_manifest: CheckpointManifest,
    outcome_artifact_sha256: str,
    targets_artifact_sha256: str,
    fit_rows: int,
    task_count: int,
    trainer_mode: str,
    production_run_claim_sha256: str | None,
) -> dict[str, object]:
    if trainer_mode not in {PRODUCTION_TRAINER_MODE, SYNTHETIC_TRAINER_MODE}:
        raise HistoryStagedPipelineError("history trainer mode is invalid")
    production = trainer_mode == PRODUCTION_TRAINER_MODE
    receipt: dict[str, object] = {
        "schema_version": HISTORY_FIT_RECEIPT_SCHEMA,
        "status": PRODUCTION_FIT_STATUS if production else SYNTHETIC_FIT_STATUS,
        "dataset": dataset,
        "claim_boundary": (
            "Fit-role history-aware OOF probabilities and fit-only bidirectional "
            "utility targets; no model-selection payload or performance metric was "
            "consumed."
        ),
        "lineage": {
            "fit_preflight_receipt_sha256": preflight_sha256,
            "fit_protocol_map_sha256": fit_map.artifact_sha256,
            "config_sha256": dict(sorted(config_sha256.items())),
            "code_sha256": dict(sorted(code_sha256.items())),
            "runtime_environment_sha256": runtime_environment_sha256,
            "execution_environment_sha256": execution_environment_sha256,
            "model_config_sha256": model_config_sha256,
            "run_config_sha256": run_config_sha256,
            "utility_config_sha256": utility_config_sha256,
            "source_identity_sha256": source_identity_sha256,
            "fold_assignment_sha256": fold_assignment_sha256,
            "fit_task_sha256": task_sha256,
            "checkpoint_manifest_sha256": checkpoint_manifest.manifest_sha256,
            "private_fit_outcome_artifact_sha256": outcome_artifact_sha256,
            "private_fit_targets_artifact_sha256": targets_artifact_sha256,
            "production_run_claim_sha256": production_run_claim_sha256,
        },
        "training_contract": {
            "protocol": HISTORY_STAGED_PROTOCOL,
            "seeds": list(EXPECTED_SEEDS),
            "outer_folds": checkpoint_manifest.outer_folds,
            "checkpoint_file_count": len(checkpoint_manifest.records),
            "fit_query_count": int(fit_rows),
            "fit_task_count": int(task_count),
            "strict_same_group_same_speaker_past_history": True,
            "bidirectional_different_set_utility": True,
            "one_oof_endpoint_probability_per_seed_and_fit_query": True,
            "one_oof_utility_context_probability_per_seed_and_fit_task": True,
            "heldout_outcomes_materialized_in_fold_callback": False,
            "selection_payload_consumed": False,
            "performance_metric_computed": False,
            "trainer_mode": trainer_mode,
            "production_receipt": production,
        },
        "public_artifact_policy": {
            "aggregate_only": True,
            "contains_row_level_values": False,
            "contains_private_paths": False,
            "contains_probabilities_targets_or_performance": False,
        },
    }
    _validate_aggregate_producer_receipt(receipt)
    return receipt


def _load_history_fit_receipt(
    path: Path,
    *,
    expected_sha256: str,
    expected: Mapping[str, object],
) -> dict[str, object]:
    expected_sha = _require_sha256(expected_sha256, "fit_receipt_sha256")
    if _file_sha256(path) != expected_sha:
        raise HistoryStagedPipelineError("history fit receipt file hash changed")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HistoryStagedPipelineError(f"cannot read history fit receipt: {error}") from error
    if not isinstance(payload, dict):
        raise HistoryStagedPipelineError("history fit receipt root is not a mapping")
    if set(payload) != {
        "schema_version",
        "status",
        "dataset",
        "claim_boundary",
        "lineage",
        "training_contract",
        "public_artifact_policy",
    }:
        raise HistoryStagedPipelineError("history fit receipt schema changed")
    if payload.get("schema_version") != HISTORY_FIT_RECEIPT_SCHEMA or payload.get(
        "status"
    ) not in {PRODUCTION_FIT_STATUS, SYNTHETIC_FIT_STATUS}:
        raise HistoryStagedPipelineError("history fit receipt identity changed")
    lineage = payload.get("lineage")
    contract = payload.get("training_contract")
    public = payload.get("public_artifact_policy")
    if not isinstance(lineage, Mapping) or not isinstance(contract, Mapping):
        raise HistoryStagedPipelineError("history fit receipt lacks lineage/contract")
    changed = sorted(
        name for name, value in expected.items() if lineage.get(name) != value
    )
    if changed:
        raise HistoryStagedPipelineError(
            f"history fit receipt lineage changed: {changed}"
        )
    required_contract = {
        "strict_same_group_same_speaker_past_history": True,
        "bidirectional_different_set_utility": True,
        "heldout_outcomes_materialized_in_fold_callback": False,
        "selection_payload_consumed": False,
        "performance_metric_computed": False,
    }
    if any(contract.get(name) is not value for name, value in required_contract.items()):
        raise HistoryStagedPipelineError("history fit receipt isolation contract changed")
    trainer_mode = contract.get("trainer_mode")
    production = trainer_mode == PRODUCTION_TRAINER_MODE
    if (
        trainer_mode not in {PRODUCTION_TRAINER_MODE, SYNTHETIC_TRAINER_MODE}
        or contract.get("production_receipt") is not production
        or (payload.get("status") == PRODUCTION_FIT_STATUS) is not production
    ):
        raise HistoryStagedPipelineError("history fit trainer identity changed")
    lineage_claim = lineage.get("production_run_claim_sha256")
    if production:
        _require_sha256(lineage_claim, "production_run_claim_sha256")
    elif lineage_claim is not None:
        raise HistoryStagedPipelineError("history fit production claim changed")
    if public != {
        "aggregate_only": True,
        "contains_row_level_values": False,
        "contains_private_paths": False,
        "contains_probabilities_targets_or_performance": False,
    }:
        raise HistoryStagedPipelineError("history fit receipt public policy changed")
    _validate_aggregate_producer_receipt(payload)
    return payload


def _produce_history_fit_only_impl(
    *,
    fit: FitRoleView,
    fit_map: FitProtocolMap,
    fit_preflight_receipt_path: str | Path,
    expected_fit_preflight_receipt_sha256: str,
    checkpoint_root: str | Path,
    outcome_artifact_path: str | Path,
    targets_artifact_path: str | Path,
    producer_receipt_path: str | Path,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    utility_config: UtilitySamplingConfig,
    config_sha256: Mapping[str, str],
    code_sha256: Mapping[str, str],
    runtime_environment_sha256: str,
    device: torch.device,
    fold_callback: HistoryFitFoldCallback | None = None,
    production_run_claim_sha256: str | None = None,
    execution_environment_sha256: str | None = None,
) -> HistoryFitProduction:
    """Produce fit OOF history probabilities without any selection parameter."""

    production_trainer = fold_callback is None
    trainer_mode = (
        PRODUCTION_TRAINER_MODE if production_trainer else SYNTHETIC_TRAINER_MODE
    )
    try:
        model_config.validate_dataset_label_order(tuple(fit.label_order))
    except (TypeError, ValueError) as error:
        raise HistoryStagedPipelineError(str(error)) from error
    receipt, preflight_sha = _verify_preflight_and_fit_map(
        fit=fit,
        fit_map=fit_map,
        fit_preflight_receipt_path=fit_preflight_receipt_path,
        expected_fit_preflight_receipt_sha256=expected_fit_preflight_receipt_sha256,
        config_sha256=config_sha256,
        code_sha256=code_sha256,
        runtime_environment_sha256=runtime_environment_sha256,
    )
    del receipt
    run_config.validate()
    utility_config_hash = _config_sha256(utility_config)
    model_config_hash = _config_sha256(model_config)
    run_config_hash = _config_sha256(run_config)
    normalized_config = _normalized_hash_mapping(config_sha256, "config_sha256")
    normalized_code = _normalized_hash_mapping(code_sha256, "code_sha256")
    runtime_hash = _require_sha256(
        runtime_environment_sha256, "runtime_environment_sha256"
    )
    execution_hash = _require_sha256(
        runtime_hash
        if execution_environment_sha256 is None
        else execution_environment_sha256,
        "execution_environment_sha256",
    )
    source_identity = _canonical_sha256(
        {
            "protocol": HISTORY_STAGED_PROTOCOL,
            "trainer_mode": trainer_mode,
            "dataset": fit.dataset,
            "fit_contract_sha256": fit.contract_sha256,
            "fit_protocol_map_sha256": fit_map.artifact_sha256,
            "model_config_sha256": model_config_hash,
            "run_config_sha256": run_config_hash,
            "utility_config_sha256": utility_config_hash,
            "config_sha256": normalized_config,
            "code_sha256": normalized_code,
            "runtime_environment_sha256": runtime_hash,
            "execution_environment_sha256": execution_hash,
        }
    )
    full_corpus = _fit_corpus_from_view(fit, model_config=model_config)
    tasks = _tasks_for_role(full_corpus, utility_config)
    if not tasks:
        raise HistoryStagedPipelineError(
            "history fit utility task set must be non-empty"
        )
    folds = _fold_assignment(fit, model_config=model_config, run_config=run_config)
    classes = len(fit.label_order)
    endpoint = np.full(
        (len(EXPECTED_SEEDS), fit.rows, len(ENDPOINT_CONTEXT_NAMES), classes),
        np.nan,
        dtype=np.float32,
    )
    utility = np.full(
        (len(EXPECTED_SEEDS), len(tasks), len(UTILITY_CONTEXT_NAMES), classes),
        np.nan,
        dtype=np.float32,
    )
    callback = fold_callback or make_real_history_fit_fold_callback(
        model_config=model_config,
        run_config=run_config,
        device=device,
    )
    root = Path(checkpoint_root)
    outcome_destination = Path(outcome_artifact_path)
    targets_destination = Path(targets_artifact_path)
    receipt_destination = Path(producer_receipt_path)
    production_claim_for_receipt: str | None = None
    if production_trainer:
        private_root = _validate_production_private_layout(
            checkpoint_root=root,
            fit_outcome=outcome_destination,
            fit_targets=targets_destination,
            fit_receipt=receipt_destination,
        )
        if production_run_claim_sha256 is None:
            raise HistoryStagedPipelineError(
                "production history fit requires a lineage-bound private claim"
            )
        expected_claim = history_production_claim_sha256(
            fit=fit,
            fit_map=fit_map,
            fit_preflight_receipt_sha256=preflight_sha,
            model_config=model_config,
            run_config=run_config,
            utility_config=utility_config,
            config_sha256=normalized_config,
            code_sha256=normalized_code,
            runtime_environment_sha256=runtime_hash,
            execution_environment_sha256=execution_hash,
        )
        if _require_sha256(
            production_run_claim_sha256, "production_run_claim_sha256"
        ) != expected_claim:
            raise HistoryStagedPipelineError("production history claim lineage changed")
        _verify_private_claim(private_root, expected_claim)
        production_claim_for_receipt = expected_claim
        if not private_root.is_dir() or any(
            entry.name
            not in {
                _PRIVATE_CLAIM_NAME,
                _PRIVATE_CHECKPOINT_NAME,
                _PRIVATE_LOCK_NAME,
            }
            for entry in private_root.iterdir()
        ):
            raise HistoryStagedPipelineError(
                "production history fit private root contains an unexpected artifact"
            )
    if root.exists() and not production_trainer:
        raise FileExistsError("history checkpoint root must not exist before fit training")
    if any(
        value.exists()
        for value in (outcome_destination, targets_destination, receipt_destination)
    ):
        raise FileExistsError("history fit output already exists")
    task_positions = {id(task): index for index, task in enumerate(tasks)}
    for seed_index, seed in enumerate(EXPECTED_SEEDS):
        for fold in range(int(run_config.outer_folds)):
            held = np.flatnonzero(folds[seed_index] == fold).astype(np.int64)
            train = np.flatnonzero(folds[seed_index] != fold).astype(np.int64)
            if not len(held) or not len(train):
                raise HistoryStagedPipelineError("history fit fold is empty")
            held_set = set(int(value) for value in held)
            held_tasks = tuple(task for task in tasks if int(task.query_index) in held_set)
            for task in held_tasks:
                if any(int(value) not in held_set for value in full_corpus.histories[int(task.query_index)]):
                    raise HistoryStagedPipelineError("history task crosses an outer fold")
            request = HistoryFitFoldRequest(
                dataset=fit.dataset,
                seed=int(seed),
                fold=int(fold),
                train_indices=train,
                heldout_indices=held,
                train_texts=tuple(fit.texts[int(index)] for index in train),
                train_audio=np.asarray(fit.audio[train]).copy(),
                train_video=np.asarray(fit.video[train]).copy(),
                train_labels=np.asarray(fit.labels[train]).copy(),
                train_group_tokens=np.asarray(fit.groups[train]).copy(),
                train_speaker_tokens=np.asarray(fit.speakers[train]).copy(),
                train_turns=np.asarray(fit.turns[train]).copy(),
                train_protocol_row_ids=np.asarray(fit.protocol_row_ids[train]).copy(),
                train_histories=tuple(fit.histories[int(index)] for index in train),
                heldout_texts=tuple(fit.texts[int(index)] for index in held),
                heldout_audio=np.asarray(fit.audio[held]).copy(),
                heldout_video=np.asarray(fit.video[held]).copy(),
                heldout_group_tokens=np.asarray(fit.groups[held]).copy(),
                heldout_speaker_tokens=np.asarray(fit.speakers[held]).copy(),
                heldout_turns=np.asarray(fit.turns[held]).copy(),
                heldout_protocol_row_ids=np.asarray(fit.protocol_row_ids[held]).copy(),
                heldout_histories=tuple(fit.histories[int(index)] for index in held),
                heldout_tasks=held_tasks,
                heldout_labels_materialized=False,
                heldout_targets_materialized=False,
                source_identity_sha256=source_identity,
                checkpoint_root=root,
            )
            output = callback(request)
            if _require_sha256(
                output.source_identity_sha256, "fold source identity"
            ) != source_identity:
                raise HistoryStagedPipelineError("history fold source identity changed")
            held_endpoint = _probability(
                output.endpoint_probability,
                (len(held), len(ENDPOINT_CONTEXT_NAMES), classes),
                "heldout_endpoint_probability",
            )
            held_utility = _probability(
                output.utility_probability,
                (len(held_tasks), len(UTILITY_CONTEXT_NAMES), classes),
                "heldout_utility_probability",
            )
            if np.isfinite(endpoint[seed_index, held]).any():
                raise HistoryStagedPipelineError("history fit row was predicted twice")
            endpoint[seed_index, held] = held_endpoint
            for local, task in enumerate(held_tasks):
                position = task_positions[id(task)]
                if np.isfinite(utility[seed_index, position]).any():
                    raise HistoryStagedPipelineError("history fit task was predicted twice")
                utility[seed_index, position] = held_utility[local]
    if not np.isfinite(endpoint).all() or not np.isfinite(utility).all():
        raise HistoryStagedPipelineError("history fit OOF coverage is incomplete")
    checkpoint_manifest = build_checkpoint_manifest(
        root, seeds=EXPECTED_SEEDS, outer_folds=int(run_config.outer_folds)
    )
    verify_complete_history_checkpoint_payloads(
        root,
        checkpoint_manifest,
        fit=fit,
        model_config=model_config,
        run_config=run_config,
        source_identity_sha256=source_identity,
        fold_by_seed_query=folds,
    )
    outcome_values: dict[str, np.ndarray] = {
        "schema_version": np.asarray(HISTORY_FIT_OUTCOME_SCHEMA),
        "dataset": np.asarray(fit.dataset),
        "dataset_label_order": np.asarray(fit.label_order),
        "seeds": np.asarray(EXPECTED_SEEDS, dtype=np.int64),
        "outer_folds": np.asarray(int(run_config.outer_folds), dtype=np.int64),
        "fit_protocol_row_ids": np.asarray(fit.protocol_row_ids, dtype=np.int64),
        "fit_query_indices": np.arange(fit.rows, dtype=np.int64),
        "fit_cluster_codes": _cluster_codes(fit.groups),
        "fit_histories_sha256": np.asarray(
            _canonical_sha256([list(row) for row in fit.histories])
        ),
        "fit_endpoint_probability_oof": endpoint,
        "fit_utility_probability_oof": utility,
        "fit_fold_by_seed_query": folds,
        "source_identity_sha256": np.asarray(source_identity),
        "checkpoint_manifest_sha256": np.asarray(checkpoint_manifest.manifest_sha256),
        "model_config_sha256": np.asarray(model_config_hash),
        "run_config_sha256": np.asarray(run_config_hash),
        "utility_config_sha256": np.asarray(utility_config_hash),
        "matrix_fit_endpoint_probability_oof_sha256": np.asarray(
            _array_sha256(endpoint)
        ),
        "matrix_fit_utility_probability_oof_sha256": np.asarray(
            _array_sha256(utility)
        ),
        "fold_assignment_sha256": np.asarray(_array_sha256(folds)),
        **_task_artifact_values("fit", tasks),
    }
    _fit_outcome_view_from_values(
        outcome_values,
        fit=fit,
        checkpoint_manifest=checkpoint_manifest,
        artifact_sha256="0" * 64,
    )
    outcome_sha = _write_npz_once(outcome_destination, outcome_values)
    targets = _utility_arrays(full_corpus, tasks, utility)
    targets_values: dict[str, np.ndarray] = {
        "schema_version": np.asarray(HISTORY_FIT_TARGETS_SCHEMA),
        "dataset": np.asarray(fit.dataset),
        "seeds": np.asarray(EXPECTED_SEEDS, dtype=np.int64),
        "fit_task_sha256": np.asarray(_task_sha256(tasks)),
        "fit_forward_utility": np.asarray(targets["forward"], dtype=np.float32),
        "fit_backward_utility": np.asarray(targets["backward"], dtype=np.float32),
        "fit_asymmetry": np.asarray(targets["asymmetry"], dtype=np.float32),
        "fit_sign_agreement": np.asarray(targets["sign_agreement"], dtype=bool),
        "source_identity_sha256": np.asarray(source_identity),
        "fit_outcome_artifact_sha256": np.asarray(outcome_sha),
    }
    for name in (
        "fit_forward_utility",
        "fit_backward_utility",
        "fit_asymmetry",
        "fit_sign_agreement",
    ):
        targets_values[f"matrix_{name}_sha256"] = np.asarray(
            _array_sha256(targets_values[name])
        )
    targets_sha = _write_npz_once(targets_destination, targets_values)
    load_history_fit_targets_view(
        targets_destination,
        expected_fit_outcome_sha256=outcome_sha,
        expected_source_identity_sha256=source_identity,
        expected_task_sha256=_task_sha256(tasks),
    )
    fit_receipt = _fit_receipt_payload(
        dataset=fit.dataset,
        preflight_sha256=preflight_sha,
        fit_map=fit_map,
        config_sha256=normalized_config,
        code_sha256=normalized_code,
        runtime_environment_sha256=runtime_hash,
        execution_environment_sha256=execution_hash,
        model_config_sha256=model_config_hash,
        run_config_sha256=run_config_hash,
        utility_config_sha256=utility_config_hash,
        source_identity_sha256=source_identity,
        fold_assignment_sha256=_array_sha256(folds),
        task_sha256=_task_sha256(tasks),
        checkpoint_manifest=checkpoint_manifest,
        outcome_artifact_sha256=outcome_sha,
        targets_artifact_sha256=targets_sha,
        fit_rows=fit.rows,
        task_count=len(tasks),
        trainer_mode=trainer_mode,
        production_run_claim_sha256=production_claim_for_receipt,
    )
    receipt_sha = _write_json_once(receipt_destination, fit_receipt)
    return HistoryFitProduction(
        outcome_artifact_path=outcome_destination.resolve(),
        outcome_artifact_sha256=outcome_sha,
        targets_artifact_path=targets_destination.resolve(),
        targets_artifact_sha256=targets_sha,
        receipt_path=receipt_destination.resolve(),
        receipt_sha256=receipt_sha,
        checkpoint_manifest=checkpoint_manifest,
        source_identity_sha256=source_identity,
        production_trainer=production_trainer,
    )


def produce_history_fit_only(
    *,
    fit: FitRoleView,
    fit_map: FitProtocolMap,
    fit_preflight_receipt_path: str | Path,
    expected_fit_preflight_receipt_sha256: str,
    checkpoint_root: str | Path,
    outcome_artifact_path: str | Path,
    targets_artifact_path: str | Path,
    producer_receipt_path: str | Path,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    utility_config: UtilitySamplingConfig,
    config_sha256: Mapping[str, str],
    code_sha256: Mapping[str, str],
    runtime_environment_sha256: str,
    device: torch.device,
    fold_callback: HistoryFitFoldCallback | None = None,
    production_run_claim_sha256: str | None = None,
    execution_environment_sha256: str | None = None,
) -> HistoryFitProduction:
    """Run fit-only production under an OS lock; test callbacks stay non-production."""

    arguments = {
        "fit": fit,
        "fit_map": fit_map,
        "fit_preflight_receipt_path": fit_preflight_receipt_path,
        "expected_fit_preflight_receipt_sha256": expected_fit_preflight_receipt_sha256,
        "checkpoint_root": checkpoint_root,
        "outcome_artifact_path": outcome_artifact_path,
        "targets_artifact_path": targets_artifact_path,
        "producer_receipt_path": producer_receipt_path,
        "model_config": model_config,
        "run_config": run_config,
        "utility_config": utility_config,
        "config_sha256": config_sha256,
        "code_sha256": code_sha256,
        "runtime_environment_sha256": runtime_environment_sha256,
        "device": device,
        "fold_callback": fold_callback,
        "production_run_claim_sha256": production_run_claim_sha256,
        "execution_environment_sha256": execution_environment_sha256,
    }
    if fold_callback is not None:
        return _produce_history_fit_only_impl(**arguments)
    private_root = Path(checkpoint_root).resolve().parent
    with history_private_run_lock(private_root):
        return _produce_history_fit_only_impl(**arguments)


def verify_history_fit_for_completion(
    *,
    fit: FitRoleView,
    fit_map: FitProtocolMap,
    fit_outcome_artifact_path: str | Path,
    expected_fit_outcome_artifact_sha256: str,
    fit_targets_artifact_path: str | Path,
    expected_fit_targets_artifact_sha256: str,
    fit_producer_receipt_path: str | Path,
    expected_fit_producer_receipt_sha256: str,
    fit_preflight_receipt_path: str | Path,
    expected_fit_preflight_receipt_sha256: str,
    checkpoint_root: str | Path,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    utility_config: UtilitySamplingConfig,
    config_sha256: Mapping[str, str],
    code_sha256: Mapping[str, str],
    runtime_environment_sha256: str,
    execution_environment_sha256: str | None = None,
) -> VerifiedHistoryFitState:
    """Verify all fit inputs and complete checkpoints before selection is opened."""

    try:
        model_config.validate_dataset_label_order(tuple(fit.label_order))
    except (TypeError, ValueError) as error:
        raise HistoryStagedPipelineError(str(error)) from error
    receipt, preflight_sha = _verify_preflight_and_fit_map(
        fit=fit,
        fit_map=fit_map,
        fit_preflight_receipt_path=fit_preflight_receipt_path,
        expected_fit_preflight_receipt_sha256=expected_fit_preflight_receipt_sha256,
        config_sha256=config_sha256,
        code_sha256=code_sha256,
        runtime_environment_sha256=runtime_environment_sha256,
    )
    checkpoint_manifest = build_checkpoint_manifest(
        checkpoint_root,
        seeds=EXPECTED_SEEDS,
        outer_folds=int(run_config.outer_folds),
    )
    outcome_path = Path(fit_outcome_artifact_path).resolve()
    outcome_sha = _require_sha256(
        expected_fit_outcome_artifact_sha256,
        "expected_fit_outcome_artifact_sha256",
    )
    if _file_sha256(outcome_path) != outcome_sha:
        raise HistoryStagedPipelineError("fit outcome artifact hash changed")
    outcome = load_history_fit_outcome_view(
        outcome_path,
        fit=fit,
        checkpoint_manifest=checkpoint_manifest,
    )
    if outcome.artifact_sha256 != outcome_sha:
        raise HistoryStagedPipelineError("fit outcome object uses another artifact")
    expected_model = _config_sha256(model_config)
    expected_run = _config_sha256(run_config)
    expected_utility = _config_sha256(utility_config)
    if (
        outcome.model_config_sha256 != expected_model
        or outcome.run_config_sha256 != expected_run
        or outcome.utility_config_sha256 != expected_utility
    ):
        raise HistoryStagedPipelineError("fit model/run/utility configuration changed")
    target_path = Path(fit_targets_artifact_path).resolve()
    target_sha = _require_sha256(
        expected_fit_targets_artifact_sha256,
        "expected_fit_targets_artifact_sha256",
    )
    if _file_sha256(target_path) != target_sha:
        raise HistoryStagedPipelineError("fit targets artifact hash changed")
    target_view = load_history_fit_targets_view(
        target_path,
        expected_fit_outcome_sha256=outcome_sha,
        expected_source_identity_sha256=outcome.source_identity_sha256,
        expected_task_sha256=outcome.tasks.task_sha256,
    )
    if (
        target_view.artifact_sha256 != target_sha
        or target_view.dataset != fit.dataset
        or target_view.forward_utility.shape
        != (len(EXPECTED_SEEDS), len(outcome.tasks))
    ):
        raise HistoryStagedPipelineError("fit targets differ from fit outcome")
    normalized_config = _normalized_hash_mapping(config_sha256, "config_sha256")
    normalized_code = _normalized_hash_mapping(code_sha256, "code_sha256")
    runtime_hash = _require_sha256(
        runtime_environment_sha256, "runtime_environment_sha256"
    )
    execution_hash = _require_sha256(
        runtime_hash
        if execution_environment_sha256 is None
        else execution_environment_sha256,
        "execution_environment_sha256",
    )
    fit_receipt_path = Path(fit_producer_receipt_path).resolve()
    fit_receipt_sha = _require_sha256(
        expected_fit_producer_receipt_sha256,
        "expected_fit_producer_receipt_sha256",
    )
    fit_receipt_payload = _load_history_fit_receipt(
        fit_receipt_path,
        expected_sha256=fit_receipt_sha,
        expected={
            "fit_preflight_receipt_sha256": preflight_sha,
            "fit_protocol_map_sha256": fit_map.artifact_sha256,
            "config_sha256": normalized_config,
            "code_sha256": normalized_code,
            "runtime_environment_sha256": runtime_hash,
            "execution_environment_sha256": execution_hash,
            "model_config_sha256": expected_model,
            "run_config_sha256": expected_run,
            "utility_config_sha256": expected_utility,
            "source_identity_sha256": outcome.source_identity_sha256,
            "fold_assignment_sha256": _array_sha256(
                outcome.fold_by_seed_query
            ),
            "fit_task_sha256": outcome.tasks.task_sha256,
            "checkpoint_manifest_sha256": checkpoint_manifest.manifest_sha256,
            "private_fit_outcome_artifact_sha256": outcome_sha,
            "private_fit_targets_artifact_sha256": target_sha,
        },
    )
    sidecars = receipt.get("sidecars")
    if not isinstance(sidecars, Mapping):
        raise HistoryStagedPipelineError("preflight receipt lacks sidecar lineage")
    selection_sidecar = sidecars.get(SELECTION_ROLE)
    if not isinstance(selection_sidecar, Mapping):
        raise HistoryStagedPipelineError("preflight receipt lacks selection feature lineage")
    selection_feature_sha = _require_sha256(
        selection_sidecar.get("feature_sha256"), "selection_feature_sha256"
    )
    verify_complete_history_checkpoint_payloads(
        checkpoint_root,
        checkpoint_manifest,
        fit=fit,
        model_config=model_config,
        run_config=run_config,
        source_identity_sha256=outcome.source_identity_sha256,
        fold_by_seed_query=outcome.fold_by_seed_query,
    )
    # Repeat the byte-level checks after all deserialisation to close the
    # validation-window replacement race.
    verify_checkpoint_manifest(checkpoint_root, checkpoint_manifest)
    if (
        _file_sha256(outcome_path) != outcome_sha
        or _file_sha256(target_path) != target_sha
        or _file_sha256(fit_receipt_path) != fit_receipt_sha
        or _file_sha256(Path(fit_preflight_receipt_path)) != preflight_sha
    ):
        raise HistoryStagedPipelineError("fit artifact changed during completion gate")
    contract = fit_receipt_payload.get("training_contract")
    production_trainer = bool(
        isinstance(contract, Mapping)
        and contract.get("trainer_mode") == PRODUCTION_TRAINER_MODE
        and contract.get("production_receipt") is True
    )
    private_root = None
    production_claim = None
    if production_trainer:
        private_root = _validate_production_private_layout(
            checkpoint_root=checkpoint_root,
            fit_outcome=outcome_path,
            fit_targets=target_path,
            fit_receipt=fit_receipt_path,
        )
        expected_claim = history_production_claim_sha256(
            fit=fit,
            fit_map=fit_map,
            fit_preflight_receipt_sha256=preflight_sha,
            model_config=model_config,
            run_config=run_config,
            utility_config=utility_config,
            config_sha256=normalized_config,
            code_sha256=normalized_code,
            runtime_environment_sha256=runtime_hash,
            execution_environment_sha256=execution_hash,
        )
        receipt_lineage = fit_receipt_payload.get("lineage")
        if (
            not isinstance(receipt_lineage, Mapping)
            or receipt_lineage.get("production_run_claim_sha256") != expected_claim
        ):
            raise HistoryStagedPipelineError(
                "history fit receipt uses another production claim"
        )
        _verify_private_claim(private_root, expected_claim)
        production_claim = expected_claim
    return VerifiedHistoryFitState(
        fit_outcome_path=outcome_path,
        fit_outcome_sha256=outcome_sha,
        fit_targets_path=target_path,
        fit_targets_sha256=target_sha,
        fit_receipt_path=fit_receipt_path,
        fit_receipt_sha256=fit_receipt_sha,
        fit_preflight_receipt_sha256=preflight_sha,
        selection_feature_sha256=selection_feature_sha,
        checkpoint_manifest=checkpoint_manifest,
        source_identity_sha256=outcome.source_identity_sha256,
        fit_outcome=outcome,
        production_trainer=production_trainer,
        private_output_root=private_root,
        execution_environment_sha256=execution_hash,
        production_run_claim_sha256=production_claim,
    )


def _assert_fit_state_unchanged(
    state: VerifiedHistoryFitState,
    *,
    checkpoint_root: str | Path,
    fit_preflight_receipt_path: str | Path,
) -> None:
    if (
        _file_sha256(state.fit_outcome_path) != state.fit_outcome_sha256
        or _file_sha256(state.fit_targets_path) != state.fit_targets_sha256
        or _file_sha256(state.fit_receipt_path) != state.fit_receipt_sha256
        or _file_sha256(Path(fit_preflight_receipt_path))
        != state.fit_preflight_receipt_sha256
    ):
        raise HistoryStagedPipelineError("fit artifact or receipt changed after verification")
    verify_checkpoint_manifest(checkpoint_root, state.checkpoint_manifest)


def verify_complete_history_checkpoint_payloads(
    checkpoint_root: str | Path,
    manifest: CheckpointManifest,
    *,
    fit: FitRoleView,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    source_identity_sha256: str,
    fold_by_seed_query: np.ndarray,
) -> None:
    """Semantically restore every fold before selection features may be opened."""

    root = Path(checkpoint_root)
    verify_checkpoint_manifest(root, manifest)
    source_identity = _require_sha256(
        source_identity_sha256, "source_identity_sha256"
    )
    folds = np.asarray(fold_by_seed_query)
    if folds.shape != (len(manifest.seeds), fit.rows):
        raise HistoryStagedPipelineError("checkpoint fold assignment shape changed")
    checkpoint_records = [
        record for record in manifest.records if record.kind == "checkpoint"
    ]
    if len(checkpoint_records) != len(manifest.seeds) * manifest.outer_folds:
        raise HistoryStagedPipelineError("checkpoint manifest lacks one payload per fold")
    by_key = {(record.seed, record.fold, record.kind): record for record in manifest.records}
    for record in checkpoint_records:
        seed_index = manifest.seeds.index(int(record.seed))
        held = np.flatnonzero(folds[seed_index] == int(record.fold)).astype(np.int64)
        train = np.flatnonzero(folds[seed_index] != int(record.fold)).astype(np.int64)
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
            validation_fraction=run_config.inner_validation_fraction,
            seed=int(record.seed),
            fold=int(record.fold),
        )
        processor_record = by_key.get((record.seed, record.fold, "text_processor"))
        if processor_record is None:
            raise HistoryStagedPipelineError("checkpoint manifest lacks a fold processor")
        processor_path = root / Path(processor_record.relative_name)
        try:
            processor_payload = joblib.load(processor_path)
        except Exception as error:
            raise HistoryStagedPipelineError(
                f"cannot deserialize fold text processor: {record.relative_name}"
            ) from error
        processor_identity = _canonical_sha256(
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
                    for name, value in asdict(run_config).items()
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
            or processor_payload.get("identity_sha256") != processor_identity
            or not isinstance(processor, FoldTextProcessor)
            or processor.fit_indices_sha256
            != _indices_sha256(split.inner_train_indices)
            or int(processor.output_dim) != int(model_config.text_dim)
            or not 1 <= int(processor.effective_dim) <= int(processor.output_dim)
        ):
            raise HistoryStagedPipelineError("fold text processor identity is invalid")
        try:
            probe = processor.transform(corpus.texts[:1])
        except Exception as error:
            raise HistoryStagedPipelineError(
                "fold text processor cannot transform its bound corpus"
            ) from error
        if probe.shape != (1, model_config.text_dim) or not np.isfinite(probe).all():
            raise HistoryStagedPipelineError("fold text processor output is invalid")
        processor_sha = _file_sha256(processor_path)
        if processor_sha != processor_record.sha256:
            raise HistoryStagedPipelineError("fold text processor changed while loading")
        expected_identity = _canonical_sha256(
            {
                "source_identity": source_identity,
                "seed": int(record.seed),
                "fold": int(record.fold),
                "inner_train": _indices_sha256(split.inner_train_indices),
                "inner_validation": _indices_sha256(split.inner_validation_indices),
                "outer_heldout": _indices_sha256(split.outer_heldout_indices),
                "processor_sha256": processor_sha,
                "model_config": asdict(model_config),
                "run_config": asdict(run_config),
            }
        )
        checkpoint_path = root / Path(record.relative_name)
        try:
            payload = _torch_load_local(checkpoint_path)
        except Exception as error:
            raise HistoryStagedPipelineError(
                f"cannot deserialize fold checkpoint: {record.relative_name}"
            ) from error
        if (
            payload.get("schema_version") != "causal_backbone_atomic_checkpoint_v2"
            or payload.get("status") != "complete"
        ):
            raise HistoryStagedPipelineError(
                "selection gate refuses a partial or unknown checkpoint payload"
            )
        if payload.get("identity_sha256") != expected_identity:
            raise HistoryStagedPipelineError(
                "checkpoint identity differs from source/fold/config/split"
            )
        rng_state = payload.get("rng_state")
        model_state = payload.get("model_state")
        best_state = payload.get("best_model_state")
        optimizer_state = payload.get("optimizer_state")
        scaler_state = payload.get("scaler_state")
        if (
            not isinstance(rng_state, Mapping)
            or set(rng_state) != {"python", "numpy", "torch_cpu", "torch_cuda"}
            or not isinstance(model_state, Mapping)
            or not model_state
            or not isinstance(best_state, Mapping)
            or not best_state
            or set(model_state) != set(best_state)
            or not isinstance(optimizer_state, Mapping)
            or not isinstance(scaler_state, Mapping)
        ):
            raise HistoryStagedPipelineError("complete checkpoint state is incomplete")
        epoch = payload.get("epoch")
        best_epoch = payload.get("best_epoch")
        best_nll = payload.get("best_validation_nll")
        bad_epochs = payload.get("bad_epochs")
        peak_mib = payload.get("peak_cuda_mib")
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
            or not isinstance(bad_epochs, int)
            or isinstance(bad_epochs, bool)
            or bad_epochs < 0
            or not isinstance(payload.get("early_stopped"), bool)
            or not isinstance(peak_mib, (int, float))
            or isinstance(peak_mib, bool)
            or not np.isfinite(float(peak_mib))
            or float(peak_mib) < 0.0
        ):
            raise HistoryStagedPipelineError("complete checkpoint stopping metadata is invalid")
        try:
            model = CausalMultimodalBackbone(model_config)
            model.load_state_dict(model_state, strict=True)
            model.load_state_dict(best_state, strict=True)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=run_config.learning_rate,
                weight_decay=run_config.weight_decay,
            )
            optimizer.load_state_dict(dict(optimizer_state))
            scaler = torch.cuda.amp.GradScaler(
                enabled=bool(run_config.use_amp and torch.cuda.is_available())
            )
            scaler.load_state_dict(dict(scaler_state))
            saved_rng = _capture_rng_state()
            try:
                _restore_rng_state(rng_state)
            finally:
                _restore_rng_state(saved_rng)
        except Exception as error:
            raise HistoryStagedPipelineError(
                "complete checkpoint cannot be strictly restored"
            ) from error
        if _file_sha256(checkpoint_path) != record.sha256:
            raise HistoryStagedPipelineError("fold checkpoint changed while loading")
        verify_checkpoint_manifest(root, manifest)


def _selection_feature_contract_sha256(selection: SelectionFeatureView) -> str:
    return _canonical_sha256(
        {
            "dataset": selection.dataset,
            "texts": _array_sha256(np.asarray(selection.texts).astype(str)),
            "audio": _array_sha256(np.asarray(selection.audio, dtype=np.float32)),
            "video": _array_sha256(np.asarray(selection.video, dtype=np.float32)),
            "groups": _array_sha256(np.asarray(selection.groups).astype(str)),
            "speakers": _array_sha256(np.asarray(selection.speakers).astype(str)),
            "turns": _array_sha256(np.asarray(selection.turns, dtype=np.int64)),
            "protocol_row_ids": _array_sha256(
                np.asarray(selection.protocol_row_ids, dtype=np.int64)
            ),
            "labels_materialized": bool(selection.labels_materialized),
        }
    )


def _fit_gate_sha256(state: VerifiedHistoryFitState) -> str:
    return _canonical_sha256(
        {
            "fit_outcome_sha256": state.fit_outcome_sha256,
            "fit_targets_sha256": state.fit_targets_sha256,
            "fit_receipt_sha256": state.fit_receipt_sha256,
            "fit_preflight_receipt_sha256": state.fit_preflight_receipt_sha256,
            "selection_feature_sha256": state.selection_feature_sha256,
            "checkpoint_manifest_sha256": state.checkpoint_manifest.manifest_sha256,
            "source_identity_sha256": state.source_identity_sha256,
            "execution_environment_sha256": state.execution_environment_sha256,
        }
    )


def materialize_history_selection_features_after_fit_gate(
    *,
    fit: FitRoleView,
    fit_state: VerifiedHistoryFitState,
    checkpoint_root: str | Path,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    fit_preflight_receipt_path: str | Path,
    dataset: str,
    sidecar_dir: str | Path,
    manifest_path: str | Path,
    config_paths: Mapping[str, str | Path],
    code_paths: Mapping[str, str | Path],
    environment: Mapping[str, object],
    execution_environment: Mapping[str, object] | None = None,
) -> VerifiedHistorySelectionView:
    """Open only the selection feature NPZ after complete payload verification.

    The selection label filename is never resolved to a filesystem path, and
    its archive is never stat'ed, hashed, opened, or deserialized here.
    """

    if dataset not in _SPECS:
        raise HistoryStagedPipelineError("dataset must be EmotionTalk or MELD")
    live_config, live_code, live_runtime, live_lineage = _live_lineage(
        config_paths=config_paths,
        code_paths=code_paths,
        environment=(
            environment if execution_environment is None else execution_environment
        ),
    )
    if live_runtime != fit_state.execution_environment_sha256:
        raise HistoryStagedPipelineError(
            "execution runtime differs from the history fit lineage"
        )
    verify_selection_feature_receipt_inputs(
        receipt_path=fit_preflight_receipt_path,
        expected_receipt_sha256=fit_state.fit_preflight_receipt_sha256,
        dataset=dataset,
        sidecar_dir=sidecar_dir,
        manifest_path=manifest_path,
        config_paths=config_paths,
        code_paths=code_paths,
        environment=environment,
    )
    if _file_sha256(Path(fit_preflight_receipt_path)) != (
        fit_state.fit_preflight_receipt_sha256
    ):
        raise HistoryStagedPipelineError("fit preflight receipt changed before feature gate")
    _assert_fit_state_unchanged(
        fit_state,
        checkpoint_root=checkpoint_root,
        fit_preflight_receipt_path=fit_preflight_receipt_path,
    )
    verify_complete_history_checkpoint_payloads(
        checkpoint_root,
        fit_state.checkpoint_manifest,
        fit=fit,
        model_config=model_config,
        run_config=run_config,
        source_identity_sha256=fit_state.source_identity_sha256,
        fold_by_seed_query=fit_state.fit_outcome.fold_by_seed_query,
    )
    receipt = _load_receipt(Path(fit_preflight_receipt_path))
    if receipt.get("dataset") != dataset:
        raise HistoryStagedPipelineError("selection dataset differs from fit receipt")
    receipt_manifest = receipt.get("manifest")
    if not isinstance(receipt_manifest, Mapping):
        raise HistoryStagedPipelineError("fit receipt lacks manifest lineage")
    manifest_file = Path(manifest_path)
    manifest_sha = _file_sha256(manifest_file)
    if manifest_sha != receipt_manifest.get("sha256"):
        raise HistoryStagedPipelineError("selection manifest differs from fit receipt")
    manifest = _read_manifest_json(manifest_file)
    spec = _SPECS[dataset]
    _validate_manifest_contract(manifest, spec)
    roles = manifest.get("roles")
    if not isinstance(roles, Mapping):
        raise HistoryStagedPipelineError("selection manifest lacks role records")
    raw = roles.get(SELECTION_ROLE)
    if not isinstance(raw, Mapping) or set(raw) != set(spec.role_record_fields):
        raise HistoryStagedPipelineError("selection manifest role record changed")
    feature_name = _plain_npz_filename(
        raw.get("feature_filename"),
        f"features_{SELECTION_ROLE}.npz",
        f"{SELECTION_ROLE}.feature",
    )
    feature_path = Path(sidecar_dir) / feature_name
    feature_sha = _file_sha256(feature_path)
    declared_feature_sha = _require_sha256(
        raw.get("feature_sha256"), f"{SELECTION_ROLE}.feature_sha256"
    )
    if (
        feature_sha != declared_feature_sha
        or feature_sha != fit_state.selection_feature_sha256
    ):
        raise HistoryStagedPipelineError("selection feature bytes changed after fit gate")
    rows = int(raw.get("rows", -1))
    groups = int(raw.get(spec.group_count_field, -1))
    history_eligible = int(raw.get("history_eligible_rows", -1))
    audio_dim = int(raw.get("audio_dimension", -1))
    video_dim = int(raw.get("video_dimension", -1))
    if (
        rows < 1
        or groups < 1
        or groups > rows
        or not 0 <= history_eligible <= rows
        or audio_dim < 1
        or video_dim < 1
    ):
        raise HistoryStagedPipelineError("selection manifest dimensions changed")
    record = SidecarRecord(
        role=SELECTION_ROLE,
        feature_path=feature_path,
        # Deliberately not the manifest's label filename.  The feature
        # materializer never reads this field, and the label path capability is
        # absent from this gate.
        label_path=Path("__sealed_selection_label_capability__"),
        feature_sha256=feature_sha,
        label_sha256=_require_sha256(
            raw.get("label_sha256"), f"{SELECTION_ROLE}.label_sha256"
        ),
        row_alignment_sha256=_require_sha256(
            raw.get("row_alignment_sha256"), f"{SELECTION_ROLE}.row_alignment_sha256"
        ),
        rows=rows,
        groups=groups,
        history_eligible_rows=history_eligible,
        audio_dimension=audio_dim,
        video_dimension=video_dim,
    )
    sidecars = HashedSidecarSet(
        dataset=dataset,
        spec=spec,
        manifest_path=manifest_file,
        manifest_sha256=manifest_sha,
        manifest=manifest,
        fit=record,
        selection=record,
    )
    view = _materialize_selection_feature(sidecars)
    if view.labels_materialized:
        raise HistoryStagedPipelineError("selection materializer exposed labels")
    normalized_sha = _selection_feature_contract_sha256(view)
    verify_selection_feature_receipt_inputs(
        receipt_path=fit_preflight_receipt_path,
        expected_receipt_sha256=fit_state.fit_preflight_receipt_sha256,
        dataset=dataset,
        sidecar_dir=sidecar_dir,
        manifest_path=manifest_path,
        config_paths=config_paths,
        code_paths=code_paths,
        environment=environment,
    )
    verify_complete_history_checkpoint_payloads(
        checkpoint_root,
        fit_state.checkpoint_manifest,
        fit=fit,
        model_config=model_config,
        run_config=run_config,
        source_identity_sha256=fit_state.source_identity_sha256,
        fold_by_seed_query=fit_state.fit_outcome.fold_by_seed_query,
    )
    _assert_fit_state_unchanged(
        fit_state,
        checkpoint_root=checkpoint_root,
        fit_preflight_receipt_path=fit_preflight_receipt_path,
    )
    if _file_sha256(feature_path) != feature_sha:
        raise HistoryStagedPipelineError("selection feature changed while materializing")
    after_config, after_code, after_runtime, after_lineage = _live_lineage(
        config_paths=config_paths,
        code_paths=code_paths,
        environment=(
            environment if execution_environment is None else execution_environment
        ),
    )
    if (
        after_config != live_config
        or after_code != live_code
        or after_runtime != live_runtime
        or after_lineage != live_lineage
    ):
        raise HistoryStagedPipelineError(
            "config/code/runtime changed while opening selection features"
        )
    return VerifiedHistorySelectionView(
        view=view,
        feature_file_sha256=feature_sha,
        normalized_feature_sha256=normalized_sha,
        fit_gate_sha256=_fit_gate_sha256(fit_state),
        checkpoint_manifest_sha256=fit_state.checkpoint_manifest.manifest_sha256,
        config_sha256=live_config,
        code_sha256=live_code,
        runtime_environment_sha256=live_runtime,
        live_lineage_sha256=live_lineage,
    )


_COMPLETE_OUTCOME_KEYS = frozenset(
    {
        "schema_version",
        "dataset",
        "dataset_label_order",
        "seeds",
        "fit_protocol_row_ids",
        "selection_protocol_row_ids",
        "fit_cluster_codes",
        "selection_cluster_codes",
        "fit_histories_sha256",
        "selection_histories_sha256",
        "fit_task_query_indices",
        "fit_task_candidate_indices",
        "fit_task_s_indptr",
        "fit_task_s_indices",
        "fit_task_t_indptr",
        "fit_task_t_indices",
        "fit_task_sha256",
        "selection_task_query_indices",
        "selection_task_candidate_indices",
        "selection_task_s_indptr",
        "selection_task_s_indices",
        "selection_task_t_indptr",
        "selection_task_t_indices",
        "selection_task_sha256",
        "fit_endpoint_probability_oof",
        "selection_endpoint_probability_fold_ensemble",
        "fit_utility_probability_oof",
        "selection_utility_probability_fold_ensemble",
        "source_identity_sha256",
        "checkpoint_manifest_sha256",
        "fit_outcome_artifact_sha256",
        "matrix_fit_endpoint_probability_oof_sha256",
        "matrix_selection_endpoint_probability_fold_ensemble_sha256",
        "matrix_fit_utility_probability_oof_sha256",
        "matrix_selection_utility_probability_fold_ensemble_sha256",
    }
)


def _outcome_free_view_from_values(
    values: Mapping[str, np.ndarray],
    *,
    fit: FitRoleView,
    selection: SelectionFeatureView,
    state: VerifiedHistoryFitState,
    artifact_sha256: str,
) -> HistoryOutcomeFreeView:
    if set(values) != set(_COMPLETE_OUTCOME_KEYS):
        missing = sorted(_COMPLETE_OUTCOME_KEYS - set(values))
        unknown = sorted(set(values) - _COMPLETE_OUTCOME_KEYS)
        raise HistoryStagedPipelineError(
            f"completed outcome schema changed: missing={missing}, unknown={unknown}"
        )
    forbidden_fragments = (
        "forward_utility",
        "backward_utility",
        "asymmetry",
        "sign_agreement",
        "accuracy",
        "macro_f1",
        "nll",
        "brier",
    )
    if any(
        fragment in name.lower()
        for name in values
        for fragment in forbidden_fragments
    ):
        raise HistoryStagedPipelineError("completed outcome contains a target or metric field")
    if (
        _single_text(values["schema_version"], "schema_version")
        != HISTORY_COMPLETE_OUTCOME_SCHEMA
        or _single_text(values["dataset"], "dataset") != fit.dataset
        or selection.dataset != fit.dataset
        or selection.labels_materialized
    ):
        raise HistoryStagedPipelineError("completed outcome identity changed")
    label_order = tuple(
        str(value) for value in np.asarray(values["dataset_label_order"]).reshape(-1)
    )
    seeds = tuple(
        int(value)
        for value in _integer_vector(values["seeds"], "seeds", unique=True)
    )
    if label_order != fit.label_order or seeds != EXPECTED_SEEDS:
        raise HistoryStagedPipelineError("completed outcome label/seed contract changed")
    fit_protocol = _integer_vector(
        values["fit_protocol_row_ids"], "fit_protocol_row_ids", unique=True
    )
    selection_protocol = _integer_vector(
        values["selection_protocol_row_ids"],
        "selection_protocol_row_ids",
        unique=True,
    )
    fit_cluster = _integer_vector(values["fit_cluster_codes"], "fit_cluster_codes")
    selection_cluster = _integer_vector(
        values["selection_cluster_codes"], "selection_cluster_codes"
    )
    if (
        not np.array_equal(fit_protocol, fit.protocol_row_ids)
        or not np.array_equal(selection_protocol, selection.protocol_row_ids)
        or not np.array_equal(fit_cluster, _cluster_codes(fit.groups))
        or not np.array_equal(selection_cluster, _cluster_codes(selection.groups))
    ):
        raise HistoryStagedPipelineError("completed outcome row alignment changed")
    fit_histories = validate_strict_past_histories(
        groups=fit.groups,
        speakers=fit.speakers,
        turns=fit.turns,
        histories=fit.histories,
    )
    selection_histories = _strict_histories(
        selection.groups, selection.speakers, selection.turns
    )
    fit_history_sha = _require_sha256(
        _single_text(values["fit_histories_sha256"], "fit histories"),
        "fit_histories_sha256",
    )
    selection_history_sha = _require_sha256(
        _single_text(values["selection_histories_sha256"], "selection histories"),
        "selection_histories_sha256",
    )
    if fit_history_sha != _canonical_sha256(
        [list(row) for row in fit_histories]
    ) or selection_history_sha != _canonical_sha256(
        [list(row) for row in selection_histories]
    ):
        raise HistoryStagedPipelineError("completed outcome history hash changed")
    fit_tasks = _decode_tasks(
        values, "fit", role_rows=fit.rows, histories=fit_histories
    )
    selection_tasks = _decode_tasks(
        values,
        "selection",
        role_rows=len(selection.texts),
        histories=selection_histories,
    )
    fit_endpoint = _probability(
        values["fit_endpoint_probability_oof"],
        (len(seeds), fit.rows, len(ENDPOINT_CONTEXT_NAMES), len(label_order)),
        "fit_endpoint_probability_oof",
    )
    selection_endpoint = _probability(
        values["selection_endpoint_probability_fold_ensemble"],
        (
            len(seeds),
            len(selection.texts),
            len(ENDPOINT_CONTEXT_NAMES),
            len(label_order),
        ),
        "selection_endpoint_probability_fold_ensemble",
    )
    fit_utility = _probability(
        values["fit_utility_probability_oof"],
        (len(seeds), len(fit_tasks), len(UTILITY_CONTEXT_NAMES), len(label_order)),
        "fit_utility_probability_oof",
    )
    selection_utility = _probability(
        values["selection_utility_probability_fold_ensemble"],
        (
            len(seeds),
            len(selection_tasks),
            len(UTILITY_CONTEXT_NAMES),
            len(label_order),
        ),
        "selection_utility_probability_fold_ensemble",
    )
    if (
        not np.array_equal(fit_endpoint, state.fit_outcome.endpoint_probability_oof)
        or not np.array_equal(fit_utility, state.fit_outcome.utility_probability_oof)
        or fit_tasks.task_sha256 != state.fit_outcome.tasks.task_sha256
    ):
        raise HistoryStagedPipelineError("completed fit probabilities differ from fit gate")
    matrices = {
        "fit_endpoint_probability_oof": fit_endpoint,
        "selection_endpoint_probability_fold_ensemble": selection_endpoint,
        "fit_utility_probability_oof": fit_utility,
        "selection_utility_probability_fold_ensemble": selection_utility,
    }
    for name, array in matrices.items():
        field = f"matrix_{name}_sha256"
        if _require_sha256(_single_text(values[field], field), field) != _array_sha256(
            np.asarray(array)
        ):
            raise HistoryStagedPipelineError(f"completed probability hash changed: {name}")
    source_identity = _require_sha256(
        _single_text(values["source_identity_sha256"], "source identity"),
        "source_identity_sha256",
    )
    checkpoint_sha = _require_sha256(
        _single_text(values["checkpoint_manifest_sha256"], "checkpoint manifest"),
        "checkpoint_manifest_sha256",
    )
    fit_outcome_sha = _require_sha256(
        _single_text(values["fit_outcome_artifact_sha256"], "fit outcome artifact"),
        "fit_outcome_artifact_sha256",
    )
    if (
        source_identity != state.source_identity_sha256
        or checkpoint_sha != state.checkpoint_manifest.manifest_sha256
        or fit_outcome_sha != state.fit_outcome_sha256
    ):
        raise HistoryStagedPipelineError("completed outcome lineage changed")
    return HistoryOutcomeFreeView(
        dataset=fit.dataset,
        label_order=label_order,
        seeds=seeds,
        fit_protocol_row_ids=fit_protocol,
        selection_protocol_row_ids=selection_protocol,
        fit_cluster_codes=fit_cluster.astype(np.int32, copy=True),
        selection_cluster_codes=selection_cluster.astype(np.int32, copy=True),
        fit_histories_sha256=fit_history_sha,
        selection_histories_sha256=selection_history_sha,
        fit_tasks=fit_tasks,
        selection_tasks=selection_tasks,
        fit_endpoint_probability_oof=fit_endpoint,
        selection_endpoint_probability_fold_ensemble=selection_endpoint,
        fit_utility_probability_oof=fit_utility,
        selection_utility_probability_fold_ensemble=selection_utility,
        source_identity_sha256=source_identity,
        checkpoint_manifest_sha256=checkpoint_sha,
        fit_outcome_artifact_sha256=fit_outcome_sha,
        fit_targets_artifact_sha256=state.fit_targets_sha256,
        artifact_sha256=_require_sha256(artifact_sha256, "artifact_sha256"),
    )


def load_history_outcome_free_view(
    path: str | Path,
    *,
    fit: FitRoleView,
    selection: SelectionFeatureView,
    state: VerifiedHistoryFitState,
) -> HistoryOutcomeFreeView:
    artifact = Path(path)
    digest = _file_sha256(artifact)
    values = _load_npz(artifact)
    result = _outcome_free_view_from_values(
        values,
        fit=fit,
        selection=selection,
        state=state,
        artifact_sha256=digest,
    )
    if _file_sha256(artifact) != digest:
        raise HistoryStagedPipelineError("completed outcome changed while validating")
    return result


def _completion_receipt_payload(
    *,
    dataset: str,
    state: VerifiedHistoryFitState,
    fit_map: FitProtocolMap,
    selection_feature_sha256: str,
    selection_histories_sha256: str,
    selection_task_sha256: str,
    private_artifact_sha256: str,
    selection_rows: int,
    selection_task_count: int,
    selection_live_lineage_sha256: str,
) -> dict[str, object]:
    completion_status = (
        PRODUCTION_COMPLETION_STATUS
        if state.production_trainer
        else SYNTHETIC_COMPLETION_STATUS
    )
    receipt: dict[str, object] = {
        "schema_version": HISTORY_COMPLETE_RECEIPT_SCHEMA,
        "status": completion_status,
        "dataset": dataset,
        "claim_boundary": (
            "Fit OOF plus model-selection feature-only history inference; no "
            "model-selection label, utility target, or performance metric was consumed."
        ),
        "lineage": {
            "fit_preflight_receipt_sha256": state.fit_preflight_receipt_sha256,
            "fit_producer_receipt_sha256": state.fit_receipt_sha256,
            "fit_protocol_map_sha256": fit_map.artifact_sha256,
            "fit_outcome_artifact_sha256": state.fit_outcome_sha256,
            "fit_targets_artifact_sha256": state.fit_targets_sha256,
            "checkpoint_manifest_sha256": state.checkpoint_manifest.manifest_sha256,
            "source_identity_sha256": state.source_identity_sha256,
            "selection_feature_sha256": selection_feature_sha256,
            "selection_histories_sha256": selection_histories_sha256,
            "selection_task_sha256": selection_task_sha256,
            "private_outcome_free_artifact_sha256": private_artifact_sha256,
            "execution_environment_sha256": state.execution_environment_sha256,
            "selection_live_lineage_sha256": _require_sha256(
                selection_live_lineage_sha256,
                "selection_live_lineage_sha256",
            ),
            "production_run_claim_sha256": state.production_run_claim_sha256,
        },
        "completion_contract": {
            "protocol": HISTORY_STAGED_PROTOCOL,
            "seeds": list(EXPECTED_SEEDS),
            "outer_folds": state.checkpoint_manifest.outer_folds,
            "fit_query_count": len(state.fit_outcome.query_indices),
            "selection_query_count": int(selection_rows),
            "selection_task_count": int(selection_task_count),
            "strict_same_group_same_speaker_past_history": True,
            "bidirectional_different_set_contexts_predicted": True,
            "complete_checkpoint_only": True,
            "selection_feature_materialized": True,
            "selection_label_materialized": False,
            "selection_label_deserialized": False,
            "selection_utility_target_computed": False,
            "evaluate_stage_run": False,
            "performance_metric_computed": False,
            "production_receipt": state.production_trainer,
            "trainer_mode": (
                PRODUCTION_TRAINER_MODE
                if state.production_trainer
                else SYNTHETIC_TRAINER_MODE
            ),
        },
        "public_artifact_policy": {
            "aggregate_only": True,
            "contains_row_level_values": False,
            "contains_private_paths": False,
            "contains_probabilities_targets_or_performance": False,
        },
    }
    _validate_aggregate_producer_receipt(receipt)
    return receipt


def complete_history_selection_outcomes(
    *,
    fit: FitRoleView,
    selection: VerifiedHistorySelectionView,
    fit_map: FitProtocolMap,
    fit_state: VerifiedHistoryFitState,
    fit_preflight_receipt_path: str | Path,
    expected_fit_preflight_receipt_sha256: str,
    checkpoint_root: str | Path,
    model_config: CausalBackboneConfig,
    run_config: BackboneRunConfig,
    utility_config: UtilitySamplingConfig,
    config_sha256: Mapping[str, str],
    code_sha256: Mapping[str, str],
    runtime_environment_sha256: str,
    execution_environment_sha256: str | None = None,
    config_paths: Mapping[str, str | Path],
    code_paths: Mapping[str, str | Path],
    environment: Mapping[str, object],
    execution_environment: Mapping[str, object] | None = None,
    sidecar_dir: str | Path,
    manifest_path: str | Path,
    selection_feature_sha256: str,
    artifact_path: str | Path,
    completion_receipt_path: str | Path,
    device: torch.device,
) -> CompletedHistoryProduction:
    """Complete feature-only selection inference using immutable checkpoints."""

    # Revalidate fit state before inspecting even the selection capability flag.
    _assert_fit_state_unchanged(
        fit_state,
        checkpoint_root=checkpoint_root,
        fit_preflight_receipt_path=fit_preflight_receipt_path,
    )
    verify_complete_history_checkpoint_payloads(
        checkpoint_root,
        fit_state.checkpoint_manifest,
        fit=fit,
        model_config=model_config,
        run_config=run_config,
        source_identity_sha256=fit_state.source_identity_sha256,
        fold_by_seed_query=fit_state.fit_outcome.fold_by_seed_query,
    )
    live_config, live_code, live_runtime, live_lineage = _live_lineage(
        config_paths=config_paths,
        code_paths=code_paths,
        environment=(
            environment if execution_environment is None else execution_environment
        ),
    )
    declared_execution = _require_sha256(
        runtime_environment_sha256
        if execution_environment_sha256 is None
        else execution_environment_sha256,
        "execution_environment_sha256",
    )
    if (
        live_config != _normalized_hash_mapping(config_sha256, "config_sha256")
        or live_code != _normalized_hash_mapping(code_sha256, "code_sha256")
        or live_runtime
        != declared_execution
        or declared_execution != fit_state.execution_environment_sha256
    ):
        raise HistoryStagedPipelineError("live config/code/runtime lineage changed")
    verify_selection_feature_receipt_inputs(
        receipt_path=fit_preflight_receipt_path,
        expected_receipt_sha256=fit_state.fit_preflight_receipt_sha256,
        dataset=fit.dataset,
        sidecar_dir=sidecar_dir,
        manifest_path=manifest_path,
        config_paths=config_paths,
        code_paths=code_paths,
        environment=environment,
    )
    receipt, preflight_sha = _verify_preflight_and_fit_map(
        fit=fit,
        fit_map=fit_map,
        fit_preflight_receipt_path=fit_preflight_receipt_path,
        expected_fit_preflight_receipt_sha256=expected_fit_preflight_receipt_sha256,
        config_sha256=config_sha256,
        code_sha256=code_sha256,
        runtime_environment_sha256=runtime_environment_sha256,
    )
    if preflight_sha != fit_state.fit_preflight_receipt_sha256:
        raise HistoryStagedPipelineError("completion uses another preflight receipt")
    # Re-open the exact fit probability artifact after the byte-level gate.
    observed_fit = load_history_fit_outcome_view(
        fit_state.fit_outcome_path,
        fit=fit,
        checkpoint_manifest=fit_state.checkpoint_manifest,
    )
    scalar_pairs = (
        (observed_fit.source_identity_sha256, fit_state.source_identity_sha256),
        (observed_fit.artifact_sha256, fit_state.fit_outcome_sha256),
        (observed_fit.model_config_sha256, _config_sha256(model_config)),
        (observed_fit.run_config_sha256, _config_sha256(run_config)),
        (observed_fit.utility_config_sha256, _config_sha256(utility_config)),
    )
    if any(left != right for left, right in scalar_pairs) or not np.array_equal(
        observed_fit.fold_by_seed_query, fit_state.fit_outcome.fold_by_seed_query
    ):
        raise HistoryStagedPipelineError("fit outcome state changed before completion")
    if not isinstance(selection, VerifiedHistorySelectionView):
        raise HistoryStagedPipelineError(
            "complete-selection requires the post-checkpoint feature capability"
        )
    if (
        selection.fit_gate_sha256 != _fit_gate_sha256(fit_state)
        or selection.checkpoint_manifest_sha256
        != fit_state.checkpoint_manifest.manifest_sha256
        or selection.feature_file_sha256 != fit_state.selection_feature_sha256
        or selection.normalized_feature_sha256
        != _selection_feature_contract_sha256(selection.view)
        or dict(selection.config_sha256) != live_config
        or dict(selection.code_sha256) != live_code
        or selection.runtime_environment_sha256 != live_runtime
        or selection.live_lineage_sha256 != live_lineage
    ):
        raise HistoryStagedPipelineError("verified selection feature capability changed")
    selection_live_lineage_sha = selection.live_lineage_sha256
    selection = selection.view
    if selection.labels_materialized:
        raise HistoryStagedPipelineError("selection labels entered complete-selection")
    if selection.dataset != fit.dataset:
        raise HistoryStagedPipelineError("fit and selection datasets differ")
    if set(np.asarray(fit.groups).astype(str)) & set(
        np.asarray(selection.groups).astype(str)
    ):
        raise HistoryStagedPipelineError("fit and selection share a group")
    if set(np.asarray(fit.protocol_row_ids, dtype=np.int64).tolist()) & set(
        np.asarray(selection.protocol_row_ids, dtype=np.int64).tolist()
    ):
        raise HistoryStagedPipelineError("fit and selection share a protocol row")
    declared_selection_sha = _require_sha256(
        selection_feature_sha256, "selection_feature_sha256"
    )
    sidecars = receipt.get("sidecars")
    selection_sidecar = (
        sidecars.get(SELECTION_ROLE) if isinstance(sidecars, Mapping) else None
    )
    if (
        declared_selection_sha != fit_state.selection_feature_sha256
        or not isinstance(selection_sidecar, Mapping)
        or selection_sidecar.get("feature_sha256") != declared_selection_sha
    ):
        raise HistoryStagedPipelineError("selection feature lineage changed")
    destination = Path(artifact_path)
    receipt_destination = Path(completion_receipt_path)
    if fit_state.production_trainer:
        private_root = _validate_production_private_layout(
            checkpoint_root=checkpoint_root,
            fit_outcome=fit_state.fit_outcome_path,
            fit_targets=fit_state.fit_targets_path,
            fit_receipt=fit_state.fit_receipt_path,
            complete_outcome=destination,
            complete_receipt=receipt_destination,
        )
        if fit_state.private_output_root != private_root:
            raise HistoryStagedPipelineError("completion uses another private root")
    if destination.exists() or receipt_destination.exists():
        raise FileExistsError("history complete-selection output already exists")
    if fit_state.checkpoint_manifest.outer_folds != int(run_config.outer_folds):
        raise HistoryStagedPipelineError("checkpoint folds differ from run config")
    # The task sampler consumes feature/history identities only.  The base
    # corpus has zero placeholder labels and is never scored.
    base_selection_corpus = _selection_corpus_from_view(
        selection,
        fit=fit,
        model_config=model_config,
        fit_speaker_reference_indices=np.arange(fit.rows, dtype=np.int64),
    )
    selection_tasks = _tasks_for_role(base_selection_corpus, utility_config)
    if not selection_tasks:
        raise HistoryStagedPipelineError(
            "history selection utility task set must be non-empty"
        )
    selection_histories_sha = _canonical_sha256(
        [list(row) for row in base_selection_corpus.histories]
    )
    classes = len(fit.label_order)
    selection_endpoint = np.zeros(
        (
            len(EXPECTED_SEEDS),
            len(selection.texts),
            len(ENDPOINT_CONTEXT_NAMES),
            classes,
        ),
        dtype=np.float64,
    )
    selection_utility = np.zeros(
        (
            len(EXPECTED_SEEDS),
            len(selection_tasks),
            len(UTILITY_CONTEXT_NAMES),
            classes,
        ),
        dtype=np.float64,
    )
    local_selection_queries = np.arange(len(selection.texts), dtype=np.int64)
    for seed_index, seed in enumerate(EXPECTED_SEEDS):
        for fold in range(int(run_config.outer_folds)):
            held = np.flatnonzero(
                fit_state.fit_outcome.fold_by_seed_query[seed_index] == fold
            ).astype(np.int64)
            train = np.flatnonzero(
                fit_state.fit_outcome.fold_by_seed_query[seed_index] != fold
            ).astype(np.int64)
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
                validation_fraction=run_config.inner_validation_fraction,
                seed=seed,
                fold=fold,
            )
            verify_checkpoint_manifest(checkpoint_root, fit_state.checkpoint_manifest)
            trained = train_one_fold_seed(
                fit_corpus,
                split,
                model_config=model_config,
                run_config=run_config,
                seed=seed,
                source_identity=fit_state.source_identity_sha256,
                checkpoint_root=Path(checkpoint_root),
                device=device,
                require_complete_checkpoint=True,
            )
            if (
                trained.summary.get("resumed_complete_checkpoint") is not True
                or trained.summary.get("resumed_partial_checkpoint") is not False
            ):
                raise HistoryStagedPipelineError(
                    "selection inference did not restore a complete checkpoint"
                )
            selection_text = trained.processor.transform(selection_corpus.texts)
            fold_endpoint = predict_current_and_all_history(
                trained.model,
                selection_corpus,
                selection_text,
                local_selection_queries,
                device=device,
                batch_size=run_config.inference_batch_size,
                max_history_items=run_config.max_history_items,
            )
            fold_utility = predict_utility_contexts(
                trained.model,
                selection_corpus,
                selection_text,
                selection_tasks,
                device=device,
                batch_size=run_config.inference_batch_size,
                max_history_items=run_config.max_history_items,
            )
            selection_endpoint[seed_index] += _probability(
                fold_endpoint,
                (
                    len(selection.texts),
                    len(ENDPOINT_CONTEXT_NAMES),
                    classes,
                ),
                "selection_fold_endpoint_probability",
            )
            selection_utility[seed_index] += _probability(
                fold_utility,
                (
                    len(selection_tasks),
                    len(UTILITY_CONTEXT_NAMES),
                    classes,
                ),
                "selection_fold_utility_probability",
            )
            verify_checkpoint_manifest(checkpoint_root, fit_state.checkpoint_manifest)
        selection_endpoint[seed_index] /= float(run_config.outer_folds)
        selection_utility[seed_index] /= float(run_config.outer_folds)
    selection_endpoint = _probability(
        selection_endpoint,
        (
            len(EXPECTED_SEEDS),
            len(selection.texts),
            len(ENDPOINT_CONTEXT_NAMES),
            classes,
        ),
        "selection_endpoint_probability_fold_ensemble",
    )
    selection_utility = _probability(
        selection_utility,
        (
            len(EXPECTED_SEEDS),
            len(selection_tasks),
            len(UTILITY_CONTEXT_NAMES),
            classes,
        ),
        "selection_utility_probability_fold_ensemble",
    )
    _assert_fit_state_unchanged(
        fit_state,
        checkpoint_root=checkpoint_root,
        fit_preflight_receipt_path=fit_preflight_receipt_path,
    )
    after_config, after_code, after_runtime, after_lineage = _live_lineage(
        config_paths=config_paths,
        code_paths=code_paths,
        environment=(
            environment if execution_environment is None else execution_environment
        ),
    )
    if (
        after_config != live_config
        or after_code != live_code
        or after_runtime != live_runtime
        or after_lineage != live_lineage
    ):
        raise HistoryStagedPipelineError(
            "config/code/runtime changed during complete-selection inference"
        )
    verify_selection_feature_receipt_inputs(
        receipt_path=fit_preflight_receipt_path,
        expected_receipt_sha256=fit_state.fit_preflight_receipt_sha256,
        dataset=fit.dataset,
        sidecar_dir=sidecar_dir,
        manifest_path=manifest_path,
        config_paths=config_paths,
        code_paths=code_paths,
        environment=environment,
    )
    complete_values: dict[str, np.ndarray] = {
        "schema_version": np.asarray(HISTORY_COMPLETE_OUTCOME_SCHEMA),
        "dataset": np.asarray(fit.dataset),
        "dataset_label_order": np.asarray(fit.label_order),
        "seeds": np.asarray(EXPECTED_SEEDS, dtype=np.int64),
        "fit_protocol_row_ids": np.asarray(fit.protocol_row_ids, dtype=np.int64),
        "selection_protocol_row_ids": np.asarray(
            selection.protocol_row_ids, dtype=np.int64
        ),
        "fit_cluster_codes": _cluster_codes(fit.groups),
        "selection_cluster_codes": _cluster_codes(selection.groups),
        "fit_histories_sha256": np.asarray(fit_state.fit_outcome.histories_sha256),
        "selection_histories_sha256": np.asarray(selection_histories_sha),
        "fit_endpoint_probability_oof": np.asarray(
            fit_state.fit_outcome.endpoint_probability_oof, dtype=np.float32
        ),
        "selection_endpoint_probability_fold_ensemble": selection_endpoint,
        "fit_utility_probability_oof": np.asarray(
            fit_state.fit_outcome.utility_probability_oof, dtype=np.float32
        ),
        "selection_utility_probability_fold_ensemble": selection_utility,
        "source_identity_sha256": np.asarray(fit_state.source_identity_sha256),
        "checkpoint_manifest_sha256": np.asarray(
            fit_state.checkpoint_manifest.manifest_sha256
        ),
        "fit_outcome_artifact_sha256": np.asarray(fit_state.fit_outcome_sha256),
        **_task_artifact_values(
            "fit",
            tuple(
                BidirectionalCoalitionTask(
                    int(query),
                    addition,
                    deletion,
                    int(candidate),
                )
                for query, candidate, addition, deletion in zip(
                    fit_state.fit_outcome.tasks.query_indices,
                    fit_state.fit_outcome.tasks.candidate_indices,
                    fit_state.fit_outcome.tasks.addition_contexts,
                    fit_state.fit_outcome.tasks.deletion_contexts,
                    strict=True,
                )
            ),
        ),
        **_task_artifact_values("selection", selection_tasks),
    }
    for name in (
        "fit_endpoint_probability_oof",
        "selection_endpoint_probability_fold_ensemble",
        "fit_utility_probability_oof",
        "selection_utility_probability_fold_ensemble",
    ):
        complete_values[f"matrix_{name}_sha256"] = np.asarray(
            _array_sha256(complete_values[name])
        )
    _outcome_free_view_from_values(
        complete_values,
        fit=fit,
        selection=selection,
        state=fit_state,
        artifact_sha256="0" * 64,
    )
    artifact_sha = _write_npz_once(destination, complete_values)
    _assert_fit_state_unchanged(
        fit_state,
        checkpoint_root=checkpoint_root,
        fit_preflight_receipt_path=fit_preflight_receipt_path,
    )
    completion_receipt = _completion_receipt_payload(
        dataset=fit.dataset,
        state=fit_state,
        fit_map=fit_map,
        selection_feature_sha256=declared_selection_sha,
        selection_histories_sha256=selection_histories_sha,
        selection_task_sha256=_task_sha256(selection_tasks),
        private_artifact_sha256=artifact_sha,
        selection_rows=len(selection.texts),
        selection_task_count=len(selection_tasks),
        selection_live_lineage_sha256=selection_live_lineage_sha,
    )
    receipt_sha = _write_json_once(receipt_destination, completion_receipt)
    return CompletedHistoryProduction(
        artifact_path=destination.resolve(),
        artifact_sha256=artifact_sha,
        receipt_path=receipt_destination.resolve(),
        receipt_sha256=receipt_sha,
        checkpoint_manifest_sha256=fit_state.checkpoint_manifest.manifest_sha256,
    )


def _receipt_count(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or int(value) < int(minimum):
        raise HistoryStagedPipelineError(
            f"history completion receipt {field} must be an integer >= {minimum}"
        )
    return int(value)


def verify_history_completion_production_attestation(
    artifact_path: str | Path,
    completion_receipt_path: str | Path,
    expected_completion_receipt_sha256: str,
) -> VerifiedHistoryCompletionAttestation:
    """Verify one canonical production completion without opening raw sidecars.

    This downstream gate intentionally accepts no selection feature or label path.
    It binds the completed outcome-free cache to its production fit receipt, private
    run claim, and the checkpoint files still present under the same private root.
    """

    artifact = Path(artifact_path).resolve()
    receipt_path = Path(completion_receipt_path).resolve()
    root = artifact.parent
    paths = _production_private_paths(root)
    _validate_production_private_layout(
        checkpoint_root=paths["checkpoint"],
        fit_outcome=paths["fit_outcome"],
        fit_targets=paths["fit_targets"],
        fit_receipt=paths["fit_receipt"],
        complete_outcome=artifact,
        complete_receipt=receipt_path,
    )
    expected_receipt_sha = _require_sha256(
        expected_completion_receipt_sha256,
        "expected_completion_receipt_sha256",
    )
    if _file_sha256(receipt_path) != expected_receipt_sha:
        raise HistoryStagedPipelineError("history completion receipt file hash changed")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HistoryStagedPipelineError(
            f"cannot read history completion receipt: {error}"
        ) from error
    top_keys = {
        "schema_version",
        "status",
        "dataset",
        "claim_boundary",
        "lineage",
        "completion_contract",
        "public_artifact_policy",
    }
    if not isinstance(receipt, dict) or set(receipt) != top_keys:
        raise HistoryStagedPipelineError("history completion receipt schema changed")
    if (
        receipt.get("schema_version") != HISTORY_COMPLETE_RECEIPT_SCHEMA
        or receipt.get("status") != PRODUCTION_COMPLETION_STATUS
        or receipt.get("claim_boundary")
        != (
            "Fit OOF plus model-selection feature-only history inference; no "
            "model-selection label, utility target, or performance metric was consumed."
        )
    ):
        raise HistoryStagedPipelineError(
            "history completion receipt is not canonical production evidence"
        )
    dataset = receipt.get("dataset")
    if dataset not in {"EmotionTalk", "MELD"}:
        raise HistoryStagedPipelineError("history completion dataset changed")
    lineage = receipt.get("lineage")
    contract = receipt.get("completion_contract")
    public = receipt.get("public_artifact_policy")
    lineage_keys = {
        "fit_preflight_receipt_sha256",
        "fit_producer_receipt_sha256",
        "fit_protocol_map_sha256",
        "fit_outcome_artifact_sha256",
        "fit_targets_artifact_sha256",
        "checkpoint_manifest_sha256",
        "source_identity_sha256",
        "selection_feature_sha256",
        "selection_histories_sha256",
        "selection_task_sha256",
        "private_outcome_free_artifact_sha256",
        "execution_environment_sha256",
        "selection_live_lineage_sha256",
        "production_run_claim_sha256",
    }
    contract_keys = {
        "protocol",
        "seeds",
        "outer_folds",
        "fit_query_count",
        "selection_query_count",
        "selection_task_count",
        "strict_same_group_same_speaker_past_history",
        "bidirectional_different_set_contexts_predicted",
        "complete_checkpoint_only",
        "selection_feature_materialized",
        "selection_label_materialized",
        "selection_label_deserialized",
        "selection_utility_target_computed",
        "evaluate_stage_run",
        "performance_metric_computed",
        "production_receipt",
        "trainer_mode",
    }
    if (
        not isinstance(lineage, dict)
        or set(lineage) != lineage_keys
        or not isinstance(contract, dict)
        or set(contract) != contract_keys
    ):
        raise HistoryStagedPipelineError(
            "history completion receipt nested schema changed"
        )
    if public != {
        "aggregate_only": True,
        "contains_row_level_values": False,
        "contains_private_paths": False,
        "contains_probabilities_targets_or_performance": False,
    }:
        raise HistoryStagedPipelineError("history completion public policy changed")
    required_true = {
        "strict_same_group_same_speaker_past_history",
        "bidirectional_different_set_contexts_predicted",
        "complete_checkpoint_only",
        "selection_feature_materialized",
        "production_receipt",
    }
    required_false = {
        "selection_label_materialized",
        "selection_label_deserialized",
        "selection_utility_target_computed",
        "evaluate_stage_run",
        "performance_metric_computed",
    }
    if (
        contract.get("protocol") != HISTORY_STAGED_PROTOCOL
        or contract.get("trainer_mode") != PRODUCTION_TRAINER_MODE
        or contract.get("seeds") != list(EXPECTED_SEEDS)
        or any(contract.get(name) is not True for name in required_true)
        or any(contract.get(name) is not False for name in required_false)
    ):
        raise HistoryStagedPipelineError(
            "history completion receipt is not production-attested"
        )
    outer_folds = _receipt_count(contract.get("outer_folds"), "outer_folds", minimum=2)
    fit_rows = _receipt_count(contract.get("fit_query_count"), "fit_query_count", minimum=1)
    selection_rows = _receipt_count(
        contract.get("selection_query_count"), "selection_query_count", minimum=1
    )
    selection_tasks = _receipt_count(
        contract.get("selection_task_count"), "selection_task_count", minimum=1
    )
    lineage_sha = {
        name: _require_sha256(lineage.get(name), name) for name in lineage_keys
    }
    claim_sha = lineage_sha["production_run_claim_sha256"]

    artifact_sha = _file_sha256(artifact)
    if artifact_sha != lineage_sha["private_outcome_free_artifact_sha256"]:
        raise HistoryStagedPipelineError("history completion artifact hash changed")
    values = _load_npz(artifact)
    if set(values) != set(_COMPLETE_OUTCOME_KEYS):
        raise HistoryStagedPipelineError("history completion artifact schema changed")
    label_order = np.asarray(values["dataset_label_order"])
    if (
        _single_text(values["schema_version"], "schema_version")
        != HISTORY_COMPLETE_OUTCOME_SCHEMA
        or _single_text(values["dataset"], "dataset") != dataset
        or label_order.ndim != 1
        or label_order.dtype.kind not in {"U", "S"}
        or len(label_order) < 2
        or tuple(_integer_vector(values["seeds"], "seeds", unique=True))
        != EXPECTED_SEEDS
    ):
        raise HistoryStagedPipelineError("history completion artifact identity changed")
    classes = len(label_order)
    fit_protocol = _integer_vector(
        values["fit_protocol_row_ids"], "fit_protocol_row_ids", unique=True
    )
    selection_protocol = _integer_vector(
        values["selection_protocol_row_ids"],
        "selection_protocol_row_ids",
        unique=True,
    )
    fit_task_count = len(
        _integer_vector(values["fit_task_query_indices"], "fit_task_query_indices")
    )
    observed_selection_task_count = len(
        _integer_vector(
            values["selection_task_query_indices"], "selection_task_query_indices"
        )
    )
    if (
        len(fit_protocol) != fit_rows
        or len(selection_protocol) != selection_rows
        or fit_task_count < 1
        or observed_selection_task_count != selection_tasks
    ):
        raise HistoryStagedPipelineError("history completion artifact counts changed")
    probability_shapes = {
        "fit_endpoint_probability_oof": (
            len(EXPECTED_SEEDS),
            fit_rows,
            len(ENDPOINT_CONTEXT_NAMES),
            classes,
        ),
        "selection_endpoint_probability_fold_ensemble": (
            len(EXPECTED_SEEDS),
            selection_rows,
            len(ENDPOINT_CONTEXT_NAMES),
            classes,
        ),
        "fit_utility_probability_oof": (
            len(EXPECTED_SEEDS),
            fit_task_count,
            len(UTILITY_CONTEXT_NAMES),
            classes,
        ),
        "selection_utility_probability_fold_ensemble": (
            len(EXPECTED_SEEDS),
            selection_tasks,
            len(UTILITY_CONTEXT_NAMES),
            classes,
        ),
    }
    for name, shape in probability_shapes.items():
        _probability(values[name], shape, name)
        matrix_field = f"matrix_{name}_sha256"
        if _require_sha256(
            _single_text(values[matrix_field], matrix_field), matrix_field
        ) != _array_sha256(np.asarray(values[name])):
            raise HistoryStagedPipelineError(
                f"history completion probability hash changed: {name}"
            )
    artifact_lineage = {
        "source_identity_sha256": _single_text(
            values["source_identity_sha256"], "source_identity_sha256"
        ),
        "checkpoint_manifest_sha256": _single_text(
            values["checkpoint_manifest_sha256"], "checkpoint_manifest_sha256"
        ),
        "fit_outcome_artifact_sha256": _single_text(
            values["fit_outcome_artifact_sha256"], "fit_outcome_artifact_sha256"
        ),
    }
    if any(
        _require_sha256(value, name) != lineage_sha[name]
        for name, value in artifact_lineage.items()
    ):
        raise HistoryStagedPipelineError("history completion artifact lineage changed")
    if _file_sha256(artifact) != artifact_sha:
        raise HistoryStagedPipelineError(
            "history completion artifact changed while validating"
        )

    try:
        checkpoint_manifest = build_checkpoint_manifest(
            paths["checkpoint"], seeds=EXPECTED_SEEDS, outer_folds=outer_folds
        )
    except (OSError, ValueError) as error:
        raise HistoryStagedPipelineError(
            f"history production checkpoint manifest is invalid: {error}"
        ) from error
    if checkpoint_manifest.manifest_sha256 != lineage_sha[
        "checkpoint_manifest_sha256"
    ]:
        raise HistoryStagedPipelineError("history production checkpoint manifest changed")
    fit_receipt_sha = lineage_sha["fit_producer_receipt_sha256"]
    fit_receipt = _load_history_fit_receipt(
        paths["fit_receipt"],
        expected_sha256=fit_receipt_sha,
        expected={
            "fit_preflight_receipt_sha256": lineage_sha[
                "fit_preflight_receipt_sha256"
            ],
            "fit_protocol_map_sha256": lineage_sha["fit_protocol_map_sha256"],
            "execution_environment_sha256": lineage_sha[
                "execution_environment_sha256"
            ],
            "source_identity_sha256": lineage_sha["source_identity_sha256"],
            "checkpoint_manifest_sha256": lineage_sha[
                "checkpoint_manifest_sha256"
            ],
            "private_fit_outcome_artifact_sha256": lineage_sha[
                "fit_outcome_artifact_sha256"
            ],
            "private_fit_targets_artifact_sha256": lineage_sha[
                "fit_targets_artifact_sha256"
            ],
            "production_run_claim_sha256": claim_sha,
        },
    )
    fit_lineage = fit_receipt.get("lineage")
    fit_contract = fit_receipt.get("training_contract")
    fit_lineage_keys = {
        "fit_preflight_receipt_sha256",
        "fit_protocol_map_sha256",
        "config_sha256",
        "code_sha256",
        "runtime_environment_sha256",
        "execution_environment_sha256",
        "model_config_sha256",
        "run_config_sha256",
        "utility_config_sha256",
        "source_identity_sha256",
        "fold_assignment_sha256",
        "fit_task_sha256",
        "checkpoint_manifest_sha256",
        "private_fit_outcome_artifact_sha256",
        "private_fit_targets_artifact_sha256",
        "production_run_claim_sha256",
    }
    fit_contract_keys = {
        "protocol",
        "seeds",
        "outer_folds",
        "checkpoint_file_count",
        "fit_query_count",
        "fit_task_count",
        "strict_same_group_same_speaker_past_history",
        "bidirectional_different_set_utility",
        "one_oof_endpoint_probability_per_seed_and_fit_query",
        "one_oof_utility_context_probability_per_seed_and_fit_task",
        "heldout_outcomes_materialized_in_fold_callback",
        "selection_payload_consumed",
        "performance_metric_computed",
        "trainer_mode",
        "production_receipt",
    }
    if (
        fit_receipt.get("dataset") != dataset
        or not isinstance(fit_lineage, dict)
        or set(fit_lineage) != fit_lineage_keys
        or not isinstance(fit_contract, dict)
        or set(fit_contract) != fit_contract_keys
        or fit_contract.get("protocol") != HISTORY_STAGED_PROTOCOL
        or fit_contract.get("trainer_mode") != PRODUCTION_TRAINER_MODE
        or fit_contract.get("production_receipt") is not True
        or fit_contract.get("seeds") != list(EXPECTED_SEEDS)
        or fit_contract.get("outer_folds") != outer_folds
        or fit_contract.get("checkpoint_file_count")
        != len(checkpoint_manifest.records)
        or fit_contract.get("fit_query_count") != fit_rows
        or fit_contract.get("fit_task_count") != fit_task_count
        or fit_lineage.get("production_run_claim_sha256") != claim_sha
    ):
        raise HistoryStagedPipelineError(
            "history fit receipt is not the claimed production source"
        )
    fit_config_sha = _normalized_hash_mapping(
        fit_lineage.get("config_sha256"), "config_sha256"
    )
    fit_code_sha = _normalized_hash_mapping(
        fit_lineage.get("code_sha256"), "code_sha256"
    )
    fit_runtime_sha = _require_sha256(
        fit_lineage.get("runtime_environment_sha256"),
        "runtime_environment_sha256",
    )
    fit_model_sha = _require_sha256(
        fit_lineage.get("model_config_sha256"), "model_config_sha256"
    )
    fit_run_sha = _require_sha256(
        fit_lineage.get("run_config_sha256"), "run_config_sha256"
    )
    fit_utility_sha = _require_sha256(
        fit_lineage.get("utility_config_sha256"), "utility_config_sha256"
    )
    if (
        _file_sha256(paths["fit_outcome"])
        != lineage_sha["fit_outcome_artifact_sha256"]
        or _file_sha256(paths["fit_targets"])
        != lineage_sha["fit_targets_artifact_sha256"]
    ):
        raise HistoryStagedPipelineError("history canonical fit artifacts changed")
    _verify_private_claim(root, claim_sha)
    try:
        verify_checkpoint_manifest(paths["checkpoint"], checkpoint_manifest)
    except (OSError, ValueError) as error:
        raise HistoryStagedPipelineError(
            f"history production checkpoint manifest changed: {error}"
        ) from error
    if (
        _file_sha256(receipt_path) != expected_receipt_sha
        or _file_sha256(artifact) != artifact_sha
        or _file_sha256(paths["fit_receipt"]) != fit_receipt_sha
        or _file_sha256(paths["fit_outcome"])
        != lineage_sha["fit_outcome_artifact_sha256"]
        or _file_sha256(paths["fit_targets"])
        != lineage_sha["fit_targets_artifact_sha256"]
    ):
        raise HistoryStagedPipelineError(
            "history production lineage changed while attesting completion"
        )
    _verify_private_claim(root, claim_sha)
    return VerifiedHistoryCompletionAttestation(
        dataset=str(dataset),
        artifact_path=artifact,
        artifact_sha256=artifact_sha,
        completion_receipt_path=receipt_path,
        completion_receipt_sha256=expected_receipt_sha,
        fit_producer_receipt_path=paths["fit_receipt"],
        fit_producer_receipt_sha256=fit_receipt_sha,
        source_identity_sha256=lineage_sha["source_identity_sha256"],
        checkpoint_manifest_sha256=checkpoint_manifest.manifest_sha256,
        production_run_claim_sha256=claim_sha,
        fit_preflight_receipt_sha256=lineage_sha[
            "fit_preflight_receipt_sha256"
        ],
        config_sha256=fit_config_sha,
        code_sha256=fit_code_sha,
        runtime_environment_sha256=fit_runtime_sha,
        execution_environment_sha256=lineage_sha[
            "execution_environment_sha256"
        ],
        model_config_sha256=fit_model_sha,
        run_config_sha256=fit_run_sha,
        utility_config_sha256=fit_utility_sha,
    )
