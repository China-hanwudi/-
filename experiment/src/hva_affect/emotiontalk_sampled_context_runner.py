"""Open-role runner for aggregate sampled-context classification diagnostics.

The expensive base-model probabilities already exist in fold checkpoints.  The
runner therefore never fits or recomputes a base classifier and never opens the
EmotionTalk label archive.  It reads only train-corpus key/split metadata from
the feature archive, rebuilds the deterministic open-role task order, verifies
the checkpoint cover, and proves that the resulting 59-D task matrices are
bitwise identical to the private utility cache.

Selection labels are recovered from the already-materialized utility targets:
for each query, exactly one of the seven classes must reproduce both cached
forward and backward NLL utilities across all its sampled tasks.  This avoids
materializing labels for calibration or internal-holdout roles from the
monolithic train label payload.  Recovered labels are used only for aggregate
diagnostics, never as model features.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .bidirectional_emotion_utility import sample_bidirectional_coalition_tasks
from .bidirectional_utility_model import (
    DEFAULT_SEEDS,
    PRIMARY_HISTORY_COVERAGE,
    BidirectionalUtilityCache,
    UtilityPredictions,
    default_model_specs,
    fit_oof_coverage_threshold,
    fit_utility_model,
    group_oof_predictions,
    load_private_oof_cache,
)
from .data_contract import ContractError, write_json_atomic
from .emotion_probability_relations import (
    bidirectional_task_order_sha256,
    emotion_class_order_sha256,
    emotion_context_schema_sha256,
    ordered_source_sha256,
)
from .emotiontalk_bidirectional_oof import (
    CONTEXT_NAMES,
    _read_json,
    _role_ranges,
    _validate_config,
    probability_task_features,
)
from .emotiontalk_contract import parse_key
from .emotiontalk_endpoint_diagnostic import assign_frame_roles
from .emotiontalk_multimodal_external import EXPECTED_ARCHIVE_FIELDS
from .emotiontalk_text_p1 import LABEL_NAMES, build_history_indices
from .meld_text_pilot import sha256_file
from .sampled_context_classification import (
    DiagnosticProvenance,
    FrozenUtilityDecision,
    SampledContextInputs,
    paired_sampled_context_model_contrast,
    sampled_context_classification_diagnostic,
)


CHECKPOINT_SCHEMA_VERSION = "emotiontalk_bidirectional_probability_checkpoint_v1"
REPORT_SCHEMA_VERSION = "emotiontalk_open_role_sampled_context_diagnostic_v1"
CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "task_positions",
        "probability",
        "base_config_sha256",
        "utility_config_sha256",
        "feature_sha256",
    }
)
EXPECTED_MODEL_NAMES = (
    "forward_only_mlp",
    "backward_only_mlp",
    "pseudo_bidirectional_same_set_mlp",
    "bidirectional_shared_mlp",
)
REFERENCE_MODEL_NAMES = EXPECTED_MODEL_NAMES[:-1]
FORBIDDEN_ROLE_TOKENS = frozenset(
    {"calibration", "holdout", "sealed", "validation", "test"}
)


class OpenRoleDiagnosticError(ContractError):
    """Raised when an open-role diagnostic contract cannot be proven."""


@dataclass(frozen=True)
class ProbabilityCheckpoint:
    positions: np.ndarray
    probability: np.ndarray


@dataclass(frozen=True)
class OpenRoleTasks:
    histories: tuple[tuple[int, ...], ...]
    fit_tasks: tuple[object, ...]
    selection_tasks: tuple[object, ...]
    fit_cluster_codes: np.ndarray
    selection_cluster_codes: np.ndarray
    expected_fold_positions: Mapping[str, np.ndarray]
    role_counts: Mapping[str, int]
    train_key_manifest_sha256: str
    dataset_identifier: str
    source_order_sha256: str
    selection_fit_assignment_sha256: str


def _single_string(value: np.ndarray, *, field: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise OpenRoleDiagnosticError(f"{field} must contain exactly one string")
    return str(array.reshape(-1)[0])


def _expected_hashes(
    base_config_path: Path,
    utility_config_path: Path,
    feature_path: Path,
) -> dict[str, str]:
    return {
        "base_config_sha256": sha256_file(base_config_path),
        "utility_config_sha256": sha256_file(utility_config_path),
        "feature_sha256": sha256_file(feature_path),
    }


def read_probability_checkpoint(
    path: Path,
    *,
    expected_hashes: Mapping[str, str],
    expected_seed_count: int,
) -> ProbabilityCheckpoint:
    """Load one whitelisted checkpoint without accepting any extra field."""

    checkpoint_path = Path(path)
    if checkpoint_path.suffix.lower() != ".npz":
        raise OpenRoleDiagnosticError("probability checkpoint must be an .npz archive")
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        if set(archive.files) != CHECKPOINT_FIELDS:
            raise OpenRoleDiagnosticError(
                f"checkpoint schema changed for {checkpoint_path.name}: {archive.files}"
            )
        if _single_string(archive["schema_version"], field="schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise OpenRoleDiagnosticError(
                f"checkpoint version changed for {checkpoint_path.name}"
            )
        for field in ("base_config_sha256", "utility_config_sha256", "feature_sha256"):
            if _single_string(archive[field], field=field) != str(expected_hashes[field]):
                raise OpenRoleDiagnosticError(
                    f"checkpoint hash mismatch for {field}: {checkpoint_path.name}"
                )
        positions = np.asarray(archive["task_positions"])
        probability = np.asarray(archive["probability"])
    if positions.dtype != np.int64 or positions.ndim != 1:
        raise OpenRoleDiagnosticError(
            f"checkpoint task positions must be one-dimensional int64: {checkpoint_path.name}"
        )
    if len(positions) and (
        np.any(positions < 0) or np.any(np.diff(positions) <= 0)
    ):
        raise OpenRoleDiagnosticError(
            f"checkpoint task positions must be unique, increasing, and non-negative: "
            f"{checkpoint_path.name}"
        )
    expected_shape = (
        int(expected_seed_count),
        len(positions),
        len(CONTEXT_NAMES),
        len(LABEL_NAMES),
    )
    if probability.dtype != np.float64 or probability.shape != expected_shape:
        raise OpenRoleDiagnosticError(
            f"checkpoint probability shape/dtype mismatch for {checkpoint_path.name}; "
            f"expected={expected_shape}/float64 got={probability.shape}/{probability.dtype}"
        )
    if not np.isfinite(probability).all():
        raise OpenRoleDiagnosticError(
            f"checkpoint contains non-finite probability: {checkpoint_path.name}"
        )
    if np.any(probability < -1e-12) or np.any(probability > 1.0 + 1e-12):
        raise OpenRoleDiagnosticError(
            f"checkpoint probabilities lie outside [0, 1]: {checkpoint_path.name}"
        )
    if not np.allclose(probability.sum(axis=3), 1.0, rtol=1e-7, atol=1e-9):
        raise OpenRoleDiagnosticError(
            f"checkpoint probabilities do not sum to one: {checkpoint_path.name}"
        )
    positions = np.array(positions, dtype=np.int64, copy=True)
    probability = np.array(probability, dtype=np.float64, copy=True)
    positions.setflags(write=False)
    probability.setflags(write=False)
    return ProbabilityCheckpoint(positions, probability)


def assemble_fit_probability_checkpoints(
    checkpoint_paths: Sequence[Path],
    *,
    expected_task_count: int,
    expected_seed_count: int,
    expected_hashes: Mapping[str, str],
    expected_positions_by_name: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Assemble fold checkpoints with exact non-overlap and full-cover checks."""

    task_count = int(expected_task_count)
    if task_count < 1:
        raise OpenRoleDiagnosticError("expected fit task count must be positive")
    paths = tuple(Path(path) for path in checkpoint_paths)
    if not paths or len({path.name for path in paths}) != len(paths):
        raise OpenRoleDiagnosticError("fit checkpoint names must be non-empty and unique")
    if expected_positions_by_name is not None and set(expected_positions_by_name) != {
        path.name for path in paths
    }:
        raise OpenRoleDiagnosticError("expected fold-position names differ from checkpoint names")
    result = np.full(
        (
            int(expected_seed_count),
            task_count,
            len(CONTEXT_NAMES),
            len(LABEL_NAMES),
        ),
        np.nan,
        dtype=np.float64,
    )
    covered = np.zeros(task_count, dtype=bool)
    for path in paths:
        checkpoint = read_probability_checkpoint(
            path,
            expected_hashes=expected_hashes,
            expected_seed_count=expected_seed_count,
        )
        positions = checkpoint.positions
        if np.any(positions >= task_count):
            raise OpenRoleDiagnosticError(
                f"fit checkpoint position exceeds task count: {path.name}"
            )
        if expected_positions_by_name is not None:
            expected = np.asarray(expected_positions_by_name[path.name], dtype=np.int64)
            if not np.array_equal(positions, expected):
                raise OpenRoleDiagnosticError(
                    f"fit checkpoint positions differ from deterministic fold: {path.name}"
                )
        if np.any(covered[positions]):
            raise OpenRoleDiagnosticError(
                f"fit checkpoint task positions overlap: {path.name}"
            )
        result[:, positions] = checkpoint.probability
        covered[positions] = True
    missing = np.flatnonzero(~covered)
    if len(missing):
        raise OpenRoleDiagnosticError(
            f"fit checkpoints do not cover all task positions; missing_count={len(missing)}"
        )
    if not np.isfinite(result).all():
        raise AssertionError("fit checkpoint assembly left non-finite cells")
    return result


def load_selection_probability_checkpoint(
    path: Path,
    *,
    expected_task_count: int,
    expected_seed_count: int,
    expected_hashes: Mapping[str, str],
) -> np.ndarray:
    """Load selection checkpoint and require the canonical 0..N-1 task order."""

    checkpoint = read_probability_checkpoint(
        path,
        expected_hashes=expected_hashes,
        expected_seed_count=expected_seed_count,
    )
    expected = np.arange(int(expected_task_count), dtype=np.int64)
    if not np.array_equal(checkpoint.positions, expected):
        raise OpenRoleDiagnosticError(
            "selection checkpoint task positions are not the canonical complete order"
        )
    return np.array(checkpoint.probability, dtype=np.float64, copy=True)


def verify_recomputed_59d_cache(
    cache: BidirectionalUtilityCache,
    *,
    fit_x: np.ndarray,
    fit_feature_names: Sequence[str],
    selection_x: np.ndarray,
    selection_feature_names: Sequence[str],
    fit_cluster_codes: np.ndarray,
    selection_cluster_codes: np.ndarray,
) -> None:
    """Prove task order by exact feature, name, width, and cluster equality."""

    fit = np.asarray(fit_x, dtype=np.float64)
    selection = np.asarray(selection_x, dtype=np.float64)
    fit_names = tuple(str(value) for value in fit_feature_names)
    selection_names = tuple(str(value) for value in selection_feature_names)
    if cache.fit.x.shape[1] != 59 or cache.selection.x.shape[1] != 59:
        raise OpenRoleDiagnosticError("private utility cache is no longer 59-dimensional")
    if fit_names != selection_names or fit_names != cache.feature_names:
        raise OpenRoleDiagnosticError("recomputed and cached feature_names are not identical")
    if fit.shape != cache.fit.x.shape or not np.array_equal(fit, cache.fit.x):
        raise OpenRoleDiagnosticError("recomputed fit 59-D task features do not match cache")
    if selection.shape != cache.selection.x.shape or not np.array_equal(
        selection, cache.selection.x
    ):
        raise OpenRoleDiagnosticError(
            "recomputed selection 59-D task features do not match cache"
        )
    if not np.array_equal(
        np.asarray(fit_cluster_codes, dtype=np.int64), cache.fit.cluster_codes
    ):
        raise OpenRoleDiagnosticError("recomputed fit cluster codes do not match cache")
    if not np.array_equal(
        np.asarray(selection_cluster_codes, dtype=np.int64), cache.selection.cluster_codes
    ):
        raise OpenRoleDiagnosticError("recomputed selection cluster codes do not match cache")


def _integer_codes(values: Sequence[object]) -> np.ndarray:
    ordered = sorted(set(str(value) for value in values))
    mapping = {value: index for index, value in enumerate(ordered)}
    return np.asarray([mapping[str(value)] for value in values], dtype=np.int64)


def _train_keys_only(feature_path: Path) -> tuple[np.ndarray, str]:
    """Read key/split metadata only; audio/video/quality arrays stay unopened."""

    with np.load(feature_path, allow_pickle=False) as archive:
        if set(archive.files) != EXPECTED_ARCHIVE_FIELDS:
            raise OpenRoleDiagnosticError(f"media feature schema changed: {archive.files}")
        keys = archive["keys"].astype(str)
        splits = archive["splits"].astype(str)
        feature_config_sha256 = _single_string(
            archive["config_sha256"], field="feature config hash"
        )
    if len(keys) != len(splits):
        raise OpenRoleDiagnosticError("feature keys and splits are not aligned")
    if np.any(splits == "test_corpus"):
        raise OpenRoleDiagnosticError("test feature rows are forbidden in the artifact")
    allowed_split_names = {"train_corpus", "val_corpus"}
    if not set(splits).issubset(allowed_split_names):
        raise OpenRoleDiagnosticError("feature artifact contains an unknown split")
    selected = keys[splits == "train_corpus"]
    if not len(selected) or len(set(selected)) != len(selected):
        raise OpenRoleDiagnosticError("train-corpus keys are empty or duplicated")
    return selected, feature_config_sha256


def _metadata_frame(keys: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key in keys:
        group, dialogue, speaker, turn = parse_key(str(key))
        rows.append(
            {
                "key": str(key),
                "group": group,
                "dialogue": dialogue,
                "speaker": speaker,
                "turn": int(turn),
            }
        )
    return pd.DataFrame(rows)


def _manifest_sha256(values: Sequence[str]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reconstruct_open_role_tasks(
    feature_path: Path,
    base_config: Mapping[str, object],
    utility_config: Mapping[str, object],
) -> tuple[OpenRoleTasks, str]:
    """Rebuild exactly the task order used by the original OOF generator."""

    keys, feature_config_sha256 = _train_keys_only(feature_path)
    full_frame = _metadata_frame(keys)
    groups_all, roles_all, _ = assign_frame_roles(
        full_frame,
        dataset="emotiontalk",
        role_protocol=str(utility_config["data_roles"]["split_protocol_id"]),
        role_ranges=_role_ranges(utility_config),
    )
    open_roles = {"base_and_utility_fit", "model_selection"}
    materialized = np.asarray([str(role) in open_roles for role in roles_all], dtype=bool)
    if not np.array_equal(
        np.unique(roles_all[materialized]),
        np.asarray(sorted(open_roles), dtype=object),
    ):
        raise OpenRoleDiagnosticError("both open roles must be present")
    work_frame = full_frame.loc[materialized].copy().reset_index(drop=True)
    work_frame["_row_id"] = np.arange(len(work_frame), dtype=np.int64)
    work_groups = groups_all[materialized]
    work_roles = roles_all[materialized]
    fit_indices = np.flatnonzero(work_roles == "base_and_utility_fit")
    selection_indices = np.flatnonzero(work_roles == "model_selection")
    if set(work_groups[fit_indices]) & set(work_groups[selection_indices]):
        raise OpenRoleDiagnosticError("fit and model-selection groups overlap")

    dataset_identifier = (
        "BAAI/Emotiontalk:train_corpus:feature_config_sha256="
        f"{feature_config_sha256}"
    )
    source_order_digest = ordered_source_sha256(
        dataset_identifier,
        tuple(str(value) for value in work_frame["key"]),
    )
    selection_fit_assignment_digest = _canonical_sha256(
        {
            "schema": "emotiontalk_selection_fit_assignment_v1",
            "probability_mode": "train_fit_only",
            "fit_role": "base_and_utility_fit",
            "prediction_role": "model_selection",
            "split_protocol_id": str(
                utility_config["data_roles"]["split_protocol_id"]
            ),
            "source_order_sha256": source_order_digest,
            "ordered_open_roles": tuple(str(value) for value in work_roles),
        }
    )

    histories = build_history_indices(work_frame)
    sampling = utility_config["counterfactual_sampling"]
    tasks = sample_bidirectional_coalition_tasks(
        histories,
        draws_per_query=int(sampling["draws_per_query"]),
        maximum_candidates=int(sampling["maximum_candidates_per_query"]),
        seed=int(sampling["seed"]),
        match_context_cardinality=bool(sampling.get("match_context_cardinality", False)),
    )
    fit_set = set(int(value) for value in fit_indices)
    selection_set = set(int(value) for value in selection_indices)
    fit_tasks = tuple(task for task in tasks if int(task.query_index) in fit_set)
    selection_tasks = tuple(task for task in tasks if int(task.query_index) in selection_set)
    if not fit_tasks or not selection_tasks:
        raise OpenRoleDiagnosticError("deterministic sampling produced an empty open role")

    fit_task_groups = [work_groups[int(task.query_index)] for task in fit_tasks]
    selection_task_groups = [work_groups[int(task.query_index)] for task in selection_tasks]
    fit_cluster_codes = _integer_codes(fit_task_groups)
    selection_cluster_codes = _integer_codes(selection_task_groups)

    fit_query = np.asarray([int(task.query_index) for task in fit_tasks], dtype=np.int64)
    fit_local = np.arange(len(fit_indices), dtype=np.int64)
    expected_positions: dict[str, np.ndarray] = {}
    splitter = GroupKFold(n_splits=int(base_config["crossfit_folds"]))
    for fold, (_, held_local) in enumerate(
        splitter.split(fit_local, groups=work_groups[fit_indices]), start=1
    ):
        held_index = fit_indices[held_local]
        expected_positions[f"fold_{fold}.npz"] = np.flatnonzero(
            np.isin(fit_query, held_index)
        ).astype(np.int64, copy=False)

    role_counts = {
        "base_and_utility_fit_rows": int(len(fit_indices)),
        "model_selection_rows": int(len(selection_indices)),
        "non_open_rows_materialized": 0,
    }
    task_material = OpenRoleTasks(
        histories=histories,
        fit_tasks=fit_tasks,
        selection_tasks=selection_tasks,
        fit_cluster_codes=fit_cluster_codes,
        selection_cluster_codes=selection_cluster_codes,
        expected_fold_positions=MappingProxyType(expected_positions),
        role_counts=MappingProxyType(role_counts),
        train_key_manifest_sha256=_manifest_sha256(keys),
        dataset_identifier=dataset_identifier,
        source_order_sha256=source_order_digest,
        selection_fit_assignment_sha256=selection_fit_assignment_digest,
    )
    return task_material, feature_config_sha256


def recover_query_labels_from_cached_utilities(
    tasks: Sequence[object],
    probability: np.ndarray,
    cached_forward: np.ndarray,
    cached_backward: np.ndarray,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-11,
) -> np.ndarray:
    """Recover the unique class reproducing both cached utility directions.

    The output is task-aligned.  Candidate classes are intersected across all
    sampled tasks belonging to the same query; ambiguity fails closed.
    """

    values = np.asarray(probability, dtype=np.float64)
    if values.ndim == 4:
        values = values.mean(axis=0)
    if values.shape != (len(tasks), len(CONTEXT_NAMES), len(LABEL_NAMES)):
        raise OpenRoleDiagnosticError("probability tensor is not aligned to sampled tasks")
    forward = np.asarray(cached_forward, dtype=np.float64)
    backward = np.asarray(cached_backward, dtype=np.float64)
    if forward.shape != (len(tasks),) or backward.shape != (len(tasks),):
        raise OpenRoleDiagnosticError("cached utilities are not aligned to sampled tasks")
    if not np.isfinite(values).all() or not np.isfinite(forward).all() or not np.isfinite(backward).all():
        raise OpenRoleDiagnosticError("label recovery inputs contain non-finite values")

    clipped = np.clip(values, 1e-12, 1.0)
    candidate_forward = np.log(clipped[:, 1, :]) - np.log(clipped[:, 0, :])
    candidate_backward = np.log(clipped[:, 2, :]) - np.log(clipped[:, 3, :])
    matches = np.isclose(candidate_forward, forward[:, None], rtol=rtol, atol=atol) & np.isclose(
        candidate_backward, backward[:, None], rtol=rtol, atol=atol
    )
    query_to_rows: dict[int, list[int]] = {}
    for row, task in enumerate(tasks):
        query_to_rows.setdefault(int(getattr(task, "query_index")), []).append(row)
    labels = np.full(len(tasks), -1, dtype=np.int64)
    for rows in query_to_rows.values():
        possible = np.ones(len(LABEL_NAMES), dtype=bool)
        for row in rows:
            possible &= matches[row]
        candidates = np.flatnonzero(possible)
        if len(candidates) != 1:
            raise OpenRoleDiagnosticError(
                "cached utilities do not identify exactly one emotion label per query"
            )
        labels[rows] = int(candidates[0])
    if np.any(labels < 0):
        raise AssertionError("query label recovery left uncovered tasks")
    return labels


def _ensemble_predictions(predictions: Sequence[UtilityPredictions]) -> UtilityPredictions:
    if not predictions:
        raise OpenRoleDiagnosticError("cannot ensemble zero utility predictions")
    forward_values = [prediction.forward for prediction in predictions]
    backward_values = [prediction.backward for prediction in predictions]
    forward = (
        None
        if any(value is None for value in forward_values)
        else np.mean(np.stack(forward_values), axis=0)
    )
    backward = (
        None
        if any(value is None for value in backward_values)
        else np.mean(np.stack(backward_values), axis=0)
    )
    if forward is not None and backward is not None:
        decision = np.minimum(forward, backward)
    elif forward is not None:
        decision = forward.copy()
    elif backward is not None:
        decision = backward.copy()
    else:
        raise AssertionError("utility prediction contains no direction")
    return UtilityPredictions(forward, backward, decision)


def _stats(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) != 5 or not np.isfinite(array).all():
        raise OpenRoleDiagnosticError("five-seed aggregate requires five finite values")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize_five_seed_diagnostics(
    diagnostics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize only policy and endpoint-comparison aggregate leaves."""

    records = tuple(diagnostics)
    if len(records) != 5:
        raise OpenRoleDiagnosticError("exactly five seed diagnostics are required")
    summary: dict[str, object] = {
        "seed_count": 5,
        "frozen_threshold": _stats(
            [float(record["decision"]["frozen_threshold"]) for record in records]
        ),
        "query_cluster_macro_selected_rate": _stats(
            [
                float(record["decision"]["query_cluster_macro_selected_rate"])
                for record in records
            ]
        ),
    }
    for branch in ("addition", "deletion"):
        first = records[0][branch]
        branch_summary: dict[str, object] = {
            "policy_metrics": {
                metric: _stats(
                    [float(record[branch]["policy_metrics"][metric]) for record in records]
                )
                for metric in ("macro_f1", "accuracy", "nll", "brier")
            },
            "relative_to_fixed_endpoints": {},
        }
        comparisons = first["relative_to_fixed_endpoints"]
        for endpoint in comparisons:
            branch_summary["relative_to_fixed_endpoints"][endpoint] = {
                metric: _stats(
                    [
                        float(
                            record[branch]["relative_to_fixed_endpoints"][endpoint][metric]
                        )
                        for record in records
                    ]
                )
                for metric in ("nll_regret", "nll_harm_rate")
            }
        summary[branch] = branch_summary
    return summary


def summarize_five_seed_paired_contrasts(
    contrasts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    records = tuple(contrasts)
    if len(records) != 5:
        raise OpenRoleDiagnosticError("exactly five paired seed contrasts are required")
    summary: dict[str, object] = {
        "seed_count": 5,
        "query_cluster_macro_disagreement_rate": _stats(
            [
                float(
                    record["decision_contrast"]["query_cluster_macro_disagreement_rate"]
                )
                for record in records
            ]
        ),
    }
    for branch in ("addition", "deletion"):
        summary[branch] = {
            "metric_delta_true_minus_reference": {
                metric: _stats(
                    [
                        float(record[branch]["metric_delta_a_minus_b"][metric])
                        for record in records
                    ]
                )
                for metric in ("macro_f1", "accuracy", "nll", "brier")
            },
            "paired_nll": {
                metric: _stats(
                    [float(record[branch]["paired_nll"][metric]) for record in records]
                )
                for metric in (
                    "a_minus_b",
                    "a_harm_rate_vs_b",
                    "a_win_rate_vs_b",
                    "tie_rate",
                )
            },
        }
    return summary


def _fit_scope_digest(hashes: Mapping[str, str], selection_checkpoint_hash: str) -> str:
    payload = "\n".join(
        [
            hashes["base_config_sha256"],
            hashes["utility_config_sha256"],
            hashes["feature_sha256"],
            selection_checkpoint_hash,
        ]
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _probability_producer_config_sha256(
    hashes: Mapping[str, str],
    *,
    feature_config_sha256: str,
) -> str:
    """Bind every frozen artifact that determines checkpoint probabilities."""

    return _canonical_sha256(
        {
            "schema": "emotiontalk_probability_producer_config_v1",
            "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
            "base_config_sha256": str(hashes["base_config_sha256"]),
            "utility_config_sha256": str(hashes["utility_config_sha256"]),
            "feature_artifact_sha256": str(hashes["feature_sha256"]),
            "feature_config_sha256": str(feature_config_sha256),
        }
    )


def _selection_task_order_sha256(
    tasks: Sequence[object],
    *,
    task_material: OpenRoleTasks,
    split_manifest_sha256: str,
    producer_config_sha256: str,
) -> str:
    """Hash selection tasks under the complete train-only lineage contract."""

    return bidirectional_task_order_sha256(
        tasks,
        dataset=task_material.dataset_identifier,
        role="model_selection",
        source_order_sha256=task_material.source_order_sha256,
        split_manifest_sha256=split_manifest_sha256,
        fold_assignment_sha256=task_material.selection_fit_assignment_sha256,
        context_schema_sha256=emotion_context_schema_sha256(),
        class_order_sha256=emotion_class_order_sha256(LABEL_NAMES),
        producer_config_sha256=producer_config_sha256,
    )


def _selection_diagnostic_inputs(
    task_material: OpenRoleTasks,
    selection_probability: np.ndarray,
    cache: BidirectionalUtilityCache,
    *,
    fit_scope_sha256: str,
    split_manifest_sha256: str,
    producer_config_sha256: str,
) -> tuple[SampledContextInputs, str]:
    tasks = task_material.selection_tasks
    labels = recover_query_labels_from_cached_utilities(
        tasks,
        selection_probability,
        cache.selection.forward,
        cache.selection.backward,
    )
    query_values = [int(getattr(task, "query_index")) for task in tasks]
    query_codes = _integer_codes(query_values)
    task_hash = _selection_task_order_sha256(
        tasks,
        task_material=task_material,
        split_manifest_sha256=split_manifest_sha256,
        producer_config_sha256=producer_config_sha256,
    )
    context_probability = {
        context: selection_probability[:, :, index, :]
        for index, context in enumerate(CONTEXT_NAMES)
    }
    inputs = SampledContextInputs(
        query_labels=labels,
        query_codes=query_codes,
        cluster_codes=task_material.selection_cluster_codes,
        context_probabilities=context_probability,
        provenance=DiagnosticProvenance(
            "train_fit_only",
            fit_scope_sha256,
            task_hash,
        ),
    )
    return inputs, task_hash


def _model_classification_diagnostics(
    cache: BidirectionalUtilityCache,
    inputs: SampledContextInputs,
    task_hash: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    specs = tuple(default_model_specs())
    if tuple(spec.name for spec in specs) != EXPECTED_MODEL_NAMES:
        raise OpenRoleDiagnosticError("default utility model set is not the registered four-model set")
    if len(DEFAULT_SEEDS) != 5 or len(set(DEFAULT_SEEDS)) != 5:
        raise OpenRoleDiagnosticError("utility diagnostic requires five distinct model seeds")
    reports: list[dict[str, object]] = []
    private: dict[str, dict[str, object]] = {}
    for spec in specs:
        seed_diagnostics: list[dict[str, object]] = []
        oof_predictions: list[UtilityPredictions] = []
        selection_predictions: list[UtilityPredictions] = []
        decisions: list[FrozenUtilityDecision] = []
        for seed in DEFAULT_SEEDS:
            oof = group_oof_predictions(cache.fit, spec, seed=int(seed), maximum_splits=5)
            threshold = fit_oof_coverage_threshold(
                oof.predictions.decision_score, PRIMARY_HISTORY_COVERAGE
            )
            fitted = fit_utility_model(cache.fit, spec, seed=int(seed))
            selection_prediction = fitted.predict(cache.selection.x)
            decision = FrozenUtilityDecision(
                selection_prediction.decision_score,
                threshold,
                task_hash,
            )
            seed_diagnostics.append(
                sampled_context_classification_diagnostic(inputs, decision)
            )
            oof_predictions.append(oof.predictions)
            selection_predictions.append(selection_prediction)
            decisions.append(decision)

        oof_ensemble = _ensemble_predictions(oof_predictions)
        selection_ensemble = _ensemble_predictions(selection_predictions)
        ensemble_threshold = fit_oof_coverage_threshold(
            oof_ensemble.decision_score, PRIMARY_HISTORY_COVERAGE
        )
        ensemble_decision = FrozenUtilityDecision(
            selection_ensemble.decision_score,
            ensemble_threshold,
            task_hash,
        )
        reports.append(
            {
                "name": spec.name,
                "mode": spec.mode,
                "utility_training_seeds": list(DEFAULT_SEEDS),
                "threshold_contract": "fit_group_oof_top_25pct_frozen_before_selection",
                "ensemble_diagnostic": sampled_context_classification_diagnostic(
                    inputs, ensemble_decision
                ),
                "five_seed_aggregate_diagnostic": summarize_five_seed_diagnostics(
                    seed_diagnostics
                ),
            }
        )
        private[spec.name] = {
            "ensemble_decision": ensemble_decision,
            "seed_decisions": tuple(decisions),
        }

    true_private = private["bidirectional_shared_mlp"]
    paired_reports: list[dict[str, object]] = []
    for reference_name in REFERENCE_MODEL_NAMES:
        reference_private = private[reference_name]
        ensemble_contrast = paired_sampled_context_model_contrast(
            inputs,
            true_private["ensemble_decision"],
            reference_private["ensemble_decision"],
        )
        seed_contrasts = [
            paired_sampled_context_model_contrast(inputs, true_decision, reference_decision)
            for true_decision, reference_decision in zip(
                true_private["seed_decisions"],
                reference_private["seed_decisions"],
                strict=True,
            )
        ]
        paired_reports.append(
            {
                "candidate": "bidirectional_shared_mlp",
                "reference": reference_name,
                "ensemble_paired_diagnostic": ensemble_contrast,
                "five_seed_aggregate_paired_diagnostic": summarize_five_seed_paired_contrasts(
                    seed_contrasts
                ),
            }
        )
    return reports, paired_reports


def _assert_aggregate_output(value: object, path: tuple[str, ...] = ()) -> None:
    forbidden_keys = {
        "decision_scores",
        "query_codes",
        "cluster_codes",
        "query_labels",
        "row_predictions",
        "task_positions",
    }
    if isinstance(value, np.ndarray):
        raise OpenRoleDiagnosticError(
            f"aggregate output contains an ndarray at {'.'.join(path)}"
        )
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            if name.lower() in forbidden_keys:
                raise OpenRoleDiagnosticError(
                    f"aggregate output contains forbidden row field {'.'.join((*path, name))}"
                )
            _assert_aggregate_output(child, (*path, name))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_aggregate_output(child, (*path, str(index)))
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise OpenRoleDiagnosticError(
            f"aggregate output contains a non-finite value at {'.'.join(path)}"
        )


def _validate_runner_signature() -> None:
    names = tuple(inspect.signature(run_open_role_sampled_context_diagnostic).parameters)
    expected = (
        "data_dir",
        "feature_path",
        "base_config_path",
        "utility_config_path",
        "private_cache_path",
        "checkpoint_dir",
        "output_path",
    )
    if names != expected:
        raise AssertionError("public runner parameters changed")
    for name in names:
        tokens = set(name.lower().replace("-", "_").split("_"))
        if tokens & FORBIDDEN_ROLE_TOKENS:
            raise AssertionError("public runner exposes a forbidden role parameter")


def run_open_role_sampled_context_diagnostic(
    data_dir: Path,
    feature_path: Path,
    base_config_path: Path,
    utility_config_path: Path,
    private_cache_path: Path,
    checkpoint_dir: Path,
    output_path: Path,
) -> dict[str, object]:
    """Run the real checkpoint-backed diagnostic without opening sealed roles."""

    paths = tuple(
        Path(value)
        for value in (
            data_dir,
            feature_path,
            base_config_path,
            utility_config_path,
            private_cache_path,
            checkpoint_dir,
            output_path,
        )
    )
    (
        data_dir,
        feature_path,
        base_config_path,
        utility_config_path,
        private_cache_path,
        checkpoint_dir,
        output_path,
    ) = paths
    if output_path.exists():
        raise FileExistsError(f"sampled-context output already exists: {output_path}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {data_dir}")
    for path in (feature_path, base_config_path, utility_config_path, private_cache_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)

    base_config = _read_json(base_config_path)
    utility_config = _read_json(utility_config_path)
    _validate_config(utility_config, base_config)
    hashes = _expected_hashes(base_config_path, utility_config_path, feature_path)
    cache = load_private_oof_cache(private_cache_path)
    if cache.source_hashes.get("base_config_sha256") != hashes["base_config_sha256"]:
        raise OpenRoleDiagnosticError("private cache base-config hash mismatch")
    if cache.source_hashes.get("utility_config_sha256") != hashes["utility_config_sha256"]:
        raise OpenRoleDiagnosticError("private cache utility-config hash mismatch")

    task_material, feature_config_sha256 = reconstruct_open_role_tasks(
        feature_path, base_config, utility_config
    )
    fold_paths = tuple(
        checkpoint_dir / f"fold_{fold}.npz"
        for fold in range(1, int(base_config["crossfit_folds"]) + 1)
    )
    expected_checkpoint_names = {path.name for path in fold_paths} | {"selection.npz"}
    actual_checkpoint_names = {path.name for path in checkpoint_dir.glob("*.npz")}
    if actual_checkpoint_names != expected_checkpoint_names:
        raise OpenRoleDiagnosticError(
            "checkpoint directory must contain exactly the deterministic folds and selection"
        )
    base_seed_count = len(tuple(base_config["seeds"]))
    fit_probability = assemble_fit_probability_checkpoints(
        fold_paths,
        expected_task_count=len(task_material.fit_tasks),
        expected_seed_count=base_seed_count,
        expected_hashes=hashes,
        expected_positions_by_name=task_material.expected_fold_positions,
    )
    selection_path = checkpoint_dir / "selection.npz"
    selection_probability = load_selection_probability_checkpoint(
        selection_path,
        expected_task_count=len(task_material.selection_tasks),
        expected_seed_count=base_seed_count,
        expected_hashes=hashes,
    )

    fit_x, fit_names = probability_task_features(
        np.mean(fit_probability, axis=0),
        task_material.fit_tasks,
        task_material.histories,
    )
    selection_x, selection_names = probability_task_features(
        np.mean(selection_probability, axis=0),
        task_material.selection_tasks,
        task_material.histories,
    )
    verify_recomputed_59d_cache(
        cache,
        fit_x=fit_x,
        fit_feature_names=fit_names,
        selection_x=selection_x,
        selection_feature_names=selection_names,
        fit_cluster_codes=task_material.fit_cluster_codes,
        selection_cluster_codes=task_material.selection_cluster_codes,
    )

    checkpoint_hashes = {
        path.name: sha256_file(path) for path in (*fold_paths, selection_path)
    }
    fit_scope_sha256 = _fit_scope_digest(hashes, checkpoint_hashes["selection.npz"])
    producer_config_sha256 = _probability_producer_config_sha256(
        hashes,
        feature_config_sha256=feature_config_sha256,
    )
    diagnostic_inputs, task_hash = _selection_diagnostic_inputs(
        task_material,
        selection_probability,
        cache,
        fit_scope_sha256=fit_scope_sha256,
        split_manifest_sha256=hashes["utility_config_sha256"],
        producer_config_sha256=producer_config_sha256,
    )
    model_reports, paired_reports = _model_classification_diagnostics(
        cache, diagnostic_inputs, task_hash
    )

    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "open_role_sampled_context_diagnostic_complete",
        "claim_boundary": (
            "These are sampled-context diagnostics on the open model-selection role. "
            "They are not a final query-level selection policy, do not establish validation/test "
            "improvement, and do not establish top-conference readiness."
        ),
        "access_contract": {
            "roles_materialized": ["base_and_utility_fit", "model_selection"],
            "non_open_role_rows_materialized": 0,
            "external_nontrain_rows_materialized": 0,
            "base_models_recomputed": False,
            "label_archive_opened": False,
            "labels_recovered_from_cached_bidirectional_utilities": True,
            "row_level_output": False,
        },
        "data_summary": {
            **dict(task_material.role_counts),
            "fit_sampled_tasks": int(len(task_material.fit_tasks)),
            "model_selection_sampled_tasks": int(len(task_material.selection_tasks)),
            "fit_task_clusters": int(len(np.unique(task_material.fit_cluster_codes))),
            "model_selection_task_clusters": int(
                len(np.unique(task_material.selection_cluster_codes))
            ),
            "base_probability_seeds": int(base_seed_count),
            "utility_model_seeds": int(len(DEFAULT_SEEDS)),
            "task_feature_count": int(cache.fit.x.shape[1]),
        },
        "alignment_audit": {
            "fit_checkpoint_exact_nonoverlapping_cover": True,
            "selection_checkpoint_canonical_complete_order": True,
            "recomputed_59d_features_bitwise_equal_cache": True,
            "feature_names_exactly_equal_cache": True,
            "cluster_codes_exactly_equal_cache": True,
            "selection_task_order_sha256": task_hash,
        },
        "source_hashes": {
            **hashes,
            "feature_config_sha256": feature_config_sha256,
            "private_cache_sha256": sha256_file(private_cache_path),
            "train_key_manifest_sha256": task_material.train_key_manifest_sha256,
            "checkpoints_sha256": checkpoint_hashes,
            "label_archive_sha256": None,
            "label_archive_hash_omission_reason": "archive_not_opened_to_protect_non_open_roles",
        },
        "models": model_reports,
        "true_bidirectional_paired_sampled_context_contrasts": paired_reports,
    }
    _assert_aggregate_output(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report, output_path.resolve())
    return report


_validate_runner_signature()
