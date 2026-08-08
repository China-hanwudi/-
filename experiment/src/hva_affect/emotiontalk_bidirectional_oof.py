from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from .bidirectional_emotion_utility import (
    BidirectionalCoalitionTask,
    bidirectional_utility_targets,
    sample_bidirectional_coalition_tasks,
)
from .data_contract import ContractError, write_json_atomic
from .emotiontalk_endpoint_diagnostic import (
    _load_materialized_labels,
    assign_frame_roles,
)
from .emotiontalk_multimodal_external import (
    Blocks,
    _align_probabilities,
    _fit_processors,
    _transform_processors,
    base_features,
    load_media_split,
    load_unlabeled_frame,
)
from .emotiontalk_text_p1 import LABEL_NAMES, build_history_indices
from .meld_text_pilot import make_classifier, sha256_file


CONTEXT_NAMES = ("s", "s_plus_candidate", "t", "t_minus_candidate")
ROLE_RANGES = {
    "base_and_utility_fit": [0, 64],
    "model_selection": [65, 79],
    "calibration": [80, 89],
    "internal_holdout_sealed": [90, 99],
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _role_ranges(config: Mapping[str, object]) -> dict[str, list[int]]:
    roles = config.get("data_roles")
    if not isinstance(roles, Mapping):
        raise ContractError("bidirectional config lacks data_roles")
    ranges = {name: roles.get(name) for name in ROLE_RANGES}
    if ranges != ROLE_RANGES:
        raise ContractError("bidirectional role ranges changed")
    if roles.get("split_protocol_id") != "scu_set_exploration_v1":
        raise ContractError("the protocol-independent frozen split id changed")
    return ROLE_RANGES.copy()


def _validate_config(config: Mapping[str, object], base_config: Mapping[str, object]) -> None:
    required = {
        "protocol",
        "status",
        "data_roles",
        "six_streams",
        "counterfactual_sampling",
        "model_comparison",
    }
    missing = required - set(config)
    if missing:
        raise ContractError(f"bidirectional config missing keys: {sorted(missing)}")
    if str(config["protocol"]) != "bidirectional_emotion_utility_v1":
        raise ContractError("unexpected bidirectional protocol")
    if "not_yet_run" not in str(config["status"]):
        raise ContractError("bidirectional exploration contract was not in an unopened state")
    _role_ranges(config)
    six_streams = config["six_streams"]
    if not isinstance(six_streams, Mapping):
        raise ContractError("six_streams must be a mapping")
    if list(six_streams.get("current", [])) != ["text", "audio", "video"]:
        raise ContractError("current six-stream contract changed")
    if list(six_streams.get("history", [])) != [
        "text_history",
        "audio_history",
        "video_history",
    ]:
        raise ContractError("history six-stream contract changed")
    sampling = config["counterfactual_sampling"]
    if not isinstance(sampling, Mapping):
        raise ContractError("counterfactual_sampling must be a mapping")
    if int(sampling.get("draws_per_query", 0)) < 1:
        raise ContractError("draws_per_query must be positive")
    if int(sampling.get("maximum_candidates_per_query", 0)) < 2:
        raise ContractError("at least two history candidates are required")
    augmentation = config.get("base_subset_augmentation")
    if not isinstance(augmentation, Mapping):
        raise ContractError("base_subset_augmentation must be frozen before running")
    if int(augmentation.get("maximum_contexts_per_query", 0)) < 2:
        raise ContractError("base subset augmentation must include multiple contexts")
    if base_config.get("sealed_split") != "test_corpus":
        raise ContractError("base config no longer seals EmotionTalk test")
    if base_config.get("primary_base") != "text_audio_video":
        raise ContractError("primary base changed")
    if int(base_config.get("crossfit_folds", 0)) < 2:
        raise ContractError("cross-fitting requires at least two folds")
    if len(base_config.get("seeds", [])) < 2:
        raise ContractError("multi-seed probability supervision is required")


def _validate_contexts(contexts: Sequence[Sequence[int]], source_rows: int) -> None:
    for context in contexts:
        values = tuple(int(value) for value in context)
        if len(values) != len(set(values)):
            raise ValueError("history context contains duplicate row indices")
        if any(value < 0 or value >= source_rows for value in values):
            raise ValueError("history context index is outside the source frame")


def aggregate_sparse_contexts(
    current: sparse.csr_matrix,
    contexts: Sequence[Sequence[int]],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Aggregate arbitrary source rows into a rectangular task matrix."""

    current = current.tocsr()
    _validate_contexts(contexts, current.shape[0])
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    counts = np.zeros(len(contexts), dtype=np.int32)
    for row, context in enumerate(contexts):
        selected = tuple(int(value) for value in context)
        counts[row] = len(selected)
        if selected:
            rows.extend([row] * len(selected))
            cols.extend(selected)
            values.extend([1.0 / len(selected)] * len(selected))
    operator = sparse.csr_matrix(
        (values, (rows, cols)),
        shape=(len(contexts), current.shape[0]),
        dtype=np.float64,
    )
    return (operator @ current).tocsr(), counts


def aggregate_dense_contexts(
    current: np.ndarray,
    contexts: Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(current)
    if current.ndim != 2:
        raise ValueError("dense current features must be two-dimensional")
    _validate_contexts(contexts, current.shape[0])
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    counts = np.zeros(len(contexts), dtype=np.int32)
    for row, context in enumerate(contexts):
        selected = tuple(int(value) for value in context)
        counts[row] = len(selected)
        if selected:
            rows.extend([row] * len(selected))
            cols.extend(selected)
            values.extend([1.0 / len(selected)] * len(selected))
    operator = sparse.csr_matrix(
        (values, (rows, cols)),
        shape=(len(contexts), current.shape[0]),
        dtype=np.float64,
    )
    return np.asarray(operator @ current, dtype=np.float64), counts


def build_context_blocks(
    current: Mapping[str, sparse.csr_matrix | np.ndarray],
    quality: np.ndarray,
    quality_names: Sequence[str],
    query_indices: Sequence[int],
    contexts: Sequence[Sequence[int]],
) -> Blocks:
    query = np.asarray(query_indices, dtype=np.int64)
    if query.ndim != 1 or len(query) != len(contexts):
        raise ValueError("query indices and contexts must be one-dimensional and aligned")
    source_rows = len(quality)
    if np.any((query < 0) | (query >= source_rows)):
        raise ValueError("query index is outside the source frame")
    if set(current) != {"text", "audio", "video"}:
        raise ValueError("current features must contain text, audio, and video")
    text_history, counts = aggregate_sparse_contexts(current["text"].tocsr(), contexts)
    audio_history, audio_counts = aggregate_dense_contexts(np.asarray(current["audio"]), contexts)
    video_history, video_counts = aggregate_dense_contexts(np.asarray(current["video"]), contexts)
    quality_history, quality_counts = aggregate_dense_contexts(np.asarray(quality), contexts)
    if not (
        np.array_equal(counts, audio_counts)
        and np.array_equal(counts, video_counts)
        and np.array_equal(counts, quality_counts)
    ):
        raise RuntimeError("context counts differ across modalities")
    return Blocks(
        current={name: value[query] for name, value in current.items()},
        history={
            "text": text_history,
            "audio": audio_history,
            "video": video_history,
        },
        quality_current=np.asarray(quality)[query],
        quality_history=quality_history,
        quality_names=tuple(str(value) for value in quality_names),
        counts=counts,
    )


def task_contexts(task: BidirectionalCoalitionTask) -> tuple[tuple[int, ...], ...]:
    candidate = int(task.candidate_index)
    return (
        tuple(task.addition_context),
        tuple(sorted(tuple(task.addition_context) + (candidate,))),
        tuple(task.deletion_context),
        tuple(value for value in task.deletion_context if int(value) != candidate),
    )


def augmented_training_rows(
    histories: Sequence[Sequence[int]],
    tasks: Sequence[BidirectionalCoalitionTask],
    allowed_indices: Sequence[int],
    *,
    maximum_contexts_per_query: int,
    seed: int,
) -> tuple[np.ndarray, list[tuple[int, ...]], np.ndarray]:
    """Create deterministic query-balanced subset augmentation for a base model."""

    allowed = {int(value) for value in allowed_indices}
    per_query: dict[int, set[tuple[int, ...]]] = {
        query: {(), tuple(int(value) for value in histories[query])} for query in allowed
    }
    for task in tasks:
        query = int(task.query_index)
        if query not in allowed:
            continue
        per_query[query].update(task_contexts(task))
    query_rows: list[int] = []
    contexts: list[tuple[int, ...]] = []
    weights: list[float] = []
    for query in sorted(allowed):
        candidates = sorted(per_query[query], key=lambda value: (len(value), value))
        for context in candidates:
            if not set(context).issubset(allowed):
                raise ContractError("a training subset crosses the fold/group boundary")
        endpoints = [(), tuple(int(value) for value in histories[query])]
        chosen = list(dict.fromkeys(endpoints))
        remainder = [value for value in candidates if value not in set(chosen)]
        room = max(0, int(maximum_contexts_per_query) - len(chosen))
        if len(remainder) > room:
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), query]))
            selected = sorted(
                (remainder[int(index)] for index in rng.choice(len(remainder), size=room, replace=False)),
                key=lambda value: (len(value), value),
            )
        else:
            selected = remainder
        chosen.extend(selected)
        per_row_weight = 1.0 / len(chosen)
        for context in chosen:
            query_rows.append(query)
            contexts.append(context)
            weights.append(per_row_weight)
    sample_weight = np.asarray(weights, dtype=np.float64)
    if len(sample_weight):
        sample_weight *= len(sample_weight) / sample_weight.sum()
    return np.asarray(query_rows, dtype=np.int64), contexts, sample_weight


def _task_arrays(
    tasks: Sequence[BidirectionalCoalitionTask],
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    query_indices: list[int] = []
    contexts: list[tuple[int, ...]] = []
    for task in tasks:
        query_indices.extend([int(task.query_index)] * len(CONTEXT_NAMES))
        contexts.extend(task_contexts(task))
    return np.asarray(query_indices, dtype=np.int64), contexts


def _predict_tasks(
    models: Sequence,
    current: Mapping[str, sparse.csr_matrix | np.ndarray],
    quality: np.ndarray,
    quality_names: Sequence[str],
    tasks: Sequence[BidirectionalCoalitionTask],
    modalities: Sequence[str],
    n_classes: int,
) -> np.ndarray:
    query_indices, contexts = _task_arrays(tasks)
    blocks = build_context_blocks(current, quality, quality_names, query_indices, contexts)
    features = base_features(blocks, modalities, use_history=True)
    result = np.empty(
        (len(models), len(tasks), len(CONTEXT_NAMES), n_classes),
        dtype=np.float64,
    )
    for seed_index, model in enumerate(models):
        probability = _align_probabilities(model, model.predict_proba(features), n_classes)
        result[seed_index] = np.asarray(probability, dtype=np.float64).reshape(
            len(tasks), len(CONTEXT_NAMES), n_classes
        )
    return result


def probability_task_features(
    probability: np.ndarray,
    tasks: Sequence[BidirectionalCoalitionTask],
    histories: Sequence[Sequence[int]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build label-free structural and probability features for utility heads."""

    probability = np.asarray(probability, dtype=np.float64)
    if probability.ndim != 3 or probability.shape[1] != len(CONTEXT_NAMES):
        raise ValueError("task probability must have shape (tasks, 4, classes)")
    if len(probability) != len(tasks):
        raise ValueError("probabilities and tasks are not aligned")
    if not np.isfinite(probability).all():
        raise ValueError("task probability contains non-finite values")
    values: list[np.ndarray] = []
    names: list[str] = []
    for context_index, context_name in enumerate(CONTEXT_NAMES):
        context_probability = probability[:, context_index]
        entropy = -(
            context_probability * np.log(np.clip(context_probability, 1e-12, 1.0))
        ).sum(axis=1)
        values.extend([context_probability.max(axis=1), entropy])
        names.extend([f"{context_name}_confidence", f"{context_name}_entropy"])
        for class_index in range(context_probability.shape[1]):
            values.append(context_probability[:, class_index])
            names.append(f"{context_name}_probability_{class_index}")
    for prefix, left_index, right_index in (
        ("forward", 1, 0),
        ("backward", 2, 3),
    ):
        delta = probability[:, left_index] - probability[:, right_index]
        values.extend([np.abs(delta).sum(axis=1), np.linalg.norm(delta, axis=1)])
        names.extend([f"{prefix}_probability_l1", f"{prefix}_probability_l2"])
        for class_index in range(delta.shape[1]):
            values.append(delta[:, class_index])
            names.append(f"{prefix}_probability_delta_{class_index}")
    addition_count = np.asarray([len(task.addition_context) for task in tasks], dtype=float)
    deletion_count = np.asarray([len(task.deletion_context) for task in tasks], dtype=float)
    overlap = np.asarray(
        [len(set(task.addition_context) & set(task.deletion_context)) for task in tasks],
        dtype=float,
    )
    union = np.asarray(
        [len(set(task.addition_context) | set(task.deletion_context)) for task in tasks],
        dtype=float,
    )
    full_count = np.asarray([len(histories[task.query_index]) for task in tasks], dtype=float)
    candidate_recency = []
    for task in tasks:
        history = tuple(int(value) for value in histories[task.query_index])
        rank = history.index(int(task.candidate_index))
        candidate_recency.append((rank + 1.0) / max(1.0, float(len(history))))
    values.extend(
        [
            np.log1p(addition_count),
            np.log1p(deletion_count),
            overlap / np.maximum(union, 1.0),
            np.log1p(full_count),
            np.asarray(candidate_recency, dtype=float),
        ]
    )
    names.extend(
        [
            "log_addition_context_count",
            "log_deletion_context_count",
            "addition_deletion_jaccard",
            "log_full_history_count",
            "candidate_recency_fraction",
        ]
    )
    matrix = np.column_stack(values).astype(np.float64, copy=False)
    if not np.isfinite(matrix).all():
        raise RuntimeError("non-finite bidirectional task feature")
    if any("label" in name or "gold" in name for name in names):
        raise AssertionError("task feature schema accidentally contains label information")
    return matrix, tuple(names)


def _targets_from_probability(
    y: np.ndarray,
    tasks: Sequence[BidirectionalCoalitionTask],
    probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query_labels = np.asarray([y[task.query_index] for task in tasks], dtype=np.int64)
    ensemble = np.mean(probability, axis=0)
    target = bidirectional_utility_targets(
        query_labels,
        ensemble[:, 0],
        ensemble[:, 1],
        ensemble[:, 2],
        ensemble[:, 3],
    )
    forward_seed: list[np.ndarray] = []
    backward_seed: list[np.ndarray] = []
    for seed_probability in probability:
        seed_target = bidirectional_utility_targets(
            query_labels,
            seed_probability[:, 0],
            seed_probability[:, 1],
            seed_probability[:, 2],
            seed_probability[:, 3],
        )
        forward_seed.append(seed_target.forward_addition)
        backward_seed.append(seed_target.backward_deletion)
    return (
        target.forward_addition,
        target.backward_deletion,
        np.asarray(forward_seed, dtype=np.float64),
        np.asarray(backward_seed, dtype=np.float64),
    )


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return float("nan")
    return float(spearmanr(left, right).statistic)


def _seed_stability(targets: np.ndarray) -> dict:
    targets = np.asarray(targets, dtype=np.float64)
    correlations = []
    for left in range(len(targets)):
        for right in range(left + 1, len(targets)):
            correlations.append(_safe_spearman(targets[left], targets[right]))
    ensemble = targets.mean(axis=0)
    sign_agreement = np.mean(np.sign(targets) == np.sign(ensemble)[None, :], axis=1)
    return {
        "seeds": int(len(targets)),
        "rows": int(targets.shape[1]),
        "pairwise_spearman_median": float(np.nanmedian(correlations)),
        "pairwise_spearman_min": float(np.nanmin(correlations)),
        "seed_to_ensemble_sign_agreement": [float(value) for value in sign_agreement],
    }


def _target_summary(forward: np.ndarray, backward: np.ndarray) -> dict:
    forward = np.asarray(forward, dtype=np.float64)
    backward = np.asarray(backward, dtype=np.float64)
    return {
        "tasks": int(len(forward)),
        "forward_mean_utility": float(forward.mean()),
        "backward_mean_utility": float(backward.mean()),
        "forward_harm_rate": float((forward < 0).mean()),
        "backward_harm_rate": float((backward < 0).mean()),
        "forward_backward_spearman": _safe_spearman(forward, backward),
        "sign_agreement": float((np.sign(forward) == np.sign(backward)).mean()),
        "mean_absolute_asymmetry": float(np.mean(np.abs(forward - backward))),
        "p90_absolute_asymmetry": float(np.quantile(np.abs(forward - backward), 0.90)),
        "nonzero_asymmetry_rate_1e_8": float((np.abs(forward - backward) > 1e-8).mean()),
    }


def _cluster_codes(groups: Sequence[str]) -> np.ndarray:
    mapping = {group: index for index, group in enumerate(sorted(set(str(v) for v in groups)))}
    return np.asarray([mapping[str(group)] for group in groups], dtype=np.int32)


def _write_private_cache(
    path: Path,
    *,
    fit_x: np.ndarray,
    fit_forward: np.ndarray,
    fit_backward: np.ndarray,
    fit_forward_seed: np.ndarray,
    fit_backward_seed: np.ndarray,
    fit_clusters: np.ndarray,
    selection_x: np.ndarray,
    selection_forward: np.ndarray,
    selection_backward: np.ndarray,
    selection_forward_seed: np.ndarray,
    selection_backward_seed: np.ndarray,
    selection_clusters: np.ndarray,
    feature_names: Sequence[str],
    hashes: Mapping[str, str],
) -> None:
    if path.exists():
        raise FileExistsError(f"private bidirectional cache already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(["emotiontalk_bidirectional_oof_cache_v1"]),
            fit_x=np.asarray(fit_x, dtype=np.float64),
            fit_forward=np.asarray(fit_forward, dtype=np.float64),
            fit_backward=np.asarray(fit_backward, dtype=np.float64),
            fit_forward_seed=np.asarray(fit_forward_seed, dtype=np.float64),
            fit_backward_seed=np.asarray(fit_backward_seed, dtype=np.float64),
            fit_cluster_codes=np.asarray(fit_clusters, dtype=np.int32),
            selection_x=np.asarray(selection_x, dtype=np.float64),
            selection_forward=np.asarray(selection_forward, dtype=np.float64),
            selection_backward=np.asarray(selection_backward, dtype=np.float64),
            selection_forward_seed=np.asarray(selection_forward_seed, dtype=np.float64),
            selection_backward_seed=np.asarray(selection_backward_seed, dtype=np.float64),
            selection_cluster_codes=np.asarray(selection_clusters, dtype=np.int32),
            feature_names=np.asarray(tuple(feature_names), dtype=str),
            base_config_sha256=np.asarray([hashes["base_config_sha256"]]),
            utility_config_sha256=np.asarray([hashes["utility_config_sha256"]]),
        )
    os.replace(temporary, path)


def _write_probability_checkpoint(
    path: Path,
    *,
    task_positions: np.ndarray,
    probability: np.ndarray,
    hashes: Mapping[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(["emotiontalk_bidirectional_probability_checkpoint_v1"]),
            task_positions=np.asarray(task_positions, dtype=np.int64),
            probability=np.asarray(probability, dtype=np.float64),
            base_config_sha256=np.asarray([hashes["base_config_sha256"]]),
            utility_config_sha256=np.asarray([hashes["utility_config_sha256"]]),
            feature_sha256=np.asarray([hashes["feature_sha256"]]),
        )
    os.replace(temporary, path)


def _load_probability_checkpoint(
    path: Path,
    *,
    expected_positions: np.ndarray,
    expected_shape: tuple[int, ...],
    hashes: Mapping[str, str],
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "schema_version",
            "task_positions",
            "probability",
            "base_config_sha256",
            "utility_config_sha256",
            "feature_sha256",
        }
        if set(archive.files) != required:
            raise ContractError(f"checkpoint schema changed: {path}")
        if str(archive["schema_version"][0]) != "emotiontalk_bidirectional_probability_checkpoint_v1":
            raise ContractError(f"checkpoint version changed: {path}")
        for name in ("base_config_sha256", "utility_config_sha256", "feature_sha256"):
            if str(archive[name][0]) != hashes[name]:
                raise ContractError(f"checkpoint hash mismatch for {name}: {path}")
        positions = archive["task_positions"]
        probability = archive["probability"]
    if positions.dtype != np.int64 or not np.array_equal(positions, expected_positions):
        raise ContractError(f"checkpoint task positions changed: {path}")
    if probability.dtype != np.float64 or probability.shape != expected_shape:
        raise ContractError(f"checkpoint probability shape/dtype changed: {path}")
    if not np.isfinite(probability).all():
        raise ContractError(f"checkpoint contains non-finite probability: {path}")
    return probability


def run_bidirectional_oof(
    data_dir: Path,
    feature_path: Path,
    base_config_path: Path,
    utility_config_path: Path,
    output_path: Path,
    private_cache_path: Path,
    checkpoint_dir: Path | None = None,
) -> dict:
    if output_path.exists():
        raise FileExistsError(f"bidirectional output already exists: {output_path}")
    if private_cache_path.exists():
        raise FileExistsError(f"private bidirectional cache already exists: {private_cache_path}")
    base_config = _read_json(base_config_path)
    utility_config = _read_json(utility_config_path)
    _validate_config(utility_config, base_config)

    hashes = {
        "base_config_sha256": sha256_file(base_config_path),
        "utility_config_sha256": sha256_file(utility_config_path),
        "feature_sha256": sha256_file(feature_path),
        "transcription_sha256": sha256_file(data_dir / "transcription.csv"),
        "labels_sha256": sha256_file(data_dir / "mm_label.npz"),
    }

    keys, audio, video, quality, quality_names, feature_config_sha = load_media_split(
        feature_path, "train_corpus"
    )
    full_frame = load_unlabeled_frame(data_dir, keys)
    groups_all, roles_all, _ = assign_frame_roles(
        full_frame,
        dataset="emotiontalk",
        role_protocol=str(utility_config["data_roles"]["split_protocol_id"]),
        role_ranges=_role_ranges(utility_config),
    )
    allowed_roles = {"base_and_utility_fit", "model_selection"}
    materialized = np.asarray([role in allowed_roles for role in roles_all], dtype=bool)
    work_frame = full_frame.loc[materialized].copy().reset_index(drop=True)
    work_frame["_row_id"] = np.arange(len(work_frame), dtype=int)
    work_audio = audio[materialized]
    work_video = video[materialized]
    work_quality = quality[materialized]
    work_groups = groups_all[materialized]
    work_roles = roles_all[materialized]
    y = _load_materialized_labels(data_dir, keys, work_frame["key"].astype(str).tolist())
    fit_indices = np.flatnonzero(work_roles == "base_and_utility_fit")
    selection_indices = np.flatnonzero(work_roles == "model_selection")
    if set(work_groups[fit_indices]) & set(work_groups[selection_indices]):
        raise ContractError("fit and model-selection groups overlap")
    if not np.array_equal(np.sort(np.unique(y[fit_indices])), np.arange(len(LABEL_NAMES))):
        raise ContractError("fit role does not contain all emotion classes")

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
    fit_tasks = [task for task in tasks if int(task.query_index) in fit_set]
    selection_tasks = [task for task in tasks if int(task.query_index) in selection_set]
    if not fit_tasks or not selection_tasks:
        raise ContractError("bidirectional task sampling produced an empty role")
    if any(set(context) == set(task.addition_context) | {task.candidate_index}
           for task in tasks for context in [task.deletion_context]):
        raise AssertionError("trivial bidirectional task escaped sampling")

    seeds = tuple(int(value) for value in base_config["seeds"])
    modalities = ("text", "audio", "video")
    n_classes = len(LABEL_NAMES)
    fit_probability = np.full(
        (len(seeds), len(fit_tasks), len(CONTEXT_NAMES), n_classes),
        np.nan,
        dtype=np.float64,
    )
    fit_task_query = np.asarray([task.query_index for task in fit_tasks], dtype=np.int64)
    splitter = GroupKFold(n_splits=int(base_config["crossfit_folds"]))
    fit_local = np.arange(len(fit_indices))
    augmentation = utility_config["base_subset_augmentation"]
    for fold, (train_local, held_local) in enumerate(
        splitter.split(fit_local, y[fit_indices], work_groups[fit_indices]), start=1
    ):
        train_index = fit_indices[train_local]
        held_index = fit_indices[held_local]
        held_mask = np.isin(fit_task_query, held_index)
        held_positions = np.flatnonzero(held_mask).astype(np.int64, copy=False)
        held_tasks = [task for task, selected in zip(fit_tasks, held_mask, strict=True) if selected]
        checkpoint_path = None if checkpoint_dir is None else checkpoint_dir / f"fold_{fold}.npz"
        expected_shape = (len(seeds), len(held_tasks), len(CONTEXT_NAMES), n_classes)
        if checkpoint_path is not None and checkpoint_path.exists():
            fit_probability[:, held_mask] = _load_probability_checkpoint(
                checkpoint_path,
                expected_positions=held_positions,
                expected_shape=expected_shape,
                hashes=hashes,
            )
            print(
                f"bidirectional OOF fold {fold} restored: held_tasks={len(held_tasks)}",
                flush=True,
            )
            continue
        train_rows, train_contexts, sample_weight = augmented_training_rows(
            histories,
            fit_tasks,
            train_index,
            maximum_contexts_per_query=int(augmentation["maximum_contexts_per_query"]),
            seed=int(augmentation["seed"]) + fold,
        )
        processors = _fit_processors(base_config, work_frame, work_audio, work_video, train_index)
        current = _transform_processors(processors, work_frame, work_audio, work_video)
        train_blocks = build_context_blocks(
            current, work_quality, quality_names, train_rows, train_contexts
        )
        train_x = base_features(train_blocks, modalities, use_history=True)
        models = []
        for seed in seeds:
            model = make_classifier(base_config, seed + fold * 1000)
            model.fit(train_x, y[train_rows], sample_weight=sample_weight)
            models.append(model)
        held_probability = _predict_tasks(
            models,
            current,
            work_quality,
            quality_names,
            held_tasks,
            modalities,
            n_classes,
        )
        fit_probability[:, held_mask] = held_probability
        if checkpoint_path is not None:
            _write_probability_checkpoint(
                checkpoint_path,
                task_positions=held_positions,
                probability=held_probability,
                hashes=hashes,
            )
        print(
            f"bidirectional OOF fold {fold} complete: "
            f"train_variants={len(train_rows)} held_tasks={len(held_tasks)}",
            flush=True,
        )
    if np.isnan(fit_probability).any():
        raise RuntimeError("incomplete bidirectional OOF probability tensor")

    selection_checkpoint = None if checkpoint_dir is None else checkpoint_dir / "selection.npz"
    selection_positions = np.arange(len(selection_tasks), dtype=np.int64)
    selection_shape = (len(seeds), len(selection_tasks), len(CONTEXT_NAMES), n_classes)
    if selection_checkpoint is not None and selection_checkpoint.exists():
        selection_probability = _load_probability_checkpoint(
            selection_checkpoint,
            expected_positions=selection_positions,
            expected_shape=selection_shape,
            hashes=hashes,
        )
        print(
            f"bidirectional model-selection prediction restored: tasks={len(selection_tasks)}",
            flush=True,
        )
    else:
        processors = _fit_processors(base_config, work_frame, work_audio, work_video, fit_indices)
        current = _transform_processors(processors, work_frame, work_audio, work_video)
        final_rows, final_contexts, final_weight = augmented_training_rows(
            histories,
            fit_tasks,
            fit_indices,
            maximum_contexts_per_query=int(augmentation["maximum_contexts_per_query"]),
            seed=int(augmentation["seed"]),
        )
        final_blocks = build_context_blocks(
            current, work_quality, quality_names, final_rows, final_contexts
        )
        final_x = base_features(final_blocks, modalities, use_history=True)
        final_models = []
        for seed in seeds:
            model = make_classifier(base_config, seed)
            model.fit(final_x, y[final_rows], sample_weight=final_weight)
            final_models.append(model)
        selection_probability = _predict_tasks(
            final_models,
            current,
            work_quality,
            quality_names,
            selection_tasks,
            modalities,
            n_classes,
        )
        if selection_checkpoint is not None:
            _write_probability_checkpoint(
                selection_checkpoint,
                task_positions=selection_positions,
                probability=selection_probability,
                hashes=hashes,
            )
        print(
            f"bidirectional model-selection prediction complete: "
            f"train_variants={len(final_rows)} tasks={len(selection_tasks)}",
            flush=True,
        )

    fit_forward, fit_backward, fit_forward_seed, fit_backward_seed = _targets_from_probability(
        y, fit_tasks, fit_probability
    )
    (
        selection_forward,
        selection_backward,
        selection_forward_seed,
        selection_backward_seed,
    ) = _targets_from_probability(y, selection_tasks, selection_probability)
    fit_x, feature_names = probability_task_features(
        np.mean(fit_probability, axis=0), fit_tasks, histories
    )
    selection_x, selection_feature_names = probability_task_features(
        np.mean(selection_probability, axis=0), selection_tasks, histories
    )
    if feature_names != selection_feature_names:
        raise RuntimeError("fit and model-selection task feature schemas differ")

    fit_task_groups = np.asarray([work_groups[task.query_index] for task in fit_tasks], dtype=object)
    selection_task_groups = np.asarray(
        [work_groups[task.query_index] for task in selection_tasks], dtype=object
    )
    _write_private_cache(
        private_cache_path,
        fit_x=fit_x,
        fit_forward=fit_forward,
        fit_backward=fit_backward,
        fit_forward_seed=fit_forward_seed,
        fit_backward_seed=fit_backward_seed,
        fit_clusters=_cluster_codes(fit_task_groups),
        selection_x=selection_x,
        selection_forward=selection_forward,
        selection_backward=selection_backward,
        selection_forward_seed=selection_forward_seed,
        selection_backward_seed=selection_backward_seed,
        selection_clusters=_cluster_codes(selection_task_groups),
        feature_names=feature_names,
        hashes=hashes,
    )

    role_counts = {
        role: {
            "rows": int(np.sum(roles_all == role)),
            "groups": int(len(set(groups_all[roles_all == role]))),
        }
        for role in ROLE_RANGES
    }
    result = {
        "protocol": str(utility_config["protocol"]),
        "status": "train_only_different_set_oof_supervision_complete; utility_model_not_yet_selected",
        "claim_boundary": (
            "This run audits whether non-trivial bidirectional utility supervision exists. "
            "It does not establish classification or safety improvement."
        ),
        "hashes": {**hashes, "feature_config_sha256": feature_config_sha},
        "roles": role_counts,
        "task_counts": {
            "fit_oof": int(len(fit_tasks)),
            "model_selection": int(len(selection_tasks)),
            "fit_groups": int(len(set(fit_task_groups))),
            "model_selection_groups": int(len(set(selection_task_groups))),
            "probability_contexts_per_task": int(len(CONTEXT_NAMES)),
            "base_seeds": int(len(seeds)),
        },
        "target_summary": {
            "fit_oof": _target_summary(fit_forward, fit_backward),
            "model_selection": _target_summary(selection_forward, selection_backward),
        },
        "seed_stability": {
            "fit_forward": _seed_stability(fit_forward_seed),
            "fit_backward": _seed_stability(fit_backward_seed),
            "model_selection_forward": _seed_stability(selection_forward_seed),
            "model_selection_backward": _seed_stability(selection_backward_seed),
        },
        "cache_contract": {
            "schema": "emotiontalk_bidirectional_oof_cache_v1",
            "numeric_dtype": "float64",
            "task_feature_count": int(fit_x.shape[1]),
            "contains_row_identifiers": False,
            "contains_gold_labels": False,
            "private_not_for_publication": True,
        },
        "sealed_audit": {
            "calibration_rows_used_for_training_or_metrics": 0,
            "internal_holdout_rows_used_for_training_or_metrics": 0,
            "validation_rows_used": 0,
            "test_rows_used": 0,
            "row_level_output_emitted": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(result, output_path.resolve())
    return result
