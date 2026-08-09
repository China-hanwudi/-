"""One-prediction-per-query reversible history-policy development runner.

The candidate utility model is trained on the existing 59-D private cache and
scores the existing sampled selection tasks.  Scores for repeated
``(query, candidate)`` pairs are averaged across coalition draws.  Every query
then reconstructs its selected history independently from its immutable,
strictly-past history; no deletion is carried to a later query.  If no candidate
clears the fit-OOF frozen threshold, that query receives current-only features.

The downstream classifier remains the registered lightweight linear SGD model
over frozen TF-IDF/PCA multimodal features.  This is therefore an open-role,
query-level development experiment, not confirmation of the new causal
multimodal backbone and not a sealed-split result.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse

from .bidirectional_utility_model import (
    DEFAULT_SEEDS,
    PRIMARY_HISTORY_COVERAGE,
    BidirectionalUtilityCache,
    default_model_specs,
    fit_oof_coverage_threshold,
    fit_utility_model,
    group_oof_predictions,
    load_private_oof_cache,
)
from .data_contract import ContractError, write_json_atomic
from .emotiontalk_bidirectional_oof import (
    _read_json,
    _role_ranges,
    _validate_config,
    augmented_training_rows,
    build_context_blocks,
    probability_task_features,
)
from .emotiontalk_endpoint_diagnostic import _load_materialized_labels, assign_frame_roles
from .emotiontalk_multimodal_external import (
    _align_probabilities,
    _fit_processors,
    _transform_processors,
    base_features,
    load_media_split,
    load_unlabeled_frame,
)
from .emotiontalk_sampled_context_runner import (
    EXPECTED_MODEL_NAMES,
    OpenRoleDiagnosticError,
    _assert_aggregate_output,
    assemble_fit_probability_checkpoints,
    load_selection_probability_checkpoint,
    reconstruct_open_role_tasks,
    recover_query_labels_from_cached_utilities,
    verify_recomputed_59d_cache,
)
from .emotiontalk_text_p1 import LABEL_NAMES, build_history_indices
from .meld_text_pilot import (
    NLL_PROBABILITY_FLOOR,
    make_classifier,
    prediction_metrics,
    sha256_file,
    true_class_loss,
)
from .negative_transfer_benchmark import describe_excess


REPORT_SCHEMA_VERSION = "emotiontalk_open_role_query_policy_v3"
MODALITIES = ("text", "audio", "video")
UTILITY_STRATEGIES = {
    "forward_only_mlp": "forward_selected_history",
    "backward_only_mlp": "backward_selected_history",
    "pseudo_bidirectional_same_set_mlp": "pseudo_selected_history",
    "bidirectional_shared_mlp": "true_bidirectional_selected_history",
}
FORBIDDEN_ROLE_TOKENS = frozenset(
    {"calibration", "holdout", "sealed", "validation", "test"}
)


class QueryPolicyContractError(ContractError):
    """Raised when a query policy, feature set, or data role is unsafe."""


@dataclass(frozen=True)
class UtilitySeedScores:
    seed: int
    threshold: float
    selection_scores: np.ndarray
    fit_query_candidate_pairs: int
    realized_fit_query_candidate_coverage: float


@dataclass(frozen=True)
class OpenRoleBaseData:
    frame: Any
    audio: np.ndarray
    video: np.ndarray
    quality: np.ndarray
    quality_names: tuple[str, ...]
    labels: np.ndarray
    groups: np.ndarray
    fit_indices: np.ndarray
    selection_indices: np.ndarray
    histories: tuple[tuple[int, ...], ...]
    label_container_rows_deserialized: int
    non_open_label_rows_deserialized: int


def aggregate_candidate_draw_scores(
    tasks: Sequence[object],
    decision_scores: np.ndarray,
) -> dict[int, dict[int, float]]:
    """Average repeated task scores for each ordered ``(query, candidate)``."""

    scores = np.asarray(decision_scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != len(tasks) or not np.isfinite(scores).all():
        raise QueryPolicyContractError("candidate task scores must be finite and task-aligned")
    buckets: dict[tuple[int, int], list[float]] = {}
    for task, score in zip(tasks, scores, strict=True):
        query = int(getattr(task, "query_index"))
        candidate = int(getattr(task, "candidate_index"))
        buckets.setdefault((query, candidate), []).append(float(score))
    result: dict[int, dict[int, float]] = {}
    for (query, candidate), values in sorted(buckets.items()):
        result.setdefault(query, {})[candidate] = float(np.mean(values))
    return result


def fit_query_candidate_coverage_threshold(
    tasks: Sequence[object],
    decision_scores: np.ndarray,
    *,
    target_coverage: float,
) -> tuple[float, int, float]:
    """Freeze coverage on the same query-candidate unit used at deployment.

    Repeated coalition draws are first averaged within each ordered
    ``(query, candidate)`` pair.  Freezing a task-row threshold and later
    applying it to these averages changes both the estimand and the realized
    coverage whenever candidate pairs have unequal draw multiplicities.
    """

    nested = aggregate_candidate_draw_scores(tasks, decision_scores)
    pair_scores = np.asarray(
        [
            nested[query][candidate]
            for query in sorted(nested)
            for candidate in sorted(nested[query])
        ],
        dtype=np.float64,
    )
    if not len(pair_scores):
        raise QueryPolicyContractError("coverage threshold requires query-candidate pairs")
    threshold = fit_oof_coverage_threshold(pair_scores, float(target_coverage))
    realized = float(np.mean(pair_scores > threshold))
    return float(threshold), int(len(pair_scores)), realized


def build_reversible_selected_contexts(
    query_indices: Sequence[int],
    histories: Sequence[Sequence[int]],
    candidate_scores: Mapping[int, Mapping[int, float]],
    *,
    threshold: float,
) -> tuple[tuple[int, ...], ...]:
    """Build a fresh selected set per query without mutating any history.

    Candidate order follows the immutable strict-past history order.  Equality
    to the frozen threshold is a fallback.  Queries with no passing candidate
    receive the empty context and therefore current-only inference.
    """

    queries = tuple(int(value) for value in query_indices)
    if len(queries) != len(set(queries)):
        raise QueryPolicyContractError("query-level inference requires one unique query row")
    frozen_threshold = float(threshold)
    if not np.isfinite(frozen_threshold):
        raise QueryPolicyContractError("utility threshold must be finite")
    source_rows = len(histories)
    contexts: list[tuple[int, ...]] = []
    for query in queries:
        if query < 0 or query >= source_rows:
            raise QueryPolicyContractError("query index is outside the history source")
        past = tuple(int(value) for value in histories[query])
        if len(past) != len(set(past)):
            raise QueryPolicyContractError("strict-past history contains duplicate candidates")
        score_map = {
            int(candidate): float(score)
            for candidate, score in candidate_scores.get(query, {}).items()
        }
        if not all(np.isfinite(value) for value in score_map.values()):
            raise QueryPolicyContractError("aggregated candidate scores contain non-finite values")
        outside = set(score_map) - set(past)
        if outside:
            raise QueryPolicyContractError(
                f"candidate score refers to a non-past row: {sorted(outside)}"
            )
        contexts.append(
            tuple(candidate for candidate in past if score_map.get(candidate, -np.inf) > frozen_threshold)
        )
    return tuple(contexts)


def coverage_matched_recency_contexts(
    query_indices: Sequence[int],
    histories: Sequence[Sequence[int]],
    reference_contexts: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Choose the same number of most-recent candidates as a reference policy."""

    queries = tuple(int(value) for value in query_indices)
    if len(queries) != len(set(queries)) or len(queries) != len(reference_contexts):
        raise QueryPolicyContractError("recency baseline inputs must contain one row per query")
    result: list[tuple[int, ...]] = []
    for query, reference in zip(queries, reference_contexts, strict=True):
        past = tuple(int(value) for value in histories[query])
        count = len(tuple(reference))
        if count > len(past):
            raise QueryPolicyContractError("reference selected more candidates than exist in history")
        result.append(tuple() if count == 0 else past[-count:])
    return tuple(result)


def validate_strict_past_contexts(
    query_indices: Sequence[int],
    contexts: Sequence[Sequence[int]],
    histories: Sequence[Sequence[int]],
) -> None:
    queries = tuple(int(value) for value in query_indices)
    if len(queries) != len(set(queries)) or len(queries) != len(contexts):
        raise QueryPolicyContractError("context inference must contain exactly one row per query")
    for query, context in zip(queries, contexts, strict=True):
        if query < 0 or query >= len(histories):
            raise QueryPolicyContractError("query index is outside the history source")
        values = tuple(int(value) for value in context)
        if len(values) != len(set(values)):
            raise QueryPolicyContractError("query context contains duplicate history rows")
        if not set(values).issubset(set(int(value) for value in histories[query])):
            raise QueryPolicyContractError("query context contains a non-past history row")


def predict_query_context_probabilities_by_model(
    models: Sequence[object],
    current: Mapping[str, sparse.csr_matrix | np.ndarray],
    quality: np.ndarray,
    quality_names: Sequence[str],
    query_indices: Sequence[int],
    contexts: Sequence[Sequence[int]],
    histories: Sequence[Sequence[int]],
    *,
    n_classes: int,
) -> np.ndarray:
    """Predict one probability row per base model and unique query/context set."""

    model_tuple = tuple(models)
    if not model_tuple:
        raise QueryPolicyContractError("query inference requires at least one base model")
    queries = np.asarray(tuple(int(value) for value in query_indices), dtype=np.int64)
    validate_strict_past_contexts(queries, contexts, histories)
    blocks = build_context_blocks(
        current,
        np.asarray(quality),
        quality_names,
        queries,
        contexts,
    )
    features = base_features(blocks, MODALITIES, use_history=True)
    per_seed: list[np.ndarray] = []
    for model in model_tuple:
        raw = np.asarray(model.predict_proba(features), dtype=np.float64)
        aligned = _align_probabilities(model, raw, int(n_classes))
        if aligned.shape != (len(queries), int(n_classes)) or not np.isfinite(aligned).all():
            raise QueryPolicyContractError("base-model query probability shape is invalid")
        per_seed.append(aligned)
    probability = np.stack(per_seed)
    if not np.allclose(probability.sum(axis=2), 1.0, rtol=1e-7, atol=1e-9):
        raise QueryPolicyContractError("query probabilities left the class simplex")
    return np.asarray(probability, dtype=np.float64)


def predict_query_context_probabilities(
    models: Sequence[object],
    current: Mapping[str, sparse.csr_matrix | np.ndarray],
    quality: np.ndarray,
    quality_names: Sequence[str],
    query_indices: Sequence[int],
    contexts: Sequence[Sequence[int]],
    histories: Sequence[Sequence[int]],
    *,
    n_classes: int,
) -> np.ndarray:
    """Predict one base-seed-ensemble row per unique query/context set."""

    per_model = predict_query_context_probabilities_by_model(
        models,
        current,
        quality,
        quality_names,
        query_indices,
        contexts,
        histories,
        n_classes=n_classes,
    )
    return np.asarray(np.mean(per_model, axis=0), dtype=np.float64)


def query_strategy_metrics(
    labels: np.ndarray,
    probability: np.ndarray,
    current_probability: np.ndarray,
    contexts: Sequence[Sequence[int]],
    histories: Sequence[Sequence[int]],
    query_indices: Sequence[int],
    cluster_codes: np.ndarray,
    *,
    ece_bins: int,
) -> dict[str, float | int]:
    """Return pooled one-row-per-query classification and safety metrics."""

    y = np.asarray(labels)
    values = np.asarray(probability, dtype=np.float64)
    current = np.asarray(current_probability, dtype=np.float64)
    clusters = np.asarray(cluster_codes)
    queries = tuple(int(value) for value in query_indices)
    if (
        y.ndim != 1
        or not np.issubdtype(y.dtype, np.integer)
        or values.shape != current.shape
        or values.shape != (len(y), len(LABEL_NAMES))
        or clusters.shape != (len(y),)
        or len(queries) != len(y)
    ):
        raise QueryPolicyContractError("query-level metric inputs are not aligned")
    if np.any((y < 0) | (y >= len(LABEL_NAMES))):
        raise QueryPolicyContractError("query label is outside the emotion class range")
    if not np.isfinite(values).all() or not np.isfinite(current).all():
        raise QueryPolicyContractError("query-level probabilities contain non-finite values")
    if np.any(values < 0.0) or np.any(current < 0.0):
        raise QueryPolicyContractError("query-level probabilities contain negative values")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9) or not np.allclose(
        current.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9
    ):
        raise QueryPolicyContractError("query-level probabilities left the class simplex")
    validate_strict_past_contexts(queries, contexts, histories)
    classification = prediction_metrics(y.astype(np.int64), values, int(ece_bins))
    excess = true_class_loss(y, values) - true_class_loss(y, current)
    excess_summary = describe_excess(excess, clusters)
    counts = np.asarray([len(tuple(context)) for context in contexts], dtype=np.int64)
    available = np.asarray([len(histories[query]) for query in queries], dtype=np.int64)
    candidate_fraction = counts / np.maximum(available, 1)
    return {
        "queries": int(len(y)),
        "pooled_macro_f1": float(classification["macro_f1"]),
        "pooled_accuracy": float(classification["accuracy"]),
        "pooled_weighted_f1": float(classification["weighted_f1"]),
        "pooled_nll": float(classification["log_loss"]),
        "pooled_brier": float(classification["brier"]),
        "mean_excess_nll_vs_current": float(excess_summary["mean_excess_loss"]),
        "harm_rate_vs_current": float(excess_summary["harm_rate"]),
        "p90_excess_nll_vs_current": float(excess_summary["p90_excess_loss"]),
        "cvar90_excess_nll_vs_current": float(excess_summary["cvar90_excess_loss"]),
        "actual_history_coverage": float(np.mean(counts > 0)),
        "mean_selected_history_count": float(np.mean(counts)),
        "mean_candidate_fraction_selected": float(np.mean(candidate_fraction)),
    }


def _numeric_summary(
    records: Sequence[Mapping[str, float | int]],
    *,
    expected_count: int = 5,
) -> dict[str, dict[str, float]]:
    rows = tuple(records)
    if len(rows) != int(expected_count):
        raise QueryPolicyContractError(
            f"numeric summary requires exactly {int(expected_count)} records"
        )
    keys = tuple(key for key in rows[0] if key != "queries")
    if any(set(row) != set(rows[0]) for row in rows):
        raise QueryPolicyContractError("utility-seed metric schemas differ")
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise QueryPolicyContractError("utility-seed metric contains non-finite values")
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    return result


def _mean_metric_record(
    records: Sequence[Mapping[str, float | int]],
    *,
    expected_count: int,
) -> dict[str, float | int]:
    rows = tuple(records)
    summary = _numeric_summary(rows, expected_count=int(expected_count))
    queries = {int(row["queries"]) for row in rows}
    if len(queries) != 1:
        raise QueryPolicyContractError("metric records contain different query counts")
    return {
        "queries": int(next(iter(queries))),
        **{key: float(value["mean"]) for key, value in summary.items()},
    }


def summarize_five_base_seed_strategy(
    records: Sequence[Mapping[str, float | int]],
) -> dict[str, object]:
    """Primary fixed-strategy estimand across five independently fitted bases."""

    return {
        "base_seed_count": 5,
        "estimand": "mean metric across five independently fitted base seeds",
        "metrics": _numeric_summary(records, expected_count=5),
    }


def summarize_joint_seed_strategy(
    record_grid: Sequence[Sequence[Mapping[str, float | int]]],
    current_records: Sequence[Mapping[str, float | int]],
    *,
    minimum_macro_f1_gain: float,
    minimum_history_coverage: float,
) -> dict[str, object]:
    """Summarize the registered 5 utility-seed x 5 base-seed grid.

    The 25 cells are the primary point-estimand unit.  They are not treated as
    25 independent samples: the five utility-seed means are retained for the
    pre-registered four-of-five development gate, and all standard deviations
    are explicitly descriptive rather than inferential.
    """

    grid = tuple(tuple(row) for row in record_grid)
    if len(grid) != 5 or any(len(row) != 5 for row in grid):
        raise QueryPolicyContractError("joint seed estimand requires a 5 by 5 metric grid")
    current_rows = tuple(current_records)
    current_mean = _mean_metric_record(current_rows, expected_count=5)
    utility_seed_means = tuple(
        _mean_metric_record(row, expected_count=5) for row in grid
    )
    flattened = tuple(record for row in grid for record in row)
    success = [
        (
            float(row["pooled_macro_f1"])
            - float(current_mean["pooled_macro_f1"])
            >= float(minimum_macro_f1_gain)
            and float(row["mean_excess_nll_vs_current"]) < 0.0
            and float(row["actual_history_coverage"])
            >= float(minimum_history_coverage)
        )
        for row in utility_seed_means
    ]
    successful = int(sum(success))
    return {
        "utility_seed_count": 5,
        "base_seed_count": 5,
        "joint_seed_grid_count": 25,
        "primary_estimand": "mean metric across the 5 utility-seed x 5 base-seed grid",
        "dependence_note": (
            "The 25 grid cells share fitted models; reported standard deviations are "
            "descriptive and are not a 25-independent-sample standard error."
        ),
        "metrics_across_25_seed_combinations": _numeric_summary(
            flattened, expected_count=25
        ),
        "metrics_across_five_utility_seed_means": _numeric_summary(
            utility_seed_means, expected_count=5
        ),
        "success_definition": {
            "unit": "each utility seed averaged across the five base seeds",
            "minimum_macro_f1_gain_vs_current": float(minimum_macro_f1_gain),
            "mean_excess_nll_vs_current_strictly_below_zero": True,
            "minimum_actual_history_coverage": float(minimum_history_coverage),
        },
        "successful_utility_seeds_out_of_five": successful,
        "meets_four_of_five": bool(successful >= 4),
    }


def summarize_utility_seed_strategy(
    records: Sequence[Mapping[str, float | int]],
    current_metrics: Mapping[str, float | int],
    *,
    minimum_macro_f1_gain: float,
    minimum_history_coverage: float,
) -> dict[str, object]:
    """Summarize five utility seeds and count jointly successful seeds."""

    rows = tuple(records)
    summary = _numeric_summary(rows)
    current_macro = float(current_metrics["pooled_macro_f1"])
    success = [
        (
            float(row["pooled_macro_f1"]) - current_macro
            >= float(minimum_macro_f1_gain)
            and float(row["mean_excess_nll_vs_current"]) < 0.0
            and float(row["actual_history_coverage"]) >= float(minimum_history_coverage)
        )
        for row in rows
    ]
    successful = int(sum(success))
    return {
        "utility_seed_count": 5,
        "metrics": summary,
        "success_definition": {
            "minimum_macro_f1_gain_vs_current": float(minimum_macro_f1_gain),
            "mean_excess_nll_vs_current_strictly_below_zero": True,
            "minimum_actual_history_coverage": float(minimum_history_coverage),
        },
        "successful_utility_seeds_out_of_five": successful,
        "meets_four_of_five": bool(successful >= 4),
    }


def _materialize_open_role_base_data(
    data_dir: Path,
    feature_path: Path,
    utility_config: Mapping[str, object],
) -> OpenRoleBaseData:
    keys, audio, video, quality, quality_names, _ = load_media_split(
        feature_path, "train_corpus"
    )
    full_frame = load_unlabeled_frame(data_dir, keys)
    groups_all, roles_all, _ = assign_frame_roles(
        full_frame,
        dataset="emotiontalk",
        role_protocol=str(utility_config["data_roles"]["split_protocol_id"]),
        role_ranges=_role_ranges(utility_config),
    )
    open_roles = {"base_and_utility_fit", "model_selection"}
    materialized = np.asarray([str(role) in open_roles for role in roles_all], dtype=bool)
    work_frame = full_frame.loc[materialized].copy().reset_index(drop=True)
    work_frame["_row_id"] = np.arange(len(work_frame), dtype=np.int64)
    work_audio = np.asarray(audio[materialized])
    work_video = np.asarray(video[materialized])
    work_quality = np.asarray(quality[materialized])
    work_groups = np.asarray(groups_all[materialized])
    work_roles = np.asarray(roles_all[materialized])
    if not set(str(value) for value in np.unique(work_roles)).issubset(open_roles):
        raise QueryPolicyContractError("a non-open role entered base-model materialization")
    labels = _load_materialized_labels(
        data_dir,
        keys,
        work_frame["key"].astype(str).tolist(),
    )
    fit_indices = np.flatnonzero(work_roles == "base_and_utility_fit")
    selection_indices = np.flatnonzero(work_roles == "model_selection")
    if set(work_groups[fit_indices]) & set(work_groups[selection_indices]):
        raise QueryPolicyContractError("fit and model-selection groups overlap")
    if not np.array_equal(np.sort(np.unique(labels[fit_indices])), np.arange(len(LABEL_NAMES))):
        raise QueryPolicyContractError("fit role does not contain all emotion classes")
    histories = build_history_indices(work_frame)
    fit_set = set(int(value) for value in fit_indices)
    selection_set = set(int(value) for value in selection_indices)
    if any(
        not set(int(value) for value in histories[int(query)]).issubset(fit_set)
        for query in fit_indices
    ):
        raise QueryPolicyContractError("fit history crosses the open-role boundary")
    if any(
        not set(int(value) for value in histories[int(query)]).issubset(selection_set)
        for query in selection_indices
    ):
        raise QueryPolicyContractError("model-selection history crosses the role boundary")
    return OpenRoleBaseData(
        frame=work_frame,
        audio=work_audio,
        video=work_video,
        quality=work_quality,
        quality_names=tuple(str(value) for value in quality_names),
        labels=np.asarray(labels, dtype=np.int64),
        groups=work_groups,
        fit_indices=fit_indices,
        selection_indices=selection_indices,
        histories=histories,
        # The upstream archive stores train_corpus labels as one pickled object
        # mapping.  np.load(...).item() necessarily materializes that complete
        # mapping even though only open-role keys are indexed below.  Preserve
        # this distinction in the public audit instead of claiming an epistemic
        # seal that the storage layout cannot provide.
        label_container_rows_deserialized=int(len(keys)),
        non_open_label_rows_deserialized=int((~materialized).sum()),
    )


def _train_linear_base_ensemble(
    data: OpenRoleBaseData,
    base_config: Mapping[str, object],
    utility_config: Mapping[str, object],
    fit_tasks: Sequence[object],
) -> tuple[tuple[object, ...], Mapping[str, sparse.csr_matrix | np.ndarray]]:
    processors = _fit_processors(
        dict(base_config), data.frame, data.audio, data.video, data.fit_indices
    )
    current = _transform_processors(processors, data.frame, data.audio, data.video)
    augmentation = utility_config["base_subset_augmentation"]
    train_rows, train_contexts, sample_weight = augmented_training_rows(
        data.histories,
        fit_tasks,
        data.fit_indices,
        maximum_contexts_per_query=int(augmentation["maximum_contexts_per_query"]),
        seed=int(augmentation["seed"]),
    )
    train_blocks = build_context_blocks(
        current,
        data.quality,
        data.quality_names,
        train_rows,
        train_contexts,
    )
    train_x = base_features(train_blocks, MODALITIES, use_history=True)
    models: list[object] = []
    for seed in base_config["seeds"]:
        model = make_classifier(dict(base_config), int(seed))
        model.fit(train_x, data.labels[train_rows], sample_weight=sample_weight)
        models.append(model)
    if len(models) != 5:
        raise QueryPolicyContractError("linear base ensemble requires five frozen seeds")
    return tuple(models), current


def _fit_utility_seed_scores(
    cache: BidirectionalUtilityCache,
    fit_tasks: Sequence[object],
) -> dict[str, tuple[UtilitySeedScores, ...]]:
    specs = tuple(default_model_specs())
    if tuple(spec.name for spec in specs) != EXPECTED_MODEL_NAMES:
        raise QueryPolicyContractError("registered four-model utility set changed")
    if len(DEFAULT_SEEDS) != 5 or len(set(DEFAULT_SEEDS)) != 5:
        raise QueryPolicyContractError("utility model seeds must contain five distinct values")
    result: dict[str, tuple[UtilitySeedScores, ...]] = {}
    for spec in specs:
        states: list[UtilitySeedScores] = []
        for seed in DEFAULT_SEEDS:
            oof = group_oof_predictions(cache.fit, spec, seed=int(seed), maximum_splits=5)
            threshold, pair_count, realized_coverage = (
                fit_query_candidate_coverage_threshold(
                    fit_tasks,
                    oof.predictions.decision_score,
                    target_coverage=PRIMARY_HISTORY_COVERAGE,
                )
            )
            fitted = fit_utility_model(cache.fit, spec, seed=int(seed))
            prediction = fitted.predict(cache.selection.x)
            states.append(
                UtilitySeedScores(
                    seed=int(seed),
                    threshold=float(threshold),
                    selection_scores=np.asarray(
                        prediction.decision_score, dtype=np.float64
                    ),
                    fit_query_candidate_pairs=int(pair_count),
                    realized_fit_query_candidate_coverage=float(realized_coverage),
                )
            )
        result[spec.name] = tuple(states)
    return result


def _source_hashes(
    data_dir: Path,
    feature_path: Path,
    base_config_path: Path,
    utility_config_path: Path,
    private_cache_path: Path,
    checkpoint_paths: Sequence[Path],
) -> dict[str, object]:
    return {
        "transcription_sha256": sha256_file(data_dir / "transcription.csv"),
        "label_archive_sha256": sha256_file(data_dir / "mm_label.npz"),
        "feature_sha256": sha256_file(feature_path),
        "base_config_sha256": sha256_file(base_config_path),
        "utility_config_sha256": sha256_file(utility_config_path),
        "private_cache_sha256": sha256_file(private_cache_path),
        "checkpoints_sha256": {
            Path(path).name: sha256_file(path) for path in checkpoint_paths
        },
    }


def run_open_role_query_policy(
    data_dir: Path,
    feature_path: Path,
    base_config_path: Path,
    utility_config_path: Path,
    private_cache_path: Path,
    checkpoint_dir: Path,
    output_path: Path,
) -> dict[str, object]:
    """Run one reversible history decision and one prediction per open query."""

    (
        data_dir,
        feature_path,
        base_config_path,
        utility_config_path,
        private_cache_path,
        checkpoint_dir,
        output_path,
    ) = tuple(
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
    if output_path.exists():
        raise FileExistsError(f"query-policy output already exists: {output_path}")
    if not data_dir.is_dir() or not checkpoint_dir.is_dir():
        raise FileNotFoundError("data-dir and checkpoint-dir must exist")
    for path in (feature_path, base_config_path, utility_config_path, private_cache_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    base_config = _read_json(base_config_path)
    utility_config = _read_json(utility_config_path)
    _validate_config(utility_config, base_config)
    expected_hashes = {
        "base_config_sha256": sha256_file(base_config_path),
        "utility_config_sha256": sha256_file(utility_config_path),
        "feature_sha256": sha256_file(feature_path),
    }
    cache = load_private_oof_cache(private_cache_path)
    if cache.source_hashes.get("base_config_sha256") != expected_hashes["base_config_sha256"]:
        raise OpenRoleDiagnosticError("private cache base-config hash mismatch")
    if cache.source_hashes.get("utility_config_sha256") != expected_hashes["utility_config_sha256"]:
        raise OpenRoleDiagnosticError("private cache utility-config hash mismatch")

    task_material, feature_config_sha256 = reconstruct_open_role_tasks(
        feature_path, base_config, utility_config
    )
    fold_paths = tuple(
        checkpoint_dir / f"fold_{fold}.npz"
        for fold in range(1, int(base_config["crossfit_folds"]) + 1)
    )
    selection_checkpoint_path = checkpoint_dir / "selection.npz"
    expected_names = {path.name for path in fold_paths} | {selection_checkpoint_path.name}
    if {path.name for path in checkpoint_dir.glob("*.npz")} != expected_names:
        raise QueryPolicyContractError("checkpoint directory schema changed")
    base_seed_count = len(tuple(base_config["seeds"]))
    fit_probability = assemble_fit_probability_checkpoints(
        fold_paths,
        expected_task_count=len(task_material.fit_tasks),
        expected_seed_count=base_seed_count,
        expected_hashes=expected_hashes,
        expected_positions_by_name=task_material.expected_fold_positions,
    )
    selection_probability = load_selection_probability_checkpoint(
        selection_checkpoint_path,
        expected_task_count=len(task_material.selection_tasks),
        expected_seed_count=base_seed_count,
        expected_hashes=expected_hashes,
    )
    fit_x, fit_names = probability_task_features(
        np.mean(fit_probability, axis=0), task_material.fit_tasks, task_material.histories
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

    data = _materialize_open_role_base_data(data_dir, feature_path, utility_config)
    if data.histories != task_material.histories:
        raise QueryPolicyContractError("materialized strict-past histories changed task order")
    recovered_selection_labels = recover_query_labels_from_cached_utilities(
        task_material.selection_tasks,
        selection_probability,
        cache.selection.forward,
        cache.selection.backward,
    )
    expected_task_labels = np.asarray(
        [data.labels[int(task.query_index)] for task in task_material.selection_tasks],
        dtype=np.int64,
    )
    if not np.array_equal(recovered_selection_labels, expected_task_labels):
        raise QueryPolicyContractError("selection labels disagree with cached utility targets")

    base_models, current = _train_linear_base_ensemble(
        data, base_config, utility_config, task_material.fit_tasks
    )
    selection_queries = tuple(int(value) for value in data.selection_indices)
    if len(selection_queries) != len(set(selection_queries)):
        raise AssertionError("selection query rows are not unique")
    empty_contexts = tuple(tuple() for _ in selection_queries)
    all_contexts = tuple(data.histories[query] for query in selection_queries)
    current_probability_by_base = predict_query_context_probabilities_by_model(
        base_models,
        current,
        data.quality,
        data.quality_names,
        selection_queries,
        empty_contexts,
        data.histories,
        n_classes=len(LABEL_NAMES),
    )
    all_probability_by_base = predict_query_context_probabilities_by_model(
        base_models,
        current,
        data.quality,
        data.quality_names,
        selection_queries,
        all_contexts,
        data.histories,
        n_classes=len(LABEL_NAMES),
    )
    selection_labels = data.labels[data.selection_indices]
    selection_clusters = np.asarray(
        [data.groups[query] for query in selection_queries], dtype=object
    )
    ece_bins = int(base_config.get("ece_bins", 15))
    current_probability = np.mean(current_probability_by_base, axis=0)
    all_probability = np.mean(all_probability_by_base, axis=0)
    current_metrics = query_strategy_metrics(
        selection_labels,
        current_probability,
        current_probability,
        empty_contexts,
        data.histories,
        selection_queries,
        selection_clusters,
        ece_bins=ece_bins,
    )
    all_metrics = query_strategy_metrics(
        selection_labels,
        all_probability,
        current_probability,
        all_contexts,
        data.histories,
        selection_queries,
        selection_clusters,
        ece_bins=ece_bins,
    )
    current_base_records = tuple(
        query_strategy_metrics(
            selection_labels,
            probability,
            probability,
            empty_contexts,
            data.histories,
            selection_queries,
            selection_clusters,
            ece_bins=ece_bins,
        )
        for probability in current_probability_by_base
    )
    all_base_records = tuple(
        query_strategy_metrics(
            selection_labels,
            probability,
            current_probability_by_base[base_index],
            all_contexts,
            data.histories,
            selection_queries,
            selection_clusters,
            ece_bins=ece_bins,
        )
        for base_index, probability in enumerate(all_probability_by_base)
    )

    utility_states = _fit_utility_seed_scores(cache, task_material.fit_tasks)
    by_strategy_grid: dict[
        str, list[tuple[dict[str, float | int], ...]]
    ] = {
        strategy: [] for strategy in UTILITY_STRATEGIES.values()
    }
    by_strategy_grid["coverage_matched_recency"] = []
    by_strategy_ensemble: dict[str, list[dict[str, float | int]]] = {
        strategy: [] for strategy in by_strategy_grid
    }
    threshold_records: dict[str, list[float]] = {name: [] for name in EXPECTED_MODEL_NAMES}
    fit_pair_count_records: dict[str, list[int]] = {
        name: [] for name in EXPECTED_MODEL_NAMES
    }
    fit_pair_coverage_records: dict[str, list[float]] = {
        name: [] for name in EXPECTED_MODEL_NAMES
    }
    for seed_index in range(5):
        true_contexts: tuple[tuple[int, ...], ...] | None = None
        for model_name in EXPECTED_MODEL_NAMES:
            state = utility_states[model_name][seed_index]
            aggregated_scores = aggregate_candidate_draw_scores(
                task_material.selection_tasks, state.selection_scores
            )
            contexts = build_reversible_selected_contexts(
                selection_queries,
                data.histories,
                aggregated_scores,
                threshold=state.threshold,
            )
            probability_by_base = predict_query_context_probabilities_by_model(
                base_models,
                current,
                data.quality,
                data.quality_names,
                selection_queries,
                contexts,
                data.histories,
                n_classes=len(LABEL_NAMES),
            )
            strategy_name = UTILITY_STRATEGIES[model_name]
            base_records = tuple(
                query_strategy_metrics(
                    selection_labels,
                    probability,
                    current_probability_by_base[base_index],
                    contexts,
                    data.histories,
                    selection_queries,
                    selection_clusters,
                    ece_bins=ece_bins,
                )
                for base_index, probability in enumerate(probability_by_base)
            )
            by_strategy_grid[strategy_name].append(base_records)
            by_strategy_ensemble[strategy_name].append(
                query_strategy_metrics(
                    selection_labels,
                    np.mean(probability_by_base, axis=0),
                    current_probability,
                    contexts,
                    data.histories,
                    selection_queries,
                    selection_clusters,
                    ece_bins=ece_bins,
                )
            )
            threshold_records[model_name].append(float(state.threshold))
            fit_pair_count_records[model_name].append(
                int(state.fit_query_candidate_pairs)
            )
            fit_pair_coverage_records[model_name].append(
                float(state.realized_fit_query_candidate_coverage)
            )
            if model_name == "bidirectional_shared_mlp":
                true_contexts = contexts
        if true_contexts is None:
            raise AssertionError("true bidirectional contexts were not constructed")
        recency_contexts = coverage_matched_recency_contexts(
            selection_queries, data.histories, true_contexts
        )
        recency_probability_by_base = predict_query_context_probabilities_by_model(
            base_models,
            current,
            data.quality,
            data.quality_names,
            selection_queries,
            recency_contexts,
            data.histories,
            n_classes=len(LABEL_NAMES),
        )
        recency_base_records = tuple(
            query_strategy_metrics(
                selection_labels,
                recency_probability,
                current_probability_by_base[base_index],
                recency_contexts,
                data.histories,
                selection_queries,
                selection_clusters,
                ece_bins=ece_bins,
            )
            for base_index, recency_probability in enumerate(
                recency_probability_by_base
            )
        )
        by_strategy_grid["coverage_matched_recency"].append(recency_base_records)
        by_strategy_ensemble["coverage_matched_recency"].append(
            query_strategy_metrics(
                selection_labels,
                np.mean(recency_probability_by_base, axis=0),
                current_probability,
                recency_contexts,
                data.histories,
                selection_queries,
                selection_clusters,
                ece_bins=ece_bins,
            )
        )

    gate = utility_config["train_only_go_gate"]
    minimum_macro_gain = float(gate["minimum_macro_f1_gain"])
    minimum_coverage = float(gate["minimum_nontrivial_coverage"])
    strategy_summaries = {
        name: summarize_joint_seed_strategy(
            record_grid,
            current_base_records,
            minimum_macro_f1_gain=minimum_macro_gain,
            minimum_history_coverage=minimum_coverage,
        )
        for name, record_grid in by_strategy_grid.items()
    }
    for name, records in by_strategy_ensemble.items():
        strategy_summaries[name]["base_seed_ensemble_diagnostic"] = (
            summarize_utility_seed_strategy(
                records,
                current_metrics,
                minimum_macro_f1_gain=minimum_macro_gain,
                minimum_history_coverage=minimum_coverage,
            )
        )
    for model_name, strategy_name in UTILITY_STRATEGIES.items():
        thresholds = np.asarray(threshold_records[model_name], dtype=np.float64)
        pair_counts = set(fit_pair_count_records[model_name])
        if len(pair_counts) != 1:
            raise AssertionError("fit query-candidate pair count changed across seeds")
        realized_coverage = np.asarray(
            fit_pair_coverage_records[model_name], dtype=np.float64
        )
        strategy_summaries[strategy_name]["fit_oof_frozen_threshold"] = {
            "unit": "query-candidate score averaged across coalition draws",
            "target_coverage": float(PRIMARY_HISTORY_COVERAGE),
            "query_candidate_pairs": int(next(iter(pair_counts))),
            "threshold_mean": float(thresholds.mean()),
            "threshold_std": float(thresholds.std(ddof=1)),
            "realized_pair_coverage_mean": float(realized_coverage.mean()),
            "realized_pair_coverage_std": float(realized_coverage.std(ddof=1)),
        }

    checkpoint_paths = (*fold_paths, selection_checkpoint_path)
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "open_role_query_level_development_complete_with_label_container_limitation",
        "claim_boundary": (
            "This is a one-prediction-per-query development experiment using a linear base "
            "model over frozen multimodal features. It is not confirmation of the new causal "
            "backbone, not validation/test evidence, and not a final top-conference claim. "
            "The upstream pickled train-label container is deserialized in full before only "
            "open-role keys are indexed, so this run does not satisfy a strict epistemic "
            "non-open-label deserialization seal."
        ),
        "policy_contract": {
            "score_reduction": "mean across coalition draws for each query-candidate pair",
            "threshold_fit_unit": (
                "fit group-OOF query-candidate score averaged across coalition draws"
            ),
            "selection_rule": (
                "aggregated query-candidate score > fit query-candidate frozen threshold"
            ),
            "empty_selection_fallback": "current_only",
            "reversibility": "each query reconstructs its set from immutable strict-past history",
            "persistent_deletion": False,
            "predictions_per_selection_query_per_strategy": 1,
            "recency_baseline": "same selected-candidate count per query as true bidirectional",
            "primary_seed_estimand": (
                "mean across the 5 utility-seed x 5 independently fitted base-seed grid"
            ),
            "base_seed_probability_ensemble_role": "secondary diagnostic only",
            "nll_probability_floor": float(NLL_PROBABILITY_FLOOR),
            "nll_identity": (
                "pooled_nll and mean_excess_nll_vs_current use the same clipped "
                "per-query true-class loss"
            ),
        },
        "access_contract": {
            "roles_used": ["base_and_utility_fit", "model_selection"],
            "non_open_role_rows_used": 0,
            "external_nontrain_rows_used": 0,
            "row_level_output": False,
            "label_container_format": "single pickled train_corpus object mapping",
            "label_container_rows_deserialized": int(
                data.label_container_rows_deserialized
            ),
            "non_open_label_rows_deserialized": int(
                data.non_open_label_rows_deserialized
            ),
            "non_open_label_keys_indexed_for_training_or_metrics": 0,
            "strict_epistemic_non_open_label_deserialization_seal_satisfied": False,
            "required_remediation_for_strict_seal": (
                "pre-materialize and hash an open-role-only label sidecar before sealing roles"
            ),
        },
        "alignment_audit": {
            "fit_checkpoint_exact_nonoverlapping_cover": True,
            "selection_checkpoint_canonical_complete_order": True,
            "recomputed_59d_features_bitwise_equal_cache": True,
            "feature_names_and_cluster_codes_equal_cache": True,
            "cached_target_label_recovery_matches_open_role_labels": True,
        },
        "experiment_counts": {
            "fit_rows": int(len(data.fit_indices)),
            "selection_queries": int(len(data.selection_indices)),
            "fit_sampled_tasks": int(len(task_material.fit_tasks)),
            "selection_sampled_tasks": int(len(task_material.selection_tasks)),
            "base_seeds": int(len(tuple(base_config["seeds"]))),
            "utility_seeds": int(len(DEFAULT_SEEDS)),
            "primary_joint_seed_grid": int(
                len(tuple(base_config["seeds"])) * len(DEFAULT_SEEDS)
            ),
            "utility_models": int(len(EXPECTED_MODEL_NAMES)),
            "task_feature_count": int(cache.fit.x.shape[1]),
        },
        "fixed_strategies": {
            "current_only": {
                "five_base_seed_primary": summarize_five_base_seed_strategy(
                    current_base_records
                ),
                "base_seed_ensemble_diagnostic": current_metrics,
            },
            "all_history": {
                "five_base_seed_primary": summarize_five_base_seed_strategy(
                    all_base_records
                ),
                "base_seed_ensemble_diagnostic": all_metrics,
            },
        },
        "utility_seed_strategy_summaries": strategy_summaries,
        "source_hashes": {
            **_source_hashes(
                data_dir,
                feature_path,
                base_config_path,
                utility_config_path,
                private_cache_path,
                checkpoint_paths,
            ),
            "feature_config_sha256": feature_config_sha256,
        },
    }
    _assert_aggregate_output(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report, output_path.resolve())
    return report


def _validate_runner_signature() -> None:
    names = tuple(inspect.signature(run_open_role_query_policy).parameters)
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
        raise AssertionError("query-policy runner parameters changed")
    for name in names:
        if set(name.lower().split("_")) & FORBIDDEN_ROLE_TOKENS:
            raise AssertionError("query-policy runner exposes a forbidden role parameter")


_validate_runner_signature()
