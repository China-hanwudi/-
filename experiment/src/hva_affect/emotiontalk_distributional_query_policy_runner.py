"""One-query evaluation for the sign-by-severity distributional utility repair.

The runner keeps the same open-role, reversible-history, one-prediction-per-query
contract as :mod:`emotiontalk_query_policy_runner`.  The only changed component
is the candidate ranking score: all four registered distributional controls are
generated once per utility seed by
``distributional_utility_repair.generate_seed_predictions``.  Repeated
coalition draws are averaged for each ordered ``(query, candidate)`` pair before
the fit-OOF coverage threshold is frozen and transferred to model selection.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn

from . import distributional_utility_repair as distributional_module
from . import emotiontalk_query_policy_runner as query_module
from .bidirectional_utility_model import load_private_oof_cache
from .data_contract import ContractError, write_json_atomic
from .distributional_utility_repair import (
    DEFAULT_OOF_FOLDS,
    DEFAULT_SEEDS,
    MODEL_MODES,
    TRUE_MODEL_NAME,
    ComponentSpec,
    DistributionalCache,
    ModelPrediction,
    generate_seed_predictions,
    load_private_cache,
    load_repair_config,
    validate_lineage_report,
)
from .emotiontalk_bidirectional_oof import (
    _read_json,
    _validate_config,
    probability_task_features,
)
from .emotiontalk_query_policy_runner import (
    _materialize_open_role_base_data,
    _source_hashes,
    _train_linear_base_ensemble,
    aggregate_candidate_draw_scores,
    build_reversible_selected_contexts,
    coverage_matched_recency_contexts,
    fit_query_candidate_coverage_threshold,
    predict_query_context_probabilities_by_model,
    query_strategy_metrics,
    summarize_five_base_seed_strategy,
    summarize_joint_seed_strategy,
    summarize_utility_seed_strategy,
)
from .emotiontalk_sampled_context_runner import (
    OpenRoleDiagnosticError,
    assemble_fit_probability_checkpoints,
    load_selection_probability_checkpoint,
    read_probability_checkpoint,
    reconstruct_open_role_tasks,
    recover_query_labels_from_cached_utilities,
    verify_recomputed_59d_cache,
)
from .emotiontalk_text_p1 import LABEL_NAMES
from .meld_text_pilot import NLL_PROBABILITY_FLOOR, sha256_file


REPORT_SCHEMA_VERSION = "emotiontalk_open_role_distributional_query_policy_v3"
TARGET_PAIR_COVERAGE = 0.25
NLL_IDENTITY_ABSOLUTE_TOLERANCE = 1e-12
MODEL_NAMES = tuple(name for name, _ in MODEL_MODES)
STRATEGY_NAMES = {
    "distributional_forward_only": "distributional_forward_selected_history",
    "distributional_backward_only": "distributional_backward_selected_history",
    "distributional_pseudo_bidirectional": "distributional_pseudo_selected_history",
    "distributional_true_bidirectional": "distributional_true_selected_history",
}
RECENCY_STRATEGY = "coverage_matched_recency"
FORBIDDEN_ROLE_TOKENS = frozenset(
    {"calibration", "holdout", "sealed", "validation", "test"}
)


class DistributionalQueryPolicyContractError(ContractError):
    """Raised when the distributional query experiment breaks its frozen contract."""


@dataclass(frozen=True)
class DistributionalSelectionState:
    """One seed/model's frozen pair threshold and selected query contexts."""

    threshold: float
    fit_query_candidate_pairs: int
    realized_fit_query_candidate_coverage: float
    contexts: tuple[tuple[int, ...], ...]


def validate_cluster_pure_fold_positions(
    positions_by_name: Mapping[str, np.ndarray],
    *,
    task_count: int,
    cluster_codes: np.ndarray,
) -> dict[str, Any]:
    """Validate a frozen checkpoint partition without regenerating fold numbers.

    ``GroupKFold(shuffle=False)`` historically used an unstable argsort for
    equal-size groups, so regenerating exact fold membership can change across
    NumPy versions even when the role rows and cluster partition are identical.
    Frozen checkpoint positions are therefore authoritative.  This validator
    still fails closed unless they are an exact, non-overlapping cover and every
    cluster is wholly contained in one fold.
    """

    count = int(task_count)
    clusters = np.asarray(cluster_codes)
    if count < 1 or clusters.shape != (count,):
        raise DistributionalQueryPolicyContractError(
            "checkpoint fold validation inputs are misaligned"
        )
    if len(positions_by_name) < 2:
        raise DistributionalQueryPolicyContractError(
            "checkpoint partition requires at least two folds"
        )
    covered = np.zeros(count, dtype=bool)
    fold_by_position = np.full(count, -1, dtype=np.int16)
    for fold_index, (name, raw_positions) in enumerate(positions_by_name.items()):
        positions = np.asarray(raw_positions)
        if positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer):
            raise DistributionalQueryPolicyContractError(
                f"checkpoint positions are invalid: {name}"
            )
        positions = positions.astype(np.int64, copy=False)
        if not len(positions) or np.any((positions < 0) | (positions >= count)):
            raise DistributionalQueryPolicyContractError(
                f"checkpoint positions are empty or out of range: {name}"
            )
        if len(np.unique(positions)) != len(positions) or np.any(covered[positions]):
            raise DistributionalQueryPolicyContractError(
                f"checkpoint positions overlap or repeat: {name}"
            )
        covered[positions] = True
        fold_by_position[positions] = int(fold_index)
    if not covered.all():
        raise DistributionalQueryPolicyContractError(
            "checkpoint positions do not exactly cover fit tasks"
        )
    for cluster in np.unique(clusters):
        cluster_folds = np.unique(fold_by_position[clusters == cluster])
        if len(cluster_folds) != 1:
            raise DistributionalQueryPolicyContractError(
                "a fit cluster is split across frozen checkpoints"
            )
    position_counts: dict[str, int] = {}
    position_hashes: dict[str, str] = {}
    for name, raw_positions in positions_by_name.items():
        positions = np.asarray(raw_positions, dtype=np.int64)
        canonical = np.ascontiguousarray(positions.astype("<i8", copy=False))
        position_counts[str(name)] = int(len(canonical))
        position_hashes[str(name)] = hashlib.sha256(
            canonical.tobytes(order="C")
        ).hexdigest()
    return {
        "fit_checkpoint_exact_nonoverlapping_cover": True,
        "fit_checkpoint_positions_unique_within_fold": True,
        "fit_checkpoint_cluster_pure_partition": True,
        "fit_checkpoint_positions_are_frozen_authority": True,
        "fit_checkpoint_fold_count": int(len(positions_by_name)),
        "fit_checkpoint_task_count": count,
        "fit_checkpoint_cluster_count": int(len(np.unique(clusters))),
        "frozen_position_count_by_fold": position_counts,
        "frozen_position_sha256_by_fold": position_hashes,
        "position_hash_canonicalization": (
            "ordered little-endian int64 task offsets in frozen task order"
        ),
    }


def summarize_fold_regeneration_sensitivity(
    frozen_positions_by_name: Mapping[str, np.ndarray],
    regenerated_positions_by_name: Mapping[str, np.ndarray],
    *,
    task_count: int,
) -> dict[str, Any]:
    """Compare a runtime-regenerated fold assignment without using it.

    The regenerated assignment is diagnostic only.  Frozen checkpoint offsets
    remain authoritative because equal-size ``GroupKFold`` group ties can change
    order across NumPy versions.
    """

    if tuple(frozen_positions_by_name) != tuple(regenerated_positions_by_name):
        raise DistributionalQueryPolicyContractError(
            "regenerated checkpoint fold names or order changed"
        )
    count = int(task_count)
    frozen_assignment = np.full(count, -1, dtype=np.int16)
    regenerated_assignment = np.full(count, -1, dtype=np.int16)
    regenerated_hashes: dict[str, str] = {}
    for fold_index, name in enumerate(frozen_positions_by_name):
        frozen = np.asarray(frozen_positions_by_name[name], dtype=np.int64)
        regenerated = np.asarray(
            regenerated_positions_by_name[name], dtype=np.int64
        )
        for positions, assignment, source in (
            (frozen, frozen_assignment, "frozen"),
            (regenerated, regenerated_assignment, "regenerated"),
        ):
            if (
                positions.ndim != 1
                or np.any((positions < 0) | (positions >= count))
                or len(np.unique(positions)) != len(positions)
                or np.any(assignment[positions] >= 0)
            ):
                raise DistributionalQueryPolicyContractError(
                    f"{source} fold sensitivity positions are invalid: {name}"
                )
            assignment[positions] = int(fold_index)
        canonical = np.ascontiguousarray(regenerated.astype("<i8", copy=False))
        regenerated_hashes[str(name)] = hashlib.sha256(
            canonical.tobytes(order="C")
        ).hexdigest()
    if np.any(frozen_assignment < 0) or np.any(regenerated_assignment < 0):
        raise DistributionalQueryPolicyContractError(
            "fold sensitivity assignments do not exactly cover fit tasks"
        )
    mismatched = int(np.count_nonzero(frozen_assignment != regenerated_assignment))
    return {
        "runtime_regenerated_fold_assignment_role": (
            "sensitivity only; never used to assemble frozen probabilities"
        ),
        "runtime_regenerated_fold_assignment_matches_frozen": bool(mismatched == 0),
        "runtime_regenerated_fold_assignment_mismatched_tasks": mismatched,
        "runtime_regenerated_position_sha256_by_fold": regenerated_hashes,
        "equal_size_group_tie_order_is_version_sensitive": True,
        "environment_drift_is_not_interpreted_as_performance_difference": True,
    }


def audit_joint_seed_nll_identity(
    record_grid_by_strategy: Mapping[
        str, Sequence[Sequence[Mapping[str, float | int]]]
    ],
    current_records: Sequence[Mapping[str, float | int]],
    *,
    absolute_tolerance: float = NLL_IDENTITY_ABSOLUTE_TOLERANCE,
) -> dict[str, Any]:
    """Prove every utility/base cell uses the matching current-base NLL."""

    expected_strategies = (*STRATEGY_NAMES.values(), RECENCY_STRATEGY)
    if tuple(record_grid_by_strategy) != expected_strategies:
        raise DistributionalQueryPolicyContractError(
            "NLL identity audit strategy order changed"
        )
    current_rows = tuple(current_records)
    if len(current_rows) != 5:
        raise DistributionalQueryPolicyContractError(
            "NLL identity audit requires five current-base records"
        )
    tolerance = float(absolute_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise DistributionalQueryPolicyContractError(
            "NLL identity audit tolerance is invalid"
        )
    maximum_by_strategy: dict[str, float] = {}
    cells_checked = 0
    for strategy_name, raw_grid in record_grid_by_strategy.items():
        grid = tuple(tuple(row) for row in raw_grid)
        if len(grid) != 5 or any(len(row) != 5 for row in grid):
            raise DistributionalQueryPolicyContractError(
                "NLL identity audit requires a 5 by 5 metric grid"
            )
        strategy_errors: list[float] = []
        for row in grid:
            for base_index, record in enumerate(row):
                pooled_difference = float(record["pooled_nll"]) - float(
                    current_rows[base_index]["pooled_nll"]
                )
                reported_excess = float(record["mean_excess_nll_vs_current"])
                error = abs(pooled_difference - reported_excess)
                if not math.isfinite(error) or error > tolerance:
                    raise DistributionalQueryPolicyContractError(
                        "pooled NLL and mean excess NLL are misaligned for "
                        f"{strategy_name} base index {base_index}"
                    )
                strategy_errors.append(error)
                cells_checked += 1
        maximum_by_strategy[str(strategy_name)] = float(max(strategy_errors))
    maximum = float(max(maximum_by_strategy.values()))
    return {
        "nll_identity": (
            "strategy pooled_nll - matching current-base pooled_nll == "
            "mean_excess_nll_vs_current"
        ),
        "nll_identity_absolute_tolerance": tolerance,
        "nll_identity_joint_cells_checked": int(cells_checked),
        "nll_identity_maximum_absolute_error": maximum,
        "nll_identity_maximum_absolute_error_by_strategy": maximum_by_strategy,
        "nll_identity_satisfied": True,
    }


def build_distributional_seed_contexts(
    fit_tasks: Sequence[object],
    selection_tasks: Sequence[object],
    fit_predictions: Mapping[str, ModelPrediction],
    selection_predictions: Mapping[str, ModelPrediction],
    query_indices: Sequence[int],
    histories: Sequence[Sequence[int]],
    *,
    target_coverage: float = TARGET_PAIR_COVERAGE,
) -> dict[str, DistributionalSelectionState]:
    """Freeze fit-pair thresholds and build reversible model-selection contexts."""

    if tuple(fit_predictions) != MODEL_NAMES or tuple(selection_predictions) != MODEL_NAMES:
        raise DistributionalQueryPolicyContractError(
            "distributional prediction mapping changed registered model order"
        )
    result: dict[str, DistributionalSelectionState] = {}
    for model_name in MODEL_NAMES:
        fit_score = np.asarray(fit_predictions[model_name].decision, dtype=np.float64)
        selection_score = np.asarray(
            selection_predictions[model_name].decision, dtype=np.float64
        )
        threshold, pair_count, realized_coverage = (
            fit_query_candidate_coverage_threshold(
                fit_tasks,
                fit_score,
                target_coverage=float(target_coverage),
            )
        )
        aggregated = aggregate_candidate_draw_scores(selection_tasks, selection_score)
        contexts = build_reversible_selected_contexts(
            query_indices,
            histories,
            aggregated,
            threshold=threshold,
        )
        result[model_name] = DistributionalSelectionState(
            threshold=float(threshold),
            fit_query_candidate_pairs=int(pair_count),
            realized_fit_query_candidate_coverage=float(realized_coverage),
            contexts=contexts,
        )
    return result


def _assert_cache_views_equal(
    distributional_cache: DistributionalCache,
    query_cache: object,
) -> None:
    """Prove the two typed loaders expose the same private cache content."""

    for split_name in ("fit", "selection"):
        distributional_split = getattr(distributional_cache, split_name)
        query_split = getattr(query_cache, split_name)
        arrays = (
            ("x", distributional_split.x, query_split.x),
            ("forward", distributional_split.forward, query_split.forward),
            ("backward", distributional_split.backward, query_split.backward),
            (
                "cluster_codes",
                distributional_split.cluster_codes,
                query_split.cluster_codes,
            ),
        )
        for field, left, right in arrays:
            if not np.array_equal(np.asarray(left), np.asarray(right)):
                raise DistributionalQueryPolicyContractError(
                    f"distributional/query cache views differ at {split_name}.{field}"
                )
    if distributional_cache.source_hashes != query_cache.source_hashes:
        raise DistributionalQueryPolicyContractError("private-cache source hashes differ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_aggregate_query_report(report: Mapping[str, Any]) -> None:
    """Fail closed on row-aligned arrays, identities, or non-JSON values."""

    forbidden_exact = {
        "predictions",
        "probabilities",
        "scores",
        "utilities",
        "labels",
        "contexts",
        "decision_scores",
        "selected",
        "selection_mask",
        "selected_indices",
        "row_order",
        "cluster_codes",
        "cluster_ids",
        "row_ids",
        "query_ids",
        "query_codes",
        "query_labels",
        "task_positions",
    }

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, np.ndarray):
            raise DistributionalQueryPolicyContractError(
                f"aggregate report contains ndarray at {'.'.join(path)}"
            )
        if isinstance(value, Mapping):
            for key, child in value.items():
                name = str(key)
                if name.lower() in forbidden_exact:
                    raise DistributionalQueryPolicyContractError(
                        f"aggregate report contains forbidden field {'.'.join((*path, name))}"
                    )
                visit(child, (*path, name))
            return
        if isinstance(value, (list, tuple)):
            if len(value) > 20:
                raise DistributionalQueryPolicyContractError(
                    f"aggregate report contains overlong list at {'.'.join(path)}"
                )
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))
            return
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            raise DistributionalQueryPolicyContractError(
                f"aggregate report contains non-finite value at {'.'.join(path)}"
            )
        if not isinstance(
            value,
            (str, int, float, bool, type(None), np.integer, np.floating),
        ):
            raise DistributionalQueryPolicyContractError(
                f"aggregate report contains non-JSON value at {'.'.join(path)}"
            )

    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise DistributionalQueryPolicyContractError("aggregate report schema changed")
    visit(report, ())


def _environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    }


def _reproducibility_manifest(
    source_hashes: Mapping[str, Any],
    environment: Mapping[str, str],
) -> str:
    material = json.dumps(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "source_hashes": source_hashes,
            "environment": environment,
            "utility_seeds": list(DEFAULT_SEEDS),
            "target_pair_coverage": TARGET_PAIR_COVERAGE,
            "nll_probability_floor": NLL_PROBABILITY_FLOOR,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def run_open_role_distributional_query_policy(
    data_dir: str | Path,
    feature_path: str | Path,
    base_config_path: str | Path,
    utility_config_path: str | Path,
    private_cache_path: str | Path,
    checkpoint_dir: str | Path,
    lineage_report_path: str | Path,
    repair_config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Run the registered distributional one-query open-role terminal check."""

    (
        data_dir,
        feature_path,
        base_config_path,
        utility_config_path,
        private_cache_path,
        checkpoint_dir,
        lineage_report_path,
        repair_config_path,
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
            lineage_report_path,
            repair_config_path,
            output_path,
        )
    )
    if output_path.exists():
        raise FileExistsError(f"distributional query output already exists: {output_path}")
    if not data_dir.is_dir() or not checkpoint_dir.is_dir():
        raise FileNotFoundError("data-dir and checkpoint-dir must exist")
    for path in (
        feature_path,
        base_config_path,
        utility_config_path,
        private_cache_path,
        lineage_report_path,
        repair_config_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    base_config = _read_json(base_config_path)
    utility_config = _read_json(utility_config_path)
    _validate_config(utility_config, base_config)
    repair_config, component_spec = load_repair_config(repair_config_path)
    if not isinstance(component_spec, ComponentSpec):
        raise DistributionalQueryPolicyContractError("repair component spec is invalid")

    expected_hashes = {
        "base_config_sha256": sha256_file(base_config_path),
        "utility_config_sha256": sha256_file(utility_config_path),
        "feature_sha256": sha256_file(feature_path),
    }
    query_cache = load_private_oof_cache(private_cache_path)
    distributional_cache = load_private_cache(private_cache_path)
    _assert_cache_views_equal(distributional_cache, query_cache)
    lineage = validate_lineage_report(lineage_report_path, distributional_cache)
    if query_cache.source_hashes.get("base_config_sha256") != expected_hashes[
        "base_config_sha256"
    ]:
        raise OpenRoleDiagnosticError("private cache base-config hash mismatch")
    if query_cache.source_hashes.get("utility_config_sha256") != expected_hashes[
        "utility_config_sha256"
    ]:
        raise OpenRoleDiagnosticError("private cache utility-config hash mismatch")

    task_material, feature_config_sha256 = reconstruct_open_role_tasks(
        feature_path,
        base_config,
        utility_config,
    )
    fold_paths = tuple(
        checkpoint_dir / f"fold_{fold}.npz"
        for fold in range(1, int(base_config["crossfit_folds"]) + 1)
    )
    selection_checkpoint_path = checkpoint_dir / "selection.npz"
    checkpoint_paths = (*fold_paths, selection_checkpoint_path)
    expected_checkpoint_names = {path.name for path in checkpoint_paths}
    if {path.name for path in checkpoint_dir.glob("*.npz")} != expected_checkpoint_names:
        raise DistributionalQueryPolicyContractError("checkpoint directory schema changed")
    base_seed_count = len(tuple(base_config["seeds"]))
    checkpoint_positions = {
        path.name: read_probability_checkpoint(
            path,
            expected_hashes=expected_hashes,
            expected_seed_count=base_seed_count,
        ).positions
        for path in fold_paths
    }
    checkpoint_partition_audit = validate_cluster_pure_fold_positions(
        checkpoint_positions,
        task_count=len(task_material.fit_tasks),
        cluster_codes=task_material.fit_cluster_codes,
    )
    fold_regeneration_sensitivity = summarize_fold_regeneration_sensitivity(
        checkpoint_positions,
        task_material.expected_fold_positions,
        task_count=len(task_material.fit_tasks),
    )
    fit_probability = assemble_fit_probability_checkpoints(
        fold_paths,
        expected_task_count=len(task_material.fit_tasks),
        expected_seed_count=base_seed_count,
        expected_hashes=expected_hashes,
        expected_positions_by_name=checkpoint_positions,
    )
    selection_probability = load_selection_probability_checkpoint(
        selection_checkpoint_path,
        expected_task_count=len(task_material.selection_tasks),
        expected_seed_count=base_seed_count,
        expected_hashes=expected_hashes,
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
        query_cache,
        fit_x=fit_x,
        fit_feature_names=fit_names,
        selection_x=selection_x,
        selection_feature_names=selection_names,
        fit_cluster_codes=task_material.fit_cluster_codes,
        selection_cluster_codes=task_material.selection_cluster_codes,
    )

    data = _materialize_open_role_base_data(
        data_dir,
        feature_path,
        utility_config,
    )
    if data.histories != task_material.histories:
        raise DistributionalQueryPolicyContractError(
            "materialized strict-past histories changed task order"
        )
    recovered_selection_labels = recover_query_labels_from_cached_utilities(
        task_material.selection_tasks,
        selection_probability,
        query_cache.selection.forward,
        query_cache.selection.backward,
    )
    expected_task_labels = np.asarray(
        [data.labels[int(task.query_index)] for task in task_material.selection_tasks],
        dtype=np.int64,
    )
    if not np.array_equal(recovered_selection_labels, expected_task_labels):
        raise DistributionalQueryPolicyContractError(
            "selection labels disagree with cached utility targets"
        )

    base_models, current = _train_linear_base_ensemble(
        data,
        base_config,
        utility_config,
        task_material.fit_tasks,
    )
    selection_queries = tuple(int(value) for value in data.selection_indices)
    if len(selection_queries) != len(set(selection_queries)):
        raise DistributionalQueryPolicyContractError(
            "selection query rows are not unique"
        )
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
        [data.groups[query] for query in selection_queries],
        dtype=object,
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

    by_strategy_grid: dict[str, list[tuple[dict[str, float | int], ...]]] = {
        strategy_name: [] for strategy_name in STRATEGY_NAMES.values()
    }
    by_strategy_grid[RECENCY_STRATEGY] = []
    by_strategy_ensemble: dict[str, list[dict[str, float | int]]] = {
        strategy_name: [] for strategy_name in by_strategy_grid
    }
    threshold_records: dict[str, list[float]] = {name: [] for name in MODEL_NAMES}
    fit_pair_count_records: dict[str, list[int]] = {name: [] for name in MODEL_NAMES}
    fit_pair_coverage_records: dict[str, list[float]] = {
        name: [] for name in MODEL_NAMES
    }

    for utility_seed in DEFAULT_SEEDS:
        fit_predictions, selection_predictions, _ = generate_seed_predictions(
            distributional_cache,
            spec=component_spec,
            training_seed=int(utility_seed),
            oof_folds=int(repair_config["group_oof_folds"]),
        )
        states = build_distributional_seed_contexts(
            task_material.fit_tasks,
            task_material.selection_tasks,
            fit_predictions,
            selection_predictions,
            selection_queries,
            data.histories,
            target_coverage=TARGET_PAIR_COVERAGE,
        )
        for model_name in MODEL_NAMES:
            state = states[model_name]
            probability_by_base = predict_query_context_probabilities_by_model(
                base_models,
                current,
                data.quality,
                data.quality_names,
                selection_queries,
                state.contexts,
                data.histories,
                n_classes=len(LABEL_NAMES),
            )
            strategy_name = STRATEGY_NAMES[model_name]
            base_records = tuple(
                query_strategy_metrics(
                    selection_labels,
                    probability,
                    current_probability_by_base[base_index],
                    state.contexts,
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
                    state.contexts,
                    data.histories,
                    selection_queries,
                    selection_clusters,
                    ece_bins=ece_bins,
                )
            )
            threshold_records[model_name].append(state.threshold)
            fit_pair_count_records[model_name].append(
                state.fit_query_candidate_pairs
            )
            fit_pair_coverage_records[model_name].append(
                state.realized_fit_query_candidate_coverage
            )

        true_contexts = states[TRUE_MODEL_NAME].contexts
        recency_contexts = coverage_matched_recency_contexts(
            selection_queries,
            data.histories,
            true_contexts,
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
                probability,
                current_probability_by_base[base_index],
                recency_contexts,
                data.histories,
                selection_queries,
                selection_clusters,
                ece_bins=ece_bins,
            )
            for base_index, probability in enumerate(recency_probability_by_base)
        )
        by_strategy_grid[RECENCY_STRATEGY].append(recency_base_records)
        by_strategy_ensemble[RECENCY_STRATEGY].append(
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

    nll_alignment_audit = audit_joint_seed_nll_identity(
        by_strategy_grid,
        current_base_records,
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
    for model_name, strategy_name in STRATEGY_NAMES.items():
        thresholds = np.asarray(threshold_records[model_name], dtype=np.float64)
        pair_counts = set(fit_pair_count_records[model_name])
        if len(pair_counts) != 1:
            raise DistributionalQueryPolicyContractError(
                "fit query-candidate pair count changed across seeds"
            )
        realized = np.asarray(
            fit_pair_coverage_records[model_name],
            dtype=np.float64,
        )
        strategy_summaries[strategy_name]["fit_oof_frozen_threshold"] = {
            "unit": "query-candidate score averaged across coalition draws",
            "target_coverage": TARGET_PAIR_COVERAGE,
            "query_candidate_pairs": int(next(iter(pair_counts))),
            "threshold_mean": float(thresholds.mean()),
            "threshold_std": float(thresholds.std(ddof=1)),
            "realized_pair_coverage_mean": float(realized.mean()),
            "realized_pair_coverage_std": float(realized.std(ddof=1)),
        }

    module_path = Path(__file__)
    runner_path = (
        module_path.parents[2]
        / "scripts"
        / "run_emotiontalk_distributional_query_policy.py"
    )
    source_hashes = {
        **_source_hashes(
            data_dir,
            feature_path,
            base_config_path,
            utility_config_path,
            private_cache_path,
            checkpoint_paths,
        ),
        "feature_config_sha256": feature_config_sha256,
        "lineage_report_sha256": _sha256(lineage_report_path),
        "repair_config_sha256": _sha256(repair_config_path),
        "implementation_sha256": _sha256(module_path),
        "runner_sha256": _sha256(runner_path) if runner_path.exists() else None,
        "query_policy_dependency_sha256": _sha256(Path(query_module.__file__)),
        "distributional_dependency_sha256": _sha256(
            Path(distributional_module.__file__)
        ),
    }
    environment = _environment()
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": (
            "open_role_distributional_query_terminal_check_complete_with_"
            "label_container_limitation"
        ),
        "claim_boundary": (
            "This is an open-role one-prediction-per-query development terminal check "
            "of distributional repair 1/3 using the frozen linear multimodal base. It is "
            "not validation/test evidence, not confirmation of the new causal backbone, "
            "and not a final top-conference claim. The upstream pickled train-label "
            "container is fully deserialized before only open-role keys are indexed."
        ),
        "policy_contract": {
            "score_family": "sign_by_severity_expected_utility",
            "score_reduction": "mean across coalition draws for each query-candidate pair",
            "threshold_fit_unit": (
                "fit group-OOF query-candidate score averaged across coalition draws"
            ),
            "selection_rule": (
                "aggregated query-candidate score > fit query-candidate frozen threshold"
            ),
            "target_fit_pair_coverage": TARGET_PAIR_COVERAGE,
            "empty_selection_fallback": "current_only",
            "reversibility": (
                "each query reconstructs its set from immutable strict-past history"
            ),
            "persistent_deletion": False,
            "predictions_per_selection_query_per_strategy": 1,
            "recency_baseline": (
                "same selected-candidate count per query as distributional true bidirectional"
            ),
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
        "repair_contract": {
            "protocol": str(repair_config["protocol"]),
            "analysis_role": str(repair_config["analysis_role"]),
            "model_names": list(MODEL_NAMES),
            "utility_seeds": list(DEFAULT_SEEDS),
            "group_oof_folds": int(DEFAULT_OOF_FOLDS),
            "component_model": component_spec.public_dict(),
            "generation_api": "distributional_utility_repair.generate_seed_predictions",
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
            **lineage,
        },
        "alignment_audit": {
            **checkpoint_partition_audit,
            **fold_regeneration_sensitivity,
            **nll_alignment_audit,
            "selection_checkpoint_canonical_complete_order": True,
            "recomputed_59d_features_bitwise_equal_cache": True,
            "feature_names_and_cluster_codes_equal_cache": True,
            "distributional_and_query_cache_views_bitwise_equal": True,
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
            "utility_models": int(len(MODEL_NAMES)),
            "task_feature_count": int(query_cache.fit.x.shape[1]),
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
        "source_hashes": source_hashes,
        "environment": environment,
        "reproducibility_manifest_sha256": _reproducibility_manifest(
            source_hashes,
            environment,
        ),
    }
    assert_aggregate_query_report(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report, output_path.resolve())
    return report


def _validate_runner_signature() -> None:
    names = tuple(inspect.signature(run_open_role_distributional_query_policy).parameters)
    expected = (
        "data_dir",
        "feature_path",
        "base_config_path",
        "utility_config_path",
        "private_cache_path",
        "checkpoint_dir",
        "lineage_report_path",
        "repair_config_path",
        "output_path",
    )
    if names != expected:
        raise AssertionError("distributional query runner parameters changed")
    for name in names:
        if set(name.lower().split("_")) & FORBIDDEN_ROLE_TOKENS:
            raise AssertionError(
                "distributional query runner exposes a forbidden role parameter"
            )


_validate_runner_signature()
