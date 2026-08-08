from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.bidirectional_utility_model import (  # noqa: E402
    BidirectionalUtilityCache,
    UtilitySplit,
)
from hva_affect.emotiontalk_sampled_context_runner import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    OpenRoleTasks,
    OpenRoleDiagnosticError,
    _selection_task_order_sha256,
    assemble_fit_probability_checkpoints,
    load_selection_probability_checkpoint,
    recover_query_labels_from_cached_utilities,
    run_open_role_sampled_context_diagnostic,
    verify_recomputed_59d_cache,
)


HASHES = {
    "base_config_sha256": "a" * 64,
    "utility_config_sha256": "b" * 64,
    "feature_sha256": "c" * 64,
}


def _probability(rows: int, seeds: int = 2) -> np.ndarray:
    values = np.full((seeds, rows, 4, 7), 1 / 7, dtype=np.float64)
    for seed in range(seeds):
        for row in range(rows):
            for context in range(4):
                preferred = (seed + row + context) % 7
                values[seed, row, context] = 0.05
                values[seed, row, context, preferred] = 0.70
    return values


def _write_checkpoint(path: Path, positions: np.ndarray, probability: np.ndarray) -> None:
    np.savez_compressed(
        path,
        schema_version=np.asarray([CHECKPOINT_SCHEMA_VERSION]),
        task_positions=np.asarray(positions, dtype=np.int64),
        probability=np.asarray(probability, dtype=np.float64),
        base_config_sha256=np.asarray([HASHES["base_config_sha256"]]),
        utility_config_sha256=np.asarray([HASHES["utility_config_sha256"]]),
        feature_sha256=np.asarray([HASHES["feature_sha256"]]),
    )


def test_fold_checkpoint_assembly_is_position_based_nonoverlapping_and_complete(tmp_path: Path) -> None:
    full = _probability(6)
    first = tmp_path / "fold_1.npz"
    second = tmp_path / "fold_2.npz"
    _write_checkpoint(first, np.asarray([1, 4]), full[:, [1, 4]])
    _write_checkpoint(second, np.asarray([0, 2, 3, 5]), full[:, [0, 2, 3, 5]])
    assembled = assemble_fit_probability_checkpoints(
        [second, first],
        expected_task_count=6,
        expected_seed_count=2,
        expected_hashes=HASHES,
        expected_positions_by_name={
            "fold_1.npz": np.asarray([1, 4]),
            "fold_2.npz": np.asarray([0, 2, 3, 5]),
        },
    )
    assert assembled.dtype == np.float64
    np.testing.assert_array_equal(assembled, full)


def test_fold_checkpoint_assembly_rejects_overlap_and_incomplete_cover(tmp_path: Path) -> None:
    full = _probability(4)
    first = tmp_path / "fold_1.npz"
    second = tmp_path / "fold_2.npz"
    _write_checkpoint(first, np.asarray([0, 1]), full[:, [0, 1]])
    _write_checkpoint(second, np.asarray([1, 2]), full[:, [1, 2]])
    with pytest.raises(OpenRoleDiagnosticError, match="overlap"):
        assemble_fit_probability_checkpoints(
            [first, second],
            expected_task_count=4,
            expected_seed_count=2,
            expected_hashes=HASHES,
        )

    _write_checkpoint(second, np.asarray([2]), full[:, [2]])
    with pytest.raises(OpenRoleDiagnosticError, match="missing_count=1"):
        assemble_fit_probability_checkpoints(
            [first, second],
            expected_task_count=4,
            expected_seed_count=2,
            expected_hashes=HASHES,
        )


def test_selection_checkpoint_requires_canonical_complete_positions(tmp_path: Path) -> None:
    full = _probability(3)
    path = tmp_path / "selection.npz"
    _write_checkpoint(path, np.asarray([0, 2]), full[:, [0, 2]])
    with pytest.raises(OpenRoleDiagnosticError, match="canonical complete order"):
        load_selection_probability_checkpoint(
            path,
            expected_task_count=3,
            expected_seed_count=2,
            expected_hashes=HASHES,
        )
    _write_checkpoint(path, np.arange(3), full)
    restored = load_selection_probability_checkpoint(
        path,
        expected_task_count=3,
        expected_seed_count=2,
        expected_hashes=HASHES,
    )
    np.testing.assert_array_equal(restored, full)


def _cache() -> tuple[BidirectionalUtilityCache, np.ndarray, tuple[str, ...]]:
    fit_x = np.arange(4 * 59, dtype=np.float64).reshape(4, 59)
    selection_x = fit_x[:2] + 1000
    fit = UtilitySplit.validated(
        fit_x,
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        np.asarray([0.2, 0.1, 0.4, 0.3]),
        np.asarray([0, 0, 1, 1]),
        label="fit",
    )
    selection = UtilitySplit.validated(
        selection_x,
        np.asarray([0.1, 0.2]),
        np.asarray([0.2, 0.1]),
        np.asarray([0, 1]),
        label="selection",
    )
    names = tuple(f"feature_{index}" for index in range(59))
    return BidirectionalUtilityCache(fit, selection, names, {}), selection_x, names


def test_recomputed_task_features_must_match_59d_cache_bitwise_and_in_order() -> None:
    cache, selection_x, names = _cache()
    verify_recomputed_59d_cache(
        cache,
        fit_x=cache.fit.x.copy(),
        fit_feature_names=names,
        selection_x=selection_x.copy(),
        selection_feature_names=names,
        fit_cluster_codes=cache.fit.cluster_codes.copy(),
        selection_cluster_codes=cache.selection.cluster_codes.copy(),
    )
    reordered = selection_x[::-1]
    with pytest.raises(OpenRoleDiagnosticError, match="selection 59-D"):
        verify_recomputed_59d_cache(
            cache,
            fit_x=cache.fit.x,
            fit_feature_names=names,
            selection_x=reordered,
            selection_feature_names=names,
            fit_cluster_codes=cache.fit.cluster_codes,
            selection_cluster_codes=cache.selection.cluster_codes,
        )


@dataclass(frozen=True)
class _Task:
    query_index: int


@dataclass(frozen=True)
class _HashTask:
    query_index: int
    addition_context: tuple[int, ...]
    deletion_context: tuple[int, ...]
    candidate_index: int


def test_selection_task_hash_passes_complete_train_only_lineage() -> None:
    tasks = (
        _HashTask(4, (1, 2), (0, 1, 2, 3), 3),
        _HashTask(7, (3,), (1, 3, 5), 5),
    )
    task_material = OpenRoleTasks(
        histories=(),
        fit_tasks=(),
        selection_tasks=tasks,
        fit_cluster_codes=np.asarray([], dtype=np.int64),
        selection_cluster_codes=np.asarray([0, 1], dtype=np.int64),
        expected_fold_positions={},
        role_counts={},
        train_key_manifest_sha256="0" * 64,
        dataset_identifier="BAAI/Emotiontalk:train_corpus:feature_config_sha256="
        + "1" * 64,
        source_order_sha256="2" * 64,
        selection_fit_assignment_sha256="3" * 64,
    )
    baseline = _selection_task_order_sha256(
        tasks,
        task_material=task_material,
        split_manifest_sha256="4" * 64,
        producer_config_sha256="5" * 64,
    )
    assert len(baseline) == 64
    changed = _selection_task_order_sha256(
        tasks,
        task_material=task_material,
        split_manifest_sha256="6" * 64,
        producer_config_sha256="5" * 64,
    )
    assert changed != baseline


def test_labels_are_recovered_from_both_cached_utility_directions_per_query() -> None:
    tasks = [_Task(10), _Task(10), _Task(20)]
    probability = np.full((1, 3, 4, 7), 0.01, dtype=np.float64)
    probability[..., 0] = 0.94
    # Query 10 uses class 2; query 20 uses class 5.  Give every context a
    # distinct ratio so both directional targets identify the class.
    for row, label in enumerate((2, 2, 5)):
        probability[0, row, 0, label] = 0.20
        probability[0, row, 1, label] = 0.50
        probability[0, row, 2, label] = 0.60
        probability[0, row, 3, label] = 0.30
        probability[0, row] /= probability[0, row].sum(axis=1, keepdims=True)
    ensemble = probability.mean(axis=0)
    rows = np.arange(3)
    labels = np.asarray([2, 2, 5])
    forward = np.log(ensemble[rows, 1, labels]) - np.log(ensemble[rows, 0, labels])
    backward = np.log(ensemble[rows, 2, labels]) - np.log(ensemble[rows, 3, labels])
    recovered = recover_query_labels_from_cached_utilities(
        tasks, probability, forward, backward
    )
    np.testing.assert_array_equal(recovered, [2, 2, 5])
    with pytest.raises(OpenRoleDiagnosticError, match="exactly one"):
        recover_query_labels_from_cached_utilities(
            tasks, probability, forward + 0.5, backward
        )


def test_public_runner_and_cli_expose_no_forbidden_role_parameters() -> None:
    parameter_names = tuple(
        inspect.signature(run_open_role_sampled_context_diagnostic).parameters
    )
    assert parameter_names == (
        "data_dir",
        "feature_path",
        "base_config_path",
        "utility_config_path",
        "private_cache_path",
        "checkpoint_dir",
        "output_path",
    )
    forbidden = {"calibration", "holdout", "sealed", "validation", "test"}
    assert all(
        not (set(name.split("_")) & forbidden) for name in parameter_names
    )

    script_path = ROOT / "scripts" / "run_emotiontalk_sampled_context_diagnostic.py"
    spec = importlib.util.spec_from_file_location("sampled_context_cli", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parser = module.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert destinations == {
        "help",
        "data_dir",
        "feature",
        "base_config",
        "utility_config",
        "cache",
        "checkpoint_dir",
        "output",
    }
    assert all(not (set(name.split("_")) & forbidden) for name in destinations)


def test_runner_fails_before_reading_any_input_when_output_exists(tmp_path: Path) -> None:
    output = tmp_path / "already.json"
    output.write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        run_open_role_sampled_context_diagnostic(
            tmp_path / "data",
            tmp_path / "features.npz",
            tmp_path / "base.json",
            tmp_path / "utility.json",
            tmp_path / "cache.npz",
            tmp_path / "checkpoints",
            output,
        )
    assert output.read_text(encoding="utf-8") == "do not overwrite"
