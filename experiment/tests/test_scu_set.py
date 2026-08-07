from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.scu_set import (  # noqa: E402
    assign_group_role,
    build_pair_features,
    conditional_utility_targets,
    sample_counterfactual_tasks,
    sequential_reversible_selection,
)


def test_group_role_assignment_is_deterministic_and_single() -> None:
    roles = {"fit": [0, 64], "selector": [65, 79], "calibration": [80, 89], "holdout": [90, 99]}
    first = assign_group_role("demo", "group-1", "v1", roles)
    second = assign_group_role("demo", "group-1", "v1", roles)
    assert first == second
    assert first[0] in roles


def test_counterfactual_tasks_never_use_candidate_inside_subset() -> None:
    histories = [(), (0,), (0, 1), (0, 1, 2), tuple(range(30))]
    first = sample_counterfactual_tasks(histories, subset_draws_per_query=4, maximum_candidates=8, seed=7)
    second = sample_counterfactual_tasks(histories, subset_draws_per_query=4, maximum_candidates=8, seed=7)
    assert first == second
    assert first
    for task in first:
        assert task.candidate_index not in task.subset_indices
        assert task.candidate_index in histories[task.query_index]
        assert set(task.subset_indices).issubset(set(histories[task.query_index]))


def test_conditional_utility_sign_matches_loss_change() -> None:
    labels = np.asarray([0, 1])
    without = np.asarray([[0.5, 0.5], [0.5, 0.5]])
    with_candidate = np.asarray([[0.8, 0.2], [0.8, 0.2]])
    target = conditional_utility_targets(labels, without, with_candidate)
    assert target[0] < 0
    assert target[1] > 0


def test_pair_features_are_label_free_finite_and_schema_stable() -> None:
    current_p = np.asarray([[0.6, 0.4], [0.3, 0.7]])
    subset_p = np.asarray([[0.5, 0.5], [0.4, 0.6]])
    with_p = np.asarray([[0.7, 0.3], [0.2, 0.8]])
    current_e = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    subset_e = np.asarray([[0.5, 0.5], [0.2, 0.8]])
    candidate_e = np.asarray([[1.0, 0.0], [0.1, 0.9]])
    matrix, names = build_pair_features(
        current_p,
        subset_p,
        with_p,
        current_e,
        subset_e,
        candidate_e,
        np.asarray([3, 5]),
        np.asarray([1, 2]),
        np.asarray([1, 3]),
    )
    assert matrix.shape == (2, len(names))
    assert np.isfinite(matrix).all()
    assert all("label" not in name and "gold" not in name for name in names)


def test_query_level_rejection_does_not_delete_future_memory() -> None:
    candidates = (1, 2, 3)

    def first_query(candidate: int, selected: tuple[int, ...]) -> float:
        return {1: 0.2, 2: -0.3, 3: 0.1}[candidate]

    def second_query(candidate: int, selected: tuple[int, ...]) -> float:
        return {1: -0.4, 2: 0.2, 3: 0.1}[candidate]

    assert sequential_reversible_selection(candidates, first_query, maximum_selected=3) == (2,)
    assert sequential_reversible_selection(candidates, second_query, maximum_selected=3) == (1,)
