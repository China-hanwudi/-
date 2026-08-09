from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.bidirectional_emotion_utility import (  # noqa: E402
    BidirectionalCoalitionTask,
    bidirectional_utility_targets,
    build_emotion_state_features,
    build_three_by_three_relation_features,
    deterministic_protocol_bucket,
    sample_bidirectional_coalition_tasks,
)


def test_trivial_bidirectional_context_is_rejected() -> None:
    with pytest.raises(ValueError, match="algebraically identical"):
        BidirectionalCoalitionTask(
            query_index=0,
            addition_context=(1, 2),
            deletion_context=(1, 2, 3),
            candidate_index=3,
        )


def test_bidirectional_sampling_is_deterministic_and_nontrivial() -> None:
    histories = [(), (0,), (0, 1), (0, 1, 2, 3), tuple(range(30))]
    first = sample_bidirectional_coalition_tasks(
        histories, draws_per_query=5, maximum_candidates=8, seed=20260808
    )
    second = sample_bidirectional_coalition_tasks(
        histories, draws_per_query=5, maximum_candidates=8, seed=20260808
    )
    assert first == second
    assert first
    for task in first:
        assert task.candidate_index not in task.addition_context
        assert task.candidate_index in task.deletion_context
        assert set(task.deletion_context) != set(task.addition_context) | {task.candidate_index}
        assert set(task.addition_context).issubset(set(histories[task.query_index]))
        assert set(task.deletion_context).issubset(set(histories[task.query_index]))


def test_size_matched_sampling_removes_cardinality_confound() -> None:
    histories = [tuple(range(8)), tuple(range(8, 16))]
    tasks = sample_bidirectional_coalition_tasks(
        histories,
        draws_per_query=12,
        maximum_candidates=8,
        seed=20260808,
        match_context_cardinality=True,
    )
    assert tasks
    for task in tasks:
        deletion_without_candidate = tuple(
            value for value in task.deletion_context if value != task.candidate_index
        )
        assert len(task.addition_context) == len(deletion_without_candidate)
        assert set(task.addition_context) != set(deletion_without_candidate)


def test_forward_and_backward_targets_are_benefit_positive_and_can_differ() -> None:
    labels = np.asarray([0, 1])
    probability_s = np.asarray([[0.40, 0.60], [0.30, 0.70]])
    probability_s_plus = np.asarray([[0.80, 0.20], [0.20, 0.80]])
    probability_t = np.asarray([[0.65, 0.35], [0.55, 0.45]])
    probability_t_minus = np.asarray([[0.50, 0.50], [0.40, 0.60]])
    targets = bidirectional_utility_targets(
        labels,
        probability_s,
        probability_s_plus,
        probability_t,
        probability_t_minus,
    )
    assert targets.forward_addition[0] > 0
    assert targets.backward_deletion[0] > 0
    assert targets.forward_addition[1] > 0
    assert targets.backward_deletion[1] < 0
    assert not np.allclose(targets.forward_addition, targets.backward_deletion)
    np.testing.assert_allclose(
        targets.asymmetry,
        targets.forward_addition - targets.backward_deletion,
    )


def test_three_by_three_relation_schema_is_label_free_and_stable() -> None:
    current = {
        "text": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        "audio": np.asarray([[0.8, 0.2], [0.2, 0.8]]),
        "video": np.asarray([[0.6, 0.4], [0.4, 0.6]]),
    }
    history = {
        "text": np.asarray([[0.9, 0.1], [0.1, 0.9]]),
        "audio": np.asarray([[0.7, 0.3], [0.3, 0.7]]),
        "video": np.asarray([[0.5, 0.5], [0.5, 0.5]]),
    }
    matrix, names = build_three_by_three_relation_features(current, history)
    assert matrix.shape == (2, 27)
    assert len(names) == 27
    assert np.isfinite(matrix).all()
    assert len({name.split("__")[0] + "__" + name.split("__")[1] for name in names}) == 9
    assert all("label" not in name and "gold" not in name for name in names)


def test_three_by_three_relation_requires_shared_projection_dimension() -> None:
    current = {
        "text": np.ones((2, 3)),
        "audio": np.ones((2, 3)),
        "video": np.ones((2, 3)),
    }
    history = {
        "text": np.ones((2, 4)),
        "audio": np.ones((2, 4)),
        "video": np.ones((2, 4)),
    }
    with pytest.raises(ValueError, match="aligned dimension"):
        build_three_by_three_relation_features(current, history)


def test_emotion_state_features_capture_vad_transition_without_labels() -> None:
    current = np.asarray([[0.8, 0.5, 0.2], [-0.4, 0.7, 0.1]])
    history = np.asarray([[0.2, 0.4, 0.3], [-0.1, 0.3, 0.0]])
    matrix, names = build_emotion_state_features(
        current,
        history,
        current_shift_probability=np.asarray([0.2, 0.8]),
        history_shift_probability=np.asarray([0.1, 0.4]),
    )
    assert matrix.shape == (2, 16)
    assert len(names) == 16
    assert np.isfinite(matrix).all()
    assert all("label" not in name and "gold" not in name for name in names)
    np.testing.assert_allclose(matrix[:, 6:9], current - history)


def test_protocol_bucket_is_deterministic() -> None:
    first = deterministic_protocol_bucket("MELD", "dia-1", "bidirectional_v1")
    second = deterministic_protocol_bucket("MELD", "dia-1", "bidirectional_v1")
    assert first == second
    assert 0 <= first <= 99
