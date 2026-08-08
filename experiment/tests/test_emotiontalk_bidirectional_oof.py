from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.bidirectional_emotion_utility import BidirectionalCoalitionTask  # noqa: E402
from hva_affect.emotiontalk_bidirectional_oof import (  # noqa: E402
    aggregate_dense_contexts,
    aggregate_sparse_contexts,
    augmented_training_rows,
    build_context_blocks,
    probability_task_features,
    task_contexts,
)


def _task() -> BidirectionalCoalitionTask:
    return BidirectionalCoalitionTask(
        query_index=4,
        addition_context=(0,),
        deletion_context=(1, 2, 3),
        candidate_index=2,
    )


def test_rectangular_context_aggregation_uses_original_source_rows() -> None:
    dense = np.arange(18, dtype=float).reshape(6, 3)
    contexts = [(0, 2), (), (5,)]
    dense_result, dense_counts = aggregate_dense_contexts(dense, contexts)
    sparse_result, sparse_counts = aggregate_sparse_contexts(sparse.csr_matrix(dense), contexts)
    expected = np.vstack([(dense[0] + dense[2]) / 2, np.zeros(3), dense[5]])
    np.testing.assert_allclose(dense_result, expected)
    np.testing.assert_allclose(sparse_result.toarray(), expected)
    np.testing.assert_array_equal(dense_counts, [2, 0, 1])
    np.testing.assert_array_equal(sparse_counts, dense_counts)
    assert dense_result.dtype == np.float64


def test_context_blocks_expand_queries_without_losing_history_alignment() -> None:
    dense = np.arange(18, dtype=float).reshape(6, 3)
    current = {
        "text": sparse.csr_matrix(dense),
        "audio": dense + 100,
        "video": dense + 200,
    }
    quality = np.arange(12, dtype=float).reshape(6, 2)
    blocks = build_context_blocks(
        current,
        quality,
        ("q0", "q1"),
        query_indices=[4, 4, 5],
        contexts=[(0,), (0, 2), ()],
    )
    assert blocks.current["text"].shape == (3, 3)
    np.testing.assert_allclose(blocks.current["audio"][0], dense[4] + 100)
    np.testing.assert_allclose(blocks.history["audio"][1], (dense[0] + dense[2]) / 2 + 100)
    np.testing.assert_array_equal(blocks.counts, [1, 2, 0])


def test_task_context_order_matches_probability_contract() -> None:
    contexts = task_contexts(_task())
    assert contexts == ((0,), (0, 2), (1, 2, 3), (1, 3))


def test_augmented_training_rows_are_query_balanced_and_fold_closed() -> None:
    histories = [(), (0,), (0, 1), (0, 1, 2), (0, 1, 2, 3)]
    rows, contexts, weights = augmented_training_rows(
        histories,
        [_task()],
        allowed_indices=np.arange(5),
        maximum_contexts_per_query=5,
        seed=7,
    )
    for query in np.unique(rows):
        assert np.isclose(weights[rows == query].sum(), weights.sum() / len(np.unique(rows)))
    assert all(set(context).issubset(set(range(5))) for context in contexts)
    with pytest.raises(Exception, match="crosses the fold"):
        augmented_training_rows(
            histories,
            [_task()],
            allowed_indices=[3, 4],
            maximum_contexts_per_query=5,
            seed=7,
        )


def test_probability_task_features_are_float64_and_label_free() -> None:
    probability = np.full((1, 4, 7), 1 / 7, dtype=np.float64)
    histories = [(), (), (), (), (0, 1, 2, 3)]
    matrix, names = probability_task_features(probability, [_task()], histories)
    assert matrix.shape[0] == 1
    assert matrix.dtype == np.float64
    assert np.isfinite(matrix).all()
    assert len(names) == matrix.shape[1]
    assert all("gold" not in name and "label" not in name for name in names)
