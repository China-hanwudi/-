from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from hva_affect.harmbench_erc_models import (  # noqa: E402
    AUDIO_DIMENSION,
    CAUSAL_GRU_ID,
    CAUSAL_GRU_PARAMETER_LIMIT,
    CURRENT_ONLY_NAMESPACE,
    CausalGRUCurrentOnlyCheckpoint,
    CausalGRUCurrentOnlyTrainer,
    CausalGRUHistoryCheckpoint,
    CausalGRUHistoryTrainer,
    DEEPSETS_PARAMETER_LIMIT,
    DEEPSETS_POOL_ID,
    DeepSetsCurrentOnlyCheckpoint,
    DeepSetsCurrentOnlyTrainer,
    DeepSetsHistoryCheckpoint,
    DeepSetsHistoryTrainer,
    ExpandedHistoryExamples,
    FROZEN_MODEL_IDS,
    HISTORY_NAMESPACE,
    HarmBenchModelError,
    LINEAR_HISTORY_SUMMARY_DIMENSION,
    LINEAR_POOL_ID,
    LinearCurrentOnlyCheckpoint,
    LinearCurrentOnlyTrainer,
    LinearHistoryCheckpoint,
    LinearHistoryTrainer,
    ProcessedRole,
    QUERY_DIMENSION,
    TEXT_DIMENSION,
    VIDEO_DIMENSION,
    fit_current_only_model,
    fit_history_model,
    fit_synthetic_current_only_model,
    fit_synthetic_history_model,
    linear_current_summary,
    linear_history_summary,
    make_current_only_trainer,
    make_history_trainer,
    normalize_expanded_history_examples,
    normalize_ordered_contexts,
    predict_current_only_model,
    predict_history_model,
    validate_probability_matrix,
)


ROWS = 8
CLASSES = 3
SEED = 29


def synthetic_arrays(
    *, rows: int = ROWS, seed: int = 101
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    text = generator.normal(size=(rows, TEXT_DIMENSION)).astype(np.float32)
    audio = generator.normal(size=(rows, AUDIO_DIMENSION)).astype(np.float32)
    video = generator.normal(size=(rows, VIDEO_DIMENSION)).astype(np.float32)
    # Make row/order differences comfortably observable without extreme values.
    text[:, 0] += np.arange(rows, dtype=np.float32) * 0.5
    audio[:, 0] -= np.arange(rows, dtype=np.float32) * 0.25
    video[:, 0] += np.arange(rows, dtype=np.float32) * 0.125
    return text, audio, video


def synthetic_role(*, rows: int = ROWS, seed: int = 101) -> ProcessedRole:
    text, audio, video = synthetic_arrays(rows=rows, seed=seed)
    return ProcessedRole(text=text, audio=audio, video=video)


def strict_past_contexts(rows: int = ROWS) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(range(max(0, query - 3), query)) for query in range(rows))


def labels(rows: int = ROWS) -> np.ndarray:
    return np.arange(rows, dtype=np.int64) % CLASSES


def _assert_probabilities(values: np.ndarray, *, rows: int = ROWS) -> None:
    assert values.shape == (rows, CLASSES)
    assert values.dtype == np.float64
    assert np.isfinite(values).all()
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)
    np.testing.assert_allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-8)
    assert values.flags.writeable is False


def test_processed_role_is_exact_float32_copied_readonly_and_row_aligned() -> None:
    text, audio, video = synthetic_arrays()
    original_text = text.copy()
    role = ProcessedRole(text=text, audio=audio, video=video)
    text[:] = 99.0
    np.testing.assert_array_equal(role.text, original_text)
    assert role.rows == ROWS
    assert role.text.flags.writeable is False
    assert role.audio.flags.writeable is False
    assert role.video.flags.writeable is False

    with pytest.raises(HarmBenchModelError, match="exact float32"):
        ProcessedRole(
            text=original_text.astype(np.float64), audio=audio, video=video
        )
    with pytest.raises(HarmBenchModelError, match="row-aligned"):
        ProcessedRole(text=original_text, audio=audio[:-1], video=video)
    bad_video = video.copy()
    bad_video[0, 0] = np.nan
    with pytest.raises(HarmBenchModelError, match="non-finite"):
        ProcessedRole(text=original_text, audio=audio, video=bad_video)


def test_frozen_family_roster_and_physical_current_only_surface() -> None:
    assert FROZEN_MODEL_IDS == (
        "hb_linear_pool_v1",
        "hb_deepsets_pool_v1",
        "hb_causal_gru_v1",
    )
    assert len(set(FROZEN_MODEL_IDS)) == 3

    current_fit_surfaces = (
        LinearCurrentOnlyTrainer.fit,
        DeepSetsCurrentOnlyTrainer.fit,
        CausalGRUCurrentOnlyTrainer.fit,
        fit_current_only_model,
    )
    current_prediction_surfaces = (
        LinearCurrentOnlyCheckpoint.predict_proba,
        DeepSetsCurrentOnlyCheckpoint.predict_proba,
        CausalGRUCurrentOnlyCheckpoint.predict_proba,
        predict_current_only_model,
    )
    for surface in (*current_fit_surfaces, *current_prediction_surfaces):
        names = set(inspect.signature(surface).parameters)
        assert "contexts" not in names
        assert "context" not in names
        assert "history" not in names

    current_types = (
        LinearCurrentOnlyCheckpoint,
        DeepSetsCurrentOnlyCheckpoint,
        CausalGRUCurrentOnlyCheckpoint,
    )
    history_types = (
        LinearHistoryCheckpoint,
        DeepSetsHistoryCheckpoint,
        CausalGRUHistoryCheckpoint,
    )
    for current, history in zip(current_types, history_types, strict=True):
        assert current is not history
        assert current.model_namespace == CURRENT_ONLY_NAMESPACE
        assert history.model_namespace == HISTORY_NAMESPACE
        assert current.model_identity != history.model_identity
        assert current.family_id == history.family_id
    assert len({checkpoint.family_id for checkpoint in current_types}) == 3


@pytest.mark.parametrize("model_id", FROZEN_MODEL_IDS)
def test_unified_fit_predict_is_seed_deterministic_and_probability_safe(
    model_id: str,
) -> None:
    features = synthetic_role()
    target = labels()
    contexts = strict_past_contexts()

    history_first = fit_synthetic_history_model(
        model_id,
        features,
        target,
        contexts,
        num_classes=CLASSES,
        seed=SEED,
        epochs=2,
    )
    history_second = fit_synthetic_history_model(
        model_id,
        features,
        target,
        contexts,
        num_classes=CLASSES,
        seed=SEED,
        epochs=2,
    )
    history_probabilities = predict_history_model(
        history_first, features, contexts
    )
    _assert_probabilities(history_probabilities)
    np.testing.assert_array_equal(
        history_probabilities,
        predict_history_model(history_second, features, contexts),
    )

    current_first = fit_synthetic_current_only_model(
        model_id,
        features,
        target,
        num_classes=CLASSES,
        seed=SEED,
        epochs=2,
    )
    current_second = fit_synthetic_current_only_model(
        model_id,
        features,
        target,
        num_classes=CLASSES,
        seed=SEED,
        epochs=2,
    )
    current_probabilities = predict_current_only_model(current_first, features)
    _assert_probabilities(current_probabilities)
    np.testing.assert_array_equal(
        current_probabilities,
        predict_current_only_model(current_second, features),
    )
    assert history_first.model_identity != current_first.model_identity
    assert type(history_first) is not type(current_first)


def test_linear_summary_is_query_mean_last_delta_and_log_count() -> None:
    text = np.zeros((4, TEXT_DIMENSION), dtype=np.float32)
    audio = np.zeros((4, AUDIO_DIMENSION), dtype=np.float32)
    video = np.zeros((4, VIDEO_DIMENSION), dtype=np.float32)
    text[:, 0] = [1.0, 3.0, 7.0, 11.0]
    audio[:, 0] = [2.0, 4.0, 8.0, 12.0]
    video[:, 0] = [5.0, 6.0, 9.0, 13.0]
    features = ProcessedRole(text=text, audio=audio, video=video)
    contexts = ((), (0,), (0, 1), (0, 2, 1))

    current = linear_current_summary(features)
    summary = linear_history_summary(features, contexts)
    assert current.shape == (4, QUERY_DIMENSION)
    assert summary.shape == (4, LINEAR_HISTORY_SUMMARY_DIMENSION)
    np.testing.assert_array_equal(summary[:, :QUERY_DIMENSION], current)

    query = current[3]
    context_matrix = current[[0, 2, 1]]
    offset = QUERY_DIMENSION
    np.testing.assert_allclose(
        summary[3, offset : 2 * offset], context_matrix.mean(axis=0)
    )
    np.testing.assert_array_equal(summary[3, 2 * offset : 3 * offset], current[1])
    np.testing.assert_array_equal(
        summary[3, 3 * offset : 4 * offset], query - current[1]
    )
    assert summary[3, -1] == pytest.approx(np.log1p(3))
    np.testing.assert_array_equal(summary[0, offset : 3 * offset], 0.0)
    np.testing.assert_array_equal(
        summary[0, 3 * offset : 4 * offset], current[0]
    )
    assert summary[0, -1] == 0.0

    checkpoint = LinearHistoryTrainer(
        num_classes=CLASSES, seed=SEED, epochs=1
    ).fit(features, np.asarray([0, 1, 2, 0]), contexts)
    assert checkpoint._estimator.loss == "log_loss"
    np.testing.assert_array_equal(checkpoint._estimator.classes_, np.arange(CLASSES))


@pytest.mark.parametrize("model_id", FROZEN_MODEL_IDS)
def test_history_fit_accepts_preexpanded_repeated_query_context_pairs(
    model_id: str,
) -> None:
    features = synthetic_role()
    query_indices = (3, 3, 4, 5, 5)
    contexts = ((0,), (0, 1), (), (1, 2), (0, 2, 4))
    target = np.asarray([0, 1, 2, 0, 1], dtype=np.int64)
    examples = normalize_expanded_history_examples(
        query_indices, contexts, feature_rows=features.rows
    )
    assert isinstance(examples, ExpandedHistoryExamples)
    assert examples.rows == len(target)
    assert examples.query_indices == query_indices
    assert examples.contexts == contexts

    checkpoint = fit_synthetic_history_model(
        model_id,
        features,
        target,
        contexts,
        num_classes=CLASSES,
        seed=SEED,
        epochs=1,
        query_indices=query_indices,
    )
    probabilities = predict_history_model(
        checkpoint,
        features,
        contexts,
        query_indices=query_indices,
    )
    _assert_probabilities(probabilities, rows=len(target))
    assert "strategy" not in inspect.signature(checkpoint.predict_proba).parameters


def test_deepsets_history_is_context_permutation_invariant() -> None:
    features = synthetic_role()
    original = strict_past_contexts()
    permuted = list(original)
    permuted[7] = (6, 4, 5)
    checkpoint = DeepSetsHistoryTrainer(
        num_classes=CLASSES, seed=SEED, epochs=2
    ).fit(features, labels(), original)
    first = checkpoint.predict_proba(features, original)
    second = checkpoint.predict_proba(features, tuple(permuted))
    np.testing.assert_array_equal(first[7], second[7])


def test_causal_gru_history_consumes_context_order_exactly() -> None:
    features = synthetic_role()
    original = strict_past_contexts()
    reversed_context = list(original)
    reversed_context[7] = tuple(reversed(original[7]))
    checkpoint = CausalGRUHistoryTrainer(
        num_classes=CLASSES, seed=SEED, epochs=2
    ).fit(features, labels(), original)
    first = checkpoint.predict_proba(features, original)
    second = checkpoint.predict_proba(features, tuple(reversed_context))
    assert not np.allclose(first[7], second[7], rtol=0.0, atol=1e-12)
    assert checkpoint._network.gru.num_layers == 1


@pytest.mark.parametrize("model_id", FROZEN_MODEL_IDS)
def test_every_history_family_supports_all_empty_contexts(model_id: str) -> None:
    features = synthetic_role()
    empty = tuple(() for _ in range(ROWS))
    checkpoint = fit_synthetic_history_model(
        model_id,
        features,
        labels(),
        empty,
        num_classes=CLASSES,
        seed=SEED,
        epochs=1,
    )
    _assert_probabilities(checkpoint.predict_proba(features, empty))


@pytest.mark.parametrize(
    ("trainer_type", "checkpoint_type", "limit"),
    (
        (DeepSetsHistoryTrainer, DeepSetsHistoryCheckpoint, DEEPSETS_PARAMETER_LIMIT),
        (
            DeepSetsCurrentOnlyTrainer,
            DeepSetsCurrentOnlyCheckpoint,
            DEEPSETS_PARAMETER_LIMIT,
        ),
        (CausalGRUHistoryTrainer, CausalGRUHistoryCheckpoint, CAUSAL_GRU_PARAMETER_LIMIT),
        (
            CausalGRUCurrentOnlyTrainer,
            CausalGRUCurrentOnlyCheckpoint,
            CAUSAL_GRU_PARAMETER_LIMIT,
        ),
    ),
)
def test_neural_parameter_gates_and_checkpoint_types(
    trainer_type: type,
    checkpoint_type: type,
    limit: int,
) -> None:
    features = synthetic_role()
    trainer = trainer_type(num_classes=CLASSES, seed=SEED, epochs=1)
    if trainer.model_namespace == HISTORY_NAMESPACE:
        checkpoint = trainer.fit(features, labels(), strict_past_contexts())
    else:
        checkpoint = trainer.fit(features, labels())
    assert isinstance(checkpoint, checkpoint_type)
    assert 0 < checkpoint.parameter_count < limit


def test_models_do_not_mutate_feature_label_or_context_inputs() -> None:
    text, audio, video = synthetic_arrays()
    role = ProcessedRole(text=text, audio=audio, video=video)
    target = labels()
    contexts = [list(values) for values in strict_past_contexts()]
    before = {
        "text": text.copy(),
        "audio": audio.copy(),
        "video": video.copy(),
        "role_text": role.text.copy(),
        "target": target.copy(),
        "contexts": deepcopy(contexts),
    }
    checkpoint = CausalGRUHistoryTrainer(
        num_classes=CLASSES, seed=SEED, epochs=1
    ).fit(role, target, contexts)
    checkpoint.predict_proba(role, contexts)
    np.testing.assert_array_equal(text, before["text"])
    np.testing.assert_array_equal(audio, before["audio"])
    np.testing.assert_array_equal(video, before["video"])
    np.testing.assert_array_equal(role.text, before["role_text"])
    np.testing.assert_array_equal(target, before["target"])
    assert contexts == before["contexts"]


def test_context_label_probability_and_factory_contracts_fail_closed() -> None:
    with pytest.raises(HarmBenchModelError, match="current query"):
        normalize_ordered_contexts(((0,), ()), rows=2)
    with pytest.raises(HarmBenchModelError, match="duplicate"):
        normalize_ordered_contexts(((), (0, 0)), rows=2)
    with pytest.raises(HarmBenchModelError, match="out-of-range"):
        normalize_ordered_contexts(((), (2,)), rows=2)
    with pytest.raises(HarmBenchModelError, match="one tuple per query"):
        normalize_ordered_contexts(((),), rows=2)

    with pytest.raises(HarmBenchModelError, match="sum to one"):
        validate_probability_matrix(
            np.full((2, 3), 0.2), expected_rows=2, num_classes=3
        )
    with pytest.raises(HarmBenchModelError, match="non-finite"):
        validate_probability_matrix(
            np.asarray([[np.nan, 0.0, 1.0]]),
            expected_rows=1,
            num_classes=3,
        )
    with pytest.raises(HarmBenchModelError, match="unknown history"):
        make_history_trainer("not_frozen", num_classes=3, seed=1)
    with pytest.raises(HarmBenchModelError, match="unknown current-only"):
        make_current_only_trainer("not_frozen", num_classes=3, seed=1)


def test_fixed_class_roster_survives_an_unobserved_fit_class() -> None:
    features = synthetic_role()
    two_observed_classes = np.arange(ROWS, dtype=np.int64) % 2
    checkpoint = LinearCurrentOnlyTrainer(
        num_classes=CLASSES, seed=SEED, epochs=2
    ).fit(features, two_observed_classes)
    probabilities = checkpoint.predict_proba(features)
    _assert_probabilities(probabilities)
    np.testing.assert_array_equal(checkpoint._estimator.classes_, np.arange(CLASSES))


def test_family_factories_return_distinct_fixed_trainer_classes() -> None:
    history = {
        model_id: type(
            make_history_trainer(
                model_id, num_classes=CLASSES, seed=SEED, epochs=1
            )
        )
        for model_id in FROZEN_MODEL_IDS
    }
    current = {
        model_id: type(
            make_current_only_trainer(
                model_id, num_classes=CLASSES, seed=SEED, epochs=1
            )
        )
        for model_id in FROZEN_MODEL_IDS
    }
    assert len(set(history.values())) == 3
    assert len(set(current.values())) == 3
    assert set(history.values()).isdisjoint(current.values())
    assert history == {
        LINEAR_POOL_ID: LinearHistoryTrainer,
        DEEPSETS_POOL_ID: DeepSetsHistoryTrainer,
        CAUSAL_GRU_ID: CausalGRUHistoryTrainer,
    }
    assert current == {
        LINEAR_POOL_ID: LinearCurrentOnlyTrainer,
        DEEPSETS_POOL_ID: DeepSetsCurrentOnlyTrainer,
        CAUSAL_GRU_ID: CausalGRUCurrentOnlyTrainer,
    }
