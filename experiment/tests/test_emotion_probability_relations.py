from __future__ import annotations

import hashlib
import io
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.emotion_probability_relations import (  # noqa: E402
    BASE_CACHE_FEATURE_NAMES,
    BASE_CACHE_SCHEMA_VERSION,
    HISTORY_CONTEXTS,
    MODALITIES,
    BaseCacheLineage,
    EmotionProbabilityBlock,
    EmotionRelationContractError,
    TrainOnlyProvenance,
    align_with_59d_task_cache,
    base_cache_lineage_sha256,
    bidirectional_task_order_sha256,
    build_emotion_probability_relations,
    dataset_identity_sha256,
    emotion_class_order_sha256,
    emotion_context_schema_sha256,
    feature_names_content_sha256,
    numeric_matrix_content_sha256,
    ordered_source_sha256,
    verify_base_59d_cache,
)


CLASSES = ("neutral", "happy", "sad", "angry", "surprised", "disgusted", "fearful")
DATASET = "BAAI/EmotionTalk@synthetic-fixed-revision"
SOURCE_IDS = ("dialogue-a/0", "dialogue-a/1", "dialogue-b/0", "dialogue-b/1")
SOURCE_HASH = ordered_source_sha256(DATASET, SOURCE_IDS)
SPLIT_HASH = "a" * 64
OOF_FOLD_HASH = "b" * 64
SELECTION_FIT_HASH = "c" * 64
PRODUCER_HASH = "d" * 64
CONTEXT_HASH = emotion_context_schema_sha256()
CLASS_HASH = emotion_class_order_sha256(CLASSES)


@dataclass(frozen=True)
class _Task:
    query_index: int
    addition_context: tuple[int, ...]
    deletion_context: tuple[int, ...]
    candidate_index: int


TASKS = (
    _Task(9, (3, 1), (5, 2, 4), 4),
    _Task(11, (), (7, 6), 6),
)


def _task_hash(
    *,
    role: str = "base_and_utility_fit",
    source_hash: str = SOURCE_HASH,
    fold_hash: str = OOF_FOLD_HASH,
    class_hash: str = CLASS_HASH,
    tasks: tuple[_Task, ...] = TASKS,
) -> str:
    return bidirectional_task_order_sha256(
        tasks,
        dataset=DATASET,
        role=role,
        source_order_sha256=source_hash,
        split_manifest_sha256=SPLIT_HASH,
        fold_assignment_sha256=fold_hash,
        context_schema_sha256=CONTEXT_HASH,
        class_order_sha256=class_hash,
        producer_config_sha256=PRODUCER_HASH,
    )


def _provenance(
    *,
    mode: str = "train_fold_oof",
    source_hash: str = SOURCE_HASH,
    fold_hash: str | None = None,
    task_hash: str | None = None,
) -> TrainOnlyProvenance:
    role = "base_and_utility_fit" if mode == "train_fold_oof" else "model_selection"
    resolved_fold = (
        OOF_FOLD_HASH if mode == "train_fold_oof" else SELECTION_FIT_HASH
    ) if fold_hash is None else fold_hash
    resolved_task = _task_hash(
        role=role,
        source_hash=source_hash,
        fold_hash=resolved_fold,
    ) if task_hash is None else task_hash
    return TrainOnlyProvenance(
        mode=mode,
        dataset=DATASET,
        role=role,
        dataset_sha256=dataset_identity_sha256(DATASET),
        source_order_sha256=source_hash,
        split_manifest_sha256=SPLIT_HASH,
        fold_assignment_sha256=resolved_fold,
        task_order_sha256=resolved_task,
        context_schema_sha256=CONTEXT_HASH,
        class_order_sha256=CLASS_HASH,
        producer_config_sha256=PRODUCER_HASH,
    )


def _probability(shift: int) -> np.ndarray:
    first = np.asarray([0.52, 0.18, 0.10, 0.08, 0.05, 0.04, 0.03])
    second = np.asarray([0.12, 0.08, 0.50, 0.10, 0.06, 0.05, 0.09])
    return np.vstack([np.roll(first, shift), np.roll(second, shift)]).astype(np.float32)


def _block(
    offset: int,
    *,
    provenance: TrainOnlyProvenance | None = None,
    class_order: tuple[str, ...] = CLASSES,
    modality_class_orders: dict[str, tuple[str, ...]] | None = None,
    probabilities: dict[str, np.ndarray] | None = None,
) -> EmotionProbabilityBlock:
    return EmotionProbabilityBlock(
        probabilities=(
            {modality: _probability(offset + index) for index, modality in enumerate(MODALITIES)}
            if probabilities is None
            else probabilities
        ),
        provenance=_provenance() if provenance is None else provenance,
        class_order=class_order,
        modality_class_orders=(
            {modality: class_order for modality in MODALITIES}
            if modality_class_orders is None
            else modality_class_orders
        ),
    )


def _history(
    *, provenance: TrainOnlyProvenance | None = None
) -> dict[str, EmotionProbabilityBlock]:
    return {
        context: _block(index + 1, provenance=provenance)
        for index, context in enumerate(HISTORY_CONTEXTS)
    }


def _cache_payload(matrix: np.ndarray, names: tuple[str, ...] = BASE_CACHE_FEATURE_NAMES) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        schema_version=np.asarray([BASE_CACHE_SCHEMA_VERSION]),
        fit_x=np.asarray(matrix, dtype=np.float64),
        selection_x=np.asarray(matrix, dtype=np.float64),
        feature_names=np.asarray(names, dtype=str),
    )
    return stream.getvalue()


def _verified_cache(
    matrix: np.ndarray,
    *,
    provenance: TrainOnlyProvenance | None = None,
    names: tuple[str, ...] = BASE_CACHE_FEATURE_NAMES,
):
    selected_provenance = _provenance() if provenance is None else provenance
    payload = _cache_payload(matrix, names)
    lineage = BaseCacheLineage(
        schema_version=BASE_CACHE_SCHEMA_VERSION,
        row_count=len(matrix),
        cache_sha256=hashlib.sha256(payload).hexdigest(),
        matrix_content_sha256=numeric_matrix_content_sha256(
            np.asarray(matrix, dtype=np.float64)
        ),
        feature_names_content_sha256=feature_names_content_sha256(names),
        provenance=selected_provenance,
        class_order=CLASSES,
    )
    return verify_base_59d_cache(
        payload,
        lineage=lineage,
        expected_lineage_sha256=base_cache_lineage_sha256(lineage),
    )


def test_builds_float64_features_and_retains_complete_provenance() -> None:
    provenance = _provenance()
    bundle = build_emotion_probability_relations(
        _block(0, provenance=provenance),
        _history(provenance=provenance),
    )
    assert bundle.matrix.shape == (2, 213)
    assert bundle.matrix.dtype == np.float64
    assert np.isfinite(bundle.matrix).all()
    assert bundle.provenance == provenance
    assert bundle.class_order == CLASSES
    assert bundle.max_normalization_correction <= 1.0e-6
    assert len(bundle.feature_names) == len(set(bundle.feature_names)) == 213
    assert len(bundle.column_groups["simple_concat"]) == 105
    assert len(bundle.column_groups["same_modality_3cell"]) == 36
    assert len(bundle.column_groups["full_9cell"]) == 108
    assert len(bundle.column_groups["simple_concat_plus_same_modality"]) == 141
    assert len(bundle.column_groups["simple_concat_plus_full_9cell"]) == 213


def test_relation_values_are_interpretable_and_signed_simplex_mean_is_avoided() -> None:
    current = _block(0)
    history = _history()
    history["candidate"] = EmotionProbabilityBlock(
        probabilities={
            "text": current.probabilities["text"],
            "audio": history["candidate"].probabilities["audio"],
            "video": history["candidate"].probabilities["video"],
        },
        provenance=current.provenance,
        class_order=CLASSES,
        modality_class_orders={modality: CLASSES for modality in MODALITIES},
    )
    bundle = build_emotion_probability_relations(current, history)
    for metric, expected in (("cosine", 1.0), ("l2", 0.0), ("mean_absolute_delta", 0.0)):
        index = bundle.feature_names.index(
            f"emotion_relation__candidate__current_text__history_text__{metric}"
        )
        np.testing.assert_allclose(bundle.matrix[:, index], expected)


def test_task_hash_canonicalizes_sets_and_binds_all_lineage_dimensions() -> None:
    mappings = [
        {
            "query_index": task.query_index,
            "addition_context": task.addition_context,
            "deletion_context": task.deletion_context,
            "candidate_index": task.candidate_index,
        }
        for task in TASKS
    ]
    baseline = _task_hash()
    assert bidirectional_task_order_sha256(
        mappings,
        dataset=DATASET,
        role="base_and_utility_fit",
        source_order_sha256=SOURCE_HASH,
        split_manifest_sha256=SPLIT_HASH,
        fold_assignment_sha256=OOF_FOLD_HASH,
        context_schema_sha256=CONTEXT_HASH,
        class_order_sha256=CLASS_HASH,
        producer_config_sha256=PRODUCER_HASH,
    ) == baseline
    reordered_members = (
        replace(TASKS[0], addition_context=tuple(reversed(TASKS[0].addition_context))),
        TASKS[1],
    )
    assert _task_hash(tasks=reordered_members) == baseline
    assert _task_hash(tasks=tuple(reversed(TASKS))) != baseline
    assert _task_hash(source_hash="e" * 64) != baseline
    assert _task_hash(role="model_selection", fold_hash=SELECTION_FIT_HASH) != baseline
    assert _task_hash(class_hash="f" * 64) != baseline


def test_mode_role_pairing_and_mixed_fold_scope_fail_closed() -> None:
    common = dict(
        dataset=DATASET,
        dataset_sha256=dataset_identity_sha256(DATASET),
        source_order_sha256=SOURCE_HASH,
        split_manifest_sha256=SPLIT_HASH,
        fold_assignment_sha256=OOF_FOLD_HASH,
        task_order_sha256=_task_hash(),
        context_schema_sha256=CONTEXT_HASH,
        class_order_sha256=CLASS_HASH,
        producer_config_sha256=PRODUCER_HASH,
    )
    with pytest.raises(EmotionRelationContractError, match="requires role"):
        TrainOnlyProvenance(
            mode="train_fold_oof",
            role="model_selection",
            **common,
        )

    history = _history()
    wrong_fold = _provenance(fold_hash="e" * 64, task_hash=_task_hash())
    history["t"] = _block(3, provenance=wrong_fold)
    with pytest.raises(EmotionRelationContractError, match="inconsistent"):
        build_emotion_probability_relations(_block(0), history)


def test_modality_class_column_declaration_and_class_hash_must_match() -> None:
    swapped = (*CLASSES[1:], CLASSES[0])
    modality_orders = {modality: CLASSES for modality in MODALITIES}
    modality_orders["audio"] = swapped
    with pytest.raises(EmotionRelationContractError, match="columns do not match"):
        _block(0, modality_class_orders=modality_orders)
    with pytest.raises(EmotionRelationContractError, match="class_order_sha256"):
        _block(0, class_order=swapped)


def test_valid_float32_softmax_is_normalized_and_large_errors_are_rejected() -> None:
    rng = np.random.default_rng(20260808)
    logits = rng.normal(size=(10_000, 7)).astype(np.float32)
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True)).astype(np.float32)
    probability = (exponent / exponent.sum(axis=1, keepdims=True)).astype(np.float32)
    assert np.any(np.abs(probability.sum(axis=1) - 1.0) > 1.01e-7)
    block = _block(
        0,
        probabilities={modality: probability for modality in MODALITIES},
    )
    for values in block.probabilities.values():
        np.testing.assert_allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-14)
    assert 0.0 < block.max_normalization_correction < 1.0e-5

    invalid = {modality: _probability(index) for index, modality in enumerate(MODALITIES)}
    invalid["text"] = invalid["text"].copy()
    invalid["text"][0, 0] += 0.01
    with pytest.raises(EmotionRelationContractError, match="sums exceed tolerance"):
        _block(0, probabilities=invalid)
    invalid["text"][0, 0] = -0.01
    with pytest.raises(EmotionRelationContractError, match="outside"):
        _block(0, probabilities=invalid)


def test_verified_cache_join_checks_payload_matrix_names_and_complete_lineage() -> None:
    base = np.arange(2 * 59, dtype=np.float64).reshape(2, 59)
    verified = _verified_cache(base)
    bundle = build_emotion_probability_relations(_block(0), _history())
    joined = align_with_59d_task_cache(
        verified,
        emotion_features=bundle,
        emotion_group="same_modality_3cell",
    )
    assert joined.matrix.shape == (2, 95)
    np.testing.assert_array_equal(joined.matrix[:, :59], base)
    assert joined.feature_names[:59] == BASE_CACHE_FEATURE_NAMES
    assert joined.provenance == bundle.provenance

    selection_provenance = _provenance(mode="train_fit_only")
    selection_bundle = build_emotion_probability_relations(
        _block(0, provenance=selection_provenance),
        _history(provenance=selection_provenance),
    )
    with pytest.raises(EmotionRelationContractError, match="differ in"):
        align_with_59d_task_cache(
            verified,
            emotion_features=selection_bundle,
            emotion_group="same_modality_3cell",
        )


def test_arbitrary_59d_row_permutation_and_self_reported_task_hash_cannot_pass() -> None:
    base = np.arange(3 * 59, dtype=np.float64).reshape(3, 59)
    payload = _cache_payload(base)
    provenance = _provenance()
    lineage = BaseCacheLineage(
        schema_version=BASE_CACHE_SCHEMA_VERSION,
        row_count=3,
        cache_sha256=hashlib.sha256(payload).hexdigest(),
        matrix_content_sha256=numeric_matrix_content_sha256(base),
        feature_names_content_sha256=feature_names_content_sha256(BASE_CACHE_FEATURE_NAMES),
        provenance=provenance,
        class_order=CLASSES,
    )
    pinned_lineage = base_cache_lineage_sha256(lineage)

    arbitrary_names = tuple(f"arbitrary_feature_{index}" for index in range(59))
    with pytest.raises(EmotionRelationContractError, match="canonical 59-D schema"):
        BaseCacheLineage(
            schema_version=BASE_CACHE_SCHEMA_VERSION,
            row_count=3,
            cache_sha256=lineage.cache_sha256,
            matrix_content_sha256=lineage.matrix_content_sha256,
            feature_names_content_sha256=feature_names_content_sha256(arbitrary_names),
            provenance=provenance,
            class_order=CLASSES,
        )

    arbitrary_payload = _cache_payload(base[:, ::-1], tuple(reversed(BASE_CACHE_FEATURE_NAMES)))
    with pytest.raises(EmotionRelationContractError, match="payload SHA-256"):
        verify_base_59d_cache(
            arbitrary_payload,
            lineage=lineage,
            expected_lineage_sha256=pinned_lineage,
        )

    permuted = base[[2, 1, 0]]
    permuted_payload = _cache_payload(permuted)
    row_wrong_lineage = replace(
        lineage,
        cache_sha256=hashlib.sha256(permuted_payload).hexdigest(),
    )
    with pytest.raises(EmotionRelationContractError, match="matrix content hash"):
        verify_base_59d_cache(
            permuted_payload,
            lineage=row_wrong_lineage,
            expected_lineage_sha256=base_cache_lineage_sha256(row_wrong_lineage),
        )

    fake_task_provenance = replace(provenance, task_order_sha256="f" * 64)
    self_reported = replace(lineage, provenance=fake_task_provenance)
    with pytest.raises(EmotionRelationContractError, match="frozen digest"):
        verify_base_59d_cache(
            payload,
            lineage=self_reported,
            expected_lineage_sha256=pinned_lineage,
        )


def test_source_order_mismatch_and_forbidden_supervision_schema_fail() -> None:
    base = np.zeros((2, 59), dtype=np.float64)
    verified = _verified_cache(base)
    wrong_source = ordered_source_sha256(DATASET, tuple(reversed(SOURCE_IDS)))
    wrong_provenance = _provenance(source_hash=wrong_source)
    wrong_bundle = build_emotion_probability_relations(
        _block(0, provenance=wrong_provenance),
        _history(provenance=wrong_provenance),
    )
    with pytest.raises(EmotionRelationContractError, match="differ in"):
        align_with_59d_task_cache(
            verified,
            emotion_features=wrong_bundle,
            emotion_group="full_9cell",
        )

    unsafe_classes = (*CLASSES[:-1], "gold_label")
    unsafe_hash = emotion_class_order_sha256(unsafe_classes)
    unsafe_provenance = replace(_provenance(), class_order_sha256=unsafe_hash)
    with pytest.raises(EmotionRelationContractError, match="forbidden supervision"):
        _block(
            0,
            provenance=unsafe_provenance,
            class_order=unsafe_classes,
        )


def test_fit_only_mode_is_valid_with_matching_selection_cache_lineage() -> None:
    provenance = _provenance(mode="train_fit_only")
    bundle = build_emotion_probability_relations(
        _block(0, provenance=provenance),
        _history(provenance=provenance),
    )
    verified = _verified_cache(np.zeros((2, 59)), provenance=provenance)
    joined = align_with_59d_task_cache(
        verified,
        emotion_features=bundle,
        emotion_group="simple_concat_plus_full_9cell",
    )
    assert joined.matrix.shape == (2, 272)
    assert joined.provenance.mode == "train_fit_only"
    assert joined.provenance.role == "model_selection"
