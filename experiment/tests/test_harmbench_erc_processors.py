from __future__ import annotations

from dataclasses import fields, replace
import inspect
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from hva_affect.emotiontalk_role_sidecar import FIT_ROLE, SELECTION_ROLE  # noqa: E402
from hva_affect.harmbench_erc_crossfit import make_shared_group_crossfit_plan  # noqa: E402
from hva_affect.harmbench_erc_open_roles import (  # noqa: E402
    FitFeatureCapability,
    OutcomeFreeRoleFeatures,
    SelectionFeatureCapability,
    make_outcome_free_role_features,
    make_synthetic_fit_feature_capability,
    make_synthetic_selection_feature_capability,
)
from hva_affect.harmbench_erc_processors import (  # noqa: E402
    FROZEN_PROCESSOR_SPEC,
    HarmBenchProcessorError,
    ProcessedRoleEmbeddings,
    ProcessorReceipt,
    ProcessorSpec,
    SharedProcessor,
    fit_shared_processor,
    transform_role_features,
    validate_processed_role_embeddings,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _features(
    role: str,
    *,
    offset: int = 0,
    texts: tuple[str, ...] = ("aaaa", "bbbb", "zzzz", "yyyy", "cccc", "dddd"),
    audio: np.ndarray | None = None,
    video: np.ndarray | None = None,
    order: np.ndarray | None = None,
    row_alignment_sha256: str | None = None,
) -> OutcomeFreeRoleFeatures:
    rows = len(texts)
    if audio is None:
        audio = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2)
    if video is None:
        video = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3) / 10.0
    if order is None:
        order = np.arange(rows, dtype=np.int64)
    order = np.asarray(order, dtype=np.int64)
    protocol_rows = np.arange(offset, offset + rows, dtype=np.int64)
    return make_outcome_free_role_features(
        dataset_id="synthetic",
        role=role,
        keys=np.asarray([f"row-{offset + index}" for index in range(rows)])[order],
        texts=np.asarray(texts, dtype=str)[order].tolist(),
        audio=np.asarray(audio, dtype=np.float32)[order],
        video=np.asarray(video, dtype=np.float32)[order],
        groups=np.asarray([f"g{index}" for index in range(rows)])[order],
        speaker_identity=np.asarray(["s0"] * rows)[order],
        turn_ids=np.arange(rows, dtype=np.int64)[order],
        protocol_row_ids=protocol_rows[order],
        row_alignment_sha256=(
            row_alignment_sha256
            if row_alignment_sha256 is not None
            else (SHA_A if role == FIT_ROLE else SHA_B)
        ),
        feature_sha256=SHA_C if role == FIT_ROLE else SHA_D,
    )


def _fit_capability(source: OutcomeFreeRoleFeatures) -> FitFeatureCapability:
    return make_synthetic_fit_feature_capability(
        fit_features=source,
        feature_manifest_sha256=SHA_C,
        synthetic_feature_projection_sha256=SHA_D,
    )


def _selection_capability(
    source: OutcomeFreeRoleFeatures,
) -> SelectionFeatureCapability:
    return make_synthetic_selection_feature_capability(
        selection_features=source,
        manifest_sha256=SHA_B,
        synthetic_feature_projection_sha256=SHA_D,
    )


def _fit_processor(
    source: OutcomeFreeRoleFeatures,
    *,
    seed: int,
    fold: int,
):
    capability = _fit_capability(source)
    plan = make_shared_group_crossfit_plan(capability)
    processor = fit_shared_processor(
        capability,
        plan,
        seed=seed,
        fold=fold,
    )
    return capability, plan, processor


def _transform(
    processor: SharedProcessor,
    source: FitFeatureCapability | SelectionFeatureCapability,
) -> ProcessedRoleEmbeddings:
    receipt = processor.receipt
    return transform_role_features(
        processor,
        source,
        expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
        expected_fit_feature_capability_sha256=receipt.source_capability_sha256,
        expected_transform_source_capability_sha256=source.capability_sha256,
        expected_crossfit_plan_sha256=receipt.crossfit_plan_sha256,
        expected_seed=receipt.seed,
        expected_fold=receipt.fold,
    )


def _high_dimensional_features(role: str, *, offset: int = 0) -> OutcomeFreeRoleFeatures:
    rows = 7
    audio = np.arange(rows * 129, dtype=np.float32).reshape(rows, 129) / 100.0
    video = np.sin(
        np.arange(rows * 131, dtype=np.float32).reshape(rows, 131) / 13.0
    )
    texts = tuple(f"speaker emotion token {index} alpha beta" for index in range(rows))
    return _features(
        role,
        offset=offset,
        texts=texts,
        audio=audio,
        video=video,
    )


def _assert_l2_safe(matrix: np.ndarray) -> None:
    assert matrix.dtype == np.float32
    assert not matrix.flags.writeable
    assert np.isfinite(matrix).all()
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    assert np.all((np.isclose(norms, 0.0)) | np.isclose(norms, 1.0, atol=2e-6))


def test_fit_api_is_outcome_free_and_spec_is_exact_model_independent() -> None:
    signature_names = set(inspect.signature(fit_shared_processor).parameters)
    public_field_names = {
        field.name
        for cls in (ProcessorSpec, ProcessorReceipt, SharedProcessor, ProcessedRoleEmbeddings)
        for field in fields(cls)
    }
    assert all("label" not in name.lower() for name in signature_names | public_field_names)
    assert all("model" not in name.lower() for name in {field.name for field in fields(ProcessorSpec)})
    with pytest.raises(TypeError):
        fit_shared_processor(
            _fit_capability(_features(FIT_ROLE)),
            make_shared_group_crossfit_plan(_fit_capability(_features(FIT_ROLE))),
            seed=17,
            fold=0,
            labels=np.asarray([0, 1]),  # type: ignore[call-arg]
        )
    with pytest.raises(HarmBenchProcessorError, match="frozen exact spec"):
        ProcessorSpec(text_max_features=49_999)
    assert FROZEN_PROCESSOR_SPEC.text_analyzer == "char_wb"
    assert FROZEN_PROCESSOR_SPEC.text_ngram_range == (2, 5)
    assert FROZEN_PROCESSOR_SPEC.text_max_features == 50_000


def test_same_seed_fold_and_rows_are_bitwise_deterministic() -> None:
    source = _high_dimensional_features(FIT_ROLE)
    fit, plan, first = _fit_processor(source, seed=17, fold=2)
    second = fit_shared_processor(fit, plan, seed=17, fold=2)
    first_output = _transform(first, fit)
    second_output = _transform(second, fit)
    assert first.receipt == second.receipt
    assert first_output.output_receipt_sha256 == second_output.output_receipt_sha256
    for modality in ("text", "audio", "video", "fusion"):
        assert np.array_equal(getattr(first_output, modality), getattr(second_output, modality))


def test_seed_changes_projection_and_receipt() -> None:
    source = _high_dimensional_features(FIT_ROLE)
    fit, plan, first = _fit_processor(source, seed=17, fold=0)
    second = fit_shared_processor(fit, plan, seed=29, fold=0)
    first_output = _transform(first, fit)
    second_output = _transform(second, fit)
    assert first.receipt.processor_receipt_sha256 != second.receipt.processor_receipt_sha256
    assert not np.array_equal(first_output.audio, second_output.audio)
    assert not np.array_equal(first_output.video, second_output.video)


def test_fit_uses_only_indexed_vocabulary_and_numeric_means() -> None:
    audio = np.asarray(
        [[0.0, 0.0], [2.0, 4.0], [100.0, 200.0], [200.0, 400.0]],
        dtype=np.float32,
    )
    video = audio + np.asarray([10.0, 20.0], dtype=np.float32)
    full_audio = np.vstack((audio, [[4.0, 8.0], [6.0, 12.0]])).astype(np.float32)
    full_video = np.vstack((video, [[14.0, 28.0], [16.0, 32.0]])).astype(np.float32)
    probe = _features(
        FIT_ROLE,
        texts=("aaaa", "bbbb", "zzzz", "yyyy", "cccc", "dddd"),
        audio=full_audio,
        video=full_video,
    )
    probe_fit = _fit_capability(probe)
    probe_plan = make_shared_group_crossfit_plan(probe_fit)
    train = probe_plan.train_indices(17, 0, fit_capability=probe_fit)
    heldout = sorted(set(range(probe.rows)) - set(train.tolist()))
    texts = ["aaaa" for _ in range(probe.rows)]
    for index in heldout:
        texts[index] = "zzzz"
    source = _features(
        FIT_ROLE,
        texts=tuple(texts),
        audio=full_audio,
        video=full_video,
    )
    fit, plan, processor = _fit_processor(source, seed=17, fold=0)
    train = plan.train_indices(17, 0, fit_capability=fit)
    heldout = sorted(set(range(source.rows)) - set(train.tolist()))
    assert all("zz" not in token for token in processor.text_vocabulary)
    assert np.allclose(processor.audio_mean, source.audio[train].mean(axis=0))
    assert np.allclose(processor.video_mean, source.video[train].mean(axis=0))
    output = _transform(processor, fit)
    for index in heldout:
        assert np.count_nonzero(output.text[index]) == 0


def test_shapes_zero_padding_float32_readonly_and_l2_safety() -> None:
    source = _features(FIT_ROLE)
    fit, _, processor = _fit_processor(source, seed=43, fold=1)
    output = _transform(processor, fit)
    assert output.text.shape == (source.rows, 256)
    assert output.audio.shape == (source.rows, 128)
    assert output.video.shape == (source.rows, 128)
    assert output.fusion.shape == (source.rows, 512)
    assert np.count_nonzero(output.text[:, processor.text_effective_dimension :]) == 0
    assert np.count_nonzero(output.audio[:, source.audio.shape[1] :]) == 0
    assert np.count_nonzero(output.video[:, source.video.shape[1] :]) == 0
    for matrix in (output.text, output.audio, output.video, output.fusion):
        _assert_l2_safe(matrix)


def test_nonfinite_capability_and_invalid_plan_calls_fail_closed() -> None:
    source = _features(FIT_ROLE)
    fit = _fit_capability(source)
    plan = make_shared_group_crossfit_plan(fit)
    bad_audio = np.asarray(source.audio).copy()
    bad_audio[0, 0] = np.nan
    corrupted = replace(source, audio=bad_audio)
    corrupted_fit = replace(fit, fit=corrupted)
    with pytest.raises(HarmBenchProcessorError, match="plan/capability changed"):
        fit_shared_processor(corrupted_fit, plan, seed=17, fold=0)
    processor = fit_shared_processor(fit, plan, seed=17, fold=0)
    selection = _features(SELECTION_ROLE, offset=100)
    selection_capability = _selection_capability(selection)
    corrupted_selection = replace(
        selection_capability,
        selection=replace(selection, audio=bad_audio),
    )
    with pytest.raises(HarmBenchProcessorError, match="role capability changed"):
        _transform(processor, corrupted_selection)
    for seed, fold in ((True, 0), (999, 0), (17, True), (17, 5)):
        with pytest.raises(HarmBenchProcessorError):
            fit_shared_processor(fit, plan, seed=seed, fold=fold)


def test_sources_outputs_and_exposed_state_are_immutable_copies() -> None:
    raw_audio = np.arange(12, dtype=np.float32).reshape(6, 2)
    source = _features(FIT_ROLE, audio=raw_audio)
    raw_audio[:] = -999.0
    assert not np.any(source.audio == -999.0)
    _, _, processor = _fit_processor(source, seed=71, fold=0)
    state_before = processor.receipt
    mean_before = processor.audio_mean
    selection = _features(SELECTION_ROLE, offset=100)
    output = _transform(processor, _selection_capability(selection))
    assert processor.receipt == state_before
    assert np.array_equal(processor.audio_mean, mean_before)
    with pytest.raises(ValueError, match="read-only"):
        output.fusion[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        output.protocol_row_ids[0] = 999
    with pytest.raises(ValueError, match="read-only"):
        mean_before[0] = 999.0


def test_receipt_binds_seed_fold_training_rows_spec_and_source_content() -> None:
    source = _features(FIT_ROLE)
    fit, plan, base = _fit_processor(source, seed=17, fold=0)
    changed_fold = fit_shared_processor(fit, plan, seed=17, fold=1)
    changed_seed = fit_shared_processor(fit, plan, seed=29, fold=0)
    changed_source = _features(
        FIT_ROLE,
        texts=("aaaa changed", "bbbb", "zzzz", "yyyy", "cccc", "dddd"),
    )
    _, _, changed_content = _fit_processor(changed_source, seed=17, fold=0)
    receipts = {
        value.receipt.processor_receipt_sha256
        for value in (base, changed_fold, changed_seed, changed_content)
    }
    assert len(receipts) == 4
    assert base.receipt.processor_spec_sha256 == FROZEN_PROCESSOR_SPEC.canonical_sha256
    assert base.receipt.source_content_sha256 == source.content_sha256
    train = plan.train_indices(17, 0, fit_capability=fit)
    assert base.receipt.train_protocol_row_ids == tuple(
        int(value) for value in source.protocol_row_ids[train]
    )
    assert base.receipt.crossfit_plan_sha256 == plan.plan_sha256
    with pytest.raises(HarmBenchProcessorError, match="receipt SHA"):
        replace(base.receipt, fold=99)


def test_transform_preserves_row_order_and_hashes_reordering() -> None:
    fit_source = _features(FIT_ROLE)
    _, _, processor = _fit_processor(fit_source, seed=101, fold=0)
    selection = _features(SELECTION_ROLE, offset=100)
    reordered = _features(
        SELECTION_ROLE,
        offset=100,
        order=np.asarray([2, 0, 1, 5, 4, 3]),
        row_alignment_sha256="e" * 64,
    )
    first = _transform(processor, _selection_capability(selection))
    second = _transform(processor, _selection_capability(reordered))
    assert np.array_equal(first.protocol_row_ids, selection.protocol_row_ids)
    assert np.array_equal(second.protocol_row_ids, reordered.protocol_row_ids)
    assert first.source_content_sha256 == selection.content_sha256
    assert second.source_content_sha256 == reordered.content_sha256
    assert first.output_row_alignment_sha256 != second.output_row_alignment_sha256
    assert first.output_receipt_sha256 != second.output_receipt_sha256
    assert first.rows == second.rows == 6


def test_transform_requires_distinct_fit_and_live_transform_source_bindings() -> None:
    fit_source = _features(FIT_ROLE)
    _, _, processor = _fit_processor(fit_source, seed=17, fold=0)
    selection = _selection_capability(_features(SELECTION_ROLE, offset=100))
    receipt = processor.receipt
    signature = inspect.signature(transform_role_features)
    for name in (
        "expected_fit_feature_capability_sha256",
        "expected_transform_source_capability_sha256",
    ):
        assert signature.parameters[name].default is inspect.Parameter.empty
    with pytest.raises(HarmBenchProcessorError, match="transform source capability"):
        transform_role_features(
            processor,
            selection,
            expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
            expected_fit_feature_capability_sha256=receipt.source_capability_sha256,
            expected_transform_source_capability_sha256="f" * 64,
            expected_crossfit_plan_sha256=receipt.crossfit_plan_sha256,
            expected_seed=receipt.seed,
            expected_fold=receipt.fold,
        )
    with pytest.raises(HarmBenchProcessorError, match="external expected binding"):
        transform_role_features(
            processor,
            selection,
            expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
            expected_fit_feature_capability_sha256="f" * 64,
            expected_transform_source_capability_sha256=selection.capability_sha256,
            expected_crossfit_plan_sha256=receipt.crossfit_plan_sha256,
            expected_seed=receipt.seed,
            expected_fold=receipt.fold,
        )


def test_internal_tfidf_transform_state_is_live_hash_bound() -> None:
    source = _features(FIT_ROLE)
    fit, _, processor = _fit_processor(source, seed=17, fold=0)
    processor._text.vectorizer._tfidf.norm = "l1"
    with pytest.raises(HarmBenchProcessorError, match="fit state"):
        _transform(processor, fit)
