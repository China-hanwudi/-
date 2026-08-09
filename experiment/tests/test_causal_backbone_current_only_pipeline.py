from __future__ import annotations

import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


torch = pytest.importorskip("torch", reason="current-only production bridge requires PyTorch")

import hva_affect.causal_backbone_current_only_pipeline as pipeline
from hva_affect.causal_backbone_current_only_pipeline import (
    CURRENT_ONLY_COMPLETION_RECEIPT_SCHEMA,
    align_selection_protocol_to_producer,
    complete_current_only_selection_probabilities,
    load_attested_history_fit_alignment_view,
    load_attested_history_producer_alignment_view,
    load_current_only_producer_alignment_view,
    make_real_current_only_fold_callback,
    verify_current_only_fit_for_completion,
)
from hva_affect.causal_backbone_evidence import (
    current_only_artifact_from_mapping,
    independent_current_only_source_identity,
)
from hva_affect.causal_backbone_evidence_runner import (
    EMOTIONTALK_LABEL_NAMES,
    EXPECTED_SEEDS,
    FIT_ROLE,
    SELECTION_ROLE,
    StageAContractError,
    load_fit_only_producer_view,
    materialize_selection_features_after_receipt,
    run_fit_preflight,
)
from hva_affect.causal_backbone_evidence_stage_b import (
    CurrentOnlyFoldOutput,
    HistoryFreeFoldRequest,
    _attest_production_current_only_fold_callback,
    align_fit_protocol_to_producer,
    load_private_npz_mapping,
    produce_independent_current_only_fit_oof,
    write_fit_only_lineage,
    write_fit_protocol_map,
)
from hva_affect.causal_multimodal_backbone import (
    CausalBackboneConfig,
    CausalMultimodalBackbone,
)
from hva_affect.causal_backbone_history_staged_pipeline import (
    HISTORY_COMPLETE_OUTCOME_SCHEMA,
    VerifiedHistoryCompletionAttestation,
)
from hva_affect.emotiontalk_causal_backbone_runner import (
    BackboneRunConfig,
    FoldTextProcessor,
    PlannedCheckpointInterruption,
    _canonical_sha256,
    _capture_rng_state,
    _indices_sha256,
    _torch_load_local,
)
from test_causal_backbone_evidence_runner import (
    ENVIRONMENT,
    _file_sha,
    _lineage_files,
    _make_emotiontalk_sidecars,
    _producer_mapping,
    _sha,
    _write_npz,
)


def _make_four_group_sidecars(root: Path) -> Path:
    manifest_path = _make_emotiontalk_sidecars(root, poison_selection=False)
    feature_path = root / f"features_{FIT_ROLE}.npz"
    with np.load(feature_path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    values["opaque_group_hashes"] = np.asarray(
        [_sha(f"fit-group-{index}") for index in range(4)]
    )
    _write_npz(feature_path, values)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["roles"][FIT_ROLE]["groups"] = 4
    manifest["roles"][FIT_ROLE]["history_eligible_rows"] = 0
    manifest["roles"][FIT_ROLE]["feature_sha256"] = _file_sha(feature_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _fixture(root: Path):
    manifest_path = _make_four_group_sidecars(root)
    configs, code = _lineage_files(root)
    receipt_path = root / "fit-receipt.json"
    preflight = run_fit_preflight(
        dataset="EmotionTalk",
        sidecar_dir=root,
        manifest_path=manifest_path,
        receipt_path=receipt_path,
        config_paths=configs,
        code_paths=code,
        environment=ENVIRONMENT,
    )
    fit_map = write_fit_protocol_map(
        preflight.fit,
        receipt_path=receipt_path,
        expected_receipt_sha256=preflight.receipt_sha256,
        output_path=root / "fit-map.npz",
    )
    lineage = write_fit_only_lineage(
        preflight.fit,
        fit_map=fit_map,
        receipt_path=receipt_path,
        expected_receipt_sha256=preflight.receipt_sha256,
        output_path=root / "fit-lineage.npz",
    )
    values = _producer_mapping(selection_poison=True)
    values["protocol_row_ids"] = np.asarray([2, 0, 3, 1, 6, 4, 7, 5], dtype=np.int64)
    values["fit_cluster_codes"] = np.asarray([0, 1, 2, 3], dtype=np.int32)
    values["selection_query_indices"] = np.asarray([0, 2, 4, 6], dtype=np.int64)
    values["selection_cluster_codes"] = np.asarray([0, 0, 1, 1], dtype=np.int32)
    receipt = preflight.receipt
    fit_sidecar = receipt["sidecars"]["fit"]
    selection_sidecar = receipt["sidecars"][SELECTION_ROLE]
    values.update(
        {
            "source_sidecar_manifest_sha256": np.asarray(
                receipt["manifest"]["sha256"]
            ),
            f"source_{FIT_ROLE}_features_sha256": np.asarray(
                fit_sidecar["feature_sha256"]
            ),
            f"source_{FIT_ROLE}_labels_sha256": np.asarray(
                fit_sidecar["label_sha256"]
            ),
            "source_model_selection_features_sha256": np.asarray(
                selection_sidecar["feature_sha256"]
            ),
            "source_model_selection_labels_sha256": np.asarray(
                selection_sidecar["label_sha256"]
            ),
        }
    )
    producer_path = root / "producer.npz"
    _write_npz(producer_path, values)
    fit_producer = load_fit_only_producer_view(producer_path)
    alignment = load_current_only_producer_alignment_view(
        producer_path, fit_producer=fit_producer
    )
    return {
        "manifest": manifest_path,
        "configs": configs,
        "code": code,
        "preflight": preflight,
        "receipt": receipt_path,
        "fit_map": fit_map,
        "lineage": lineage,
        "producer_path": producer_path,
        "fit_producer": fit_producer,
        "alignment": alignment,
    }


def _produce_fake_fit(
    root: Path,
    values: dict,
    *,
    partial_checkpoint: tuple[int, int] | None = None,
    production_attested: bool = True,
    bad_checkpoint_identity: tuple[int, int] | None = None,
):
    preflight = values["preflight"]
    model, run_config = _tiny_configs()
    baseline_run = replace(run_config, subset_dropout_probability=0.0)
    private_root = root / "current-only-private"
    hashes = {
        "model": _sha("model"),
        "run": _sha("run"),
        "code": _sha("code"),
        "runtime": _sha("runtime"),
    }
    claim = pipeline.current_only_production_claim_sha256(
        fit=preflight.fit,
        fit_map=values["fit_map"],
        lineage=values["lineage"],
        fit_preflight_receipt_sha256=preflight.receipt_sha256,
        model_config=model,
        run_config=run_config,
        model_config_sha256=hashes["model"],
        run_config_sha256=hashes["run"],
        source_code_sha256=hashes["code"],
        runtime_environment_sha256=hashes["runtime"],
    )
    pipeline.claim_or_resume_current_only_private_root(
        private_root,
        production_claim_sha256=claim,
        allow_resume=False,
    )
    paths = pipeline.current_only_private_paths(private_root)
    checkpoint_root = paths["checkpoint"]
    current_identity = independent_current_only_source_identity(
        producer_source_identity_sha256=values["lineage"].source_identity_sha256,
        model_config=model,
        run_config=run_config,
        rows=preflight.fit.rows,
        data_contract_sha256=values["fit_map"].fit_arrays_contract_sha256,
    )

    def callback(request):
        run = (
            request.checkpoint_root
            / f"seed_{request.seed:05d}"
            / f"fold_{request.fold:02d}"
        )
        run.mkdir(parents=True, exist_ok=True)
        status = (
            "partial"
            if partial_checkpoint == (request.seed, request.fold)
            else "complete"
        )
        corpus = pipeline._corpus_from_fold_request(request, model_config=model)
        split = pipeline._split_from_outer_partition(
            corpus,
            outer_train=request.train_indices,
            heldout=request.heldout_indices,
            validation_fraction=baseline_run.inner_validation_fraction,
            seed=request.seed,
            fold=request.fold,
        )
        vectorizer = TfidfVectorizer(
            analyzer=baseline_run.text_analyzer,
            ngram_range=(baseline_run.text_ngram_min, baseline_run.text_ngram_max),
            min_df=baseline_run.text_min_df,
            max_df=baseline_run.text_max_df,
            max_features=baseline_run.text_max_features,
            sublinear_tf=baseline_run.text_sublinear_tf,
            dtype=np.float32,
        )
        text_matrix = vectorizer.fit_transform(
            [corpus.texts[int(index)] for index in split.inner_train_indices]
        )
        effective = min(model.text_dim, max(1, int(text_matrix.shape[1]) - 1))
        svd = None
        if text_matrix.shape[1] > 1:
            svd = TruncatedSVD(
                n_components=effective,
                n_iter=baseline_run.text_svd_n_iter,
                algorithm="randomized",
                random_state=int(request.seed),
            )
            svd.fit(text_matrix)
        processor = FoldTextProcessor(
            vectorizer=vectorizer,
            svd=svd,
            output_dim=model.text_dim,
            effective_dim=effective,
            fit_indices_sha256=_indices_sha256(split.inner_train_indices),
        )
        processor_identity = _canonical_sha256(
            {
                "source_identity": current_identity,
                "fold": int(request.fold),
                "seed": int(request.seed),
                "inner_train_indices_sha256": _indices_sha256(
                    split.inner_train_indices
                ),
                "text_dim": model.text_dim,
                "text_settings": {
                    name: value
                    for name, value in asdict(baseline_run).items()
                    if name.startswith("text_")
                },
            }
        )
        processor_path = run / "text_processor.joblib"
        joblib.dump(
            {
                "schema_version": "fold_local_text_svd_v1",
                "identity_sha256": processor_identity,
                "processor": processor,
            },
            processor_path,
        )
        processor_sha = _file_sha(processor_path)
        checkpoint_identity = _canonical_sha256(
            {
                "source_identity": current_identity,
                "seed": int(request.seed),
                "fold": int(request.fold),
                "inner_train": _indices_sha256(split.inner_train_indices),
                "inner_validation": _indices_sha256(
                    split.inner_validation_indices
                ),
                "outer_heldout": _indices_sha256(split.outer_heldout_indices),
                "processor_sha256": processor_sha,
                "model_config": asdict(model),
                "run_config": asdict(baseline_run),
            }
        )
        backbone = CausalMultimodalBackbone(model)
        optimizer = torch.optim.AdamW(
            backbone.parameters(),
            lr=baseline_run.learning_rate,
            weight_decay=baseline_run.weight_decay,
        )
        state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in backbone.state_dict().items()
        }
        torch.save(
            {
                "schema_version": "causal_backbone_atomic_checkpoint_v2",
                "status": status,
                "identity_sha256": (
                    _sha("wrong-checkpoint-identity")
                    if bad_checkpoint_identity == (request.seed, request.fold)
                    else checkpoint_identity
                ),
                "epoch": 0,
                "model_state": state,
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": {},
                "best_model_state": {
                    name: tensor.clone() for name, tensor in state.items()
                },
                "best_epoch": 0,
                "best_validation_nll": 1.0,
                "bad_epochs": 0,
                "early_stopped": False,
                "peak_cuda_mib": 0.0,
                "rng_state": _capture_rng_state(),
            },
            run / "checkpoint.pt",
        )
        probability = np.zeros(
            (len(request.heldout_indices), len(preflight.fit.label_order)),
            dtype=np.float32,
        )
        for local, fit_row in enumerate(request.heldout_indices):
            probability[local, int(fit_row) % len(preflight.fit.label_order)] = 1.0
        return CurrentOnlyFoldOutput(probability, current_identity)

    folds = np.tile(
        np.asarray([0, 0, 1, 1], dtype=np.int32), (len(EXPECTED_SEEDS), 1)
    )
    callback_for_run = (
        _attest_production_current_only_fold_callback(
            callback,
            model_config_semantic_sha256=pipeline._model_config_semantic_sha256(
                model
            ),
            run_config_semantic_sha256=pipeline._run_config_semantic_sha256(
                run_config
            ),
        )
        if production_attested
        else callback
    )
    produced = produce_independent_current_only_fit_oof(
        fit=preflight.fit,
        fit_map=values["fit_map"],
        lineage=values["lineage"],
        fit_preflight_receipt_path=values["receipt"],
        expected_fit_preflight_receipt_sha256=preflight.receipt_sha256,
        fold_by_seed_row=folds,
        outer_folds=2,
        checkpoint_root=checkpoint_root,
        artifact_path=paths["fit_artifact"],
        producer_receipt_path=paths["fit_receipt"],
        model_config_sha256=hashes["model"],
        run_config_sha256=hashes["run"],
        model_config_semantic_sha256=pipeline._model_config_semantic_sha256(model),
        run_config_semantic_sha256=pipeline._run_config_semantic_sha256(run_config),
        source_code_sha256=hashes["code"],
        runtime_environment_sha256=hashes["runtime"],
        fold_callback=callback_for_run,
        production_run_claim_sha256=claim if production_attested else None,
    )
    context = SimpleNamespace(
        checkpoint_root=checkpoint_root,
        model=model,
        run_config=run_config,
        hashes=hashes,
        claim=claim,
        paths=paths,
    )
    return produced, context


def _tiny_configs() -> tuple[CausalBackboneConfig, BackboneRunConfig]:
    model = CausalBackboneConfig(
        text_dim=4,
        audio_dim=3,
        video_dim=2,
        d_model=8,
        num_heads=2,
        num_layers=1,
        ffn_dim=16,
        num_speakers=16,
        max_turns=32,
        max_relative_turn=8,
        num_classes=7,
        dropout=0.0,
        parameter_limit=100_000,
    )
    run = BackboneRunConfig(
        outer_folds=2,
        inner_validation_fraction=0.25,
        max_epochs=1,
        early_stopping_patience=1,
        batch_size=2,
        inference_batch_size=4,
        gradient_accumulation_steps=1,
        max_history_items=8,
        use_amp=False,
        text_min_df=1,
        text_max_df=1.0,
        text_max_features=32,
        text_svd_n_iter=2,
    )
    return model, run


def _fit_gate_bindings(context: SimpleNamespace) -> dict[str, object]:
    return {
        "model_config": context.model,
        "run_config": context.run_config,
        "model_config_sha256": context.hashes["model"],
        "run_config_sha256": context.hashes["run"],
        "source_code_sha256": context.hashes["code"],
        "runtime_environment_sha256": context.hashes["runtime"],
        "production_run_claim_sha256": context.claim,
        "device": torch.device("cpu"),
    }


def test_completion_alignment_loader_leaves_outcome_arrays_opaque(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    alignment = values["alignment"]
    assert alignment.selection_query_indices.tolist() == [0, 2, 4, 6]
    assert alignment.selection_cluster_codes.tolist() == [0, 0, 1, 1]
    # All other selection fields in the producer are object-array poison.  A
    # successful load proves the completion view did not deserialize them.
    selection = materialize_selection_features_after_receipt(
        receipt_path=values["receipt"],
        expected_receipt_sha256=values["preflight"].receipt_sha256,
        dataset="EmotionTalk",
        sidecar_dir=tmp_path,
        manifest_path=values["manifest"],
        config_paths=values["configs"],
        code_paths=values["code"],
        environment=ENVIRONMENT,
    )
    combined, positions = align_selection_protocol_to_producer(selection, alignment)
    assert set(combined.tolist()) == {0, 2, 4, 6}
    assert sorted(positions.tolist()) == [0, 1, 2, 3]
    assert selection.labels_materialized is False


def test_attested_staged_history_views_replace_legacy_producer_path(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path)
    fit_protocol = np.asarray(
        values["preflight"].fit.protocol_row_ids, dtype=np.int64
    )
    selection_protocol = np.arange(100, 104, dtype=np.int64)
    source_identity = _sha("staged-history-source")
    checkpoint_manifest = _sha("staged-history-checkpoints")
    artifact_path = tmp_path / "history-complete-outcome.npz"
    _write_npz(
        artifact_path,
        {
            "schema_version": np.asarray(HISTORY_COMPLETE_OUTCOME_SCHEMA),
            "dataset": np.asarray("EmotionTalk"),
            "dataset_label_order": np.asarray(
                values["preflight"].fit.label_order
            ),
            "seeds": np.asarray(EXPECTED_SEEDS, dtype=np.int64),
            "fit_protocol_row_ids": fit_protocol,
            "selection_protocol_row_ids": selection_protocol,
            "fit_cluster_codes": np.arange(len(fit_protocol), dtype=np.int32),
            "selection_cluster_codes": np.asarray([0, 0, 1, 1], dtype=np.int32),
            "source_identity_sha256": np.asarray(source_identity),
            "checkpoint_manifest_sha256": np.asarray(checkpoint_manifest),
        },
    )
    attestation = VerifiedHistoryCompletionAttestation(
        dataset="EmotionTalk",
        artifact_path=artifact_path.resolve(),
        artifact_sha256=_file_sha(artifact_path),
        completion_receipt_path=(tmp_path / "history-complete-receipt.json"),
        completion_receipt_sha256=_sha("completion-receipt"),
        fit_producer_receipt_path=(tmp_path / "history-fit-receipt.json"),
        fit_producer_receipt_sha256=_sha("fit-producer-receipt"),
        source_identity_sha256=source_identity,
        checkpoint_manifest_sha256=checkpoint_manifest,
        production_run_claim_sha256=_sha("history-production-claim"),
        fit_preflight_receipt_sha256=values["preflight"].receipt_sha256,
        config_sha256={"config": _sha("config")},
        code_sha256={"code": _sha("code")},
        runtime_environment_sha256=_sha("preflight-runtime"),
        execution_environment_sha256=_sha("execution-runtime"),
        model_config_sha256=_sha("history-model"),
        run_config_sha256=_sha("history-run"),
        utility_config_sha256=_sha("history-utility"),
    )
    fit_view = load_attested_history_fit_alignment_view(
        attestation,
        fit_preflight_receipt_path=values["receipt"],
        expected_fit_preflight_receipt_sha256=(
            values["preflight"].receipt_sha256
        ),
    )
    assert fit_view.producer_file_sha256 == attestation.artifact_sha256
    assert np.array_equal(fit_view.protocol_row_ids, fit_protocol)
    _, fit_positions = align_fit_protocol_to_producer(
        values["fit_map"], fit_view
    )
    assert sorted(fit_positions.tolist()) == list(range(len(fit_protocol)))

    alignment = load_attested_history_producer_alignment_view(
        attestation,
        fit_producer=fit_view,
    )
    assert alignment.producer_file_sha256 == attestation.artifact_sha256
    assert alignment.selection_query_indices.tolist() == [4, 5, 6, 7]
    assert alignment.protocol_row_ids.tolist() == [
        *fit_protocol.tolist(),
        *selection_protocol.tolist(),
    ]

    artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")
    with pytest.raises(
        pipeline.CurrentOnlyPipelineError,
        match="attested history artifact changed",
    ):
        load_attested_history_producer_alignment_view(
            attestation,
            fit_producer=fit_view,
        )


def test_real_fit_callback_passes_only_train_labels_to_history_free_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, run = _tiny_configs()
    request = HistoryFreeFoldRequest(
        dataset="EmotionTalk",
        seed=17,
        fold=0,
        train_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        heldout_indices=np.asarray([4, 5, 6, 7], dtype=np.int64),
        train_texts=("a", "b", "c", "d"),
        train_audio=np.zeros((4, 3), dtype=np.float32),
        train_video=np.zeros((4, 2), dtype=np.float32),
        train_labels=np.asarray([1, 2, 3, 4], dtype=np.int64),
        train_group_tokens=np.asarray(["g0", "g0", "g1", "g1"]),
        train_speaker_tokens=np.asarray(["s0", "s0", "s1", "s1"]),
        train_turns=np.asarray([0, 1, 0, 1], dtype=np.int64),
        train_histories=((), (), (), ()),
        heldout_texts=("e", "f", "g", "h"),
        heldout_audio=np.zeros((4, 3), dtype=np.float32),
        heldout_video=np.zeros((4, 2), dtype=np.float32),
        heldout_group_tokens=np.asarray(["g2", "g2", "g3", "g3"]),
        heldout_speaker_tokens=np.asarray(["s2", "s2", "s3", "s3"]),
        heldout_turns=np.asarray([0, 1, 0, 1], dtype=np.int64),
        heldout_histories=((), (), (), ()),
        heldout_labels_materialized=False,
        fit_lineage_source_identity_sha256="7" * 64,
        checkpoint_root=Path("unused"),
        model_config_sha256="8" * 64,
        run_config_sha256="9" * 64,
    )
    calls = []

    def fake_train(corpus, split, **kwargs):
        calls.append(kwargs)
        assert not any(corpus.histories)
        assert corpus.labels[split.outer_heldout_indices].tolist() == [0, 0, 0, 0]
        assert sorted(corpus.labels[split.inner_train_indices].tolist() + corpus.labels[split.inner_validation_indices].tolist()) == [1, 2, 3, 4]
        assert np.all(corpus.speaker_ids[split.outer_heldout_indices] == 0)
        assert np.all(corpus.speaker_ids[split.inner_train_indices] > 0)
        assert np.all(corpus.speaker_ids[split.inner_validation_indices] > 0)
        assert kwargs["require_complete_checkpoint"] is False
        assert kwargs["data_contract_sha256"] == "b" * 64
        return SimpleNamespace(source_identity_sha256="a" * 64, trained=object())

    def fake_predict(_trained, corpus, queries, **_kwargs):
        assert not any(corpus.histories)
        return np.full((len(queries), 7), 1.0 / 7.0, dtype=np.float32)

    monkeypatch.setattr(pipeline, "train_independent_current_only_fold_seed", fake_train)
    monkeypatch.setattr(pipeline, "predict_independent_current_only_probability", fake_predict)
    callback = make_real_current_only_fold_callback(
        model_config=model,
        run_config=run,
        data_contract_sha256="b" * 64,
        device=torch.device("cpu"),
    )
    output = callback(request)
    assert output.probability.shape == (4, 7)
    assert output.current_only_source_identity_sha256 == "a" * 64
    assert len(calls) == 1


def test_current_only_vad_targets_are_bound_to_dataset_label_order() -> None:
    model, _ = _tiny_configs()
    enabled = replace(
        model,
        affect_relation_mode="primary_history_relation",
        affect_relation_use_vad_features=True,
        auxiliary_vad_weight=0.1,
        emotion_label_order=EMOTIONTALK_LABEL_NAMES,
    )
    pipeline._validate_current_only_model_label_contract(
        enabled, EMOTIONTALK_LABEL_NAMES
    )
    with pytest.raises(
        pipeline.CurrentOnlyPipelineError, match="VAD supervision label order"
    ):
        pipeline._validate_current_only_model_label_contract(
            replace(enabled, emotion_label_order=tuple(reversed(EMOTIONTALK_LABEL_NAMES))),
            EMOTIONTALK_LABEL_NAMES,
        )


def test_complete_selection_is_checkpoint_only_and_strategy_consumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path)
    produced_fit, fit_context = _produce_fake_fit(tmp_path, values)
    state = verify_current_only_fit_for_completion(
        fit_artifact_path=produced_fit.artifact_path,
        fit_producer_receipt_path=produced_fit.receipt_path,
        expected_fit_producer_receipt_sha256=produced_fit.receipt_sha256,
        checkpoint_root=fit_context.checkpoint_root,
        fit=values["preflight"].fit,
        fit_map=values["fit_map"],
        lineage=values["lineage"],
        producer=values["fit_producer"],
        fit_preflight_receipt_path=values["receipt"],
        expected_fit_preflight_receipt_sha256=values["preflight"].receipt_sha256,
        outer_folds=2,
        **_fit_gate_bindings(fit_context),
    )
    selection_label = tmp_path / f"labels_{SELECTION_ROLE}.npz"
    selection_label.unlink()
    opened = []
    original_load = np.load

    def guarded_load(path, *args, **kwargs):
        resolved = Path(path)
        opened.append(resolved)
        if resolved == selection_label:
            raise AssertionError("complete-selection deserialized selection labels")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_load)
    selection = materialize_selection_features_after_receipt(
        receipt_path=values["receipt"],
        expected_receipt_sha256=values["preflight"].receipt_sha256,
        dataset="EmotionTalk",
        sidecar_dir=tmp_path,
        manifest_path=values["manifest"],
        config_paths=values["configs"],
        code_paths=values["code"],
        environment=ENVIRONMENT,
    )
    model, run = fit_context.model, fit_context.run_config
    load_calls = []
    active_speaker_mapping_sha = []

    class Processor:
        def transform(self, texts):
            return np.zeros((len(texts), model.text_dim), dtype=np.float32)

    def fake_complete_loader(corpus, split, **kwargs):
        load_calls.append(kwargs)
        outer_train = np.concatenate(
            [split.inner_train_indices, split.inner_validation_indices]
        )
        training_speakers = set(corpus.speaker_identity[outer_train].tolist())
        heldout_only = np.asarray(
            [
                value not in training_speakers
                for value in corpus.speaker_identity[split.outer_heldout_indices]
            ],
            dtype=bool,
        )
        assert np.all(corpus.speaker_ids[split.outer_heldout_indices][heldout_only] == 0)
        active_speaker_mapping_sha.append(corpus.speaker_mapping_sha256)
        assert kwargs["require_complete_checkpoint"] is True
        assert kwargs["data_contract_sha256"] == values["fit_map"].fit_arrays_contract_sha256
        trained = SimpleNamespace(
            summary={
                "resumed_complete_checkpoint": True,
                "resumed_partial_checkpoint": False,
            },
            processor=Processor(),
            model=object(),
        )
        return SimpleNamespace(
            source_identity_sha256=state.current_only_source_identity_sha256,
            trained=trained,
        )

    def fake_selection_predict(_model, corpus, _text, queries, contexts, **_kwargs):
        assert corpus.label_access_mode.endswith("never_scored")
        assert np.all(corpus.labels == 0)
        assert all(not value for value in contexts)
        assert corpus.speaker_mapping_sha256 == active_speaker_mapping_sha[-1]
        return np.full((len(queries), 7), 1.0 / 7.0, dtype=np.float32)

    monkeypatch.setattr(
        pipeline, "train_independent_current_only_fold_seed", fake_complete_loader
    )
    monkeypatch.setattr(pipeline, "predict_one_probability_per_query", fake_selection_predict)
    completed = complete_current_only_selection_probabilities(
        fit=values["preflight"].fit,
        selection=selection,
        fit_map=values["fit_map"],
        lineage=values["lineage"],
        fit_producer=values["fit_producer"],
        producer=values["alignment"],
        fit_state=state,
        fit_preflight_receipt_path=values["receipt"],
        expected_fit_preflight_receipt_sha256=values["preflight"].receipt_sha256,
        checkpoint_root=fit_context.checkpoint_root,
        model_config=model,
        run_config=run,
        model_config_sha256=fit_context.hashes["model"],
        run_config_sha256=fit_context.hashes["run"],
        source_code_sha256=fit_context.hashes["code"],
        runtime_environment_sha256=fit_context.hashes["runtime"],
        production_run_claim_sha256=fit_context.claim,
        selection_feature_sha256=values["preflight"].receipt["sidecars"][
            SELECTION_ROLE
        ]["feature_sha256"],
        artifact_path=fit_context.paths["complete_artifact"],
        completion_receipt_path=fit_context.paths["complete_receipt"],
        device=torch.device("cpu"),
    )
    assert selection_label not in opened
    assert len(load_calls) == len(EXPECTED_SEEDS) * 2
    cache = load_private_npz_mapping(completed.artifact_path)
    artifact = current_only_artifact_from_mapping(cache, values["alignment"])
    assert artifact.split_probability("fit").shape == (5, 4, 7)
    assert artifact.split_probability("model_selection").shape == (5, 4, 7)
    _, producer_positions = align_fit_protocol_to_producer(
        values["fit_map"], values["fit_producer"]
    )
    for fit_row, producer_position in enumerate(producer_positions):
        assert np.all(
            artifact.fit_probability[
                :, int(producer_position), fit_row % len(artifact.label_order)
            ]
            == 1.0
        )
    receipt = json.loads(completed.receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == CURRENT_ONLY_COMPLETION_RECEIPT_SCHEMA
    assert receipt["completion_contract"]["complete_checkpoint_only"] is True
    assert receipt["completion_contract"]["selection_label_deserialized"] is False
    assert receipt["completion_contract"]["selection_label_file_accessed"] is False
    assert receipt["completion_contract"]["performance_metric_computed"] is False
    serialized = json.dumps(receipt)
    assert str(tmp_path) not in serialized
    assert '"probabilities"' not in serialized
    assert '"labels"' not in serialized


def test_incomplete_checkpoint_tree_fails_before_completion(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    produced_fit, fit_context = _produce_fake_fit(tmp_path, values)
    missing = (
        fit_context.checkpoint_root / "seed_00017" / "fold_00" / "checkpoint.pt"
    )
    missing.unlink()
    with pytest.raises(StageAContractError, match="missing"):
        verify_current_only_fit_for_completion(
            fit_artifact_path=produced_fit.artifact_path,
            fit_producer_receipt_path=produced_fit.receipt_path,
            expected_fit_producer_receipt_sha256=produced_fit.receipt_sha256,
            checkpoint_root=fit_context.checkpoint_root,
            fit=values["preflight"].fit,
            fit_map=values["fit_map"],
            lineage=values["lineage"],
            producer=values["fit_producer"],
            fit_preflight_receipt_path=values["receipt"],
            expected_fit_preflight_receipt_sha256=values["preflight"].receipt_sha256,
            outer_folds=2,
            **_fit_gate_bindings(fit_context),
        )


def test_partial_checkpoint_fails_semantic_gate_before_selection_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path)
    produced_fit, fit_context = _produce_fake_fit(
        tmp_path, values, partial_checkpoint=(17, 0)
    )
    selection_feature = tmp_path / f"features_{SELECTION_ROLE}.npz"
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path.resolve() == selection_feature.resolve():
            raise AssertionError("selection feature opened before complete checkpoint gate")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(
        pipeline.CurrentOnlyPipelineError, match="not semantically complete"
    ):
        verify_current_only_fit_for_completion(
            fit_artifact_path=produced_fit.artifact_path,
            fit_producer_receipt_path=produced_fit.receipt_path,
            expected_fit_producer_receipt_sha256=produced_fit.receipt_sha256,
            checkpoint_root=fit_context.checkpoint_root,
            fit=values["preflight"].fit,
            fit_map=values["fit_map"],
            lineage=values["lineage"],
            producer=values["fit_producer"],
            fit_preflight_receipt_path=values["receipt"],
            expected_fit_preflight_receipt_sha256=values[
                "preflight"
            ].receipt_sha256,
            outer_folds=2,
            **_fit_gate_bindings(fit_context),
        )


def test_completion_rejects_selection_cluster_partition_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path)
    produced_fit, fit_context = _produce_fake_fit(tmp_path, values)
    state = verify_current_only_fit_for_completion(
        fit_artifact_path=produced_fit.artifact_path,
        fit_producer_receipt_path=produced_fit.receipt_path,
        expected_fit_producer_receipt_sha256=produced_fit.receipt_sha256,
        checkpoint_root=fit_context.checkpoint_root,
        fit=values["preflight"].fit,
        fit_map=values["fit_map"],
        lineage=values["lineage"],
        producer=values["fit_producer"],
        fit_preflight_receipt_path=values["receipt"],
        expected_fit_preflight_receipt_sha256=values["preflight"].receipt_sha256,
        outer_folds=2,
        **_fit_gate_bindings(fit_context),
    )
    selection = materialize_selection_features_after_receipt(
        receipt_path=values["receipt"],
        expected_receipt_sha256=values["preflight"].receipt_sha256,
        dataset="EmotionTalk",
        sidecar_dir=tmp_path,
        manifest_path=values["manifest"],
        config_paths=values["configs"],
        code_paths=values["code"],
        environment=ENVIRONMENT,
    )
    bad_alignment = replace(
        values["alignment"],
        selection_cluster_codes=np.arange(
            len(values["alignment"].selection_cluster_codes), dtype=np.int64
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "train_independent_current_only_fold_seed",
        lambda *_args, **_kwargs: pytest.fail(
            "checkpoint loader ran after selection cluster mismatch"
        ),
    )
    model, run = fit_context.model, fit_context.run_config
    with pytest.raises(
        pipeline.CurrentOnlyPipelineError, match="selection clusters differ"
    ):
        complete_current_only_selection_probabilities(
            fit=values["preflight"].fit,
            selection=selection,
            fit_map=values["fit_map"],
            lineage=values["lineage"],
            fit_producer=values["fit_producer"],
            producer=bad_alignment,
            fit_state=state,
            fit_preflight_receipt_path=values["receipt"],
            expected_fit_preflight_receipt_sha256=values[
                "preflight"
            ].receipt_sha256,
            checkpoint_root=fit_context.checkpoint_root,
            model_config=model,
            run_config=run,
            model_config_sha256=fit_context.hashes["model"],
            run_config_sha256=fit_context.hashes["run"],
            source_code_sha256=fit_context.hashes["code"],
            runtime_environment_sha256=fit_context.hashes["runtime"],
            production_run_claim_sha256=fit_context.claim,
            selection_feature_sha256=state.selection_feature_sha256,
            artifact_path=fit_context.paths["complete_artifact"],
            completion_receipt_path=fit_context.paths["complete_receipt"],
            device=torch.device("cpu"),
        )


def test_completion_rejects_fit_artifact_replaced_after_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path)
    produced_fit, fit_context = _produce_fake_fit(tmp_path, values)
    state = verify_current_only_fit_for_completion(
        fit_artifact_path=produced_fit.artifact_path,
        fit_producer_receipt_path=produced_fit.receipt_path,
        expected_fit_producer_receipt_sha256=produced_fit.receipt_sha256,
        checkpoint_root=fit_context.checkpoint_root,
        fit=values["preflight"].fit,
        fit_map=values["fit_map"],
        lineage=values["lineage"],
        producer=values["fit_producer"],
        fit_preflight_receipt_path=values["receipt"],
        expected_fit_preflight_receipt_sha256=values["preflight"].receipt_sha256,
        outer_folds=2,
        **_fit_gate_bindings(fit_context),
    )
    selection = materialize_selection_features_after_receipt(
        receipt_path=values["receipt"],
        expected_receipt_sha256=values["preflight"].receipt_sha256,
        dataset="EmotionTalk",
        sidecar_dir=tmp_path,
        manifest_path=values["manifest"],
        config_paths=values["configs"],
        code_paths=values["code"],
        environment=ENVIRONMENT,
    )
    state.artifact_path.write_bytes(b"replaced-after-fit-gate")
    monkeypatch.setattr(
        pipeline,
        "train_independent_current_only_fold_seed",
        lambda *_args, **_kwargs: pytest.fail("checkpoint loader ran after fit mutation"),
    )
    model, run = fit_context.model, fit_context.run_config
    with pytest.raises(
        pipeline.CurrentOnlyPipelineError,
        match="fit probability artifact/receipt changed after verification",
    ):
        complete_current_only_selection_probabilities(
            fit=values["preflight"].fit,
            selection=selection,
            fit_map=values["fit_map"],
            lineage=values["lineage"],
            fit_producer=values["fit_producer"],
            producer=values["alignment"],
            fit_state=state,
            fit_preflight_receipt_path=values["receipt"],
            expected_fit_preflight_receipt_sha256=values["preflight"].receipt_sha256,
            checkpoint_root=fit_context.checkpoint_root,
            model_config=model,
            run_config=run,
            model_config_sha256=fit_context.hashes["model"],
            run_config_sha256=fit_context.hashes["run"],
            source_code_sha256=fit_context.hashes["code"],
            runtime_environment_sha256=fit_context.hashes["runtime"],
            production_run_claim_sha256=fit_context.claim,
            selection_feature_sha256=state.selection_feature_sha256,
            artifact_path=fit_context.paths["complete_artifact"],
            completion_receipt_path=fit_context.paths["complete_receipt"],
            device=torch.device("cpu"),
        )


def test_fit_gate_rejects_synthetic_callback_receipt(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    produced, fit_context = _produce_fake_fit(
        tmp_path, values, production_attested=False
    )
    with pytest.raises(
        pipeline.CurrentOnlyPipelineError,
        match="crossed isolation contract|not production evidence",
    ):
        verify_current_only_fit_for_completion(
            fit_artifact_path=produced.artifact_path,
            fit_producer_receipt_path=produced.receipt_path,
            expected_fit_producer_receipt_sha256=produced.receipt_sha256,
            checkpoint_root=fit_context.checkpoint_root,
            fit=values["preflight"].fit,
            fit_map=values["fit_map"],
            lineage=values["lineage"],
            producer=values["fit_producer"],
            fit_preflight_receipt_path=values["receipt"],
            expected_fit_preflight_receipt_sha256=values["preflight"].receipt_sha256,
            outer_folds=2,
            **_fit_gate_bindings(fit_context),
        )


def test_fit_gate_rehashes_live_arrays_before_checkpoint_restore(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    produced, fit_context = _produce_fake_fit(tmp_path, values)
    values["preflight"].fit.audio[0, 0] += np.float32(1.0)
    with pytest.raises(
        pipeline.CurrentOnlyPipelineError, match="live fit arrays changed"
    ):
        verify_current_only_fit_for_completion(
            fit_artifact_path=produced.artifact_path,
            fit_producer_receipt_path=produced.receipt_path,
            expected_fit_producer_receipt_sha256=produced.receipt_sha256,
            checkpoint_root=fit_context.checkpoint_root,
            fit=values["preflight"].fit,
            fit_map=values["fit_map"],
            lineage=values["lineage"],
            producer=values["fit_producer"],
            fit_preflight_receipt_path=values["receipt"],
            expected_fit_preflight_receipt_sha256=values["preflight"].receipt_sha256,
            outer_folds=2,
            **_fit_gate_bindings(fit_context),
        )


def test_fit_gate_rejects_semantically_wrong_checkpoint_identity(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path)
    produced, fit_context = _produce_fake_fit(
        tmp_path, values, bad_checkpoint_identity=(17, 0)
    )
    with pytest.raises(
        pipeline.CurrentOnlyPipelineError,
        match="identity differs from source/fold/config/split",
    ):
        verify_current_only_fit_for_completion(
            fit_artifact_path=produced.artifact_path,
            fit_producer_receipt_path=produced.receipt_path,
            expected_fit_producer_receipt_sha256=produced.receipt_sha256,
            checkpoint_root=fit_context.checkpoint_root,
            fit=values["preflight"].fit,
            fit_map=values["fit_map"],
            lineage=values["lineage"],
            producer=values["fit_producer"],
            fit_preflight_receipt_path=values["receipt"],
            expected_fit_preflight_receipt_sha256=values["preflight"].receipt_sha256,
            outer_folds=2,
            **_fit_gate_bindings(fit_context),
        )


def test_fit_gate_rejects_runtime_lineage_change(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    produced, fit_context = _produce_fake_fit(tmp_path, values)
    bindings = _fit_gate_bindings(fit_context)
    bindings["runtime_environment_sha256"] = _sha("changed-runtime")
    with pytest.raises(
        pipeline.CurrentOnlyPipelineError, match="completion claim lineage changed"
    ):
        verify_current_only_fit_for_completion(
            fit_artifact_path=produced.artifact_path,
            fit_producer_receipt_path=produced.receipt_path,
            expected_fit_producer_receipt_sha256=produced.receipt_sha256,
            checkpoint_root=fit_context.checkpoint_root,
            fit=values["preflight"].fit,
            fit_map=values["fit_map"],
            lineage=values["lineage"],
            producer=values["fit_producer"],
            fit_preflight_receipt_path=values["receipt"],
            expected_fit_preflight_receipt_sha256=values["preflight"].receipt_sha256,
            outer_folds=2,
            **bindings,
        )


def test_private_claim_resume_and_os_lock_fail_closed(tmp_path: Path) -> None:
    private_root = tmp_path / "claimed-current-only"
    claim = _sha("one-exact-production-lineage")
    created = pipeline.claim_or_resume_current_only_private_root(
        private_root,
        production_claim_sha256=claim,
        allow_resume=False,
    )
    assert created == private_root.resolve()
    assert pipeline.claim_or_resume_current_only_private_root(
        private_root,
        production_claim_sha256=claim,
        allow_resume=True,
    ) == private_root.resolve()
    with pytest.raises(
        pipeline.CurrentOnlyPipelineError, match="claim lineage changed"
    ):
        pipeline.claim_or_resume_current_only_private_root(
            private_root,
            production_claim_sha256=_sha("different-lineage"),
            allow_resume=True,
        )
    with pipeline._ExclusiveCurrentOnlyFitLock(private_root.resolve()):
        with pytest.raises(
            pipeline.CurrentOnlyPipelineError, match="holds the private-root lock"
        ):
            with pipeline._ExclusiveCurrentOnlyFitLock(private_root.resolve()):
                pytest.fail("a second fit lock was acquired")
    with pipeline._ExclusiveCurrentOnlyFitLock(private_root.resolve()):
        pass


def test_real_no_vad_interrupted_resume_matches_uninterrupted_and_selection_inference(
    tmp_path: Path,
) -> None:
    model, base_run = _tiny_configs()
    assert model.auxiliary_vad_weight == 0.0
    run_config = replace(
        base_run,
        max_epochs=2,
        early_stopping_patience=3,
        subset_dropout_probability=0.0,
    )
    request = HistoryFreeFoldRequest(
        dataset="EmotionTalk",
        seed=17,
        fold=0,
        train_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        heldout_indices=np.asarray([4, 5, 6, 7], dtype=np.int64),
        train_texts=(
            "calm alpha one",
            "calm alpha two",
            "bright beta one",
            "bright beta two",
        ),
        train_audio=np.asarray(
            [[0.1, 0.0, 0.2], [0.2, 0.1, 0.0], [0.0, 0.3, 0.1], [0.1, 0.2, 0.3]],
            dtype=np.float32,
        ),
        train_video=np.asarray(
            [[0.1, 0.2], [0.2, 0.1], [0.3, 0.0], [0.0, 0.3]],
            dtype=np.float32,
        ),
        train_labels=np.asarray([0, 1, 2, 3], dtype=np.int64),
        train_group_tokens=np.asarray(["g0", "g0", "g1", "g1"]),
        train_speaker_tokens=np.asarray(["s0", "s0", "s1", "s1"]),
        train_turns=np.asarray([0, 1, 0, 1], dtype=np.int64),
        train_histories=((), (), (), ()),
        heldout_texts=(
            "quiet gamma one",
            "quiet gamma two",
            "sharp delta one",
            "sharp delta two",
        ),
        heldout_audio=np.asarray(
            [[0.3, 0.1, 0.0], [0.2, 0.2, 0.1], [0.0, 0.1, 0.3], [0.1, 0.0, 0.2]],
            dtype=np.float32,
        ),
        heldout_video=np.asarray(
            [[0.2, 0.2], [0.1, 0.3], [0.3, 0.1], [0.2, 0.0]],
            dtype=np.float32,
        ),
        heldout_group_tokens=np.asarray(["g2", "g2", "g3", "g3"]),
        heldout_speaker_tokens=np.asarray(["s2", "s2", "s3", "s3"]),
        heldout_turns=np.asarray([0, 1, 0, 1], dtype=np.int64),
        heldout_histories=((), (), (), ()),
        heldout_labels_materialized=False,
        fit_lineage_source_identity_sha256="7" * 64,
        checkpoint_root=tmp_path / "resumed-checkpoints",
        model_config_sha256="8" * 64,
        run_config_sha256="9" * 64,
    )
    corpus = pipeline._corpus_from_fold_request(request, model_config=model)
    split = pipeline._split_from_outer_partition(
        corpus,
        outer_train=request.train_indices,
        heldout=request.heldout_indices,
        validation_fraction=run_config.inner_validation_fraction,
        seed=request.seed,
        fold=request.fold,
    )
    common = {
        "producer_source_identity_sha256": request.fit_lineage_source_identity_sha256,
        "model_config": model,
        "run_config": run_config,
        "seed": request.seed,
        "device": torch.device("cpu"),
        "data_contract_sha256": "b" * 64,
    }
    with pytest.raises(PlannedCheckpointInterruption):
        pipeline.train_independent_current_only_fold_seed(
            corpus,
            split,
            checkpoint_root=request.checkpoint_root,
            test_interrupt_after_epoch=0,
            **common,
        )
    resumed = pipeline.train_independent_current_only_fold_seed(
        corpus,
        split,
        checkpoint_root=request.checkpoint_root,
        **common,
    )
    assert resumed.trained.summary["resumed_partial_checkpoint"] is True
    uninterrupted = pipeline.train_independent_current_only_fold_seed(
        corpus,
        split,
        checkpoint_root=tmp_path / "uninterrupted-checkpoints",
        **common,
    )
    resumed_probability = pipeline.predict_independent_current_only_probability(
        resumed,
        corpus,
        request.heldout_indices,
        device=torch.device("cpu"),
        batch_size=run_config.inference_batch_size,
        max_history_items=run_config.max_history_items,
    )
    uninterrupted_probability = pipeline.predict_independent_current_only_probability(
        uninterrupted,
        corpus,
        request.heldout_indices,
        device=torch.device("cpu"),
        batch_size=run_config.inference_batch_size,
        max_history_items=run_config.max_history_items,
    )
    np.testing.assert_allclose(
        resumed_probability,
        uninterrupted_probability,
        rtol=0.0,
        atol=1.0e-7,
    )
    payload = _torch_load_local(resumed.trained.checkpoint_path)
    strict_model = CausalMultimodalBackbone(model)
    strict_model.load_state_dict(payload["model_state"], strict=True)
    strict_model.load_state_dict(payload["best_model_state"], strict=True)
    restored = pipeline.train_independent_current_only_fold_seed(
        corpus,
        split,
        checkpoint_root=request.checkpoint_root,
        require_complete_checkpoint=True,
        **common,
    )
    assert restored.trained.summary["resumed_complete_checkpoint"] is True
    mapping, mapping_sha = pipeline._fit_speaker_mapping(
        "EmotionTalk",
        request.train_speaker_tokens,
        num_speakers=model.num_speakers,
    )
    selection_corpus = pipeline._make_corpus(
        dataset="EmotionTalk",
        texts=("unseen epsilon one", "unseen epsilon two"),
        audio=np.asarray([[0.1, 0.1, 0.2], [0.2, 0.0, 0.1]], dtype=np.float32),
        video=np.asarray([[0.2, 0.1], [0.1, 0.2]], dtype=np.float32),
        labels=np.zeros(2, dtype=np.int64),
        groups=np.asarray(["selection-g0", "selection-g1"]),
        speakers=np.asarray(["new-speaker-0", "new-speaker-1"]),
        turns=np.asarray([0, 0], dtype=np.int64),
        role=SELECTION_ROLE,
        model_config=model,
        fit_speaker_mapping=mapping,
        speaker_mapping_sha256=mapping_sha,
        label_access_mode="selection_features_only_zero_placeholder_labels_never_scored",
    )
    selection_text = restored.trained.processor.transform(selection_corpus.texts)
    selection_probability = pipeline.predict_one_probability_per_query(
        restored.trained.model,
        selection_corpus,
        selection_text,
        np.arange(2, dtype=np.int64),
        ((), ()),
        device=torch.device("cpu"),
        batch_size=run_config.inference_batch_size,
        max_history_items=run_config.max_history_items,
    )
    assert selection_probability.shape == (2, len(EMOTIONTALK_LABEL_NAMES))
    assert np.isfinite(selection_probability).all()
    np.testing.assert_allclose(selection_probability.sum(axis=1), 1.0, atol=1.0e-6)
    assert selection_corpus.label_access_mode.endswith("never_scored")
    assert np.all(selection_corpus.labels == 0)
