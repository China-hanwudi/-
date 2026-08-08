from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch", reason="causal backbone runner tests require PyTorch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.causal_multimodal_backbone import (  # noqa: E402
    CausalBackboneConfig,
    CausalMultimodalBackbone,
)
from hva_affect.data_contract import ContractError  # noqa: E402
from hva_affect.emotiontalk_causal_backbone_runner import (  # noqa: E402
    BackboneRunConfig,
    FIT_ROLE,
    OpenRoleCorpus,
    PlannedCheckpointInterruption,
    SELECTION_ROLE,
    UtilitySamplingConfig,
    _corpus_contract_sha256,
    _nll,
    _role_assignment_sha256,
    classification_metrics,
    create_verified_corpus_provenance,
    execute_crossfit_backbone,
    fit_fold_text_processor,
    make_crossfit_splits,
    pack_query_contexts,
    predict_one_probability_per_query,
    train_one_fold_seed,
)
from hva_affect.meld_text_pilot import true_class_loss  # noqa: E402


def tiny_model_config() -> CausalBackboneConfig:
    return CausalBackboneConfig(
        text_dim=8,
        audio_dim=10,
        video_dim=12,
        d_model=16,
        num_heads=4,
        num_layers=1,
        ffn_dim=24,
        num_speakers=4,
        max_turns=32,
        max_relative_turn=8,
        num_classes=7,
        dropout=0.0,
    )


def tiny_run_config() -> BackboneRunConfig:
    return BackboneRunConfig(
        outer_folds=2,
        inner_validation_fraction=0.25,
        max_epochs=1,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
        batch_size=8,
        inference_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        label_smoothing=0.0,
        subset_dropout_probability=0.25,
        max_history_items=8,
        use_amp=False,
        max_cuda_memory_mib=7800,
        text_analyzer="char",
        text_ngram_min=1,
        text_ngram_max=2,
        text_min_df=1,
        text_max_df=1.0,
        text_max_features=256,
        text_sublinear_tf=True,
        text_svd_n_iter=3,
    )


def synthetic_corpus() -> OpenRoleCorpus:
    rng = np.random.default_rng(20260808)
    keys: list[str] = []
    texts: list[str] = []
    labels: list[int] = []
    groups: list[str] = []
    roles: list[str] = []
    buckets: list[int] = []
    speakers: list[int] = []
    turns: list[int] = []
    histories: list[tuple[int, ...]] = []
    for group_index in range(11):
        role = FIT_ROLE if group_index < 8 else SELECTION_ROLE
        prior: list[int] = []
        for turn in range(4):
            row = len(keys)
            keys.append(f"synthetic_{group_index:02d}_{turn:02d}")
            texts.append(
                f"group{group_index} turn{turn} calm happy sad angry surprise disgust fear"
            )
            labels.append((group_index + turn) % 7)
            groups.append(f"dialogue_{group_index:02d}")
            roles.append(role)
            buckets.append(10 if role == FIT_ROLE else 70)
            speakers.append(group_index % 2 + 1)
            turns.append(turn)
            histories.append(tuple(prior))
            prior.append(row)
    rows = len(keys)
    corpus = OpenRoleCorpus(
        keys=np.asarray(keys),
        texts=tuple(texts),
        audio=rng.normal(size=(rows, 10)).astype(np.float32),
        video=rng.normal(size=(rows, 12)).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups),
        roles=np.asarray(roles),
        buckets=np.asarray(buckets, dtype=np.int16),
        speaker_ids=np.asarray(speakers, dtype=np.int64),
        turn_ids=np.asarray(turns, dtype=np.int64),
        histories=tuple(histories),
        speaker_identity=np.asarray(
            [f"speaker_{group_index % 2}" for group_index in range(11) for _ in range(4)]
        ),
        speaker_mapping_sha256="e" * 64,
        label_access_mode="synthetic_strict_physical_sidecars",
    )
    corpus.validate(tiny_model_config())
    return corpus


def synthetic_provenance(corpus: OpenRoleCorpus):
    manifest_sha = "a" * 64
    return create_verified_corpus_provenance(
        dataset_id="SyntheticEmotion",
        manifest_schema="synthetic_manifest_v1",
        manifest_status="strict_sidecars_verified",
        manifest_sha256=manifest_sha,
        source_hashes={
            "sidecar_manifest": manifest_sha,
            "synthetic_input": "b" * 64,
        },
        label_order=tuple(f"label_{index}" for index in range(tiny_model_config().num_classes)),
        role_rows={
            FIT_ROLE: int(len(corpus.role_indices(FIT_ROLE))),
            SELECTION_ROLE: int(len(corpus.role_indices(SELECTION_ROLE))),
        },
        audio_dim=corpus.audio.shape[1],
        video_dim=corpus.video.shape[1],
        role_assignment_sha256=_role_assignment_sha256(corpus),
        speaker_mapping_sha256=corpus.speaker_mapping_sha256,
        corpus_contract_sha256=_corpus_contract_sha256(corpus),
        verification_origin="synthetic_contract_test",
    )


def test_crossfit_is_group_disjoint_and_model_selection_is_inference_only() -> None:
    corpus = synthetic_corpus()
    splits = make_crossfit_splits(
        corpus, outer_folds=2, validation_fraction=0.25, seed=17
    )
    held = []
    for split in splits:
        partitions = (
            split.inner_train_indices,
            split.inner_validation_indices,
            split.outer_heldout_indices,
        )
        group_sets = [set(corpus.groups[indices]) for indices in partitions]
        assert not group_sets[0] & group_sets[1]
        assert not group_sets[0] & group_sets[2]
        assert not group_sets[1] & group_sets[2]
        assert all(set(corpus.roles[indices]) == {FIT_ROLE} for indices in partitions)
        held.extend(split.outer_heldout_indices.tolist())
    assert set(held) == set(corpus.role_indices(FIT_ROLE).tolist())
    assert not set(held) & set(corpus.role_indices(SELECTION_ROLE).tolist())


def test_context_packer_rejects_future_and_cross_group_rows() -> None:
    corpus = synthetic_corpus()
    processor = fit_fold_text_processor(
        corpus.texts,
        np.arange(8),
        output_dim=8,
        config=tiny_run_config(),
        seed=17,
    )
    text = processor.transform(corpus.texts)
    packed = pack_query_contexts(
        corpus, text, [3], [[0, 2]], max_history_items=8
    )
    assert packed["text_features"].shape == (1, 3, 8)
    assert packed["history_mask"].tolist() == [[True, True, False]]
    with pytest.raises(ContractError, match="non-history"):
        pack_query_contexts(corpus, text, [2], [[3]], max_history_items=8)
    with pytest.raises(ContractError, match="non-history"):
        pack_query_contexts(corpus, text, [3], [[4]], max_history_items=8)


def test_corpus_validation_fails_on_latent_future_leakage() -> None:
    corpus = synthetic_corpus()
    histories = list(corpus.histories)
    histories[2] = (3,)
    invalid = replace(corpus, histories=tuple(histories))
    with pytest.raises(ContractError, match="future turn"):
        invalid.validate(tiny_model_config())


def test_corpus_validation_rejects_cross_speaker_history() -> None:
    corpus = synthetic_corpus()
    speaker_identity = np.asarray(corpus.speaker_identity).copy()
    speaker_identity[2] = "different_speaker"
    invalid = replace(corpus, speaker_identity=speaker_identity)
    with pytest.raises(ContractError, match="speaker boundary"):
        invalid.validate(tiny_model_config())


def test_one_query_one_selected_context_probability_interface() -> None:
    corpus = synthetic_corpus()
    config = tiny_run_config()
    processor = fit_fold_text_processor(
        corpus.texts,
        corpus.role_indices(FIT_ROLE)[:16],
        output_dim=8,
        config=config,
        seed=17,
    )
    text = processor.transform(corpus.texts)
    model = CausalMultimodalBackbone(tiny_model_config()).eval()
    probability = predict_one_probability_per_query(
        model,
        corpus,
        text,
        [2, 3],
        [(0,), (0, 2)],
        device=torch.device("cpu"),
        batch_size=2,
        max_history_items=8,
    )
    assert probability.shape == (2, 7)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)


def test_verified_provenance_rejects_post_verification_forgery() -> None:
    corpus = synthetic_corpus()
    provenance = synthetic_provenance(corpus)
    provenance.validate(corpus, tiny_model_config())
    with pytest.raises(ContractError, match="strict physical"):
        replace(provenance, strict_role_feature_sidecars=False).validate(
            corpus, tiny_model_config()
        )
    with pytest.raises(ContractError, match="role rows"):
        replace(
            provenance,
            role_rows={FIT_ROLE: 1, SELECTION_ROLE: 1},
        ).validate(corpus, tiny_model_config())
    forged_hashes = dict(provenance.source_hashes)
    forged_hashes["synthetic_input"] = "f" * 64
    with pytest.raises(ContractError, match="changed after verification"):
        replace(provenance, source_hashes=forged_hashes).validate(
            corpus, tiny_model_config()
        )


def test_tiny_end_to_end_cache_provenance_and_atomic_resume(tmp_path: Path) -> None:
    corpus = synthetic_corpus()
    repository = tmp_path / "public_repo"
    repository.mkdir()
    private = tmp_path / "private_artifacts"
    public = repository / "results" / "causal.json"
    kwargs = dict(
        provenance=synthetic_provenance(corpus),
        model_config=tiny_model_config(),
        run_config=tiny_run_config(),
        sampling_config=UtilitySamplingConfig(1, 4, 20260808, True),
        seeds=(17,),
        private_output_dir=private,
        repository_root=repository,
        device=torch.device("cpu"),
    )
    report = execute_crossfit_backbone(
        corpus, public_output_path=public, **kwargs
    )
    assert report["training"]["parameter_count"] < 2_000_000
    assert report["performance_claim_gate"]["authorized"] is False
    assert report["data_boundary"]["strict_role_feature_sidecars"] is True
    assert report["data_boundary"]["strict_role_label_sidecars"] is True
    assert report["training"]["seeds"] == [17]
    assert len(report["training"]["fold_partitions"]) == 2
    assert report["provenance"]["input_hashes"]["synthetic_input"] == "b" * 64
    assert report["dataset_label_order"] == [f"label_{index}" for index in range(7)]
    assert report["verified_manifest"]["schema_version"] == "synthetic_manifest_v1"
    assert report["rows_and_groups"]["fit_history_eligible_rows"] == 24
    assert report["rows_and_groups"]["model_selection_history_eligible_rows"] == 9
    assert report["probability_protocol"]["current_only_semantics"] == (
        "same_trained_model_empty_history_intervention_"
        "not_independently_trained_baseline"
    )
    assert report["feature_contract"]["actual_dimensions"] == {
        "text": 8,
        "audio": 10,
        "video": 12,
    }
    assert report["provenance"]["internal_code_hashes_start"] == report["provenance"][
        "internal_code_hashes_end"
    ]
    assert report["provenance"]["runtime_environment_sha256"]
    assert report["public_artifact_policy"][
        "contains_row_level_keys_predictions_utilities_or_embeddings"
    ] is False
    cache = private / "emotiontalk_causal_backbone_oof_v1.npz"
    assert cache.is_file() and public.is_file()
    with np.load(cache, allow_pickle=False) as archive:
        assert str(archive["utility_source"]).startswith("recomputed_from_causal")
        assert archive["fit_endpoint_probability_oof"].shape[0] == 1
        assert np.isfinite(archive["fit_forward_utility"]).all()
        assert str(archive["checkpoint_manifest_sha256"]).strip()
        assert str(archive["matrix_fit_endpoint_probability_oof_sha256"]).strip()
        assert archive["dataset_label_order"].tolist() == [
            f"label_{index}" for index in range(7)
        ]
    assert len(report["provenance"]["private_matrix_hashes"]) == 8
    public_payload = json.loads(public.read_text(encoding="utf-8"))
    assert "private_artifacts" not in json.dumps(public_payload)

    resumed_public = repository / "results" / "causal_resumed.json"
    resumed = execute_crossfit_backbone(
        corpus, public_output_path=resumed_public, **kwargs
    )
    assert all(
        row["resumed_complete_checkpoint"]
        for row in resumed["training"]["fold_runs"]
    )


def test_private_artifacts_cannot_be_written_inside_public_repo(tmp_path: Path) -> None:
    corpus = synthetic_corpus()
    repository = tmp_path / "repo"
    repository.mkdir()
    with pytest.raises(ContractError, match="outside"):
        execute_crossfit_backbone(
            corpus,
            provenance=synthetic_provenance(corpus),
            model_config=tiny_model_config(),
            run_config=tiny_run_config(),
            sampling_config=UtilitySamplingConfig(1, 4, 20260808, True),
            seeds=(17,),
            private_output_dir=repository / "private",
            public_output_path=repository / "report.json",
            repository_root=repository,
            device=torch.device("cpu"),
        )


def test_public_aggregate_is_write_once_without_overwrite_escape(tmp_path: Path) -> None:
    corpus = synthetic_corpus()
    repository = tmp_path / "repo"
    repository.mkdir()
    public = repository / "report.json"
    public.write_text('{"existing": true}\n', encoding="utf-8")
    before = public.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        execute_crossfit_backbone(
            corpus,
            provenance=synthetic_provenance(corpus),
            model_config=tiny_model_config(),
            run_config=tiny_run_config(),
            sampling_config=UtilitySamplingConfig(1, 4, 20260808, True),
            seeds=(17,),
            private_output_dir=tmp_path / "private",
            public_output_path=public,
            repository_root=repository,
            device=torch.device("cpu"),
        )
    assert public.read_bytes() == before


def test_partial_checkpoint_resume_is_tensor_and_prediction_equivalent(
    tmp_path: Path,
) -> None:
    corpus = synthetic_corpus()
    split = make_crossfit_splits(
        corpus, outer_folds=2, validation_fraction=0.25, seed=17
    )[0]
    model_config = replace(tiny_model_config(), dropout=0.15)
    run_config = replace(
        tiny_run_config(), max_epochs=3, early_stopping_patience=4
    )
    kwargs = dict(
        corpus=corpus,
        split=split,
        model_config=model_config,
        run_config=run_config,
        seed=17,
        source_identity="f" * 64,
        device=torch.device("cpu"),
    )
    uninterrupted = train_one_fold_seed(
        checkpoint_root=tmp_path / "uninterrupted", **kwargs
    )
    with pytest.raises(PlannedCheckpointInterruption, match="planned interruption"):
        train_one_fold_seed(
            checkpoint_root=tmp_path / "resumed",
            test_interrupt_after_epoch=0,
            **kwargs,
        )
    resumed = train_one_fold_seed(checkpoint_root=tmp_path / "resumed", **kwargs)
    assert resumed.summary["resumed_partial_checkpoint"] is True
    assert resumed.summary["best_epoch"] == uninterrupted.summary["best_epoch"]
    assert resumed.summary["best_validation_nll"] == uninterrupted.summary[
        "best_validation_nll"
    ]
    uninterrupted_state = uninterrupted.model.state_dict()
    resumed_state = resumed.model.state_dict()
    assert set(uninterrupted_state) == set(resumed_state)
    assert all(
        torch.equal(uninterrupted_state[name], resumed_state[name])
        for name in uninterrupted_state
    )
    contexts = [corpus.histories[int(index)] for index in split.outer_heldout_indices]
    uninterrupted_probability = predict_one_probability_per_query(
        uninterrupted.model,
        corpus,
        uninterrupted.text_features,
        split.outer_heldout_indices,
        contexts,
        device=torch.device("cpu"),
        batch_size=32,
        max_history_items=8,
    )
    resumed_probability = predict_one_probability_per_query(
        resumed.model,
        corpus,
        resumed.text_features,
        split.outer_heldout_indices,
        contexts,
        device=torch.device("cpu"),
        batch_size=32,
        max_history_items=8,
    )
    np.testing.assert_array_equal(resumed_probability, uninterrupted_probability)


def test_complete_checkpoint_only_mode_never_creates_or_resumes_training(
    tmp_path: Path,
) -> None:
    corpus = synthetic_corpus()
    split = make_crossfit_splits(
        corpus, outer_folds=2, validation_fraction=0.25, seed=17
    )[0]
    kwargs = dict(
        corpus=corpus,
        split=split,
        model_config=tiny_model_config(),
        run_config=replace(
            tiny_run_config(), max_epochs=2, early_stopping_patience=3
        ),
        seed=17,
        source_identity="e" * 64,
        device=torch.device("cpu"),
    )

    missing_root = tmp_path / "missing"
    with pytest.raises(ContractError, match="cannot create"):
        train_one_fold_seed(
            checkpoint_root=missing_root,
            require_complete_checkpoint=True,
            **kwargs,
        )
    assert not missing_root.exists()

    partial_root = tmp_path / "partial"
    with pytest.raises(PlannedCheckpointInterruption):
        train_one_fold_seed(
            checkpoint_root=partial_root,
            test_interrupt_after_epoch=0,
            **kwargs,
        )
    with pytest.raises(ContractError, match="refuses a partial"):
        train_one_fold_seed(
            checkpoint_root=partial_root,
            require_complete_checkpoint=True,
            **kwargs,
        )

    complete_root = tmp_path / "complete"
    trained = train_one_fold_seed(checkpoint_root=complete_root, **kwargs)
    loaded = train_one_fold_seed(
        checkpoint_root=complete_root,
        require_complete_checkpoint=True,
        **kwargs,
    )
    assert loaded.summary["resumed_complete_checkpoint"] is True
    assert all(
        torch.equal(trained.model.state_dict()[name], loaded.model.state_dict()[name])
        for name in trained.model.state_dict()
    )


def test_run_fails_if_internal_code_hash_changes_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hva_affect.emotiontalk_causal_backbone_runner as runner

    corpus = synthetic_corpus()
    observed = iter(({"runner.py": "1" * 64}, {"runner.py": "2" * 64}))
    monkeypatch.setattr(runner, "_internal_code_hashes", lambda: next(observed))
    repository = tmp_path / "repo"
    repository.mkdir()
    public = repository / "report.json"
    with pytest.raises(ContractError, match="code changed"):
        execute_crossfit_backbone(
            corpus,
            provenance=synthetic_provenance(corpus),
            model_config=tiny_model_config(),
            run_config=tiny_run_config(),
            sampling_config=UtilitySamplingConfig(1, 4, 20260808, True),
            seeds=(17,),
            private_output_dir=tmp_path / "private",
            public_output_path=public,
            repository_root=repository,
            device=torch.device("cpu"),
        )
    assert not public.exists()


def test_classification_metrics_follow_dataset_label_order_width() -> None:
    labels = np.asarray([0, 1, 2], dtype=np.int64)
    probability = np.eye(3, dtype=np.float64)
    metrics = classification_metrics(labels, probability)
    assert metrics["macro_f1"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_emotiontalk_cli_exposes_only_manifest_sidecar_inputs() -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_emotiontalk_causal_backbone.py"),
        "--help",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    help_text = completed.stdout
    options = set(re.findall(r"--[a-z][a-z0-9-]*", help_text))
    assert "--sidecar-dir" in help_text
    assert "--sidecar-manifest" in help_text
    for forbidden in (
        "--data-dir",
        "--features",
        "--role-sidecar",
        "--allow-single-pickle-fallback",
        "--overwrite-public",
        "--dev",
        "--test",
        "--calibration",
        "--holdout",
    ):
        assert forbidden not in options


def test_published_runner_configuration_matches_feature_dimensions() -> None:
    payload = json.loads(
        (ROOT / "configs" / "carma_causal_backbone_v1.json").read_text(encoding="utf-8")
    )
    model = CausalBackboneConfig.from_mapping(payload)
    runner = BackboneRunConfig.from_mapping(payload)
    assert (model.text_dim, model.audio_dim, model.video_dim) == (256, 1536, 768)
    assert runner.outer_folds == 5
    assert runner.max_cuda_memory_mib < 8192
    assert CausalMultimodalBackbone(model).parameter_count() < 2_000_000
    assert payload["status"] == "frozen_open_role_production_contract_not_performance_evidence"
    assert payload["runtime_contract"]["sealed_test_labels_must_remain_unopened"] is True
    assert "sealed_test_labels_required" not in payload["runtime_contract"]


def test_nll_uses_shared_one_e_minus_twelve_probability_floor() -> None:
    labels = np.asarray([0, 1], dtype=np.int64)
    probability = np.asarray(
        [
            [1.0e-30, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
            [0.5, 1.0e-20, 0.1, 0.1, 0.1, 0.1, 0.1],
        ],
        dtype=np.float64,
    )
    probability /= probability.sum(axis=1, keepdims=True)
    expected = float(true_class_loss(labels, probability).mean())
    assert _nll(labels, probability) == pytest.approx(expected, rel=0.0, abs=0.0)
    assert expected == pytest.approx(-np.log(1.0e-12), rel=0.0, abs=1.0e-12)
