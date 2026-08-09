from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch", reason="MELD causal loader uses backbone config")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.causal_multimodal_backbone import (  # noqa: E402
    CausalBackboneConfig,
    CausalMultimodalBackbone,
)
from hva_affect.data_contract import ContractError  # noqa: E402
from hva_affect.emotiontalk_causal_backbone_runner import (  # noqa: E402
    FIT_ROLE,
    FROZEN_ROLE_RANGES,
    SELECTION_ROLE,
    BackboneRunConfig,
    UtilitySamplingConfig,
    execute_crossfit_backbone,
)
from hva_affect.meld_causal_backbone_loader import (  # noqa: E402
    load_meld_open_role_corpus,
    preflight_meld_causal_backbone_inputs,
)
from hva_affect.meld_multimodal_sidecar import (  # noqa: E402
    EMOTION_TO_INDEX,
    MANIFEST_SCHEMA_VERSION,
    PROTOCOL,
    ROLE_NAMES,
    SIDECAR_SCHEMA_VERSION,
)
from hva_affect.scu_set import assign_group_role  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config(audio_dim: int = 4, video_dim: int = 6) -> CausalBackboneConfig:
    return CausalBackboneConfig(
        text_dim=8,
        audio_dim=audio_dim,
        video_dim=video_dim,
        d_model=16,
        num_heads=4,
        num_layers=1,
        ffn_dim=24,
        num_speakers=8,
        max_turns=32,
        max_relative_turn=8,
        num_classes=7,
        dropout=0.0,
    )


def run_config() -> BackboneRunConfig:
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


def dialogues_for_role(role: str, count: int) -> list[int]:
    result: list[int] = []
    for dialogue in range(100_000):
        observed, _ = assign_group_role(
            "MELD", dialogue, "scu_set_exploration_v1", FROZEN_ROLE_RANGES
        )
        if observed == role:
            result.append(dialogue)
        if len(result) == count:
            return result
    raise AssertionError("could not find enough synthetic MELD dialogues")


def write_pair(
    root: Path,
    role: str,
    dialogues: list[int],
    *,
    speaker_code: int,
    protocol_offset: int,
    audio_dim: int = 4,
    video_dim: int = 6,
) -> dict[str, object]:
    utterances: list[str] = []
    dialogue_codes: list[int] = []
    speakers: list[int] = []
    orders: list[int] = []
    labels: list[int] = []
    for dialogue in dialogues:
        for order in range(4):
            utterances.append(
                f"dialogue {dialogue} turn {order} neutral joy sadness anger"
            )
            dialogue_codes.append(dialogue)
            speakers.append(speaker_code)
            orders.append(order)
            labels.append((dialogue + order) % len(EMOTION_TO_INDEX))
    rows = len(utterances)
    alignment = hashlib.sha256(
        f"{role}:{dialogues}:{speaker_code}".encode("utf-8")
    ).hexdigest()
    feature = root / f"features_{role}.npz"
    label = root / f"labels_{role}.npz"
    np.savez(
        feature,
        schema_version=np.asarray([SIDECAR_SCHEMA_VERSION]),
        role=np.asarray([role]),
        row_alignment_sha256=np.asarray([alignment]),
        utterances=np.asarray(utterances),
        audio_mean_std=np.arange(rows * audio_dim, dtype=np.float32).reshape(
            rows, audio_dim
        ),
        video_mean_std=np.arange(rows * video_dim, dtype=np.float32).reshape(
            rows, video_dim
        ),
        dialogue_codes=np.asarray(dialogue_codes, dtype=np.int64),
        speaker_codes=np.asarray(speakers, dtype=np.int64),
        utterance_order=np.asarray(orders, dtype=np.int64),
        protocol_row_ids=np.arange(protocol_offset, protocol_offset + rows, dtype=np.int64),
    )
    np.savez(
        label,
        schema_version=np.asarray([SIDECAR_SCHEMA_VERSION]),
        role=np.asarray([role]),
        row_alignment_sha256=np.asarray([alignment]),
        labels=np.asarray(labels, dtype=np.int64),
    )
    return {
        "feature_filename": feature.name,
        "label_filename": label.name,
        "rows": rows,
        "dialogues": len(dialogues),
        "history_eligible_rows": 3 * len(dialogues),
        "audio_dimension": audio_dim,
        "video_dimension": video_dim,
        "feature_sha256": sha(feature),
        "label_sha256": sha(label),
        "row_alignment_sha256": alignment,
    }


def sealed_record(role: str, audio_dim: int, video_dim: int) -> dict[str, object]:
    return {
        "feature_filename": f"features_{role}.npz",
        "label_filename": f"labels_{role}.npz",
        "rows": 1,
        "dialogues": 1,
        "history_eligible_rows": 0,
        "audio_dimension": audio_dim,
        "video_dimension": video_dim,
        "feature_sha256": "c" * 64,
        "label_sha256": "d" * 64,
        "row_alignment_sha256": "e" * 64,
    }


def make_fixture(
    root: Path,
    *,
    fit_dialogue_count: int = 1,
    selection_dialogue_count: int = 1,
    selection_speaker: int = 99,
    audio_dim: int = 4,
    video_dim: int = 6,
) -> tuple[Path, Path, dict]:
    root.mkdir(parents=True, exist_ok=True)
    sidecars = root / "sidecars_v2"
    sidecars.mkdir()
    roles = {
        FIT_ROLE: write_pair(
            sidecars,
            FIT_ROLE,
            dialogues_for_role(FIT_ROLE, fit_dialogue_count),
            speaker_code=7,
            protocol_offset=0,
            audio_dim=audio_dim,
            video_dim=video_dim,
        ),
        SELECTION_ROLE: write_pair(
            sidecars,
            SELECTION_ROLE,
            dialogues_for_role(SELECTION_ROLE, selection_dialogue_count),
            speaker_code=selection_speaker,
            protocol_offset=4 * fit_dialogue_count,
            audio_dim=audio_dim,
            video_dim=video_dim,
        ),
        "calibration": sealed_record("calibration", audio_dim, video_dim),
        "internal_holdout": sealed_record("internal_holdout", audio_dim, video_dim),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "role_separated_train_sidecars_created_and_hashed",
        "dataset_id": "MELD",
        "split_protocol_id": "scu_set_exploration_v1",
        "label_order": list(EMOTION_TO_INDEX),
        "claim_boundary": "synthetic strict MELD train-only sidecar fixture",
        "source_contract": {
            "train_csv_sha256": "a" * 64,
            "train_pickle_sha256": "b" * 64,
            "official_csv_is_authoritative_label_source": True,
            "embedded_pickle_label_used_for_training_or_metrics": False,
            "embedded_pickle_label_consistency_checked_by_trusted_custodian": True,
            "embedded_pickle_label_mismatch_statistics_exposed": False,
            "missing_feature_rows": 0,
            "extra_feature_rows": 0,
        },
        "feature_contract": {
            "audio_mean_std_columns": audio_dim,
            "video_mean_std_columns": video_dim,
            "numeric_dtype": "float32",
            "strict_same_dialogue_same_speaker_past_history_supported": True,
            "protocol_row_identity": "zero_based_official_train_csv_row_index",
        },
        "seal_contract": {
            "features_and_labels_are_in_separate_archives": True,
            "each_role_has_a_separate_label_archive": True,
            "allow_pickle_required_to_load_sidecars": False,
            "open_role_runner_may_load_only": [FIT_ROLE, SELECTION_ROLE],
            "calibration_and_internal_holdout_remain_unopened_by_model_runners": True,
        },
        "roles": roles,
        "config_sha256": "f" * 64,
        "public_content_audit": {
            "contains_labels_or_class_counts": False,
            "contains_utterances_or_embeddings": False,
            "contains_dialogue_speaker_or_row_identifiers": False,
            "contains_private_absolute_paths": False,
        },
    }
    manifest_path = root / "manifest_v2.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return sidecars, manifest_path, manifest


def test_meld_loader_requires_manifest_and_opens_only_four_open_role_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sidecars, manifest_path, _, = make_fixture(tmp_path)
    # Files may exist for sealed roles, but the model loader must not open them.
    np.savez(sidecars / "features_calibration.npz", forbidden=np.asarray([1]))
    np.savez(sidecars / "labels_calibration.npz", forbidden=np.asarray([1]))
    real_load = np.load
    opened: list[str] = []

    def audited_load(path, *args, **kwargs):
        opened.append(Path(path).name)
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", audited_load)
    corpus, provenance = load_meld_open_role_corpus(
        sidecar_dir=sidecars,
        manifest_path=manifest_path,
        model_config=config(),
    )
    expected = {
        f"features_{FIT_ROLE}.npz",
        f"labels_{FIT_ROLE}.npz",
        f"features_{SELECTION_ROLE}.npz",
        f"labels_{SELECTION_ROLE}.npz",
    }
    assert set(opened) == expected
    assert all(opened.count(name) == 1 for name in expected)
    assert len(corpus.keys) == 8
    assert set(corpus.roles.astype(str)) == {FIT_ROLE, SELECTION_ROLE}
    assert sorted(map(len, corpus.histories)) == [0, 0, 1, 1, 2, 2, 3, 3]
    fit = corpus.role_indices(FIT_ROLE)
    selection = corpus.role_indices(SELECTION_ROLE)
    assert set(corpus.speaker_ids[fit]) == {1}
    assert set(corpus.speaker_ids[selection]) == {0}
    assert provenance.manifest_schema == MANIFEST_SCHEMA_VERSION
    assert provenance.label_order == tuple(EMOTION_TO_INDEX)
    assert provenance.audio_dim == 4 and provenance.video_dim == 6
    assert provenance.source_hashes["sidecar_manifest"] == sha(manifest_path)
    assert provenance.source_hashes["trusted_source_train_csv"] == "a" * 64
    assert corpus.protocol_row_ids.tolist() == list(range(8))


def test_meld_loader_signature_has_no_arbitrary_role_or_sealed_paths() -> None:
    import inspect

    assert set(inspect.signature(load_meld_open_role_corpus).parameters) == {
        "sidecar_dir",
        "manifest_path",
        "model_config",
    }


@pytest.mark.parametrize("kind", ["feature", "label"])
def test_meld_loader_rejects_copied_renamed_role_file(
    tmp_path: Path, kind: str
) -> None:
    sidecars, manifest_path, manifest = make_fixture(tmp_path)
    prefix = "features" if kind == "feature" else "labels"
    source = sidecars / f"{prefix}_{FIT_ROLE}.npz"
    target = sidecars / f"{prefix}_{SELECTION_ROLE}.npz"
    target.write_bytes(source.read_bytes())
    field = "feature_sha256" if kind == "feature" else "label_sha256"
    manifest["roles"][SELECTION_ROLE][field] = sha(target)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ContractError, match="role does not match"):
        load_meld_open_role_corpus(
            sidecar_dir=sidecars,
            manifest_path=manifest_path,
            model_config=config(),
        )


def test_meld_loader_rejects_real_file_hash_or_sealed_label_stat_drift(
    tmp_path: Path,
) -> None:
    sidecars, manifest_path, manifest = make_fixture(tmp_path)
    feature = sidecars / f"features_{FIT_ROLE}.npz"
    feature.write_bytes(feature.read_bytes() + b"changed")
    with pytest.raises(ContractError, match="hash differs"):
        load_meld_open_role_corpus(
            sidecar_dir=sidecars,
            manifest_path=manifest_path,
            model_config=config(),
        )
    # Restore a clean fixture and prove that sealed-role mismatch statistics
    # are forbidden in the model-facing manifest.
    other = tmp_path / "other"
    sidecars, manifest_path, manifest = make_fixture(other)
    manifest["source_contract"]["embedded_pickle_label_mismatch_statistics_exposed"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ContractError, match="mismatch statistics"):
        load_meld_open_role_corpus(
            sidecar_dir=sidecars,
            manifest_path=manifest_path,
            model_config=config(),
        )


def test_selection_speakers_do_not_change_fit_only_mapping(tmp_path: Path) -> None:
    first_dir, first_manifest, _ = make_fixture(tmp_path / "first", selection_speaker=99)
    second_dir, second_manifest, _ = make_fixture(tmp_path / "second", selection_speaker=123)
    first, first_provenance = load_meld_open_role_corpus(
        sidecar_dir=first_dir, manifest_path=first_manifest, model_config=config()
    )
    second, second_provenance = load_meld_open_role_corpus(
        sidecar_dir=second_dir, manifest_path=second_manifest, model_config=config()
    )
    assert first_provenance.speaker_mapping_sha256 == second_provenance.speaker_mapping_sha256
    np.testing.assert_array_equal(
        first.speaker_ids[first.role_indices(FIT_ROLE)],
        second.speaker_ids[second.role_indices(FIT_ROLE)],
    )
    assert set(first.speaker_ids[first.role_indices(SELECTION_ROLE)]) == {0}
    assert set(second.speaker_ids[second.role_indices(SELECTION_ROLE)]) == {0}


def test_meld_executor_report_binds_label_order_manifest_dimensions_and_history_counts(
    tmp_path: Path,
) -> None:
    sidecars, manifest_path, _ = make_fixture(
        tmp_path / "data", fit_dialogue_count=4, selection_dialogue_count=2
    )
    corpus, provenance = load_meld_open_role_corpus(
        sidecar_dir=sidecars, manifest_path=manifest_path, model_config=config()
    )
    repository = tmp_path / "repo"
    repository.mkdir()
    report = execute_crossfit_backbone(
        corpus,
        provenance=provenance,
        model_config=config(),
        run_config=run_config(),
        sampling_config=UtilitySamplingConfig(1, 4, 20260808, True),
        seeds=(17,),
        private_output_dir=tmp_path / "private",
        public_output_path=repository / "meld_report.json",
        repository_root=repository,
        device=torch.device("cpu"),
        private_cache_filename="meld_test_oof.npz",
    )
    assert report["dataset"] == "MELD"
    assert report["split"] == "official_train_open_roles_only"
    assert report["performance_claim_gate"]["authorized"] is False
    assert report["dataset_label_order"] == list(EMOTION_TO_INDEX)
    assert report["verified_manifest"]["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert report["verified_manifest"]["sha256"] == sha(manifest_path)
    assert report["feature_contract"]["actual_dimensions"] == {
        "text": 8,
        "audio": 4,
        "video": 6,
    }
    assert report["rows_and_groups"]["fit_history_eligible_rows"] == 12
    assert report["rows_and_groups"]["model_selection_history_eligible_rows"] == 6
    rendered = json.dumps(report)
    assert "embedded_pickle_label_mismatch_count" not in rendered
    assert report["probability_protocol"]["current_only_semantics"].endswith(
        "not_independently_trained_baseline"
    )
    assert report["artifact_role"].startswith("probability_and_utility_producer_only")


def test_nonperformance_preflight_never_deserializes_selection_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sidecars, manifest_path, _ = make_fixture(
        tmp_path, fit_dialogue_count=4, selection_dialogue_count=2
    )
    real_load = np.load
    opened: list[str] = []

    def audited_load(path, *args, **kwargs):
        opened.append(Path(path).name)
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", audited_load)
    report = preflight_meld_causal_backbone_inputs(
        sidecar_dir=sidecars,
        manifest_path=manifest_path,
        model_config=config(),
        run_config=run_config(),
        sampling_config=UtilitySamplingConfig(1, 4, 20260808, True),
    )
    assert f"labels_{FIT_ROLE}.npz" in opened
    assert f"labels_{SELECTION_ROLE}.npz" not in opened
    assert f"features_{SELECTION_ROLE}.npz" in opened
    assert not any("calibration" in name or "holdout" in name for name in opened)
    assert report["data_access"]["selection_labels_deserialized"] is False
    assert report["forward_contract"]["metrics_or_utilities_computed"] is False
    assert report["performance_claim_authorized"] is False


def test_meld_production_config_and_cli_contract() -> None:
    payload = json.loads(
        (ROOT / "configs" / "carma_causal_backbone_meld_v1.json").read_text(
            encoding="utf-8"
        )
    )
    production = CausalBackboneConfig.from_mapping(payload)
    assert (production.text_dim, production.audio_dim, production.video_dim) == (
        256,
        64,
        4096,
    )
    assert CausalMultimodalBackbone(production).parameter_count() < 2_000_000
    assert payload["status"] == "frozen_open_role_production_contract_not_performance_evidence"
    assert payload["runtime_contract"]["sealed_test_labels_must_remain_unopened"] is True
    path = ROOT / "scripts" / "run_meld_causal_backbone.py"
    spec = importlib.util.spec_from_file_location("meld_causal_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    destinations = {action.dest for action in module.build_parser()._actions}
    assert destinations == {
        "help",
        "sidecar_dir",
        "sidecar_manifest",
        "backbone_config",
        "utility_config",
        "confirmatory_config",
        "private_output_dir",
        "public_output",
        "device",
    }
    assert not destinations & {
        "fit_feature",
        "fit_label",
        "selection_feature",
        "selection_label",
        "dev",
        "test",
        "calibration",
        "holdout",
    }
    preflight_path = ROOT / "scripts" / "preflight_meld_causal_backbone.py"
    preflight_spec = importlib.util.spec_from_file_location(
        "meld_preflight_cli", preflight_path
    )
    preflight_module = importlib.util.module_from_spec(preflight_spec)
    assert preflight_spec.loader is not None
    preflight_spec.loader.exec_module(preflight_module)
    preflight_destinations = {
        action.dest for action in preflight_module.build_parser()._actions
    }
    assert preflight_destinations == {
        "help",
        "sidecar_dir",
        "sidecar_manifest",
        "backbone_config",
        "utility_config",
    }
