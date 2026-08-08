from __future__ import annotations

import hashlib
import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.bidirectional_emotion_utility import (  # noqa: E402
    BidirectionalCoalitionTask,
)
from hva_affect.causal_multimodal_backbone import CausalBackboneConfig  # noqa: E402
from hva_affect.emotion_probability_relations import (  # noqa: E402
    HISTORY_CONTEXTS,
    MODALITIES,
    EmotionProbabilityBlock,
    TrainOnlyProvenance,
    bidirectional_task_order_sha256,
    dataset_identity_sha256,
    emotion_class_order_sha256,
    emotion_context_schema_sha256,
    ordered_source_sha256,
    numeric_matrix_content_sha256,
)
from hva_affect.emotion_relation_vad_repair import (  # noqa: E402
    BASE_SEEDS,
    BASE_59D_PARAMETER_COUNT,
    CAPACITY_CONTROL_EXPANSION,
    CAPACITY_CONTROL_MATRIX_SHA256,
    CAPACITY_CONTROL_SPEC,
    CAPACITY_CONTROL_SPEC_SHA256,
    CAPACITY_CONTROL_VARIANT,
    CLASS_ORDER,
    FIT_GATE_MACRO_F1_GAIN,
    FIT_GATE_REQUIRED_SEEDS,
    FIT_GATE_STAGE_ACCESS_CONTRACT,
    FIT_INTERNAL_GATE_NAMESPACE,
    FIT_INTERNAL_GATE_SPLIT_SPEC_SHA256,
    PRIMARY_VARIANT,
    PRIMARY_OR_CONTROL_PARAMETER_COUNT,
    REGISTERED_MODEL_CONFIG_SHA256,
    REGISTERED_OUTPUT_FILENAME,
    REGISTERED_OUTPUT_PATH,
    REGISTERED_OUTPUT_REPOSITORY_RELATIVE_PATH,
    REGISTERED_SIDECAR_MANIFEST_SHA256,
    VAD_COORDINATES,
    VAD_COORDINATE_SHA256,
    VARIANT_ORDER,
    VARIANT_WIDTHS,
    EmotionRelationVADRepairError,
    ProjectorSpec,
    RolePosteriorGrid,
    _deterministic_base_cache_payload,
    _fit_gate_variant_seed_scores,
    _producer_config_sha256,
    _selection_variant_scores,
    build_fit_internal_gate_corpus,
    build_role_feature_set,
    build_vad_state_transition_features,
    capacity_control_expansion_matrix,
    deterministic_fit_internal_gate_split,
    fit_fit_role_group_oof_posteriors,
    fit_full_fit_predict_selection_posteriors,
    fit_gate_train_predict_gate_eval_posteriors,
    fit_only_gate_decision,
    load_emotiontalk_fit_gate_stage,
    load_emotion_relation_vad_repair_config,
    materialize_selection_labels_after_fit_gate,
    run_emotion_relation_vad_repair,
    vad_coordinate_sha256,
)
from hva_affect.bidirectional_utility_model import trainable_parameter_count  # noqa: E402
from hva_affect.emotiontalk_causal_backbone_runner import (  # noqa: E402
    FIT_ROLE,
    SELECTION_ROLE,
    OpenRoleCorpus,
    _corpus_contract_sha256,
    _role_assignment_sha256,
    create_verified_corpus_provenance,
)
import hva_affect.emotion_relation_vad_repair as repair3  # noqa: E402
from hva_affect.emotiontalk_role_sidecar import (  # noqa: E402
    FROZEN_ROLE_RANGES,
    OPEN_ROLES,
    PROTOCOL as ROLE_SIDECAR_PROTOCOL,
    prepare_emotiontalk_role_sidecars,
)
from hva_affect.scu_set import assign_group_role  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "emotion_relation_vad_repair_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _group_for_role(role: str) -> tuple[str, str]:
    for value in range(1, 100_000):
        group = f"G{value:05d}"
        dialogue = "01"
        observed, _ = assign_group_role(
            "EmotionTalk",
            f"{group}/{dialogue}",
            "scu_set_exploration_v1",
            FROZEN_ROLE_RANGES,
        )
        if observed == role:
            return group, dialogue
    raise AssertionError("no synthetic group found for role")


def _physical_sidecar_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, CausalBackboneConfig]:
    keys: list[str] = []
    labels: dict[str, dict[str, int]] = {}
    for role in OPEN_ROLES:
        group, dialogue = _group_for_role(role)
        speaker = "01" if role == FIT_ROLE else "99"
        for turn in range(3):
            key = f"{group}_{dialogue}_{speaker}_{turn:03d}"
            keys.append(key)
            labels[key] = {"emo": turn, "val": 0}
    label_archive = tmp_path / "labels_source.npz"
    np.savez(label_archive, train_corpus=np.asarray(labels, dtype=object))
    media = tmp_path / "media_source.npz"
    np.savez(
        media,
        keys=np.asarray(keys),
        splits=np.asarray(["train_corpus"] * len(keys)),
        audio_features=np.arange(len(keys) * 2, dtype=np.float32).reshape(len(keys), 2),
        video_features=np.arange(len(keys) * 3, dtype=np.float32).reshape(len(keys), 3),
        quality=np.zeros((len(keys), 1), dtype=np.float32),
        quality_names=np.asarray(["q"]),
        config_sha256=np.asarray("d" * 64),
    )
    transcription = tmp_path / "transcription.csv"
    transcription.write_text(
        "name,chinese\n"
        + "".join(f"{key}.wav,synthetic-{index}\n" for index, key in enumerate(keys)),
        encoding="utf-8",
    )
    config = tmp_path / "sidecar_config.json"
    config.write_text(
        json.dumps(
            {
                "protocol": ROLE_SIDECAR_PROTOCOL,
                "status": "frozen_before_trusted_generation",
                "dataset_id": "EmotionTalk",
                "split_protocol_id": "scu_set_exploration_v1",
                "roles": FROZEN_ROLE_RANGES,
                "source_sha256": {
                    "label_archive": _sha(label_archive),
                    "media_features": _sha(media),
                    "transcription": _sha(transcription),
                },
            }
        ),
        encoding="utf-8",
    )
    repository = tmp_path / "repository"
    repository.mkdir()
    private = tmp_path / "private" / "emotiontalk"
    manifest_path = repository / "manifest.json"
    prepare_emotiontalk_role_sidecars(
        label_archive_path=label_archive,
        feature_path=media,
        transcription_path=transcription,
        config_path=config,
        private_output_dir=private,
        public_manifest_path=manifest_path,
        repository_root=repository,
    )
    model_config = CausalBackboneConfig(
        text_dim=8,
        audio_dim=2,
        video_dim=3,
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
    return private, manifest_path, model_config


def _replace_selection_payloads_with_corrupt_bytes(
    private: Path,
    manifest_path: Path,
    *,
    synchronize_manifest_hash: bool,
) -> tuple[Path, Path]:
    selection_feature = private / f"features_{SELECTION_ROLE}.npz"
    selection_label = private / f"labels_{SELECTION_ROLE}.npz"
    selection_feature.write_bytes(b"not-an-npz-selection-feature-payload")
    selection_label.write_bytes(b"not-an-npz-selection-label-payload")
    if synchronize_manifest_hash:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["roles"][SELECTION_ROLE]["feature_sha256"] = _sha(selection_feature)
        manifest["roles"][SELECTION_ROLE]["label_sha256"] = _sha(selection_label)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return selection_feature, selection_label


def _synthetic_corpus(
    *, groups_per_role: int = 5, rows_per_group: int = 7
) -> tuple[OpenRoleCorpus, CausalBackboneConfig, object]:
    rows = 2 * groups_per_role * rows_per_group
    keys: list[str] = []
    texts: list[str] = []
    labels: list[int] = []
    groups: list[str] = []
    roles: list[str] = []
    buckets: list[int] = []
    turns: list[int] = []
    histories: list[tuple[int, ...]] = []
    speaker_identity: list[str] = []
    audio = np.empty((rows, 3), dtype=np.float32)
    video = np.empty((rows, 2), dtype=np.float32)
    row = 0
    for role_index, role in enumerate((FIT_ROLE, SELECTION_ROLE)):
        for group_index in range(groups_per_role):
            group = f"role-{role_index}-group-{group_index}"
            group_start = row
            for turn in range(rows_per_group):
                label = turn % len(CLASS_ORDER)
                keys.append(hashlib.sha256(f"key-{row}".encode()).hexdigest())
                texts.append(
                    f"emotiontoken{label} stablephrase classword{label} group{group_index}"
                )
                labels.append(label)
                groups.append(group)
                roles.append(role)
                buckets.append(group_index if role == FIT_ROLE else 65 + group_index)
                turns.append(turn)
                histories.append(tuple(range(group_start, row)))
                speaker_identity.append(hashlib.sha256(f"speaker-{group}".encode()).hexdigest())
                audio[row] = (label, group_index / 10.0, turn / 10.0)
                video[row] = (label / 10.0, (6 - label) / 10.0)
                row += 1
    speaker_mapping_hash = hashlib.sha256(b"synthetic-speaker-map").hexdigest()
    corpus = OpenRoleCorpus(
        keys=np.asarray(keys),
        texts=tuple(texts),
        audio=audio,
        video=video,
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups),
        roles=np.asarray(roles),
        buckets=np.asarray(buckets, dtype=np.int16),
        speaker_ids=np.ones(rows, dtype=np.int64),
        turn_ids=np.asarray(turns, dtype=np.int64),
        histories=tuple(histories),
        protocol_row_ids=np.arange(rows, dtype=np.int64),
        speaker_identity=np.asarray(speaker_identity),
        speaker_mapping_sha256=speaker_mapping_hash,
        label_access_mode="synthetic_contract_test",
    )
    model_config = CausalBackboneConfig(
        audio_dim=3,
        video_dim=2,
        num_speakers=4,
        max_turns=64,
        num_classes=7,
    )
    corpus.validate(model_config)
    manifest_hash = hashlib.sha256(b"synthetic-manifest").hexdigest()
    provenance = create_verified_corpus_provenance(
        dataset_id="EmotionTalk-synthetic",
        manifest_schema="emotiontalk_role_separated_sidecars_v2",
        manifest_status="synthetic_contract_test",
        manifest_sha256=manifest_hash,
        source_hashes={
            "sidecar_manifest": manifest_hash,
            "fit_features": hashlib.sha256(b"fit-features").hexdigest(),
            "fit_labels": hashlib.sha256(b"fit-labels").hexdigest(),
            "model_selection_features": hashlib.sha256(b"selection-features").hexdigest(),
            "model_selection_labels": hashlib.sha256(b"selection-labels").hexdigest(),
        },
        label_order=CLASS_ORDER,
        role_rows={
            FIT_ROLE: groups_per_role * rows_per_group,
            SELECTION_ROLE: groups_per_role * rows_per_group,
        },
        audio_dim=3,
        video_dim=2,
        role_assignment_sha256=_role_assignment_sha256(corpus),
        speaker_mapping_sha256=speaker_mapping_hash,
        corpus_contract_sha256=_corpus_contract_sha256(corpus),
        verification_origin="synthetic_contract_test",
    )
    provenance.validate(corpus, model_config)
    return corpus, model_config, provenance


def _posterior(corpus: OpenRoleCorpus, role: str) -> RolePosteriorGrid:
    rows = corpus.role_indices(role)
    probability: dict[str, np.ndarray] = {}
    for modality_index, modality in enumerate(MODALITIES):
        values = np.full(
            (len(BASE_SEEDS), len(rows), len(CLASS_ORDER)),
            0.04,
            dtype=np.float64,
        )
        for seed_index in range(len(BASE_SEEDS)):
            for local, corpus_row in enumerate(rows):
                peak = (int(corpus_row) + modality_index + seed_index) % len(CLASS_ORDER)
                values[seed_index, local, peak] = 0.76
        probability[modality] = values
    folds = (
        np.arange(len(rows), dtype=np.int32) % 5
        if role == FIT_ROLE
        else np.full(len(rows), -1, dtype=np.int32)
    )
    return RolePosteriorGrid(
        role=role,
        row_indices=rows,
        probabilities=probability,
        base_seeds=BASE_SEEDS,
        fold_by_local_row=folds,
        fold_assignment_sha256=hashlib.sha256(f"fold-{role}".encode()).hexdigest(),
        producer_config_sha256=_producer_config_sha256(ProjectorSpec()),
    )


def _tasks(corpus: OpenRoleCorpus, role: str) -> tuple[BidirectionalCoalitionTask, ...]:
    role_rows = corpus.role_indices(role)
    result: list[BidirectionalCoalitionTask] = []
    for group in np.unique(corpus.groups[role_rows]):
        members = role_rows[corpus.groups[role_rows] == group]
        result.append(
            BidirectionalCoalitionTask(
                query_index=int(members[4]),
                addition_context=(int(members[0]),),
                deletion_context=(int(members[1]), int(members[2])),
                candidate_index=int(members[2]),
            )
        )
    return tuple(result)


def _permuted_corpus(corpus: OpenRoleCorpus, permutation: np.ndarray) -> OpenRoleCorpus:
    permutation = np.asarray(permutation, dtype=np.int64)
    old_to_new = {int(old): new for new, old in enumerate(permutation.tolist())}
    histories = tuple(
        tuple(old_to_new[int(candidate)] for candidate in corpus.histories[int(old)])
        for old in permutation
    )
    protocol_rows = np.asarray(corpus.protocol_row_ids, dtype=np.int64)
    speaker_identity = np.asarray(corpus.speaker_identity, dtype=str)
    return OpenRoleCorpus(
        keys=np.asarray(corpus.keys[permutation]).copy(),
        texts=tuple(corpus.texts[int(row)] for row in permutation),
        audio=np.asarray(corpus.audio[permutation]).copy(),
        video=np.asarray(corpus.video[permutation]).copy(),
        labels=np.asarray(corpus.labels[permutation]).copy(),
        groups=np.asarray(corpus.groups[permutation]).copy(),
        roles=np.asarray(corpus.roles[permutation]).copy(),
        buckets=np.asarray(corpus.buckets[permutation]).copy(),
        speaker_ids=np.asarray(corpus.speaker_ids[permutation]).copy(),
        turn_ids=np.asarray(corpus.turn_ids[permutation]).copy(),
        histories=histories,
        protocol_row_ids=np.asarray(protocol_rows[permutation]).copy(),
        speaker_identity=np.asarray(speaker_identity[permutation]).copy(),
        speaker_mapping_sha256=corpus.speaker_mapping_sha256,
        label_access_mode=corpus.label_access_mode,
    )


def _relation_provenance(rows: int = 2) -> TrainOnlyProvenance:
    dataset = "EmotionTalk/synthetic@fixed"
    source_hash = ordered_source_sha256(
        dataset, tuple(f"source-{index}" for index in range(rows + 3))
    )
    split_hash = hashlib.sha256(b"split").hexdigest()
    fold_hash = hashlib.sha256(b"fold").hexdigest()
    producer_hash = hashlib.sha256(b"producer").hexdigest()
    class_hash = emotion_class_order_sha256(CLASS_ORDER)
    context_hash = emotion_context_schema_sha256()
    tasks = (
        BidirectionalCoalitionTask(3, (0,), (1, 2), 2),
        BidirectionalCoalitionTask(4, (0,), (1, 3), 3),
    )
    task_hash = bidirectional_task_order_sha256(
        tasks,
        dataset=dataset,
        role=FIT_ROLE,
        source_order_sha256=source_hash,
        split_manifest_sha256=split_hash,
        fold_assignment_sha256=fold_hash,
        context_schema_sha256=context_hash,
        class_order_sha256=class_hash,
        producer_config_sha256=producer_hash,
    )
    return TrainOnlyProvenance(
        mode="train_fold_oof",
        dataset=dataset,
        role=FIT_ROLE,
        dataset_sha256=dataset_identity_sha256(dataset),
        source_order_sha256=source_hash,
        split_manifest_sha256=split_hash,
        fold_assignment_sha256=fold_hash,
        task_order_sha256=task_hash,
        context_schema_sha256=context_hash,
        class_order_sha256=class_hash,
        producer_config_sha256=producer_hash,
    )


def _block(probability: np.ndarray, provenance: TrainOnlyProvenance) -> EmotionProbabilityBlock:
    return EmotionProbabilityBlock(
        probabilities={modality: probability for modality in MODALITIES},
        provenance=provenance,
        class_order=CLASS_ORDER,
        modality_class_orders={modality: CLASS_ORDER for modality in MODALITIES},
    )


def test_frozen_config_registers_primary_vad_hash_and_fit_only_gate() -> None:
    frozen = load_emotion_relation_vad_repair_config(CONFIG_PATH)
    assert frozen.raw["primary_variant"] == PRIMARY_VARIANT
    assert tuple(frozen.raw["variant_order"]) == VARIANT_ORDER
    assert frozen.raw["variant_widths"] == dict(VARIANT_WIDTHS)
    assert frozen.raw["vad"]["coordinate_sha256"] == VAD_COORDINATE_SHA256
    assert vad_coordinate_sha256() == VAD_COORDINATE_SHA256
    gate = frozen.raw["fit_only_open_gate"]
    assert gate["minimum_macro_f1_gain_vs_each_reference"] == FIT_GATE_MACRO_F1_GAIN
    assert gate["accuracy_minimum_gain_vs_each_reference"] == 0.0
    assert gate["reference_variants"] == [
        "base_59d_only",
        CAPACITY_CONTROL_VARIANT,
    ]
    assert gate["minimum_successful_utility_seeds_out_of_five"] == FIT_GATE_REQUIRED_SEEDS
    assert gate["selection_prediction_if_fail"] is False
    assert gate["selection_label_scoring_if_fail"] is False
    assert frozen.projector.text_sublinear_tf is True
    assert frozen.sampling.draws_per_query == 8
    assert frozen.sampling.maximum_candidates == 16
    assert frozen.sampling.seed == 20260808
    assert frozen.sampling.match_context_cardinality is True
    assert frozen.raw["data_access_staging"] == dict(FIT_GATE_STAGE_ACCESS_CONTRACT)
    assert frozen.raw["registered_output_filename"] == REGISTERED_OUTPUT_FILENAME
    assert frozen.raw["registered_output_repository_relative_path"] == (
        REGISTERED_OUTPUT_REPOSITORY_RELATIVE_PATH
    )
    assert REGISTERED_OUTPUT_PATH == (
        ROOT / "artifacts" / REGISTERED_OUTPUT_FILENAME
    ).resolve()
    assert frozen.raw["registered_input_locks"] == {
        "sidecar_manifest_sha256": REGISTERED_SIDECAR_MANIFEST_SHA256,
        "model_config_sha256": REGISTERED_MODEL_CONFIG_SHA256,
        "enforced_before_sidecar_deserialization_or_model_training": True,
    }
    assert frozen.raw["optional_teacher"]["fit_oof_eligibility"] == {
        "minimum_successful_seeds_out_of_five": 4,
        "minimum_macro_f1_gain": FIT_GATE_MACRO_F1_GAIN,
        "nll_maximum_worsening": 0.0,
        "accuracy_minimum_gain": 0.0,
        "same_seed_macro_nll_accuracy_intersection": True,
    }
    selection_source = inspect.getsource(repair3.evaluate_model_selection)
    assert 'primary["pooled_accuracy"]' in selection_source
    assert 'current_mean["pooled_accuracy"]' in selection_source
    assert 'reference["pooled_accuracy"]' in selection_source


def test_fit_only_gate_is_same_seed_six_condition_intersection_four_of_five() -> None:
    passed, per_seed = fit_only_gate_decision(
        [0.002, 0.003, 0.004, 0.0021, 0.10],
        [0.0, -0.1, -0.01, 1e-13, 1e-6],
        [0.0, 0.1, 0.01, 0.0, 0.5],
        [0.003, 0.003, 0.004, 0.0021, 0.10],
        [0.0, -0.1, -0.01, 1e-13, -0.1],
        [0.0, 0.1, 0.01, 0.0, 0.5],
    )
    assert passed is True
    assert per_seed == (True, True, True, True, False)
    failed, failed_per_seed = fit_only_gate_decision(
        [0.003] * 5,
        [-0.1] * 5,
        [0.0] * 5,
        [0.003] * 5,
        [-0.1] * 5,
        [0.0, 0.0, 0.0, -1e-9, -1e-9],
    )
    assert failed is False
    assert failed_per_seed == (True, True, True, False, False)
    with pytest.raises(EmotionRelationVADRepairError):
        fit_only_gate_decision(
            [0.1] * 4,
            [0.0] * 5,
            [0.0] * 5,
            [0.1] * 5,
            [0.0] * 5,
            [0.0] * 5,
        )


def test_capacity_control_hash_rank_isometry_and_parameter_counts() -> None:
    canonical = json.dumps(
        dict(CAPACITY_CONTROL_SPEC),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    assert hashlib.sha256(canonical).hexdigest() == CAPACITY_CONTROL_SPEC_SHA256
    expansion = capacity_control_expansion_matrix()
    np.testing.assert_array_equal(expansion, CAPACITY_CONTROL_EXPANSION)
    assert expansion.shape == (59, 299)
    assert np.linalg.matrix_rank(expansion) == 59
    np.testing.assert_allclose(
        expansion @ expansion.T,
        np.eye(59),
        rtol=0.0,
        atol=5e-16,
    )
    assert numeric_matrix_content_sha256(expansion) == CAPACITY_CONTROL_MATRIX_SHA256
    assert trainable_parameter_count(299, (32, 16), 2) == (
        PRIMARY_OR_CONTROL_PARAMETER_COUNT
    )
    assert trainable_parameter_count(59, (32, 16), 2) == BASE_59D_PARAMETER_COUNT


def test_fit_internal_group_assignment_is_row_order_invariant_and_complete() -> None:
    corpus, model_config, _ = _synthetic_corpus(groups_per_role=10)
    first = deterministic_fit_internal_gate_split(corpus)
    permutation = np.random.default_rng(20260808).permutation(len(corpus.keys))
    permuted = _permuted_corpus(corpus, permutation)
    permuted.validate(model_config)
    second = deterministic_fit_internal_gate_split(permuted)
    assert first.namespace == FIT_INTERNAL_GATE_NAMESPACE
    assert first.split_spec_sha256 == FIT_INTERNAL_GATE_SPLIT_SPEC_SHA256
    assert first.train_groups == second.train_groups
    assert first.eval_groups == second.eval_groups
    assert first.group_assignment_sha256 == second.group_assignment_sha256
    assert first.row_assignment_sha256 == second.row_assignment_sha256
    assert not set(first.train_groups) & set(first.eval_groups)
    fit_groups = set(corpus.groups[corpus.role_indices(FIT_ROLE)].astype(str))
    assert set(first.train_groups) | set(first.eval_groups) == fit_groups
    gate_corpus = build_fit_internal_gate_corpus(
        corpus, first, model_config=model_config
    )
    for group in np.unique(gate_corpus.groups):
        assert len(set(gate_corpus.roles[gate_corpus.groups == group].astype(str))) == 1
    public = json.dumps(first.aggregate_attestation(), sort_keys=True)
    assert not any(group in public for group in fit_groups)
    assert not any(str(key) in public for key in corpus.keys)


@pytest.mark.parametrize("failure", ["missing_class", "no_history"])
def test_fit_internal_gate_is_no_go_when_partition_is_ineligible(failure: str) -> None:
    corpus, model_config, _ = _synthetic_corpus(groups_per_role=10)
    split = deterministic_fit_internal_gate_split(corpus)
    if failure == "missing_class":
        labels = np.asarray(corpus.labels).copy()
        labels[list(split.eval_rows)] = 0
        changed = replace(corpus, labels=labels)
    else:
        changed = replace(corpus, histories=tuple(tuple() for _ in corpus.histories))
    ineligible = deterministic_fit_internal_gate_split(changed)
    assert ineligible.eligible is False
    with pytest.raises(EmotionRelationVADRepairError, match="no_go"):
        build_fit_internal_gate_corpus(changed, ineligible, model_config=model_config)


def test_gate_eval_projector_fit_never_receives_gate_eval_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, model_config, _ = _synthetic_corpus(groups_per_role=10)
    split = deterministic_fit_internal_gate_split(corpus)
    gate_corpus = build_fit_internal_gate_corpus(
        corpus, split, model_config=model_config
    )
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_prepare(_corpus, _modality, train_rows, predict_rows, _spec):
        train = np.asarray(train_rows, dtype=np.int64)
        predict = np.asarray(predict_rows, dtype=np.int64)
        calls.append((train.copy(), predict.copy()))
        return np.zeros((len(train), 1)), np.zeros((len(predict), 1))

    class FakeProjector:
        classes_ = np.arange(len(CLASS_ORDER), dtype=np.int64)

        def fit(self, values, labels):
            assert len(values) == len(labels)
            assert set(np.asarray(labels).tolist()) == set(range(len(CLASS_ORDER)))
            return self

        def predict_proba(self, values):
            return np.full(
                (len(values), len(CLASS_ORDER)),
                1.0 / len(CLASS_ORDER),
                dtype=np.float64,
            )

    monkeypatch.setattr(repair3, "_prepare_modality_features", fake_prepare)
    monkeypatch.setattr(repair3, "_new_projector", lambda *_: FakeProjector())
    train_oof = fit_fit_role_group_oof_posteriors(gate_corpus, ProjectorSpec())
    calls.clear()
    eval_prediction = fit_gate_train_predict_gate_eval_posteriors(
        gate_corpus,
        ProjectorSpec(),
        gate_train_oof=train_oof,
    )
    assert eval_prediction.role == SELECTION_ROLE
    assert calls
    for train_rows, predict_rows in calls:
        assert set(gate_corpus.roles[train_rows].astype(str)) == {FIT_ROLE}
        assert set(gate_corpus.roles[predict_rows].astype(str)) == {SELECTION_ROLE}


def test_gate_eval_label_and_feature_perturbations_obey_training_firewall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, model_config, verified = _synthetic_corpus(groups_per_role=10)
    split = deterministic_fit_internal_gate_split(corpus)
    gate_corpus = build_fit_internal_gate_corpus(
        corpus, split, model_config=model_config
    )
    train_posterior = _posterior(gate_corpus, FIT_ROLE)
    eval_posterior = _posterior(gate_corpus, SELECTION_ROLE)
    train_tasks = _tasks(gate_corpus, FIT_ROLE)
    eval_tasks = _tasks(gate_corpus, SELECTION_ROLE)
    train_features = build_role_feature_set(
        gate_corpus,
        verified,
        train_posterior,
        train_tasks,
        outcome_labels_allowed=True,
    )
    eval_features = build_role_feature_set(
        gate_corpus,
        verified,
        eval_posterior,
        eval_tasks,
        outcome_labels_allowed=False,
    )
    seen_fit_groups: list[set[str]] = []

    def fake_oof(train_split, tasks, task_labels, *_args, **_kwargs):
        assert tasks == train_tasks
        assert np.array_equal(task_labels, train_features.task_labels)
        seen_fit_groups.append(
            set(gate_corpus.groups[[task.query_index for task in tasks]].astype(str))
        )
        return SimpleNamespace(
            predictions=SimpleNamespace(
                decision_score=np.linspace(-1.0, 1.0, len(tasks), dtype=np.float64)
            ),
            fold_by_row=np.arange(len(tasks), dtype=np.int32) % 5,
        )

    def fake_threshold(tasks, scores, *, target_coverage):
        assert tasks == train_tasks
        np.testing.assert_array_equal(
            scores, np.linspace(-1.0, 1.0, len(tasks), dtype=np.float64)
        )
        assert target_coverage == 0.25
        return 0.125, len(tasks), 0.25

    class FakeFitted:
        def __init__(self, spec, seed, width):
            self.spec = spec
            self.seed = seed
            self.x_scaler = SimpleNamespace(
                mean_=np.zeros(width), scale_=np.ones(width)
            )
            self.target_mean = np.zeros(2)
            self.target_scale = np.ones(2)
            self.estimator = SimpleNamespace(
                coefs_=[
                    np.zeros((width, 32)),
                    np.zeros((32, 16)),
                    np.zeros((16, 2)),
                ],
                intercepts_=[np.zeros(32), np.zeros(16), np.zeros(2)],
            )
            self.parameter_count = trainable_parameter_count(width, (32, 16), 2)

        def predict(self, values):
            weights = np.arange(1, values.shape[1] + 1, dtype=np.float64)
            return SimpleNamespace(decision_score=np.asarray(values) @ weights)

    def fake_fit(train_split, tasks, task_labels, spec, _balance, *, seed):
        assert tasks == train_tasks
        assert np.array_equal(train_split.x, train_features.variants[PRIMARY_VARIANT].x)
        assert np.array_equal(task_labels, train_features.task_labels)
        return FakeFitted(spec, seed, train_split.x.shape[1])

    monkeypatch.setattr(repair3, "group_oof_class_balanced_predictions", fake_oof)
    monkeypatch.setattr(repair3, "fit_query_candidate_coverage_threshold", fake_threshold)
    monkeypatch.setattr(repair3, "fit_class_balanced_utility_model", fake_fit)
    spec = next(
        value
        for value in load_emotion_relation_vad_repair_config(CONFIG_PATH).utility_specs
        if value.name == "class_balanced_true_bidirectional_mlp"
    )
    balance = load_emotion_relation_vad_repair_config(CONFIG_PATH).balance
    original = _fit_gate_variant_seed_scores(
        train_features, eval_features, PRIMARY_VARIANT, spec, balance
    )

    changed_labels = np.asarray(gate_corpus.labels).copy()
    eval_rows = gate_corpus.role_indices(SELECTION_ROLE)
    changed_labels[eval_rows] = (changed_labels[eval_rows] + 3) % len(CLASS_ORDER)
    label_changed_corpus = replace(gate_corpus, labels=changed_labels)
    label_changed_features = build_role_feature_set(
        label_changed_corpus,
        verified,
        eval_posterior,
        eval_tasks,
        outcome_labels_allowed=False,
    )
    after_label_change = _fit_gate_variant_seed_scores(
        train_features, label_changed_features, PRIMARY_VARIANT, spec, balance
    )
    assert [state.training_artifact_sha256 for state in original] == [
        state.training_artifact_sha256 for state in after_label_change
    ]
    assert [state.threshold_commitment_sha256 for state in original] == [
        state.threshold_commitment_sha256 for state in after_label_change
    ]
    for before, after in zip(original, after_label_change, strict=True):
        np.testing.assert_array_equal(before.gate_eval_scores, after.gate_eval_scores)

    changed_probabilities = {
        modality: np.roll(values, shift=1, axis=2)
        for modality, values in eval_posterior.probabilities.items()
    }
    feature_changed_posterior = RolePosteriorGrid(
        role=SELECTION_ROLE,
        row_indices=eval_posterior.row_indices,
        probabilities=changed_probabilities,
        base_seeds=eval_posterior.base_seeds,
        fold_by_local_row=eval_posterior.fold_by_local_row,
        fold_assignment_sha256=eval_posterior.fold_assignment_sha256,
        producer_config_sha256=eval_posterior.producer_config_sha256,
    )
    feature_changed = build_role_feature_set(
        gate_corpus,
        verified,
        feature_changed_posterior,
        eval_tasks,
        outcome_labels_allowed=False,
    )
    after_feature_change = _fit_gate_variant_seed_scores(
        train_features, feature_changed, PRIMARY_VARIANT, spec, balance
    )
    assert [state.training_artifact_sha256 for state in original] == [
        state.training_artifact_sha256 for state in after_feature_change
    ]
    assert [state.threshold_commitment_sha256 for state in original] == [
        state.threshold_commitment_sha256 for state in after_feature_change
    ]
    assert any(
        not np.array_equal(before.gate_eval_scores, after.gate_eval_scores)
        for before, after in zip(original, after_feature_change, strict=True)
    )
    assert seen_fit_groups
    assert all(groups <= set(split.train_groups) for groups in seen_fit_groups)


def test_vad_is_posterior_expectation_and_current_minus_context_transition() -> None:
    provenance = _relation_provenance()
    happy = np.zeros((2, len(CLASS_ORDER)), dtype=np.float64)
    happy[:, CLASS_ORDER.index("happy")] = 1.0
    sad = np.zeros_like(happy)
    sad[:, CLASS_ORDER.index("sad")] = 1.0
    current = _block(happy, provenance)
    history = {
        context: _block(sad, provenance)
        for context in HISTORY_CONTEXTS
    }
    matrix, names = build_vad_state_transition_features(current, history)
    assert matrix.shape == (2, 27)
    assert names[:3] == (
        "vad_state__current__valence",
        "vad_state__current__arousal",
        "vad_state__current__dominance",
    )
    np.testing.assert_allclose(
        matrix[:, :3], np.tile(VAD_COORDINATES["happy"], (2, 1))
    )
    np.testing.assert_allclose(
        matrix[:, 3:6], np.tile(VAD_COORDINATES["sad"], (2, 1))
    )
    np.testing.assert_allclose(
        matrix[:, 15:18],
        np.tile(
            np.asarray(VAD_COORDINATES["happy"])
            - np.asarray(VAD_COORDINATES["sad"]),
            (2, 1),
        ),
    )
    assert not any("gold" in name or "label" in name for name in names)


def test_builds_exact_299d_primary_and_all_explanatory_ablation_widths() -> None:
    corpus, _, verified = _synthetic_corpus()
    features = build_role_feature_set(
        corpus,
        verified,
        _posterior(corpus, FIT_ROLE),
        _tasks(corpus, FIT_ROLE),
    )
    assert features.provenance.mode == "train_fold_oof"
    assert features.vad_coordinate_sha256 == VAD_COORDINATE_SHA256
    assert tuple(features.variants) == VARIANT_ORDER
    for variant in VARIANT_ORDER:
        assert features.variants[variant].x.shape == (5, VARIANT_WIDTHS[variant])
        assert len(features.feature_names[variant]) == VARIANT_WIDTHS[variant]
        assert not any(
            token in name.lower()
            for name in features.feature_names[variant]
            for token in ("gold", "label", "target")
        )
    assert features.base_probability_by_seed.shape == (5, 5, 4, 7)


def test_feature_matrix_is_independent_of_gold_label_values() -> None:
    corpus, _, verified = _synthetic_corpus()
    posterior = _posterior(corpus, FIT_ROLE)
    tasks = _tasks(corpus, FIT_ROLE)
    original = build_role_feature_set(corpus, verified, posterior, tasks)
    changed_labels = (np.asarray(corpus.labels) + 1) % len(CLASS_ORDER)
    changed = replace(corpus, labels=changed_labels)
    rebuilt = build_role_feature_set(changed, verified, posterior, tasks)
    for variant in VARIANT_ORDER:
        np.testing.assert_array_equal(
            original.variants[variant].x, rebuilt.variants[variant].x
        )
    assert not np.array_equal(original.task_labels, rebuilt.task_labels)


def test_selection_posterior_does_not_read_selection_labels() -> None:
    corpus, _, _ = _synthetic_corpus()
    spec = ProjectorSpec()
    fit_oof = fit_fit_role_group_oof_posteriors(corpus, spec)
    first = fit_full_fit_predict_selection_posteriors(corpus, spec, fit_oof=fit_oof)
    changed_labels = np.asarray(corpus.labels).copy()
    selection = corpus.role_indices(SELECTION_ROLE)
    changed_labels[selection] = (changed_labels[selection] + 3) % len(CLASS_ORDER)
    changed = replace(corpus, labels=changed_labels)
    second = fit_full_fit_predict_selection_posteriors(changed, spec, fit_oof=fit_oof)
    for modality in MODALITIES:
        np.testing.assert_array_equal(
            first.probabilities[modality], second.probabilities[modality]
        )


def test_base_cache_serialization_is_deterministic_and_role_specific() -> None:
    matrix = np.arange(3 * 59, dtype=np.float64).reshape(3, 59)
    fit_first = _deterministic_base_cache_payload(matrix, FIT_ROLE)
    fit_second = _deterministic_base_cache_payload(matrix, FIT_ROLE)
    selection = _deterministic_base_cache_payload(matrix, SELECTION_ROLE)
    assert fit_first == fit_second
    assert fit_first != selection


def test_stage1_hash_verifies_but_never_opens_corrupt_selection_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private, manifest_path, model_config = _physical_sidecar_fixture(tmp_path)
    selection_feature, selection_label = _replace_selection_payloads_with_corrupt_bytes(
        private,
        manifest_path,
        synchronize_manifest_hash=True,
    )
    real_load = np.load
    opened: list[str] = []

    def audited_load(path, *args, **kwargs):
        name = Path(path).name
        opened.append(name)
        if Path(path).resolve() in {
            selection_feature.resolve(),
            selection_label.resolve(),
        }:
            raise AssertionError("Stage 1 must not np.load a selection payload")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", audited_load)
    staged = load_emotiontalk_fit_gate_stage(
        sidecar_dir=private,
        manifest_path=manifest_path,
        model_config=model_config,
    )
    assert opened == [
        f"features_{FIT_ROLE}.npz",
        f"labels_{FIT_ROLE}.npz",
    ]
    assert selection_feature.name not in opened
    assert selection_label.name not in opened
    assert staged.provenance.selection_feature_hash_verified is True
    assert staged.provenance.selection_feature_payload_opened is False
    assert staged.provenance.selection_feature_deserialized is False
    assert staged.provenance.selection_label_hash_verified is True
    assert staged.provenance.selection_label_payload_opened is False
    assert staged.provenance.selection_label_deserialized is False
    assert (
        staged.provenance.source_hashes[f"{SELECTION_ROLE}_features"]
        == _sha(selection_feature)
    )
    assert (
        staged.provenance.source_hashes[f"{SELECTION_ROLE}_labels"]
        == _sha(selection_label)
    )
    assert set(staged.corpus.roles.astype(str)) == {FIT_ROLE}
    assert len(staged.corpus.keys) == staged.provenance.role_rows[FIT_ROLE]
    staged.provenance.validate(staged.corpus, model_config)


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        (f"features_{SELECTION_ROLE}.npz", "feature sidecar hash differs"),
        (f"labels_{SELECTION_ROLE}.npz", "label sidecar hash differs"),
    ],
)
def test_stage1_rejects_selection_hash_drift_before_any_np_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    message: str,
) -> None:
    private, manifest_path, model_config = _physical_sidecar_fixture(tmp_path)
    (private / filename).write_bytes(b"unsynchronised-corrupt-selection-payload")
    real_load = np.load
    opened: list[str] = []

    def audited_load(path, *args, **kwargs):
        opened.append(Path(path).name)
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", audited_load)
    with pytest.raises(EmotionRelationVADRepairError, match=message):
        load_emotiontalk_fit_gate_stage(
            sidecar_dir=private,
            manifest_path=manifest_path,
            model_config=model_config,
        )
    assert opened == []


def test_stage2_materializes_and_realigns_selection_labels_after_gate(
    tmp_path: Path,
) -> None:
    private, manifest_path, model_config = _physical_sidecar_fixture(tmp_path)
    staged = load_emotiontalk_fit_gate_stage(
        sidecar_dir=private,
        manifest_path=manifest_path,
        model_config=model_config,
    )
    corpus, provenance = materialize_selection_labels_after_fit_gate(
        staged,
        sidecar_dir=private,
        manifest_path=manifest_path,
        model_config=model_config,
    )
    fit_rows = corpus.role_indices(FIT_ROLE)
    selection_rows = corpus.role_indices(SELECTION_ROLE)
    np.testing.assert_array_equal(
        corpus.labels[fit_rows], staged.corpus.labels
    )
    assert set(staged.corpus.roles.astype(str)) == {FIT_ROLE}
    assert np.all((corpus.labels[selection_rows] >= 0) & (corpus.labels[selection_rows] < 7))
    assert provenance.manifest_sha256 == staged.provenance.manifest_sha256
    assert dict(provenance.source_hashes) == dict(staged.provenance.source_hashes)
    provenance.validate(corpus, model_config)


def test_runner_has_only_verified_sidecar_interface_and_gates_selection_call_order() -> None:
    parameters = tuple(inspect.signature(run_emotion_relation_vad_repair).parameters)
    assert parameters == (
        "sidecar_dir",
        "manifest_path",
        "model_config_path",
        "repair_config_path",
        "output_path",
    )
    assert not {
        "raw_npz",
        "transcription",
        "pickle",
        "data_dir",
        "sealed_role_path",
    } & set(parameters)
    runner_source = inspect.getsource(run_emotion_relation_vad_repair)
    assert "load_emotiontalk_fit_gate_stage(" in runner_source
    assert runner_source.index("if not fit_gate_passed") < runner_source.index(
        "materialize_selection_labels_after_fit_gate("
    ) < runner_source.index(
        "fit_full_fit_predict_selection_posteriors("
    )
    selection_source = inspect.getsource(_selection_variant_scores)
    assert "fit_class_balanced_seed_scores(" in selection_source
    module_source = (ROOT / "src" / "hva_affect" / "emotion_relation_vad_repair.py").read_text(
        encoding="utf-8"
    )
    assert "\n    provisional = VerifiedCorpusProvenance(" not in module_source


def test_config_rejects_split_capacity_and_input_lock_drift(tmp_path: Path) -> None:
    original = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mutations = (
        ("fit_internal_gate_split", "namespace", "wrong-namespace"),
        ("capacity_control", "matrix_sha256", "0" * 64),
        ("registered_input_locks", "model_config_sha256", "1" * 64),
    )
    for index, (section, field, value) in enumerate(mutations):
        changed = json.loads(json.dumps(original))
        changed[section][field] = value
        path = tmp_path / f"drift-{index}.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(EmotionRelationVADRepairError):
            load_emotion_relation_vad_repair_config(path)


def test_input_hash_drift_and_output_path_shopping_fail_before_any_np_load_or_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sidecar_dir = tmp_path / "sidecars"
    sidecar_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    model = tmp_path / "model.json"
    model.write_text("{}", encoding="utf-8")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("drift must fail before payload load or model fit")

    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(repair3, "load_emotiontalk_fit_gate_stage", forbidden)
    monkeypatch.setattr(repair3, "fit_fit_role_group_oof_posteriors", forbidden)
    with pytest.raises(EmotionRelationVADRepairError, match="output path"):
        run_emotion_relation_vad_repair(
            sidecar_dir,
            manifest,
            model,
            CONFIG_PATH,
            tmp_path / "alternate-parent" / REGISTERED_OUTPUT_FILENAME,
        )
    registered_for_test = (tmp_path / REGISTERED_OUTPUT_FILENAME).resolve()
    monkeypatch.setattr(repair3, "REGISTERED_OUTPUT_PATH", registered_for_test)
    with pytest.raises(EmotionRelationVADRepairError, match="SHA-256 drifted"):
        run_emotion_relation_vad_repair(
            sidecar_dir,
            manifest,
            model,
            CONFIG_PATH,
            registered_for_test,
        )


def test_model_config_toctou_is_rejected_before_sidecar_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sidecar_dir = tmp_path / "sidecars"
    sidecar_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    model = tmp_path / "model.json"
    model.write_text(
        '{"model":{"audio_dim":3,"video_dim":2,"num_speakers":4,'
        '"max_turns":64,"num_classes":7}}',
        encoding="utf-8",
    )
    output = (tmp_path / REGISTERED_OUTPUT_FILENAME).resolve()
    monkeypatch.setattr(repair3, "REGISTERED_OUTPUT_PATH", output)
    monkeypatch.setattr(repair3, "_assert_registered_input_locks", lambda *_: None)
    real_load_json = repair3._load_json

    def mutate_after_read(path: Path, *, name: str):
        payload = real_load_json(path, name=name)
        if name == "causal backbone config":
            Path(path).write_text("{}", encoding="utf-8")
        return payload

    monkeypatch.setattr(repair3, "_load_json", mutate_after_read)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("TOCTOU must fail before sidecar load or model fit")

    monkeypatch.setattr(repair3, "load_emotiontalk_fit_gate_stage", forbidden)
    monkeypatch.setattr(repair3, "fit_fit_role_group_oof_posteriors", forbidden)
    with pytest.raises(EmotionRelationVADRepairError, match="model_config changed"):
        run_emotion_relation_vad_repair(
            sidecar_dir,
            manifest,
            model,
            CONFIG_PATH,
            output,
        )


def test_optional_teacher_report_whitelists_fields_and_drops_row_payload() -> None:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["optional_teacher"]["malicious_rows"] = [
        {"row_id": "secret", "label": 3, "probability": [0.1, 0.9]}
    ]
    raw["optional_teacher"]["fit_oof_eligibility"]["row_like_payload"] = [1, 2, 3]
    report = repair3._optional_teacher_report(raw)
    rendered = json.dumps(report, sort_keys=True)
    assert "malicious" not in rendered
    assert "row_like_payload" not in rendered
    assert "secret" not in rendered
    assert set(report) == {
        "status",
        "model_revision",
        "role",
        "fit_oof_eligibility",
        "selection_can_choose_teacher",
        "selection_can_choose_layer",
        "selection_can_choose_class_mapping",
        "johnson_chinese_model",
    }


def test_failed_fit_gate_behaviorally_prevents_selection_prediction_and_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, _, verified = _synthetic_corpus(groups_per_role=10)
    sidecar_dir = tmp_path / "sidecars"
    sidecar_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    model_config = tmp_path / "model.json"
    model_config.write_text(
        '{"model":{"audio_dim":3,"video_dim":2,"num_speakers":4,'
        '"max_turns":64,"num_classes":7}}',
        encoding="utf-8",
    )
    output = tmp_path / REGISTERED_OUTPUT_FILENAME
    monkeypatch.setattr(repair3, "REGISTERED_OUTPUT_PATH", output.resolve())
    fit_posterior = _posterior(corpus, FIT_ROLE)
    fit_features = build_role_feature_set(
        corpus, verified, fit_posterior, _tasks(corpus, FIT_ROLE)
    )
    monkeypatch.setattr(
        repair3,
        "load_emotiontalk_fit_gate_stage",
        lambda **_: SimpleNamespace(corpus=corpus, provenance=verified, manifest={}),
    )
    monkeypatch.setattr(repair3, "_assert_registered_input_locks", lambda *_: None)
    monkeypatch.setattr(
        repair3,
        "fit_fit_role_group_oof_posteriors",
        lambda *_: fit_posterior,
    )
    monkeypatch.setattr(
        repair3,
        "_sample_tasks_for_role",
        lambda _corpus, role, _sampling: _tasks(_corpus, role),
    )
    monkeypatch.setattr(
        repair3,
        "build_role_feature_set",
        lambda *_, **__: fit_features,
    )
    monkeypatch.setattr(
        repair3,
        "fit_gate_train_predict_gate_eval_posteriors",
        lambda *_, **__: fit_posterior,
    )
    monkeypatch.setattr(
        repair3,
        "evaluate_fit_only_open_gate",
        lambda *_args, **_kwargs: (
            False,
            {
                "passed": False,
                "selection_prediction_generated": False,
                "selection_label_scored": False,
            },
        ),
    )

    def forbidden_selection_call(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("selection prediction must not run after a failed fit gate")

    monkeypatch.setattr(
        repair3,
        "fit_full_fit_predict_selection_posteriors",
        forbidden_selection_call,
    )
    monkeypatch.setattr(
        repair3,
        "materialize_selection_labels_after_fit_gate",
        forbidden_selection_call,
    )
    report = run_emotion_relation_vad_repair(
        sidecar_dir,
        manifest,
        model_config,
        CONFIG_PATH,
        output,
    )
    assert report["status"] == "fit_only_gate_no_go_no_selection_predictions"
    assert report["model_selection"]["executed"] is False
    assert report["model_selection"]["selection_prediction_generated"] is False
    assert report["model_selection"]["selection_label_scored"] is False
    assert output.is_file()


def test_gate_fail_runner_accepts_hash_synced_corrupt_selection_payloads_without_opening_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private, manifest_path, _ = _physical_sidecar_fixture(tmp_path)
    selection_feature, selection_label = _replace_selection_payloads_with_corrupt_bytes(
        private,
        manifest_path,
        synchronize_manifest_hash=True,
    )
    model_config_path = tmp_path / "model.json"
    model_config_path.write_text(
        json.dumps(
            {
                "model": {
                    "audio_dim": 2,
                    "video_dim": 3,
                    "num_speakers": 4,
                    "max_turns": 32,
                    "num_classes": 7,
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / REGISTERED_OUTPUT_FILENAME
    monkeypatch.setattr(repair3, "REGISTERED_OUTPUT_PATH", output.resolve())
    real_load = np.load
    opened: list[str] = []

    def audited_load(path, *args, **kwargs):
        name = Path(path).name
        opened.append(name)
        if Path(path).resolve() in {
            selection_feature.resolve(),
            selection_label.resolve(),
        }:
            raise AssertionError("gate-fail runner must not np.load selection payloads")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", audited_load)
    fit_posterior = SimpleNamespace(producer_config_sha256="a" * 64)
    fit_features = SimpleNamespace(
        provenance=SimpleNamespace(task_order_sha256="b" * 64),
        base_cache_lineage_sha256="c" * 64,
    )
    monkeypatch.setattr(
        repair3,
        "fit_fit_role_group_oof_posteriors",
        lambda *_: fit_posterior,
    )
    monkeypatch.setattr(
        repair3,
        "_sample_tasks_for_role",
        lambda *_: (object(),),
    )
    monkeypatch.setattr(
        repair3,
        "build_role_feature_set",
        lambda *_, **__: fit_features,
    )
    fake_split = SimpleNamespace(
        group_assignment_sha256="d" * 64,
        row_assignment_sha256="e" * 64,
        aggregate_attestation=lambda: {
            "namespace": FIT_INTERNAL_GATE_NAMESPACE,
            "eligible": True,
        },
    )
    monkeypatch.setattr(
        repair3,
        "deterministic_fit_internal_gate_split",
        lambda *_: fake_split,
    )
    monkeypatch.setattr(
        repair3,
        "build_fit_internal_gate_corpus",
        lambda corpus, *_args, **_kwargs: corpus,
    )
    monkeypatch.setattr(
        repair3,
        "fit_gate_train_predict_gate_eval_posteriors",
        lambda *_, **__: fit_posterior,
    )
    monkeypatch.setattr(repair3, "_assert_registered_input_locks", lambda *_: None)
    monkeypatch.setattr(
        repair3,
        "evaluate_fit_only_open_gate",
        lambda *_args, **_kwargs: (
            False,
            {
                "passed": False,
                "selection_prediction_generated": False,
                "selection_label_scored": False,
            },
        ),
    )

    def forbidden_after_gate(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("selection materialization/prediction must not run")

    monkeypatch.setattr(
        repair3,
        "materialize_selection_labels_after_fit_gate",
        forbidden_after_gate,
    )
    monkeypatch.setattr(
        repair3,
        "fit_full_fit_predict_selection_posteriors",
        forbidden_after_gate,
    )
    report = run_emotion_relation_vad_repair(
        private,
        manifest_path,
        model_config_path,
        CONFIG_PATH,
        output,
    )
    assert opened == [
        f"features_{FIT_ROLE}.npz",
        f"labels_{FIT_ROLE}.npz",
    ]
    assert selection_feature.name not in opened
    assert selection_label.name not in opened
    assert report["status"] == "fit_only_gate_no_go_no_selection_predictions"
    assert report["fit_gate_stage_provenance"]["selection_label_hash_verified"] is True
    assert report["fit_gate_stage_provenance"]["selection_label_deserialized"] is False
    assert report["fit_gate_stage_provenance"]["selection_feature_payload_opened"] is False
    assert report["fit_gate_stage_provenance"]["selection_label_payload_opened"] is False
    assert report["access_contract"]["stage_1_selection_feature_payload_opened"] is False
    assert report["access_contract"]["stage_1_selection_feature_payload_deserialized"] is False
    assert report["access_contract"]["stage_1_selection_label_payload_opened"] is False
    assert report["access_contract"]["stage_1_selection_label_payload_deserialized"] is False
    assert report["access_contract"]["selection_label_payload_deserialized_after_fit_gate"] is False
    assert report["access_contract"]["access_event_sequence"][-1].startswith(
        "fit_gate_failed"
    )
    assert output.is_file()
