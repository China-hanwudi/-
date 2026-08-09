from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch", reason="EmotionTalk runner loader uses backbone config")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.causal_multimodal_backbone import CausalBackboneConfig  # noqa: E402
from hva_affect.data_contract import ContractError  # noqa: E402
from hva_affect.emotiontalk_causal_backbone_runner import (  # noqa: E402
    load_emotiontalk_open_role_corpus,
)
from hva_affect.emotiontalk_role_sidecar import (  # noqa: E402
    FEATURE_SCHEMA,
    FIT_ROLE,
    FROZEN_ROLE_RANGES,
    LABEL_SCHEMA,
    MANIFEST_SCHEMA,
    OPEN_ROLES,
    PROTOCOL,
    SELECTION_ROLE,
    load_emotiontalk_role_sidecars,
    prepare_emotiontalk_role_sidecars,
)
from hva_affect.scu_set import assign_group_role  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_for_role(role: str) -> tuple[str, str]:
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
    raise AssertionError("no group for role")


def sources(
    tmp_path: Path, *, selection_speaker: str = "99"
) -> tuple[Path, Path, Path, list[str]]:
    keys: list[str] = []
    labels: dict[str, dict[str, int]] = {}
    for role in OPEN_ROLES:
        group, dialogue = group_for_role(role)
        speaker = "01" if role == FIT_ROLE else selection_speaker
        for turn in range(3):
            key = f"{group}_{dialogue}_{speaker}_{turn:03d}"
            keys.append(key)
            labels[key] = {"emo": turn, "val": 0}
    label_path = tmp_path / "mm_label.npz"
    np.savez(label_path, train_corpus=np.asarray(labels, dtype=object))
    feature_path = tmp_path / "full_media_features.npz"
    np.savez(
        feature_path,
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
        + "".join(f"{key}.wav,文本-{index}\n" for index, key in enumerate(keys)),
        encoding="utf-8",
    )
    return label_path, feature_path, transcription, keys


def config_path(
    tmp_path: Path, label_path: Path, feature_path: Path, transcription: Path
) -> Path:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "status": "frozen_before_trusted_generation",
                "dataset_id": "EmotionTalk",
                "split_protocol_id": "scu_set_exploration_v1",
                "roles": FROZEN_ROLE_RANGES,
                "source_sha256": {
                    "label_archive": sha(label_path),
                    "media_features": sha(feature_path),
                    "transcription": sha(transcription),
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def prepare_fixture(tmp_path: Path) -> tuple[Path, Path, dict, list[str], tuple[Path, Path, Path]]:
    label_path, feature_path, transcription, keys = sources(tmp_path)
    config = config_path(tmp_path, label_path, feature_path, transcription)
    repository = tmp_path / "repo"
    repository.mkdir()
    private = tmp_path / "private" / "emotiontalk_v2"
    public = repository / "manifest_v2.json"
    manifest = prepare_emotiontalk_role_sidecars(
        label_archive_path=label_path,
        feature_path=feature_path,
        transcription_path=transcription,
        config_path=config,
        private_output_dir=private,
        public_manifest_path=public,
        repository_root=repository,
    )
    return private, public, manifest, keys, (label_path, feature_path, transcription)


def tiny_model_config() -> CausalBackboneConfig:
    return CausalBackboneConfig(
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


def test_trusted_generation_writes_four_private_sidecars_and_aggregate_manifest(
    tmp_path: Path,
) -> None:
    private, public, manifest, source_keys, _ = prepare_fixture(tmp_path)
    assert manifest["schema_version"] == MANIFEST_SCHEMA
    assert manifest["seal_contract"]["calibration_holdout_validation_test_sidecars_created"] is False
    assert manifest["public_content_audit"]["contains_labels_or_class_counts"] is False
    expected = {
        f"features_{FIT_ROLE}.npz",
        f"labels_{FIT_ROLE}.npz",
        f"features_{SELECTION_ROLE}.npz",
        f"labels_{SELECTION_ROLE}.npz",
    }
    assert {path.name for path in private.iterdir()} == expected
    rendered = public.read_text(encoding="utf-8")
    assert all(key not in rendered for key in source_keys)
    arrays, loaded_manifest = load_emotiontalk_role_sidecars(
        sidecar_dir=private, manifest_path=public
    )
    assert loaded_manifest == manifest
    assert set(arrays) == set(OPEN_ROLES)
    for role in OPEN_ROLES:
        assert arrays[role].audio.shape == (3, 2)
        assert arrays[role].video.shape == (3, 3)
        with np.load(private / f"features_{role}.npz", allow_pickle=False) as archive:
            assert str(archive["schema_version"]) == FEATURE_SCHEMA
        with np.load(private / f"labels_{role}.npz", allow_pickle=False) as archive:
            assert str(archive["schema_version"]) == LABEL_SCHEMA


def test_model_loader_opens_only_manifest_four_files_and_not_sealed_or_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private, public, _, _, sources_paths = prepare_fixture(tmp_path)
    sealed = private / "features_calibration.npz"
    np.savez(sealed, forbidden=np.asarray([1]))
    real_load = np.load
    opened: list[str] = []

    def audited_load(path, *args, **kwargs):
        opened.append(Path(path).name)
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", audited_load)
    corpus, provenance = load_emotiontalk_open_role_corpus(
        sidecar_dir=private,
        manifest_path=public,
        model_config=tiny_model_config(),
    )
    expected = {
        f"features_{FIT_ROLE}.npz",
        f"labels_{FIT_ROLE}.npz",
        f"features_{SELECTION_ROLE}.npz",
        f"labels_{SELECTION_ROLE}.npz",
    }
    assert set(opened) == expected
    assert all(opened.count(name) == 1 for name in expected)
    assert sealed.name not in opened
    assert all(path.name not in opened for path in sources_paths)
    fit = corpus.role_indices(FIT_ROLE)
    selection = corpus.role_indices(SELECTION_ROLE)
    assert set(corpus.speaker_ids[fit]) == {1}
    assert set(corpus.speaker_ids[selection]) == {0}
    assert provenance.speaker_mapping_sha256 == corpus.speaker_mapping_sha256
    assert sorted(map(len, corpus.histories)) == [0, 0, 1, 1, 2, 2]


@pytest.mark.parametrize("kind", ["feature", "label"])
def test_copied_renamed_role_sidecar_is_rejected(tmp_path: Path, kind: str) -> None:
    private, public, manifest, _, _ = prepare_fixture(tmp_path)
    suffix = "features" if kind == "feature" else "labels"
    source = private / f"{suffix}_{FIT_ROLE}.npz"
    target = private / f"{suffix}_{SELECTION_ROLE}.npz"
    shutil.copyfile(source, target)
    hash_field = "feature_sha256" if kind == "feature" else "label_sha256"
    manifest["roles"][SELECTION_ROLE][hash_field] = sha(target)
    public.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ContractError, match="role changed"):
        load_emotiontalk_role_sidecars(sidecar_dir=private, manifest_path=public)


def test_manifest_hash_drift_is_rejected(tmp_path: Path) -> None:
    private, public, manifest, _, _ = prepare_fixture(tmp_path)
    manifest["roles"][FIT_ROLE]["feature_sha256"] = "f" * 64
    public.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ContractError, match="hash differs"):
        load_emotiontalk_role_sidecars(sidecar_dir=private, manifest_path=public)


@pytest.mark.parametrize(
    ("kind", "field", "hash_field", "message"),
    [
        (
            "features",
            "source_feature_config_sha256",
            "feature_sha256",
            "source config",
        ),
        ("labels", "source_label_sha256", "label_sha256", "source hash"),
    ],
)
def test_sidecar_source_cross_binding_cannot_be_reregistered(
    tmp_path: Path,
    kind: str,
    field: str,
    hash_field: str,
    message: str,
) -> None:
    private, public, manifest, _, _ = prepare_fixture(tmp_path)
    path = private / f"{kind}_{FIT_ROLE}.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    payload[field] = np.asarray("9" * 64)
    np.savez(path, **payload)
    manifest["roles"][FIT_ROLE][hash_field] = sha(path)
    public.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ContractError, match=message):
        load_emotiontalk_role_sidecars(sidecar_dir=private, manifest_path=public)


def test_manifest_seal_and_public_audit_fields_are_exact(tmp_path: Path) -> None:
    private, public, manifest, _, _ = prepare_fixture(tmp_path)
    manifest["public_content_audit"]["contains_labels_or_class_counts"] = True
    public.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ContractError, match="public-content audit"):
        load_emotiontalk_role_sidecars(sidecar_dir=private, manifest_path=public)


def test_hash_drift_fails_before_sidecar_write(tmp_path: Path) -> None:
    label_path, feature_path, transcription, _ = sources(tmp_path)
    config = config_path(tmp_path, label_path, feature_path, transcription)
    transcription.write_text("changed", encoding="utf-8")
    private = tmp_path / "private"
    with pytest.raises(ContractError, match="hash changed"):
        prepare_emotiontalk_role_sidecars(
            label_archive_path=label_path,
            feature_path=feature_path,
            transcription_path=transcription,
            config_path=config,
            private_output_dir=private,
            public_manifest_path=tmp_path / "manifest.json",
        )
    assert not private.exists()


def test_private_sidecar_inside_repository_is_rejected(tmp_path: Path) -> None:
    label_path, feature_path, transcription, _ = sources(tmp_path)
    config = config_path(tmp_path, label_path, feature_path, transcription)
    repository = tmp_path / "repo"
    repository.mkdir()
    with pytest.raises(ContractError, match="outside"):
        prepare_emotiontalk_role_sidecars(
            label_archive_path=label_path,
            feature_path=feature_path,
            transcription_path=transcription,
            config_path=config,
            private_output_dir=repository / "open_roles",
            public_manifest_path=repository / "manifest.json",
            repository_root=repository,
        )
