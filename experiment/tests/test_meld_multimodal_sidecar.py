from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.meld_multimodal_sidecar import (  # noqa: E402
    EMOTION_TO_INDEX,
    MANIFEST_SCHEMA_VERSION,
    MeldSidecarContractError,
    PROTOCOL,
    _speaker_token_code,
    _pool_record,
    load_meld_role_sidecar,
    prepare_meld_role_sidecars,
)
from hva_affect.scu_set import assign_group_role  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _groups_covering_roles() -> list[int]:
    ranges = {
        "base_and_utility_fit": [0, 64],
        "model_selection": [65, 79],
        "calibration": [80, 89],
        "internal_holdout": [90, 99],
    }
    found: dict[str, int] = {}
    for group in range(10_000):
        role, _ = assign_group_role("MELD", group, "scu_set_exploration_v1", ranges)
        found.setdefault(role, group)
        if len(found) == 4:
            return [found[name] for name in ranges]
    raise AssertionError("could not synthesize one group per role")


def _write_fixture(
    tmp_path: Path, *, sealed_speaker_suffix: str = "default"
) -> tuple[Path, Path, Path]:
    groups = _groups_covering_roles()
    rows = []
    records = {}
    emotions = tuple(EMOTION_TO_INDEX)
    for index, group in enumerate(groups):
        for order in range(2):
            key = f"{group}_{order}"
            emotion = emotions[(index + order) % len(emotions)]
            rows.append(
                {
                    "Utterance": f"synthetic utterance {index} {order}",
                    "Speaker": (
                        "same-speaker"
                        if index < 2
                        else f"sealed-{sealed_speaker_suffix}-{index}"
                    ),
                    "Emotion": emotion,
                    "Dialogue_ID": group,
                    "Utterance_ID": order,
                }
            )
            records[key] = {
                "token_ids": [1.0, 2.0],
                "audio_features": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                "video_features": np.asarray(
                    [[5.0, 6.0, 7.0], [9.0, 10.0, 11.0]], dtype=np.float64
                ),
                # Deliberately wrong for one row: official CSV must win.
                "label": 6 if index == 0 and order == 0 else EMOTION_TO_INDEX[emotion],
            }
    csv_path = tmp_path / "train_sent_emo.csv"
    pickle_path = tmp_path / "train.pkl"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with pickle_path.open("wb") as handle:
        pickle.dump(records, handle, protocol=3)
    config = {
        "protocol": PROTOCOL,
        "status": "frozen_before_role_sidecar_generation",
        "dataset_id": "MELD",
        "split_protocol_id": "scu_set_exploration_v1",
        "roles": {
            "base_and_utility_fit": [0, 64],
            "model_selection": [65, 79],
            "calibration": [80, 89],
            "internal_holdout": [90, 99],
        },
        "source_contract": {
            "train_csv_sha256": _sha(csv_path),
            "train_pickle_sha256": _sha(pickle_path),
            "allowed_missing_feature_rows": 0,
            "official_csv_is_authoritative_label_source": True,
            "embedded_pickle_label_is_trusted_custodian_consistency_audit_only": True,
            "embedded_label_mismatch_statistics_must_not_be_published": True,
            "dev_and_test_sources_forbidden": True,
        },
        "feature_contract": {
            "audio_dimension": 2,
            "video_dimension": 3,
            "pooling": "utterance_sequence_mean_and_population_std",
            "text_source": "official_train_csv_utterance",
            "speaker_token": (
                "sha256_63bit_population_independent_then_fit_only_runner_mapping_with_oov_zero"
            ),
            "numeric_dtype": "float32",
            "history_rule": (
                "same_dialogue_and_same_speaker_and_strictly_lower_Utterance_ID"
            ),
        },
        "output_contract": {
            "write_once": True,
            "one_feature_archive_per_role": True,
            "one_label_archive_per_role": True,
            "allow_pickle": False,
            "private_not_for_repository": True,
            "public_output": "aggregate manifest only",
            "legacy_v1_directory_must_not_be_overwritten": True,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return csv_path, pickle_path, config_path


def test_pooling_is_mean_plus_population_std_and_fails_on_misalignment() -> None:
    record = {
        "token_ids": [1, 2],
        "audio_features": np.asarray([[1.0, 3.0], [5.0, 7.0]]),
        "video_features": np.asarray([[2.0, 4.0, 6.0], [6.0, 8.0, 10.0]]),
    }
    audio, video = _pool_record(record, audio_dimension=2, video_dimension=3)
    np.testing.assert_allclose(audio, [3.0, 5.0, 2.0, 2.0])
    np.testing.assert_allclose(video, [4.0, 6.0, 8.0, 2.0, 2.0, 2.0])
    assert audio.dtype == np.float32 and video.dtype == np.float32
    broken = dict(record)
    broken["token_ids"] = [1]
    with pytest.raises(MeldSidecarContractError, match="not aligned"):
        _pool_record(broken, audio_dimension=2, video_dimension=3)


def test_preparation_writes_role_separated_archives_and_official_labels_win(
    tmp_path: Path,
) -> None:
    csv_path, pickle_path, config_path = _write_fixture(tmp_path)
    output = tmp_path / "sidecars"
    public_manifest = tmp_path / "manifest_v2.json"
    report = prepare_meld_role_sidecars(
        train_csv_path=csv_path,
        train_pickle_path=pickle_path,
        config_path=config_path,
        private_output_dir=output,
        public_manifest_path=public_manifest,
    )
    assert report["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert set(report["roles"]) == {
        "base_and_utility_fit",
        "model_selection",
        "calibration",
        "internal_holdout",
    }
    assert report["source_contract"][
        "embedded_pickle_label_consistency_checked_by_trusted_custodian"
    ] is True
    assert report["source_contract"]["embedded_pickle_label_mismatch_statistics_exposed"] is False
    assert "mismatch_count" not in json.dumps(report)
    assert report["seal_contract"]["each_role_has_a_separate_label_archive"] is True
    loaded = load_meld_role_sidecar(
        output / "features_base_and_utility_fit.npz",
        output / "labels_base_and_utility_fit.npz",
        expected_role="base_and_utility_fit",
    )
    assert len(loaded.labels) == 2
    # The first synthetic row is neutral in the official CSV despite the
    # intentionally wrong embedded pickle label.
    assert int(loaded.labels[0]) == EMOTION_TO_INDEX["neutral"]
    assert loaded.audio_mean_std.shape == (2, 4)
    assert loaded.video_mean_std.shape == (2, 6)
    assert report["roles"]["base_and_utility_fit"]["history_eligible_rows"] == 1
    assert int(loaded.speaker_codes[0]) == _speaker_token_code("same-speaker")
    assert loaded.protocol_row_ids.tolist()[:2] == [0, 1]
    assert not (output / "manifest.json").exists()
    with pytest.raises(FileExistsError, match="already exists|not empty"):
        prepare_meld_role_sidecars(
            train_csv_path=csv_path,
            train_pickle_path=pickle_path,
            config_path=config_path,
            private_output_dir=output,
            public_manifest_path=public_manifest,
        )


def test_hash_mismatch_fails_before_unregistered_pickle_is_opened(tmp_path: Path) -> None:
    csv_path, pickle_path, config_path = _write_fixture(tmp_path)
    pickle_path.write_bytes(b"not a trusted pickle")
    with pytest.raises(MeldSidecarContractError, match="pickle hash differs"):
        prepare_meld_role_sidecars(
            train_csv_path=csv_path,
            train_pickle_path=pickle_path,
            config_path=config_path,
            private_output_dir=tmp_path / "out",
            public_manifest_path=tmp_path / "manifest.json",
        )


def test_fit_speaker_token_is_independent_of_sealed_role_speaker_population(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    csv_a, pickle_a, config_a = _write_fixture(first, sealed_speaker_suffix="A")
    csv_b, pickle_b, config_b = _write_fixture(second, sealed_speaker_suffix="B")
    out_a = first / "sidecars"
    out_b = second / "sidecars"
    prepare_meld_role_sidecars(
        train_csv_path=csv_a,
        train_pickle_path=pickle_a,
        config_path=config_a,
        private_output_dir=out_a,
        public_manifest_path=first / "manifest.json",
    )
    prepare_meld_role_sidecars(
        train_csv_path=csv_b,
        train_pickle_path=pickle_b,
        config_path=config_b,
        private_output_dir=out_b,
        public_manifest_path=second / "manifest.json",
    )
    fit_a = load_meld_role_sidecar(
        out_a / "features_base_and_utility_fit.npz",
        out_a / "labels_base_and_utility_fit.npz",
        expected_role="base_and_utility_fit",
    )
    fit_b = load_meld_role_sidecar(
        out_b / "features_base_and_utility_fit.npz",
        out_b / "labels_base_and_utility_fit.npz",
        expected_role="base_and_utility_fit",
    )
    np.testing.assert_array_equal(fit_a.speaker_codes, fit_b.speaker_codes)
    assert set(fit_a.speaker_codes.tolist()) == {_speaker_token_code("same-speaker")}


def test_private_meld_sidecars_cannot_be_written_inside_repository(tmp_path: Path) -> None:
    csv_path, pickle_path, config_path = _write_fixture(tmp_path)
    repository = tmp_path / "repo"
    repository.mkdir()
    with pytest.raises(MeldSidecarContractError, match="outside repository"):
        prepare_meld_role_sidecars(
            train_csv_path=csv_path,
            train_pickle_path=pickle_path,
            config_path=config_path,
            private_output_dir=repository / "private_sidecars",
            public_manifest_path=repository / "manifest.json",
            repository_root=repository,
        )


def test_cli_has_no_dev_or_test_input() -> None:
    path = ROOT / "scripts" / "prepare_meld_multimodal_sidecars.py"
    spec = importlib.util.spec_from_file_location("meld_sidecar_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    destinations = {action.dest for action in module.build_parser()._actions}
    assert destinations == {
        "help",
        "train_csv",
        "train_pickle",
        "config",
        "private_output_dir",
        "public_manifest",
    }
    assert "dev" not in destinations and "test" not in destinations
