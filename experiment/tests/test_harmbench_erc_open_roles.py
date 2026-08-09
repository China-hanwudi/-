from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from hva_affect.emotiontalk_role_sidecar import FIT_ROLE, SELECTION_ROLE  # noqa: E402
from hva_affect.harmbench_erc_open_roles import (  # noqa: E402
    HarmBenchOpenRoleError,
    OutcomeFreeRoleFeatures,
    make_open_role_capabilities,
    make_outcome_free_role_features,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _features(role: str, *, offset: int = 0) -> OutcomeFreeRoleFeatures:
    prefix = "g" if role == FIT_ROLE else "h"
    return make_outcome_free_role_features(
        dataset_id="synthetic",
        role=role,
        keys=np.asarray([f"row-{offset + index}" for index in range(6)]),
        texts=[f"text {index}" for index in range(6)],
        audio=np.arange(18, dtype=np.float32).reshape(6, 3) / 10.0,
        video=np.arange(12, dtype=np.float32).reshape(6, 2) / 10.0,
        groups=np.asarray(
            [f"{prefix}0", f"{prefix}0", f"{prefix}0", f"{prefix}0", f"{prefix}1", f"{prefix}1"]
        ),
        speaker_identity=np.asarray(["a", "b", "a", "a", "a", "a"]),
        turn_ids=np.asarray([0, 1, 2, 3, 0, 1]),
        protocol_row_ids=np.arange(offset, offset + 6, dtype=np.int64),
        row_alignment_sha256=SHA_A if role == FIT_ROLE else SHA_B,
        feature_sha256=SHA_C if role == FIT_ROLE else SHA_D,
    )


def test_prediction_capability_has_no_label_surface_and_is_immutable() -> None:
    fit = _features(FIT_ROLE)
    selection = _features(SELECTION_ROLE, offset=100)
    source_labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    capability = make_open_role_capabilities(
        fit_features=fit,
        fit_labels=source_labels,
        fit_label_sha256=SHA_A,
        selection_features=selection,
        label_order=("a", "b", "c"),
        manifest_sha256=SHA_B,
    )
    assert "labels" not in {field.name for field in fields(type(capability.selection))}
    assert not hasattr(capability.selection, "labels")
    assert capability.selection_label_archive_opened is False
    assert capability.selection_label_archive_hashed is False
    source_labels[:] = 2
    assert capability.fit.labels.tolist() == [0, 1, 2, 0, 1, 2]
    with pytest.raises(ValueError, match="read-only"):
        capability.selection.audio[0, 0] = 99.0


def test_same_speaker_history_is_strict_past_not_dialogue_all_past() -> None:
    values = _features(FIT_ROLE)
    assert values.same_speaker_histories[0] == ()
    assert values.same_speaker_histories[1] == ()
    assert values.same_speaker_histories[2] == (0,)
    assert values.same_speaker_histories[3] == (0, 2)
    assert 1 not in values.same_speaker_histories[3]
    assert values.same_speaker_histories[5] == (4,)
    assert values.history_eligible.tolist() == [False, False, True, True, False, True]


def test_factory_rejects_misalignment_duplicate_rows_and_nonfinite_media() -> None:
    common = dict(
        dataset_id="synthetic",
        role=FIT_ROLE,
        keys=np.asarray(["a", "b"]),
        texts=["a", "b"],
        audio=np.zeros((2, 2), dtype=np.float32),
        video=np.zeros((2, 2), dtype=np.float32),
        groups=np.asarray(["g", "g"]),
        speaker_identity=np.asarray(["s", "s"]),
        turn_ids=np.asarray([0, 1]),
        protocol_row_ids=np.asarray([0, 1]),
        row_alignment_sha256=SHA_A,
        feature_sha256=SHA_B,
    )
    with pytest.raises(HarmBenchOpenRoleError, match="row-aligned"):
        make_outcome_free_role_features(**{**common, "texts": ["a"]})
    with pytest.raises(HarmBenchOpenRoleError, match="unique"):
        make_outcome_free_role_features(
            **{**common, "protocol_row_ids": np.asarray([0, 0])}
        )
    bad_audio = np.zeros((2, 2), dtype=np.float32)
    bad_audio[0, 0] = np.nan
    with pytest.raises(HarmBenchOpenRoleError, match="non-finite"):
        make_outcome_free_role_features(**{**common, "audio": bad_audio})


def test_capability_rejects_selection_labels_or_wrong_roles() -> None:
    fit = _features(FIT_ROLE)
    selection = _features(SELECTION_ROLE, offset=100)
    with pytest.raises(HarmBenchOpenRoleError, match="integer dtype"):
        make_open_role_capabilities(
            fit_features=fit,
            fit_labels=np.asarray([0.0] * 6),
            fit_label_sha256=SHA_A,
            selection_features=selection,
            label_order=("a", "b"),
            manifest_sha256=SHA_B,
        )
    with pytest.raises(HarmBenchOpenRoleError, match="outside"):
        make_open_role_capabilities(
            fit_features=fit,
            fit_labels=np.asarray([0, 1, 2, 0, 1, 2]),
            fit_label_sha256=SHA_A,
            selection_features=selection,
            label_order=("a", "b"),
            manifest_sha256=SHA_B,
        )
    with pytest.raises(HarmBenchOpenRoleError, match="role identities"):
        make_open_role_capabilities(
            fit_features=fit,
            fit_labels=np.asarray([0, 1, 0, 1, 0, 1]),
            fit_label_sha256=SHA_A,
            selection_features=_features(FIT_ROLE, offset=100),
            label_order=("a", "b"),
            manifest_sha256=SHA_B,
        )


def test_capability_hash_binds_fit_labels_and_selection_content() -> None:
    fit = _features(FIT_ROLE)
    selection = _features(SELECTION_ROLE, offset=100)
    first = make_open_role_capabilities(
        fit_features=fit,
        fit_labels=np.asarray([0, 1, 2, 0, 1, 2]),
        fit_label_sha256=SHA_A,
        selection_features=selection,
        label_order=("a", "b", "c"),
        manifest_sha256=SHA_B,
    )
    second = make_open_role_capabilities(
        fit_features=fit,
        fit_labels=np.asarray([0, 1, 2, 0, 2, 1]),
        fit_label_sha256=SHA_A,
        selection_features=selection,
        label_order=("a", "b", "c"),
        manifest_sha256=SHA_B,
    )
    assert first.capability_sha256 != second.capability_sha256


def test_capability_rejects_group_crossing_role_boundary() -> None:
    fit = _features(FIT_ROLE)
    selection = _features(SELECTION_ROLE, offset=100)
    overlapping = make_outcome_free_role_features(
        dataset_id="synthetic",
        role=SELECTION_ROLE,
        keys=selection.keys,
        texts=selection.texts,
        audio=selection.audio,
        video=selection.video,
        groups=np.asarray(["g0"] * selection.rows),
        speaker_identity=selection.speaker_identity,
        turn_ids=selection.turn_ids,
        protocol_row_ids=selection.protocol_row_ids,
        row_alignment_sha256=selection.row_alignment_sha256,
        feature_sha256=selection.feature_sha256,
    )
    with pytest.raises(HarmBenchOpenRoleError, match="split an independent group"):
        make_open_role_capabilities(
            fit_features=fit,
            fit_labels=np.asarray([0, 1, 2, 0, 1, 2]),
            fit_label_sha256=SHA_A,
            selection_features=overlapping,
            label_order=("a", "b", "c"),
            manifest_sha256=SHA_B,
        )
