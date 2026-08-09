from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import gc
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import weakref

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from test_harmbench_erc_production_training import (  # noqa: E402
    _current_kwargs,
    _history_kwargs,
    production_contract,
)
from hva_affect.emotiontalk_role_sidecar import SELECTION_ROLE  # noqa: E402
from hva_affect.harmbench_erc_checkpoint_artifact import (  # noqa: E402
    HarmBenchCheckpointArtifactError,
    load_checkpoint_artifact,
    publish_checkpoint_artifact,
    restore_current_only_checkpoint_artifact,
    restore_history_checkpoint_artifact,
)
from hva_affect.harmbench_erc_contexts import (  # noqa: E402
    FIT_HELDOUT_OOF_CONTEXT_ROLE,
    SELECTION_CONTEXT_ROLE,
    STRICT_PAST_STRATEGY_IDS,
    build_strict_past_context_roster,
)
from hva_affect.harmbench_erc_open_roles import (  # noqa: E402
    make_outcome_free_role_features,
    make_synthetic_selection_feature_capability,
)
from hva_affect.harmbench_erc_processors import transform_role_features  # noqa: E402
from hva_affect.harmbench_erc_models import (  # noqa: E402
    CAUSAL_GRU_ID,
    CURRENT_ONLY_NAMESPACE,
    DEEPSETS_POOL_ID,
    HISTORY_NAMESPACE,
    LINEAR_POOL_ID,
    ProductionCurrentOnlyCheckpoint,
    ProductionHistoryCheckpoint,
    fit_current_only_model,
    fit_history_model,
    predict_production_current_only,
    predict_production_history,
)


MODEL_IDS = (LINEAR_POOL_ID, DEEPSETS_POOL_ID, CAUSAL_GRU_ID)


def _heldout_roster(contract: object) -> object:
    return build_strict_past_context_roster(
        contract.fit_features,
        contract.fit_features,
        contract.processed,
        contract.receipt,
        contract.plan,
        training_seed=contract.seed,
        fold=contract.fold,
        context_role=FIT_HELDOUT_OOF_CONTEXT_ROLE,
        strategy_id=STRICT_PAST_STRATEGY_IDS[0],
        expected_fit_plan_capability_sha256=(
            contract.fit_features.capability_sha256
        ),
        expected_source_capability_sha256=(
            contract.fit_features.capability_sha256
        ),
        expected_processor_receipt_sha256=(
            contract.receipt.processor_receipt_sha256
        ),
        expected_processed_output_receipt_sha256=(
            contract.processed.output_receipt_sha256
        ),
        expected_crossfit_plan_sha256=contract.plan.plan_sha256,
    )


@pytest.mark.parametrize("model_id", MODEL_IDS)
@pytest.mark.parametrize("model_namespace", (HISTORY_NAMESPACE, CURRENT_ONLY_NAMESPACE))
def test_six_restart_paths_are_prediction_equivalent_after_original_is_dropped(
    tmp_path: Path,
    production_contract: object,
    model_id: str,
    model_namespace: str,
) -> None:
    contract = production_contract
    if model_namespace == HISTORY_NAMESPACE:
        roster = _heldout_roster(contract)
        assert any(count == 0 for count in roster.context_counts)
        assert any(count > 0 for count in roster.context_counts)
        by_protocol = {
            int(value): index
            for index, value in enumerate(contract.processed.protocol_row_ids)
        }
        physical_queries = tuple(
            by_protocol[value] for value in roster.query_protocol_row_ids
        )
        assert physical_queries != tuple(range(len(physical_queries)))
        original = fit_history_model(
            model_id,
            contract.fit_training,
            contract.processed,
            contract.receipt,
            contract.plan,
            contract.rosters,
            contract.examples,
            **_history_kwargs(contract),
        )
        expected = predict_production_history(
            original,
            contract.fit_features,
            contract.fit_features,
            contract.processed,
            contract.receipt,
            contract.plan,
            roster,
        )
    else:
        roster = None
        original = fit_current_only_model(
            model_id,
            contract.fit_training,
            contract.processed,
            contract.receipt,
            contract.plan,
            contract.independence,
            **_current_kwargs(contract),
        )
        expected = predict_production_current_only(
            original,
            contract.fit_features,
            contract.fit_features,
            contract.processed,
            contract.receipt,
            contract.plan,
        )

    artifact = publish_checkpoint_artifact(tmp_path, original)
    receipt_path = artifact.receipt_path
    receipt_file_sha256 = artifact.receipt_file_sha256
    original_reference = weakref.ref(original)
    del original, artifact
    gc.collect()
    assert original_reference() is None

    loaded = load_checkpoint_artifact(
        receipt_path,
        expected_receipt_file_sha256=receipt_file_sha256,
    )
    if model_namespace == HISTORY_NAMESPACE:
        restored = restore_history_checkpoint_artifact(
            loaded,
            contract.fit_training,
            contract.processor,
            contract.processed,
            contract.plan,
            contract.rosters,
            contract.examples,
        )
        assert isinstance(restored, ProductionHistoryCheckpoint)
        observed = predict_production_history(
            restored,
            contract.fit_features,
            contract.fit_features,
            contract.processed,
            contract.receipt,
            contract.plan,
            roster,
        )
    else:
        restored = restore_current_only_checkpoint_artifact(
            loaded,
            contract.fit_training,
            contract.processor,
            contract.processed,
            contract.plan,
            contract.independence,
        )
        assert isinstance(restored, ProductionCurrentOnlyCheckpoint)
        observed = predict_production_current_only(
            restored,
            contract.fit_features,
            contract.fit_features,
            contract.processed,
            contract.receipt,
            contract.plan,
        )
    np.testing.assert_array_equal(observed, expected)
    if model_id == LINEAR_POOL_ID:
        with pytest.raises(HarmBenchCheckpointArtifactError, match="prediction-only"):
            restored.checkpoint._estimator.partial_fit(  # type: ignore[attr-defined]
                np.zeros((1, 1)), np.zeros(1)
            )
    else:
        assert restored.checkpoint._network.training is False  # type: ignore[attr-defined]
        assert all(
            parameter.requires_grad is False
            for parameter in restored.checkpoint._network.parameters()  # type: ignore[attr-defined]
        )


def test_restore_and_prediction_surfaces_have_no_free_identity_or_raw_indices() -> None:
    forbidden_restore = {
        "model_id",
        "model_namespace",
        "family_id",
        "num_classes",
        "class_order",
        "seed",
        "fold",
        "sha256",
    }
    for surface in (
        restore_history_checkpoint_artifact,
        restore_current_only_checkpoint_artifact,
    ):
        assert forbidden_restore.isdisjoint(inspect.signature(surface).parameters)
    assert {"contexts", "query_indices"}.isdisjoint(
        inspect.signature(predict_production_history).parameters
    )
    current_names = set(
        inspect.signature(predict_production_current_only).parameters
    )
    assert all("context" not in name and "history" not in name for name in current_names)


def test_restore_rejects_live_fit_processor_output_plan_and_context_tampering(
    tmp_path: Path,
    production_contract: object,
) -> None:
    contract = production_contract
    history = fit_history_model(
        LINEAR_POOL_ID,
        contract.fit_training,
        contract.processed,
        contract.receipt,
        contract.plan,
        contract.rosters,
        contract.examples,
        **_history_kwargs(contract),
    )
    history_root = tmp_path / "history"
    history_root.mkdir()
    history_artifact = publish_checkpoint_artifact(history_root, history)
    current = fit_current_only_model(
        LINEAR_POOL_ID,
        contract.fit_training,
        contract.processed,
        contract.receipt,
        contract.plan,
        contract.independence,
        **_current_kwargs(contract),
    )
    current_root = tmp_path / "current"
    current_root.mkdir()
    current_artifact = publish_checkpoint_artifact(current_root, current)

    bad_fit = replace(contract.fit_training, capability_sha256="0" * 64)
    with pytest.raises(HarmBenchCheckpointArtifactError, match="lineage"):
        restore_current_only_checkpoint_artifact(
            current_artifact,
            bad_fit,
            contract.processor,
            contract.processed,
            contract.plan,
            contract.independence,
        )
    bad_processed = replace(
        contract.processed, output_receipt_sha256="0" * 64
    )
    with pytest.raises(HarmBenchCheckpointArtifactError, match="lineage"):
        restore_current_only_checkpoint_artifact(
            current_artifact,
            contract.fit_training,
            contract.processor,
            bad_processed,
            contract.plan,
            contract.independence,
        )
    bad_plan = replace(contract.plan, plan_sha256="0" * 64)
    with pytest.raises(HarmBenchCheckpointArtifactError, match="lineage"):
        restore_current_only_checkpoint_artifact(
            current_artifact,
            contract.fit_training,
            contract.processor,
            contract.processed,
            bad_plan,
            contract.independence,
        )
    bad_processor = deepcopy(contract.processor)
    object.__setattr__(bad_processor.receipt, "fit_state_sha256", "0" * 64)
    with pytest.raises(HarmBenchCheckpointArtifactError, match="lineage"):
        restore_current_only_checkpoint_artifact(
            current_artifact,
            contract.fit_training,
            bad_processor,
            contract.processed,
            contract.plan,
            contract.independence,
        )
    bad_independence = deepcopy(contract.independence)
    object.__setattr__(bad_independence, "roster_sha256", "0" * 64)
    with pytest.raises(HarmBenchCheckpointArtifactError, match="independence"):
        restore_current_only_checkpoint_artifact(
            current_artifact,
            contract.fit_training,
            contract.processor,
            contract.processed,
            contract.plan,
            bad_independence,
        )
    bad_examples = replace(contract.examples, example_sha256="0" * 64)
    with pytest.raises(HarmBenchCheckpointArtifactError, match="context"):
        restore_history_checkpoint_artifact(
            history_artifact,
            contract.fit_training,
            contract.processor,
            contract.processed,
            contract.plan,
            contract.rosters,
            bad_examples,
        )


def test_typed_prediction_supports_feature_only_selection_without_label_surface(
    production_contract: object,
) -> None:
    contract = production_contract
    rows = 4
    selection_features = make_outcome_free_role_features(
        dataset_id="synthetic",
        role=SELECTION_ROLE,
        keys=[f"selection:dialogue:{index}" for index in range(rows)],
        texts=[f"unseen affect selection turn {index}" for index in range(rows)],
        audio=np.asarray(
            [[1.0, index + 1.0, index + 2.0, -1.0] for index in range(rows)],
            dtype=np.float32,
        ),
        video=np.asarray(
            [[2.0, index + 2.0, (index + 1.0) ** 2, 1.0] for index in range(rows)],
            dtype=np.float32,
        ),
        groups=["selection_dialogue"] * rows,
        speaker_identity=["selection_speaker"] * rows,
        turn_ids=np.arange(rows, dtype=np.int64),
        protocol_row_ids=np.arange(9_000, 9_000 + rows, dtype=np.int64),
        row_alignment_sha256="1" * 64,
        feature_sha256="2" * 64,
    )
    selection = make_synthetic_selection_feature_capability(
        selection_features=selection_features,
        manifest_sha256="3" * 64,
        synthetic_feature_projection_sha256=(
            contract.fit_features.cross_role_feature_roster_receipt
            .fit_feature_projection_sha256
        ),
    )
    processed = transform_role_features(
        contract.processor,
        selection,
        expected_processor_receipt_sha256=(
            contract.receipt.processor_receipt_sha256
        ),
        expected_fit_feature_capability_sha256=(
            contract.fit_features.capability_sha256
        ),
        expected_transform_source_capability_sha256=selection.capability_sha256,
        expected_crossfit_plan_sha256=contract.plan.plan_sha256,
        expected_seed=contract.seed,
        expected_fold=contract.fold,
    )
    roster = build_strict_past_context_roster(
        contract.fit_features,
        selection,
        processed,
        contract.receipt,
        contract.plan,
        training_seed=contract.seed,
        fold=contract.fold,
        context_role=SELECTION_CONTEXT_ROLE,
        strategy_id=STRICT_PAST_STRATEGY_IDS[0],
        expected_fit_plan_capability_sha256=(
            contract.fit_features.capability_sha256
        ),
        expected_source_capability_sha256=selection.capability_sha256,
        expected_processor_receipt_sha256=(
            contract.receipt.processor_receipt_sha256
        ),
        expected_processed_output_receipt_sha256=processed.output_receipt_sha256,
        expected_crossfit_plan_sha256=contract.plan.plan_sha256,
    )
    history = fit_history_model(
        LINEAR_POOL_ID,
        contract.fit_training,
        contract.processed,
        contract.receipt,
        contract.plan,
        contract.rosters,
        contract.examples,
        **_history_kwargs(contract),
    )
    current = fit_current_only_model(
        LINEAR_POOL_ID,
        contract.fit_training,
        contract.processed,
        contract.receipt,
        contract.plan,
        contract.independence,
        **_current_kwargs(contract),
    )
    history_probabilities = predict_production_history(
        history,
        contract.fit_features,
        selection,
        processed,
        contract.receipt,
        contract.plan,
        roster,
    )
    current_probabilities = predict_production_current_only(
        current,
        contract.fit_features,
        selection,
        processed,
        contract.receipt,
        contract.plan,
    )
    assert history_probabilities.shape == (rows, 3)
    assert current_probabilities.shape == (rows, 3)
    assert not hasattr(selection, "labels")


def _run_linear_subprocess_smoke() -> None:
    contract = production_contract.__wrapped__()
    original = fit_current_only_model(
        LINEAR_POOL_ID,
        contract.fit_training,
        contract.processed,
        contract.receipt,
        contract.plan,
        contract.independence,
        **_current_kwargs(contract),
    )
    expected = predict_production_current_only(
        original,
        contract.fit_features,
        contract.fit_features,
        contract.processed,
        contract.receipt,
        contract.plan,
    )
    with tempfile.TemporaryDirectory() as directory:
        artifact = publish_checkpoint_artifact(Path(directory), original)
        path = artifact.receipt_path
        digest = artifact.receipt_file_sha256
        del artifact, original
        gc.collect()
        loaded = load_checkpoint_artifact(
            path, expected_receipt_file_sha256=digest
        )
        restored = restore_current_only_checkpoint_artifact(
            loaded,
            contract.fit_training,
            contract.processor,
            contract.processed,
            contract.plan,
            contract.independence,
        )
        observed = predict_production_current_only(
            restored,
            contract.fit_features,
            contract.fit_features,
            contract.processed,
            contract.receipt,
            contract.plan,
        )
    np.testing.assert_array_equal(observed, expected)


def test_linear_restart_works_in_a_fresh_python_process() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(SOURCE), str(Path(__file__).parent))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from test_harmbench_erc_checkpoint_restore import "
                "_run_linear_subprocess_smoke; _run_linear_subprocess_smoke()"
            ),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
