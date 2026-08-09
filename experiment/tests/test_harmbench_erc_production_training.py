from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import textwrap

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import hva_affect.harmbench_erc_models as model_module  # noqa: E402
from hva_affect.emotiontalk_role_sidecar import FIT_ROLE  # noqa: E402
from hva_affect.harmbench_erc_contexts import (  # noqa: E402
    CURRENT_ONLY_STRATEGY_ID,
    FIT_TRAIN_CONTEXT_ROLE,
    STRICT_PAST_STRATEGY_IDS,
    build_strict_past_context_roster,
)
from hva_affect.harmbench_erc_crossfit import (  # noqa: E402
    make_context_training_examples,
    make_shared_group_crossfit_plan,
)
from hva_affect.harmbench_erc_models import (  # noqa: E402
    CURRENT_ONLY_NAMESPACE,
    HISTORY_NAMESPACE,
    LINEAR_POOL_ID,
    HarmBenchModelError,
    ProductionCurrentOnlyCheckpoint,
    ProductionHistoryCheckpoint,
    aggregate_context_roster_sha256,
    class_order_sha256,
    fit_current_only_model,
    fit_history_model,
)
from hva_affect.harmbench_erc_open_roles import (  # noqa: E402
    make_fit_role_capability,
    make_outcome_free_role_features,
    make_synthetic_fit_feature_capability,
)
from hva_affect.harmbench_erc_processors import (  # noqa: E402
    fit_shared_processor,
    transform_role_features,
)


FORBIDDEN_RAW_TRAINING_NAMES = frozenset(
    {
        "fit_synthetic_history_model",
        "fit_synthetic_current_only_model",
        "make_history_trainer",
        "make_current_only_trainer",
        "LinearHistoryTrainer",
        "LinearCurrentOnlyTrainer",
        "DeepSetsHistoryTrainer",
        "DeepSetsCurrentOnlyTrainer",
        "CausalGRUHistoryTrainer",
        "CausalGRUCurrentOnlyTrainer",
    }
)


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


@pytest.fixture(scope="module")
def production_contract() -> SimpleNamespace:
    groups: list[str] = []
    speakers: list[str] = []
    turns: list[int] = []
    keys: list[str] = []
    texts: list[str] = []
    audio: list[list[float]] = []
    video: list[list[float]] = []
    for group_index in range(5):
        group = f"dialogue_{group_index}"
        for local_index in range(5):
            row = len(groups)
            groups.append(group)
            speakers.append(f"{group}:speaker_{local_index % 2}")
            turns.append(local_index)
            keys.append(f"fit:{group}:{local_index}")
            texts.append(
                f"synthetic affect {group_index} turn {local_index} token_{row}"
            )
            audio.append(
                [
                    float(group_index + 1),
                    float(local_index + 1),
                    float(row + 1),
                    float((-1) ** row),
                ]
            )
            video.append(
                [
                    float(local_index + 2),
                    float(group_index + 2),
                    float((row + 1) ** 2),
                    float((-1) ** local_index),
                ]
            )
    rows = len(groups)
    features = make_outcome_free_role_features(
        dataset_id="synthetic",
        role=FIT_ROLE,
        keys=keys,
        texts=texts,
        audio=np.asarray(audio, dtype=np.float32),
        video=np.asarray(video, dtype=np.float32),
        groups=groups,
        speaker_identity=speakers,
        turn_ids=np.asarray(turns, dtype=np.int64),
        protocol_row_ids=np.arange(1_000, 1_000 + rows, dtype=np.int64),
        row_alignment_sha256="a" * 64,
        feature_sha256="b" * 64,
    )
    fit_features = make_synthetic_fit_feature_capability(
        fit_features=features,
        feature_manifest_sha256="c" * 64,
        synthetic_feature_projection_sha256="d" * 64,
    )
    label_order = ("negative", "neutral", "positive")
    fit_training = make_fit_role_capability(
        fit_feature_capability=fit_features,
        fit_labels=np.arange(rows, dtype=np.int64) % len(label_order),
        fit_label_sha256="e" * 64,
        label_order=label_order,
        fit_manifest_sha256="f" * 64,
    )
    plan = make_shared_group_crossfit_plan(fit_features)
    seed = plan.training_seed_ids[0]
    fold = 0
    processor = fit_shared_processor(fit_features, plan, seed=seed, fold=fold)
    receipt = processor.receipt
    processed = transform_role_features(
        processor,
        fit_features,
        expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
        expected_fit_feature_capability_sha256=fit_features.capability_sha256,
        expected_transform_source_capability_sha256=fit_features.capability_sha256,
        expected_crossfit_plan_sha256=plan.plan_sha256,
        expected_seed=seed,
        expected_fold=fold,
    )
    rosters = {}
    for strategy in STRICT_PAST_STRATEGY_IDS:
        rosters[strategy] = build_strict_past_context_roster(
            fit_features,
            fit_features,
            processed,
            receipt,
            plan,
            training_seed=seed,
            fold=fold,
            context_role=FIT_TRAIN_CONTEXT_ROLE,
            strategy_id=strategy,
            expected_fit_plan_capability_sha256=fit_features.capability_sha256,
            expected_source_capability_sha256=fit_features.capability_sha256,
            expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
            expected_processed_output_receipt_sha256=(
                processed.output_receipt_sha256
            ),
            expected_crossfit_plan_sha256=plan.plan_sha256,
        )
    roster_shas = {
        strategy: roster.roster_sha256 for strategy, roster in rosters.items()
    }
    examples = make_context_training_examples(
        rosters,
        fit_features,
        processed,
        receipt,
        plan,
        training_seed=seed,
        fold=fold,
        expected_fit_feature_capability_sha256=fit_features.capability_sha256,
        expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
        expected_processed_output_receipt_sha256=processed.output_receipt_sha256,
        expected_crossfit_plan_sha256=plan.plan_sha256,
        expected_context_roster_sha256_by_strategy=roster_shas,
    )
    independence = build_strict_past_context_roster(
        fit_features,
        fit_features,
        processed,
        receipt,
        plan,
        training_seed=seed,
        fold=fold,
        context_role=FIT_TRAIN_CONTEXT_ROLE,
        strategy_id=CURRENT_ONLY_STRATEGY_ID,
        expected_fit_plan_capability_sha256=fit_features.capability_sha256,
        expected_source_capability_sha256=fit_features.capability_sha256,
        expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
        expected_processed_output_receipt_sha256=processed.output_receipt_sha256,
        expected_crossfit_plan_sha256=plan.plan_sha256,
    )
    return SimpleNamespace(
        fit_features=fit_features,
        fit_training=fit_training,
        plan=plan,
        seed=seed,
        fold=fold,
        processor=processor,
        receipt=receipt,
        processed=processed,
        rosters=rosters,
        roster_shas=roster_shas,
        examples=examples,
        independence=independence,
    )


def _history_kwargs(contract: SimpleNamespace) -> dict[str, object]:
    return {
        "training_seed": contract.seed,
        "fold": contract.fold,
        "epochs": 1,
        "expected_fit_training_capability_sha256": (
            contract.fit_training.capability_sha256
        ),
        "expected_fit_feature_capability_sha256": (
            contract.fit_features.capability_sha256
        ),
        "expected_processor_receipt_sha256": (
            contract.receipt.processor_receipt_sha256
        ),
        "expected_processed_output_receipt_sha256": (
            contract.processed.output_receipt_sha256
        ),
        "expected_crossfit_plan_sha256": contract.plan.plan_sha256,
        "expected_context_roster_sha256_by_strategy": contract.roster_shas,
        "expected_context_training_examples_sha256": (
            contract.examples.example_sha256
        ),
    }


def _current_kwargs(contract: SimpleNamespace) -> dict[str, object]:
    result = _history_kwargs(contract)
    result.pop("expected_context_roster_sha256_by_strategy")
    result.pop("expected_context_training_examples_sha256")
    result["expected_independence_roster_sha256"] = (
        contract.independence.roster_sha256
    )
    return result


def test_production_fit_surfaces_do_not_accept_labels_or_class_count(
    production_contract: SimpleNamespace,
) -> None:
    assert FORBIDDEN_RAW_TRAINING_NAMES.isdisjoint(model_module.__all__)
    for surface in (fit_history_model, fit_current_only_model):
        names = set(inspect.signature(surface).parameters)
        assert "labels" not in names
        assert "num_classes" not in names
    with pytest.raises(TypeError):
        fit_history_model(
            LINEAR_POOL_ID,
            production_contract.fit_training,
            production_contract.processed,
            production_contract.receipt,
            production_contract.plan,
            production_contract.rosters,
            production_contract.examples,
            labels=np.zeros(1, dtype=np.int64),  # type: ignore[call-arg]
            **_history_kwargs(production_contract),
        )
    with pytest.raises(TypeError):
        fit_current_only_model(
            LINEAR_POOL_ID,
            production_contract.fit_training,
            production_contract.processed,
            production_contract.receipt,
            production_contract.plan,
            production_contract.independence,
            num_classes=99,  # type: ignore[call-arg]
            **_current_kwargs(production_contract),
        )


def test_production_facades_do_not_call_raw_or_synthetic_training_surfaces(
) -> None:
    expected_private_core = {
        "fit_history_model": "_fit_history_array_core",
        "fit_current_only_model": "_fit_current_only_array_core",
    }
    for surface in (fit_history_model, fit_current_only_model):
        tree = ast.parse(textwrap.dedent(inspect.getsource(surface)))
        calls = {
            name
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for name in [_call_name(node)]
            if name is not None
        }
        assert FORBIDDEN_RAW_TRAINING_NAMES.isdisjoint(calls)
        assert expected_private_core[surface.__name__] in calls


def test_production_scripts_cannot_import_or_reference_raw_training_surfaces(
) -> None:
    scripts_root = Path(__file__).resolve().parents[1] / "scripts"
    violations: list[str] = []
    for path in sorted(scripts_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name in FORBIDDEN_RAW_TRAINING_NAMES:
                        violations.append(
                            f"{path.name}:{node.lineno}:import:{alias.name}"
                        )
            elif (
                isinstance(node, ast.Attribute)
                and node.attr in FORBIDDEN_RAW_TRAINING_NAMES
            ):
                violations.append(f"{path.name}:{node.lineno}:attribute:{node.attr}")
            elif isinstance(node, ast.Call):
                name = _call_name(node)
                if name in FORBIDDEN_RAW_TRAINING_NAMES:
                    violations.append(f"{path.name}:{node.lineno}:call:{name}")
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in FORBIDDEN_RAW_TRAINING_NAMES
            ):
                violations.append(f"{path.name}:{node.lineno}:dynamic:{node.value}")
    assert violations == []


def test_history_labels_are_mechanically_derived_for_repeated_queries(
    production_contract: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    original = model_module._fit_history_array_core

    def capture(model_id, features, target, contexts, **kwargs):
        captured["target"] = np.asarray(target).copy()
        captured["query_indices"] = tuple(kwargs["query_indices"])
        captured["feature_rows"] = features.rows
        return original(model_id, features, target, contexts, **kwargs)

    monkeypatch.setattr(model_module, "_fit_history_array_core", capture)
    result = fit_history_model(
        LINEAR_POOL_ID,
        production_contract.fit_training,
        production_contract.processed,
        production_contract.receipt,
        production_contract.plan,
        production_contract.rosters,
        production_contract.examples,
        **_history_kwargs(production_contract),
    )
    assert isinstance(result, ProductionHistoryCheckpoint)
    assert result.model_namespace == HISTORY_NAMESPACE
    assert result.class_order == production_contract.fit_training.fit.label_order
    assert result.class_order_sha256 == class_order_sha256(
        result.class_order,
        dataset_id=result.dataset_id,
        fit_training_capability_sha256=result.fit_training_capability_sha256,
    )
    assert result.context_roster_manifest_sha256 == (
        aggregate_context_roster_sha256(
            production_contract.examples.context_roster_sha256_by_strategy
        )
    )
    query_indices = captured["query_indices"]
    by_protocol = {
        int(value): index
        for index, value in enumerate(production_contract.processed.protocol_row_ids)
    }
    full_query_indices = tuple(
        by_protocol[value]
        for value in production_contract.examples.query_protocol_row_ids
    )
    expected = production_contract.fit_training.fit.labels[
        np.asarray(full_query_indices, dtype=np.int64)
    ]
    np.testing.assert_array_equal(captured["target"], expected)
    repeated = {
        query for query in query_indices if query_indices.count(query) > 1
    }
    assert repeated
    for query in repeated:
        observed = {
            int(expected[index])
            for index, value in enumerate(query_indices)
            if value == query
        }
        positions = [index for index, value in enumerate(query_indices) if value == query]
        assert observed == {int(expected[positions[0]])}
    train = set(
        production_contract.plan.train_indices(
            production_contract.seed,
            production_contract.fold,
            fit_capability=production_contract.fit_features,
        ).tolist()
    )
    assert captured["feature_rows"] == len(train)
    assert captured["feature_rows"] < production_contract.processed.rows
    assert set(query_indices).issubset(set(range(len(train))))


def test_current_only_fit_binds_class_order_and_zero_consumption(
    production_contract: SimpleNamespace,
) -> None:
    result = fit_current_only_model(
        LINEAR_POOL_ID,
        production_contract.fit_training,
        production_contract.processed,
        production_contract.receipt,
        production_contract.plan,
        production_contract.independence,
        **_current_kwargs(production_contract),
    )
    assert isinstance(result, ProductionCurrentOnlyCheckpoint)
    assert result.model_namespace == CURRENT_ONLY_NAMESPACE
    assert result.class_order == production_contract.fit_training.fit.label_order
    assert result.class_order_sha256 == class_order_sha256(
        result.class_order,
        dataset_id=result.dataset_id,
        fit_training_capability_sha256=result.fit_training_capability_sha256,
    )
    assert result.context_count == 0
    assert result.history_consumption_count == 0
    assert result.independence_roster_sha256 == (
        production_contract.independence.roster_sha256
    )
    with pytest.raises(HarmBenchModelError, match="seed differs"):
        replace(result, training_seed=production_contract.plan.training_seed_ids[1])


def test_production_fit_rejects_stale_authority_and_wrong_independence_proof(
    production_contract: SimpleNamespace,
) -> None:
    bad = _history_kwargs(production_contract)
    bad["expected_fit_training_capability_sha256"] = "0" * 64
    with pytest.raises(HarmBenchModelError, match="external binding"):
        fit_history_model(
            LINEAR_POOL_ID,
            production_contract.fit_training,
            production_contract.processed,
            production_contract.receipt,
            production_contract.plan,
            production_contract.rosters,
            production_contract.examples,
            **bad,
        )
    with pytest.raises(HarmBenchModelError, match="independence proof"):
        fit_current_only_model(
            LINEAR_POOL_ID,
            production_contract.fit_training,
            production_contract.processed,
            production_contract.receipt,
            production_contract.plan,
            production_contract.rosters[STRICT_PAST_STRATEGY_IDS[0]],
            **_current_kwargs(production_contract),
        )
