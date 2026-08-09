from __future__ import annotations

from dataclasses import replace
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.harmbench_erc_contract import (  # noqa: E402
    load_development_protocol,
    validate_development_protocol,
)
from hva_affect.harmbench_erc_inference import (  # noqa: E402
    bind_production_probability_panel,
    bootstrap_cell_metrics,
    bootstrap_paired_strategy_contrast,
    factorize_cluster_keys,
    frozen_inference_spec,
    make_production_shared_cluster_bootstrap_plan,
    make_shared_cluster_bootstrap_plan,
    probability_panel_sha256,
    sampled_query_indices,
    validate_production_probability_panel,
    validate_production_shared_plan,
    validate_shared_plan,
)
from hva_affect.harmbench_erc_metrics import HarmBenchMetricError  # noqa: E402


DATASET = "synthetic_dialogues"
CONFIG = ROOT / "configs" / "harmbench_erc_v1_draft.json"


def panel_fixture() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    labels = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)
    current_one = np.asarray(
        [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.4, 0.6],
            [0.55, 0.45],
            [0.7, 0.3],
        ]
    )
    candidate_one = np.asarray(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.3, 0.7],
        ]
    )
    current = np.stack([current_one, current_one * 0.98 + 0.01], axis=0)
    candidate = np.stack([candidate_one, candidate_one * 0.98 + 0.01], axis=0)
    eligible = np.asarray([False, True, True, True, True, True])
    clusters = np.asarray(["a", "a", "b", "b", "c", "c"], dtype=object)
    row_ids = np.arange(len(labels), dtype=np.int64)
    return labels, current, candidate, eligible, clusters, row_ids


def plan_fixture(random_seed: int = 19):
    _, _, _, _, clusters, row_ids = panel_fixture()
    return make_shared_cluster_bootstrap_plan(
        DATASET,
        row_ids,
        clusters,
        training_seed_count=2,
        replicates=100,
        random_seed=random_seed,
    )


def production_fixture():
    labels, current, candidate, eligible, clusters, row_ids = panel_fixture()
    scales = (1.0, 0.99, 0.98, 0.97, 0.96)
    current = np.stack(
        [current[0] * scale + (1.0 - scale) / current.shape[-1] for scale in scales]
    )
    candidate = np.stack(
        [candidate[0] * scale + (1.0 - scale) / candidate.shape[-1] for scale in scales]
    )
    contract = load_development_protocol(CONFIG)
    plan = make_production_shared_cluster_bootstrap_plan(
        contract,
        DATASET,
        row_ids,
        clusters,
        training_seed_ids=[17, 29, 43, 71, 101],
    )
    return contract, plan, labels, current, candidate, eligible, clusters, row_ids


def test_composite_clusters_cannot_collide_through_delimiters_or_types() -> None:
    values = np.asarray(
        [["x/y", "z"], ["x", "y/z"], ["x/y", "z"], ["x", "y/z"]],
        dtype=object,
    )
    codes, members = factorize_cluster_keys(values)
    assert len(members) == 2
    assert np.array_equal(codes, np.asarray([0, 1, 0, 1]))
    typed_codes, typed_members = factorize_cluster_keys(
        np.asarray([True, 1, 1.0], dtype=object)
    )
    assert len(typed_members) == 3
    assert np.array_equal(typed_codes, np.asarray([0, 1, 2]))


def test_shared_plan_is_seed_deterministic_and_keeps_clusters_whole() -> None:
    first = plan_fixture(19)
    second = plan_fixture(19)
    assert first.alignment_sha256 == second.alignment_sha256
    assert first.plan_sha256 == second.plan_sha256
    assert np.array_equal(first.seed_draws, second.seed_draws)
    assert np.array_equal(first.cluster_draws, second.cluster_draws)
    sampled = sampled_query_indices(first, 0)
    for code in np.unique(first.cluster_draws[0]):
        member = first.cluster_members[int(code)]
        assert np.sum(np.isin(sampled, member)) % len(member) == 0


def test_current_against_itself_has_exact_zero_point_and_interval() -> None:
    labels, current, _, eligible, clusters, row_ids = panel_fixture()
    plan = plan_fixture(23)
    report = bootstrap_cell_metrics(
        DATASET,
        row_ids,
        clusters,
        labels,
        current,
        current,
        eligible,
        selected=eligible,
        plan=plan,
    )
    for endpoint in (
        "delta_macro_f1",
        "delta_accuracy",
        "delta_mean_nll",
        "population_mean_regret",
        "population_p90_regret",
        "population_cvar90_regret",
        "population_harm_rate_gt_0_05",
    ):
        assert report["point"][endpoint] == pytest.approx(0.0)
        assert report["bootstrap"][endpoint]["ci95_low"] == pytest.approx(0.0)
        assert report["bootstrap"][endpoint]["ci95_high"] == pytest.approx(0.0)
    assert report["alignment_contract"]["bootstrap_plan_sha256"] == plan.plan_sha256


def test_cell_bootstrap_reports_candidate_improvement_without_seed_vectors() -> None:
    labels, current, candidate, eligible, clusters, row_ids = panel_fixture()
    plan = plan_fixture(29)
    report = bootstrap_cell_metrics(
        DATASET,
        row_ids,
        clusters,
        labels,
        current,
        candidate,
        eligible,
        selected=eligible,
        plan=plan,
    )
    assert report["point"]["delta_macro_f1"] > 0.0
    assert report["point"]["population_mean_regret"] < 0.0
    assert report["point"]["population_harm_rate_gt_0_05"] >= 0.0
    assert report["bootstrap"]["population_harm_rate_gt_0_05"]["finite_fraction"] == 1.0
    assert report["inference_contract"]["cluster_count"] == 3
    assert "seed_results" not in str(report)


def test_paired_contrast_reuses_draws_and_returns_left_minus_right() -> None:
    labels, current, candidate, eligible, clusters, row_ids = panel_fixture()
    plan = plan_fixture(31)
    contrast = bootstrap_paired_strategy_contrast(
        DATASET,
        row_ids,
        clusters,
        labels,
        current,
        candidate,
        eligible,
        current,
        eligible,
        eligible,
        plan,
    )
    assert contrast["point"]["delta_macro_f1"] > 0.0
    assert contrast["point"]["population_mean_regret"] < 0.0
    assert contrast["paired_on"] == "same_training_seed_draw_and_whole_cluster_draw"
    assert contrast["alignment_contract"]["bootstrap_plan_sha256"] == plan.plan_sha256


def test_same_length_row_reordering_is_rejected_by_alignment_digest() -> None:
    labels, current, candidate, eligible, clusters, row_ids = panel_fixture()
    plan = plan_fixture(37)
    with pytest.raises(HarmBenchMetricError, match="alignment differs"):
        bootstrap_cell_metrics(
            DATASET,
            row_ids[::-1],
            clusters,
            labels,
            current,
            candidate,
            eligible,
            eligible,
            plan,
        )


def test_forged_or_malformed_plan_is_rejected() -> None:
    plan = plan_fixture(41)
    bad_seed_draws = np.array(plan.seed_draws, copy=True)
    bad_seed_draws[0, 0] = plan.training_seed_count
    with pytest.raises(HarmBenchMetricError, match="out-of-range"):
        validate_shared_plan(replace(plan, seed_draws=bad_seed_draws))
    overlap = (
        np.asarray([0, 1]),
        np.asarray([1, 2]),
        np.asarray([3, 4, 5]),
    )
    with pytest.raises(HarmBenchMetricError, match="disjoint"):
        validate_shared_plan(replace(plan, cluster_members=overlap))


def test_too_many_no_history_bootstrap_replicates_fail_closed() -> None:
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    current_one = np.asarray([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])
    candidate_one = np.asarray([[0.7, 0.3], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6]])
    current = np.stack([current_one, current_one], axis=0)
    candidate = np.stack([candidate_one, candidate_one], axis=0)
    clusters = np.asarray(["eligible", "eligible", "none", "none"], dtype=object)
    row_ids = np.arange(4, dtype=np.int64)
    eligible = np.asarray([True, True, False, False])
    plan = make_shared_cluster_bootstrap_plan(
        "sparse_history",
        row_ids,
        clusters,
        training_seed_count=2,
        replicates=100,
        random_seed=43,
    )
    with pytest.raises(HarmBenchMetricError, match="finite bootstrap fraction"):
        bootstrap_cell_metrics(
            "sparse_history",
            row_ids,
            clusters,
            labels,
            current,
            candidate,
            eligible,
            eligible,
            plan,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_seed_count", 2.5),
        ("replicates", 100.5),
        ("random_seed", 7.1),
    ],
)
def test_plan_integer_controls_do_not_silently_truncate(field: str, value: float) -> None:
    _, _, _, _, clusters, row_ids = panel_fixture()
    kwargs = {"training_seed_count": 2, "replicates": 100, "random_seed": 7}
    kwargs[field] = value
    with pytest.raises(HarmBenchMetricError, match="exact integer"):
        make_shared_cluster_bootstrap_plan(DATASET, row_ids, clusters, **kwargs)


def test_production_plan_is_contract_bound_and_has_no_runtime_controls() -> None:
    contract, plan, _, _, _, _, _, _ = production_fixture()
    spec = frozen_inference_spec(contract)
    validated = validate_production_shared_plan(contract, plan)
    assert validated.shared_plan.training_seed_count == 5
    assert validated.shared_plan.replicates == 10000
    assert validated.shared_plan.random_seed == 20260810
    assert validated.training_seed_ids == (17, 29, 43, 71, 101)
    assert validated.protocol_sha256 == spec.protocol_sha256
    assert validated.inference_spec_sha256 == spec.spec_sha256
    with pytest.raises(HarmBenchMetricError, match="generic bootstrap plans"):
        validate_production_shared_plan(contract, plan_fixture())


@pytest.mark.parametrize(
    "seed_ids",
    [
        [17, 29, 43, 71],
        [17, 29, 43, 71, 101, 103],
        [29, 17, 43, 71, 101],
        [17, 29, 43, 71, 103],
    ],
)
def test_production_plan_rejects_seed_identity_or_order_drift(seed_ids: list[int]) -> None:
    _, _, _, _, _, _, clusters, row_ids = production_fixture()
    contract = load_development_protocol(CONFIG)
    with pytest.raises(HarmBenchMetricError, match="seed identities or order"):
        make_production_shared_cluster_bootstrap_plan(
            contract,
            DATASET,
            row_ids,
            clusters,
            training_seed_ids=seed_ids,
        )


def test_semantically_valid_but_unpinned_protocol_cannot_enter_production() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["research_question"] += " Changed after the production pin."
    changed = validate_development_protocol(copy.deepcopy(payload))
    with pytest.raises(HarmBenchMetricError, match="not the pinned production draft"):
        frozen_inference_spec(changed)


def test_probability_tensor_is_bound_to_row_and_seed_order() -> None:
    contract, plan, _, current, _, _, clusters, row_ids = production_fixture()
    panel = bind_production_probability_panel(
        contract,
        plan,
        row_ids,
        clusters,
        model_id="model_a",
        strategy_id="independent_current_only",
        values=current,
        expected_array_sha256=probability_panel_sha256(current),
    )
    assert validate_production_probability_panel(
        contract, plan, row_ids, clusters, panel
    ) is panel

    with pytest.raises(HarmBenchMetricError, match="tensor SHA-256"):
        validate_production_probability_panel(
            contract,
            plan,
            row_ids,
            clusters,
            replace(panel, values=panel.values[:, ::-1, :]),
        )
    with pytest.raises(HarmBenchMetricError, match="tensor SHA-256"):
        validate_production_probability_panel(
            contract,
            plan,
            row_ids,
            clusters,
            replace(panel, values=panel.values[::-1]),
        )
    changed = np.array(panel.values, copy=True)
    changed[0, 0] = changed[0, 0][::-1]
    with pytest.raises(HarmBenchMetricError, match="tensor SHA-256"):
        validate_production_probability_panel(
            contract, plan, row_ids, clusters, replace(panel, values=changed)
        )
    with pytest.raises(HarmBenchMetricError, match="binding changed"):
        validate_production_probability_panel(
            contract, plan, row_ids[::-1], clusters, panel
        )


def test_probability_builder_requires_producer_pinned_array_hash() -> None:
    contract, plan, _, current, _, _, clusters, row_ids = production_fixture()
    reordered = current[:, ::-1, :]
    with pytest.raises(HarmBenchMetricError, match="producer-pinned"):
        bind_production_probability_panel(
            contract,
            plan,
            row_ids,
            clusters,
            model_id="model_a",
            strategy_id="independent_current_only",
            values=reordered,
            expected_array_sha256=probability_panel_sha256(current),
        )
