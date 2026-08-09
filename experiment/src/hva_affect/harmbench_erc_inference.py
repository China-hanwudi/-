"""Shared seed-by-cluster inference for HarmBench-ERC.

Every model and strategy on one dataset must reuse the same validated
``SharedClusterBootstrapPlan``.  The plan is bound to the dataset and ordered
row/cluster alignment, preserving paired comparisons while keeping complete
dialogue/person clusters together.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

import numpy as np

from .harmbench_erc_contract import (
    DevelopmentProtocolContract,
    EXPECTED_BOOTSTRAP_REPLICATES,
    EXPECTED_BOOTSTRAP_SEED,
    EXPECTED_CONFIDENCE_INTERVAL,
    EXPECTED_HARM_THRESHOLDS,
    EXPECTED_MINIMUM_FINITE_BOOTSTRAP_FRACTION,
    EXPECTED_NLL_PROBABILITY_FLOOR,
    EXPECTED_TAIL_ALPHA,
    EXPECTED_TRAINING_SEEDS,
    canonical_protocol_bytes,
)
from .harmbench_erc_metrics import (
    HarmBenchMetricError,
    NLL_EPSILON,
    DEFAULT_HARM_THRESHOLDS,
    DEFAULT_TAIL_ALPHA,
    classification_metrics,
    ensure_finite_public_tree,
    evaluate_frozen_policy,
    hybrid_probability,
    validated_labels,
    validated_probability,
)


MINIMUM_BOOTSTRAP_REPLICATES = 100
MINIMUM_FINITE_BOOTSTRAP_FRACTION = 0.95
PINNED_DEVELOPMENT_PROTOCOL_SHA256 = (
    "0e30c634ac3b0ebd10b3238ad307d9cb276b7691d108d1cb30b9c5867a699512"
)
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CELL_ENDPOINTS = (
    "delta_macro_f1",
    "delta_accuracy",
    "delta_mean_nll",
    "delta_mean_brier",
    "coverage",
    "population_mean_regret",
    "population_p90_regret",
    "population_cvar90_regret",
    "population_harm_rate_gt_0",
    "population_harm_rate_gt_0_05",
    "conditional_mean_regret",
    "conditional_p90_regret",
    "conditional_cvar90_regret",
    "conditional_harm_rate_gt_0",
    "conditional_harm_rate_gt_0_05",
    "eligible_break_rate",
    "eligible_rescue_rate",
    "selected_break_rate",
    "selected_rescue_rate",
)


@dataclass(frozen=True)
class SharedClusterBootstrapPlan:
    dataset_id: str
    alignment_sha256: str
    plan_sha256: str
    random_seed: int
    replicates: int
    training_seed_count: int
    query_count: int
    cluster_count: int
    seed_draws: np.ndarray
    cluster_draws: np.ndarray
    cluster_members: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class FrozenInferenceSpec:
    protocol_sha256: str
    spec_sha256: str
    training_seed_ids: tuple[int, ...]
    replicates: int
    bootstrap_seed: int
    minimum_finite_bootstrap_fraction: float
    nll_probability_floor: float
    harm_thresholds_nats: tuple[float, ...]
    tail_alpha: float
    confidence_interval: str


@dataclass(frozen=True)
class ProductionSharedClusterBootstrapPlan:
    protocol_sha256: str
    inference_spec_sha256: str
    training_seed_ids: tuple[int, ...]
    shared_plan: SharedClusterBootstrapPlan
    binding_sha256: str


@dataclass(frozen=True)
class ProductionProbabilityPanel:
    protocol_sha256: str
    inference_spec_sha256: str
    bootstrap_plan_sha256: str
    dataset_id: str
    alignment_sha256: str
    training_seed_ids: tuple[int, ...]
    model_id: str
    strategy_id: str
    shape: tuple[int, int, int]
    array_sha256: str
    binding_sha256: str
    values: np.ndarray


def _exact_integer(value: object, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise HarmBenchMetricError(f"{name} must be an exact integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise HarmBenchMetricError(f"{name} must be at least {minimum}")
    return result


def _dataset_identifier(value: object) -> str:
    if not isinstance(value, str) or not IDENTIFIER_PATTERN.fullmatch(value):
        raise HarmBenchMetricError("dataset_id must be a short opaque identifier")
    return value


def _opaque_identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_PATTERN.fullmatch(value):
        raise HarmBenchMetricError(f"{name} must be a short opaque identifier")
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise HarmBenchMetricError(f"{name} must be a lowercase SHA-256")
    return value


def _canonical_digest(value: dict[str, object]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def frozen_inference_spec(contract: object) -> FrozenInferenceSpec:
    """Derive the sole production inference spec from the pinned draft contract."""

    if not isinstance(contract, DevelopmentProtocolContract):
        raise HarmBenchMetricError("production inference requires a validated protocol")
    observed_protocol_sha = hashlib.sha256(
        canonical_protocol_bytes(contract.payload)
    ).hexdigest()
    if observed_protocol_sha != contract.canonical_sha256:
        raise HarmBenchMetricError("validated protocol payload/hash binding changed")
    if observed_protocol_sha != PINNED_DEVELOPMENT_PROTOCOL_SHA256:
        raise HarmBenchMetricError("protocol is not the pinned production draft")
    roster = contract.payload["model_roster_gate"]
    paired = contract.payload["paired_estimands"]
    metric = contract.payload["metric_contract"]
    inference = contract.payload["inference_contract"]
    training_seed_ids = tuple(roster["training_seeds"])
    descriptor = {
        "protocol_sha256": observed_protocol_sha,
        "training_seed_ids": list(training_seed_ids),
        "replicates": inference["bootstrap_replicates"],
        "bootstrap_seed": inference["bootstrap_seed"],
        "minimum_finite_bootstrap_fraction": inference[
            "minimum_finite_bootstrap_fraction"
        ],
        "nll_probability_floor": paired["probability_floor_for_nll"],
        "harm_thresholds_nats": [
            paired["primary_harm_threshold_nats"],
            paired["practical_harm_sensitivity_threshold_nats"],
        ],
        "tail_alpha": metric["tail_alpha"],
        "confidence_interval": inference["confidence_interval"],
    }
    if (
        training_seed_ids != tuple(EXPECTED_TRAINING_SEEDS)
        or descriptor["replicates"] != EXPECTED_BOOTSTRAP_REPLICATES
        or descriptor["bootstrap_seed"] != EXPECTED_BOOTSTRAP_SEED
        or descriptor["minimum_finite_bootstrap_fraction"]
        != EXPECTED_MINIMUM_FINITE_BOOTSTRAP_FRACTION
        or descriptor["nll_probability_floor"] != EXPECTED_NLL_PROBABILITY_FLOOR
        or tuple(descriptor["harm_thresholds_nats"]) != EXPECTED_HARM_THRESHOLDS
        or descriptor["tail_alpha"] != EXPECTED_TAIL_ALPHA
        or descriptor["confidence_interval"] != EXPECTED_CONFIDENCE_INTERVAL
        or NLL_EPSILON != EXPECTED_NLL_PROBABILITY_FLOOR
        or tuple(DEFAULT_HARM_THRESHOLDS) != EXPECTED_HARM_THRESHOLDS
        or DEFAULT_TAIL_ALPHA != EXPECTED_TAIL_ALPHA
        or MINIMUM_FINITE_BOOTSTRAP_FRACTION
        != EXPECTED_MINIMUM_FINITE_BOOTSTRAP_FRACTION
    ):
        raise HarmBenchMetricError("runtime constants differ from the frozen protocol")
    return FrozenInferenceSpec(
        protocol_sha256=observed_protocol_sha,
        spec_sha256=_canonical_digest(descriptor),
        training_seed_ids=training_seed_ids,
        replicates=EXPECTED_BOOTSTRAP_REPLICATES,
        bootstrap_seed=EXPECTED_BOOTSTRAP_SEED,
        minimum_finite_bootstrap_fraction=EXPECTED_MINIMUM_FINITE_BOOTSTRAP_FRACTION,
        nll_probability_floor=EXPECTED_NLL_PROBABILITY_FLOOR,
        harm_thresholds_nats=EXPECTED_HARM_THRESHOLDS,
        tail_alpha=EXPECTED_TAIL_ALPHA,
        confidence_interval=EXPECTED_CONFIDENCE_INTERVAL,
    )


def _typed_scalar(value: object, *, name: str) -> tuple[str, str]:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        raise HarmBenchMetricError(f"{name} contains a missing component")
    if isinstance(value, bool):
        return ("bool", "true" if value else "false")
    if isinstance(value, int):
        return ("int", str(value))
    if isinstance(value, float):
        if not np.isfinite(value):
            raise HarmBenchMetricError(f"{name} contains a non-finite component")
        return ("float", value.hex())
    if isinstance(value, str):
        if len(value) > 512:
            raise HarmBenchMetricError(f"{name} contains an overlong component")
        return ("str", value)
    raise HarmBenchMetricError(
        f"{name} contains unsupported component type {type(value).__name__}"
    )


def _typed_rows(values: object, *, name: str) -> list[tuple[tuple[str, str], ...]]:
    raw = np.asarray(values, dtype=object)
    if raw.ndim == 1:
        rows = [(_typed_scalar(value, name=name),) for value in raw.tolist()]
    elif raw.ndim == 2 and raw.shape[1] >= 1:
        rows = [
            tuple(_typed_scalar(value, name=name) for value in row)
            for row in raw.tolist()
        ]
    else:
        raise HarmBenchMetricError(f"{name} must be a non-empty one- or two-dimensional array")
    if not rows:
        raise HarmBenchMetricError(f"{name} must contain at least one query")
    return rows


def factorize_cluster_keys(values: object) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Factorize typed scalar/composite keys without delimiter or type collisions."""

    rows = _typed_rows(values, name="clusters")
    mapping: dict[tuple[tuple[str, str], ...], int] = {}
    codes = np.empty(len(rows), dtype=np.int64)
    for index, key in enumerate(rows):
        if key not in mapping:
            mapping[key] = len(mapping)
        codes[index] = mapping[key]
    members = tuple(np.flatnonzero(codes == code) for code in range(len(mapping)))
    return codes, members


def alignment_sha256(
    dataset_id: object, protocol_row_ids: object, clusters: object
) -> str:
    dataset = _dataset_identifier(dataset_id)
    row_keys = _typed_rows(protocol_row_ids, name="protocol_row_ids")
    if any(len(key) != 1 for key in row_keys):
        raise HarmBenchMetricError("protocol_row_ids must be a one-dimensional vector")
    if len(set(row_keys)) != len(row_keys):
        raise HarmBenchMetricError("protocol_row_ids must be unique")
    cluster_keys = _typed_rows(clusters, name="clusters")
    if len(cluster_keys) != len(row_keys):
        raise HarmBenchMetricError("protocol_row_ids and clusters have different lengths")
    payload = {
        "dataset_id": dataset,
        "ordered_protocol_row_ids": row_keys,
        "ordered_cluster_keys": cluster_keys,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plan_digest(
    *,
    dataset_id: str,
    alignment: str,
    random_seed: int,
    replicates: int,
    training_seed_count: int,
    query_count: int,
    cluster_count: int,
    seed_draws: np.ndarray,
    cluster_draws: np.ndarray,
    cluster_members: tuple[np.ndarray, ...],
) -> str:
    descriptor = {
        "dataset_id": dataset_id,
        "alignment_sha256": alignment,
        "random_seed": random_seed,
        "replicates": replicates,
        "training_seed_count": training_seed_count,
        "query_count": query_count,
        "cluster_count": cluster_count,
    }
    digest = hashlib.sha256(
        json.dumps(
            descriptor,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    digest.update(np.asarray(seed_draws, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(cluster_draws, dtype="<i8").tobytes(order="C"))
    for member in cluster_members:
        encoded = np.asarray(member, dtype="<i8")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded.tobytes(order="C"))
    return digest.hexdigest()


def make_shared_cluster_bootstrap_plan(
    dataset_id: object,
    protocol_row_ids: object,
    clusters: object,
    *,
    training_seed_count: int,
    replicates: int,
    random_seed: int,
) -> SharedClusterBootstrapPlan:
    dataset = _dataset_identifier(dataset_id)
    seeds = _exact_integer(training_seed_count, name="training_seed_count", minimum=2)
    repetitions = _exact_integer(
        replicates, name="replicates", minimum=MINIMUM_BOOTSTRAP_REPLICATES
    )
    rng_seed = _exact_integer(random_seed, name="random_seed", minimum=0)
    alignment = alignment_sha256(dataset, protocol_row_ids, clusters)
    _, members = factorize_cluster_keys(clusters)
    if len(members) < 2:
        raise HarmBenchMetricError("at least two independent clusters are required")
    query_count = int(sum(len(member) for member in members))
    row_ids = np.asarray(protocol_row_ids, dtype=object)
    if row_ids.ndim != 1 or len(row_ids) != query_count:
        raise HarmBenchMetricError("protocol_row_ids must align with all cluster rows")
    rng = np.random.default_rng(rng_seed)
    seed_draws = rng.integers(
        0, seeds, size=(repetitions, seeds), dtype=np.int64
    )
    cluster_draws = rng.integers(
        0, len(members), size=(repetitions, len(members)), dtype=np.int64
    )
    digest = _plan_digest(
        dataset_id=dataset,
        alignment=alignment,
        random_seed=rng_seed,
        replicates=repetitions,
        training_seed_count=seeds,
        query_count=query_count,
        cluster_count=len(members),
        seed_draws=seed_draws,
        cluster_draws=cluster_draws,
        cluster_members=members,
    )
    seed_draws.setflags(write=False)
    cluster_draws.setflags(write=False)
    for member in members:
        member.setflags(write=False)
    plan = SharedClusterBootstrapPlan(
        dataset_id=dataset,
        alignment_sha256=alignment,
        plan_sha256=digest,
        random_seed=rng_seed,
        replicates=repetitions,
        training_seed_count=seeds,
        query_count=query_count,
        cluster_count=len(members),
        seed_draws=seed_draws,
        cluster_draws=cluster_draws,
        cluster_members=members,
    )
    validate_shared_plan(plan)
    return plan


def validate_shared_plan(plan: object) -> SharedClusterBootstrapPlan:
    if not isinstance(plan, SharedClusterBootstrapPlan):
        raise HarmBenchMetricError("bootstrap plan must use the validated typed contract")
    dataset = _dataset_identifier(plan.dataset_id)
    if not SHA256_PATTERN.fullmatch(plan.alignment_sha256) or not SHA256_PATTERN.fullmatch(
        plan.plan_sha256
    ):
        raise HarmBenchMetricError("bootstrap plan digest is malformed")
    seed = _exact_integer(plan.random_seed, name="random_seed", minimum=0)
    replicates = _exact_integer(
        plan.replicates, name="replicates", minimum=MINIMUM_BOOTSTRAP_REPLICATES
    )
    seed_count = _exact_integer(
        plan.training_seed_count, name="training_seed_count", minimum=2
    )
    query_count = _exact_integer(plan.query_count, name="query_count", minimum=1)
    cluster_count = _exact_integer(plan.cluster_count, name="cluster_count", minimum=2)
    if not isinstance(plan.seed_draws, np.ndarray) or plan.seed_draws.dtype.kind not in "iu":
        raise HarmBenchMetricError("seed draws must be an integer ndarray")
    if plan.seed_draws.shape != (replicates, seed_count):
        raise HarmBenchMetricError("seed draw shape changed")
    if np.any(plan.seed_draws < 0) or np.any(plan.seed_draws >= seed_count):
        raise HarmBenchMetricError("seed draw contains an out-of-range index")
    if not isinstance(plan.cluster_draws, np.ndarray) or plan.cluster_draws.dtype.kind not in "iu":
        raise HarmBenchMetricError("cluster draws must be an integer ndarray")
    if plan.cluster_draws.shape != (replicates, cluster_count):
        raise HarmBenchMetricError("cluster draw shape changed")
    if np.any(plan.cluster_draws < 0) or np.any(plan.cluster_draws >= cluster_count):
        raise HarmBenchMetricError("cluster draw contains an out-of-range index")
    if not isinstance(plan.cluster_members, tuple) or len(plan.cluster_members) != cluster_count:
        raise HarmBenchMetricError("cluster member roster changed")
    normalized_members: list[np.ndarray] = []
    for member in plan.cluster_members:
        if (
            not isinstance(member, np.ndarray)
            or member.ndim != 1
            or not len(member)
            or member.dtype.kind not in "iu"
        ):
            raise HarmBenchMetricError("cluster members must be non-empty integer vectors")
        if np.any(member < 0) or np.any(member >= query_count):
            raise HarmBenchMetricError("cluster member contains an out-of-range query")
        normalized_members.append(np.asarray(member, dtype=np.int64))
    cover = np.concatenate(normalized_members)
    if len(cover) != query_count or not np.array_equal(
        np.sort(cover), np.arange(query_count, dtype=np.int64)
    ):
        raise HarmBenchMetricError("cluster members must be disjoint and exactly cover queries")
    expected = _plan_digest(
        dataset_id=dataset,
        alignment=plan.alignment_sha256,
        random_seed=seed,
        replicates=replicates,
        training_seed_count=seed_count,
        query_count=query_count,
        cluster_count=cluster_count,
        seed_draws=plan.seed_draws,
        cluster_draws=plan.cluster_draws,
        cluster_members=tuple(normalized_members),
    )
    if expected != plan.plan_sha256:
        raise HarmBenchMetricError("bootstrap plan digest mismatch")
    return plan


def _production_plan_binding(
    *,
    spec: FrozenInferenceSpec,
    training_seed_ids: tuple[int, ...],
    shared_plan_sha256: str,
) -> str:
    return _canonical_digest(
        {
            "protocol_sha256": spec.protocol_sha256,
            "inference_spec_sha256": spec.spec_sha256,
            "training_seed_ids": list(training_seed_ids),
            "shared_plan_sha256": shared_plan_sha256,
        }
    )


def make_production_shared_cluster_bootstrap_plan(
    contract: object,
    dataset_id: object,
    protocol_row_ids: object,
    clusters: object,
    *,
    training_seed_ids: object,
) -> ProductionSharedClusterBootstrapPlan:
    """Build the exact 5-seed/10,000-draw plan with no runtime overrides."""

    spec = frozen_inference_spec(contract)
    if not isinstance(training_seed_ids, (list, tuple)) or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in training_seed_ids
    ):
        raise HarmBenchMetricError("training_seed_ids must be an exact integer sequence")
    normalized_ids = tuple(int(value) for value in training_seed_ids)
    if normalized_ids != spec.training_seed_ids:
        raise HarmBenchMetricError("training seed identities or order changed")
    shared = make_shared_cluster_bootstrap_plan(
        dataset_id,
        protocol_row_ids,
        clusters,
        training_seed_count=len(normalized_ids),
        replicates=spec.replicates,
        random_seed=spec.bootstrap_seed,
    )
    return ProductionSharedClusterBootstrapPlan(
        protocol_sha256=spec.protocol_sha256,
        inference_spec_sha256=spec.spec_sha256,
        training_seed_ids=normalized_ids,
        shared_plan=shared,
        binding_sha256=_production_plan_binding(
            spec=spec,
            training_seed_ids=normalized_ids,
            shared_plan_sha256=shared.plan_sha256,
        ),
    )


def validate_production_shared_plan(
    contract: object, plan: object
) -> ProductionSharedClusterBootstrapPlan:
    spec = frozen_inference_spec(contract)
    if not isinstance(plan, ProductionSharedClusterBootstrapPlan):
        raise HarmBenchMetricError("generic bootstrap plans cannot enter production inference")
    if (
        plan.protocol_sha256 != spec.protocol_sha256
        or plan.inference_spec_sha256 != spec.spec_sha256
        or plan.training_seed_ids != spec.training_seed_ids
    ):
        raise HarmBenchMetricError("production plan/protocol binding changed")
    shared = validate_shared_plan(plan.shared_plan)
    if (
        shared.training_seed_count != len(spec.training_seed_ids)
        or shared.replicates != spec.replicates
        or shared.random_seed != spec.bootstrap_seed
    ):
        raise HarmBenchMetricError("production plan controls differ from the frozen spec")
    expected = _production_plan_binding(
        spec=spec,
        training_seed_ids=plan.training_seed_ids,
        shared_plan_sha256=shared.plan_sha256,
    )
    if plan.binding_sha256 != expected:
        raise HarmBenchMetricError("production plan binding digest mismatch")
    return plan


def _sampled_query_indices_unchecked(
    plan: SharedClusterBootstrapPlan, replicate: int
) -> np.ndarray:
    return np.concatenate(
        [plan.cluster_members[int(code)] for code in plan.cluster_draws[replicate]]
    )


def sampled_query_indices(
    plan: SharedClusterBootstrapPlan, replicate: int
) -> np.ndarray:
    validated = validate_shared_plan(plan)
    index = _exact_integer(replicate, name="replicate", minimum=0)
    if index >= validated.replicates:
        raise HarmBenchMetricError("replicate index is out of range")
    return _sampled_query_indices_unchecked(validated, index)


def _probability_panel(values: object, *, name: str) -> np.ndarray:
    panel = np.asarray(values, dtype=np.float64)
    if panel.ndim != 3 or panel.shape[0] < 2:
        raise HarmBenchMetricError(f"{name} must have shape [training_seeds>=2, queries, classes]")
    for seed in range(panel.shape[0]):
        validated_probability(panel[seed], name=f"{name}[{seed}]")
    return panel


def probability_panel_sha256(values: object) -> str:
    """Hash one canonical float64 seed×query×class probability tensor."""

    panel = _probability_panel(values, name="probability_panel")
    normalized = np.ascontiguousarray(panel, dtype="<f8")
    descriptor = {
        "dtype": "float64-little-endian",
        "shape": list(normalized.shape),
    }
    digest = hashlib.sha256(
        json.dumps(
            descriptor,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _probability_panel_binding(
    *,
    production_plan: ProductionSharedClusterBootstrapPlan,
    dataset_id: str,
    alignment: str,
    model_id: str,
    strategy_id: str,
    shape: tuple[int, int, int],
    array_sha256: str,
) -> str:
    return _canonical_digest(
        {
            "protocol_sha256": production_plan.protocol_sha256,
            "inference_spec_sha256": production_plan.inference_spec_sha256,
            "bootstrap_plan_sha256": production_plan.shared_plan.plan_sha256,
            "dataset_id": dataset_id,
            "ordered_row_alignment_sha256": alignment,
            "training_seed_ids": list(production_plan.training_seed_ids),
            "model_id": model_id,
            "strategy_id": strategy_id,
            "shape": list(shape),
            "array_sha256": array_sha256,
        }
    )


def bind_production_probability_panel(
    contract: object,
    production_plan: object,
    protocol_row_ids: object,
    clusters: object,
    *,
    model_id: object,
    strategy_id: object,
    values: object,
    expected_array_sha256: object,
) -> ProductionProbabilityPanel:
    """Bind a producer-pinned tensor hash to row order, seed order and plan."""

    plan = validate_production_shared_plan(contract, production_plan)
    dataset = plan.shared_plan.dataset_id
    observed_alignment = alignment_sha256(dataset, protocol_row_ids, clusters)
    if observed_alignment != plan.shared_plan.alignment_sha256:
        raise HarmBenchMetricError("probability panel row/cluster alignment changed")
    model = _opaque_identifier(model_id, name="model_id")
    strategy = _opaque_identifier(strategy_id, name="strategy_id")
    expected_array = _sha256(expected_array_sha256, name="expected_array_sha256")
    panel = np.array(
        _probability_panel(values, name="probability_panel"),
        dtype=np.float64,
        order="C",
        copy=True,
    )
    if panel.shape[:2] != (
        len(plan.training_seed_ids),
        plan.shared_plan.query_count,
    ):
        raise HarmBenchMetricError("probability panel does not match seed/query identities")
    observed_array = probability_panel_sha256(panel)
    if observed_array != expected_array:
        raise HarmBenchMetricError("probability tensor differs from the producer-pinned SHA-256")
    shape = tuple(int(value) for value in panel.shape)
    binding = _probability_panel_binding(
        production_plan=plan,
        dataset_id=dataset,
        alignment=observed_alignment,
        model_id=model,
        strategy_id=strategy,
        shape=shape,
        array_sha256=observed_array,
    )
    panel.setflags(write=False)
    return ProductionProbabilityPanel(
        protocol_sha256=plan.protocol_sha256,
        inference_spec_sha256=plan.inference_spec_sha256,
        bootstrap_plan_sha256=plan.shared_plan.plan_sha256,
        dataset_id=dataset,
        alignment_sha256=observed_alignment,
        training_seed_ids=plan.training_seed_ids,
        model_id=model,
        strategy_id=strategy,
        shape=shape,
        array_sha256=observed_array,
        binding_sha256=binding,
        values=panel,
    )


def validate_production_probability_panel(
    contract: object,
    production_plan: object,
    protocol_row_ids: object,
    clusters: object,
    panel: object,
) -> ProductionProbabilityPanel:
    plan = validate_production_shared_plan(contract, production_plan)
    if not isinstance(panel, ProductionProbabilityPanel):
        raise HarmBenchMetricError("production probability input requires a typed binding")
    observed_alignment = alignment_sha256(
        plan.shared_plan.dataset_id, protocol_row_ids, clusters
    )
    if (
        panel.protocol_sha256 != plan.protocol_sha256
        or panel.inference_spec_sha256 != plan.inference_spec_sha256
        or panel.bootstrap_plan_sha256 != plan.shared_plan.plan_sha256
        or panel.dataset_id != plan.shared_plan.dataset_id
        or panel.alignment_sha256 != observed_alignment
        or observed_alignment != plan.shared_plan.alignment_sha256
        or panel.training_seed_ids != plan.training_seed_ids
    ):
        raise HarmBenchMetricError("probability panel production binding changed")
    model = _opaque_identifier(panel.model_id, name="model_id")
    strategy = _opaque_identifier(panel.strategy_id, name="strategy_id")
    values = _probability_panel(panel.values, name="probability_panel")
    shape = tuple(int(value) for value in values.shape)
    if shape != panel.shape or shape[:2] != (
        len(plan.training_seed_ids),
        plan.shared_plan.query_count,
    ):
        raise HarmBenchMetricError("probability panel shape binding changed")
    observed_array = probability_panel_sha256(values)
    if observed_array != panel.array_sha256:
        raise HarmBenchMetricError("probability tensor SHA-256 changed")
    expected_binding = _probability_panel_binding(
        production_plan=plan,
        dataset_id=panel.dataset_id,
        alignment=panel.alignment_sha256,
        model_id=model,
        strategy_id=strategy,
        shape=shape,
        array_sha256=observed_array,
    )
    if expected_binding != panel.binding_sha256:
        raise HarmBenchMetricError("probability panel binding digest mismatch")
    return panel


def _boolean_panel(values: object, *, seeds: int, queries: int, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim == 1:
        if len(raw) != queries or raw.dtype.kind != "b":
            raise HarmBenchMetricError(f"{name} must be boolean and aligned to queries")
        return np.broadcast_to(raw.astype(bool, copy=False), (seeds, queries))
    if raw.ndim == 2:
        if raw.shape != (seeds, queries) or raw.dtype.kind != "b":
            raise HarmBenchMetricError(
                f"{name} must be boolean with shape [training_seeds, queries]"
            )
        return raw.astype(bool, copy=False)
    raise HarmBenchMetricError(
        f"{name} must be one-dimensional or [training_seeds, queries]"
    )


def _prepare_cell(
    dataset_id: object,
    protocol_row_ids: object,
    clusters: object,
    labels: object,
    current_probability: object,
    strategy_probability: object,
    history_eligible: object,
    selected: object,
    plan: SharedClusterBootstrapPlan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    validated_plan = validate_shared_plan(plan)
    dataset = _dataset_identifier(dataset_id)
    observed_alignment = alignment_sha256(dataset, protocol_row_ids, clusters)
    if dataset != validated_plan.dataset_id or observed_alignment != validated_plan.alignment_sha256:
        raise HarmBenchMetricError("cell dataset/row/cluster alignment differs from bootstrap plan")
    _, observed_members = factorize_cluster_keys(clusters)
    if len(observed_members) != validated_plan.cluster_count or any(
        not np.array_equal(left, right)
        for left, right in zip(observed_members, validated_plan.cluster_members)
    ):
        raise HarmBenchMetricError("cell cluster membership differs from bootstrap plan")

    current = _probability_panel(current_probability, name="current_probability")
    strategy = _probability_panel(strategy_probability, name="strategy_probability")
    if current.shape != strategy.shape:
        raise HarmBenchMetricError("current and strategy probability panels differ")
    seeds, queries, classes = current.shape
    if seeds != validated_plan.training_seed_count or queries != validated_plan.query_count:
        raise HarmBenchMetricError("probability panel does not match the shared bootstrap plan")
    y_true = validated_labels(labels, queries=queries, classes=classes)
    eligible_raw = np.asarray(history_eligible)
    if (
        eligible_raw.ndim != 1
        or len(eligible_raw) != queries
        or eligible_raw.dtype.kind != "b"
    ):
        raise HarmBenchMetricError("history_eligible must be a boolean query vector")
    eligible = eligible_raw.astype(bool, copy=False)
    masks = _boolean_panel(selected, seeds=seeds, queries=queries, name="selected")
    if np.any(masks & ~eligible[None, :]):
        raise HarmBenchMetricError("selected contains queries without strictly past history")
    return y_true, current, strategy, eligible, masks


def _metric_path(report: dict[str, Any], *path: str) -> float | None:
    value: Any = report
    for key in path:
        value = value[key]
    return None if value is None else float(value)


def _flatten_report(report: dict[str, Any]) -> dict[str, float | None]:
    flattened = {
        "delta_macro_f1": _metric_path(report, "hybrid_minus_current", "macro_f1"),
        "delta_accuracy": _metric_path(report, "hybrid_minus_current", "accuracy"),
        "delta_mean_nll": _metric_path(report, "hybrid_minus_current", "mean_nll"),
        "delta_mean_brier": _metric_path(report, "hybrid_minus_current", "mean_brier"),
        "coverage": _metric_path(report, "nll_regret", "coverage"),
        "population_mean_regret": _metric_path(
            report, "nll_regret", "population", "mean_regret"
        ),
        "population_p90_regret": _metric_path(
            report, "nll_regret", "population", "p90_regret"
        ),
        "population_cvar90_regret": _metric_path(
            report, "nll_regret", "population", "cvar90_regret"
        ),
        "population_harm_rate_gt_0": _metric_path(
            report, "nll_regret", "population", "harm_rate", "greater_than_0"
        ),
        "population_harm_rate_gt_0_05": _metric_path(
            report, "nll_regret", "population", "harm_rate", "greater_than_0.05"
        ),
        "conditional_mean_regret": _metric_path(
            report, "nll_regret", "conditional_on_used", "mean_regret"
        ),
        "conditional_p90_regret": _metric_path(
            report, "nll_regret", "conditional_on_used", "p90_regret"
        ),
        "conditional_cvar90_regret": _metric_path(
            report, "nll_regret", "conditional_on_used", "cvar90_regret"
        ),
        "conditional_harm_rate_gt_0": _metric_path(
            report,
            "nll_regret",
            "conditional_on_used",
            "harm_rate",
            "greater_than_0",
        ),
        "conditional_harm_rate_gt_0_05": _metric_path(
            report,
            "nll_regret",
            "conditional_on_used",
            "harm_rate",
            "greater_than_0.05",
        ),
        "eligible_break_rate": _metric_path(
            report,
            "classification_transitions",
            "history_eligible_queries",
            "history_breaks_correct_current",
        ),
        "eligible_rescue_rate": _metric_path(
            report,
            "classification_transitions",
            "history_eligible_queries",
            "history_rescues_wrong_current",
        ),
        "selected_break_rate": _metric_path(
            report,
            "classification_transitions",
            "history_selected_queries",
            "history_breaks_correct_current",
        ),
        "selected_rescue_rate": _metric_path(
            report,
            "classification_transitions",
            "history_selected_queries",
            "history_rescues_wrong_current",
        ),
    }
    if tuple(flattened) != CELL_ENDPOINTS:
        raise AssertionError("cell endpoint order changed")
    return flattened


def _empty_history_seed_report(
    labels: np.ndarray,
    current: np.ndarray,
    strategy: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float | None]:
    if np.any(selected):
        raise HarmBenchMetricError("a no-history bootstrap sample cannot select history")
    hybrid = hybrid_probability(current, strategy, selected)
    current_metrics = classification_metrics(labels, current)
    hybrid_metrics = classification_metrics(labels, hybrid)
    empty: dict[str, float | None] = {
        "delta_macro_f1": float(hybrid_metrics["macro_f1"] - current_metrics["macro_f1"]),
        "delta_accuracy": float(hybrid_metrics["accuracy"] - current_metrics["accuracy"]),
        "delta_mean_nll": float(hybrid_metrics["mean_nll"] - current_metrics["mean_nll"]),
        "delta_mean_brier": float(
            hybrid_metrics["mean_brier"] - current_metrics["mean_brier"]
        ),
    }
    for endpoint in (
        "coverage",
        "population_mean_regret",
        "population_p90_regret",
        "population_cvar90_regret",
        "population_harm_rate_gt_0",
        "population_harm_rate_gt_0_05",
        "conditional_mean_regret",
        "conditional_p90_regret",
        "conditional_cvar90_regret",
        "conditional_harm_rate_gt_0",
        "conditional_harm_rate_gt_0_05",
        "eligible_break_rate",
        "eligible_rescue_rate",
        "selected_break_rate",
        "selected_rescue_rate",
    ):
        empty[endpoint] = None
    return empty


def _evaluate_seed(
    labels: np.ndarray,
    current: np.ndarray,
    strategy: np.ndarray,
    eligible: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float | None]:
    if not np.any(eligible):
        return _empty_history_seed_report(labels, current, strategy, selected)
    return _flatten_report(
        evaluate_frozen_policy(labels, current, strategy, eligible, selected)
    )


def _mean_or_none(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def _point_estimates(
    labels: np.ndarray,
    current: np.ndarray,
    strategy: np.ndarray,
    eligible: np.ndarray,
    masks: np.ndarray,
) -> dict[str, float | None]:
    reports = [
        _evaluate_seed(labels, current[seed], strategy[seed], eligible, masks[seed])
        for seed in range(current.shape[0])
    ]
    return {
        endpoint: _mean_or_none([report[endpoint] for report in reports])
        for endpoint in reports[0]
    }


def _bootstrap_samples(
    labels: np.ndarray,
    current: np.ndarray,
    strategy: np.ndarray,
    eligible: np.ndarray,
    masks: np.ndarray,
    plan: SharedClusterBootstrapPlan,
) -> dict[str, np.ndarray]:
    point_endpoints = tuple(_point_estimates(labels, current, strategy, eligible, masks))
    samples = {
        endpoint: np.full(plan.replicates, np.nan, dtype=np.float64)
        for endpoint in point_endpoints
    }
    for replicate in range(plan.replicates):
        queries = _sampled_query_indices_unchecked(plan, replicate)
        seed_reports = [
            _evaluate_seed(
                labels[queries],
                current[int(seed), queries],
                strategy[int(seed), queries],
                eligible[queries],
                masks[int(seed), queries],
            )
            for seed in plan.seed_draws[replicate]
        ]
        for endpoint in point_endpoints:
            value = _mean_or_none([report[endpoint] for report in seed_reports])
            if value is not None:
                samples[endpoint][replicate] = value
    return samples


def _summarize_samples(
    values: np.ndarray,
    *,
    point_value: float | None,
) -> dict[str, float | int | bool | None]:
    finite = values[np.isfinite(values)]
    fraction = float(len(finite) / len(values))
    if point_value is not None and fraction < MINIMUM_FINITE_BOOTSTRAP_FRACTION:
        raise HarmBenchMetricError(
            "finite bootstrap fraction fell below the frozen minimum: "
            f"{fraction:.6f} < {MINIMUM_FINITE_BOOTSTRAP_FRACTION:.6f}"
        )
    if point_value is None:
        return {
            "bootstrap_mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "finite_replicates": int(len(finite)),
            "finite_fraction": fraction,
            "minimum_finite_fraction": MINIMUM_FINITE_BOOTSTRAP_FRACTION,
            "minimum_finite_fraction_gate_applicable": False,
        }
    return {
        "bootstrap_mean": float(np.mean(finite)),
        "ci95_low": float(np.quantile(finite, 0.025)),
        "ci95_high": float(np.quantile(finite, 0.975)),
        "finite_replicates": int(len(finite)),
        "finite_fraction": fraction,
        "minimum_finite_fraction": MINIMUM_FINITE_BOOTSTRAP_FRACTION,
        "minimum_finite_fraction_gate_applicable": True,
    }


def bootstrap_cell_metrics(
    dataset_id: object,
    protocol_row_ids: object,
    clusters: object,
    labels: object,
    current_probability: object,
    strategy_probability: object,
    history_eligible: object,
    selected: object,
    plan: SharedClusterBootstrapPlan,
) -> dict[str, object]:
    y_true, current, strategy, eligible, masks = _prepare_cell(
        dataset_id,
        protocol_row_ids,
        clusters,
        labels,
        current_probability,
        strategy_probability,
        history_eligible,
        selected,
        plan,
    )
    point = _point_estimates(y_true, current, strategy, eligible, masks)
    samples = _bootstrap_samples(y_true, current, strategy, eligible, masks, plan)
    report: dict[str, object] = {
        "alignment_contract": {
            "dataset_id": plan.dataset_id,
            "alignment_sha256": plan.alignment_sha256,
            "bootstrap_plan_sha256": plan.plan_sha256,
        },
        "metric_contract": {
            "nll_probability_floor": NLL_EPSILON,
            "harm_thresholds_nats": list(DEFAULT_HARM_THRESHOLDS),
            "tail_alpha": DEFAULT_TAIL_ALPHA,
        },
        "inference_contract": {
            "unit": "training_seed_crossed_with_whole_cluster",
            "training_seed_count": plan.training_seed_count,
            "cluster_count": plan.cluster_count,
            "replicates": plan.replicates,
            "random_seed": plan.random_seed,
            "shared_plan_required_for_all_dataset_cells": True,
            "minimum_finite_bootstrap_fraction": MINIMUM_FINITE_BOOTSTRAP_FRACTION,
            "invalid_replicates_silently_redrawn": False,
        },
        "point": point,
        "bootstrap": {
            endpoint: _summarize_samples(values, point_value=point[endpoint])
            for endpoint, values in samples.items()
        },
    }
    ensure_finite_public_tree(report)
    return report


def bootstrap_paired_strategy_contrast(
    dataset_id: object,
    protocol_row_ids: object,
    clusters: object,
    labels: object,
    current_probability: object,
    left_strategy_probability: object,
    left_selected: object,
    right_strategy_probability: object,
    right_selected: object,
    history_eligible: object,
    plan: SharedClusterBootstrapPlan,
) -> dict[str, object]:
    left = _prepare_cell(
        dataset_id,
        protocol_row_ids,
        clusters,
        labels,
        current_probability,
        left_strategy_probability,
        history_eligible,
        left_selected,
        plan,
    )
    right = _prepare_cell(
        dataset_id,
        protocol_row_ids,
        clusters,
        labels,
        current_probability,
        right_strategy_probability,
        history_eligible,
        right_selected,
        plan,
    )
    if not np.array_equal(left[0], right[0]) or not np.array_equal(left[3], right[3]):
        raise HarmBenchMetricError("left and right cells are not aligned")
    left_point = _point_estimates(*left)
    right_point = _point_estimates(*right)
    left_samples = _bootstrap_samples(*left, plan)
    right_samples = _bootstrap_samples(*right, plan)

    point: dict[str, float | None] = {}
    bootstrap: dict[str, object] = {}
    for endpoint in left_point:
        if left_point[endpoint] is None or right_point[endpoint] is None:
            point[endpoint] = None
        else:
            point[endpoint] = float(left_point[endpoint] - right_point[endpoint])
        difference = left_samples[endpoint] - right_samples[endpoint]
        bootstrap[endpoint] = _summarize_samples(
            difference, point_value=point[endpoint]
        )
    report: dict[str, object] = {
        "alignment_contract": {
            "dataset_id": plan.dataset_id,
            "alignment_sha256": plan.alignment_sha256,
            "bootstrap_plan_sha256": plan.plan_sha256,
        },
        "metric_contract": {
            "nll_probability_floor": NLL_EPSILON,
            "harm_thresholds_nats": list(DEFAULT_HARM_THRESHOLDS),
            "tail_alpha": DEFAULT_TAIL_ALPHA,
        },
        "contrast_direction": "left_minus_right",
        "paired_on": "same_training_seed_draw_and_whole_cluster_draw",
        "point": point,
        "bootstrap": bootstrap,
    }
    ensure_finite_public_tree(report)
    return report


def bootstrap_production_cell_metrics(
    contract: object,
    production_plan: object,
    protocol_row_ids: object,
    clusters: object,
    labels: object,
    current_probability: object,
    strategy_probability: object,
    history_eligible: object,
    selected: object,
) -> dict[str, object]:
    """Production-only cell entrypoint requiring plan and tensor receipts."""

    plan = validate_production_shared_plan(contract, production_plan)
    current = validate_production_probability_panel(
        contract, plan, protocol_row_ids, clusters, current_probability
    )
    strategy = validate_production_probability_panel(
        contract, plan, protocol_row_ids, clusters, strategy_probability
    )
    if current.model_id != strategy.model_id:
        raise HarmBenchMetricError(
            "current and strategy panels must come from the same model family"
        )
    return bootstrap_cell_metrics(
        plan.shared_plan.dataset_id,
        protocol_row_ids,
        clusters,
        labels,
        current.values,
        strategy.values,
        history_eligible,
        selected,
        plan.shared_plan,
    )


def bootstrap_production_paired_strategy_contrast(
    contract: object,
    production_plan: object,
    protocol_row_ids: object,
    clusters: object,
    labels: object,
    current_probability: object,
    left_strategy_probability: object,
    left_selected: object,
    right_strategy_probability: object,
    right_selected: object,
    history_eligible: object,
) -> dict[str, object]:
    """Production paired contrast with one shared plan and bound tensors."""

    plan = validate_production_shared_plan(contract, production_plan)
    current = validate_production_probability_panel(
        contract, plan, protocol_row_ids, clusters, current_probability
    )
    left = validate_production_probability_panel(
        contract, plan, protocol_row_ids, clusters, left_strategy_probability
    )
    right = validate_production_probability_panel(
        contract, plan, protocol_row_ids, clusters, right_strategy_probability
    )
    if len({current.model_id, left.model_id, right.model_id}) != 1:
        raise HarmBenchMetricError("paired panels must come from the same model family")
    return bootstrap_paired_strategy_contrast(
        plan.shared_plan.dataset_id,
        protocol_row_ids,
        clusters,
        labels,
        current.values,
        left.values,
        left_selected,
        right.values,
        right_selected,
        history_eligible,
        plan.shared_plan,
    )
