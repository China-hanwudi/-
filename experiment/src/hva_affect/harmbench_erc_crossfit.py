"""Outcome-free shared group cross-fitting for HarmBench-ERC model families."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from .harmbench_erc_contract import EXPECTED_TRAINING_SEEDS
from .harmbench_erc_contexts import (
    FIT_TRAIN_CONTEXT_ROLE,
    STRICT_PAST_STRATEGY_IDS,
    StrictPastContextRoster,
    validate_strict_past_context_roster,
)
from .harmbench_erc_open_roles import (
    FitFeatureCapability,
    OutcomeFreeRoleFeatures,
    validate_fit_feature_capability,
)

if TYPE_CHECKING:  # pragma: no cover - annotations only; processors import crossfit.
    from .harmbench_erc_processors import (
        ProcessedRoleEmbeddings,
        ProcessorReceipt,
    )


class HarmBenchCrossfitError(ValueError):
    """Raised when a fold or augmentation plan can leak groups or outcomes."""


EXPECTED_OUTER_FOLDS = 5
CONTEXT_TRAINING_EXAMPLE_SCHEMA = "harmbench_erc_context_training_examples_v2"


@dataclass(frozen=True)
class SharedGroupCrossfitPlan:
    dataset_id: str
    training_seed_ids: tuple[int, ...]
    outer_folds: int
    query_count: int
    group_count: int
    protocol_row_ids_sha256: str
    group_alignment_sha256: str
    source_capability_sha256: str
    fold_assignment: np.ndarray
    plan_sha256: str

    def heldout_indices(
        self,
        training_seed: int,
        fold: int,
        *,
        fit_capability: FitFeatureCapability,
    ) -> np.ndarray:
        _, heldout = resolve_shared_group_crossfit_indices(
            self,
            fit_capability,
            training_seed=training_seed,
            fold=fold,
        )
        return heldout

    def train_indices(
        self,
        training_seed: int,
        fold: int,
        *,
        fit_capability: FitFeatureCapability,
    ) -> np.ndarray:
        train, _ = resolve_shared_group_crossfit_indices(
            self,
            fit_capability,
            training_seed=training_seed,
            fold=fold,
        )
        return train


@dataclass(frozen=True)
class ContextTrainingExamples:
    schema_version: str
    dataset_id: str
    training_seed: int
    fold: int
    fit_feature_capability_sha256: str
    cross_role_feature_roster_sha256: str
    processor_receipt_sha256: str
    processed_output_receipt_sha256: str
    crossfit_plan_sha256: str
    fit_train_protocol_row_ids_sha256: str
    fit_heldout_protocol_row_ids_sha256: str
    context_roster_sha256_by_strategy: tuple[tuple[str, str], ...]
    query_protocol_row_ids: tuple[int, ...]
    context_protocol_row_ids: tuple[tuple[int, ...], ...]
    source_strategies: tuple[tuple[str, ...], ...]
    example_count: int
    example_sha256: str


def _sha256(value: object, *, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise HarmBenchCrossfitError(f"{name} must be a lowercase SHA-256")
    return digest


def _typed_string_vector(values: object, *, name: str) -> np.ndarray:
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or not len(raw):
        raise HarmBenchCrossfitError(f"{name} must be a non-empty vector")
    if any(not isinstance(value, (str, np.str_)) or not str(value) for value in raw):
        raise HarmBenchCrossfitError(f"{name} must contain non-empty strings")
    result = np.asarray([str(value) for value in raw])
    result.setflags(write=False)
    return result


def _canonical_vector_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in np.asarray(values).reshape(-1):
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _group_tie(dataset_id: str, seed: int, group: str) -> str:
    return hashlib.sha256(
        f"harmbench_group_fold_v1\x1f{dataset_id}\x1f{seed}\x1f{group}".encode(
            "utf-8"
        )
    ).hexdigest()


def _balanced_group_assignment(
    *, dataset_id: str, groups: np.ndarray, seed: int, folds: int
) -> np.ndarray:
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(unique) < folds:
        raise HarmBenchCrossfitError("fewer independent groups than outer folds")
    ordered = sorted(
        range(len(unique)),
        key=lambda index: (
            -int(counts[index]),
            _group_tie(dataset_id, seed, str(unique[index])),
        ),
    )
    loads = [0] * folds
    group_folds = np.empty(len(unique), dtype=np.int16)
    for group_index in ordered:
        minimum = min(loads)
        candidates = [fold for fold, load in enumerate(loads) if load == minimum]
        fold = min(
            candidates,
            key=lambda value: hashlib.sha256(
                f"{_group_tie(dataset_id, seed, str(unique[group_index]))}\x1f{value}".encode()
            ).hexdigest(),
        )
        group_folds[group_index] = fold
        loads[fold] += int(counts[group_index])
    if any(load == 0 for load in loads):
        raise HarmBenchCrossfitError("balanced assignment produced an empty fold")
    return group_folds[inverse]


def _plan_descriptor(
    *,
    features: OutcomeFreeRoleFeatures,
    seeds: tuple[int, ...],
    capability_sha: str,
    groups: np.ndarray,
    assignments: np.ndarray,
) -> dict[str, object]:
    return {
        "dataset_id": features.dataset_id,
        "role": features.role,
        "training_seed_ids": list(seeds),
        "outer_folds": EXPECTED_OUTER_FOLDS,
        "query_count": features.rows,
        "group_count": len(set(groups.tolist())),
        "protocol_row_ids_sha256": _canonical_vector_sha256(
            features.protocol_row_ids
        ),
        "group_alignment_sha256": _canonical_vector_sha256(groups),
        "source_capability_sha256": capability_sha,
        "fold_assignment": np.asarray(assignments, dtype=np.int16).astype(int).tolist(),
    }


def _descriptor_sha256(descriptor: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            descriptor,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def make_shared_group_crossfit_plan(
    fit_capability: FitFeatureCapability,
    *,
    training_seed_ids: Sequence[int] = EXPECTED_TRAINING_SEEDS,
    outer_folds: int = EXPECTED_OUTER_FOLDS,
) -> SharedGroupCrossfitPlan:
    """Create one outcome-free fold plan shared by every model family."""

    try:
        fit_capability = validate_fit_feature_capability(fit_capability)
    except ValueError as error:
        raise HarmBenchCrossfitError(f"invalid fit capability: {error}") from error
    features = fit_capability.fit
    raw_seeds = tuple(training_seed_ids)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw_seeds
    ):
        raise HarmBenchCrossfitError("training seeds must be exact integers")
    seeds = tuple(int(value) for value in raw_seeds)
    if seeds != tuple(EXPECTED_TRAINING_SEEDS) or len(set(seeds)) != len(seeds):
        raise HarmBenchCrossfitError("training seed roster changed")
    if isinstance(outer_folds, (bool, np.bool_)) or not isinstance(
        outer_folds, (int, np.integer)
    ) or int(outer_folds) != EXPECTED_OUTER_FOLDS:
        raise HarmBenchCrossfitError("outer fold count changed")
    capability_sha = _sha256(
        fit_capability.capability_sha256, name="source_capability_sha256"
    )
    groups = _typed_string_vector(features.groups, name="groups")
    if len(groups) != features.rows:
        raise HarmBenchCrossfitError("group vector differs from feature rows")
    assignments = np.stack(
        [
            _balanced_group_assignment(
                dataset_id=features.dataset_id,
                groups=groups,
                seed=seed,
                folds=EXPECTED_OUTER_FOLDS,
            )
            for seed in seeds
        ]
    ).astype(np.int16, copy=False)
    descriptor = _plan_descriptor(
        features=features,
        seeds=seeds,
        capability_sha=capability_sha,
        groups=groups,
        assignments=assignments,
    )
    plan_sha = _descriptor_sha256(descriptor)
    assignments.setflags(write=False)
    result = SharedGroupCrossfitPlan(
        dataset_id=features.dataset_id,
        training_seed_ids=seeds,
        outer_folds=EXPECTED_OUTER_FOLDS,
        query_count=features.rows,
        group_count=descriptor["group_count"],
        protocol_row_ids_sha256=descriptor["protocol_row_ids_sha256"],
        group_alignment_sha256=descriptor["group_alignment_sha256"],
        source_capability_sha256=capability_sha,
        fold_assignment=assignments,
        plan_sha256=plan_sha,
    )
    validate_shared_group_crossfit_plan(result, fit_capability)
    return result


def validate_shared_group_crossfit_plan(
    plan: SharedGroupCrossfitPlan, fit_capability: FitFeatureCapability
) -> None:
    if not isinstance(plan, SharedGroupCrossfitPlan):
        raise HarmBenchCrossfitError("invalid crossfit plan type")
    try:
        fit_capability = validate_fit_feature_capability(fit_capability)
    except ValueError as error:
        raise HarmBenchCrossfitError(f"invalid fit capability: {error}") from error
    features = fit_capability.fit
    if plan.source_capability_sha256 != fit_capability.capability_sha256:
        raise HarmBenchCrossfitError("crossfit source capability identity changed")
    if plan.dataset_id != features.dataset_id or plan.query_count != features.rows:
        raise HarmBenchCrossfitError("crossfit dataset/row identity changed")
    if plan.training_seed_ids != tuple(EXPECTED_TRAINING_SEEDS):
        raise HarmBenchCrossfitError("crossfit seed identity changed")
    if plan.outer_folds != EXPECTED_OUTER_FOLDS:
        raise HarmBenchCrossfitError("crossfit fold count changed")
    assignment = np.asarray(plan.fold_assignment)
    if assignment.shape != (len(EXPECTED_TRAINING_SEEDS), features.rows):
        raise HarmBenchCrossfitError("crossfit assignment shape changed")
    if assignment.dtype.kind not in "iu" or np.any(
        (assignment < 0) | (assignment >= EXPECTED_OUTER_FOLDS)
    ):
        raise HarmBenchCrossfitError("crossfit assignment contains an invalid fold")
    if assignment.flags.writeable:
        raise HarmBenchCrossfitError("crossfit assignment must be immutable")
    groups = np.asarray(features.groups, dtype=str)
    for seed_index in range(len(EXPECTED_TRAINING_SEEDS)):
        if set(assignment[seed_index].tolist()) != set(range(EXPECTED_OUTER_FOLDS)):
            raise HarmBenchCrossfitError("crossfit seed does not cover every fold")
        for group in set(groups.tolist()):
            observed = set(assignment[seed_index][groups == group].tolist())
            if len(observed) != 1:
                raise HarmBenchCrossfitError("crossfit split an independent group")
    expected_assignment = np.stack(
        [
            _balanced_group_assignment(
                dataset_id=features.dataset_id,
                groups=groups,
                seed=seed,
                folds=EXPECTED_OUTER_FOLDS,
            )
            for seed in plan.training_seed_ids
        ]
    ).astype(np.int16, copy=False)
    if not np.array_equal(assignment, expected_assignment):
        raise HarmBenchCrossfitError("crossfit assignment differs from deterministic plan")
    descriptor = _plan_descriptor(
        features=features,
        seeds=plan.training_seed_ids,
        capability_sha=_sha256(
            plan.source_capability_sha256, name="source_capability_sha256"
        ),
        groups=groups,
        assignments=assignment,
    )
    exact_fields = {
        "group_count": plan.group_count,
        "protocol_row_ids_sha256": plan.protocol_row_ids_sha256,
        "group_alignment_sha256": plan.group_alignment_sha256,
    }
    if any(descriptor[name] != value for name, value in exact_fields.items()):
        raise HarmBenchCrossfitError("crossfit descriptor binding changed")
    if _descriptor_sha256(descriptor) != plan.plan_sha256:
        raise HarmBenchCrossfitError("crossfit plan SHA differs from live assignment")


def _exact_seed_fold(
    plan: SharedGroupCrossfitPlan,
    *,
    training_seed: object,
    fold: object,
) -> tuple[int, int, int]:
    if isinstance(training_seed, (bool, np.bool_)) or not isinstance(
        training_seed, (int, np.integer)
    ):
        raise HarmBenchCrossfitError("training seed must be an exact integer")
    if isinstance(fold, (bool, np.bool_)) or not isinstance(fold, (int, np.integer)):
        raise HarmBenchCrossfitError("fold must be an exact integer")
    seed_value = int(training_seed)
    fold_value = int(fold)
    if seed_value not in plan.training_seed_ids:
        raise HarmBenchCrossfitError("training seed is outside the frozen roster")
    if fold_value not in range(EXPECTED_OUTER_FOLDS):
        raise HarmBenchCrossfitError("fold is outside the frozen roster")
    return seed_value, fold_value, plan.training_seed_ids.index(seed_value)


def resolve_shared_group_crossfit_indices(
    plan: SharedGroupCrossfitPlan,
    fit_capability: FitFeatureCapability,
    *,
    training_seed: object,
    fold: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Live-validate the capability/plan before deriving one fold's row indices."""

    validate_shared_group_crossfit_plan(plan, fit_capability)
    seed_value, fold_value, seed_index = _exact_seed_fold(
        plan,
        training_seed=training_seed,
        fold=fold,
    )
    del seed_value
    assignment = np.asarray(plan.fold_assignment)
    train = np.flatnonzero(assignment[seed_index] != fold_value).astype(np.int64)
    heldout = np.flatnonzero(assignment[seed_index] == fold_value).astype(np.int64)
    if not len(train) or not len(heldout):
        raise HarmBenchCrossfitError("crossfit produced an empty train or heldout fold")
    train.setflags(write=False)
    heldout.setflags(write=False)
    return train, heldout


def make_context_training_examples(
    context_rosters: Mapping[str, StrictPastContextRoster],
    fit_capability: FitFeatureCapability,
    processed_features: "ProcessedRoleEmbeddings",
    processor_receipt: "ProcessorReceipt",
    crossfit_plan: SharedGroupCrossfitPlan,
    *,
    training_seed: object,
    fold: object,
    expected_fit_feature_capability_sha256: str,
    expected_processor_receipt_sha256: str,
    expected_processed_output_receipt_sha256: str,
    expected_crossfit_plan_sha256: str,
    expected_context_roster_sha256_by_strategy: Mapping[str, str],
) -> ContextTrainingExamples:
    """Build fit-train augmentation only from live-validated context rosters.

    The caller cannot choose query or allowed rows.  The fit partition is
    derived from ``crossfit_plan`` and every strategy roster is revalidated
    against the live feature capability, processor output, receipt, seed, and
    fold before its protocol-row IDs enter the examples.
    """

    try:
        fit_capability = validate_fit_feature_capability(fit_capability)
        validate_shared_group_crossfit_plan(crossfit_plan, fit_capability)
    except ValueError as error:
        raise HarmBenchCrossfitError(
            f"invalid fit capability or crossfit plan: {error}"
        ) from error
    expected_fit_sha = _sha256(
        expected_fit_feature_capability_sha256,
        name="expected_fit_feature_capability_sha256",
    )
    if fit_capability.capability_sha256 != expected_fit_sha:
        raise HarmBenchCrossfitError(
            "fit capability differs from the external training binding"
        )
    expected_plan_sha = _sha256(
        expected_crossfit_plan_sha256,
        name="expected_crossfit_plan_sha256",
    )
    if crossfit_plan.plan_sha256 != expected_plan_sha:
        raise HarmBenchCrossfitError(
            "crossfit plan differs from the external training binding"
        )
    seed_value, fold_value, _ = _exact_seed_fold(
        crossfit_plan,
        training_seed=training_seed,
        fold=fold,
    )
    if not isinstance(context_rosters, Mapping) or tuple(context_rosters) != (
        STRICT_PAST_STRATEGY_IDS
    ):
        raise HarmBenchCrossfitError("context strategy roster/order changed")
    if not isinstance(expected_context_roster_sha256_by_strategy, Mapping) or tuple(
        expected_context_roster_sha256_by_strategy
    ) != STRICT_PAST_STRATEGY_IDS:
        raise HarmBenchCrossfitError(
            "external context receipt roster/order changed"
        )

    validated_rosters: list[StrictPastContextRoster] = []
    roster_receipts: list[tuple[str, str]] = []
    for strategy in STRICT_PAST_STRATEGY_IDS:
        roster = context_rosters[strategy]
        expected_roster_sha = _sha256(
            expected_context_roster_sha256_by_strategy[strategy],
            name=f"expected_context_roster_sha256_by_strategy[{strategy}]",
        )
        try:
            validated = validate_strict_past_context_roster(
                roster,
                fit_capability,
                fit_capability,
                processed_features,
                processor_receipt,
                crossfit_plan,
                training_seed=seed_value,
                fold=fold_value,
                context_role=FIT_TRAIN_CONTEXT_ROLE,
                strategy_id=strategy,
                expected_fit_plan_capability_sha256=expected_fit_sha,
                expected_source_capability_sha256=expected_fit_sha,
                expected_processor_receipt_sha256=(
                    expected_processor_receipt_sha256
                ),
                expected_processed_output_receipt_sha256=(
                    expected_processed_output_receipt_sha256
                ),
                expected_crossfit_plan_sha256=expected_plan_sha,
                expected_context_roster_sha256=expected_roster_sha,
            )
        except (TypeError, ValueError) as error:
            raise HarmBenchCrossfitError(
                f"invalid fit-train context roster for {strategy}: {error}"
            ) from error
        validated_rosters.append(validated)
        roster_receipts.append((strategy, expected_roster_sha))

    queries = validated_rosters[0].query_protocol_row_ids
    if any(roster.query_protocol_row_ids != queries for roster in validated_rosters):
        raise HarmBenchCrossfitError("context rosters differ in derived training queries")
    output_queries: list[int] = []
    output_contexts: list[tuple[int, ...]] = []
    output_sources: list[tuple[str, ...]] = []
    descriptor_rows: list[dict[str, object]] = []
    for query_position, query_protocol_row_id in enumerate(queries):
        grouped: dict[tuple[int, ...], list[str]] = {}
        for strategy, roster in zip(
            STRICT_PAST_STRATEGY_IDS, validated_rosters, strict=True
        ):
            context = roster.context_protocol_row_ids[query_position]
            grouped.setdefault(context, []).append(strategy)
        for context, source_ids in grouped.items():
            output_queries.append(int(query_protocol_row_id))
            output_contexts.append(context)
            output_sources.append(tuple(source_ids))
            descriptor_rows.append(
                {
                    "query_protocol_row_id": int(query_protocol_row_id),
                    "context_protocol_row_ids": list(context),
                    "strategies": source_ids,
                }
            )
    first_roster = validated_rosters[0]
    descriptor = {
        "schema_version": CONTEXT_TRAINING_EXAMPLE_SCHEMA,
        "dataset_id": fit_capability.dataset_id,
        "training_seed": seed_value,
        "fold": fold_value,
        "fit_feature_capability_sha256": expected_fit_sha,
        "cross_role_feature_roster_sha256": (
            fit_capability.cross_role_feature_roster_sha256
        ),
        "processor_receipt_sha256": _sha256(
            expected_processor_receipt_sha256,
            name="expected_processor_receipt_sha256",
        ),
        "processed_output_receipt_sha256": _sha256(
            expected_processed_output_receipt_sha256,
            name="expected_processed_output_receipt_sha256",
        ),
        "crossfit_plan_sha256": expected_plan_sha,
        "fit_train_protocol_row_ids_sha256": (
            first_roster.fit_train_protocol_row_ids_sha256
        ),
        "fit_heldout_protocol_row_ids_sha256": (
            first_roster.fit_heldout_protocol_row_ids_sha256
        ),
        "context_roster_sha256_by_strategy": [
            list(item) for item in roster_receipts
        ],
        "rows": descriptor_rows,
    }
    digest = hashlib.sha256(
        json.dumps(
            descriptor,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return ContextTrainingExamples(
        schema_version=CONTEXT_TRAINING_EXAMPLE_SCHEMA,
        dataset_id=fit_capability.dataset_id,
        training_seed=seed_value,
        fold=fold_value,
        fit_feature_capability_sha256=expected_fit_sha,
        cross_role_feature_roster_sha256=(
            fit_capability.cross_role_feature_roster_sha256
        ),
        processor_receipt_sha256=descriptor["processor_receipt_sha256"],
        processed_output_receipt_sha256=descriptor[
            "processed_output_receipt_sha256"
        ],
        crossfit_plan_sha256=expected_plan_sha,
        fit_train_protocol_row_ids_sha256=(
            first_roster.fit_train_protocol_row_ids_sha256
        ),
        fit_heldout_protocol_row_ids_sha256=(
            first_roster.fit_heldout_protocol_row_ids_sha256
        ),
        context_roster_sha256_by_strategy=tuple(roster_receipts),
        query_protocol_row_ids=tuple(output_queries),
        context_protocol_row_ids=tuple(output_contexts),
        source_strategies=tuple(output_sources),
        example_count=len(output_queries),
        example_sha256=digest,
    )


def validate_context_training_examples(
    examples: ContextTrainingExamples,
    context_rosters: Mapping[str, StrictPastContextRoster],
    fit_capability: FitFeatureCapability,
    processed_features: "ProcessedRoleEmbeddings",
    processor_receipt: "ProcessorReceipt",
    crossfit_plan: SharedGroupCrossfitPlan,
    *,
    training_seed: object,
    fold: object,
    expected_fit_feature_capability_sha256: str,
    expected_processor_receipt_sha256: str,
    expected_processed_output_receipt_sha256: str,
    expected_crossfit_plan_sha256: str,
    expected_context_roster_sha256_by_strategy: Mapping[str, str],
    expected_context_training_examples_sha256: str,
) -> ContextTrainingExamples:
    """Live-rebuild and compare a context-example receipt at consumption time."""

    if not isinstance(examples, ContextTrainingExamples):
        raise HarmBenchCrossfitError(
            "examples must be a ContextTrainingExamples receipt"
        )
    expected_example_sha = _sha256(
        expected_context_training_examples_sha256,
        name="expected_context_training_examples_sha256",
    )
    if examples.example_sha256 != expected_example_sha:
        raise HarmBenchCrossfitError(
            "context training examples differ from the external binding"
        )
    rebuilt = make_context_training_examples(
        context_rosters,
        fit_capability,
        processed_features,
        processor_receipt,
        crossfit_plan,
        training_seed=training_seed,
        fold=fold,
        expected_fit_feature_capability_sha256=(
            expected_fit_feature_capability_sha256
        ),
        expected_processor_receipt_sha256=expected_processor_receipt_sha256,
        expected_processed_output_receipt_sha256=(
            expected_processed_output_receipt_sha256
        ),
        expected_crossfit_plan_sha256=expected_crossfit_plan_sha256,
        expected_context_roster_sha256_by_strategy=(
            expected_context_roster_sha256_by_strategy
        ),
    )
    for item in fields(ContextTrainingExamples):
        if getattr(examples, item.name) != getattr(rebuilt, item.name):
            raise HarmBenchCrossfitError(
                f"context training examples differ from live derivation: {item.name}"
            )
    return examples


__all__ = [
    "CONTEXT_TRAINING_EXAMPLE_SCHEMA",
    "ContextTrainingExamples",
    "EXPECTED_OUTER_FOLDS",
    "HarmBenchCrossfitError",
    "SharedGroupCrossfitPlan",
    "make_context_training_examples",
    "make_shared_group_crossfit_plan",
    "resolve_shared_group_crossfit_indices",
    "validate_context_training_examples",
    "validate_shared_group_crossfit_plan",
]
