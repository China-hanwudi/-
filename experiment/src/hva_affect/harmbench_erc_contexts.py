"""Receipt-bound, outcome-free strict-past contexts for HarmBench-ERC.

The public builders in this module never accept caller-supplied groups,
histories, embeddings, query rows, or allowed rows.  All identity/time fields
come from a live-validated feature capability, all similarity inputs come from
a live-validated processor output, and every fit partition comes from the
shared whole-group cross-fit plan.

Imports of the neighbouring HarmBench modules are intentionally local.  The
cross-fit module imports :data:`STRICT_PAST_STRATEGY_IDS`, so eager imports in
the opposite direction would create an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from numbers import Integral
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from .harmbench_erc_protocol_v2 import (
    EXPECTED_ANCHOR_STRATEGY_ID,
    EXPECTED_CONTEXT_ROSTER_ORDER,
    EXPECTED_HISTORY_STRATEGY_ORDER,
    HarmBenchProtocolV2Error,
    STRATEGY_RULE_VERSION,
    get_context_strategy_contract,
    strategy_rule_sha256,
)

if TYPE_CHECKING:  # pragma: no cover - annotations only; see module docstring.
    from .harmbench_erc_crossfit import SharedGroupCrossfitPlan
    from .harmbench_erc_open_roles import (
        FitFeatureCapability,
        OutcomeFreeRoleFeatures,
        SelectionFeatureCapability,
    )
    from .harmbench_erc_processors import (
        ProcessedRoleEmbeddings,
        ProcessorReceipt,
    )


class HarmBenchContextError(ValueError):
    """Raised when context provenance or strict-past semantics change."""


CONTEXT_ROSTER_SCHEMA = "harmbench_erc_strict_past_context_roster_v2"
CURRENT_ONLY_STRATEGY_ID = EXPECTED_ANCHOR_STRATEGY_ID
STRICT_PAST_STRATEGY_IDS = EXPECTED_HISTORY_STRATEGY_ORDER
CONTEXT_STRATEGY_IDS = EXPECTED_CONTEXT_ROSTER_ORDER
REQUIRED_MODALITY_IDS = ("text", "audio", "video")
FROZEN_TOP_K = 3

FIT_TRAIN_CONTEXT_ROLE = "fit_train"
FIT_HELDOUT_OOF_CONTEXT_ROLE = "fit_heldout_oof"
SELECTION_CONTEXT_ROLE = "selection_prediction"
CONTEXT_ROLE_IDS = (
    FIT_TRAIN_CONTEXT_ROLE,
    FIT_HELDOUT_OOF_CONTEXT_ROLE,
    SELECTION_CONTEXT_ROLE,
)

STRICT_PAST_INVARIANT_IDS = (
    "same_source_role",
    "same_independent_group",
    "strictly_lower_turn_id",
    "derived_partition_only",
    "canonical_turn_then_protocol_order",
    "unique_context_protocol_rows",
)
CURRENT_ONLY_INVARIANT_ID = "zero_context_and_history_consumption"
SHA256_LENGTH = 64


def _valid_sha256(value: object, *, name: str) -> str:
    digest = str(value)
    if (
        len(digest) != SHA256_LENGTH
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise HarmBenchContextError(f"{name} must be a lowercase SHA-256")
    return digest


def _canonical_json_sha256(payload: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(
            dict(payload),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise HarmBenchContextError(f"context receipt is not canonical JSON: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def _canonical_array_sha256(values: object) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\x00")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\x00")
    if array.dtype.kind in {"U", "S", "O"}:
        for value in array.astype(str).reshape(-1):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    else:
        canonical = np.ascontiguousarray(array)
        if canonical.dtype.byteorder == ">" or (
            canonical.dtype.byteorder == "=" and not np.little_endian
        ):
            canonical = canonical.byteswap().view(canonical.dtype.newbyteorder("<"))
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _exact_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise HarmBenchContextError(f"{name} must be an exact integer")
    result = int(value)
    if result < 0:
        raise HarmBenchContextError(f"{name} must be non-negative")
    return result


def _integer_tuple(
    values: object,
    *,
    name: str,
    nonempty: bool,
    unique: bool = True,
) -> tuple[int, ...]:
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise HarmBenchContextError(f"{name} must be an integer sequence") from error
    normalized = tuple(
        _exact_nonnegative_integer(value, name=f"{name} entry") for value in raw
    )
    if nonempty and not normalized:
        raise HarmBenchContextError(f"{name} must not be empty")
    if unique and len(normalized) != len(set(normalized)):
        raise HarmBenchContextError(f"{name} must contain unique rows")
    return normalized


def _contexts_tuple(values: object) -> tuple[tuple[int, ...], ...]:
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise HarmBenchContextError("context_protocol_row_ids must be a sequence") from error
    return tuple(
        _integer_tuple(value, name=f"context_protocol_row_ids[{index}]", nonempty=False)
        for index, value in enumerate(raw)
    )


def _context_hash(contexts: Sequence[Sequence[int]]) -> str:
    return _canonical_json_sha256(
        {"context_protocol_row_ids": [list(context) for context in contexts]}
    )


def _invariant_ids(strategy_id: str) -> tuple[str, ...]:
    if strategy_id == CURRENT_ONLY_STRATEGY_ID:
        return (*STRICT_PAST_INVARIANT_IDS, CURRENT_ONLY_INVARIANT_ID)
    return STRICT_PAST_INVARIANT_IDS


def _strategy_rule_fields(strategy_id: str) -> dict[str, object]:
    """Resolve the exact v2 rule into immutable receipt fields.

    This function accepts only a registered identifier.  No builder or
    validator surface accepts caller-supplied ranking, top-k, or fallback
    controls.
    """

    try:
        rule = get_context_strategy_contract(strategy_id)
        digest = strategy_rule_sha256(rule)
    except HarmBenchProtocolV2Error as error:
        raise HarmBenchContextError(
            "context strategy is outside the frozen v2 roster"
        ) from error
    modality_order = rule["modality_order"]
    if not isinstance(modality_order, tuple):
        raise HarmBenchContextError("registered modality order is not immutable")
    return {
        "strategy_rule_version": STRATEGY_RULE_VERSION,
        "strategy_rule_sha256": digest,
        "candidate_scope": rule["candidate_scope"],
        "strict_past_required": rule["strict_past"],
        "top_k": rule["top_k"],
        "ranking": rule["ranking"],
        "ranking_tie": rule["ranking_tie"],
        "zero_vector_policy": rule["zero_vector"],
        "modality_order": modality_order,
        "duplicate_skip_policy": rule["duplicate_skip"],
        "emission_order": rule["emission_order"],
        "empty_fallback": rule["empty_fallback"],
    }


@dataclass(frozen=True)
class StrictPastContextRoster:
    """Immutable protocol-row roster for one fold, partition, and strategy.

    Contexts contain protocol-row IDs, never mutable physical row positions.
    ``fit_train_protocol_row_ids_sha256`` and
    ``fit_heldout_protocol_row_ids_sha256`` bind both internally derived sides
    of the fold even for selection prediction.
    """

    schema_version: str
    dataset_id: str
    source_role: str
    context_role: str
    strategy_id: str
    strategy_rule_version: str
    strategy_rule_sha256: str
    candidate_scope: str
    strict_past_required: bool
    top_k: int | None
    ranking: str
    ranking_tie: str
    zero_vector_policy: str
    modality_order: tuple[str, ...]
    duplicate_skip_policy: str
    emission_order: str
    empty_fallback: str
    training_seed: int
    fold: int
    fit_plan_capability_sha256: str
    source_capability_sha256: str
    cross_role_feature_roster_sha256: str
    source_content_sha256: str
    source_row_alignment_sha256: str
    processor_receipt_sha256: str
    processed_output_receipt_sha256: str
    processed_output_row_alignment_sha256: str
    crossfit_plan_sha256: str
    fit_train_protocol_row_ids_sha256: str
    fit_heldout_protocol_row_ids_sha256: str
    query_protocol_row_ids: tuple[int, ...]
    query_protocol_row_ids_sha256: str
    context_protocol_row_ids: tuple[tuple[int, ...], ...]
    context_protocol_row_ids_sha256: str
    context_counts: tuple[int, ...]
    query_count: int
    total_context_count: int
    history_consumption_count: int
    strict_past_invariant_ids: tuple[str, ...]
    roster_sha256: str

    def __post_init__(self) -> None:
        _validate_roster_receipt_shape(self)


def _roster_descriptor(roster: StrictPastContextRoster) -> dict[str, object]:
    return {
        "schema_version": roster.schema_version,
        "dataset_id": roster.dataset_id,
        "source_role": roster.source_role,
        "context_role": roster.context_role,
        "strategy_id": roster.strategy_id,
        "strategy_rule_version": roster.strategy_rule_version,
        "strategy_rule_sha256": roster.strategy_rule_sha256,
        "candidate_scope": roster.candidate_scope,
        "strict_past_required": roster.strict_past_required,
        "top_k": roster.top_k,
        "ranking": roster.ranking,
        "ranking_tie": roster.ranking_tie,
        "zero_vector_policy": roster.zero_vector_policy,
        "modality_order": list(roster.modality_order),
        "duplicate_skip_policy": roster.duplicate_skip_policy,
        "emission_order": roster.emission_order,
        "empty_fallback": roster.empty_fallback,
        "training_seed": roster.training_seed,
        "fold": roster.fold,
        "fit_plan_capability_sha256": roster.fit_plan_capability_sha256,
        "source_capability_sha256": roster.source_capability_sha256,
        "cross_role_feature_roster_sha256": (
            roster.cross_role_feature_roster_sha256
        ),
        "source_content_sha256": roster.source_content_sha256,
        "source_row_alignment_sha256": roster.source_row_alignment_sha256,
        "processor_receipt_sha256": roster.processor_receipt_sha256,
        "processed_output_receipt_sha256": (
            roster.processed_output_receipt_sha256
        ),
        "processed_output_row_alignment_sha256": (
            roster.processed_output_row_alignment_sha256
        ),
        "crossfit_plan_sha256": roster.crossfit_plan_sha256,
        "fit_train_protocol_row_ids_sha256": (
            roster.fit_train_protocol_row_ids_sha256
        ),
        "fit_heldout_protocol_row_ids_sha256": (
            roster.fit_heldout_protocol_row_ids_sha256
        ),
        "query_protocol_row_ids": list(roster.query_protocol_row_ids),
        "query_protocol_row_ids_sha256": roster.query_protocol_row_ids_sha256,
        "context_protocol_row_ids": [
            list(context) for context in roster.context_protocol_row_ids
        ],
        "context_protocol_row_ids_sha256": (
            roster.context_protocol_row_ids_sha256
        ),
        "context_counts": list(roster.context_counts),
        "query_count": roster.query_count,
        "total_context_count": roster.total_context_count,
        "history_consumption_count": roster.history_consumption_count,
        "strict_past_invariant_ids": list(roster.strict_past_invariant_ids),
    }


def _validate_roster_receipt_shape(roster: StrictPastContextRoster) -> None:
    if roster.schema_version != CONTEXT_ROSTER_SCHEMA:
        raise HarmBenchContextError("context roster schema changed")
    if not isinstance(roster.dataset_id, str) or not roster.dataset_id:
        raise HarmBenchContextError("context roster dataset identity is empty")
    if not isinstance(roster.source_role, str) or not roster.source_role:
        raise HarmBenchContextError("context roster source role is empty")
    if roster.context_role not in CONTEXT_ROLE_IDS:
        raise HarmBenchContextError("context role is outside the frozen roster")
    if roster.strategy_id not in CONTEXT_STRATEGY_IDS:
        raise HarmBenchContextError("context strategy is outside the frozen roster")
    expected_rule_fields = _strategy_rule_fields(roster.strategy_id)
    for name, expected in expected_rule_fields.items():
        if getattr(roster, name) != expected:
            raise HarmBenchContextError(f"context strategy rule changed: {name}")
    _exact_nonnegative_integer(roster.training_seed, name="training_seed")
    _exact_nonnegative_integer(roster.fold, name="fold")
    if (
        not isinstance(roster.query_protocol_row_ids, tuple)
        or not isinstance(roster.context_protocol_row_ids, tuple)
        or any(
            not isinstance(context, tuple)
            for context in roster.context_protocol_row_ids
        )
        or not isinstance(roster.context_counts, tuple)
        or not isinstance(roster.strict_past_invariant_ids, tuple)
        or not isinstance(roster.modality_order, tuple)
    ):
        raise HarmBenchContextError("context roster row containers must be immutable tuples")
    for name in (
        "fit_plan_capability_sha256",
        "strategy_rule_sha256",
        "source_capability_sha256",
        "cross_role_feature_roster_sha256",
        "source_content_sha256",
        "source_row_alignment_sha256",
        "processor_receipt_sha256",
        "processed_output_receipt_sha256",
        "processed_output_row_alignment_sha256",
        "crossfit_plan_sha256",
        "fit_train_protocol_row_ids_sha256",
        "fit_heldout_protocol_row_ids_sha256",
        "query_protocol_row_ids_sha256",
        "context_protocol_row_ids_sha256",
        "roster_sha256",
    ):
        _valid_sha256(getattr(roster, name), name=name)

    query_ids = _integer_tuple(
        roster.query_protocol_row_ids,
        name="query_protocol_row_ids",
        nonempty=True,
    )
    contexts = _contexts_tuple(roster.context_protocol_row_ids)
    counts = _integer_tuple(
        roster.context_counts,
        name="context_counts",
        nonempty=True,
        unique=False,
    )
    if len(contexts) != len(query_ids) or len(counts) != len(query_ids):
        raise HarmBenchContextError("context roster arrays are not query-aligned")
    if counts != tuple(len(context) for context in contexts):
        raise HarmBenchContextError("context counts differ from context rows")
    query_count = _exact_nonnegative_integer(roster.query_count, name="query_count")
    total_context_count = _exact_nonnegative_integer(
        roster.total_context_count, name="total_context_count"
    )
    history_consumption_count = _exact_nonnegative_integer(
        roster.history_consumption_count,
        name="history_consumption_count",
    )
    if query_count != len(query_ids):
        raise HarmBenchContextError("query count differs from ordered query rows")
    total = sum(counts)
    if total_context_count != total:
        raise HarmBenchContextError("total context count differs from context rows")
    if history_consumption_count != total:
        raise HarmBenchContextError("history consumption differs from context rows")
    if roster.strategy_id == CURRENT_ONLY_STRATEGY_ID and (
        total != 0
        or any(contexts)
        or any(counts)
        or roster.history_consumption_count != 0
    ):
        raise HarmBenchContextError(
            "current-only must have zero context and history consumption"
        )
    if tuple(roster.strict_past_invariant_ids) != _invariant_ids(roster.strategy_id):
        raise HarmBenchContextError("strict-past invariant roster changed")
    if _canonical_array_sha256(np.asarray(query_ids, dtype=np.int64)) != (
        roster.query_protocol_row_ids_sha256
    ):
        raise HarmBenchContextError("query protocol-row hash changed")
    if _context_hash(contexts) != roster.context_protocol_row_ids_sha256:
        raise HarmBenchContextError("context protocol-row hash changed")
    if _canonical_json_sha256(_roster_descriptor(roster)) != roster.roster_sha256:
        raise HarmBenchContextError("context roster receipt changed")


@dataclass(frozen=True)
class _ResolvedContextInputs:
    fit_capability: object
    source_capability: object
    features: object
    processed: object
    processor_receipt: object
    plan: object
    training_seed: int
    fold: int
    train_indices: np.ndarray
    heldout_indices: np.ndarray
    query_indices: np.ndarray


def _validate_processor_receipt(
    processor_receipt: object,
    *,
    fit_capability: object,
    plan: object,
    training_seed: int,
    fold: int,
    train_indices: np.ndarray,
    expected_processor_receipt_sha256: str,
) -> object:
    from .harmbench_erc_processors import ProcessorReceipt

    if not isinstance(processor_receipt, ProcessorReceipt):
        raise HarmBenchContextError("processor_receipt must be a ProcessorReceipt")
    try:
        rebuilt = ProcessorReceipt(
            **{
                item.name: getattr(processor_receipt, item.name)
                for item in fields(ProcessorReceipt)
            }
        )
    except ValueError as error:
        raise HarmBenchContextError(f"processor receipt changed: {error}") from error
    fit_features = fit_capability.fit
    expected_train_ids = tuple(
        int(fit_features.protocol_row_ids[index]) for index in train_indices
    )
    exact_bindings = {
        "dataset_id": fit_features.dataset_id,
        "source_role": fit_features.role,
        "seed": training_seed,
        "fold": fold,
        "source_capability_sha256": fit_capability.capability_sha256,
        "cross_role_feature_roster_sha256": (
            fit_capability.cross_role_feature_roster_sha256
        ),
        "crossfit_plan_sha256": plan.plan_sha256,
        "train_protocol_row_ids": expected_train_ids,
        "source_row_alignment_sha256": fit_features.row_alignment_sha256,
        "source_content_sha256": fit_features.content_sha256,
        "processor_receipt_sha256": _valid_sha256(
            expected_processor_receipt_sha256,
            name="expected_processor_receipt_sha256",
        ),
    }
    if any(getattr(rebuilt, name) != value for name, value in exact_bindings.items()):
        raise HarmBenchContextError("processor receipt differs from live fold binding")
    return rebuilt


def _resolve_live_inputs(
    fit_plan_capability: object,
    source_capability: object,
    processed_features: object,
    processor_receipt: object,
    crossfit_plan: object,
    *,
    training_seed: object,
    fold: object,
    context_role: str,
    expected_fit_plan_capability_sha256: str,
    expected_source_capability_sha256: str,
    expected_processor_receipt_sha256: str,
    expected_processed_output_receipt_sha256: str,
    expected_crossfit_plan_sha256: str,
) -> _ResolvedContextInputs:
    from .harmbench_erc_crossfit import (
        SharedGroupCrossfitPlan,
        resolve_shared_group_crossfit_indices,
        validate_shared_group_crossfit_plan,
    )
    from .harmbench_erc_open_roles import (
        FitFeatureCapability,
        SelectionFeatureCapability,
        validate_fit_feature_capability,
        validate_selection_feature_capability,
    )
    from .harmbench_erc_processors import validate_processed_role_embeddings

    if context_role not in CONTEXT_ROLE_IDS:
        raise HarmBenchContextError("context role is outside the frozen roster")
    seed_value = _exact_nonnegative_integer(training_seed, name="training_seed")
    fold_value = _exact_nonnegative_integer(fold, name="fold")
    try:
        fit_capability = validate_fit_feature_capability(fit_plan_capability)
    except ValueError as error:
        raise HarmBenchContextError(f"fit-plan capability changed: {error}") from error
    if fit_capability.capability_sha256 != _valid_sha256(
        expected_fit_plan_capability_sha256,
        name="expected_fit_plan_capability_sha256",
    ):
        raise HarmBenchContextError("fit-plan capability differs from external binding")
    if not isinstance(crossfit_plan, SharedGroupCrossfitPlan):
        raise HarmBenchContextError("crossfit_plan must be a SharedGroupCrossfitPlan")
    try:
        validate_shared_group_crossfit_plan(crossfit_plan, fit_capability)
        train_indices, heldout_indices = resolve_shared_group_crossfit_indices(
            crossfit_plan,
            fit_capability,
            training_seed=seed_value,
            fold=fold_value,
        )
    except ValueError as error:
        raise HarmBenchContextError(f"crossfit plan changed: {error}") from error
    if crossfit_plan.plan_sha256 != _valid_sha256(
        expected_crossfit_plan_sha256, name="expected_crossfit_plan_sha256"
    ):
        raise HarmBenchContextError("crossfit plan differs from external binding")

    expected_source_sha = _valid_sha256(
        expected_source_capability_sha256,
        name="expected_source_capability_sha256",
    )
    try:
        if isinstance(source_capability, FitFeatureCapability):
            source = validate_fit_feature_capability(source_capability)
            features = source.fit
            if context_role == SELECTION_CONTEXT_ROLE:
                raise HarmBenchContextError(
                    "selection context requires a selection feature capability"
                )
            if source.capability_sha256 != fit_capability.capability_sha256:
                raise HarmBenchContextError(
                    "fit context source differs from the plan capability"
                )
        elif isinstance(source_capability, SelectionFeatureCapability):
            source = validate_selection_feature_capability(source_capability)
            features = source.selection
            if context_role != SELECTION_CONTEXT_ROLE:
                raise HarmBenchContextError(
                    "fit context requires the fit feature capability"
                )
        else:
            raise HarmBenchContextError(
                "source_capability must be fit-feature or selection-feature only"
            )
    except HarmBenchContextError:
        raise
    except ValueError as error:
        raise HarmBenchContextError(f"source capability changed: {error}") from error
    if source.capability_sha256 != expected_source_sha:
        raise HarmBenchContextError("source capability differs from external binding")
    if source.dataset_id != fit_capability.dataset_id:
        raise HarmBenchContextError("source and fit-plan datasets differ")
    if source.cross_role_feature_roster_sha256 != (
        fit_capability.cross_role_feature_roster_sha256
    ):
        raise HarmBenchContextError("source and fit-plan feature rosters differ")
    if isinstance(source, SelectionFeatureCapability):
        fit_features = fit_capability.fit
        if set(features.groups.tolist()).intersection(fit_features.groups.tolist()):
            raise HarmBenchContextError("selection source shares a fit independent group")
        if set(features.protocol_row_ids.tolist()).intersection(
            fit_features.protocol_row_ids.tolist()
        ):
            raise HarmBenchContextError("selection source shares a fit protocol row")
        if set(features.keys.tolist()).intersection(fit_features.keys.tolist()):
            raise HarmBenchContextError("selection source shares a fit row key")

    receipt = _validate_processor_receipt(
        processor_receipt,
        fit_capability=fit_capability,
        plan=crossfit_plan,
        training_seed=seed_value,
        fold=fold_value,
        train_indices=train_indices,
        expected_processor_receipt_sha256=expected_processor_receipt_sha256,
    )
    try:
        processed = validate_processed_role_embeddings(
            processed_features,
            expected_source_capability_sha256=source.capability_sha256,
            expected_processor_receipt_sha256=receipt.processor_receipt_sha256,
            expected_output_receipt_sha256=_valid_sha256(
                expected_processed_output_receipt_sha256,
                name="expected_processed_output_receipt_sha256",
            ),
        )
    except ValueError as error:
        raise HarmBenchContextError(f"processed feature output changed: {error}") from error
    if (
        processed.dataset_id != features.dataset_id
        or processed.role != features.role
        or processed.source_content_sha256 != features.content_sha256
        or processed.source_row_alignment_sha256 != features.row_alignment_sha256
        or processed.cross_role_feature_roster_sha256
        != source.cross_role_feature_roster_sha256
        or not np.array_equal(processed.protocol_row_ids, features.protocol_row_ids)
    ):
        raise HarmBenchContextError("processed output differs from live source alignment")

    fit_groups = np.asarray(fit_capability.fit.groups, dtype=str)
    train_group_set = set(fit_groups[train_indices].tolist())
    heldout_group_set = set(fit_groups[heldout_indices].tolist())
    if train_group_set.intersection(heldout_group_set):
        raise HarmBenchContextError("crossfit plan split an independent group")
    if context_role == FIT_TRAIN_CONTEXT_ROLE:
        query_indices = np.asarray(train_indices, dtype=np.int64)
    elif context_role == FIT_HELDOUT_OOF_CONTEXT_ROLE:
        query_indices = np.asarray(heldout_indices, dtype=np.int64)
    else:
        query_indices = np.arange(features.rows, dtype=np.int64)
    query_indices = np.asarray(
        sorted(
            query_indices.tolist(),
            key=lambda index: int(features.protocol_row_ids[index]),
        ),
        dtype=np.int64,
    )
    query_indices.setflags(write=False)
    return _ResolvedContextInputs(
        fit_capability=fit_capability,
        source_capability=source,
        features=features,
        processed=processed,
        processor_receipt=receipt,
        plan=crossfit_plan,
        training_seed=seed_value,
        fold=fold_value,
        train_indices=train_indices,
        heldout_indices=heldout_indices,
        query_indices=query_indices,
    )


def _cosine_score(matrix: np.ndarray, query: int, candidate: int) -> float:
    query_vector = np.asarray(matrix[query], dtype=np.float64)
    candidate_vector = np.asarray(matrix[candidate], dtype=np.float64)
    denominator = float(np.linalg.norm(query_vector) * np.linalg.norm(candidate_vector))
    if denominator == 0.0:
        return 0.0
    score = float(np.dot(query_vector, candidate_vector) / denominator)
    if not np.isfinite(score):
        raise HarmBenchContextError("cosine scoring produced a non-finite value")
    return score


def _canonical_indices(features: object, indices: Sequence[int]) -> tuple[int, ...]:
    return tuple(
        sorted(
            (int(value) for value in indices),
            key=lambda index: (
                int(features.turn_ids[index]),
                int(features.protocol_row_ids[index]),
            ),
        )
    )


def _dialogue_history(
    features: object,
    *,
    query: int,
    allowed_indices: set[int],
) -> tuple[int, ...]:
    candidates = (
        candidate
        for candidate in allowed_indices
        if features.groups[candidate] == features.groups[query]
        and int(features.turn_ids[candidate]) < int(features.turn_ids[query])
    )
    return _canonical_indices(features, candidates)


def _select_context_indices(
    resolved: _ResolvedContextInputs,
    *,
    query: int,
    strategy_id: str,
    allowed_indices: set[int],
) -> tuple[int, ...]:
    features = resolved.features
    processed = resolved.processed
    rule = _strategy_rule_fields(strategy_id)
    if strategy_id == CURRENT_ONLY_STRATEGY_ID:
        return ()
    history = _dialogue_history(
        features,
        query=query,
        allowed_indices=allowed_indices,
    )
    if strategy_id == "dialogue_all_past":
        return history
    if strategy_id == "same_speaker_all_past":
        return tuple(
            candidate
            for candidate in history
            if features.speaker_identity[candidate] == features.speaker_identity[query]
        )
    if strategy_id == "recent_k3":
        top_k = rule["top_k"]
        if type(top_k) is not int:
            raise HarmBenchContextError("recent_k3 registered top-k is invalid")
        ranked = sorted(
            history,
            key=lambda candidate: (
                -int(features.turn_ids[candidate]),
                int(features.protocol_row_ids[candidate]),
            ),
        )
        return _canonical_indices(features, ranked[:top_k])
    if strategy_id == "similarity_top3":
        top_k = rule["top_k"]
        if type(top_k) is not int:
            raise HarmBenchContextError("similarity_top3 registered top-k is invalid")
        ranked = sorted(
            history,
            key=lambda candidate: (
                -_cosine_score(processed.fusion, query, candidate),
                int(features.protocol_row_ids[candidate]),
            ),
        )
        return _canonical_indices(features, ranked[:top_k])
    if strategy_id == "modality_balanced_top3":
        if not history:
            return ()
        top_k = rule["top_k"]
        modality_order = rule["modality_order"]
        if type(top_k) is not int or modality_order != REQUIRED_MODALITY_IDS:
            raise HarmBenchContextError(
                "modality-balanced registered top-k or modality order is invalid"
            )
        rankings = {
            modality: tuple(
                sorted(
                    history,
                    key=lambda candidate: (
                        -_cosine_score(
                            getattr(processed, modality), query, candidate
                        ),
                        int(features.protocol_row_ids[candidate]),
                    ),
                )
            )
            for modality in modality_order
        }
        target = min(top_k, len(history))
        selected: list[int] = []
        selected_protocol_rows: set[int] = set()
        depth = 0
        while len(selected) < target:
            for modality in modality_order:
                candidate = rankings[modality][depth]
                protocol_row = int(features.protocol_row_ids[candidate])
                if protocol_row not in selected_protocol_rows:
                    selected.append(candidate)
                    selected_protocol_rows.add(protocol_row)
                    if len(selected) == target:
                        break
            depth += 1
        return _canonical_indices(features, selected)
    raise HarmBenchContextError("context strategy is outside the frozen roster")


def _build_from_resolved(
    resolved: _ResolvedContextInputs,
    *,
    context_role: str,
    strategy_id: str,
) -> StrictPastContextRoster:
    if strategy_id not in CONTEXT_STRATEGY_IDS:
        raise HarmBenchContextError("context strategy is outside the frozen roster")
    features = resolved.features
    if context_role == FIT_TRAIN_CONTEXT_ROLE:
        allowed = {int(value) for value in resolved.train_indices}
    elif context_role == FIT_HELDOUT_OOF_CONTEXT_ROLE:
        allowed = {int(value) for value in resolved.heldout_indices}
    elif context_role == SELECTION_CONTEXT_ROLE:
        allowed = set(range(features.rows))
    else:  # Defensive even though _resolve_live_inputs already checked it.
        raise HarmBenchContextError("context role is outside the frozen roster")
    query_ids = tuple(
        int(features.protocol_row_ids[index]) for index in resolved.query_indices
    )
    context_indices = tuple(
        _select_context_indices(
            resolved,
            query=int(query),
            strategy_id=strategy_id,
            allowed_indices=allowed,
        )
        for query in resolved.query_indices
    )
    context_ids = tuple(
        tuple(int(features.protocol_row_ids[index]) for index in context)
        for context in context_indices
    )
    counts = tuple(len(context) for context in context_ids)
    strategy_rule = _strategy_rule_fields(strategy_id)
    fit_features = resolved.fit_capability.fit
    train_ids = np.asarray(
        fit_features.protocol_row_ids[resolved.train_indices], dtype=np.int64
    )
    heldout_ids = np.asarray(
        fit_features.protocol_row_ids[resolved.heldout_indices], dtype=np.int64
    )
    descriptor = {
        "schema_version": CONTEXT_ROSTER_SCHEMA,
        "dataset_id": features.dataset_id,
        "source_role": features.role,
        "context_role": context_role,
        "strategy_id": strategy_id,
        **{
            name: list(value) if name == "modality_order" else value
            for name, value in strategy_rule.items()
        },
        "training_seed": resolved.training_seed,
        "fold": resolved.fold,
        "fit_plan_capability_sha256": (
            resolved.fit_capability.capability_sha256
        ),
        "source_capability_sha256": (
            resolved.source_capability.capability_sha256
        ),
        "cross_role_feature_roster_sha256": (
            resolved.source_capability.cross_role_feature_roster_sha256
        ),
        "source_content_sha256": features.content_sha256,
        "source_row_alignment_sha256": features.row_alignment_sha256,
        "processor_receipt_sha256": (
            resolved.processor_receipt.processor_receipt_sha256
        ),
        "processed_output_receipt_sha256": (
            resolved.processed.output_receipt_sha256
        ),
        "processed_output_row_alignment_sha256": (
            resolved.processed.output_row_alignment_sha256
        ),
        "crossfit_plan_sha256": resolved.plan.plan_sha256,
        "fit_train_protocol_row_ids_sha256": _canonical_array_sha256(train_ids),
        "fit_heldout_protocol_row_ids_sha256": _canonical_array_sha256(heldout_ids),
        "query_protocol_row_ids": list(query_ids),
        "query_protocol_row_ids_sha256": _canonical_array_sha256(
            np.asarray(query_ids, dtype=np.int64)
        ),
        "context_protocol_row_ids": [list(context) for context in context_ids],
        "context_protocol_row_ids_sha256": _context_hash(context_ids),
        "context_counts": list(counts),
        "query_count": len(query_ids),
        "total_context_count": sum(counts),
        "history_consumption_count": sum(counts),
        "strict_past_invariant_ids": list(_invariant_ids(strategy_id)),
    }
    return StrictPastContextRoster(
        schema_version=CONTEXT_ROSTER_SCHEMA,
        dataset_id=features.dataset_id,
        source_role=features.role,
        context_role=context_role,
        strategy_id=strategy_id,
        strategy_rule_version=str(strategy_rule["strategy_rule_version"]),
        strategy_rule_sha256=str(strategy_rule["strategy_rule_sha256"]),
        candidate_scope=str(strategy_rule["candidate_scope"]),
        strict_past_required=bool(strategy_rule["strict_past_required"]),
        top_k=strategy_rule["top_k"],
        ranking=str(strategy_rule["ranking"]),
        ranking_tie=str(strategy_rule["ranking_tie"]),
        zero_vector_policy=str(strategy_rule["zero_vector_policy"]),
        modality_order=tuple(strategy_rule["modality_order"]),
        duplicate_skip_policy=str(strategy_rule["duplicate_skip_policy"]),
        emission_order=str(strategy_rule["emission_order"]),
        empty_fallback=str(strategy_rule["empty_fallback"]),
        training_seed=resolved.training_seed,
        fold=resolved.fold,
        fit_plan_capability_sha256=resolved.fit_capability.capability_sha256,
        source_capability_sha256=resolved.source_capability.capability_sha256,
        cross_role_feature_roster_sha256=(
            resolved.source_capability.cross_role_feature_roster_sha256
        ),
        source_content_sha256=features.content_sha256,
        source_row_alignment_sha256=features.row_alignment_sha256,
        processor_receipt_sha256=(
            resolved.processor_receipt.processor_receipt_sha256
        ),
        processed_output_receipt_sha256=resolved.processed.output_receipt_sha256,
        processed_output_row_alignment_sha256=(
            resolved.processed.output_row_alignment_sha256
        ),
        crossfit_plan_sha256=resolved.plan.plan_sha256,
        fit_train_protocol_row_ids_sha256=descriptor[
            "fit_train_protocol_row_ids_sha256"
        ],
        fit_heldout_protocol_row_ids_sha256=descriptor[
            "fit_heldout_protocol_row_ids_sha256"
        ],
        query_protocol_row_ids=query_ids,
        query_protocol_row_ids_sha256=descriptor[
            "query_protocol_row_ids_sha256"
        ],
        context_protocol_row_ids=context_ids,
        context_protocol_row_ids_sha256=descriptor[
            "context_protocol_row_ids_sha256"
        ],
        context_counts=counts,
        query_count=len(query_ids),
        total_context_count=sum(counts),
        history_consumption_count=sum(counts),
        strict_past_invariant_ids=_invariant_ids(strategy_id),
        roster_sha256=_canonical_json_sha256(descriptor),
    )


def build_strict_past_context_roster(
    fit_plan_capability: FitFeatureCapability,
    source_capability: FitFeatureCapability | SelectionFeatureCapability,
    processed_features: ProcessedRoleEmbeddings,
    processor_receipt: ProcessorReceipt,
    crossfit_plan: SharedGroupCrossfitPlan,
    *,
    training_seed: int,
    fold: int,
    context_role: str,
    strategy_id: str,
    expected_fit_plan_capability_sha256: str,
    expected_source_capability_sha256: str,
    expected_processor_receipt_sha256: str,
    expected_processed_output_receipt_sha256: str,
    expected_crossfit_plan_sha256: str,
) -> StrictPastContextRoster:
    """Build one provenance-complete context roster without outcomes.

    Fit train/heldout rows and selection rows are derived internally from the
    typed capabilities and plan.  No labels or caller-selected row indices are
    accepted.  For ``independent_current_only`` the output is mechanically
    constrained to empty contexts and zero history consumption.
    """

    resolved = _resolve_live_inputs(
        fit_plan_capability,
        source_capability,
        processed_features,
        processor_receipt,
        crossfit_plan,
        training_seed=training_seed,
        fold=fold,
        context_role=context_role,
        expected_fit_plan_capability_sha256=(
            expected_fit_plan_capability_sha256
        ),
        expected_source_capability_sha256=expected_source_capability_sha256,
        expected_processor_receipt_sha256=expected_processor_receipt_sha256,
        expected_processed_output_receipt_sha256=(
            expected_processed_output_receipt_sha256
        ),
        expected_crossfit_plan_sha256=expected_crossfit_plan_sha256,
    )
    return _build_from_resolved(
        resolved,
        context_role=context_role,
        strategy_id=strategy_id,
    )


def _assert_live_roster_invariants(
    roster: StrictPastContextRoster,
    resolved: _ResolvedContextInputs,
) -> None:
    features = resolved.features
    if roster.context_role == FIT_TRAIN_CONTEXT_ROLE:
        allowed_indices = {int(value) for value in resolved.train_indices}
    elif roster.context_role == FIT_HELDOUT_OOF_CONTEXT_ROLE:
        allowed_indices = {int(value) for value in resolved.heldout_indices}
    elif roster.context_role == SELECTION_CONTEXT_ROLE:
        allowed_indices = set(range(features.rows))
    else:
        raise HarmBenchContextError("context role is outside the frozen roster")
    by_protocol = {
        int(protocol_id): index
        for index, protocol_id in enumerate(features.protocol_row_ids)
    }
    expected_query_ids = tuple(
        int(features.protocol_row_ids[index]) for index in resolved.query_indices
    )
    try:
        observed_query_ids = tuple(roster.query_protocol_row_ids)
        observed_contexts = tuple(roster.context_protocol_row_ids)
    except TypeError as error:
        raise HarmBenchContextError("context roster rows are not sequences") from error
    if observed_query_ids != expected_query_ids:
        raise HarmBenchContextError("query rows differ from the derived partition")
    if len(observed_contexts) != len(observed_query_ids):
        raise HarmBenchContextError("contexts are not query-aligned")
    for query_id, context_ids in zip(observed_query_ids, observed_contexts, strict=True):
        if query_id not in by_protocol:
            raise HarmBenchContextError("query protocol row is outside the source")
        query = by_protocol[query_id]
        try:
            context_tuple = tuple(context_ids)
        except TypeError as error:
            raise HarmBenchContextError("context rows are not a sequence") from error
        if len(context_tuple) != len(set(context_tuple)):
            raise HarmBenchContextError("context contains duplicate protocol rows")
        context_indices: list[int] = []
        for context_id in context_tuple:
            if context_id not in by_protocol:
                raise HarmBenchContextError("context protocol row is outside the source")
            candidate = by_protocol[context_id]
            if candidate not in allowed_indices:
                raise HarmBenchContextError("context crosses the derived partition")
            if features.groups[candidate] != features.groups[query]:
                raise HarmBenchContextError("context crosses an independent group")
            if int(features.turn_ids[candidate]) >= int(features.turn_ids[query]):
                raise HarmBenchContextError("context contains a current or future turn")
            context_indices.append(candidate)
        if tuple(context_indices) != _canonical_indices(features, context_indices):
            raise HarmBenchContextError("context order is not canonical strict-past order")
    if roster.strategy_id == CURRENT_ONLY_STRATEGY_ID and (
        any(observed_contexts)
        or roster.total_context_count != 0
        or roster.history_consumption_count != 0
    ):
        raise HarmBenchContextError(
            "current-only must have zero context and history consumption"
        )


def validate_strict_past_context_roster(
    roster: StrictPastContextRoster,
    fit_plan_capability: FitFeatureCapability,
    source_capability: FitFeatureCapability | SelectionFeatureCapability,
    processed_features: ProcessedRoleEmbeddings,
    processor_receipt: ProcessorReceipt,
    crossfit_plan: SharedGroupCrossfitPlan,
    *,
    training_seed: int,
    fold: int,
    context_role: str,
    strategy_id: str,
    expected_fit_plan_capability_sha256: str,
    expected_source_capability_sha256: str,
    expected_processor_receipt_sha256: str,
    expected_processed_output_receipt_sha256: str,
    expected_crossfit_plan_sha256: str,
    expected_context_roster_sha256: str,
) -> StrictPastContextRoster:
    """Live-revalidate a roster and every upstream capability/receipt."""

    if not isinstance(roster, StrictPastContextRoster):
        raise HarmBenchContextError("roster must be a StrictPastContextRoster")
    if roster.roster_sha256 != _valid_sha256(
        expected_context_roster_sha256,
        name="expected_context_roster_sha256",
    ):
        raise HarmBenchContextError("context roster differs from external binding")
    resolved = _resolve_live_inputs(
        fit_plan_capability,
        source_capability,
        processed_features,
        processor_receipt,
        crossfit_plan,
        training_seed=training_seed,
        fold=fold,
        context_role=context_role,
        expected_fit_plan_capability_sha256=(
            expected_fit_plan_capability_sha256
        ),
        expected_source_capability_sha256=expected_source_capability_sha256,
        expected_processor_receipt_sha256=expected_processor_receipt_sha256,
        expected_processed_output_receipt_sha256=(
            expected_processed_output_receipt_sha256
        ),
        expected_crossfit_plan_sha256=expected_crossfit_plan_sha256,
    )
    _assert_live_roster_invariants(roster, resolved)
    _validate_roster_receipt_shape(roster)
    expected = _build_from_resolved(
        resolved,
        context_role=context_role,
        strategy_id=strategy_id,
    )
    for item in fields(StrictPastContextRoster):
        if getattr(roster, item.name) != getattr(expected, item.name):
            raise HarmBenchContextError(
                f"context roster differs from live derivation: {item.name}"
            )
    return roster


__all__ = [
    "CONTEXT_ROLE_IDS",
    "CONTEXT_ROSTER_SCHEMA",
    "CONTEXT_STRATEGY_IDS",
    "CURRENT_ONLY_STRATEGY_ID",
    "FIT_HELDOUT_OOF_CONTEXT_ROLE",
    "FIT_TRAIN_CONTEXT_ROLE",
    "FROZEN_TOP_K",
    "HarmBenchContextError",
    "REQUIRED_MODALITY_IDS",
    "SELECTION_CONTEXT_ROLE",
    "STRICT_PAST_INVARIANT_IDS",
    "STRICT_PAST_STRATEGY_IDS",
    "StrictPastContextRoster",
    "build_strict_past_context_roster",
    "validate_strict_past_context_roster",
]
