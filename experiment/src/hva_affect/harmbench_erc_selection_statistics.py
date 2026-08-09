"""Pure-memory aggregate statistics for the HarmBench-ERC selection audit.

The irreversible evaluator owns files, the attempt marker, and the single
label deserialisation.  This module begins only after that boundary.  Its
public analysis entry point accepts a loader-minted capability and returns an
aggregate-only report: no labels, probabilities, row identifiers, group
tokens, per-seed values, resampling draws, or paths can leave this module.

The frozen analysis is deliberately inflexible.  It evaluates the exact two
selection datasets, three co-primary model families, one current-only anchor,
five history strategies, and five training seeds from protocol v2.  All row
metrics are computed separately for each seed before aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Mapping, Sequence

import numpy as np

from . import harmbench_erc_prediction_artifact as _prediction
from . import harmbench_erc_selection_labels as _labels
from . import harmbench_erc_selection_prelabel as _prelabel
from .harmbench_erc_metrics import (
    classification_metrics,
    empirical_upper_cvar,
    paired_true_class_regret,
    regret_sign_severity_profile,
    top_label_expected_calibration_error,
)
from .harmbench_erc_protocol_v2 import (
    EXPECTED_ANCHOR_STRATEGY_ID,
    EXPECTED_CONTEXT_ROSTER_ORDER,
    EXPECTED_HISTORY_STRATEGY_ORDER,
    EXPECTED_MODEL_ORDER,
    EXPECTED_SELECTION_DATASETS,
    EXPECTED_TRAINING_SEEDS,
    PROTOCOL_V2_CANONICAL_SHA256,
)


class HarmBenchSelectionStatisticsError(ValueError):
    """Raised when the sealed input or aggregate output violates protocol."""


SELECTION_STATISTICS_SCHEMA = "harmbench_erc_selection_statistics_aggregate_v1"
JOINT_INPUT_SCHEMA = "harmbench_erc_joint_selection_evaluation_inputs_v1"
EXPLORATORY_STATUS = "previously_observed_selection_exploratory"
PRIMARY_HISTORY_STRATEGY_ID = "same_speaker_all_past"
PRIMARY_METRIC_ORDER = ("Macro-F1", "mean-regret")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_810
RANDOMIZATION_EXACT_MAX_CLUSTERS = 20
RANDOMIZATION_MC_REPLICATES = 100_000
RANDOMIZATION_SEED = 20_260_811
MINIMUM_FINITE_BOOTSTRAP_FRACTION = 0.95
HOLM_ALPHA = 0.05
MACRO_F1_PRACTICAL_MINIMUM = 0.005
MEAN_REGRET_PRACTICAL_MINIMUM = 0.01
HARM_THRESHOLDS = (0.0, 0.05)
TAIL_ALPHA = 0.90
ECE_BINS = 15
DEPTH_STRATA = (
    ("depth_1", 1, 1),
    ("depth_2_3", 2, 3),
    ("depth_4_7", 4, 7),
    ("depth_ge_8", 8, None),
)

_DATASET_SEAL = object()
_MODEL_SEAL = object()
_STRATEGY_SEAL = object()
_JOINT_SEAL = object()
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_FORBIDDEN_OUTPUT_KEY_PARTS = (
    "label",
    "probabilit",
    "row_id",
    "query_id",
    "group_token",
    "cluster_id",
    "seed_draw",
    "cluster_draw",
    "path",
    "embedding",
    "speaker",
    "dialogue_id",
    "raw_text",
)


def _exact_keys(value: object, expected: set[str], *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise HarmBenchSelectionStatisticsError(f"{name} must be a mapping")
    observed = {str(key) for key in value}
    if observed != expected:
        raise HarmBenchSelectionStatisticsError(
            f"{name} schema changed: missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )
    return value


def _readonly_copy(value: object, *, dtype: object) -> np.ndarray:
    contiguous = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    result = result.reshape(contiguous.shape)
    result.setflags(write=False)
    return result


def _probability_panel(value: object, *, queries: int, classes: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (len(EXPECTED_TRAINING_SEEDS), queries, classes):
        raise HarmBenchSelectionStatisticsError(
            f"{name} must have exact shape [5, queries, classes]"
        )
    if raw.dtype.kind != "f":
        raise HarmBenchSelectionStatisticsError(f"{name} must use a floating dtype")
    panel = np.asarray(raw, dtype=np.float64)
    if (
        not np.isfinite(panel).all()
        or np.any(panel < 0.0)
        or np.any(panel > 1.0)
        or not np.allclose(panel.sum(axis=2), 1.0, rtol=0.0, atol=1e-6)
    ):
        raise HarmBenchSelectionStatisticsError(
            f"{name} contains invalid probability rows"
        )
    return _readonly_copy(panel, dtype=np.float64)


def _unicode_vector(value: object, *, length: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or len(raw) != length or raw.dtype.kind not in {"U", "S"}:
        raise HarmBenchSelectionStatisticsError(
            f"{name} must be an aligned Unicode/string vector"
        )
    vector = np.asarray(raw, dtype=np.str_)
    if any(not str(item) or "\x00" in str(item) for item in vector.tolist()):
        raise HarmBenchSelectionStatisticsError(f"{name} contains an empty/unsafe token")
    return _readonly_copy(vector, dtype=vector.dtype)


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class _StrategyInputs:
    strategy_id: str
    probability: np.ndarray = field(repr=False, compare=False)
    use_history_mask: np.ndarray = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _STRATEGY_SEAL:
            raise HarmBenchSelectionStatisticsError(
                "strategy inputs can only be minted by the joint-input loader"
            )
        if self.probability.flags.writeable or self.use_history_mask.flags.writeable:
            raise HarmBenchSelectionStatisticsError("strategy arrays must be immutable")


@dataclass(frozen=True)
class _ModelInputs:
    model_id: str
    current_probability: np.ndarray = field(repr=False, compare=False)
    strategies: tuple[_StrategyInputs, ...] = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _MODEL_SEAL:
            raise HarmBenchSelectionStatisticsError(
                "model inputs can only be minted by the joint-input loader"
            )
        if self.current_probability.flags.writeable:
            raise HarmBenchSelectionStatisticsError("current probabilities must be immutable")


@dataclass(frozen=True)
class _DatasetInputs:
    dataset_id: str
    labels: np.ndarray = field(repr=False, compare=False)
    protocol_row_ids: np.ndarray = field(repr=False, compare=False)
    group_tokens: np.ndarray = field(repr=False, compare=False)
    class_tokens: np.ndarray = field(repr=False, compare=False)
    dialogue_history_eligible: np.ndarray = field(repr=False, compare=False)
    dialogue_depth: np.ndarray = field(repr=False, compare=False)
    models: tuple[_ModelInputs, ...] = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _DATASET_SEAL:
            raise HarmBenchSelectionStatisticsError(
                "dataset inputs can only be minted by the joint-input loader"
            )
        arrays = (
            self.labels,
            self.protocol_row_ids,
            self.group_tokens,
            self.class_tokens,
            self.dialogue_history_eligible,
            self.dialogue_depth,
        )
        if any(array.flags.writeable for array in arrays):
            raise HarmBenchSelectionStatisticsError("dataset arrays must be immutable")


@dataclass(frozen=True)
class JointSelectionEvaluationInputs:
    """Opaque, loader-sealed in-memory input for the one-shot statistics pass."""

    schema_version: str
    protocol_canonical_sha256: str
    dataset_order: tuple[str, ...]
    model_order: tuple[str, ...]
    strategy_order: tuple[str, ...]
    training_seed_order: tuple[int, ...]
    source_kind: str
    _input_sha256: str = field(repr=False)
    _datasets: tuple[_DatasetInputs, ...] = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _JOINT_SEAL:
            raise HarmBenchSelectionStatisticsError(
                "joint inputs can only be created by the module-internal loader"
            )


def _joint_input_digest(datasets: Sequence[_DatasetInputs]) -> str:
    digest = hashlib.sha256()
    digest.update(PROTOCOL_V2_CANONICAL_SHA256.encode("ascii"))
    for dataset in datasets:
        digest.update(dataset.dataset_id.encode("utf-8"))
        for array in (
            dataset.labels,
            dataset.protocol_row_ids,
            dataset.group_tokens,
            dataset.class_tokens,
            dataset.dialogue_history_eligible,
            dataset.dialogue_depth,
        ):
            digest.update(_array_digest(array).encode("ascii"))
        for model in dataset.models:
            digest.update(model.model_id.encode("ascii"))
            digest.update(_array_digest(model.current_probability).encode("ascii"))
            for strategy in model.strategies:
                digest.update(strategy.strategy_id.encode("ascii"))
                digest.update(_array_digest(strategy.probability).encode("ascii"))
                digest.update(_array_digest(strategy.use_history_mask).encode("ascii"))
    return digest.hexdigest()


def _factorize_groups(values: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    index_by_token: dict[str, int] = {}
    codes = np.empty(len(values), dtype=np.int64)
    members: list[list[int]] = []
    for index, raw in enumerate(values.tolist()):
        token = str(raw)
        code = index_by_token.get(token)
        if code is None:
            code = len(members)
            index_by_token[token] = code
            members.append([])
        codes[index] = code
        members[code].append(index)
    immutable_members = tuple(
        _readonly_copy(np.asarray(item, dtype=np.int64), dtype=np.int64)
        for item in members
    )
    return _readonly_copy(codes, dtype=np.int64), immutable_members


def _mint_dataset_inputs(dataset_id: str, source: object) -> _DatasetInputs:
    value = _exact_keys(
        source,
        {
            "labels",
            "protocol_row_ids",
            "group_tokens",
            "class_tokens",
            "dialogue_history_eligible",
            "dialogue_depth",
            "models",
        },
        name=f"dataset[{dataset_id}]",
    )
    raw_rows = np.asarray(value["protocol_row_ids"])
    if (
        raw_rows.ndim != 1
        or not len(raw_rows)
        or raw_rows.dtype.kind not in "iu"
        or raw_rows.dtype.kind == "b"
    ):
        raise HarmBenchSelectionStatisticsError("protocol rows must be integer vector")
    rows = _readonly_copy(raw_rows, dtype=np.int64)
    if len(set(rows.tolist())) != len(rows):
        raise HarmBenchSelectionStatisticsError("protocol rows must be unique")
    q = len(rows)
    groups = _unicode_vector(value["group_tokens"], length=q, name="group_tokens")
    raw_classes = np.asarray(value["class_tokens"])
    if raw_classes.ndim != 1 or len(raw_classes) < 2:
        raise HarmBenchSelectionStatisticsError("class_tokens must contain at least 2 classes")
    classes = _unicode_vector(
        raw_classes, length=len(raw_classes), name="class_tokens"
    )
    if len(set(classes.tolist())) != len(classes):
        raise HarmBenchSelectionStatisticsError("class tokens must be unique")
    raw_labels = np.asarray(value["labels"])
    if (
        raw_labels.shape != (q,)
        or raw_labels.dtype.kind not in "iu"
        or raw_labels.dtype.kind == "b"
    ):
        raise HarmBenchSelectionStatisticsError("labels must be aligned integers")
    labels = _readonly_copy(raw_labels, dtype=np.int64)
    if np.any(labels < 0) or np.any(labels >= len(classes)):
        raise HarmBenchSelectionStatisticsError("labels contain an out-of-range class")
    raw_eligible = np.asarray(value["dialogue_history_eligible"])
    if raw_eligible.shape != (q,) or raw_eligible.dtype != np.dtype("bool"):
        raise HarmBenchSelectionStatisticsError("E_dialogue must be an exact boolean vector")
    eligible = _readonly_copy(raw_eligible, dtype=np.bool_)
    if not np.any(eligible):
        raise HarmBenchSelectionStatisticsError("E_dialogue population is empty")
    raw_depth = np.asarray(value["dialogue_depth"])
    if raw_depth.shape != (q,) or raw_depth.dtype.kind not in "iu" or raw_depth.dtype.kind == "b":
        raise HarmBenchSelectionStatisticsError("dialogue depth must be aligned integers")
    depth = _readonly_copy(raw_depth, dtype=np.int64)
    if np.any(depth < 0) or not np.array_equal(depth > 0, eligible):
        raise HarmBenchSelectionStatisticsError(
            "dialogue depth must be positive exactly on E_dialogue"
        )
    if len(set(groups.tolist())) < 1:
        raise HarmBenchSelectionStatisticsError("dataset cluster roster is empty")

    model_sources = _exact_keys(
        value["models"], set(EXPECTED_MODEL_ORDER), name=f"models[{dataset_id}]"
    )
    models: list[_ModelInputs] = []
    for model_id in EXPECTED_MODEL_ORDER:
        model_value = _exact_keys(
            model_sources[model_id],
            {"current_probability", "strategies"},
            name=f"model[{dataset_id},{model_id}]",
        )
        current = _probability_panel(
            model_value["current_probability"],
            queries=q,
            classes=len(classes),
            name=f"current_probability[{dataset_id},{model_id}]",
        )
        strategy_sources = _exact_keys(
            model_value["strategies"],
            set(EXPECTED_HISTORY_STRATEGY_ORDER),
            name=f"strategies[{dataset_id},{model_id}]",
        )
        strategies: list[_StrategyInputs] = []
        for strategy_id in EXPECTED_HISTORY_STRATEGY_ORDER:
            strategy_value = _exact_keys(
                strategy_sources[strategy_id],
                {"probability", "use_history_mask"},
                name=f"strategy[{dataset_id},{model_id},{strategy_id}]",
            )
            probability = _probability_panel(
                strategy_value["probability"],
                queries=q,
                classes=len(classes),
                name=f"probability[{dataset_id},{model_id},{strategy_id}]",
            )
            raw_mask = np.asarray(strategy_value["use_history_mask"])
            if raw_mask.shape != (len(EXPECTED_TRAINING_SEEDS), q) or raw_mask.dtype != np.dtype("bool"):
                raise HarmBenchSelectionStatisticsError(
                    "history-use mask must have exact boolean shape [5, queries]"
                )
            mask = _readonly_copy(raw_mask, dtype=np.bool_)
            if not np.all(mask == mask[0, :][None, :]):
                raise HarmBenchSelectionStatisticsError(
                    "history-use mask changed across training seeds"
                )
            if np.any(mask & ~eligible[None, :]):
                raise HarmBenchSelectionStatisticsError(
                    "history-use mask exceeds E_dialogue"
                )
            if strategy_id == "dialogue_all_past" and not np.array_equal(
                mask, np.broadcast_to(eligible, mask.shape)
            ):
                raise HarmBenchSelectionStatisticsError(
                    "dialogue_all_past coverage must exactly equal E_dialogue"
                )
            if not np.array_equal(probability[~mask], current[~mask]):
                raise HarmBenchSelectionStatisticsError(
                    "effective history probability lacks exact current-only fallback"
                )
            strategies.append(
                _StrategyInputs(
                    strategy_id=strategy_id,
                    probability=probability,
                    use_history_mask=mask,
                    _seal=_STRATEGY_SEAL,
                )
            )
        models.append(
            _ModelInputs(
                model_id=model_id,
                current_probability=current,
                strategies=tuple(strategies),
                _seal=_MODEL_SEAL,
            )
        )
    return _DatasetInputs(
        dataset_id=dataset_id,
        labels=labels,
        protocol_row_ids=rows,
        group_tokens=groups,
        class_tokens=classes,
        dialogue_history_eligible=eligible,
        dialogue_depth=depth,
        models=tuple(models),
        _seal=_DATASET_SEAL,
    )


def _mint_joint_inputs(
    raw_datasets: object, *, source_kind: str
) -> JointSelectionEvaluationInputs:
    if source_kind not in {"attempt_bound_activated_labels", "trusted_synthetic_fixture_only"}:
        raise HarmBenchSelectionStatisticsError("joint-input source kind is invalid")
    sources = _exact_keys(
        raw_datasets, set(EXPECTED_SELECTION_DATASETS), name="joint datasets"
    )
    datasets = tuple(
        _mint_dataset_inputs(dataset_id, sources[dataset_id])
        for dataset_id in EXPECTED_SELECTION_DATASETS
    )
    digest = _joint_input_digest(datasets)
    return JointSelectionEvaluationInputs(
        schema_version=JOINT_INPUT_SCHEMA,
        protocol_canonical_sha256=PROTOCOL_V2_CANONICAL_SHA256,
        dataset_order=tuple(EXPECTED_SELECTION_DATASETS),
        model_order=tuple(EXPECTED_MODEL_ORDER),
        strategy_order=tuple(EXPECTED_CONTEXT_ROSTER_ORDER),
        training_seed_order=tuple(EXPECTED_TRAINING_SEEDS),
        source_kind=source_kind,
        _input_sha256=digest,
        _datasets=datasets,
        _seal=_JOINT_SEAL,
    )


def _make_trusted_synthetic_joint_selection_evaluation_inputs(
    raw_datasets: object,
) -> JointSelectionEvaluationInputs:
    """Private synthetic-only factory for tests; never a real-data ingest seam."""

    return _mint_joint_inputs(
        raw_datasets, source_kind="trusted_synthetic_fixture_only"
    )


def load_joint_selection_evaluation_inputs(
    attempt: _prelabel.AttemptStartedCapability,
    activated_labels: Sequence[_labels.ActivatedSelectionLabelCapability],
) -> JointSelectionEvaluationInputs:
    """Mint the in-memory capability after marker fsync and one-time label load.

    No label file is opened here.  The activated capabilities already contain
    immutable in-memory labels; the attempt capability supplies the exact live
    36-artifact roster and 30 fallback-completed effective pairs.
    """

    # Revalidate the attempt marker and prelabel bundle at the loader boundary.
    # This may reopen outcome-free receipts/predictions, but never a label NPZ;
    # the statistics core below remains strictly pure-memory.
    if (
        type(attempt) is not _prelabel.AttemptStartedCapability
        or attempt._seal is not _prelabel._ATTEMPT_SEAL  # noqa: SLF001
        or type(attempt._prelabel) is not _prelabel.LoadedSelectionPrelabelBundle  # noqa: SLF001
        or attempt._prelabel._seal is not _prelabel._BUNDLE_SEAL  # noqa: SLF001
    ):
        raise HarmBenchSelectionStatisticsError(
            "a loader-sealed attempt capability is required"
        )
    try:
        live_attempt = _prelabel._revalidate_attempt_started_capability(  # noqa: SLF001
            attempt
        )
    except (TypeError, ValueError, OSError) as error:
        raise HarmBenchSelectionStatisticsError(
            "attempt marker/prelabel live revalidation failed"
        ) from error
    if (
        live_attempt.protocol_canonical_sha256
        != live_attempt._prelabel.protocol_canonical_sha256  # noqa: SLF001
        or live_attempt.prelabel_bundle_file_sha256
        != live_attempt._prelabel.bundle_file_sha256  # noqa: SLF001
        or live_attempt.prelabel_receipt_file_sha256
        != live_attempt._prelabel.receipt_file_sha256  # noqa: SLF001
        or live_attempt.marker.get("prelabel_bundle_file_sha256")
        != live_attempt.prelabel_bundle_file_sha256
        or live_attempt.marker.get("prelabel_receipt_file_sha256")
        != live_attempt.prelabel_receipt_file_sha256
        or live_attempt.marker.get("protocol_canonical_sha256")
        != live_attempt.protocol_canonical_sha256
    ):
        raise HarmBenchSelectionStatisticsError(
            "attempt/prelabel in-memory binding changed"
        )
    if live_attempt.protocol_canonical_sha256 != PROTOCOL_V2_CANONICAL_SHA256:
        raise HarmBenchSelectionStatisticsError("attempt protocol pin changed")
    if (
        isinstance(activated_labels, (str, bytes))
        or not isinstance(activated_labels, Sequence)
        or len(activated_labels) != len(EXPECTED_SELECTION_DATASETS)
    ):
        raise HarmBenchSelectionStatisticsError(
            "exactly two activated label capabilities are required"
        )
    labels_by_dataset: dict[str, _labels.ActivatedSelectionLabelCapability] = {}
    for raw in activated_labels:
        try:
            live_label = _labels._revalidate_activated_selection_labels(raw)  # noqa: SLF001
        except (TypeError, ValueError) as error:
            raise HarmBenchSelectionStatisticsError(
                "labels must be loader-activated after the attempt marker"
            ) from error
        if live_label.dataset_id in labels_by_dataset:
            raise HarmBenchSelectionStatisticsError("duplicate activated label dataset")
        labels_by_dataset[live_label.dataset_id] = live_label
    if set(labels_by_dataset) != set(EXPECTED_SELECTION_DATASETS):
        raise HarmBenchSelectionStatisticsError("activated label dataset roster changed")
    if len({item.artifact_file_sha256 for item in labels_by_dataset.values()}) != 2 or len(
        {item.manifest_file_sha256 for item in labels_by_dataset.values()}
    ) != 2:
        raise HarmBenchSelectionStatisticsError(
            "activated label files/manifests must remain distinct"
        )
    try:
        _prelabel._validate_activated_label_suite_for_attempt(  # noqa: SLF001
            live_attempt,
            tuple(labels_by_dataset[dataset_id] for dataset_id in EXPECTED_SELECTION_DATASETS),
        )
    except (TypeError, ValueError, OSError) as error:
        raise HarmBenchSelectionStatisticsError(
            "activated labels are not the exact consumed ticket suite for this attempt"
        ) from error

    manifests_by_dataset = {
        item.dataset_id: item
        for item in live_attempt._prelabel._label_manifests  # noqa: SLF001
    }
    if set(manifests_by_dataset) != set(EXPECTED_SELECTION_DATASETS):
        raise HarmBenchSelectionStatisticsError("attempt label-manifest roster changed")
    for dataset_id in EXPECTED_SELECTION_DATASETS:
        activated = labels_by_dataset[dataset_id]
        manifest = manifests_by_dataset[dataset_id]
        if (
            activated.role != _labels.SELECTION_LABEL_ROLE
            or activated.dataset_id != manifest.dataset_id
            or activated.rows != manifest.rows
            or activated.ordered_protocol_row_alignment_sha256
            != manifest.ordered_protocol_row_alignment_sha256
            or activated.class_order_sha256 != manifest.class_order_sha256
            or activated.artifact_file_sha256 != manifest.artifact_file_sha256
            or activated.manifest_file_sha256 != manifest.manifest_file_sha256
            or activated.protocol_canonical_sha256
            != live_attempt.protocol_canonical_sha256
            or activated.attempt_marker_file_sha256
            != live_attempt.marker_file_sha256
            or activated.prelabel_bundle_file_sha256
            != live_attempt.prelabel_bundle_file_sha256
            or activated.prelabel_receipt_file_sha256
            != live_attempt.prelabel_receipt_file_sha256
        ):
            raise HarmBenchSelectionStatisticsError(
                "activated labels are not the exact attempt-bound sidecars"
            )

    artifacts: dict[tuple[str, str, str], _prediction.LoadedPredictionArtifact] = {}
    for raw in live_attempt._prelabel._prediction_artifacts:  # noqa: SLF001
        if (
            type(raw) is not _prediction.LoadedPredictionArtifact
            or raw._seal is not _prediction._LOADED_SEAL  # noqa: SLF001
            or raw.role != _prediction.SELECTION_ROLE
        ):
            raise HarmBenchSelectionStatisticsError(
                "the exact prediction roster is not loader-sealed"
            )
        try:
            artifact = _prediction._revalidate_loaded_prediction_artifact(  # noqa: SLF001
                raw, expected_role=_prediction.SELECTION_ROLE
            )
        except (TypeError, ValueError, OSError) as error:
            raise HarmBenchSelectionStatisticsError(
                "hidden prediction artifact failed live revalidation"
            ) from error
        key = (artifact.dataset_id, artifact.model_id, artifact.strategy_id)
        if key in artifacts:
            raise HarmBenchSelectionStatisticsError("duplicate prediction roster cell")
        artifacts[key] = artifact
    expected_artifact_keys = {
        (dataset_id, model_id, strategy_id)
        for dataset_id in EXPECTED_SELECTION_DATASETS
        for model_id in EXPECTED_MODEL_ORDER
        for strategy_id in EXPECTED_CONTEXT_ROSTER_ORDER
    }
    if set(artifacts) != expected_artifact_keys:
        raise HarmBenchSelectionStatisticsError("exact 36-artifact roster changed")
    artifact_values = tuple(artifacts.values())
    if (
        len({item.artifact_file_sha256 for item in artifact_values}) != 36
        or len({item.receipt_file_sha256 for item in artifact_values}) != 36
        or len({item.panel_sha256 for item in artifact_values}) != 36
        or len({(item.artifact_path, item.receipt_path) for item in artifact_values})
        != 36
    ):
        raise HarmBenchSelectionStatisticsError(
            "prediction artifacts/files/panels are not unique"
        )

    bundle_artifact_records: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for dataset_record in live_attempt._prelabel.bundle["datasets"]:  # noqa: SLF001
        if not isinstance(dataset_record, Mapping):
            raise HarmBenchSelectionStatisticsError("prelabel dataset binding changed")
        for record in dataset_record["prediction_artifacts"]:
            if not isinstance(record, Mapping):
                raise HarmBenchSelectionStatisticsError("prelabel artifact binding changed")
            key = (
                str(record["dataset_id"]),
                str(record["model_id"]),
                str(record["strategy_id"]),
            )
            if key in bundle_artifact_records:
                raise HarmBenchSelectionStatisticsError("duplicate prelabel artifact binding")
            bundle_artifact_records[key] = record
    if set(bundle_artifact_records) != expected_artifact_keys:
        raise HarmBenchSelectionStatisticsError("prelabel artifact binding roster changed")
    for key, artifact in artifacts.items():
        record = bundle_artifact_records[key]
        if (
            record.get("artifact_file_sha256") != artifact.artifact_file_sha256
            or record.get("receipt_file_sha256") != artifact.receipt_file_sha256
            or record.get("panel_sha256") != artifact.panel_sha256
            or record.get("receipt_payload_sha256")
            != _prediction.public_prediction_receipt_sha256(artifact.receipt)
            or tuple(record.get("training_seed_ids", ()))
            != tuple(EXPECTED_TRAINING_SEEDS)
            or record.get("fold_count") != 5
            or record.get("entry_count") != 25
            or artifact.receipt.get("probability_sha256")
            != _prediction._array_sha256(artifact.probabilities)  # noqa: SLF001
            or artifact.receipt.get("per_fold_probability_sha256")
            != _prediction._array_sha256(artifact.per_fold_probabilities)  # noqa: SLF001
            or artifact.receipt.get("query_roster_sha256")
            != _prediction._array_sha256(artifact.query_protocol_row_ids)  # noqa: SLF001
            or artifact.receipt.get("group_roster_sha256")
            != _prediction._array_sha256(artifact.group_tokens)  # noqa: SLF001
            or artifact.receipt.get("context_count_sha256")
            != _prediction._array_sha256(artifact.context_count)  # noqa: SLF001
            or artifact.receipt.get("strategy_context_nonempty_sha256")
            != _prediction._array_sha256(artifact.strategy_context_nonempty)  # noqa: SLF001
            or artifact.receipt.get("dialogue_history_eligible_sha256")
            != _prediction._array_sha256(artifact.dialogue_history_eligible)  # noqa: SLF001
        ):
            raise HarmBenchSelectionStatisticsError(
                "hidden prediction artifact differs from the frozen prelabel binding"
            )

    pairs: dict[tuple[str, str, str], _prediction.EffectiveHistoryCurrentPair] = {}
    for raw in live_attempt._prelabel._effective_pairs:  # noqa: SLF001
        if (
            type(raw) is not _prediction.EffectiveHistoryCurrentPair
            or raw._seal is not _prediction._PAIR_SEAL  # noqa: SLF001
        ):
            raise HarmBenchSelectionStatisticsError(
                "effective history/current fallback pair is not sealed"
            )
        try:
            pair = _prediction._revalidate_effective_history_current_pair(  # noqa: SLF001
                raw
            )
        except (TypeError, ValueError, OSError) as error:
            raise HarmBenchSelectionStatisticsError(
                "hidden effective pair failed live reconstruction/revalidation"
            ) from error
        key = (pair.dataset_id, pair.model_id, pair.history_strategy_id)
        if key in pairs:
            raise HarmBenchSelectionStatisticsError("duplicate effective pair")
        pairs[key] = pair
    expected_pair_keys = {
        (dataset_id, model_id, strategy_id)
        for dataset_id in EXPECTED_SELECTION_DATASETS
        for model_id in EXPECTED_MODEL_ORDER
        for strategy_id in EXPECTED_HISTORY_STRATEGY_ORDER
    }
    if set(pairs) != expected_pair_keys:
        raise HarmBenchSelectionStatisticsError("exact 30 effective-pair roster changed")
    if len({pair.pair_receipt_sha256 for pair in pairs.values()}) != 30:
        raise HarmBenchSelectionStatisticsError("effective-pair receipts are not unique")
    for key, pair in pairs.items():
        dataset_id, model_id, strategy_id = key
        expected_history = artifacts[(dataset_id, model_id, strategy_id)]
        expected_current = artifacts[
            (dataset_id, model_id, EXPECTED_ANCHOR_STRATEGY_ID)
        ]
        if (
            pair._history_artifact is not expected_history  # noqa: SLF001
            or pair._current_artifact is not expected_current  # noqa: SLF001
            or pair.history_artifact_file_sha256
            != expected_history.artifact_file_sha256
            or pair.current_artifact_file_sha256
            != expected_current.artifact_file_sha256
            or pair.pair_receipt_sha256
            != _prediction._canonical_sha256(pair.receipt)  # noqa: SLF001
            or pair.effective_probability_sha256
            != _prediction._array_sha256(pair.probabilities)  # noqa: SLF001
            or pair.use_history_mask_sha256
            != _prediction._array_sha256(pair.use_history_mask)  # noqa: SLF001
            or pair.dialogue_history_eligible_sha256
            != _prediction._array_sha256(pair.dialogue_history_eligible)  # noqa: SLF001
        ):
            raise HarmBenchSelectionStatisticsError(
                "hidden effective pair differs from its exact artifact lineage"
            )

    raw_datasets: dict[str, object] = {}
    for dataset_id in EXPECTED_SELECTION_DATASETS:
        label = labels_by_dataset[dataset_id]
        reference = artifacts[
            (dataset_id, EXPECTED_MODEL_ORDER[0], EXPECTED_ANCHOR_STRATEGY_ID)
        ]
        if (
            not np.array_equal(label.protocol_row_ids, reference.query_protocol_row_ids)
            or not np.array_equal(label.class_tokens, reference.class_tokens)
            or label.class_order_sha256 != reference.class_order_sha256
        ):
            raise HarmBenchSelectionStatisticsError(
                "activated labels differ from the frozen row/class alignment"
            )
        depth_reference = artifacts[
            (dataset_id, EXPECTED_MODEL_ORDER[0], "dialogue_all_past")
        ]
        depth = np.asarray(depth_reference.context_count[0, 0, :], dtype=np.int64)
        if (
            not np.all(
                depth_reference.context_count
                == depth[None, None, :]
            )
            or not np.array_equal(
                depth_reference.strategy_context_nonempty,
                depth_reference.context_count > 0,
            )
            or not np.array_equal(depth > 0, reference.dialogue_history_eligible)
        ):
            raise HarmBenchSelectionStatisticsError(
                "dialogue_all_past depth is inconsistent across 25 entries"
            )
        for other_model_id in EXPECTED_MODEL_ORDER[1:]:
            other = artifacts[(dataset_id, other_model_id, "dialogue_all_past")]
            if not np.all(other.context_count == depth[None, None, :]):
                raise HarmBenchSelectionStatisticsError(
                    "dialogue depth changed across co-primary models"
                )
        model_values: dict[str, object] = {}
        for model_id in EXPECTED_MODEL_ORDER:
            current = artifacts[(dataset_id, model_id, EXPECTED_ANCHOR_STRATEGY_ID)]
            strategies: dict[str, object] = {}
            for strategy_id in EXPECTED_HISTORY_STRATEGY_ORDER:
                pair = pairs[(dataset_id, model_id, strategy_id)]
                if (
                    pair.current_artifact_file_sha256 != current.artifact_file_sha256
                    or pair.current_receipt_file_sha256 != current.receipt_file_sha256
                ):
                    raise HarmBenchSelectionStatisticsError(
                        "effective pair does not use the unique matching current anchor"
                    )
                strategies[strategy_id] = {
                    "probability": pair.probabilities,
                    "use_history_mask": pair.use_history_mask,
                }
            model_values[model_id] = {
                "current_probability": current.probabilities,
                "strategies": strategies,
            }
        raw_datasets[dataset_id] = {
            "labels": label.labels,
            "protocol_row_ids": reference.query_protocol_row_ids,
            "group_tokens": reference.group_tokens,
            "class_tokens": reference.class_tokens,
            "dialogue_history_eligible": reference.dialogue_history_eligible,
            "dialogue_depth": depth,
            "models": model_values,
        }
    return _mint_joint_inputs(
        raw_datasets, source_kind="attempt_bound_activated_labels"
    )


def _revalidate_joint_inputs(value: object) -> JointSelectionEvaluationInputs:
    if (
        type(value) is not JointSelectionEvaluationInputs
        or value._seal is not _JOINT_SEAL
        or value.schema_version != JOINT_INPUT_SCHEMA
        or value.protocol_canonical_sha256 != PROTOCOL_V2_CANONICAL_SHA256
        or value.dataset_order != tuple(EXPECTED_SELECTION_DATASETS)
        or value.model_order != tuple(EXPECTED_MODEL_ORDER)
        or value.strategy_order != tuple(EXPECTED_CONTEXT_ROSTER_ORDER)
        or value.training_seed_order != tuple(EXPECTED_TRAINING_SEEDS)
        or value.source_kind
        not in {"attempt_bound_activated_labels", "trusted_synthetic_fixture_only"}
        or len(value._datasets) != len(EXPECTED_SELECTION_DATASETS)
        or any(type(item) is not _DatasetInputs or item._seal is not _DATASET_SEAL for item in value._datasets)
        or _joint_input_digest(value._datasets) != value._input_sha256
    ):
        raise HarmBenchSelectionStatisticsError(
            "a live module-loader-sealed JointSelectionEvaluationInputs is required"
        )
    return value


def _model_by_id(dataset: _DatasetInputs, model_id: str) -> _ModelInputs:
    index = EXPECTED_MODEL_ORDER.index(model_id)
    model = dataset.models[index]
    if model.model_id != model_id:
        raise HarmBenchSelectionStatisticsError("sealed model order changed")
    return model


def _strategy_by_id(model: _ModelInputs, strategy_id: str) -> _StrategyInputs:
    index = EXPECTED_HISTORY_STRATEGY_ORDER.index(strategy_id)
    strategy = model.strategies[index]
    if strategy.strategy_id != strategy_id:
        raise HarmBenchSelectionStatisticsError("sealed strategy order changed")
    return strategy


def _macro_f1_from_confusion(confusion: np.ndarray) -> np.ndarray:
    """Vectorized all-frozen-class Macro-F1 for arrays ending in [C,C]."""

    values = np.asarray(confusion, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != values.shape[-2]:
        raise HarmBenchSelectionStatisticsError("confusion tensor is not square")
    true_positive = np.diagonal(values, axis1=-2, axis2=-1)
    denominator = values.sum(axis=-2) + values.sum(axis=-1)
    per_class = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=denominator > 0.0,
    )
    return np.mean(per_class, axis=-1, dtype=np.float64)


def _cluster_confusions(
    labels: np.ndarray,
    predictions: np.ndarray,
    cluster_codes: np.ndarray,
    *,
    clusters: int,
    classes: int,
) -> np.ndarray:
    if predictions.ndim != 2 or predictions.shape[1] != len(labels):
        raise HarmBenchSelectionStatisticsError("prediction/label alignment changed")
    result = np.empty(
        (predictions.shape[0], clusters, classes, classes), dtype=np.int64
    )
    base = cluster_codes * (classes * classes) + labels * classes
    length = clusters * classes * classes
    for seed_index in range(predictions.shape[0]):
        linear = base + predictions[seed_index]
        result[seed_index] = np.bincount(linear, minlength=length).reshape(
            clusters, classes, classes
        )
    return result


def _depth_member(depth: np.ndarray, minimum: int, maximum: int | None) -> np.ndarray:
    member = depth >= minimum
    if maximum is not None:
        member &= depth <= maximum
    return member


def _point_cell(
    dataset: _DatasetInputs,
    model: _ModelInputs,
    strategy: _StrategyInputs,
) -> dict[str, object]:
    eligible = np.asarray(dataset.dialogue_history_eligible, dtype=np.bool_)
    labels = dataset.labels[eligible]
    current = model.current_probability[:, eligible, :]
    history = strategy.probability[:, eligible, :]
    regrets = np.empty((len(EXPECTED_TRAINING_SEEDS), len(labels)), dtype=np.float64)
    macro_differences = np.empty(len(EXPECTED_TRAINING_SEEDS), dtype=np.float64)
    ece_differences = np.empty(len(EXPECTED_TRAINING_SEEDS), dtype=np.float64)
    cvar_values = np.empty(len(EXPECTED_TRAINING_SEEDS), dtype=np.float64)
    for seed_index in range(len(EXPECTED_TRAINING_SEEDS)):
        regrets[seed_index] = paired_true_class_regret(
            labels, current[seed_index], history[seed_index]
        )
        current_metrics = classification_metrics(labels, current[seed_index])
        history_metrics = classification_metrics(labels, history[seed_index])
        macro_differences[seed_index] = (
            history_metrics["macro_f1"] - current_metrics["macro_f1"]
        )
        ece_differences[seed_index] = top_label_expected_calibration_error(
            labels, history[seed_index], bins=ECE_BINS
        ) - top_label_expected_calibration_error(
            labels, current[seed_index], bins=ECE_BINS
        )
        cvar_values[seed_index] = empirical_upper_cvar(
            regrets[seed_index], alpha=TAIL_ALPHA
        )
    flattened = regrets.reshape(-1)
    sign_severity = regret_sign_severity_profile(flattened)
    selected = strategy.use_history_mask[0, eligible]
    eligible_groups = dataset.group_tokens[eligible]
    depth = dataset.dialogue_depth[eligible]
    strata: list[dict[str, object]] = []
    estimable: list[tuple[str, float]] = []
    for stratum_id, minimum, maximum in DEPTH_STRATA:
        member = _depth_member(depth, minimum, maximum)
        queries = int(member.sum())
        clusters = int(len(set(eligible_groups[member].tolist())))
        is_estimable = clusters >= 2
        mean_regret = (
            float(np.mean(regrets[:, member], dtype=np.float64))
            if is_estimable
            else None
        )
        if mean_regret is not None:
            estimable.append((stratum_id, mean_regret))
        strata.append(
            {
                "stratum_id": stratum_id,
                "status": "estimable" if is_estimable else "not_estimable",
                "queries": queries,
                "independent_clusters": clusters,
                "seed_query_observations": int(queries * len(EXPECTED_TRAINING_SEEDS)),
                "mean_regret": mean_regret,
            }
        )
    if estimable:
        worst_id, worst_value = max(estimable, key=lambda item: item[1])
    else:
        worst_id, worst_value = None, None
    return {
        "macro_f1_difference": float(np.mean(macro_differences)),
        "mean_regret": float(np.mean(regrets, dtype=np.float64)),
        "secondary": {
            "ece_difference": float(np.mean(ece_differences)),
            "cvar90_regret": float(np.mean(cvar_values)),
            "harm_rate_gt_0": float(np.mean(flattened > HARM_THRESHOLDS[0])),
            "harm_rate_gt_0_05": float(np.mean(flattened > HARM_THRESHOLDS[1])),
            "coverage": {
                "eligible_queries": int(len(labels)),
                "history_context_nonempty_queries": int(selected.sum()),
                "rate": float(np.mean(selected)),
            },
            "sign_severity": {
                "seed_query_observations": int(sign_severity["queries"]),
                "counts": dict(sign_severity["counts"]),
                "rates": dict(sign_severity["rates"]),
            },
            "depth_strata": strata,
            "worst_depth_stratum": {
                "status": "estimable" if estimable else "not_estimable",
                "estimable_strata": int(len(estimable)),
                "stratum_id": worst_id,
                "mean_regret": worst_value,
            },
        },
        "_regrets": regrets,
    }


@dataclass(frozen=True)
class _BootstrapPlan:
    seed_weights: np.ndarray = field(repr=False)
    cluster_weights: tuple[np.ndarray, ...] = field(repr=False)
    plan_sha256: str


def _resample_weights(draws: np.ndarray, categories: int) -> np.ndarray:
    rows = draws.shape[0]
    weights = np.zeros((rows, categories), dtype=np.uint32)
    row_index = np.repeat(np.arange(rows, dtype=np.int64), draws.shape[1])
    np.add.at(weights, (row_index, draws.reshape(-1)), 1)
    weights.setflags(write=False)
    return weights


def _make_bootstrap_plan(datasets: Sequence[_DatasetInputs]) -> _BootstrapPlan:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seed_draws = rng.integers(
        0,
        len(EXPECTED_TRAINING_SEEDS),
        size=(BOOTSTRAP_REPLICATES, len(EXPECTED_TRAINING_SEEDS)),
        dtype=np.int16,
    )
    seed_weights = _resample_weights(seed_draws, len(EXPECTED_TRAINING_SEEDS))
    cluster_weights: list[np.ndarray] = []
    digest = hashlib.sha256()
    digest.update(b"harmbench_erc_shared_bootstrap_v2")
    digest.update(seed_draws.tobytes(order="C"))
    for dataset in datasets:
        groups = dataset.group_tokens
        _, members = _factorize_groups(groups)
        cluster_count = len(members)
        draws = rng.integers(
            0,
            cluster_count,
            size=(BOOTSTRAP_REPLICATES, cluster_count),
            dtype=np.int32,
        )
        digest.update(dataset.dataset_id.encode("ascii"))
        digest.update(draws.tobytes(order="C"))
        cluster_weights.append(_resample_weights(draws, cluster_count))
    return _BootstrapPlan(
        seed_weights=seed_weights,
        cluster_weights=tuple(cluster_weights),
        plan_sha256=digest.hexdigest(),
    )


def _bootstrap_cell_samples(
    dataset: _DatasetInputs,
    model: _ModelInputs,
    strategy: _StrategyInputs,
    cluster_weights: np.ndarray,
    seed_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    eligible = dataset.dialogue_history_eligible
    labels = np.asarray(dataset.labels[eligible], dtype=np.int64)
    all_codes, all_members = _factorize_groups(dataset.group_tokens)
    codes = np.asarray(all_codes[eligible], dtype=np.int64)
    members = all_members
    cluster_count = len(members)
    classes = len(dataset.class_tokens)
    sizes = np.bincount(codes, minlength=cluster_count).astype(np.float64)
    sampled_queries = cluster_weights @ sizes

    current = model.current_probability[:, eligible, :]
    history = strategy.probability[:, eligible, :]
    current_predictions = np.argmax(current, axis=2)
    history_predictions = np.argmax(history, axis=2)
    current_confusion = _cluster_confusions(
        labels,
        current_predictions,
        codes,
        clusters=cluster_count,
        classes=classes,
    )
    history_confusion = _cluster_confusions(
        labels,
        history_predictions,
        codes,
        clusters=cluster_count,
        classes=classes,
    )
    regrets = np.empty((len(EXPECTED_TRAINING_SEEDS), len(labels)), dtype=np.float64)
    for seed_index in range(len(EXPECTED_TRAINING_SEEDS)):
        regrets[seed_index] = paired_true_class_regret(
            labels, current[seed_index], history[seed_index]
        )
    regret_cluster_sum = np.empty(
        (len(EXPECTED_TRAINING_SEEDS), cluster_count), dtype=np.float64
    )
    for seed_index in range(len(EXPECTED_TRAINING_SEEDS)):
        regret_cluster_sum[seed_index] = np.bincount(
            codes, weights=regrets[seed_index], minlength=cluster_count
        )

    macro_samples = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    regret_samples = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    batch_size = 512
    for start in range(0, BOOTSTRAP_REPLICATES, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_REPLICATES)
        weights = np.asarray(cluster_weights[start:stop], dtype=np.float64)
        seed_weight = np.asarray(seed_weights[start:stop], dtype=np.float64)
        current_cm = np.einsum(
            "bg,sgij->bsij", weights, current_confusion, optimize=True
        )
        history_cm = np.einsum(
            "bg,sgij->bsij", weights, history_confusion, optimize=True
        )
        macro_by_seed = _macro_f1_from_confusion(history_cm) - _macro_f1_from_confusion(
            current_cm
        )
        macro_samples[start:stop] = np.sum(
            macro_by_seed * seed_weight, axis=1
        ) / len(EXPECTED_TRAINING_SEEDS)
        regret_sum = weights @ regret_cluster_sum.T
        denominator = sampled_queries[start:stop, None]
        regret_by_seed = np.divide(
            regret_sum,
            denominator,
            out=np.full_like(regret_sum, np.nan, dtype=np.float64),
            where=denominator > 0.0,
        )
        regret_samples[start:stop] = np.sum(
            regret_by_seed * seed_weight, axis=1
        ) / len(EXPECTED_TRAINING_SEEDS)
        empty = sampled_queries[start:stop] <= 0.0
        macro_samples[start:stop][empty] = np.nan
    return macro_samples, regret_samples


def _bootstrap_summary(values: np.ndarray, *, point: float) -> dict[str, object]:
    finite = values[np.isfinite(values)]
    fraction = float(len(finite) / len(values))
    if fraction < MINIMUM_FINITE_BOOTSTRAP_FRACTION:
        raise HarmBenchSelectionStatisticsError(
            "finite bootstrap replicate fraction fell below 0.95"
        )
    return {
        "point": float(point),
        "ci95_low": float(np.quantile(finite, 0.025)),
        "ci95_high": float(np.quantile(finite, 0.975)),
        "finite_replicates": int(len(finite)),
        "finite_fraction": fraction,
    }


def _shared_randomization_p_values(
    datasets: Sequence[_DatasetInputs],
    point_cells: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> tuple[dict[tuple[str, str, str], float], dict[str, object]]:
    """Shared whole-cluster paired swaps for all frozen strategy cells.

    Only the six same-speaker entries enter Holm; scope-control and secondary
    p-values remain descriptive and cannot expand or alter that family.
    """

    dataset_prepared: list[dict[str, object]] = []
    cluster_counts: list[int] = []
    for dataset in datasets:
        all_codes, all_members = _factorize_groups(dataset.group_tokens)
        eligible = dataset.dialogue_history_eligible
        codes = np.asarray(all_codes[eligible], dtype=np.int64)
        labels = np.asarray(dataset.labels[eligible], dtype=np.int64)
        cluster_count = len(all_members)
        cluster_counts.append(cluster_count)
        model_values: dict[str, object] = {}
        for model_id in EXPECTED_MODEL_ORDER:
            model = _model_by_id(dataset, model_id)
            current = model.current_probability[:, eligible, :]
            classes = len(dataset.class_tokens)
            current_confusion = _cluster_confusions(
                labels,
                np.argmax(current, axis=2),
                codes,
                clusters=cluster_count,
                classes=classes,
            )
            strategy_values: dict[str, object] = {}
            for strategy_id in EXPECTED_HISTORY_STRATEGY_ORDER:
                strategy = _strategy_by_id(model, strategy_id)
                history = strategy.probability[:, eligible, :]
                regrets = np.empty(
                    (len(EXPECTED_TRAINING_SEEDS), len(labels)), dtype=np.float64
                )
                for seed_index in range(len(EXPECTED_TRAINING_SEEDS)):
                    regrets[seed_index] = paired_true_class_regret(
                        labels, current[seed_index], history[seed_index]
                    )
                regret_by_cluster = np.bincount(
                    codes,
                    weights=regrets.sum(axis=0, dtype=np.float64),
                    minlength=cluster_count,
                ).astype(np.float64)
                history_confusion = _cluster_confusions(
                    labels,
                    np.argmax(history, axis=2),
                    codes,
                    clusters=cluster_count,
                    classes=classes,
                )
                strategy_values[strategy_id] = {
                    "regret_by_cluster": regret_by_cluster,
                    "history_confusion": history_confusion,
                }
            model_values[model_id] = {
                "queries": len(labels),
                "current_confusion": current_confusion,
                "strategies": strategy_values,
            }
        dataset_prepared.append(
            {
                "dataset_id": dataset.dataset_id,
                "cluster_count": cluster_count,
                "models": model_values,
            }
        )

    combined_clusters = int(sum(cluster_counts))
    if combined_clusters <= RANDOMIZATION_EXACT_MAX_CLUSTERS:
        method = "exact"
        assignments_total = 1 << combined_clusters
        rng: np.random.Generator | None = None
        reported_seed: int | None = None
    else:
        method = "monte_carlo"
        assignments_total = RANDOMIZATION_MC_REPLICATES
        rng = np.random.default_rng(RANDOMIZATION_SEED)
        reported_seed = RANDOMIZATION_SEED

    observed: dict[tuple[str, str, str], float] = {}
    for model_id in EXPECTED_MODEL_ORDER:
        for strategy_id in EXPECTED_HISTORY_STRATEGY_ORDER:
            macro = 0.0
            regret = 0.0
            for dataset_index, dataset in enumerate(datasets):
                prepared = dataset_prepared[dataset_index]["models"][model_id]
                assert isinstance(prepared, Mapping)
                prepared_strategy = prepared["strategies"][strategy_id]
                assert isinstance(prepared_strategy, Mapping)
                current_cm = np.asarray(prepared["current_confusion"]).sum(axis=1)
                history_cm = np.asarray(prepared_strategy["history_confusion"]).sum(axis=1)
                macro += 0.5 * float(
                    np.mean(
                        _macro_f1_from_confusion(history_cm)
                        - _macro_f1_from_confusion(current_cm)
                    )
                )
                regret += 0.5 * float(
                    np.asarray(prepared_strategy["regret_by_cluster"]).sum()
                    / (len(EXPECTED_TRAINING_SEEDS) * int(prepared["queries"]))
                )
            observed[(model_id, strategy_id, "Macro-F1")] = abs(macro)
            observed[(model_id, strategy_id, "mean-regret")] = abs(regret)
            expected_macro = 0.5 * sum(
                float(
                    point_cells[(dataset.dataset_id, model_id, strategy_id)][
                        "macro_f1_difference"
                    ]
                )
                for dataset in datasets
            )
            expected_regret = 0.5 * sum(
                float(
                    point_cells[(dataset.dataset_id, model_id, strategy_id)][
                        "mean_regret"
                    ]
                )
                for dataset in datasets
            )
            if not math.isclose(
                abs(expected_macro),
                observed[(model_id, strategy_id, "Macro-F1")],
                abs_tol=1e-12,
                rel_tol=0.0,
            ) or not math.isclose(
                abs(expected_regret),
                observed[(model_id, strategy_id, "mean-regret")],
                abs_tol=1e-12,
                rel_tol=0.0,
            ):
                raise HarmBenchSelectionStatisticsError(
                    "cluster randomization point statistic differs from the frozen point estimate"
                )

    exceed = {key: 0 for key in observed}
    plan_hash = hashlib.sha256()
    plan_hash.update(b"harmbench_erc_shared_cluster_randomization_v2")
    plan_hash.update(method.encode("ascii"))
    plan_hash.update(np.asarray(cluster_counts, dtype=np.int64).tobytes())
    batch_size = 2048
    offsets = np.cumsum([0, *cluster_counts], dtype=np.int64)
    for start in range(0, assignments_total, batch_size):
        stop = min(start + batch_size, assignments_total)
        count = stop - start
        if method == "exact":
            integers = np.arange(start, stop, dtype=np.uint64)
            shifts = np.arange(combined_clusters, dtype=np.uint64)
            bits = ((integers[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
        else:
            assert rng is not None
            bits = rng.integers(
                0, 2, size=(count, combined_clusters), dtype=np.uint8
            )
        plan_hash.update(bits.tobytes(order="C"))
        for model_id in EXPECTED_MODEL_ORDER:
            for strategy_id in EXPECTED_HISTORY_STRATEGY_ORDER:
                joint_regret = np.zeros(count, dtype=np.float64)
                joint_macro = np.zeros(count, dtype=np.float64)
                for dataset_index, _dataset in enumerate(datasets):
                    prepared = dataset_prepared[dataset_index]["models"][model_id]
                    assert isinstance(prepared, Mapping)
                    prepared_strategy = prepared["strategies"][strategy_id]
                    assert isinstance(prepared_strategy, Mapping)
                    local_bits = np.asarray(
                        bits[:, offsets[dataset_index] : offsets[dataset_index + 1]],
                        dtype=np.float64,
                    )
                    regret_cluster = np.asarray(
                        prepared_strategy["regret_by_cluster"], dtype=np.float64
                    )
                    original_regret = float(regret_cluster.sum())
                    permuted_regret = original_regret - 2.0 * (
                        local_bits @ regret_cluster
                    )
                    joint_regret += 0.5 * permuted_regret / (
                        len(EXPECTED_TRAINING_SEEDS) * int(prepared["queries"])
                    )

                    current_cluster = np.asarray(
                        prepared["current_confusion"], dtype=np.float64
                    )
                    history_cluster = np.asarray(
                        prepared_strategy["history_confusion"], dtype=np.float64
                    )
                    delta = (current_cluster - history_cluster).transpose(
                        1, 0, 2, 3
                    )
                    delta_flat = delta.reshape(delta.shape[0], -1)
                    adjustment = (local_bits @ delta_flat).reshape(
                        count, *history_cluster.sum(axis=1).shape
                    )
                    history_total = history_cluster.sum(axis=1)[None, ...]
                    current_total = current_cluster.sum(axis=1)[None, ...]
                    candidate = history_total + adjustment
                    anchor = current_total - adjustment
                    dataset_macro = np.mean(
                        _macro_f1_from_confusion(candidate)
                        - _macro_f1_from_confusion(anchor),
                        axis=1,
                    )
                    joint_macro += 0.5 * dataset_macro
                exceed[(model_id, strategy_id, "Macro-F1")] += int(
                    np.sum(
                        np.abs(joint_macro)
                        >= observed[(model_id, strategy_id, "Macro-F1")]
                    )
                )
                exceed[(model_id, strategy_id, "mean-regret")] += int(
                    np.sum(
                        np.abs(joint_regret)
                        >= observed[(model_id, strategy_id, "mean-regret")]
                    )
                )

    p_values: dict[tuple[str, str, str], float] = {}
    for key in observed:
        if method == "exact":
            p_values[key] = float(exceed[key] / assignments_total)
        else:
            p_values[key] = float(
                (1 + exceed[key]) / (1 + assignments_total)
            )
    contract = {
        "method": method,
        "combined_typed_clusters": combined_clusters,
        "assignments": assignments_total,
        "random_seed": reported_seed,
        "test_side": "two_sided_absolute_equal_dataset_weight",
        "shared_assignment_plan_sha256": plan_hash.hexdigest(),
    }
    return p_values, contract


def _holm_step_down(
    raw_p_values: Sequence[float], *, alpha: float = HOLM_ALPHA
) -> tuple[list[float], list[bool]]:
    values = np.asarray(raw_p_values, dtype=np.float64)
    if (
        values.shape != (len(EXPECTED_MODEL_ORDER) * len(PRIMARY_METRIC_ORDER),)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or float(alpha) != HOLM_ALPHA
    ):
        raise HarmBenchSelectionStatisticsError("Holm family is not the exact six tests")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    rejections_sorted = np.zeros(len(values), dtype=np.bool_)
    still_rejecting = True
    for rank, original_index in enumerate(order):
        multiplier = len(values) - rank
        running = max(running, multiplier * float(values[original_index]))
        adjusted_sorted[rank] = min(1.0, running)
        if still_rejecting and values[original_index] <= alpha / multiplier:
            rejections_sorted[rank] = True
        else:
            still_rejecting = False
    adjusted = np.empty(len(values), dtype=np.float64)
    rejected = np.empty(len(values), dtype=np.bool_)
    for rank, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[rank]
        rejected[original_index] = rejections_sorted[rank]
    return adjusted.tolist(), rejected.tolist()


def _compute_statistics_core(
    sealed_inputs: JointSelectionEvaluationInputs,
) -> dict[str, object]:
    inputs = _revalidate_joint_inputs(sealed_inputs)
    datasets = inputs._datasets
    point_cells: dict[tuple[str, str, str], dict[str, object]] = {}
    for dataset in datasets:
        for model_id in EXPECTED_MODEL_ORDER:
            model = _model_by_id(dataset, model_id)
            for strategy_id in EXPECTED_HISTORY_STRATEGY_ORDER:
                strategy = _strategy_by_id(model, strategy_id)
                point_cells[(dataset.dataset_id, model_id, strategy_id)] = _point_cell(
                    dataset, model, strategy
                )

    bootstrap_plan = _make_bootstrap_plan(datasets)
    bootstrap_samples: dict[tuple[str, str, str, str], np.ndarray] = {}
    dataset_records: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(datasets):
        _codes, members = _factorize_groups(dataset.group_tokens)
        model_records: list[dict[str, object]] = []
        for model_id in EXPECTED_MODEL_ORDER:
            model = _model_by_id(dataset, model_id)
            strategy_records: list[dict[str, object]] = []
            for strategy_id in EXPECTED_HISTORY_STRATEGY_ORDER:
                strategy = _strategy_by_id(model, strategy_id)
                point = point_cells[(dataset.dataset_id, model_id, strategy_id)]
                macro_samples, regret_samples = _bootstrap_cell_samples(
                    dataset,
                    model,
                    strategy,
                    bootstrap_plan.cluster_weights[dataset_index],
                    bootstrap_plan.seed_weights,
                )
                bootstrap_samples[
                    (dataset.dataset_id, model_id, strategy_id, "Macro-F1")
                ] = macro_samples
                bootstrap_samples[
                    (dataset.dataset_id, model_id, strategy_id, "mean-regret")
                ] = regret_samples
                strategy_records.append(
                    {
                        "strategy_id": strategy_id,
                        "macro_f1_difference": _bootstrap_summary(
                            macro_samples,
                            point=float(point["macro_f1_difference"]),
                        ),
                        "mean_regret": _bootstrap_summary(
                            regret_samples, point=float(point["mean_regret"])
                        ),
                        "secondary": point["secondary"],
                    }
                )
            model_records.append(
                {"model_id": model_id, "strategies": strategy_records}
            )
        dataset_records.append(
            {
                "dataset_id": dataset.dataset_id,
                "queries": int(len(dataset.labels)),
                "classes": int(len(dataset.class_tokens)),
                "independent_clusters": int(len(members)),
                "eligible_queries": int(dataset.dialogue_history_eligible.sum()),
                "eligible_independent_clusters": int(
                    len(set(dataset.group_tokens[dataset.dialogue_history_eligible].tolist()))
                ),
                "models": model_records,
            }
        )

    randomization_p, randomization_contract = _shared_randomization_p_values(
        datasets, point_cells
    )
    joint_records: list[dict[str, object]] = []
    for model_id in EXPECTED_MODEL_ORDER:
        for strategy_id in EXPECTED_HISTORY_STRATEGY_ORDER:
            metric_summaries: dict[str, object] = {}
            for metric_id, point_key in (
                ("Macro-F1", "macro_f1_difference"),
                ("mean-regret", "mean_regret"),
            ):
                joint_samples = 0.5 * (
                    bootstrap_samples[
                        (
                            EXPECTED_SELECTION_DATASETS[0],
                            model_id,
                            strategy_id,
                            metric_id,
                        )
                    ]
                    + bootstrap_samples[
                        (
                            EXPECTED_SELECTION_DATASETS[1],
                            model_id,
                            strategy_id,
                            metric_id,
                        )
                    ]
                )
                point = 0.5 * sum(
                    float(
                        point_cells[(dataset_id, model_id, strategy_id)][point_key]
                    )
                    for dataset_id in EXPECTED_SELECTION_DATASETS
                )
                metric_summaries[metric_id] = _bootstrap_summary(
                    joint_samples, point=point
                )
            is_primary = strategy_id == PRIMARY_HISTORY_STRATEGY_ID
            joint_records.append(
                {
                    "model_id": model_id,
                    "strategy_id": strategy_id,
                    "macro_f1_difference": metric_summaries["Macro-F1"],
                    "mean_regret": metric_summaries["mean-regret"],
                    "randomization": {
                        "status": (
                            "evaluated_primary_family"
                            if is_primary
                            else "evaluated_secondary_not_in_primary_family"
                        ),
                        "macro_f1_raw_p_value": randomization_p[
                            (model_id, strategy_id, "Macro-F1")
                        ],
                        "mean_regret_raw_p_value": randomization_p[
                            (model_id, strategy_id, "mean-regret")
                        ],
                    },
                }
            )

    joint_by_key = {
        (str(row["model_id"]), str(row["strategy_id"])): row
        for row in joint_records
    }
    raw_family = [
        randomization_p[(model_id, PRIMARY_HISTORY_STRATEGY_ID, metric_id)]
        for model_id in EXPECTED_MODEL_ORDER
        for metric_id in PRIMARY_METRIC_ORDER
    ]
    adjusted, rejected = _holm_step_down(raw_family)
    family_records: list[dict[str, object]] = []
    family_index = 0
    for model_id in EXPECTED_MODEL_ORDER:
        joint = joint_by_key[(model_id, PRIMARY_HISTORY_STRATEGY_ID)]
        for metric_id in PRIMARY_METRIC_ORDER:
            if metric_id == "Macro-F1":
                point = float(joint["macro_f1_difference"]["point"])
                practical_minimum = MACRO_F1_PRACTICAL_MINIMUM
            else:
                point = float(joint["mean_regret"]["point"])
                practical_minimum = MEAN_REGRET_PRACTICAL_MINIMUM
            family_records.append(
                {
                    "hypothesis_id": f"H_{model_id}_{metric_id}",
                    "model_id": model_id,
                    "metric_id": metric_id,
                    "raw_p_value": float(raw_family[family_index]),
                    "holm_adjusted_p_value": float(adjusted[family_index]),
                    "holm_rejected": bool(rejected[family_index]),
                    "point": point,
                    "absolute_practical_minimum": practical_minimum,
                    "substantive_gate_pass": bool(
                        rejected[family_index]
                        and abs(point) >= practical_minimum
                    ),
                }
            )
            family_index += 1

    no_harm_cells: list[dict[str, object]] = []
    for dataset_record in dataset_records:
        dataset_id = str(dataset_record["dataset_id"])
        for model_record in dataset_record["models"]:
            model_id = str(model_record["model_id"])
            strategy_record = next(
                row
                for row in model_record["strategies"]
                if row["strategy_id"] == PRIMARY_HISTORY_STRATEGY_ID
            )
            macro_low = float(strategy_record["macro_f1_difference"]["ci95_low"])
            regret_high = float(strategy_record["mean_regret"]["ci95_high"])
            finite_ok = bool(
                strategy_record["macro_f1_difference"]["finite_fraction"]
                >= MINIMUM_FINITE_BOOTSTRAP_FRACTION
                and strategy_record["mean_regret"]["finite_fraction"]
                >= MINIMUM_FINITE_BOOTSTRAP_FRACTION
            )
            passed = bool(
                finite_ok
                and macro_low >= -MACRO_F1_PRACTICAL_MINIMUM
                and regret_high <= MEAN_REGRET_PRACTICAL_MINIMUM
            )
            no_harm_cells.append(
                {
                    "dataset_id": dataset_id,
                    "model_id": model_id,
                    "macro_f1_ci95_low": macro_low,
                    "required_macro_f1_ci95_low": -MACRO_F1_PRACTICAL_MINIMUM,
                    "mean_regret_ci95_high": regret_high,
                    "required_mean_regret_ci95_high": MEAN_REGRET_PRACTICAL_MINIMUM,
                    "finite_fraction_gate_pass": finite_ok,
                    "cell_pass": passed,
                }
            )

    report: dict[str, object] = {
        "schema_version": SELECTION_STATISTICS_SCHEMA,
        "protocol_canonical_sha256": PROTOCOL_V2_CANONICAL_SHA256,
        "selection_result_status": EXPLORATORY_STATUS,
        "confirmatory_claim": False,
        "calibration_role": False,
        "holdout_role": False,
        "validation_role": False,
        "official_test_role": False,
        "analysis_contract": {
            "dataset_order": list(EXPECTED_SELECTION_DATASETS),
            "model_order": list(EXPECTED_MODEL_ORDER),
            "strategy_order": list(EXPECTED_CONTEXT_ROSTER_ORDER),
            "primary_history_strategy_id": PRIMARY_HISTORY_STRATEGY_ID,
            "primary_metric_order": list(PRIMARY_METRIC_ORDER),
            "training_seed_count": len(EXPECTED_TRAINING_SEEDS),
            "row_metric_order": "per_training_seed_and_query_before_seed_aggregation",
            "joint_weighting": "equal_weight_two_dataset_mean",
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "random_seed": BOOTSTRAP_SEED,
                "confidence_interval": "percentile_2.5_97.5",
                "minimum_finite_fraction": MINIMUM_FINITE_BOOTSTRAP_FRACTION,
                "whole_original_cluster_resampling": True,
                "shared_plan_across_all_cells_metrics_strategies": True,
                "shared_plan_sha256": bootstrap_plan.plan_sha256,
            },
            "randomization": randomization_contract,
            "holm": {
                "family_size": len(family_records),
                "familywise_alpha": HOLM_ALPHA,
                "stable_declared_order": True,
                "winner_selection_permitted": False,
            },
            "ece_secondary_metric_bins": ECE_BINS,
            "cvar_tail_alpha": TAIL_ALPHA,
        },
        "datasets": dataset_records,
        "joint_cells": joint_records,
        "primary_holm_family": family_records,
        "no_harm_gate": {
            "cells": no_harm_cells,
            "all_six_cells_pass": bool(all(row["cell_pass"] for row in no_harm_cells)),
            "confirmatory_authority_granted": False,
        },
    }
    validate_selection_statistics_report(report)
    return report


def _exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise HarmBenchSelectionStatisticsError(f"{name} must be a boolean")
    return value


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise HarmBenchSelectionStatisticsError(f"{name} must be an exact integer")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise HarmBenchSelectionStatisticsError(f"{name} must be a finite float")
    return value


def _validate_interval_summary(value: object, *, name: str) -> None:
    row = _exact_keys(
        value,
        {"point", "ci95_low", "ci95_high", "finite_replicates", "finite_fraction"},
        name=name,
    )
    point = _finite_float(row["point"], name=f"{name}.point")
    low = _finite_float(row["ci95_low"], name=f"{name}.ci95_low")
    high = _finite_float(row["ci95_high"], name=f"{name}.ci95_high")
    if low > high:
        raise HarmBenchSelectionStatisticsError(f"{name} CI is reversed")
    _ = point
    finite = _exact_int(
        row["finite_replicates"], name=f"{name}.finite_replicates", minimum=1
    )
    fraction = _finite_float(row["finite_fraction"], name=f"{name}.finite_fraction")
    if finite > BOOTSTRAP_REPLICATES or not (
        MINIMUM_FINITE_BOOTSTRAP_FRACTION <= fraction <= 1.0
    ) or not math.isclose(
        fraction, finite / BOOTSTRAP_REPLICATES, rel_tol=0.0, abs_tol=1e-15
    ):
        raise HarmBenchSelectionStatisticsError(f"{name} finite fraction changed")


def _validate_secondary(value: object, *, name: str) -> None:
    row = _exact_keys(
        value,
        {
            "ece_difference",
            "cvar90_regret",
            "harm_rate_gt_0",
            "harm_rate_gt_0_05",
            "coverage",
            "sign_severity",
            "depth_strata",
            "worst_depth_stratum",
        },
        name=name,
    )
    _finite_float(row["ece_difference"], name=f"{name}.ece_difference")
    _finite_float(row["cvar90_regret"], name=f"{name}.cvar90_regret")
    for key in ("harm_rate_gt_0", "harm_rate_gt_0_05"):
        rate = _finite_float(row[key], name=f"{name}.{key}")
        if not 0.0 <= rate <= 1.0:
            raise HarmBenchSelectionStatisticsError(f"{name}.{key} is not a rate")
    coverage = _exact_keys(
        row["coverage"],
        {"eligible_queries", "history_context_nonempty_queries", "rate"},
        name=f"{name}.coverage",
    )
    eligible = _exact_int(
        coverage["eligible_queries"], name=f"{name}.coverage.eligible_queries", minimum=1
    )
    selected = _exact_int(
        coverage["history_context_nonempty_queries"],
        name=f"{name}.coverage.history_context_nonempty_queries",
    )
    rate = _finite_float(coverage["rate"], name=f"{name}.coverage.rate")
    if selected > eligible or not math.isclose(
        rate, selected / eligible, rel_tol=0.0, abs_tol=1e-15
    ):
        raise HarmBenchSelectionStatisticsError(f"{name}.coverage is inconsistent")
    sign = _exact_keys(
        row["sign_severity"],
        {"seed_query_observations", "counts", "rates"},
        name=f"{name}.sign_severity",
    )
    observations = _exact_int(
        sign["seed_query_observations"],
        name=f"{name}.sign_severity.seed_query_observations",
        minimum=1,
    )
    expected_bins = {
        "substantial_benefit",
        "small_benefit",
        "exact_zero_including_fallback",
        "small_harm",
        "substantial_harm",
    }
    counts = _exact_keys(sign["counts"], expected_bins, name=f"{name}.sign_severity.counts")
    rates = _exact_keys(sign["rates"], expected_bins, name=f"{name}.sign_severity.rates")
    count_total = 0
    for bin_id in expected_bins:
        count = _exact_int(counts[bin_id], name=f"{name}.counts.{bin_id}")
        bin_rate = _finite_float(rates[bin_id], name=f"{name}.rates.{bin_id}")
        if not math.isclose(
            bin_rate, count / observations, rel_tol=0.0, abs_tol=1e-15
        ):
            raise HarmBenchSelectionStatisticsError(f"{name} sign rate changed")
        count_total += count
    if count_total != observations:
        raise HarmBenchSelectionStatisticsError(f"{name} sign bins are not exhaustive")
    strata = row["depth_strata"]
    if not isinstance(strata, list) or len(strata) != len(DEPTH_STRATA):
        raise HarmBenchSelectionStatisticsError(f"{name} depth strata changed")
    estimable_values: list[tuple[str, float]] = []
    for observed, (stratum_id, _minimum, _maximum) in zip(strata, DEPTH_STRATA):
        stratum = _exact_keys(
            observed,
            {
                "stratum_id",
                "status",
                "queries",
                "independent_clusters",
                "seed_query_observations",
                "mean_regret",
            },
            name=f"{name}.{stratum_id}",
        )
        if stratum["stratum_id"] != stratum_id:
            raise HarmBenchSelectionStatisticsError(f"{name} depth order changed")
        queries = _exact_int(stratum["queries"], name=f"{name}.{stratum_id}.queries")
        clusters = _exact_int(
            stratum["independent_clusters"], name=f"{name}.{stratum_id}.clusters"
        )
        observations = _exact_int(
            stratum["seed_query_observations"],
            name=f"{name}.{stratum_id}.observations",
        )
        if observations != queries * len(EXPECTED_TRAINING_SEEDS):
            raise HarmBenchSelectionStatisticsError(f"{name} stratum count changed")
        if clusters >= 2:
            if stratum["status"] != "estimable":
                raise HarmBenchSelectionStatisticsError(f"{name} estimable stratum suppressed")
            mean_regret = _finite_float(
                stratum["mean_regret"], name=f"{name}.{stratum_id}.mean_regret"
            )
            estimable_values.append((stratum_id, mean_regret))
        elif stratum["status"] != "not_estimable" or stratum["mean_regret"] is not None:
            raise HarmBenchSelectionStatisticsError(
                f"{name} non-estimable stratum was imputed/merged"
            )
    worst = _exact_keys(
        row["worst_depth_stratum"],
        {"status", "estimable_strata", "stratum_id", "mean_regret"},
        name=f"{name}.worst_depth_stratum",
    )
    if _exact_int(worst["estimable_strata"], name=f"{name}.estimable_strata") != len(
        estimable_values
    ):
        raise HarmBenchSelectionStatisticsError(f"{name} estimable-strata count changed")
    if estimable_values:
        expected_id, expected_value = max(estimable_values, key=lambda item: item[1])
        if (
            worst["status"] != "estimable"
            or worst["stratum_id"] != expected_id
            or _finite_float(worst["mean_regret"], name=f"{name}.worst_mean")
            != expected_value
        ):
            raise HarmBenchSelectionStatisticsError(f"{name} worst stratum changed")
    elif (
        worst["status"] != "not_estimable"
        or worst["stratum_id"] is not None
        or worst["mean_regret"] is not None
    ):
        raise HarmBenchSelectionStatisticsError(f"{name} invented a worst stratum")


def _privacy_and_finite_scan(value: object, *, path: str = "$") -> None:
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise HarmBenchSelectionStatisticsError(f"non-finite output at {path}")
        return
    if isinstance(value, str):
        if (
            _WINDOWS_PATH.match(value)
            or value.startswith(("/", "file://"))
            or "\\" in value
        ):
            raise HarmBenchSelectionStatisticsError(f"private path-like output at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise HarmBenchSelectionStatisticsError(f"non-string output key at {path}")
            lowered = key.lower()
            if any(part in lowered for part in _FORBIDDEN_OUTPUT_KEY_PARTS):
                raise HarmBenchSelectionStatisticsError(
                    f"privacy-forbidden output key at {path}.{key}"
                )
            _privacy_and_finite_scan(child, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        if len(value) > 128:
            raise HarmBenchSelectionStatisticsError(f"oversized output sequence at {path}")
        for index, child in enumerate(value):
            _privacy_and_finite_scan(child, path=f"{path}[{index}]")
        return
    raise HarmBenchSelectionStatisticsError(
        f"unsupported aggregate output type at {path}: {type(value).__name__}"
    )


def validate_selection_statistics_report(report: object) -> None:
    """Fail closed on schema drift, non-finite values, or row-level leakage."""

    root = _exact_keys(
        report,
        {
            "schema_version",
            "protocol_canonical_sha256",
            "selection_result_status",
            "confirmatory_claim",
            "calibration_role",
            "holdout_role",
            "validation_role",
            "official_test_role",
            "analysis_contract",
            "datasets",
            "joint_cells",
            "primary_holm_family",
            "no_harm_gate",
        },
        name="selection statistics report",
    )
    if (
        root["schema_version"] != SELECTION_STATISTICS_SCHEMA
        or root["protocol_canonical_sha256"] != PROTOCOL_V2_CANONICAL_SHA256
        or root["selection_result_status"] != EXPLORATORY_STATUS
        or any(
            _exact_bool(root[key], name=key)
            for key in (
                "confirmatory_claim",
                "calibration_role",
                "holdout_role",
                "validation_role",
                "official_test_role",
            )
        )
    ):
        raise HarmBenchSelectionStatisticsError("exploratory status boundary changed")
    contract = _exact_keys(
        root["analysis_contract"],
        {
            "dataset_order",
            "model_order",
            "strategy_order",
            "primary_history_strategy_id",
            "primary_metric_order",
            "training_seed_count",
            "row_metric_order",
            "joint_weighting",
            "bootstrap",
            "randomization",
            "holm",
            "ece_secondary_metric_bins",
            "cvar_tail_alpha",
        },
        name="analysis_contract",
    )
    if (
        contract["dataset_order"] != list(EXPECTED_SELECTION_DATASETS)
        or contract["model_order"] != list(EXPECTED_MODEL_ORDER)
        or contract["strategy_order"] != list(EXPECTED_CONTEXT_ROSTER_ORDER)
        or contract["primary_history_strategy_id"] != PRIMARY_HISTORY_STRATEGY_ID
        or contract["primary_metric_order"] != list(PRIMARY_METRIC_ORDER)
        or contract["training_seed_count"] != len(EXPECTED_TRAINING_SEEDS)
        or contract["row_metric_order"]
        != "per_training_seed_and_query_before_seed_aggregation"
        or contract["joint_weighting"] != "equal_weight_two_dataset_mean"
        or contract["ece_secondary_metric_bins"] != ECE_BINS
        or contract["cvar_tail_alpha"] != TAIL_ALPHA
    ):
        raise HarmBenchSelectionStatisticsError("analysis contract changed")
    bootstrap = _exact_keys(
        contract["bootstrap"],
        {
            "replicates",
            "random_seed",
            "confidence_interval",
            "minimum_finite_fraction",
            "whole_original_cluster_resampling",
            "shared_plan_across_all_cells_metrics_strategies",
            "shared_plan_sha256",
        },
        name="bootstrap contract",
    )
    if (
        bootstrap["replicates"] != BOOTSTRAP_REPLICATES
        or bootstrap["random_seed"] != BOOTSTRAP_SEED
        or bootstrap["confidence_interval"] != "percentile_2.5_97.5"
        or bootstrap["minimum_finite_fraction"] != MINIMUM_FINITE_BOOTSTRAP_FRACTION
        or bootstrap["whole_original_cluster_resampling"] is not True
        or bootstrap["shared_plan_across_all_cells_metrics_strategies"] is not True
        or not isinstance(bootstrap["shared_plan_sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", bootstrap["shared_plan_sha256"])
    ):
        raise HarmBenchSelectionStatisticsError("bootstrap contract changed")
    randomization = _exact_keys(
        contract["randomization"],
        {
            "method",
            "combined_typed_clusters",
            "assignments",
            "random_seed",
            "test_side",
            "shared_assignment_plan_sha256",
        },
        name="randomization contract",
    )
    clusters = _exact_int(
        randomization["combined_typed_clusters"], name="combined_typed_clusters", minimum=1
    )
    if clusters <= RANDOMIZATION_EXACT_MAX_CLUSTERS:
        if (
            randomization["method"] != "exact"
            or randomization["assignments"] != 1 << clusters
            or randomization["random_seed"] is not None
        ):
            raise HarmBenchSelectionStatisticsError("exact randomization changed")
    elif (
        randomization["method"] != "monte_carlo"
        or randomization["assignments"] != RANDOMIZATION_MC_REPLICATES
        or randomization["random_seed"] != RANDOMIZATION_SEED
    ):
        raise HarmBenchSelectionStatisticsError("Monte Carlo randomization changed")
    if (
        randomization["test_side"] != "two_sided_absolute_equal_dataset_weight"
        or not isinstance(randomization["shared_assignment_plan_sha256"], str)
        or not re.fullmatch(
            r"[0-9a-f]{64}", randomization["shared_assignment_plan_sha256"]
        )
    ):
        raise HarmBenchSelectionStatisticsError("randomization semantics changed")
    holm = _exact_keys(
        contract["holm"],
        {"family_size", "familywise_alpha", "stable_declared_order", "winner_selection_permitted"},
        name="Holm contract",
    )
    if dict(holm) != {
        "family_size": 6,
        "familywise_alpha": HOLM_ALPHA,
        "stable_declared_order": True,
        "winner_selection_permitted": False,
    }:
        raise HarmBenchSelectionStatisticsError("Holm contract changed")

    datasets = root["datasets"]
    if not isinstance(datasets, list) or len(datasets) != 2:
        raise HarmBenchSelectionStatisticsError("dataset report roster changed")
    dataset_primary_summaries: dict[tuple[str, str], tuple[Mapping[str, object], Mapping[str, object]]] = {}
    for dataset_record, dataset_id in zip(datasets, EXPECTED_SELECTION_DATASETS):
        dataset_row = _exact_keys(
            dataset_record,
            {
                "dataset_id",
                "queries",
                "classes",
                "independent_clusters",
                "eligible_queries",
                "eligible_independent_clusters",
                "models",
            },
            name=f"dataset report {dataset_id}",
        )
        if dataset_row["dataset_id"] != dataset_id:
            raise HarmBenchSelectionStatisticsError("dataset report order changed")
        for key in (
            "queries",
            "classes",
            "independent_clusters",
            "eligible_queries",
            "eligible_independent_clusters",
        ):
            _exact_int(dataset_row[key], name=f"{dataset_id}.{key}", minimum=1)
        models = dataset_row["models"]
        if not isinstance(models, list) or len(models) != len(EXPECTED_MODEL_ORDER):
            raise HarmBenchSelectionStatisticsError("dataset model roster changed")
        for model_record, model_id in zip(models, EXPECTED_MODEL_ORDER):
            model_row = _exact_keys(
                model_record, {"model_id", "strategies"}, name=f"model {model_id}"
            )
            if model_row["model_id"] != model_id:
                raise HarmBenchSelectionStatisticsError("model report order changed")
            strategies = model_row["strategies"]
            if not isinstance(strategies, list) or len(strategies) != len(
                EXPECTED_HISTORY_STRATEGY_ORDER
            ):
                raise HarmBenchSelectionStatisticsError("history strategy roster changed")
            for strategy_record, strategy_id in zip(
                strategies, EXPECTED_HISTORY_STRATEGY_ORDER
            ):
                strategy_row = _exact_keys(
                    strategy_record,
                    {"strategy_id", "macro_f1_difference", "mean_regret", "secondary"},
                    name=f"strategy {strategy_id}",
                )
                if strategy_row["strategy_id"] != strategy_id:
                    raise HarmBenchSelectionStatisticsError("strategy report order changed")
                _validate_interval_summary(
                    strategy_row["macro_f1_difference"],
                    name=f"{dataset_id}.{model_id}.{strategy_id}.macro",
                )
                _validate_interval_summary(
                    strategy_row["mean_regret"],
                    name=f"{dataset_id}.{model_id}.{strategy_id}.regret",
                )
                _validate_secondary(
                    strategy_row["secondary"],
                    name=f"{dataset_id}.{model_id}.{strategy_id}.secondary",
                )
                if strategy_id == PRIMARY_HISTORY_STRATEGY_ID:
                    dataset_primary_summaries[(dataset_id, model_id)] = (
                        strategy_row["macro_f1_difference"],
                        strategy_row["mean_regret"],
                    )

    joint = root["joint_cells"]
    if not isinstance(joint, list) or len(joint) != 15:
        raise HarmBenchSelectionStatisticsError("joint cell roster changed")
    expected_joint = [
        (model_id, strategy_id)
        for model_id in EXPECTED_MODEL_ORDER
        for strategy_id in EXPECTED_HISTORY_STRATEGY_ORDER
    ]
    joint_primary_rows: dict[str, Mapping[str, object]] = {}
    for cell, (model_id, strategy_id) in zip(joint, expected_joint):
        row = _exact_keys(
            cell,
            {"model_id", "strategy_id", "macro_f1_difference", "mean_regret", "randomization"},
            name="joint cell",
        )
        if row["model_id"] != model_id or row["strategy_id"] != strategy_id:
            raise HarmBenchSelectionStatisticsError("joint cell order changed")
        _validate_interval_summary(row["macro_f1_difference"], name="joint macro")
        _validate_interval_summary(row["mean_regret"], name="joint regret")
        expected_macro_point = 0.5 * sum(
            float(dataset_primary_summaries[(dataset_id, model_id)][0]["point"])
            if strategy_id == PRIMARY_HISTORY_STRATEGY_ID
            else float(
                next(
                    strategy
                    for dataset_row in datasets
                    if dataset_row["dataset_id"] == dataset_id
                    for model_row in dataset_row["models"]
                    if model_row["model_id"] == model_id
                    for strategy in model_row["strategies"]
                    if strategy["strategy_id"] == strategy_id
                )["macro_f1_difference"]["point"]
            )
            for dataset_id in EXPECTED_SELECTION_DATASETS
        )
        expected_regret_point = 0.5 * sum(
            float(dataset_primary_summaries[(dataset_id, model_id)][1]["point"])
            if strategy_id == PRIMARY_HISTORY_STRATEGY_ID
            else float(
                next(
                    strategy
                    for dataset_row in datasets
                    if dataset_row["dataset_id"] == dataset_id
                    for model_row in dataset_row["models"]
                    if model_row["model_id"] == model_id
                    for strategy in model_row["strategies"]
                    if strategy["strategy_id"] == strategy_id
                )["mean_regret"]["point"]
            )
            for dataset_id in EXPECTED_SELECTION_DATASETS
        )
        if not math.isclose(
            float(row["macro_f1_difference"]["point"]),
            expected_macro_point,
            rel_tol=0.0,
            abs_tol=1e-15,
        ) or not math.isclose(
            float(row["mean_regret"]["point"]),
            expected_regret_point,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise HarmBenchSelectionStatisticsError(
                "joint point is not the equal-weight two-dataset estimate"
            )
        random = _exact_keys(
            row["randomization"],
            {"status", "macro_f1_raw_p_value", "mean_regret_raw_p_value"},
            name="joint randomization",
        )
        expected_status = (
            "evaluated_primary_family"
            if strategy_id == PRIMARY_HISTORY_STRATEGY_ID
            else "evaluated_secondary_not_in_primary_family"
        )
        if random["status"] != expected_status:
            raise HarmBenchSelectionStatisticsError("randomization strategy role changed")
        for key in ("macro_f1_raw_p_value", "mean_regret_raw_p_value"):
            p_value = _finite_float(random[key], name=key)
            if not 0.0 <= p_value <= 1.0:
                raise HarmBenchSelectionStatisticsError("invalid randomization p-value")
        if strategy_id == PRIMARY_HISTORY_STRATEGY_ID:
            joint_primary_rows[model_id] = row

    family = root["primary_holm_family"]
    if not isinstance(family, list) or len(family) != 6:
        raise HarmBenchSelectionStatisticsError("Holm family size changed")
    family_index = 0
    family_rows: list[Mapping[str, object]] = []
    for model_id in EXPECTED_MODEL_ORDER:
        for metric_id in PRIMARY_METRIC_ORDER:
            row = _exact_keys(
                family[family_index],
                {
                    "hypothesis_id",
                    "model_id",
                    "metric_id",
                    "raw_p_value",
                    "holm_adjusted_p_value",
                    "holm_rejected",
                    "point",
                    "absolute_practical_minimum",
                    "substantive_gate_pass",
                },
                name="Holm hypothesis",
            )
            if (
                row["hypothesis_id"] != f"H_{model_id}_{metric_id}"
                or row["model_id"] != model_id
                or row["metric_id"] != metric_id
            ):
                raise HarmBenchSelectionStatisticsError("Holm hypothesis order changed")
            raw_p = _finite_float(row["raw_p_value"], name="raw p")
            adjusted_p = _finite_float(row["holm_adjusted_p_value"], name="adjusted p")
            point = _finite_float(row["point"], name="Holm point")
            minimum = _finite_float(
                row["absolute_practical_minimum"], name="practical minimum"
            )
            rejected = _exact_bool(row["holm_rejected"], name="Holm rejected")
            substantive = _exact_bool(
                row["substantive_gate_pass"], name="substantive gate"
            )
            expected_minimum = (
                MACRO_F1_PRACTICAL_MINIMUM
                if metric_id == "Macro-F1"
                else MEAN_REGRET_PRACTICAL_MINIMUM
            )
            if (
                not 0.0 <= raw_p <= 1.0
                or not raw_p <= adjusted_p <= 1.0
                or minimum != expected_minimum
                or substantive != (rejected and abs(point) >= minimum)
            ):
                raise HarmBenchSelectionStatisticsError("Holm/gate value changed")
            joint_primary = joint_primary_rows[model_id]
            if metric_id == "Macro-F1":
                expected_raw = joint_primary["randomization"]["macro_f1_raw_p_value"]
                expected_point = joint_primary["macro_f1_difference"]["point"]
            else:
                expected_raw = joint_primary["randomization"]["mean_regret_raw_p_value"]
                expected_point = joint_primary["mean_regret"]["point"]
            if raw_p != expected_raw or point != expected_point:
                raise HarmBenchSelectionStatisticsError(
                    "Holm hypothesis differs from its joint primary cell"
                )
            family_rows.append(row)
            family_index += 1
    expected_adjusted, expected_rejected = _holm_step_down(
        [float(row["raw_p_value"]) for row in family_rows]
    )
    for index, row in enumerate(family_rows):
        if (
            float(row["holm_adjusted_p_value"]) != expected_adjusted[index]
            or bool(row["holm_rejected"]) != expected_rejected[index]
        ):
            raise HarmBenchSelectionStatisticsError(
                "Holm step-down correction was not computed over the exact family"
            )

    no_harm = _exact_keys(
        root["no_harm_gate"],
        {"cells", "all_six_cells_pass", "confirmatory_authority_granted"},
        name="no-harm gate",
    )
    cells = no_harm["cells"]
    if not isinstance(cells, list) or len(cells) != 6:
        raise HarmBenchSelectionStatisticsError("no-harm cell roster changed")
    expected_cells = [
        (dataset_id, model_id)
        for dataset_id in EXPECTED_SELECTION_DATASETS
        for model_id in EXPECTED_MODEL_ORDER
    ]
    observed_passes: list[bool] = []
    for cell, (dataset_id, model_id) in zip(cells, expected_cells):
        row = _exact_keys(
            cell,
            {
                "dataset_id",
                "model_id",
                "macro_f1_ci95_low",
                "required_macro_f1_ci95_low",
                "mean_regret_ci95_high",
                "required_mean_regret_ci95_high",
                "finite_fraction_gate_pass",
                "cell_pass",
            },
            name="no-harm cell",
        )
        if row["dataset_id"] != dataset_id or row["model_id"] != model_id:
            raise HarmBenchSelectionStatisticsError("no-harm cell order changed")
        macro_low = _finite_float(row["macro_f1_ci95_low"], name="no-harm macro")
        regret_high = _finite_float(row["mean_regret_ci95_high"], name="no-harm regret")
        expected_macro_summary, expected_regret_summary = dataset_primary_summaries[
            (dataset_id, model_id)
        ]
        if (
            macro_low != expected_macro_summary["ci95_low"]
            or regret_high != expected_regret_summary["ci95_high"]
        ):
            raise HarmBenchSelectionStatisticsError(
                "no-harm cell is not bound to the same-origin dataset CI"
            )
        if (
            row["required_macro_f1_ci95_low"] != -MACRO_F1_PRACTICAL_MINIMUM
            or row["required_mean_regret_ci95_high"]
            != MEAN_REGRET_PRACTICAL_MINIMUM
        ):
            raise HarmBenchSelectionStatisticsError("no-harm thresholds changed")
        finite_ok = _exact_bool(
            row["finite_fraction_gate_pass"], name="finite gate"
        )
        cell_pass = _exact_bool(row["cell_pass"], name="cell pass")
        expected_finite_ok = bool(
            float(expected_macro_summary["finite_fraction"])
            >= MINIMUM_FINITE_BOOTSTRAP_FRACTION
            and float(expected_regret_summary["finite_fraction"])
            >= MINIMUM_FINITE_BOOTSTRAP_FRACTION
        )
        expected_pass = bool(
            expected_finite_ok
            and macro_low >= -MACRO_F1_PRACTICAL_MINIMUM
            and regret_high <= MEAN_REGRET_PRACTICAL_MINIMUM
        )
        if finite_ok != expected_finite_ok or cell_pass != expected_pass:
            raise HarmBenchSelectionStatisticsError("no-harm cell gate changed")
        observed_passes.append(cell_pass)
    if (
        _exact_bool(no_harm["all_six_cells_pass"], name="all-six gate")
        != all(observed_passes)
        or no_harm["confirmatory_authority_granted"] is not False
    ):
        raise HarmBenchSelectionStatisticsError("no-harm aggregate authority changed")
    _privacy_and_finite_scan(report)


def evaluate_selection_statistics(
    sealed_inputs: JointSelectionEvaluationInputs,
) -> dict[str, object]:
    """Run the frozen production analysis on attempt-bound activated labels."""

    inputs = _revalidate_joint_inputs(sealed_inputs)
    if inputs.source_kind != "attempt_bound_activated_labels":
        raise HarmBenchSelectionStatisticsError(
            "synthetic fixtures cannot enter the production statistics entry point"
        )
    return _compute_statistics_core(inputs)


def _evaluate_trusted_synthetic_selection_statistics(
    sealed_inputs: JointSelectionEvaluationInputs,
) -> dict[str, object]:
    """Private test seam; synthetic outputs have no publication authority."""

    inputs = _revalidate_joint_inputs(sealed_inputs)
    if inputs.source_kind != "trusted_synthetic_fixture_only":
        raise HarmBenchSelectionStatisticsError("synthetic test seam requires synthetic input")
    return _compute_statistics_core(inputs)


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "EXPLORATORY_STATUS",
    "HarmBenchSelectionStatisticsError",
    "JointSelectionEvaluationInputs",
    "RANDOMIZATION_EXACT_MAX_CLUSTERS",
    "RANDOMIZATION_MC_REPLICATES",
    "RANDOMIZATION_SEED",
    "SELECTION_STATISTICS_SCHEMA",
    "evaluate_selection_statistics",
    "load_joint_selection_evaluation_inputs",
    "validate_selection_statistics_report",
]
