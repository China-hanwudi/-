"""Outcome-facing metrics for the frozen HarmBench-ERC protocol.

The module is intentionally data-source agnostic.  It accepts only aligned
labels, probabilities, history eligibility and an already-frozen selection
mask/threshold.  It never chooses an official-test operating point.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

import numpy as np
from sklearn.metrics import f1_score


class HarmBenchMetricError(ValueError):
    """Raised when an input violates the benchmark metric contract."""


PROBABILITY_TOLERANCE = 1e-6
NLL_EPSILON = 1e-12
DEFAULT_HARM_THRESHOLDS = (0.0, 0.05)
DEFAULT_TAIL_ALPHA = 0.90
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "seed_results",
        "query_ids",
        "row_ids",
        "cluster_ids",
        "labels",
        "predictions",
        "probabilities",
        "embeddings",
        "speaker_ids",
        "dialogue_ids",
        "texts",
        "paths",
    }
)
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:")
MAXIMUM_PUBLIC_SEQUENCE_LENGTH = 256


def _finite_vector(values: object, *, name: str, dtype: object = np.float64) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1 or not len(array):
        raise HarmBenchMetricError(f"{name} must be a non-empty one-dimensional vector")
    if not np.isfinite(array).all():
        raise HarmBenchMetricError(f"{name} contains missing or non-finite values")
    return array


def validated_probability(values: object, *, name: str) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim != 2 or not probability.shape[0] or probability.shape[1] < 2:
        raise HarmBenchMetricError(f"{name} must have shape [queries, classes>=2]")
    if not np.isfinite(probability).all():
        raise HarmBenchMetricError(f"{name} contains missing or non-finite values")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise HarmBenchMetricError(f"{name} contains values outside [0, 1]")
    row_sum = probability.sum(axis=1)
    if not np.allclose(row_sum, 1.0, rtol=0.0, atol=PROBABILITY_TOLERANCE):
        raise HarmBenchMetricError(f"{name} rows do not sum to one")
    return probability


def validated_labels(values: object, *, queries: int, classes: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != queries:
        raise HarmBenchMetricError("labels must align one-to-one with probability rows")
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise HarmBenchMetricError("labels must use an integer dtype")
    labels = raw.astype(np.int64, copy=False)
    if np.any(labels < 0) or np.any(labels >= classes):
        raise HarmBenchMetricError("labels contain an out-of-range class index")
    return labels


def _boolean_vector(values: object, *, name: str, queries: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != queries or raw.dtype.kind != "b":
        raise HarmBenchMetricError(f"{name} must be a boolean vector aligned to queries")
    return raw.astype(bool, copy=False)


def _aligned_probabilities(
    labels: object,
    current_probability: object,
    strategy_probability: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = validated_probability(current_probability, name="current_probability")
    strategy = validated_probability(strategy_probability, name="strategy_probability")
    if strategy.shape != current.shape:
        raise HarmBenchMetricError("current and strategy probability shapes differ")
    y_true = validated_labels(labels, queries=current.shape[0], classes=current.shape[1])
    return y_true, current, strategy


def true_class_nll(labels: object, probability: object) -> np.ndarray:
    values = validated_probability(probability, name="probability")
    y_true = validated_labels(labels, queries=values.shape[0], classes=values.shape[1])
    true_probability = values[np.arange(len(y_true)), y_true]
    return -np.log(np.clip(true_probability, NLL_EPSILON, 1.0))


def paired_true_class_regret(
    labels: object,
    current_probability: object,
    strategy_probability: object,
) -> np.ndarray:
    """Return strategy NLL minus independently trained current-only NLL."""

    y_true, current, strategy = _aligned_probabilities(
        labels, current_probability, strategy_probability
    )
    current_nll = -np.log(
        np.clip(current[np.arange(len(y_true)), y_true], NLL_EPSILON, 1.0)
    )
    strategy_nll = -np.log(
        np.clip(strategy[np.arange(len(y_true)), y_true], NLL_EPSILON, 1.0)
    )
    return strategy_nll - current_nll


def multiclass_brier_per_query(labels: object, probability: object) -> np.ndarray:
    values = validated_probability(probability, name="probability")
    y_true = validated_labels(labels, queries=values.shape[0], classes=values.shape[1])
    one_hot = np.zeros_like(values)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return np.sum((values - one_hot) ** 2, axis=1)


def empirical_upper_cvar(values: object, *, alpha: float = DEFAULT_TAIL_ALPHA) -> float:
    """Exact empirical upper-tail CVaR with fractional boundary mass.

    For n observations, the upper tail has mass ``(1-alpha) * n``.  The
    function averages the largest observations using a fractional weight at
    the boundary when that mass is non-integral.  This remains correct when
    many observations tie at the quantile (notably zero fallback regret).
    """

    if not 0.0 <= float(alpha) < 1.0:
        raise HarmBenchMetricError("alpha must be in [0, 1)")
    vector = _finite_vector(values, name="values")
    tail_mass = (1.0 - float(alpha)) * len(vector)
    if tail_mass <= 0.0:
        raise HarmBenchMetricError("upper-tail mass must be positive")
    descending = np.sort(vector)[::-1]
    whole = int(np.floor(tail_mass))
    fraction = float(tail_mass - whole)
    total = float(descending[:whole].sum()) if whole else 0.0
    if fraction > 0.0:
        total += fraction * float(descending[whole])
    return float(total / tail_mass)


def classification_metrics(labels: object, probability: object) -> dict[str, float]:
    values = validated_probability(probability, name="probability")
    y_true = validated_labels(labels, queries=values.shape[0], classes=values.shape[1])
    prediction = np.argmax(values, axis=1)
    return {
        "macro_f1": float(
            f1_score(
                y_true,
                prediction,
                labels=np.arange(values.shape[1]),
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(np.mean(prediction == y_true)),
        "mean_nll": float(np.mean(true_class_nll(y_true, values))),
        "mean_brier": float(np.mean(multiclass_brier_per_query(y_true, values))),
    }


def hybrid_probability(
    current_probability: object,
    strategy_probability: object,
    selected: object,
) -> np.ndarray:
    current = validated_probability(current_probability, name="current_probability")
    strategy = validated_probability(strategy_probability, name="strategy_probability")
    if current.shape != strategy.shape:
        raise HarmBenchMetricError("current and strategy probability shapes differ")
    mask = _boolean_vector(selected, name="selected", queries=current.shape[0])
    return np.where(mask[:, None], strategy, current)


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability))


def _validate_frozen_metric_parameters(
    harm_thresholds: Sequence[float], tail_alpha: float
) -> tuple[float, ...]:
    thresholds = tuple(float(value) for value in harm_thresholds)
    if thresholds != DEFAULT_HARM_THRESHOLDS:
        raise HarmBenchMetricError(
            f"harm thresholds are frozen at {DEFAULT_HARM_THRESHOLDS}"
        )
    if float(tail_alpha) != DEFAULT_TAIL_ALPHA:
        raise HarmBenchMetricError(f"tail alpha is frozen at {DEFAULT_TAIL_ALPHA}")
    return thresholds


def describe_regret(
    regret: object,
    selected: object,
    *,
    harm_thresholds: Sequence[float] = DEFAULT_HARM_THRESHOLDS,
    tail_alpha: float = DEFAULT_TAIL_ALPHA,
) -> dict[str, object]:
    values = _finite_vector(regret, name="regret")
    mask = _boolean_vector(selected, name="selected", queries=len(values))
    thresholds = _validate_frozen_metric_parameters(harm_thresholds, tail_alpha)

    population = np.where(mask, values, 0.0)
    used = values[mask]
    used_harm = {
        f"greater_than_{threshold:.12g}": float(np.mean(used > threshold))
        if len(used)
        else None
        for threshold in thresholds
    }
    population_harm = {
        f"greater_than_{threshold:.12g}": float(np.mean(mask & (values > threshold)))
        for threshold in thresholds
    }
    return {
        "eligible_queries": int(len(values)),
        "used_queries": int(mask.sum()),
        "coverage": float(np.mean(mask)),
        "population": {
            "mean_regret": float(np.mean(population)),
            "p90_regret": _quantile(population, 0.90),
            "cvar90_regret": empirical_upper_cvar(population, alpha=tail_alpha),
            "harm_rate": population_harm,
        },
        "conditional_on_used": {
            "mean_regret": float(np.mean(used)) if len(used) else None,
            "p90_regret": _quantile(used, 0.90) if len(used) else None,
            "cvar90_regret": empirical_upper_cvar(used, alpha=tail_alpha)
            if len(used)
            else None,
            "harm_rate": used_harm,
        },
    }


def evaluate_frozen_policy(
    labels: object,
    current_probability: object,
    strategy_probability: object,
    history_eligible: object,
    selected: object,
    *,
    harm_thresholds: Sequence[float] = DEFAULT_HARM_THRESHOLDS,
    tail_alpha: float = DEFAULT_TAIL_ALPHA,
) -> dict[str, object]:
    y_true, current, strategy = _aligned_probabilities(
        labels, current_probability, strategy_probability
    )
    _validate_frozen_metric_parameters(harm_thresholds, tail_alpha)
    eligible = _boolean_vector(
        history_eligible, name="history_eligible", queries=len(y_true)
    )
    use_history = _boolean_vector(selected, name="selected", queries=len(y_true))
    if np.any(use_history & ~eligible):
        raise HarmBenchMetricError("selected contains queries without strictly past history")
    if not np.any(eligible):
        raise HarmBenchMetricError("no history-eligible queries are available")

    hybrid = hybrid_probability(current, strategy, use_history)
    current_metrics = classification_metrics(y_true, current)
    hybrid_metrics = classification_metrics(y_true, hybrid)
    regret = paired_true_class_regret(y_true, current, strategy)[eligible]
    regret_report = describe_regret(
        regret,
        use_history[eligible],
        harm_thresholds=harm_thresholds,
        tail_alpha=tail_alpha,
    )
    current_brier = multiclass_brier_per_query(y_true, current)
    strategy_brier = multiclass_brier_per_query(y_true, strategy)
    brier_regret = strategy_brier[eligible] - current_brier[eligible]
    brier_used = use_history[eligible]
    brier_population = np.where(brier_used, brier_regret, 0.0)
    brier_conditional = brier_regret[brier_used]
    prediction_current = np.argmax(current, axis=1)
    prediction_hybrid = np.argmax(hybrid, axis=1)
    broken = (prediction_current == y_true) & (prediction_hybrid != y_true)
    rescued = (prediction_current != y_true) & (prediction_hybrid == y_true)

    def transition_rates(mask: np.ndarray) -> dict[str, float | None]:
        denominator = int(mask.sum())
        return {
            "denominator_queries": denominator,
            "history_breaks_correct_current": float(np.mean(broken[mask]))
            if denominator
            else None,
            "history_rescues_wrong_current": float(np.mean(rescued[mask]))
            if denominator
            else None,
        }

    return {
        "metric_contract": {
            "nll_probability_floor": NLL_EPSILON,
            "harm_thresholds_nats": list(DEFAULT_HARM_THRESHOLDS),
            "tail_alpha": DEFAULT_TAIL_ALPHA,
            "tail_estimator": "exact_empirical_upper_tail_with_fractional_boundary_mass",
        },
        "current_only": current_metrics,
        "hybrid": hybrid_metrics,
        "hybrid_minus_current": {
            key: float(hybrid_metrics[key] - current_metrics[key])
            for key in current_metrics
        },
        "nll_regret": regret_report,
        "brier_regret": {
            "eligible_queries": int(eligible.sum()),
            "used_queries": int(brier_used.sum()),
            "coverage": float(np.mean(brier_used)),
            "population": {
                "mean_regret": float(np.mean(brier_population)),
                "p90_regret": _quantile(brier_population, 0.90),
                "cvar90_regret": empirical_upper_cvar(
                    brier_population, alpha=DEFAULT_TAIL_ALPHA
                ),
                "harm_rate_greater_than_0": float(
                    np.mean(brier_used & (brier_regret > 0.0))
                ),
            },
            "conditional_on_used": {
                "mean_regret": float(np.mean(brier_conditional))
                if len(brier_conditional)
                else None,
                "p90_regret": _quantile(brier_conditional, 0.90)
                if len(brier_conditional)
                else None,
                "cvar90_regret": empirical_upper_cvar(
                    brier_conditional, alpha=DEFAULT_TAIL_ALPHA
                )
                if len(brier_conditional)
                else None,
                "harm_rate_greater_than_0": float(
                    np.mean(brier_conditional > 0.0)
                )
                if len(brier_conditional)
                else None,
            },
        },
        "classification_transitions": {
            "all_queries": transition_rates(np.ones(len(y_true), dtype=bool)),
            "history_eligible_queries": transition_rates(eligible),
            "history_selected_queries": transition_rates(use_history),
        },
    }


def evaluate_frozen_thresholds(
    labels: object,
    current_probability: object,
    strategy_probability: object,
    history_eligible: object,
    risk_score: object,
    thresholds: Sequence[float],
    *,
    harm_thresholds: Sequence[float] = DEFAULT_HARM_THRESHOLDS,
    tail_alpha: float = DEFAULT_TAIL_ALPHA,
) -> dict[str, object]:
    current = validated_probability(current_probability, name="current_probability")
    score = _finite_vector(risk_score, name="risk_score")
    if len(score) != current.shape[0]:
        raise HarmBenchMetricError("risk_score must align one-to-one with queries")
    eligible = _boolean_vector(
        history_eligible, name="history_eligible", queries=current.shape[0]
    )
    frozen = tuple(float(value) for value in thresholds)
    if not frozen or not np.isfinite(frozen).all() or len(set(frozen)) != len(frozen):
        raise HarmBenchMetricError("thresholds must be a non-empty unique finite sequence")

    entries: dict[str, object] = {}
    for threshold in frozen:
        selected = eligible & (score <= threshold)
        entries[f"{threshold:.17g}"] = {
            "frozen_threshold": threshold,
            "evaluation": evaluate_frozen_policy(
                labels,
                current,
                strategy_probability,
                eligible,
                selected,
                harm_thresholds=harm_thresholds,
                tail_alpha=tail_alpha,
            ),
        }
    return {
        "selection_rule": "use_history_when_risk_score_less_than_or_equal_to_frozen_threshold",
        "official_test_top_k_reselection_permitted": False,
        "thresholds": entries,
    }


def ensure_finite_public_tree(value: object, *, path: str = "$") -> None:
    """Second-line JSON/privacy scan; an exact public schema is still required."""

    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, str):
        lowered = value.lower()
        if len(value) > 4096:
            raise HarmBenchMetricError(f"public string is unexpectedly long at {path}")
        if (
            WINDOWS_ABSOLUTE_PATH.match(value)
            or value.startswith(("/", "\\\\", "~/", "~\\"))
            or lowered.startswith("file://")
            or "/users/" in lowered
            or "\\users\\" in lowered
        ):
            raise HarmBenchMetricError(f"public metric tree contains a local path at {path}")
        return
    if isinstance(value, float):
        if not np.isfinite(value):
            raise HarmBenchMetricError(f"public metric tree contains non-finite value at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise HarmBenchMetricError(f"public metric tree has a non-string key at {path}")
            if key in FORBIDDEN_PUBLIC_KEYS:
                raise HarmBenchMetricError(
                    f"public metric tree contains forbidden key {key} at {path}"
                )
            ensure_finite_public_tree(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        if len(value) > MAXIMUM_PUBLIC_SEQUENCE_LENGTH:
            raise HarmBenchMetricError(f"public sequence is unexpectedly long at {path}")
        for index, child in enumerate(value):
            ensure_finite_public_tree(child, path=f"{path}[{index}]")
        return
    raise HarmBenchMetricError(
        f"public metric tree contains unsupported type {type(value).__name__} at {path}"
    )
