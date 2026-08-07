"""Leakage-safe MELD audio+text history-utility feasibility experiment.

The same multimodal base predictions define the utility target for both
selectors.  Therefore the text-meta versus audio-augmented comparison tests
whether acoustic evidence improves utility prediction, rather than conflating
the comparison with a different emotion classifier.  MELD test is rejected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .meld_audio_contract import AUDIO_FEATURE_NAMES, load_official_split, sha256_file
from .meld_text_pilot import (
    _align_probabilities,
    _build_vectorizer,
    aggregate_custom_sets,
    build_history_indices,
    cluster_bootstrap_difference,
    make_classifier,
    prediction_metrics,
    true_class_loss,
)


TEXT_SELECTOR_FEATURE_NAMES = (
    "log_history_count",
    "current_history_text_cosine",
    "history_on_confidence",
    "history_zero_confidence",
    "history_on_entropy",
    "history_zero_entropy",
    "probability_l1_shift",
    "probability_l2_shift",
    "prediction_agreement",
    *[f"history_on_prob_{index}" for index in range(7)],
    *[f"history_zero_prob_{index}" for index in range(7)],
)

AUDIO_SELECTOR_FEATURE_NAMES = (
    *TEXT_SELECTOR_FEATURE_NAMES,
    "current_history_audio_cosine",
    "current_history_audio_l1",
    "current_history_audio_l2",
    "current_history_audio_max_abs",
    *[f"audio_current_{name}" for name in AUDIO_FEATURE_NAMES],
    *[f"audio_history_{name}" for name in AUDIO_FEATURE_NAMES],
    *[f"audio_absdiff_{name}" for name in AUDIO_FEATURE_NAMES],
)


def load_aligned_split(csv_path: Path, audio_npz: Path) -> tuple[pd.DataFrame, np.ndarray]:
    if "test" in csv_path.name.casefold() or "test" in audio_npz.name.casefold():
        raise ValueError("MELD test is sealed")
    frame = load_official_split(csv_path)
    with np.load(audio_npz, allow_pickle=False) as payload:
        keys = payload["media_key"].astype(str)
        features = payload["audio_features"].astype(np.float64)
        names = tuple(payload["feature_names"].astype(str).tolist())
    if names != tuple(AUDIO_FEATURE_NAMES):
        raise ValueError("audio feature-name drift")
    if features.shape != (len(keys), len(AUDIO_FEATURE_NAMES)):
        raise ValueError("audio feature geometry mismatch")
    if len(set(keys.tolist())) != len(keys):
        raise ValueError("duplicate audio keys")
    if not np.isfinite(features).all():
        raise ValueError("non-finite audio features")
    lookup = {key: index for index, key in enumerate(keys)}
    keep = frame["media_key"].isin(lookup).to_numpy()
    aligned = frame.loc[keep].copy().reset_index(drop=True)
    aligned["_row_id"] = np.arange(len(aligned), dtype=int)
    order = np.asarray([lookup[key] for key in aligned["media_key"].astype(str)], dtype=int)
    return aligned, features[order]


def aggregate_dense_history(
    current: np.ndarray, histories: Sequence[Sequence[int]]
) -> tuple[np.ndarray, np.ndarray]:
    history = np.zeros_like(current, dtype=np.float64)
    counts = np.zeros(len(histories), dtype=np.int32)
    for query, selected in enumerate(histories):
        selected = tuple(int(index) for index in selected)
        counts[query] = len(selected)
        if selected:
            history[query] = current[list(selected)].mean(axis=0)
    return history, counts


def base_features(
    current_text: sparse.csr_matrix,
    history_text: sparse.csr_matrix,
    counts: np.ndarray,
    current_audio: np.ndarray,
    history_audio: np.ndarray,
) -> sparse.csr_matrix:
    return sparse.hstack(
        [
            current_text,
            history_text,
            sparse.csr_matrix(np.log1p(counts).reshape(-1, 1)),
            sparse.csr_matrix(current_audio),
            sparse.csr_matrix(history_audio),
        ],
        format="csr",
    )


def zero_base_features(
    current_text: sparse.csr_matrix, current_audio: np.ndarray
) -> sparse.csr_matrix:
    return base_features(
        current_text,
        sparse.csr_matrix(current_text.shape, dtype=current_text.dtype),
        np.zeros(current_text.shape[0], dtype=np.int32),
        current_audio,
        np.zeros_like(current_audio),
    )


def text_selector_features(
    current: sparse.csr_matrix,
    history: sparse.csr_matrix,
    counts: np.ndarray,
    actual_probability: np.ndarray,
    zero_probability: np.ndarray,
) -> np.ndarray:
    dot = np.asarray(current.multiply(history).sum(axis=1)).ravel()
    current_norm = np.sqrt(np.asarray(current.multiply(current).sum(axis=1)).ravel())
    history_norm = np.sqrt(np.asarray(history.multiply(history).sum(axis=1)).ravel())
    cosine = dot / np.maximum(current_norm * history_norm, 1e-12)
    actual_entropy = -(actual_probability * np.log(np.clip(actual_probability, 1e-12, 1))).sum(1)
    zero_entropy = -(zero_probability * np.log(np.clip(zero_probability, 1e-12, 1))).sum(1)
    delta = actual_probability - zero_probability
    output = np.column_stack(
        [
            np.log1p(counts),
            cosine,
            actual_probability.max(axis=1),
            zero_probability.max(axis=1),
            actual_entropy,
            zero_entropy,
            np.abs(delta).sum(axis=1),
            np.sqrt((delta**2).sum(axis=1)),
            (actual_probability.argmax(1) == zero_probability.argmax(1)).astype(float),
            actual_probability,
            zero_probability,
        ]
    )
    if output.shape[1] != len(TEXT_SELECTOR_FEATURE_NAMES):
        raise RuntimeError("text selector feature geometry mismatch")
    return output


def audio_augmented_selector_features(
    text_features: np.ndarray,
    current_audio: np.ndarray,
    history_audio: np.ndarray,
) -> np.ndarray:
    difference = current_audio - history_audio
    current_norm = np.linalg.norm(current_audio, axis=1)
    history_norm = np.linalg.norm(history_audio, axis=1)
    cosine = np.sum(current_audio * history_audio, axis=1) / np.maximum(
        current_norm * history_norm, 1e-12
    )
    absolute = np.abs(difference)
    output = np.column_stack(
        [
            text_features,
            cosine,
            absolute.mean(axis=1),
            np.linalg.norm(difference, axis=1),
            absolute.max(axis=1),
            current_audio,
            history_audio,
            absolute,
        ]
    )
    if output.shape[1] != len(AUDIO_SELECTOR_FEATURE_NAMES):
        raise RuntimeError("audio selector feature geometry mismatch")
    return output


def _risk_regressor(config: dict, loss: str, seed: int, quantile: float | None = None):
    cfg = config["risk_model"]
    kwargs = dict(
        loss=loss,
        learning_rate=float(cfg["learning_rate"]),
        max_iter=int(cfg["max_iter"]),
        max_leaf_nodes=int(cfg["max_leaf_nodes"]),
        min_samples_leaf=int(cfg["min_samples_leaf"]),
        l2_regularization=float(cfg["l2_regularization"]),
        random_state=int(seed),
    )
    if quantile is not None:
        kwargs["quantile"] = float(quantile)
    return HistGradientBoostingRegressor(**kwargs)


def _risk_classifier(config: dict, seed: int) -> HistGradientBoostingClassifier:
    cfg = config["risk_model"]
    return HistGradientBoostingClassifier(
        learning_rate=float(cfg["learning_rate"]),
        max_iter=int(cfg["max_iter"]),
        max_leaf_nodes=int(cfg["max_leaf_nodes"]),
        min_samples_leaf=int(cfg["min_samples_leaf"]),
        l2_regularization=float(cfg["l2_regularization"]),
        class_weight="balanced",
        random_state=int(seed),
    )


def _fit_risk_ensemble(
    features: np.ndarray,
    target: np.ndarray,
    fit_indices: np.ndarray,
    config: dict,
) -> dict[str, list]:
    models: dict[str, list] = {"mean": [], "quantile": [], "harm": []}
    for seed in config["seeds"]:
        mean = _risk_regressor(config, "squared_error", int(seed))
        quantile = _risk_regressor(
            config, "quantile", int(seed), quantile=float(config["risk_quantile"])
        )
        harm = _risk_classifier(config, int(seed))
        mean.fit(features[fit_indices], target[fit_indices])
        quantile.fit(features[fit_indices], target[fit_indices])
        harm.fit(features[fit_indices], (target[fit_indices] > 0).astype(int))
        models["mean"].append(mean)
        models["quantile"].append(quantile)
        models["harm"].append(harm)
    return models


def _risk_predictions(models: dict[str, list], features: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": np.mean([model.predict(features) for model in models["mean"]], axis=0),
        "quantile": np.mean(
            [model.predict(features) for model in models["quantile"]], axis=0
        ),
        "harm": np.mean(
            [model.predict_proba(features)[:, 1] for model in models["harm"]], axis=0
        ),
    }


def _safe_spearman(truth: np.ndarray, prediction: np.ndarray) -> float:
    value = float(spearmanr(truth, prediction).statistic)
    return value if np.isfinite(value) else 0.0


def _selector_metrics(target: np.ndarray, prediction: dict[str, np.ndarray]) -> dict:
    harm = (target > 0).astype(int)
    return {
        "harm_auc": float(roc_auc_score(harm, prediction["harm"])),
        "harm_balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(harm, (prediction["harm"] >= 0.5).astype(int))
        ),
        "mean_prediction_spearman": _safe_spearman(target, prediction["mean"]),
    }


def _cluster_bootstrap_selector_gain(
    target: np.ndarray,
    text_prediction: dict[str, np.ndarray],
    audio_prediction: dict[str, np.ndarray],
    groups: np.ndarray,
    replicates: int,
    seed: int,
) -> dict:
    unique = np.unique(groups)
    rows = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    auc_gain: list[float] = []
    spearman_gain: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([rows[group] for group in sampled])
        truth = target[indices]
        harm = (truth > 0).astype(int)
        if np.unique(harm).size < 2:
            continue
        auc_gain.append(
            roc_auc_score(harm, audio_prediction["harm"][indices])
            - roc_auc_score(harm, text_prediction["harm"][indices])
        )
        spearman_gain.append(
            _safe_spearman(truth, audio_prediction["mean"][indices])
            - _safe_spearman(truth, text_prediction["mean"][indices])
        )
    if not auc_gain or not spearman_gain:
        raise RuntimeError("selector bootstrap produced no valid replicates")
    return {
        "valid_replicates": len(auc_gain),
        "harm_auc_gain_ci_low": float(np.quantile(auc_gain, 0.025)),
        "harm_auc_gain_ci_high": float(np.quantile(auc_gain, 0.975)),
        "spearman_gain_ci_low": float(np.quantile(spearman_gain, 0.025)),
        "spearman_gain_ci_high": float(np.quantile(spearman_gain, 0.975)),
    }


def run_audio_text_risk(
    train_csv: Path,
    dev_csv: Path,
    train_audio_npz: Path,
    dev_audio_npz: Path,
    config_path: Path,
) -> tuple[dict, pd.DataFrame]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    train, train_audio_raw = load_aligned_split(train_csv, train_audio_npz)
    dev, dev_audio_raw = load_aligned_split(dev_csv, dev_audio_npz)
    train_histories = build_history_indices(train)
    dev_histories = build_history_indices(dev)
    train_counts_raw = np.asarray([len(values) for values in train_histories])
    dev_counts_raw = np.asarray([len(values) for values in dev_histories])
    eligible_train = train_counts_raw > 0
    eligible_dev = dev_counts_raw > 0

    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train["Emotion"].astype(str))
    dev_y = encoder.transform(dev["Emotion"].astype(str))
    n_classes = len(encoder.classes_)
    if n_classes != 7:
        raise ValueError(f"expected seven MELD emotions, found {n_classes}")
    groups = train["Dialogue_ID"].astype(str).to_numpy()

    oof_text = np.full((len(train), len(TEXT_SELECTOR_FEATURE_NAMES)), np.nan)
    oof_audio = np.full((len(train), len(AUDIO_SELECTOR_FEATURE_NAMES)), np.nan)
    oof_actual_loss = np.full(len(train), np.nan)
    oof_zero_loss = np.full(len(train), np.nan)
    splitter = GroupKFold(n_splits=int(config["crossfit_folds"]))
    for fold, (fit_index, held_index) in enumerate(
        splitter.split(np.zeros(len(train)), train_y, groups), start=1
    ):
        vectorizer = _build_vectorizer(config)
        vectorizer.fit(train.iloc[fit_index]["Utterance"].astype(str))
        current_text = vectorizer.transform(train["Utterance"].astype(str)).tocsr()
        history_text, counts = aggregate_custom_sets(current_text, train_histories)
        scaler = StandardScaler().fit(train_audio_raw[fit_index])
        current_audio = np.clip(
            scaler.transform(train_audio_raw),
            -float(config["audio_standardization_clip"]),
            float(config["audio_standardization_clip"]),
        )
        history_audio, audio_counts = aggregate_dense_history(current_audio, train_histories)
        if not np.array_equal(counts, audio_counts):
            raise RuntimeError("text/audio history count mismatch")
        actual = base_features(
            current_text, history_text, counts, current_audio, history_audio
        )
        zero = zero_base_features(current_text, current_audio)
        fit_x = sparse.vstack([actual[fit_index], zero[fit_index]], format="csr")
        fit_y = np.concatenate([train_y[fit_index], train_y[fit_index]])
        model = make_classifier(config, int(config["seeds"][0]) + fold)
        model.fit(fit_x, fit_y)
        actual_probability = _align_probabilities(
            model, model.predict_proba(actual[held_index]), n_classes
        )
        zero_probability = _align_probabilities(
            model, model.predict_proba(zero[held_index]), n_classes
        )
        text_meta = text_selector_features(
            current_text[held_index],
            history_text[held_index],
            counts[held_index],
            actual_probability,
            zero_probability,
        )
        oof_text[held_index] = text_meta
        oof_audio[held_index] = audio_augmented_selector_features(
            text_meta, current_audio[held_index], history_audio[held_index]
        )
        oof_actual_loss[held_index] = true_class_loss(
            train_y[held_index], actual_probability
        )
        oof_zero_loss[held_index] = true_class_loss(train_y[held_index], zero_probability)

    if np.isnan(oof_text[eligible_train]).any() or np.isnan(oof_audio[eligible_train]).any():
        raise RuntimeError("incomplete cross-fitted selector features")
    target = oof_actual_loss - oof_zero_loss
    selector_indices = np.flatnonzero(eligible_train)
    split_selector = GroupShuffleSplit(
        n_splits=1,
        test_size=float(config["selector_calibration_fraction"]),
        random_state=int(config["bootstrap_seed"]),
    )
    fit_local, calibration_local = next(
        split_selector.split(selector_indices, groups=groups[eligible_train])
    )
    selector_fit = selector_indices[fit_local]
    selector_calibration = selector_indices[calibration_local]
    selector_models = {
        "text_meta": _fit_risk_ensemble(oof_text, target, selector_fit, config),
        "audio_augmented": _fit_risk_ensemble(oof_audio, target, selector_fit, config),
    }

    vectorizer = _build_vectorizer(config)
    train_current_text = vectorizer.fit_transform(train["Utterance"].astype(str)).tocsr()
    dev_current_text = vectorizer.transform(dev["Utterance"].astype(str)).tocsr()
    train_history_text, train_counts = aggregate_custom_sets(
        train_current_text, train_histories
    )
    dev_history_text, dev_counts = aggregate_custom_sets(dev_current_text, dev_histories)
    scaler = StandardScaler().fit(train_audio_raw)
    clip = float(config["audio_standardization_clip"])
    train_audio = np.clip(scaler.transform(train_audio_raw), -clip, clip)
    dev_audio = np.clip(scaler.transform(dev_audio_raw), -clip, clip)
    train_history_audio, train_audio_counts = aggregate_dense_history(
        train_audio, train_histories
    )
    dev_history_audio, dev_audio_counts = aggregate_dense_history(dev_audio, dev_histories)
    if not np.array_equal(train_counts, train_audio_counts) or not np.array_equal(
        dev_counts, dev_audio_counts
    ):
        raise RuntimeError("final text/audio history count mismatch")
    train_actual = base_features(
        train_current_text,
        train_history_text,
        train_counts,
        train_audio,
        train_history_audio,
    )
    dev_actual = base_features(
        dev_current_text, dev_history_text, dev_counts, dev_audio, dev_history_audio
    )
    train_zero = zero_base_features(train_current_text, train_audio)
    dev_zero = zero_base_features(dev_current_text, dev_audio)
    fit_history_x = sparse.vstack([train_actual, train_zero], format="csr")
    fit_history_y = np.concatenate([train_y, train_y])
    actual_probabilities: list[np.ndarray] = []
    zero_probabilities: list[np.ndarray] = []
    current_probabilities: list[np.ndarray] = []
    per_seed: list[dict] = []
    for seed in config["seeds"]:
        history_model = make_classifier(config, int(seed))
        history_model.fit(fit_history_x, fit_history_y)
        actual = _align_probabilities(
            history_model, history_model.predict_proba(dev_actual), n_classes
        )
        zero = _align_probabilities(
            history_model, history_model.predict_proba(dev_zero), n_classes
        )
        current_model = make_classifier(config, int(seed))
        current_model.fit(train_zero, train_y)
        current = _align_probabilities(
            current_model, current_model.predict_proba(dev_zero), n_classes
        )
        actual_probabilities.append(actual)
        zero_probabilities.append(zero)
        current_probabilities.append(current)
        per_seed.append(
            {
                "seed": int(seed),
                "current_only": prediction_metrics(dev_y, current, int(config["ece_bins"])),
                "within_model_zero": prediction_metrics(dev_y, zero, int(config["ece_bins"])),
                "full_history": prediction_metrics(dev_y, actual, int(config["ece_bins"])),
            }
        )
    actual_probability = np.mean(actual_probabilities, axis=0)
    zero_probability = np.mean(zero_probabilities, axis=0)
    current_probability = np.mean(current_probabilities, axis=0)
    actual_loss = true_class_loss(dev_y, actual_probability)
    zero_loss = true_class_loss(dev_y, zero_probability)
    dev_target = actual_loss - zero_loss

    dev_text_meta = text_selector_features(
        dev_current_text,
        dev_history_text,
        dev_counts,
        actual_probability,
        zero_probability,
    )
    dev_audio_meta = audio_augmented_selector_features(
        dev_text_meta, dev_audio, dev_history_audio
    )
    feature_sets = {"text_meta": oof_text, "audio_augmented": oof_audio}
    dev_feature_sets = {"text_meta": dev_text_meta, "audio_augmented": dev_audio_meta}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    corrections: dict[str, float] = {}
    selector_reports: dict[str, dict] = {}
    policies: dict[str, dict] = {}
    dev_groups = dev["Dialogue_ID"].astype(str).to_numpy()

    def policy_report(use_history: np.ndarray) -> dict:
        use_history = use_history & eligible_dev
        probability = zero_probability.copy()
        probability[use_history] = actual_probability[use_history]
        policy_loss = true_class_loss(dev_y, probability)
        difference = policy_loss[eligible_dev] - zero_loss[eligible_dev]
        used = use_history[eligible_dev]
        selected = dev_target[use_history]
        bootstrap = cluster_bootstrap_difference(
            difference,
            dev_groups[eligible_dev],
            int(config["bootstrap_replicates"]),
            int(config["bootstrap_seed"]) + int(round(10000 * used.mean())),
        )
        return {
            "history_coverage": float(used.mean()),
            "metrics": prediction_metrics(dev_y, probability, int(config["ece_bins"])),
            "harm_rate_all_eligible": float((difference > 0).mean()),
            "harm_rate_among_used": float((selected > 0).mean()) if used.any() else None,
            "mean_excess_among_used": float(selected.mean()) if used.any() else None,
            "p90_excess_among_used": float(np.quantile(selected, 0.90)) if used.any() else None,
            "p99_excess_among_used": float(np.quantile(selected, 0.99)) if used.any() else None,
            "cluster_bootstrap_policy_minus_zero": bootstrap,
        }

    quantile = float(config["risk_quantile"])
    for name in ("text_meta", "audio_augmented"):
        calibration_prediction = _risk_predictions(
            selector_models[name], feature_sets[name][selector_calibration]
        )
        residual = target[selector_calibration] - calibration_prediction["quantile"]
        conformal_level = min(
            1.0, np.ceil((len(residual) + 1) * quantile) / len(residual)
        )
        correction = float(np.quantile(residual, conformal_level, method="higher"))
        corrections[name] = correction
        prediction = _risk_predictions(selector_models[name], dev_feature_sets[name])
        prediction["upper"] = prediction["quantile"] + correction
        predictions[name] = prediction
        metrics = _selector_metrics(dev_target[eligible_dev], {
            key: value[eligible_dev] for key, value in prediction.items() if key != "upper"
        })
        metrics.update(
            {
                "conformal_correction_nats": correction,
                "calibration_upper_coverage": float(
                    (target[selector_calibration] <= calibration_prediction["quantile"] + correction).mean()
                ),
                "validation_upper_coverage": float(
                    (dev_target[eligible_dev] <= prediction["upper"][eligible_dev]).mean()
                ),
            }
        )
        selector_reports[name] = metrics
        name_policies = {
            "conformal_q90_upper_below_zero": policy_report(prediction["upper"] < 0)
        }
        calibration_upper = calibration_prediction["quantile"] + correction
        for coverage in config["coverage_targets"]:
            threshold = float(np.quantile(calibration_upper, float(coverage)))
            report = policy_report(prediction["upper"] <= threshold)
            report["calibration_risk_threshold"] = threshold
            name_policies[f"calibration_target_{int(round(100 * float(coverage)))}pct"] = report
        policies[name] = name_policies

    text_metrics = selector_reports["text_meta"]
    audio_metrics = selector_reports["audio_augmented"]
    gain = {
        "harm_auc_gain": audio_metrics["harm_auc"] - text_metrics["harm_auc"],
        "mean_prediction_spearman_gain": (
            audio_metrics["mean_prediction_spearman"]
            - text_metrics["mean_prediction_spearman"]
        ),
    }
    gain.update(
        _cluster_bootstrap_selector_gain(
            dev_target[eligible_dev],
            {key: value[eligible_dev] for key, value in predictions["text_meta"].items()},
            {key: value[eligible_dev] for key, value in predictions["audio_augmented"].items()},
            dev_groups[eligible_dev],
            int(config["bootstrap_replicates"]),
            int(config["bootstrap_seed"]) + 909,
        )
    )
    gate = config["gates"]
    audio_gain_pass = bool(
        gain["harm_auc_gain"] >= float(gate["minimum_audio_gain_auc"])
        and gain["mean_prediction_spearman_gain"]
        >= float(gate["minimum_audio_gain_spearman"])
        and gain["harm_auc_gain_ci_low"]
        > float(gate["minimum_audio_gain_auc_ci_low"])
        and gain["spearman_gain_ci_low"]
        > float(gate["minimum_audio_gain_spearman_ci_low"])
    )
    primary = policies["audio_augmented"]["conformal_q90_upper_below_zero"]
    primary_bootstrap = primary["cluster_bootstrap_policy_minus_zero"]
    deployment_pass = bool(
        primary["history_coverage"] >= float(gate["minimum_nontrivial_coverage"])
        and primary["harm_rate_among_used"] is not None
        and primary["harm_rate_among_used"] <= float(gate["maximum_selected_harm_rate"])
        and primary["mean_excess_among_used"] is not None
        and primary["mean_excess_among_used"] <= float(gate["maximum_mean_excess_nll"])
        and primary_bootstrap["mean_excess_loss_ci_high"]
        <= float(gate["maximum_mean_excess_ci_high"])
    )

    per_query = pd.DataFrame(
        {
            "Dialogue_ID": dev["Dialogue_ID"].to_numpy(),
            "Utterance_ID": dev["Utterance_ID"].to_numpy(),
            "Speaker": dev["Speaker"].astype(str).to_numpy(),
            "gold_emotion": dev["Emotion"].astype(str).to_numpy(),
            "history_count": dev_counts,
            "eligible_history": eligible_dev,
            "zero_loss": zero_loss,
            "full_history_loss": actual_loss,
            "excess_nll": dev_target,
            "text_predicted_mean": predictions["text_meta"]["mean"],
            "text_predicted_harm": predictions["text_meta"]["harm"],
            "text_predicted_upper": predictions["text_meta"]["upper"],
            "audio_predicted_mean": predictions["audio_augmented"]["mean"],
            "audio_predicted_harm": predictions["audio_augmented"]["harm"],
            "audio_predicted_upper": predictions["audio_augmented"]["upper"],
        }
    )
    return (
        {
            "protocol": config["protocol"],
            "scope": config["scope"],
            "data": {
                "train_rows": int(len(train)),
                "dev_rows": int(len(dev)),
                "eligible_train": int(eligible_train.sum()),
                "eligible_dev": int(eligible_dev.sum()),
                "selector_fit_rows": int(len(selector_fit)),
                "selector_calibration_rows": int(len(selector_calibration)),
                "missing_train_audio_keys": ["dia125_utt3.mp4"],
                "missing_dev_audio_keys": ["dia110_utt7.mp4"],
                "train_audio_npz_sha256": sha256_file(train_audio_npz),
                "dev_audio_npz_sha256": sha256_file(dev_audio_npz),
                "test_opened": False,
            },
            "label_order": encoder.classes_.tolist(),
            "base_models": {
                "current_only": prediction_metrics(
                    dev_y, current_probability, int(config["ece_bins"])
                ),
                "within_model_zero": prediction_metrics(
                    dev_y, zero_probability, int(config["ece_bins"])
                ),
                "full_history": {
                    **prediction_metrics(dev_y, actual_probability, int(config["ece_bins"])),
                    "harm_rate_eligible": float((dev_target[eligible_dev] > 0).mean()),
                    "mean_excess_nll_eligible": float(dev_target[eligible_dev].mean()),
                },
                "per_seed": per_seed,
            },
            "selector_feature_dimensions": {
                "text_meta": len(TEXT_SELECTOR_FEATURE_NAMES),
                "audio_augmented": len(AUDIO_SELECTOR_FEATURE_NAMES),
            },
            "selector_metrics": selector_reports,
            "audio_incremental_gain": gain,
            "policies": policies,
            "gates": {
                "audio_incremental_signal": {
                    "pass": audio_gain_pass,
                    "thresholds": gate,
                },
                "calibrated_safe_fallback": {
                    "pass": deployment_pass,
                    "thresholds": gate,
                },
                "overall_multimodal_method_go": {
                    "pass": bool(audio_gain_pass and deployment_pass)
                },
            },
            "config": config,
            "limitations": [
                "This is real audio plus official text, not a three-modality experiment; visual bytes are absent.",
                "Two official utterances lack audio in the public transport and are explicitly excluded.",
                "Cross-fitted base targets use one seed per fold; final dev base and risk models use five-seed ensembles.",
                "Conformal coverage is marginal over the held-out calibration groups, not a per-speaker guarantee.",
                "Dev is used once for the frozen route decision; MELD test remains sealed.",
            ],
        },
        per_query,
    )


def write_outputs(result: dict, per_query: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    query_output = output.with_name(output.stem + "_per_query.csv.gz")
    per_query.to_csv(query_output, index=False, compression="gzip")
