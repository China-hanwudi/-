"""Cross-fitted point-utility and tail-risk MVP for EmotionTalk text."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from .data_contract import write_json_atomic
from .emotiontalk_text_p1 import (
    _vectorizer,
    build_history_indices,
    load_emotiontalk_split,
)
from .meld_text_pilot import (
    _align_probabilities,
    aggregate_custom_sets,
    cluster_bootstrap_difference,
    combine_features,
    make_classifier,
    prediction_metrics,
    sha256_file,
    true_class_loss,
    zero_history_features,
)


FEATURE_NAMES = (
    "log_history_count",
    "current_history_cosine",
    "history_on_confidence",
    "history_zero_confidence",
    "history_on_entropy",
    "history_zero_entropy",
    "probability_l1_shift",
    "probability_l2_shift",
    "prediction_agreement",
    "history_on_prob_0",
    "history_on_prob_1",
    "history_on_prob_2",
    "history_on_prob_3",
    "history_on_prob_4",
    "history_on_prob_5",
    "history_on_prob_6",
    "history_zero_prob_0",
    "history_zero_prob_1",
    "history_zero_prob_2",
    "history_zero_prob_3",
    "history_zero_prob_4",
    "history_zero_prob_5",
    "history_zero_prob_6",
)


def selector_features(
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
    features = np.column_stack(
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
    if features.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("selector feature geometry mismatch")
    return features


def _risk_model(config: dict, *, loss: str, quantile: float | None = None):
    cfg = config["risk_model"]
    kwargs = {
        "loss": loss,
        "learning_rate": float(cfg["learning_rate"]),
        "max_iter": int(cfg["max_iter"]),
        "max_leaf_nodes": int(cfg["max_leaf_nodes"]),
        "min_samples_leaf": int(cfg["min_samples_leaf"]),
        "l2_regularization": float(cfg["l2_regularization"]),
        "random_state": int(config["bootstrap_seed"]),
    }
    if quantile is not None:
        kwargs["quantile"] = quantile
    return HistGradientBoostingRegressor(**kwargs)


def _harm_model(config: dict) -> HistGradientBoostingClassifier:
    cfg = config["risk_model"]
    return HistGradientBoostingClassifier(
        learning_rate=float(cfg["learning_rate"]),
        max_iter=int(cfg["max_iter"]),
        max_leaf_nodes=int(cfg["max_leaf_nodes"]),
        min_samples_leaf=int(cfg["min_samples_leaf"]),
        l2_regularization=float(cfg["l2_regularization"]),
        class_weight="balanced",
        random_state=int(config["bootstrap_seed"]),
    )


def run_point_risk(data_dir: Path, config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    train = load_emotiontalk_split(data_dir, "train_corpus")
    validation = load_emotiontalk_split(data_dir, "val_corpus")
    train_histories = build_history_indices(train)
    validation_histories = build_history_indices(validation)
    train_counts_raw = np.asarray([len(values) for values in train_histories])
    validation_counts_raw = np.asarray([len(values) for values in validation_histories])
    eligible_train = train_counts_raw > 0
    eligible_validation = validation_counts_raw > 0

    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train["emotion"].astype(str))
    validation_y = encoder.transform(validation["emotion"].astype(str))
    n_classes = len(encoder.classes_)
    groups = (
        train["group"].astype(str) + "_" + train["dialogue"].astype(str)
    ).to_numpy()

    oof_features = np.full((len(train), len(FEATURE_NAMES)), np.nan)
    oof_actual_loss = np.full(len(train), np.nan)
    oof_zero_loss = np.full(len(train), np.nan)
    splitter = GroupKFold(n_splits=int(config["crossfit_folds"]))
    for fold, (fit_index, held_index) in enumerate(
        splitter.split(np.zeros(len(train)), train_y, groups), start=1
    ):
        vectorizer = _vectorizer(config)
        vectorizer.fit(train.iloc[fit_index]["text"].astype(str))
        current = vectorizer.transform(train["text"].astype(str)).tocsr()
        history, counts = aggregate_custom_sets(current, train_histories)
        actual = combine_features(current, history, counts)
        zero = zero_history_features(current)
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
        oof_features[held_index] = selector_features(
            current[held_index],
            history[held_index],
            counts[held_index],
            actual_probability,
            zero_probability,
        )
        oof_actual_loss[held_index] = true_class_loss(train_y[held_index], actual_probability)
        oof_zero_loss[held_index] = true_class_loss(train_y[held_index], zero_probability)

    if np.isnan(oof_features[eligible_train]).any():
        raise RuntimeError("incomplete cross-fit features")
    target = oof_actual_loss - oof_zero_loss
    selector_indices = np.flatnonzero(eligible_train)
    selector_groups = groups[eligible_train]
    split_selector = GroupShuffleSplit(
        n_splits=1,
        test_size=float(config["selector_calibration_fraction"]),
        random_state=int(config["bootstrap_seed"]),
    )
    fit_local, calibration_local = next(
        split_selector.split(selector_indices, groups=selector_groups)
    )
    selector_fit = selector_indices[fit_local]
    selector_calibration = selector_indices[calibration_local]

    mean_model = _risk_model(config, loss="squared_error")
    quantile = float(config["risk_quantile"])
    quantile_model = _risk_model(config, loss="quantile", quantile=quantile)
    harm_model = _harm_model(config)
    mean_model.fit(oof_features[selector_fit], target[selector_fit])
    quantile_model.fit(oof_features[selector_fit], target[selector_fit])
    harm_model.fit(oof_features[selector_fit], (target[selector_fit] > 0).astype(int))

    calibration_q = quantile_model.predict(oof_features[selector_calibration])
    calibration_residual = target[selector_calibration] - calibration_q
    conformal_level = min(
        1.0,
        np.ceil((len(calibration_residual) + 1) * quantile)
        / len(calibration_residual),
    )
    conformal_correction = float(
        np.quantile(calibration_residual, conformal_level, method="higher")
    )

    vectorizer = _vectorizer(config)
    train_current = vectorizer.fit_transform(train["text"].astype(str)).tocsr()
    validation_current = vectorizer.transform(validation["text"].astype(str)).tocsr()
    train_history, train_counts = aggregate_custom_sets(train_current, train_histories)
    validation_history, validation_counts = aggregate_custom_sets(
        validation_current, validation_histories
    )
    train_actual = combine_features(train_current, train_history, train_counts)
    validation_actual = combine_features(
        validation_current, validation_history, validation_counts
    )
    zero_train = zero_history_features(train_current)
    zero_validation = zero_history_features(validation_current)
    fit_x = sparse.vstack([train_actual, zero_train], format="csr")
    fit_y = np.concatenate([train_y, train_y])
    actual_probabilities = []
    zero_probabilities = []
    for seed in config["seeds"]:
        model = make_classifier(config, int(seed))
        model.fit(fit_x, fit_y)
        actual_probabilities.append(
            _align_probabilities(model, model.predict_proba(validation_actual), n_classes)
        )
        zero_probabilities.append(
            _align_probabilities(model, model.predict_proba(zero_validation), n_classes)
        )
    actual_probability = np.mean(actual_probabilities, axis=0)
    zero_probability = np.mean(zero_probabilities, axis=0)
    validation_features = selector_features(
        validation_current,
        validation_history,
        validation_counts,
        actual_probability,
        zero_probability,
    )
    actual_loss = true_class_loss(validation_y, actual_probability)
    zero_loss = true_class_loss(validation_y, zero_probability)
    validation_target = actual_loss - zero_loss
    predicted_mean = mean_model.predict(validation_features)
    predicted_quantile = quantile_model.predict(validation_features)
    predicted_upper = predicted_quantile + conformal_correction
    predicted_harm = harm_model.predict_proba(validation_features)[:, 1]

    validation_groups = (
        validation["group"].astype(str)
        + "_"
        + validation["dialogue"].astype(str)
    ).to_numpy()
    always_history_harm = float((validation_target[eligible_validation] > 0).mean())

    def policy_report(use_history: np.ndarray) -> dict:
        use_history = use_history & eligible_validation
        probability = zero_probability.copy()
        probability[use_history] = actual_probability[use_history]
        policy_loss = true_class_loss(validation_y, probability)
        difference = policy_loss[eligible_validation] - zero_loss[eligible_validation]
        used = use_history[eligible_validation]
        bootstrap = cluster_bootstrap_difference(
            difference,
            validation_groups[eligible_validation],
            int(config["bootstrap_replicates"]),
            int(config["bootstrap_seed"]) + int(10000 * used.mean()),
        )
        selected_difference = validation_target[use_history]
        return {
            "history_coverage": float(used.mean()),
            "metrics": prediction_metrics(
                validation_y, probability, int(config["ece_bins"])
            ),
            "harm_rate_all_eligible": float((difference > 0).mean()),
            "harm_rate_among_used": (
                float((selected_difference > 0).mean()) if used.any() else None
            ),
            "mean_excess_among_used": (
                float(selected_difference.mean()) if used.any() else None
            ),
            "p90_excess_among_used": (
                float(np.quantile(selected_difference, 0.90)) if used.any() else None
            ),
            "cluster_bootstrap_policy_minus_zero": bootstrap,
        }

    policies = {
        "conformal_q90_upper_below_zero": policy_report(predicted_upper < 0),
    }
    calibration_upper = (
        quantile_model.predict(oof_features[selector_calibration]) + conformal_correction
    )
    for coverage in (0.10, 0.25, 0.50):
        threshold = float(np.quantile(calibration_upper, coverage))
        item = policy_report(predicted_upper <= threshold)
        item["calibration_risk_threshold"] = threshold
        policies[f"calibration_target_{int(coverage * 100)}pct"] = item

    calibration_target = target[selector_calibration]
    calibration_coverage = float((calibration_target <= calibration_upper).mean())
    validation_quantile_coverage = float(
        (validation_target[eligible_validation] <= predicted_upper[eligible_validation]).mean()
    )
    harm_truth = validation_target[eligible_validation] > 0
    selector_metrics = {
        "harm_auc": float(
            roc_auc_score(harm_truth.astype(int), predicted_harm[eligible_validation])
        ),
        "harm_balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(
                harm_truth.astype(int),
                (predicted_harm[eligible_validation] >= 0.5).astype(int),
            )
        ),
        "mean_prediction_spearman": float(
            spearmanr(
                validation_target[eligible_validation], predicted_mean[eligible_validation]
            ).statistic
        ),
        "calibration_upper_coverage": calibration_coverage,
        "validation_upper_coverage": validation_quantile_coverage,
        "conformal_correction_nats": conformal_correction,
    }

    gate_cfg = config["go_gate"]
    primary = policies["conformal_q90_upper_below_zero"]
    primary_bootstrap = primary["cluster_bootstrap_policy_minus_zero"]
    go = bool(
        primary["history_coverage"] >= float(gate_cfg["min_history_coverage"])
        and primary_bootstrap["mean_excess_loss_ci_high"]
        <= float(gate_cfg["max_mean_excess_ci_high"])
        and always_history_harm - primary["harm_rate_all_eligible"]
        >= float(gate_cfg["min_harm_reduction_vs_always_history"])
    )
    return {
        "protocol": config["protocol"],
        "scope": config["scope"],
        "data": {
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "eligible_train": int(eligible_train.sum()),
            "eligible_validation": int(eligible_validation.sum()),
            "selector_fit_rows": int(len(selector_fit)),
            "selector_calibration_rows": int(len(selector_calibration)),
            "test_opened": False,
            "transcription_sha256": sha256_file(data_dir / "transcription.csv"),
            "labels_sha256": sha256_file(data_dir / "mm_label.npz"),
        },
        "selector_features": list(FEATURE_NAMES),
        "selector_metrics": selector_metrics,
        "baselines": {
            "always_zero": prediction_metrics(
                validation_y, zero_probability, int(config["ece_bins"])
            ),
            "always_history": {
                **prediction_metrics(
                    validation_y, actual_probability, int(config["ece_bins"])
                ),
                "harm_rate_eligible": always_history_harm,
                "mean_excess_loss_eligible": float(
                    validation_target[eligible_validation].mean()
                ),
            },
        },
        "policies": policies,
        "gates": {
            "point_utility_tail_risk_go": {
                "pass": go,
                "thresholds": gate_cfg,
            }
        },
        "config": config,
        "limitations": [
            "Text-only MVP; audio/video are unavailable.",
            "Cross-fitted base labels use one seed per fold; final validation base uses five seeds.",
            "Conformalized quantile coverage is marginal over calibration groups, not a formal per-speaker guarantee.",
            "Validation is used once for route triage; test is sealed.",
        ],
    }


def write_output(result: dict, output: Path) -> None:
    write_json_atomic(result, output.resolve())
