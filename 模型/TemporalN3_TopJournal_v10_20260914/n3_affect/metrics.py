"""Paper-grade metric suite for the v9 evaluation protocol.

This module is deliberately dependency-light (numpy only) so the same numbers
can be recomputed from a saved ``predictions.npz`` on any machine.

Classification
--------------
weighted_f1, macro_f1, accuracy, balanced_accuracy, per_class
{precision, recall, f1, support}, macro_precision, macro_recall, MCC,
confusion_matrix, top2_accuracy, top3_accuracy, nll, brier, ece plus the raw
reliability bins.

Regression
----------
mae, mse, rmse, pearson, spearman, ccc, r2 and Acc-2 / F1-2 under **four
explicit, documented binarisation protocols** so a table note can name the one
that is actually reported:

``gt0``
    positive iff ``y > 0``.  The official CH-SIMS v2 convention: a zero label
    is a NON-positive sample.
``ge0``
    positive iff ``y >= 0``.  The "has0" convention used by most CMU-MOSEI
    leaderboards (MMSA/MMIM lineage).
``non0``
    drop ``y == 0`` samples, then positive iff ``y > 0``.
``zero``
    positive iff ``y != 0``.  The MOSEI "non-zero" convention.

Every regression report carries all four so the choice is auditable instead of
implicit.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _as1d(x: Iterable[float] | np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    return a


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks with ties shared, matching scipy.stats.rankdata."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def pearson(y: np.ndarray, p: np.ndarray) -> float:
    y = _as1d(y)
    p = _as1d(p)
    if y.size < 2:
        return 0.0
    ys = y.std()
    ps = p.std()
    if ys < EPS or ps < EPS:
        return 0.0
    return float(np.corrcoef(p, y)[0, 1])


def spearman(y: np.ndarray, p: np.ndarray) -> float:
    y = _as1d(y)
    p = _as1d(p)
    if y.size < 2:
        return 0.0
    return pearson(_ranks(y), _ranks(p))


def ccc(y: np.ndarray, p: np.ndarray) -> float:
    """Concordance correlation coefficient (Lin, 1989).

    ``2 * rho * sy * sp / (sy^2 + sp^2 + (my - mp)^2)``
    """
    y = _as1d(y)
    p = _as1d(p)
    if y.size < 2:
        return 0.0
    my, mp = y.mean(), p.mean()
    vy, vp = y.var(), p.var()
    if vy < EPS or vp < EPS:
        return 0.0
    sy, sp = math.sqrt(vy), math.sqrt(vp)
    rho = pearson(y, p)
    denom = vy + vp + (my - mp) ** 2
    if denom < EPS:
        return 0.0
    return float(2.0 * rho * sy * sp / denom)


def r2_score(y: np.ndarray, p: np.ndarray) -> float:
    y = _as1d(y)
    p = _as1d(p)
    if y.size == 0:
        return 0.0
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot < EPS:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def binary_metrics(y: np.ndarray, p: np.ndarray, mode: str) -> dict[str, float]:
    """Acc-2 / F1-2 under an explicitly named binarisation protocol."""
    y = _as1d(y)
    p = _as1d(p)
    if mode == "gt0":
        pos_t, pos_p = y > 0, p > 0
    elif mode == "ge0":
        pos_t, pos_p = y >= 0, p >= 0
    elif mode == "non0":
        keep = y != 0
        y, p = y[keep], p[keep]
        pos_t, pos_p = y > 0, p > 0
    elif mode == "zero":
        pos_t, pos_p = y != 0, p != 0
    else:
        raise ValueError(f"unknown binarisation mode: {mode}")

    n = int(y.size)
    tp = int((pos_t & pos_p).sum())
    fp = int((~pos_t & pos_p).sum())
    fn = int((pos_t & ~pos_p).sum())
    tn = int((~pos_t & ~pos_p).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1_pos = 2 * prec * rec / max(prec + rec, EPS)
    prec_n = tn / max(tn + fn, 1)
    rec_n = tn / max(tn + fp, 1)
    f1_neg = 2 * prec_n * rec_n / max(prec_n + rec_n, EPS)
    out = {
        "n_considered": float(n),
        "acc2": float(acc * 100.0),
        "f1_2_positive": float(f1_pos * 100.0),
        "f1_2_macro": float(0.5 * (f1_pos + f1_neg) * 100.0),
        "precision_positive": float(prec),
        "recall_positive": float(rec),
        "positive_rate_true": float(pos_t.mean()) if n else 0.0,
        "positive_rate_pred": float(pos_p.mean()) if n else 0.0,
    }
    return out


def regression_report(y: np.ndarray, p: np.ndarray, *,
                      clip: tuple[float, float] | None = None,
                      threshold_modes: Sequence[str] = ("gt0", "ge0", "non0", "zero")) -> dict:
    """Full regression block.  ``clip`` selects the headline prediction scale."""
    y = _as1d(y)
    p_raw = _as1d(p)
    p = np.clip(p_raw, clip[0], clip[1]) if clip is not None else p_raw
    err = p - y
    rep: dict = {
        "n": int(y.size),
        "clip_applied": list(clip) if clip is not None else None,
        "mae": float(np.abs(err).mean()) if y.size else 0.0,
        "mse": float((err ** 2).mean()) if y.size else 0.0,
        "rmse": float(math.sqrt((err ** 2).mean())) if y.size else 0.0,
        "pearson": pearson(y, p),
        "spearman": spearman(y, p),
        "ccc": ccc(y, p),
        "r2": r2_score(y, p),
        "pred_mean": float(p.mean()) if y.size else 0.0,
        "pred_std": float(p.std()) if y.size else 0.0,
        "label_mean": float(y.mean()) if y.size else 0.0,
        "label_std": float(y.std()) if y.size else 0.0,
        "bias": float(err.mean()) if y.size else 0.0,
    }
    if clip is not None:
        rep["mae_unclipped"] = float(np.abs(p_raw - y).mean()) if y.size else 0.0
        rep["pearson_unclipped"] = pearson(y, p_raw)
        rep["ccc_unclipped"] = ccc(y, p_raw)
        rep["r2_unclipped"] = r2_score(y, p_raw)
    for mode in threshold_modes:
        rep[f"binary_{mode}"] = binary_metrics(y, p, mode)
    # error buckets, matching the "per emotion interval error analysis" ask
    if y.size:
        qs = np.quantile(y, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        buckets = []
        for lo, hi in zip(qs[:-1], qs[1:]):
            sel = (y >= lo) & (y <= hi) if hi == qs[-1] else (y >= lo) & (y < hi)
            if sel.sum() == 0:
                continue
            buckets.append({"y_lo": float(lo), "y_hi": float(hi), "n": int(sel.sum()),
                            "mae": float(np.abs(err[sel]).mean())})
        rep["error_by_label_quintile"] = buckets
    return rep


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def confusion_matrix(y: Sequence[int], p: Sequence[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, q in zip(y, p):
        ti, qi = int(t), int(q)
        if 0 <= ti < num_classes and 0 <= qi < num_classes:
            cm[ti, qi] += 1
    return cm


def per_class_prf(cm: np.ndarray) -> list[dict[str, float]]:
    k = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred = cm.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return [
        {"class_index": i, "precision": float(precision[i]), "recall": float(recall[i]),
         "f1": float(f1[i]), "support": int(support[i]), "predicted": int(pred[i])}
        for i in range(k)
    ]


def matthews_cc(cm: np.ndarray) -> float:
    """Gorodkin multiclass MCC (equals the binary MCC when K == 2)."""
    cm = cm.astype(np.float64)
    t_k = cm.sum(axis=1)
    p_k = cm.sum(axis=0)
    c = float(np.trace(cm))
    s = float(cm.sum())
    num = c * s - float(np.dot(p_k, t_k))
    den = math.sqrt(max((s * s - float(np.dot(p_k, p_k))) * (s * s - float(np.dot(t_k, t_k))), 0.0))
    return float(num / den) if den > EPS else 0.0


def expected_calibration_error(probs: np.ndarray, y: Sequence[int], n_bins: int = 15) -> dict:
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if probs.ndim != 2 or probs.shape[0] != y.size or y.size == 0:
        return {"ece": 0.0, "mce": 0.0, "bins": []}
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bins: list[dict] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if not sel.any():
            continue
        n_b = int(sel.sum())
        acc_b = float(correct[sel].mean())
        conf_b = float(conf[sel].mean())
        gap = abs(acc_b - conf_b)
        ece += n_b / y.size * gap
        mce = max(mce, gap)
        bins.append({"lo": float(lo), "hi": float(hi), "n": n_b,
                     "accuracy": acc_b, "confidence": conf_b, "gap": gap})
    return {"ece": float(ece), "mce": float(mce), "bins": bins}


def classification_report(logits: np.ndarray, y: Sequence[int], num_classes: int,
                          class_names: Sequence[str] | None = None,
                          n_bins: int = 15) -> dict:
    logits = np.asarray(logits, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
        logits = logits.reshape(-1, num_classes)
    if logits.ndim == 2 and logits.shape[1] != num_classes:
        raise ValueError(f"expected {num_classes} logit columns, got {logits.shape[1]}")
    if logits.shape[0] != y_arr.size:
        raise ValueError("logits/labels length mismatch")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.clip(exp.sum(axis=1, keepdims=True), EPS, None)
    preds = probs.argmax(axis=1)
    return classification_report_from_probs(probs, preds, y_arr, num_classes, class_names, n_bins)


def classification_report_from_probs(probs: np.ndarray, preds: np.ndarray, y: np.ndarray,
                                     num_classes: int,
                                     class_names: Sequence[str] | None = None,
                                     n_bins: int = 15) -> dict:
    probs = np.asarray(probs, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.int64).reshape(-1)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    cm = confusion_matrix(y, preds, num_classes)
    pc = per_class_prf(cm)
    names = list(class_names) if class_names is not None else [f"class_{i}" for i in range(num_classes)]
    if len(names) != num_classes:
        raise ValueError("class_names length must equal num_classes")
    for row, name in zip(pc, names):
        row["class_name"] = name
    f1 = np.array([r["f1"] for r in pc], dtype=np.float64)
    sup = np.array([r["support"] for r in pc], dtype=np.float64)
    rec = np.array([r["recall"] for r in pc], dtype=np.float64)
    pre = np.array([r["precision"] for r in pc], dtype=np.float64)
    acc = float(np.mean(y == preds)) if y.size else 0.0
    top2 = topk_accuracy(probs, y, 2)
    top3 = topk_accuracy(probs, y, 3)
    nll = float(-np.log(np.clip(probs[np.arange(y.size), np.clip(y, 0, num_classes - 1)], EPS, None)).mean()) if y.size else 0.0
    onehot = np.zeros_like(probs)
    if y.size:
        onehot[np.arange(y.size), np.clip(y, 0, num_classes - 1)] = 1.0
    brier = float(((probs - onehot) ** 2).mean()) if y.size else 0.0
    majority = float(sup.max() / sup.sum()) if sup.sum() else 0.0
    return {
        "n": int(y.size),
        "num_classes": int(num_classes),
        "class_names": names,
        "accuracy": float(acc),
        "balanced_accuracy": float(rec.mean()) if num_classes else 0.0,
        "macro_f1": float(f1.mean()) if num_classes else 0.0,
        "weighted_f1": float((f1 * sup).sum() / sup.sum()) if sup.sum() else 0.0,
        "macro_precision": float(pre.mean()) if num_classes else 0.0,
        "macro_recall": float(rec.mean()) if num_classes else 0.0,
        "mcc": matthews_cc(cm),
        "top2_accuracy": top2,
        "top3_accuracy": top3,
        "nll": nll,
        "brier": brier,
        "majority_class_rate": majority,
        "per_class": pc,
        "confusion_matrix": cm.tolist(),
        "calibration": expected_calibration_error(probs, y, n_bins=n_bins),
    }


def topk_accuracy(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    if y.size == 0:
        return 0.0
    k = min(k, probs.shape[1])
    top = np.argsort(-probs, axis=1)[:, :k]
    return float((top == y[:, None]).any(axis=1).mean())


# ---------------------------------------------------------------------------
# multi-seed aggregation
# ---------------------------------------------------------------------------
def aggregate(values: Sequence[float], *, confidence: float = 0.95) -> dict:
    """mean / std / 95% CI.  Uses the t distribution when scipy is present and a
    conservative normal approximation otherwise (documented in the report)."""
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None,
                "ci_method": "none"}
    n = a.size
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    method = "normal"
    tcrit = 1.959964
    if n > 1:
        try:  # prefer the exact t quantile
            from scipy import stats as _st  # type: ignore
            tcrit = float(_st.t.ppf(0.5 + confidence / 2.0, df=n - 1))
            method = "t"
        except Exception:
            # small-sample t table fallback
            table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
                     7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
            tcrit = table.get(n - 1, 1.959964)
            method = "t_table"
    return {"n": n, "mean": mean, "std": std, "sem": se,
            "ci95_low": mean - tcrit * se, "ci95_high": mean + tcrit * se,
            "ci_method": method, "ci_level": confidence, "values": [float(v) for v in a]}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, *, higher_is_better: bool,
                     n_boot: int = 10000, seed: int = 12345) -> dict:
    """Paired bootstrap over samples.  ``a`` = our model, ``b`` = baseline.

    Returns the mean paired difference (a - b) and a two-sided bootstrap CI for
    it.  A CI that excludes 0 is the claim we want to be able to make.
    """
    a = _as1d(a)
    b = _as1d(b)
    if a.size != b.size or a.size == 0:
        return {"n": 0, "mean_diff": None, "ci95_low": None, "ci95_high": None,
                "p_two_sided": None, "method": "paired_bootstrap"}
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    frac_le = float((boots <= 0).mean())
    frac_ge = float((boots >= 0).mean())
    p = 2.0 * min(frac_le, frac_ge)
    obs = float(d.mean())
    return {"n": int(d.size), "mean_diff": obs,
            "ci95_low": float(lo), "ci95_high": float(hi),
            "p_two_sided": float(min(1.0, p)),
            "fraction_bootstrap_le_zero": frac_le,
            "fraction_bootstrap_ge_zero": frac_ge,
            "excludes_zero": bool(lo > 0 or hi < 0),
            "direction_favours": ("model" if (obs > 0) == higher_is_better else "baseline"),
            "method": "paired_bootstrap", "n_boot": n_boot,
            "higher_is_better": bool(higher_is_better)}


def wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    a = _as1d(a)
    b = _as1d(b)
    if a.size != b.size or a.size == 0:
        return {"n": 0, "statistic": None, "p_two_sided": None, "method": "wilcoxon"}
    d = a - b
    nz = d[d != 0]
    if nz.size == 0:
        return {"n": 0, "statistic": 0.0, "p_two_sided": 1.0, "method": "wilcoxon",
                "note": "all paired differences are exactly zero"}
    try:
        from scipy import stats as _st  # type: ignore
        stat, p = _st.wilcoxon(d)
        return {"n": int(nz.size), "statistic": float(stat), "p_two_sided": float(p),
                "method": "wilcoxon_scipy"}
    except Exception:
        r = _ranks(np.abs(nz))
        w_plus = float(r[nz > 0].sum())
        n = nz.size
        mu = n * (n + 1) / 4.0
        sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0) if n > 0 else 0.0
        z = (w_plus - mu) / sd if sd > EPS else 0.0
        p = math.erfc(abs(z) / math.sqrt(2.0))
        return {"n": int(n), "statistic": float(w_plus), "p_two_sided": float(p),
                "method": "wilcoxon_normal_approx", "z": float(z)}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = _as1d(a)
    b = _as1d(b)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return 0.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    if pooled < EPS:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def holm_bonferroni(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down correction for a family of comparisons."""
    items = sorted(((k, v) for k, v in pvalues.items() if v is not None),
                   key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    rejected_so_far = True
    for i, (k, p) in enumerate(items):
        thresh = alpha / max(m - i, 1)
        if rejected_so_far and p <= thresh:
            rej = True
        else:
            rej = False
            rejected_so_far = False
        out[k] = {"p_raw": float(p), "threshold": float(thresh),
                  "p_adjusted": float(min(1.0, p * max(m - i, 1))), "reject_null": bool(rej)}
    return {"family_size": m, "alpha": alpha, "method": "holm_bonferroni", "results": out}
