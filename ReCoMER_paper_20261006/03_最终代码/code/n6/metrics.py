"""Paper-grade metric suite for model6 (numpy only, no sklearn).

Classification: accuracy, weighted_f1, macro_f1, balanced_accuracy, MCC,
per-class precision/recall/f1/support, macro precision/recall, confusion
matrix.

Regression: mae, rmse, pearson, spearman, ccc, r2 and Acc-2 / F1-2 under the
four explicitly named binarisation protocols used by the reference suite
(``gt0`` is the CH-SIMS official "sign agreement" convention and the headline
acc2/f12 reported here).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np

EPS = 1e-12

M3ED_CLASS_NAMES = ("Happy", "Neutral", "Sad", "Disgust", "Anger", "Fear",
                    "Surprise")


def _as1d(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


# ---------------------------------------------------------------------------
# regression pieces (formulas as in the reference metrics.py)
# ---------------------------------------------------------------------------
def pearson(y: np.ndarray, p: np.ndarray) -> float:
    y = _as1d(y)
    p = _as1d(p)
    if y.size < 2:
        return 0.0
    if y.std() < EPS or p.std() < EPS:
        return 0.0
    return float(np.corrcoef(p, y)[0, 1])


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and sx[j] == sx[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(y: np.ndarray, p: np.ndarray) -> float:
    y = _as1d(y)
    p = _as1d(p)
    if y.size < 2:
        return 0.0
    return pearson(_ranks(y), _ranks(p))


def ccc(y: np.ndarray, p: np.ndarray) -> float:
    y = _as1d(y)
    p = _as1d(p)
    if y.size < 2:
        return 0.0
    vy, vp = y.var(), p.var()
    if vy < EPS or vp < EPS:
        return 0.0
    my, mp = y.mean(), p.mean()
    rho = pearson(y, p)
    denom = vy + vp + (my - mp) ** 2
    if denom < EPS:
        return 0.0
    return float(2.0 * rho * math.sqrt(vy * vp) / denom)


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


def binary_metrics(y: np.ndarray, p: np.ndarray, mode: str) -> Dict[str, float]:
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
        raise ValueError("unknown binarisation mode: %s" % mode)
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
    return {
        "n": int(y.size),
        "acc2": float(acc * 100.0),
        "f1_2_positive": float(f1_pos * 100.0),
        "f1_2_macro": float(0.5 * (f1_pos + f1_neg) * 100.0),
        "precision_positive": float(prec),
        "recall_positive": float(rec),
    }


def regression_report(y, p, threshold_modes: Sequence[str] = ("gt0", "ge0", "non0", "zero")) -> Dict:
    y = _as1d(y)
    p = _as1d(p)
    err = p - y
    rep: Dict[str, object] = {
        "n": int(y.size),
        "mae": float(np.abs(err).mean()) if y.size else 0.0,
        "mse": float((err ** 2).mean()) if y.size else 0.0,
        "rmse": float(math.sqrt((err ** 2).mean())) if y.size else 0.0,
        "pearson": pearson(y, p),
        "spearman": spearman(y, p),
        "ccc": ccc(y, p),
        "r2": r2_score(y, p),
        "bias": float(err.mean()) if y.size else 0.0,
    }
    for mode in threshold_modes:
        rep["binary_%s" % mode] = binary_metrics(y, p, mode)
    gt0 = rep["binary_gt0"]
    rep["acc2"] = gt0["acc2"]
    rep["f12"] = gt0["f1_2_macro"]
    return rep


# ---------------------------------------------------------------------------
# classification pieces
# ---------------------------------------------------------------------------
def confusion_matrix(y: Sequence[int], p: Sequence[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, q in zip(y, p):
        ti, qi = int(t), int(q)
        if 0 <= ti < num_classes and 0 <= qi < num_classes:
            cm[ti, qi] += 1
    return cm


def per_class_prf(cm: np.ndarray) -> List[Dict[str, float]]:
    k = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred = cm.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return [
        {"class_index": i, "precision": float(precision[i]),
         "recall": float(recall[i]), "f1": float(f1[i]),
         "support": int(support[i]), "predicted": int(pred[i])}
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
    den = math.sqrt(max(
        (s * s - float(np.dot(p_k, p_k))) * (s * s - float(np.dot(t_k, t_k))), 0.0))
    return float(num / den) if den > EPS else 0.0


def classification_report(logits, y, num_classes: int,
                          class_names: Sequence[str] = M3ED_CLASS_NAMES) -> Dict:
    logits = np.asarray(logits, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.clip(exp.sum(axis=1, keepdims=True), EPS, None)
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(y_arr, preds, num_classes)
    pc = per_class_prf(cm)
    names = list(class_names)[:num_classes]
    for row, name in zip(pc, names):
        row["class_name"] = name
    f1 = np.array([r["f1"] for r in pc], dtype=np.float64)
    sup = np.array([r["support"] for r in pc], dtype=np.float64)
    rec = np.array([r["recall"] for r in pc], dtype=np.float64)
    pre = np.array([r["precision"] for r in pc], dtype=np.float64)
    acc = float(np.mean(y_arr == preds)) if y_arr.size else 0.0
    nll = float(-np.log(np.clip(
        probs[np.arange(y_arr.size), np.clip(y_arr, 0, num_classes - 1)],
        EPS, None)).mean()) if y_arr.size else 0.0
    return {
        "n": int(y_arr.size),
        "num_classes": int(num_classes),
        "accuracy": acc,
        "balanced_accuracy": float(rec.mean()) if num_classes else 0.0,
        "macro_f1": float(f1.mean()) if num_classes else 0.0,
        "weighted_f1": float((f1 * sup).sum() / sup.sum()) if sup.sum() else 0.0,
        "macro_precision": float(pre.mean()) if num_classes else 0.0,
        "macro_recall": float(rec.mean()) if num_classes else 0.0,
        "mcc": matthews_cc(cm),
        "nll": nll,
        "per_class": pc,
        "confusion_matrix": cm.tolist(),
    }


def per_group_weighted_f1(y, pred, groups, num_classes: int) -> Dict[str, float]:
    """Weighted F1 computed independently per group (e.g. per series).

    ``groups`` is a per-sample label (any hashable); returns {group: wf1}.
    Groups with no positive support anywhere yield 0.0.
    """
    y = np.asarray(y).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    g = np.asarray(groups).reshape(-1)
    out: Dict[str, float] = {}
    for s in sorted(set(g.tolist()), key=str):
        sel = g == s
        if int(sel.sum()) == 0:
            continue
        cm = confusion_matrix(y[sel], pred[sel], num_classes)
        pc = per_class_prf(cm)
        f1 = np.array([r["f1"] for r in pc], dtype=np.float64)
        sup = np.array([r["support"] for r in pc], dtype=np.float64)
        out[str(s)] = float((f1 * sup).sum() / sup.sum()) if sup.sum() else 0.0
    return out
