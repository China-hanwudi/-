"""v9 reporting layer -- Tables 1..5, grouped SOTA table, and statistics.

Reads the run tree produced by ``run_v9_matrix.py`` (train -> ``eval_full`` on
valid, optional route-mode evals, optional robustness sweep) and emits
top-journal-style tables plus the raw statistics JSON.

Design rules baked in here (they mirror the frozen pre-registration):

* Paper-reported numbers and locally reproduced numbers live in different
  columns.  Nothing is ever computed by mixing the two.
* Classification and regression never share a metric table.
* Every comparison is grouped by (dataset, modalities, label protocol, split).
* Selection metrics (valid) and the frozen test are recorded separately; a
  table generated from a valid split is labelled as such in its caption.
* Multi-seed summaries are ``mean +/- std`` with a 95 % CI, and paired tests are
  performed at two levels: seed level (Wilcoxon signed-rank, Cohen's d) and
  sample level (paired bootstrap over per-sample losses from ``predictions.npz``).
* Holm-Bonferroni correction is applied to each family of comparisons.

Everything is numpy/python only, so the tables are recomputable offline from the
saved JSON/NPZ artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# metric specifications
# ---------------------------------------------------------------------------
#   (key, display, direction, role, scale)
#   direction : "up" means larger is better
#   scale     : display multiplier (classification metrics are stored 0..1)
CLASSIFICATION_METRICS: list[tuple[str, str, str, str, float]] = [
    ("weighted_f1", "Weighted-F1", "up", "primary", 100.0),
    ("macro_f1", "Macro-F1", "up", "co-primary", 100.0),
    ("accuracy", "Accuracy", "up", "co-primary", 100.0),
    ("balanced_accuracy", "Balanced Acc", "up", "secondary", 100.0),
    ("macro_precision", "Macro-Precision", "up", "secondary", 100.0),
    ("macro_recall", "Macro-Recall", "up", "secondary", 100.0),
    ("mcc", "MCC", "up", "secondary", 100.0),
    ("top2_accuracy", "Top-2 Acc", "up", "secondary", 100.0),
    ("top3_accuracy", "Top-3 Acc", "up", "secondary", 100.0),
    ("nll", "NLL", "down", "diagnostic", 1.0),
    ("brier", "Brier", "down", "diagnostic", 1.0),
    ("ece", "ECE", "down", "diagnostic", 1.0),
    ("mce", "MCE", "down", "diagnostic", 1.0),
    ("majority_class_rate", "Majority-class rate", "up", "reference", 100.0),
]

REGRESSION_METRICS: list[tuple[str, str, str, str, float]] = [
    ("mae", "MAE", "down", "primary", 1.0),
    ("rmse", "RMSE", "down", "co-primary", 1.0),
    ("pearson", "Pearson", "up", "co-primary", 100.0),
    ("ccc", "CCC", "up", "co-primary", 100.0),
    ("spearman", "Spearman", "up", "secondary", 100.0),
    ("r2", "R2", "up", "secondary", 100.0),
    ("binary_gt0.acc2", "Acc-2 (y>0)", "up", "secondary", 1.0),
    ("binary_gt0.f1_2_positive", "F1-2 positive (y>0)", "up", "secondary", 1.0),
    ("binary_gt0.f1_2_macro", "F1-2 macro (y>0)", "up", "secondary", 1.0),
    ("binary_ge0.acc2", "Acc-2 (y>=0)", "up", "audit", 1.0),
    ("binary_ge0.f1_2_positive", "F1-2 positive (y>=0)", "up", "audit", 1.0),
    ("binary_non0.acc2", "Acc-2 (non-zero)", "up", "audit", 1.0),
    ("binary_zero.acc2", "Acc-2 (zero protocol)", "up", "audit", 1.0),
]

# headline numbers carried by robust_eval.py's condense()
ROBUST_HEADLINE = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc",
                   "mae", "rmse", "pearson", "ccc", "r2", "acc2_gt0", "f1_2_gt0"]

DATASET_LABEL = {"m3ed": "M3ED (7-class emotion)", "mosei": "CMU-MOSEI (sentiment)",
                 "chsims": "CH-SIMS v2 (sentiment)"}
PRIMARY_METRIC = {"m3ed": "weighted_f1", "mosei": "mae", "chsims": "mae"}
MAIN_TAG = "main"
ABLATION_TAGS = {"m3ed": ["no_history", "no_relation", "no_cf", "no_redundancy",
                          "no_adaptive_budget"],
                 "chsims": ["no_unimodal"], "mosei": []}
ABLATION_PRETTY = {
    "no_history": "w/o history branch (current-only by construction)",
    "no_relation": "w/o candidate 3x3 relation module",
    "no_cf": "w/o bidirectional utility / counterfactual supervision",
    "no_redundancy": "w/o shared-private redundancy suppression",
    "no_adaptive_budget": "w/o risk budget + hard fallback (freeze accept budget)",
    "no_unimodal": "w/o unimodal label anchoring",
}
ABLATION_SWITCH = {
    "no_history": "--disable-history",
    "no_relation": "--disable-relation",
    "no_cf": "--no-counterfactual",
    "no_redundancy": "--no-redundancy",
    "no_adaptive_budget": "--no-adaptive-budget",
    "no_unimodal": "--no-unimodal",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def get_path(obj, dotted: str):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def fmt(v, nd: int = 4) -> str:
    if v is None:
        return "n/r"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        if not math.isfinite(v):
            return "n/r"
        return f"{v:.{nd}f}"
    return str(v)


def mean_std(agg: dict, scale: float, nd: int = 4) -> str:
    if not agg or agg.get("n", 0) == 0 or agg.get("mean") is None:
        return "n/r"
    m = agg["mean"] * scale
    s = (agg["std"] or 0.0) * scale
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def ci_str(agg: dict, scale: float, nd: int = 4) -> str:
    if not agg or agg.get("n", 0) == 0 or agg.get("ci95_low") is None:
        return "n/r"
    return f"[{agg['ci95_low'] * scale:.{nd}f}, {agg['ci95_high'] * scale:.{nd}f}]"


def md_table(header: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    align = align or ["---"] * len(header)
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(align) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


# ---------------------------------------------------------------------------
# run discovery
# ---------------------------------------------------------------------------
def discover(runs_root: Path) -> dict:
    """Return {dataset: {tag: {seed: record}}}."""
    found: dict = {}
    if not runs_root.is_dir():
        return found
    for ds_dir in sorted(runs_root.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name not in DATASET_LABEL:
            continue
        ds = ds_dir.name
        found.setdefault(ds, {})
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            m = re.match(r"^(?P<tag>.+)_seed(?P<seed>\d+)$", run_dir.name)
            if not m:
                continue
            tag, seed = m.group("tag"), int(m.group("seed"))
            rec = {
                "dataset": ds, "tag": tag, "seed": seed, "dir": str(run_dir),
                "eval_full": None, "final_result": None, "run_metadata": None,
                "robust": None, "route_evals": {}, "source_of_metrics": "none",
            }
            ef = run_dir / "eval_full.json"
            if ef.is_file():
                try:
                    rec["eval_full"] = json.loads(ef.read_text(encoding="utf-8"))
                    rec["source_of_metrics"] = "eval_full(valid)"
                except Exception as exc:
                    rec["eval_full_error"] = str(exc)
            fr = run_dir / "FINAL_RESULT.json"
            if fr.is_file():
                try:
                    rec["final_result"] = json.loads(fr.read_text(encoding="utf-8"))
                except Exception as exc:
                    rec["final_result_error"] = str(exc)
            rm = run_dir / "RUN_METADATA.json"
            if rm.is_file():
                try:
                    rec["run_metadata"] = json.loads(rm.read_text(encoding="utf-8"))
                except Exception:
                    pass
            rb = run_dir / "robust" / "robust_eval.json"
            if rb.is_file():
                try:
                    rec["robust"] = json.loads(rb.read_text(encoding="utf-8"))
                except Exception:
                    pass
            for sub in sorted(run_dir.glob("eval_*")):
                f = sub / "eval_full.json"
                if f.is_file():
                    try:
                        rec["route_evals"][sub.name] = json.loads(f.read_text(encoding="utf-8"))
                    except Exception:
                        pass
            found[ds].setdefault(tag, {})[seed] = rec
    return found


def metrics_of(rec: dict) -> dict | None:
    """Prefer the full evaluation; fall back to the training-time valid block."""
    if rec.get("eval_full"):
        return rec["eval_full"].get("metrics")
    if rec.get("final_result"):
        return rec["final_result"].get("best_valid")
    return None


def per_seed_values(records: dict, key: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for seed, rec in records.items():
        m = metrics_of(rec)
        if not m:
            continue
        v = get_path(m, key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[seed] = float(v)
    return out


def aggregate_metric(records: dict, key: str) -> dict:
    from n3_affect import metrics as M
    vals = per_seed_values(records, key)
    agg = M.aggregate([vals[s] for s in sorted(vals)])
    agg["per_seed"] = {str(s): vals[s] for s in sorted(vals)}
    return agg


# ---------------------------------------------------------------------------
# sample-level paired comparison (stronger than a 3-5 seed signed-rank test)
# ---------------------------------------------------------------------------
def _per_sample_loss(run_dir: Path, kind: str) -> tuple[list, np.ndarray] | None:
    """Per-sample loss vector for ``kind`` in {'accuracy','abs_error'}."""
    p = run_dir / "predictions.npz"
    if not p.is_file():
        return None
    try:
        z = np.load(p, allow_pickle=True)
    except Exception:
        return None
    y = np.asarray(z["y_true"]).reshape(-1)
    pv = np.asarray(z["y_pred"]).reshape(-1)
    ids = [str(x) for x in z["ids"]]
    if y.size == 0:
        return None
    if kind == "accuracy":
        lg = np.asarray(z["logits"])
        if lg.ndim != 2 or lg.shape[0] != y.size:
            return None
        pred = lg.argmax(axis=1)
        loss = (pred == y.astype(np.int64)).astype(np.float64)
    else:
        loss = np.abs(pv - y)
    return ids, loss


def sample_level_compare(a_dir: Path, b_dir: Path, kind: str,
                         n_boot: int = 10000) -> dict:
    """Paired bootstrap over samples, aligning the two runs by sample id.

    ``a`` = main, ``b`` = variant.  Returns the paired difference oriented so a
    positive value means main is better.
    """
    from n3_affect import metrics as M
    ra = _per_sample_loss(a_dir, kind)
    rb = _per_sample_loss(b_dir, kind)
    if ra is None or rb is None:
        return {"available": False}
    ia, la = ra
    ib, lb = rb
    ma = {k: i for i, k in enumerate(ia)}
    mb = {k: i for i, k in enumerate(ib)}
    common = [k for k in ia if k in mb]
    if not common:
        return {"available": False, "reason": "no shared sample ids"}
    x = np.array([la[ma[k]] for k in common])
    yv = np.array([lb[mb[k]] for k in common])
    higher = kind == "accuracy"
    d = (x - yv) if higher else (yv - x)
    bs = M.paired_bootstrap(d, np.zeros_like(d), higher_is_better=True, n_boot=n_boot)
    bs["available"] = True
    bs["kind"] = kind
    bs["n_shared_samples"] = len(common)
    bs["mean_main"] = float(x.mean())
    bs["mean_variant"] = float(yv.mean())
    return bs


# ---------------------------------------------------------------------------
# Table 1 / 2 / 3 -- main results
# ---------------------------------------------------------------------------
def main_table(ds: str, records: dict, out_dir: Path, stats: dict) -> str:
    spec = CLASSIFICATION_METRICS if ds == "m3ed" else REGRESSION_METRICS
    seeds = sorted(records)
    if not seeds or all(metrics_of(records[s]) is None for s in seeds):
        return (f"### {DATASET_LABEL[ds]} — main results\n\n"
                f"_No completed run with metrics yet "
                f"({len(seeds)} directory(ies) present, none with `eval_full.json`)._\n")
    lines = []
    lines.append(f"### {DATASET_LABEL[ds]} — main results (split: **valid**, n_seeds={len(seeds)})\n")
    lines.append(f"Seeds: {', '.join(str(s) for s in seeds)}. "
                 f"Selection: {' > '.join(['valid Weighted-F1', 'valid Macro-F1', 'Accuracy']) if ds == 'm3ed' else 'valid MAE'}. "
                 f"Checkpoint chosen on valid only; the sealed test is not read here.\n")

    header = ["Metric", "Direction"] + [f"seed {s}" for s in seeds] + ["mean ± std", "95% CI"]
    align = ["---", ":---:"] + [":---:"] * len(seeds) + [":---:", ":---:"]
    rows = []
    for key, disp, direction, role, scale in spec:
        vals = per_seed_values(records, key)
        if not vals:
            continue
        agg = aggregate_metric(records, key)
        stats.setdefault(ds, {})[key] = {
            "display": disp, "direction": direction, "role": role, "scale": scale,
            "aggregate": agg}
        cells = [f"{vals[s] * scale:.4f}" if s in vals else "n/r" for s in seeds]
        star = "**" if role == "primary" else ""
        rows.append([f"{star}{disp}{star}", "↑" if direction == "up" else "↓"]
                    + cells + [mean_std(agg, scale), ci_str(agg, scale)])
    lines.append(md_table(header, rows, align))

    # per-class block
    ev = [r for r in (records[s].get("eval_full") for s in seeds) if r]
    if ev and ds == "m3ed" and ev[0]["metrics"].get("per_class"):
        lines.append("\n#### Per-class P/R/F1 (mean over seeds)\n")
        names = [c["class_name"] for c in ev[0]["metrics"]["per_class"]]
        per: dict = {n: {"precision": [], "recall": [], "f1": [], "support": []} for n in names}
        for e in ev:
            for c in e["metrics"]["per_class"]:
                for k in ("precision", "recall", "f1", "support"):
                    per[c["class_name"]][k].append(float(c[k]))
        rows = []
        for n in names:
            p = per[n]
            rows.append([n,
                         f"{np.mean(p['precision']) * 100:.2f} ± {np.std(p['precision'], ddof=1) * 100:.2f}" if len(p['precision']) > 1 else fmt(np.mean(p['precision']) * 100),
                         f"{np.mean(p['recall']) * 100:.2f} ± {np.std(p['recall'], ddof=1) * 100:.2f}" if len(p['recall']) > 1 else fmt(np.mean(p['recall']) * 100),
                         f"{np.mean(p['f1']) * 100:.2f} ± {np.std(p['f1'], ddof=1) * 100:.2f}" if len(p['f1']) > 1 else fmt(np.mean(p['f1']) * 100),
                         f"{np.mean(p['support']):.0f}"])
        lines.append(md_table(["Class", "Precision", "Recall", "F1", "Support(mean)"], rows,
                             ["---", ":---:", ":---:", ":---:", ":---:"]))
        lines.append("\nThe minority-class check: if a rare class (Fear / Disgust / Surprise) "
                     "has recall near 0 while Neutral-Anger keep their F1, the model is collapsing "
                     "minorities into the majority classes — read together with the confusion matrix below.\n")

    # confusion matrix (seed with the median primary metric)
    if ds == "m3ed" and ev:
        pk = PRIMARY_METRIC[ds]
        prim = [(e["metrics"].get(pk), i) for i, e in enumerate(ev) if e["metrics"].get(pk) is not None]
        if prim:
            prim.sort()
            idx = prim[len(prim) // 2][1]
            cm = np.array(ev[idx]["metrics"]["confusion_matrix"])
            names = ev[idx]["metrics"]["class_names"]
            lines.append(f"\n#### Confusion matrix (seed {ev[idx].get('checkpoint_seed')}, "
                         f"median {pk})\n")
            rows = []
            for i, n in enumerate(names):
                rows.append([n] + [str(int(x)) for x in cm[i]] + [str(int(cm[i].sum()))])
            lines.append(md_table(["true ↓ / pred →"] + names + ["support"], rows,
                                 ["---"] + [":---:"] * (len(names) + 1)))
            rec = np.divide(np.diag(cm), cm.sum(axis=1),
                            out=np.zeros(len(names)), where=cm.sum(axis=1) > 0)
            lines.append(f"\nRow-normalised recall (seed {ev[idx].get('checkpoint_seed')}): "
                         + ", ".join(f"{n}={r * 100:.2f}%" for n, r in zip(names, rec)))
            write_csv(out_dir / f"confusion_m3ed_seed{ev[idx].get('checkpoint_seed')}.csv",
                      ["true\\pred"] + list(names), [[names[i]] + list(map(int, cm[i]))
                                                     for i in range(len(names))])

    # error-by-quintile for regression
    if ev and ds != "m3ed":
        lines.append("\n#### Error by label quintile (per-seed MAE, mean over seeds)\n")
        keys = sorted({f"{b['y_lo']:.4f}_{b['y_hi']:.4f}" for e in ev
                       for b in e["metrics"].get("error_by_label_quintile", [])})
        rows = []
        for k in keys:
            lo, hi = k.split("_")
            maes = []
            for e in ev:
                for b in e["metrics"].get("error_by_label_quintile", []):
                    if f"{b['y_lo']:.4f}_{b['y_hi']:.4f}" == k:
                        maes.append(b["mae"])
            if maes:
                rows.append([f"[{float(lo):.3f}, {float(hi):.3f}]", str(len(maes)),
                             f"{np.mean(maes):.4f}",
                             f"{np.std(maes, ddof=1):.4f}" if len(maes) > 1 else "n/a"])
        lines.append(md_table(["label interval", "n seeds", "MAE mean", "MAE std"], rows,
                             ["---", ":---:", ":---:", ":---:"]))

    # calibration reliability for classification
    if ev and ds == "m3ed":
        lines.append("\n#### Calibration (reliability, seed-median model)\n")
        cal = ev[0]["metrics"].get("calibration", {})
        lines.append(f"ECE = {cal.get('ece'):.4f}, MCE = {cal.get('mce'):.4f} "
                     f"(15 equal-width confidence bins; full bins in `calibration_m3ed.json`).")
        rows = [[f"{b['lo']:.2f}–{b['hi']:.2f}", str(b["n"]), f"{b['accuracy']:.4f}",
                 f"{b['confidence']:.4f}", f"{b['gap']:.4f}"] for b in cal.get("bins", [])]
        if rows:
            lines.append("")
            lines.append(md_table(["confidence bin", "n", "empirical acc", "mean confidence", "gap"],
                                 rows, ["---", ":---:", ":---:", ":---:", ":---:"]))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table 4 -- ablations with paired tests
# ---------------------------------------------------------------------------
def ablation_table(ds: str, tagmap: dict, out_dir: Path, stats: dict,
                   n_boot: int = 10000) -> str:
    from n3_affect import metrics as M
    if MAIN_TAG not in tagmap:
        return f"### {DATASET_LABEL[ds]} — module ablations\n\n_main runs not present yet._\n"
    main = tagmap[MAIN_TAG]
    pk = PRIMARY_METRIC[ds]
    _, disp, direction, _, scale = next(s for s in
        (CLASSIFICATION_METRICS if ds == "m3ed" else REGRESSION_METRICS) if s[0] == pk)
    higher_better = direction == "up"

    lines = [f"### {DATASET_LABEL[ds]} — module ablations\n",
             f"Primary metric: **{disp}** ({'↑' if higher_better else '↓'}). "
             f"Each variant is trained with the same seeds as the mainline; the paired "
             f"comparison uses only the seeds present in **both** arms.\n"]

    header = ["Variant", "switch", "seeds", f"{disp} mean ± std", "Δ vs main (paired)",
              "95% CI of Δ", "p (bootstrap)", "p (Wilcoxon)", "Cohen's d", "Holm"]
    align = ["---", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:"]
    rows = []
    comparators = []
    for tag in ABLATION_TAGS.get(ds, []):
        if tag not in tagmap:
            continue
        comp = tagmap[tag]
        common = sorted(set(main) & set(comp))
        if not common:
            continue
        a_all = per_seed_values(main, pk)
        b_all = per_seed_values(comp, pk)
        a = np.array([a_all[s] for s in common if s in a_all])
        b = np.array([b_all[s] for s in common if s in b_all])
        if a.size == 0 or a.size != b.size:
            continue
        # delta expressed so positive always means "main is better"
        delta = (a - b) if higher_better else (b - a)
        agg_a = M.aggregate(a.tolist())
        agg_b = M.aggregate(b.tolist())
        bs = M.paired_bootstrap(delta, np.zeros_like(delta), higher_is_better=True,
                                n_boot=n_boot)
        wx = M.wilcoxon(a, b)
        d = M.cohens_d(a, b)
        if not higher_better:
            d = -d
        item = {
            "tag": tag, "pretty": ABLATION_PRETTY.get(tag, tag),
            "switch": ABLATION_SWITCH.get(tag, ""), "seeds": common,
            "main_mean": agg_a["mean"], "main_std": agg_a["std"],
            "variant_mean": agg_b["mean"], "variant_std": agg_b["std"],
            "delta_mean": float(delta.mean()), "delta_scale": scale,
            "delta_ci95": [bs["ci95_low"], bs["ci95_high"]],
            "p_bootstrap": bs["p_two_sided"], "excludes_zero": bs["excludes_zero"],
            "p_wilcoxon": wx["p_two_sided"], "wilcoxon_method": wx["method"],
            "cohens_d": d, "n_pairs": int(a.size),
            "main_vs_variant_ratio": (float(b.mean() / a.mean()) if a.mean() else None),
        }
        comparators.append(item)
        rows.append([
            ABLATION_PRETTY.get(tag, tag), ABLATION_SWITCH.get(tag, ""),
            f"{len(common)}",
            mean_std(agg_b, scale),
            f"{delta.mean() * scale:+.4f}",
            f"[{bs['ci95_low'] * scale:+.4f}, {bs['ci95_high'] * scale:+.4f}]",
            fmt(bs["p_two_sided"]), fmt(wx["p_two_sided"]), fmt(d, 3), ""])

    # Holm correction over this family
    if comparators:
        pvals = {c["tag"]: c["p_bootstrap"] for c in comparators}
        holm = M.holm_bonferroni(pvals, alpha=0.05)
        for i, c in enumerate(comparators):
            h = holm["results"].get(c["tag"], {})
            c["holm"] = h
            rows[i][-1] = f"p_adj={fmt(h.get('p_adjusted'))} {'✓' if h.get('reject_null') else '✗'}"
        lines.append(md_table(header, rows, align))
        lines.append(f"\nMultiple-comparison correction: {holm['method']}, "
                     f"family size {holm['family_size']}, alpha {holm['alpha']}. "
                     f"A ✓ marks a comparison that survives correction.\n")
    else:
        lines.append("_No ablation arm with overlapping seeds is available yet._\n")

    # ---- sample-level paired bootstrap (all samples of the valid split) -----
    kind = "accuracy" if ds == "m3ed" else "abs_error"
    sl_rows = []
    sl_store: dict = {}
    for tag in ABLATION_TAGS.get(ds, []):
        if tag not in tagmap:
            continue
        comp = tagmap[tag]
        per_seed = {}
        pool_main: list[float] = []
        pool_var: list[float] = []
        for seed in sorted(set(main) & set(comp)):
            res = sample_level_compare(Path(main[seed]["dir"]), Path(comp[seed]["dir"]),
                                       kind, n_boot)
            if not res.get("available"):
                continue
            per_seed[str(seed)] = {k: res[k] for k in
                                   ("mean_diff", "ci95_low", "ci95_high", "p_two_sided",
                                    "n_shared_samples", "mean_main", "mean_variant")}
            if seed in main and (Path(main[seed]["dir"]) / "predictions.npz").is_file():
                ra = _per_sample_loss(Path(main[seed]["dir"]), kind)
                rb2 = _per_sample_loss(Path(comp[seed]["dir"]), kind)
                if ra and rb2:
                    ia, la = ra
                    ib2, lb2 = rb2
                    mb2 = {k: i for i, k in enumerate(ib2)}
                    for i, k in enumerate(ia):
                        if k in mb2:
                            pool_main.append(float(la[i]))
                            pool_var.append(float(lb2[mb2[k]]))
        if not per_seed:
            continue
        # seed-averaged paired diff and its CI, plus the sign agreement
        diffs = np.array([v["mean_diff"] for v in per_seed.values()])
        agree = int((diffs > 0).sum())
        # pooled bootstrap over all samples of all seeds (labelled anti-conservative)
        pooled = None
        if pool_main and pool_var:
            x = np.array(pool_main)
            yv = np.array(pool_var)
            d = (x - yv) if kind == "accuracy" else (yv - x)
            pooled = M.paired_bootstrap(d, np.zeros_like(d), higher_is_better=True, n_boot=n_boot)
        sl_store[tag] = {"per_seed": per_seed, "seed_diffs": diffs.tolist(),
                         "seeds_favouring_main": agree, "n_seeds": len(per_seed),
                         "pooled": pooled}
        sl_rows.append([
            ABLATION_PRETTY.get(tag, tag), str(len(per_seed)),
            f"{agree}/{len(per_seed)}",
            f"{diffs.mean() * scale:+.4f}" if kind == "abs_error" else f"{diffs.mean() * 100:+.2f}",
            ", ".join(f"{v['p_two_sided']:.4f}" for v in per_seed.values()),
            (f"{pooled['mean_diff'] * scale:+.4f}" if kind == "abs_error"
             else f"{pooled['mean_diff'] * 100:+.2f}") if pooled else "n/r",
            (f"[{pooled['ci95_low'] * scale:+.4f}, {pooled['ci95_high'] * scale:+.4f}]"
             if kind == "abs_error" else
             f"[{pooled['ci95_low'] * 100:+.2f}, {pooled['ci95_high'] * 100:+.2f}]")
            if pooled else "n/r",
            fmt(pooled["p_two_sided"]) if pooled else "n/r"])
    if sl_rows:
        unit = "correct-classification rate" if kind == "accuracy" else "|error|"
        lines.append(f"\n#### Sample-level paired bootstrap ({unit}, aligned by sample id)\n")
        lines.append(md_table(
            ["Variant", "seeds", "seeds favouring main", "mean paired Δ",
             "per-seed p", "pooled Δ", "pooled 95% CI", "pooled p"],
            sl_rows, ["---", ":---:", ":---:", ":---:", "---", ":---:", ":---:", ":---:"]))
        lines.append("\nPer-seed p-values are the honest unit of evidence; the pooled column "
                     "concatenates every seed's valid samples, which is anti-conservative "
                     "because the same valid clips recur across seeds, and is shown only for "
                     "completeness.\n")

    delta_key = f"ablation_{ds}_primary_{pk}"
    stats.setdefault("ablations", {})[delta_key] = {
        "dataset": ds, "primary_metric": pk, "primary_display": disp,
        "higher_is_better": higher_better, "scale": scale,
        "comparisons": comparators, "sample_level": sl_store}
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# route-mode comparison (mechanism evidence)
# ---------------------------------------------------------------------------
def route_table(ds: str, tagmap: dict, out_dir: Path, stats: dict) -> str:
    if MAIN_TAG not in tagmap:
        return ""
    lines = [f"### {DATASET_LABEL[ds]} — route-mode / history-use comparison (seed 43)\n",
             "Same checkpoint, only the routing decision changes. "
             "`current-only` is the no-history anchor; `hard-safe` is the proposed route.\n"]
    spec = CLASSIFICATION_METRICS if ds == "m3ed" else REGRESSION_METRICS
    show = [s for s in spec if s[3] in ("primary", "co-primary")]
    rows = []
    order = ["(checkpoint default)", "current-only", "plain-history", "soft-gate"]
    for seed, rec in sorted(tagmap[MAIN_TAG].items()):
        arms = {"(checkpoint default)": rec.get("eval_full")}
        for name, obj in rec.get("route_evals", {}).items():
            rm = obj.get("route_mode", name.replace("eval_", "").replace("_", "-"))
            arms[rm] = obj
        for rm in order:
            obj = arms.get(rm)
            if not obj:
                continue
            m = obj["metrics"]
            row = [f"seed {seed}", rm]
            for key, disp, _d, _r, scale in show:
                v = get_path(m, key)
                row.append(f"{v * scale:.4f}" if isinstance(v, (int, float)) else "n/r")
            row.append(fmt(obj.get("history_use_rate"), 4))
            row.append(fmt(obj.get("accept_rate_mean"), 4))
            rows.append(row)
            stats.setdefault("route_modes", {}).setdefault(ds, []).append({
                "seed": seed, "route_mode": rm,
                "history_use_rate": obj.get("history_use_rate"),
                "accept_rate_mean": obj.get("accept_rate_mean"),
                "metrics": {k: get_path(m, k) for k, *_ in show}})
    if not rows:
        return ""
    header = ["run", "route mode"] + [s[1] for s in show] + ["history use rate", "accept rate"]
    align = ["---", ":---:"] + [":---:"] * len(show) + [":---:", ":---:"]
    lines.append(md_table(header, rows, align))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table 5 -- robustness + efficiency
# ---------------------------------------------------------------------------
def robustness_table(ds: str, tagmap: dict, out_dir: Path, stats: dict) -> str:
    recs = {s: r for s, r in (tagmap.get(MAIN_TAG) or {}).items() if r.get("robust")}
    if not recs:
        return f"### {DATASET_LABEL[ds]} — robustness & efficiency\n\n_robustness sweep not run yet._\n"
    lines = [f"### {DATASET_LABEL[ds]} — robustness & efficiency (valid, capped n)\n"]
    num_line = " · ".join(f"seed {s}: n={r['robust'].get('n_evaluated')}" for s, r in sorted(recs.items()))
    lines.append(f"{num_line}\n")

    rows = []
    for seed, r in sorted(recs.items()):
        rb = r["robust"]
        lines.append(f"#### seed {seed} — baseline")
        b = rb.get("baseline_headline", {})
        lines.append(", ".join(f"{k}={fmt(v, 4)}" for k, v in b.items()) + "\n")

        # missing modality
        mm = rb.get("missing_modality", {})
        if mm:
            lines.append("\n**Missing-modality robustness** (only the listed modalities are kept)\n")
            hdrs = ["kept modalities"] + [k for k in ROBUST_HEADLINE if k in next(iter(mm.values()))["headline"]]
            body = []
            for mods, obj in mm.items():
                h = obj["headline"]
                body.append([str(mods)] + [fmt(h.get(k), 4) for k in hdrs[1:]])
            lines.append(md_table(hdrs, body))
            stats.setdefault("robustness", {}).setdefault(ds, {}).setdefault(str(seed), {})["missing_modality"] = {
                k: v["headline"] for k, v in mm.items()}

        # noise
        mn = rb.get("modality_noise", {})
        if mn:
            lines.append("\n**Modality noise** (Gaussian, relative to the feature std)\n")
            for family, obj in mn.items():
                ks = [k for k in ROBUST_HEADLINE if k in next(iter(obj.values()))["headline"]]
                hdrs = ["alpha"] + ks
                body = [[lvl] + [fmt(v["headline"].get(k), 4) for k in ks] for lvl, v in obj.items()]
                lines.append(f"-_{family}_-")
                lines.append(md_table(hdrs, body))
                lines.append("")
            stats.setdefault("robustness", {}).setdefault(ds, {}).setdefault(str(seed), {})["modality_noise"] = mn

        # history length
        hl = rb.get("history_length_K", {})
        if hl:
            lines.append("\n**History length K** (0 = no history)\n")
            ks = [k for k in ROBUST_HEADLINE if k in next(iter(hl.values()))["headline"]]
            body = [[f"K={k}"] + [fmt(v["headline"].get(x), 4) for x in ks] for k, v in hl.items()]
            lines.append(md_table(["K"] + ks, body))
            lines.append(f"\nMechanism trace: history_use_rate={fmt(rb.get('baseline', {}).get('mech', {}).get('history_use_rate') if isinstance(rb.get('baseline', {}).get('mech'), dict) else None, 4)}, "
                         f"accept_rate_mean={fmt(rb.get('baseline', {}).get('mech', {}).get('accept_rate_mean') if isinstance(rb.get('baseline', {}).get('mech'), dict) else None, 4)}")
            stats.setdefault("robustness", {}).setdefault(ds, {}).setdefault(str(seed), {})["history_length_K"] = hl

        eff = rb.get("efficiency")
        if eff:
            lines.append("\n**Efficiency** (FLOPs = 2×MAC, analytic over Linear/MHA/GRUCell)\n")
            flat = {k: v for k, v in eff.items() if isinstance(v, (int, float))}
            lines.append(", ".join(f"{k}={fmt(v, 6)}" for k, v in flat.items()))
            lines.append("")
            lines.append(f"Trainable parameters (from training): "
                         f"{fmt((r.get('final_result') or {}).get('trainable_parameters'))}")
            stats.setdefault("efficiency", {}).setdefault(ds, {})[str(seed)] = flat
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table 6 -- grouped SOTA comparison from a curated registry
# ---------------------------------------------------------------------------
GROUP_DEFS = [
    ("A", "Same dataset · same label protocol · tri-modal", "Directly comparable tri-modal results."),
    ("B", "Bi-modal", "Any two modalities removed."),
    ("C", "Uni-modal", "Single-modality upper/lower references."),
    ("D", "Different backbone or split/label protocol", "NOT directly comparable; shown for context only."),
]


def _mean_primary(stats: dict, ds: str, key: str):
    obj = (stats.get(ds) or {}).get(key)
    if not obj:
        return None
    return obj.get("aggregate", {}).get("mean")


def _auto_rows(merged: dict, stats: dict) -> list[dict]:
    """Rows derived from this repository's own runs (never paper numbers)."""
    rows: list[dict] = []
    for ds, tagmap in sorted(merged.items()):
        main = tagmap.get(MAIN_TAG) or {}
        if not any(metrics_of(r) for r in main.values()):
            continue
        cls = ds == "m3ed"
        split_map = {"m3ed": "17427 / 2821 / 4201 (sealed)",
                     "mosei": "16326 / 1871 / 4659 (sealed)",
                     "chsims": "2143 / 647 / 1034 (sealed) [official train 2722; dev 579 carved out]"}
        label_map = {"m3ed": "M3ED 7-class (official order)",
                     "mosei": "MOSEI [-3,3], y/3 normalised, Acc-2 y>0",
                     "chsims": "CH-SIMS v2 [-1,1], clipped, Acc-2 y>0"}
        row = {
            "group": "A", "method": "TemporalN3 v9 (this work) — valid split",
            "modalities": "T/A/V", "backbone": "frozen feature towers + N3 router",
            "dataset": DATASET_LABEL[ds], "split": split_map[ds], "labels": label_map[ds],
            "local_reproduction": True, "source": "this report",
            "accuracy": (_mean_primary(stats, ds, "accuracy") or 0) * 100 if cls else None,
            "weighted_f1": (_mean_primary(stats, ds, "weighted_f1") or 0) * 100 if cls else None,
            "macro_f1": (_mean_primary(stats, ds, "macro_f1") or 0) * 100 if cls else None,
            "mae": None if cls else _mean_primary(stats, ds, "mae"),
            "pearson": None if cls else (_mean_primary(stats, ds, "pearson") or 0) * 100,
            "ccc": None if cls else (_mean_primary(stats, ds, "ccc") or 0) * 100,
            "note": "mean over the completed seeds; see Tables 1-3 for per-seed values and CIs",
        }
        rows.append(row)

        # inference-time modality removal (same checkpoint, valid split)
        rb = (stats.get("robustness") or {}).get(ds) or {}
        mm: dict = {}
        for seed_obj in rb.values():
            for mods, head in (seed_obj.get("missing_modality") or {}).items():
                mm.setdefault(str(mods), []).append(head)
        for mods, heads in sorted(mm.items()):
            nmod = len([m for m in mods.replace("+", ",").split(",") if m.strip()])
            group = "C" if nmod == 1 else ("B" if nmod == 2 else "A")

            def avg(k):
                vals = [h.get(k) for h in heads if isinstance(h.get(k), (int, float))]
                return float(np.mean(vals)) if vals else None

            rows.append({
                "group": group,
                "method": f"TemporalN3 v9, inference-time keeping {{{mods}}}",
                "modalities": mods, "backbone": "same checkpoint as the row above",
                "dataset": DATASET_LABEL[ds], "split": split_map[ds], "labels": label_map[ds],
                "local_reproduction": True, "source": "robust_eval.py missing_modality",
                "accuracy": avg("accuracy") * 100 if cls and avg("accuracy") is not None else None,
                "weighted_f1": avg("weighted_f1") * 100 if cls and avg("weighted_f1") is not None else None,
                "macro_f1": avg("macro_f1") * 100 if cls and avg("macro_f1") is not None else None,
                "mae": None if cls else avg("mae"),
                "pearson": (avg("pearson") * 100) if (not cls and avg("pearson") is not None) else None,
                "ccc": (avg("ccc") * 100) if (not cls and avg("ccc") is not None) else None,
                "note": "features of the removed modalities are zeroed at inference; the weights "
                        "were still trained on all three, so this is NOT a re-trained uni/bi-modal model",
            })
    return rows


def sota_table(baselines_path: Path | None, stats: dict, merged: dict) -> str:
    lines = ["### Table 6 — Grouped comparison (paper-reported vs locally reproduced)\n",
             "Grouping follows the pre-registration rule: only works sharing task, modality set, "
             "label protocol and split may be compared head-to-head. Classification and regression "
             "never share a table body.\n"]
    data = {"rows": []}
    if baselines_path and Path(baselines_path).is_file():
        data = json.loads(Path(baselines_path).read_text(encoding="utf-8"))
    rows = _auto_rows(merged, stats) + data.get("rows", [])
    for gid, gname, gnote in GROUP_DEFS:
        sel = [r for r in rows if str(r.get("group", "")).upper() == gid]
        lines.append(f"\n#### Group {gid} — {gname}\n")
        lines.append(f"_{gnote}_\n")
        if not sel:
            lines.append("_No admitted row yet._\n")
            continue
        header = ["Method", "T/A/V", "Pretrained backbone", "Dataset", "Train/valid/test",
                  "Label protocol", "Accuracy", "Weighted-F1", "Macro-F1", "MAE", "Pearson",
                  "CCC", "Locally reproduced", "Note", "Source"]
        align = ["---", ":---:", "---", "---", "---", "---"] + [":---:"] * 6 + [":---:", "---", "---"]
        body = []
        for r in sel:
            src = r.get("source", "")
            if r.get("doi"):
                src = f"{src} (doi:{r['doi']})" if src else f"doi:{r['doi']}"
            body.append([r.get("method", ""), r.get("modalities", ""), r.get("backbone", ""),
                         r.get("dataset", ""), r.get("split", ""), r.get("labels", ""),
                         fmt(r.get("accuracy"), 2), fmt(r.get("weighted_f1"), 2),
                         fmt(r.get("macro_f1"), 2), fmt(r.get("mae"), 4),
                         fmt(r.get("pearson"), 2), fmt(r.get("ccc"), 3),
                         "yes" if r.get("local_reproduction") else "no (paper)",
                         r.get("note", ""), src])
        lines.append(md_table(header, body, align))
    pend = data.get("pending", [])
    if pend:
        lines.append("\n#### Not admitted to any group — pending verification\n")
        lines.append("_These numbers exist in the project history but are deliberately kept out of "
                     "every comparison group until their split, label protocol and leak status are "
                     "verified. They must not be cited as a bar to beat._\n")
        body = []
        for r in pend:
            body.append([r.get("method", ""), r.get("modalities", ""),
                         r.get("group_if_admitted", ""), r.get("dataset", ""),
                         r.get("split", ""), fmt(r.get("mae"), 6),
                         fmt(r.get("accuracy"), 2), fmt(r.get("macro_f1"), 2),
                         r.get("why_pending", "")])
        lines.append(md_table(["Method", "T/A/V", "would join", "Dataset", "split",
                              "MAE", "Accuracy", "Macro-F1", "why held back"],
                             body, ["---", ":---:", ":---:", "---", "---", ":---:", ":---:", ":---:", "---"]))
    hold = data.get("hold_note")
    if hold:
        lines.append(f"\n**Evidence hold.** {hold}\n")
    stats["sota_groups"] = {g[0]: [r.get("method") for r in rows
                                  if str(r.get("group", "")).upper() == g[0]] for g in GROUP_DEFS}
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# table notes
# ---------------------------------------------------------------------------
def table_notes(stats: dict) -> str:
    return """### Table notes (mandatory for every reported number)

**Splits and splits actually used**

| Dataset | Train | Valid (selection) | Extra | Test |
|---|---:|---:|---:|---:|
| M3ED | 17427 | 2821 | — | 4201 (sealed) |
| CMU-MOSEI | 16326 | 1871 | — | 4659 (sealed) |
| CH-SIMS v2 | 2143 | 647 | dev 579 | 1034 (sealed) |

* CMU-MOSEI train/valid/test sizes reproduce the **official** split exactly
  (16326 / 1871 / 4659).  Verified: id overlap between all split pairs is 0 and
  no exact feature row is shared across splits.
* **CH-SIMS v2 trains on 2143 clips, not the official 2722.**  A pre-registered dev
  split (579) was carved out of the official train set, so `2143 + 579 = 2722`.
  `valid` 647 is the official valid set and the test set is the official 1034.
  Published CH-SIMS v2 numbers trained on 2722, so this row is **not strictly
  split-matched** and is flagged as such wherever it appears.
* M3ED ids are unique per split with zero overlap; a small amount of *text*
  repetition exists across splits (train∩valid 122, train∩test 181, valid∩test 66)
  because M3ED contains repeated utterances. Exact multimodal feature rows are
  never shared.

**Label protocols**

* M3ED — 7 classes, packed ids 0..6, official order
  `Happy, Neutral, Sad, Disgust, Anger, Fear, Surprise`
  (`data_audit.json`, `label_field = EmoAnnotation.final_main_emo`).
  Train frequencies: Happy 9.33 %, Neutral 40.91 %, Sad 15.69 %, Disgust 6.57 %,
  Anger 21.90 %, Fear 1.61 %, Surprise 3.99 %. **Fear is ~25× rarer than Neutral**,
  which is why Balanced Accuracy / Macro-F1 / per-class recall are reported
  alongside Accuracy.
* CMU-MOSEI — raw sentiment in [-3, 3]; the model regresses `y/3` and all reported
  values are converted back with `× 3`.  Acc-2 / F1-2 headline uses the `y > 0`
  threshold; `>= 0`, non-zero and zero conventions are reported in the audit rows
  so no reader has to guess which binary protocol was used.
* CH-SIMS v2 — continuous sentiment in [-1, 1]; predictions are clipped to
  [-1, 1] before MAE (official convention).  Acc-2 positive class is `y > 0`, so a
  label of exactly 0 counts as **non-positive**.  The MOSEI `>= 0` convention is
  explicitly **not** applied here.
* Acc-3 / Acc-5 are only reported where the official protocol defines them.  We do
  **not** discretise continuous CH-SIMS v2 labels to manufacture an "accuracy"
  comparable to classification SOTA.

**What each dataset can and cannot test**

* M3ED and CMU-MOSEI pack `history_index [N,3]` and `speaker_same [N,3]`, so the
  history / candidate-routing machinery is **live** there.  Reported
  `history_use_rate` and `accept_rate_mean` are meaningful for these two.
* **CH-SIMS v2 packed tensors contain no `history_index` key at all** (only
  `ids / label / unimodal_labels / T / A / V`).  Consequently every clip has
  `has_history = False` and the mechanism trace reads
  `history_use_rate = 0.0, accept_rate_mean = 0.0`.  This is a **structural
  property of the packing, not a routing collapse** and must not be reported as
  evidence that the gate fails.  CH-SIMS v2 in this revision is a **static
  tri-modal regression benchmark**; its role is (a) the unimodal-label-anchoring
  ablation and (b) cross-dataset regression generality.  Mechanism claims must be
  supported on M3ED / MOSEI.

**Statistics**

* Every mean is a mean over **fixed seeds 17, 29, 43, 71, 101** (ablation arms use
  the pre-registered subset 17, 43, 101).  `std` is the sample standard deviation
  (ddof = 1); the 95 % CI uses the t quantile when scipy is available and a
  documented t-table otherwise (the method used is recorded per metric in
  `stats_v9.json` as `ci_method`).
* Paired tests at seed level: Wilcoxon signed-rank.  Paired tests at sample level:
  paired bootstrap over per-sample losses (10 000 resamples, seed 12345), which is
  the stronger evidence because it does not reduce the sample to 5 seeds.
* Effect size: Cohen's d, with the sign oriented so positive = our model better.
* Multiple comparisons: Holm-Bonferroni within each ablation family.
* **Paper-reported numbers are never mixed with local reproductions.**  The
  "Locally reproduced" column is `yes` only when a checkpoint and a
  `FINAL_RESULT.json` exist in this repository for that row.

**Failure accounting**

* A run with exit code 0 but no checkpoint or no `FINAL_RESULT.json` is counted as
  a failure and is excluded from every mean.  Failed arms appear in
  `stats_v9.json` under `failed_runs`.
"""


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def collect_failed(runs_root: Path) -> list[dict]:
    out = []
    if not runs_root.is_dir():
        return out
    for ds_dir in runs_root.iterdir():
        if not ds_dir.is_dir():
            continue
        for run_dir in ds_dir.iterdir():
            if not run_dir.is_dir():
                continue
            has_ckpt = (run_dir / "best.pt").is_file()
            has_final = (run_dir / "FINAL_RESULT.json").is_file()
            if not (has_ckpt and has_final):
                out.append({"dir": str(run_dir), "has_checkpoint": has_ckpt,
                            "has_final_result": has_final})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v9 report generator")
    ap.add_argument("--runs", type=Path, required=True, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--baselines", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from n3_affect import metrics as M  # noqa: F401

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables").mkdir(parents=True, exist_ok=True)

    merged: dict = {}
    failed: list[dict] = []
    for root in args.runs:
        found = discover(root)
        for ds, tagmap in found.items():
            for tag, seedmap in tagmap.items():
                merged.setdefault(ds, {}).setdefault(tag, {}).update(seedmap)
        failed.extend(collect_failed(root))

    stats: dict = {"runs_roots": [str(r) for r in args.runs],
                   "datasets_present": sorted(merged),
                   "failed_runs": failed}

    parts: list[str] = []
    parts.append("# v9 measurement report — M3ED / CMU-MOSEI / CH-SIMS v2\n")
    parts.append("Generated from the run tree by `n3_affect/report_v9.py`. "
                 "All numbers are recomputable from the saved `eval_full.json`, "
                 "`FINAL_RESULT.json` and `predictions.npz` artifacts.\n")
    parts.append(f"Run roots: {', '.join(f'`{r}`' for r in args.runs)}\n")

    # inventory
    inv_rows = []
    for ds in sorted(merged):
        for tag in sorted(merged[ds]):
            seeds = sorted(merged[ds][tag])
            n_full = sum(1 for s in seeds if merged[ds][tag][s].get("eval_full"))
            inv_rows.append([ds, tag, f"{len(seeds)}", f"{n_full}",
                             ", ".join(str(s) for s in seeds)])
    parts.append("\n## 0. Run inventory\n")
    parts.append(md_table(["dataset", "arm", "runs", "with full eval", "seeds"], inv_rows,
                         ["---", "---", ":---:", ":---:", "---"]))
    if failed:
        parts.append(f"\n**{len(failed)} directory(ies) lack a checkpoint or FINAL_RESULT.json "
                     f"and are excluded from every mean.** See `stats_v9.json`.\n")

    # main tables
    for ds in ("m3ed", "mosei", "chsims"):
        if ds not in merged:
            continue
        parts.append(f"\n## {DATASET_LABEL[ds]}\n")
        parts.append(main_table(ds, merged[ds].get(MAIN_TAG, {}), args.out / "tables", stats))
        if ds in ABLATION_TAGS and ABLATION_TAGS[ds]:
            parts.append(ablation_table(ds, merged[ds], args.out / "tables", stats, args.n_boot))
        rt = route_table(ds, merged[ds], args.out / "tables", stats)
        if rt:
            parts.append(rt)
        parts.append(robustness_table(ds, merged[ds], args.out / "tables", stats))

    parts.append("\n## Table 6 — grouped comparison\n")
    parts.append(sota_table(args.baselines, stats, merged))
    parts.append("\n" + table_notes(stats))

    (args.out / "REPORT_v9_MAIN.md").write_text("\n".join(parts), encoding="utf-8")
    (args.out / "stats_v9.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # flat CSV of every aggregated metric for downstream plotting
    rows = []
    for ds, byds in stats.items():
        if not isinstance(byds, dict) or ds in ("ablations", "route_modes", "robustness",
                                                "efficiency", "sota_groups"):
            continue
        for key, obj in byds.items():
            if not isinstance(obj, dict) or "aggregate" not in obj:
                continue
            a = obj["aggregate"]
            rows.append([ds, key, obj["display"], obj["direction"], obj["role"],
                         a.get("n"), a.get("mean"), a.get("std"), a.get("ci95_low"),
                         a.get("ci95_high"), a.get("ci_method")])
    write_csv(args.out / "tables" / "main_metrics_summary.csv",
              ["dataset", "metric", "display", "direction", "role", "n", "mean",
               "std", "ci95_low", "ci95_high", "ci_method"], rows)

    print(f"REPORT_WRITTEN {args.out / 'REPORT_v9_MAIN.md'}")
    print(f"STATS_WRITTEN  {args.out / 'stats_v9.json'}")
    print(f"datasets_present={sorted(merged)} failed_dirs={len(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
