# -*- coding: utf-8 -*-
"""汇总 paper_experiments_20261004 四数据集 × 5 臂 × 10 seeds 矩阵。
输出 four_dataset_matrix_summary.json：
- 每臂每数据集的 mean±sd（valid 主指标 + 辅助指标）
- 配对 seed 级差异（evidence 臂 vs uniform_nohist / uniform_history）：mean、95%CI、sign-flip p
仅读取 FINAL_RESULT.json，不触碰 test.pt。
"""
import json
import math
import numpy as np
from pathlib import Path

ROOT = Path("/root/autodl-tmp/paper_experiments_20261004/runs")
DS = [("M3ED_strong", "cls"), ("MELD", "cls"), ("IEMOCAP", "cls"), ("MOSEI_full", "reg")]
ARMS = ["uniform_nohist", "uniform_history", "evidence_closed", "evidence_solo", "evidence_mpath"]
SEEDS = [7, 13, 17, 23, 29, 37, 43, 53, 71, 101]


def load(ds, arm, seed):
    f = ROOT / f"{ds}__{arm}__s{seed}" / "FINAL_RESULT.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    assert d.get("test_read") is False, f"test_read true in {f}"
    return d["best_valid"]


def sign_flip_p(gains):
    """双侧符号置换检验（与既有汇总同风格）。"""
    g = np.array(gains, dtype=float)
    n = len(g)
    if n == 0 or np.all(g == 0):
        return 1.0
    k = int((g > 0).sum())
    # p = P(|wins - n/2| >= |k - n/2|) under Binomial(n, 0.5), two-sided
    from math import comb
    p = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n * 2
    return min(1.0, p)


def paired(a, b):
    """seed 级配对差异 a-b，含 95% CI 与符号检验（不假设正态）。"""
    gains = [x - y for x, y in zip(a, b)]
    arr = np.array(gains, dtype=float)
    lo, hi = np.percentile(arr, [2.5, 97.5]) if len(arr) >= 4 else (float("nan"),) * 2
    return {"mean": float(arr.mean()), "ci95": [float(lo), float(hi)],
            "wins": int((arr > 0).sum()), "n": len(arr),
            "sign_flip_p_two_sided": sign_flip_p(gains), "gains": gains}


summary = {}
for ds, task in DS:
    metric = "mae" if task == "reg" else "weighted_f1"
    arms_data = {}
    missing = []
    for arm in ARMS:
        vals, aux = [], []
        for s in SEEDS:
            bv = load(ds, arm, s)
            if bv is None:
                missing.append(f"{arm}/s{s}")
                continue
            vals.append(bv[metric])
            if task == "reg":
                aux.append({"mae": bv["mae"], "rmse": bv["rmse"], "pearson": bv["pearson"]})
            else:
                aux.append({"weighted_f1": bv["weighted_f1"], "accuracy": bv["accuracy"],
                            "macro_f1": bv["macro_f1"]})
        arms_data[arm] = {
            "mean": float(np.mean(vals)) if vals else None,
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals), "aux_means": {k: float(np.mean([a[k] for a in aux])) for k in aux[0]} if aux else {},
        }
    comp = {}
    for arm in ["uniform_history", "evidence_closed", "evidence_solo", "evidence_mpath"]:
        va = [load(ds, arm, s)[metric] for s in SEEDS if load(ds, arm, s)]
        vb0 = [load(ds, "uniform_nohist", s)[metric] for s in SEEDS if load(ds, "uniform_nohist", s)]
        vbh = [load(ds, "uniform_history", s)[metric] for s in SEEDS if load(ds, "uniform_history", s)]
        # 配对需同 seed 对齐
        pa = {s: load(ds, arm, s)[metric] for s in SEEDS if load(ds, arm, s)}
        p0 = {s: load(ds, "uniform_nohist", s)[metric] for s in SEEDS if load(ds, "uniform_nohist", s)}
        ph = {s: load(ds, "uniform_history", s)[metric] for s in SEEDS if load(ds, "uniform_history", s)}
        common0 = sorted(set(pa) & set(p0))
        commonh = sorted(set(pa) & set(ph))
        # 对 MAE，下降为改善 => 差异取 b - a（正=改善）；对 WF1 取 a - b
        if task == "reg":
            comp[arm + "_vs_B0"] = paired([p0[s] - pa[s] for s in common0], [0.0] * len(common0))
            comp[arm + "_vs_uniform_history"] = paired([ph[s] - pa[s] for s in commonh], [0.0] * len(commonh))
        else:
            comp[arm + "_vs_B0"] = paired([pa[s] for s in common0], [p0[s] for s in common0])
            comp[arm + "_vs_uniform_history"] = paired([pa[s] for s in commonh], [ph[s] for s in commonh])
    summary[ds] = {"task": task, "metric": metric, "seeds": SEEDS,
                   "arms": arms_data, "comparisons": comp,
                   "missing": missing}

out = {"scope": "four-dataset five-arm ten-seed matrix (paper_20261004 + fill_20261006)",
       "protocol": "valid-only, train/valid never test; uniform arms utility=uniform; "
                   "evidence arms shapley+evidence router bounded; B0 = uniform_nohist",
       "datasets": summary}
with open("/root/autodl-tmp/paper_experiments_20261004/four_dataset_matrix_summary.json", "w") as f:
    json.dump(out, f, indent=1)

# 控制台简表
for ds, task in DS:
    m = summary[ds]["metric"]
    print(f"== {ds} ({m}) ==")
    for arm in ARMS:
        a = summary[ds]["arms"][arm]
        print(f"  {arm:18s} {a['mean']:.4f} ± {a['sd']:.4f}  (n={a['n']})")
    for k, c in summary[ds]["comparisons"].items():
        print(f"  {k:36s} mean={c['mean']:+.4f} CI=[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] "
              f"wins={c['wins']}/{c['n']} p={c['sign_flip_p_two_sided']:.4f}")
print("missing:", {d: summary[d]["missing"] for d, _ in DS if summary[d]["missing"]})
