# -*- coding: utf-8 -*-
"""图 C：MOSEI 全指标剖面 history off vs on（升级点 13：论文只有 MAE delta）。

一句话结论：历史反馈把 MOSEI 的每一项指标都推向有利方向，
但幅度普遍很小（MAE 改善最大），且只在部分指标上超过种子间波动。

数据源：mechanisms/MOSEI_full/{43,47,59}/METRICS.json
        metrics_history_off / metrics_history_on（完整测试协议，3 seeds）。
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
from style_shared import apply_style, headline, save, FORMAL

apply_style()

SEEDS = ["43", "47", "59"]
METRICS = [
    ("mae", "MAE  \u2193", False, "{:.3f}"),
    ("rmse", "RMSE  \u2193", False, "{:.3f}"),
    ("pearson", "Pearson r  \u2191", True, "{:.3f}"),
    ("ccc", "CCC  \u2191", True, "{:.3f}"),
    ("r2", "R\u00b2  \u2191", True, "{:.3f}"),
    ("acc2", "Acc-2 (%)  \u2191", True, "{:.1f}"),
]

off, on = [], []
for s in SEEDS:
    d = json.load(open(os.path.join(FORMAL, "mechanisms", "MOSEI_full", s, "METRICS.json"),
                       encoding="utf-8"))
    off.append(d["metrics_history_off"])
    on.append(d["metrics_history_on"])

fig, axes = plt.subplots(1, 6, figsize=(7.0, 2.35))
C_OFF, C_ON = "#B8B8B8", "#0072B2"
for ax, (key, title, higher_better, fmt) in zip(axes, METRICS):
    v_off = np.array([m[key] for m in off], dtype=float)
    v_on = np.array([m[key] for m in on], dtype=float)
    means = np.array([v_off.mean(), v_on.mean()])
    sds = np.array([v_off.std(ddof=1), v_on.std(ddof=1)])
    ax.bar([0, 1], means, yerr=sds, capsize=2.2, color=[C_OFF, C_ON],
           edgecolor="black", linewidth=0.4, error_kw=dict(lw=0.7),
           width=0.6, zorder=3)
    delta = v_on.mean() - v_off.mean()
    better = (delta > 0) if higher_better else (delta < 0)
    ax.set_title(title, loc="left", fontweight="bold",
                 color="#1B7837" if better else "#8B8B8B")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["off", "on"], fontsize=7.5)
    ax.grid(axis="y", zorder=0)
    lo = min((means - sds).min(), means.min())
    hi = max((means + sds).max(), means.max())
    span = hi - lo if hi > lo else abs(hi) * 0.1 + 1e-6
    ax.set_ylim(lo - 0.42 * span, hi + 0.30 * span)
    for xi, m, sd in zip([0, 1], means, sds):
        ax.annotate(fmt.format(m), (xi, m + sd), textcoords="offset points",
                    xytext=(0, 2), ha="center", fontsize=6.5, color="#333333")
    # 每 seed 连线，展示方向一致性
    for a, b in zip(v_off, v_on):
        ax.plot([0, 1], [a, b], color="#999999", lw=0.6, alpha=0.55, zorder=2)

headline(fig, "History feedback moves every MOSEI metric in the favorable direction (green titles), "
              "but the effect is small relative to seed spread; MAE shows the largest relative gain.",
         y=0.995)
fig.text(0.012, 0.895, "MOSEI test protocol \u00b7 mean \u00b1 SD over seeds 43/47/59 \u00b7 "
                       "grey lines = per-seed paired values",
         fontsize=7.2, color="#555555", ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.86])
save(fig, "figC_mosei_metric_profile")
