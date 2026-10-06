# -*- coding: utf-8 -*-
"""图 E v2：正式 M3ED 测试逐类 F1，双对照显著性（等权融合 + cRBEF 专家）。

一句话结论：融合相对等权在 Sad 上显著改善（少数类救援）；相对专家是
"Happy/Fear 上显著更好、Sad 上显著更差"的再分配，而非全面占优。

数据源：服务器 subgroup_perclass_results.json 的 A_per_class
        （逐对话 bootstrap，协议对齐，1000 次重采样）。
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
from style_shared import PALETTE, apply_style, headline, save

apply_style()

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "subgroup_perclass_results.json"),
                   encoding="utf-8"))["A_per_class"]
CLASSES = d["class_names"]
support = [358, 1855, 734, 218, 736, 65, 235]
ARMS = ["MHnoU", "cRBEF", "equal_weight", "ReCoMER"]
LABELS = {"MHnoU": "MHnoU branch", "cRBEF": "cRBEF (fixed)",
          "equal_weight": "Equal-weight", "ReCoMER": "ReCoMER"}

fig, ax = plt.subplots(figsize=(7.0, 3.1))
x = np.arange(len(CLASSES))
w = 0.19
for i, a in enumerate(ARMS):
    mean = np.array(d["per_class"][a]["f1_mean"]) * 100
    sd = np.array(d["per_class"][a]["f1_sd"]) * 100
    bars = ax.bar(x + (i - 1.5) * w, mean, width=w * 0.92, yerr=sd,
                  capsize=1.8, color=PALETTE[a], edgecolor="black",
                  linewidth=0.4, error_kw=dict(lw=0.6),
                  label=LABELS[a], zorder=3)
    if a == "ReCoMER":
        for b in bars:
            b.set_linewidth(1.2)

reco_mean = np.array(d["per_class"]["ReCoMER"]["f1_mean"]) * 100
reco_sd = np.array(d["per_class"]["ReCoMER"]["f1_sd"]) * 100
boot_eq = d["bootstrap"]["ReCoMER_minus_equal_weight"]
boot_cr = d["bootstrap"]["ReCoMER_minus_cRBEF"]

for ci, xc in enumerate(x):
    ytop = reco_mean[ci] + reco_sd[ci]
    # vs 等权：绿色星号（CI 不含 0）
    lo, hi = boot_eq["ci_lo"][ci] * 100, boot_eq["ci_hi"][ci] * 100
    if lo > 0 or hi < 0:
        ax.annotate("*", (xc + 1.5 * w, ytop), textcoords="offset points",
                    xytext=(0, 3), fontsize=8.5, ha="center",
                    color="#1B7837", fontweight="bold")
    # vs cRBEF：上/下三角（差值为正=融合更好 ▲绿，为负 ▲红置于更高处）
    lo2, hi2 = boot_cr["ci_lo"][ci] * 100, boot_cr["ci_hi"][ci] * 100
    if lo2 > 0:
        ax.annotate("\u25b2", (xc + 1.5 * w, ytop), textcoords="offset points",
                    xytext=(-11, 3), fontsize=5.8, ha="center", color="#1B7837")
    elif hi2 < 0:
        ax.annotate("\u25bc", (xc + 1.5 * w, ytop), textcoords="offset points",
                    xytext=(-11, 3), fontsize=5.8, ha="center", color="#C0392B")

ax.annotate("* green star: ReCoMER vs equal-weight, bootstrap CI excludes 0\n"
            "\u25b2/\u25bc green/red: ReCoMER vs fixed expert, CI excludes 0",
            (0.995, 0.885), xycoords="axes fraction", ha="right",
            fontsize=6.4, color="#444444")

ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(CLASSES, support)],
                   fontsize=7.4)
ax.set_ylabel("F1 (%)")
ax.set_ylim(0, 86)
ax.grid(axis="y", zorder=0)
ax.legend(loc="lower left", frameon=False, ncol=4, fontsize=7.2,
          bbox_to_anchor=(0.0, 1.01), columnspacing=1.1,
          handletextpad=0.5, borderaxespad=0.0)

headline(fig, "Per class, the fusion redistributes rather than uniformly improves: significant gains over\n"
              "equal weight on Sad, and a trade against the fixed expert \u2014 better on Happy and Fear, worse on Sad.")
fig.text(0.012, 0.862, "Frozen M3ED formal test \u00b7 mean \u00b1 SD over seeds 43/47/59 (cRBEF fixed) \u00b7 "
                       "1,000 dialogue-level bootstrap resamples",
         fontsize=7.2, color="#555555", ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.832])
save(fig, "figE_perclass_m3ed")
