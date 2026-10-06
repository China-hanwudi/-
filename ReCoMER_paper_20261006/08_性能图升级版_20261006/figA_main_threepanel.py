# -*- coding: utf-8 -*-
"""图 A（三面板版）：主结果 Weighted F1 / Accuracy / ECE。

一句话结论：ReCoMER 在 accuracy 与校准上领先所有融合规则；
WF1 上与固定专家的差距（−0.60 pp，CI [−1.39, +0.19]）在 seed 噪声范围内。

数据源：05_实验结果_汇总/ReCoMER_正式测试与消融_汇总/SUMMARY.json
（full_test.<arm>.<metric>.mean/sample_sd；calibration.<arm>[0].ECE15_equal_width）
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
from style_shared import PALETTE, apply_style, headline, save, FORMAL

apply_style()

d = json.load(open(os.path.join(FORMAL, "SUMMARY.json"), encoding="utf-8"))
ft = d["full_test"]
cal = d["calibration"]

ARMS = ["MHnoU", "AV", "TAV", "equal_weight", "ReCoMER", "cRBEF"]
SHORT = {"MHnoU": "MHnoU", "AV": "AV", "TAV": "TAV",
         "equal_weight": "Equal w.", "ReCoMER": "ReCoMER",
         "cRBEF": "cRBEF*"}

fig = plt.figure(figsize=(7.0, 2.3))
XS = [0.055, 0.055 + 0.29 + 0.035, 0.055 + 2 * (0.29 + 0.035)]
axes = [fig.add_axes([x0, 0.15, 0.29, 0.55]) for x0 in XS]


def style_axis(ax, arms, title):
    ax.set_title(title, loc="left", fontweight="bold", fontsize=8.4, pad=2.5)
    ax.set_xticks(np.arange(len(arms)))
    ax.set_xticklabels([SHORT[a] for a in arms], fontsize=6.3,
                       rotation=18, ha="right")
    ax.grid(axis="y", zorder=0)
    ax.tick_params(axis="y", labelsize=7.2)


# ---- 面板 1：Weighted F1 (%) ----
means = np.array([ft[a]["weighted_f1"]["mean"] for a in ARMS]) * 100
sds = np.array([ft[a]["weighted_f1"]["sample_sd"] for a in ARMS]) * 100
bars = axes[0].bar(np.arange(len(ARMS)), means, yerr=sds, capsize=1.8,
                   color=[PALETTE[a] for a in ARMS], edgecolor="black",
                   linewidth=0.4, error_kw=dict(lw=0.6, ecolor="#555555"),
                   zorder=3)
bars[ARMS.index("ReCoMER")].set_linewidth(1.4)
style_axis(axes[0], ARMS, "Weighted F1 (%)  \u2191")
axes[0].set_ylim(0, (means + sds).max() * 1.15)
for xi, m, sd in zip(np.arange(len(ARMS)), means, sds):
    axes[0].annotate(f"{m:.1f}", (xi, m + sd), textcoords="offset points",
                     xytext=(0, 2.0), ha="center", fontsize=6.4, color="#333333")

# ---- 面板 2：Accuracy (%) ----
means = np.array([ft[a]["accuracy"]["mean"] for a in ARMS]) * 100
sds = np.array([ft[a]["accuracy"]["sample_sd"] for a in ARMS]) * 100
bars = axes[1].bar(np.arange(len(ARMS)), means, yerr=sds, capsize=1.8,
                   color=[PALETTE[a] for a in ARMS], edgecolor="black",
                   linewidth=0.4, error_kw=dict(lw=0.6, ecolor="#555555"),
                   zorder=3)
bars[ARMS.index("ReCoMER")].set_linewidth(1.4)
style_axis(axes[1], ARMS, "Accuracy (%)  \u2191")
axes[1].set_ylim(0, (means + sds).max() * 1.15)
for xi, m, sd in zip(np.arange(len(ARMS)), means, sds):
    axes[1].annotate(f"{m:.1f}", (xi, m + sd), textcoords="offset points",
                     xytext=(0, 2.0), ha="center", fontsize=6.4, color="#333333")

# ---- 面板 3：ECE (15 bins)，仅 4 个融合级分支 ----
ECE_ARMS = ["MHnoU", "equal_weight", "ReCoMER", "cRBEF"]
vals = np.array([cal[a][0]["ECE15_equal_width"] for a in ECE_ARMS])
bars = axes[2].bar(np.arange(len(ECE_ARMS)), vals,
                   color=[PALETTE[a] for a in ECE_ARMS], edgecolor="black",
                   linewidth=0.4, zorder=3)
bars[ECE_ARMS.index("ReCoMER")].set_linewidth(1.4)
style_axis(axes[2], ECE_ARMS, "ECE (15 bins)  \u2193")
axes[2].set_ylim(0, vals.max() * 1.22)
for xi, v in zip(np.arange(len(ECE_ARMS)), vals):
    axes[2].annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                     xytext=(0, 2.0), ha="center", fontsize=6.4, color="#333333")

headline(fig, "ReCoMER leads every fusion rule on accuracy and calibration; its WF1 gap to the\n"
              "fixed expert (\u22120.60 pp, CI [\u22121.39, +0.19]) is within seed noise.")
fig.text(0.012, 0.80, "Frozen M3ED formal test \u00b7 4,201 samples \u00b7 mean \u00b1 SD over 3 paired seeds "
                      "(cRBEF/TAV/AV fixed checkpoints; bold outline = ReCoMER)",
         fontsize=7.0, color="#555555", ha="left", va="top")
save(fig, "figA_main_threepanel")

# ---- 控制台核对 ----
print("WF1:", " ".join(f"{ft[a]['weighted_f1']['mean']*100:.1f}" for a in ARMS))
print("Acc:", " ".join(f"{ft[a]['accuracy']['mean']*100:.1f}" for a in ARMS))
print("ECE:", " ".join(f"{cal[a][0]['ECE15_equal_width']:.4f}" for a in ECE_ARMS))
