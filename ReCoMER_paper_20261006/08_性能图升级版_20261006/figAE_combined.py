# -*- coding: utf-8 -*-
"""合并图 AE（figure* 双栏版）：上排 = 主结果三面板（WF1/Acc/ECE），
下排 = 逐类 F1 vs 同协议复现基线。

数据源：
- 05_实验结果_汇总/ReCoMER_正式测试与消融_汇总/SUMMARY.json（full_test / calibration）
- subgroup_perclass_results.json（A_per_class，0-1 尺度）
- baselines_m3ed.json（models.*.per_class_mean/per_class_sd，0-100 百分数）
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
from style_shared import PALETTE, apply_style, headline, save, FORMAL

apply_style()
BASE = os.path.dirname(os.path.abspath(__file__))

# ================= 数据 =================
d = json.load(open(os.path.join(FORMAL, "SUMMARY.json"), encoding="utf-8"))
ft, cal = d["full_test"], d["calibration"]

R = json.load(open(os.path.join(BASE, "subgroup_perclass_results.json"), encoding="utf-8"))
B = json.load(open(os.path.join(BASE, "baselines_m3ed.json"), encoding="utf-8"))

ARMS = ["MHnoU", "AV", "TAV", "equal_weight", "ReCoMER", "cRBEF"]
SHORT = {"MHnoU": "MHnoU", "AV": "AV", "TAV": "TAV",
         "equal_weight": "Equal w.", "ReCoMER": "ReCoMER", "cRBEF": "cRBEF*"}

A = R["A_per_class"]
classes = A["class_names"]
support = A["per_class"]["ReCoMER"]["support"]
BASELINE_COLORS = {"MLP": "#BBBBBB", "DialogueRNN": "#888888"}
series = []
for name in ["MLP", "DialogueRNN"]:
    series.append((name, name, BASELINE_COLORS[name],
                   np.array(B["models"][name]["per_class_mean"]),
                   np.array(B["models"][name]["per_class_sd"])))
for arm, lab in [("MHnoU", "MHnoU branch"), ("cRBEF", "cRBEF (fixed)"),
                 ("equal_weight", "Equal-weight"), ("ReCoMER", "ReCoMER")]:
    series.append((arm, lab, PALETTE[arm],
                   np.array(A["per_class"][arm]["f1_mean"]) * 100,
                   np.array(A["per_class"][arm]["f1_sd"]) * 100))

# ================= 版式 =================
fig = plt.figure(figsize=(7.0, 4.6))
TOP_Y, TOP_H = 0.485, 0.35
XS = [0.055, 0.055 + 0.29 + 0.03, 0.055 + 2 * (0.29 + 0.03)]
axes_top = [fig.add_axes([x0, TOP_Y, 0.29, TOP_H]) for x0 in XS]
axE = fig.add_axes([0.055, 0.09, 0.92, 0.35])


def style_axis(ax, arms, title):
    ax.set_title(title, loc="left", fontweight="bold", fontsize=7.9, pad=2.5)
    ax.set_xticks(np.arange(len(arms)))
    ax.set_xticklabels([SHORT[a] for a in arms], fontsize=5.9,
                       rotation=18, ha="right")
    ax.grid(axis="y", zorder=0)
    ax.tick_params(axis="y", labelsize=6.7)


# ---- 上排面板 1：Weighted F1 (%) ----
means = np.array([ft[a]["weighted_f1"]["mean"] for a in ARMS]) * 100
sds = np.array([ft[a]["weighted_f1"]["sample_sd"] for a in ARMS]) * 100
bars = axes_top[0].bar(np.arange(len(ARMS)), means, yerr=sds, capsize=1.6,
                       color=[PALETTE[a] for a in ARMS], edgecolor="black",
                       linewidth=0.4, error_kw=dict(lw=0.55, ecolor="#555555"),
                       zorder=3)
bars[ARMS.index("ReCoMER")].set_linewidth(1.4)
style_axis(axes_top[0], ARMS, "Weighted F1 (%)  \u2191")
axes_top[0].set_ylim(0, (means + sds).max() * 1.15)
for xi, m, sd in zip(np.arange(len(ARMS)), means, sds):
    axes_top[0].annotate(f"{m:.1f}", (xi, m + sd), textcoords="offset points",
                         xytext=(0, 1.5), ha="center", fontsize=5.9, color="#333333")

# ---- 上排面板 2：Accuracy (%) ----
means = np.array([ft[a]["accuracy"]["mean"] for a in ARMS]) * 100
sds = np.array([ft[a]["accuracy"]["sample_sd"] for a in ARMS]) * 100
bars = axes_top[1].bar(np.arange(len(ARMS)), means, yerr=sds, capsize=1.6,
                       color=[PALETTE[a] for a in ARMS], edgecolor="black",
                       linewidth=0.4, error_kw=dict(lw=0.55, ecolor="#555555"),
                       zorder=3)
bars[ARMS.index("ReCoMER")].set_linewidth(1.4)
style_axis(axes_top[1], ARMS, "Accuracy (%)  \u2191")
axes_top[1].set_ylim(0, (means + sds).max() * 1.15)

# ---- 上排面板 3：ECE (15 bins)，仅 4 臂 ----
ECE_ARMS = ["MHnoU", "equal_weight", "ReCoMER", "cRBEF"]
vals = np.array([cal[a][0]["ECE15_equal_width"] for a in ECE_ARMS])
bars = axes_top[2].bar(np.arange(len(ECE_ARMS)), vals,
                       color=[PALETTE[a] for a in ECE_ARMS], edgecolor="black",
                       linewidth=0.4, zorder=3)
bars[ECE_ARMS.index("ReCoMER")].set_linewidth(1.4)
style_axis(axes_top[2], ECE_ARMS, "ECE (15 bins)  \u2193")
axes_top[2].set_ylim(0, vals.max() * 1.22)
for xi, v in zip(np.arange(len(ECE_ARMS)), vals):
    axes_top[2].annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                         xytext=(0, 1.5), ha="center", fontsize=5.9, color="#333333")

# ---- 下排：逐类 F1 vs 基线 ----
x = np.arange(len(classes))
nser = len(series)
w = 0.13
for i, (key, lab, color, m, sd) in enumerate(series):
    off = (i - (nser - 1) / 2.0) * w
    bars = axE.bar(x + off, m, width=w * 0.9, yerr=sd, capsize=1.4,
                   color=color, edgecolor="black", linewidth=0.35,
                   error_kw=dict(lw=0.5, ecolor="#555555"), label=lab, zorder=3)
    if key == "ReCoMER":
        for b in bars:
            b.set_linewidth(1.3)
    for xi, v, s in zip(x + off, m, sd):
        axE.annotate(f"{v:.1f}", (xi, v + s), textcoords="offset points",
                     xytext=(0, 1.2), fontsize=5.8, color="#666666",
                     ha="center", va="bottom", rotation=90, zorder=5)
axE.set_xticks(x)
axE.set_xticklabels([f"{c}\n(n={nn})" for c, nn in zip(classes, support)],
                    fontsize=6.8)
axE.set_ylabel("Per-class F1 (%)", fontsize=7.5)
axE.set_xlim(-0.6, len(classes) - 0.4)
axE.set_ylim(0, 92)
axE.tick_params(axis="y", labelsize=6.7)
axE.grid(axis="y", zorder=0)
axE.legend(ncol=3, frameon=False, loc="upper right", fontsize=6.2,
           handlelength=1.2, columnspacing=0.9, bbox_to_anchor=(1.0, 1.02))

# ================= 标题层 =================
headline(fig, "ReCoMER leads every fusion rule on accuracy and calibration, and exceeds the same-protocol\n"
              "reproduced baselines on every class; its WF1 gap to the fixed expert (\u22120.60 pp) is within seed noise.")
fig.text(0.012, 0.905, "Frozen M3ED formal test \u00b7 4,201 samples \u00b7 mean \u00b1 SD over 3 paired seeds "
                       "(cRBEF/TAV/AV fixed checkpoints; bold outline = ReCoMER; baselines: 3 seeds on identical features and split)",
         fontsize=7.0, color="#555555", ha="left", va="top")

save(fig, "figAE_combined")

# ---- 控制台核对 ----
print("WF1:", " ".join(f"{ft[a]['weighted_f1']['mean']*100:.1f}" for a in ARMS))
print("ECE ReCoMER:", cal["ReCoMER"][0]["ECE15_equal_width"])
rc = dict(zip(classes, A["per_class"]["ReCoMER"]["f1_mean"]))
print("ReCoMER Happy:", rc["Happy"] * 100, " Fear:", rc["Fear"] * 100)
