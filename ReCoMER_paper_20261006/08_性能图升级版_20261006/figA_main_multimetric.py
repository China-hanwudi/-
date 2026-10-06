# -*- coding: utf-8 -*-
"""图 A：主结果多指标面板（升级点 12：主表只有 WF1 一个数）。

一句话结论：ReCoMER 在 accuracy / NLL / 校准上领先所有融合规则，
WF1 上与固定专家差距在噪声范围内。

数据源：SUMMARY.json full_test（六分支 WF1/macro-F1/Acc/NLL，3 paired seeds）
        + calibration（ECE15 / Brier，4 分支）。
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
PANELS = [("weighted_f1", "Weighted F1 (%)  \u2191", 100.0),
          ("macro_f1", "Macro F1 (%)  \u2191", 100.0),
          ("accuracy", "Accuracy (%)  \u2191", 100.0),
          ("nll", "NLL  \u2193", 1.0)]

fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0))
axes = axes.ravel()

for ax, (key, title, scale) in zip(axes, PANELS):
    means = np.array([ft[a][key]["mean"] * scale for a in ARMS])
    sds = np.array([ft[a][key]["sample_sd"] * scale for a in ARMS])
    x = np.arange(len(ARMS))
    bars = ax.bar(x, means, yerr=sds, capsize=2.2,
                  color=[PALETTE[a] for a in ARMS],
                  edgecolor="black", linewidth=0.4, error_kw=dict(lw=0.7),
                  zorder=3)
    bars[ARMS.index("ReCoMER")].set_linewidth(1.4)  # 高亮完整系统
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[a] for a in ARMS], fontsize=6.4,
                       rotation=18, ha="right")
    ax.grid(axis="y", zorder=0)
    span = (means + sds).max() - max(0, (means - sds).min())
    ax.set_ylim(0, (means + sds).max() + 0.16 * span)
    for xi, m, sd in zip(x, means, sds):
        v = f"{m:.1f}" if scale > 1 else f"{m:.2f}"
        ax.annotate(v, (xi, m + sd), textcoords="offset points",
                    xytext=(0, 2.5), ha="center", fontsize=6.6, color="#333333")

# 第 5、6 面板：校准（仅 4 个融合级分支有数据）
for ax, (key, ttl) in zip(axes[4:],
                          [("ECE15_equal_width", "ECE (15 bins)  \u2193"),
                           ("multiclass_Brier_sum", "Multiclass Brier  \u2193")]):
    arms = ["MHnoU", "equal_weight", "ReCoMER", "cRBEF"]
    vals = np.array([cal[a][0][key] for a in arms])
    x = np.arange(len(arms))
    bars = ax.bar(x, vals, color=[PALETTE[a] for a in arms],
                  edgecolor="black", linewidth=0.4, zorder=3)
    bars[arms.index("ReCoMER")].set_linewidth(1.4)
    ax.set_title(ttl, loc="left", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[a] for a in arms], fontsize=6.4,
                       rotation=18, ha="right")
    ax.grid(axis="y", zorder=0)
    ax.set_ylim(0, vals.max() * 1.24)
    for xi, v in zip(x, vals):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                    xytext=(0, 2.5), ha="center", fontsize=6.6, color="#333333")

headline(fig, "ReCoMER leads every fusion rule on accuracy, NLL and calibration on the frozen M3ED test;\n"
              "its WF1 gap to the fixed expert (\u22120.60 pp, CI [\u22121.39, +0.19]) is within seed noise.")
fig.text(0.012, 0.892, "Frozen M3ED formal test \u00b7 4,201 samples \u00b7 mean \u00b1 SD over 3 paired seeds "
                       "(cRBEF/TAV/AV checkpoints fixed; bold outline = ReCoMER; *fixed checkpoint)",
         fontsize=7.0, color="#555555", ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.872])
save(fig, "figA_main_multimetric")
