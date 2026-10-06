# -*- coding: utf-8 -*-
"""图 H 左（单栏版）：IEMOCAP / MELD 测试 WF1 柱状图，供 LaTeX 双栏插入。

数据源：transfer_iemocap.json / transfer_meld.json
（mean_fusion.<arm>.wf1/wf1_sd；arms_test_expert.<arm>.wf1 取最强专家臂 text）。
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
from style_shared import PALETTE, apply_style, save

apply_style()
BASE = os.path.dirname(os.path.abspath(__file__))

iem = json.load(open(os.path.join(BASE, "transfer_iemocap.json"), encoding="utf-8"))
mel = json.load(open(os.path.join(BASE, "transfer_meld.json"), encoding="utf-8"))

EXPERT_COLOR = "#D9D9D9"


def best_expert(rec):
    arms = rec["arms_test_expert"]
    name = max(arms, key=lambda k: arms[k]["wf1"])
    return name, arms[name]["wf1"] * 100


def fusion(rec, arm):
    mf = rec["mean_fusion"][arm]
    return mf["wf1"] * 100, mf["wf1_sd"] * 100


groups = [
    ("IEMOCAP \u00b7 4-class \u00b7 Qwen3 text",
     [fusion(iem, "MHnoU"), (best_expert(iem)[1], 0.0), fusion(iem, "equal_weight"),
      fusion(iem, "ReCoMER_gate")]),
    ("MELD \u00b7 7-class \u00b7 T=1024 text",
     [fusion(mel, "MHnoU"), (best_expert(mel)[1], 0.0), fusion(mel, "equal_weight"),
      fusion(mel, "ReCoMER_gate")]),
]
BAR_COLORS = [PALETTE["MHnoU"], EXPERT_COLOR, PALETTE["equal_weight"], PALETTE["ReCoMER"]]
BAR_LABELS = ["MHnoU", "Expert (text)", "Equal-weight", "ReCoMER gate"]

fig = plt.figure(figsize=(3.3, 2.5))
ax = fig.add_axes([0.11, 0.15, 0.87, 0.62])

centers = [0.0, 1.55]
w = 0.19
for gi, (note, vals) in enumerate(groups):
    c = centers[gi]
    for bi, (v, s) in enumerate(vals):
        bar = ax.bar(c + (bi - 1.5) * w, v, width=w * 0.9,
                     color=BAR_COLORS[bi], edgecolor="black", linewidth=0.35,
                     yerr=s if s > 0 else None, capsize=1.5,
                     error_kw=dict(lw=0.5, ecolor="#555555"),
                     label=BAR_LABELS[bi] if gi == 0 else None, zorder=3)
        if bi == 3:
            for b in bar:
                b.set_linewidth(1.2)
        ax.annotate(f"{v:.1f}", (c + (bi - 1.5) * w, v + s),
                    textcoords="offset points", xytext=(0, 1.5),
                    fontsize=6.0, color="#555555", ha="center", va="bottom",
                    rotation=90 if gi == 0 else 0, zorder=5)
    ax.annotate(note, (c, 1.03), xycoords=("data", "axes fraction"),
                fontsize=6.4, color="#555555", ha="center", va="bottom")

ax.set_title("Test WF1 (%), 3 seeds", loc="left", fontweight="bold",
             fontsize=8.4, pad=13)
ax.set_xticks(centers)
ax.set_xticklabels(["IEMOCAP", "MELD"], fontsize=7.5)
ax.set_xlim(-0.65, 2.2)
ax.set_ylim(60, 84)
ax.tick_params(axis="y", labelsize=7)
ax.grid(axis="y", zorder=0)
ax.legend(ncol=2, frameon=False, loc="upper left", fontsize=6.0,
          handlelength=1.1, columnspacing=0.8, bbox_to_anchor=(0.0, 1.03))

save(fig, "figH_left_bars")
for name, vals in groups:
    print(name, " | ".join(f"{v:.2f}" + (f"\u00b1{s:.2f}" if s else "") for v, s in vals))
