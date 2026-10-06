# -*- coding: utf-8 -*-
"""图 H 右（单栏版）：融合增益 vs 输入强度差，供 LaTeX 双栏插入。

数据源：transfer_iemocap.json / transfer_meld.json（mean_fusion、arms_test_expert）；
M3ED 点来自正文正式测试（MHnoU 45.43 / cRBEF 58.60 / ReCoMER 58.00）。
IEMOCAP 增益 +0.8 = 3-seed 汇总 ReCoMER_gate (73.73) − text 专家 (72.95)。
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from style_shared import SIG_OK, SIG_MARGINAL, apply_style, save

apply_style()
BASE = os.path.dirname(os.path.abspath(__file__))

fig = plt.figure(figsize=(3.3, 2.5))
ax = fig.add_axes([0.17, 0.15, 0.78, 0.66])

pts = [
    (1.40, 0.8, SIG_OK, "IEMOCAP", "+0.8", "all 3 seeds positive"),
    (1.10, 0.0, SIG_MARGINAL, "MELD", "+0.0", None),
    (13.17, -0.6, "#8B5A5A", "M3ED", "-0.6", None),
]
for gap, gain, color, name, val, note in pts:
    ax.scatter([gap], [gain], s=55, color=color, edgecolor="black",
               linewidth=0.6, zorder=4)
    dy = 9 if name == "IEMOCAP" else (-15 if name == "M3ED" else 9)
    ax.annotate(f"{name} {val}", (gap, gain), textcoords="offset points",
                xytext=(6, dy), fontsize=6.5, color=color, fontweight="bold",
                va="center")
    if note:
        ax.annotate(note, (gap, gain), textcoords="offset points",
                    xytext=(6, dy + 10), fontsize=5.8, color=color,
                    va="center", style="italic")

ax.axhline(0, color="#444444", lw=0.9, ls="--", zorder=2)
ax.set_title("When does fusion add value?", loc="left", fontweight="bold",
             fontsize=8.4)
ax.set_xlabel("input gap (pp)", fontsize=7)
ax.set_ylabel("fusion gain (pp)", fontsize=7)
ax.set_xlim(0, 15)
ax.set_ylim(-1.6, 2.2)
ax.tick_params(labelsize=7)
ax.grid(axis="y", zorder=0)

save(fig, "figH_right_scatter")
