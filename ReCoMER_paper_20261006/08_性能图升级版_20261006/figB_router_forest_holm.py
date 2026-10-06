# -*- coding: utf-8 -*-
"""图 B：十 seed 证据路由的 Holm 校正森林图（升级点 14：论文只标 CI 星号）。

一句话结论：Shapley 路由在 MOSEI 上对所有对照显著且 10/10 全胜；
MELD / M3ED 的增益方向依赖对照选择，且部分在 Holm 校正后失去显著性。

数据源：main_crossdataset_summary.json（meld / m3ed）
        main_mosei_summary.json（mosei）
        comparisons[*].mean_gain / ci95 / wins / holm_p_current_complete_family
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
from style_shared import SIG_OK, SIG_MARGINAL, SIG_NS, apply_style, headline, save, RES

apply_style()

DS = [("mosei", "MOSEI (MAE reduction \u2191 better)", "main_mosei_summary.json"),
      ("meld", "MELD (WF1 gain \u2191 better)", "main_crossdataset_summary.json"),
      ("m3ed", "M3ED (WF1 gain \u2191 better)", "main_crossdataset_summary.json")]

rows = []  # (dataset_key, control_label, gain, lo, hi, wins, n, holm_p)
CTRL_LABEL = {"uniform": "vs uniform prior", "constant_task": "vs constant task prior",
              "mlp_task": "vs MLP task prior"}
for ds_key, ds_label, fname in DS:
    d = json.load(open(os.path.join(RES, fname), encoding="utf-8"))["datasets"][ds_key]
    for ctrl in ["uniform", "constant_task", "mlp_task"]:
        c = d["comparisons"][ctrl]
        g, (lo, hi) = c["mean_gain"], c["ci95"]
        rows.append((ds_key, ds_label, CTRL_LABEL[ctrl], g * 100, lo * 100, hi * 100,
                     c["wins"], c["n"], c["holm_p_current_complete_family"]))

fig, ax = plt.subplots(figsize=(7.0, 3.6))
ypos, ylabels, ycolors = [], [], []
group_first = {}
y = 0
last_ds = None
for ds_key, ds_label, ctrl, g, lo, hi, wins, n, p in rows:
    if last_ds is not None and ds_key != last_ds:
        y -= 0.6  # 数据集分组间隔
    color = SIG_OK if p < 0.05 else (SIG_MARGINAL if p < 0.10 else SIG_NS)
    ax.plot([lo, hi], [y, y], color=color, lw=2.2, solid_capstyle="round", zorder=3)
    ax.scatter([g], [y], s=34, color=color, zorder=4, edgecolor="black", linewidth=0.5)
    sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("\u2020" if p < 0.10 else "n.s."))
    ax.annotate(f"{g:+.2f} [{lo:+.2f}, {hi:+.2f}]   {wins}/{n} wins   Holm p = {p:.4f} {sig}",
                (hi, y), textcoords="offset points", xytext=(6, -2.5),
                fontsize=6.9, color=color, va="center")
    ypos.append(y); ylabels.append(ctrl); ycolors.append(color)
    group_first.setdefault(ds_key, y)  # 记录每组首行行位
    y -= 1
    last_ds = ds_key

ax.axvline(0, color="#444444", lw=0.9, zorder=2)
ax.set_yticks(ypos)
ax.set_yticklabels(ylabels, fontsize=7.6)
for tick, c in zip(ax.get_yticklabels(), ycolors):
    tick.set_color(c)
# 数据集分组标题：放在每组首行上方、绘图区左缘（左侧空白区）
for ds_key, ds_label, _ in DS:
    ax.annotate(ds_label, (-1.12, group_first[ds_key] + 0.42),
                fontsize=8.3, fontweight="bold", ha="left", va="bottom",
                color="#333333")
ax.set_xlabel("EvidenceRouter gain over control (pp, or MAE\u00d7100), paired per seed")
ax.set_xlim(-1.15, 1.8)
ax.set_ylim(min(ypos) - 0.7, max(ypos) + 0.85)
ax.grid(axis="x", zorder=0)

from matplotlib.lines import Line2D
legend_items = [
    Line2D([0], [0], color=SIG_OK, lw=2.2, label="significant after Holm (p < 0.05)"),
    Line2D([0], [0], color=SIG_MARGINAL, lw=2.2,
           label="marginal (0.05 \u2264 p < 0.10, \u2020)"),
    Line2D([0], [0], color=SIG_NS, lw=2.2, label="not significant (p \u2265 0.10)"),
]
fig.legend(handles=legend_items, loc="lower center", ncol=3, frameon=False,
           fontsize=6.9, bbox_to_anchor=(0.585, 0.008), columnspacing=1.4,
           handlelength=1.6)

headline(fig, "Shapley-anchored routing is significantly better than every control on MOSEI (10/10 seeds);\n"
              "on MELD and M3ED the gain depends on the control and partly loses significance after Holm correction.")
fig.text(0.012, 0.885, "Fixed-backbone ten-seed protocol (seeds 7\u2013101) \u00b7 paired seed-level 95% CI \u00b7 "
                       "Holm correction over the full comparison family",
         fontsize=7.2, color="#555555", ha="left")
fig.tight_layout(rect=[0.155, 0.075, 1, 0.87])
save(fig, "figB_router_forest_holm")
