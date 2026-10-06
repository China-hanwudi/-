# -*- coding: utf-8 -*-
"""图 E：正式 M3ED 测试逐类 F1 + 配对 bootstrap 差异（补 item 12 残余：per-class）。

一句话结论：融合在类间重新分配误差——对 cRBEF 显著换来 Happy/Fear、显著失去 Sad；
对等权融合则显著提升 Sad。

数据源：subgroup_perclass_results.json（服务器 remote_analysis.py 生成；
bootstrap 以完整对话为重采样单位，与正式协议一致，1000 次）。
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
from style_shared import PALETTE, SIG_OK, SIG_NS, apply_style, headline, save

apply_style()

R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "subgroup_perclass_results.json"), encoding="utf-8"))
A = R["A_per_class"]
classes = A["class_names"]
support = A["per_class"]["ReCoMER"]["support"]
ARMS = ["MHnoU", "cRBEF", "equal_weight", "ReCoMER"]

fig = plt.figure(figsize=(7.0, 5.3))
ax1 = fig.add_axes([0.075, 0.545, 0.90, 0.315])
ax2 = fig.add_axes([0.075, 0.075, 0.90, 0.385])

# ---- 上：逐类 F1 分组柱状 ----
x = np.arange(len(classes))
w = 0.2
for i, a in enumerate(ARMS):
    m = np.array(A["per_class"][a]["f1_mean"]) * 100
    sd = np.array(A["per_class"][a]["f1_sd"]) * 100
    bars = ax1.bar(x + (i - 1.5) * w, m, width=w * 0.9, yerr=sd, capsize=1.8,
                   color=PALETTE[a], edgecolor="black", linewidth=0.35,
                   error_kw=dict(lw=0.6), label=a, zorder=3)
    if a == "ReCoMER":
        for b in bars:
            b.set_linewidth(1.2)
ax1.set_title("Per-class F1 (%) on the frozen M3ED formal test", loc="left",
              fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(classes, support)], fontsize=7.2)
ax1.set_ylabel("F1 (%)")
ax1.grid(axis="y", zorder=0)
ax1.legend(ncol=4, frameon=False, loc="upper right", fontsize=7.2)
ax1.set_ylim(0, 84)

# ---- 下：配对 bootstrap 逐类差异森林 ----
rows = []
for pair_key, pair_label in [("ReCoMER_minus_cRBEF", "ReCoMER \u2212 cRBEF"),
                             ("ReCoMER_minus_equal_weight", "ReCoMER \u2212 equal weight")]:
    b = A["bootstrap"][pair_key]
    for ci, cls in enumerate(classes):
        rows.append((pair_label, cls, b["diff_mean"][ci] * 100,
                     b["ci_lo"][ci] * 100, b["ci_hi"][ci] * 100))
ypos, ylabels, ycolors = [], [], []
y = 0
last_pair = None
pair_first = {}
for pair, cls, g, lo, hi in rows:
    if last_pair is not None and pair != last_pair:
        y -= 0.8
    sig = (lo > 0) or (hi < 0)
    color = SIG_OK if sig else SIG_NS
    ax2.plot([lo, hi], [y, y], color=color, lw=2.0, solid_capstyle="round", zorder=3)
    ax2.scatter([g], [y], s=26, color=color, zorder=4, edgecolor="black", linewidth=0.5)
    ax2.annotate(f"{g:+.1f} [{lo:+.1f}, {hi:+.1f}]" + (" *" if sig else ""),
                 (hi, y), textcoords="offset points", xytext=(5, -2.5),
                 fontsize=6.4, color=color, va="center")
    ypos.append(y); ylabels.append(cls); ycolors.append(color)
    pair_first.setdefault(pair, y)
    y -= 1
    last_pair = pair

ax2.axvline(0, color="#444444", lw=0.9, zorder=2)
ax2.set_yticks(ypos)
ax2.set_yticklabels(ylabels, fontsize=6.9)
for t, c in zip(ax2.get_yticklabels(), ycolors):
    t.set_color(c)
for pair, yf in pair_first.items():
    ax2.annotate(pair, (-32, yf + 0.38), fontsize=8.2, fontweight="bold",
                 ha="left", va="bottom", color="#333333", annotation_clip=False)
ax2.set_xlim(-32, 30)
ax2.set_ylim(min(ypos) - 0.7, max(ypos) + 0.95)
ax2.set_xlabel("per-class F1 difference (pp), dialogue bootstrap 95% CI (* = CI excludes 0)")
ax2.grid(axis="x", zorder=0)

headline(fig, "Fusion redistributes per-class errors: against cRBEF it significantly gains Happy/Fear but loses Sad;\n"
              "against equal-weight fusion it significantly improves Sad (+3.8 pp).")
fig.text(0.012, 0.902, "Frozen M3ED formal test \u00b7 mean \u00b1 SD over 3 paired seeds (cRBEF fixed checkpoint) \u00b7 "
                       "bootstrap on seed-43 predictions, whole-dialogue resampling (n=1,000)",
         fontsize=7.0, color="#555555", ha="left")
save(fig, "figE_perclass_f1_bootstrap")
