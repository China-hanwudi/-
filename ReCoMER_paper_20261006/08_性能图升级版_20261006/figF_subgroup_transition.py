# -*- coding: utf-8 -*-
"""图 F：延续 vs 转折子群分析（item 5：论文 4.8 的 claim 此前没有任何表/图支撑）。

一句话结论：历史反馈稳定提升延续句（3 数据集 3/3 种子为正）；
在转折句上 MELD/IEMOCAP 中性、M3ED 一致变差——"转折风险"是数据集依赖的，不是普遍规律。

数据源：subgroup_perclass_results.json B_subgroup
（mechanisms/*/SAMPLE_DIAGNOSTICS.npz 的 transition_group：0=label 延续，1=label 转折；
 deployed 预测在 history on/off 下的逐样本精度差，3 seeds 43/47/59）。
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
from style_shared import SIG_OK, SIG_NS, apply_style, headline, save

apply_style()

R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "subgroup_perclass_results.json"), encoding="utf-8"))
B = R["B_subgroup"]

DS = [("MELD", "MELD"), ("M3ED_textQwen", "M3ED (text-Qwen)"), ("IEMOCAP", "IEMOCAP")]

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), sharey=True)
plt.subplots_adjust(left=0.085, right=0.985, top=0.72, bottom=0.19, wspace=0.14)

for ax, (ds_key, ds_name) in zip(axes, DS):
    d = B[ds_key]
    means, sds, labels = [], [], []
    for gkey, glabel in [("0.0", "continuation"), ("1.0", "label flip")]:
        g = d[gkey]
        delta = (np.array(g["acc_history_on"]) - np.array(g["acc_history_off"])) * 100
        means.append(delta.mean())
        sds.append(delta.std(ddof=1))
        flip_rate = np.mean(g["history_flip_rate"]) * 100
        labels.append(f"{glabel}\nn={int(np.mean(g['n']))}\nflips {flip_rate:.1f}%")
    x = np.arange(2)
    colors = [SIG_OK if m > 0 else "#8B8B8B" for m in means]
    ax.bar(x, means, yerr=sds, capsize=2.6, color=colors, width=0.5,
           edgecolor="black", linewidth=0.4, error_kw=dict(lw=0.7), zorder=3)
    rng = np.random.default_rng(11)
    for gi, gkey in enumerate(["0.0", "1.0"]):
        g = d[gkey]
        delta = (np.array(g["acc_history_on"]) - np.array(g["acc_history_off"])) * 100
        ax.scatter(x[gi] + rng.uniform(-0.09, 0.09, size=len(delta)), delta,
                   s=15, color="#555555", alpha=0.6, zorder=4, edgecolor="none")
    ax.axhline(0, color="#444444", lw=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_title(ds_name, loc="left", fontweight="bold")
    ax.grid(axis="y", zorder=0)
    for xi, m in zip(x, means):
        ax.annotate(f"{m:+.2f} pp", (xi, m), textcoords="offset points",
                    xytext=(0, 9 if m >= 0 else -15), ha="center", fontsize=6.9,
                    fontweight="bold", color="#1B7837" if m > 0 else "#666666")

axes[0].set_ylabel("accuracy change (pp)\nhistory ON \u2212 history OFF")
axes[0].set_ylim(-1.0, 0.85)

headline(fig, "History feedback consistently helps continuation utterances (9/9 dataset\u00d7seed pairs positive);\n"
              "on emotional transitions it is neutral on MELD/IEMOCAP but consistently harmful on M3ED (3/3 seeds).",
         y=0.995)
fig.text(0.012, 0.845, "MHnoU deployed branch, test protocol \u00b7 mean \u00b1 SD over seeds 43/47/59 \u00b7 "
                       "grey dots = per-seed values \u00b7 \u201cflips\u201d = share of predictions changed when history is toggled",
         fontsize=7.0, color="#555555", ha="left")
save(fig, "figF_subgroup_transition")
