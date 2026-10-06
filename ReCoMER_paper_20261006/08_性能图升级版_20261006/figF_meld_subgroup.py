# -*- coding: utf-8 -*-
"""图 F：MELD 延续/转折子群分析（补"硬伤"清单第 5 项：4.8 节的 claim 此前无图表支撑）。

一句话结论：历史反馈对延续句有一致的小幅收益（3/3 seed 准确率提升），
对转折句基本中性——论文"转折句被历史拖垮"的表述应降级为"无收益"。

数据源：服务器 mechanisms/MELD/{43,47,59}/SAMPLE_DIAGNOSTICS.npz 的
        transition_group / history_on/off_logits -> subgroup_meld.json。
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
from style_shared import apply_style, headline, save

apply_style()

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "subgroup_meld.json"), encoding="utf-8"))
seeds = sorted(d.keys())
C_ON, C_OFF = "#0072B2", "#B8B8B8"

GROUPS = []
for gkey, gname in [("continuation", "Continuation utterances\n(same emotion as previous)"),
                    ("transition", "Transition utterances\n(emotion changed)")]:
    acc_on = np.array([d[s][gkey]["acc_history_on"] * 100 for s in seeds])
    acc_off = np.array([d[s][gkey]["acc_history_off"] * 100 for s in seeds])
    delta = acc_on - acc_off
    wins = int((delta > 0).sum())
    GROUPS.append((gkey, gname, acc_on, acc_off, delta, wins))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
for ax, (gkey, gname, acc_on, acc_off, delta, wins) in zip(axes, GROUPS):
    x = np.arange(len(seeds))
    ax.bar(x - 0.19, acc_off, width=0.36, color=C_OFF, edgecolor="black",
           linewidth=0.4, label="history off", zorder=3)
    ax.bar(x + 0.19, acc_on, width=0.36, color=C_ON, edgecolor="black",
           linewidth=0.4, label="history on", zorder=3)
    mean_txt = f"mean {delta.mean():+.2f} pts, {wins}/{len(seeds)} seeds improve"
    title_color = "#1B7837" if delta.mean() > 0 and wins == len(seeds) else "#333333"
    ax.set_title(f"{gname}\n{mean_txt}", loc="left", fontweight="bold",
                 fontsize=8.2, color=title_color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=7.5)
    base = min(acc_off.min(), acc_on.min()) - 1.4
    ax.set_ylim(base, max(acc_off.max(), acc_on.max()) + 1.8)
    ax.grid(axis="y", zorder=0)
    for xi, vn, dt in zip(x, acc_on, delta):
        ax.annotate(f"{dt:+.2f}", (xi + 0.19, vn), textcoords="offset points",
                    xytext=(0, 2.5), ha="center", fontsize=6.8,
                    color="#1B7837" if dt > 0 else "#8B8B8B", fontweight="bold")

axes[0].set_ylabel("Accuracy (%)")
axes[0].legend(loc="upper left", frameon=False, fontsize=7.2,
               bbox_to_anchor=(0.0, 1.02))

headline(fig, "History feedback consistently helps continuation utterances (+0.31 acc pts, 3/3 seeds)\n"
              "but is neutral on transitions (+0.05 acc pts) \u2014 a scope condition, not a failure mode.")
fig.text(0.012, 0.862, "MELD test protocol \u00b7 seeds 43/47/59 \u00b7 continuation n=976, transition n=1,353 \u00b7 "
                       "MHnoU branch with history on vs off",
         fontsize=7.2, color="#555555", ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.832])
save(fig, "figF_meld_subgroup")
