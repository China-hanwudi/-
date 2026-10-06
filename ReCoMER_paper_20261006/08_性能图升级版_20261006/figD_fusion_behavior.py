# -*- coding: utf-8 -*-
"""图 D：外层融合行为——校准强度 eta 与模态权重稳定性（升级点 15：融合行为不可见）。

一句话结论：学习到的融合是保守的——模态权重始终贴近均匀先验，
且正式测试 3 个种子中有 1 个被校准选为 eta=0（完全退回等权融合）。

数据源：SUMMARY.json full_test_etas（正式测试 3 种子的 eta）
        main_crossdataset_summary.json / main_mosei_summary.json 的
        pairs[*].evidence_mean_weight（十 seed 的 T/A/V 平均权重）。
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
from style_shared import apply_style, headline, save, RES, FORMAL

apply_style()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.85),
                               gridspec_kw={"width_ratios": [1, 2.1],
                                            "wspace": 0.30})

# ---- 左：正式测试的校准强度 eta ----
d = json.load(open(os.path.join(FORMAL, "SUMMARY.json"), encoding="utf-8"))
etas = d["full_test_etas"]                      # {"43": 1.0, "47": 0.0, "59": 1.0}
seeds = sorted(etas, key=int)
vals = [etas[s] for s in seeds]
ax1.bar(seeds, vals, color=["#0072B2" if v > 0 else "#8B8B8B" for v in vals],
        width=0.5, edgecolor="black", linewidth=0.4, zorder=3)
ax1.set_title("Calibrated strength \u03b7 (formal test)", loc="left",
              fontweight="bold", fontsize=8.6)
ax1.set_ylim(-0.08, 1.30)
ax1.set_yticks([0, 0.5, 1.0])
ax1.set_ylabel("\u03b7")
ax1.grid(axis="y", zorder=0)
for s, v in zip(seeds, vals):
    note = "full\ncorrection" if v > 0 else "falls back to\nequal weight"
    ax1.annotate(note, (s, v), textcoords="offset points",
                 xytext=(0, 7 if v > 0 else 11),
                 ha="center", fontsize=6.6,
                 color="#0072B2" if v > 0 else "#666666")

# ---- 右：十 seed 模态权重分布（3 数据集）----
DS = [("mosei", "MOSEI", "main_mosei_summary.json"),
      ("meld", "MELD", "main_crossdataset_summary.json"),
      ("m3ed", "M3ED", "main_crossdataset_summary.json")]
MOD = ["Text", "Audio", "Visual"]
C_MOD = ["#0072B2", "#E69F00", "#009E73"]
rng = np.random.default_rng(7)
for gi, (ds_key, ds_name, fname) in enumerate(DS):
    dj = json.load(open(os.path.join(RES, fname), encoding="utf-8"))["datasets"][ds_key]
    W = np.array([p["evidence_mean_weight"] for p in dj["pairs"]])  # [10, 3]
    base = gi * 1.0
    for mi in range(3):
        x = base + mi * 0.24 + rng.uniform(-0.03, 0.03, size=len(W))
        ax2.scatter(x, W[:, mi], s=13, color=C_MOD[mi], alpha=0.65,
                    edgecolor="none", zorder=3)
        ax2.plot([base + mi * 0.24 - 0.09, base + mi * 0.24 + 0.09],
                 [W[:, mi].mean()] * 2, color="black", lw=1.6, zorder=4)
    if gi < 2:
        ax2.axvline(base + 0.83, color="#CCCCCC", lw=0.6, ls="--", zorder=1)
ax2.axhline(1 / 3, color="#444444", lw=0.9, ls=":", zorder=2)
ax2.annotate("uniform prior 1/3", (0.02, 1 / 3), xycoords=("axes fraction", "data"),
             textcoords="offset points", xytext=(2, 3), fontsize=6.6, color="#444444")
ax2.set_xticks([gi * 1.0 + 0.24 for gi in range(3)])
ax2.set_xticklabels([n for _, n, _ in DS])
ax2.set_ylim(0.20, 0.46)
ax2.set_ylabel("learned mean modality weight")
ax2.set_title("EvidenceRouter weights (ten seeds)", loc="left",
              fontweight="bold", fontsize=8.6)
ax2.grid(axis="y", zorder=0)
handles = [plt.Line2D([0], [0], marker="o", ls="", color=c, markersize=5, label=m)
           for c, m in zip(C_MOD, MOD)]
ax2.legend(handles=handles, loc="upper center", frameon=False, ncol=3,
           bbox_to_anchor=(0.5, -0.15), columnspacing=1.1, handletextpad=0.25,
           fontsize=7.0)

headline(fig, "The learned outer fusion is conservative: \u03b7 calibration falls back to equal weight on 1 of 3\n"
              "formal-test seeds, and evidence weights stay close to the uniform prior on all datasets.")
fig.text(0.012, 0.862, "Left: frozen M3ED formal test (cRBEF fixed seed 17), \u03b7 = 0 means equal weight \u00b7 "
                       "Right: fixed-backbone ten-seed protocol, black bars = mean",
         fontsize=7.2, color="#555555", ha="left")
# tight_layout 与轴外图例冲突，改用手动边距
fig.subplots_adjust(left=0.065, right=0.995, top=0.77, bottom=0.21, wspace=0.30)
save(fig, "figD_fusion_behavior")
