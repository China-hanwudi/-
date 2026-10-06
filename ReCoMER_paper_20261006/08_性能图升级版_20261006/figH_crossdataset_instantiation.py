# -*- coding: utf-8 -*-
"""图 H：跨数据集实例化双面板。

左面板：IEMOCAP / MELD 测试 WF1（mean ± SD over 3 seeds, fixed-α gate）。
右面板："when fusion works" —— 融合增益 vs 输入强度差（最强专家 − MHnoU 分支）。

数据源：transfer_iemocap.json / transfer_meld.json；M3ED 点来自正文正式测试
（MHnoU 45.43 / cRBEF 58.60 / ReCoMER 58.00）。

注意：右面板 IEMOCAP 增益 +0.8 pp = 3-seed 汇总 ReCoMER_gate (73.73) − text 专家
(72.95)；逐 seed 增益 +1.74 / +0.47 / +0.14，三 seed 均为正。
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
from style_shared import (PALETTE, SIG_OK, SIG_MARGINAL, apply_style, headline, save)

apply_style()
BASE = os.path.dirname(os.path.abspath(__file__))

iem = json.load(open(os.path.join(BASE, "transfer_iemocap.json"), encoding="utf-8"))
mel = json.load(open(os.path.join(BASE, "transfer_meld.json"), encoding="utf-8"))

EXPERT_COLOR = "#D9D9D9"   # 专家臂（text pack，单次评估，无 seed 方差）

def best_expert(rec):
    """测试集上最强的单专家臂及其 wf1。"""
    arms = rec["arms_test_expert"]
    name = max(arms, key=lambda k: arms[k]["wf1"])
    return name, arms[name]["wf1"] * 100

def fusion_bars(rec):
    mf = rec["mean_fusion"]
    return [(mf["MHnoU"]["wf1"] * 100, mf["MHnoU"]["wf1_sd"] * 100),
            (mf["equal_weight"]["wf1"] * 100, mf["equal_weight"]["wf1_sd"] * 100),
            (mf["ReCoMER_gate"]["wf1"] * 100, mf["ReCoMER_gate"]["wf1_sd"] * 100)]

iem_exp_name, iem_exp = best_expert(iem)     # text, 72.95
mel_exp_name, mel_exp = best_expert(mel)     # text, 66.16
iem_m, iem_eq, iem_rc = fusion_bars(iem)
mel_m, mel_eq, mel_rc = fusion_bars(mel)

fig = plt.figure(figsize=(7.0, 2.9))
axL = fig.add_axes([0.07, 0.17, 0.42, 0.52])
axR = fig.add_axes([0.585, 0.17, 0.385, 0.52])

# ---- 左面板：两组柱状（IEMOCAP / MELD） ----
groups = [
    ("IEMOCAP", "4-class \u00b7 Qwen3-2560 text pack",
     [iem_m, (iem_exp, 0.0), iem_eq, iem_rc]),
    ("MELD", "7-class \u00b7 T=1024 text pack",
     [mel_m, (mel_exp, 0.0), mel_eq, mel_rc]),
]
BAR_COLORS = [PALETTE["MHnoU"], EXPERT_COLOR, PALETTE["equal_weight"], PALETTE["ReCoMER"]]
BAR_LABELS = ["MHnoU branch", f"Expert ({iem_exp_name}, best arm)",
              "Equal-weight", "ReCoMER (fixed-\u03b1 gate)"]
centers = [0.0, 1.55]
w = 0.19
for gi, (name, note, vals) in enumerate(groups):
    c = centers[gi]
    for bi, (v, s) in enumerate(vals):
        bar = axL.bar(c + (bi - 1.5) * w, v, width=w * 0.9,
                      color=BAR_COLORS[bi], edgecolor="black", linewidth=0.35,
                      yerr=s if s > 0 else None, capsize=1.6,
                      error_kw=dict(lw=0.55, ecolor="#555555"),
                      label=BAR_LABELS[bi] if gi == 0 else None, zorder=3)
        if bi == 3:
            for b in bar:
                b.set_linewidth(1.3)
        axL.annotate(f"{v:.1f}", (c + (bi - 1.5) * w, v + s),
                     textcoords="offset points", xytext=(0, 1.5),
                     fontsize=6.5, color="#666666", ha="center", va="bottom",
                     zorder=5)
    axL.annotate(note, (c, 1.03), xycoords=("data", "axes fraction"),
                 fontsize=6.6, color="#555555", ha="center", va="bottom")

axL.set_title("Test WF1 (%) \u00b7 mean \u00b1 SD over 3 seeds", loc="left",
              fontweight="bold", pad=12)
axL.set_xticks(centers)
axL.set_xticklabels([g[0] for g in groups], fontsize=8)
axL.set_xlim(-0.65, 2.2)
axL.set_ylim(60, 84)
axL.grid(axis="y", zorder=0)
axL.legend(ncol=2, frameon=False, loc="upper left", fontsize=6.4,
           handlelength=1.2, columnspacing=0.9, bbox_to_anchor=(0.0, 1.04))

# ---- 右面板：增益 vs 输入强度差 ----
# (gap pp, gain pp, color, dataset, sig, extra note)
# IEMOCAP 增益 = 3-seed 汇总 ReCoMER_gate (73.73) − text 专家 (72.95) = +0.78 ≈ +0.8；
# 逐 seed 增益 +1.74 / +0.47 / +0.14，三 seed 均为正。
pts = [
    (1.4, 0.8, SIG_OK, "IEMOCAP", "sig", "all 3 seeds positive"),
    (1.1, 0.0, SIG_MARGINAL, "MELD", "ns", None),
    (13.2, -0.6, "#B23A2F", "M3ED", "ns", None),
]
for gap, gain, color, name, sig, note in pts:
    axR.scatter([gap], [gain], s=60, color=color, edgecolor="black",
                linewidth=0.6, zorder=4)
    dy = 8 if name != "M3ED" else -14
    axR.annotate(f"{name}  {gain:+.1f} pp ({sig})", (gap, gain),
                 textcoords="offset points", xytext=(7, dy),
                 fontsize=7.0, color=color, fontweight="bold", va="center")
    if note:
        axR.annotate(note, (gap, gain), textcoords="offset points",
                     xytext=(7, dy + 11), fontsize=6.2, color=color,
                     va="center", style="italic")
axR.axhline(0, color="#444444", lw=0.9, zorder=2)
axR.set_title("When does fusion add value?", loc="left", fontweight="bold")
axR.set_xlabel("input gap (pp): strongest expert \u2212 MHnoU branch", fontsize=7.6)
axR.set_ylabel("fusion gain (pp):\nReCoMER \u2212 expert", fontsize=7.6)
axR.set_xlim(0, 15.8)
axR.set_ylim(-2.4, 3.4)
axR.grid(axis="y", zorder=0)

headline(fig, "Instantiated on new datasets, the conservative gate fusion adds value when branch and\n"
              "expert are matched (IEMOCAP, +0.8 pp over 3 seeds), and degrades gracefully when they are not")
fig.text(0.012, 0.875, "Left: test WF1, mean \u00b1 SD over 3 seeds, fixed-\u03b1 gate (\u03b1 from valid) \u00b7 "
                       "Right: gain vs input-strength gap (branch vs strongest expert), same protocol \u00b7\n"
                       "M3ED point from the frozen formal test \u00b7 expert arm has no seed variance (single checkpoint)",
         fontsize=7.0, color="#555555", ha="left", va="top")
save(fig, "figH_crossdataset_instantiation")

# ---- 控制台核对 ----
print(f"IEMOCAP: MHnoU {iem_m[0]:.2f}±{iem_m[1]:.2f} | expert {iem_exp_name} {iem_exp:.2f} | "
      f"eq {iem_eq[0]:.2f}±{iem_eq[1]:.2f} | ReCoMER {iem_rc[0]:.2f}±{iem_rc[1]:.2f}")
print(f"MELD:    MHnoU {mel_m[0]:.2f}±{mel_m[1]:.2f} | expert {mel_exp_name} {mel_exp:.2f} | "
      f"eq {mel_eq[0]:.2f}±{mel_eq[1]:.2f} | ReCoMER {mel_rc[0]:.2f}±{mel_rc[1]:.2f}")
print(f"gaps: IEMOCAP {iem_exp - iem_m[0]:.2f} | MELD {mel_exp - mel_m[0]:.2f} | M3ED {58.60 - 45.43:.2f}")
print(f"gains(json): IEMOCAP {iem_rc[0] - iem_exp:+.2f} | MELD {mel_rc[0] - 65.74808551689778:+.2f} (vs crbef_gate) | M3ED {58.00 - 58.60:+.2f}")
