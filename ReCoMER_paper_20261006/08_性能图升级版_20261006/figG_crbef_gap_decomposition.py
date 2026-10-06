# -*- coding: utf-8 -*-
"""图 G：cRBEF > ReCoMER 的归因分解图（修正版）。
面板 a：每 seed 四分枝 WF1（标 eta）；
面板 b：争议样本流向条 + 丢失样本类别构成；
面板 c：丢失样本上 MHnoU 错误置信度 vs cRBEF 正确类置信度分布。
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
from style_shared import PALETTE, apply_style, headline, save

apply_style()
BASE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(BASE, "crbef_vs_recomer_decomposition.json"), encoding="utf-8"))
SEEDS = ["43", "47", "59"]
CLASSES = ["Happy", "Neutral", "Sad", "Disgust", "Anger", "Fear", "Surprise"]

fig = plt.figure(figsize=(7.0, 3.2))
ax1 = fig.add_axes([0.065, 0.14, 0.29, 0.58])
ax2 = fig.add_axes([0.405, 0.55, 0.27, 0.17])
ax2b = fig.add_axes([0.405, 0.14, 0.27, 0.34])
ax3 = fig.add_axes([0.745, 0.14, 0.235, 0.58])

# ---- a：每 seed 四分枝 ----
x = np.arange(len(SEEDS))
w = 0.2
arm_keys = [("cRBEF", "cRBEF"), ("MHnoU", "MHnoU"), ("equal_weight", "Equal w."),
            ("probabilities", "ReCoMER")]
for i, (key, lab) in enumerate(arm_keys):
    vals = [R["seeds"][s][key] * 100 for s in SEEDS]
    bars = ax1.bar(x + (i - 1.5) * w, vals, width=w * 0.9,
                   color=PALETTE[key if key in PALETTE else "equal_weight"],
                   edgecolor="black", linewidth=0.35, label=lab, zorder=3)
    if key == "probabilities":
        for b in bars:
            b.set_linewidth(1.2)
ax1.set_title("Per-seed test WF1 (%)", loc="left", fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels([f"seed {s}\n\u03b7={R['seeds'][s]['eta']:g}" for s in SEEDS], fontsize=7.2)
ax1.set_ylim(41, 66)
ax1.grid(axis="y", zorder=0)
ax1.legend(fontsize=5.6, ncol=1, frameon=False, loc="upper left", handlelength=1.1, labelspacing=0.25)
for s_i, s in enumerate(SEEDS):
    gap = (R["seeds"][s]["probabilities"] - R["seeds"][s]["cRBEF"]) * 100
    ax1.annotate(f"{gap:+.2f}", (s_i + 1.5 * w, R["seeds"][s]["probabilities"] * 100),
                 textcoords="offset points", xytext=(2, 3), fontsize=6.6,
                 color="#B23A2F", fontweight="bold")

# ---- b：流向条 + 丢失类别 ----
f = R["pooled"]["flow"]
lost = R["pooled"]["lost_by_class"]
total = f["crbef_ok_mh_bad"]
kept_rc = f["crbef_ok_mh_bad_rc_kept"]
lost_n = total - kept_rc

ax2.barh([0.5], [kept_rc], color="#1B7837", edgecolor="black", lw=0.4, height=0.55, zorder=3)
ax2.barh([0.5], [lost_n], left=kept_rc, color="#B23A2F", edgecolor="black", lw=0.4,
         height=0.55, zorder=3)
ax2.annotate(f"kept {kept_rc:.0f} ({kept_rc/total:.0%})", (kept_rc/2, 0.5), ha="center",
             va="center", fontsize=6.6, color="white", fontweight="bold")
ax2.annotate(f"lost {lost_n:.0f} ({lost_n/total:.0%})", (kept_rc+lost_n/2, 0.5),
             ha="center", va="center", fontsize=6.6, color="white", fontweight="bold")
ax2.annotate(f"cRBEF \u2713 \u2227 MHnoU \u2717 (n={total:.0f})",
             (0, 1.28), fontsize=7.2, fontweight="bold", color="#333333", va="center")
ax2.set_yticks([])
ax2.set_xlim(0, total * 1.02)
ax2.set_ylim(-0.15, 1.45)
ax2.set_xticks([])
for sp in ["left", "right", "top"]:
    ax2.spines[sp].set_visible(False)

cats = ["Sad", "Anger", "Neutral", "Surprise", "Disgust", "Happy", "Fear"]
vals = np.array([lost[c] for c in cats])
cols = ["#B23A2F" if c in ("Sad", "Anger") else "#B9B9B4" for c in cats]
ax2b.barh(np.arange(len(cats))[::-1], vals, color=cols, edgecolor="black", lw=0.3,
          height=0.6, zorder=3)
for i, (c, v) in enumerate(zip(cats, vals)):
    ax2b.annotate(f"{c} {v:.0f}", (v + 4, len(cats) - 1 - i), fontsize=6.4, va="center",
                  color="#333333")
ax2b.set_yticks([])
ax2b.set_xlim(0, vals.max() * 1.5)
ax2b.set_title("lost samples by true class", loc="left", fontweight="bold", fontsize=8.2)
ax2b.set_xlabel("samples", fontsize=7)

# ---- c：置信度分布 ----
mh_lost, cr_lost = [], []
for s in SEEDS:
    d = np.load(os.path.join(BASE, "full_test_local", s, "PREDICTIONS.npz"))
    yt = d["y_true"]
    arms = {k: d[k] for k in ["MHnoU", "cRBEF", "probabilities"]}
    cr, mh, rc = arms["cRBEF"].argmax(1), arms["MHnoU"].argmax(1), arms["probabilities"].argmax(1)
    lost_mask = (cr == yt) & (mh != yt) & (rc != yt)
    mh_lost += list(arms["MHnoU"][lost_mask].max(1))
    cr_lost += list(arms["cRBEF"][lost_mask][np.arange(lost_mask.sum()), yt[lost_mask]])

bins = np.linspace(0.2, 1.0, 17)
ax3.hist(mh_lost, bins=bins, density=True, alpha=0.55, color="#B23A2F",
         label="MHnoU wrong-class conf.", zorder=3)
ax3.hist(cr_lost, bins=bins, density=True, alpha=0.55, color="#E69F00",
         label="cRBEF true-class conf.", zorder=3)
ax3.axvline(np.mean(mh_lost), color="#B23A2F", lw=1.2, ls="--")
ax3.axvline(np.mean(cr_lost), color="#E69F00", lw=1.2, ls="--")
ax3.annotate(f"{np.mean(mh_lost):.2f}", (np.mean(mh_lost), 5.3), fontsize=6.4,
             color="#B23A2F", ha="center")
ax3.annotate(f"{np.mean(cr_lost):.2f}", (np.mean(cr_lost), 4.9), fontsize=6.4,
             color="#9A7A00", ha="center")
ax3.set_title("Confidence on lost samples", loc="left", fontweight="bold", fontsize=8.6)
ax3.set_xlabel("probability", fontsize=7)
ax3.set_ylabel("density", fontsize=7)
ax3.set_ylim(0, 6)
ax3.legend(fontsize=5.6, frameon=False, loc="upper left")
ax3.grid(axis="y", zorder=0)

headline(fig, "The 0.60 pp gap = one \u03b7-fallback seed (\u22120.85) + bounded-correction residuals:\n"
              "MHnoU is confidently wrong (0.63) exactly where the correction cannot rescue.",
         y=0.995)
fig.text(0.012, 0.872, "Frozen M3ED formal test \u00b7 4,201 samples \u00b7 3 paired seeds (cRBEF fixed seed 17)",
         fontsize=7.0, color="#555555", ha="left")
save(fig, "figG_crbef_gap_decomposition")
