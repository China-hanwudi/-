# -*- coding: utf-8 -*-
"""图 E 升级版上半：正式 M3ED 测试逐类 F1 + 总体 WF1，ReCoMER vs 同协议复现基线。

一句话结论：ReCoMER 在全部 7 个类与总体 WF1 上超过同协议复现的 MLP / DialogueRNN。

数据源：
- subgroup_perclass_results.json（MHnoU/cRBEF/equal_weight/ReCoMER 逐类 f1_mean/f1_sd，0-1 尺度）
- baselines_m3ed.json（MLP/DialogueRNN，per_class_mean/per_class_sd 为 0-100 百分数；
  models.*.wf1_sd 为总体 WF1 的 seed 间 SD）
- full_test_local/{43,47,59}/PREDICTIONS.npz（重新计算各臂总体 WF1，sklearn weighted F1，
  argmax 口径；sample SD, ddof=1，与 SUMMARY.json 的 sample_sd 一致）
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from style_shared import PALETTE, apply_style, headline, save

apply_style()
BASE = os.path.dirname(os.path.abspath(__file__))

R = json.load(open(os.path.join(BASE, "subgroup_perclass_results.json"), encoding="utf-8"))
B = json.load(open(os.path.join(BASE, "baselines_m3ed.json"), encoding="utf-8"))

A = R["A_per_class"]
classes = A["class_names"]
support = A["per_class"]["ReCoMER"]["support"]

# ---- 总体 WF1（3 seeds，sample SD）----
SEEDS = ["43", "47", "59"]
ARM_NPZ = {"MHnoU": "MHnoU", "cRBEF": "cRBEF", "equal_weight": "equal_weight",
           "ReCoMER": "probabilities"}
overall, overall_sd = {}, {}
for arm, key in ARM_NPZ.items():
    vals = []
    for s in SEEDS:
        d = np.load(os.path.join(BASE, "full_test_local", s, "PREDICTIONS.npz"))
        vals.append(f1_score(d["y_true"], d[key].argmax(1), average="weighted") * 100)
    overall[arm] = float(np.mean(vals))
    overall_sd[arm] = float(np.std(vals, ddof=1))

# 6 个系列：(键, 图例名, 颜色, 均值list, sd list) —— 逐类 7 值 + 第 8 组 Overall WF1
BASELINE_COLORS = {"MLP": "#BBBBBB", "DialogueRNN": "#888888"}
series = []
for name in ["MLP", "DialogueRNN"]:
    m = list(B["models"][name]["per_class_mean"])            # 已是百分数
    sd = list(B["models"][name]["per_class_sd"])             # 已是百分数
    m.append(B["models"][name]["wf1_mean"] * 100)            # Overall WF1
    sd.append(B["models"][name]["wf1_sd"] * 100)
    series.append((name, name, BASELINE_COLORS[name], np.array(m), np.array(sd)))
for arm, lab in [("MHnoU", "MHnoU branch"), ("cRBEF", "cRBEF (fixed)"),
                 ("equal_weight", "Equal-weight"), ("ReCoMER", "ReCoMER")]:
    m = list(np.array(A["per_class"][arm]["f1_mean"]) * 100)
    sd = list(np.array(A["per_class"][arm]["f1_sd"]) * 100)
    m.append(overall[arm])
    sd.append(overall_sd[arm])
    series.append((arm, lab, PALETTE[arm], np.array(m), np.array(sd)))

fig = plt.figure(figsize=(7.0, 2.6))
ax = fig.add_axes([0.06, 0.18, 0.92, 0.62])

# 7 个类位于 0..6，Overall 组与第 7 组留 1.6 倍组距
x = np.array(list(range(len(classes))) + [len(classes) - 1 + 1.6])
n = len(series)
w = 0.13
for i, (key, lab, color, m, sd) in enumerate(series):
    off = (i - (n - 1) / 2.0) * w
    bars = ax.bar(x + off, m, width=w * 0.9, yerr=sd, capsize=1.6,
                  color=color, edgecolor="black", linewidth=0.35,
                  error_kw=dict(lw=0.55, ecolor="#555555"), label=lab, zorder=3)
    if key == "ReCoMER":
        for b in bars:
            b.set_linewidth(1.3)
    # 柱顶数值标签：竖排（90°），避免同组相邻柱标签横向重叠
    for xi, v, s in zip(x + off, m, sd):
        ax.annotate(f"{v:.1f}", (xi, v + s), textcoords="offset points",
                    xytext=(0, 1.5), fontsize=6.5, color="#666666",
                    ha="center", va="bottom", rotation=90, zorder=5)

labels = [f"{c}\n(n={nn})" for c, nn in zip(classes, support)] + ["Overall\nWF1"]
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.2)
ax.set_ylabel("Per-class F1 (%)")
ax.set_xlim(-0.6, x[-1] + 0.5)
ax.set_ylim(0, 92)
ax.grid(axis="y", zorder=0)
ax.legend(ncol=3, frameon=False, loc="upper right", fontsize=7.0,
          handlelength=1.2, columnspacing=0.9, bbox_to_anchor=(1.0, 1.04))

headline(fig, "ReCoMER exceeds the same-protocol reproduced baselines on every class and on overall\n"
              "WF1 (58.00 vs MLP 44.73 and DialogueRNN 41.61)")
fig.text(0.012, 0.845, "Frozen M3ED formal test \u00b7 4,201 samples \u00b7 baselines: 3 seeds on identical features/split; "
                       "ReCoMER/cRBEF: 3 paired seeds (cRBEF fixed checkpoint)",
         fontsize=7.0, color="#555555", ha="left")
save(fig, "figE_perclass_with_baselines")

# ---- 控制台核对 ----
print("Overall WF1 (%):", {k: round(v, 2) for k, v in overall.items()},
      "| sd:", {k: round(v, 2) for k, v in overall_sd.items()})
print("MLP", round(B["models"]["MLP"]["wf1_mean"] * 100, 2),
      "| DialogueRNN", round(B["models"]["DialogueRNN"]["wf1_mean"] * 100, 2))
print("ReCoMER Fear:", round(series[5][3][5], 1))
