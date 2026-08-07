from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Noto Sans SC', 'Arial', 'SimHei', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.sans-serif': ['Microsoft YaHei', 'Noto Sans SC', 'Arial', 'SimHei', 'DejaVu Sans', 'Liberation Sans']})
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
RESULT = REPOSITORY / "results" / "emotiontalk_multimodal_external_v1.json"
OUT = REPOSITORY / "assets" / "emotiontalk_external_confirmation"
SOURCE = REPOSITORY / "results" / "emotiontalk_external_source_data.csv"

METHODS = ["text", "text_audio", "text_video", "text_audio_video"]
LABELS = ["文本", "文本+音频", "文本+视频", "三模态"]
COLORS = ["#B4C0E4", "#7884B4", "#A7C9C8", "#484878"]
BLUE = "#484878"
RED = "#C44E52"
GREEN = "#2E8B57"
GREY = "#777777"


def panel_label(ax, label: str) -> None:
    ax.text(-0.13, 1.06, label, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom")


def write_source(rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with SOURCE.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    rows: list[dict] = []
    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.04, 1.0], height_ratios=[1.0, 1.05])

    # a — natural harm, hero evidence
    ax = fig.add_subplot(grid[0, 0])
    rates, lows, highs = [], [], []
    for method, label in zip(METHODS, LABELS, strict=True):
        item = result["base_modality_ablation"][method]["eligible_regret"]
        rates.append(100 * item["harm_rate"])
        lows.append(100 * item["harm_rate_ci_low"])
        highs.append(100 * item["harm_rate_ci_high"])
        rows.append({"panel": "a", "method": label, "metric": "harm_rate_pct", "estimate": 100 * item["harm_rate"], "ci_low": 100 * item["harm_rate_ci_low"], "ci_high": 100 * item["harm_rate_ci_high"], "n": 1770})
    y = np.arange(len(LABELS))
    ax.errorbar(rates, y, xerr=[np.asarray(rates) - lows, np.asarray(highs) - rates], fmt="none", ecolor=GREY, elinewidth=1.1, capsize=2.5, zorder=1)
    ax.scatter(rates, y, s=40, c=COLORS, edgecolors="#333333", linewidths=0.6, zorder=2)
    ax.axvline(20, color=RED, linestyle="--", linewidth=0.9, alpha=0.8)
    ax.text(20.5, 3.42, "预冻结最低门 20%", color=RED, fontsize=6.2, va="top")
    for value, row in zip(rates, y, strict=True):
        ax.text(value + 1.0, row, f"{value:.1f}%", va="center", fontsize=6.5)
    ax.set_yticks(y, LABELS)
    ax.invert_yaxis()
    ax.set_xlim(15, 48)
    ax.set_xlabel("历史造成逐查询NLL上升的比例（%）")
    ax.set_title("自然历史伤害在四种模态设置中均存在", loc="left", fontweight="bold")
    panel_label(ax, "a")

    # b — selector metrics
    sub = grid[0, 1].subgridspec(1, 2, wspace=0.12)
    auc_ax = fig.add_subplot(sub[0, 0])
    rho_ax = fig.add_subplot(sub[0, 1])
    aucs, rhos = [], []
    for method, label in zip(METHODS, LABELS, strict=True):
        item = result["selector_metrics"][method]
        aucs.append(item["harm_auc"])
        rhos.append(item["mean_prediction_spearman"])
        rows.extend([
            {"panel": "b", "method": label, "metric": "harm_auc", "estimate": item["harm_auc"], "n": 1770},
            {"panel": "b", "method": label, "metric": "spearman", "estimate": item["mean_prediction_spearman"], "n": 1770},
        ])
    x = np.arange(len(LABELS))
    auc_ax.bar(x, aucs, color=COLORS, edgecolor="#333333", linewidth=0.5)
    auc_ax.axhline(0.65, color=RED, linestyle="--", linewidth=0.9)
    auc_ax.set_ylim(0.54, 0.70)
    auc_ax.set_ylabel("Harm AUC")
    auc_ax.set_xticks(x, ["T", "T+A", "T+V", "T+A+V"], rotation=35, ha="right")
    auc_ax.set_title("伤害分类AUC", fontsize=7.2)
    for index, value in enumerate(aucs):
        auc_ax.text(index, value + 0.003, f"{value:.3f}", ha="center", va="bottom", fontsize=5.7, rotation=90)
    rho_ax.bar(x, rhos, color=COLORS, edgecolor="#333333", linewidth=0.5)
    rho_ax.axhline(0, color="#333333", linewidth=0.7)
    rho_ax.set_ylim(-0.02, 0.22)
    rho_ax.set_ylabel("Spearman ρ")
    rho_ax.set_xticks(x, ["T", "T+A", "T+V", "T+A+V"], rotation=35, ha="right")
    rho_ax.set_title("连续regret排序", fontsize=7.2)
    for index, value in enumerate(rhos):
        rho_ax.text(index, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=5.7, rotation=90)
    auc_ax.text(-0.26, 1.11, "b", transform=auc_ax.transAxes, fontsize=9, fontweight="bold")

    # c — risk/coverage
    ax = fig.add_subplot(grid[1, 0])
    policy = result["risk_coverage_policies"]["text_audio_video"]
    curve = []
    for key, item in policy.items():
        if not key.startswith("calibration_target_"):
            continue
        boot = item["cluster_bootstrap_policy_minus_current"]
        curve.append((100 * item["history_coverage"], boot["mean_excess_loss"], boot["mean_excess_loss_ci_low"], boot["mean_excess_loss_ci_high"]))
    curve.sort()
    coverage = np.asarray([item[0] for item in curve])
    mean = np.asarray([item[1] for item in curve])
    low = np.asarray([item[2] for item in curve])
    high = np.asarray([item[3] for item in curve])
    ax.fill_between(coverage, low, high, color="#B4C0E4", alpha=0.45, linewidth=0)
    ax.plot(coverage, mean, color=BLUE, marker="o", markersize=3.2, linewidth=1.2)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axvline(10, color=RED, linestyle="--", linewidth=0.9)
    strict = policy["strict_conformal_q90_upper_below_zero"]
    strict_boot = strict["cluster_bootstrap_policy_minus_current"]
    sx, sy = 100 * strict["history_coverage"], strict_boot["mean_excess_loss"]
    ax.scatter([sx], [sy], color=RED, marker="X", s=45, zorder=4)
    ax.annotate(f"严格q90：{sx:.2f}%（9条）", (sx, sy), xytext=(12, 13), textcoords="offset points", fontsize=6.2, color=RED, arrowprops={"arrowstyle": "-", "color": RED, "lw": 0.7})
    ax.set_xlim(-1, 94)
    ax.set_xlabel("使用历史的覆盖率（%）")
    ax.set_ylabel("策略相对current-only的均值超额NLL")
    ax.set_title("严格安全回退未达到非平凡覆盖", loc="left", fontweight="bold")
    panel_label(ax, "c")
    for x_value, m, lo, hi in curve:
        rows.append({"panel": "c", "method": "T+A+V selector", "metric": "policy_mean_excess_nll", "coverage_pct": x_value, "estimate": m, "ci_low": lo, "ci_high": hi, "n": 1770})
    rows.append({"panel": "c", "method": "strict_q90", "metric": "policy_mean_excess_nll", "coverage_pct": sx, "estimate": sy, "ci_low": strict_boot["mean_excess_loss_ci_low"], "ci_high": strict_boot["mean_excess_loss_ci_high"], "n": 1770})

    # d — restricted permutation control
    outer = fig.add_subplot(grid[1, 1])
    outer.set_axis_off()
    outer.set_title("真实配对历史显著优于受限置换", loc="left", fontweight="bold", pad=4)
    panel_label(outer, "d")
    ax = outer.inset_axes([0.0, 0.58, 1.0, 0.30])
    pairing = result["restricted_history_permutation_control"]
    advantage = pairing["actual_advantage_over_restricted_permutation_nats"]
    ax.errorbar([advantage["mean"]], [0], xerr=[[advantage["mean"] - advantage["ci_low"]], [advantage["ci_high"] - advantage["mean"]]], fmt="o", color=GREEN, ecolor=GREEN, capsize=3, markersize=5)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.text(advantage["mean"], 0.13, f"+{advantage['mean']:.3f} nats", ha="center", fontsize=6.5, color=GREEN)
    ax.set_yticks([0], ["真实历史优势"])
    ax.set_xlim(-0.08, 0.65)
    ax.set_ylim(-0.25, 0.25)
    ax.set_xlabel("置换NLL − 真实历史NLL（nats）")
    bar_ax = outer.inset_axes([0.0, 0.08, 1.0, 0.30])
    actual_harm = result["base_modality_ablation"]["text_audio_video"]["eligible_regret"]["harm_rate"]
    perm_harm = pairing["permutation_mean_harm_vs_current"]
    bar_ax.barh([1, 0], [100 * actual_harm, 100 * perm_harm], height=0.52, color=[BLUE, "#D8D8D8"], edgecolor="#333333", linewidth=0.5)
    bar_ax.text(100 * actual_harm + 1, 1, f"{100*actual_harm:.1f}%", va="center", fontsize=6.3)
    bar_ax.text(100 * perm_harm + 1, 0, f"{100*perm_harm:.1f}%", va="center", fontsize=6.3)
    bar_ax.set_yticks([1, 0], ["真实历史", "受限置换"])
    bar_ax.set_xlim(0, 72)
    bar_ax.set_xlabel("相对current-only的伤害率（%）")
    rows.extend([
        {"panel": "d", "method": "actual_vs_permutation", "metric": "nll_advantage", "estimate": advantage["mean"], "ci_low": advantage["ci_low"], "ci_high": advantage["ci_high"], "n": 1770},
        {"panel": "d", "method": "actual_history", "metric": "harm_rate_pct", "estimate": 100 * actual_harm, "n": 1770},
        {"panel": "d", "method": "restricted_permutation", "metric": "harm_rate_pct", "estimate": 100 * perm_harm, "n": 1770},
    ])

    fig.suptitle("EmotionTalk三模态外部确认：问题成立，但严格安全回退仍失败", fontsize=10, fontweight="bold")
    fig.text(0.01, -0.030, "n=1,770个有同说话人历史的validation查询；误差线/阴影为dialogue聚类bootstrap 95% CI（2,000次）；base为5种子集成；置换20次。", fontsize=6.0, color="#555555")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_source(rows)
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
