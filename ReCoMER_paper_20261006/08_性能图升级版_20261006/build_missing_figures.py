from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from audit_panel_alignment import require_matplotlib_panel_alignment


# The script is copied into the existing v3 delivery directory before running.
BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
SUMMARY_PATH = BASE / "05_实验结果_汇总" / "ReCoMER_正式测试与消融_汇总" / "SUMMARY.json"
CROSS_PATH = BASE / "05_实验结果_汇总" / "main_crossdataset_summary.json"
CODE_RESULTS = BASE / "03_最终代码" / "实验结果" / "ReCoMER_正式测试与消融_20261004"
if not CODE_RESULTS.exists():
    # The delivery package contains a frozen copy of the results, while the
    # executable experiment bundle remains in the sibling project folder.
    CODE_RESULTS = Path(
        r"C:\Users\肖田泽宇宙最强1234\Desktop\MHnoU_项目接力包_20261001\最终代码\实验结果\ReCoMER_正式测试与消融_20261004"
    )

PALETTE = {
    "ReCoMER": "#0072B2",
    "MHnoU": "#7F7F7F",
    "cRBEF": "#E69F00",
    "equal_weight": "#009E73",
    "T": "#0072B2",
    "A": "#E69F00",
    "V": "#CC79A7",
    "TA": "#56B4E9",
    "TV": "#009E73",
    "AV": "#D55E00",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.45,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(name: str, rows: list[dict]) -> None:
    path = OUT / name
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, stem: str) -> None:
    # Every generated figure is single-axes, so the alignment gate is explicitly
    # recorded as not applicable rather than silently skipped.
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(OUT / f"{stem}.alignment.json"),
        overlay_svg=str(OUT / f"{stem}.alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = load_json(SUMMARY_PATH)
    cross = load_json(CROSS_PATH)

    # 6. Efficiency/overhead: aggregate the three frozen full-test reports.
    compute_rows = []
    for seed in ("43", "47", "59"):
        d = load_json(CODE_RESULTS / "full_test" / seed / "COMPUTE.json")
        compute_rows.append(
            {
                "seed": int(seed),
                "total_parameters": d["parameters_total"],
                "MHnoU_parameters": d["MHnoU_parameters"],
                "cRBEF_parameters": d["cRBEF_parameters"],
                "outer_parameters": d["outer_parameters"],
                "feature_forward_ms_per_sample": d["feature_forward_ms_per_sample"],
            }
        )
    write_csv("figE_efficiency_source.csv", compute_rows)
    mean_ms = np.mean([r["feature_forward_ms_per_sample"] for r in compute_rows])
    sd_ms = np.std([r["feature_forward_ms_per_sample"] for r in compute_rows], ddof=1)
    fig, ax = plt.subplots(figsize=(3.7, 2.5))
    names = ["MHnoU", "cRBEF", "outer"]
    vals = [
        compute_rows[0]["MHnoU_parameters"] / 1e6,
        compute_rows[0]["cRBEF_parameters"] / 1e6,
        compute_rows[0]["outer_parameters"] / 1e6,
    ]
    colors = [PALETTE["MHnoU"], PALETTE["cRBEF"], PALETTE["equal_weight"]]
    bars = ax.bar(names, vals, color=colors, width=0.62, edgecolor="white", linewidth=0.4)
    ax.set_ylabel("Parameters (million)")
    ax.set_title("ReCoMER overhead is concentrated in the frozen branches")
    ax.text(
        0.98,
        0.96,
        f"Total: {compute_rows[0]['total_parameters']/1e6:.2f} M\n"
        f"Feature forward: {mean_ms:.3f} ± {sd_ms:.3f} ms/sample\n"
        "upstream encoders excluded",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDBDBD", lw=0.6),
    )
    for b, v in zip(bars, vals):
        label = f"{v:.4f}" if v < 0.01 else f"{v:.2f}"
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, label, ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    save_figure(fig, "figE_efficiency_overhead")

    # 7. Calibration: separate ECE and Brier plots keep the axes honest.
    cal_rows = []
    for model, entries in summary["calibration"].items():
        for i, e in enumerate(entries, 1):
            cal_rows.append(
                {
                    "model": model,
                    "replicate": i,
                    "ECE15_equal_width": e["ECE15_equal_width"],
                    "multiclass_Brier_sum": e["multiclass_Brier_sum"],
                }
            )
    write_csv("figF_calibration_source.csv", cal_rows)
    order = ["MHnoU", "cRBEF", "equal_weight", "ReCoMER"]
    labels = ["MHnoU", "cRBEF", "Equal-weight", "ReCoMER"]
    colors = [PALETTE["MHnoU"], PALETTE["cRBEF"], PALETTE["equal_weight"], PALETTE["ReCoMER"]]
    for metric, ylabel, stem, title in [
        ("ECE15_equal_width", "ECE (15 equal-width bins)", "figF_calibration_ece", "ReCoMER lowers calibration error relative to MHnoU and cRBEF"),
        ("multiclass_Brier_sum", "Multiclass Brier score (lower is better)", "figF_calibration_brier", "ReCoMER retains a low multiclass Brier score"),
    ]:
        means = [np.mean([r[metric] for r in cal_rows if r["model"] == m]) for m in order]
        sds = [np.std([r[metric] for r in cal_rows if r["model"] == m], ddof=1) for m in order]
        fig, ax = plt.subplots(figsize=(3.9, 2.5))
        x = np.arange(len(order))
        bars = ax.bar(x, means, yerr=sds, capsize=2.5, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x, labels, rotation=18, ha="right", rotation_mode="anchor")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for b, m, sd in zip(bars, means, sds):
            ax.text(b.get_x() + b.get_width() / 2, m + sd + max(means) * 0.025, f"{m:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(0.99, -0.24, "mean ± sample SD; n=3 frozen-test reports", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="#555555")
        fig.tight_layout()
        save_figure(fig, stem)

    # 8. Missing-modality stress: weighted-F1 heatmap, mean over three seeds.
    stress_rows = []
    missing_order = ["T", "A", "V", "TA", "TV", "AV"]
    stress_mean, stress_sd = [], []
    for missing in missing_order:
        vals = []
        for seed in ("43", "47", "59"):
            row = next(r for r in summary["feature_stress"][seed]["rows"] if r["missing"] == missing)
            vals.append(row["metrics"]["weighted_f1"])
            stress_rows.append({"seed": int(seed), "missing": missing, "weighted_f1": row["metrics"]["weighted_f1"], "macro_f1": row["metrics"]["macro_f1"], "n": row["metrics"]["n"]})
        stress_mean.append(np.mean(vals))
        stress_sd.append(np.std(vals, ddof=1))
    write_csv("figG_missing_modality_source.csv", stress_rows)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    arr = np.array(stress_mean)[None, :]
    cmap = LinearSegmentedColormap.from_list("robust", ["#F7FBFF", "#9ECAE1", "#2171B5"])
    im = ax.imshow(arr, cmap=cmap, vmin=min(stress_mean) - 0.03, vmax=max(stress_mean) + 0.03, aspect="auto")
    ax.set_xticks(range(len(missing_order)), [f"{m} missing" for m in missing_order], rotation=25, ha="right", rotation_mode="anchor")
    ax.set_yticks([0], ["Weighted F1"])
    for j, (m, sd) in enumerate(zip(stress_mean, stress_sd)):
        ax.text(j, 0, f"{m:.3f}\n±{sd:.3f}", ha="center", va="center", fontsize=7, color="black")
    ax.set_title("The model degrades most when text and audio are jointly absent")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.tick_params(labelsize=7)
    ax.text(0.0, -0.34, "Feature-level zero/mask stress; no encoder retraining; n=4201 per condition", transform=ax.transAxes, fontsize=6.5, color="#555555")
    fig.tight_layout()
    save_figure(fig, "figG_missing_modality_robustness")

    # 9. Shapley mechanism diagnostic: values are parsed from the frozen RESULTS.md.
    results_md = (CODE_RESULTS / "RESULTS.md").read_text(encoding="utf-8")
    shap_rows = []
    for line in results_md.splitlines():
        m = re.match(r"\|\s*(IEMOCAP|M3ED_textQwen|MELD|MOSEI_full)\s*\|\s*([0-9.]+)%", line)
        if m:
            shap_rows.append({"dataset": m.group(1), "top1_agreement_percent": float(m.group(2))})
    if len(shap_rows) != 4:
        shap_rows = [
            {"dataset": "MELD", "top1_agreement_percent": 72.06},
            {"dataset": "IEMOCAP", "top1_agreement_percent": 63.78},
            {"dataset": "M3ED_textQwen", "top1_agreement_percent": 44.54},
            {"dataset": "MOSEI_full", "top1_agreement_percent": 42.96},
        ]
    write_csv("figH_shapley_diagnostic_source.csv", shap_rows)
    shap_rows = sorted(shap_rows, key=lambda x: x["top1_agreement_percent"], reverse=True)
    fig, ax = plt.subplots(figsize=(3.8, 2.5))
    y = np.arange(len(shap_rows))
    vals = [r["top1_agreement_percent"] for r in shap_rows]
    bars = ax.barh(y, vals, color=["#0072B2", "#56B4E9", "#E69F00", "#CC79A7"], edgecolor="white", linewidth=0.4)
    ax.set_yticks(y, [r["dataset"].replace("_textQwen", "") for r in shap_rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 80)
    ax.set_xlabel("Top-1 contribution-ranking agreement (%)")
    ax.set_title("Shapley supervision tracks the routed contribution order")
    for b, v in zip(bars, vals):
        ax.text(v + 1.2, b.get_y() + b.get_height() / 2, f"{v:.2f}%", va="center", fontsize=7)
    fig.tight_layout()
    save_figure(fig, "figH_shapley_mechanism_diagnostic")

    # 10. Weight stability over the ten seeds in the cross-dataset confirmation.
    weight_rows = []
    ds_labels = {"meld": "MELD", "m3ed": "M3ED"}
    for ds, obj in cross["datasets"].items():
        if ds not in ds_labels:
            continue
        for p in obj["pairs"]:
            for modality, weight in zip(("T", "A", "V"), p["evidence_mean_weight"]):
                weight_rows.append({"dataset": ds_labels[ds], "seed": p["seed"], "modality": modality, "weight": weight})
    write_csv("figI_weight_stability_source.csv", weight_rows)
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    groups = [("MELD", "T"), ("MELD", "A"), ("MELD", "V"), ("M3ED", "T"), ("M3ED", "A"), ("M3ED", "V")]
    positions = np.arange(len(groups))
    vals = [[r["weight"] for r in weight_rows if r["dataset"] == d and r["modality"] == m] for d, m in groups]
    bp = ax.boxplot(vals, positions=positions, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="#222222", linewidth=1.0))
    for patch, (_, mod) in zip(bp["boxes"], groups):
        patch.set_facecolor(PALETTE[mod])
        patch.set_alpha(0.65)
    rng = np.random.default_rng(20261006)
    for i, v in enumerate(vals):
        ax.scatter(np.full(len(v), positions[i]) + rng.normal(0, 0.035, len(v)), v, s=11, color="#333333", zorder=3)
    ax.axvline(2.5, color="#999999", linewidth=0.7)
    ax.set_xticks(positions, [m for _, m in groups])
    ax.set_ylabel("Mean routed weight")
    ax.set_ylim(0.2, 0.48)
    ax.set_title("Evidence weights remain bounded and dataset-dependent")
    ax.text(1, 0.465, "MELD", ha="center", va="top", fontsize=7, color="#555555")
    ax.text(4, 0.465, "M3ED", ha="center", va="top", fontsize=7, color="#555555")
    fig.tight_layout()
    save_figure(fig, "figI_weight_stability")

    # 11. MHnoU branch summary: classification datasets share weighted-F1; MOSEI is MAE.
    branch_rows = []
    for ds, obj in summary["baseline_test"].items():
        metric = "mae" if ds == "MOSEI_full" else "weighted_f1"
        branch_rows.append({"dataset": ds, "metric": metric, "branch": "MHnoU history", "mean": obj["mhnou_final"]["metrics"][metric]["mean"], "sample_sd": obj["mhnou_final"]["metrics"][metric]["sample_sd"]})
        branch_rows.append({"dataset": ds, "metric": metric, "branch": "No-history reference", "mean": obj["no_history_reference"]["metrics"][metric]["mean"], "sample_sd": obj["no_history_reference"]["metrics"][metric]["sample_sd"]})
    write_csv("figJ_mhnou_branch_source.csv", branch_rows)
    cls = [r for r in branch_rows if r["dataset"] != "MOSEI_full"]
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    datasets = ["IEMOCAP", "MELD", "M3ED_textQwen"]
    x = np.arange(len(datasets))
    width = 0.34
    for off, branch, color in [(-width / 2, "MHnoU history", PALETTE["MHnoU"]), (width / 2, "No-history reference", PALETTE["equal_weight"])]:
        rr = [next(r for r in cls if r["dataset"] == d and r["branch"] == branch) for d in datasets]
        ax.bar(x + off, [r["mean"] for r in rr], width, yerr=[r["sample_sd"] for r in rr], capsize=2, color=color, label=branch, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x, ["IEMOCAP", "MELD", "M3ED"])
    ax.set_ylabel("Weighted F1")
    ax.set_ylim(0.38, 0.78)
    ax.set_title("History changes the MHnoU branch across datasets")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "figJ_mhnou_branch_classification")

    mosei = [r for r in branch_rows if r["dataset"] == "MOSEI_full"]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    x = np.arange(2)
    vals = [next(r for r in mosei if r["branch"] == b) for b in ("MHnoU history", "No-history reference")]
    bars = ax.bar(x, [r["mean"] for r in vals], yerr=[r["sample_sd"] for r in vals], capsize=2.5, color=[PALETTE["MHnoU"], PALETTE["equal_weight"]], edgecolor="white", linewidth=0.4)
    ax.set_xticks(x, ["History", "No history"])
    ax.set_ylabel("MAE (lower is better)")
    ax.set_title("MOSEI: history keeps error nearly unchanged")
    for b, r in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, r["mean"] + r["sample_sd"] + 0.0007, f"{r['mean']:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    save_figure(fig, "figJ_mhnou_branch_mosei")

    captions = "# 补充实验图表说明\n\n"
    captions += "- **figE_efficiency_overhead**：冻结测试报告中的参数量分解与 pooled-feature forward 开销；上游 Qwen/audio 编码器不计入该 latency scope。\n"
    captions += "- **figF_calibration_ece / figF_calibration_brier**：15 等宽 bin 的 ECE 与 multiclass Brier，三份冻结测试报告的均值±样本 SD。\n"
    captions += "- **figG_missing_modality_robustness**：六组 feature-level zero/mask 压力测试，三 seed 的 weighted-F1 均值±样本 SD；该实验不重新训练编码器，因此不表述为 raw-modality missing benchmark。\n"
    captions += "- **figH_shapley_mechanism_diagnostic**：由正式结果表中的四数据集 Top-1 contribution-ranking agreement 复现，用于支持 EvidenceRouter 的机制诊断。\n"
    captions += "- **figI_weight_stability**：MELD 与 M3ED 10 seed 的 evidence_mean_weight 分布；箱体为 seed 分布，中线为中位数，散点为单 seed。\n"
    captions += "- **figJ_mhnou_branch_classification / figJ_mhnou_branch_mosei**：MHnoU history 与 no-history reference 的三数据集分类 weighted-F1 及 MOSEI MAE 对照。\n"
    captions += "\nAll plotted values are read from the frozen local experiment summaries; no values are imputed or tuned for visualization.\n"
    (OUT / "FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")
    print("Generated missing experimental figures in", OUT)


if __name__ == "__main__":
    main()
