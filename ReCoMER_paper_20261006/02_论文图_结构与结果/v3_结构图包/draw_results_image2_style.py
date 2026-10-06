"""Exact-data results figures with the Image 2 style study as visual reference.

The Image 2 output is not used as a quantitative source: all plotted values are
loaded from results_data_image2_style.json. The figure uses direct labels,
restrained modality colours, and green/red directional cues suggested by the
provided Feishu guide.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "results_data_image2_style.json").read_text(encoding="utf-8"))

INK = "#233642"
MUTED = "#6C7A82"
GRID = "#D9E1E5"
BLUE = "#4E8FCB"
ORANGE = "#D9924B"
TEAL = "#4BA9A7"
PURPLE = "#8877C9"
GREEN = "#2E9B63"
RED = "#CE5B5B"
GOLD = "#D59B32"
PALE_BLUE = "#EAF2F8"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.7,
    "xtick.color": INK,
    "ytick.color": INK,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def panel_title(ax, label, title):
    ax.text(0.0, 1.07, f"{label}  {title}", transform=ax.transAxes,
            ha="left", va="bottom", color=INK, fontsize=11, fontweight="bold")


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.85)
    ax.set_axisbelow(True)


def save_all(fig, stem):
    for ext, kwargs in [("png", {"dpi": 600}), ("tiff", {"dpi": 600}), ("pdf", {}), ("svg", {})]:
        fig.savefig(ROOT / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.08, **kwargs)
    plt.close(fig)


def draw_overview():
    fig = plt.figure(figsize=(12.0, 6.2), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.05, 1.0],
                          left=0.055, right=0.985, top=0.90, bottom=0.12,
                          wspace=0.30, hspace=0.58)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    # Panel A: main result.
    main = DATA["main_m3ed_test"]
    labels = [d["model"] for d in main]
    values = np.array([d["wf1"] for d in main])
    errs = np.array([d["sd"] for d in main])
    colours = ["#A4B1B8", BLUE, ORANGE, TEAL, PURPLE, GOLD]
    x = np.arange(len(labels))
    ax_a.bar(x, values, yerr=errs, capsize=3, color=colours, edgecolor="white", linewidth=0.7,
             error_kw={"ecolor": INK, "elinewidth": 0.8, "capthick": 0.8})
    ax_a.set_ylim(40, 61.5)
    ax_a.set_xticks(x, labels, rotation=28, ha="right")
    ax_a.set_ylabel("M3ED test WF1 (%)")
    ax_a.set_yticks(np.arange(40, 62, 5))
    style_axis(ax_a)
    for i, (v, e) in enumerate(zip(values, errs)):
        ax_a.text(i, v + max(e, 0.15) + 0.35, f"{v:.2f}", ha="center", va="bottom",
                  fontsize=9, fontweight="bold", color=INK)
    ax_a.text(4, 60.9, "+0.607 pp vs equal-weight", color=PURPLE, fontsize=8.4,
              ha="center", va="top", fontweight="bold")
    panel_title(ax_a, "(a)", "Frozen M3ED test comparison")

    # Panel B: conditional module effects.
    modules = ["History feedback", "EvidenceRouter", "History abstention"]
    datasets = ["M3ED", "MELD", "IEMOCAP", "MOSEI"]
    matrix = np.full((len(modules), len(datasets)), np.nan)
    for row in DATA["module_effects_pp"]:
        matrix[modules.index(row["module"]), datasets.index(row["dataset"])] = row["effect_pp"]
    im = ax_b.imshow(matrix, cmap="RdYlGn", vmin=-0.75, vmax=0.75, aspect="auto")
    ax_b.set_xticks(np.arange(len(datasets)), datasets)
    ax_b.set_yticks(np.arange(len(modules)), ["History\nfeedback", "Evidence\nRouter", "History\nabstention"])
    ax_b.tick_params(length=0, labelsize=8.5)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            if np.isfinite(matrix[r, c]):
                v = matrix[r, c]
                ax_b.text(c, r, f"{v:+.3f}", ha="center", va="center", fontsize=8.3,
                          fontweight="bold", color="white" if abs(v) > 0.38 else INK)
            else:
                ax_b.text(c, r, "—", ha="center", va="center", fontsize=9, color=MUTED)
    for spine in ax_b.spines.values(): spine.set_visible(False)
    ax_b.set_title("positive = better (WF1 ↑ or MAE ↓)", fontsize=8.2, color=MUTED, pad=7)
    panel_title(ax_b, "(b)", "Module effects across datasets (pp)")

    # Panel C: integrated paired deltas with bootstrap intervals.
    ab = DATA["integrated_deltas_pp"]
    ab = list(reversed(ab))
    y = np.arange(len(ab))
    d = np.array([r["delta_pp"] for r in ab])
    lo = np.array([r["ci_low"] for r in ab])
    hi = np.array([r["ci_high"] for r in ab])
    xerr = np.vstack([d - lo, hi - d])
    ax_c.axvline(0, color=INK, lw=0.8)
    ax_c.errorbar(d, y, xerr=xerr, fmt="none", ecolor="#7D8A90", elinewidth=1.0,
                  capsize=2, zorder=2)
    ax_c.scatter(d, y, s=42, c=[GREEN if v >= 0 else RED for v in d], zorder=3,
                 edgecolors="white", linewidths=0.6)
    ax_c.set_yticks(y, [r["control"] for r in ab], fontsize=8.2)
    ax_c.set_xlabel("Full model − control (WF1 pp)")
    ax_c.set_xlim(-1.05, 1.15)
    ax_c.grid(axis="x", color=GRID, lw=0.6)
    ax_c.spines[["top", "right", "left"]].set_visible(False)
    ax_c.tick_params(axis="y", length=0)
    ax_c.text(0.98, 0.02, "95% dialogue-bootstrap CI", transform=ax_c.transAxes,
              ha="right", va="bottom", fontsize=7.5, color=MUTED)
    panel_title(ax_c, "(c)", "Integrated ablation deltas")

    fig.suptitle("ReCoMER results: main comparison, conditional module gains and ablations",
                 x=0.055, ha="left", y=0.965, fontsize=13, fontweight="bold", color=INK)
    fig.text(0.055, 0.045,
             "Bars and effects use the recorded protocols; module effects are positive when the metric improves.",
             ha="left", va="bottom", fontsize=8, color=MUTED)
    save_all(fig, "07_ReCoMER_results_overview_image2_style")


def draw_exact_feature():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"wspace": 0.48})
    # Exact-feature M3ED controls.
    ax = axes[0]
    rows = DATA["exact_feature_m3ed"]
    labels = [r["arm"] for r in rows]
    v = np.array([r["wf1"] for r in rows])
    e = np.array([r["sd"] for r in rows])
    x = np.arange(len(rows))
    ax.bar(x, v, yerr=e, capsize=3, color=["#A4B1B8", TEAL, BLUE, PURPLE, ORANGE], edgecolor="white")
    ax.set_xticks(x, labels, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("M3ED exact-feature test WF1 (%)")
    ax.set_ylim(52.4, 54.9)
    ax.set_yticks(np.arange(52.5, 55.0, 0.5))
    style_axis(ax)
    for i, z in enumerate(v): ax.text(i, z + e[i] + 0.03, f"{z:.2f}", ha="center", va="bottom", fontsize=8.2)
    panel_title(ax, "(a)", "Exact-feature M3ED controls")

    ax = axes[1]
    rows = DATA["exact_feature_crbef"]
    labels = [r["arm"] for r in rows]
    v = np.array([r["wf1"] for r in rows])
    e = np.array([r["sd"] for r in rows])
    x = np.arange(len(rows))
    ax.bar(x, v, yerr=e, capsize=3, color=[BLUE, TEAL, ORANGE, PURPLE, GOLD], edgecolor="white")
    ax.set_xticks(x, labels, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("WF1 (%)")
    ax.set_ylim(38, 57)
    ax.set_yticks(np.arange(40, 58, 5))
    style_axis(ax)
    for i, z in enumerate(v): ax.text(i, z + e[i] + 0.2, f"{z:.2f}", ha="center", va="bottom", fontsize=8.2)
    panel_title(ax, "(b)", "Exact-feature cRBEF replication")
    fig.suptitle("Exact-feature protocol: supplementary reproducibility checks", x=0.06, ha="left",
                 y=1.02, fontsize=13, fontweight="bold", color=INK)
    fig.text(0.06, -0.05, "Values are means ± population SD over the recorded seeds.", fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.26, top=0.80)
    save_all(fig, "08_ReCoMER_exact_feature_checks_image2_style")


if __name__ == "__main__":
    draw_overview()
    draw_exact_feature()
    print("Wrote Image 2-style exact-data figures to", ROOT)
