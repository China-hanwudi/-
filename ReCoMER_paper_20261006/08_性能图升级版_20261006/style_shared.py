# -*- coding: utf-8 -*-
"""统一视觉语言：全部性能升级图共用调色板、字体与版式规范。
设计原则（遵循课题组图表方法论）：
- 颜色 3-5 种、低饱和；同一分支在所有图中同色同名；
- 每个 panel 自带指标名与方向箭头，图可脱离正文阅读；
- 大标题 = 一句话结论（读者看完应记住什么）。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 调色板（Okabe-Ito 低饱和变体）----
PALETTE = {
    "MHnoU": "#8B8B8B",
    "AV": "#CC79A7",
    "TAV": "#56B4E9",
    "equal_weight": "#009E73",
    "cRBEF": "#E69F00",
    "ReCoMER": "#0072B2",
}

ARM_LABELS = {
    "MHnoU": "MHnoU branch",
    "AV": "AV",
    "TAV": "TAV",
    "equal_weight": "Equal-weight",
    "cRBEF": "cRBEF (fixed)",
    "ReCoMER": "ReCoMER",
}

# 显著性三态色（森林图）
SIG_OK = "#1B7837"    # Holm p < 0.05
SIG_MARGINAL = "#E69F00"  # 0.05 <= p < 0.10
SIG_NS = "#8B8B8B"    # p >= 0.10

BASE = r"C:\Users\肖田泽宇宙最强1234\Desktop\论文\论文结构图_v3_具体表达_20261004"
RES = BASE + r"\05_实验结果_汇总"
FORMAL = RES + r"\ReCoMER_正式测试与消融_汇总"


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.5,
    })


def headline(fig, text, y=0.995):
    """图顶部的一句话结论标题（va=top，避免与副标题重叠）。"""
    fig.suptitle(text, fontsize=10, fontweight="bold", x=0.012, y=y,
                 ha="left", va="top")


def save(fig, name):
    import os

    out = os.path.join(BASE, "08_性能图升级版_20261006")
    fig.savefig(os.path.join(out, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved:", name)
