# -*- coding: utf-8 -*-
"""实验数据盘点看板（PNG 一图版，替代未渲染的 canvas）。
版式：标题 → 总览条 → 四章节卡片 → 必补(红)/可砍(黄)/放弃(灰)三栏 → 页脚。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import textwrap

plt.rcParams["font.family"] = ["Noto Sans SC", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 调色（低饱和）
C_DONE = "#1B7837"
C_MUST = "#B23A2F"
C_CUT = "#C98A12"
C_SKIP = "#7F7F7F"
C_FILL = "#FAFAF7"
C_STROKE = "#DDDDD6"
C_TEXT = "#2B2B28"
C_SUB = "#6E6E68"

W, H = 16, 9
fig = plt.figure(figsize=(W, H), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor("white")


def wrap(s, n):
    return "\n".join(textwrap.wrap(s, n, break_long_words=True, replace_whitespace=False))


def card(x, y, w, h, title, title_color=C_TEXT, fill=C_FILL, stroke=C_STROKE, title_fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.1",
                                fc=fill, ec=stroke, lw=1.0, mutation_aspect=0.5))
    ax.text(x + 1.6, y + h - 2.6, title, fontsize=title_fs, fontweight="bold",
            color=title_color, va="top")


# ---------------- 标题 ----------------
ax.text(2.5, 97, "ReCoMER 实验部分数据盘点", fontsize=21, fontweight="bold", color=C_TEXT, va="top")
ax.text(2.5, 92.8, "按论文章节 4.1–4.6 映射 · 四级分类：已具备 14 / 已做可砍 4 / 必须补 2 / 不必强求 4（今日补齐 7 项：图 A–F + 四数据集矩阵）",
        fontsize=10, color=C_SUB, va="top")
ax.text(2.5, 90.3, "来源：论文工作区 2026-10-06 全量核查（交接文档 02 + GPU 服务器 paper_experiments_20261004 实查）",
        fontsize=8, color=C_SUB, va="top")

# ---------------- 总览条 ----------------
ov_y, ov_h = 79.5, 8.5
kpis = [("24", "实验数据项", C_TEXT), ("14", "已具备", C_DONE), ("2", "必须补", C_MUST), ("8", "可砍/放弃", C_CUT)]
kw = 10.5
for i, (v, lab, col) in enumerate(kpis):
    x = 2.5 + i * (kw + 1.2)
    ax.add_patch(FancyBboxPatch((x, ov_y), kw, ov_h, boxstyle="round,pad=0.4,rounding_size=1.1",
                                fc=C_FILL, ec=C_STROKE, lw=1.0, mutation_aspect=0.5))
    ax.text(x + kw / 2, ov_y + ov_h - 1.6, v, fontsize=19, fontweight="bold", color=col, ha="center", va="top")
    ax.text(x + kw / 2, ov_y + 1.7, lab, fontsize=9, color=C_SUB, ha="center", va="bottom")

# 堆叠条
bx, bw = 50.5, 47
segs = [("已具备 14", 14, C_DONE), ("可砍 4", 4, C_CUT), ("必补 2", 2, C_MUST), ("不必强求 4", 4, C_SKIP)]
ax.text(bx, ov_y + ov_h - 0.6, "24 项总体构成", fontsize=10.5, fontweight="bold", color=C_TEXT, va="top")
bar_y, bar_h = ov_y + 2.2, 3.2
cx = bx
for name, v, col in segs:
    w = bw * v / 24
    ax.add_patch(Rectangle((cx, bar_y), w, bar_h, fc=col, ec="white", lw=0.8))
    if v >= 4:
        ax.text(cx + w / 2, bar_y + bar_h / 2, f"{name}", fontsize=8.5, color="white",
                ha="center", va="center", fontweight="bold")
    cx += w
for name, v, col in segs:
    if v < 4:
        ax.plot([bx + 1, bx + 2.2], [bar_y - 1.1, bar_y - 1.1], color=col, lw=5, solid_capstyle="butt")
        ax.text(bx + 3, bar_y - 1.5, f"{name}", fontsize=8, color=C_SUB, va="center")
        bx += 12

# ---------------- 章节卡片（4 列） ----------------
CH = [
    ("4.1–4.2 · Setup 与主对比", [
        (1, "正式测试六分支 + 配对 bootstrap"),
        (1, "逐类 F1 + 对话级 bootstrap（图 E）"),
        (1, "多指标面板 WF1/Acc/NLL/ECE/Brier（图 A）"),
    ]),
    ("4.3 · 组件隔离研究", [
        (1, "历史反馈十 seed（3 数据集）"),
        (1, "EvidenceRouter 十 seed + Holm 校正（图 B）"),
        (1, "弃权变体 3-seed 修正协议 B1/E5/E6/E7"),
    ]),
    ("4.4 · 整合消融与审计", [
        (1, "六个整合控制 + 对话 bootstrap"),
        (1, "cRBEF/MHnoU splice 信息流审计"),
        (1, "目标梯度审计（单模态头无梯度）"),
    ]),
    ("4.5–4.6 · Exact-feature 与失败边界", [
        (1, "四数据集 × 5 臂 × 10 seed 矩阵（今日补齐）"),
        (1, "延续/转折子群分析（图 F，今日补齐）"),
        (0, "同协议外部 SOTA 对比表 ← 唯一缺口"),
    ]),
]
cy, chh, cw, gap = 46, 30.5, 23.6, 0.9
for i, (title, items) in enumerate(CH):
    x = 2.5 + i * (cw + gap)
    card(x, cy, cw, chh, title)
    yy = cy + chh - 6.2
    for ok, txt in items:
        col = C_DONE if ok else C_MUST
        ax.text(x + 1.6, yy, "✓" if ok else "×", fontsize=11, color=col, fontweight="bold", va="top")
        ax.text(x + 4.2, yy, wrap(txt, 15), fontsize=8.6, color=C_TEXT if ok else C_MUST,
                va="top", fontweight="normal" if ok else "bold", linespacing=1.35)
        yy -= 7.6 if len(txt) > 15 else 5.6

# ---------------- 底部三栏 ----------------
by, bh = 3.5, 39
colw = 32.2
# 必补（红）
x = 2.5
ax.add_patch(FancyBboxPatch((x, by), colw, bh, boxstyle="round,pad=0.4,rounding_size=1.1",
                            fc="#FDF6F5", ec="#E3C4C0", lw=1.0, mutation_aspect=0.5))
ax.text(x + 1.6, by + bh - 2.4, "必须补（2 项）", fontsize=11, fontweight="bold", color=C_MUST, va="top")
ax.text(x + 1.6, by + bh - 6.4, "① 同协议外部 SOTA 对比表", fontsize=9.6, fontweight="bold", color=C_TEXT, va="top")
ax.text(x + 1.6, by + bh - 9.2, wrap("性能叙事的唯一外部锚点。收敛到 M3ED 上 2–3 个可复现强基线（HiRoC/DF-ERC 类），用现有特征包复现。服务器有 meld_sota_control 残件可借鉴。", 20),
        fontsize=8.4, color=C_SUB, va="top", linespacing=1.4)
ax.text(x + 1.6, by + bh - 22.6, "② “完整系统 vs 最强专家”归因段落", fontsize=9.6, fontweight="bold", color=C_TEXT, va="top")
ax.text(x + 1.6, by + bh - 25.4, wrap("58.00 < 58.60 需归因：MHnoU 分支（45.43）拖低候选集、η 校准 1/3 种子退回等权、有界修正容量受限。不需新实验，把减分项改写成边界分析。", 20),
        fontsize=8.4, color=C_SUB, va="top", linespacing=1.4)

# 可砍（黄）
x = 2.5 + colw + 1.1
ax.add_patch(FancyBboxPatch((x, by), colw, bh, boxstyle="round,pad=0.4,rounding_size=1.1",
                            fc="#FCF8EF", ec="#E6D9B8", lw=1.0, mutation_aspect=0.5))
ax.text(x + 1.6, by + bh - 2.4, "已做但不必要放（4 项）", fontsize=11, fontweight="bold", color=C_CUT, va="top")
cuts = [
    "首轮开发集数字（OUTER/valid 那批）——已被正式测试取代，禁止进论文",
    "gate 六公式敏感性、ordering controls——CI 全跨 0，附录带过",
    "semantic/sparsemax 筛查、jointsoft 臂——筛查信号/全负，一句带过",
    "PCA、特征压力测试——附录；压力测试须改成标准鲁棒性实验才报",
    "evidence_mpath 臂——四数据集不显著，只留 closed/solo；脚本自比较产物不报告",
]
yy = by + bh - 6.2
for c in cuts:
    ax.text(x + 1.6, yy, "•", fontsize=9, color=C_CUT, va="top")
    ax.text(x + 3.4, yy, wrap(c, 21), fontsize=8.2, color=C_TEXT, va="top", linespacing=1.35)
    yy -= 2.4 + 1.6 * (len(wrap(c, 21).split("\n")) - 1) + 1.4

# 放弃（灰）
x = 2.5 + 2 * (colw + 1.1)
ax.add_patch(FancyBboxPatch((x, by), colw, bh, boxstyle="round,pad=0.4,rounding_size=1.1",
                            fc="#F7F7F5", ec=C_STROKE, lw=1.0, mutation_aspect=0.5))
ax.text(x + 1.6, by + bh - 2.4, "没做但不值得强求（放弃）", fontsize=11, fontweight="bold", color=C_SKIP, va="top")
skips = [
    ("cRBEF 跨数据集专家（item 2 核心）", "需按原始 4096 维契约重做特征+训练，多日工程且谱系仍不清；改叙事：M3ED 完整系统 + 其余组件级证据"),
    ("弃权变体扩 10 seeds", "E6≈E5、硬拒绝 0 触发，扩 seed 无新信息；做软门控归因更值"),
    ("CMU-MOSI / CH-SIMS_v2", "原计划即标 additional，与核心 claim 无关"),
    ("完整 OOF 证据链", "工程审计性质，rebuttal 阶段补，不阻塞投稿"),
]
yy = by + bh - 6.2
for t, r in skips:
    ax.text(x + 1.6, yy, t, fontsize=8.8, fontweight="bold", color=C_TEXT, va="top")
    n = wrap(r, 21).count("\n") + 1
    ax.text(x + 1.6, yy - 2.6, wrap(r, 21), fontsize=8.0, color=C_SUB, va="top", linespacing=1.35)
    yy -= 2.6 + 1.65 * n + 2.6

# ---------------- 页脚 ----------------
ax.text(2.5, 1.8, "完整数据：08_性能图升级版_20261006/four_dataset_matrix_summary.json ｜ 图表产物：图 A–F（同目录）｜ 服务器：paper_experiments_20261004（80 补种子 runs 零失败）",
        fontsize=7.5, color=C_SUB, va="bottom")

fig.savefig("audit_board_20261006.png", bbox_inches="tight", facecolor="white")
fig.savefig("audit_board_20261006.pdf", bbox_inches="tight", facecolor="white")
print("saved audit_board_20261006.png")
