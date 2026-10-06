# -*- coding: utf-8 -*-
"""实验数据盘点工作簿：总览 + 章节映射 + 四级分类总表 + 必补计划 + 矩阵快照。"""
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

OUT = "实验数据盘点_20261006.xlsx"

FONT = "Microsoft YaHei"
C_TITLE = "2B2B28"
C_HEADER_FILL = "3A3A36"
C_HEADER_TXT = "FFFFFF"
C_CAT = {
    "已具备": ("E8F3EA", "1B7837"),
    "必补": ("F9ECEA", "B23A2F"),
    "可砍": ("FBF3E2", "9A6A0A"),
    "放弃": ("F2F2EF", "6E6E68"),
}
C_ACCENT = "1F5FA8"
THIN = Side(style="thin", color="D8D8D2")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

wb = openpyxl.Workbook()

def style_title(ws, cell, text, size=16):
    ws[cell] = text
    ws[cell].font = Font(name=FONT, size=size, bold=True, color=C_TITLE)

def style_sub(ws, cell, text):
    ws[cell] = text
    ws[cell].font = Font(name=FONT, size=9, color="6E6E68")

def header_row(ws, row, headers, widths=None):
    for i, h in enumerate(headers, start=1):
        c = ws.cell(row=row, column=i, value=h)
        c.font = Font(name=FONT, size=10, bold=True, color=C_HEADER_TXT)
        c.fill = PatternFill("solid", fgColor=C_HEADER_FILL)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BORDER
    if widths:
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = w
    ws.row_dimensions[row].height = 22

def body_cell(ws, row, col, value, wrap=True, bold=False, color=C_TITLE, align="left", fill=None, num_fmt=None):
    c = ws.cell(row=row, column=col, value=value)
    c.font = Font(name=FONT, size=9.5, bold=bold, color=color)
    c.alignment = Alignment(horizontal=align, vertical="top", wrap_text=wrap)
    c.border = BORDER
    if fill:
        c.fill = PatternFill("solid", fgColor=fill)
    if num_fmt:
        c.number_format = num_fmt
    return c

# ================= Sheet 1 总览 =================
ws = wb.active
ws.title = "总览"
ws.sheet_view.showGridLines = False
for col, w in zip("ABCDE", [14, 22, 46, 30, 30]):
    ws.column_dimensions[col].width = w
style_title(ws, "B2", "ReCoMER 实验部分数据盘点", 18)
style_sub(ws, "B3", "来源：论文工作区 2026-10-06 全量核查（交接文档 02 + GPU 服务器 paper_experiments_20261004 实查）")
style_sub(ws, "B4", "口径：valid-only，test_read=False 逐 run 断言；四级分类见「四级分类总表」")

kpis = [("实验数据项", "=COUNTA(四级分类总表!B3:B26)", C_TITLE),
        ("已具备", '=COUNTIF(四级分类总表!A3:A26,"已具备")', "1B7837"),
        ("必须补", '=COUNTIF(四级分类总表!A3:A26,"必补")', "B23A2F"),
        ("可砍+放弃", '=COUNTIF(四级分类总表!A3:A26,"可砍")+COUNTIF(四级分类总表!A3:A26,"放弃")', "9A6A0A")]
for i, (lab, formula, colr) in enumerate(kpis):
    r = 6
    c1 = ws.cell(row=r, column=2 + i, value=lab)
    c1.font = Font(name=FONT, size=10, bold=True, color="6E6E68")
    c1.alignment = Alignment(horizontal="center")
    c2 = ws.cell(row=r + 1, column=2 + i, value=formula)
    c2.font = Font(name=FONT, size=22, bold=True, color=colr)
    c2.alignment = Alignment(horizontal="center")

style_sub(ws, "B10", "一句话结论：今天之后实验数据只剩一个真缺口——同协议外部 SOTA 对比表（+ 一段“完整系统 vs 最强专家”归因文字）。")
ws.merge_cells("B10:E10")
ws["B10"].font = Font(name=FONT, size=11, bold=True, color="B23A2F")
ws["B10"].alignment = Alignment(wrap_text=True, vertical="top")
ws.row_dimensions[10].height = 30

idx = [("章节映射", "实验部分需要哪些数据、按 4.1–4.6 章节的逐项状态"),
       ("四级分类总表", "24 项数据的分类、理由与处置建议（本工作簿的数据主表）"),
       ("必补计划", "仅剩的两项必补工作及执行方案"),
       ("四数据集矩阵", "item 3 补齐后的完整矩阵（四数据集 × 5 臂 × 10 seeds）")]
style_sub(ws, "B12", "工作表索引")
ws["B12"].font = Font(name=FONT, size=10, bold=True, color=C_TITLE)
for i, (name, desc) in enumerate(idx):
    body_cell(ws, 13 + i, 2, name, bold=True)
    body_cell(ws, 13 + i, 3, desc)

# ================= Sheet 2 章节映射 =================
ws = wb.create_sheet("章节映射")
ws.sheet_view.showGridLines = False
style_title(ws, "A1", "实验部分数据需求 × 现状（按论文章节）", 14)
style_sub(ws, "A2", "状态口径：✅ 已具备 ｜ ⚠️ 已有但规格受限（可用）｜ ❌ 缺失")
header_row(ws, 3, ["论文章节", "需要的数据", "状态", "载体（图表/表）", "备注"],
           [12, 52, 10, 22, 40])
rows = [
    ("4.1 Setup", "数据集引用、协议说明、seed 列表", "✅", "正文", "MELD/M3ED/IEMOCAP/MOSEI 引用已挂"),
    ("4.2 主对比", "正式 M3ED 测试六分支 + 配对 bootstrap", "✅", "Table 1 + 图 A", "图 A 增加 Acc/NLL/ECE/Brier 多指标"),
    ("4.2 主对比", "逐类 F1 + 逐类配对 bootstrap CI", "✅", "图 E（今日）", "对 cRBEF：+Happy/Fear、−Sad；对等权：+Sad（显著）"),
    ("4.3 组件隔离", "历史反馈十 seed（3 数据集）", "✅", "Table 2", "条件性正效应，措辞已对齐"),
    ("4.3 组件隔离", "EvidenceRouter 十 seed + Holm 校正", "✅", "Table 3 + 图 B（今日）", "MOSEI 显著；MELD/M3ED 边缘诚实标注"),
    ("4.3 组件隔离", "弃权变体完整矩阵", "⚠️", "Table 4", "3-seed 修正协议已冻结；扩 10 seed 不值得强求"),
    ("4.4 整合消融", "六个整合控制 + 对话 bootstrap", "✅", "Table 5", "弃权项 −0.487 显著，其余 CI 跨 0"),
    ("4.4 审计", "信息流 / 目标梯度 / 身份审计", "✅", "正文 + 附录", "splice 审计通过；单模态头无梯度已声明"),
    ("4.5 exact-feature", "M3ED 十 seed 五臂对照", "✅", "Table 6 + 图 9", "与正式结果严格分离表述"),
    ("4.5 exact-feature", "四数据集 B0 矩阵（item 3）", "✅", "四数据集矩阵表（今日）", "80 补种子 runs 零失败；evidence_solo 10/10 显著"),
    ("4.6 失败边界", "延续/转折子群分析（item 5）", "✅", "图 F（今日）", "延续句 9/9 为正；转折句仅 M3ED 一致为负"),
    ("4.6/全文", "同协议外部 SOTA 对比表", "❌", "缺", "唯一顶刊门槛级缺口；方案见「必补计划」"),
]
r = 4
for sec, need, st, carrier, note in rows:
    body_cell(ws, r, 1, sec, bold=True)
    body_cell(ws, r, 2, need)
    fill, colr = {"✅": C_CAT["已具备"], "⚠️": C_CAT["可砍"], "❌": C_CAT["必补"]}[st]
    body_cell(ws, r, 3, st, align="center", bold=True, color=colr, fill=fill)
    body_cell(ws, r, 4, carrier)
    body_cell(ws, r, 5, note)
    ws.row_dimensions[r].height = 26
    r += 1

# ================= Sheet 3 四级分类总表 =================
ws = wb.create_sheet("四级分类总表")
ws.sheet_view.showGridLines = False
style_title(ws, "A1", "24 项实验数据：四级分类总表", 14)
style_sub(ws, "A2", "分类口径：已具备=进论文；可砍=已做但不必放；必补=投稿前必须完成；放弃=不值得强求")
header_row(ws, 3, ["分类", "项目", "状态说明 / 理由", "处置建议"], [10, 38, 46, 34])
items = [
    ("已具备", "正式 M3ED 测试六分支 + 配对 bootstrap", "58.00±0.34；vs 等权 +0.607 [0.250,1.005] 显著", "Table 1 + 正文"),
    ("已具备", "多指标面板（WF1/Acc/NLL/ECE/Brier）", "校准与 NLL 领先所有融合规则", "图 A（今日）"),
    ("已具备", "逐类 F1 + 逐类配对 bootstrap CI", "融合重分配类间误差：+Happy/Fear、−Sad（vs cRBEF）；+Sad（vs 等权）", "图 E（今日）"),
    ("已具备", "历史反馈十 seed 隔离研究", "MOSEI 最稳；M3ED 7/10；MELD CI 跨 0", "Table 2，条件性措辞"),
    ("已具备", "EvidenceRouter 十 seed + Holm 校正", "MOSEI 对全部对照显著 10/10；MELD/M3ED 边缘", "Table 3 + 图 B（今日）"),
    ("已具备", "弃权变体 3-seed 修正协议（B1/E5/E6/E7）", "E5 稳定优于 B1；E6≈E5；E7 无优势", "Table 4"),
    ("已具备", "六个整合控制消融", "去弃权 −0.487 显著；其余 CI 跨 0", "Table 5"),
    ("已具备", "信息流 / 目标梯度 / 身份审计", "splice 通过；梯度分离确认", "正文 + 附录"),
    ("已具备", "M3ED exact-feature 十 seed 五臂", "排除特征协议混淆", "Table 6 + 图 9"),
    ("已具备", "四数据集 × 5 臂 × 10 seed 矩阵（item 3）", "今日补齐 80 runs 零失败；evidence_solo 唯一一致显著臂", "四数据集矩阵表（今日）"),
    ("已具备", "延续/转折子群分析（item 5）", "延续句 9/9 为正；转折句仅 M3ED 3/3 为负", "图 F（今日）+ 4.6 节支撑"),
    ("已具备", "MOSEI 全指标剖面", "历史反馈六项指标全向有利但幅度小", "图 C / 附录"),
    ("已具备", "融合行为证据（η 与模态权重）", "η=0 回退 1/3 种子；权重贴近均匀先验", "图 D（今日）"),
    ("已具备", "校准指标（ECE15 / Brier）", "ReCoMER ECE 0.046，优于 MHnoU/cRBEF", "并入图 A"),
    ("必补", "同协议外部 SOTA 对比表", "性能叙事无任何外部锚点；DialogueRNN 51.66 只是 context 参照", "复现 2–3 个 M3ED 强基线，方案见「必补计划」"),
    ("必补", "“完整系统 vs 最强专家”归因段落", "58.00 < 58.60 需要机制层面解释，否则 4.2 叙事不闭环", "补写归因段：分支拖累/η 回退/有界容量，不需新实验"),
    ("可砍", "首轮开发集数字（OUTER/valid 61.76 那批）", "已被正式测试取代；交接文档明令禁止与之矛盾的表述", "绝不进论文"),
    ("可砍", "gate 六公式敏感性 + ordering controls", "全部 CI 跨 0，属探索性", "附录一句带过"),
    ("可砍", "semantic/sparsemax 筛查 + jointsoft 臂", "单 seed 筛查信号 / 全面为负", "正文一句即可"),
    ("可砍", "PCA 图、特征压力测试、mpath 臂", "诊断性/不显著；压力测试非标准鲁棒性基准", "附录或不报"),
    ("放弃", "cRBEF 跨数据集专家（item 2 核心）", "需按原始 4096 维契约重做特征+训练，多日工程且谱系仍不清", "改叙事：M3ED 完整系统 + 其余数据集组件级证据"),
    ("放弃", "弃权变体扩到 10 seeds", "E6≈E5、硬拒绝 0 触发，扩 seed 无新信息", "改做软门控归因分析"),
    ("放弃", "CMU-MOSI / CH-SIMS_v2 扩展", "原计划即标 additional", "不做"),
    ("放弃", "完整 OOF 证据链", "工程审计性质", "rebuttal 阶段补，不阻塞投稿"),
]
r = 4
for cat, proj, why, act in items:
    fill, colr = C_CAT[cat]
    body_cell(ws, r, 1, cat, bold=True, color=colr, fill=fill, align="center")
    body_cell(ws, r, 2, proj, bold=True)
    body_cell(ws, r, 3, why)
    body_cell(ws, r, 4, act)
    ws.row_dimensions[r].height = 30
    r += 1

# ================= Sheet 4 必补计划 =================
ws = wb.create_sheet("必补计划")
ws.sheet_view.showGridLines = False
style_title(ws, "A1", "仅剩的两项必补工作", 14)
header_row(ws, 3, ["工作", "为什么要补", "执行方案", "工作量估计"], [26, 40, 52, 18])
plans = [
    ("同协议外部 SOTA 对比表",
     "顶刊门槛：没有同协议外部基线，58.00 这个数字在审稿人眼里没有坐标；DialogueRNN 51.66 是不同特征/协议的 context 参照，不能当锚点",
     "收敛到 M3ED 上 2–3 个可复现强基线（HiRoC / DF-ERC 类）；用现有特征包复现，同一划分同一指标；服务器有 meld_sota_control 残件可借鉴；绝不把 valid 分数与文献 test 分数直接对比",
     "2–4 天（含调参与核对）"),
    ("“完整系统 vs 最强专家”归因段落",
     "58.00 < 58.60 若不解释，4.2 的方法动机（融合增益）与实验结果互相矛盾，审稿人必然追问",
     "归因三要素：① MHnoU 分支仅 45.43，拉低候选集质量；② η 校准在 1/3 种子退回等权；③ 有界修正（λ=0.3）容量受限。写成对融合边界条件的讨论段，不需新实验",
     "0.5 天（纯写作）"),
]
r = 4
for name, why, how, cost in plans:
    body_cell(ws, r, 1, name, bold=True, color="B23A2F")
    body_cell(ws, r, 2, why)
    body_cell(ws, r, 3, how)
    body_cell(ws, r, 4, cost, align="center")
    ws.row_dimensions[r].height = 72
    r += 1

# ================= Sheet 5 四数据集矩阵 =================
ws = wb.create_sheet("四数据集矩阵")
ws.sheet_view.showGridLines = False
style_title(ws, "A1", "四数据集 × 5 臂 × 10 seeds 完整矩阵（item 3，今日补齐）", 14)
style_sub(ws, "A2", "valid-only · test_read=False 逐 run 断言 · 主指标：cls=weighted_f1 ↑，reg=MAE ↓；配对为 seed 级")
header_row(ws, 3, ["数据集", "主指标", "B0 uniform_nohist", "uniform_history", "evidence_closed", "evidence_solo", "evidence_mpath",
                   "evidence_solo vs B0（均值 [95% CI]，胜场，p）"],
           [14, 10, 18, 18, 18, 18, 18, 40])
mat = [
    ("M3ED_strong", "WF1 ↑", 0.5880, 0.5901, 0.5896, 0.6030, 0.5920, "+1.50 pp [+0.91, +2.05]，10/10，p=0.002"),
    ("MELD", "WF1 ↑", 0.6347, 0.6334, 0.6338, 0.6409, 0.6345, "+0.63 pp [−0.12, +1.21]，8/10，p=0.109；vs uniform_history +0.75 pp [+0.20, +1.34]，10/10，p=0.002"),
    ("IEMOCAP", "WF1 ↑", 0.6854, 0.6895, 0.6891, 0.6973, 0.6920, "+1.20 pp，10/10，p=0.002"),
    ("MOSEI_full", "MAE ↓", 0.5108, 0.5091, 0.5090, 0.5068, 0.5112, "MAE 改善 0.0040，8/10，p=0.109"),
]
r = 4
for ds, met, b0, uh, ec, es, em, key in mat:
    body_cell(ws, r, 1, ds, bold=True)
    body_cell(ws, r, 2, met, align="center")
    for j, v in enumerate([b0, uh, ec, es, em]):
        c = body_cell(ws, r, 3 + j, v, align="center", num_fmt="0.0000")
        if j == 3:
            c.font = Font(name=FONT, size=9.5, bold=True, color="1B7837")
    body_cell(ws, r, 8, key)
    ws.row_dimensions[r].height = 30
    r += 1
style_sub(ws, f"A{r+1}", "结论：evidence_solo（deploy=solo_weighted）是唯一一致显著的正增益臂；evidence_closed（论文当前 closed_loop 部署）四数据集均不显著。写进论文前与导师确认部署口径。")
ws.merge_cells(start_row=r + 1, start_column=1, end_row=r + 1, end_column=8)
ws.cell(row=r + 1, column=1).alignment = Alignment(wrap_text=True, vertical="top")
ws.cell(row=r + 1, column=1).font = Font(name=FONT, size=9.5, bold=True, color="B23A2F")
ws.row_dimensions[r + 1].height = 30

wb.save(OUT)
print("saved", OUT)
