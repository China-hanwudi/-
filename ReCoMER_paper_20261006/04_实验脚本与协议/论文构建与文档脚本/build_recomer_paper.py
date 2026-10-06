# -*- coding: utf-8 -*-
"""Assemble a clean English ReCoMER manuscript from the supplied project files."""
from io import BytesIO
from pathlib import Path
import glob
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(r"C:\Users\肖田泽宇宙最强1234\Desktop\MHnoU_项目接力包_20261001")
PAPER = Path(r"C:\Users\肖田泽宇宙最强1234\Desktop\论文")
FIG_DIR = PAPER / "01_本轮定稿_7张"
RESULT_DIR = ROOT / "结构图" / "论文结构图_v3_具体表达_20261004"
OUT = PAPER / "ReCoMER_完整论文_20261005.docx"


def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tcPr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_border(cell, color="D9D9D9", sz="4"):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    borders = tcPr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:" + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), sz)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_cell_text(cell, text, bold=False, color="000000", size=9):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if len(str(text)) < 20 else WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(str(text))
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(doc, rows, caption=None, widths=None):
    if caption:
        p = doc.add_paragraph(style="Caption")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(caption)
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = table.rows[i].cells[j]
            set_cell_border(cell)
            set_cell_text(cell, value, bold=(i == 0), color=("FFFFFF" if i == 0 else "000000"), size=8.5)
            if i == 0:
                set_cell_shading(cell, "355C7D")
            elif i % 2 == 0:
                set_cell_shading(cell, "F2F6F8")
    if widths:
        for row in table.rows:
            for j, width in enumerate(widths):
                row.cells[j].width = Inches(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def add_caption(doc, text):
    p = doc.add_paragraph(style="Caption")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(text)
    return p


def add_figure(doc, path, caption, width=6.25):
    if not path.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width))
    add_caption(doc, caption)


def add_body(doc, text, italic=False):
    if not text.strip():
        return
    p = doc.add_paragraph(style="Body Text")
    p.paragraph_format.first_line_indent = Inches(0.18)
    p.paragraph_format.line_spacing = 1.05
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text.strip())
    r.italic = italic
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_after = Pt(3)
    p.add_run(text.strip())


def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    p.paragraph_format.keep_with_next = True
    return p


def configure_styles(doc):
    sec = doc.sections[0]
    sec.top_margin = Inches(0.68)
    sec.bottom_margin = Inches(0.65)
    sec.left_margin = Inches(0.72)
    sec.right_margin = Inches(0.72)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.font.size = Pt(9.5)
    body = styles["Body Text"]
    body.font.name = "Times New Roman"
    body._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    body._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    body.font.size = Pt(9.5)
    for name, size in [("Heading 1", 13), ("Heading 2", 11), ("Heading 3", 10)]:
        s = styles[name]
        s.font.name = "Times New Roman"
        s._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
        s._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
        s.font.size = Pt(size)
        s.font.bold = True
        s.font.color.rgb = RGBColor(0, 0, 0)
        s.paragraph_format.space_before = Pt(8 if name == "Heading 1" else 5)
        s.paragraph_format.space_after = Pt(3)
    title_style = styles["Title"]
    title_style.font.name = "Times New Roman"
    title_style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    title_style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    title_style.font.size = Pt(17)
    title_style.font.bold = True
    title_style.font.color.rgb = RGBColor(0, 0, 0)
    title_style.paragraph_format.space_after = Pt(4)
    style_ppr = title_style._element.get_or_add_pPr()
    style_pbdr = style_ppr.find(qn("w:pBdr"))
    if style_pbdr is not None:
        style_ppr.remove(style_pbdr)
    cap = styles["Caption"]
    cap.font.name = "Times New Roman"
    cap._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    cap._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    cap.font.size = Pt(8.5)
    cap.font.italic = False
    cap.font.color.rgb = RGBColor(40, 40, 40)
    cap.paragraph_format.space_before = Pt(1)
    cap.paragraph_format.space_after = Pt(6)
    # Page number field.
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_run = footer.add_run("ReCoMER  |  ")
    footer_run.font.size = Pt(8)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    footer._p.append(fld)


def first_english_block(doc):
    paras = [p for p in doc.paragraphs]
    out = []
    seen_english = False
    for p in paras:
        t = p.text.strip()
        if t == "English":
            seen_english = True
            continue
        if seen_english and (t == "中文" or t.startswith("中文")):
            break
        if seen_english:
            out.append(p)
    return out


def add_method_from_source(doc, figures=None):
    src = sorted(PAPER.glob("ReCoMER_Method*2700*.docx"))[0]
    source = Document(str(src))
    for p in first_english_block(source):
        text = p.text.strip()
        if not text:
            continue
        if text.startswith("3."):
            level = 1 if re.match(r"3\.\d+\s", text) else 2
            add_heading(doc, text, level=level)
            if figures and text.split()[0] in figures:
                path, caption, width = figures[text.split()[0]]
                add_figure(doc, path, caption, width)
            continue
        if re.fullmatch(r"\(?\d+\)?", text):
            drawing = p._p.xpath(".//w:drawing")
            if drawing:
                blip = drawing[0].xpath(".//a:blip")
                if blip:
                    rid = blip[0].get(qn("r:embed"))
                    part = source.part.related_parts.get(rid)
                    if part:
                        para = doc.add_paragraph()
                        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        run = para.add_run()
                        run.add_picture(BytesIO(part.blob), width=Inches(3.85))
                        add_caption(doc, f"Equation {text.strip().strip('()')}")
                        continue
        add_body(doc, text)


def add_related_work(doc):
    rel = next(PAPER.glob("04_Related_Work*.txt"))
    text = rel.read_text(encoding="utf-8")
    eng = text.split("二、中文", 1)[0]
    eng = eng.split("2. Related Work", 1)[1]
    for raw in eng.splitlines():
        line = raw.strip()
        if not line or line.startswith("2026") or line.startswith("一、English"):
            continue
        if line.startswith("2."):
            add_heading(doc, line, level=2 if line.count(".") > 1 else 1)
        elif line.startswith("Recent multimodal"):
            add_body(doc, line)
        elif not line.startswith("三、"):
            add_body(doc, line)
    add_body(doc, "Recent conversation models also provide complementary context baselines. DialogueCRN [14] performs iterative contextual reasoning over retrieved emotional clues, DialogueGCN [15] models self- and inter-speaker dependencies as a graph, and COSMIC [16] injects commonsense relations such as intents, events and mental states. TelME [21] uses language-led cross-modal distillation and shifting fusion to strengthen non-verbal students. These methods improve contextual representation or cross-modal transfer, whereas ReCoMER focuses on the joint control of history admission, contribution supervision and an independent prediction-level expert.")


def add_experiments(doc):
    exp = next(PAPER.glob("05_Experiments*.md"))
    lines = exp.read_text(encoding="utf-8").splitlines()
    add_heading(doc, "4 Experiments", level=1)
    i = 1
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("# 5. Experiments"):
            i += 1
            continue
        if line.startswith("## Data and audit sources"):
            break
        if line.startswith("## "):
            heading = line[3:].strip().replace("5.", "4.", 1)
            add_heading(doc, heading, level=2)
            result_figures = {
                "4.2": (RESULT_DIR / "07_ReCoMER_results_overview_image2_style.png", "Figure 8. Formal M3ED comparison, conditional module effects and paired integrated ablations.", 6.25),
                "4.7": (RESULT_DIR / "08_ReCoMER_exact_feature_checks_image2_style.png", "Figure 9. Exact-feature M3ED controls and cRBEF replication.", 6.25),
                "4.8": (RESULT_DIR / "09_ReCoMER_gate_sensitivity_image2_style.png", "Figure 10. Frozen cRBEF gate-formula sensitivity; this is a discrete formula robustness check rather than a continuous hyperparameter sweep.", 6.25),
                "4.9": (RESULT_DIR / "10_ReCoMER_modality_pca_image2_style.png", "Figure 11. PCA projections of exact M3ED test features for text, audio and vision, colored by ground-truth emotion.", 6.25),
            }
            key = heading.split()[0]
            if key in result_figures:
                add_figure(doc, *result_figures[key])
                if key == "4.9":
                    add_figure(doc, RESULT_DIR / "11_ReCoMER_confusion_matrices_image2_style.png", "Figure 12. Mean row-normalized confusion structure for MHnoU, cRBEF and ReCoMER across the three formal M3ED test seeds.", 6.25)
            i += 1
            continue
        if line.startswith("**Table "):
            cap = line.strip("*")
            table_lines = []
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            while j < len(lines) and lines[j].strip().startswith("|"):
                table_lines.append(lines[j].strip())
                j += 1
            if len(table_lines) >= 2:
                rows = []
                for tl in table_lines:
                    cells = [c.strip() for c in tl.strip("|").split("|")]
                    if all(set(c) <= set("-:") for c in cells):
                        continue
                    rows.append(cells)
                if rows:
                    add_table(doc, rows, cap, widths=[2.7] + [1.0] * (len(rows[0]) - 1))
            i = j
            continue
        if line.startswith("-"):
            add_bullet(doc, line[1:].strip())
            i += 1
            continue
        if line.startswith("**") and line.endswith("**"):
            add_body(doc, line.strip("*"), italic=True)
            i += 1
            continue
        if not line.startswith("|"):
            add_body(doc, line)
        i += 1


def add_figures_and_tables(doc):
    add_figure(doc, FIG_DIR / "00_概念动机_v02_定稿.png", "Figure 1. Motivation: modality evidence and historical context can be useful, redundant or misleading depending on the utterance.", 5.9)
    add_heading(doc, "3 Method", level=1)
    add_heading(doc, "3.1 Task formulation and framework overview", level=2)
    add_body(doc, "Given an utterance at turn t, ReCoMER predicts an emotion label from current text, audio and visual features together with a bounded history window. The framework keeps two prediction paths structurally separate: MHnoU performs selective contextual reasoning, while cRBEF supplies an independent probability expert. A bounded outer correction combines complete predictions only after both paths have produced their own class distributions.")
    add_figure(doc, FIG_DIR / "01_ReCoMER总体框架_v05_定稿.png", "Figure 2. ReCoMER overview. MHnoU and cRBEF remain parallel; the outer module performs bounded class-wise fusion.", 6.1)
    add_method_from_source(doc)
    # Insert the remaining conceptual figures near their corresponding method sections.
    # The source method text is complete; these figures are added as visual anchors after the method block.
    add_figure(doc, FIG_DIR / "02_MHnoU_v03_定稿.png", "Figure 3. MHnoU: feedback-conditioned reassessment, contribution routing and history admission.", 5.9)
    add_figure(doc, FIG_DIR / "03_SABER_v03_定稿.png", "Figure 4. SABER: Shapley-anchored contribution supervision followed by bounded modality routing.", 5.9)
    add_figure(doc, FIG_DIR / "04_NCHA_v03_定稿.png", "Figure 5. NCHA: real history competes with an explicit null candidate before hard admission.", 5.9)
    add_figure(doc, FIG_DIR / "05_cRBEF_v02_定稿.png", "Figure 6. cRBEF: prior-relative audio-visual evidence is regulated by a reliability gate.", 5.9)
    add_figure(doc, FIG_DIR / "06_局部类别校正_v04_定稿.png", "Figure 7. Local class-wise fusion: candidate-restricted correction with an exact equal-weight fallback.", 5.9)


def add_results_figures(doc):
    add_figure(doc, RESULT_DIR / "07_ReCoMER_results_overview_image2_style.png", "Figure 8. Formal M3ED comparison, conditional module effects and paired integrated ablations.", 6.25)
    add_figure(doc, RESULT_DIR / "08_ReCoMER_exact_feature_checks_image2_style.png", "Figure 9. Exact-feature M3ED controls and cRBEF replication.", 6.25)
    add_figure(doc, RESULT_DIR / "09_ReCoMER_gate_sensitivity_image2_style.png", "Figure 10. Frozen cRBEF gate-formula sensitivity; this is a discrete formula robustness check rather than a continuous hyperparameter sweep.", 6.25)
    add_figure(doc, RESULT_DIR / "10_ReCoMER_modality_pca_image2_style.png", "Figure 11. PCA projections of exact M3ED test features for text, audio and vision, colored by ground-truth emotion.", 6.25)
    add_figure(doc, RESULT_DIR / "11_ReCoMER_confusion_matrices_image2_style.png", "Figure 12. Mean row-normalized confusion structure for MHnoU, cRBEF and ReCoMER across the three formal M3ED test seeds.", 6.25)


def add_conclusion(doc):
    add_heading(doc, "5 Conclusion", level=1)
    conc = next(PAPER.glob("06_Conclusion*.md"))
    text = conc.read_text(encoding="utf-8")
    eng = text.split("## 中文", 1)[0]
    for line in eng.splitlines():
        t = line.strip()
        if not t or t.startswith("# 6."):
            continue
        add_body(doc, t)


def add_references(doc):
    add_heading(doc, "References", level=1)
    refs = [
        "[1] Majumder, N., Poria, S., Hazarika, D., Mihalcea, R., Gelbukh, A., and Cambria, E. DialogueRNN: An Attentive RNN for Emotion Detection in Conversations. AAAI, 2019, 33(01), 6818–6825.",
        "[2] Hu, J., Liu, Y., Zhao, J., and Jin, Q. MMGCN: Multimodal Fusion via Deep Graph Convolution Network for Emotion Recognition in Conversation. ACL-IJCNLP, 2021, 5666–5675.",
        "[3] Chen, F., Shao, J., Zhu, S., and Shen, H. T. Multivariate, Multi-Frequency and Multimodal: Rethinking Graph Neural Networks for Emotion Recognition in Conversation. CVPR, 2023, 10761–10770.",
        "[4] Zhang, T. and Tan, Z. ECERC: Evidence-Cause Attention Network for Multi-Modal Emotion Recognition in Conversation. ACL, 2025, 2064–2077.",
        "[5] Li, G., Guo, H., Liu, M., Wei, Z., Gu, B., Gu, T., and Ma, D. HiRoC: Selective History Routing and Disagreement-Aware Calibration for Multimodal Conversational Emotion Recognition. Journal of King Saud University Computer and Information Sciences, 38, 770, 2026.",
        "[6] Li, B., Fei, H., Liao, L., Zhao, Y., Teng, C., Chua, T.-S., Ji, D., and Li, F. Revisiting Disentanglement and Fusion on Modality and Context in Conversational Multimodal Emotion Recognition. ACM Multimedia, 2023.",
        "[7] Peng, X., Wei, Y., Deng, A., Wang, D., and Hu, D. Balanced Multimodal Learning via On-the-Fly Gradient Modulation. CVPR, 2022, 8238–8247.",
        "[8] Fan, Y., Xu, W., Wang, H., Wang, J., and Guo, S. PMR: Prototypical Modal Rebalance for Multimodal Learning. CVPR, 2023, 20029–20038.",
        "[9] He, K., Chen, B., Ding, Y., Li, F., Teng, C., and Ji, D. PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis. AAAI, 2026, 40(37), 30960–30968.",
        "[10] Fang, Y., Huang, W., Wan, G., Su, K., and Ye, M. EMOE: Modality-Specific Enhanced Dynamic Emotion Experts. CVPR, 2025, 14314–14324.",
        "[11] Bi, L., Zhang, Y., Wang, L., Niu, Y., and Zhao, H. Two Challenges, One Solution: Robust Multimodal Learning through Dynamic Modality Recognition and Enhancement. Findings of EMNLP, 2025, 12855–12867.",
        "[12] Gao, Z., Jiang, X., Xu, X., Shen, F., Li, Y., and Shen, H. T. Embracing Unimodal Aleatoric Uncertainty for Robust Multimodal Fusion. CVPR, 2024, 26876–26885.",
        "[13] Ovanger, O., Harris, L., and Keitt, T. H. Adaptive Evidence Weighting for Audio-Spatiotemporal Fusion. arXiv:2602.03817, 2026.",
        "[14] Hu, D., Wei, L., and Huai, X. DialogueCRN: Contextual Reasoning Networks for Emotion Recognition in Conversations. ACL-IJCNLP, 2021, 7042–7052.",
        "[15] Ghosal, D., Majumder, N., Poria, S., Chhaya, N., and Gelbukh, A. DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation. EMNLP-IJCNLP, 2019.",
        "[16] Ghosal, D., Majumder, N., Gelbukh, A., Mihalcea, R., and Poria, S. COSMIC: COmmonSense knowledge for eMotion Identification in Conversations. Findings of EMNLP, 2020, 2470–2481.",
        "[17] Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., and Mihalcea, R. MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations. ACL, 2019, 527–536.",
        "[18] Zhao, J., Zhang, T., Hu, J., Liu, Y., Jin, Q., Wang, X., and Li, H. M3ED: Multi-modal Multi-scene Multi-label Emotional Dialogue Database. ACL, 2022, 5699–5710.",
        "[19] Busso, C., Bulut, M., Lee, C.-C., Kazemzadeh, A., Mower, E., Kim, S., Chang, J. N., Lee, S., and Narayanan, S. S. IEMOCAP: Interactive Emotional Dyadic Motion Capture Database. Language Resources and Evaluation, 42(4), 335–359, 2008.",
        "[20] Zadeh, A. B., Liang, P. P., Poria, S., Cambria, E., and Morency, L.-P. Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph. ACL, 2018, 2236–2246.",
        "[21] Yun, T., Lim, H., Lee, J., and Song, M. TelME: Teacher-leading Multimodal Fusion Network for Emotion Recognition in Conversation. NAACL-HLT, 2024, 82–95.",
    ]
    for ref in refs:
        p = doc.add_paragraph(style="Body Text")
        p.paragraph_format.left_indent = Inches(0.12)
        p.paragraph_format.first_line_indent = Inches(-0.12)
        p.paragraph_format.space_after = Pt(3)
        p.add_run(ref)


def main():
    doc = Document()
    configure_styles(doc)
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(3)
    title.add_run("ReCoMER: Reliability and Context Modeling for Multimodal Conversational Emotion Recognition")
    title._p.get_or_add_pPr().find(qn("w:pBdr"))
    ppr = title._p.get_or_add_pPr()
    pbdr = ppr.find(qn("w:pBdr"))
    if pbdr is not None:
        ppr.remove(pbdr)
    for run in title.runs:
        run.font.color.rgb = RGBColor(0, 0, 0)
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.paragraph_format.space_after = Pt(14)
    r = sub.add_run("Anonymous Submission")
    r.italic = True
    r.font.size = Pt(10)

    add_heading(doc, "Abstract", level=1)
    add_body(doc, "Multimodal conversational emotion recognition requires a model to decide not only how to combine text, audio and vision, but also whether historical context and an external prediction should be trusted for the current utterance. We present ReCoMER, a reliability- and context-aware framework that combines MHnoU with the independent cRBEF expert. MHnoU integrates feedback-conditioned context reassessment, Shapley-anchored bounded evidence routing and null-candidate history abstention. cRBEF constructs prior-relative audio-visual evidence and regulates it with a reliability gate. A bounded local class-wise fusion module then combines the two complete predictions while preserving an exact equal-weight fallback. On the frozen formal M3ED test, ReCoMER reaches 58.00% WF1 and improves over matched equal-weight fusion by 0.607 percentage points. Module-isolation experiments show conditional gains on MOSEI, MELD and IEMOCAP, while also exposing negative or neutral regimes. The resulting framework separates prediction, history admission, contribution estimation and expert fusion, making both its improvements and failure boundaries auditable.")
    add_body(doc, "Keywords: multimodal emotion recognition; emotion recognition in conversation; modality contribution; history abstention; reliability-aware fusion")

    add_heading(doc, "1 Introduction", level=1)
    intro = next(PAPER.glob("ReCoMER_Introduction*.docx"))
    for p in first_english_block(Document(str(intro))):
        t = p.text.strip()
        if not t or t == "ReCoMER Introduction":
            continue
        if t.startswith("The main contributions"):
            add_body(doc, t)
            continue
        if t.startswith("We propose ReCoMER") or t.startswith("We develop MHnoU") or t.startswith("We incorporate cRBEF"):
            add_bullet(doc, t)
        else:
            add_body(doc, t)
    add_figure(doc, FIG_DIR / "00_概念动机_v02_定稿.png", "Figure 1. Motivation for selective context and modality evidence control.", 5.9)

    add_heading(doc, "2 Related Work", level=1)
    add_related_work(doc)

    # Method and figures.
    add_heading(doc, "3 Method", level=1)
    method_figures = {
        "3.1": (FIG_DIR / "01_ReCoMER总体框架_v05_定稿.png", "Figure 2. ReCoMER framework overview.", 6.1),
        "3.2": (FIG_DIR / "02_MHnoU_v03_定稿.png", "Figure 3. MHnoU history feedback, contribution routing and history admission.", 5.9),
        "3.3": (FIG_DIR / "03_SABER_v03_定稿.png", "Figure 4. SABER contribution supervision and bounded routing.", 5.9),
        "3.4": (FIG_DIR / "04_NCHA_v03_定稿.png", "Figure 5. NCHA null-candidate history abstention.", 5.9),
        "3.5": (FIG_DIR / "05_cRBEF_v02_定稿.png", "Figure 6. cRBEF prior-relative audio-visual evidence and reliability gating.", 5.9),
        "3.6": (FIG_DIR / "06_局部类别校正_v04_定稿.png", "Figure 7. Bounded local class-wise fusion and equal-weight fallback.", 5.9),
    }
    add_method_from_source(doc, method_figures)

    add_experiments(doc)
    add_conclusion(doc)
    add_references(doc)

    # Keep document metadata neutral for later author completion.
    doc.core_properties.title = "ReCoMER: Reliability and Context Modeling for Multimodal Conversational Emotion Recognition"
    doc.core_properties.author = ""
    doc.core_properties.subject = "Multimodal conversational emotion recognition"
    doc.save(str(OUT))
    print(OUT)


if __name__ == "__main__":
    main()
