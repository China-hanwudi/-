from pathlib import Path
import re
from docx import Document


SRC = Path(r"C:\Users\肖田泽宇宙最强1234\Desktop\论文\ReCoMER_Method_中英双语_20261004_扩展版.docx")
OUT = SRC.with_name("ReCoMER_Method_中英双语_20261004_逻辑修订版.docx")


EN_PREFIX = {
    7: "A standard history-attention block has a concrete failure mode: it assigns mass to every valid-looking slot even when a previous utterance is irrelevant or only partially observed. FCCR is introduced to make history a conditional correction rather than an unconditional input. ",
    9: "The design still needs a constraint because an unconstrained feedback residual could amplify a noisy history representation. ",
    11: "A second failure mode is that missing tokens can distort a joint head if masking is applied only at the input boundary. ",
    13: "Uniform averaging or a free learned gate has a specific weakness: it cannot explain why a modality is useful for one utterance but unreliable for another. SABER is designed to replace this opaque preference with evidence anchored to sample-level utility. ",
    14: "To prevent the router target from becoming a hand-written heuristic, we define a measurable subset utility. ",
    16: "The raw subset utilities still have incomparable scales across samples, so the Shapley target must be normalized before it supervises the router. ",
    18: "The resulting scores also require a deployment constraint: unrestricted weights could let one noisy modality dominate or assign negative mass to a present modality. ",
    20: "This bound is necessary for an interpretable ablation and for stable inference. ",
    22: "Ordinary attention has no abstention option: even when all history is harmful, it still distributes probability over available slots. NCHA is introduced to make the model decide whether a historical update should exist at all. ",
    24: "The null candidate alone is not sufficient, because a soft null score does not define a stable deployment decision. ",
    26: "A soft null mass also needs a structural constraint; otherwise rejected history could re-enter through another memory path. ",
    28: "The hard inference rule is therefore a deliberate safety boundary, not an additional prediction target. ",
    30: "A single MHnoU predictor can still be confidently wrong when lexical context dominates a nonverbal cue. cRBEF is introduced as an independent external expert so that the final system receives a second, differently constructed opinion. ",
    31: "The external branch must first be kept structurally independent; otherwise the proposed second opinion would merely duplicate MHnoU features. ",
    32: "The split between TAV and AV is also a constraint: the AV branch must remain text-free if it is to provide independent nonverbal evidence. ",
    33: "A naive probability average cannot express class-specific support, because the AV branch may support one class while contradicting another. ",
    35: "The gate is therefore constrained to use reliability statistics rather than raw feature concatenation. ",
    36: "The final evidence equation makes the intended limiting cases explicit. ",
    38: "After obtaining two complete peer predictions, simple averaging still has a concrete limitation: it cannot use the internal reference changes or the relative modality evidence to resolve a disagreement. ",
    40: "Because equal averaging is intentionally conservative, the corrector needs a separate evidence channel that explains a disagreement without changing either expert. ",
    42: "An unconstrained MLP over concatenated predictions could learn arbitrary class transfers and overfit the fitting split. ",
    44: "Even a well-trained scorer can be unsafe if it is allowed to move probability to every class. ",
    46: "A correction that can replace the baseline completely would also erase the reliable cases of the two experts. ",
    48: "The final constraint is a recoverable baseline: the fitted mechanism must be able to return exactly to equal averaging when evidence is insufficient. ",
}

EN_SUFFIX = {
    7: " We write this feedback-conditioned update in Eq. (2): the first term is the current modality state, the second term is the masked history summary, and γ_m controls the residual strength.",
    9: " Eq. (3) therefore describes a single reassessment step: auxiliary modality logits are computed after feedback, and the resulting evidence is passed to the router before joint prediction.",
    14: " Equation (4) makes this definition explicit: the utility of a subset is the log probability of the observed class, with the empty utility fixed to zero.",
    16: " Equation (5) then converts these marginal gains into a standardized three-component supervision target.",
    18: " Equation (6) maps the standardized estimates to bounded weights and keeps the support of the available-modality mask separate from the learned preference.",
    22: " Equation (7) defines the utility-derived keep probability and its role in calibrating the empty candidate.",
    24: " Equation (8) shows the second attention normalization: real history and the null candidate compete, but only real history contributes a value vector.",
    26: " Equation (9) expresses the straight-through hard decision used during training and the binary decision used at inference.",
    33: " Equation (10) uses this evidence to form a class-wise log-probability correction rather than a global scalar average.",
    38: " Equation (11) records this equal-probability baseline explicitly so that the learned module has a transparent reference.",
    40: " Equation (12) defines the centered relative evidence supplied to the outer corrector.",
    42: " Equation (13) standardizes these 23 features and produces a bounded class-wise residual score.",
    44: " Equation (14) reallocates only the probability mass inside the candidate set.",
    46: " Equation (15) interpolates the corrected distribution with the baseline using the selected η.",
}

ZH_PREFIX = {
    65: "普通历史注意力存在一个具体不足：只要历史槽位有效，就会分配注意力，即使上一轮话语与当前情绪无关或只包含部分模态。FCCR 的设计目标是把历史变成有条件的校正，而不是无条件输入。",
    67: "该设计还需要约束，因为不受限制的反馈残差可能放大噪声历史表示。",
    69: "另一个问题是，如果只在输入端做掩码，缺失 token 仍可能在联合预测头中改变表示。",
    71: "均匀平均或自由学习的门控有一个具体不足：它无法解释同一模态为什么对一个话语有用、对另一个话语却不可靠。SABER 因此将模态偏好锚定到样本级效用证据上。",
    72: "为了避免路由器监督变成人为指定的启发式规则，我们先定义可测量的子集效用。",
    74: "不同样本的原始子集效用尺度并不一致，因此 Shapley 目标在监督路由器之前还必须进行标准化。",
    76: "得到的分数还需要部署约束，否则噪声模态可能无限制地主导预测，甚至让有效模态得到负质量。",
    78: "这一有界设计既是稳定推理的需要，也是后续消融中可解释的控制条件。",
    80: "普通注意力没有弃权选项：即使所有历史都可能有害，也会在有效槽位之间分配概率。NCHA 的设计目的就是让模型先判断是否应该产生历史更新。",
    82: "但仅有空候选仍然不够，因为软空候选分数不能直接给出稳定的部署决策。",
    84: "空候选质量还需要结构性约束，否则被拒绝的历史可能通过另一条记忆路径重新进入模型。",
    86: "因此，推理阶段的硬门是安全边界，而不是额外的预测标签。",
    88: "即使 MHnoU 已经建模历史，当词义线索压过非语言线索时，单一主预测器仍可能高置信度地出错。cRBEF 被设计为独立外部专家，为最终系统提供结构不同的第二意见。",
    89: "外部分支首先必须保持结构独立，否则所谓第二意见只是在重复 MHnoU 特征。",
    90: "TAV 与 AV 的分支划分同样是一种约束：如果 AV 要提供独立的非语言证据，它就不能接收文本输入。",
    91: "简单的概率平均无法表达类别级支持，因为 AV 可能支持某个类别、同时反驳另一个类别。",
    93: "因此，门控被约束为只使用可靠性统计量，而不是直接拼接原始特征。",
    94: "最终证据公式明确给出了两个极限情况。",
    96: "得到两个完整专家预测后，直接平均仍有一个具体不足：它不能利用内部参考变化和相对模态证据来解决专家分歧。",
    98: "由于等权平均本身是保守基线，校正器还需要独立证据通道来解释分歧，但不能改写任何一个专家。",
    100: "如果把预测直接交给不受限制的 MLP，模型可能学习任意类别转移并在拟合集上过拟合。",
    102: "即使评分器训练良好，允许它对所有类别重新分配概率也可能产生不安全的跳变。",
    104: "如果校正可以完全替代基线，两个专家原本可靠的样本也可能被破坏。",
    106: "最后还需要一个可恢复的基线约束：当证据不足时，机制必须能够严格返回等权平均。",
}

ZH_SUFFIX = {
    65: "公式（2）给出这一反馈更新：第一项是当前模态状态，第二项是带掩码的历史摘要，γ_m 控制残差强度。",
    67: "因此，公式（3）表示一次反馈后的重评估：先由辅助模态头产生 logits，再把更新后的证据交给路由器和联合预测器。",
    72: "公式（4）明确规定，子集效用是真实类别的对数概率，空集效用固定为零。",
    74: "公式（5）将这些边际增益转换为三个模态的标准化监督目标。",
    76: "公式（6）把标准化估计映射为有界权重，并使可用性掩码与学习到的偏好保持独立。",
    80: "公式（7）定义效用驱动的保留概率及其对空候选校准的作用。",
    82: "公式（8）给出第二次注意力归一化：真实历史与空候选竞争，但只有真实历史提供值向量。",
    84: "公式（9）表示训练时的直通硬决策以及推理时使用的二值决策。",
    91: "公式（10）使用该证据生成类别级对数概率校正，而不是全局标量平均。",
    96: "公式（11）显式记录等权概率基线，使学习模块拥有透明参照。",
    98: "公式（12）定义输入外层校正器的中心化相对证据。",
    100: "公式（13）对 23 个特征进行标准化并输出有界的类别级残差。",
    102: "公式（14）只重新分配候选集合内部的概率质量。",
    104: "公式（15）使用选定的 η 在校正分布与基线之间插值。",
}


def split_sentences(text):
    return re.split(r"(?<=[.!?。！？])\s+", text.strip())


def set_text(paragraph, text):
    if not paragraph.runs:
        paragraph.add_run(text)
        return
    paragraph.runs[0].text = text
    for run in paragraph.runs[1:]:
        run.text = ""


def main():
    doc = Document(SRC)
    english_ids = list(EN_PREFIX)
    chinese_ids = list(ZH_PREFIX)
    # Add the requested logic to every prose paragraph, then remove one
    # redundant closing sentence so the English 3.2--3.6 block stays within
    # the previously requested 2,500--3,000 word range.
    for idx in english_ids:
        text = EN_PREFIX[idx] + doc.paragraphs[idx].text
        parts = split_sentences(text)
        if len(parts) > 3:
            text = " ".join(parts[:-1])
        text += EN_SUFFIX.get(idx, "")
        set_text(doc.paragraphs[idx], text)
    for idx in chinese_ids:
        text = ZH_PREFIX[idx] + doc.paragraphs[idx].text
        parts = split_sentences(text)
        if len(parts) > 3:
            text = "".join(parts[:-1])
        text += ZH_SUFFIX.get(idx, "")
        set_text(doc.paragraphs[idx], text)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
