# 6. Conclusion

Multimodal conversational emotion recognition is difficult because the reliability of historical context and individual modalities changes from one utterance to another. ReCoMER addresses this problem with a structured conditional-fusion framework: MHnoU provides the history-aware main prediction, EvidenceRouter estimates sample-specific modality contributions with Shapley-anchored supervision, the history module can abstain through an explicit empty candidate, and cRBEF supplies an external probability expert that is combined only at the outer fusion stage. This separation makes the information flow auditable and allows the contribution of each component to be tested independently.

Under the frozen formal M3ED test protocol, ReCoMER achieved 58.00% WF1 and improved over matched equal-weight fusion by 0.607 percentage points. The module-isolation studies further showed conditional benefits: historical feedback produced its clearest gain on MOSEI, EvidenceRouter improved MOSEI and MELD under the fixed-backbone protocol, and the corrected abstention variants improved IEMOCAP and MOSEI relative to the original empty-candidate design. The cRBEF/MHnoU information-flow checks also confirmed that cRBEF acts as an external expert rather than an unintended input to the MHnoU modality router.

These results define the scope of the contribution. The fixed cRBEF expert remained slightly above the complete ReCoMER result in the formal test; EvidenceRouter was not beneficial on the tested M3ED representation; and the abstention variants did not improve every dataset. The evidence therefore does not support a claim of universal gains from every module. Instead, it indicates that history compensation, contribution routing and expert fusion are most useful when the corresponding context or modality evidence is reliable, and that their effects depend on the feature regime and evaluation protocol.

Overall, ReCoMER provides a reproducible and interpretable way to combine historical context, modality-specific evidence and an independent expert in multimodal conversational prediction. Its main value is the explicit separation of prediction, history admission, contribution estimation and expert fusion, together with diagnostics that expose when these signals help or fail. Future work should evaluate all variants under a single unified feature protocol, complete same-protocol external baselines and subgroup analyses, and calibrate history rejection and expert weighting for more stable cross-dataset transfer.

## 中文参考译文

多模态对话情感识别的难点在于：历史上下文和各模态信息的可靠性会随话语样本发生变化。ReCoMER 针对这一问题构建了条件融合框架：由 MHnoU 负责基于历史信息的主预测，由带有 Shapley 监督的 EvidenceRouter 估计样本级模态贡献，由显式空候选实现历史信息的拒绝，由 cRBEF 作为独立的概率专家，并且只在外层融合阶段与主模型结合。这样的结构将各部分的信息流分开，使每个模块都能够被单独检验。

在冻结的 M3ED 正式测试协议下，ReCoMER 的 WF1 为 58.00%，比匹配的等权融合高 0.607 个百分点。模块实验进一步表明，历史反馈在 MOSEI 上的收益最清楚，EvidenceRouter 在固定骨干网络条件下改善了 MOSEI 和 MELD，修正后的历史弃权变体相对于原始空候选设计在 IEMOCAP 和 MOSEI 上有所改善。cRBEF 与 MHnoU 的信息流检查也表明，cRBEF 确实作为外部专家参与融合，而不是意外地进入 MHnoU 的模态路由器。

这些结果同时限定了本文结论的范围：正式测试中固定的 cRBEF 专家仍略高于完整 ReCoMER；EvidenceRouter 在当前 M3ED 特征表示上没有带来收益；历史弃权变体也没有在所有数据集上提升。因此，本文不能宣称每个模块都具有普遍增益。更准确的结论是，历史补偿、贡献路由和专家融合在对应的上下文或模态证据可靠时更有价值，其效果会受到特征设置和评估协议的影响。

总体而言，ReCoMER 提供了一种可复现、可解释的融合方式，将历史上下文、模态证据和独立专家结合到多模态对话预测中。它的主要价值在于明确区分预测、历史信息接纳、贡献估计和专家融合，并通过诊断分析揭示这些信号何时有效、何时失效。后续工作应在统一特征协议下完成全部变体评估、补充同协议外部基线和子群分析，并进一步校准历史拒绝与专家权重，以提高跨数据集迁移的稳定性。
