# 补充实验图表说明

- **figE_efficiency_overhead**：冻结测试报告中的参数量分解与 pooled-feature forward 开销；上游 Qwen/audio 编码器不计入该 latency scope。
- **figF_calibration_ece / figF_calibration_brier**：15 等宽 bin 的 ECE 与 multiclass Brier，三份冻结测试报告的均值±样本 SD。
- **figG_missing_modality_robustness**：六组 feature-level zero/mask 压力测试，三 seed 的 weighted-F1 均值±样本 SD；该实验不重新训练编码器，因此不表述为 raw-modality missing benchmark。
- **figH_shapley_mechanism_diagnostic**：由正式结果表中的四数据集 Top-1 contribution-ranking agreement 复现，用于支持 EvidenceRouter 的机制诊断。
- **figI_weight_stability**：MELD 与 M3ED 10 seed 的 evidence_mean_weight 分布；箱体为 seed 分布，中线为中位数，散点为单 seed。
- **figJ_mhnou_branch_classification / figJ_mhnou_branch_mosei**：MHnoU history 与 no-history reference 的三数据集分类 weighted-F1 及 MOSEI MAE 对照。

All plotted values are read from the frozen local experiment summaries; no values are imputed or tuned for visualization.
