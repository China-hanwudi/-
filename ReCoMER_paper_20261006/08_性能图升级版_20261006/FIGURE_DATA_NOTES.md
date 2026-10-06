# 补充实验图表数据说明

本批图表按用户给出的六项缺口组织，并从冻结的本地实验汇总中直接读取数值：

- **效率/开销**：三份正式测试 `COMPUTE.json` 的参数量与 feature-level forward latency；上游文本/音频编码器不在该 latency scope 内。
- **校准**：15 等宽 bin 的 ECE 与 multiclass Brier，使用三份冻结测试报告的均值和样本标准差。
- **缺失模态鲁棒性**：T/A/V/TA/TV/AV 六组 feature-level zero/mask stress，未重新训练编码器；因此图注明确写成压力测试，不冒充 raw-modality missing benchmark。
- **Shapley 机制诊断**：采用正式结果表中的四数据集 Top-1 contribution-ranking agreement（MELD 72.06%、IEMOCAP 63.78%、M3ED 44.54%、MOSEI 42.96%）。
- **模态权重稳定性**：MELD 与 M3ED 的十 seed `evidence_mean_weight`，箱体显示 seed 分布，散点显示单 seed。
- **MHnoU 分支汇总**：分类数据集使用 weighted-F1，MOSEI 使用 MAE；history 与 no-history reference 分开绘制，避免混用指标。

所有源数据 CSV 与图注均随图保存于本目录；没有为绘图调参、插值或补造实验结果。
