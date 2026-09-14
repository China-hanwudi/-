# 当前活动主线（TemporalN3 v10 exploratory）

本仓库当前唯一活动模型主线是 `模型/` 下的 TemporalN3 v10，服务于：

- M3ED：七分类，valid Weighted-F1 → Macro-F1 选择 checkpoint；
- CMU-MOSEI：连续情感回归，标签按 `y/3` 归一化，valid MAE 选择 checkpoint；
- CH-SIMS_v2：连续情感回归，按 packed train/valid 合同运行。

`docs/` 中涉及 MELD、IEMOCAP、EmotionTalk 的内容属于历史实验记录和证据，不是当前训练入口、基准或 SOTA 结果。当前活动代码不含这些数据集的训练/评估入口。

v10 是 post-v9-test exploratory model，只允许根据 train/valid 诊断开发。旧 test 不得再次用于选择或调参；正式结论需要新的封存评估协议。

仓库不包含原始数据、预计算特征、封存 test、训练 checkpoint 或私有凭据。正式训练必须在独立输出目录完成，并记录代码、配置、数据清单 hash 和随机种子。
