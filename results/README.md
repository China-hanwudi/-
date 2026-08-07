# 结果文件说明

| 文件 | 内容 | 是否包含单条样本 |
|---|---|---|
| `emotiontalk_multimodal_external_v1.json` | 一次性EmotionTalk validation聚合结果、聚类CI、模态消融、selector、风险覆盖、质量敏感性和判门 | 否 |
| `emotiontalk_multimodal_external_v1_train_only.json` | validation打开前的训练与calibration诊断、配置/特征/bundle哈希 | 否 |
| `emotiontalk_external_source_data.csv` | `assets/emotiontalk_external_confirmation.png`使用的聚合作图源数据 | 否 |

未公开的文件包括逐查询CSV、说话人/对话键、媒体索引、原始转写、音视频、特征NPZ和训练bundle。

主结果JSON的解释合同要求：工程成功不等于CARMA假设通过；所有预冻结门和消融必须无论方向完整报告；质量分层不得用于静默删除样本；test仍保持封存。
