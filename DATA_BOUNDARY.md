# 数据与发布边界

本仓库是公开科研代码和聚合证据快照，不是数据镜像。

## 可以进入仓库

- 自行编写的实验代码、合同测试和冻结配置；
- 不含单条样本身份或媒体内容的聚合统计结果；
- 由聚合结果生成的科研图与作图源数据；
- 数据集官方入口、许可状态和复现说明；
- 不包含受限材料正文的 SHA-256 复现标识。

## 禁止进入仓库

- EmotionTalk、MELD、IEMOCAP、CPED、M3ED 的原始或重分发文本、标签、音频、视频和压缩包；
- 逐查询结果表、说话人/对话键、转写表、媒体索引和可恢复单条样本的信息；
- WavLM、DINOv2 或其他模型权重；
- 音视频派生特征（`.npz`/`.npy`）、训练 bundle（`.joblib`）和 checkpoints；
- IEMOCAP release form、DUA、伦理材料、授权邮件或机构信息；
- GitHub/Hugging Face token、密码、Cookie、私钥、环境变量文件；
- 微信聊天、截图、未获授权的共同作者材料和私人路径。

## 当前公开结果的边界

`results/emotiontalk_multimodal_external_v1.json` 是冻结 validation 的聚合与分组统计结果，不含原始媒体或逐查询记录。`results/emotiontalk_external_source_data.csv` 只包含公开图件所需的聚合数值。

EmotionTalk test 和 MELD test 尚未启封。任何后续 test 评估都必须在论文方案、代码、阈值和统计门最终冻结后另行执行，不能反向调参。

## 数据获取

研究者必须从数据集官方入口自行申请或下载，并独立遵守许可、署名、使用和再分发条件。仓库的 MIT License 只覆盖本仓库自行编写的代码和文档，不改变第三方数据、模型或媒体的许可证。
