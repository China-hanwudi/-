# MELD聚合结果快照

本目录只发布MELD train/dev Pilot的聚合统计和复现标识，不发布原始文本、标签、音频、逐查询结果、说话人／对话键或派生音频特征。

## 文件

- `meld_audio_text_pilot_summary.json`：从冻结结果提取的机器可读聚合摘要。

## 证据边界

- 协议：`meld_audio_text_feasibility_v1`。
- 数据：MELD官方train/dev文本＋真实WAV轻量声学特征。
- 可对齐样本：train 9,988条、dev 1,108条；有历史dev查询765条。
- 随机种子：17、29、43、71、101。
- 置信区间：2,000次按dialogue聚类bootstrap，seed 20260805。
- MELD test：未打开。
- 冻结完整结果JSON SHA-256：`ccc6b1937e7d68eda9033c646152e472d1c294cc76c3e60a6a88cdae94f51943`。
- 独立重复运行得到相同SHA-256；1,108×15逐查询表逐值相同、最大绝对差为0。逐查询表因数据发布边界不上传。

结果解释见[`../../docs/04_MELD音频文本Pilot结果.md`](../../docs/04_MELD音频文本Pilot结果.md)。
