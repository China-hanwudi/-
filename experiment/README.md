# 实验代码说明

本目录公开当前 EmotionTalk 三模态外部确认的核心实现、冻结配置和不依赖原始数据的合同测试。

2026-08-08 新增教师三点整合后的 train-only 方法合同：

- `configs/bidirectional_emotion_utility_v1.json`：不同集合双向效用、六流情感编码、3×3关系和GO/STOP门；
- `src/hva_affect/bidirectional_emotion_utility.py`：非平凡联盟任务、双向目标、3×3关系与VAD/shift特征；
- `tests/test_bidirectional_emotion_utility.py`：退化双向、标签泄漏、维度与可重复性合同。

这些文件表示新方法**可以开始生成 train-only OOF 子集监督**，不表示新方法已经通过真实数据验证。

## 目录

- `configs/`：媒体特征与一次性validation协议；
- `src/hva_affect/`：时间安全历史、base模型、selector、校准和统计实现；
- `scripts/`：编码器基准、特征提取、train/freeze/validate和制图入口；
- `tests/`：使用合成小数组验证历史置零、模态schema、严格过去和受限置换等合同。

## 安装与无数据测试

建议使用 Python 3.11/3.12：

```powershell
python -m venv .venv
$researchPython = (Resolve-Path '.venv\Scripts\python.exe').Path
& $researchPython -m pip install --upgrade pip
& $researchPython -m pip install -r experiment\requirements-multimodal.txt
& $researchPython -m pytest experiment\tests -q
```

提取WavLM/DINOv2媒体特征时还需安装与本机CUDA兼容的PyTorch，再安装：

```powershell
& $researchPython -m pip install -r experiment\requirements-emotiontalk-media.txt
```

## 完整实验的外部输入

完整复现需要研究者自行合法获得：

- EmotionTalk官方元数据、转写和标签容器；
- 原始音频与视频归档；
- 固定revision的 `microsoft/wavlm-base-plus`；
- 固定revision的 `facebook/dinov2-small`。

这些文件、模型权重、派生特征和训练bundle均不在仓库中。

## 三阶段冻结运行

准备好 `emotiontalk_media_features_v1.npz` 后，从仓库根目录依次执行：

```powershell
& $researchPython experiment\scripts\run_emotiontalk_multimodal_external.py train `
  --data-dir <EmotionTalk-mm-process目录>

& $researchPython experiment\scripts\run_emotiontalk_multimodal_external.py freeze `
  --data-dir <EmotionTalk-mm-process目录>

& $researchPython experiment\scripts\run_emotiontalk_multimodal_external.py validate `
  --data-dir <EmotionTalk-mm-process目录>
```

`validate`被设计为一次性写入：结果文件已存在时拒绝覆盖；freeze manifest、配置和bundle不一致时拒绝运行；test split请求会fail closed。

## 已公开的冻结证据

- 配置：`configs/emotiontalk_multimodal_external_v1.json`；
- 媒体特征合同：`configs/emotiontalk_media_features_v1.json`；
- 聚合validation结果：`../results/emotiontalk_multimodal_external_v1.json`；
- 作图源数据：`../results/emotiontalk_external_source_data.csv`；
- 完整解释：`../docs/05_EmotionTalk三模态外部确认结果.md`。

逐查询表、特征NPZ、bundle和原始数据不公开。参见 [`../DATA_BOUNDARY.md`](../DATA_BOUNDARY.md)。
