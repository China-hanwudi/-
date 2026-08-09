# 模型 / ComposerN3

本目录是面向仓库 N3 主线的**可训练情感分类模型包**。

- 默认骨干：**ComposerN3**（六路 N3 可训练网络，可直接冒烟）
- **推荐文本塔（已上传权重）：** [EmoBERTa-base](artifacts/pretrained/emoberta-base)（`tae898/emoberta-base`，对话情感，MELD/IEMOCAP 导向，上游 MIT）

选型对照见 [`artifacts/pretrained/MODEL_CARD.md`](artifacts/pretrained/MODEL_CARD.md)。

> 边界见 [`DATA_BOUNDARY.md`](../DATA_BOUNDARY.md)（科研协作版）：鼓励上传开源权重与自训 checkpoint；仍禁止未授权的数据集原文镜像与密钥。

## 权重 vs 调参

- **权重**：模型参数数字（本仓已放 EmoBERTa 的 `pytorch_model.bin`）
- **超参数**：lr、batch size 等（在 `configs/`）
- 你们继续训出的 N3 头 → 放 `artifacts/checkpoints/`

## 结构

```text
模型/
  artifacts/pretrained/emoberta-base/  # 已入库推荐权重（Git LFS）
  artifacts/checkpoints/               # 自训 N3
  configs/n3_train_v1.json
  n3_affect/
  tests/
```

## 启用 EmoBERTa

```python
from n3_affect import N3TrainConfig, N3EmotionModel
cfg = N3TrainConfig(text_tower="emoberta_base")  # 优先读本地 artifacts
model = N3EmotionModel(cfg)
```

## 冒烟（默认不加载 EmoBERTa）

```powershell
cd 模型
python -m pytest tests -q
python -m n3_affect.train --epochs 1 --steps-per-epoch 2
```
