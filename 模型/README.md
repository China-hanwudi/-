# 模型 / ComposerN3 + 主线 Qwen3-Omni

- **主线唯一大模型：** `Qwen/Qwen3-Omni-30B-A3B-Instruct`（`text_tower=qwen3_omni_30b_a3b`）
- **N3 融合骨干：** ComposerN3
- **支线：** `emoberta_base`（仓内 LFS）、`composer_n3`

> Omni 权重很大，**禁止整包进 Git**。本机下载：`python -m n3_affect.download_qwen`  
> 已取消 Qwen3-4B 等其它千问主线布置，避免体积膨胀。

## 主线

```python
from n3_affect import N3TrainConfig, N3EmotionModel
cfg = N3TrainConfig.from_json("configs/n3_train_v1.json")
assert cfg.text_tower == "qwen3_omni_30b_a3b"
model = N3EmotionModel(cfg)
```

## 冒烟（不下载 Omni）

```powershell
cd 模型
python -m pytest tests -q
python -m n3_affect.train --text-tower composer_n3 --epochs 1 --steps-per-epoch 2
```
