# 模型 / ComposerN3 + 主线千问

- **主线大模型：** `Qwen/Qwen3-4B-Instruct-2507`（`text_tower=qwen3_4b`，Apache-2.0，**免费**开源权重，本机从 Hugging Face 下载）
- **N3 融合骨干：** ComposerN3（六路 / 3×3 / 效用门控）
- **支线保留：** `emoberta_base`（已入库 LFS）、`composer_n3`（纯特征）、`xlm_roberta_large`

> 4B 千问体积大，**不整包进 Git**；首次主线训练会自动拉权重到本机缓存，也可放到 `artifacts/pretrained/qwen3-4b-instruct-2507/`。

## 主线用法

```python
from n3_affect import N3TrainConfig, N3EmotionModel
# 读 configs/n3_train_v1.json 时 default_mode 已是 qwen3_4b
cfg = N3TrainConfig.from_json("configs/n3_train_v1.json")
assert cfg.text_tower == "qwen3_4b"
model = N3EmotionModel(cfg)
```

## 支线用法

```python
cfg = N3TrainConfig(text_tower="emoberta_base")  # 本地 artifacts
cfg = N3TrainConfig(text_tower="composer_n3")    # 冒烟 / 特征 sidecar
```

## 冒烟（不下载千问）

```powershell
cd 模型
python -m pytest tests -q
python -m n3_affect.train --text-tower composer_n3 --epochs 1 --steps-per-epoch 2
```

可选：把千问缓存到仓库外或 `artifacts/pretrained/qwen3-4b-instruct-2507/`：

```powershell
python -m n3_affect.download_qwen
```
