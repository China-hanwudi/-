# 模型 / ComposerN3 + 主线 Qwen3-Omni

> **2026-08-13 代码同步警告：**最新实验要求 Qwen 分别处理文本、音频和视频，并使用 K=3/masks、emotion-only、dev Weighted-F1 best 与 `STOP_BEFORE_TEST`。当前代码只把 Qwen 接入 `text_tower`，A/V 仍为外部特征投影，配置和正式 trainer 也未完成上述同步。**不要直接用当前 `n3_train_v1.json` 启动本轮正式训练。**完整差异和实现 Gate 见 [`../docs/14_最新执行基线与GitHub旧方案差异_2026-08-13.md`](../docs/14_最新执行基线与GitHub旧方案差异_2026-08-13.md)。

- **目标主线唯一大模型：** `Qwen/Qwen3-Omni-30B-A3B-Instruct`（当前仓库键仍为 `text_tower=qwen3_omni_30b_a3b`，待升级为可审计的三模态 extractor）
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
