# 模型 / ComposerN3（当前可训练架构）

本目录是**当前仓库默认可训练实现**，架构与本地训练用模型包同构。完整说明见：

**[`docs/17_ComposerN3当前实现架构_对齐E模型_2026-08-14.md`](../docs/17_ComposerN3当前实现架构_对齐E模型_2026-08-14.md)**

---

## 架构摘要

```text
冻 Qwen3-Omni thinker（文本 2048-d，可离线）
+ 音视频侧车（1536 / 768）
    → ModalityProjector → z∈R^128
    → 共享 3×3 当前×历史关系
    → 多效应双向效用 + 可学习 mix
    → 两级门控（模态 / 联合风险）
    → Transformer 上下文 → 7 类情感 + VAD 辅助
```

| 项 | 当前实现 |
|---|---|
| 主线大模型 | `Qwen/Qwen3-Omni-30B-A3B-Instruct`（**仅文本塔**；权重不进 Git） |
| 六路 | `T_t,A_t,V_t,T_h,A_h,V_h`，历史 `K=3` + masks |
| 骨干 | ComposerN3（`n3_affect/model.py`） |
| 正式数据入口 | `generate_meld_manifests` → `extract_meld_features` → `train_meld` / `eval_meld` |
| 支线 | `emoberta_base`（仓内 LFS）、`composer_n3` 合成冒烟 |

> Omni 太大：**禁止整包进 Git**。本机/服务器：`python -m n3_affect.download_qwen` 或 `local_omni_path.txt` 指向已有权重。

---

## 与 docs/14–16「Phase A/B」的关系

`docs/14`–`16` 描述的是**尚未默认落地**的升级合同（Qwen 原生三模态 `e`、A0/A1 emotion-only、`phi_k` 理论条件化等）。

- **现在按本目录代码与 docs/17 架构训练 / 改代码。**  
- 不要用 Phase A/B 文档否定当前 ComposerN3 六路+效用+门控实现。  
- 将来若切换到 Phase A/B，应单开实现与新配置，并改写本 README。

---

## MELD 实跑（推荐）

```bash
cd 模型
# 或 bash run_meld_v3.sh（先改脚本内数据根路径）

python -m n3_affect.generate_meld_manifests --out-dir /path/manifests
python -m n3_affect.extract_meld_features \
  --manifests /path/manifests/train.csv /path/manifests/val.csv /path/manifests/test.csv \
  --out-dirs /path/feat/train /path/feat/val /path/feat/test
python -m n3_affect.train_meld \
  --train-manifest /path/manifests/train.csv \
  --train-features /path/feat/train \
  --out-dir /path/train_out
python -m n3_affect.eval_meld \
  --checkpoint /path/train_out/checkpoints/best.pt \
  --manifest /path/manifests/val.csv \
  --features /path/feat/val
```

`train_meld`：**只用官方 train**；早停用 train 内 monitor，不用官方 test 调参。test 评估须另有授权。

---

## 合成冒烟（不加载 Omni）

```powershell
cd 模型
python -m pytest tests -q
python -m n3_affect.train --text-tower composer_n3 --epochs 1 --steps-per-epoch 2
```

---

## 构造示例

```python
from n3_affect import N3TrainConfig, N3EmotionModel
cfg = N3TrainConfig.from_json("configs/n3_train_v1.json")
assert cfg.text_tower == "qwen3_omni_30b_a3b"
assert cfg.text_dim == 2048
model = N3EmotionModel(cfg)
```
