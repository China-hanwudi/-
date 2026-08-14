# 模型 / ComposerN3 + 主线 Qwen3-Omni

> **2026-08-13 代码同步警告：**最新实验要求 Qwen 分别处理文本、音频和视频，并使用严格过去 `K=3`/三类 masks。Phase A 为 emotion-only、dev Weighted-F1 best，结束于 `STOP_BEFORE_TEST_A`；Phase B 才启用情感状态/动力学、双向效用和两级门控，冻结后结束于 `STOP_BEFORE_TEST`。当前代码只把 Qwen 接入 `text_tower`，A/V 仍为外部特征投影，配置和正式 trainer 也未完成上述同步。**不要直接用当前 `n3_train_v1.json` 启动本轮正式训练。**完整差异和实现 Gate 见 [`../docs/14_最新执行基线与GitHub旧方案差异_2026-08-13.md`](../docs/14_最新执行基线与GitHub旧方案差异_2026-08-13.md)。

- **目标主线唯一大模型：** `Qwen/Qwen3-Omni-30B-A3B-Instruct`（当前仓库键仍为 `text_tower=qwen3_omni_30b_a3b`，待升级为可审计的三模态 extractor）
- **N3 融合骨干：** ComposerN3
- **支线：** `emoberta_base`（仓内 LFS）、`composer_n3`

> Omni 权重很大，**禁止整包进 Git**。本机下载：`python -m n3_affect.download_qwen`  
> 已取消 Qwen3-4B 等其它千问主线布置，避免体积膨胀。

## 目标协议（尚待实现）

```text
raw x
→ frozen Qwen per-modality/per-candidate hidden e + provenance
→ trainable P_T/P_A/P_V
→ Z_current=[B,3,128], Z_history=[B,K=3,3,128]
→ Phase A: independently trained A0 current-only
           + frozen-formula A1 plain-history (CE-only)
→ Phase B: group-OOF emotion posterior/VAD/confidence
           → strict-past dynamics + metadata/quality/masks
           → R_k^0 → phi_k → conditioned R_k
           → S≠R bidirectional utility → modality/candidate gates
```

`phi_k` 必须直接连接条件化逐候选 `R_k`、效用头、模态门和候选联合门四条路径。fit/train gold 只允许用于训练损失和 held-out-group OOF evaluator 内的 utility target；dev gold 只进隔离 metric/model-selection evaluator；test/outer gold 只进授权后的独立 write-once evaluator。任何 gold 都禁止作为模型输入，也禁止用 dev/test gold 类别查 VAD 后回灌模型。情感专用编码器仅作替代表征 baseline。

Phase A 的 A1 只允许对候选话语表示做无参数 mask-safe mean；无历史时在 A1 classifier 前硬切独立 A0-best logits/probabilities。Phase B 无候选通过或风险失败也硬切 A0，不得用 history/N3 checkpoint 的空历史近似或混合概率。dev gold 只进入隔离的 best-checkpoint evaluator；正式 test/outer gold 只进入授权后的独立 write-once evaluator。

## 已同步的 MELD 实跑入口（来自本地 `E:\模型`，2026-08-14）

下列文件**真实存在**，用于官方 MELD 划分上的离线特征 + N3 训练/评估（与上方「目标协议」仍有差距，**不能**直接宣称 Phase A/B Gate 已完成）：

| 入口 | 作用 |
|---|---|
| `n3_affect/generate_meld_manifests.py` | 从已解压 MELD 生成 train/val/test manifest |
| `n3_affect/extract_meld_features.py` | 冻 Qwen thinker **文本**隐状态；音频/视频为 librosa / torchvision 侧车特征（非 Omni 原生 A/V 三路） |
| `n3_affect/meld_dataset.py` | 读取离线特征 batch |
| `n3_affect/train_meld.py` | 仅用官方 train；train 内 10% monitor 早停（不用官方 val/test 调参） |
| `n3_affect/eval_meld.py` | checkpoint 评估 |
| `run_meld_v3.sh` / `run_meld_resume.sh` / `run_meld_retrain.sh` / `n3_affect/run_pipeline.sh` | 服务器一键流水线（路径默认 `/root/肖田泽最强/`） |

Omni 权重与 `local_omni_path.txt` **不进 Git**；服务器上指向 `/data/shared/qwen/Qwen3-Omni-30B-A3B-Instruct`。

示例（服务器；先改脚本内数据根路径）：

```bash
cd 模型
bash run_meld_v3.sh
# 或分步：
# python -m n3_affect.generate_meld_manifests --out-dir ...
# python -m n3_affect.extract_meld_features --manifests ... --out-dirs ...
# python -m n3_affect.train_meld --train-manifest ... --train-features ... --out-dir ...
```

## 遗留构造示例（仅 smoke）

```python
from n3_affect import N3TrainConfig, N3EmotionModel
cfg = N3TrainConfig.from_json("configs/n3_train_v1.json")
assert cfg.text_tower == "qwen3_omni_30b_a3b"
model = N3EmotionModel(cfg)
```

## 遗留合成冒烟（不下载 Omni）

```powershell
cd 模型
python -m pytest tests -q
python -m n3_affect.train --text-tower composer_n3 --epochs 1 --steps-per-epoch 2
```

上述命令只验证旧接口仍可运行。`train_meld` 链证明「真实 MELD + 冻 Qwen 文本特征 + N3」可跑，**仍不证明**完整 Qwen 三模态 `x→e→z`、`K=3` masks、Phase A `STOP_BEFORE_TEST_A` 或 Phase B 理论条件化已实现。
