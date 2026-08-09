# 模型 / ComposerN3

本目录是面向仓库 N3 主线的**可训练情感分类模型包**。默认内置骨干为 **ComposerN3**（按本仓库冻结协议由 Cursor Composer 设计的六路建模网络）；可选文本大模型塔为 **Qwen2.5-0.5B-Instruct**（中英双语，适配 EmotionTalk + MELD）。

> 不上传权重、特征或原始数据。遵守根目录 [`DATA_BOUNDARY.md`](../DATA_BOUNDARY.md)。

## 结构

```text
模型/
  README.md
  requirements.txt
  configs/n3_train_v1.json
  n3_affect/          # 可 import 的训练包
  tests/              # 前向 / 训练冒烟测试
```

## 模型在做什么

对齐 README / `docs/12` 的 N3 路径：

1. **六路输入**：`T_t/A_t/V_t` 与严格过去 `T_h/A_h/V_h` 分路投影，不做早期融合塌缩  
2. **共享 3×3 关系**：九种当前–历史模态对，共享低秩双线性 + 类型嵌入  
3. **双向边际效用**：预测 `U_T/U_A/U_V/U_joint`，并构造 `U_cross`  
4. **两级门控**：模态级保留 → 联合风险门；不通过则回退 **current-only**  
5. **主损失**：情绪分类 `L_emotion`，辅以效用与 VAD

## 内置大模型选择

| 模式 | ID | 说明 |
|---|---|---|
| **默认** | `composer_n3` | 仓库原生可训练骨干，无需下载 HF 权重，可直接冒烟训练 |
| 可选 | `Qwen/Qwen2.5-0.5B-Instruct` | 冻结文本塔 + 可训投影；需本机 `transformers`，权重只存本地 |

选择理由：ComposerN3 与现有 sidecar / SVD+WavLM+DINO 特征合同兼容；Qwen2.5-0.5B 覆盖中英对话且体积适合 LoRA / 冻塔微调。

## 快速跑通（合成数据）

在仓库根目录或本目录：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r 模型\requirements.txt
cd 模型
..\ .venv\Scripts\python.exe -m n3_affect.train --epochs 1 --steps-per-epoch 2
..\ .venv\Scripts\python.exe -m pytest tests -q
```

默认只写指标卡，不写 `.pt` checkpoint。真实数据训练时请把特征和权重放在仓库外路径。

## 与 `experiment/` 的关系

- 本包实现 N3 **可训练骨架**，不替代已冻结的 HarmBench / N2 合同测试  
- 真实 six-way producer、官方 test 启封与确认性评估仍按主 README 边界执行  
- 可把本包的 `logits` / gate 概率接到后续 fit-only OOF 流水线

## 配置

见 [`configs/n3_train_v1.json`](configs/n3_train_v1.json)。官方 test 评估授权默认为 `false`。
