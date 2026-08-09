# 模型 / ComposerN3

本目录是面向仓库 N3 主线的**可训练情感分类模型包**。

- 默认骨干：**ComposerN3**（按本仓库冻结协议设计的六路网络，无需下载外部权重即可训练冒烟）
- 增强文本塔：**XLM-RoBERTa-large**（`FacebookAI/xlm-roberta-large`，MIT，中英多语，比原先 Qwen2.5-0.5B 强得多）

> 不上传权重、特征或原始数据。遵守根目录 [`DATA_BOUNDARY.md`](../DATA_BOUNDARY.md)。

## 「权重」是什么？（和「调参」不是一回事）

| 说法 | 是什么 | 例子 | 要不要进 Git |
|---|---|---|---|
| **权重 / 参数（parameters）** | 模型里已经学到的那一大堆数字，决定「看到输入后怎么输出」 | `.pt` / `.bin` / `.safetensors` 文件里的矩阵 | **不要**（体积大，且常有再分发限制） |
| **超参数（hyperparameters）** | 训练前你们人手设定的旋钮，用来控制怎么学 | 学习率、batch size、损失权重 λ、层数 | **可以**（写在 `configs/*.json`） |
| 你们说的「后调参」 | 通常指调**超参数**，或在验证集上选模型；不是把权重文件提交进仓库 | 改 `lr`、换 seed、消融开关 | 配置可提交；选中的大权重仍放本地 |

一句话：**权重 ≈ 模型脑子里的记忆（学出来的参数）；调参 ≈ 你们拧的训练旋钮（超参数）。**  
仓库禁止提交的「权重」，就是预训练/训练得到的那些大文件，不是 `n3_train_v1.json` 里的 lr。

启用 XLM-R 时：本机会从 Hugging Face **下载**其权重到本地缓存；冻结主干、只训练我们的小投影与 N3 头。下载物不要 `git add`。

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

## 内置 / 可选编码器

| 模式 | ID | 说明 |
|---|---|---|
| **默认** | `composer_n3` | 仓库原生可训练骨干，无需下载，可直接冒烟 |
| **增强** | `xlm_roberta_large` → `FacebookAI/xlm-roberta-large` | 约 5.5 亿参数冻塔；MIT；覆盖 EmotionTalk(中) + MELD(英) |

为何选它：开源许可清晰、分类/表示能力远强于 0.5B 聊天小模型、多语对话情感场景常用，比闭源 API 更利于复现与合作分工（Cursor / Codex 只负责写代码，不充当训练 backbone）。

启用增强塔（需本机已装 `transformers`，会下载权重到本地）：

```python
from n3_affect import N3TrainConfig, N3EmotionModel
cfg = N3TrainConfig(text_tower="xlm_roberta_large")
model = N3EmotionModel(cfg)
```

## 快速跑通（合成数据，默认 ComposerN3）

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r 模型\requirements.txt
cd 模型
..\ .venv\Scripts\python.exe -m n3_affect.train --epochs 1 --steps-per-epoch 2
..\ .venv\Scripts\python.exe -m pytest tests -q
```

默认只写指标卡，不写 `.pt` checkpoint。真实数据与 HF 权重请放在仓库外。

## 与 `experiment/` 的关系

- 本包实现 N3 **可训练骨架**，不替代已冻结的 HarmBench / N2 合同测试  
- 真实 six-way producer、官方 test 启封与确认性评估仍按主 README 边界执行  

## 配置

见 [`configs/n3_train_v1.json`](configs/n3_train_v1.json)。官方 test 评估授权默认为 `false`。
