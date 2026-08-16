# N3 情感历史效用控制：六路建模 × 双向边际效用 × 真实分类增益

> **当前可训练主架构（ACTIVE）：ComposerN3**——与本地训练用模型包同构。  
> 详解：[ComposerN3 当前实现架构](docs/17_ComposerN3当前实现架构_对齐E模型_2026-08-14.md) · 代码：[`模型/`](模型/README.md)

```text
冻 Qwen3-Omni thinker（文本）+ 音视频侧车特征
  → 六路投影 z∈R^128
  → 共享 3×3 关系 → 多效应双向效用 → 两级门控
  → Transformer 上下文 → 情感分类（+ VAD 辅助）
```

- **HarmBench/ERC：**辅助评估与安全合同，不是当前主方法。  
- **最终判据：**真实情感分类指标；工程冒烟通过不等于方法成功。  
- **`docs/14`–`16`：**升级向的 Phase A/B（Qwen 原生三模态、`phi_k` 等）**目标合同**，默认**不是**当前代码架构；与实现冲突时，以 `模型/n3_affect` + docs/17 为准。

## 当前实现架构（按代码，2026-08-14）

| 层 | 实现要点 |
|---|---|
| 特征 | 离线：`extract_meld_features` — Qwen 文本 2048-d；A/V 为 librosa / 帧统计侧车 |
| 输入维 | `text_dim=2048`, `audio_dim=1536`, `video_dim=768` → `d_model=128` |
| 历史 | 严格过去最多 `K=3`，`history_mask` / `modality_mask` |
| 骨干 | `SharedThreeByThree` + `BidirectionalUtilityHeads` + `TwoLevelGate` + Transformer |
| 训练 | `train_meld.py`：仅官方 **train**；train 内 monitor 早停 |
| 主线 LLM | `Qwen3-Omni-30B-A3B-Instruct` **只作文本塔**（权重不进 Git） |

数据状态（机器侧，不进公开仓）：MELD 可走上述链；IEMOCAP / EmotionTalk 另按数据集文档推进。三数据集**分别**训练评估，不默认一套权重证明三套数据。

已解压 ≠ 可直接训：仍需 manifest → 特征 → `train_meld`。入口见 [`模型/README.md`](模型/README.md)。

![ComposerN3 当前可训练架构](assets/composer_n3_current_architecture_20260817.svg)

上图与当前 `模型/n3_affect` 实现对齐：Qwen 只作文本塔，音视频使用侧车特征。远期 **docs/14 Phase A/B 目标流程**仍保留为历史设计资料：[PNG](assets/n3_qwen_omni_experiment_workflow_20260813.png) · [PPTX](assets/n3_qwen_omni_experiment_workflow_20260813.pptx)。Phase B 设计来源见 [docs/12](docs/12_N3候选方案_要求对照与冻结协议_2026-08-09.md)。

## N3 两阶段实验框架

| 阶段 | 核心处理 | 输出与判定边界 |
|---|---|---|
| Phase A1. 冻结三模态提取 | 原始 `x` 经 Qwen3-Omni 分模态/分候选得到 hidden `e`；分路缓存 provenance；`ModalityProjector(e) → z∈R^128` | 明确 N3 消费的是 `z` 而不是原始媒体；证明每路 Qwen 来源可追溯 |
| Phase A2. 严格历史与基线训练 | 候选先取最近 `K=3`，槽位按 oldest→newest；独立训练 A0 current-only 与冻结公式的 A1 plain-history；dev Weighted-F1 选 best | 无历史硬切 A0；得到每个数据集自己的两个管线基线并停在 `STOP_BEFORE_TEST_A`；不作完整 N3 机制结论 |
| Phase B0. 情感状态与动力学 | 从冻结 `e` 开始 group-OOF 全部可学习状态 producer，得到后验、VAD、置信度；严格过去同 actor 轨迹计算惯性/恢复，跨 actor 单列 interaction shift，并计算 current/history 冲突 | 形成无评估标签泄漏的逐候选 `a_k`，缺少合法来源则 fail closed |
| Phase B1. 条件化候选级 3×3 | 对 `h_1`、`h_2`、`h_3` **逐候选**先计算九种基础关系 `R_k^0`，再用 `phi_k` 得到条件化关系 `R_k` | 避免 `R_k/phi_k` 循环定义；保留候选和模态可归因性；禁止先聚合三条历史 |
| Phase B2. 真实双向边际效用 | 在不同的 `S`/`R` 背景下分别估计各候选的文本、音频、视频加入收益与删除风险，再估计联合效用和不可加残差 | `U_T/U_A/U_V`、`U_joint` 与 `U_cross` |
| Phase B3. 两级门控与确认 | `phi_k` 直接进入模态门和候选联合门；失败时硬切独立 A0 checkpoint；至少 5 seeds、分组配对 95% CI、完整基线/消融 | 只有真实分类指标和完整合取门通过，才支持 N3 有效性主张 |

### 双向边际效用处理什么

它不是只对“已经融合好的历史向量”做一次比较。N3 同时包含两层比较：

1. **模态级效用：**只改变候选历史的一种模态，分别计算文本、音频、视频历史的前向加入收益和后向删除风险；
2. **联合级效用：**同时改变该候选的三种模态，判断融合后的整体收益，并用 `U_cross` 表示联合效用不能被三个单模态效用简单相加解释的协同或冲突。

对候选历史 `h` 的模态 `m∈{T,A,V}`，冻结比较中其余模态和历史背景：

```text
M_m_forward  = loss(q; S_m) - loss(q; S_m + h^m)
M_m_backward = loss(q; R_m) - loss(q; R_m - h^m)
U_m = (M_m_forward, M_m_backward)
```

其中 `M_forward > 0` 表示加入该模态有益，`M_backward > 0` 表示删除后更好、继续保留该模态存在风险。前向背景与后向背景必须来自预注册的不同集合状态，禁止用同一差值改符号伪造“双向”。

## 为什么这是情感领域方法

N3 的任务、表示、理论约束和成功标准都绑定情感识别，而不是通用历史筛选：

- 主任务始终是当前话语的情绪分类，主损失为 `L_emotion`；
- 当前主干是冻结 Qwen3-Omni 的三路可审计表示；情感专用文本、音频和视觉编码器作为公平的替代表征 baseline，用来检验结论是否依赖特定表征；
- 离散情绪后验和 VAD 由训练角色内的冻结/fit-only 头预测；惯性、转折、恢复与跨模态冲突由这些预测及严格过去轨迹计算，禁止把 gold 类别直接查表后作为输入；
- 上述情感状态与动力学变量组成 `a_k`，再与基础关系 `R_k^0` 形成 `phi_k`；`phi_k` 直接条件化 `R_k`、效用头、模态门和候选联合门。只有真实接入 forward、产生可审计输出且对应消融有效，才算“融入情感理论”；
- 最终成功必须体现为真实 Accuracy、Weighted-F1 等情感分类指标提升，并通过去 VAD、去惯性/转折/恢复、去跨模态冲突、去双向效用、去两级门控等主模型组件消融验证；情感专用编码器只做替代表征 baseline，不作为当前 Qwen 主模型中可“移除”的组件。

因此，即使一个通用 selector 能预测“某段历史是否有用”，如果完整 N3 不能提高真实情感分类，它也不能支持本项目的核心主张。

## 确认性实验路径

```text
按 speaker/dialogue/session 分组交叉拟合
    ↓
只在 fit 内生成 N3 效用监督和 OOF 预测
    ↓
冻结源码、配置、指标、效应阈值、统计合同与 protocol SHA
    ↓
生成不含评估标签的 outcome-free 预测产物
    ↓
由一次性 label-only evaluator 计算最终结果
    ↓
MELD + EmotionTalk + IEMOCAP 多数据集证据（正负结果均报告）
```

主要规则：

- 以 speaker、dialogue 或 session 为统计和切分单位，禁止把同一人物或会话泄漏到训练与评估两侧；
- MELD 主指标预注册为 Weighted-F1，同时报告 Accuracy、Macro-F1、NLL、Brier、ECE、历史伤害率、CVaR 和风险—覆盖；
- 至少 5 个随机种子，并使用分组配对区间判断提升是否稳定；
- 完整 N3 必须优于 independent current-only 和最强历史基线，并在移除情感理论变量、模态级/联合双向效用、两级门控或逐候选 3×3 关系后出现预期下降；情感专用编码器以替代表征 baseline 单独比较；
- 已观察的模型选择结果只作探索性证据，不能在继续调参后包装成确认性成功。
- IEMOCAP 采用预注册 outer-Session 五折；每个 Session 仅在对应 fold 内 held-out，不能声称五个 Session 在整个开发期全局从未使用。

## 数据集与当前角色

| 数据集 | N3 中的角色 | 当前状态 | 官方入口 |
|---|---|---|---|
| MELD | 标准英文多方对话基准；首个 Phase A 管线修复对象 | 原始数据可用；旧 run 为 `invalid_preliminary_run`；待修复后重训，test 封存 | [GitHub](https://github.com/declare-lab/MELD) |
| EmotionTalk | 中文三模态外部证据与较深同说话人历史 | 新一轮原始包上传中；尚未完成解压和新管线审计；旧结果仅作历史探索 | [GitHub](https://github.com/NKU-HLT/EmotionTalk) · [Hugging Face](https://huggingface.co/datasets/BAAI/Emotiontalk) |
| IEMOCAP | 第三个独立数据集；适合 session/speaker 隔离及情感转折/恢复 | 已获 USC 官方授权且归档/解压/媒体完整性 PASS；下一步做 manifest、标签和 Session 五折 preflight，尚未训练 | [USC](https://sail.usc.edu/iemocap/) · [Release](https://sail.usc.edu/iemocap/iemocap_release.htm) |

数据集的详细许可状态与备用顺序见 [数据集与许可状态](docs/03_数据集与许可状态.md)。IEMOCAP 不是可从非官方仓库随意下载或重新分发的数据；无学校邮箱不改变其授权要求。

## 数据与发布边界

完整规定见 [DATA_BOUNDARY.md](DATA_BOUNDARY.md)（科研协作版）。

**本轮执行可提交：**代码、配置、数据 schema、无原文的聚合审计、聚合指标、流程文档和合成测试。

**本轮执行禁止提交：**MELD/EmotionTalk/IEMOCAP 原始或解压数据、逐样本 manifest/标签、派生特征、Qwen 或训练权重、服务器 checkpoint/日志、服务器地址和私有绝对路径，以及任何账号、密钥或未脱敏隐私材料。数据已合法取得或已经解压也不会改变这条发布边界。

原始数据请用 [`datasets/`](datasets/) 在仓库外下载。本轮 MELD/IEMOCAP 执行产生或依赖的 Qwen/训练权重只在仓库外按哈希引用，不上传到本分支；`模型/artifacts/` 中已有的历史公开权重资产不因此自动成为本轮执行产物。

### 协作者如何取得数据

本项目对**原始数据集**仍采用“官方来源下载＋本地校验”。详细目录规范见 [官方数据获取与本地目录规范](datasets/README.md)。开源预训练权重可直接使用本仓 `模型/artifacts/pretrained/`。

```powershell
# 默认只取 MELD train/dev 标注；test 保持 evaluator-only
$dataRoot = 'D:\N3_data'
powershell -NoProfile -ExecutionPolicy Bypass -File datasets\scripts\download_official_data.ps1 `
  -Dataset MELD `
  -Destination (Join-Path $dataRoot 'MELD\e8cedf27b5d2877e198332c957127e16eb214afe')

# EmotionTalk 需先执行 hf auth login 并由本人接受 gated 条款
powershell -NoProfile -ExecutionPolicy Bypass -File datasets\scripts\download_official_data.ps1 `
  -Dataset EmotionTalk `
  -Destination (Join-Path $dataRoot 'EmotionTalk\adbc17fc944e8cf2873643906160c6ca0259ab61')
```

加入 `-IncludeMedia` 才会下载大体积音视频；MELD 的 `-IncludeTest` 只允许指定 evaluator 使用。下载所得 **原始** archive、标签和媒体仍不得提交进本仓；训好的**自有权重**按 `DATA_BOUNDARY.md` 可放入 `模型/artifacts/`。

## 快速验证框架

在 Windows PowerShell 中：

```powershell
python -m venv .venv
$python = (Resolve-Path '.venv\Scripts\python.exe').Path
& $python -m pip install --upgrade pip
& $python -m pip install -r experiment\requirements-harmbench.txt
& $python -m pytest experiment\tests\test_harmbench_erc*.py -q
```

当前已完成的框架测试结果为 **418 passed**，另有一个非阻断的 sklearn 零方差警告。该结果证明现有接口、泄漏防护和合同测试通过，不代表 N3 已经取得真实数据分类增益。

## 关键文件

- [最新 Phase A/Phase B 执行基线](docs/14_最新执行基线与GitHub旧方案差异_2026-08-13.md)：当前唯一执行顺序、逐数据集 Gate、实现差距与 test 边界；
- [MELD 已解压数据后续执行全流程](docs/15_MELD_已解压数据后续执行全流程_2026-08-13.md)：从用户批准路径、只读 preflight 到 Phase A/Phase B 两个停止点及独立授权 test 的逐 Gate 操作单；
- [IEMOCAP 已解压数据后续执行全流程](docs/16_IEMOCAP_已解压数据后续执行全流程_2026-08-13.md)：从 Session/媒体审计、外层 Session 五折 manifest 到两阶段训练与五折交叉验证汇总的逐 Gate 操作单；
- [最新两阶段流程图（PPTX）](assets/n3_qwen_omni_experiment_workflow_20260813.pptx)：可编辑的 Phase A/Phase B 架构与实验流程；
- [N3 要求对照与冻结协议](docs/12_N3候选方案_要求对照与冻结协议_2026-08-09.md)：六路表示、3×3 交互、双向效用、两级门控、成功门和多数据集验证规则；
- [前三项创新新颖性审计](docs/13_CARMA-Affect_前三项创新_新颖性审计_2026-08-07.md)：创新边界、相关工作映射与不可过度声称的内容；
- [N3/HarmBench 候选冻结配置](experiment/configs/harmbench_erc_v2_candidate.json)：当前机器可检验的候选配置；
- [依赖清单](experiment/requirements-harmbench.txt)：框架测试依赖；
- [数据与发布边界](DATA_BOUNDARY.md)：允许和禁止进入公开仓库的内容。

## 当前实现边界

以下是截至 2026-08-09 的历史实现边界（最新状态见本文顶部的 2026-08-13 基线）：

- N3 协议、框架图、配置合同和合成/防泄漏测试已经进入本分支；
- 六路真实 producer、三类情感领域编码器及完整真实数据训练流水线尚未全部完成；
- MELD 与 EmotionTalk 的官方 test 仍保持封存；
- IEMOCAP 已取得官方许可并通过归档/解压/媒体完整性检查，但 manifest、标签、Session 五折和 Qwen 单样本 preflight 尚未完成，仍未训练；
- 因此当前可以表述为“**N3 实验框架已搭建并冻结关键合同**”，不能表述为“**N3 已被证明有效**”。

本分支下一步先关闭 Phase A 的 Qwen 三模态 producer、K=3/mask、emotion-only trainer、dev Weighted-F1 best 与 test-deny 实现差距；各数据集 Phase A 通过后先停在 `STOP_BEFORE_TEST_A`，再实现并审计 Phase B 的逐 `h_k` 3×3、情感变量、真实双向效用和两级门控。Phase B 冻结后停在 `STOP_BEFORE_TEST`；正式评估仍须逐数据集按各自协议独立授权且 write-once。
