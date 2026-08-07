# CARMA-Affect 顶会证据推进计划

## Goal

在不访问当前/未来标签、不泄漏人物或时间信息的前提下，建立并验证“个体历史何时造成情感预测负迁移、如何预测这种伤害并安全回退”的可复现研究证据；最终交付满足顶会审稿标准的多数据集结果，或形成有充分否定证据支撑的 benchmark/问题论文路线。

## Current Phase

Phase 1 — 预注册与 benchmark 冻结（in_progress）

## Next Step

在不读取 90–99 内部封存组的前提下，完成 EmotionTalk train-only 端点目标诊断：0–64 训练、65–79 选型、80–89 校准；只有符号—严重度错配在独立选型组复现，才进入完整随机历史子集增强。

## Confirmatory Evidence Gate

只有同时满足以下条件，才可称为“具备顶会投稿证据”；不保证录用：

1. 至少 2 个独立来源数据集完成确认实验，目标为 3 个（MELD、IEMOCAP、EmotionTalk；许可失败时使用预先声明的替代集）。
2. 严格按对话/人物/时间协议切分；历史仅来自查询时刻之前；所有选择器监督由 out-of-fold/cross-fitting 产生。
3. 与 current-only、all-history、强检索/记忆基线及校准回退基线公平比较。
4. 锁定主指标：情感预测 Macro-F1、相对 current-only regret、历史伤害率、q90/worst-group regret、fallback coverage-risk、ECE/Brier。
5. 确认实验至少 5 个随机种子；报告均值、95% CI、配对效应量与 Holm 校正；按对话或人物做层级 bootstrap，禁止把 utterance 当作完全独立样本。
6. 主方法必须在至少 2 个数据集上降低负迁移且 95% CI 不跨 0；同时不能以明显牺牲 Macro-F1 或最差群体为代价。
7. 完成消融、标签置乱、未来历史注入检测、重复人物检测、时间反转、状态突变与恢复、模态缺失/噪声压力测试。
8. 代码、配置、随机种子、环境、聚合结果与失败实验均可复现；未授权原始数据和派生受限特征不公开。

## GPT Use Policy

- GPT 只能作为预先定义的文本零样本基线、冻结特征/教师或解释性辅助模块；不得读取测试标签、未来话语、人物跨切分资料或完整未授权音视频。
- GPT 方案必须与非 GPT 参数量/信息量可比基线比较，并记录模型快照、提示词、温度、重试和成本。
- GPT 输出必须缓存并版本化；确认实验禁止根据测试结果改提示词。
- 若没有可用 API 凭据或数据许可，先实现可离线运行的接口和 mock/开源替代基线，不伪造 GPT 实验数据。

## Phases

### Phase 0 — 复现实验与资产审计

Status: completed

- 识别实际训练/评估入口、冻结配置、结果散列和测试。
- 盘点 MELD、EmotionTalk、IEMOCAP 数据可用性与许可边界。
- 记录 GPU/CPU、Python/CUDA、依赖和磁盘预算。
- 复现合同测试及一个冻结结果入口。

### Phase 1 — 预注册与 benchmark 冻结

Status: in_progress

- 冻结研究问题、假设、数据切分、主要/次要指标和统计方案。
- 建立 schema、数据卡、泄漏单元测试和负迁移 benchmark。
- 把探索集/确认集及开发轮次明确隔离。

### Phase 2 — 强基线与可行性复现

Status: pending

- current-only、all-history、recency、similarity、top-k、oracle upper bound。
- 文本、音频、视频及三模态公平对照。
- 复核伤害预测是否存在可学习信号，以及回退的理论可达上限。

### Phase 3 — 模型探索与 GPT 审计

Status: pending

- 条件边际效用选择器、可校准风险头、选择性预测/安全回退。
- 测试分层、可逆历史门控、set/sequence 归因和不确定性建模。
- 评估 GPT 零样本、冻结 embedding/教师蒸馏三种角色；只保留严格优于等成本基线者。
- 所有探索结果完整登记，不以测试集表现选型。

### Phase 4 — 锁定方法与确认实验

Status: pending

- 根据开发集预先锁定一版方法、超参和提示词。
- 在未触碰确认集上运行至少 5 seeds × 至少 2 数据集。
- 生成层级 bootstrap、效应量、Holm 校正、校准与 worst-group 报告。

### Phase 5 — 反证、消融与鲁棒性

Status: pending

- 运行标签置乱、时间反转、未来泄漏、状态突变/恢复、模态缺失、人物偏差检查。
- 区分“检索质量”“历史因果效用”和“选择器校准”的贡献。
- 对失败和负结果做等强度报告。

### Phase 6 — 论文证据包与发布

Status: pending

- 形成主表、置信区间、图表、数据卡、模型卡、复现实验清单。
- 进行模拟审稿，按 critical/important/minor 分级修正。
- 更新草稿 PR；仅在证据门槛满足后形成投稿结论。

## Decision Rules

- 当前 CARMA 严格 q90 安全回退已冻结为 FAIL；未经新确认实验不得改写为成功。
- 当前方法论文路线为 STOP，benchmark 路线为 GO；只有 Phase 2–4 的新证据可触发重新评估。
- 同一失败最多尝试三种不同修复路径；之后记录为阻塞或否定结果，不无限调参。
- 探索实验与确认实验目录、配置和结果表分离；确认结果一旦读取，不再回到同一确认集调参。

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| 初始仓库缺少持续计划文件 | 1 | 创建 task_plan.md、findings.md、progress.md |
| PowerShell 默认编码导致中文 Markdown 乱码 | 1 | 后续读取显式使用 UTF-8，不把乱码误判为文件损坏 |
| `System.IO.Path.GetRelativePath` 在当前 PowerShell/.NET 不可用 | 1 | 使用字符串前缀裁剪或 Python 只读枚举，不重复原命令 |
| 后台实验轮询只看到 Python shim CPU 很低 | 1 | 用 Win32_Process 检查子进程，确认实际 `python3.11.exe` 正在计算 |
| 在 PowerShell 中误用 Bash heredoc `<<` 读取 JSON | 1 | 改用 PowerShell here-string 管道到 Python stdin，不重复 Bash 语法 |
| EmotionTalk benchmark 误把 dialogue 单列当全局 cluster | 1 | 首轮 CI 作废；改用 `group/dialogue` 复合键并新增单元测试后重跑 |
| PowerShell here-string 插值将 Unicode 私有绝对路径传给 Python 时乱码 | 1 | 从私有 experiment 目录运行并使用相对路径，避免跨层 Unicode 插值 |

## Decisions Made

| Date | Decision | Rationale |
|---|---|---|
| 2026-08-07 | 以历史负迁移 benchmark 为当前主路线，CARMA 方法作为待重构候选 | 已冻结 q90 安全回退失败，但历史伤害与多模态预测增量成立 |
| 2026-08-07 | 将 GPT 限定为基线、冻结特征/教师或解释模块 | 保持核心问题不变并防止标签/时间泄漏与不可复现提示调参 |
| 2026-08-07 | 主探索模型选择 SCU-Set，RCPS 作为安全层 | SCU-Set直接学习逐候选/集合边际效用并支持可逆重用；单纯改阈值创新不足 |
| 2026-08-07 | SCU 目标升级为 sign×severity/quantile 分布建模 | 双数据集探索显示低伤害概率不等于低平均regret，单一分类或均值目标均不足 |
| 2026-08-07 | Phase 0 通过并进入 Phase 1 | MELD 第三次独立重跑与冻结结果字节/解压内容完全一致；32 项私有工程测试和 28 项公开新增合同均通过 |
