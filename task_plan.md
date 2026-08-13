# CARMA-Affect N3 候选方案顶会证据推进计划

## Goal

在不访问当前/未来评估标签、不泄漏人物或时间信息的前提下，建立并验证 N3 候选方案：当前以冻结 Qwen3-Omni 分别表示当前与历史的文本、音频和视频，通过共享参数的 3×3 当前—历史交互、模态级与联合级真实双向边际效用以及两级安全门控，选择能够真正提高当前情感分类性能的历史信息；情感专用编码器保留为后续 baseline/消融。最终结论必须来自冻结协议后的未观察数据角色或预注册外部确认数据；HarmBench 只保留为辅助 benchmark/备选论文路线。

## Highest-priority override（2026-08-13）

详细合同见 [`docs/14_最新执行基线与GitHub旧方案差异_2026-08-13.md`](docs/14_最新执行基线与GitHub旧方案差异_2026-08-13.md)。本节优先于 2026-08-09 及更早记录；旧记录不得删除或改写。

- [x] 冻结当前主干为 `Qwen3-Omni-30B-A3B-Instruct`；文本、音频、视频均由其离线抽取，同时保持 `T_t/A_t/V_t/T_h/A_h/V_h` 分路和原始隐藏维度缓存。
- [x] 冻结 N3 输入合同：`K=3` 严格过去；`current_modality_mask[B,3]`、`history_mask[B,K]`、`history_modality_mask[B,K,3]`；无历史逐样本精确回退 current-only；内部 `ModalityProjector` 输出 `d_model=128`。
- [x] 冻结第一轮训练为 emotion-only：`utility_loss_weight=0`、`vad_loss_weight=0`；按 dev Weighted-F1 选择 best；train+dev 后必须 `STOP_BEFORE_TEST`。
- [x] 将 MELD 旧运行标记为 `invalid_preliminary_run`：只作故障诊断，不进论文、不用于调参；主要缺陷为 Qwen 只编码文本、视频 94.71% 全零、随机冻结文本投影、伪 VAD/utility target、错误 best 规则及自动 test。
- [ ] MELD 依序完成：数据清单 → 单样本 Qwen 三模态证明 → 32 train/8 dev 冒烟 → 全量特征审计 → emotion-only train+dev → `STOP_BEFORE_TEST`。
- [x] IEMOCAP 官方归档已校验并完整解压，Session1–5 和 WAV/AVI 抽检通过；状态从“等待授权”改为“数据完整性 PASS、尚未训练”。
- [ ] IEMOCAP 完成四分类 `angry/happy(hap+exc)/sad/neutral`、约 5531 条目标审计、固定 Session 五折、时间戳视频对齐、严格过去 K=3 manifest；在此之前不得训练。
- [ ] EmotionTalk 继续上传；到盘后依次做 archive、解压、manifest、媒体、mask 和 Qwen 三模态冒烟审计；不得假设已完成。
- [ ] 关闭 GitHub 代码差距：当前仓库仍只把 Qwen 接在文本塔上，utility/VAD 配置仍为旧值，K=3 masks、dev Weighted-F1 best 和 test-deny 正式 trainer 尚未同步；未关闭前不得直接用仓库旧配置重训。
- [ ] 三数据集采用同一冻结框架分别训练/验证/测试并对照总结；跨数据集零样本迁移仅作额外实验，不能替代三数据集内验证。
- [ ] 正式 test 继续封存；只有源码、配置、特征 manifest、best checkpoint 和统计合同冻结后，才可对每个数据集单独一次性授权。

## Highest-priority override（2026-08-09）

- 最高研究依据：用户提供的四条连贯研究要求及仓库内冻结协议。
- 正向方法唯一主线正式命名为 **N3 候选方案**；老师前三条要求作为待检验的方法贡献，第四条作为不可绕过的真实情感分类成功门。
- HarmBench 仅保留为：历史负迁移评价工具、旧 N2 失败原因分析、N3 辅助 benchmark，以及正向方法失败时的备选论文路线；不得替代 N3。
- 已结束的 10,000 次 bootstrap / 100,000 次 randomization 只封存一次；禁止重复运行，禁止继续扩展 HarmBench 真实实验。
- 既有 N2、HarmBench、selector repair 与负结果证据全部保留，不删除、不覆盖、不改写。
- IEMOCAP 预注册为 N3 的第三个独立外部确认数据集，只能在 N3 结构、超参数、效用阈值和统计合同冻结后运行，不得用于模型选择或调参，结果无论正负均报告。
- IEMOCAP 若因授权或预注册六路协议不可满足而失败，替代顺序固定为 `CPED → M3ED`；只按预先定义的许可/数据可行性门切换，禁止按结果选数据集。
- 在新的 protocol ID、模型、指标、效应阈值、统计方法和公开模板完全冻结前，MELD/EmotionTalk official test、validation、calibration、internal holdout 与任何未观察标签继续封存；真实受限数据禁止发送给 GPT/API/外部服务。

## Current Phase

N3 Phase 0 — 主线纠偏、协议/许可审计与预注册冻结（in_progress）

### N3 立即执行门

- [x] 确认 10k bootstrap / 100k randomization 已自然结束且当前无相关统计进程；不重跑。
- [ ] 完成“老师四条要求—当前 N2 实现—缺口—N3 修改方案”逐项对照。
- [ ] 完成情感领域编码器的 checkpoint、代码/权重/训练数据许可、数据适配、8GB VRAM/16GB RAM/约 53.5 GiB 磁盘预算与可复现性审计。
- [ ] 完成 IEMOCAP 的官方授权、数据结构、标签协议、session 级说话人隔离和六路模态可用性审计；同时审计 CPED、M3ED 的固定替代条件。
- [ ] 冻结 N3 protocol ID、六路接口、情感理论变量、共享 3×3 交互、模态级/联合级双向效用、两级门控、损失、超参数、阈值、15 项基线/消融、指标、统计合同、成功门和公开模板。
- [ ] 仅在 synthetic contract 与 fit 角色实现、测试和审计 N3；不得读取任何封存评估标签。
- [ ] 完成正式运行 readiness 审计后停止；没有单独授权不得解封任何未观察标签。

### 历史 N2/HarmBench 记录（保留，不再作为当前主线）

### Causal evidence gate 子任务（2026-08-08）

- [x] 独立 history-stripped current-only 身份、checkpoint 与概率合同
- [x] fit-OOF 冻结 25% query-candidate operating point 与 deterministic tie rule
- [x] 每个 query 等选中基数的 most-recent recency 对照
- [x] 五 seed×共享整簇 bootstrap CI 与整簇配对 randomization p 值
- [x] 完整预声明 Holm family 与 accuracy no-harm/non-inferiority gate
- [x] aggregate-only exact public whitelist 与 EmotionTalk+MELD hash-bound index
- [x] synthetic/相关/full regression、compileall 与 diff check
- [x] 实现独立 current-only artifact producer/CLI 与 selection-feature-only 概率 cache
- [x] 消除 current-only fit 的完整 history producer 冷启动依赖：fit-lineage/fit-map 可在 selection feature/label 均物理缺失时完成
- [x] 修复第一轮 complete-selection High：不访问 selection label、绑定 preflight config/code、绑定 feature-sidecar cluster 分区
- [x] 修复第二轮 production High：processor/model/identity strict-load、live lineage、固定 trainer、仓库外 private root 与原子 no-clobber writer
- [x] 将 history producer 拆出 fit-only 与 selection-feature-only；独立 evaluate 仍待后续接线
- [x] 通过新的独立只读审计后，才决定是否授权正式 open-role 长跑

### N2 Affect-Relation Causal Backbone（2026-08-08）

- [x] 接入严格过去 3×3 当前/历史模态关系与固定 VAD 辅助表示
- [x] 冻结 EmotionTalk/MELD 的 full、同容量 current-only、no-VAD、no-3×3 共八份可执行配置
- [x] no-VAD 禁止携带 VAD 标签顺序/辅助损失；VAD 标签顺序必须与已验证数据清单逐项一致
- [x] 四变体参数量严格一致：EmotionTalk 1,540,191；MELD 1,838,815，均小于 2M
- [x] 实现 versioned immutable source snapshot：clean detached worktree、commit/tree、递归 source set、repository-external write-once manifest
- [x] 关闭 source-key consumer High、单数据集 verifier performance-gate High 与双数据集 joint freeze，并完成全仓独立审计
- [x] 完成 history production CLI 第二轮修复、current-only 对称审计、全仓回归与最终独立只读 N2 审计
- [x] 推送 freeze `de056c3`，创建 clean detached worktree/外部 source snapshot，并通过 MELD 最坏形状 batch=64 CUDA backward smoke
- [x] 在新 freeze 后完成八变体 `fit-preflight`、`fit-lineage-create` 与独立 `fit-lineage-validate`；全程未打开 selection payload、未训练、未计算性能
- [x] 完成首个真实 `EmotionTalk/full history-fit`：25/25 checkpoints、fit OOF、双向 utility targets 与 producer receipt 均发布，selection payload 未消费
- [x] 完成 `EmotionTalk/full history-complete-selection`：25 套 checkpoint-only 恢复、selection-feature-only cache 与 receipt 发布，selection label 未访问
- [x] 完成每数据集唯一的 EmotionTalk full-anchor `current-only-fit`：25/25 folds、独立 fit OOF 与 receipt 发布，历史和 selection payload 均未消费
- [x] 绑定 full history completion 完成 EmotionTalk current-only selection cache：2,682 selection queries，selection label 未物化
- [x] 完成 `EmotionTalk/no_vad history-fit`：25/25 folds、fit OOF/targets/receipt 发布，selection payload 未消费
- [x] 完成 `EmotionTalk/no_vad history-complete-selection`：25 套 complete checkpoint 恢复、2,682 条 selection query 特征缓存与 receipt 发布，selection label 未访问
- [x] 完成 `EmotionTalk/no_history_3x3 history-fit`：25/25 folds、fit OOF/targets/receipt 发布，selection payload 未消费
- [x] 完成 `EmotionTalk/no_history_3x3 history-complete-selection`：25 套 complete checkpoint 恢复、2,682 条 selection query 特征缓存与 receipt 发布，selection label 未访问
- [x] 完成 `EmotionTalk/capacity_control history-fit`：25/25 folds、fit OOF/targets/receipt 发布，selection payload 未消费
- [x] 完成 `EmotionTalk/capacity_control history-complete-selection`：25 套 complete checkpoint 恢复、2,682 条 selection query 特征缓存与 receipt 发布，selection label 未访问
- [x] 完成 `MELD/full history-fit`：25/25 folds、fit OOF/targets/receipt 发布，selection payload 未消费
- [x] 完成 `MELD/full history-complete-selection`：25 套 complete checkpoint 恢复、1,419 条 selection query 特征缓存与 receipt 发布，selection label 未访问
- [x] 完成 MELD 唯一 full-anchor `current-only-fit`：25/25 folds、独立 fit OOF/receipt 发布，训练与推理历史均为 0，selection payload 未消费
- [x] 完成 MELD 唯一 full-anchor `current-only-complete-selection`：25 套 complete checkpoint 恢复、1,419 条 selection query 特征缓存，selection label 未访问
- [x] 完成 `MELD/no_vad history-fit`：25/25 folds、fit OOF/targets/receipt 发布，selection payload 未消费
- [x] 完成 `MELD/no_vad history-complete-selection`：25 套 complete checkpoint 恢复、1,419 条 selection query 特征缓存，selection label 未访问
- [x] 完成 `MELD/no_history_3x3 history-fit`：25/25 folds、fit OOF/targets/receipt 发布，selection payload 未消费
- [x] 完成 `MELD/no_history_3x3 history-complete-selection`：25 套 complete checkpoint 恢复、1,419 条 selection query / 3,880 个 context 的 feature-only 缓存与 receipt 发布，selection label、utility target 与性能未消费
- [x] 完成最后一个 `MELD/capacity_control history-fit`：25/25 folds、fit OOF/targets/receipt 发布，selection payload 未消费
- [x] 完成 `MELD/capacity_control history-complete-selection`：25 套 complete checkpoint 恢复、1,419 条 selection query / 3,880 个 context 的 feature-only 缓存与 receipt 发布，selection label、utility target 与性能未消费
- [x] 生成 EmotionTalk 与 MELD 各四份 `strategy-complete-selection`，完成八份 outcome-free strategy roster
  - [x] MELD/full
  - [x] MELD/no_vad
  - [x] MELD/no_history_3x3
  - [x] MELD/capacity_control
  - [x] EmotionTalk/full
  - [x] EmotionTalk/no_vad
  - [x] EmotionTalk/no_history_3x3
  - [x] EmotionTalk/capacity_control
- [x] 在 exact 八份 strategy 认证后，分别运行 MELD 与 EmotionTalk `evaluate-model-selection`，再执行双数据集 joint freeze
  - [x] MELD：`model_selection_gate_passed=false`；prospective power `0.8127`（power gate=true），七项性能门全部失败；H1–H5 中仅 H3 emotion-constraint increment 经 Holm 校正显著
  - [x] EmotionTalk：attempt 1 在标签访问前 fail closed；clean-closure attempt 2 exit 0，`model_selection_gate_passed=false`、prospective power `0.2201`（power gate=false），仅 regret-vs-current 门通过；H2/H5 Holm 拒绝但 H1、最强参考、安全、稳定性与功效均未通过
  - [x] 双数据集 hash-bound joint freeze：`joint_model_selection_freeze_passed=false`，calibration workflow、holdout、validation 与 test 均未获授权
- [x] 完成公开层最终评估、EmotionTalk 独立只读审计、三 public JSON 散列/隐私扫描、冻结 source closure 复核与 evaluator/joint 专项 `58 passed`

### Selector repair 关闭记录（2026-08-08）

- [x] Repair 1 distributional：NO-GO
- [x] Repair 2 class-balanced：NO-GO
- [x] Repair 3 emotion/VAD/3×3：fit-only gate NO-GO（0/5 utility seeds；未打开 model-selection）
- [x] 按“三种不同修复后停止”规则终止该 selector 模型族，不再进行 Repair 4 或结果驱动调参
- [x] 提交并推送 Repair 3 aggregate-only、write-once 结果与审计记录（commit `f61aaa0`）

### GPT baseline gate（2026-08-08）

- [x] 真实受限数据云 GPT：NO-GO（无凭据、外传授权、DPA/ZDR 与冻结 adapter）
- [x] 本地 7B+/gpt-oss：NO-GO（8GB 显存、内存与磁盘预算不足以支撑可重复长历史实验）
- [x] 实现并测试 synthetic-only、零网络、write-once 的 GPT adapter 合同（仅未来接口，不是性能实验）
- [ ] 在未来授权齐备前，只运行等信息 TF-IDF/SVD 与小型 causal backbone 离线基线

## Task-local best_skill card（2026-08-08）

```text
best_skill: supervised-ML empirical pipeline + open-role query policy + reproducible publication evidence
train_signal: true bidirectional supervision improves surrogate utility ordering, but the current mean-NLL selector lowers query Macro-F1; distributional repair improves task-level RMSE yet has not passed absolute safety
selection_split: EmotionTalk frozen model-selection groups 65–79 under scu_set_exploration_v1
heldout_gate: no reads from groups 80–89/90–99 or validation/test; contract tests, hash/schema checks, cluster-level inference, then an independently sourced second dataset after method lock
accepted_patterns: fit-only cross-fitting; shared-cluster paired inference; query-level one-prediction estimand; exact current/all/coverage-matched-recency comparisons; public aggregate-only outputs
rejected_patterns: mean-MLP utility tuning; label-derived inference features; pooled-utterance uncertainty; threshold selection on sealed roles; post-hoc GPT prompt/model selection
patch_scope: one target/model-family or one metric-contract correction per candidate run
reject_if: candidate fails to improve query Macro-F1 without positive excess NLL, fails the predeclared seed gate, or cannot outperform the matching recency/all-history strong baselines; after three distinct repair families, stop this selector route
```

## Next Step

严格按以下顺序推进且不并行解封评估角色：

1. 封存 HarmBench 已结束统计与中断工程审计；不实现 label access ticket，不接 terminal evaluator，不重跑统计或真实实验。
2. 完成老师四条要求对照表，并把 N3 结构性缺口转成可测试合同。
3. 完成情感领域编码器与 IEMOCAP/CPED/M3ED 许可、数据结构、标签和资源审计。
4. 冻结 N3 protocol ID、结构、超参数、效用阈值、统计合同、15 项基线/消融与成功门。
5. 只在 synthetic contract 与 fit 角色实现/测试；不打开 MELD/EmotionTalk 的任何封存角色，也不运行 IEMOCAP。
6. 形成正式运行 readiness 报告并停止，等待对未观察数据角色的单独授权。

HarmBench 现有工程证据保持封存：最终 protocol v2 pin=`58630569e7cb518b3b04fc9029bd5c78c56e409fee6ae2f36bc0a90143fc4f9a`，相邻回归分别有 `217 passed`、`81 passed`、`83 passed`、`122 passed`、`44 passed`、统计专项 `12 passed in 12.79s`，curator durability `26 passed in 0.89s -W error`。这些数字只证明合成/工程合同，不是模型性能或真实标签访问证据。中断的 prelabel/evaluator 文件保持原状并标记未完成。绝不暂存 `jzmq.dll` 或 `libzmq-mt-4_3_6.dll`。

## Confirmatory Evidence Gate

只有同时满足以下条件，才可称为“具备顶会投稿证据”；不保证录用：

1. 至少 2 个独立来源数据集完成确认实验，目标为 3 个（MELD、IEMOCAP、EmotionTalk；许可失败时使用预先声明的替代集）。
2. 严格按对话/人物/时间协议切分；历史仅来自查询时刻之前；所有选择器监督由 out-of-fold/cross-fitting 产生。
3. 与 current-only、all-history、强检索/记忆基线及校准回退基线公平比较。
4. MELD 锁定 Weighted-F1 为主指标；同时报告 Macro-F1、Accuracy、NLL、Brier、ECE、历史负迁移率、CVaR 和风险—覆盖曲线。外部确认集需在冻结协议中预先指定与标签映射一致的主指标，不得看到结果后变更。
5. 确认实验至少 5 个随机种子；报告均值、95% CI、配对效应量与 Holm 校正；按对话或人物做层级 bootstrap，禁止把 utterance 当作完全独立样本。
6. 完整 N3 必须同时满足：Accuracy 高于 independent current-only；Weighted-F1 高于最强历史基线；Macro-F1 不明显下降；配对 95% CI 支持提升；至少 5 seeds 方向基本一致；双向效用伤害率低于单向效用；去情感编码器、双向效用、模态级效用或 3×3 后性能下降。增益必须来自真实情感分类，不能只来自效用 AUC/伤害率。
7. 完成消融、标签置乱、未来历史注入检测、重复人物检测、时间反转、状态突变与恢复、模态缺失/噪声压力测试。
8. 代码、配置、随机种子、环境、聚合结果与失败实验均可复现；未授权原始数据和派生受限特征不公开。

## GPT Use Policy

- GPT 只能作为预先定义的文本零样本基线、冻结特征/教师或解释性辅助模块；不得读取测试标签、未来话语、人物跨切分资料或完整未授权音视频。
- GPT 方案必须与非 GPT 参数量/信息量可比基线比较，并记录模型快照、提示词、温度、重试和成本。
- GPT 输出必须缓存并版本化；确认实验禁止根据测试结果改提示词。
- 若没有可用 API 凭据或数据许可，先实现可离线运行的接口和 mock/开源替代基线，不伪造 GPT 实验数据。

## N3 Mainline Phases

### N3 Phase 0 — 纠偏、对照、许可与协议冻结

Status: in_progress

- 封存 HarmBench 收尾状态与中断文件，不再扩展。
- 完成老师四条要求对照、情感编码器审计和 IEMOCAP/固定替代集审计。
- 冻结 protocol、结构、超参数、阈值、统计与公开模板。

### N3 Phase 1 — Synthetic contract 与 fit-only 实现

Status: pending

- 实现六路接口、情感变量、共享 3×3、模态/联合双向效用和两级门控。
- 所有效用标签仅由 fit 内 group cross-fitting/OOF 生成。
- 运行泄漏、集合反事实、模态归因、缺失模态、参数预算和确定性合同测试。

### N3 Phase 2 — 冻结基线/消融与运行就绪审计

Status: pending

- 固定 15 项最低基线/消融、5 seeds、配对统计、效应阈值和 no-harm 门。
- 核验模型权重许可、数据授权、磁盘/显存/时长预算与 write-once 输出路径。
- 生成 readiness 报告；未获独立授权时在此停止。

### N3 Phase 3 — 预注册外部确认（需单独授权）

Status: pending

- IEMOCAP 仅在冻结后作为第三独立外部确认集运行；不得调参，正负结果均报告。
- IEMOCAP 授权/六路协议失败时，按 `CPED → M3ED` 固定顺序替代。
- 不因任何中间性能改变模型、阈值、标签映射、统计或数据集顺序。

### N3 Phase 4 — 多数据集结论、反证与论文证据包

Status: pending

- 只按冻结成功门综合 MELD、独立确认角色和预注册外部数据。
- 完整报告负结果、缺失模态/噪声/转折恢复/跨模态冲突压力测试。
- 若正向方法失败，才评估 HarmBench 备选论文路线；不得改写 N3 失败。

## Historical N2/HarmBench Phases（preserved）

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
| curator 只读检索再次假定不存在的 `meld_causal_backbone.py`，并使并行读取整体退出 1 | 1 | 无文件/数据访问；用 `rg --files`/已有 import 定位真实 schema 在 `meld_causal_backbone_loader.py` 与 `meld_multimodal_sidecar.py`，后续 curator 检索只使用已验证精确路径 |
| class-order SHA 的组合 `rg` 把带冒号的第二模式解析成 Windows 路径，返回非法文件名 | 1 | 无文件修改；改为读取已定位的 `_class_order_sha256` 精确行段，确认其 canonical JSON 绑定，后续多模式使用单一 `-e` 或分别检索 |
| curator schema 并行只读审计假定存在 `meld_causal_backbone_runner.py`，`rg` 因该路径不存在退出 1 | 1 | 无文件修改；先用 `rg --files` 定位 MELD 实际模块，再只读精确文件，禁止重复假定文件名 |
| 尝试给已完成 selection-label 代理追加 curator-ingest 子任务时并发槽已被 statistics 子代理占满，协作工具返回 `agent thread limit reached` | 1 | 未创建任务、无文件修改；保留 ingest 为后续动作，等待当前 prelabel/statistics 任务释放槽后再派发，不重复占槽 |
| sign×severity key rename 的首个组合补丁把测试文件上下文误放在 metrics 文件更新块中，apply_patch 原子拒绝 | 1 | 无部分写入；先用 `rg -A/-B` 读取两个精确区段，再分别对源文件和测试文件做小补丁 |
| Windows 上将 `harmbench_erc_*.py` 作为 `rg` 位置参数触发非法路径，导致并行只读源码审计在启动阶段退出 | 1 | 无文件修改；改用 `rg -g 'harmbench_erc_*.py'` 的 glob 过滤，禁止重复原命令 |
| 联合模型回归在 typed roster 代理正处于 `make_fit_feature_capability` 签名迁移的中间状态触发，导致 18 failed/27 errors；根因是共享工作区并行编辑而非模型逻辑结论 | 1 | 停止基于中间 API 修补；要求该代理完成明确 synthetic helper 与所有相邻 fixture 迁移，待其最终报告后再复跑 |
| 新 crossfit heldout-roster 攻击测试正确 fail-closed，但实际先在“query rows differ from derived partition”失败，测试只匹配 `context_role` 导致 1 项假阴性 | 1 | 保留更强的分区现场校验，将断言改为外层稳定错误 `invalid fit-train context roster` 后复跑 |
| processor cache parent identity 首版把目录 `st_size` 纳入稳定身份；并发 writer 创建各自临时子目录会合法改变该值，使 4 线程测试出现 1 项假阳性失败 | 1 | 不重复用可变目录大小作身份；保留 `st_dev/st_ino/file_attributes` 作为父目录对象身份，再复跑并发与完整专项 |
| 首次把 combined open-role capability 重构为 fit/selection 两阶段时，大补丁因函数返回上下文与当前文件不完全一致而被原子拒绝 | 1 | 无部分写入；改为先读取精确 dataclass/function 区段，再分成小补丁依次加入 typed capabilities、factory 与 loader，避免重复大范围替换 |
| 恢复后首次用 Windows Store 系统 Python 运行 HarmBench 专项时缺少 `scikit-learn`，3 个测试模块在 collection 阶段失败 | 1 | 未执行测试主体、未修改结果；该环境此前已知只供轻量脚本，改为定位并使用旧实验已审计的独立 `.venv`，不在系统 Python 临时安装依赖 |
| production probability seed-axis mutation 测试首版把同一 seed 概率复制 5 次，反转 seed 轴数值不变而未触发 SHA 差异 | 1 | 仅合成测试失败、无科研结果；把 fixture 改为五个可区分且仍满足 simplex 的概率面板后，query/seed/bit 三类 mutation 均被拒绝 |
| row-order mutation 已正确 fail closed，但测试正则预期 `alignment changed`，实现统一报 `production binding changed` | 1 | 仅错误消息断言不匹配；保持更一般的实现错误，不放宽验证，测试改为匹配 `binding changed` 后通过 |
| 在 `functions.exec` 的 JavaScript 组装 PowerShell 行数命令时，两次因 JS 字符串/PowerShell 反引号混用触发脚本解析错误 | 2 | 两次均未执行 shell、未修改文件；改用 JS 数组生成路径列表，并以 `[char]9` 代替 PowerShell 反引号制表符后成功。后续嵌套脚本避免在 JS template literal 中直接写 PowerShell backtick |
| 首次 MELD/full strategy 只读审计命令在内插表达式中混用赋值与管道，触发 PowerShell `Missing closing ')'` parser error | 1 | 无文件修改；把 stderr 读取拆为独立变量后重跑，只读审计成功，不重复原语法 |
| 当前 Windows PowerShell 的 `ConvertFrom-Json` 不支持 `-Depth` 参数 | 2 | 未修改任何文件；改用无 `-Depth` 的定向字段读取，并再次登记禁止重复该参数 |
| 首次组合 planning patch 使用了错误的 findings/progress 标题锚点 | 1 | `apply_patch` 原子拒绝且无部分写入；读取真实 UTF-8 标题后分文件重试 |
| 并行 `rg --files`/缺失 config 探测中有一个只读子命令以“无匹配”退出 1，使组合工具被标记失败 | 1 | 无文件修改；改用 `rg --files | Select-String` 与精确文件名只读定位，显式允许空结果 |
| 初始仓库缺少持续计划文件 | 1 | 创建 task_plan.md、findings.md、progress.md |
| PowerShell 默认编码导致中文 Markdown 乱码 | 1 | 后续读取显式使用 UTF-8，不把乱码误判为文件损坏 |
| `System.IO.Path.GetRelativePath` 在当前 PowerShell/.NET 不可用 | 1 | 使用字符串前缀裁剪或 Python 只读枚举，不重复原命令 |
| 后台实验轮询只看到 Python shim CPU 很低 | 1 | 用 Win32_Process 检查子进程，确认实际 `python3.11.exe` 正在计算 |
| 在 PowerShell 中误用 Bash heredoc `<<` 读取 JSON | 1 | 改用 PowerShell here-string 管道到 Python stdin，不重复 Bash 语法 |
| EmotionTalk benchmark 误把 dialogue 单列当全局 cluster | 1 | 首轮 CI 作废；改用 `group/dialogue` 复合键并新增单元测试后重跑 |
| PowerShell here-string 插值将 Unicode 私有绝对路径传给 Python 时乱码 | 1 | 从私有 experiment 目录运行并使用相对路径，避免跨层 Unicode 插值 |
| 首次端点诊断把无历史行纳入 utility selector 拟合 | 1 | 第一折完成前终止无效运行；恢复旧协议的 `history_count>0` eligible 约束，31 项测试通过后重启 |
| 私有端点缓存首版把 selector 特征量化为 float32 | 1 | direct-mean 复核发生轻微漂移，hurdle 结果作废并私有隔离；缓存改为无损 float64 后重建 |
| PowerPoint COM 新建/打开演示文稿返回 `0x80048240` | 3 | 停止重复 COM 尝试；使用 `@oai/artifact-tool` 生成可编辑 PPTX，PDF 改由最终渲染经 ReportLab 导出并回渲验证 |
| 本机未安装 draw.io 桌面端 | 1 | 无法使用 live draw.io 后端；保留原生 PowerPoint 可编辑形状路线 |
| Poppler shim 无法直接处理含日文目录的输入路径 | 1 | 将最终 PDF 临时复制到 ASCII 临时目录后调用真实 `pdftoppm.exe`，完成回渲检查 |
| 一次多文件 `apply_patch` 因 findings 标题上下文不匹配而整体拒绝 | 1 | 先用 UTF-8 读取真实文件头，再按稳定英文标题分别插入；未发生部分写入 |
| 因果 backbone 初版发布配置把真实 1,536 维 WavLM mean+std 写为 768 维 | 1 | 只读核对私有特征 shape 后同步修正默认值与发布配置；仍低于 2M 参数门 |
| v2 初版把 oracle opportunity regret 当作相对 fallback 的安全 regret，且 ensemble 排名与独立 seed bootstrap estimand 不一致 | 1 | 结果生成前停止运行；拆分 excess NLL/oracle regret，排名与 CI 统一为五独立运行均值，改用 seed→cluster 嵌套配对 bootstrap 后重启 |
| 跨三文件 `apply_patch` 因 progress 标题上下文匹配失败而整体拒绝 | 3 | 三次均确认无部分写入；已改为逐文件、最小标题锚点补丁并成功，后续禁止组合计划三文件提交 |
| 再次把含 Unicode 的私有绝对路径插入 PowerShell here-string 后传给 Python，触发已知乱码错误 | 2 | 未修改文件；立即改回私有 experiment 工作目录加 ASCII 相对路径。后续所有私有 Python stdin 脚本禁止嵌入该绝对路径 |
| 单文件 progress 标题-only 插入仍因隐式相邻上下文校验失败 | 1 | 改为显式替换文件前三行（标题、空行、首节标题）后成功；后续顶部插入采用三行锚点 |
| distributional query 首轮重算的 GroupKFold positions 与冻结 fit checkpoint 不一致 | 1 | 根因是 NumPy 2.3.1→1.26.4 / sklearn 环境下等长 group 的非稳定 tie-order 漂移；runner fail-closed。修复以已散列 checkpoint positions 为权威，并验证 exact cover、无重复、边界、cluster purity 与 59-D bitwise 绑定；不改模型、阈值或 seed |
| class-balanced 首轮运行的报告 schema 缺逐 utility-seed 指标与组合 reproducibility manifest | 1 | 在结果生成且任何 model-selection 数字可见前中止；保留 abort 日志，仅补报告/provenance 后按完全相同冻结配置重跑 |
| 为判断 EmotionTalk 文本语言而直接格式化 `transcription.csv` 前 8 行，意外显示同一对话的 `emotion` 列 | 1 | 立即停止；key-only SHA 角色审计确认该对话为 bucket 16（fit），未触及 65–99，但这些标签仍不得进入特征、配置、模型选择或论证。后续文本查看先显式列白名单并由 sidecar fail-closed |
| PowerShell `foreach {...} |` 语法产生 empty pipe parser error（元数据审计与 sidecar 目标核验各一次） | 2 | 固定先把 `foreach` 结果赋给变量，再单独管道格式化；两次均未产生文件或外部状态变化，后续禁止再用内联 foreach-pipe |
| 试探的 ACL Anthology `2022.wassa-1.22` 不是 XLM-EMO；Semantic Scholar 无 key 查询又返回 429 | 1 | 明确作废该文献 ID，不重试受限 API；后续用模型仓库 citation metadata、Crossref/ACL title search 或已安装学术检索工具交叉核验 |
| 请求 XLM-EMO GitHub 的 `README.md` 返回 404 | 1 | 通过 GitHub contents API 确认实际文件为 `README.rst`，随后只读核验；未重复错误路径 |
| Crossref DOI 精确入口不支持 `select` 参数 | 1 | 去掉列表路由专用参数后重查同一 DOI，成功核验 PAD 理论文献元数据；未把失败响应当成检索成功 |
| Windows PowerShell 将传给 `rg` 的 `run_*` / `test_*` 路径通配符按字面路径处理 | 1 | 不重复该写法；后续只传目录并用 `rg -g 'pattern'` 过滤，已有命中不作为完整审计结果 |
| causal runner 在切 0–79 前读取完整 EmotionTalk train/validation 特征；MELD loader 未以 manifest/真实文件 hash 绑定来源 | 1 | 独立审计在任何真实训练前阻断；必须生成物理隔离的 EmotionTalk open-role feature sidecar，并让 MELD loader 重算 alignment、核验冻结 manifest/hash、提供正式 CLI，测试不得使用虚假 hash 代替来源证明 |
| 一次组合 `rg` 因 PowerShell 字符串转义形成未闭合正则 | 1 | 未修改文件；改为多个 `rg -F`/`rg -e` 字面检索并成功完成陈旧接口审计 |
| 组合只读命令中第二个 `rg` 无命中返回 exit 1，导致工具整体标记失败 | 1 | 已确认前置输出完整且未修改文件；后续对“可能无命中”的审计使用 `Select-String`/显式允许空结果，不把无命中误判为代码失败 |
| 用全局 Python 运行 causal runner 专项时因缺 PyTorch 导致整文件 skipped 且退出非零 | 1 | 未产生训练或文件修改；改用已冻结旧实验 `.venv`，同一专项 `16 passed`。后续依赖 torch 的测试固定使用该环境 |
| 冻结旧实验 `.venv` 未安装 `ruff` | 1 | 不为格式检查污染冻结训练环境；改用 `py_compile`、JSON parser、`git diff --check` 与确定性尾随空白扫描 |
| 从仓库根目录直接运行新增专项测试时未设置 `PYTHONPATH=experiment/src`，导致 `ModuleNotFoundError: hva_affect` | 1 | 未产生训练或结果；改为解析同级冻结 `.venv` 的相对路径，并显式设置仓库 `experiment/src` 为 `PYTHONPATH` 后再运行 |
| 首次冻结暂存后的 `git diff --cached --check` 检出 6 个新增文件 EOF 多余空行 | 1 | 未提交、未推送、未运行真实 gate；对稳定文件用 `apply_patch` 去除多余空行，等待并行 Stage-B 收尾后重新暂存并全量复核 |
| 用 `Measure-Object -Property` 直接传脚本块统计 staged numstat 时 PowerShell 拒绝 Hashtable property | 1 | 文件大小与禁用扩展名审计已完成且未修改文件；后续先将 numstat 映射成数值对象再求和，不重复该语法 |
| 只读检索误写不存在的 `causal_backbone_stage_b.py` 文件名 | 1 | 无文件修改；通过 `rg --files` 定位实际文件为 `causal_backbone_evidence_stage_b.py`，后续使用精确路径 |
| DLL 元数据审计末尾的 `git check-ignore` 因两个文件未被忽略而返回 exit 1 | 1 | 前置只读元数据/散列输出完整且无文件修改；单独复核确认它们是指向 Aha IPC 临时目录的运行时符号链接，后续显式排除暂存，不把“未忽略”重复当命令错误 |
| 再次尝试跨三份 planning 文件组合补丁，因 `findings.md` 标题误写导致整体拒绝 | 1 | 确认无部分写入后按真实 UTF-8 标题逐文件应用；后续 planning 更新严格执行单文件补丁 |
| 新增并发 public-writer 测试时只注册了 1 个 Holm 假设，而合同要求至少 2 个 | 1 | 未写科研结果；补齐第二个预声明 mean-regret 假设后，同一并发 writer 测试与 fit-receipt 并发测试均通过（`2 passed`） |
| 再次误用 PowerShell 内联 `foreach {...} |` 枚举八份配置，触发 empty-pipe parser error | 3 | 无文件/结果修改；立即改为 `$rows = foreach (...) {...}; $rows | ...` 并成功只读。后续配置枚举只使用该两阶段写法或 Python，不再使用内联 foreach-pipe |
| 再次把 `carma_affect_relation_*_v1.json` 作为 `rg` 字面路径，随后修正的检索又因零命中退出 1 | 2 | 两次均为只读且无状态变化；后续可能零命中的配置检索固定用 `Get-ChildItem ... | Select-String`，不再向 `rg` 传 Windows 通配路径，也不把零命中当错误 |
| 合并磁盘与 `Win32_VideoController` CIM 查询超过默认 10 秒 | 1 | 无状态变化；拆为轻量 `Get-PSDrive` 与 `nvidia-smi --query-gpu` 后成功，不重复慢 CIM 路径 |
| 已登记禁止的 Windows `rg` 路径通配符写法被再次误用，测试文件搜索返回路径语法错误 | 3 | 三次均为只读失败且无状态变化；本轮已改用目录参数加 `rg --glob` 成功。达到 3-strike，后续所有 Windows `rg` 调用禁止任何路径通配符，只允许目录参数配 `--glob/-g`，命令模板必须直接复用而不再手写 glob 路径 |
| 将长输出命令直接管道到 `Select-Object -First` 导致消费端提前关闭、整体返回失败 | 2 | 两次均为只读且前置输出可见、无文件修改；后续必须先完整捕获到 PowerShell 变量，再对变量执行 `Select-Object -First`，禁止把 `rg`/CLI 原生进程直接接到提前关闭的消费端 |
| 首次八变体 `fit-lineage-validate` 把 shell 超时错误设为 1 秒，读验证进程被工具终止 | 1 | 验证入口只读且未修改任何产物；把 shell 内部超时调整为 300 秒后再运行，不改变科研参数或 write-once root |
| 第二次 `fit-lineage-validate` 手工把保留名 `production_source_snapshot_v1` 重复加入 `--config` | 1 | CLI 在读取训练/性能数据前 fail closed；核对 `_bind_source_snapshot_config` 后移除重复参数，由三个显式 snapshot 参数自动绑定同一 manifest，再运行八项全部通过 |
| 检查新日志路径时再次使用已禁止的 PowerShell 内联 `foreach {...} |` | 4 | 只读 parser error、无状态变化；立即改为先赋值 `$rows=foreach (...) {...}` 再单独管道。后续所有枚举检查改用该两阶段模板，不再手写内联 pipe |
| 首次 `EmotionTalk/full history-fit` 的 PowerShell 包装器设置 `ErrorActionPreference=Stop`，把 PyTorch AMP FutureWarning 的 stderr 误判为致命错误 | 1 | Python 在首个 fold 训练前被包装器中止；保留 claim、lock、processor 与零长度日志，不删除或覆盖。使用完全相同 frozen 参数、同一 root 和 CLI `--resume`，仅把 native stderr 策略改为 Continue 并写入新日志 |
| current-only 完成后散列枚举先把 `FileInfo` 投影成不含 `FullName` 的对象，再错误读取 `$f.FullName` | 1 | 只读 `Get-FileHash` 参数错误、无文件修改；保留原对象用于散列、另建投影用于显示后重跑，artifact/receipt SHA 与 stdout 一致 |
| 并行只读审计在最终 GO 消息后又导入冻结包，重新生成 ignored `__pycache__`；首次 no-history-3×3 启动被源码闭包在 0.805 秒 fail closed | 1 | 未创建训练 root、未读取 payload、无训练进程；把两批可再生缓存精确移到 run quarantine，等所有审计进程结束后再次验证 `.py`-only 闭包。下一次使用新日志且仍按首次运行、不加 `--resume` |
| 续接时误把顶层 `functions.wait` 当作 `functions.exec` 内的嵌套工具调用 | 1 | 立即改用顶层 wait；活跃训练 cell 未受影响、无文件或进程状态变化 |
| 元数据枚举再次误写 PowerShell 内联 `foreach {...} |` | 7 | 只读 parser error、无状态变化；最近一次发生在收口 UTF-8 枚举，立即改回 `$rows=@(); foreach (...) {...}; $rows | ...` 两阶段模板并成功完成七文件验证 |
| EmotionTalk 首次 `evaluate-model-selection` 前的独立验签导入在冻结源码闭包生成 ignored `__pycache__`，触发 `production package bootstrap integrity check failed closed` | 1 | 入口在首次项目包 import、参数解析、payload/label archive 访问与性能计算前退出；stdout 为空，private/public 目标均未创建。保留 attempt 1 日志，将 28 个 `.pyc` 完整移入 run quarantine，重验 `.py`-only closure 后以相同冻结协议和新日志启动 attempt 2，不使用 `--resume`、不覆盖失败现场 |
| 首次 MELD aggregate verifier 的 PowerShell→Python 原生命令引号被剥离，Python 将环境变量键误作名称 | 1 | verifier 尚未执行且无文件修改；把 `-c` 代码改为 PowerShell 双引号包裹、Python 内部单引号键后，以首次 stdout 的固定 receipt SHA 验签通过 |
| 安全策略拒绝对 frozen `__pycache__` 做递归及逐文件删除 | 2 | 两次均未删除任何内容；改用经过绝对路径、内容类型和目标父目录验证的 `Move-Item`，把完整缓存可恢复地移入 run quarantine |
| 对 0 字节 EmotionTalk stderr 调用 `[regex]::Matches($null, ...)` 触发只读 ArgumentNullException | 1 | stdout 与 stderr 元数据已完整读取、评估不受影响；以文件长度 0 作为空日志证据，后续正则统计前先把 null 归一为空字符串 |
| 收口专项测试首次同时使用 `python -I` 与外部 `PYTHONPATH`，隔离模式按设计忽略该路径，导致两个模块在 collection 阶段 `ModuleNotFoundError` | 1 | 未执行测试主体、未修改科研代码或结果；对仓库工作副本改用普通解释器加显式 `PYTHONPATH=experiment/src`，同两份专项最终 `58 passed`。正式冻结入口仍保持 `python -I` |
| S1 中间态 compile 命令把 JavaScript 局部变量误当作 PowerShell 环境变量 `$env:CARMA_PY` | 1 | shell 在解释器启动前失败、未修改代码或结果；改为显式引用冻结科研 `.venv` 的绝对解释器路径，同四模块 `py_compile` 通过，后续不跨脚本作用域假定环境变量存在 |

## Decisions Made

| Date | Decision | Rationale |
|---|---|---|
| 2026-08-07 | 以历史负迁移 benchmark 为当前主路线，CARMA 方法作为待重构候选 | 已冻结 q90 安全回退失败，但历史伤害与多模态预测增量成立 |
| 2026-08-07 | 将 GPT 限定为基线、冻结特征/教师或解释模块 | 保持核心问题不变并防止标签/时间泄漏与不可复现提示调参 |
| 2026-08-07 | 主探索模型选择 SCU-Set，RCPS 作为安全层 | SCU-Set直接学习逐候选/集合边际效用并支持可逆重用；单纯改阈值创新不足 |
| 2026-08-07 | SCU 目标升级为 sign×severity/quantile 分布建模 | 双数据集探索显示低伤害概率不等于低平均regret，单一分类或均值目标均不足 |
| 2026-08-07 | Phase 0 通过并进入 Phase 1 | MELD 第三次独立重跑与冻结结果字节/解压内容完全一致；32 项私有工程测试和 28 项公开新增合同均通过 |
| 2026-08-07 | 直接端点 mean utility 路线暂停，启动两部式 repair 1/3 | 独立模型选择组中 harm AUC=0.728、目标跨 seed 稳定，但直接 mean Spearman=-0.002；伤害发生与严重度不能由单一均值头替代 |
| 2026-08-08 | 教师前三点合并为一项组合创新，不分别宣称首创 | 情感理论、六流和模态交互均有高重叠邻近工作；潜在新意在不同集合双向效用、train-only OOF、3×3关系与校准回退的完整闭环 |
| 2026-08-08 | 强制 `T != S union {h_i}` | 若 `T=S union {h_i}`，前向加入与后向删除代数相同，所谓双向无新增信息 |
| 2026-08-08 | 端点 hurdle 不再作为进入新方法的充分依据 | 可见 hurdle 结果来自已隔离 float32 缓存且关键门失败；必须重新生成无损不同集合 OOF 监督 |
| 2026-08-08 | repair 2/3 class-balanced 判定 NO-GO，不再调参 | 虽把 true Macro-F1 从 0.52723 修到 0.54000并胜过current，但accuracy下降，且落后all-history/recency/backward；严格三参考联合门仅2/5 |
| 2026-08-08 | repair 3/3 固定为情感概率3×3关系＋VAD特征增强，full为预声明primary | 前两条修复只改变效用目标/类平衡，尚未真实实现老师要求的情感理论与六流关系；复用repair2 policy可把增量归因于新特征而非再次换目标 |
| 2026-08-08 | repair 1/3 distributional query-level 判定 NO-GO，不再调参 | true excess NLL显著改善但Macro-F1未高于current且比同coverage recency低0.01729，0/5过预注册门；forward/backward各4/5不能转化为双向成功 |
| 2026-08-08 | 下一独立模型族预选 Affect-Relation Causal Backbone（N2），plain causal/quantile selector（N1）作强基线 | N2 最直接落实情感理论、当前/历史3×3关系和双向效用；必须先冻结同参数control、no-VAD/no-3×3消融与fit-only nested gate，不能依据selection结果调结构 |
| 2026-08-09 | N2 production 工程与统计链单项 GO，但性能结论仍未知 | source/joint/CLI 独立审计与最终 N2 审计均无 P0/P1；504 项全仓回归通过。只有新 freeze、detached snapshot 与 CUDA smoke 后才授权真实训练，且 model-selection 前不得改结构/阈值 |
| 2026-08-09 | N2 Stage-A 八变体 lineage 门通过，授权顺序启动 EmotionTalk/full fit-only history 训练 | 八份 receipt/map/lineage 均由 frozen source snapshot 再验证，选择标签/特征 payload 未打开且 worktree clean；该授权只覆盖 fit-only 训练，不授权 model-selection、calibration、holdout 或 test |
| 2026-08-09 | 保持既定 D/E/C history 分配，并把未来 MELD current-only 调度到 D 盘 | 十个训练产品保守约 18.0 GiB；调整 current-only 的物理路径可把 D/E 最终余量平衡到约 3.3/4.5 GiB，不改变数据、模型、种子或统计协议；每个新 root 前仍须重新查盘 |
| 2026-08-09 | 冻结 MELD N2 model-selection 为单数据集 NO-GO，不据此修补 N2 | prospective power 0.8127 达标，但七门全部失败，H1/H2 均未 Holm 拒绝；仅 H3 显著不足以支持整体方法或解封后续角色 |
| 2026-08-09 | 冻结 N2 双数据集 joint 为 NO-GO，终止在当前角色上继续调参 | EmotionTalk gate=false 且 power=0.2201，MELD gate=false；joint verifier 给出三个预注册失败原因并拒绝 calibration/holdout/test 授权。后续只能使用全新 protocol/数据角色或转 benchmark/否证路线 |
