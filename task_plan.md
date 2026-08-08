# CARMA-Affect 顶会证据推进计划

## Goal

在不访问当前/未来标签、不泄漏人物或时间信息的前提下，建立并验证“个体历史何时造成情感预测负迁移、如何预测这种伤害并安全回退”的可复现研究证据；最终交付满足顶会审稿标准的多数据集结果，或形成有充分否定证据支撑的 benchmark/问题论文路线。

## Current Phase

Phase 1 — 预注册与 benchmark 冻结（in_progress）

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
- [ ] 只在新 freeze commit 推送后运行 fit-only OOF gate

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

Repair 3 已以 0/5 永久 NO-GO 结束且不得重跑。N2 的 source snapshot、history/current/strategy/evaluator、单数据集 performance gate 与双数据集 joint freeze 已关闭全部已知 P0/P1；根任务全仓回归为 504 passed，最终独立 N2 审计为 167 passed。当前唯一下一动作是完成隐私/大文件/禁用文件审计并推送新 freeze；随后从该 commit 创建 clean detached worktree、仓库外 source manifest，并先做不读取真实标签的最大形状 CUDA smoke。只有 smoke 证明 batch=64 可行后才创建跨盘全新 write-once roots并启动真实 N2 open-role 训练；若 OOM，只能在任何性能结果可见前冻结等价 batch=32 应急配置并重新走完整 freeze。真实性能失败必须保留并进入 underpowered/否证路线，不得据结果修改 N2 结构、参考或阈值。

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
| 当前 Windows PowerShell 的 `ConvertFrom-Json` 不支持 `-Depth` 参数 | 1 | 未修改任何文件；改用 `python -m json.tool` 与无 `-Depth` 的定向字段读取，不重复该参数 |
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
| 已登记禁止的 Windows `rg` 路径通配符写法被再次误用，测试文件搜索返回路径语法错误 | 2 | 只读失败且无状态变化；立即改为目录参数加 `rg -g 'test_*.py'` 并成功，后续审计固定使用该形式 |
| 将 CLI `--help` 直接管道到 `Select-Object -First` 导致消费端提前关闭、并行检查整体返回失败 | 1 | 无文件修改；改为完整捕获 help 文本后再取首行，compileall、JSON、diff 与 isolated CLI 四项均重新独立通过 |

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
