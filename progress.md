# CARMA-Affect Progress Log

## 2026-08-08 — 教师三点整合与新流程冻结

- 完成文献新颖性边界复核：三个单点不能分别主张首创，方法主张收敛为“不同集合双向效用 × 情感理论 × 3×3关系 × 校准回退”。
- 生成新的单页科研全流程图；PPTX 使用原生可编辑形状，另导出 PNG 和单页 PDF。
- PowerPoint COM 新建和打开均重复返回 HRESULT `0x80048240`，draw.io 桌面端未安装；按技能降级到 `@oai/artifact-tool`，不影响最终可编辑性。
- PPTX 经独立 PowerPoint/LibreOffice 渲染工具回渲，`slides_test.py` 报告无越界；PDF 经 pypdf 检查为 1 页、960×540 pt，并由 Poppler 回渲人工检查通过。
- 新增 `bidirectional_emotion_utility.py`：非平凡双向联盟任务、benefit-positive `u+`/`u-`、不同集合采样、3×3关系和 VAD/shift 特征。
- 新增冻结配置 `bidirectional_emotion_utility_v1.json`，明确角色切分、数据集、基线、指标、消融、train-only 与确认性 GO 门。
- 新增 7 项合成合同测试；全部公开测试为 `41 passed in 2.74s`。
- 新增完整科研流程文档与约 3 分钟讲解稿；README、证据索引、当前进度与实验说明同步更新。
- 新增 readiness 聚合文件，明确当前只证明工程/研究合同就绪，不是模型性能证据。
- 下一步是完成无损 float64 train-only OOF base cache 并产生四组不同集合概率；80–89 calibration、90–99 internal holdout 和 external test 继续封存。

## 2026-08-07 — Phase 0 完成并进入预注册阶段

- MELD 独立完整重跑完成，stderr 为空；新聚合 JSON 与原始/既有 repeat 的 SHA-256 均为 `ccc6b1937e7d68eda9033c646152e472d1c294cc76c3e60a6a88cdae94f51943`。
- 三份 MELD per-query gzip 解压后均为 218,468 bytes，SHA-256 均为 `cc6073a1f45b4d9898fc47d5b0b687b2edd35fcac4d522290c575953ffec6a8f`；逐字节一致。
- Phase 0 复现与资产审计判定完成；研究转入 Phase 1。EmotionTalk/MELD test 继续封存，当前结果只作探索。
- 修正 SCU 角色名称，明确 0–64 同时用于 base/utility 拟合，65–79 只做模型选择，80–89 校准，90–99 内部 holdout 保持封存。
- 新增双数据集 benchmark 探索报告；公开输出仍为 aggregate-only，不包含逐查询、对话、说话人或标签记录。
- 冻结 EmotionTalk train-only 端点诊断配置：5-fold group cross-fit、5 个 base seeds、5 个 risk seeds；94 个复合对话组用于独立模型选择，59 个校准组和 55 个内部 holdout 组不读取性能。
- 第一次端点训练在第一折完成前被主动终止：代码审计发现 utility selector 拟合未排除无历史查询。未生成结果文件；修正为仅 `history_count>0` 后，公开合同测试增至 `31 passed` 并重新启动。
- 修正后的 EmotionTalk 三模态端点诊断完成：模型选择组 2,442 个有历史查询、94 个复合对话 cluster；校准、内部 holdout、validation、test 均未用于训练或指标。
- 多 seed target 稳定性通过：fit/model-selection 的 pairwise Spearman 中位数为 0.907/0.918，平均多数同号一致率为 0.960/0.961。
- 伤害分类信号通过（AUC 0.728），直接 mean utility 信号失败（Spearman -0.002）；总体进入随机子集增强的联合门为 FAIL。
- 25%/50% 覆盖出现符号—严重度偏好反转：harm 头显著降低伤害率，但因罕见大伤害使平均 regret 更差。冻结 repair 1/3 为两部式 hurdle：分别建模伤害概率、伤害幅度和收益幅度。
- 为避免每次修复都重跑 base，新增只保存在私有 artifacts 的无标识符端点缓存；缓存只含特征、效用目标、history count 与整数 cluster code，不发布。
- 首版缓存将 selector 特征压成 float32，导致缓存上 direct-mean Spearman 与原始 float64 入口轻微不一致。该 hurdle 运行判为协议无效，结果与缓存已移入私有 quarantine；缓存合同改为 float64 无损保存后重建。

## 2026-08-07 — 持续科研目标启动

- 用户授权继续按照预定方案开展科研，并允许修改模型或评估 GPT 模型。
- 创建长期目标：形成满足顶会证据门槛的多数据集、无泄漏、统计可靠结果；若方法不成立，则形成有充分否定证据的 benchmark/替代路线。
- 启用 planning-with-files、scientific-critical-thinking、statistical-analysis 工作流。
- 检查发现仓库原先不存在 task_plan.md、findings.md、progress.md，现已建立。
- 下一动作：仓库、数据、环境与复现入口审计。
- 已运行 session catchup；未发现需恢复的未同步内容。
- 已确认仓库主目录结构和无额外 AGENTS.md 约束。
- 已完成仓库文件清单与公开数据边界初读；识别出 EmotionTalk 主执行链和 5 组测试。
- 记录错误：PowerShell 默认编码读取中文 Markdown 出现 mojibake；后续命令统一指定 `-Encoding UTF8` 或 Python UTF-8 模式。
- 已审阅当前方法、冻结配置和下一步文档；确认 q90 gate、5-fold cross-fitting、5 seeds 和 test fail-closed 约束。
- 已把“逐候选/集合效用与可逆重用”登记为后续方法重构的主要缺口。
- 完成本机首轮环境盘点：RTX 4070 Laptop 8GB、C盘约28GB可用、系统Python缺训练依赖，但旧实验目录存在独立 `.venv`。
- GPT 调用条件审计：当前无 OpenAI/Azure OpenAI key，因此不会伪造或声称已完成 GPT 实验。
- 发现旧研究目录保留了比公开仓库更完整的 MELD/EmotionTalk 实验工程，下一步精确审计可复现资产。
- 核验旧 `.venv`：包含 CUDA PyTorch、Transformers、scikit-learn 与完整科学计算栈，可直接用于复现。
- 发现 EmotionTalk 特征/模型 bundle 与 MELD 聚合结果资产；下一步记录精确文件名、散列、split 与可重复入口。
- 确认 EmotionTalk 转写与 162.7MB 冻结三模态特征在本机；原始 Audio/Multimodal 归档未找到。
- 确认 MELD train/dev/test 标注均在本机，但 test 继续封存。
- 记录错误：PowerShell 环境不支持 `Path.GetRelativePath`，artifact 相对路径枚举改用字符串前缀裁剪。
- 已成功用替代命令生成完整 artifact 清单。
- 重新运行旧实验工程全部合同测试：`32 passed in 3.90s`。
- 确认 EmotionTalk 冻结 bundle/manifest/per-query 与 MELD repeat 资产均在本机。
- 独立核对 MELD 复现审计：聚合 JSON 字节相同，per-query 1108 行逐值相同，最大数值差 0。
- 独立核对 EmotionTalk freeze manifest：config、features、bundle 散列一致，公开 validation 结果散列一致。
- 查明 MELD 复现入口和输入合同：官方 train/dev CSV + D盘 35维 handcrafted audio NPZ；test 未读取。
- 下一动作：验证 D盘输入散列后，在新输出目录独立重跑 MELD 音频文本风险实验。
- MELD train/dev CSV 与两份 audio NPZ 的 SHA-256 均与冻结 preflight 一致。
- 已启动全新 `artifacts/repro_current/` 独立复现实验；输出隔离，未覆盖旧结果，test 未作为参数传入。
- 审计 EmotionTalk/MELD per-query schema 与规模；确认它们适合统计诊断但不能直接支撑逐候选新模型。
- 复现实验仍在运行。首次轮询只观察到 Windows Store Python shim（PID 24104），真实计算进程是其子进程 `python3.11.exe`；后续同时监控进程树与输出文件。
- 审计核心代码与 EmotionTalk bundle 结构；确认新逐候选模型需要重新生成 train-only cross-fit 中间工件，不能直接复用冻结 bundle 训练。
- 读取既有独立 HVA-Affect 计划与 findings，继承 MELD 跨 split 污染、EmotionTalk speaker 泛化和 IEMOCAP 许可硬门，避免重复踩坑。
- 重新读取当前计划后确认仍处 Phase 0；独立 MELD 重跑持续占用真实子进程 CPU，尚未产出结果或错误日志，因此不提前判定失败。
- 已核对公开依赖与测试风格，统一负迁移 benchmark 可在现有环境内实现且无需额外大包。
- 新增 `negative_transfer_benchmark_v1` 冻结探索协议、统一聚合评估模块、CLI 和 5 项合成合同测试。
- 新模块覆盖自然伤害、重尾/CVaR、cluster tail concentration、history-depth、current/all/oracle、风险—覆盖、strict upper 与 dialogue bootstrap；输出合同禁止行级/cluster 标识符。
- 新 benchmark 单元测试与原合同测试共同通过：`22 passed in 5.11s`；全部 experiment 源码/脚本 compileall 通过。
- 已启动 EmotionTalk aggregate-only benchmark（2000 次 dialogue bootstrap），输出目标位于公开 `results/`，运行日志留在私有工作目录。
- MELD 独立重跑与 EmotionTalk benchmark 均有真实 Python 子进程运行，暂未产生错误日志；继续并行推进协议与代码，不提前终止长计算。
- 盘点 D/E 盘和 Hugging Face 缓存：剩余空间有限，仅已有 WavLM/DINOv2；当前不下载 7B+ LLM。
- 依据 scientific-brainstorming 完成四候选独立生成、预定义标准、区间评分与对抗审查。
- 决定主探索 SCU-Set，RCPS 为安全层/强基线，regime 模型为压力测试，GPT Critic 延期。
- 新增完整模型决策记录与 `scu_set_exploration_v1.json` train-only 探索合同；内部 holdout 10% 继续封存。
- 新增配置全部通过 JSON 解析，`git diff --check` 通过；新增文件尚未提交，符合先验证后发布顺序。
- EmotionTalk aggregate benchmark 输出文件已生成，下一步核验运行日志、隐私合同与核心统计。
- 记录读取错误：PowerShell 不支持 Bash heredoc；未修改任何结果文件，改用 here-string stdin。
- 完成首轮 benchmark 点估计审计并发现复合 cluster bug：adapter 只用了 dialogue，错误合并跨 group 的同号对话。
- 首轮结果的 cluster CI 明确作废；将保留到私有 quarantine 供审计，修复为 group/dialogue 后重新运行，不把错误结果上传。
- 已将错误 aggregate JSON 移入私有 quarantine，可恢复且不再位于公开仓库。
- benchmark 已支持复合 cluster，新增防止跨 group 同号 dialogue 合并的合同测试；修复后 `23 passed in 5.23s` 且 compileall 通过。
- 已重新启动修复后的 EmotionTalk 2000次复合对话 bootstrap，并并行启动 MELD 2000次 dialogue bootstrap；两者均只写聚合结果。
- 审计 SCU 与现有 train_only 的接口差距：必须新增 fold 内随机子集增强、多seed OOF稳定性检查和专用 pair-task 工件，不能把端点模型直接外推到任意子集。
- 新增 SCU-Set 纯合同模块：确定性 group 角色切分、候选上限/随机子集任务、效用目标、无标签 pair features、逐步且可逆的上界选择。
- 新增 5 项 SCU 合成测试，特别检查候选不在子集、效用符号、特征不含 label/gold、查询间不永久删除记忆。
- SCU 与全部公开合同测试共同通过：`28 passed in 5.24s`；compileall 通过。
- 修复后的 EmotionTalk benchmark 和 MELD benchmark 均已完成，stderr 为 0；独立 MELD 原始模型重跑仍在计算。
- 聚合合同交叉核验首次因 PowerShell→Python 的 Unicode 绝对路径插值乱码失败；结果文件未修改，改为在私有目录使用相对路径。
- 修复后的 EmotionTalk/MELD aggregate benchmark 完成统计审计与隐私交叉核验，两个数据集均 PASS。
- 发现符号—严重度目标错配：harm 概率排序与 mean utility 排序对伤害率和平均 regret 给出相反选择；已登记 H1–H3，待 train-only/holdout 反证。
- MELD 独立重跑仍有真实 CPU 进展、无错误日志；长运行不影响其他 Phase 0 工作。
