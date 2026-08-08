# CARMA-Affect Progress Log

## 2026-08-08 — Repair 3 post-run 聚合审计启动

- 确认运行基线为已推送 freeze commit `fddcda76ae326602cd6717eb95251d0c2bd24bff`。
- 只读计算 Repair 3 结果 SHA-256：`cbdb69b81db27195c86f032cdd263c17718ad5842b363c6b0f4afaee69a45504`。
- 定向解析确认 fit-only gate 为 NO-GO、0/5 seeds；model-selection 未执行，所有 sealed roles 保持未打开。
- 未重跑实验、未删除工件、未读取 row/group identifiers 或封存 payload。
- 正在并行完成 aggregate schema/privacy/hash 审计与 Causal Stage-B producer/bridge 实现。
- 独立 post-run 审计结论为 artifact integrity/privacy PASS、Repair 3 method gate NO-GO；顶层 schema、aggregate-only privacy、canonical manifest 与十项本地来源 hash 均通过。

## 2026-08-08 — Repair 3 真实运行前 High 闭环与主进程复核

- 在任何真实 sidecar payload 或 fit-gate 结果打开前，冻结无标签 opaque-group 80/20 fit-internal gate：namespace `CARMA-Affect/Repair3/fit-internal-gate/v1`，SHA 排序、ceil 20%、无分层/无 salt 搜索；projector、utility 与阈值仅在 gate-train 拟合，gate-eval 只在预测承诺后评分。
- 新增可重算的 rank-59 59→299 等信息容量对照：descriptor SHA-256 `b8d18ce77d49f4b3bc59bd68c6b6ecf3fee136769cc234744eaa293f2a47e522`，matrix SHA-256 `20eb5664adacd98324261590d360d1ad2e30306dc0d90e0b622a388f2e1b1f36`；primary/control 均 10,162 参数，59D reference 为 2,482 参数。
- fit gate 现要求同一 utility seed 同时相对 59D 与 capacity control 满足 `ΔMacro-F1≥0.002`、`ΔNLL≤1e-12`、`Δaccuracy≥0`，且至少 4/5 seeds；selection 与 optional-teacher gate 同样要求 accuracy no-harm。
- 冻结输入 manifest/model SHA、唯一 write-once 输出、配置 TOCTOU 多阶段复核与报告白名单。实现代理报告 Repair 3 专项 24 passed、相关 71 passed、全仓 244 passed；主进程另以显式 `PYTHONPATH=experiment/src` 独立联合复跑 Repair 3 + causal runner 专项为 `34 passed in 9.62s`。
- Repair 3 最终代码/config/test/runner SHA-256：`55e00d59b77798828140dace58a1bc290a085d13791310085a0c8bdd04590a03` / `0e664554fb6859c634da54439f8f98ed91c8274ddb0d8699ed2854accd0ce781` / `a08fd9249f26681a281cf238b0b60280598961d90124ade34cbd6abdc24565ea` / `74229e4c6cfb73b9c2039d39e81db65a706140df791715ff8fd8c1db6c10ba9d`。
- 仅做字节级预检确认 `D:/HVA-Affect_data/EmotionTalk/carma_open_role_sidecars_v2` 四文件存在且 SHA-256 与公开 manifest 完全一致；没有反序列化任何 payload。真实 gate 继续等待独立审计 GO。
- 首次从仓库根运行专项测试时遗漏 `PYTHONPATH`，仅产生 import collection error、未训练或写结果；已在 task plan 错误表记录，并改为解析同级冻结 `.venv` 加显式源码路径后通过。
- 主进程全仓独立复核完成：`244 passed, 11 warnings in 46.22s`；warnings 仅为合成小迭代 MLP 未收敛提示和既有 PyTorch AMP deprecation。`compileall`、38 个公开 JSON 解析、`git diff --check` 与源码/配置/Markdown 尾随空白扫描全部通过。
- Causal Stage-B Part 1 在冻结快照前收口：修复 current-only callback 可见 heldout label 与 utility callback 可见 heldout target 的两个 fail-closed 缺口，新增私有 fit-map 文件级重验及合成 E2E 测试；module/test SHA-256 为 `d2b0b42045e22099d179a4428eb2fa5d90f118839e1c0052335fef32293dd384` / `9f8bd52df10af3d4d37703f039c3172484cf0074e1c4635823c6c1cdfd204840`。
- 加入 Stage-B 后，主进程再次全仓复核为 `245 passed, 11 warnings in 44.93s`；compileall、38 JSON 解析、工作树/暂存 diff-check、尾随空白、禁用扩展名与单文件 >1 MiB 检查均通过。未读取真实 sidecar、未训练。

## 2026-08-08 — 长期目标续跑与两项独立复核

- 已恢复持久计划与此前科研状态，并建立持续目标：在无泄漏、强基线、多数据集与统计门全部满足前不宣称顶会证据完成。
- 独立重跑 Repair 3 Stage-1 定向测试：`14 passed in 10.14s`。确认 selection feature/label 在 gate 前均只做字节级 hash、不反序列化；staged isolation 单项 GO。
- 独立重跑 causal evidence 专项：`19 passed in 9.20s`。确认整簇配对 randomization、Holm raw p、accuracy no-harm 和单数据集不可发表边界；验证层单项 GO，训练/bridge 仍 NO-GO。
- 启动 Repair 3 下一轮实现：fit 内确定性 group-hash train/eval 隔离、rank-59 299D 同容量对照、accuracy 同 seed 联合门与正式 manifest hash pin。当前仍禁止真实 sidecar 性能运行。
- 记录协议恢复：fit split namespace 采用上一轮根任务最新裁决 `CARMA-Affect/Repair3/fit-internal-gate/v1`。容量 spec 的旧交接 hash 无可重算 descriptor，已在任何真实结果前重冻结为可复现 canonical SHA-256 `b8d18ce77d49f4b3bc59bd68c6b6ecf3fee136769cc234744eaa293f2a47e522`；矩阵内容 SHA-256 保持 `20eb5664adacd98324261590d360d1ad2e30306dc0d90e0b622a388f2e1b1f36`。
- 按 OpenAI 官方文档复核 GPT 路线：最新模型族可提供 `gpt-5.6-sol/terra/luna`，但本机无 API 凭据；默认 abuse-monitoring 可能保留内容最多 30 天，受限 EmotionTalk 文本在无外传授权与 ZDR/MAM 条件前继续禁止发送。GPT 保留为未来冻结文本基线/教师，不阻塞当前离线主实验。
- causal bridge 只读审计确认：现有 cache 不能无重训产生 independent-current-only，也缺少 utility OOF/selection score、load-only checkpoint manifest 与 staged contexts。已并行启动纯合成 Stage-A 合同实现；在 Stage-B selection completion 与真实训练接线完成前，causal 正式长跑仍 NO-GO。

## 2026-08-08 — Repair 3 最终修复与 causal bridge 并行推进

- Repair 3 的 fit 内隔离 namespace 固定为 `CARMA-Affect/Repair3/fit-internal-gate/v1`；已通知实现与审计任务统一该值并增加漂移测试。
- 只读初检确认 staged loader 主体已存在；真实性能运行继续阻断，等待 group-hash gate-train/gate-eval、等参数 299D control、accuracy no-harm 与同一 4/5 seed 交集全部闭环。
- causal evidence 后续任务已启动：补独立 history-stripped current-only producer/CLI 与 selected/matched-recency query 概率 bridge；当前只允许合成合同测试，不允许真实训练或封存角色读取。
- 运行资源复核：RTX 4070 8GB 当前空闲（约 7.84GB 可用），未发现 Python/pytest 训练进程；C/D/E 盘剩余约 24.84/10.56/9.84GB。资源足够做合同测试与小模型 gate，但仍不适合下载或本地训练大型 GPT 级模型。
- 确认性合同在任何封存结果产生前完成预执行修正：主参考集合加入 current/all-history/coverage-matched-recency/forward/backward；bootstrap 改为五 seed×共享整簇 crossed 设计；Holm 原始 p 值固定整簇配对 randomization；accuracy no-harm 固定为点差 ≥0、CI 下界 ≥−0.005，并要求相对 current 与冻结最强参考均通过。
- `test_confirmatory_contract.py` 新增四组 fail-closed 回归；专项 `17 passed`，配置 JSON parse 与 Python compile 均通过。
- causal runner 新增 `require_complete_checkpoint` 只读门：缺 processor/checkpoint、partial checkpoint 均拒绝，只有 identity 完全匹配的 complete checkpoint 可恢复；新增三态合同测试。冻结 torch 环境专项 `16 passed, 5 warnings`。
- causal bridge 独立只读审计进一步确认生产链必须拆为 `fit`→`complete-selection`→`evaluate`；现有双角色 loader 会过早打开 selection payload，且 producer typed view 缺 decision scores、histories 与逐 checkpoint records，因此正式长跑继续 NO-GO。
- evidence 层新增 exact-schema `build_current_only_artifact_mapping`：复制 producer 行/cluster 对齐，固定 history 消耗为 0，绑定不同 source/checkpoint identity，重算矩阵哈希与 independence attestation，并在返回前自校验；仅是合成生产 API，尚未运行真实训练。
- current-only builder 新增正向 exact-key、自校验及 history source/checkpoint 复用拒绝测试；冻结 torch 环境 evidence 专项更新为 `20 passed`。
- confirmatory+evidence+causal-runner 联合回归为 `53 passed, 5 warnings`；warning 仅为 PyTorch GradScaler 旧 API 提示，无失败或训练结果产生。
- accuracy no-harm 现有两个显式、不可删减的配置对照（vs independent current-only、vs 冻结最强 admissible baseline），并新增按 confirmatory config 文件 SHA 加载的 evidence gate；丢失任一对照、哈希漂移或放宽 0/−0.005 门均 fail closed。confirmatory+evidence 专项 `38 passed`。
- `.gitignore` 补充排除 `output/pptx` 的 inspect NDJSON 与渲染临时目录，防止科研代码提交混入本地 QA 中间件；最终可编辑/发布文件不受该规则影响。
- 分支仍为 `codex/carma-affect-research-status-20260807`，跟踪 `hanwudi` 同名远端；新 ignore 规则已验证命中 inspect、渲染目录和 Office 临时锁文件，后续提交不会误纳入这些本地中间件。
- Repair 3 实现代理报告三项 High 均已落盘，定向 `23 passed`、相关 `70 passed`；全仓一度出现 6 个 causal staged-receipt 失败，原因是公开聚合键误名为 `fit_contract.groups`。causal 代理已改成纯计数 `group_count`，未放宽隐私白名单或公开 group id，正在复跑。

## 2026-08-08 — causal evidence 验证层完成，正式长跑仍阻断

- 新增 `causal_backbone_evidence.py` 与 19 项 synthetic contract tests；独立 current-only、fit-OOF 25% operating point、逐 query matched-recency、五 seed×共享整簇 bootstrap CI、paired whole-cluster randomization、Holm、accuracy no-harm、严格公开 schema 和双数据集索引均已实现。
- Holm 原始 p 值现来自整簇配对随机化检验：每个 cluster 一次 swap，五个训练 seed 共享；Macro-F1/accuracy 每次重算。未中心化 bootstrap tail proportion 已完全移出 p 值路径。
- 公开单数据集报告固定为 `single_dataset_not_publishable`；EmotionTalk 与 MELD 缺一时 cross-dataset index 直接拒绝，双数据集齐全也只形成 open-role 索引，不授权 method success。
- 回归结果：证据专项 19 passed；证据+producer/loader/query-policy/utility/confirmatory 81 passed；全仓 217 passed；`compileall`、`git diff --check` 与新增文件尾随空白检查通过。
- 仍未实现独立 current-only 五 seed×fold 产物生成 runner/CLI，也未把策略上下文概率生产接到实际 producer checkpoints；因此真实训练长跑继续 **NO-GO**，等待代码冻结后的独立只读审计。

## 2026-08-08 — causal evidence 层进入统计修正

- evidence 模块已开始实现独立 current-only、冻结25% operating point、等基数recency、shared-cluster bootstrap、Holm与accuracy no-harm；尚未启动正式训练。
- 审计阻断了把未中心化 bootstrap tail proportion 直接作为 Holm p 值的做法；要求改用整簇配对 randomization/permutation 或验证过的 null-centered bootstrap，并明确单数据集结果不能触发顶会成功。

## 2026-08-08 — Repair 3 真实运行继续阻断

- 独立审计的 30 项纯合成测试通过，但发现 gate 前完整 loader 会打开 model-selection features/labels；这属于物理隔离失败，真实 Repair 3 未启动。
- 已要求实现 staged loader、selection payload 访问事件/attestation、完整超参锁定，并在修复后再次独立审计；只有审计 GO 才允许唯一一次真实 fit gate。
- 同时登记 299D vs 59D 的输入层容量混杂；在真实运行前需加入或冻结等参数/等信息负对照，不能把潜在增益直接等同于情感理论特征有效。
- 进一步审计发现现有 base-OOF→utility-OOF 不是 nested，held-group 标签可间接影响 meta 训练特征；已把 fit 内 group-hash gate-train/gate-eval 隔离列为真实运行前 High 修复项。
- 已在见到 Repair 3 结果前要求把 accuracy no-harm 纳入 fit/model-selection 判定，避免再次出现只改善 Macro-F1、却降低准确率仍被判成功的目标错位。

## 2026-08-08 — 严格 v2 sidecar 已生成并通过零性能核验

- EmotionTalk 新建 `D:/HVA-Affect_data/EmotionTalk/carma_open_role_sidecars_v2`，没有覆盖旧 v1；公开 manifest `results/emotiontalk_open_role_sidecar_v2_manifest.json` SHA-256 `bbd843876fa051c5426d0d56870adc939cdf71e1e8eaf552880ab4f89d47f530`。fit 9,549 行/368 groups，model-selection 2,682 行/99 groups；只生成四个开放角色文件，未创建 calibration/holdout/validation/test sidecar。
- EmotionTalk loader 无性能复核通过：12,231 行、11,300 条有历史，manifest/provenance attestation `086c1663f1041ab4c707a5e862c51d470bd176130cd751cd41e851bb9e840ab6`；`sealed_role_arrays_opened=false`、`validation_or_test_opened=false`。
- MELD 新建 `D:/HVA-Affect_data/MELD/carma_sidecars_v2`，旧 `carma_sidecars_v1` 未覆盖；公开 manifest `results/meld_multimodal_role_sidecars_v2_manifest.json` SHA-256 `7b12632066d20dc252c0d0d58ecc72e2d1ceefe015972ac4d73c1d0570826f99`。CPU non-performance preflight 通过：fit 6,606 行/679 dialogues，selection features 1,419 行/150 dialogues；selection label 只核验文件 hash、未反序列化，calibration/holdout/dev/test 均未打开。
- 最新 causal 四专项 40 passed、全仓 192 passed；Repair3 相关独立回归 40 passed 并 compileall 通过。causal producer 获独立 GO，但仍明确不是论文性能证据；独立 current-only、matched recency、冻结 operating point、seed×cluster CI 与 multiplicity 尚待完成。

## 2026-08-08 — causal-backbone 独立审计阻断项闭环

- EmotionTalk 与 MELD 正式 runner 均只接受 `sidecar_dir + manifest`；物理 role sidecars、manifest/hash/schema/alignment、fit-only speaker/OOV、RNG 完整 resume、write-once 报告与 producer-only 声明已进入代码和合同测试。
- 最后一项 High 已修：公开报告按数据集区分 split，EmotionTalk 保持 `train_corpus_open_roles_only`，MELD 明确为 `official_train_open_roles_only`；新增 MELD 端到端报告回归断言。
- MELD 文档补齐 v2 新目录生成、CPU-only preflight 与 formal producer 命令；历史 `carma_sidecars_v1` 明确禁止覆盖或用于正式 runner。preflight 哈希但不反序列化 model-selection labels，不打开 calibration/holdout/dev/test，不计算指标或效用。
- 本轮修改后的相关 runner/loader 测试为 `24 passed in 20.94s`；`compileall` 与 `git diff --check` 通过。主任务此前独立全仓回归为 `192 passed`；最终只读审计已请求。
- 该闭环只解除“可以安全做结构预检/开放角色 producer”的工程阻断，不产生模型性能数据，也不授权准确率、Macro-F1 或顶会方法成功主张。

## 2026-08-08 — repair 1/3 distributional query-level 最终 NO-GO

- 修复折叠环境漂移后，以原 producer 精确环境 Python 3.11.9 / NumPy 2.3.1 / SciPy 1.16.2 / sklearn 1.7.2 完成 attempt2；stderr=0。结果 `emotiontalk_distributional_query_policy_v3_model_selection.json` SHA-256 `16f7ffcb940f0206f7b6756bbd272fb60dd4f6a2ec77e8fbcecf9305bd54ae17`。
- current Macro-F1/NLL 为 0.531589/1.611305；all-history 为 0.541436/1.584738；coverage-matched recency 为 0.548742/1.591291 且 5/5 过门。distributional true 为 Macro-F1 0.531454、NLL 1.558586、excess -0.052719、coverage 0.588973，但 0/5 过门。
- true 相对 current Macro-F1 -0.000136，相对同 coverage recency -0.017289；说明修复只改善概率损失，没有改善主要分类指标。forward/backward 各 4/5 不能支持 true 双向创新。125-cell NLL 恒等最大误差 `6.66e-16`，18 个来源/实现 hash 无不一致，aggregate-only audit 通过。

## 2026-08-08 — 冻结情感 teacher / GPT 路线判定

- 决定仅把固定 commit 的 XLM-EMO 作为 repair3 的 fit-only gated secondary feature；四类概率不能冒充本任务七类监督，selection 不参与模型/层/映射选择。Chinese-Emotion-Small 因数据与复现证据不足被淘汰为主 teacher。
- 当前 GPT 路线 NO-GO：无 OpenAI/Azure 凭据，也无受限对话文本外传授权、DPA/保留策略与稳定 snapshot 合同。不会借用当前 Codex 会话批量标注或伪造 GPT 数字；未来只有在固定 snapshot、temperature=0、去标识 fit-only 输入、请求/响应缓存与本地等信息基线齐备后才可作为次级基线。

## 2026-08-08 — repair 2/3 class-balanced 最终 NO-GO

- 真实结果 `emotiontalk_class_balanced_utility_repair_v1_model_selection.json` 已原子写入，SHA-256 `547e54e1e6f525944eb6715e760e4a0af78abb9b4d0bea7d1940b4fdc0be1cf4`；内嵌 reproducibility manifest `776d7492ba4febecdcb91d2dc23d5f505a5d4345735728168561ef386d08d632` 独立重算一致。
- true 五乘五均值：Macro-F1 0.539997、accuracy 0.636548、NLL 1.581836、excess NLL -0.029469、coverage 0.544030。相对 current Macro +0.008407，但 accuracy -0.008700；相对 all/recency/backward 的 Macro 分别 -0.001439/-0.003460/-0.003869。
- 预注册 current gate 为 4/5，但同时胜过 current/all/coverage-matched recency 的严格诊断仅 2/5；true 对 backward 的 Macro 仅 1/5、accuracy 0/5。按规则判定 NO-GO，不调参、不重跑、不打开封存角色。
- 156 项 NLL 恒等最大误差 `6.97e-16`；stderr 25 条冻结迭代上限 warning、无 traceback/error；专项 11 passed、当时全仓 161 passed。单一 pickle 整体反序列化限制仍使 strict epistemic seal=false，因此该结果只能是开放角色探索证据。

## 2026-08-08 — distributional query-policy 续跑恢复

- 已从前一轮完整恢复：query-policy v2 的方法级结论为 NO-GO；NLL 口径已统一为 `true_class_loss` 在 `1e-12` 概率下限下的均值，旧 v2 报告保持不覆盖。
- 新的 distributional query runner 已实现七策略、5 utility seeds × 5 base seeds、一查询一预测、聚合输出与完整 provenance；首轮真实运行因 NumPy 版本下 `GroupKFold` 等长组排序差异触发 checkpoint position fail-closed，未生成结果。
- 冻结 checkpoint positions 现作为权威折叠，并增加 exact-cover、无重叠、越界、cluster-purity、逐折 canonical position SHA-256 及 runtime fold-regeneration sensitivity；模型、阈值、种子和角色均未改变。
- 新增 125 个 strategy × utility-seed × matching base-seed 单元的 NLL 恒等式 fail-closed 审计：`strategy pooled_nll - matching current-base pooled_nll == mean_excess_nll_vs_current`，绝对容差 `1e-12`，只公开最大误差等聚合字段。
- 找回原 producer 精确环境 `.venv`：Python 3.11.9、NumPy 2.3.1、SciPy 1.16.2、scikit-learn 1.7.2；专项 `8 passed`，全套 `164 passed`。下一动作是在新的私有 attempt-2 日志目录用该环境真实重跑；80–89、90–99、validation/test 继续封存。

## 2026-08-08 — 修复路线运行时合同审计

- causal backbone 独立审计发现真实运行阻断项：EmotionTalk loader 在角色过滤前载入整份 train/validation 特征，与“80–99/validation no-read”合同和公开报告表述不一致；MELD loader 也未绑定冻结 manifest 与真实文件 hash，row alignment 证明不足。因此 GPU 训练继续禁止，先把开放角色特征物理分离并强化 MELD provenance。
- 为确认 EmotionTalk 语言而格式化原始 `transcription.csv` 前 8 行时，意外连同 `emotion` 列显示；8 行均来自同一对话。操作立即停止；key-only SHA 角色审计确认该对话属于 bucket 16（fit），未触及 model-selection/calibration/holdout。标签值仍不用于任何特征、超参、模型选择或科学结论；后续文本抽样改为显式列白名单。
- distributional query-level 首次真实运行在读取 `fold_1.npz` 时因 checkpoint positions 与当前环境重算折叠不一致而 fail-closed；根因定位为原 producer 的 NumPy 2.3.1/sklearn 1.7.2 与当前 NumPy 1.26.4 对等长 group 的非稳定排序差异。没有生成结果或修改模型；修复仅把散列绑定的冻结 checkpoint positions 作为权威，并补 exact-cover/无重复/cluster-purity/59-D bitwise 验证。
- class-balanced repair 首次真实运行尚未生成结果、也未见任何 model-selection 数字时，发现公开报告缺逐 utility-seed 指标与组合 reproducibility manifest；决定中止并登记为 report-contract abort，只补报告字段、hash 与环境清单，再以完全相同配置重跑。
- causal backbone 已统一逐样本 `true_class_loss` 的 `1e-12` NLL 口径并参数化 dataset/cache 身份；误启动的空运行在首轮训练前停止，未产生训练输出或模型选择信息。

## 2026-08-08 — MELD 三模态开放角色 sidecar 完成

- 核验本机已有的 MELD MM-Align train 三模态特征：9,988 行，音频序列 32 维、视频序列 2,048 维；仅使用官方 train CSV，不打开 dev/test。
- 新增 `meld_multimodal_sidecar.py`、冻结配置、CLI 与 4 项合成合同测试；源文件哈希不符时在 unpickle 前 fail-closed，官方 CSV 标签优先于第三方 pickle 标签。
- 真实写入四个角色各自的特征/标签 sidecar；开放 fit/model-selection 分别为 6,606/1,419 行，并有 4,755/1,015 条同 dialogue、同 speaker、严格过去历史查询。
- 校准与内部 holdout sidecar 已分文件并继续封存；新模型 runner 只允许接收 fit/model-selection 路径。下一步复用参数化 causal-backbone 核心，在 MELD 开放角色重新生成 backbone-relative 双向效用。

## 2026-08-08 — 查询级均值 MLP 路线终判与 v3 统计完成

- one-query-one-prediction 真实开放角色运行完成，`stderr=0`；报告 SHA-256 `c5a81a951b492734ae72ccdc060c059929b0fc5343c00f0a82831311b8b8dc0f`，主估计量为 25 个 utility/base seed 组合。
- 真双向 query policy 的 mean excess NLL -0.02667、coverage 53.80%，但 Macro-F1 0.52723 低于 current-only 0.53159，0/5 utility seeds 通过联合门；all-history Macro-F1 0.54144，coverage-matched recency 0.54298 且 4/5 通过。
- 结论：均值 MLP 虽能降低平均 NLL，却不能提高 Macro-F1，且历史选择质量被 recency 支配；现有路线 STOP，distributional repair 1/3 已启动，class-balanced repair 2/3 开始实现。
- v3 crossed/shared-cluster 与精确共同 25% 诊断真实重跑完成，`stderr=0`，SHA-256 `a625e966b286d955a0c3c379e67639cfbd6ce260b47d422f8cb2b32918d9779a`；true 相对 forward/backward/pseudo 的 crossed CI 和 exact-coverage CI 均保持小于 0，但仅支持 surrogate utility 增量。
- distributional repair 1/3 真实任务级运行完成，`stderr=0`，SHA-256 `fa9190a9c080a5078b12ad039cca7b3b0ef8f4cc33b613754eb30084dada81c2`；25% excess 从旧 true MLP 的 +0.00691 降到 +0.00437但仍有害，10% 仅 cluster-macro 点估计略负而 row mean/selected-only 仍正。下一步接入 query-level 后再作 repair 终判。

## 2026-08-08 — 真实 sampled-context 分类门完成

- 真实开放角色 sampled-context 运行完成，`stderr=0`；聚合报告 SHA-256 `39f3fac9d02f4f9e1773181e4de697e2091ee9210b846fcfd2ba0bcb2cef14a4`，未访问 calibration、internal holdout、validation/test，也未输出行级数据。
- 真双向在 sampled task 上相对三个 utility 对照有小幅 NLL 优势，但 addition 相对 `always S` 的 Macro-F1 下降 0.00344、NLL regret +0.00456；deletion 相对更强 `always T-h` 的 Macro-F1 下降 0.00409、NLL regret +0.00546。
- sampled-context 分类/绝对安全门判为 NO-GO；保留查询级策略作为不同估计量的最终开放角色检查，随后按预登记 repair 1/3 转入 sign×severity distributional utility。
- 查询级 runner 独立审计修复 task-row/聚合 pair 阈值错配，并把主估计量升级为 5×5 utility/base seed 网格；我复跑 query+sampled+relations 相关测试为 `37 passed`，真实 query-level 开放角色运行已启动。
- runner 如实标记单 pickle 标签容器导致严格字节级封存不成立；非开放标签从未被索引或用于训练/指标，但正式确认前必须先冻结 open-role-only label sidecar。
- utility v3 crossed/shared-cluster bootstrap 与共同 25% coverage 诊断实现完成；我复跑专项为 `13 passed`，待 query 运行资源释放后生成新的 v3 报告。

## 2026-08-08 — 修正后 v2 四模型结果完成并审计

- 有效 v2 运行完成，`stderr=0`；公开 aggregate-only 报告为 `results/emotiontalk_bidirectional_utility_models_v2_model_selection.json`，SHA-256 `d4ef35eeff45e71e31a99c2c6f9953a55c2ad6d2e99ac00ee23b6a059fee442e`。
- 五独立 seed 平均 excess NLL 排名为：different-set bidirectional 0.006911、backward 0.012213、pseudo 0.013453、forward 0.013465；真正双向相对三对照均 5/5 seed 获胜。
- 10,000 次 seed→cluster 嵌套配对 bootstrap 的三组 excess-NLL 差值 CI 均严格小于 0；相对同参数量伪双向为 -0.006542，95% CI [-0.008901, -0.003995]。
- 绝对安全仍未通过：真正双向五个 seed 与 ensemble 的 excess NLL 都大于 0；因此仅继续开放角色的 sampled-context 和查询级策略诊断，不开启任何封存角色。
- 同步完成情感概率 provenance P0 修复：完整 source/split/fold/task/context/class/producer lineage、真实 59D cache 内容验证、float32 simplex 容差与负向测试；专项 10 passed、代理报告全套 101 passed。
- 独立只读复算确认排名、五 seed 胜数、excess-NLL/RMSE CI 方向、参数匹配与全部现有哈希一致；结论仍是开发 GO、绝对安全/解封 NO-GO。
- 审计识别 seed×cluster 交叉结构与 v2 独立嵌套 cluster 重抽的潜在 CI 偏窄风险，以及不同模型实际 coverage 不同的公平性风险；已启动 v3 shared-cluster crossed bootstrap、cluster-only 和共同 25% coverage 诊断实现，旧 v2 报告保持不覆盖。
- sampled-context provenance API 已完成兼容修复；我复跑 relations/classification/runner 相关测试为 `26 passed`，随后启动真实开放角色 aggregate-only sampled-context 运行。
- 对私有不同集合 cache 做 aggregate-only oracle 审计：selection 正 strict utility 率 46.16%，10%/25% oracle excess NLL 为 -0.04320/-0.05085，证明仍有较大可学习上限；把 sign×severity/distributional utility 登记为均值 MLP 失败后的独立 repair 路径。

## 2026-08-08 — v2 伪双向与层级统计运行启动

- 新增与真正双向模型同为两头、同参数量的 `pseudo_bidirectional_same_set_mlp`；两个训练目标均为 forward utility，明确模拟 `T=S∪{h}` 时反向目标退化为同一代数效用，不含独立反向集合信息。
- 新增 10,000 次配对训练种子×完整对话聚类 bootstrap，分别报告真双向相对 forward、backward 和伪双向的 policy-regret/strict-RMSE 差值置信区间、5/5 seed 胜数及阈值迁移后的覆盖率差。
- v2 报告将写入输入 cache SHA-256、实现源码 SHA-256 与源配置哈希；旧 v1 结果保留为可审计开发记录，不覆盖。
- 修正 sampled-context 分类指标定义：主 Macro-F1/accuracy 在 query 聚合后按全部 query 池化计算，cluster 作为重采样单位和次要宏平均诊断，避免把小型单类对话的 F1 当作主指标。
- 发现发布 backbone 的 audio 输入误写为 768 维；真实 EmotionTalk 冻结 WavLM 是 mean+std 1,536 维，已把配置和默认值修为 1,536，模型仍严格少于 2M 参数。
- 全套公开回归在修改后为 `89 passed`，仅两个小型合成 Adam 达到迭代上限的 warning。
- 首次 v2 启动后，独立审计发现旧 `policy_regret` 是相对逐任务 oracle 的机会损失，不是相对 fallback 的 excess NLL，且 CI 对应五次独立运行均值而原排名使用非线性 ensemble；已在任何结果文件产生前停止该运行。
- 已把安全主量改为 `cluster_macro_excess_nll_vs_fallback=-policy_utility`，oracle opportunity regret 只作次要诊断；排名和 CI 统一为五次独立运行的均值，bootstrap 改为冻结定义的 seed→cluster 嵌套配对，种子同步为 `(17,29,43,71,101)`。
- 25% 字段改为 fit-OOF **目标**覆盖率并显式报告迁移后的实际覆盖率；加入上游 OOF lineage、cache/代码/配置/环境哈希校验。修正后的真实 v2 已重新启动，日志位于私有 `bidirectional_utility_model_run_v2_corrected_estimand/`。

## 2026-08-08 — 严格 25% fit-OOF 阈值效用模型完成

- 有效的第三次效用模型运行完成，`stderr=0`；公开聚合结果写入 `results/emotiontalk_bidirectional_utility_models_v1_model_selection.json`，SHA-256 为 `cdfdbf39d08ff3374d0d04093d14abb3ab163d3a4f22d237f1bb58501b642b5c`。
- 输入严格为 58,976 个 fit OOF 任务、16,212 个 model-selection 任务、59 个无标签特征；378/94 个聚类，未读取 calibration、internal holdout、validation 或 test，未输出行级记录。
- 五种子集成排名：真正不同集合双向共享 MLP 的 model-selection 聚类宏平均 policy regret 为 0.06640，优于 backward-only 的 0.07042 和 forward-only 的 0.07206；严格双向效用 RMSE 分别为 0.35360、0.36516、0.36953。
- 双向模型在 5/5 独立种子上均同时优于两个单向模型；但其 model-selection 策略效用仍为负（聚类宏平均 −0.00689），fit 阈值迁移后的覆盖率为 27.59%，尚未证明安全或分类提升。
- 当前结果只通过“方向性模型选择信号”预检；尚缺退化伪双向参数匹配对照、配对聚类置信区间和情感分类 Macro-F1/accuracy，因此 calibration 与所有封存角色继续关闭。
- 下一动作：补齐伪双向和配对统计，并利用现有四上下文概率 checkpoint 构建不泄漏的任务级分类诊断；若仍有增量，再实现真实查询级可逆历史策略。

## 2026-08-08 — 不同集合双向 OOF 实验启动

- 重新运行全部公开合同测试，原始 41 项通过；新增 rectangular context aggregation、query-balanced subset augmentation、等基数不同成员采样、float64/label-free task feature 等测试后为 `47 passed`。
- 新增 `emotiontalk_bidirectional_oof.py` 与 CLI，真实实现 fold 内随机历史子集增强及 `P(S)`、`P(S+h_i)`、`P(T)`、`P(T-h_i)` 五 seed OOF 生成；缓存与 checkpoint 均强制 float64、哈希校验、原子写入且不含行级身份。
- 修复审稿审计发现的 split 漂移风险：新实验继续使用冻结的 `scu_set_exploration_v1` split id，不允许因为方法/协议名称变化而把已观察 group 重新分配到封存角色。
- 修复集合大小混杂：主实验强制 addition context 与 deletion-without-candidate 等基数、成员不同；退化伪双向留作后续负对照。
- 完成真实任务量预检：fit 58,976 tasks，model-selection 16,212 tasks，增强训练 64,220 rows，预计五 seed 四概率 80.3 MiB。
- 已启动完整 EmotionTalk different-set OOF 后台实验；每折完成即写私有 checkpoint，避免系统重启导致已完成折丢失。公开输出仅在全部折、完整性检查和私有 cache 成功后原子生成。
- 完整 5-fold OOF 与 model-selection 推理已经成功结束，`stderr=0`；五折分别覆盖 11,780、11,768、11,786、11,836、11,806 个 held-out tasks，随后对 16,212 个 model-selection tasks 完成五 seed 推理。
- 新的真实结果证明 forward/backward utility 不是代数重复且跨角色稳定复现；公开聚合结果已写入 `results/emotiontalk_bidirectional_oof_v1_model_selection.json`，私有 float64 cache 与 5 个 fold + selection checkpoint 均通过 schema、hash、shape、dtype、finite 和 forbidden-field 核验。
- 下一动作：在私有 cache 上运行 forward-only、backward-only、退化伪双向与共享双输出模型的五独立 seed 公平比较；只有预测和选择门通过后才构造情感领域特征与 3×3 消融。

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
