# CARMA-Affect Research Findings

## Causal Stage-B Part 1 合成合同结论（2026-08-08）

- 新增独立 current-only fit OOF producer、私有 fit protocol-row mapping 与 fit-only utility OOF score producer 的 Stage-B 合同；所有真实 selection/evaluate 接线仍未执行。
- 首轮 callback 设计审计发现 heldout label/target 可能通过请求对象进入训练回调；在任何真实 sidecar 或训练前已修复：current-only callback 只接收 train labels，utility callback 只接收 train forward/backward targets，heldout 部分仅提供无标签特征与索引。
- 私有 fit-map 现在同时绑定 preflight receipt、文件 SHA 与 fit array contract，拒绝仅在内存伪造、改名替换或 receipt 漂移的映射。
- 合成端到端测试覆盖 current-only/utility artifact happy path 与 heldout label/target 不可见边界；Stage-B module/test SHA-256 为 `d2b0b42045e22099d179a4428eb2fa5d90f118839e1c0052335fef32293dd384` / `9f8bd52df10af3d4d37703f039c3172484cf0074e1c4635823c6c1cdfd204840`。
- 该模块只证明 Part 1 的接口与 provenance 合同，不代表独立 current-only 已真实训练，也不代表 selected/recency bridge、evaluate 或双数据集联合推断已经完成。

## Repair 3 Stage-1 与统计层独立复核（2026-08-08，继续中）

- Stage-1 最终候选 source/test/config SHA-256 分别为 `56debb696ed1fd6b0cf5ba111f8a5a3a27c1cff16040a5f8142bc064debd2e6b`、`76c79908cca82e16e352ccde92a2c8908a476abc5fa3df9f93f7bf93e74567a7`、`e7deab0c42dc0c5e480f6f8a0566679f4db0872114eecb87bc37d94b7549f921`；根任务独立重跑定向测试为 `14 passed`。代码与行为测试共同证明 gate 前只反序列化 fit features/labels，selection features/labels 仅做 SHA-256，坏 payload 在同步 manifest hash 后也不会被 `np.load` 打开。该结论只把 staged physical isolation 单项改为 GO，未授权真实运行。
- 真实 fit gate 仍需消除 base-OOF→utility-OOF 非 nested 泄漏。上一轮根任务的最新冻结裁决把 fit 内分组 namespace 定为 `CARMA-Affect/Repair3/fit-internal-gate/v1`；与较早交接摘要中的建议名冲突时，以此最新裁决为准，并要求 config/code/test/report 全链一致。
- 59D→299D 等信息容量对照的 expansion matrix 已独立重算：`numeric_matrix_content_sha256`（shape/dtype canonical header、NUL、C-order bytes）为 `20eb5664adacd98324261590d360d1ad2e30306dc0d90e0b622a388f2e1b1f36`，rank=59，`max|EE^T-I|=2.22e-16`。交接中的旧 spec hash `d0cf...` 无法复原 canonical descriptor，不能硬贴；在任何真实结果可见前已重冻结可重算 canonical descriptor SHA-256 `b8d18ce77d49f4b3bc59bd68c6b6ecf3fee136769cc234744eaa293f2a47e522`，旧值仅作为 handoff-only superseded 记录。
- causal evidence 固定 source/test SHA-256 为 `36b758f0df217dc07abf80f268eae18eca8b8ceb38d727af96b4ee1fdad88666` / `1db6b3353c6706195905917371873551b8250176a740173d3dd3b78b98f18b21`；根任务独立重跑 `19 passed`。整簇 paired randomization 的 swap 在五 seed 间共享，非线性 Macro-F1/accuracy 每次重算，exact/Monte Carlo p 值口径合格；public exact whitelist 与 `single_dataset_not_publishable` 边界合格。该验证层单项 GO，但 producer bridge、独立 current-only 训练产物和真正跨数据集联合推断仍未接线，因此正式长跑继续 NO-GO。

## GPT 直接接入的当前官方边界（2026-08-08）

- 本机环境只检查变量名后确认没有 OpenAI/Azure OpenAI API 凭据；没有读取或输出任何密钥值。当前 Hugging Face cache 也没有 GPT 或 XLM-EMO 权重。因此本轮不能产生真实 GPT 实验数据。
- OpenAI 官方最新模型指南当前把 `gpt-5.6-sol` 列为旗舰，把 `gpt-5.6-terra` 列为智能/成本平衡、`gpt-5.6-luna` 列为高吞吐成本优先；三者支持 Responses、Chat Completions 与 Batch。若未来获合规授权，研究用批量文本教师优先评估 `gpt-5.6-terra`，但只能作为预冻结文本基线/辅助教师，不能替代多模态双向效用核心。
- 官方数据控制文档说明：API 数据默认不用于训练，但 abuse-monitoring logs 默认可能保留客户内容最多 30 天；ZDR/MAM 需要组织获批。Responses 默认还可能有 30 天 application-state retention，ZDR 下 `store` 强制为 false。EmotionTalk/受限对话文本在未获得数据外传授权、DPA/保留策略和 ZDR/MAM 条件前不得发送。来源：`https://developers.openai.com/api/docs/guides/latest-model.md`、`https://developers.openai.com/api/docs/models/gpt-5.6-terra.md`、`https://developers.openai.com/api/docs/guides/your-data.md`（访问 2026-08-08）。

## Causal producer→evidence bridge 的只读 gap 审计（2026-08-08）

- 现有 history producer 的 `current_only` 只是同一 history-trained backbone 的空上下文干预，不能在不重训时冒充 independent-current-only。合格产物必须新增 5 seeds × outer folds 的 history-stripped checkpoint/processor、fit OOF、selection fold-ensemble 与独立 manifest；训练和推理 histories 均物理为空且 subset dropout 为 0。
- typed producer cache 当前丢弃 utility probability/target 数组，且只存 histories hash、不存可重建 contexts；producer 也没有 utility-model fit OOF/selection decision score。evidence 的冻结 operating point 因而尚无真实 score source。
- selected/recency 概率需要 load-only 恢复 history checkpoints；现有入口在缺 checkpoint 时可能训练，正式 bridge 必须在任何反序列化前验证逐文件 manifest，并保证缺件/错 hash 时拒绝且训练调用次数为 0。
- 最小正式流程需三阶段：`fit` 只打开 fit sidecar并写 write-once receipt，selection 四 payload 只做 SHA；`complete-selection` 验证 receipt/代码/环境/配置/sidecar/checkpoint 后才打开 selection 并生成 current/selected/等基数 recency 概率；`evaluate` 最后加载 evaluation labels 并只发布 aggregate report。EmotionTalk/MELD 的现有 loader 均不能直接满足这一 staged 入口。
- 已把只读结论转化为独立 Stage-A 实现任务：只做 typed producer-view、fit receipt、checkpoint manifest、utility OOF score schema与合成 fail-closed 测试；不读取真实 sidecar或训练真实数据。

## Confirmatory contract 预运行审计缺口（2026-08-08）

- `carma_confirmatory_analysis_v1.json` 把 accuracy 仅列为 mandatory secondary，尚未把已冻结的 accuracy no-harm（点差 ≥0、95% CI 下界 ≥−0.005）写入成功门；这与 causal evidence 层和老师“情感任务有效性”要求不一致，必须在任何 calibration/holdout/正式结果前修正并加 fail-closed 测试。
- confirmatory bootstrap 仍登记为先 resample training seed、再独立 resample dataset cluster；五 seeds 实际共享同一批 clusters，旧写法会重复 v2 的共享 cluster 难度被平均掉风险。正式合同应改为 seed×shared whole-cluster crossed bootstrap，并用 paired whole-cluster randomization 生成 Holm 原始 p 值。
- 当前唯一 primary contrast 只允许从 forward/backward 中冻结参考，但开放角色结果显示 all-history 与 coverage-matched recency 更强。方法成功门必须同时不劣/胜过 current-only、all-history、coverage-matched recency 和最强单向基线，否则会用弱参考夸大结果。
- 已实现的 evidence API 固定 randomization 为 10,000 次、seed `20260829`（cluster≤16 时精确枚举），bootstrap/检验点估计强制一致；accuracy gate builder 只接受点差 0 与 CI 下界 −0.005。confirmatory 合同应直接绑定这些已测试语义，避免另造不一致统计口径。

## Repair 3 修复中间审计（2026-08-08，未授权真实运行）

- 当前实现已出现两阶段数据访问主体：fit gate 前仅物化 fit 数组，model-selection feature/label 只做字节 SHA-256；gate 通过后才进入 selection materialization。该事实仍需最终独立测试与代码哈希确认。
- 容量负对照已开始进入实现，包含 capacity-matched utility spec；最终门必须要求同一至少 4/5 utility seeds 同时胜过原 59D reference 与无新增信息的 299D capacity control，而不是分别满足两个 4/5 集合。
- fit 内隔离 namespace 由根任务冻结为 `CARMA-Affect/Repair3/fit-internal-gate/v1`；配置、实现、测试和公开 provenance 必须全链一致，真实性能运行前不得再漂移。

## Causal evidence 层统计修复结论（2026-08-08）

- CI 与检验现已严格分离：效应量及 95% CI 使用五训练 seed×共享整簇 crossed bootstrap；Holm 的原始 p 值使用 paired whole-cluster randomization，cluster swap 在五 seed 上共享且非线性分类指标逐 assignment 重算。强效应方向与 sharp-null super-uniformity 均有确定性测试。
- current-only 合同要求物理清空训练/推理 history、独立 source identity、独立 checkpoint manifest、矩阵 hash 与 independence attestation；history-trained backbone 的 empty-history endpoint 会被拒绝为基线。
- accuracy 是强制 no-harm gate：点差 ≥0 且 95% CI 下界 ≥−0.005；−0.005 只是假定的非劣界，不是“提高”证据。只有点差与 CI 下界均 >0 才可描述 accuracy 提高，Macro-F1 不能覆盖 accuracy gate 失败。
- PUBLIC_EVIDENCE_SCHEMA 改为递归 exact whitelist；任意 `opaque_ids`、无害命名下的 row hashes、五 seed 数值向量、labels/probability vectors 或本地路径都会被拒绝，公开 writer 只接受受控 builder 的固定 schema/status。
- 当前模块是验证与统计 gate，不是完整生产链：尚缺独立 current-only artifact producer/CLI，以及用实际 history checkpoint 为 selected/matched-recency 各生成一次 query 概率的桥接 runner。因此这些合同通过不等于可以启动真实长跑或形成顶会数据。
- 代码复核显示 history producer 已保存五 seed 的 fit-OOF/selection-fold-ensemble endpoint 概率、四种 utility context 概率、任务 CSR 与 checkpoint manifest；它足够训练/冻结 selector，但没有保存“冻结策略选中的 query context”概率。最小 bridge 必须从 hash-bound checkpoints 重新加载每折模型，对 selected 与 matched-recency 各做一次 one-query inference，不能用已有 endpoint 概率拼接或平均冒充。
- 独立 current-only 可复用同一训练骨架，但必须构造 histories 全空的独立 corpus identity、独立 checkpoint namespace/manifest，并关闭 history subset dropout 的有效路径；仅从 history-trained checkpoint 调用 empty context 不满足合同。
- 现有 `train_one_fold_seed` 是“训练或恢复”接口：在 checkpoint 缺失或仅部分完成时会继续训练。策略概率 bridge 必须使用新增的 complete-checkpoint-only 加载门，不能因路径错误悄然重训；否则 bridge 的模型身份、运行成本和可复现性会漂移。该门还必须在任何 processor 新建前检查，避免 inference-only 入口写入新工件。
- EmotionTalk 与 MELD 正式入口在载入各自 v2 sidecar 后共享同一 `OpenRoleCorpus`、`VerifiedCorpusProvenance`、`BackboneRunConfig` 和 `execute_crossfit_backbone` 核心。因此 current-only/bridge 应实现一个数据集无关核心与薄 CLI 分派，不复制两套训练统计逻辑；数据集差异只留在 manifest loader 与已冻结维度检查。
- 现有 runner 测试已覆盖 partial-checkpoint resume 的张量与概率等价性，可在同一合成语料上增加 complete-checkpoint-only 的三态合同：缺失工件拒绝、partial 拒绝、complete 只读恢复成功。这比只在 bridge 源码中检查路径更能证明不会静默训练。

## Causal evidence 层的统计审计（2026-08-08，修复中）

- 新 evidence 模块已把独立训练的 history-stripped current-only、fit-OOF 25% operating point、逐 query 等基数 recency、五 seed×共享整簇 bootstrap、Holm family、accuracy no-harm 与 aggregate-only 校验分离；这纠正了把 history-trained backbone 的空历史干预冒充独立 current-only 的问题。
- 当前初稿把未中心化 percentile bootstrap 中“效应落在零点不利侧的比例”直接当作 Holm 原始 p 值；该量不是自动校准的零假设检验。已要求 CI 保留 shared-cluster crossed bootstrap，而 Holm p 值改用预冻结、整簇配对、五 seed 共享 swap 的 randomization/permutation（或经过验证的 null-centered bootstrap），并为非线性 Macro-F1 重算指标。
- 单数据集 evidence 报告不能触发顶会方法成功；MELD 与 EmotionTalk 必须由 hash-bound cross-dataset aggregator 同时判定通过。若跨数据集聚合尚未实现，公开 schema 必须显式标记 `single_dataset_not_publishable`。
- accuracy 预冻结为强制 no-harm：点差至少 0，配对 95% CI 下界至少 −0.005；这只能支持“未明显伤害 accuracy”。只有点差与 CI 都大于 0 时，才可写“accuracy 提高”。

## Repair 3 独立审计的运行阻断（2026-08-08，修复中）

- 当前公开入口仍在 fit-only gate 前调用完整 `load_emotiontalk_open_role_corpus`，该 loader 会反序列化 fit/model-selection 两个角色的 features 与 labels；因此即使随后不评分 selection，也不满足物理 fit-only 开门合同，真实 Repair 3 暂为 **NO-GO**。
- 合格修复必须把加载拆成两阶段：gate 前只打开 fit feature/label；model-selection feature/label 文件只做 manifest 与字节 SHA-256 校验，不反序列化 payload；只有 gate 通过后才加载 selection。失败报告还必须明确记录 selection feature/label payload 均未打开、未反序列化，并提供有顺序的阶段 attestation。
- 配置冻结还需锁死 counterfactual sampling 的 seed/draws/candidate cap/cardinality 与 `text_sublinear_tf`，否则 runner 仍存在未登记自由度。
- 299D primary 与 59D reference 使用相同隐藏层但前者多 7,680 个输入层权重；若无等信息/等参数负对照，任何增益都不能完全归因于 3×3/VAD 特征内容。该容量混杂必须在真实 fit gate 前预先处理或明确降低主张强度。
- 当前 fit gate 还复用了全 fit 的 base OOF 特征再做 utility OOF，属于非 nested stacking：utility held-group 可能通过其它训练行的 base producer 间接受到该组标签影响。最小修复候选是在 fit 角色内预先按 group hash 固定 gate-train/gate-eval；base 与 utility/阈值只在 gate-train 拟合，gate-eval 只做推理和一次评分。若不消除该路径，fit gate 的泛化增益不可信。
- 原 fit gate 只把 Macro-F1 与 NLL 纳入判定，accuracy 仅报告；这可能重复 Repair 2 的“Macro-F1 上升但 accuracy 明显下降”。为回应老师的情感任务有效性要求，真实 Repair 3 前必须预先把 accuracy no-harm 纳入同一 4/5 seed gate；Macro-F1 仍可作为类别不均衡 ERC 的主端点，但不能把明显牺牲 accuracy 的结果宣称为整体预测改善。

## 严格开放角色数据边界与 Repair 3 实现状态（2026-08-08）

- EmotionTalk v2 现在是 fit features/labels 与 model-selection features/labels 四个物理 `allow_pickle=False` 文件；runner 不再接收上游 media/transcription/pickle 路径。公开 manifest SHA-256 为 `bbd843876fa051c5426d0d56870adc939cdf71e1e8eaf552880ab4f89d47f530`，封存角色数组未创建也未打开。
- MELD v2 以官方 train CSV 为标签源、固定 MM-Align train pickle 的真实 SHA-256，并保留官方 CSV 原始 row identity；non-performance preflight 没有反序列化 selection label。公开 manifest SHA-256 为 `7b12632066d20dc252c0d0d58ecc72e2d1ceefe015972ac4d73c1d0570826f99`。
- Repair 3 已冻结为 299D primary：59D task features + 105D 七类后验拼接 + 108D 完整 3×3 current/history relation + 27D VAD state/transition。三路内部七类 posterior 对 fit 做 whole-group OOF；VAD 只由预测 posterior 乘固定锚点获得，gold emotion 不进入推理特征。
- fit-only 开门在真实结果前固定为：full 相对同一 class-balanced true-bidirectional 59D reference 至少 4/5 utility seeds 的 Macro-F1 增益 ≥0.002，且 NLL 不恶化。失败则不得生成或评分 selection prediction；成功后 primary 仍固定，消融只能解释，不能选模型。

## Causal-backbone 审计后的证据边界（2026-08-08）

- 形式化开放角色 runner 的输入边界已收窄到 manifest 绑定的物理 role sidecars；任意原始路径、单 pickle fallback、覆盖公开报告及 calibration/holdout/dev/test 参数均不可用。
- MELD 非性能 preflight 可验证 fit 全输入、model-selection features、任务构造和未训练四上下文前向，同时只对 model-selection label 文件做字节哈希而不反序列化。它证明结构/来源合同可运行，不证明模型有效。
- `current_only` 明确是同一已训练 backbone 的空历史干预，不是独立训练的 current-only 强基线；正式性能主张仍需独立 current-only、coverage-matched recency、冻结 query policy、seed×cluster 配对 CI 与预声明多重比较校正。
- MELD 报告的 `official_train_open_roles_only` 仅表示官方 train 内 fit/model-selection 角色，不表示 official dev/test 被访问。v1 sidecars 属历史工件，正式 runner 只接受新生成且 schema/hash 绑定的 v2。

## Repair 3 的预冻结 PAD/VAD 锚点来源（2026-08-08）

- Repair 3 只把 PAD/VAD 当作**理论启发的固定设计坐标**，不当作个体真实心理状态：neutral `(0,0,0)`、happy `(.81,.51,.46)`、sad `(-.63,-.27,-.33)`、angry `(-.51,.59,.25)`、surprised `(.40,.67,-.13)`、disgusted `(-.60,.35,.11)`、fearful `(-.64,.60,-.43)`。坐标只与内部七类 group-OOF posterior 做期望；禁止使用 gold emotion，禁止在 model-selection 上选择或调整映射。
- 理论元数据已由 Crossref 的 DOI 精确入口核验：Albert Mehrabian, “Pleasure-arousal-dominance: A general framework for describing and measuring individual differences in Temperament”, *Current Psychology* 14(4), 261–292 (1996), DOI `10.1007/BF02686918`。查询端点 `https://api.crossref.org/works/10.1007%2FBF02686918`，访问日期 2026-08-08；Crossref 只核验书目信息，具体坐标在代码/配置中仍须表述为 Mehrabian-style 预注册设计锚点并保存 canonical SHA-256，不能夸称为本数据集估计真值。

## Distributional repair 1/3 最终结果（2026-08-08）

- sign×severity distributional selector 明显降低 NLL：true 五乘五均值 NLL 1.558586、相对 current excess -0.052719；但 Macro-F1 0.531454 基本等于且略低于 current 0.531589。
- 同 coverage 的 recency 达到 Macro-F1 0.548742、5/5 过门；true 低 0.017289 且 0/5 过门。结论是 distributional loss repair 改善校准/概率损失，却没有解决主要分类决策，也没有证明双向效用优于简单近期历史。
- forward/backward 各自 4/5 只能说明单向 utility 排序仍含信号，不能把两者的成功外推为 true bidirectional 成功。repair 1/3 冻结为 NO-GO，结果 SHA-256 `16f7ffcb940f0206f7b6756bbd272fb60dd4f6a2ec77e8fbcecf9305bd54ae17`。

## Class-balanced repair 2/3 最终结果（2026-08-08）

- class-balanced true policy 相对旧 true mean-MLP 把 Macro-F1 从 0.527232 提高到 0.539997（+1.2765 percentage points），并相对 current-only 提高 0.008407，同时把 mean excess NLL 降到 -0.029469；说明类不平衡确实是旧 selector 失败的一部分，而不是完全没有可学信号。
- 但结果不是方法成功：accuracy 相对 current 下降 0.008700；Macro-F1 相对 all-history、coverage-matched recency、class-balanced backward 分别低 0.001439、0.003460、0.003869。严格三参考联合门只有 2/5 utility seeds，true 对 backward 只有 1/5 Macro 胜出、0/5 accuracy 胜出。
- 因此可以写“class balancing repairs part of the classification loss mismatch”，不能写“双向边际效用优于单向/强历史基线”。该 repair 冻结为 NO-GO，结果 SHA-256 `547e54e1e6f525944eb6715e760e4a0af78abb9b4d0bea7d1940b4fdc0be1cf4`。

## 情感领域冻结文本模型候选（2026-08-08；探索，不是结果）

- EmotionTalk 文本为中文；因此英文专用 `j-hartmann/emotion-english-distilroberta-base` 不适合作为主情感模型。
- Hugging Face 只读 API 初筛发现 `tabularisai/multilingual-emotion-classification` 明确标注覆盖 `zh`、基于 XLM-R、截至查询时约 49,911 downloads；但其标签显示 synthetic-data 与 CC-BY-NC-4.0，必须先审计模型卡、类别语义、论文与许可，不能仅凭下载量采用。
- 更稳健的冻结候选是 `MilaNLProc/xlm-emo-t`：commit `a6ee7c9fad08d60204e7ae437d41d392381496f0`、非 gated、截至查询约 467,158 downloads，XLM-T/XLM-R 架构，固定输出 anger/fear/joy/sadness 四类，模型卡引用 Bianchi et al. 的 XLM-EMO（WASSA 2022），面向多语零样本/低资源情感识别。模型卡只写“19 languages”而未列清单，当前尚未确认训练/评估语言是否含中文；不能仅因 tokenizer 多语就宣称中文有效。其模型卡也未给出简洁 SPDX license，使用前必须继续核对原论文、上游 XLM-T、训练集与 Twitter 数据限制。
- `tabularisai` 候选是 11 类多标签（含 anger/contempt/disgust/fear/frustration/gratitude/joy/love/neutral/sadness/surprise），commit `a2b9b4d9640c53e84ab7abdd2a66146d3dafb10c`；优点是明确支持中文并覆盖本任务多数类别，缺点是 2026 synthetic LLM data、CC-BY-NC、域外验证不足。因此只可作为探索性 frozen teacher/feature，不应优先于 XLM-EMO，也不能用其自报 test F1 替代本项目开放角色结果。
- 中文专用候选 `Johnson8187/Chinese-Emotion-Small`：commit `2c04ce86de44d232f0fbe31413868eb31d791aea`、MIT、非 gated，基于 mDeBERTa-v3-base，8 类为平淡/关切/开心/愤怒/悲伤/疑问/惊奇/厌恶。优点是中文对话域和类别可解释；缺点是只有约 4,000 条自行标注繁体中文样本、下载量约 589、模型卡无独立 benchmark 表，且 concern/questioning 与本项目 fear 不对齐。它只能作为预注册的次级冻结 teacher，不宜单独承担核心方法证据。
- 较稳妥的模型选择协议是：在 **fit role 内**预先比较 XLM-EMO 与 Chinese-Emotion-Small 的冻结情感表示/概率是否有基本信号；不依据 model-selection 选择模型。随后只把预先胜出的一个带入 65–79 的 relation/theory repair，并与无情感模型、无 VAD、无 3×3 的等信息消融比较。
- 两个候选都约 1.13 GB（XLM-EMO 1,126,445,744 bytes；Chinese-Emotion-Small 1,135,925,877 bytes），合计约 2.26 GB。本机当前 C/D/E 余量约 26.12/10.79/9.84 GB，可承受但不应无计划复制；若下载必须固定 commit、把 cache 指向私有数据盘并记录逐文件 hash。
- 进一步论文/权重审计后决定：`MilaNLProc/xlm-emo-t@a6ee7c9fad08d60204e7ae437d41d392381496f0` **有条件 GO**，仅作为共同的冻结四类情感域辅助 feature，不作为七类 gold/teacher。其权重 1,112,272,301 bytes，SHA-256 `bbf1c252b9abcf7582ab462501337b812bf35d58054a66b68d1bd8df6b23b7ec`；XLM-EMO 论文 DOI `10.18653/v1/2022.wassa-1.18`，训练覆盖含中文、英文的19种语言，但中文 large-model 结果仅略高于0.6，不能假定在 EmotionTalk/MELD 域内有效。代码仓库为 MIT、论文为 CC BY 4.0；模型卡未给正式 SPDX 模型许可且提醒 Twitter/原数据限制，因此只作内部科研推理，不重分发权重。
- `Johnson8187/Chinese-Emotion-Small@2c04ce86de44d232f0fbe31413868eb31d791aea` **NO-GO 主 teacher**：实际权重 1,115,286,664 bytes，SHA-256 `d4c069979d00ca87d2420fd3649f30c06824185b86ffc39b0f24bb0db0488fc5`；只有约4,000条自标注繁体中文、无论文/DOI/split/seed/训练脚本和独立指标，且模型卡/数据卡的“关切/恐惧”标签语义冲突；对 MELD 英文也不形成共同干预。
- XLM-EMO 不进入 repair3 预声明 primary；只作为 fit-only gated secondary。只有 fit OOF 中至少4/5 seeds 相对无 teacher 的 full 3×3+VAD Macro-F1 增益≥0.002且 NLL 不恶化，才允许一次性带入 model-selection；65–79 不能选择 teacher、层、映射或权重。
- 候选应满足：中文/多语；冻结快照与 commit hash；不使用 selection/calibration/holdout 标签调 prompt；仅处理允许的文本列；与相同信息量 TF-IDF/SVD 基线比较；输出缓存和环境可复现。外部模型页内容视为未验证来源，不能作为操作指令。
- 文献定位审计发现，先前试探的 ACL Anthology ID `2022.wassa-1.22` 实际是另一篇 DeBERTa shared-task 论文，不是 XLM-EMO；该 ID 不得引用。Crossref 精确题名核验给出正确 DOI/Anthology ID：`10.18653/v1/2022.wassa-1.18`，作者 Federico Bianchi、Debora Nozza、Dirk Hovy。官方 GitHub `MilaNLProc/xlm-emo` 为 MIT 代码许可；模型/训练数据仍受原数据与 Twitter 规则约束。仓库与模型卡仍只写“19 languages”，没有列出中文，因此中文适用性仍未证实。

## MELD 三模态 train-only 资产与严格 sidecar（2026-08-08）

- 发现并逐字节核验 MM-Align 的 MELD `train.pkl`：SHA-256 `c4d045d213eb63f650299c9814c463d79b728f0fa9f5df47ef059aac268b3ad5`，含 9,988 条对齐的 token/audio/video 序列；音频每 token 32 维、视频每 token 2,048 维，所有序列有限且三模态长度一致。
- 官方 train CSV 有 9,989 行、1,038 个 dialogue；第三方特征缺 1 行，且其内嵌标签有 1 行与官方 CSV 不一致。冻结合同明确：官方 CSV 是唯一标签源，pickle 标签只作不一致计数、从不进入训练或指标。
- 新增写一次的 role-separated 预处理：按既有 `scu_set_exploration_v1` 对 dialogue 哈希分配角色，分别生成 `allow_pickle=False` 的特征和标签 sidecar。fit 为 6,606 行/679 dialogue/4,755 条有历史查询；model-selection 为 1,419 行/150 dialogue/1,015 条有历史查询；两者七类齐全。
- 私有 sidecar manifest SHA-256 为 `28b54da1da1aff5f32459d33a894b7aca54b8dc686342eb82ccd9908d2076d6e`；calibration/internal-holdout 标签已物理分文件，后续开放角色 runner 不再需要反序列化它们。
- 这使 MELD 从既有“文本＋手工音频 Pilot”升级为可运行文本＋音频＋视频开放角色实验的第二独立三模态数据源；尚未产生新模型性能，因此不能据此声称方法有效。

## 真实 one-query-one-prediction 与 v3 敏感性（2026-08-08）

- 查询级 v2 开放角色运行完成，`stderr=0`；2,630 个 model-selection queries 每策略每 seed 恰好一条预测，主估计量为 5 utility × 5 base seeds。报告 SHA-256 为 `c5a81a951b492734ae72ccdc060c059929b0fc5343c00f0a82831311b8b8dc0f`。
- current-only 的 Macro-F1/accuracy/NLL 为 0.53159/0.64525/1.61130；all-history 为 0.54144/0.65278/1.58474，mean excess NLL -0.02657、harm rate 32.59%。all-history 在当前开放角色上是必须认真对待的强基线，而不是默认有害的弱对照。
- 真双向选择策略 coverage 53.80%、平均选 1.10 条历史、mean excess NLL -0.02667、harm rate 30.20%，说明 query 聚合后平均 NLL 安全性相对 current-only 转为正信号；但 Macro-F1 仅 0.52723、accuracy 0.63208，分别较 current-only 下降 0.00436/0.01317，五个 utility seeds 中 0/5 满足联合门。
- 真双向还被强基线支配：all-history 在近似相同 mean excess NLL 下 Macro-F1 高 0.01420；coverage-matched recency 在相同 coverage/每 query 相同候选数下 Macro-F1 0.54298、accuracy 0.65059、harm rate 24.53%，4/5 seeds 通过，而真双向为 0/5。当前模型“选择哪条历史”的质量不如简单最近历史。
- forward/pseudo 的 mean NLL 改善略大，但 Macro-F1 仍未过门；这进一步表明普通 per-query NLL utility 与 Macro-F1 目标错位。现有均值 MLP 路线正式 **STOP**，repair 2 必须对 utility 训练做 class-balanced weighting，推理特征仍禁止 gold label。
- utility v3 重跑 SHA-256 `a625e966b286d955a0c3c379e67639cfbd6ce260b47d422f8cb2b32918d9779a`。shared-cluster crossed CI 在冻结 fit 阈值点仍全部小于 0：true vs forward [-0.01041,-0.00292]、backward [-0.00933,-0.00136]、pseudo [-0.01013,-0.00303]；精确共同 25% score-only 诊断也分别为 [-0.00977,-0.00266]、[-0.00949,-0.00270]、[-0.00941,-0.00305]。因此相对 surrogate utility 增量不是由 v2 偏窄 bootstrap 或覆盖漂移单独造成，但它仍未转化为分类成功。

## Distributional repair 1/3 任务级结果（2026-08-08）

- sign×severity 三组件真实开放角色运行完成，`stderr=0`；报告 `results/emotiontalk_distributional_utility_repair_v1_model_selection.json` 的 SHA-256 为 `fa9190a9c080a5078b12ad039cca7b3b0ef8f4cc33b613754eb30084dada81c2`。
- 真双向仍排名第一：25% fit 阈值迁移后的 cluster-macro excess NLL 0.00437，优于 pseudo 0.00967、forward 0.01071、backward 0.01267；相对三个对照的 shared-cluster crossed CI 均小于 0，且 5/5 seeds 获胜。严格 utility RMSE 0.34790，优于旧均值 MLP 的 0.35774。
- 但主 25% **绝对安全仍失败**：cluster-macro excess NLL +0.00437、row mean +0.00501、selected harm rate 46.96%。它只是把伤害缩小，未把符号翻为安全收益。
- 探索性 10% 点出现聚类/行加权分歧：cluster-macro excess 均值 -0.00092，但五 seed 最大值 +0.00026；row-mean excess +0.00056，selected-only excess +0.00505，selected harm rate 52.00%。不能根据单一聚类宏平均点估计宣称安全，也不能用探索覆盖决定确认成功。
- repair 1/3 当前判定为 **任务级部分改善、主门 NO-GO**；尚需用相同 distributional 分数运行 query-candidate 聚合的一查询一预测分类诊断。若 Macro-F1 仍不胜 recency/all-history，则该 repair 正式停止。

## 真实 sampled-context 分类诊断（2026-08-08）

- 开放角色 aggregate-only 运行完成，`stderr=0`；报告 `results/emotiontalk_sampled_context_diagnostic_v1_model_selection.json` 的 SHA-256 为 `39f3fac9d02f4f9e1773181e4de697e2091ee9210b846fcfd2ba0bcb2cef14a4`。输入为 9,817 fit 行、2,630 model-selection 行、58,976/16,212 sampled tasks 与 378/94 clusters；未物化非开放角色，标签归档未打开。
- 真双向 ensemble 的 addition 策略相对 `always S` **未通过分类/安全门**：Macro-F1 0.55343 vs 0.55687（-0.00344），accuracy 0.66168 vs 0.66216，NLL 1.44130 vs 1.43725，mean NLL regret +0.00456。
- deletion 策略相对 `always T` 有改善（Macro-F1 0.55027 vs 0.54747；NLL regret -0.04645），但相对更强的 `always T-h` 反而更差（Macro-F1 0.55027 vs 0.55436；NLL regret +0.00546）。因此不能把“优于较弱固定端点”包装成选择策略成功。
- 五 utility seeds 平均下，真双向相对 forward/backward/pseudo 的 sampled-context NLL 均略低；但 addition Macro-F1 差值分别约 -0.00008、+0.00039、+0.00045，远低于 0.002 开发门且方向不稳定。不同集合反向监督的代理 NLL 增量尚未转化为可声明的情感分类提升。
- 该诊断按设计把同 query 的多个随机上下文概率平均，不能代替 one-query-one-prediction。决策：sampled-context 门为 **NO-GO**；继续运行查询级真实集合策略仅用于判定是否存在聚合后的挽救信号，不开启 calibration/holdout/test。

## 查询级 runner 与 v3 统计审计（2026-08-08）

- 查询级阈值口径已修正：fit OOF 先按 `(query,candidate)` 对多次 coalition draw 的预测取均值，再在同一部署单位上冻结覆盖阈值；禁止把 task-row 阈值直接用于聚合分数。
- 主估计量已改为 5 utility seeds × 5 独立 base seeds 的 25 格均值；五 base 概率 ensemble 只作次要诊断。每个策略、每个 selection query 都恰好一条预测，并复核严格过去历史、角色内历史、概率 simplex 和标签范围。
- 查询级运行仍有明确封存限制：`mm_label.npz` 是单一 pickle 字典，加载开放标签会反序列化完整 train 容器，虽然代码只索引 0–79 且 80–99 使用量为零。因此当前只能称 operational use seal，`strict_epistemic_non_open_label_deserialization_seal_satisfied=false`；正式确认前必须由预先冻结的数据准备步骤生成并哈希 open-role-only label sidecar。
- utility v3 已加入 seed×共享-cluster crossed bootstrap（主要开放敏感性）、cluster 独立嵌套 legacy v2 敏感性，以及不看标签的 model-selection 精确共同 25% coverage transductive 诊断；旧 v2 结果受 schema 保护，不会被覆盖。

## 修正后 v2 四模型开发门结果（2026-08-08）

- 在 94 个开放 model-selection 对话聚类、5 个独立 utility seeds 上，真正不同集合双向模型的平均 `cluster_macro_excess_nll_vs_fallback` 为 0.006911，优于 backward-only 0.012213、伪双向同参数量对照 0.013453 和 forward-only 0.013465；5/5 seeds 均胜过三个对照。
- seed→cluster 嵌套配对 bootstrap（10,000 次）的候选减对照差值均小于 0：相对 forward 为 -0.006554，95% CI [-0.009378, -0.003840]；相对 backward 为 -0.005302，[-0.008303, -0.002520]；相对伪双向为 -0.006542，[-0.008901, -0.003995]。严格双向 utility RMSE 的三个 CI 也全部小于 0。
- 伪双向与真正双向均为 2,482 参数、双输出头；因此当前开发证据排除了“仅因多一个输出头/参数量”这一简单解释，并支持不同集合 backward target 含可学习增量。
- 绝对安全门仍失败：真正双向五个独立 seed 的 excess NLL 全为正，均值 0.006911；五种子 ensemble 为 0.005703。也就是说它虽比对照少伤害，仍比 current-only fallback 更差，不能称为安全提升。
- fit-OOF 25% 目标覆盖迁移后的 ensemble 实际覆盖为 25.83%；独立 seed 覆盖范围为 21.19%–32.88%。不同模型覆盖率不完全相同，因此这不是严格共同覆盖率比较。
- 决策：方向监督开发门为 **GO**，分类/查询级安全门为 **NO-GO/待检验**；calibration、internal holdout、validation 与 test 继续封存。下一步必须产生 one-query-one-prediction 的 Macro-F1/accuracy/NLL/Brier、excess NLL、harm rate 与 coverage，再决定是否保留当前选择器。
- 独立统计审计追加限制：五个训练 seed 共享同一批 94 个 clusters，实际是 seed×cluster 交叉结构；v2 字面 seed→cluster 嵌套 bootstrap 为每个 seed 独立重抽 cluster，可能把共享 cluster 难度误差平均掉并使 CI 偏窄。必须补共享-cluster crossed bootstrap 与 cluster-only 敏感性后才能把 CI 当成稳健推断。
- 独立审计还确认迁移后四模型 ensemble coverage 分别为 true 25.83%、forward 27.68%、backward 24.13%、pseudo 28.36%。因此 v2 只能说明“各自 fit-OOF 冻结阈值下 true 最好”；不能声称在严格共同 25% coverage 下最好。后续增加不看标签的精确共同覆盖率诊断，并在确认阶段报告 coverage-risk 曲线。
- `bootstrap_probability_difference_below_zero=1.0` 只表示本次 10,000/10,000 个重采样差值小于 0，不是数学概率 1，也不是 p 值。三个开发 contrasts 未做 multiplicity 校正，不能转写为确认性显著性结论。
- provenance 仍需加固：报告断言应验证 ranking/seed/selected/contrast 数值恒等式，复现文件缺失必须 fail closed，并用不泄露身份的 group-set digest 证明 fit 与 selection 无 group 重叠。
- 私有 cache 的 aggregate-only 可达上限复核显示，model-selection strict utility 正值率为 46.16%，但均值为 -0.05162，属于符号近均衡、负尾更重的分布；fit 的正值率为 45.15%、均值 -0.05981，跨角色形态稳定。
- 真实 utility oracle 在 model-selection 的 10%/25% 覆盖可达 mean excess NLL -0.04320/-0.05085（cluster-macro -0.05114/-0.05880），而当前 true bidirectional 为正 excess。故失败不是“无可用正历史”，而是现有 59D 均值 MLP 的排序/校准误差；下一条不同修复路径应为 sign×severity/distributional utility，而不是继续微调同一均值头。

## 第二轮模型与诊断合同（2026-08-08）

- 指标勘误：v1 的 `policy_regret` 实际定义为 `max(u,0)-I(select)u`，应称为 **oracle opportunity regret**；它天然非负，不能证明相对 fallback 安全。相对 fallback 的真实 excess NLL 是 `-I(select)u=-policy_utility`。两模型的配对差值因 oracle 常数抵消而数值相同，但绝对安全解释完全不同。
- 开发排名现已改为五个独立 utility seed 的平均 excess NLL；非线性五种子 ensemble 只作诊断，从而使点估计与 seed→cluster bootstrap CI 对应同一 estimand。
- 为排除“双输出容量”而非“独立反向集合信息”造成增益，新增同参数量伪双向负对照：两个头都学习 forward 目标；真正双向仍分别学习 `u+` 与来自不同集合的 `u−`。
- 模型选择推断升级为训练种子×对话聚类的配对层级 bootstrap；该统计只用于开放 model-selection 预检，不替代后续冻结确认分析。
- 新增严格因果、任意历史子集掩码的多模态 Transformer 骨架；EmotionTalk 实际输入合同为 text SVD 256、WavLM mean+std 1,536、DINOv2 768，参数量仍低于 2M。
- 新增无金标情感输入的 3×3 情感概率关系合同和 sampled-context 分类诊断。前者仍需训练折内的文本/音频/视频七类情感概率生成器，当前只有特征构造与对齐合同，尚无真实数值结果。
- sampled-context 诊断可快速检验 add/fallback 与 retain/delete 是否改善 Macro-F1/accuracy/NLL，但多个随机上下文会在 query 内平均，因此只属于中间证据；最终主张仍要求一个真实查询级可逆历史集合。

## 不同集合双向效用的首个真实模型选择结果（2026-08-08）

- 在 25% fit-OOF **目标覆盖率**协议下，五种子集成的 model-selection 聚类宏平均 oracle opportunity regret 排名为：different-set bidirectional 0.06640、backward-only 0.07042、forward-only 0.07206；双向相对最强单向的点估计改善为 0.00402（5.71%）。这是机会损失诊断，不是绝对安全指标。
- 严格双向目标的聚类宏平均 RMSE 同样排序为 0.35360、0.36516、0.36953；五个独立种子均给出双向优于两个单向的 regret 点估计，说明反向不同集合监督具有可复现的预测增量。
- 该增量尚不是最终安全收益：双向策略选择的真实严格效用仍为负（聚类宏平均 −0.00689），oracle-regret 仍为正，且 fit 阈值迁移到 model-selection 后实际覆盖率漂移到 27.59%。
- 现有比较缺少预先计划的“退化伪双向”参数匹配对照，也没有配对聚类 bootstrap；因此只能登记为开发信号，不能称为 H1 最终通过。
- 分类意义仍未建立。下一诊断必须用不读取封存角色的四上下文概率，比较同一任务上的 add/fallback 与 remove/retain 决策对 Macro-F1、accuracy、NLL 和 harm rate 的影响；最终仍需查询级可逆历史集合策略。

## 持续科研审计与不同集合 OOF 启动（2026-08-08）

- 顶会证据审计判定：当前只能支持“历史负迁移问题/benchmark 值得研究”，教师提出的三项方法创新仍为 `0/3` 实证通过；旧 endpoint 的 harm AUC 为 0.728，但 mean utility Spearman 约为 -0.002，不能作为新方法成功证据。
- 发现并在真实实验前修复两个关键混杂：数据角色不再随新模型协议名变化，统一复用冻结的 `scu_set_exploration_v1` split id；不同集合任务强制 `|S|=|T\{h_i}|` 且成员不同，避免把集合大小效应误当成双向不对称。
- 新增可断点恢复的 EmotionTalk different-set runner：每个训练折内先做 query-balanced history-subset augmentation，再生成 5 seed float64 `P(S)`、`P(S+h_i)`、`P(T)`、`P(T-h_i)` OOF 概率；calibration、internal holdout、validation 与 test 均保持封存。
- 真实预检得到：12,447 个可物化 train-role utterances；9,817 fit 行、2,630 model-selection 行；在等基数不同成员约束下生成 58,976 个 fit OOF 任务与 16,212 个 model-selection 任务；基础模型增强训练行约 64,220，五 seed 四概率约 80.3 MiB。
- 完整 OOF 已成功结束且 `stderr=0`。fit OOF 的 forward/backward Spearman=0.6453、sign agreement=0.8107、mean |asymmetry|=0.1479 nats；独立 model-selection 角色分别为 0.6720、0.8226、0.1317 nats，99.99% 以上任务具有非零不对称。该结果支持“不同集合双向监督含新增信息”，但尚不支持“新方法提高分类或安全”。
- 五个 base seed 的 utility target 稳定：fit forward/backward pairwise Spearman 中位数约 0.878/0.877，model-selection 约 0.903/0.902；每 seed 与 ensemble 的符号一致率约 93%–95%。因此观察到的不对称不太可能只是单一 base seed 噪声。
- 私有 cache 为 32,985,539 bytes（SHA-256 `d754d9d3af36fddc9c3f935983ecd77c921163c1d33f5615e46871bed47901c9`），包含 58,976×59 fit 特征与 16,212×59 selection 特征、双向 targets 和 5-seed targets；所有连续数组均为 finite float64，不含 key、query/candidate index、speaker、gold label 或行级 group id。
- GPT 路线可行但当前无 API key：最合理角色为固定快照文本基线、结构化情感/VAD teacher、冻结 embedding 或蒸馏来源；不能替代双向效用主方法，也不能补齐音频/视频证据。论文主实验若使用 GPT，应优先固定日期快照并在许可/ZDR/预算门后执行。
- 数据硬门：EmotionTalk 当前只有冻结 WavLM/DINOv2 派生特征、无本地原始音视频；MELD 当前没有完整视觉证据；IEMOCAP 尚无获许可数据。因此 3×3 六流跨数据集确认仍需要第二个完整三模态对话数据集。

## 教师三点整合后的新方法合同（2026-08-08）

- 三个单点均不能独立安全主张为首创：情感理论/领域模型已有成熟工作，六流与模态两两交互高度拥挤，加入/删除归因也有邻近方法。
- 潜在创新被收窄为完整闭环：不同集合 forward-addition/backward-deletion、train-only OOF 反事实监督、情感状态约束、3×3 当前—历史关系、校准可逆选择与 current-only 回退。
- 最关键的数学约束是 `T != S union {h_i}`；否则 `u_plus` 与 `u_minus` 代数相同。该约束已进入数据类验证和合成测试，而不是只写在文档中。
- 新增六流共享投影后的 3×3 关系接口，每格输出 cosine、L2、signed mean delta，共 27 个无标签特征；不同模态原始维度未对齐时拒绝计算。
- 新增 VAD 状态、VAD 变化和情感转移概率接口；gold emotion 明确禁止作为特征。
- 新冻结协议的真实性能尚未产生。当前缺少完整 float64 `P(S)`、`P(S+h_i)`、`P(T)`、`P(T-h_i)` OOF 缓存，因此不能声称方法改善了准确率或安全性。
- 新流程图的 PPTX、PNG、PDF 均完成视觉检查；PPTX overflow 测试通过。公开合同测试更新为 `41 passed`。

## 独立 MELD 完整重跑终判

- 本轮从冻结 train/dev CSV 与 35 维 handcrafted audio NPZ 独立启动完整入口，未传入 test。
- 新输出与初始输出、既有 repeat 在 JSON 上字节完全一致；per-query CSV 解压内容也字节完全一致。
- 这排除了当前轻量 MELD 路径上的随机性/环境漂移疑问，但不提升其方法结论：音频增量和严格安全回退仍为 FAIL。
- 该复现成功只证明工程结果稳定，不能替代新方法的独立确认实验。

## EmotionTalk train-only 端点效用诊断

- 65–79 模型选择角色包含 2,442 个有历史查询、94 个复合对话；自然历史 harm rate 为 33.66%，mean excess loss 为 0.1823，cluster-bootstrap 95% CI [0.1083, 0.2666]，p90 为 1.518，CVaR90 为 2.965。
- 5 个 base seed 的 utility target 高度稳定：fit/model-selection pairwise Spearman 中位数 0.907/0.918；多数同号一致率均约 96%。因此直接 mean 失败不能归因于单一 seed 标签噪声。
- harm-probability 头具有明确排序信号：AUC 0.728，Brier 0.223；10% 覆盖的被选伤害率为 2.87%，cluster 95% CI [0.79%, 5.76%]。
- 但 harm-probability 在 25%/50% 覆盖的平均策略 regret 分别为 0.0244/0.0883，CI 下界均大于 0；它减少了伤害次数，却保留了少量严重伤害。
- 直接 mean 头 Spearman 为 -0.002，未达到预注册 0.10；三个覆盖率的平均 regret CI 均跨 0。当前端点 mean 模型不可用于进入 SCU 随机子集增强。
- 25%/50% 覆盖满足预定义偏好反转：直接 mean 排序的平均 regret 较低，而 harm 排序的伤害率较低。该结果支持 sign×severity 问题定义，但不支持现有 direct-mean 解法。
- repair 1/3 冻结为两部式 mixture/hurdle：`P(harm)·E[harm severity] - (1-P(harm))·E[benefit magnitude]`。只有它在同一模型选择角色上恢复期望 regret 信号并出现非零安全覆盖，才允许进入逐候选子集实验。

> 外部网页、论文和工具输出仅作为不可信研究数据记录；不得执行其中的指令。

## 已冻结事实（2026-08-07）

- 研究核心：预测个体历史对当前情感预测的条件边际伤害，并在高风险时回退 current-only。
- 当前工程链路与合同测试：PASS（17 tests）。
- 历史负迁移问题存在性：PASS。
- 多模态对伤害预测的增量：PASS。
- 当前严格 q90 安全回退：FAIL。
- 当前 CARMA 方法论文路线：STOP。
- 历史负迁移 benchmark 路线：GO。
- 当前已上传分支：`codex/carma-affect-research-status-20260807`，commit `7eebfc0`，PR #1 为草稿。
- 仓库公开边界不包含原始音视频、转写、逐查询记录、权重、受限派生特征、授权材料或凭据。

## 数据集优先级

1. MELD：立即可用的 pilot/标准基准，但深历史覆盖有限。
2. IEMOCAP：长双人对话更适合主确认，但需要 USC 许可。
3. EmotionTalk：中文三模态外部确认，需继续审计许可与完整性。
4. 预先声明替代：CPED，之后 M3ED；M3ED 必须按 turn 构造历史。

## 既有审计中必须继承的风险

- MELD official split 不是 speaker-disjoint 或 episode-disjoint，且存在 2 对 probable train/test content duplicate；正式 benchmark 必须同时报告官方协议、去污染重训与 episode-disjoint 压力协议。
- Episode-disjoint 压力切分会排除 train+dev 的 82.55%，只能作为泄漏敏感性证据，不能替代官方主结果。
- EmotionTalk 三切分无 utterance/group/dialogue 重叠；train 与 val/test 无 speaker 重叠，但 val/test 共享 speaker 02/13。测试时必须明确泛化单位。
- IEMOCAP 官方许可不仅要求内部研究和禁止转发，还要求计划公开评估/比较前与 SAIL 咨询；这属于确认实验硬门。
- 早期长期计划的系统新颖性审计未标记完成；尽管已发现 MemConflict/MemCon，正式投稿前仍需可复现的最新邻近工作矩阵。

## GPT 初步判断

- 不应直接用 GPT 取代整个多模态模型：它不能自动解决严格时间安全、个体历史归因和校准回退。
- 最合理的试验角色是：文本零样本强基线；冻结文本 embedding；在训练折内生成软标签的教师；最终仍需非 GPT 的可复现安全选择器。
- GPT 结果如果依赖闭源模型快照和提示词，需要版本、成本、缓存与数据传输合规记录。

## 模型候选决策（proposal，不是发现）

- 主候选为 SCU-Set：用 train-only OOF 随机历史子集构造逐候选、逐集合的模型相对边际效用监督，再用小型 set-conditioned utility model 预测 q10/q50/q90 与伤害概率。
- RCPS/coverage-constrained risk control 作为安全层与强基线，不能单独包装成核心创新。
- Latent regime/change-point 只作为状态突变/恢复压力测试候选；演员短对话不足以支撑真实心理状态主张。
- GPT Memory Critic 当前延期：无 API key、可复现性与隐私风险较高。若未来运行，必须与小型多语 embedding 等信息基线比较。
- SCU-Set 的效用必须表述为“相对于冻结 base 的条件决策效用”，不得冒充历史的真实心理因果效应。

## 待验证

- 本机实际具备哪些数据、特征、GPU 和依赖。
- 冻结 EmotionTalk 结果能否从当前仓库入口完整复现。
- q90 失败源于信号不足、目标定义、校准、分布移位还是样本量。
- GPT/LLM 是否提供超过传统文本编码器的稳定增量。

## 本机环境与资产

- GPU：NVIDIA GeForce RTX 4070 Laptop，8,188 MiB；足够冻结编码器、小型神经 base 与参数高效微调，不适合本地训练大型 GPT 级模型。
- 系统 Python 3.11.9；全局仅有 numpy/pandas/pytest，缺 scipy、scikit-learn、torch、transformers、openai、pingouin、statsmodels。
- 既有研究目录 `HVA-Affect_科研全流程_2026-08-04/experiment/.venv` 已包含 SciPy 等依赖，应优先复用并核验，避免污染全局环境。
- C 盘剩余约 28.1 GB；必须控制模型缓存、特征和 checkpoints，不能盲目下载大型模型/数据。
- 未检测到 OPENAI/Azure OpenAI API key，也未检测到 Hugging Face token；当前不能声称运行 GPT 实验，后续先做离线接口与开源/缓存基线。
- 完整工作目录比公开仓库多出 MELD 配置、音频合同、运行脚本、外部基线代码以及 EmotionTalk 元数据/实验资产；需精确审计哪些原始数据与派生特征真实可用。
- 旧实验虚拟环境可直接运行：numpy 2.3.1、pandas 3.0.3、SciPy 1.16.2、scikit-learn 1.7.2、PyTorch 2.11.0+cu128、Transformers 4.57.6、pytest 8.4.2。
- 旧实验 `artifacts/` 中存在约 155 MiB、142 MiB、101 MiB、51 MiB 等 EmotionTalk 特征/模型资产，以及 MELD 聚合结果；说明复现无需重新下载全部编码器，但仍需验证散列和 schema。
- 外部 `conv-emotion` 基线仓库含大量预提取 MELD/IEMOCAP 特征；其许可、数据来源、人物/时间切分与可公开性尚未审计，不能直接当作确认数据使用。
- D盘仅余约 11.66GB，E盘约 10.57GB；`D:/HVA-Affect_data` 已占约 19.84GB。大型 7B+ 本地语言模型不适合当前磁盘/8GB显存预算。
- Hugging Face 缓存已有固定 WavLM-base-plus 与 DINOv2-small，可复用；没有文本 embedding/LLM 缓存。若做非 GPT 文本增强，应优先选择小型多语 embedding 模型并记录下载散列。
- EmotionTalk 可用资产确认：官方 `transcription.csv` 存在，冻结三模态特征 `emotiontalk_media_features_v1.npz` 存在（162,682,156 bytes）；未在科研任务目录发现 `Audio.tar` 或 `Multimodal.tar` 原始归档。
- MELD 官方 train/dev/test 标注 CSV 均存在；test 文件存在不等于允许启封，当前仍按协议把 test 视为封存确认集。
- 精确 artifacts 清单首次尝试因 Windows PowerShell 所用 .NET 不提供 `System.IO.Path.GetRelativePath` 而失败；后续用字符串前缀裁剪或 Python 只读枚举。

## 复现资产状态

- 旧实验工程合同测试实际为 32 项，已于本轮重新运行并全部通过（3.90s）；公开仓库只携带其中 17 项。
- EmotionTalk 冻结复现资产完整存在：特征 NPZ、train-only summary、148.8MB joblib bundle、freeze manifest、validation 聚合 JSON 与 per-query 压缩表。
- MELD 复现资产完整存在：文本 feasibility、cross-fit selector、negative controls、音频文本 risk、per-query 表及一份 repeat 复现结果。
- 既有 `reproducibility_audit.json` 与 repeat 输出表明此前做过一次 MELD 音频文本复现；本轮还需独立核对其散列/数值一致性，而不能仅信任文件名。
- MELD 原始与 repeat 聚合 JSON 字节级相同，SHA-256 均为 `ccc6b1937e7d68eda9033c646152e472d1c294cc76c3e60a6a88cdae94f51943`；1108 行 per-query 解压后逐值完全相同、最大数值差 0。gzip 文件字节散列不同仅反映容器元数据，不能误判为结果不复现。
- EmotionTalk freeze manifest 与当前 config、feature NPZ、bundle 的 SHA-256 全部一致；冻结 validation 结果散列为 `bd2dba738a2de326f165859b0e4dd101c623a31c635b309dfbca1de4fa1c9a01`。
- EmotionTalk freeze manifest 明确 `test_policy=sealed`，validation 仅授权一次运行；现有结果可审计但不能继续用其调阈值后称为确认。
- MELD 音频文本脚本要求显式传入 train/dev 官方 CSV 与两份 derived audio NPZ；preflight 记录数据位于 `D:/HVA-Affect_data/MELD/derived/`。
- MELD train 交集 9,988、dev 交集 1,108，各缺 1 个官方 key；音频为 35 维 handcrafted PCM，不是 WavLM。parquet 内 gold label 从未读取，测试 split 未打开。
- 该轻量音频实验只能支撑“真实音频工程可行/轻量音频无明显增量”，不能用于否定更强语音表征。

## Per-query 资产可用于的探索范围

- EmotionTalk validation per-query 表：1,908 行、35 列；包含 dialogue/speaker/turn、history_count、current/full loss、受限置换损失、四种 selector 输出与媒体质量字段。约 1,770 行有历史（由冻结报告给出）。
- MELD dev per-query 表：1,108 行、15 列；包含 dialogue/utterance/speaker、history_count、current/full loss 及文本/音频 selector 输出。
- 两表均含可识别对话或说话人键，严格禁止上传；公开仓库只保留聚合统计。
- 两表没有完整 base 类别概率和逐条历史候选特征，因此可用于 benchmark/统计再分析与策略诊断，但不足以独立训练新的逐候选方法；新方法需从 train/cross-fit 中间产物或原始特征重新生成监督。

## EmotionTalk 统一 benchmark 首轮诊断

- 首轮 aggregate 输出的点估计再次显示明显重尾：有历史查询 1,770，伤害率 33.90%，median excess 为负但 mean 为正，p90 约 1.10 nats、CVaR90 约 2.10 nats；oracle 可在 66.10% 覆盖下达到平均 regret -0.238。
- 三模态 mean-risk 排序在固定覆盖诊断中出现明显潜力，而 strict q90 仍只覆盖 0.51% 且被放行样本平均 excess 为正。这提示“均值排序信号存在、tail/calibration 安全头失败”，但因为来源是已查看的 validation，只能作为探索线索。
- 发现本轮新 benchmark adapter 的 cluster 错误：EmotionTalk 的 `dialogue` 不是全局唯一，必须用 `group/dialogue` 复合键。首轮输出把 71 个对话错误合并为 48 个 cluster，因此所有 cluster bootstrap CI 与 cluster concentration 暂时作废；点估计不受此错误影响。

## 修复后双数据集探索结果

- 修复后 EmotionTalk 为 71 个 cluster，MELD 为 103 个 dialogue；两份 aggregate JSON 均通过与私有 per-query 的行数、均值、伤害率精确交叉核验，并通过无行级/cluster 标识符输出检查。
- EmotionTalk：伤害率 33.90%，mean excess 0.0564（dialogue-bootstrap 95% CI -0.0213, 0.1400），p90 1.098、CVaR90 2.101；正 regret 的 33.97% 集中在最差 10% 对话，显示重尾和聚类风险。
- MELD：伤害率 43.27%，mean excess 0.0842（95% CI -0.0594, 0.2225），p90 1.623、CVaR90 3.162；正 regret 的 41.49% 集中在最差 10% 对话。
- 两数据集均出现重要的**符号—严重度错配**探索现象：harm-probability 排序能降低被选样本的伤害发生率，却可能使平均策略 regret 为正；mean-utility 排序能使平均 regret 为负，却仍保留较高伤害率。
- 例：MELD 文本 harm-probability 在 10% 覆盖的 mean policy regret 为 +0.0415（cluster 95% CI 0.0143, 0.0709），而 mean-risk 排序为 -0.0700（95% CI -0.0981, -0.0361），但 mean-risk 被选样本伤害率仍为 51.3%。
- EmotionTalk 三模态 mean-risk 在 10/25/50% 覆盖的 mean policy regret 均为负且 cluster CI 上界小于 0；同一模型的 harm-probability 在 25/50% 覆盖反而为正。这说明“预测是否受伤”和“控制期望/尾部损失”不是同一目标。
- 以上均来自已查看的 validation/dev，属于**探索性诊断而非确认性新发现**。下一步必须在 train-only 角色切分与封存 internal holdout 上预注册验证符号—严重度错配。

## 新的可证伪机制假设

- H1：历史负迁移是零膨胀/混合重尾结果；仅优化 `P(excess>0)` 会忽略少量大伤害的严重度。
- H2：仅优化 conditional mean 会降低平均 regret，但不能保证 harm incidence 或 q90/CVaR。
- H3：逐候选条件分布模型（sign × severity/quantiles）配合 coverage-constrained risk control，才可能同时控制平均 regret、伤害率与尾部风险。

## 代码可扩展性

- EmotionTalk 核心实现 756 行，已模块化为 processor、history blocks、base features、selector features、cross-fit train、freeze/verify、restricted permutation、cluster bootstrap 与 validate_once。
- 现有 `base_features` 接受整体 history 或 donor replacement；尚无逐候选 history tensor/set 接口，但可在不破坏现有冻结路径的前提下新增模块。
- 冻结 joblib bundle 仅保存 processors、base models、selectors 和散列/行数；不保存 OOF probability、selector features 或逐候选监督。因此新模型必须重新运行 train-only cross-fitting，并输出专门的开发工件。
- MELD 风险代码同样模块化，适合先建立统一 benchmark adapter，再实现更复杂方法，避免在两个数据集复制统计逻辑。
- 公开实验依赖已固定 numpy/pandas/SciPy/scikit-learn/pytest/matplotlib 版本，足以实现统一 benchmark 与层级 bootstrap，无需新增重量依赖。
- 现有测试采用合成数组验证合同，适合新增“聚类 bootstrap 不拆分对话、策略未使用样本 regret 必为 0、oracle 上界、无标识符输出”等 benchmark 单元测试。

## SCU-Set 实现审计

- 现有 EmotionTalk train-only cross-fit 每折只训练一个 OOF base（seed0+fold），而最终 base 才训练 5 seeds；现有效用监督可能包含单模型噪声。SCU 探索必须比较单 seed 与多 seed OOF 目标稳定性。
- 现有 base 只在“空历史”和“全历史”两种端点上训练；直接拿它评估任意历史子集属于分布外插值。SCU 必须在 fit fold 内加入随机 history-subset augmentation 后，才能生成可信的子集边际效用。
- `build_blocks` 已支持任意 custom history index sets，因此无需改动冻结原路径；可新建独立 SCU 模块生成 subset blocks。
- EmotionTalk train cross-fit 的 group 已正确使用 `group_dialogue` 复合键；SCU 的角色切分和 bootstrap 必须复用相同复合单位。
- 候选 utility 模型不宜直接吞 25k 维稀疏 TF-IDF。最小可行输入应使用 base 概率变化、候选/集合多模态几何、recency/count 与质量差，小型 MLP 即可在 8GB GPU 内训练。

## 仓库初始审计

- 当前仓库根目录包含 `assets/`、`docs/`、`experiment/`、`results/`，没有额外 `AGENTS.md` 约束。
- 当前目录是 Git worktree（`.git` 为文件），后续必须保留现有分支与用户改动。
- planning-with-files session catchup 未报告未同步上下文。
- 仓库共有 43 个已发布研究文件；核心可执行面集中在 `experiment/scripts/`、`experiment/src/hva_affect/` 和 5 组合同测试。
- 已提供 EmotionTalk 四份冻结配置、媒体特征提取、编码器基准、三模态外部验证和结果作图脚本；MELD 仅有模块与文档，没有独立公开运行脚本。
- 公开结果明确把 EmotionTalk validation 视为一次性冻结评估，EmotionTalk test 与 MELD test 尚未启封。
- Windows PowerShell 默认解码导致中文 README 首次读取出现乱码；文件本身未证明损坏，后续强制 UTF-8 读取。

## 当前实现合同

- EmotionTalk 主配置已冻结：5 个 base seeds、5 个 risk seeds、5-fold dialogue GroupKFold、20% 独立 calibration、q90 conformal 风险上界。
- 主模型是冻结 WavLM/DINOv2 + 字符 TF-IDF + 线性逻辑分类器；selector 是 HistGradientBoosting 的均值、q90 和伤害分类三头组合。
- 主风险目标为 `Delta L = NLL(full-history) - NLL(current-only)`；正值表示历史伤害。
- 严格 q90 gate 预注册要求：历史覆盖率至少 10%、策略平均 excess regret 的聚类 95% CI 上界不大于 0、被使用历史的伤害率至少下降 5%。当前冻结结果未过门。
- 当前实现只判断聚合历史整体是否有害，尚缺逐候选效用、集合非加性交互、逐步集合构造与可逆重用；这正是方法重构的核心缺口。
- test split 在代码与配置中 fail-closed；EmotionTalk validation 已消耗，不得继续以其结果调参后声称确认性结论。
