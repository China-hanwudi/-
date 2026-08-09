# CARMA-Affect N2 双数据集 model-selection 最终评估

日期：2026-08-09

阶段：冻结的非确认性 model-selection 评估

结论状态：**N2 正向方法门未通过；后续数据角色保持封存**

## 一、结论摘要

冻结的 N2 `carma_bidirectional_full` 在 MELD 和 EmotionTalk 上都没有同时胜过预先选择的最强可接受历史基线 `all_history`，也没有满足安全性、稳定性和双数据集功效的合取条件。MELD 七项性能门全部失败；EmotionTalk 七项中仅“相对 current-only 的 mean-regret 95% CI 上界不大于 0”通过。两个数据集的 `model_selection_gate_passed` 因而均为 `false`，双数据集 joint freeze 也为 `false`。[证据：E1–E3]

这不是“VAD/情感约束无效”或“3×3 模态—历史关系无效”的证明。MELD 的 H3 给出一个只限 model-selection 的正向机制信号；EmotionTalk 的 H3/H4 未拒绝零假设，但其 prospective power 仅为 0.2201，不能把非显著结果解释为零效应或组件无效。[证据：E1、E2]

当前结果只支持一个受限结论：**在这两个已经观察过的 model-selection 角色、冻结实现、冻结阈值和冻结统计合同下，N2 未能超过最强历史基线并通过预注册门。** calibration outcome、internal holdout、validation、external test 以及确认性方法成功声明均未获授权。[证据：E3、E4]

## 二、公开证据与边界

本评估只读取以下公开聚合文件和仓库内研究记录；未读取任何 private artifact、逐样本标签、预测、概率、cluster 向量或 seed 向量。

| 证据 ID | 文件 | SHA-256 / 用途 |
|---|---|---|
| E1 | [`results/carma_n2_meld_model_selection_reference_freeze_v1.json`](../results/carma_n2_meld_model_selection_reference_freeze_v1.json) | `7ef638b9491be96ec2ea1345173228eeb582c11def341eb70cbcce6b77657a40`；MELD 公开聚合结果 |
| E2 | [`results/carma_n2_emotiontalk_model_selection_reference_freeze_v1.json`](../results/carma_n2_emotiontalk_model_selection_reference_freeze_v1.json) | `27b4bad50e477a6f08e002eec6aed0eacb4a1d8d57b65a69d5142764fe08bda2`；EmotionTalk 公开聚合结果 |
| E3 | [`results/carma_n2_joint_model_selection_freeze_v1.json`](../results/carma_n2_joint_model_selection_freeze_v1.json) | `47ec55d4b1a60be5b574812fd0c82713685efc07083734ab86c1e890421d36a5`；双数据集 joint freeze |
| E4 | [`findings.md`](../findings.md) | evaluator、fail-closed、正式 verifier 与解释边界记录 |
| E5 | [`task_plan.md`](../task_plan.md) | 冻结流程、已完成任务及后续研究约束 |
| E6 | [`progress.md`](../progress.md) | 运行进度与审计轨迹 |

E1/E2 明确把自身限定为“model-selection-only reference freeze and prospective sensitivity”，不是确认性性能证据。两份报告均为 aggregate-only，且公开合同标记 `confirmatory_claim_authorized=false` 与 `calibration_holdout_or_test_access_authorized=false`。[证据：E1、E2]

## 三、评估合同

- 主候选：`carma_bidirectional_full`；每个数据集均使用 5 个训练 seed。MELD 包含 1,419 个 query、150 个 cluster，其中 1,015 个有历史 query、128 个有历史 cluster；EmotionTalk 包含 2,682 个 query、99 个 cluster，其中 2,484 个有历史 query、99 个有历史 cluster。[证据：E1、E2]
- 冻结参考选择规则为：先按五 seed 平均 model-selection Macro-F1，继而按 accuracy、mean regret 和模型 ID 排序。两个数据集最终都冻结 `all_history` 为最强参考。[证据：E1、E2]
- 95% CI 使用五 seed × 共享整 cluster 抽样的 crossed bootstrap（10,000 次，seed `20260808`）；假设检验使用五 seed 共享 swap 的配对整 cluster randomization（10,000 次，seed `20260829`）。H1–H5 采用 Holm–Bonferroni，familywise alpha 为 0.05。[证据：E1、E2]
- prospective sensitivity 针对事先冻结的绝对 Macro-F1 效应 0.005；它不是使用已观察效应计算的 post-hoc power。[证据：E1、E2]

## 四、主要点估计

下表均为五 seed 平均。mean regret 越低越好；history-harm rate 只对使用历史的策略有解释意义。[证据：E1、E2]

| 数据集 | 方法 | Macro-F1 | Accuracy | Mean regret | History-harm rate |
|---|---|---:|---:|---:|---:|
| MELD | N2 full | 0.313179 | 0.531642 | -0.000641 | 0.482167 |
| MELD | current-only | 0.306209 | 0.528964 | 0.000000 | — |
| MELD | all-history | 0.315850 | 0.533897 | -0.009094 | 0.483350 |
| EmotionTalk | N2 full | 0.531113 | 0.667487 | -0.055051 | 0.435749 |
| EmotionTalk | current-only | 0.505978 | 0.651529 | 0.000000 | — |
| EmotionTalk | all-history | 0.546120 | 0.671588 | -0.074579 | 0.426973 |

最关键的强基线对比是 full − all-history。MELD 的 Macro-F1 差为 -0.002671，95% CI [-0.007904, 0.001694]；mean-regret 差为 +0.008453，95% CI [0.002946, 0.014540]。EmotionTalk 的 Macro-F1 差为 -0.015007，95% CI [-0.026216, -0.003708]；mean-regret 差为 +0.019528，95% CI [0.010478, 0.028949]。因此 EmotionTalk 在两项主量上都被 `all_history` 明确支配；MELD 的 Macro-F1 区间跨零，但 regret 相对 `all_history` 明确更差。[证据：E1、E2]

## 五、七项合取性能门

七项门彼此独立计算，并按合取规则判断；任一失败都会阻止性能门通过。[证据：E1、E2、E4]

| 门 | 冻结判据 | MELD | EmotionTalk |
|---|---|---|---|
| G1 Macro-F1 点增益 | full − frozen reference ≥ 0.005 | **失败**：-0.002671 | **失败**：-0.015007 |
| G2 Macro-F1 区间 | 上述差值 95% CI 下界 > 0 | **失败**：[-0.007904, 0.001694] | **失败**：[-0.026216, -0.003708] |
| G3 regret vs current | full − current 的 95% CI 上界 ≤ 0 | **失败**：-0.000641，CI [-0.011799, 0.010083] | **通过**：-0.055051，CI [-0.074846, -0.037432] |
| G4 regret vs frozen reference | full − reference 的 95% CI 上界 ≤ 0 | **失败**：+0.008453，CI [0.002946, 0.014540] | **失败**：+0.019528，CI [0.010478, 0.028949] |
| G5 history-harm 相对降幅 | 相对 frozen reference 至少降低 5% | **失败**：降低 0.2446% | **失败**：降低 -2.0554%，即 harm 更高 |
| G6 accuracy no-harm 合取 | 对 current 和 frozen reference 均要求点差 ≥ 0 且 CI 下界 ≥ -0.005 | **失败**：vs current +0.002678，CI [-0.005878, 0.011215]；vs reference -0.002255，CI [-0.007363, 0.002365] | **失败**：vs current 通过（+0.015958，CI [0.005684, 0.026949]）；vs reference 失败（-0.004101，CI [-0.010755, 0.002931]） |
| G7 同 seed 联合成功 | 5 个 seed 中至少 4 个同时满足 Macro-F1 优于 reference 且 regret vs current ≤ 0 | **失败**：0/5 | **失败**：0/5 |

汇总：MELD 为 0/7，EmotionTalk 为 1/7。accuracy no-harm 是安全门，不等同于“accuracy 提高”的证据。[证据：E1、E2]

完整 `model_selection_gate_passed` 还要求 H1 与 H2 在 Holm 校正后同时拒绝。MELD 的 H1/H2 均未拒绝；EmotionTalk 只有 H2 拒绝。因此即使不考虑其他失败门，两个数据集仍都不能通过完整 model-selection gate。[证据：E1–E4]

## 六、H1–H5 Holm family

差值方向均为 full − 对照；H1、H3、H4、H5 的备择方向为大于 0，H2 的备择方向为小于 0。CI 为 percentile bootstrap 区间；p 值来自配对整 cluster randomization，`p_Holm` 为 Holm 校正值。[证据：E1、E2]

### MELD

| 假设与对照 | 点差 | 95% CI | raw p | p_Holm | 拒绝 |
|---|---:|---:|---:|---:|---|
| H1：Macro-F1 vs all-history | -0.002671 | [-0.007904, 0.001694] | 0.952005 | 0.952005 | 否 |
| H2：mean regret vs current-only | -0.000641 | [-0.011799, 0.010083] | 0.429857 | 0.859714 | 否 |
| H3：Macro-F1 vs no-VAD | +0.011421 | [-0.005391, 0.029637] | 0.002300 | 0.011499 | **是** |
| H4：Macro-F1 vs no-history-3×3 | +0.001870 | [-0.006571, 0.011561] | 0.193381 | 0.580142 | 否 |
| H5：Macro-F1 vs current-only | +0.006970 | [-0.005277, 0.020999] | 0.028997 | 0.115988 | 否 |

MELD 只有 H3 在 Holm family 中被拒绝。其 randomization p 值支持冻结 selection 上的正向情感约束增量诊断，但 percentile CI 跨零，而且整体七门、H1 与 H2 均未通过；因此 H3 不能被包装成确认性方法成功，也不能抵消强基线对比失败。[证据：E1]

### EmotionTalk

| 假设与对照 | 点差 | 95% CI | raw p | p_Holm | 拒绝 |
|---|---:|---:|---:|---:|---|
| H1：Macro-F1 vs all-history | -0.015007 | [-0.026216, -0.003708] | 0.999400 | 0.999400 | 否 |
| H2：mean regret vs current-only | -0.055051 | [-0.074846, -0.037432] | 0.000100 | 0.000500 | **是** |
| H3：Macro-F1 vs no-VAD | +0.002069 | [-0.016752, 0.022306] | 0.371963 | 0.743926 | 否 |
| H4：Macro-F1 vs no-history-3×3 | +0.001492 | [-0.006555, 0.010121] | 0.234977 | 0.704930 | 否 |
| H5：Macro-F1 vs current-only | +0.025135 | [0.009372, 0.042893] | 0.000100 | 0.000500 | **是** |

EmotionTalk 的 H2/H5 表明 full 相对 current-only 有明确改善，但 H1 显示它没有胜过更强的 `all_history`。H3/H4 仅为“未获支持”，不是等效性检验，也不是组件零效应或组件无效的证明。[证据：E2]

## 七、prospective power 与 joint freeze

| 数据集 | 预设效应 | Prospective power | 最低要求 | Power gate | Model-selection gate |
|---|---:|---:|---:|---|---|
| MELD | 0.005 Macro-F1 | 0.8127 | 0.80 | 通过 | 失败 |
| EmotionTalk | 0.005 Macro-F1 | 0.2201 | 0.80 | 失败 | 失败 |

MELD 的功效通过只说明该设计对预设 0.005 效应具有目标灵敏度，不能补救其性能门失败。EmotionTalk 功效不足，因此尤其不能用 H3/H4 的非显著结果作否定性机制结论。[证据：E1、E2]

joint predicate 要求两个数据集的 model-selection gate 和 prospective power gate 全部通过。最终 `joint_model_selection_freeze_passed=false`，精确失败原因为：[证据：E3]

1. `EmotionTalk:prospective_power_below_0.80`
2. `EmotionTalk:upstream_model_selection_gate_failed`
3. `MELD:upstream_model_selection_gate_failed`

## 八、独立审计状态

- MELD evaluator、EmotionTalk attempt 2 和 joint freeze 的正式 verifier 均通过；joint run 与 verifier 均 exit 0。EmotionTalk attempt 1 因验签过程生成 `__pycache__`，在首次项目包 import、参数解析和标签访问前 fail-closed，未生成 public/private 结果；清理审计生成的 bytecode 后，attempt 2 在相同冻结合同下完成。[证据：E4]
- 本文编写时另做了只读公开层一致性审计：重新计算三份 public JSON 的 SHA-256；验证 JSON 可解析、required dataset roster 为 MELD/EmotionTalk、两者冻结参考均为 `all_history`；用公开方法均值复算 H1–H5 点差、两项 accuracy 点差、相对 reference regret 与 history-harm 相对降幅；再核对七门布尔值、0/5 同 seed 结果、prospective power 及 joint failure reasons。公开层检查一致，未访问 private artifact。
- 该审计只能证明公开报告内部一致、散列与冻结记录一致；它不把 model-selection 结果升级为 calibration、holdout、test 或外部泛化证据。

## 九、阶段授权

由于 joint predicate 失败，以下项目均为 **未授权 / 保持封存**：[证据：E3、E4]

- separate calibration-stage workflow；
- calibration outcome 访问；
- internal holdout 解封；
- validation 解封或计分；
- external test 解封；
- confirmatory method-success 声明。

已经观察过的 MELD/EmotionTalk model-selection 结果不得回流用于修改 N2 后在同一 selection 上重新声称确认性检验。任何后续正向方法主张都需要新的 protocol ID、全新且合法的数据角色/数据集、重新冻结的分析合同；否则应转入强基线支配、历史负迁移和风险尾部的 benchmark/否证路线。[证据：E3–E5]

## 十、严谨表述边界

可以表述：

- “冻结 N2 在 MELD 与 EmotionTalk 的 model-selection 角色上均未通过预注册联合门。”
- “EmotionTalk 上 N2 相对 current-only 有改善，但被更强的 all-history 基线支配。”
- “MELD 的 H3 提供 model-selection 机制诊断信号，但不足以支持整体方法成功。”
- “EmotionTalk 的 H3/H4 未获支持，且设计对 0.005 效应功效不足。”

不得表述：

- “VAD/情感理论组件已被证明无效”或“3×3 关系已被证明无效”；
- “CARMA-Affect 已提高两个数据集的准确率”或“已超过最强基线”；
- “N2 已通过确认性验证”或“具有 test/外部泛化效果”；
- “非显著即无效”“功效通过即性能通过”，或把五个训练 seed 当作五个独立受试样本。

最终判定：**N2 当前不是可用于顶会正向方法论文的成功证据；它是一份经过冻结、可审计的负结果与强基线诊断。** 是否形成投稿价值，取决于后续能否在新数据角色上预注册复现正向结果，或把现有失败系统化为历史负迁移、强基线支配与 fail-closed 评估协议的 benchmark 贡献。
