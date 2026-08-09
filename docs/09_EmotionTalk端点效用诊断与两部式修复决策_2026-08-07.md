# EmotionTalk 端点效用诊断与两部式修复决策

日期：2026-08-07

协议：`emotiontalk_scu_endpoint_diagnostic_v1`
性质：train-only 模型选择探索；不是最终确认实验

## 1. 为什么先做端点诊断

完整 SCU-Set 需要大量随机历史子集和逐候选训练。在投入该成本前，必须先回答：空历史与全历史之间的效用目标是否稳定、伤害符号和平均严重度是否都可预测。若端点目标都不成立，逐候选模型只会放大噪声。

## 2. 严格角色隔离

| 角色 | 复合对话组 | 行数 | 本轮用途 |
|---|---:|---:|---|
| base/utility fit（0–64） | 379 | 9,817 | 5-fold group cross-fit、risk 头拟合 |
| model selection（65–79） | 94 | 2,630 | 独立诊断；其中 2,442 行有历史 |
| calibration（80–89） | 59 | 1,572 | 未用于训练或指标 |
| internal holdout（90–99） | 55 | 1,394 | 未用于训练或指标 |

EmotionTalk train 标签存储在一个整体字典容器中。实现为键集合审计解包该容器，但只把 fit/model-selection 键对应的值转成训练/评估数组；校准和 holdout 的性能没有被计算。Validation 和 test 均未读取。

## 3. 目标稳定性：通过

| 统计 | fit | model selection |
|---|---:|---:|
| 有历史查询 | 9,061 | 2,442 |
| seed 两两效用排序相关中位数 | 0.907 | 0.918 |
| 最小两两相关 | 0.900 | 0.895 |
| 多数同号平均一致率 | 95.96% | 96.13% |
| 五 seed 全部同号比例 | 86.09% | 86.00% |

因此 endpoint utility 不是由某一个随机 seed 偶然制造的标签。后续模型失败应优先解释为“可学习目标形式不对”或“特征不足”，而不是简单增加 seed。

## 4. 独立模型选择结果

自然使用全历史时：

- harm rate：33.66%；
- mean excess loss：0.1823，dialogue-bootstrap 95% CI [0.1083, 0.2666]；
- p90：1.518；CVaR90：2.965；
- 最差 10% 对话贡献 32.17% 的正 regret。

两个风险目标表现完全不同：

| 目标 | 排序指标 | 结果 | 门槛 |
|---|---|---:|---:|
| 直接 mean utility | Spearman | -0.002 | ≥0.10，失败 |
| harm probability | AUC | 0.728 | ≥0.55，通过 |
| harm probability | Brier | 0.223 | 诊断值 |

在 10% 覆盖，harm 头把被选伤害率降到 2.87%（95% CI [0.79%, 5.76%]），但平均策略 regret 的 CI 仍跨 0。在 25% 和 50% 覆盖，它的伤害率仍显著低于直接 mean 排序，但平均 regret 分别为 0.0244 和 0.0883，且 CI 下界大于 0。

## 5. 这意味着什么

伤害分类器学到了“多数情况下会不会受伤”，却没有识别少数高严重度事件。直接均值回归器也未学到稳定排序。于是出现预定义偏好反转：

- harm 目标减少伤害次数；
- direct mean 排序在 25%/50% 覆盖的平均 regret 相对更低；
- 两者选择集合的 Jaccard 仅为 0.162/0.353。

这支持“符号与严重度需要分开建模”的问题判断，但当前联合门仍为 FAIL，所以不能直接进入随机历史子集 SCU。

## 6. 冻结的 repair 1/3

下一候选为两部式 mixture/hurdle：

\[
\widehat{E[r\mid x]}
=P(r>0\mid x)\widehat{E[r\mid r>0,x]}
-(1-P(r>0\mid x))\widehat{E[-r\mid r<0,x]}.
\]

三个模块分别预测伤害概率、伤害幅度和收益幅度。幅度使用 `log1p` 目标，并按 fit 角色 99.5% 分位裁剪预测，减少重尾外推。进入随机子集增强必须同时满足：

1. mixture expected-regret Spearman ≥ 0.10；
2. 相对 direct mean Spearman 增量 ≥ 0.05；
3. harm AUC ≥ 0.55；
4. 10%/25%/50% 至少一个覆盖率的 cluster-bootstrap mean-regret CI 上界 ≤ 0。

该修复仍只使用 model-selection 角色选型。80–89 calibration 与 90–99 internal holdout 会继续封存，直到方法和阈值冻结。
