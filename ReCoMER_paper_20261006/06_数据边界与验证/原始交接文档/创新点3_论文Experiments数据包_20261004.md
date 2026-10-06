# 创新点三论文 Experiments 数据包与补实验清单

日期：2026-10-04

## 一、本文档用途

本文档面向论文 Experiments 部分的写作，整理创新点三目前可以使用的实验数据，并明确还缺哪些数据。

创新点三的论文主张应限定为：

> 模型在历史候选与空候选之间进行句级自动竞争；当历史证据可能有害时，模型可以选择空候选，并在该决策下同时切断历史反馈残差和原始历史 token 路径。

目前的实验证据支持“机制已经实现，并在部分数据集上表现出正向信号”，但还不支持“所有数据集普遍提升”。因此写作时必须同时呈现 IEMOCAP/MOSEI 的正向结果、MELD 的基本持平和 M3ED 的下降边界。

---

## 二、论文 Experiments 建议结构

建议把创新点三放在 Experiments 中的一个独立小节，结构如下：

### 4.x Experimental setup

说明：

- 任务：多模态对话情感识别；
- 输入：当前文本、音频、视觉表示及最多 K 个历史槽位；
- 核心机制：真实历史候选与空候选共同竞争；
- B1/E5/E6/E7 的定义；
- 数据集划分和指标；
- 3 个随机种子；
- test 在结构冻结前不使用。

### 4.x Main comparison

报告 B1、E5、E6、E7 在 MELD、M3ED、IEMOCAP、MOSEI 上的结果。

这一小节只回答：新弃权设计是否相对原始空候选机制表现更好或更稳。

### 4.x Ablation of the abstention design

报告：

- B0：无历史；
- B1：原始空候选；
- E5：utility + softmax + base features；
- E6：加入 semantic features；
- E7：sparsemax；
- U0/U1：是否使用 train-only utility cache。

这一小节回答：性能变化来自历史本身、空候选机制、utility 校准、语义特征，还是归一化方式。

### 4.x Mechanism diagnostics

报告：

- 实际硬弃权率；
- 连续空候选概率；
- 历史保留率；
- 历史残差范数；
- 历史 token mask 比例；
- 同标签/异标签历史分组；
- 有历史/无历史分组；
- 情绪延续/翻转分组。

这一小节证明模型确实在“选择是否使用历史”，而不是只改变了一个软权重。

### 4.x Sensitivity and failure boundary

至少报告：

- null-bias β；
- hard threshold；
- softmax temperature；
- utility loss weight；
- 历史槽位数量；
- 不同数据集上的 gate 分布。

M3ED 当前下降必须在这里作为边界案例保留，不能只放在补充材料里而在正文写成全面有效。

---

## 三、术语统一表

| 论文中统一使用 | 含义 | 不建议混用的写法 |
|---|---|---|
| historical candidates | 历史句候选 | history tokens、past samples |
| empty candidate | 空候选 | null class、reject class（除非首次解释） |
| abstention | 弃权 | refusal、history rejection |
| hard abstention | 硬弃权 | hard reject，除非定义一致 |
| keep probability | 历史保留概率 | history confidence |
| hard keep rate | 硬决策历史保留率 | keep ratio（首次需定义） |
| utility calibrator | 历史效用校准器 | utility gate |
| residual path | 历史反馈残差路径 | feedback shortcut |
| raw history token path | 原始历史 token 路径 | joint history branch |
| B1 | 原始空候选对照 | legacy baseline |
| E5 | utility + softmax + base features | main model（在冻结前不要直接叫 final model） |
| E6 | utility + softmax + semantic features | semantic model |
| E7 | utility + sparsemax | sparse model |

数据集和指标统一写法：

- MELD：Weighted-F1，↑；
- M3ED：Weighted-F1，↑；
- IEMOCAP：Weighted-F1，↑；
- MOSEI：MAE，↓。

---

## 四、当前可以直接使用的主结果

### 表 1：Innovation 3 的跨数据集对照结果

| Dataset | Metric | B1 | E5 | E6 | E7 |
|---|---|---:|---:|---:|---:|
| MELD | inner-dev Weighted-F1 ↑ | 0.5967 ± 0.0069 | 0.5970 ± 0.0090 | 0.5968 ± 0.0088 | 0.5957 ± 0.0091 |
| M3ED | valid Weighted-F1 ↑ | 0.561040 ± 0.0012 | 0.558256 ± 0.0008 | 0.558256 ± 0.0008 | 0.558650 ± 0.0023 |
| IEMOCAP | valid Weighted-F1 ↑ | 0.724835 ± 0.0022 | 0.730898 ± 0.0065 | 0.730898 ± 0.0065 | 0.730378 ± 0.0058 |
| MOSEI | valid MAE ↓ | 0.563965 ± 0.0026 | 0.562650 ± 0.0028 | 0.562651 ± 0.0028 | 0.563299 ± 0.0028 |

表注建议：

> Values are mean ± population standard deviation over three random seeds. MELD uses a dialogue-disjoint fit/inner-dev split; M3ED, IEMOCAP and MOSEI use their fixed train/valid partitions. Test data were not used during model selection or mechanism analysis.

### 表 2：相对 B1 的绝对变化

| Dataset | E5 − B1 | E6 − B1 | E7 − B1 | 方向 |
|---|---:|---:|---:|---|
| MELD | +0.0003 | +0.0001 | −0.0010 | Weighted-F1 基本持平 |
| M3ED | −0.002784 | −0.002784 | −0.002390 | Weighted-F1 下降 |
| IEMOCAP | +0.006063 | +0.006063 | +0.005543 | Weighted-F1 提升 |
| MOSEI | −0.001315 | −0.001314 | −0.000666 | MAE 改善 |

注意：MOSEI 是 MAE，数值越低越好；其负变化表示改善，不能和 Weighted-F1 的正变化直接混用。

---

## 五、当前结果在正文中可以怎么解释

### 5.1 可写入正文的核心观察

目前最稳妥的结果叙述是：

1. E5 在 IEMOCAP 和 MOSEI 上相对 B1 表现更好；
2. MELD 上 E5 与 B1 基本持平；
3. M3ED 上 E5/E6/E7 均低于 B1；
4. E6 与 E5 几乎一致，当前没有证据证明 semantic features 带来独立增益；
5. E7 没有显示出相对 E5 的稳定优势。

### 5.2 不能写入正文的强结论

不要写：

- “The proposed abstention mechanism consistently improves all datasets.”
- “E5 is universally superior to the original design.”
- “Sparsemax provides a better abstention decision.”
- “The model learns the correct causal utility of history.”
- “The abstention rate proves that the model rejects harmful history.”

这些说法要么被 M3ED 结果反驳，要么缺少 utility-cache、分组指标或因果对照支持。

### 5.3 建议使用的英文结果段落骨架

可按照以下逻辑写，不要在没有补数据前直接填成最终稿：

> We first compared the original empty-candidate design (B1) with three corrected variants that retain the same historical candidates while modifying the empty-candidate decision. E5 uses utility-calibrated logits with softmax normalization, E6 adds semantic history features, and E7 replaces softmax with sparsemax. Across the three-seed evaluation, E5 improved over B1 on IEMOCAP and MOSEI, remained close to B1 on MELD, but underperformed B1 on M3ED. Thus, the current results support a dataset-dependent effect rather than a universal improvement. The near-identical E5 and E6 scores also indicate that the additional semantic features have not yet demonstrated an independent gain.

中文含义：

> 我们首先比较原始空候选设计与三个修正版。修正版都保留相同的历史候选，只改变空候选决策。E5 在 IEMOCAP 和 MOSEI 上相对 B1 改善，在 MELD 上接近 B1，在 M3ED 上低于 B1。因此目前证据支持的是数据集相关的效果，而不是普遍提升。E5 与 E6 几乎相同，也说明额外语义特征尚未体现独立收益。

---

## 六、当前可以放进论文的机制诊断数据

### 表 3：E5 seed=7 的弃权诊断

| Dataset | empty_mean | empty_any_rate | keep_prob | hard_keep |
|---|---:|---:|---:|---:|
| M3ED | 0.0576 | 0.0610 | 0.9982 | 0.9447 |
| IEMOCAP | 0.1318 | 0.1112 | 0.9636 | 0.8888 |
| MOSEI | 0.5275 | 0.5265 | 0.9638 | 0.4842 |

需要在论文中定义：

- empty_mean：空候选的平均连续概率；
- empty_any_rate：出现空候选倾向的样本比例；
- keep_prob：连续历史保留概率；
- hard_keep：最终二值决策的历史保留比例；
- hard_abstain_rate：1 − hard_keep。

目前缺少 MELD 的同口径诊断表，建议补出后再在正文中做完整跨数据集比较。

### 表 4：历史质量统计

| Dataset | Samples with history | Mean history slots | Samples with same-label history | Slot-level match |
|---|---:|---:|---:|---:|
| MELD | 0.8972 | 2.3968 | 0.6121 | 0.3676 |
| M3ED | 0.9447 | 2.5168 | 0.7538 | 0.5845 |
| IEMOCAP | 0.8888 | 1.8401 | 0.8485 | 0.7743 |
| MOSEI | 0.4842 | 1.1737 | 0.2759 | 0.1362 |

这张表适合放在机制诊断或补充材料。正文只需保留一句：

> The datasets differed substantially in historical evidence quality, with IEMOCAP providing more label-consistent history and MOSEI containing fewer and less frequently matching historical slots.

这句话仍需配套表格或图。

---

## 七、还必须补什么

以下项目按论文写作优先级排列。

### P0：没有这些就不建议写成完整主实验结论

#### 1. B0 无历史对照

目前主表只有 B1/E5/E6/E7，缺少统一协议下的 B0。

需要补：

| Dataset | B0 metric | B1 metric | E5 metric |
|---|---:|---:|---:|
| MELD | [待补] | 0.5967 ± 0.0069 | 0.5970 ± 0.0090 |
| M3ED | [待补] | 0.561040 ± 0.0012 | 0.558256 ± 0.0008 |
| IEMOCAP | [待补] | 0.724835 ± 0.0022 | 0.730898 ± 0.0065 |
| MOSEI | [待补] | 0.563965 ± 0.0026 | 0.562650 ± 0.0028 |

B0 用来回答“历史本身是否有用”，B1 用来回答“原始空候选是否优于单纯历史机制”，E5 用来回答“校准后的弃权是否进一步改善”。

#### 2. 完整硬弃权率表

需要为 MELD、M3ED、IEMOCAP、MOSEI 输出同一 seed、同一 checkpoint 口径下的：

- empty_mean；
- empty_any_rate；
- hard_keep；
- hard_abstain_rate；
- history token mask ratio；
- history residual norm。

当前只有 E5 seed=7 的部分数据，不足以支撑完整机制图。

#### 3. 3 seed 的诊断均值和离散程度

不能只报 seed=7 的 gate 行为。至少补：

| Dataset | Variant | hard_abstain_rate mean ± std | keep_prob mean ± std | residual norm |
|---|---|---:|---:|---:|
| MELD | E5 | [待补] | [待补] | [待补] |
| M3ED | E5 | [待补] | [待补] | [待补] |
| IEMOCAP | E5 | [待补] | [待补] | [待补] |
| MOSEI | E5 | [待补] | [待补] | [待补] |

### P1：没有这些就无法充分解释机制

#### 4. 分组性能

至少按以下组报告 E5 与 B1：

- 有历史 vs 无历史；
- 有同标签历史 vs 无同标签历史；
- 历史槽位数 0、1、2、3；
- 情绪延续 vs 情绪翻转；
- hard keep vs hard abstain。

核心表格格式：

| Dataset | Subgroup | B1 | E5 | E5 − B1 | n |
|---|---:|---:|---:|---:|---:|
| MELD | same-label history | [待补] | [待补] | [待补] | [待补] |
| MELD | different-label history | [待补] | [待补] | [待补] | [待补] |
| M3ED | same-label history | [待补] | [待补] | [待补] | [待补] |
| IEMOCAP | same-label history | [待补] | [待补] | [待补] | [待补] |
| MOSEI | no matching history | [待补] | [待补] | [待补] | [待补] |

这组数据决定模型是不是“该拒时拒绝”，而不是仅仅改变平均分数。

#### 5. history_override 机制对照

需要在固定 checkpoint 上报告：

- normal：模型自行选择；
- override=0：强制弃权；
- override=1：强制保留。

每个数据集至少给出主指标和历史残差范数。若 override=0 后扰动历史输入不会改变输出，才能作为硬断路证据。

### P2：论文完整性和稳健性需要

#### 6. utility cache 对照

当前多数据集主结果主要是端到端任务损失训练，不要写成“已由 utility teacher 监督”。

需要补：

- U0：E5，不使用 utility cache；
- U1：E5，使用 train-only utility cache；
- 两者使用相同 seed、划分、训练预算；
- 分别报告主任务指标和 gate 诊断。

#### 7. 参数敏感性

至少选择一个主数据集和一个辅助数据集，扫描：

- null-bias β：0.5、1.0、2.0；
- hard threshold：0.3、0.5、0.7；
- temperature τ：0.5、1.0、2.0；
- utility loss weight：0、0.1、0.3、0.5。

每次只改变一个参数。正文只放趋势图，完整表放补充材料。

#### 8. 统计检验和置信区间

当前只有 3 seed 的均值 ± 标准差。建议正式版本至少补：

- 逐 seed 数值；
- 95% confidence interval；
- paired seed difference；
- 若进行显著性检验，说明检验对象是 seed-level paired difference，而不是把样本级预测当作独立重复。

在 3 seed 下，显著性检验的解释应保守，不能用一个 p 值掩盖较大的 seed 方差。

---

## 八、建议制作的图表

### 主文建议保留

#### Figure X：创新点三结构图

展示：

    historical candidates + empty candidate
        → automatic competition
        → keep / abstain
        → residual path and token path

重点在图中明确画出：空候选获胜时两条历史路径都被切断。

#### Table X：跨数据集主结果

使用表 1，报告 B1/E5/E6/E7。指标方向写在列名中。

#### Figure Y：历史质量与弃权行为

推荐做四个数据集的分组柱状图或点图：

- historical quality；
- hard abstain rate；
- E5 − B1；
- 以数据集为横轴统一比较。

这张图可以直观看出 MOSEI 的低历史质量与更高弃权倾向，以及 M3ED 的“历史质量较好但性能下降”。

### 补充材料建议放置

- 完整 3 seed 逐 seed 表；
- B0 对照；
- history_override 对照；
- 参数敏感性；
- 逐样本 gate 分布；
- M3ED 失败案例；
- residual norm 和 token mask 的完整统计；
- utility cache U0/U1。

---

## 九、论文写作时的证据分配

| 内容 | 主文 | 补充材料 | 目前状态 |
|---|---|---|---|
| B1/E5/E6/E7 主结果 | 必须 | 完整逐 seed | 已有 3 seed 汇总 |
| B0 无历史 | 必须 | 完整日志 | 缺统一矩阵 |
| 硬弃权率 | 必须简报 | 完整诊断 | 只部分完成 |
| 历史质量 | 简要 | 完整表 | 已有 |
| 同标签/异标签分组 | 至少一张图或一句关键结果 | 完整表 | 缺 |
| utility cache | 若作为方法主张则必须 | U0/U1 完整表 | 缺全数据集正式矩阵 |
| 参数敏感性 | 趋势图 | 完整扫描 | 缺 |
| M3ED 失败边界 | 必须提及 | 详细分析 | 缺逐样本分析 |
| test 最终结果 | 结构冻结后 | 完整结果 | 尚未进行 |

---

## 十、当前阶段可以先写的 Experiments 文字

### 10.1 实验设置段

可以先写：

> We evaluated the sentence-level history abstention mechanism on MELD, M3ED, IEMOCAP and MOSEI. For each sample, the model retained the original historical candidates and appended an explicit empty candidate. B1 denotes the original empty-candidate competition, whereas E5 calibrates the empty-candidate logit using utility-related features before softmax normalization. E6 additionally uses semantic history features, and E7 replaces softmax with sparsemax. MELD was evaluated with a dialogue-disjoint fit/inner-dev split; the other datasets used their fixed train/valid partitions. All reported values are means and population standard deviations over three random seeds, and test data were not used for model selection.

### 10.2 主结果段

> The corrected variants showed dataset-dependent behavior. E5 improved over B1 on IEMOCAP by 0.006063 Weighted-F1 and reduced MOSEI MAE by 0.001315, while remaining close to B1 on MELD (+0.0003 Weighted-F1). In contrast, E5 decreased M3ED Weighted-F1 by 0.002784. E6 produced nearly identical results to E5, providing no independent evidence for an additional semantic-feature gain, whereas E7 did not consistently outperform the softmax variant. These results indicate that the abstention mechanism can be beneficial when historical evidence is unreliable, but its effect is not universal across datasets.

### 10.3 机制诊断段

暂时不要写成最终稿，直到补齐 MELD 和三 seed 的硬弃权率。可以先保留结构：

> To verify that the gain was associated with history selection rather than a soft rescaling effect, we measured the continuous empty-candidate probability, the hard keep rate and the history-path statistics. E5 retained most historical evidence on M3ED and IEMOCAP, whereas it made substantially more hard abstention decisions on MOSEI, which contained fewer and less frequently matching historical slots. The complete comparison requires the same diagnostic statistics for all seeds and datasets.

---

## 十一、创新点三目前的论文结论边界

### 可以写成的结论

- The mechanism preserves automatic competition between historical candidates and an explicit empty candidate.
- Hard abstention removes both the historical residual path and the raw historical-token path.
- The corrected model shows positive signals on IEMOCAP and MOSEI.
- The effect is dataset-dependent.
- M3ED reveals a failure boundary that motivates further utility supervision and gate calibration.

### 暂时不能写成的结论

- The mechanism universally improves multimodal conversational emotion recognition.
- The semantic feature extension is independently beneficial.
- Sparsemax is superior to softmax.
- The gate has recovered the true causal utility of history.
- The current 3-seed matrix is sufficient for final generalization claims.

---

## 十二、下一步最小执行清单

如果现在优先完成论文 Experiments，建议按以下顺序补数据：

1. 先补 B0 在四个数据集上的统一协议结果。
2. 补四个数据集、三个 seed 的 E5 hard_abstain_rate、keep_prob、history residual norm 和 token mask ratio。
3. 补 E5/B1 的 same-label 与 different-label 分组指标。
4. 对 M3ED 做失败样本分析，至少输出 20 个代表性样本的候选、空候选分数和最终决策。
5. 完成 U0/U1 utility-cache 对照。
6. 代码和协议冻结后，再跑参数敏感性。
7. 最后才做 test 评估并写最终主结果。

若时间非常紧，最低可先完成第 1–3 项；没有这三项，正文只能写“候选模型筛选结果”，还不能完整证明“自动弃权确实解决了错误历史”。

---

## 十三、数据来源

本数据包使用以下材料整理：

- 本地状态：代码/创新点3_句级弃权改进/MULTIDATASET_STATUS_20261003.md
- 执行状态：代码/创新点3_句级弃权改进/EXECUTION_STATUS.md
- 有效数据汇总：文档/创新点3_实验数据整理_20261004.md
- 设计规范：文档/创新点3_后续改造详细对照_给新窗口.md
- 实验协议：文档/创新点3_详细执行与消融方案_20261002.md
- 论文相关工作：论文/04_Related_Work_第三版_英中文_20261004.txt


