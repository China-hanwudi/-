# EmotionTalk三模态外部确认终判报告

**研究任务：** CARMA-Affect／历史负迁移研究路线外部确认

**冻结协议：** `emotiontalk_multimodal_external_v1`

**数据版本：** `BAAI/Emotiontalk@adbc17fc944e8cf2873643906160c6ca0259ab61`

**终验日期：** 2026-08-07
**证据状态：** validation一次性终验完成；test继续封存

## 一、结论先行

最终结论不是“CARMA被多模态挽救”，而是：**问题成立，当前安全回退方法仍失败；负迁移benchmark路线可以继续，CARMA方法型路线暂不应投稿。**

- **自然历史伤害复现：PASS。** 三模态full-history相对三模态current-only在1,770个有历史查询中伤害率为33.90%，dialogue聚类95% CI为29.99%–38.18%。
- **多模态selector增量：PASS。** 三模态harm AUC为0.6773；相对文本selector提高0.0871，95% CI为0.0508–0.1235。Spearman提高0.1778，95% CI为0.0961–0.2548。
- **严格q90安全回退：FAIL。** 只覆盖9/1,770条，即0.51%，远低于10%最低门；所用历史的平均超额NLL仍为+0.0863 nats，策略总体均值超额NLL为+0.00044，95% CI上界为+0.00143。
- **历史配对特异性：PASS。** 真实历史相对20次受限置换历史平均获得0.4339 nats NLL优势，95% CI为0.3394–0.5525；置换历史的伤害率为60.62%，显著高于真实历史的33.90%。
- **路线终判：** `carma_method_route_go = false`；`negative_transfer_benchmark_route_go = true`。

## 二、核心研究问题

本阶段只回答三个预冻结问题：

1. 在新的中文三模态对话数据上，个体历史是否仍会对一部分查询造成自然负迁移？
2. 音频和视频信息是否能在同一三模态伤害目标上提高伤害预测？
3. 经过独立calibration的严格q90风险上界，能否在非零且非平凡覆盖下安全使用历史？

核心目标定义为：对每个有历史的查询，计算三模态full-history相对三模态current-only的超额负对数似然：

\[
\Delta L_t=L(y_t,\hat y_t^{\mathrm{full\ history}})-L(y_t,\hat y_t^{\mathrm{current\ only}}).
\]

当\(\Delta L_t>0\)时，历史伤害当前预测；当\(\Delta L_t<0\)时，历史改善当前预测。selector在推理时不能访问\(y_t\)，只能预测这一伤害风险。

## 三、数据与无标签特征合同

- 官方固定版本共19,250条三模态话语；Phase 19只使用train 15,413条和validation 1,908条，test特征和标签均未打开。
- validation中有1,770条查询具有同dialogue、同speaker且严格更早的历史，占92.77%。
- 文本只使用官方`transcription.csv`中文转写。
- 音频只使用原始WAV经`microsoft/wavlm-base-plus`生成的无EmotionTalk标签表示：最后隐藏层时间均值与标准差，共1,536维。
- 视频只使用主MP4经`facebook/dinov2-small`生成的无EmotionTalk标签表示：4个均匀帧、最大正脸加边界裁剪、无脸时保留全帧，CLS均值与标准差共768维。
- 所有归档JSON、`audio.csv`情绪caption、`emotion_result`以及模态特定情绪标签均禁止进入模型。
- train+validation的17,321条音视频特征精确对齐、0失败、0非有限值、0 test行；特征SHA-256为`d9d8c40c4f1325a68d0b9488c6a54846d29234606039135ac959aeeeedc439c2`。

## 四、预冻结验证流程

1. 按dialogue做5折GroupKFold，保证同一对话不跨base模型fit／held-out。
2. 每折只在fit dialogue上拟合中文TF-IDF、音频标准化+96维PCA、视频标准化+96维PCA。
3. 在held-out dialogue上同时生成current-only和full-history概率、逐查询超额NLL与selector几何特征。
4. 四个selector消融均预测同一个三模态超额NLL目标：text、text+audio、text+video、text+audio+video。
5. 对有历史的OOF查询再按dialogue拆分selector fit 11,341条与独立calibration 2,901条。
6. base与风险模型均采用5个随机种子；q90风险上界用独立calibration做conformal修正。
7. 代码、配置、音视频特征、转写、标签容器与训练bundle共同写入freeze manifest后，validation只执行一次。
8. 按dialogue进行2,000次聚类bootstrap；全部预冻结门无论正负均报告。

## 五、主结果

### 5.1 模态消融与自然负迁移

| Base输入 | Current-only NLL | Full-history NLL | Current／Full Macro-F1 | 历史伤害率（95% CI） | 有历史查询均值超额NLL（95% CI） | p90／p99 regret |
|---|---:|---:|---:|---:|---:|---:|
| 文本 | 1.5159 | 1.4026 | 0.3773／0.4115 | 28.93%（24.30%–33.45%） | -0.1221（-0.1793至-0.0655） | 0.5366／1.7485 |
| 文本+音频 | 1.3748 | 1.3479 | 0.4422／0.4650 | 29.27%（25.35%–33.69%） | -0.0290（-0.0941至0.0369） | 0.8742／3.0649 |
| 文本+视频 | 1.6136 | 1.7125 | 0.3812／0.3916 | 39.10%（34.41%–43.66%） | +0.1066（0.0362–0.1807） | 1.0835／3.0198 |
| 三模态 | 1.4993 | 1.5516 | 0.4577／0.4916 | 33.90%（29.99%–38.18%） | +0.0564（-0.0216至0.1360） | 1.0977／3.5204 |

必须同时解释两个现象：三模态full-history把Macro-F1从0.4577提高到0.4916，但NLL从1.4993恶化到1.5516，并伤害33.90%的个体查询。这说明平均分类指标与逐查询概率风险并不等价，也是本benchmark的核心价值。

### 5.2 Selector模态增量

| Selector输入 | Harm AUC | Balanced Accuracy | Spearman ρ | Validation q90上界覆盖率 |
|---|---:|---:|---:|---:|
| 文本 | 0.5903 | 0.5552 | 0.0102 | 90.56% |
| 文本+音频 | 0.6380 | 0.5922 | 0.1154 | 90.73% |
| 文本+视频 | 0.5886 | 0.5520 | 0.0535 | 91.58% |
| 三模态 | 0.6773 | 0.6199 | 0.1879 | 92.20% |

三模态相对文本的AUC增量为+0.0871（95% CI 0.0508–0.1235），Spearman增量为+0.1778（95% CI 0.0961–0.2548），两项均超过预冻结点估计门且bootstrap下界为正。因此，**音视频确实提供了与历史伤害有关的增量信号，但该信号尚不足以支持严格安全回退。**

### 5.3 严格q90回退为什么失败

严格规则只在预测伤害q90上界小于0时使用历史。validation结果为：

- 覆盖9/1,770条，0.51%，低于10%最低门；
- 使用历史的9条中1条受损，伤害率11.11%；
- 9条的平均超额NLL为+0.0863，p90为+0.1663，p99为+0.7673；
- 相对current-only的总体均值超额NLL为+0.00044，95% CI为-0.00006至+0.00143，上界仍大于0。

因此失败同时来自“覆盖过低”和“安全性未证实”，不能表述为只差一个阈值。固定10%–90%覆盖策略属于风险—覆盖描述，不可用于事后替换预冻结主门。

### 5.4 受限历史置换负控制

每个validation查询优先从同speaker、不同dialogue、相同历史深度分箱抽取历史聚合表示，共20次固定seed置换；所有置换预测均在读取validation标签前完成。

- 真实历史相对置换历史的NLL优势为+0.4339 nats，95% CI 0.3394–0.5525；
- 真实历史伤害率33.90%，置换历史伤害率60.62%；
- 置换历史相对current-only的平均超额NLL为+0.4903。

这证明结果并非“随便加任意历史都会一样”，支持建立真实历史配对的负迁移benchmark；它不等于证明当前selector已经能可靠找到有用历史。

## 六、质量敏感性分析

所有质量样本均保留在主分析中，分层只用于识别测量条件的影响：

- 四帧均检出正脸：n=1,047，伤害率32.38%，均值超额NLL +0.0007，95% CI跨0。
- 部分帧检出正脸：n=668，伤害率36.53%，均值超额NLL +0.1451，95% CI 0.0368–0.2505。
- 四帧均未检出正脸：n=55，伤害率30.91%，区间较宽；没有删除。
- 单声道：n=67，伤害率20.90%，均值超额NLL -0.1995；双声道：n=1,703，伤害率34.41%，均值超额NLL +0.0664。单声道样本少，不能据此作因果解释。
- validation有历史查询中没有16 kHz源音频，全部为44.1 kHz；因此16 kHz分层在本终验上不可估计，不应伪造比较结论。

## 七、可信度与审稿风险

### 已控制

- 严格同dialogue、同speaker、过去时序历史；
- dialogue组外cross-fitting，所有预处理只在fit组拟合；
- 独立selector calibration；
- validation前代码／配置／bundle哈希冻结；
- validation只运行一次，结果文件禁止覆盖；
- 5随机种子、dialogue聚类bootstrap、逐查询结果和受限置换负控制；
- test继续封存；
- 质量异常不静默删除。

### 仍有限制

- EmotionTalk是演员对话，不能直接外推真实临床或自然长期生活场景；
- conformal覆盖只保证calibration dialogue上的边际性质，不是逐speaker安全保证；
- 预训练表示是冻结特征，未验证端到端多模态编码器；
- validation说话人并非完全与其他split隔离，当前证据主要支持查询级、对话级风险结论；
- 多个base消融是预注册诊断，不能把其中最有利的一项事后改成主结果；
- 本阶段未启封test，最终论文若使用test，必须作为新的独立终验，且不能反向调参。

## 八、路线终判与下一步

| 路线 | 终判 | 依据 | 下一步 |
|---|---|---|---|
| CARMA／安全回退方法论文 | STOP | 严格q90只覆盖0.51%，总体均值regret上界仍为正 | 不在EmotionTalk validation继续调阈值；需要新方法或新数据上的重新预注册 |
| 历史负迁移benchmark／问题论文 | GO | 33.90%自然伤害、重尾regret、跨数据复现、真实历史优于受限置换 | 组织MELD+EmotionTalk双数据证据；申请IEMOCAP作第三数据集 |
| “多模态能否提高伤害预测”机制结论 | 有条件支持 | AUC与Spearman增量的配对聚类CI均为正 | 作为benchmark中的分析贡献，不夸大为安全部署能力 |
| EmotionTalk test终验 | 暂不执行 | 当前目标是validation外部确认，test仍封存 | 论文方案、代码和统计门完成最终预注册后再独立启封 |

建议把论文核心问题收窄为：

> 在纵向个性化情感预测中，加入真实个人历史何时会提高或损害当前预测？现有可观测的多模态风险信号，能否在不访问当前标签的条件下支持非平凡且安全的历史使用？

当前证据对前半问给出明确肯定，对后半问给出明确否定。这个“问题成立、方法仍未解决”的组合更适合高质量benchmark／诊断论文，而不是继续包装现有CARMA为成功方法。

## 九、最小复现与哈希

```powershell
experiment\.venv\Scripts\python.exe -m pytest experiment/tests -q
experiment\.venv\Scripts\python.exe experiment/scripts/run_emotiontalk_multimodal_external.py train --data-dir experiment/external/EmotionTalk_metadata/EmotionTalk/dataset/mm-process
experiment\.venv\Scripts\python.exe experiment/scripts/run_emotiontalk_multimodal_external.py freeze --data-dir experiment/external/EmotionTalk_metadata/EmotionTalk/dataset/mm-process
experiment\.venv\Scripts\python.exe experiment/scripts/run_emotiontalk_multimodal_external.py validate --data-dir experiment/external/EmotionTalk_metadata/EmotionTalk/dataset/mm-process
```

关键SHA-256：

- 配置：`fdd5cd3065c33360eb25837feb4e791296183e514d6a74c8017e8d378465a90e`
- 三模态特征：`d9d8c40c4f1325a68d0b9488c6a54846d29234606039135ac959aeeeedc439c2`
- 训练bundle：`3f66f5383e9f79b8b6c756e069d003a0a8e396548ca2ba7e9a110430ec3b2204`
- Freeze manifest：`f50cde004b747d7e2e26452bf6ac5673bccd8b135ac6da09c9ef2af3087d44e4`
- Validation结果JSON：`bd2dba738a2de326f165859b0e4dd101c623a31c635b309dfbca1de4fa1c9a01`
- 逐查询表：`a370c9aa378bcc07a6644169a4a8fbe30b592f43e41c742dffcdab4c8eb969ee`

## 十、最终判断

**实验工程全流程：PASS。** 数据、无标签特征、时间安全历史、cross-fitting、calibration、一次性validation、聚类统计、置换控制、多种子、质量敏感性和复现产物均已跑通。

**CARMA科学假设：FAIL。** 严格q90安全回退没有获得非平凡覆盖，也没有证明策略均值regret不高于0。
**历史负迁移研究价值：PASS。** EmotionTalk三模态外部证据复现自然伤害，证明真实历史配对具有特异性，并显示多模态信息能提高伤害可预测性，但仍不足以实现安全使用。
