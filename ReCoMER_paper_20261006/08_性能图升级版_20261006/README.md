# 性能图升级版（2026-10-06）

按课题组图表方法论重做的 4 张实验图，对应"论文里有但规格不够顶刊"的第 12–15 项。
**只新增图，未改动论文正文（main.tex）。**

## 设计规范（全部图共用）

- 每张图顶部大标题 = 一句话结论（读者看完应记住什么）；
- 副标题注明协议、样本量、seed 数、误差棒含义，保证 self-contained；
- 调色板固定（`style_shared.py`）：ReCoMER 蓝 `#0072B2`、cRBEF 琥珀 `#E69F00`、
  等权绿 `#009E73`、MHnoU 灰 `#8B8B8B`、显著绿 `#1B7837`、边缘琥珀、不显著灰；
- 每张图由同名 `.py` 脚本生成（300 dpi PNG + 矢量 PDF），改图改脚本后重跑即可。

## 图清单

| 文件 | 升级点 | 一句话结论 | 数据源 |
|---|---|---|---|
| figA_main_multimetric | 12（主表只有 WF1） | ReCoMER 在 accuracy/NLL/校准上领先所有融合规则，WF1 与固定专家差距在噪声内 | `SUMMARY.json` full_test + calibration |
| figB_router_forest_holm | 14（无校正后 p 值） | 路由在 MOSEI 对所有对照显著（10/10）；MELD/M3ED 方向依赖对照，部分 Holm 后失显著 | `main_crossdataset_summary.json` / `main_mosei_summary.json` |
| figC_mosei_metric_profile | 13（MOSEI 只有 MAE delta） | 历史反馈把 MOSEI 每项指标推向有利方向，但幅度小于种子间波动 | `mechanisms/MOSEI_full/{43,47,59}/METRICS.json` |
| figD_fusion_behavior | 15（融合行为不可见） | 学习到的融合是保守的：1/3 种子 η 退回等权，权重贴近均匀先验 | `SUMMARY.json` full_test_etas + 十 seed evidence_mean_weight |
| figE_perclass_m3ed | 12 残余（per-class） | 融合增益来自少数类救援：Sad 显著、Fear 边缘，多数类不变；Fear 未解决 | 服务器 `full_test/{43,47,59}/PREDICTIONS.npz` → `perclass_m3ed.json` |
| figF_meld_subgroup | 硬伤第 5 项（子群分析） | 历史反馈对延续句有一致小幅收益（3/3），对转折句中性 | 服务器 `mechanisms/MELD/*/SAMPLE_DIAGNOSTICS.npz` 的 transition_group → `subgroup_meld.json` |

## 服务器分析管线（2026-10-06 新增）

- `server_exec.py` / `srun.py`：paramiko 执行远程命令；`sftp.py`：文件传输
  （本机无 sshpass，SFTP  put 对该服务器有兼容问题，统一走 base64+SSH 管道）。
- `server_analysis.py`：在服务器 `/root/recomer_remaining_tests_20261004_side/` 运行，
  输出 `perclass_m3ed.json`（逐类 F1 + 逐对话 bootstrap）与 `subgroup_meld.json`。
- figE 星标规则：3 个 seed 中配对 bootstrap CI 不含 0 的个数（Sad 2/3、Fear 1/3）。
- **figF 与论文 4.8 节措辞冲突**：正文写"转折句被错误历史状态拖垮"，
  数据是"转折句无收益"（mean +0.05，2/3 seed 为正），改论文时须降级表述。

## 诚实性备注（写图注时必须保留）

1. **figA**：cRBEF/TAV/AV 为固定 checkpoint（sd=0），ReCoMER/等权为 3 paired seeds；
   WF1 差距 −0.60 pp 的 CI 跨 0，标题措辞已与证据强度对齐，不要再拔高。
2. **figB**：p 值为 Holm 族校正后结果；MELD vs uniform（p=0.0586）与
   M3ED vs uniform（p=0.0586）只能写 marginal，不能写 significant。
3. **figC**：只有 3 seeds，且是 mechanisms 测试协议（非 10-seed valid 协议）；
   "幅度小于种子波动"必须在图注或正文说明，否则会被误读为强增益。
4. **figD**：η=0（seed 47）意味着该种子实际输出=等权融合，这是校准选择的结果，
   不能删去或改写成提升。
5. **figA 未覆盖 item 12 的 per-class 部分**：正式 M3ED 测试的逐类指标需要服务器上的
   `CR17_TEST_PREDICTIONS.npz` / `full_test/` 预测文件，本地汇总只有聚合指标；
   补上逐类 F1 柱状图后 item 12 才算完整。

## 复现

```bash
cd 08_性能图升级版_20261006
python figA_main_multimetric.py
python figB_router_forest_holm.py
python figC_mosei_metric_profile.py
python figD_fusion_behavior.py
```

---

## 2026-10-06 第二批：服务器数据分析产物（图 E / 图 F）

服务器：`root@connect.westd.seetacloud.com:30558`（RTX 5090，数据在
`/root/recomer_remaining_tests_20261004_side`）。
分析脚本：`remote_analysis.py`（已上传服务器），结果：`subgroup_perclass_results.json`。

| 文件 | 对应缺口 | 一句话结论 |
|---|---|---|
| figE_perclass_f1_bootstrap | item 12 残余（per-class） | 融合重分配类间误差：对 cRBEF 显著 +Happy/Fear、显著 −Sad；对等权显著 +Sad |
| figF_subgroup_transition | item 5（子群分析） | 历史反馈 9/9 延续句为正；转折句 MELD/IEMOCAP 中性、M3ED 3/3 为负 |

### item 1–4 在服务器上的执行状态（2026-10-06 勘查）

- **item 1（外部 SOTA 表）**：未执行。需要选定基线并在同协议下复现；服务器有旧 `meld_sota_control` 残件可借鉴，属多日工程，需先定基线清单。
- **item 2（三数据集完整 ReCoMER）**：被设计阻塞。服务器 `recomer_validation_20261004_side/PLAN.json` scope 明确写着 "complete ReCoMER awaits original cRBEF expert predictions"——cRBEF 专家权重只有 M3ED 版，其他数据集需要按原始特征契约（4096d 文本等）重做特征提取 + 专家训练。
- **item 3（四数据集 B0 矩阵）**：IEMOCAP 的 10-seed 矩阵缺失；十 seed 矩阵 harness（integrated_H）在旧服务器 `/data/emo/...`，本机只有 3seed×2lr 验证矩阵（48 runs）。需移植脚本 + 冒烟测试后可排队。
- **item 4（弃权诊断矩阵）**：Table 4 的 B1/E5/E6/E7 是已冻结的 3-seed 修正协议；扩充诊断矩阵需 abstention 训练 harness（查 `run_data_followup_20261004.py` / `run_targeted_ablation_H.sh`），同样需先移植验证。

### figF 对论文 4.8 节 claim 的影响（重要）

论文写"MELD 上历史帮助延续句、转折句传播错误先验"。数据细化后：
- 延续句受益：**三个数据集全部 3/3 种子为正**（比论文 claim 更强）；
- 转折句：MELD 上实际为**中性**（+0.05 pp，2/3 种子微正），IEMOCAP 也是正的；
  **只有 M3ED 转折句一致为负**（−0.27 pp，3/3 种子）。
- 建议 4.8 节措辞改为"转折风险是数据集依赖的"（M3ED 上有证据，MELD 上无），
  figF 可作该节的支撑图。

---

## 2026-10-06 item 3 执行记录（GPU 服务器）

**关键发现**：item 3 所需的四数据集矩阵实际由 `paper_experiments_20261004`
（autodl-tmp）承载，而非旧服务器的 integrated_H harness。该队列 2026-10-05 完成，
覆盖 M3ED_strong/MELD/IEMOCAP/MOSEI_full × 5 臂（uniform_nohist=B0、uniform_history、
evidence_closed、evidence_solo、evidence_mpath）× 6 seeds（29,37,43,53,71,101），
缺 seeds 7/13/17/23。

**执行**：新建 `run_ablation_queue_fill_20261006.sh`（与原脚本逐字同配置，仅改 SEEDS），
setsid 后台运行 80 个补齐 run。冒烟通过：M3ED_strong uniform_nohist s7
valid WF1=0.5873、test_read=False、TRAIN_COMPLETE，与原队列量级一致。
日志：`/root/autodl-tmp/paper_experiments_20261004/logs/fill_20261006.out`，
完成标记 `FILL_QUEUE_20261006_COMPLETE`。

**结果**（`four_dataset_matrix_summary.json`，四数据集 × 5 臂 × 10 seeds 全齐）：

- **evidence_solo（deploy=solo_weighted）是唯一一致显著的正增益臂**：
  M3ED +1.50 pp（10/10，p=0.002，CI 不含 0）；MELD 对 uniform_history +0.75 pp（10/10，p=0.002）；
  IEMOCAP +1.20 pp vs B0（10/10，p=0.002）；MOSEI MAE 均值最优（8/10 改善）。
- **evidence_closed（论文当前的 closed_loop 部署）在四个数据集上均无显著增益**
  （+0.16/−0.08/+0.37/+0.18，CI 全部含 0）。
- 含义：本矩阵支持把默认部署从 closed_loop 改为 solo_weighted 的论证，
  或与正式测试协议（closed_loop）的差异做专门讨论。写进论文前需与导师确认口径。

---

## 2026-10-06 第三批：论文实验部分整体重排（已编译进论文）

交付物：`../ReCoMER_CVPR_修改版_v2_20261006.pdf`（LaTeX 工程 `ReCoMER_CVPR_LaTeX/`，tectonic 编译，0 引用/标签问题）。

### 正文图表（Section 4）
| 位置 | 内容 | 来源 |
|---|---|---|
| 4.2 | Table 1（重制：+MLP/DialogueRNN 同协议复现行，cRBEF 标注为 frozen expert reference） | baselines_m3ed.json |
| 4.2 | 图：三面板多指标（WF1/Acc/ECE，figA 瘦身版） | SUMMARY.json |
| 4.2 | 图：逐类 F1 × 6 系列含复现基线（figE 升级版） | subgroup_perclass + baselines_m3ed |
| 4.3 | Table 2 历史反馈；Table 3 路由（带 Holm p） | 原表 + main_crossdataset |
| 4.4 | Table 4 整合消融 | 原表 |
| 4.5 | Table 5 四数据集 × 5 臂 × 10 seed 矩阵 | four_dataset_matrix_summary |
| 4.6 | Table 6 跨数据集实例化 + 图 figH 双面板（WF1 柱 + 何时融合有效散点） | transfer_*.json |
| 4.7 | 图 figF 延续/转折子群（三数据集） | subgroup_perclass_results |

### 附录
弃权变体表（含变体定义）、exact-feature 控制表与图、figB Holm 森林图、figG 差距分解、
figC MOSEI 剖面、figD 融合行为、fig10-12 敏感性/PCA/混淆矩阵。

### 文字要点
- 4.2 新增：同协议复现基线对比（+13.3/+16.4 pp，明确声明非 SOTA claim）；
  cRBEF 差距的三句机制归因（指向附录 figG）。
- 4.6 新增跨数据集实例化小节：IEMOCAP +0.8 pp（3 seed 全正，诚实口径）、
  MELD 机制正确但受 T=1024 契约限制；附"输入强度差 vs 融合增益"规律图。
- 4.7 子群结论改为"转折风险是数据集依赖的"（M3ED 3/3 为负，MELD/IEMOCAP 中性）。
- pending 清单更新：四数据集 B0 矩阵与子群分析已完成并入正文。

---

## 2026-10-06 最终版：正文 8 页达成

`ReCoMER_CVPR_修改版_v3_20261006.pdf`：正文恰好 8 页（参考文献自第 9 页起），全篇无空栏、无重复标签、无未定义引用。

为达标做的结构决策（均保留信息）：
- Table 1 并入 fig:combined：逐类图新增 "Overall WF1" 组（8 组柱），表的 8 行数字全部入图，AV/TAV 披露移入图注；
- figA 三面板（WF1/Acc/ECE）整体在附录（fig:multimetric），正文 accuracy/校准声明引用它；
- figH（跨数据集双面板）在附录，正文保留全部数字与规律句；
- 四数据集矩阵、路由/历史/消融/弃权表均在附录，正文为数字+指针；
- 2/3/4/5 节做了不改意思的删繁就简（与 3.6 重复的超参句、变体定义移附录表注等）。
