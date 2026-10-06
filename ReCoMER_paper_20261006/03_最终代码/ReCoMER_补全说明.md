# ReCoMER 完整实现与训练（2026-10-04）

首轮已完成三个种子的完整模型训练并下载本地。模型路径及四路结果见
`ReCoMER_首轮结果说明.md`；完整权重位于 `实验结果/ReCoMER_完整模型_20261004/seed*/recomer.pt`。

本次接入的模型是 **现有 MHnoU（三个部件）＋独立 cRBEF（TAV、rich AV、可靠性门控）＋local_current**。
MHnoU 的原模型、训练和损失文件没有被替换。导入的 cRBEF 来自用户提供的
`CRBEF_uniform_h_complete_candidate_20261003.zip`，专家定义和权重逐文件核对了原包 SHA-256。
原包的旧 `uniform_h` 权重和旧融合器没有作为当前模型使用。

## 已补齐的文件

| 文件 | 作用 |
| --- | --- |
| `code/n6/crbef_source/head.py`、`rich.py` | 原包中的专家网络定义，原样保留 |
| `code/crbef_assets/m3ed/` | TAV、AV、CR_gate 权重、原始标准化统计和来源记录 |
| `code/n6/crbef_expert.py` | cRBEF 的独立特征输入、可靠性门控和 ID 对齐 |
| `code/n6/recomer.py` | 完整推理对象、输入证据构造、统一模型保存与加载 |
| `code/tools/recomer_predict.py` | 使用原始专家特征或明确指定的专家预测缓存，生成同批输入/预测 |
| `code/tools/train_recomer_fusion.py` | 在 FIT 上拟合融合器，在分开的 CAL 上选择融合强度 |
| `实验结果/run_complete_recomer_20261004.py` | 与 cRBEF 源划分对齐的三种子训练和四路比较 |

## 推理过程

1. MHnoU 接收自己的当前 T/A/V 和历史，输出预测。
2. 同一个 MHnoU checkpoint 关闭历史掩码，得到无历史参考和三个单模态 logits。
3. 独立 cRBEF 接收原有 CR 特征及 rich audio，计算 TAV、AV 和可靠性门控后的预测。
4. 构造 `P=[cRBEF, MHnoU, TAV_reference, same_checkpoint_no_history_reference]`。
5. 从无历史单模态 logits 构造相对证据 `C`，交给重新拟合的 `local_current`。

原包的 CR 特征、rich audio 与 MHnoU 的特征是不同表示。不能把 MHnoU 的音频
或文本向量按维度裁剪后冒充专家特征。所有输入都需要按同一个 utterance ID 对齐。
监督标签不会传入统一模型的推理路径。未提供拟合后的融合器时，统一模型拒绝输出
所谓的最终 ReCoMER 预测。

## 当前专家的适用范围

本次导入的专家是 **M3ED 七分类**，类别顺序为
`Happy, Neutral, Sad, Disgust, Anger, Fear, Surprise`。
它不能直接用于 MELD/IEMOCAP 的不同类别定义，也不能直接用于 MOSEI 回归。
其他数据集需要各自的 cRBEF 专家与相应融合训练。

## 模型保存与使用

训练完成后，每个 MHnoU/融合种子会产生一个 `recomer.pt`，内部包含：

- 最新 MHnoU 配置和权重；
- cRBEF 两个专家、门控、先验和视觉标准化统计；
- local_current 权重、标准化统计和选定的 eta；
- 类别顺序与训练来源记录。

在 `code/` 目录使用：

```python
from n6.recomer import ReCoMER

model = ReCoMER.from_bundle("/path/to/recomer.pt", device="cuda")
# batch 是 MHnoU 当前/历史输入；expert_features 是同 ID 的原始专家特征。
out = model(batch, expert_features)
probabilities = out["fused"]
```

命令行先看 `python -m tools.recomer_predict --help`，完整模型支持 `--bundle`。
重新拟合融合器的入口是 `python -m tools.train_recomer_fusion --help`。

## 首轮训练协议与结果边界

服务器独立运行目录：`/root/recomer_complete_20261004_side`。
这次使用原包的四个源分组折和一个 final 分组，分别重新训练最新 MHnoU，种子为
43、47、59，共 15 次 MHnoU 拟合和 3 次外层融合拟合。学习率 2e-4 来自之前三种子
验证集均值比较。FIT 为 9741 个分组折外样本；CAL 为另外 1981 个样本；
另有 3931 个已经被研究过的 OUTER 开发样本。可用的官方 valid 也属于开发评估。

完整模型会比较 MHnoU、cRBEF、等权融合和重新拟合的 ReCoMER。
三个重复改变 MHnoU 和融合种子；原包 cRBEF 的源权重/源种子 17 保持固定。
它们不是三个全系统独立重训种子。原始上游教师和特征的历史监督曝光仍然存在，
这次匹配新的任务头划分不能使其变成全系统 OOF。官方 test 没有参与本次训练或选择。

由于服务器未找回原始训练 CR h 特征，本轮融合训练使用可核对的原包分组专家预测缓存。
这能完成最新 MHnoU 与 cRBEF 的实际融合训练；从原始媒体提取全部专家特征仍属于
外部数据准备。特征级专家推理已经通过原包真实权重和归档合成输入的数值一致性检查。

## 验证

`python -m unittest discover -s tests -v` 包括以下新增检查：
原始 cRBEF 权重输出重放、推理不读取历史标签、关闭历史参考不受历史特征变化影响、
完整模型包保存/加载后的预测一致，以及未拟合融合器不能冒充完整预测。
合成接口验证不能用作性能结果。
