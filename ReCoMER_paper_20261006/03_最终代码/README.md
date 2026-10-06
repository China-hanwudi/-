# 最终代码：MHnoU 与完整 ReCoMER

2026-10-04 已补齐独立 cRBEF 专家定义、原包权重、统一 `ReCoMER` 推理对象、
融合训练和单文件模型保存。以 `ReCoMER_补全说明.md` 为当前完整模型入口说明。
MHnoU 原主干保留；原包旧 `uniform_h` 不用于替换它。
三个种子的完整模型已实际训练并保存在本地，四路开发结果与权重位置见
`ReCoMER_首轮结果说明.md`。

这份代码以创新点三句级历史弃权改进版为主干，接入创新点二的
`EvidenceRouter`，并提供创新点一的 `closed_loop` 总框架路径。

## 三个部件的关系

- 创新点一：`--deploy closed_loop`。历史准入后的当前表示进入 joint head，
  joint 输出承担最终预测；closed-loop 路径不再把同一段历史同时作为 pooled
  feedback 和 raw history token 计入。
- 创新点二：`--gate-architecture evidence --bounded-w`。EvidenceRouter 使用
  模态局部证据和跨模态上下文预测贡献分数，继续使用逐样本 Shapley、中心化
  bounded 权重和缺失模态重归一化。
- 创新点三：`--history-abstain-variant utility_softmax`。真实历史槽与空候选
  共同竞争；空候选胜出时同时切断历史反馈和 raw history token 路径。

## 推荐的整合训练入口

```text
python -m n6.train \
  --data <pack> --out <run> --seed <seed> \
  --utility shapley --gate-architecture evidence \
  --gate-detach-inputs --detach-utility-path --bounded-w \
  --bounded-lambda 0.3 --history-abstain-variant utility_softmax \
  --deploy closed_loop
```

`legacy`、`solo_weighted` 等默认值仍保留，用于兼容旧 checkpoint 和隔离消融。
创新点二的冻结底座门控训练入口为 `python -m n6.train_gate`。

## cRBEF v2 外部专家与 local_current 融合

本副本现在默认使用压缩包中的 `local_current` 外层融合。cRBEF 内部仍保留
自己的可靠性门控；外层先取 cRBEF 与 MHnoU 的等权基线，再用 23 个类别级特征
和一个共享小网络对候选类别进行有界的局部概率重分配：

```text
q = (p_cRBEF + p_MHnoU) / 2
fused = q + 0.25 * eta * (local_class_correction(q, P, C) - q)
```

其中 `P` 的四列必须按 `[cRBEF, MHnoU, TAV_reference, no_history_reference]`
提供，`C` 为 `[T, A, V]` 的相对模态证据，形状分别为 `[N,4,C]` 和 `[N,3,C]`。
候选集合外的类别不被重新分配，修正幅度也受到限制。未拟合的校正器退化为
等权基线，不能作为正式性能结果。

先准备两个模型的同批样本预测和一个 `fusion_inputs.npz`：

```text
P:             [N,4,C] float32
C:             [N,3,C] float32
ids:           [N]      （建议提供）
class_names:   [C]      （建议提供）
```

然后运行：

```text
cd code
python tools/fuse_crbef.py \
  --main <MHnoU运行目录>/valid_logits.npz \
  --crbef <cRBEF运行目录>/valid_predictions.npz \
  --fusion-inputs <fusion_inputs.npz> \
  --out <输出目录>/crbef_fused.npz \
  --num-classes 7 --gate-checkpoint <输出目录>/local_current.pt
```

校正器必须在训练折或 OOF 预测上单独拟合，再用于验证/测试。拟合时增加：

```text
--fit-gate --fit-main <训练或OOF的MHnoU预测.npz> \
           --fit-crbef <训练或OOF的cRBEF预测.npz> \
           --fit-fusion-inputs <训练或OOF的fusion_inputs.npz> \
           --save-gate <输出目录>/local_current.pt
```

如需兼容旧的单标量 cRBEF 外部门控，可显式加 `--fusion crbef_gate`；该模式
不需要 `fusion_inputs.npz`。融合结果会写入 `.npz`，同时生成同名 `.json`，
其中包含融合公式、类别校正幅度、候选集合比例和分类指标。

## 检查

在 `code` 目录执行：

```text
python -m unittest discover -s tests -v
```

现有历史路径测试和新增整合路径测试均覆盖：空候选硬弃权、残差切断、
EvidenceRouter 输出、bounded 权重归一化以及 closed-loop 不重复计入 raw history。

此前版本仅完成代码整合与 CPU smoke/unit 验证。当前补全版本新增特征级完整模型
和三种子分组训练入口；实际训练状态、模型和结果见实验记录。开发评估与正式
测试结果需分别报告，不能沿用旧包成绩作为最新 ReCoMER 的成绩。
