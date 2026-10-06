# 最终代码 H：整合模型实验摘要（valid-only）

运行目录：`/data/emo/肖田泽科研/模型/model6_integrated_H_20261003/runs/matrix`。
所有结果由 `FINAL_RESULT.json` 自动汇总；本批次不读取 test.pt。
指标：M3ED/MELD 为 valid weighted-F1（↑），MOSEI 为 valid MAE（↓）。

## 汇总

| 数据集 | 臂 | n | 均值 | SD | 相对 uniform_h 的均值差 | 95% CI(差值) | 改善种子数 | test_read |
|---|---|---:|---:|---:|---:|---:|---:|---|
| M3ED | nohist | 3 | 0.585275 | 0.006012 | +0.002147 | [-0.016538, +0.020833] | 2/3 | False,False,False |
| M3ED | uniform_h | 3 | 0.583128 | 0.003454 | +0.000000 | [+0.000000, +0.000000] | 0/3 | False,False,False |
| M3ED | history_softmax | 3 | 0.586803 | 0.005012 | +0.003675 | [-0.013930, +0.021280] | 2/3 | False,False,False |
| M3ED | semantic | 1 | 0.577207 | 0.000000 | -0.002187 | [+nan, +nan] | 0/1 | False |
| M3ED | sparsemax | 1 | 0.578085 | 0.000000 | -0.001309 | [+nan, +nan] | 0/1 | False |
| M3ED | integrated | 3 | 0.584271 | 0.006122 | +0.001143 | [-0.006858, +0.009144] | 2/3 | False,False,False |
| M3ED | integrated_solo | 3 | 0.601315 | 0.001725 | +0.018187 | [+0.006504, +0.029870] | 3/3 | False,False,False |
| M3ED | integrated_jointsoft | 2 | 0.569181 | 0.002346 | -0.015814 | [-0.021499, -0.010129] | 0/2 | False,False |
| M3ED | jointsoft_true | 1 | 0.570539 | 0.000000 | -0.008854 | [+nan, +nan] | 0/1 | False |
| MELD | nohist | 3 | 0.581425 | 0.000974 | +0.002043 | [-0.013189, +0.017274] | 2/3 | False,False,False |
| MELD | uniform_h | 3 | 0.579383 | 0.005760 | +0.000000 | [+0.000000, +0.000000] | 0/3 | False,False,False |
| MELD | history_softmax | 3 | 0.576669 | 0.004418 | -0.002714 | [-0.026228, +0.020800] | 1/3 | False,False,False |
| MELD | semantic | 1 | 0.583156 | 0.000000 | +0.009812 | [+nan, +nan] | 1/1 | False |
| MELD | sparsemax | 1 | 0.585024 | 0.000000 | +0.011680 | [+nan, +nan] | 1/1 | False |
| MELD | integrated | 3 | 0.581892 | 0.003389 | +0.002509 | [-0.013324, +0.018342] | 1/3 | False,False,False |
| MELD | integrated_solo | 3 | 0.579562 | 0.004742 | +0.000180 | [-0.011522, +0.011882] | 2/3 | False,False,False |
| MELD | integrated_jointsoft | 2 | 0.577383 | 0.000664 | -0.005019 | [-0.029726, +0.019688] | 0/2 | False,False |
| MELD | jointsoft_true | 1 | 0.576329 | 0.000000 | +0.002985 | [+nan, +nan] | 1/1 | False |
| MOSEI | nohist | 3 | 0.563654 | 0.005291 | -0.000271 | [-0.017080, +0.016538] | 1/3 | False,False,False |
| MOSEI | uniform_h | 3 | 0.563383 | 0.002650 | +0.000000 | [+0.000000, +0.000000] | 0/3 | False,False,False |
| MOSEI | history_softmax | 3 | 0.558094 | 0.002541 | +0.005289 | [-0.006822, +0.017400] | 2/3 | False,False,False |
| MOSEI | semantic | 1 | 0.562130 | 0.000000 | +0.003148 | [+nan, +nan] | 1/1 | False |
| MOSEI | sparsemax | 1 | 0.562504 | 0.000000 | +0.002774 | [+nan, +nan] | 1/1 | False |
| MOSEI | integrated | 3 | 0.561348 | 0.001725 | +0.002035 | [-0.000657, +0.004727] | 3/3 | False,False,False |
| MOSEI | integrated_solo | 3 | 0.562700 | 0.004767 | +0.000683 | [-0.017086, +0.018452] | 2/3 | False,False,False |
| MOSEI | integrated_jointsoft | 2 | 0.562274 | 0.002197 | +0.000162 | [-0.006547, +0.006870] | 1/2 | False,False |
| MOSEI | jointsoft_true | 1 | 0.556996 | 0.000000 | +0.008282 | [+nan, +nan] | 1/1 | False |

## 逐 seed 原始指标

| 数据集 | seed | nohist | uniform_h | history_softmax | semantic | sparsemax | integrated | integrated_solo | integrated_jointsoft | jointsoft_true |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M3ED | 7 | 0.584223 | 0.579393 | 0.586987 | 0.577207 | 0.578085 | 0.577207 | 0.603000 | nan | 0.570539 |
| M3ED | 13 | 0.591744 | 0.583784 | 0.591721 | nan | nan | 0.588027 | 0.599552 | 0.567522 | nan |
| M3ED | 17 | 0.579859 | 0.586207 | 0.581701 | nan | nan | 0.587580 | 0.601392 | 0.570840 | nan |
| MELD | 7 | 0.582189 | 0.573344 | 0.578555 | 0.583156 | 0.585024 | 0.583156 | 0.574702 | nan | 0.576329 |
| MELD | 13 | 0.580329 | 0.579988 | 0.579830 | nan | nan | 0.578052 | 0.584177 | 0.576913 | nan |
| MELD | 17 | 0.581758 | 0.584816 | 0.571621 | nan | nan | 0.584468 | 0.579808 | 0.577853 | nan |
| MOSEI | 7 | 0.557772 | 0.565278 | 0.558194 | 0.562130 | 0.562504 | 0.562130 | 0.557518 | nan | 0.556996 |
| MOSEI | 13 | 0.568023 | 0.564517 | 0.555506 | nan | nan | 0.562544 | 0.563685 | 0.563828 | nan |
| MOSEI | 17 | 0.565167 | 0.560354 | 0.560584 | nan | nan | 0.559371 | 0.566898 | 0.560721 | nan |

审计：共 60 个完整结果；`test_read=true` 或异常状态条目数 = 0。
