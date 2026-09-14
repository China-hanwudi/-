# TemporalN3 New-Dataset Mainline (v9)

本版本唯一活动数据集是 **M3ED、CMU-MOSEI、CH-SIMS_v2**。历史数据集代码已从活动路径移除，不再作为基准、训练入口或 SOTA 对比来源。

## 模型与数据契约

- M3ED：七类分类，输入维度 T/A/V=`768/1024/342`，主选择指标为 valid Weighted-F1，其次 Macro-F1、Accuracy、loss。
- CMU-MOSEI：连续回归，输入维度 T/A/V=`768/74/35`，标签按 `y/3` 归一化，报告值为 `3*u`，主指标为 valid MAE。
- CH-SIMS_v2：连续回归，输入维度从 packed manifest 读取（当前 `768/25/177`），使用 `train_chsims.py`，封存 test 永不读取。

对话历史统一由 packed `history_index` 构造为 oldest→newest、右对齐 K=3；索引越界、未来索引和无效槽位全部 mask 掉。无历史时 `hard-safe` 必须与 `current-only` 完全一致。

## 入口

```text
python -m n3_affect.train_m3ed --data /data/emo/肖田泽科研/数据/M3ED/packed --out runs/m3ed/seed_017 --seed 17
python -m n3_affect.train_mosei --data /data/emo/肖田泽科研/数据/MOSEI/packed --out runs/mosei/seed_017 --seed 17
python -m n3_affect.train_chsims --source . --data /data/emo/肖田泽科研/数据/CH-SIMS_v2/v7_packed --out runs/chsims/seed_017 --seed 17
```

正式训练前必须完成 `--smoke 1`、`python -m compileall n3_affect`、数据 hash/manifest 审计和 GPU 空闲复核。所有 run 独立保存 `best.pt`、`FINAL_RESULT.json`、配置/数据 hash 和 runtime 元数据；本版本当前只完成工程修订，未声称正式性能。
