# MELD Pilot复现代码

本目录包含MELD官方文本＋真实WAV轻量声学特征Pilot的核心实现、冻结配置和不依赖真实数据的合同测试。

## 内容

- `configs/meld_audio_text_feasibility_v1.json`：预冻结实验配置、随机种子、统计门与数据来源标识；
- `src/hva_affect/data_contract.py`：MELD切分、时间顺序和同说话人历史合同；
- `src/hva_affect/meld_audio_contract.py`：官方CSV／WAV对齐与35维轻量声学特征合同；
- `src/hva_affect/meld_text_pilot.py`：current-only、zero-history与full-history文本反事实基础实现；
- `src/hva_affect/meld_audio_text_risk.py`：cross-fitting、point-risk selector、conformal q90门控、聚类bootstrap与输出；
- `scripts/`：数据预检和冻结实验入口；
- `tests/`：不读取MELD原始数据的合同单元测试。

## 安装与无数据测试

```powershell
python -m venv .venv
$meldPython = (Resolve-Path '.venv\Scripts\python.exe').Path
& $meldPython -m pip install -r experiment\requirements-multimodal.txt
& $meldPython -m pytest experiment\tests -q
```

## 完整实验

研究者需从官方来源合法取得MELD train/dev标签和真实WAV／派生声学特征。仓库不重分发原始文本、标签、音频、逐查询表或NPZ特征。

```powershell
& $meldPython experiment\scripts\run_meld_audio_text_risk.py `
  --train-csv <MELD标签目录>\train_sent_emo.csv `
  --dev-csv <MELD标签目录>\dev_sent_emo.csv `
  --train-audio <派生特征目录>\meld_train_audio_handcrafted_v1.npz `
  --dev-audio <派生特征目录>\meld_dev_audio_handcrafted_v1.npz `
  --config experiment\configs\meld_audio_text_feasibility_v1.json `
  --output <输出目录>\meld_audio_text_risk_v1.json
```

脚本会在同一目录派生写出`meld_audio_text_risk_v1_per_query.csv.gz`。MELD test在本Pilot中封存；不要将test路径传给该流程，也不要把逐查询输出提交到公开仓库。

聚合结果见[`../results/meld/`](../results/meld/)，完整解释见[`../docs/04_MELD音频文本Pilot结果.md`](../docs/04_MELD音频文本Pilot结果.md)。
