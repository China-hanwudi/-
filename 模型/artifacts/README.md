# 自有训练产物（可提交）

按根目录 `DATA_BOUNDARY.md`：此处可放置**团队训练得到的** N3 / ComposerN3 checkpoint 与配套指标卡。

- 可以：`*.pt` / `*.pth` / `*.safetensors`、聚合 `metrics.json`、说明 Markdown  
- 不可以：MELD / EmotionTalk / IEMOCAP 原始数据或可还原样本的特征库  
- 单文件 > 100MB 请启用 Git LFS  

第三方预训练塔（如 XLM-RoBERTa-large）请协作者自行从 Hugging Face 下载，不必整包镜像到本目录。
