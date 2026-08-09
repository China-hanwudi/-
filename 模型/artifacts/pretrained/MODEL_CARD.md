# Main-line vs branch text towers

## Main line (current)

| Field | Value |
|---|---|
| Key | `qwen3_4b` |
| Hugging Face | `Qwen/Qwen3-4B-Instruct-2507` |
| License | Apache-2.0（开源权重**免费**下载使用） |
| Role | N3 主线文本大模型（冻塔 + 训 N3 头） |
| In git? | **否**（约数 GB，请本机 `python -m n3_affect.download_qwen` 或 HF 缓存） |

## Branch (kept)

| Key | Model | In git |
|---|---|---|
| `emoberta_base` | `tae898/emoberta-base` | 是（`emoberta-base/`，Git LFS） |
| `composer_n3` | 无外部 LLM | 不需要 |
| `xlm_roberta_large` | `FacebookAI/xlm-roberta-large` | 否（按需下载） |
