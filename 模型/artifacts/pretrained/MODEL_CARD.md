# Pretrained text tower: EmoBERTa-base

| Field | Value |
|---|---|
| Selected for | CARMA-Affect / N3 dialogue emotion (MELD-aligned ERC) |
| Hugging Face id | `tae898/emoberta-base` |
| Upstream paper | Kim & Vossen, *EmoBERTa: Speaker-Aware Emotion Recognition in Conversation with RoBERTa* (2021) |
| Upstream code license | MIT ([tae898/erc](https://github.com/tae898/erc/)) |
| Why this weight | Domain match for **Emotion Recognition in Conversation**; reported on **MELD / IEMOCAP**; MIT-friendly for research mirrors; base size is uploadable via Git LFS (unlike multi-GB XLM-R large) |
| Not chosen | Qwen2.5-0.5B (too weak / chat-general); XLM-R-large (strong multilingual but ~2GB+ and not emotion-specialized); Hartmann DistilRoBERTa (strong 7-class EN emotion, less ERC/MELD-specific than EmoBERTa) |

## Files

Snapshot of the public HF revision stored under `emoberta-base/` for offline / collaborator bootstrap. Prefer citing the upstream model card and paper in publications.

## Load

```python
from transformers import AutoModel, AutoTokenizer
path = "模型/artifacts/pretrained/emoberta-base"
tok = AutoTokenizer.from_pretrained(path)
enc = AutoModel.from_pretrained(path)
```

N3 package optional tower key: `emoberta_base` (see `模型/configs/n3_train_v1.json`).
