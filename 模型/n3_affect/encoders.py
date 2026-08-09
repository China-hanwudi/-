"""Six-way modality encoders for N3.

Main-line text tower: ``qwen3_4b`` → ``Qwen/Qwen3-4B-Instruct-2507`` (Apache-2.0).
Branch towers: ``emoberta_base`` (vendored), ``xlm_roberta_large``, or
feature-only ``composer_n3`` projectors.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from .config import N3TrainConfig

HF_TEXT_TOWERS = {
    "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
    "emoberta_base": "tae898/emoberta-base",
    "xlm_roberta_large": "FacebookAI/xlm-roberta-large",
}

# Causal LMs lack a BERT-style CLS; use masked mean pooling.
CAUSAL_TEXT_TOWERS = frozenset({"qwen3_4b"})


class ModalityProjector(nn.Module):
    def __init__(self, in_dim: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class OptionalHFTextTower(nn.Module):
    """Frozen Hugging Face encoder/LM -> d_model."""

    def __init__(self, model_source: str, d_model: int, *, causal: bool = False) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HF text towers require transformers. "
                "pip install 'transformers>=4.51' safetensors sentencepiece accelerate"
            ) from exc
        self.causal = causal
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if causal:
            lm = AutoModelForCausalLM.from_pretrained(
                model_source,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            # Prefer the underlying transformer body when present.
            self.backbone = getattr(lm, "model", None) or getattr(lm, "transformer", None) or lm
        else:
            self.backbone = AutoModel.from_pretrained(model_source, trust_remote_code=True)
        for p in self.backbone.parameters():
            p.requires_grad = False
        hidden = int(getattr(self.backbone.config, "hidden_size"))
        self.proj = nn.Linear(hidden, d_model)

    def forward_texts(self, texts: list[str], device: torch.device) -> Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            out = self.backbone(**encoded)
            hidden = out.last_hidden_state
            if self.causal:
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                pooled = hidden[:, 0]
        return self.proj(pooled)


class SixWayEncoders(nn.Module):
    """Encode current and history streams independently (no early fusion)."""

    def __init__(self, cfg: N3TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.text_proj = ModalityProjector(cfg.text_dim, cfg.d_model, cfg.dropout)
        self.audio_proj = ModalityProjector(cfg.audio_dim, cfg.d_model, cfg.dropout)
        self.video_proj = ModalityProjector(cfg.video_dim, cfg.d_model, cfg.dropout)
        self.hf_text: OptionalHFTextTower | None = None
        if cfg.text_tower in HF_TEXT_TOWERS:
            source = cfg.resolved_text_model_source()
            self.hf_text = OptionalHFTextTower(
                source,
                cfg.d_model,
                causal=cfg.text_tower in CAUSAL_TEXT_TOWERS,
            )

    def forward(
        self,
        batch: Mapping[str, Tensor],
        texts_current: list[str] | None = None,
        texts_history: list[str] | None = None,
    ) -> dict[str, Tensor]:
        if self.hf_text is not None and texts_current is not None and texts_history is not None:
            device = batch["T_t"].device
            z_tt = self.hf_text.forward_texts(texts_current, device)
            z_th = self.hf_text.forward_texts(texts_history, device)
        else:
            z_tt = self.text_proj(batch["T_t"])
            z_th = self.text_proj(batch["T_h"])
        return {
            "T_t": z_tt,
            "A_t": self.audio_proj(batch["A_t"]),
            "V_t": self.video_proj(batch["V_t"]),
            "T_h": z_th,
            "A_h": self.audio_proj(batch["A_h"]),
            "V_h": self.video_proj(batch["V_h"]),
        }
