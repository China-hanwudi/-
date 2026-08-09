"""Six-way modality encoders for N3.

Default ``composer_n3`` uses learnable projectors over pre-extracted features.
Optional ``emoberta_base`` loads the vendored ERC encoder under
``模型/artifacts/pretrained/emoberta-base`` (MIT, MELD/IEMOCAP-oriented).
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from .config import N3TrainConfig

HF_TEXT_TOWERS = {
    "emoberta_base": "tae898/emoberta-base",
    "xlm_roberta_large": "FacebookAI/xlm-roberta-large",
}


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
    """Frozen Hugging Face encoder -> d_model."""

    def __init__(self, model_source: str, d_model: int) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HF text towers require transformers. "
                "pip install transformers safetensors sentencepiece"
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.backbone = AutoModel.from_pretrained(model_source)
        for p in self.backbone.parameters():
            p.requires_grad = False
        hidden = int(self.backbone.config.hidden_size)
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
            pooled = out.last_hidden_state[:, 0]
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
            self.hf_text = OptionalHFTextTower(source, cfg.d_model)

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
