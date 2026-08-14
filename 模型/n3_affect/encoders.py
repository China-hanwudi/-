"""Six-way modality encoders for N3.

Main-line tower: ``qwen3_omni_30b_a3b`` → ``Qwen/Qwen3-Omni-30B-A3B-Instruct``.
Branch: ``emoberta_base`` (vendored), ``xlm_roberta_large``, ``composer_n3``.

Weights are never vendored for Omni (too large); load from HF id or local cache only.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from .config import N3TrainConfig

HF_TEXT_TOWERS = {
    "qwen3_omni_30b_a3b": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "emoberta_base": "tae898/emoberta-base",
    "xlm_roberta_large": "FacebookAI/xlm-roberta-large",
}

CAUSAL_TEXT_TOWERS = frozenset({"qwen3_omni_30b_a3b"})


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
        # Omni-30B needs bf16 + device_map offload on 32GB GPUs (e.g. RTX 5090).
        load_kwargs: dict = {"trust_remote_code": True}
        if causal or "Omni" in str(model_source) or "omni" in str(model_source).lower():
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"
            load_kwargs["low_cpu_mem_usage"] = True
        if causal:
            try:
                from transformers import Qwen3OmniMoeForConditionalGeneration

                lm = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    model_source, **load_kwargs
                )
                if hasattr(lm, "disable_talker"):
                    lm.disable_talker()
                # Thinker text tower hidden size.
                thinker = getattr(lm, "thinker", None)
                self.backbone = (
                    getattr(thinker, "model", None)
                    or getattr(lm, "model", None)
                    or getattr(lm, "transformer", None)
                    or lm
                )
            except Exception:
                try:
                    lm = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)
                except Exception:
                    # Omni checkpoints may need AutoModel + remote code.
                    lm = AutoModel.from_pretrained(model_source, **load_kwargs)
                self.backbone = getattr(lm, "model", None) or getattr(lm, "transformer", None) or lm
        else:
            self.backbone = AutoModel.from_pretrained(model_source, trust_remote_code=True)
        for p in self.backbone.parameters():
            p.requires_grad = False
        cfg = getattr(self.backbone, "config", None)
        hidden = None
        if cfg is not None:
            hidden = getattr(cfg, "hidden_size", None)
            if hidden is None and hasattr(cfg, "text_config"):
                hidden = getattr(cfg.text_config, "hidden_size", None)
            if hidden is None and hasattr(cfg, "thinker_config"):
                tcfg = cfg.thinker_config
                text_cfg = getattr(tcfg, "text_config", tcfg)
                hidden = getattr(text_cfg, "hidden_size", None)
        if hidden is None:
            hidden = 2048  # Qwen3-Omni thinker text default
        self.proj = nn.Linear(int(hidden), d_model)

    def forward_texts(self, texts: list[str], device: torch.device) -> Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        # With device_map="auto", feed inputs to the first parameter device.
        try:
            first_device = next(self.backbone.parameters()).device
        except StopIteration:
            first_device = device
        if first_device.type == "meta":
            first_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoded = {k: v.to(first_device) for k, v in encoded.items()}
        with torch.no_grad():
            out = self.backbone(**encoded)
            hidden = out.last_hidden_state
            if self.causal:
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                pooled = hidden[:, 0]
        return self.proj(pooled.to(self.proj.weight.device))


class SixWayEncoders(nn.Module):
    """Encode current and history streams independently (no early fusion).

    Applies history_mask and modality_mask so invalid history slots and
    missing modalities do not contribute to downstream layers.
    """

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
        history_mask = batch.get("history_mask")
        modality_mask = batch.get("modality_mask")
        if self.hf_text is not None and texts_current is not None and texts_history is not None:
            device = batch["T_t"].device
            z_tt = self.hf_text.forward_texts(texts_current, device)
            z_th = self.hf_text.forward_texts(texts_history, device)
        else:
            z_tt = self.text_proj(batch["T_t"])
            z_th = self.text_proj(batch["T_h"])
        z_at = self.audio_proj(batch["A_t"])
        z_vt = self.video_proj(batch["V_t"])
        z_ah = self.audio_proj(batch["A_h"])
        z_vh = self.video_proj(batch["V_h"])

        if modality_mask is not None:
            z_tt = z_tt * modality_mask[:, 0:1]
            z_at = z_at * modality_mask[:, 1:2]
            z_vt = z_vt * modality_mask[:, 2:3]
            if history_mask is not None:
                slot_weight = modality_mask[:, 0:1].unsqueeze(-1) * history_mask.unsqueeze(-1)
                z_th = z_th * slot_weight
                slot_weight = modality_mask[:, 1:2].unsqueeze(-1) * history_mask.unsqueeze(-1)
                z_ah = z_ah * slot_weight
                slot_weight = modality_mask[:, 2:3].unsqueeze(-1) * history_mask.unsqueeze(-1)
                z_vh = z_vh * slot_weight
            else:
                z_th = z_th * modality_mask[:, 0:1].unsqueeze(-1)
                z_ah = z_ah * modality_mask[:, 1:2].unsqueeze(-1)
                z_vh = z_vh * modality_mask[:, 2:3].unsqueeze(-1)
        elif history_mask is not None:
            z_th = z_th * history_mask.unsqueeze(-1)
            z_ah = z_ah * history_mask.unsqueeze(-1)
            z_vh = z_vh * history_mask.unsqueeze(-1)

        return {
            "T_t": z_tt,
            "A_t": z_at,
            "V_t": z_vt,
            "T_h": z_th,
            "A_h": z_ah,
            "V_h": z_vh,
        }
