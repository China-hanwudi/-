"""End-to-end N3 emotion classifier for training."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import N3TrainConfig
from .encoders import SixWayEncoders
from .gating import TwoLevelGate
from .relation import SharedThreeByThree
from .utility import BidirectionalUtilityHeads


class TheoryAuxHead(nn.Module):
    """Lightweight VAD regression head (fit-only auxiliary)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Linear(d_model, 3)

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(self.net(x))


class N3EmotionModel(nn.Module):
    """ComposerN3: six-way → 3×3 → utility → two-level gate → emotion logits."""

    def __init__(self, cfg: N3TrainConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or N3TrainConfig()
        self.cfg.validate()
        d = self.cfg.d_model
        self.encoders = SixWayEncoders(self.cfg)
        self.current_fuse = nn.Sequential(
            nn.Linear(d * 3, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )
        self.relation = SharedThreeByThree(d, self.cfg.relation_rank, self.cfg.dropout)
        self.utility = BidirectionalUtilityHeads(d, self.cfg.gate_hidden, self.cfg.dropout)
        self.gate = TwoLevelGate(d, self.cfg.gate_hidden, self.cfg.dropout)
        self.context = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=self.cfg.num_heads,
                dim_feedforward=self.cfg.ffn_dim,
                dropout=self.cfg.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            ),
            num_layers=self.cfg.num_layers,
        )
        self.classifier = nn.Linear(d, self.cfg.num_classes)
        self.vad_head = TheoryAuxHead(d)

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        batch: Mapping[str, Tensor],
        texts_current: list[str] | None = None,
        texts_history: list[str] | None = None,
    ) -> dict[str, Tensor]:
        streams = self.encoders(batch, texts_current=texts_current, texts_history=texts_history)
        current_pool = self.current_fuse(
            torch.cat([streams["T_t"], streams["A_t"], streams["V_t"]], dim=-1)
        )
        relation_vec, relation_grid = self.relation(streams)
        utilities = self.utility(current_pool, relation_vec)
        gated = self.gate(
            current_pool,
            {"T_h": streams["T_h"], "A_h": streams["A_h"], "V_h": streams["V_h"]},
            utilities,
        )
        tokens = torch.stack(
            [
                streams["T_t"],
                streams["A_t"],
                streams["V_t"],
                gated["representation"],
            ],
            dim=1,
        )
        encoded = self.context(tokens)
        pooled = encoded.mean(dim=1)
        logits = self.classifier(pooled)
        vad = self.vad_head(pooled)
        return {
            "logits": logits,
            "probs": F.softmax(logits, dim=-1),
            "vad": vad,
            "relation_grid": relation_grid,
            **utilities,
            **gated,
        }

    def predict_label(self, batch: Mapping[str, Tensor]) -> list[str]:
        with torch.no_grad():
            out = self.forward(batch)
            idx = out["logits"].argmax(dim=-1).tolist()
        return [self.cfg.emotion_label_order[i] for i in idx]

    def export_card(self) -> dict[str, Any]:
        return {
            "model_name": "ComposerN3",
            "protocol_id": "n3_train_v1",
            "text_tower": self.cfg.text_tower,
            "trainable_parameters": self.count_trainable_parameters(),
            "parameter_budget": self.cfg.parameter_budget,
            "num_classes": self.cfg.num_classes,
            "emotion_label_order": list(self.cfg.emotion_label_order),
        }
