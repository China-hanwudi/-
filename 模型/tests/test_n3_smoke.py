from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from n3_affect.config import N3TrainConfig
from n3_affect.losses import n3_total_loss
from n3_affect.model import N3EmotionModel
from n3_affect.train import train_one_run


def _batch(cfg: N3TrainConfig, n: int = 4) -> dict[str, torch.Tensor]:
    return {
        "T_t": torch.randn(n, cfg.text_dim),
        "A_t": torch.randn(n, cfg.audio_dim),
        "V_t": torch.randn(n, cfg.video_dim),
        "T_h": torch.randn(n, cfg.text_dim),
        "A_h": torch.randn(n, cfg.audio_dim),
        "V_h": torch.randn(n, cfg.video_dim),
    }


def test_forward_shapes() -> None:
    cfg = N3TrainConfig()
    model = N3EmotionModel(cfg)
    out = model(_batch(cfg))
    assert out["logits"].shape == (4, 7)
    assert out["relation_grid"].shape == (4, 3, 3)
    assert out["U_T"].shape == (4, 2)
    assert out["U_cross"].shape == (4, 1)
    assert out["joint_keep_prob"].shape == (4, 1)


def test_loss_backward() -> None:
    cfg = N3TrainConfig()
    model = N3EmotionModel(cfg)
    batch = _batch(cfg)
    labels = torch.randint(0, 7, (4,))
    out = model(batch)
    losses = n3_total_loss(out, labels, cfg, vad_targets=torch.randn(4, 3))
    losses["loss"].backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_config_from_json() -> None:
    path = ROOT / "configs" / "n3_train_v1.json"
    cfg = N3TrainConfig.from_json(path)
    assert cfg.text_tower == "composer_n3"
    assert cfg.hf_text_model_id == "FacebookAI/xlm-roberta-large"
    assert cfg.num_classes == 7


def test_xlm_tower_config_validates() -> None:
    cfg = N3TrainConfig(text_tower="xlm_roberta_large")
    cfg.validate()
    assert "xlm-roberta-large" in cfg.hf_text_model_id


def test_smoke_train() -> None:
    cfg = N3TrainConfig(max_epochs=1, batch_size=4, seed=17)
    card = train_one_run(cfg, steps_per_epoch=1)
    assert card["model_name"] == "ComposerN3"
    assert card["history"][0]["loss"] > 0
