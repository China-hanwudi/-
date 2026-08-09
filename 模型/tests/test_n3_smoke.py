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
    cfg = N3TrainConfig(text_tower="composer_n3")
    model = N3EmotionModel(cfg)
    out = model(_batch(cfg))
    assert out["logits"].shape == (4, 7)
    assert out["relation_grid"].shape == (4, 3, 3)
    assert out["U_T"].shape == (4, 2)
    assert out["U_cross"].shape == (4, 1)
    assert out["joint_keep_prob"].shape == (4, 1)


def test_loss_backward() -> None:
    cfg = N3TrainConfig(text_tower="composer_n3")
    model = N3EmotionModel(cfg)
    batch = _batch(cfg)
    labels = torch.randint(0, 7, (4,))
    out = model(batch)
    losses = n3_total_loss(out, labels, cfg, vad_targets=torch.randn(4, 3))
    losses["loss"].backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_config_from_json_mainline_qwen() -> None:
    path = ROOT / "configs" / "n3_train_v1.json"
    cfg = N3TrainConfig.from_json(path)
    assert cfg.text_tower == "qwen3_omni_30b_a3b"
    assert "Qwen3-Omni-30B-A3B" in cfg.hf_text_model_id
    assert cfg.num_classes == 7


def test_branch_towers_validate() -> None:
    for key in ("composer_n3", "emoberta_base", "qwen3_omni_30b_a3b", "xlm_roberta_large"):
        cfg = N3TrainConfig(text_tower=key)
        cfg.validate()


def test_smoke_train() -> None:
    cfg = N3TrainConfig(text_tower="composer_n3", max_epochs=1, batch_size=4, seed=17)
    card = train_one_run(cfg, steps_per_epoch=1)
    assert card["model_name"] == "ComposerN3"
    assert card["history"][0]["loss"] > 0
