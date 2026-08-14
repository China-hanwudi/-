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
from n3_affect.utility import EFFECT_ORDER, representation_deltas


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
    assert out["U_mix"].shape == (4, 1)
    assert out["mix_weights"].shape == (4, len(EFFECT_ORDER))
    assert out["joint_keep_prob"].shape == (4, 1)
    assert torch.allclose(out["mix_weights"].sum(dim=-1), torch.ones(4), atol=1e-5)


def test_representation_deltas_reconstruct() -> None:
    n, d = 3, 8
    z = {k: torch.randn(n, d) for k in ("TAV", "0AV", "T0V", "TA0", "T00", "0A0", "00V", "000")}
    dlt = representation_deltas(z)
    rebuilt = (
        z["000"]
        + dlt["T_bwd"]
        + dlt["A_bwd"]
        + dlt["V_bwd"]
        + dlt["TA"]
        + dlt["TV"]
        + dlt["AV"]
        + dlt["cross"]
    )
    assert torch.allclose(rebuilt, z["TAV"], atol=1e-5)
    assert torch.allclose(dlt["T_fwd"], z["TAV"] - z["0AV"], atol=1e-5)


def test_mix_logits_get_emotion_grad() -> None:
    cfg = N3TrainConfig(text_tower="composer_n3")
    model = N3EmotionModel(cfg)
    labels = torch.randint(0, 7, (4,))
    out = model(_batch(cfg))
    losses = n3_total_loss(out, labels, cfg)
    losses["loss"].backward()
    assert model.utility.mix_logits.grad is not None
    assert model.utility.mix_logits.grad.abs().sum() > 0


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
