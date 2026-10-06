import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent / "code"
sys.path.insert(0, str(ROOT))

from n6.config import M6Config
from n6.losses import total_loss
from n6.model import UGFModel


def main():
    torch.manual_seed(7)
    bsz, k, classes = 4, 3, 7
    cfg = M6Config(
        text_dim=768, audio_dim=2048, video_dim=342,
        task="cls", num_classes=classes, utility="shapley",
        use_history=True, history_k=k, use_bounded_w=True,
        bounded_lambda=0.3, contrib_correct=True,
        detach_utility_path=True, utility_head_version=3,
    )
    cfg.validate()
    model = UGFModel(cfg)
    batch = {
        "T_t": torch.randn(bsz, 768),
        "A_t": torch.randn(bsz, 2048),
        "V_t": torch.randn(bsz, 342),
        "T_h": torch.randn(bsz, k, 768),
        "A_h": torch.randn(bsz, k, 2048),
        "V_h": torch.randn(bsz, k, 342),
        "history_mask": torch.ones(bsz, k),
        "history_modality_mask": torch.ones(bsz, k, 3),
        "modality_mask": torch.ones(bsz, 3),
    }
    y = torch.randint(0, classes, (bsz,))
    out = model(batch)
    phi = model.measure_shapley(batch, y)
    loss = total_loss(out, y, cfg, phi=phi, mu=out["utility_mu"])
    loss["loss"].backward()
    assert out["weights"].shape == (bsz, 3)
    assert out["deployed"].shape == (bsz, classes)
    assert torch.isfinite(loss["loss"])
    assert model.contrib_corrector is not None
    print("SMOKE_OK", tuple(out["weights"].shape), tuple(out["deployed"].shape),
          float(loss["loss"]))


if __name__ == "__main__":
    main()
