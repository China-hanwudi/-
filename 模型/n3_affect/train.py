"""Synthetic smoke trainer + CLI entry for ComposerN3.

Real multimodal features must stay outside git (see DATA_BOUNDARY.md).
This script trains on random tensors shaped like the repo sidecars so the
architecture is executable without uploading datasets or checkpoints.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Allow `python -m n3_affect.train` from the 模型 directory.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from n3_affect.config import N3TrainConfig
from n3_affect.losses import n3_total_loss
from n3_affect.model import N3EmotionModel


class SyntheticSixWayDataset(Dataset):
    def __init__(self, cfg: N3TrainConfig, size: int = 64) -> None:
        self.cfg = cfg
        self.size = size
        g = torch.Generator().manual_seed(cfg.seed)
        self.T_t = torch.randn(size, cfg.text_dim, generator=g)
        self.A_t = torch.randn(size, cfg.audio_dim, generator=g)
        self.V_t = torch.randn(size, cfg.video_dim, generator=g)
        self.T_h = torch.randn(size, cfg.text_dim, generator=g)
        self.A_h = torch.randn(size, cfg.audio_dim, generator=g)
        self.V_h = torch.randn(size, cfg.video_dim, generator=g)
        self.y = torch.randint(0, cfg.num_classes, (size,), generator=g)
        self.vad = torch.tanh(torch.randn(size, 3, generator=g))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "T_t": self.T_t[idx],
            "A_t": self.A_t[idx],
            "V_t": self.V_t[idx],
            "T_h": self.T_h[idx],
            "A_h": self.A_h[idx],
            "V_h": self.V_h[idx],
            "label": self.y[idx],
            "vad": self.vad[idx],
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_run(cfg: N3TrainConfig, steps_per_epoch: int | None = None) -> dict:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = N3EmotionModel(cfg).to(device)
    n_params = model.count_trainable_parameters()
    if n_params > cfg.parameter_budget:
        print(
            f"warning: trainable params {n_params} exceed budget {cfg.parameter_budget}",
            file=sys.stderr,
        )
    ds = SyntheticSixWayDataset(cfg, size=max(cfg.batch_size * 4, 32))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = []
    model.train()
    for epoch in range(cfg.max_epochs):
        running = 0.0
        n = 0
        for step, batch in enumerate(loader):
            if steps_per_epoch is not None and step >= steps_per_epoch:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")
            vad = batch.pop("vad")
            out = model(batch)
            losses = n3_total_loss(out, labels, cfg, vad_targets=vad)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += float(losses["loss"].item())
            n += 1
        avg = running / max(n, 1)
        history.append({"epoch": epoch, "loss": avg})
        print(f"epoch={epoch} loss={avg:.4f} params={n_params} device={device}")
    card = model.export_card()
    card["history"] = history
    card["device"] = str(device)
    return card


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train ComposerN3 (synthetic smoke or config)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "n3_train_v1.json",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--text-tower",
        default=None,
        choices=["composer_n3", "qwen3_4b", "emoberta_base", "xlm_roberta_large"],
        help="Override config text tower (use composer_n3 for smoke without downloading Qwen)",
    )
    parser.add_argument("--steps-per-epoch", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = N3TrainConfig.from_json(args.config) if args.config.exists() else N3TrainConfig()
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.seed is not None:
        cfg.seed = args.seed
    if args.text_tower is not None:
        cfg.text_tower = args.text_tower
    cfg.validate()

    card = train_one_run(cfg, steps_per_epoch=args.steps_per_epoch)
    text = json.dumps(card, ensure_ascii=False, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        # Never write .pt weights into the git tree by default.
        print(f"wrote metrics card only (no checkpoint): {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
