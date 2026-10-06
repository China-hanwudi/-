import json
import sys
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parent / "code"
sys.path.insert(0, str(CODE))

from n6.data import open_split
from n6.evaluate import load_checkpoint


def corr(x, y):
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / (x.square().sum().sqrt() * y.square().sum().sqrt()).clamp_min(1e-9))


def main(ckpt, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_checkpoint(ckpt, device)
    ds = open_split(data, "valid.pt", task=cfg.task)
    mus, phis = [], []
    for st in range(0, ds.n, 64):
        b, y = ds.batch(range(st, min(st + 64, ds.n)))
        b = {k: v.to(device) for k, v in b.items()}
        y = y.to(device)
        with torch.no_grad():
            out = model(b)
        phi = model.measure_shapley(b, y, standardize=True)
        mus.append(out["utility_mu"].detach().cpu())
        phis.append(phi.detach().cpu())
    mu = torch.cat(mus)
    phi = torch.cat(phis)
    print(json.dumps({
        "ckpt": str(ckpt),
        "pearson_by_modality": [corr(mu[:, i], phi[:, i]) for i in range(3)],
        "mu_mean": mu.mean(0).tolist(),
        "phi_mean": phi.mean(0).tolist(),
        "mu_std": mu.std(0).tolist(),
        "phi_std": phi.std(0).tolist(),
    }))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
