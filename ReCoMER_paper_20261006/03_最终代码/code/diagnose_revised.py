import json
import sys
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parent / "code"
sys.path.insert(0, str(CODE))

from n6.data import open_split
from n6.evaluate import load_checkpoint


def main(ckpt, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_checkpoint(ckpt, device)
    ds = open_split(data, "valid.pt", task=cfg.task)
    sums = torch.zeros(3, dtype=torch.float64)
    mu_sum = torch.zeros(3, dtype=torch.float64)
    n = 0
    for st in range(0, ds.n, 64):
        b, y = ds.batch(range(st, min(st + 64, ds.n)))
        b = {k: v.to(device) for k, v in b.items()}
        y = y.to(device)
        out = model(b)
        bs = y.shape[0]
        sums += out["weights"].detach().cpu().double().sum(dim=0)
        mu_sum += out["utility_mu"].detach().cpu().double().sum(dim=0)
        n += bs
    gamma = model.contrib_corrector.gamma.detach().cpu().tolist()
    print(json.dumps({
        "ckpt": str(ckpt),
        "weights_mean": (sums / n).tolist(),
        "mu_mean": (mu_sum / n).tolist(),
        "gamma": gamma,
        "bounded_lambda": cfg.bounded_lambda,
        "contrib_correct": cfg.contrib_correct,
        "detach_utility_path": cfg.detach_utility_path,
    }))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
