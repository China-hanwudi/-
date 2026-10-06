"""Read-only gradient audit; no optimizer or state updates."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "代码/创新点2_门控对照/code"))
from n6.data import open_split
from n6.evaluate import load_checkpoint
from n6.losses import total_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    rows = []
    for job in json.loads(Path(args.jobs).read_text(encoding="utf-8")):
        model, cfg, _ = load_checkpoint(job["ckpt"], torch.device("cpu"))
        ds = open_split(job["data"], "valid.pt", task=cfg.task)
        params = list(model.encoders.parameters())
        gen = torch.Generator().manual_seed(20261002)
        ix = torch.randperm(ds.n, generator=gen)
        cosines = []
        for k in range(3):
            b, y = ds.batch(ix[k*64:(k+1)*64].tolist())
            phi = model.measure_shapley(b, y)
            out = model(b)
            ls = total_loss(out, y, cfg, phi=phi)
            task_g = torch.autograd.grad(ls["task"], params, retain_graph=True)
            utility_g = torch.autograd.grad(ls["utility"], params)
            task_v = torch.cat([g.reshape(-1) for g in task_g])
            utility_v = torch.cat([g.reshape(-1) for g in utility_g])
            cosines.append(float(F.cosine_similarity(task_v, utility_v, dim=0)))
        row = {"name": job["name"], "cosines_task_utility_encoder": cosines}
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
