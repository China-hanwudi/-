import json
import sys
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parent / "代码" / "创新点2_门控对照" / "code"
sys.path.insert(0, str(CODE))

from n6.data import open_split  # noqa: E402
from n6.evaluate import load_checkpoint  # noqa: E402


MODS = ("T", "A", "V")


def main(ckpt, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_checkpoint(ckpt, device)
    model.eval()
    ds = open_split(data, "valid.pt", task=cfg.task)
    sums = {
        "n": 0,
        "weight": torch.zeros(3, dtype=torch.float64),
        "phi": torch.zeros(3, dtype=torch.float64),
        "weight_entropy": 0.0,
        "weight_max": 0.0,
        "match_phi": 0,
        "match_solo": 0,
        "oracle_top": torch.zeros(3, dtype=torch.float64),
        "group_weight": torch.zeros(3, 3, dtype=torch.float64),
        "group_n": torch.zeros(3, dtype=torch.float64),
        "best_solo_u": 0.0,
        "deployed_u": 0.0,
        "uniform_u": 0.0,
        "group_best_solo_u": torch.zeros(3, dtype=torch.float64),
        "group_deployed_u": torch.zeros(3, dtype=torch.float64),
        "group_uniform_u": torch.zeros(3, dtype=torch.float64),
    }
    with torch.no_grad():
        for st in range(0, ds.n, 64):
            b, y = ds.batch(range(st, min(st + 64, ds.n)))
            b = {k: v.to(device) for k, v in b.items()}
            y = y.to(device)
            out = model(b)
            phi = model.measure_shapley(b, y, standardize=False)
            weights = out["weights"].float()
            solo_stack = out["solo_stack"].float()
            deployed = out["deployed"].float()
            if cfg.task == "cls":
                solo_u = torch.stack([
                    torch.log_softmax(solo_stack[:, i, :], dim=-1)
                    .gather(1, y.view(-1, 1)).squeeze(1)
                    for i in range(3)], dim=1)
                dep_u = torch.log_softmax(deployed, dim=-1) \
                    .gather(1, y.view(-1, 1)).squeeze(1)
                uniform_logits = solo_stack.mean(dim=1)
                uni_u = torch.log_softmax(uniform_logits, dim=-1) \
                    .gather(1, y.view(-1, 1)).squeeze(1)
            else:
                solo_u = -torch.abs(solo_stack.squeeze(-1) - y.view(-1, 1))
                dep_u = -torch.abs(deployed.squeeze(-1) - y)
                uni_u = -torch.abs(solo_stack.mean(dim=1).squeeze(-1) - y)

            mask = b["modality_mask"].bool()
            phi = phi.masked_fill(~mask, float("-inf"))
            solo_u = solo_u.masked_fill(~mask, float("-inf"))
            weights = weights * mask.float()
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            weights = weights / denom
            top_w = weights.argmax(dim=1)
            top_phi = phi.argmax(dim=1)
            top_solo = solo_u.argmax(dim=1)
            best_solo_u = solo_u.max(dim=1).values
            sums["n"] += len(y)
            sums["weight"] += weights.double().sum(0).cpu()
            finite_phi = torch.where(torch.isfinite(phi), phi, torch.zeros_like(phi))
            sums["phi"] += finite_phi.double().sum(0).cpu()
            sums["match_phi"] += int((top_w == top_phi).sum())
            sums["match_solo"] += int((top_w == top_solo).sum())
            ent = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(1)
            sums["weight_entropy"] += float(ent.sum())
            sums["weight_max"] += float(weights.max(1).values.sum())
            for m in range(3):
                sel = top_phi == m
                cnt = int(sel.sum())
                if cnt:
                    sums["oracle_top"][m] += cnt
                    sums["group_n"][m] += cnt
                    sums["group_weight"][m] += weights[sel].double().sum(0).cpu()
                    sums["group_best_solo_u"][m] += best_solo_u[sel].double().sum().cpu()
                    sums["group_deployed_u"][m] += dep_u[sel].double().sum().cpu()
                    sums["group_uniform_u"][m] += uni_u[sel].double().sum().cpu()
            sums["best_solo_u"] += float(best_solo_u.sum())
            sums["deployed_u"] += float(dep_u.sum())
            sums["uniform_u"] += float(uni_u.sum())

    n = float(sums["n"])
    group_n = sums["group_n"].clamp_min(1)
    print(json.dumps({
        "ckpt": str(ckpt),
        "dataset": cfg.dataset_id,
        "task": cfg.task,
        "n": int(n),
        "mean_pred_weight": (sums["weight"] / n).tolist(),
        "mean_raw_phi": (sums["phi"] / n).tolist(),
        "weight_entropy": sums["weight_entropy"] / n,
        "weight_max": sums["weight_max"] / n,
        "top_match_raw_phi": sums["match_phi"] / n,
        "top_match_solo_utility": sums["match_solo"] / n,
        "oracle_top_fraction_raw_phi": (sums["oracle_top"] / n).tolist(),
        "mean_weight_when_raw_phi_top": (sums["group_weight"] / group_n[:, None]).tolist(),
        "utility_all": {
            "best_solo": sums["best_solo_u"] / n,
            "deployed": sums["deployed_u"] / n,
            "uniform_solo": sums["uniform_u"] / n,
        },
        "utility_by_raw_phi_top": {
            "n": sums["group_n"].tolist(),
            "best_solo": (sums["group_best_solo_u"] / group_n).tolist(),
            "deployed": (sums["group_deployed_u"] / group_n).tolist(),
            "uniform_solo": (sums["group_uniform_u"] / group_n).tolist(),
        },
    }))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
