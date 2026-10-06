"""Fit latest local_current on aligned fit bank, select eta on separate CAL."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from n6.crbef_fusion import LocalCurrentConfig, fit_local_current
from n6.crbef_expert import M3ED_CLASSES
from n6.metrics import classification_report
from n6.recomer import ReCoMER


def load_inputs(path):
    with np.load(path, allow_pickle=False) as f:
        z = {k: f[k].copy() for k in ("P", "C", "ids", "source", "y_true", "class_names")}
    if tuple(z["class_names"]) != M3ED_CLASSES:
        raise ValueError("class order mismatch")
    n = len(z["ids"])
    if (z["P"].shape != (n, 4, 7) or z["C"].shape != (n, 3, 7)
            or z["y_true"].shape != (n,) or z["source"].shape != (n,)):
        raise ValueError("invalid complete-model input shapes")
    if n == 0 or len(set(z["ids"])) != n:
        raise ValueError("input bank must have unique nonempty IDs")
    if (not np.isfinite(z["P"]).all() or not np.isfinite(z["C"]).all()
            or (z["P"] < 0).any() or not np.allclose(z["P"].sum(-1), 1, atol=2e-4)):
        raise ValueError("invalid probabilities/evidence")
    if not np.issubdtype(z["y_true"].dtype, np.integer) or not ((z["y_true"] >= 0) & (z["y_true"] < 7)).all():
        raise ValueError("invalid M3ED targets")
    return z


def four_way(model, z):
    with torch.no_grad():
        p = model(torch.from_numpy(z["P"]), torch.from_numpy(z["C"])).exp().numpy()
    probs = {"MHnoU": z["P"][:, 1], "cRBEF": z["P"][:, 0],
             "equal_weight": z["P"][:, :2].mean(1), "ReCoMER": p}
    reports = {k: classification_report(np.log(v.clip(1e-9)), z["y_true"].tolist(),
                                       7, M3ED_CLASSES) for k, v in probs.items()}
    return reports, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--cal", type=Path, required=True)
    ap.add_argument("--mhnou", type=Path, required=True)
    ap.add_argument("--expert-assets", type=Path, default=Path("crbef_assets/m3ed"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    fit, cal = load_inputs(args.fit), load_inputs(args.cal)
    if set(fit["ids"]) & set(cal["ids"]) or set(fit["source"]) & set(cal["source"]):
        raise ValueError("FIT and CAL must be sample/source disjoint")
    torch.manual_seed(args.seed)
    fuser, stats = fit_local_current(torch.from_numpy(fit["P"]), torch.from_numpy(fit["C"]),
                                    torch.from_numpy(fit["y_true"]), LocalCurrentConfig(7), epochs=args.epochs)
    selected, best, choices = 0.0, None, []
    for eta in (0.0, 0.25, 0.5, 1.0):
        fuser.cfg.eta = eta
        reports, _ = four_way(fuser, cal)
        r = reports["ReCoMER"]
        key = (r["weighted_f1"], r["macro_f1"], r["accuracy"])
        choices.append(dict(eta=eta, metrics=r))
        if best is None or key > best:
            best, selected = key, eta
    fuser.cfg.eta = selected
    ck = dict(cfg=fuser.cfg.__dict__, model=fuser.state_dict(), class_names=list(M3ED_CLASSES),
              seed=args.seed, fit_stats=stats, fit_ids=fit["ids"].tolist(), cal_ids=cal["ids"].tolist())
    torch.save(ck, args.out / "local_current.pt")
    complete = ReCoMER.from_files(args.mhnou, args.expert_assets, args.out / "local_current.pt")
    complete.save_bundle(args.out / "recomer.pt", dict(seed=args.seed, fit=str(args.fit), cal=str(args.cal),
                         scope="head-excluded development; archived upstream exposure remains",
                         full_system_OOF=False, official_test_evaluated=False))
    reports, _ = four_way(fuser, cal)
    receipt = dict(seed=args.seed, eta=selected, fit_rows=len(fit["ids"]), cal_rows=len(cal["ids"]),
                   choices=choices, four_way_cal=reports, fit_stats=stats,
                   fit_cal_source_disjoint=True, old_uniform_h_used=False,
                   full_system_OOF=False, official_test_evaluated=False)
    (args.out / "TRAIN_RESULT.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(dict(status="COMPLETE_RECOMER_SAVED", eta=selected, bundle=str(args.out / "recomer.pt"))))


if __name__ == "__main__":
    main()
