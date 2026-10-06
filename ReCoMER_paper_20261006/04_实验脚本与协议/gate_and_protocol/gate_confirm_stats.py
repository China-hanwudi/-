"""Paired seed uncertainty for a frozen-base MOSEI confirmation."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

SEEDS = (7, 13, 17, 23, 29, 37, 43, 53, 71, 101)
PILOT = (17, 29, 43)


def stats(gains):
    gains = np.asarray(gains, dtype=float)
    n = len(gains)
    mean = float(gains.mean())
    se = float(gains.std(ddof=1)/np.sqrt(n))
    q = float(t.ppf(0.975, n-1))
    signs = np.asarray(list(itertools.product((-1, 1), repeat=n)))
    perm_means = (signs*gains).mean(1)
    p = float(np.mean(np.abs(perm_means) >= abs(mean)-1e-12))
    return dict(n=n, mean_gain=mean, ci95=[mean-q*se, mean+q*se],
                wins=int((gains>0).sum()), sign_flip_p_two_sided=p,
                gains=gains.tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    args = p.parse_args()
    runs = {}
    for variant in ("constant_task", "mlp_task", "evidence_task"):
        runs[variant] = {}
        for seed in SEEDS:
            file = args.root / "mosei" / ("seed"+str(seed)) / variant / "FINAL_RESULT.json"
            if not file.exists():
                raise SystemExit("incomplete confirmation: %s" % file)
            runs[variant][seed] = json.loads(file.read_text(encoding="utf-8"))
    out = dict(seed_order=list(SEEDS), pilot_seeds=list(PILOT),
               scope="training-seed variation on the same valid split; test sealed",
               comparisons={}, means={})
    for variant in runs:
        out["means"][variant] = float(np.mean([runs[variant][s]["best_valid"]["mae"] for s in SEEDS]))
    out["means"]["uniform"] = float(np.mean([runs["evidence_task"][s]["base_valid"]["mae"] for s in SEEDS]))
    for reference in ("uniform", "constant_task", "mlp_task"):
        gains = []
        for seed in SEEDS:
            ref = (runs["evidence_task"][seed]["base_valid"]["mae"] if reference == "uniform"
                   else runs[reference][seed]["best_valid"]["mae"])
            gains.append(ref - runs["evidence_task"][seed]["best_valid"]["mae"])
        r = stats(gains)
        r["nonpilot_7_seeds"] = stats([g for s,g in zip(SEEDS,gains) if s not in PILOT])
        out["comparisons"][reference] = r
    # Three pre-specified references for this selected recipe, not an
    # adjustment covering the preceding architecture/hyperparameter screen.
    ordered = sorted(out["comparisons"], key=lambda k: out["comparisons"][k]["sign_flip_p_two_sided"])
    adjusted = 0.0
    for i, name in enumerate(ordered):
        r = out["comparisons"][name]
        adjusted = max(adjusted, min(1.0, (3-i)*r["sign_flip_p_two_sided"]))
        r["holm_p_for_3_confirm_references"] = adjusted
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
