"""Summarize paired gate-only screens without treating validation search as confirmation."""
import argparse
import json
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    args = p.parse_args()
    datasets = {}
    for dataset in ("meld", "mosei", "m3ed"):
        groups = {}
        for f in sorted((args.root / dataset).glob("seed*/*/FINAL_RESULT.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            key = "mae" if dataset == "mosei" else "weighted_f1"
            base, new = r["base_valid"][key], r["best_valid"][key]
            gain = base-new if dataset == "mosei" else new-base
            groups.setdefault(r["variant"], []).append(dict(
                seed=r["seed"], baseline=base, value=new, gain=gain,
                best_epoch=r["best_epoch_index"]))
        datasets[dataset] = {variant: dict(
            n=len(rows), mean_gain=statistics.mean(r["gain"] for r in rows),
            wins=sum(r["gain"] > 0 for r in rows),
            mean_value=statistics.mean(r["value"] for r in rows),
            pairs=rows) for variant, rows in groups.items()}
    print(json.dumps(dict(status="exploratory 3-seed screen",
                          higher_gain_is_better=True, datasets=datasets), indent=2))


if __name__ == "__main__":
    main()
