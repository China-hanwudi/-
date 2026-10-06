import json
import sys
from pathlib import Path


def main(root):
    root = Path(root)
    for p in sorted(root.glob("seed*/FINAL_RESULT.json")):
        d = json.loads(p.read_text())
        b = d["best_valid"]
        print(p.parent.name, d["seed"], "wf1=%.6f" % b["weighted_f1"],
              "best_epoch=%s" % d["best_epoch_index"],
              "epochs=%s" % d["epochs_completed"])


if __name__ == "__main__":
    main(sys.argv[1])
