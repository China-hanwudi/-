import glob
import json
import math
import os
import statistics

ROOT = "/data/emo/肖田泽科研/模型/model6_integrated_H_20261003/runs/matrix"
OUT = "/data/emo/肖田泽科研/模型/model6_integrated_H_20261003/summary"
os.makedirs(OUT, exist_ok=True)
DATASETS = {
    "M3ED": ("cls", ["nohist", "uniform_h", "history_softmax", "semantic", "sparsemax", "integrated", "integrated_solo", "integrated_jointsoft", "jointsoft_true"]),
    "MELD": ("cls", ["nohist", "uniform_h", "history_softmax", "semantic", "sparsemax", "integrated", "integrated_solo", "integrated_jointsoft", "jointsoft_true"]),
    "MOSEI": ("reg", ["nohist", "uniform_h", "history_softmax", "semantic", "sparsemax", "integrated", "integrated_solo", "integrated_jointsoft", "jointsoft_true"]),
}
SEEDS = [7, 13, 17]

rows = []
for ds, (task, arms) in DATASETS.items():
    for arm in arms:
        for seed in SEEDS:
            path = os.path.join(ROOT, f"{ds}_{arm}_s{seed}", "FINAL_RESULT.json")
            if not os.path.exists(path):
                continue
            d = json.load(open(path, encoding="utf-8"))
            b = d.get("best_valid", {})
            metric = b.get("weighted_f1") if task == "cls" else b.get("mae")
            rows.append({
                "dataset": ds, "task": task, "arm": arm, "seed": seed,
                "metric": metric, "direction": "higher" if task == "cls" else "lower",
                "epochs": d.get("epochs_completed"), "status": d.get("status"),
                "test_read": bool(d.get("test_read")), "path": path,
            })

json.dump(rows, open(os.path.join(OUT, "all_runs.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def mean_sd(vals):
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

def ci95(vals):
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    m, sd = mean_sd(vals)
    try:
        from scipy.stats import t
        crit = float(t.ppf(0.975, len(vals) - 1))
    except Exception:
        crit = 1.96
    half = crit * sd / math.sqrt(len(vals))
    return m - half, m + half

lines = [
    "# 最终代码 H：整合模型实验摘要（valid-only）",
    "",
    "运行目录：`/data/emo/肖田泽科研/模型/model6_integrated_H_20261003/runs/matrix`。",
    "所有结果由 `FINAL_RESULT.json` 自动汇总；本批次不读取 test.pt。",
    "指标：M3ED/MELD 为 valid weighted-F1（↑），MOSEI 为 valid MAE（↓）。",
    "",
    "## 汇总",
    "",
    "| 数据集 | 臂 | n | 均值 | SD | 相对 uniform_h 的均值差 | 95% CI(差值) | 改善种子数 | test_read |",
    "|---|---|---:|---:|---:|---:|---:|---:|---|",
]

for ds, (task, arms) in DATASETS.items():
    base = {r["seed"]: r["metric"] for r in rows if r["dataset"] == ds and r["arm"] == "uniform_h"}
    for arm in arms:
        rs = [r for r in rows if r["dataset"] == ds and r["arm"] == arm]
        vals = [r["metric"] for r in rs]
        if not vals:
            continue
        m, sd = mean_sd(vals)
        diffs = []
        for r in rs:
            if r["seed"] in base:
                # positive means improvement for either task
                raw = r["metric"] - base[r["seed"]]
                diffs.append(raw if task == "cls" else -raw)
        dm = statistics.mean(diffs) if diffs else float("nan")
        lo, hi = ci95(diffs) if diffs else (float("nan"), float("nan"))
        improved = sum(1 for x in diffs if x > 0)
        tr = ",".join(str(r["test_read"]) for r in rs)
        lines.append(f"| {ds} | {arm} | {len(vals)} | {m:.6f} | {sd:.6f} | {dm:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {improved}/{len(diffs)} | {tr} |")

lines += [
    "",
    "## 逐 seed 原始指标",
    "",
    "| 数据集 | seed | nohist | uniform_h | history_softmax | semantic | sparsemax | integrated | integrated_solo | integrated_jointsoft | jointsoft_true |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for ds, (task, arms) in DATASETS.items():
    by = {(r["arm"], r["seed"]): r["metric"] for r in rows if r["dataset"] == ds}
    for seed in SEEDS:
        vals = [by.get((a, seed), float("nan")) for a in ["nohist", "uniform_h", "history_softmax", "semantic", "sparsemax", "integrated", "integrated_solo", "integrated_jointsoft", "jointsoft_true"]]
        fmt = ["nan" if isinstance(v, float) and math.isnan(v) else f"{v:.6f}" for v in vals]
        lines.append(f"| {ds} | {seed} | " + " | ".join(fmt) + " |")

bad = [r for r in rows if r["test_read"] or r["status"] not in ("TRAIN_COMPLETE", "SMOKE_PASS")]
lines += ["", f"审计：共 {len(rows)} 个完整结果；`test_read=true` 或异常状态条目数 = {len(bad)}。"]
open(os.path.join(OUT, "SUMMARY.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

print("rows", len(rows), "bad", len(bad))
print("wrote", os.path.join(OUT, "SUMMARY.md"))
