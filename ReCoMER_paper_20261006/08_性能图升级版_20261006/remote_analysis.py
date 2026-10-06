# -*- coding: utf-8 -*-
"""服务器端分析（CPU 即可）：
A) 正式 M3ED 测试逐类 F1 + 配对 bootstrap CI（补 item 12 残余）
B) 延续/转折子群分析（item 5）：history on/off 在各子群的差异

输出 subgroup_perclass_results.json 到当前目录。
"""
import json
import numpy as np
from collections import defaultdict

rng = np.random.default_rng(20261006)
CLASS_NAMES = ["Happy", "Neutral", "Sad", "Disgust", "Anger", "Fear", "Surprise"]
BASE = "/root/recomer_remaining_tests_20261004_side"


def per_class_prf(y_true, y_pred, C):
    out = []
    for c in range(C):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        sup = int((y_true == c).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out.append({"precision": p, "recall": r, "f1": f, "support": sup})
    return out


def wf1(y_true, y_pred, C):
    # weighted F1 = sklearn-style: per-class F1 weighted by support
    pcs = per_class_prf(y_true, y_pred, C)
    sup = np.array([x["support"] for x in pcs], dtype=float)
    f1 = np.array([x["f1"] for x in pcs], dtype=float)
    return float((f1 * sup).sum() / sup.sum())


def dialogue_ids(ids):
    """样本 id 前缀即对话 id（协议：重采样单位为完整对话）。"""
    return np.array([str(i).rsplit("_", 1)[0] for i in ids])


# ---------- A) 正式测试逐类 F1 ----------
seeds = ["43", "47", "59"]
arms = ["MHnoU", "cRBEF", "equal_weight", "probabilities"]  # probabilities=ReCoMER
arm_label = {"MHnoU": "MHnoU", "cRBEF": "cRBEF",
             "equal_weight": "equal_weight", "probabilities": "ReCoMER"}

per_seed = {}
y_true_ref, ids_ref = None, None
for s in seeds:
    d = np.load(f"{BASE}/full_test/{s}/PREDICTIONS.npz")
    y_true = d["y_true"]
    ids = d["ids"]
    per_seed[s] = {"y_true": y_true, "ids": ids}
    for a in arms:
        per_seed[s][a] = d[a].argmax(axis=1)
    if y_true_ref is None:
        y_true_ref, ids_ref = y_true, ids

C = 7
per_class = {arm_label[a]: [] for a in arms}
for s in seeds:
    yt = per_seed[s]["y_true"]
    for a in arms:
        per_class[arm_label[a]].append(per_class_prf(yt, per_seed[s][a], C))

# 跨 seed 均值±sd（逐类 F1）
per_class_summary = {}
for name, lst in per_class.items():
    arr = np.array([[x["f1"] for x in run] for run in lst])  # [3,7]
    sup = np.array([x["support"] for x in lst[0]])
    per_class_summary[name] = {
        "f1_mean": arr.mean(axis=0).tolist(),
        "f1_sd": arr.std(axis=0, ddof=1).tolist(),
        "support": sup.tolist(),
    }

def per_class_f1_vec(y_true, y_pred):
    return np.array([x["f1"] for x in per_class_prf(y_true, y_pred, C)])


def paired_boot(armA, armB, n_boot=1000):
    yt = per_seed["43"]["y_true"]
    ids = per_seed["43"]["ids"]
    dlg = dialogue_ids(ids)
    ud = np.unique(dlg)
    predA = per_seed["43"][armA]
    predB = per_seed["43"][armB]
    diffs = np.zeros((n_boot, C))
    for b in range(n_boot):
        samp = rng.choice(ud, size=len(ud), replace=True)
        mask = np.isin(dlg, samp)
        diffs[b] = per_class_f1_vec(yt[mask], predA[mask]) - \
                   per_class_f1_vec(yt[mask], predB[mask])
    lo, hi = np.percentile(diffs, [2.5, 97.5], axis=0)
    return {"diff_mean": diffs.mean(axis=0).tolist(),
            "ci_lo": lo.tolist(), "ci_hi": hi.tolist()}


boot = {
    "ReCoMER_minus_cRBEF": paired_boot("probabilities", "cRBEF"),
    "ReCoMER_minus_equal_weight": paired_boot("probabilities", "equal_weight"),
}

result_A = {"class_names": CLASS_NAMES,
            "per_class": per_class_summary, "bootstrap": boot}

# ---------- B) 子群分析（transition_group）----------
DS = ["IEMOCAP", "M3ED_textQwen", "MELD", "MOSEI_full"]
subgroup = {}
for ds in DS:
    agg = defaultdict(lambda: defaultdict(list))
    tg_vals = set()
    for s in seeds:
        try:
            d = np.load(f"{BASE}/mechanisms/{ds}/{s}/SAMPLE_DIAGNOSTICS.npz")
        except FileNotFoundError:
            continue
        tg = d["transition_group"]
        labels = d["labels"]
        deployed = d["deployed"].argmax(axis=1)
        acc = (deployed == labels)
        for g in np.unique(tg):
            m = tg == g
            tg_vals.add(float(g))
            agg[float(g)]["n"].append(int(m.sum()))
            agg[float(g)]["acc"].append(float(acc[m].mean()))
            # 历史效应：off/on logits 的 NLL 差 与 off/on 预测一致率
            l_off = d["history_off_logits"][m]
            l_on = d["history_on_logits"][m]
            pred_off = l_off.argmax(axis=1)
            pred_on = l_on.argmax(axis=1)
            agg[float(g)]["history_flip_rate"].append(float((pred_off != pred_on).mean()))
            # history on/off accuracy
            agg[float(g)]["acc_history_on"].append(float((pred_on == labels[m]).mean()))
            agg[float(g)]["acc_history_off"].append(float((pred_off == labels[m]).mean()))
    subgroup[ds] = {str(k): {kk: vv for kk, vv in v.items()} for k, v in agg.items()}
    subgroup[ds]["_groups_present"] = sorted(tg_vals)

out = {"A_per_class": result_A, "B_subgroup": subgroup,
       "meta": {"seeds": seeds, "n_boot": 1000,
                "bootstrap_unit": "whole dialogue (protocol-aligned)",
                "note": "cRBEF fixed checkpoint; bootstrap describes test-dialogue uncertainty"}}
with open(f"{BASE}/subgroup_perclass_results.json", "w") as f:
    json.dump(out, f, indent=1)
print("saved subgroup_perclass_results.json")
print(json.dumps({k: subgroup[k]["_groups_present"] for k in DS}))
