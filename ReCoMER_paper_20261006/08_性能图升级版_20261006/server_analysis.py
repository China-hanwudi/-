# -*- coding: utf-8 -*-
"""在服务器上运行：
A) 正式 M3ED 测试逐类指标 + ReCoMER 对照的配对 bootstrap（按对话重采样）
B) MELD 延续/转折子群分析（transition_group 已在 SAMPLE_DIAGNOSTICS 中）
输出 perclass_m3ed.json 与 subgroup_meld.json。
"""
import json
import os
import numpy as np

BASE = "/root/recomer_remaining_tests_20261004_side"
SEEDS = ["43", "47", "59"]
ARMS = ["MHnoU", "cRBEF", "equal_weight", "ReCoMER"]
CLASS_NAMES = ["Happy", "Neutral", "Sad", "Disgust", "Anger", "Fear", "Surprise"]


def per_class_f1(y_true, probs, num_classes=7):
    pred = probs.argmax(axis=1)
    f1s, prec, rec, sup = [], [], [], []
    for c in range(num_classes):
        tp = int(((pred == c) & (y_true == c)).sum())
        fp = int(((pred == c) & (y_true != c)).sum())
        fn = int(((pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        prec.append(p); rec.append(r); f1s.append(f)
        sup.append(int((y_true == c).sum()))
    return f1s, prec, rec, sup


def dialogue_ids(ids):
    """diaN_uttM -> diaN"""
    out = []
    for s in ids:
        s = str(s)
        out.append(s.split("_utt")[0] if "_utt" in s else s)
    return np.array(out)


def bootstrap_paired_diff(y_true, prob_a, prob_b, dgroups, n_boot=1000, seed=0):
    """每对话重采样，返回 F1 向量差的 (mean, lo, hi)；按类返回 [7,3]。"""
    rng = np.random.default_rng(seed)
    dlg = np.unique(dgroups)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(dlg, size=len(dlg), replace=True)
        mask = np.isin(dgroups, pick)
        fa, _, _, _ = per_class_f1(y_true[mask], prob_a[mask])
        fb, _, _, _ = per_class_f1(y_true[mask], prob_b[mask])
        diffs.append(np.array(fa) - np.array(fb))
    diffs = np.array(diffs)
    lo = np.percentile(diffs, 2.5, axis=0)
    hi = np.percentile(diffs, 97.5, axis=0)
    return diffs.mean(axis=0), lo, hi


def main():
    # ---------- A. M3ED per-class ----------
    per_seed = {}
    for s in SEEDS:
        d = np.load(os.path.join(BASE, "full_test", s, "PREDICTIONS.npz"))
        y = d["y_true"]
        arms = {a: d[a] for a in ["MHnoU", "cRBEF", "equal_weight"]}
        arms["ReCoMER"] = d["probabilities"]
        entry = {}
        for a in ARMS:
            f1s, prec, rec, sup = per_class_f1(y, arms[a])
            entry[a] = {"f1": f1s, "precision": prec, "recall": rec}
        dg = dialogue_ids(d["ids"])
        boots = {}
        for ctrl in ["equal_weight", "cRBEF"]:
            m, lo, hi = bootstrap_paired_diff(y, arms["ReCoMER"], arms[ctrl],
                                              dg, seed=int(s))
            boots[ctrl] = {"mean": m.tolist(), "lo": lo.tolist(),
                           "hi": hi.tolist()}
        per_seed[s] = {"support": [int((y == c).sum()) for c in range(7)],
                       "arms": entry, "boot_ReCoMER_minus": boots,
                       "n_dialogues": int(len(np.unique(dg)))}

    # 跨 seed 汇总
    summary = {}
    for a in ARMS:
        arr = np.array([per_seed[s]["arms"][a]["f1"] for s in SEEDS])
        summary[a] = {"f1_mean": arr.mean(axis=0).tolist(),
                      "f1_sd": arr.std(axis=0).tolist()}
    out_a = {"class_names": CLASS_NAMES, "per_seed": per_seed,
             "summary": summary}
    with open(os.path.join(BASE, "perclass_m3ed.json"), "w") as f:
        json.dump(out_a, f, indent=1)

    # ---------- B. MELD subgroup ----------
    sub = {}
    for s in SEEDS:
        d = np.load(os.path.join(BASE, "mechanisms", "MELD", s,
                                 "SAMPLE_DIAGNOSTICS.npz"))
        y = d["labels"]
        tg = d["transition_group"].astype(int)
        deployed = d["deployed"]
        on = d["history_on_logits"]
        off = d["history_off_logits"]
        def wf1(y_true, logits):
            z = logits
            z = z - z.max(axis=1, keepdims=True)
            p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
            return per_class_f1(y_true, p)
        entry = {}
        for g, gname in [(0, "continuation"), (1, "transition")]:
            m = tg == g
            f_on, _, _, _ = wf1(y[m], on[m])
            f_off, _, _, _ = wf1(y[m], off[m])
            acc_on = float((on[m].argmax(1) == y[m]).mean())
            acc_off = float((off[m].argmax(1) == y[m]).mean())
            # deployed = 最终 MHnoU 部署输出
            fd, _, _, _ = per_class_f1(y[m], deployed[m])
            entry[gname] = {
                "n": int(m.sum()),
                "acc_history_on": acc_on, "acc_history_off": acc_off,
                "wf1_history_on_mean": float(np.mean(f_on)),
                "wf1_history_off_mean": float(np.mean(f_off)),
                "wf1_deployed": float(np.mean(fd)),
            }
        # 全体
        f_on, _, _, _ = wf1(y, on); f_off, _, _, _ = wf1(y, off)
        entry["all"] = {
            "n": int(len(y)),
            "acc_history_on": float((on.argmax(1) == y).mean()),
            "acc_history_off": float((off.argmax(1) == y).mean()),
            "wf1_history_on_mean": float(np.mean(f_on)),
            "wf1_history_off_mean": float(np.mean(f_off)),
        }
        sub[s] = entry
    with open(os.path.join(BASE, "subgroup_meld.json"), "w") as f:
        json.dump(sub, f, indent=1)

    print("DONE perclass + subgroup")


if __name__ == "__main__":
    main()
