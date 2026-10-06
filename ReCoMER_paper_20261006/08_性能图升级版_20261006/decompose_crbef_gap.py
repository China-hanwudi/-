# -*- coding: utf-8 -*-
"""cRBEF > ReCoMER 的样本级归因分解（正式 M3ED 测试，3 paired seeds）。
输出 crbef_vs_recomer_decomposition.json + 控制台报告。
"""
import json
import os
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_test_local")
SEEDS = ["43", "47", "59"]
CLASSES = ["Happy", "Neutral", "Sad", "Disgust", "Anger", "Fear", "Surprise"]


def wf1(y_true, y_pred, C=7):
    sup = np.bincount(y_true, minlength=C).astype(float)
    f1s = []
    for c in range(C):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float((np.array(f1s) * sup).sum() / sup.sum()), f1s


report = {"seeds": {}, "pooled": {}}
preds = {}
for s in SEEDS:
    d = np.load(os.path.join(BASE, s, "PREDICTIONS.npz"))
    m = json.load(open(os.path.join(BASE, s, "METRICS.json"), encoding="utf-8"))
    eta = m.get("eta")
    yt = d["y_true"]
    arms = {k: d[k] for k in ["MHnoU", "cRBEF", "equal_weight", "probabilities"]}
    preds[s] = (yt, arms, eta)

    row = {"eta": eta}
    for k, prob in arms.items():
        row[k] = wf1(yt, prob.argmax(1))[0]
    report["seeds"][s] = row

# ---- 池化（三种子等权平均 WF1，与 SUMMARY 口径一致）----
pooled = {}
for k in ["MHnoU", "cRBEF", "equal_weight", "probabilities"]:
    vals = [report["seeds"][s][k] for s in SEEDS]
    pooled[k] = float(np.mean(vals))
report["pooled"]["wf1"] = pooled

# ---- 逐类（池化三种子平均）----
per_class = {}
for k in ["MHnoU", "cRBEF", "equal_weight", "probabilities"]:
    allf1 = np.array([wf1(preds[s][0], preds[s][1][k].argmax(1))[1] for s in SEEDS])
    per_class[k] = allf1.mean(0).tolist()
report["pooled"]["per_class_f1"] = {k: dict(zip(CLASSES, v)) for k, v in per_class.items()}

# ---- 样本级归因（用 seed 43 代表，三种子分别算再汇总）----
flow = {"crbef_ok": 0, "crbef_ok_mh_bad": 0,
        "crbef_ok_mh_bad_eq_kept": 0, "crbef_ok_mh_bad_rc_kept": 0,
        "crbef_ok_mh_bad_rc_lost_but_eq_kept": 0,
        "mh_conf_on_lost": [], "cr_conf_on_lost": [], "mh_conf_on_kept": [], "cr_conf_on_kept": []}
oracle_gain = []
lost_by_class = np.zeros(7)
crbef_wins_by_class = np.zeros(7)
for s in SEEDS:
    yt, arms, eta = preds[s]
    cr, mh, eq, rc = (arms["cRBEF"].argmax(1), arms["MHnoU"].argmax(1),
                      arms["equal_weight"].argmax(1), arms["probabilities"].argmax(1))
    cr_ok = cr == yt
    mh_ok = mh == yt
    eq_ok = eq == yt
    rc_ok = rc == yt
    flow["crbef_ok"] += int(cr_ok.sum())
    both = cr_ok & ~mh_ok
    flow["crbef_ok_mh_bad"] += int(both.sum())
    kept_eq = both & eq_ok
    kept_rc = both & rc_ok
    flow["crbef_ok_mh_bad_eq_kept"] += int(kept_eq.sum())
    flow["crbef_ok_mh_bad_rc_kept"] += int(kept_rc.sum())
    flow["crbef_ok_mh_bad_rc_lost_but_eq_kept"] += int((kept_eq & ~rc_ok).sum())
    lost = both & ~rc_ok
    for i in np.where(lost)[0]:
        lost_by_class[yt[i]] += 1
        flow["mh_conf_on_lost"].append(float(arms["MHnoU"][i].max()))
        flow["cr_conf_on_lost"].append(float(arms["cRBEF"][i][cr[i]]))
    for i in np.where(kept_rc)[0]:
        flow["mh_conf_on_kept"].append(float(arms["MHnoU"][i].max()))
        flow["cr_conf_on_kept"].append(float(arms["cRBEF"][i][cr[i]]))
    for c in range(7):
        crbef_wins_by_class[c] += int((cr_ok & (yt == c)).sum())
    # oracle：逐样本选 cRBEF/MHnoU 中置信度高者的预测
    conf_cr = arms["cRBEF"].max(1)
    conf_mh = arms["MHnoU"].max(1)
    oracle = np.where(conf_cr >= conf_mh, cr, mh)
    oracle_gain.append(wf1(yt, oracle)[0])

flow = {k: (float(np.mean(v)) if isinstance(v, list) and v and isinstance(v[0], float) else v)
        for k, v in flow.items()}
report["pooled"]["flow"] = flow
report["pooled"]["lost_by_class"] = dict(zip(CLASSES, lost_by_class.tolist()))
report["pooled"]["crbef_correct_by_class"] = dict(zip(CLASSES, crbef_wins_by_class.tolist()))
report["pooled"]["oracle_pick_wf1_mean"] = float(np.mean(oracle_gain))
report["pooled"]["wf1_gap_rc_minus_crbef"] = pooled["probabilities"] - pooled["cRBEF"]
report["pooled"]["wf1_gap_eq_minus_crbef"] = pooled["equal_weight"] - pooled["cRBEF"]

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crbef_vs_recomer_decomposition.json")
json.dump(report, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print("== 每 seed ==")
for s in SEEDS:
    r = report["seeds"][s]
    print(f"seed{s} eta={r['eta']}: cRBEF={r['cRBEF']:.4f} MHnoU={r['MHnoU']:.4f} "
          f"eq={r['equal_weight']:.4f} ReCoMER={r['probabilities']:.4f} "
          f"gap={r['probabilities']-r['cRBEF']:+.4f}")
print("== 池化 ==")
print({k: round(v, 4) for k, v in pooled.items()})
print("gap ReCoMER-cRBEF:", round(report["pooled"]["wf1_gap_rc_minus_crbef"], 4),
      "| eq-cRBEF:", round(report["pooled"]["wf1_gap_eq_minus_crbef"], 4))
print("oracle(conf pick):", round(report["pooled"]["oracle_pick_wf1_mean"], 4))
print("== 逐类 F1（池化）==")
for c in CLASSES:
    print(f"{c:8s} MHnoU={per_class['MHnoU'][CLASSES.index(c)]:.3f} "
          f"cRBEF={per_class['cRBEF'][CLASSES.index(c)]:.3f} "
          f"eq={per_class['equal_weight'][CLASSES.index(c)]:.3f} "
          f"ReCoMER={per_class['probabilities'][CLASSES.index(c)]:.3f}")
print("== cRBEF对/MHnoU错的样本流向（3种子合计）==")
f = flow
print(f"cRBEF 对总数={f['crbef_ok']}, 其中MHnoU错={f['crbef_ok_mh_bad']}")
print(f"  等权保住={f['crbef_ok_mh_bad_eq_kept']} ({f['crbef_ok_mh_bad_eq_kept']/f['crbef_ok_mh_bad']:.1%})",
      f" ReCoMER保住={f['crbef_ok_mh_bad_rc_kept']} ({f['crbef_ok_mh_bad_rc_kept']/f['crbef_ok_mh_bad']:.1%})",
      f" 其中等权保住但ReCoMER弄丢={f['crbef_ok_mh_bad_rc_lost_but_eq_kept']}")
print("丢失样本的类别:", report["pooled"]["lost_by_class"])
print(f"丢失样本置信度: MHnoU={f['mh_conf_on_lost']:.3f} vs cRBEF={f['cr_conf_on_lost']:.3f}")
print(f"保住样本置信度: MHnoU={f['mh_conf_on_kept']:.3f} vs cRBEF={f['cr_conf_on_kept']:.3f}")
print("saved", out)
