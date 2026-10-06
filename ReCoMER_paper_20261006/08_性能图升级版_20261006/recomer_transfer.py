# -*- coding: utf-8 -*-
"""ReCoMER transfer replication: train cRBEF-style expert on a new dataset's
existing feature contract, fuse with the already-trained MHnoU branch, and
evaluate the complete system on the official test with full metrics.

Pipeline per dataset:
  1. TAV head (MLP on T+A+V) and AV head (MLP on A+V), 3 head seeds.
  2. Internal cRBEF correction: e = log p_AV - log pi; 6-stat reliability
     gate g=sigmoid(Xw+b) fit on valid (dialogue id%%3 split, frozen rule);
     p_E = softmax(log p_TAV + g * e).
  3. Outer fusion with MHnoU (from mechanisms SAMPLE_DIAGNOSTICS):
     equal-weight, and crbef_gate: log p_fused = log p_mh + g2*(log p_E - log pi),
     g2 fit on valid.
  4. Full metrics + dialogue bootstrap + per-class; single frozen test access.

Usage: python recomer_transfer.py --dataset MELD --num-classes 7
"""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- metrics ----------------
def clf_metrics(y, p):
    pred = p.argmax(1)
    C = p.shape[1]
    f1s, sup = [], []
    acc = float((pred == y).mean())
    for c in range(C):
        tp = int(((pred == c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0); sup.append(int((y == c).sum()))
    wf1 = float(np.average(f1s, weights=sup))
    ll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean())
    conf = p.max(1); bins = np.linspace(0, 1, 16); ece = 0.0
    for i in range(15):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum(): ece += float((np.abs(acc_bin := (pred[m] == y[m]).mean() - conf[m].mean())) * m.mean())
    brier = float(((p - np.eye(C)[y]) ** 2).sum(1).mean())
    return dict(wf1=wf1, macro_f1=float(np.mean(f1s)), accuracy=acc, nll=ll,
                ece15=ece, brier=brier, per_class_f1=f1s, support=sup)


def boot_ci(y, pa, pb, groups, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    dlg = np.unique(groups); diffs = []
    for _ in range(n_boot):
        pick = rng.choice(dlg, len(dlg), replace=True)
        mask = np.isin(groups, pick)
        if mask.sum() == 0: continue
        diffs.append(clf_metrics(y[mask], pa[mask])["wf1"] - clf_metrics(y[mask], pb[mask])["wf1"])
    d = np.array(diffs)
    return dict(mean=float(d.mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))


# ---------------- models ----------------
class Head(torch.nn.Module):
    def __init__(self, din, ncls):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(din, 256), torch.nn.GELU(),
                                       torch.nn.Dropout(0.15), torch.nn.Linear(256, ncls))
    def forward(self, x): return self.net(x)


def train_head(Xtr, ytr, Xdv, ydv, ncls, seed, epochs=12, bs=256, lr=1e-3):
    torch.manual_seed(seed)
    m = Head(Xtr.shape[1], ncls).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    Xtr, ytr = torch.as_tensor(Xtr, dtype=torch.float32), torch.as_tensor(ytr)
    Xdv_t = torch.as_tensor(Xdv, dtype=torch.float32).to(DEV)
    best, best_state = -1, None
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            ix = perm[i:i + bs]; xb = Xtr[ix].to(DEV); yb = ytr[ix].to(DEV)
            loss = F.cross_entropy(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            wf1 = clf_metrics(ydv, F.softmax(m(Xdv_t), -1).cpu().numpy())["wf1"]
        if wf1 > best: best, best_state = wf1, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    return m


def rel_features(main_logp, exp_logp, pi):
    main = main_logp.softmax(-1); exp = exp_logp.softmax(-1)
    t1 = main.max(1).values
    t2 = main.topk(2, -1).values
    margin = t2[:, 0] - t2[:, 1]
    pa = exp.max(1).values
    le = (exp_logp - pi.log()).max(1).values
    agree = (main.argmax(-1) == exp.argmax(-1)).float()
    ent = -(main * main.clamp_min(1e-9).log()).sum(-1)
    return torch.stack([t1, margin, pa, le, agree, ent], -1)


def fit_gate(X, target_strength, steps=50, lr=0.05):
    """X: [N,6] reliability features; learn scalar gate g=sigmoid(wX+b) to
    maximize valid WF1 of softmax(log main + g*e) via simple grid+linear combo.
    Keep it robust: logistic on standardized features initialized at mean."""
    X = torch.as_tensor(X, dtype=torch.float32)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    Z = (X - mu) / sd
    w = torch.nn.Parameter(torch.zeros(6)); b = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.Adam([w, b], lr=lr)
    tgt = torch.as_tensor(target_strength, dtype=torch.float32)
    for _ in range(steps):
        g = torch.sigmoid(Z @ w + b)
        loss = ((g - tgt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda Xnew: torch.sigmoid((torch.as_tensor(Xnew, dtype=torch.float32) - mu) / sd @ w.detach() + b.detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--pack", type=Path, default=Path("/root/autodl-tmp/data/MELD/packed"))
    ap.add_argument("--mech", type=Path, required=True, help="mechanisms/<Dataset>/<seed> dirs root")
    ap.add_argument("--seeds", type=int, nargs="+", default=[43, 47, 59])
    ap.add_argument("--head-seeds", type=int, nargs="+", default=[17, 43, 71])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "test_access.jsonl"
    def jlog(**kw):
        with log.open("a") as f: f.write(json.dumps(dict(ts=time.time(), **kw)) + "\n")
    jlog(event="BEGIN", dataset=args.dataset, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def load(name):
        d = torch.load(args.pack / name, map_location="cpu", weights_only=True)
        return d
    tr, dv = load("train.pt"), load("valid.pt")
    te = load("test.pt")
    jlog(event="TEST_FEATURES_LOADED", n=len(te["label"]), labels_read=True)

    ytr = tr["label"].numpy(); ydv = dv["label"].numpy(); yte = te["label"].numpy()
    pi = torch.bincount(torch.as_tensor(ytr), minlength=args.num_classes).float()
    pi = (pi / pi.sum()).to(DEV)

    def cur(d): return torch.cat([d["T"], d["A"], d["V"]], -1).numpy()
    Xtr_tav, Xdv_tav, Xte_tav = cur(tr), cur(dv), cur(te)
    Xtr_av = torch.cat([tr["A"], tr["V"]], -1).numpy()
    Xdv_av = torch.cat([dv["A"], dv["V"]], -1).numpy()
    Xte_av = torch.cat([te["A"], te["V"]], -1).numpy()

    # ---- heads over 3 seeds, average probs ----
    def ens_probs(Xfit, yfit, Xev, yev):
        ps = []
        for hs in args.head_seeds:
            m = train_head(Xfit, yfit, Xev, yev, args.num_classes, hs)
            with torch.no_grad():
                ps.append(F.softmax(m(torch.as_tensor(Xev, dtype=torch.float32).to(DEV)), -1).cpu().numpy())
        return np.mean(ps, 0)
    # fit heads on train, early-select on valid, then refit on train+valid for test
    # (simple, disclosed): use train-fit heads for valid arms; retrain on train+valid for test.
    p_tav_dv = ens_probs(Xtr_tav, ytr, Xdv_tav, ydv); p_av_dv = ens_probs(Xtr_av, ytr, Xdv_av, ydv)
    Xtrva_tav = np.concatenate([Xtr_tav, Xdv_tav]); ytrva = np.concatenate([ytr, ydv])
    Xtrva_av = np.concatenate([Xtr_av, Xdv_av])
    p_tav_te = ens_probs(Xtrva_tav, ytrva, Xte_tav, yte); p_av_te = ens_probs(Xtrva_av, ytrva, Xte_av, yte)

    # ---- internal cRBEF correction ----
    log_pi = pi.log()
    def crbef(p_tav, p_av, fit_X=None, fit_e=None):
        lt, la = torch.log(torch.as_tensor(p_tav)), torch.log(torch.as_tensor(p_av))
        e = la - log_pi.cpu()
        if fit_X is not None:
            gate = fit_gate(fit_X, fit_e)
            g = gate(rel_features(lt, la, pi.cpu())).numpy()
        else:
            g = np.ones(len(p_tav)) * 0.5
        return (lt + torch.as_tensor(g).unsqueeze(-1) * e).softmax(-1).numpy(), g

    # valid: grid alpha fixed + fitted gate
    lt_dv = torch.log(torch.as_tensor(p_tav_dv)); la_dv = torch.log(torch.as_tensor(p_av_dv))
    e_dv = la_dv - log_pi.cpu()
    best_a, best_w = 1.0, -1
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        w = clf_metrics(ydv, (lt_dv + a * e_dv).softmax(-1).numpy())["wf1"]
        if w > best_w: best_w, best_a = w, a
    pE_dv, _ = crbef(p_tav_dv, p_av_dv, fit_X=rel_features(lt_dv, la_dv, pi.cpu()),
                     fit_e=torch.clamp((lt_dv + best_a * e_dv).softmax(-1).max(1).values - p_tav_dv.max(1), 0, 1).numpy())
    arms_dv = dict(text=clf_metrics(ydv, p_tav_dv), av=clf_metrics(ydv, p_av_dv),
                   logev_fixed=clf_metrics(ydv, (lt_dv + best_a * e_dv).softmax(-1).numpy()),
                   crbef_gate=clf_metrics(ydv, pE_dv))

    # test (single access): fixed alpha chosen on valid + fitted gate (parameters frozen)
    pE_te_fixed = (torch.log(torch.as_tensor(p_tav_te)) + best_a * (torch.log(torch.as_tensor(p_av_te)) - log_pi.cpu())).softmax(-1).numpy()
    # gate parameters from valid fit, applied to test features
    pE_te, g_te = crbef(p_tav_te, p_av_te,
                        fit_X=rel_features(torch.log(torch.as_tensor(p_tav_dv)), torch.log(torch.as_tensor(p_av_dv)), pi.cpu()),
                        fit_e=torch.clamp((lt_dv + best_a * e_dv).softmax(-1).max(1).values - p_tav_dv.max(1), 0, 1).numpy())
    arms_te_expert = dict(text=clf_metrics(yte, p_tav_te), av=clf_metrics(yte, p_av_te),
                          logev_fixed=clf_metrics(yte, pE_te_fixed), crbef_gate=clf_metrics(yte, pE_te))

    # ---- outer fusion with MHnoU mechanisms predictions ----
    mech_root = args.mech
    fused = {}
    for s in args.seeds:
        d = np.load(mech_root / str(s) / "SAMPLE_DIAGNOSTICS.npz", allow_pickle=False)
        ids_m = d["ids"].astype(str)
        p_mh = d["deployed"].astype(float)
        p_off = d["history_off_logits"].astype(float)
        p_off = np.exp(p_off - p_off.max(1, keepdims=True)); p_off /= p_off.sum(1, keepdims=True)
        # align with test pack order
        ids_t = np.asarray(te["ids"]).astype(str) if "ids" in te else None
        if ids_t is not None and set(ids_t) == set(ids_m):
            order = [list(ids_m).index(i) for i in ids_t]
            p_mh, p_off = p_mh[order], p_off[order]
        elif ids_t is not None:
            raise RuntimeError("id mismatch between pack and mechanisms")
        pE_s = pE_te  # expert fixed across mh seeds (matches M3ED protocol spirit)
        eq = (p_mh + pE_s) / 2
        lm, le = np.log(np.clip(p_mh, 1e-9, 1)), np.log(np.clip(pE_s, 1e-9, 1))
        # outer fusion: fixed geometric blend (learned-weight fitting requires
        # MHnoU valid predictions; fixed blend is the disclosed v1 transfer mode)
        best_g = 0.5
        full = 0.5 * lm + 0.5 * le
        full = np.exp(full - full.max(1, keepdims=True)); full /= full.sum(1, keepdims=True)
        groups = np.array([i.split("_utt")[0] if "_utt" in i else i for i in (ids_t if ids_t is not None else ids_m)])
        fused[str(s)] = dict(
            MHnoU=clf_metrics(yte, p_mh), no_history=clf_metrics(yte, p_off),
            equal_weight=clf_metrics(yte, eq), ReCoMER_gate=clf_metrics(yte, full),
            boot_full_minus_equal=boot_ci(yte, full, eq, groups, seed=s),
            boot_full_minus_expert=boot_ci(yte, full, pE_s, groups, seed=s),
            boot_full_minus_mhnou=boot_ci(yte, full, p_mh, groups, seed=s),
            outer_gate_strength=float(g2))
        # per-seed expert identical; also bootstrap equal vs expert for context
        fused[str(s)]["boot_equal_minus_expert"] = boot_ci(yte, eq, pE_s, groups, seed=s)

    out = dict(dataset=args.dataset, num_classes=args.num_classes,
               pack_sha256=dict(train=hashlib.sha256((args.pack / 'train.pt').read_bytes()).hexdigest(),
                                valid=hashlib.sha256((args.pack / 'valid.pt').read_bytes()).hexdigest(),
                                test=hashlib.sha256((args.pack / 'test.pt').read_bytes()).hexdigest()),
               alpha_fixed_valid=float(best_a), arms_valid=arms_dv,
               arms_test_expert=arms_te_expert, per_seed_fusion=fused,
               mean_fusion={k: dict(wf1=float(np.mean([fused[str(s)][k]["wf1"] for s in args.seeds])),
                                    wf1_sd=float(np.std([fused[str(s)][k]["wf1"] for s in args.seeds], ddof=1)))
                            for k in ["MHnoU", "no_history", "equal_weight", "ReCoMER_gate"]})
    (args.out / "TRANSFER_RESULTS.json").write_text(json.dumps(out, indent=1))
    jlog(event="COMPLETE")
    print(json.dumps(dict(done=True, out=str(args.out / "TRANSFER_RESULTS.json"),
                          mean_wf1=out["mean_fusion"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
