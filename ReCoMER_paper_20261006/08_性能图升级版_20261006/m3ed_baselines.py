# -*- coding: utf-8 -*-
"""Classic ERC baselines on the M3ED_textQwen official-split pack, for a
per-class comparison against ReCoMER on M3ED. Models: utterance MLP and a
DialogueRNN reimplementation (dyadic). Test accessed once per model family.

Output: /root/baselines_m3ed_20261006/baseline_results.json
"""
import hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PACK = Path("/root/autodl-tmp/data/M3ED_textQwen/packed")
OUT = Path("/root/baselines_m3ed_20261006")
OUT.mkdir(exist_ok=True)
NCLS = 7
NAMES = ["Happy", "Neutral", "Sad", "Disgust", "Anger", "Fear", "Surprise"]
SEEDS = [17, 43, 71]


def jlog(**kw):
    with (OUT / "test_access.jsonl").open("a") as f:
        f.write(json.dumps(dict(ts=time.time(), **kw)) + "\n")


def clf_metrics(y, p):
    pred = p.argmax(1)
    f1s, sup = [], []
    for c in range(NCLS):
        tp = int(((pred == c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0); sup.append(int((y == c).sum()))
    return dict(wf1=float(np.average(f1s, weights=sup)), macro_f1=float(np.mean(f1s)),
                accuracy=float((pred == y).mean()), per_class_f1=f1s, support=sup)


def load(sp):
    d = torch.load(PACK / f"{sp}.pt", map_location="cpu", weights_only=True)
    X = torch.cat([d["T"], d["A"], d["V"]], -1).numpy().astype(np.float32)
    y = d["label"].numpy()
    ids = np.asarray(d["ids"]).astype(str)
    spk = np.array([0 if i.startswith("A_") else 1 for i in ids])
    dlg = np.array(["_".join(i.split("_")[1:3]) for i in ids])
    return X, y, ids, spk, dlg


Xtr, ytr, _, spk_tr, dlg_tr = load("train")
Xdv, ydv, _, spk_dv, dlg_dv = load("valid")
Xte, yte, ids_te, spk_te, dlg_te = load("test")
# standardize features with train statistics (raw hidden-state scales vary widely)
mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
Xtr = (Xtr - mu) / sd; Xdv = (Xdv - mu) / sd; Xte = (Xte - mu) / sd
jlog(event="TEST_LOADED", n=len(yte), pack_sha256=hashlib.sha256((PACK / "test.pt").read_bytes()).hexdigest())


# ---------------- utterance MLP ----------------
def mlp_run(seed):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.GELU(), nn.Dropout(0.15),
                      nn.Linear(256, NCLS)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    Xt = torch.as_tensor(Xtr); yt = torch.as_tensor(ytr)
    Xv = torch.as_tensor(Xdv).to(DEV)
    best, state = -1, None
    for ep in range(15):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            ix = perm[i:i + 256]
            loss = F.cross_entropy(m(Xt[ix].to(DEV)), yt[ix].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            w = clf_metrics(ydv, F.softmax(m(Xv), -1).cpu().numpy())["wf1"]
        if w > best: best, state = w, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        p = F.softmax(m(torch.as_tensor(Xte).to(DEV)), -1).cpu().numpy()
    return clf_metrics(yte, p)


# ---------------- DialogueRNN (dyadic reimplementation) ----------------
class DialogueRNN(nn.Module):
    def __init__(self, din, d=100):
        super().__init__()
        self.proj = nn.Linear(din, d)
        self.ctx = nn.GRUCell(d, d)
        self.party = nn.ModuleList([nn.GRUCell(d, d), nn.GRUCell(d, d)])
        self.glob = nn.GRUCell(3 * d, d)
        self.emo = nn.GRUCell(d, d)
        self.att = nn.Linear(2 * d, d)
        self.out = nn.Linear(d, NCLS)

    def forward(self, xs, speakers):
        # xs: [L, din]; speakers: [L] in {0,1}; sequential over the dialogue
        d = self.proj(xs)
        c = torch.zeros(d.shape[1], device=xs.device)
        p = [torch.zeros(d.shape[1], device=xs.device) for _ in range(2)]
        e = torch.zeros(d.shape[1], device=xs.device)
        g_list = []
        outs = []
        for t in range(d.shape[0]):
            c = self.ctx(d[t], c)
            s = int(speakers[t])
            p[s] = self.party[s](d[t], p[s])
            g = self.glob(torch.cat([c, p[s], e]), g_list[-1] if g_list else torch.zeros_like(c))
            # attention over previous global states
            if g_list:
                G = torch.stack(g_list)
                a = torch.softmax((self.att(torch.cat([g.expand(len(g_list), -1), G], -1)) * g).sum(-1), 0)
                ctx = (a.unsqueeze(-1) * G).sum(0)
            else:
                ctx = g
            e = self.emo(ctx, e)
            outs.append(self.out(e))
            g_list.append(g)
        return torch.stack(outs)


def dialogue_run(seed, epochs=25, lr=1e-4, bs=8):
    torch.manual_seed(seed); np.random.seed(seed)
    m = DialogueRNN(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    tr_dl = sorted(set(dlg_tr)); dv_dl = sorted(set(dlg_dv))
    def batch(dl, X, y, spk, dlg, idxs):
        seqs, labs, spks, lens = [], [], [], []
        for i in idxs:
            msk = np.where(dlg == dl[i])[0]
            seqs.append(torch.as_tensor(X[msk])); labs.append(y[msk])
            spks.append(spk[msk]); lens.append(len(msk))
        L = max(lens)
        Xb = torch.zeros(len(idxs), L, X.shape[1]); Yb = np.full((len(idxs), L), -1)
        Sb = np.zeros((len(idxs), L), dtype=int)
        for r in range(len(idxs)):
            Xb[r, :lens[r]] = seqs[r]; Yb[r, :lens[r]] = labs[r]; Sb[r, :lens[r]] = spks[r]
        return Xb.to(DEV), torch.as_tensor(Yb).to(DEV), Sb, lens
    best, state = -1, None
    for ep in range(epochs):
        m.train(); order = np.random.permutation(len(tr_dl))
        for i in range(0, len(order), bs):
            Xb, Yb, Sb, lens = batch(tr_dl, Xtr, ytr, spk_tr, dlg_tr, order[i:i + bs])
            opt.zero_grad(); loss = 0.0; nb = 0
            for r in range(Xb.shape[0]):
                L = lens[r]
                logits = m(Xb[r, :L], Sb[r, :L])
                loss = loss + F.cross_entropy(logits, Yb[r, :L]); nb += 1
            (loss / nb).backward(); opt.step()
        m.eval(); pv = []
        with torch.no_grad():
            for i in range(0, len(dv_dl), 16):
                Xb, Yb, Sb, lens = batch(dv_dl, Xdv, ydv, spk_dv, dlg_dv, np.arange(i, min(i + 16, len(dv_dl))))
                for r in range(Xb.shape[0]):
                    L = lens[r]
                    pv.append(F.softmax(m(Xb[r, :L], Sb[r, :L]), -1).cpu().numpy())
        pv = np.concatenate(pv)
        # valid ids order: concatenated dialogue by dialogue in dv_dl order
        msk = np.concatenate([np.where(dlg_dv == k)[0] for k in dv_dl])
        w = clf_metrics(ydv[msk], pv)["wf1"]
        if w > best: best, state = w, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); m.eval(); pt = []
    te_dl = sorted(set(dlg_te))
    with torch.no_grad():
        for i in range(0, len(te_dl), 16):
            Xb, Yb, Sb, lens = batch(te_dl, Xte, yte, spk_te, dlg_te, np.arange(i, min(i + 16, len(te_dl))))
            for r in range(Xb.shape[0]):
                L = lens[r]
                pt.append(F.softmax(m(Xb[r, :L], Sb[r, :L]), -1).cpu().numpy())
    pt = np.concatenate(pt)
    msk = np.concatenate([np.where(dlg_te == k)[0] for k in te_dl])
    return clf_metrics(yte[msk], pt)


if __name__ == "__main__":
    res = {}
    per_seed = {}
    for name, fn in [("MLP", mlp_run), ("DialogueRNN", dialogue_run)]:
        entries = []
        for s in SEEDS:
            t0 = time.time()
            entries.append(fn(s))
            print(name, s, "done", round(time.time() - t0), "s", entries[-1]["wf1"], flush=True)
        jlog(event="TEST_EVALUATED", model=name, seeds=SEEDS)
        arr = np.array([e["per_class_f1"] for e in entries])
        res[name] = dict(
            wf1_mean=float(np.mean([e["wf1"] for e in entries])),
            wf1_sd=float(np.std([e["wf1"] for e in entries], ddof=1)),
            acc_mean=float(np.mean([e["accuracy"] for e in entries])),
            per_class_mean=(arr.mean(0) * 100).round(2).tolist(),
            per_class_sd=(arr.std(0, ddof=1) * 100).round(2).tolist(),
            class_names=NAMES)
        per_seed[name] = entries
    (OUT / "baseline_results.json").write_text(json.dumps(
        dict(models=res, per_seed=per_seed, seeds=SEEDS,
             note="baselines trained on M3ED_textQwen pack (official M3ED split, Qwen-family features); "
                  "test accessed once per family with access log"), indent=1))
    print(json.dumps(res), flush=True)
