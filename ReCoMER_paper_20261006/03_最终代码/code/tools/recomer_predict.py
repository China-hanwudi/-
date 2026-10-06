"""Export aligned complete-model inputs, or infer using a trained ReCoMER.

Run from code/: python -m tools.recomer_predict --help
Original CR features and their cached predictions are explicit alternatives.
No labels enter either model. Training pipelines use only train.pt/valid.pt.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from n6.crbef_expert import CRBEFFeatureStore
from n6.data import open_split
from n6.recomer import ReCoMER, combine_inputs, mhnou_reference_outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", choices=("train", "valid"), default="train")
    ap.add_argument("--mhnou", type=Path)
    ap.add_argument("--expert-assets", type=Path, default=Path("crbef_assets/m3ed"))
    ap.add_argument("--fusion", type=Path)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--expert-cache", type=Path, help="audited cRBEF bank with ids and P[:,0/2]")
    ap.add_argument("--cr-features", type=Path)
    ap.add_argument("--rich-features", type=Path)
    ap.add_argument("--ids-json", type=Path, help="ordered ID list for one training/development role")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit("refusing to overwrite " + str(args.out))
    if args.bundle and (args.mhnou or args.fusion):
        raise SystemExit("choose --bundle or separate model checkpoints")
    if not args.bundle and not args.mhnou:
        raise SystemExit("provide --mhnou or --bundle")
    if args.expert_cache and (args.cr_features or args.rich_features):
        raise SystemExit("choose cached predictions or original feature files")
    model = (ReCoMER.from_bundle(args.bundle, args.device) if args.bundle else
             ReCoMER.from_files(args.mhnou, args.expert_assets, args.fusion, args.device))
    ds = open_split(args.data, args.split + ".pt")
    all_ids = [str(v) for v in ds.raw["ids"]]
    lookup = CRBEFFeatureStore._lookup(all_ids, "MHnoU pack")
    ids = json.loads(args.ids_json.read_text(encoding="utf-8")) if args.ids_json else all_ids
    ids = [str(v) for v in ids]
    if not ids or len(set(ids)) != len(ids) or any(v not in lookup for v in ids):
        raise SystemExit("requested IDs must be unique and present in the MHnoU split")
    cache = None
    if args.expert_cache:
        with np.load(args.expert_cache, allow_pickle=False) as f:
            cache = {k: f[k].copy() for k in ("ids", "P")}
        if cache["P"].shape != (len(cache["ids"]), 4, 7):
            raise SystemExit("expert cache must use the supplied M3ED P=[CR,MH,TAV,nohist] contract")
        ci = CRBEFFeatureStore._lookup(cache["ids"], "expert cache")
        if any(v not in ci for v in ids):
            raise SystemExit("expert cache misses requested IDs")
        p = cache["P"][:, [0, 2]]
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1, atol=2e-4):
            raise SystemExit("invalid cached expert probabilities")
    else:
        if not args.cr_features or not args.rich_features:
            raise SystemExit("provide --expert-cache or both original feature files")
        store = CRBEFFeatureStore(args.cr_features, args.rich_features)
    rows = {k: [] for k in ("P", "C", "L0", "Lh", "M")}
    fused = []
    for s in range(0, len(ids), args.batch_size):
        current = ids[s:s + args.batch_size]
        b, _ = ds.batch([lookup[v] for v in current])
        b = {k: v.to(args.device) for k, v in b.items()}
        if cache is None:
            out = model.build_inputs(b, store.batch(current))
        else:
            mh = mhnou_reference_outputs(model.mhnou, b)
            cp = torch.as_tensor(cache["P"][[ci[v] for v in current]], device=args.device)
            P, C = combine_inputs(mh, dict(CRBEF=cp[:, 0], TAV=cp[:, 2]))
            out = dict(P=P, **mh)
        for k in rows:
            rows[k].append(out[k].cpu().numpy())
        if model.fusion is not None:
            with torch.no_grad():
                fused.append(model.fusion(out["P"], out["C"]).exp().cpu().numpy())
    values = {k: np.concatenate(v) for k, v in rows.items()}
    values.update(ids=np.asarray(ids), class_names=np.asarray(model.expert.class_names),
                  source=np.asarray([v.split("_")[1] for v in ids]),
                  y_true=ds.raw["label"][[lookup[v] for v in ids]].numpy())
    if fused:
        values["fused"] = np.concatenate(fused)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **values)
    receipt = dict(n=len(ids), mhnou=str(args.mhnou), bundle=str(args.bundle), split=args.split,
                   class_names=list(model.expert.class_names), expert_cache=str(args.expert_cache),
                   original_features_used=cache is None, fused=model.fusion is not None,
                   labels_used_by_forward=False, no_history_reference="same MHnoU checkpoint",
                   old_uniform_h_used=False)
    args.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
