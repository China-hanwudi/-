"""Evaluation entry for model6 (valid split only).

Loads a run's ``best.pt``, rebuilds the model from the stored config and
evaluates the DEPLOYED output on the packed **valid** split (the sealed test
split is never referenced; ``n6.data.open_split`` refuses anything that is not
train.pt / valid.pt and this entry point only ever asks for valid.pt).

Flags:
* ``--oracle-utility`` -- additionally deploy the measured exact-Shapley
  weights: per-sample RAW (unstandardised) phi -> softmax(phi / tau) with
  the eps floor, applied to the SOLO_SUM path (always) and, for
  ``joint_softgate`` checkpoints, as an oracle token weighting of the joint
  path (valid-only analysis).
* ``--missing-modality-curves`` -- for each modality m and noise level
  p in {0.25, 0.5, 0.75, 1.0}, corrupt m in the valid features (``zero`` the
  whole vector for a p-fraction of samples, and ``noise`` = add Gaussian
  noise of std p * per-feature std), re-evaluate the deployed metrics, and
  emit a JSON table.  Pass ``--ckpt-alt`` to compare a second checkpoint
  (e.g. a u0-uniform vs a u2-shapley run) in the same table.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import M6Config
from .data import open_final_test, open_split
from .model import UGFModel, MODALITIES
from .train import evaluate

MMC_LEVELS = (0.25, 0.5, 0.75, 1.0)


def load_checkpoint(ckpt_path, device):
    """Load best.pt, transparently upgrading legacy single-head checkpoints.

    Round-1/2 checkpoints store the utility head as ``utility_head.net.3``
    (single [3,d] layer); iteration-3 code builds a (mu, sigma) double head.
    The shim remaps ``net.3`` into ``head_mu`` and keeps the zero-initialised
    ``head_sigma`` (zero sigma => deploy weights bit-identical to the old
    softmax(mu/tau) behaviour).  Unrecognised layouts are refused with a
    clear message.
    """
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = M6Config.from_dict(ck["cfg"])
    model = UGFModel(cfg)
    sd = ck["model"]
    if model.utility_head is not None:
        if cfg.gate_architecture in ("evidence", "constant"):
            # Revised Innovation-2 routers have their own explicit layouts;
            # do not run the legacy UtilityHead shim on them.
            model.load_state_dict(sd, strict=True)
        else:
            has_v2 = any(k.startswith("utility_head.head_mu.") for k in sd)
            has_v1 = any(k.startswith("utility_head.net.3.") for k in sd)
            if has_v2:
                model.load_state_dict(sd, strict=True)
            elif has_v1:
                remapped = {}
                for k, v in sd.items():
                    if k.startswith("utility_head.net.3."):
                        remapped["utility_head.head_mu."
                                 + k[len("utility_head.net.3."):]] = v
                    else:
                        remapped[k] = v
                missing, unexpected = model.load_state_dict(remapped, strict=False)
                allowed = {"utility_head.head_sigma.weight",
                           "utility_head.head_sigma.bias"}
                if set(missing) != allowed or unexpected:
                    raise ValueError(
                        "legacy utility-head shim failed for %s: missing=%s "
                        "unexpected=%s" % (ckpt_path, missing, unexpected))
                cfg.utility_head_version = 1   # record the upgrade for the report
            else:
                raise ValueError(
                    "unrecognised UtilityHead layout in %s: found neither "
                    "utility_head.head_mu.* (v2) nor utility_head.net.3.* (v1) "
                    "keys" % ckpt_path)
    else:
        model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model, cfg, ck


@torch.no_grad()
def evaluate_oracle(model, ds, device, bs: int, cfg: M6Config) -> Dict:
    """Deploy the measured exact-Shapley weights (valid-only analysis).

    Per valid batch the RAW (unstandardised) per-sample Shapley vector
    ``phi [B,3]`` is measured; the deployed weights are ``softmax(phi / tau)``
    with the eps floor and renormalisation applied per sample.  Two paths are
    reported: ``solo_sum`` (the weights applied to the solo logits, always)
    and, for ``joint_softgate`` checkpoints, ``joint_softgate`` (the same
    weights scale the tokens into the joint head -- oracle token weighting).
    """
    from .datasetspec import class_names_for
    from .metrics import classification_report, regression_report

    def report_of(logits_list):
        logits = np.concatenate(logits_list, axis=0)
        if cfg.task == "cls":
            return classification_report(
                logits, ys, cfg.num_classes,
                class_names_for(cfg, getattr(ds, "path", None)))
        return regression_report(np.asarray(ys), logits.reshape(-1))

    was_training = model.training
    model.eval()
    ys: List[float] = []
    solo_logits_all: List[np.ndarray] = []
    joint_logits_all: List[np.ndarray] = []
    w_sum = torch.zeros(3, dtype=torch.float64)
    w_count = 0
    try:
        for s in range(0, ds.n, bs):
            b, y = ds.batch(range(s, min(s + bs, ds.n)))
            b = {k: v.to(device) for k, v in b.items()}
            y = y.to(device)
            out = model(b)
            phi = model.measure_shapley(b, y, standardize=False)   # [B,3] raw
            w = torch.softmax(phi / cfg.tau, dim=-1)
            w = w.clamp_min(cfg.eps_floor)
            w = w / w.sum(dim=-1, keepdim=True)
            w_sum += w.sum(dim=0).detach().cpu().double()
            w_count += int(w.shape[0])
            solo_dep = (w.unsqueeze(-1).to(out["solo_stack"].dtype)
                        * out["solo_stack"]).sum(dim=1)
            solo_logits_all.append(solo_dep.detach().float().cpu().numpy())
            if cfg.deploy in ("joint_softgate", "closed_loop"):
                toks, msk = model.build_tokens(
                    out["cur_embs"], out["hist_embs"], b, token_weight=w,
                    include_history_tokens=(cfg.deploy != "closed_loop"))
                joint_dep = model.joint(toks, msk)
                joint_logits_all.append(
                    joint_dep.detach().float().cpu().numpy())
            ys.extend(y.cpu().tolist())
    finally:
        if was_training:
            model.train()
    block: Dict[str, object] = {
        "solo_sum": report_of(solo_logits_all),
        "mean_oracle_weights": (w_sum / max(w_count, 1)).tolist(),
    }
    if cfg.deploy in ("joint_softgate", "closed_loop"):
        block["joint_softgate"] = report_of(joint_logits_all)
    return block


@torch.no_grad()
def evaluate_corrupted(model, ds, device, bs: int, cfg: M6Config,
                       modality: str, p: float, mode: str,
                       generator: torch.Generator) -> Dict:
    """Re-evaluate the deployed output after corrupting one valid modality."""
    key = {"T": "T", "A": "A", "V": "V"}[modality]
    orig = ds.raw[key]
    if mode == "zero":
        drop = torch.rand(ds.n, generator=generator) < p
        corrupted = orig.clone()
        corrupted[drop] = 0.0
    elif mode == "noise":
        std = orig.std(dim=0, keepdim=True)
        noise = torch.randn(orig.shape, generator=generator) * (p * std)
        corrupted = orig + noise
    else:
        raise ValueError(mode)
    ds.raw[key] = corrupted
    try:
        m, _ = evaluate(model, ds, device, bs, cfg)
    finally:
        ds.raw[key] = orig
    return m


def run_report(ckpt_path, args, device) -> Dict:
    model, cfg, ck = load_checkpoint(ckpt_path, device)
    tag = "u_" + cfg.utility
    valid = (open_split(args.data, "valid.pt", task=cfg.task)
             if args.split == "valid" else
             open_final_test(args.data, task=cfg.task))
    bs = args.batch_size
    base, _ = evaluate(model, valid, device, bs, cfg)
    report: Dict[str, object] = {
        "ckpt": str(ckpt_path),
        "ckpt_tag": tag,
        "split": args.split,
        "n": valid.n,
        "seed": ck.get("seed"),
        "epoch": ck.get("epoch"),
        "candidate_kind": ck.get("candidate_kind"),
        "task": cfg.task,
        "utility_mode": cfg.utility,
        "utility_head_version": cfg.utility_head_version,
        "modules": {"m1": bool(cfg.use_m1), "m2": bool(cfg.use_m2),
                    "m3": bool(cfg.use_m3)},
        "metrics": base,
    }
    if args.oracle_utility:
        report["oracle_utility"] = evaluate_oracle(model, valid, device, bs, cfg)
    if args.missing_modality_curves:
        rows: List[Dict[str, object]] = [{
            "ckpt_tag": tag, "mode": "clean", "modality": None, "p": 0.0,
            "metrics": base,
        }]
        gen = torch.Generator().manual_seed(20260919)
        for mode in ("zero", "noise"):
            for modality in MODALITIES:
                for p in MMC_LEVELS:
                    m = evaluate_corrupted(model, valid, device, bs, cfg,
                                           modality, p, mode, gen)
                    rows.append({"ckpt_tag": tag, "mode": mode,
                                 "modality": modality, "p": p, "metrics": m})
        report["missing_modality_curves"] = rows
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--ckpt-alt", type=Path, default=None,
                    help="optional second checkpoint for the u0-vs-u2 curve table")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--split", choices=["valid", "test"], default="valid",
                    help="valid for model selection; test only for final ceremony")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--oracle-utility", action="store_true")
    ap.add_argument("--missing-modality-curves", action="store_true")
    ap.add_argument("--out", type=Path, default=None,
                    help="optional path to also write the JSON report")
    args = ap.parse_args(argv)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device))

    t0 = time.time()
    report = run_report(args.ckpt, args, device)
    if args.ckpt_alt is not None:
        report = {"primary": report, "alt": run_report(args.ckpt_alt, args, device)}
    report["elapsed_sec"] = round(time.time() - t0, 1)
    report["test_read"] = args.split == "test"
    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
