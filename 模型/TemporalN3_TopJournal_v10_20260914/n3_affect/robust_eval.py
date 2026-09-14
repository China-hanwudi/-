"""Robustness and efficiency evaluation for the v10 exploratory mainline.

Four modes, all inference-only (no training, no valid-based selection):

``missing``
    Keep only the named *current* modalities (features and mask) and report the
    degradation.  Seven subsets: T, A, V, TA, TV, AV, TAV.  Zeroing the encoded token is
    exactly the representation the model already uses for a missing modality
    (same convention as ``_ablated_current_logits``).
``noise``
    Add Gaussian noise proportional to each modality's own standard deviation.
    All-modality sweep plus an audio-only sweep (audio is the known weak stream
    at 25-d eGeMAPS / 74-d pooled MOSEI features).
``history``
    History-length sweep K = 0,1,2,3 by masking all but the K most recent
    canonical slots.
``efficiency``
    Trainable parameters, analytic FLOPs estimate (Linear + attention + GRUCell),
    median single-sample inference latency, throughput, and peak accelerator
    memory during a real training step.

Everything is written to ``robust_eval.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

DEFAULT_DATA = Path(os.environ.get("DATA_ROOT", "data"))


# ---------------------------------------------------------------------------
# FLOPs accounting
# ---------------------------------------------------------------------------
def analytic_flops(model: nn.Module, batch: dict, route_mode: str) -> dict:
    total = {"linear": 0, "attention": 0, "recurrent": 0}

    def linear_hook(mod: nn.Linear, inp, out):
        if not inp:
            return
        x = inp[0]
        n = x.numel() // max(mod.in_features, 1)
        total["linear"] += 2 * n * mod.in_features * mod.out_features

    def mha_hook(mod: nn.MultiheadAttention, inp, out):
        if not inp:
            return
        x = inp[0]
        if x.dim() != 3:
            return
        b, l, d = x.shape
        total["attention"] += 2 * b * l * d * 3 * d      # qkv projection
        total["attention"] += 2 * b * l * l * d * 2      # scores + weighted values
        total["attention"] += 2 * b * l * d * d          # output projection

    def gru_hook(mod: nn.GRUCell, inp, out):
        if not inp:
            return
        h = inp[0]
        n = h.numel() // max(mod.input_size, 1)
        total["recurrent"] += 2 * n * 3 * (mod.input_size * mod.hidden_size
                                           + mod.hidden_size * mod.hidden_size)

    handles = []
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(linear_hook))
        elif isinstance(mod, nn.MultiheadAttention):
            handles.append(mod.register_forward_hook(mha_hook))
        elif isinstance(mod, nn.GRUCell):
            handles.append(mod.register_forward_hook(gru_hook))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(batch, route_mode=route_mode)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    total["total"] = sum(total.values())
    total["unit"] = "FLOPs (2 x MAC) for one forward pass of the given batch"
    total["method"] = ("analytic estimate over nn.Linear, nn.MultiheadAttention and "
                       "nn.GRUCell call sites; embeddings and elementwise ops excluded")
    return total


# ---------------------------------------------------------------------------
# perturbation helpers
# ---------------------------------------------------------------------------
def zero_modalities(batch: dict, mods: tuple[str, ...]) -> dict:
    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    for name in mods:
        b[f"{name}_t"] = torch.zeros_like(b[f"{name}_t"])
        b[f"{name}_h"] = torch.zeros_like(b[f"{name}_h"])
    if "modality_mask" in b:
        for name in mods:
            b["modality_mask"][:, "TAV".index(name)] = 0.0
    if "history_modality_mask" in b:
        for name in mods:
            b["history_modality_mask"][:, :, "TAV".index(name)] = 0.0
    return b


def keep_modalities(batch: dict, kept: tuple[str, ...]) -> dict:
    """Return a batch in which exactly ``kept`` T/A/V streams remain."""
    unknown = set(kept) - set("TAV")
    if unknown or not kept:
        raise ValueError(f"kept must be a non-empty T/A/V subset, got {kept!r}")
    removed = tuple(name for name in "TAV" if name not in kept)
    return zero_modalities(batch, removed)


def add_noise(batch: dict, mods: tuple[str, ...], alpha: float, generator: torch.Generator) -> dict:
    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    for name in mods:
        for key in (f"{name}_t", f"{name}_h"):
            x = b[key]
            if not x.is_floating_point() or x.numel() == 0:
                continue
            # per-feature scale, so a 768-d text stream and a 25-d eGeMAPS stream
            # are perturbed by a comparable *relative* amount
            scale = x.std(dim=0, keepdim=True).clamp_min(1e-6)
            b[key] = x + alpha * scale * torch.randn(x.shape, generator=generator,
                                                     device=x.device, dtype=x.dtype)
    return b


def truncate_history(batch: dict, keep: int) -> dict:
    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    hm = b.get("history_mask")
    if hm is None:
        return b
    slots = hm.size(1)
    if keep >= slots:
        return b
    if keep <= 0:
        b["history_mask"] = torch.zeros_like(hm)
        for name in "TAV":
            b[f"{name}_h"] = torch.zeros_like(b[f"{name}_h"])
        if "history_modality_mask" in b:
            b["history_modality_mask"] = torch.zeros_like(b["history_modality_mask"])
        return b
    # canonical order is oldest -> newest, right aligned; the K most recent live
    # in the rightmost K slots.
    cut = slots - keep
    b["history_mask"] = torch.cat([torch.zeros_like(hm[:, :cut]), hm[:, cut:]], dim=1)
    if "history_modality_mask" in b:
        hmm = b["history_modality_mask"]
        b["history_modality_mask"] = torch.cat(
            [torch.zeros_like(hmm[:, :cut]), hmm[:, cut:]], dim=1)
    return b


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def score(model, split_obj, batch_fn, n, batch_size, device, route_mode,
          transform=None, seed: int = 0) -> dict:
    from n3_affect import metrics as M
    model.eval()
    model.current_epoch = 10 ** 6
    # The generator must live on the same device as the tensors it feeds:
    # a CPU generator feeding CUDA tensors raises
    # "Expected a 'cuda' device type for generator but found 'cpu'".
    gen = torch.Generator(device=torch.device(device)).manual_seed(seed)
    preds, trues, logits = [], [], []
    for start in range(0, n, batch_size):
        ix = torch.arange(start, min(start + batch_size, n))
        batch, y = batch_fn(ix)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        if transform is not None:
            batch = transform(batch, gen)
        out = model(batch, route_mode=route_mode)
        if "logits" in out:
            logits.append(out["logits"].detach().float().cpu().numpy())
            preds.append(out["logits"].argmax(-1).detach().cpu().numpy())
        else:
            preds.append(out["prediction"].detach().float().reshape(-1).cpu().numpy())
        trues.append(y.detach().float().reshape(-1).cpu().numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    if logits:
        return M.classification_report(np.concatenate(logits), true.astype(np.int64), 7,
                                       class_names=("Happy", "Neutral", "Sad", "Disgust",
                                                    "Anger", "Fear", "Surprise"))
    return M.regression_report(true, pred, clip=(-1.0, 1.0) if split_obj.__class__.__name__ == "ChSimsPacked" else None)


def condense(rep: dict) -> dict:
    """Headline numbers only, for compact robustness tables."""
    keys = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc",
            "mae", "rmse", "pearson", "ccc", "r2")
    out = {k: rep[k] for k in keys if k in rep}
    if "binary_gt0" in rep:
        out["acc2_gt0"] = rep["binary_gt0"]["acc2"]
        out["f1_2_gt0"] = rep["binary_gt0"]["f1_2_positive"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Robustness + efficiency evaluation")
    ap.add_argument("--dataset", required=True, choices=["m3ed", "mosei", "chsims"])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--route-mode", default="hard-safe")
    ap.add_argument("--modes", default="missing,noise,history,efficiency")
    ap.add_argument("--noise-levels", default="0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--latency-batches", type=int, default=20)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(args.source.resolve()))
    from n3_affect.eval_full import DATASET_LAYOUT, build_model, build_split, sha256_file

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    path = args.data / DATASET_LAYOUT[args.dataset][args.split]
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, cfg, task = build_model(args.dataset, ckpt)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    split_obj, batch_fn, _ = build_split(args.dataset, path)
    n = min(split_obj.n, 2048)  # robustness sweeps are comparative; cap the cost
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    levels = [float(x) for x in args.noise_levels.split(",") if x.strip()]

    out: dict = {
        "dataset": args.dataset, "split": args.split, "route_mode": args.route_mode,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_seed": ckpt.get("seed"), "checkpoint_epoch": ckpt.get("epoch"),
        "device": str(device), "n_evaluated": int(n), "task": task,
    }

    t0 = time.time()
    base = score(model, split_obj, batch_fn, n, args.batch_size, device, args.route_mode)
    out["baseline"] = base
    out["baseline_headline"] = condense(base)

    if "missing" in modes:
        out["missing_modality"] = {}
        for mods in (("T",), ("A",), ("V",), ("T", "A"), ("T", "V"), ("A", "V"), ("T", "A", "V")):
            rep = score(model, split_obj, batch_fn, n, args.batch_size, device,
                        args.route_mode, transform=lambda b, g, m=mods: keep_modalities(b, m))
            out["missing_modality"]["+".join(mods)] = {"headline": condense(rep)}

    if "noise" in modes:
        out["modality_noise"] = {"all_modalities": {}, "audio_only": {}}
        for a in levels:
            rep = score(model, split_obj, batch_fn, n, args.batch_size, device, args.route_mode,
                        transform=lambda b, g, aa=a: add_noise(b, ("T", "A", "V"), aa, g))
            out["modality_noise"]["all_modalities"][f"alpha={a:g}"] = {"headline": condense(rep)}
        for a in levels:
            rep = score(model, split_obj, batch_fn, n, args.batch_size, device, args.route_mode,
                        transform=lambda b, g, aa=a: add_noise(b, ("A",), aa, g))
            out["modality_noise"]["audio_only"][f"alpha={a:g}"] = {"headline": condense(rep)}

    if "history" in modes:
        out["history_length_K"] = {}
        for k in (0, 1, 2, 3):
            rep = score(model, split_obj, batch_fn, n, args.batch_size, device, args.route_mode,
                        transform=lambda b, g, kk=k: truncate_history(b, kk))
            out["history_length_K"][str(k)] = {"headline": condense(rep)}

    if "efficiency" in modes:
        ix = torch.arange(0, min(args.batch_size, split_obj.n))
        probe_batch, _ = batch_fn(ix)
        probe_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in probe_batch.items()}
        flops = analytic_flops(model, probe_batch, args.route_mode)
        # latency: median over single-batch forwards
        lat = []
        for i in range(args.latency_batches):
            s = (i * args.batch_size) % max(split_obj.n - args.batch_size, 1)
            b, _ = batch_fn(torch.arange(s, s + args.batch_size))
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.no_grad():
                model(b, route_mode=args.route_mode)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t) / float(b["T_t"].size(0)))
        lat = np.asarray(lat)
        # peak memory of one real training step
        peak_train_mib = None
        rss_delta_mib = None
        try:
            import resource
            rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            rss0 = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        bb, yy = batch_fn(torch.arange(0, min(args.batch_size, split_obj.n)))
        bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in bb.items()}
        yy = yy.to(device)
        try:
            if task == "classification":
                from n3_affect.losses import n3_total_loss
                o = model(bb, route_mode=args.route_mode)
                o["cf_measured_targets"] = model.measure_counterfactual_utility(bb, yy)
                loss = n3_total_loss(o, yy, cfg)["loss"]
            else:
                from n3_affect.regression_losses import n3_regression_loss
                o = model(bb, route_mode=args.route_mode)
                o["cf_measured_targets"] = model.measure_counterfactual_utility(bb, yy)
                loss = n3_regression_loss(o, yy, cfg)["loss"]
            loss.backward()
            if device.type == "cuda":
                peak_train_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        except Exception as exc:  # pragma: no cover - diagnostics only
            peak_train_mib = None
            out.setdefault("warnings", []).append(f"training-step memory probe failed: {exc}")
        finally:
            model.zero_grad(set_to_none=True)
            model.eval()
        if rss0 is not None:
            try:
                import resource
                rss_delta_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 - rss0
            except Exception:
                rss_delta_mib = None
        out["efficiency"] = {
            "trainable_parameters": model.count_trainable_parameters(),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "flops": flops,
            "batch_size_profiled": int(probe_batch["T_t"].size(0)),
            "latency_ms_per_sample_median": float(np.median(lat) * 1000.0),
            "latency_ms_per_sample_p90": float(np.quantile(lat, 0.9) * 1000.0),
            "throughput_samples_per_sec": float(1.0 / max(np.median(lat), 1e-9)),
            "peak_training_memory_mib": peak_train_mib,
            "rss_delta_mib_during_training_step": rss_delta_mib,
            "n_latency_batches": len(lat),
        }

    out["elapsed_sec"] = round(time.time() - t0, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "robust_eval.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("ROBUST_OK", json.dumps({"dataset": args.dataset,
                                   "baseline": out.get("baseline_headline"),
                                   "efficiency": out.get("efficiency")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
