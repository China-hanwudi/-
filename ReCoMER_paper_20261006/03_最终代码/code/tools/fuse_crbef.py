"""Fuse an MHnoU prediction file with an external cRBEF-v2 prediction file.

Example (run from ``最终代码/code``)::

    python tools/fuse_crbef.py \
      --main runs/mhnou/valid_logits.npz \
      --crbef runs/crbef_v2/valid_predictions.npz \
      --out runs/fused/crbef_fused.npz \
      --num-classes 7 --fit-gate --save-gate runs/fused/gate.pt

Both files may contain ``logits`` or probabilities under ``probs``.  If
``ids`` are present they are used to align rows before fusion.  The gate is
trained on the supplied labels only when ``--fit-gate`` is requested; without
that flag the module uses its bounded half-strength initial gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from n6.crbef_fusion import (CRBEFFusionConfig, CRBEFExternalFusion,
                             LocalCurrentConfig, LocalCurrentFusion,
                             fit_gate, fit_local_current)
from n6.metrics import classification_report


def _array_key(z: np.lib.npyio.NpzFile, requested: Optional[str]) -> str:
    if requested:
        if requested not in z.files:
            raise SystemExit("key %r not found in %s" % (requested, z.fid))
        return requested
    for key in ("logits", "fused_logits", "probs", "probabilities",
                "prediction", "predictions", "CRBEF", "NEW_BASE"):
        if key in z.files:
            return key
    candidates = [k for k in z.files if k not in {"ids", "y_true", "labels"}
                  and np.asarray(z[k]).ndim >= 2]
    if len(candidates) == 1:
        return candidates[0]
    raise SystemExit("cannot infer prediction key; pass --main-key/--crbef-key")


def load_predictions(path: Path, requested: Optional[str]
                     ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], str, Optional[list]]:
    with np.load(path, allow_pickle=False) as z:
        key = _array_key(z, requested)
        arr = np.asarray(z[key])
        # train.py writes [checkpoint, sample, class].  The last checkpoint is
        # the deployed candidate and is the useful default for fusion.
        if arr.ndim == 3:
            arr = arr[-1]
        if arr.ndim != 2:
            raise SystemExit("%s:%s must be [N,C], got %s" %
                             (path, key, arr.shape))
        ids = np.asarray(z["ids"]).astype(str) if "ids" in z.files else None
        labels = None
        for label_key in ("y_true", "labels", "label"):
            if label_key in z.files:
                labels = np.asarray(z[label_key])
                break
        class_names = None
        for names_key in ("class_names", "classes", "class_order"):
            if names_key in z.files:
                class_names = [str(v) for v in np.asarray(z[names_key]).tolist()]
                break
        return arr.astype(np.float32), ids, labels, key, class_names


def _np_probabilities(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if np.isfinite(arr).all() and (arr >= -1e-6).all():
        total = arr.sum(axis=1, keepdims=True)
        if np.allclose(total, 1.0, atol=2e-4, rtol=2e-4):
            return np.clip(arr, 1e-7, None).astype(np.float32)
    arr = arr - arr.max(axis=1, keepdims=True)
    exp = np.exp(arr)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def load_local_inputs(path: Path):
    """Load the four-way P tensor and modality evidence C for local_current.

    Preferred format is ``P[N,4,C]`` plus ``C[N,3,C]`` (or the descriptive
    key ``modality_evidence``).  For convenience, the package-style
    ``p_tav``/``b0`` keys are also accepted; a and b are taken from the two
    prediction files passed to this CLI.
    """
    with np.load(path, allow_pickle=False) as z:
        if "P" in z.files:
            P = np.asarray(z["P"], dtype=np.float32)
        else:
            if "p_tav" not in z.files:
                raise SystemExit("fusion inputs require P or p_tav")
            b0_key = "b0" if "b0" in z.files else "main_no_history"
            if b0_key not in z.files:
                raise SystemExit("fusion inputs require b0 or main_no_history")
            P = None
            p_tav = np.asarray(z["p_tav"], dtype=np.float32)
            b0 = np.asarray(z[b0_key], dtype=np.float32)
        c_key = "C" if "C" in z.files else "modality_evidence"
        if c_key not in z.files:
            raise SystemExit("fusion inputs require C or modality_evidence")
        C = np.asarray(z[c_key], dtype=np.float32)
        ids = np.asarray(z["ids"]).astype(str) if "ids" in z.files else None
        class_names = None
        for names_key in ("class_names", "classes", "class_order"):
            if names_key in z.files:
                class_names = [str(v) for v in np.asarray(z[names_key]).tolist()]
                break
    return P, (locals().get("p_tav"), locals().get("b0")), C, ids, class_names


def align_local_inputs(P, refs, C, aux_ids, target_ids,
                       main_np, expert_np):
    p_tav, b0 = refs
    if aux_ids is not None:
        if target_ids is None:
            raise SystemExit("fusion inputs have ids but prediction files do not")
        pos = {str(v): i for i, v in enumerate(aux_ids.tolist())}
        if len(pos) != len(aux_ids):
            raise SystemExit("fusion input ids contain duplicates")
        missing = [v for v in target_ids if str(v) not in pos]
        if missing:
            raise SystemExit("fusion inputs miss aligned prediction ids")
        order = np.asarray([pos[str(v)] for v in target_ids], dtype=np.int64)
        if P is not None:
            P = P[order]
        else:
            p_tav, b0 = p_tav[order], b0[order]
        C = C[order]
    elif len(C) != len(main_np):
        raise SystemExit("fusion inputs row count does not match predictions")
    if P is None:
        P = np.stack([_np_probabilities(expert_np), _np_probabilities(main_np),
                      _np_probabilities(p_tav), _np_probabilities(b0)], axis=1)
    if P.ndim != 3 or P.shape[1] != 4 or P.shape[0] != len(main_np):
        raise SystemExit("fusion input P must be [N,4,C] and match predictions")
    if C.ndim != 3 or C.shape[0] != len(main_np) or C.shape[1] != 3:
        raise SystemExit("fusion input C must be [N,3,C] and match predictions")
    p_main = _np_probabilities(main_np)
    p_expert = _np_probabilities(expert_np)
    if not np.allclose(_np_probabilities(P[:, 0]), p_expert, atol=2e-3, rtol=2e-3):
        raise SystemExit("fusion P[:,0] does not match cRBEF predictions")
    if not np.allclose(_np_probabilities(P[:, 1]), p_main, atol=2e-3, rtol=2e-3):
        raise SystemExit("fusion P[:,1] does not match MHnoU predictions")
    return P.astype(np.float32), C.astype(np.float32)


def align(main: np.ndarray, main_ids: Optional[np.ndarray],
          expert: np.ndarray, expert_ids: Optional[np.ndarray],
          labels: Optional[np.ndarray], labels_from_expert: bool = False):
    if main_ids is None or expert_ids is None:
        if main.shape[0] != expert.shape[0]:
            raise SystemExit("prediction row counts differ and ids are missing")
        return main, expert, labels, None
    main_ids = main_ids.tolist()
    expert_ids = expert_ids.tolist()
    if len(main_ids) != len(set(map(str, main_ids))):
        raise SystemExit("MHnoU prediction ids contain duplicates")
    if len(expert_ids) != len(set(map(str, expert_ids))):
        raise SystemExit("cRBEF prediction ids contain duplicates")
    expert_pos = {str(v): i for i, v in enumerate(expert_ids)}
    missing = [v for v in main_ids if str(v) not in expert_pos]
    if missing:
        raise SystemExit("cRBEF predictions miss %d MHnoU ids (first=%r)" %
                         (len(missing), missing[0]))
    order = np.asarray([expert_pos[str(v)] for v in main_ids], dtype=np.int64)
    if labels is not None and labels_from_expert:
        labels = labels[order]
    if labels is not None and labels.shape[0] != len(main_ids):
        raise SystemExit("labels are not aligned with prediction rows")
    return main, expert[order], labels, np.asarray(main_ids, dtype=str)


def parse_prior(value: Optional[str], classes: int) -> Optional[torch.Tensor]:
    if not value:
        return None
    try:
        vals = [float(v.strip()) for v in value.split(",")]
    except ValueError as exc:
        raise SystemExit("--class-prior must be comma-separated numbers") from exc
    if len(vals) != classes or any(v < 0 for v in vals) or sum(vals) <= 0:
        raise SystemExit("--class-prior must contain %d non-negative values" % classes)
    return torch.tensor(vals, dtype=torch.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=Path, required=True,
                    help="MHnoU valid_logits.npz or another prediction file")
    ap.add_argument("--crbef", type=Path, required=True,
                    help="cRBEF-v2 prediction .npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fusion", choices=("local_current", "crbef_gate"),
                    default="local_current",
                    help="outer fusion method; local_current is the supplied newer method")
    ap.add_argument("--fusion-inputs", type=Path, default=None,
                    help="local_current .npz containing P[N,4,C] and C[N,3,C]")
    ap.add_argument("--fit-fusion-inputs", type=Path, default=None,
                    help="train/OOF local_current P/C inputs used to fit the corrector")
    ap.add_argument("--main-key", default=None)
    ap.add_argument("--crbef-key", default=None)
    ap.add_argument("--num-classes", type=int, default=None)
    ap.add_argument("--class-order", default=None,
                    help="optional comma-separated class order to require")
    ap.add_argument("--class-prior", default=None,
                    help="optional train prior, e.g. 0.1,0.2,...")
    ap.add_argument("--fit-gate", action="store_true",
                    help="fit the gate from --fit-main/--fit-crbef train or OOF files")
    ap.add_argument("--fit-main", type=Path, default=None,
                    help="train/OOF MHnoU predictions used only to fit the gate")
    ap.add_argument("--fit-crbef", type=Path, default=None,
                    help="train/OOF cRBEF predictions used only to fit the gate")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-gate", type=float, default=1.0)
    ap.add_argument("--initial-gate", type=float, default=0.5)
    ap.add_argument("--eta", type=float, default=1.0,
                    help="local_current correction strength; select on a separate CAL split")
    ap.add_argument("--gate-checkpoint", type=Path, default=None)
    ap.add_argument("--save-gate", type=Path, default=None)
    args = ap.parse_args(argv)

    main_np, main_ids, main_labels, main_key, main_classes = load_predictions(args.main, args.main_key)
    expert_np, expert_ids, expert_labels, expert_key, expert_classes = load_predictions(args.crbef, args.crbef_key)
    if main_classes is not None and expert_classes is not None and main_classes != expert_classes:
        raise SystemExit("MHnoU and cRBEF class order metadata differ")
    expected_classes = ([v.strip() for v in args.class_order.split(",")]
                        if args.class_order else None)
    declared_classes = main_classes or expert_classes
    if expected_classes is not None and declared_classes is not None and expected_classes != declared_classes:
        raise SystemExit("prediction class order does not match --class-order")
    labels_from_expert = main_labels is None and expert_labels is not None
    main_np, expert_np, labels, out_ids = align(
        main_np, main_ids, expert_np, expert_ids,
        main_labels if main_labels is not None else expert_labels,
        labels_from_expert=labels_from_expert)
    if main_np.shape != expert_np.shape:
        raise SystemExit("aligned predictions must have identical [N,C] shape")
    classes = int(args.num_classes or main_np.shape[1])
    if main_np.shape[1] != classes:
        raise SystemExit("--num-classes does not match prediction files")
    prior = parse_prior(args.class_prior, classes)

    if args.fusion == "local_current":
        if args.fusion_inputs is None:
            raise SystemExit("local_current requires --fusion-inputs with P[N,4,C] and C[N,3,C]")
        if args.gate_checkpoint is not None and args.fit_gate:
            raise SystemExit("choose --fit-gate or --gate-checkpoint, not both")
        P, refs, evidence, aux_ids, aux_classes = load_local_inputs(args.fusion_inputs)
        if aux_classes is not None and declared_classes is not None and aux_classes != declared_classes:
            raise SystemExit("fusion input class order differs from predictions")
        P, evidence = align_local_inputs(P, refs, evidence, aux_ids, out_ids,
                                         main_np, expert_np)
        local_cfg = LocalCurrentConfig(num_classes=classes, eta=args.eta)
        local_stats = {}
        if args.fit_gate:
            if args.fit_main is None or args.fit_crbef is None or args.fit_fusion_inputs is None:
                raise SystemExit("local_current fitting requires --fit-main, --fit-crbef, and --fit-fusion-inputs")
            fit_main_np, fit_main_ids, fit_labels_main, _, fit_main_classes = load_predictions(
                args.fit_main, args.main_key)
            fit_expert_np, fit_expert_ids, fit_labels_expert, _, fit_expert_classes = load_predictions(
                args.fit_crbef, args.crbef_key)
            if (fit_main_classes is not None and fit_expert_classes is not None
                    and fit_main_classes != fit_expert_classes):
                raise SystemExit("fit MHnoU and cRBEF class order metadata differ")
            fit_labels = (fit_labels_main if fit_labels_main is not None
                          else fit_labels_expert)
            fit_from_expert = fit_labels_main is None and fit_labels_expert is not None
            fit_main_np, fit_expert_np, fit_labels, fit_ids = align(
                fit_main_np, fit_main_ids, fit_expert_np, fit_expert_ids,
                fit_labels, labels_from_expert=fit_from_expert)
            if fit_labels is None:
                raise SystemExit("local_current fitting requires y_true/labels")
            fit_P, fit_refs, fit_evidence, fit_aux_ids, fit_aux_classes = load_local_inputs(
                args.fit_fusion_inputs)
            if fit_aux_classes is not None and declared_classes is not None and fit_aux_classes != declared_classes:
                raise SystemExit("fit fusion input class order differs from eval predictions")
            fit_P, fit_evidence = align_local_inputs(
                fit_P, fit_refs, fit_evidence, fit_aux_ids, fit_ids,
                fit_main_np, fit_expert_np)
            if fit_P.shape[2] != classes:
                raise SystemExit("fit fusion input class count does not match eval files")
            model, local_stats = fit_local_current(
                torch.from_numpy(fit_P), torch.from_numpy(fit_evidence),
                torch.from_numpy(fit_labels).long(), local_cfg,
                epochs=args.epochs, lr=args.lr)
        else:
            model = LocalCurrentFusion(local_cfg)
        if args.gate_checkpoint:
            payload = torch.load(args.gate_checkpoint, map_location="cpu")
            saved_cfg = payload.get("cfg", {}) if isinstance(payload, dict) else {}
            if isinstance(saved_cfg, dict):
                for key in ("eta", "correction_fraction", "delta_scale"):
                    if key in saved_cfg:
                        setattr(local_cfg, key, float(saved_cfg[key]))
            model = LocalCurrentFusion(local_cfg)
            state = payload.get("model", payload.get("state_dict", payload)) \
                if isinstance(payload, dict) else payload
            model.load_state_dict(state, strict=True)
            model.eval()
        with torch.no_grad():
            fused, details = model(torch.from_numpy(P), torch.from_numpy(evidence),
                                    return_details=True)
        fused_np = fused.cpu().numpy().astype(np.float32)
        fused_prob = details["probabilities"].cpu().numpy().astype(np.float32)
        report = {
            "fusion": "local_current",
            "main": str(args.main), "crbef": str(args.crbef),
            "fusion_inputs": str(args.fusion_inputs),
            "class_order": declared_classes or expected_classes,
            "n": int(fused_np.shape[0]), "num_classes": classes,
            "formula": "q=(p_cRBEF+p_MHnoU)/2; bounded class-wise local correction",
            "features": "23 per-class features; P=[cRBEF,MHnoU,TAV_reference,no_history_reference], C=[T,A,V]",
            "eta": float(local_cfg.eta),
            "correction_fraction": float(local_cfg.correction_fraction),
            "delta_scale": float(local_cfg.delta_scale),
            "fitted": bool(args.fit_gate or args.gate_checkpoint),
            "fit_main": str(args.fit_main) if args.fit_main else None,
            "fit_crbef": str(args.fit_crbef) if args.fit_crbef else None,
            "fit_fusion_inputs": str(args.fit_fusion_inputs) if args.fit_fusion_inputs else None,
            "fit_stats": local_stats,
            "mean_abs_delta": float(details["delta"].abs().mean()),
            "candidate_fraction": float(details["candidate_mask"].mean()),
        }
        if labels is not None and labels.shape[0] == fused_np.shape[0]:
            report["metrics"] = classification_report(
                fused_np, labels.astype(np.int64).tolist(), classes,
                ["class_%d" % i for i in range(classes)])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out, fused_logits=fused_np, probabilities=fused_prob,
            delta=details["delta"].cpu().numpy().astype(np.float32),
            candidate_mask=details["candidate_mask"].cpu().numpy().astype(np.float32),
            y_true=labels if labels is not None else np.asarray([]),
            ids=out_ids if out_ids is not None else np.asarray([]))
        if args.save_gate:
            args.save_gate.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"cfg": local_cfg.__dict__, "model": model.state_dict(),
                        "report": report}, args.save_gate)
        report_path = args.out.with_suffix(args.out.suffix + ".json")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    cfg = CRBEFFusionConfig(num_classes=classes, max_gate=args.max_gate,
                            initial_gate=args.initial_gate)
    main_t = torch.from_numpy(main_np)
    expert_t = torch.from_numpy(expert_np)
    gate_stats = {}
    if args.fit_gate:
        if args.fit_main is None or args.fit_crbef is None:
            raise SystemExit("--fit-gate requires --fit-main and --fit-crbef; "
                             "fit on train/OOF predictions, not eval rows")
        if args.gate_checkpoint is not None:
            raise SystemExit("choose --fit-gate or --gate-checkpoint, not both")
        fit_main_np, fit_main_ids, fit_labels_main, _, fit_main_classes = load_predictions(
            args.fit_main, args.main_key)
        fit_expert_np, fit_expert_ids, fit_labels_expert, _, fit_expert_classes = load_predictions(
            args.fit_crbef, args.crbef_key)
        if (fit_main_classes is not None and fit_expert_classes is not None
                and fit_main_classes != fit_expert_classes):
            raise SystemExit("fit MHnoU and cRBEF class order metadata differ")
        if declared_classes is not None:
            fit_declared = fit_main_classes or fit_expert_classes
            if fit_declared is not None and fit_declared != declared_classes:
                raise SystemExit("fit and eval prediction class orders differ")
        fit_from_expert = fit_labels_main is None and fit_labels_expert is not None
        fit_labels = (fit_labels_main if fit_labels_main is not None
                      else fit_labels_expert)
        fit_main_np, fit_expert_np, fit_labels, _ = align(
            fit_main_np, fit_main_ids, fit_expert_np, fit_expert_ids,
            fit_labels, labels_from_expert=fit_from_expert)
        if fit_labels is None:
            raise SystemExit("fit files require y_true/labels")
        if fit_main_np.shape[1] != classes or fit_expert_np.shape != fit_main_np.shape:
            raise SystemExit("fit prediction class shapes do not match eval files")
        model, gate_stats = fit_gate(
            torch.from_numpy(fit_main_np), torch.from_numpy(fit_expert_np),
            torch.from_numpy(fit_labels).long(), cfg, class_prior=prior,
            epochs=args.epochs, lr=args.lr)
    else:
        model = CRBEFExternalFusion(cfg, class_prior=prior)
    if args.gate_checkpoint:
        payload = torch.load(args.gate_checkpoint, map_location="cpu")
        if isinstance(payload, dict) and isinstance(payload.get("cfg"), dict):
            saved_max_gate = payload["cfg"].get("max_gate")
            if saved_max_gate is not None:
                cfg.max_gate = float(saved_max_gate)
        if isinstance(payload, dict) and "model" in payload:
            payload = payload["model"]
        model.load_state_dict(payload, strict=True)
        model.eval()
    with torch.no_grad():
        fused, gate = model(main_t, expert_t, return_gate=True)
    fused_np = fused.cpu().numpy().astype(np.float32)
    gate_np = gate.cpu().numpy().astype(np.float32)
    report = {
        "main": str(args.main), "crbef": str(args.crbef),
        "main_key": main_key, "crbef_key": expert_key,
        "class_order": declared_classes or expected_classes,
        "n": int(fused_np.shape[0]), "num_classes": classes,
        "formula": "log_p_mh + gate * (log_p_crbef - log_class_prior)",
        "gate_bound": [0.0, float(cfg.max_gate)],
        "gate_mean": float(gate_np.mean()),
        "gate_min": float(gate_np.min()),
        "gate_max": float(gate_np.max()),
        "gate_fit": bool(args.fit_gate or args.gate_checkpoint),
        "fit_main": str(args.fit_main) if args.fit_main else None,
        "fit_crbef": str(args.fit_crbef) if args.fit_crbef else None,
        "gate_fit_stats": gate_stats,
    }
    if labels is not None and labels.shape[0] == fused_np.shape[0]:
        report["metrics"] = classification_report(
            fused_np, labels.astype(np.int64).tolist(), classes,
            ["class_%d" % i for i in range(classes)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, fused_logits=fused_np, gate=gate_np,
                        y_true=labels if labels is not None else np.asarray([]),
                        ids=out_ids if out_ids is not None else np.asarray([]))
    if args.save_gate:
        args.save_gate.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": cfg.__dict__, "model": model.state_dict(),
                    "report": report}, args.save_gate)
    report_path = args.out.with_suffix(args.out.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
