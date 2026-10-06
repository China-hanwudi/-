"""Train latest MHnoU on matching source folds, refit cRBEF outer fusion.

Archived expert heads are reused. This is source-head-excluded development,
not full-system OOF and not an official test result. Old paused workflows
are read-only dependencies and are never resumed by this script.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--source-bank", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--valid-expert-cache", type=Path)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()
    root, source, data = args.root.resolve(), args.source_bank.resolve(), args.data.resolve()
    code = root / "code"
    sys.path.insert(0, str(code))
    from n6.crbef_expert import M3ED_CLASSES
    from n6.data import open_split
    from n6.crbef_fusion import LocalCurrentConfig, LocalCurrentFusion
    from n6.recomer import ReCoMER
    from tools.train_recomer_fusion import four_way, load_inputs
    torch.set_num_threads(2)
    started = time.time()
    locks = []
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    def status(phase, **extra):
        x = dict(phase=phase, pid=os.getpid(), seconds=round(time.time() - started),
                 old_paused_workflow_resumed=False, official_test_evaluated=False, **extra)
        save(root / "STATUS.json", x)
        print(json.dumps(x), flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    def run(cmd, name):
        with (logs / (name + ".log")).open("w") as f:
            subprocess.run(cmd, cwd=code, env=env, stdout=f, stderr=subprocess.STDOUT,
                           check=True, timeout=3600)
    try:
        for name in ("GPU_RESEARCH.lock", "FUSION_RESEARCH.lock"):
            h = (source.parent / name).open("a+")
            fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(h)
        if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip():
            raise RuntimeError("GPU occupied; existing process will not be interrupted")
        if (root / "PLAN.json").exists():
            raise RuntimeError("one-shot plan exists; inspect prior run before restarting")
        contracts = json.loads((source / "CONTRACTS.json").read_text())
        ds = open_split(data, "train.pt")
        ids = [str(v) for v in ds.raw["ids"]]
        if len(set(ids)) != len(ids):
            raise ValueError("pack IDs are not unique")
        lut = {v: i for i, v in enumerate(ids)}
        train_hash = sha(data / "train.pt")
        manifests = root / "manifests"
        manifests.mkdir(exist_ok=False)
        jobs = contracts["jobs"]
        for job in jobs:
            fit, dev = set(job["fit_ids"]), set(job["selection_ids"])
            if not (fit | dev).issubset(lut) or fit & dev:
                raise ValueError("source contract incompatible with current pack")
            if {v.split("_")[1] for v in fit} & {v.split("_")[1] for v in dev}:
                raise ValueError("model fit/selection sources overlap")
            for role, target in job["predictions"].items():
                if (fit | dev) & set(target):
                    raise ValueError("target IDs overlap upstream task-head fit/selection")
                if {v.split("_")[1] for v in fit | dev} & {v.split("_")[1] for v in target}:
                    raise ValueError("target sources overlap upstream task-head fit/selection")
                save(manifests / (job["name"] + "_" + role + "_ids.json"), target)
                bank = source / "pairs" / job["name"] / (role + "_INPUTS.npz")
                labels = source / "pairs" / job["name"] / (role + "_LABELS.npz")
                with np.load(bank, allow_pickle=False) as z, np.load(labels, allow_pickle=False) as y:
                    if not np.array_equal(z["ids"], y["ids"]):
                        raise ValueError("expert prediction/label IDs differ")
                    current_y = ds.raw["label"][[lut[str(v)] for v in z["ids"]]].numpy()
                    if not np.array_equal(current_y, y["labels"]):
                        raise ValueError("expert/current pack target mapping differs")
            selected = fit | dev
            save(manifests / (job["name"] + ".json"), dict(version="dialogue_fit_inner_dev_v1",
                train_sha256=train_hash, fit_indices=[lut[v] for v in job["fit_ids"]],
                inner_dev_indices=[lut[v] for v in job["selection_ids"]],
                excluded_indices=[i for i, v in enumerate(ids) if v not in selected]))
        # Check that history never crosses fit/dev/target sources or dialogues.
        hi = ds.raw["history_index"].numpy()
        for i, row in enumerate(hi):
            current = ids[i].split("_")
            for j in row:
                if 0 <= j < len(ids) and ids[int(j)].split("_")[1:-1] != current[1:-1]:
                    raise ValueError("history crosses a source/dialogue")
        save(root / "PLAN.json", dict(scope="latest MHnoU + original archived cRBEF + refitted local_current",
             seeds=[43, 47, 59], mh_lr=args.lr, mh_fits=15, source_head_folds=4,
             source_head_seed=17, train_sha256=train_hash, source_contract_sha256=sha(source / "CONTRACTS.json"),
             code_sha256={str(p.relative_to(code)): sha(p) for p in code.rglob("*.py")},
             evaluation="reused OUTER development; official valid is additional development if supplied",
             full_system_OOF=False, upstream_teacher_exposure_known=True, official_test_evaluated=False,
             old_uniform_h_used=False, old_fusion_weights_used=False,
             lr_selection="mean official-valid WF1 across all three MH seeds in preceding validation grid",
             feature_scope="archived original CR predictions; raw train CR h features not recovered"))
        run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], "unit_tests")
        summary = []
        for seed in (43, 47, 59):
            inputs = root / "inputs" / str(seed)
            inputs.mkdir(parents=True, exist_ok=False)
            for job in jobs:
                name = job["name"]
                out = root / "mhnou" / str(seed) / name
                status("TRAINING_MHNOU_MATCHED_FOLD", seed=seed, fold=name)
                cmd = [sys.executable, "-u", "-m", "n6.train", "--data", str(data), "--out", str(out),
                       "--dialogue-manifest", str(manifests / (name + ".json")), "--seed", str(seed),
                       "--lr", str(args.lr), "--epochs", "20", "--batch-size", "256", "--swa-window", "3",
                       "--patience", "8", "--loss-patience", "0", "--min-delta", "0.0001", "--device", "cuda",
                       "--utility", "shapley", "--gate-architecture", "evidence", "--gate-detach-inputs",
                       "--detach-utility-path", "--bounded-w", "--bounded-lambda", "0.3",
                       "--history-abstain-variant", "utility_softmax", "--deploy", "closed_loop"]
                run(cmd, f"seed{seed}_{name}_train")
                for role in job["predictions"]:
                    target = inputs / (name + "_" + role + ".npz")
                    status("EXPORTING_MATCHED_INPUTS", seed=seed, fold=name, role=role)
                    run([sys.executable, "-m", "tools.recomer_predict", "--data", str(data),
                         "--mhnou", str(out / "best.pt"), "--expert-cache",
                         str(source / "pairs" / name / (role + "_INPUTS.npz")), "--ids-json",
                         str(manifests / (name + "_" + role + "_ids.json")), "--device", "cuda",
                         "--out", str(target)], f"seed{seed}_{name}_{role}_export")
            bank = [load_inputs(inputs / (j["name"] + "_fusion_fit.npz")) for j in jobs if j["name"] != "final"]
            joined = {k: (bank[0][k] if k == "class_names" else np.concatenate([z[k] for z in bank])) for k in bank[0]}
            if len(joined["ids"]) != 9741 or len(set(joined["ids"])) != 9741:
                raise ValueError("OOF bank coverage is incorrect")
            np.savez_compressed(inputs / "FIT.npz", **joined)
            fusedir = root / "recomer" / str(seed)
            status("FITTING_COMPLETE_RECOMER", seed=seed)
            run([sys.executable, "-m", "tools.train_recomer_fusion", "--fit", str(inputs / "FIT.npz"),
                 "--cal", str(inputs / "final_fusion_cal.npz"), "--mhnou",
                 str(root / "mhnou" / str(seed) / "final/best.pt"), "--seed", str(seed),
                 "--out", str(fusedir)], f"seed{seed}_fusion_train")
            ck = torch.load(fusedir / "local_current.pt", map_location="cpu", weights_only=True)
            fuser = LocalCurrentFusion(LocalCurrentConfig(**ck["cfg"]))
            fuser.load_state_dict(ck["model"])
            outer = load_inputs(inputs / "final_outer.npz")
            reports, probabilities = four_way(fuser, outer)
            save(fusedir / "OUTER_DEVELOPMENT_RESULT.json", dict(seed=seed, scope="previously reused train-source development",
                 metrics=reports, full_system_OOF=False, official_test_evaluated=False))
            np.savez_compressed(fusedir / "OUTER_PREDICTIONS.npz", ids=outer["ids"], probabilities=probabilities,
                                y_true=outer["y_true"], class_names=outer["class_names"])
            result = dict(seed=seed, bundle=str(fusedir / "recomer.pt"), outer=reports)
            if args.valid_expert_cache:
                # The optional cache is generated by the same CR17 source weights.
                with np.load(args.valid_expert_cache, allow_pickle=False) as z:
                    if tuple(z["class_names"]) != M3ED_CLASSES:
                        raise ValueError("official-valid CR cache class order differs")
                    p = np.stack([z["CR17"], z["CR17"], z["TAV"], z["TAV"]], axis=1)
                    np.savez_compressed(inputs / "VALID_CR_ONLY.npz", ids=z["ids"], P=p)
                run([sys.executable, "-m", "tools.recomer_predict", "--data", str(data), "--split", "valid",
                     "--bundle", str(fusedir / "recomer.pt"), "--expert-cache", str(inputs / "VALID_CR_ONLY.npz"),
                     "--device", "cuda", "--out", str(inputs / "VALID.npz")], f"seed{seed}_valid_export")
                valid = load_inputs(inputs / "VALID.npz")
                vr, _ = four_way(fuser, valid)
                save(fusedir / "VALID_DEVELOPMENT_RESULT.json", dict(seed=seed, metrics=vr,
                     scope="official valid reused for development; recovered CR feature route", official_test_evaluated=False))
                result["valid"] = vr
            # Verify that the self-contained bundle reproduces the fitted fuser.
            restored = ReCoMER.from_bundle(fusedir / "recomer.pt")
            with torch.no_grad():
                replay = restored.fusion(torch.from_numpy(outer["P"]), torch.from_numpy(outer["C"])).exp().numpy()
            if not np.array_equal(probabilities, replay):
                raise ValueError("complete model bundle replay differs")
            summary.append(result)
            save(root / "RESULTS.json", dict(seeds=summary, full_system_OOF=False,
                                            official_test_evaluated=False, cRBEF_seed_fixed=17))
            status("COMPLETE_MODEL_SAVED", seed=seed, bundle=str(fusedir / "recomer.pt"))
        status("COMPLETE", complete_recomer_bundles=3, four_way_comparisons=True)
    except BaseException as exc:
        status("FAILED", error=str(exc))
        raise
    finally:
        for h in locks:
            h.close()


if __name__ == "__main__":
    main()
