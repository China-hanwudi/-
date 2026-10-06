import ast
import copy
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, os.environ.get("GATE_CODE", str(Path(__file__).resolve().parent / "代码/创新点2_门控对照/code")))
from n6.data import open_split
from n6.evaluate import load_checkpoint
from n6.model import UGFModel
from n6.train_gate import rank_loss


def method_ast(path, name):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UGFModel")
    return ast.dump(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name))


def main():
    torch.set_num_threads(2)
    root = Path("/data/emo/肖田泽科研/模型/model6_innovation_validation_20260930/experiments")
    original = root.parent / "code/n6/model.py"
    new = Path(os.environ["GATE_CODE"]) / "n6/model.py"
    for name in ("measure_shapley", "deploy_weights"):
        assert method_ast(original, name) == method_ast(new, name), name
    for tag, ckpt, data in [
        ("mosei", root/"mosei_loop_20260930/runs/mosei_h/seed17/best.pt", "/data/emo/肖田泽科研/数据/MOSEI/packed"),
        ("meld", root/"meld_mechanism_20260930/runs/uniform_h/seed17/best.pt", "/data/emo/肖田泽科研/数据/MELD/packed"),
        ("m3ed", root/"runs/uniform_h/seed17/best.pt", "/data/emo/肖田泽科研/数据/M3ED/packed_audio_e2_zh_hubert_large")]:
        base, cfg, _ = load_checkpoint(ckpt, torch.device("cpu"))
        ds = open_split(data, "valid.pt", cfg.task)
        b, y = ds.batch(range(16))
        c = copy.deepcopy(cfg)
        c.utility = "shapley"
        c.use_bounded_w = True
        c.bounded_lambda = 0.3
        c.gate_architecture = "evidence"
        c.gate_detach_inputs = True
        c.detach_utility_path = True
        c.utility_head_version = 5
        c.contrib_correct = False
        c.validate()
        model = UGFModel(c).eval()
        missing, extra = model.load_state_dict(base.state_dict(), strict=False)
        assert not extra and all(n.startswith("utility_head.") for n in missing)
        out = model(b)
        assert torch.allclose(out["deployed"], base(b)["deployed"], atol=1e-6)
        phi = model.measure_shapley(b, y)
        assert torch.equal(phi, base.measure_shapley(b, y))
        loss = (out["utility_mu"] - phi).square().mean() + 0.1*rank_loss(out["utility_mu"], phi, b["modality_mask"])
        loss.backward()
        assert all(p.grad is None for p in model.encoders.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.utility_head.parameters())
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"check.pt"
            torch.save({"cfg": c.to_dict(), "model": model.state_dict()}, path)
            loaded, _, _ = load_checkpoint(path, torch.device("cpu"))
            assert torch.equal(loaded(b)["deployed"], out["deployed"])
        # Present-modality masking remains finite, including all-missing.
        masked = dict(b, modality_mask=b["modality_mask"].clone())
        masked["modality_mask"][0] = 0
        masked["modality_mask"][1] = torch.tensor([1., 0., 0.])
        mo = model(masked)
        assert torch.isfinite(mo["deployed"]).all()
        assert mo["deployed"][0].abs().sum() == 0
        assert torch.allclose(mo["deployed"][1], mo["solo_stack"][1, 0])
        score = torch.randn(16, 3)
        assert torch.allclose(model.deploy_weights(score), model.deploy_weights(10*score), atol=1e-6)
        assert model.deploy_weights(score).max() <= (1.3/3 + 1e-6)
        print("PASS", tag, "teacher/formula unchanged; gate gradients isolated; reload/masks valid", flush=True)
    # Existing revised checkpoints still strict-load.
    for ckpt in [root/"revised_gate_confirm/shapley_cc_lam03/seed17/best.pt",
                 root/"revised_crossdataset_20261001/meld/seed17/best.pt"]:
        load_checkpoint(ckpt, torch.device("cpu"))
    print("PASS legacy v3/v4 checkpoint compatibility", flush=True)


if __name__ == "__main__":
    main()
