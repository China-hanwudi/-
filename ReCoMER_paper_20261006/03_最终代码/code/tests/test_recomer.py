"""Real cRBEF weight replay and complete ReCoMER inference/bundle checks."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from n6.config import M6Config
from n6.model import UGFModel
from n6.crbef_expert import CRBEFExpert
from n6.crbef_fusion import LocalCurrentConfig, LocalCurrentFusion
from n6.recomer import ReCoMER

ROOT = Path(__file__).resolve().parents[1]


class ReCoMERTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        with np.load(ROOT / "tests/fixtures/crbef/SYNTHETIC_FEATURE_FIXTURE.npz", allow_pickle=False) as z:
            cls.fixture = {k: z[k].copy() for k in z.files}
        cls.assets = ROOT / "crbef_assets/m3ed"

    def inputs(self):
        f = self.fixture
        batch = {k[len("native__"):]: torch.from_numpy(v) for k, v in f.items() if k.startswith("native__")}
        x = {k: f["cr__" + k] for k in ("h", "a", "pa", "pv")}
        x.update(v=f["native__V_t"], A_mean=f["rich__A_mean"], A_local4=f["rich__A_local4"])
        return batch, x

    def complete(self, fusion=True):
        torch.manual_seed(12)
        cfg = M6Config(text_dim=768, audio_dim=2048, video_dim=342, task="cls", num_classes=7,
                       class_names=["Happy", "Neutral", "Sad", "Disgust", "Anger", "Fear", "Surprise"],
                       dataset_id="m3ed", d_model=24, ff_dim=72, use_history=True, history_k=3,
                       use_abstain=True, history_abstain_variant="utility_softmax", utility="shapley",
                       deploy="closed_loop", use_bounded_w=True, gate_architecture="evidence",
                       gate_detach_inputs=True, detach_utility_path=True, history_feature_dim=4)
        cfg.validate()
        return ReCoMER(UGFModel(cfg), CRBEFExpert(self.assets),
                       LocalCurrentFusion(LocalCurrentConfig(7)) if fusion else None)

    def test_original_expert_weights_match_archived_feature_forward(self):
        _, x = self.inputs()
        out = CRBEFExpert(self.assets)(x)
        with np.load(ROOT / "tests/fixtures/crbef/SYNTHETIC_EXPECTED.npz", allow_pickle=False) as z:
            np.testing.assert_allclose(out["CRBEF"].numpy(), z["CRBEF"], atol=3e-7, rtol=2e-6)
            np.testing.assert_allclose(out["CR_gate"].numpy(), z["CR_gate"], atol=3e-7, rtol=2e-6)

    def test_complete_prediction_ignores_history_labels_and_cuts_history(self):
        b, x = self.inputs()
        m = self.complete()
        out = m(b, x)
        relabeled = dict(b, history_label=torch.full((8, 3), 999, dtype=torch.long))
        again = m(relabeled, x)
        torch.testing.assert_close(out["fused"], again["fused"], rtol=0, atol=0)
        altered = dict(b)
        for k in ("T_h", "A_h", "V_h"):
            altered[k] = b[k] + 1000
        # Current-history predictions may change; the disabled-history reference cannot.
        changed = m.build_inputs(altered, x)
        torch.testing.assert_close(out["no_history"], changed["no_history"], rtol=0, atol=0)
        torch.testing.assert_close(out["C"], changed["C"], rtol=0, atol=0)
        torch.testing.assert_close(out["fused"].sum(1), torch.ones(8))

    def test_bundle_roundtrip_restores_all_three_modules(self):
        b, x = self.inputs()
        m = self.complete()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "recomer.pt"
            m.save_bundle(p, dict(scope="synthetic interface test"))
            restored = ReCoMER.from_bundle(p)
            torch.testing.assert_close(m(b, x)["fused"], restored(b, x)["fused"], atol=0, rtol=0)
            self.assertTrue(all(not p.requires_grad for p in restored.expert.parameters()))

    def test_unfitted_fusion_is_not_returned_as_complete_prediction(self):
        b, x = self.inputs()
        m = self.complete(fusion=False)
        with self.assertRaises(RuntimeError):
            m(b, x)


if __name__ == "__main__":
    unittest.main()
