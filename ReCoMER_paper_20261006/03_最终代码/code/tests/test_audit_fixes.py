import unittest

import torch

from n6.config import M6Config
from n6.model import EvidenceRouter, UGFModel


class AuditFixTest(unittest.TestCase):
    def test_evidence_router_receives_history_state(self):
        torch.manual_seed(3)
        router = EvidenceRouter(12, 3, "cls", history_feature_dim=4)
        for p in router.scorer[-1].parameters():
            p.data.normal_(0.0, 0.1)
        cur = {m: torch.randn(2, 12) for m in ("T", "A", "V")}
        solo = {m: torch.randn(2, 3) for m in ("T", "A", "V")}
        mask = torch.ones(2, 3)
        z = torch.zeros(2, 4)
        o = torch.ones(2, 4)
        mu0, _ = router(cur, solo, mask, z)
        mu1, _ = router(cur, solo, mask, o)
        self.assertEqual(tuple(mu0.shape), (2, 3))
        self.assertGreater(float((mu0 - mu1).abs().max()), 0.0)

    def test_mpath_respects_history_override(self):
        cfg = M6Config(
            text_dim=4, audio_dim=4, video_dim=4, d_model=12,
            num_heads=3, ff_dim=24, num_classes=3, utility="uniform",
            use_history=True, history_k=2, use_mpath=True,
        )
        model = UGFModel(cfg).eval()
        model.mpath.gamma.data.fill_(1.0)
        b = {
            "T_t": torch.randn(2, 4), "A_t": torch.randn(2, 4),
            "V_t": torch.randn(2, 4),
            "T_h": torch.randn(2, 2, 4), "A_h": torch.randn(2, 2, 4),
            "V_h": torch.randn(2, 2, 4),
            "history_mask": torch.ones(2, 2),
            "history_modality_mask": torch.ones(2, 2, 3),
            "modality_mask": torch.ones(2, 3),
        }
        _, _, closed = model.encode(b, history_override=0)
        _, _, opened = model.encode(b, history_override=1)
        self.assertTrue(torch.allclose(closed["corr_m"], torch.zeros_like(closed["corr_m"])))
        self.assertGreater(float(opened["corr_m"].abs().max()), 0.0)

    def test_m1_rejects_weighted_joint_deployment(self):
        cfg = M6Config(use_m1=True, deploy="closed_loop")
        with self.assertRaises(ValueError):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()
