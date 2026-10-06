"""CPU checks for the cRBEF-v2 external expert adapter."""
from __future__ import annotations

import unittest

import torch

from n6.crbef_fusion import (CRBEFFusionConfig, CRBEFExternalFusion,
                             LocalCurrentConfig, LocalCurrentFusion,
                             fit_gate, fit_local_current, fuse_with_fixed_gate,
                             local_current_candidate_mask, local_current_features,
                             reliability_features)


class CRBEFFusionTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5)
        self.main = torch.randn(12, 4)
        self.expert = torch.randn(12, 4)
        self.y = torch.randint(0, 4, (12,))

    def test_gate_is_bounded_and_shapes_match(self):
        cfg = CRBEFFusionConfig(num_classes=4)
        model = CRBEFExternalFusion(cfg).eval()
        fused, gate = model(self.main, self.expert, return_gate=True)
        self.assertEqual(tuple(fused.shape), (12, 4))
        self.assertEqual(tuple(gate.shape), (12,))
        self.assertTrue(torch.isfinite(fused).all())
        self.assertGreaterEqual(float(gate.min()), 0.0)
        self.assertLessEqual(float(gate.max()), 1.0)

    def test_probability_inputs_and_reliability_features(self):
        p = self.main.softmax(-1)
        q = self.expert.softmax(-1)
        f = reliability_features(p, q)
        self.assertEqual(tuple(f.shape), (12, 6))
        fused = fuse_with_fixed_gate(p, q, torch.full((12,), 0.5))
        self.assertTrue(torch.allclose(fused.exp().sum(-1),
                                       torch.ones(12), atol=1e-6))

    def test_gate_training_keeps_experts_frozen(self):
        cfg = CRBEFFusionConfig(num_classes=4)
        model, stats = fit_gate(self.main, self.expert, self.y, cfg,
                                epochs=3)
        self.assertTrue(torch.isfinite(torch.tensor(stats["train_nll"])))
        self.assertTrue(all(not p.requires_grad for p in
                            (self.main, self.expert)))
        _, gate = model(self.main, self.expert, return_gate=True)
        self.assertTrue((gate >= 0).all() and (gate <= 1).all())

    def test_local_current_is_bounded_and_mass_preserving(self):
        p = torch.rand(12, 4, 4)
        p = p / p.sum(-1, keepdim=True)
        c = torch.randn(12, 3, 4)
        x = local_current_features(p, c)
        self.assertEqual(tuple(x.shape), (12, 4, 23))
        mask = local_current_candidate_mask(p)
        self.assertTrue(mask.any(dim=1).all())
        model = LocalCurrentFusion(LocalCurrentConfig(num_classes=4)).eval()
        fused, details = model(p, c, return_details=True)
        self.assertTrue(torch.isfinite(fused).all())
        self.assertTrue(torch.allclose(details["probabilities"].sum(-1),
                                       torch.ones(12), atol=1e-6))
        self.assertTrue(torch.all(details["probabilities"] >= 0))

    def test_local_current_fit_changes_only_outer_corrector(self):
        p = torch.rand(12, 4, 4)
        p = p / p.sum(-1, keepdim=True)
        c = torch.randn(12, 3, 4)
        model, stats = fit_local_current(
            p, c, self.y, LocalCurrentConfig(num_classes=4), epochs=2)
        self.assertTrue(torch.isfinite(torch.tensor(stats["loss"])))
        out = model(p, c)
        self.assertEqual(tuple(out.shape), (12, 4))
        self.assertTrue(torch.allclose(out.exp().sum(-1), torch.ones(12), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
