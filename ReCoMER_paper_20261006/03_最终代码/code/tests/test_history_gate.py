"""Functional tests for sentence-level history admission.

Run from this code directory with:
  C:\\ProgramData\\anaconda3\\python.exe -m unittest tests.test_history_gate
"""
from __future__ import annotations

import unittest

import torch

from n6.config import M6Config
from n6.model import UGFModel


def config(mode: str = "sentence", execution: str = "hard") -> M6Config:
    return M6Config(
        text_dim=4, audio_dim=5, video_dim=6, d_model=12, num_heads=6,
        ff_dim=24, num_classes=3, dropout=0.0, utility="uniform",
        history_k=3, history_gate_mode=mode,
        history_gate_execution=execution, history_gate_warmup_epochs=0,
        history_gate_soft_epochs=0,
    )


def batch(k: int = 3) -> dict:
    torch.manual_seed(11)
    b = 3
    return {
        "T_t": torch.randn(b, 4), "A_t": torch.randn(b, 5),
        "V_t": torch.randn(b, 6),
        "T_h": torch.randn(b, k, 4), "A_h": torch.randn(b, k, 5),
        "V_h": torch.randn(b, k, 6),
        "history_mask": torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0]])[:, :k],
        "history_modality_mask": torch.ones(b, k, 3),
        "modality_mask": torch.ones(b, 3),
        "speaker_same": torch.ones(b, k),
    }


class HistoryGateTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = UGFModel(config()).eval()
        # A nonzero gamma makes this a true interruption test rather than an
        # accidental pass caused by LayerScale's zero initialisation.
        for pool in self.model.hist_pool.values():
            pool.gamma.data.fill_(1.0)

    def test_closed_gate_blocks_history_everywhere(self) -> None:
        b = batch()
        off = self.model(b, history_override=0)
        changed = {name: value.clone() for name, value in b.items()}
        for name in ("T_h", "A_h", "V_h"):
            changed[name].add_(101.0)
        off_changed = self.model(changed, history_override=0)
        self.assertTrue(torch.equal(off["deployed"], off_changed["deployed"]))
        self.assertTrue(torch.equal(off["joint"], off_changed["joint"]))
        self.assertTrue(torch.equal(off["token_mask"], off_changed["token_mask"]))
        self.assertTrue(torch.equal(off["token_mask"][:, 3:], torch.zeros_like(off["token_mask"][:, 3:])))

    def test_open_gate_uses_history(self) -> None:
        b = batch()
        on = self.model(b, history_override=1)
        changed = {name: value.clone() for name, value in b.items()}
        changed["T_h"].add_(9.0)
        changed_on = self.model(changed, history_override=1)
        self.assertGreater(float((on["joint"] - changed_on["joint"]).abs().max()), 0.0)

    def test_empty_history_is_safe(self) -> None:
        b = batch()
        b["history_mask"].zero_()
        out = self.model(b)
        self.assertFalse(torch.isnan(out["deployed"]).any())
        self.assertTrue(torch.equal(out["history_gate_hard"], torch.zeros(3)))
        self.assertTrue(torch.equal(out["token_mask"][:, 3:], torch.zeros_like(out["token_mask"][:, 3:])))

    def test_gate_features_and_gradients(self) -> None:
        model = UGFModel(config()).train()
        for pool in model.hist_pool.values():
            pool.gamma.data.fill_(1.0)
        model.set_history_gate_epoch(10)
        out = model(batch())
        self.assertEqual(tuple(out["history_gate_features"].shape), (3, 24))
        loss = out["deployed"].square().mean() + torch.nn.functional.binary_cross_entropy_with_logits(
            out["history_gate_logit"], torch.ones(3))
        loss.backward()
        self.assertIsNotNone(model.history_gate.net[0].weight.grad)
        self.assertTrue(torch.isfinite(model.history_gate.net[0].weight.grad).all())

    def test_none_mode_has_no_active_gate(self) -> None:
        model = UGFModel(config(mode="none")).eval()
        out = model(batch())
        self.assertIsNone(out["history_gate_logit"])
        self.assertIsNone(out["history_gate_applied"])
        self.assertTrue(all(pool.null_emb is None for pool in model.hist_pool.values()))

    def test_legacy_mode_has_its_own_null_parameters(self) -> None:
        model = UGFModel(config(mode="legacy_null")).eval()
        self.assertTrue(all(pool.null_emb is not None for pool in model.hist_pool.values()))

    def test_utility_empty_decision_disconnects_residual(self) -> None:
        cfg = config(mode="none")
        cfg.history_abstain_variant = "utility_softmax"
        model = UGFModel(cfg).eval()
        for pool in model.hist_pool.values():
            pool.gamma.data.fill_(1.0)
        b = batch()
        out = model(b)
        out_off = model(b, history_override=0)
        self.assertTrue(torch.equal(out_off["token_mask"][:, 3:],
                                    torch.zeros_like(out_off["token_mask"][:, 3:])))
        self.assertTrue(torch.equal(out_off["history_gate_hard"],
                                    torch.zeros_like(out_off["history_gate_hard"])))
        # The override must close the residual path used by solo/deployed too.
        changed = {name: value.clone() for name, value in b.items()}
        for name in ("T_h", "A_h", "V_h"):
            changed[name].add_(101.0)
        off_changed = model(changed, history_override=0)
        self.assertTrue(torch.equal(out_off["deployed"], off_changed["deployed"]))


if __name__ == "__main__":
    unittest.main()
