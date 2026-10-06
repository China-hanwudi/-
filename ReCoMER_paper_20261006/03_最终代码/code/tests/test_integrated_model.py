"""Small CPU checks for the integrated Innovation-1/2/3 path."""
from __future__ import annotations

import unittest

import torch

from n6.config import M6Config
from n6.model import UGFModel


def make_batch():
    torch.manual_seed(23)
    b, k = 4, 3
    return {
        "T_t": torch.randn(b, 4), "A_t": torch.randn(b, 5),
        "V_t": torch.randn(b, 6),
        "T_h": torch.randn(b, k, 4), "A_h": torch.randn(b, k, 5),
        "V_h": torch.randn(b, k, 6),
        "history_mask": torch.ones(b, k),
        "history_modality_mask": torch.ones(b, k, 3),
        "modality_mask": torch.ones(b, 3),
        "speaker_same": torch.ones(b, k),
    }


class IntegratedModelTest(unittest.TestCase):
    def test_evidence_router_closed_loop(self):
        cfg = M6Config(
            text_dim=4, audio_dim=5, video_dim=6, d_model=12,
            num_heads=6, ff_dim=24, num_classes=3, dropout=0.0,
            utility="shapley", deploy="closed_loop",
            gate_architecture="evidence", use_bounded_w=True,
            gate_detach_inputs=True, detach_utility_path=True,
            history_abstain_variant="utility_softmax")
        model = UGFModel(cfg).eval()
        out = model(make_batch())
        self.assertEqual(tuple(out["deployed"].shape), (4, 3))
        self.assertEqual(tuple(out["weights"].shape), (4, 3))
        self.assertTrue(torch.allclose(
            out["weights"].sum(dim=-1), torch.ones(4), atol=1e-6))
        # closed_loop has only the three admitted current tokens; raw history
        # is already represented in cur_embs and must not be counted twice.
        self.assertEqual(out["token_mask"].shape[1], 3)

    def test_closed_loop_override_disconnects_history(self):
        cfg = M6Config(
            text_dim=4, audio_dim=5, video_dim=6, d_model=12,
            num_heads=6, ff_dim=24, num_classes=3, dropout=0.0,
            utility="uniform", deploy="closed_loop",
            history_abstain_variant="utility_softmax")
        model = UGFModel(cfg).eval()
        for pool in model.hist_pool.values():
            pool.gamma.data.fill_(1.0)
        batch = make_batch()
        off = model(batch, history_override=0)
        changed = {k: v.clone() for k, v in batch.items()}
        for key in ("T_h", "A_h", "V_h"):
            changed[key].add_(101.0)
        off_changed = model(changed, history_override=0)
        self.assertTrue(torch.equal(off["deployed"], off_changed["deployed"]))


if __name__ == "__main__":
    unittest.main()
