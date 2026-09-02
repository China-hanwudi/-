import torch

from temporal_n3.modules import (
    BatchedCandidateThreeByThree,
    CandidateTwoLevelGate,
    UtilityRiskBottleneck,
    authorize_next_round,
)
from temporal_n3.policy import TemporalPolicy, initial_candidates, refill_after_filter, strata
from temporal_n3.model import TemporalResamplingN3
from temporal_n3.config import TemporalN3Config


def test_candidate_grid_masks_and_bounded_fallback():
    torch.manual_seed(7)
    batch, candidates, dims = 2, 6, 16
    streams = {f"{name}_{side}": torch.randn(batch, dims) if side == "t" else torch.randn(batch, candidates, dims) for name in "TAV" for side in ("t", "h")}
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]], dtype=torch.float32)
    relation, grid = BatchedCandidateThreeByThree(dims, 4, 0.0)(streams, mask)
    utilities = {name: torch.randn(batch, candidates, 2) for name in ("U_T", "U_A", "U_V", "U_joint")}
    utilities["U_mix"] = torch.randn(batch, candidates, 1)
    bottleneck = UtilityRiskBottleneck(dims, 12, 0.0)(relation, utilities, torch.zeros(batch, candidates, 2))
    gated = CandidateTwoLevelGate(dims, 12, 0.0)(torch.randn(batch, dims), {f"{name}_h": streams[f"{name}_h"] for name in "TAV"}, utilities, bottleneck, mask)
    assert grid.shape == (batch, candidates, 3, 3)
    assert torch.all(gated["candidate_keep_prob"][1, 2:] == 0)
    assert authorize_next_round(2, False, 0.8, 0.5, True, 2.0, 0.8).action == "authorize_resample"
    assert authorize_next_round(3, False, 9, 9, True, 9, 1).action == "fallback_current_only"


def test_temporal_policy_is_bounded_and_never_refills_old_candidates():
    policy = TemporalPolicy(max_candidates=6)
    initial = initial_candidates(20, policy)
    refilled = refill_after_filter(20, initial, initial[:2], policy)
    assert len(initial) == 6 and len(refilled) == 6
    assert set(refilled).intersection(initial) <= set(initial[:2])
    older, middle, recent = strata(20)
    assert (len(older), len(middle), len(recent)) == (10, 6, 4)


def test_model_fails_closed_without_authorization_and_on_empty_history():
    cfg = TemporalN3Config(text_dim=8, audio_dim=8, video_dim=8, d_model=8, relation_rank=4, gate_hidden=8, num_classes=4)
    model = TemporalResamplingN3(cfg)
    streams = {
        "T_t": torch.randn(2, 8), "A_t": torch.randn(2, 8), "V_t": torch.randn(2, 8),
        "T_h": torch.randn(2, 3, 8), "A_h": torch.randn(2, 3, 8), "V_h": torch.randn(2, 3, 8),
    }
    batch = {**streams, "history_mask": torch.tensor([[1, 1, 1], [0, 0, 0]], dtype=torch.float32)}
    out = model(batch, torch.zeros(2, 3, 2))
    assert torch.equal(out["history_used"], torch.zeros(2, 1, dtype=torch.bool))
    assert torch.allclose(out["logits"], out["current_only_logits"])
    out = model(batch, torch.zeros(2, 3, 2), history_authorized=torch.tensor([1, 1]))
    assert torch.equal(out["history_used"].view(-1), torch.tensor([True, False]))
