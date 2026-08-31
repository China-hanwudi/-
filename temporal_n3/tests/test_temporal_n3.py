import torch

from temporal_n3.modules import (
    BatchedCandidateThreeByThree,
    CandidateTwoLevelGate,
    UtilityRiskBottleneck,
    authorize_next_round,
)
from temporal_n3.policy import TemporalPolicy, initial_candidates, refill_after_filter, strata


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
