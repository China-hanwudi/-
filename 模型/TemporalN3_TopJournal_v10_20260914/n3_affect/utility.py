"""Multi-item measured marginal utilities with learned mixing weights.

Current-time Möbius terms, history unimodal bidirectional terms, and
current-history temporal interactions. Each effect has its own softmax
weight, trained by emotion CE. Mix weights also form a vector residual
injected into the gated representation so they receive a real gradient.
A hinge on the peak weight prevents collapse to a single effect.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


EFFECT_ORDER = (
    "T_fwd",
    "T_bwd",
    "A_fwd",
    "A_bwd",
    "V_fwd",
    "V_bwd",
    "TA",
    "TV",
    "AV",
    "cross",
    "Th_fwd",
    "Th_bwd",
    "Ah_fwd",
    "Ah_bwd",
    "Vh_fwd",
    "Vh_bwd",
    "T_time",
    "A_time",
    "V_time",
)


def _zero_like(x: Tensor) -> Tensor:
    return torch.zeros_like(x)


def fuse_variants(fuse: nn.Module, t: Tensor, a: Tensor, v: Tensor) -> dict[str, Tensor]:
    z = _zero_like(t)
    za = _zero_like(a)
    zv = _zero_like(v)
    cat = torch.cat
    return {
        "TAV": fuse(cat([t, a, v], dim=-1)),
        "0AV": fuse(cat([z, a, v], dim=-1)),
        "T0V": fuse(cat([t, za, v], dim=-1)),
        "TA0": fuse(cat([t, a, zv], dim=-1)),
        "T00": fuse(cat([t, za, zv], dim=-1)),
        "0A0": fuse(cat([z, a, zv], dim=-1)),
        "00V": fuse(cat([z, za, v], dim=-1)),
        "000": fuse(cat([z, za, zv], dim=-1)),
    }


def representation_deltas(z: dict[str, Tensor]) -> dict[str, Tensor]:
    return {
        "T_fwd": z["TAV"] - z["0AV"],
        "T_bwd": z["T00"] - z["000"],
        "A_fwd": z["TAV"] - z["T0V"],
        "A_bwd": z["0A0"] - z["000"],
        "V_fwd": z["TAV"] - z["TA0"],
        "V_bwd": z["00V"] - z["000"],
        "TA": z["TA0"] - z["T00"] - z["0A0"] + z["000"],
        "TV": z["T0V"] - z["T00"] - z["00V"] + z["000"],
        "AV": z["0AV"] - z["0A0"] - z["00V"] + z["000"],
        "cross": (
            z["TAV"]
            - z["TA0"]
            - z["T0V"]
            - z["0AV"]
            + z["T00"]
            + z["0A0"]
            + z["00V"]
            - z["000"]
        ),
    }


def temporal_delta(time_fuse: nn.Module, current: Tensor, history: Tensor) -> Tensor:
    """Inclusion-exclusion interaction between current and history of one modality."""
    zc = _zero_like(current)
    zh = _zero_like(history)
    cat = torch.cat
    f_ch = time_fuse(cat([current, history], dim=-1))
    f_c0 = time_fuse(cat([current, zh], dim=-1))
    f_0h = time_fuse(cat([zc, history], dim=-1))
    f_00 = time_fuse(cat([zc, zh], dim=-1))
    return f_ch - f_c0 - f_0h + f_00


def all_effect_deltas(
    current_z: dict[str, Tensor],
    history_z: dict[str, Tensor],
    time_fuse: nn.Module,
    t: Tensor,
    a: Tensor,
    v: Tensor,
    t_h: Tensor,
    a_h: Tensor,
    v_h: Tensor,
) -> dict[str, Tensor]:
    cur = representation_deltas(current_z)
    hist = representation_deltas(history_z)
    return {
        **cur,
        "Th_fwd": hist["T_fwd"],
        "Th_bwd": hist["T_bwd"],
        "Ah_fwd": hist["A_fwd"],
        "Ah_bwd": hist["A_bwd"],
        "Vh_fwd": hist["V_fwd"],
        "Vh_bwd": hist["V_bwd"],
        "T_time": temporal_delta(time_fuse, t, t_h),
        "A_time": temporal_delta(time_fuse, a, a_h),
        "V_time": temporal_delta(time_fuse, v, v_h),
    }


class BidirectionalUtilityHeads(nn.Module):
    """Score 19 measured effects and mix them with per-effect learned weights."""

    def __init__(self, d_model: int, hidden: int, dropout: float, mix_tau: float = 1.25) -> None:
        super().__init__()
        self.mix_tau = float(mix_tau)
        self.proj = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(d_model, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                    nn.Tanh(),
                )
                for name in EFFECT_ORDER
            }
        )
        self.mix_logits = nn.Parameter(torch.zeros(len(EFFECT_ORDER)))

    def mixing_weights(self) -> Tensor:
        return F.softmax(self.mix_logits / max(self.mix_tau, 1e-6), dim=0)

    def mixing_weights_dict(self) -> dict[str, float]:
        w = self.mixing_weights().detach().cpu()
        return {name: float(w[i]) for i, name in enumerate(EFFECT_ORDER)}

    def mix_parameters(self) -> list[nn.Parameter]:
        return [self.mix_logits]

    def forward(self, deltas: dict[str, Tensor], context: Tensor | None = None) -> dict[str, Tensor]:
        del context
        scored: dict[str, Tensor] = {}
        stacked = []
        for name in EFFECT_ORDER:
            u = self.proj[name](deltas[name])
            scored[name] = u
            stacked.append(u)
        mix = self.mixing_weights().view(1, -1)
        stacked_t = torch.cat(stacked, dim=-1)
        mix = mix.expand(stacked_t.size(0), -1)
        u_mix = (stacked_t * mix).sum(dim=-1, keepdim=True)
        stacked_d = torch.stack([deltas[name] for name in EFFECT_ORDER], dim=1)
        delta_mix = (stacked_d * mix.unsqueeze(-1)).sum(dim=1)
        u_t = torch.cat([scored["T_fwd"], scored["T_bwd"]], dim=-1)
        u_a = torch.cat([scored["A_fwd"], scored["A_bwd"]], dim=-1)
        u_v = torch.cat([scored["V_fwd"], scored["V_bwd"]], dim=-1)
        u_joint = torch.cat([u_mix, scored["cross"]], dim=-1)
        return {
            "U_T": u_t,
            "U_A": u_a,
            "U_V": u_v,
            "U_joint": u_joint,
            "U_cross": scored["cross"],
            "U_mix": u_mix,
            "U_TA": scored["TA"],
            "U_TV": scored["TV"],
            "U_AV": scored["AV"],
            "mix_weights": mix,
            "delta_mix": delta_mix,
        }
