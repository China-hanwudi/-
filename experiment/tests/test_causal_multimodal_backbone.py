from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch", reason="causal backbone tests require PyTorch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.causal_multimodal_backbone import (  # noqa: E402
    CausalBackboneConfig,
    CausalMultimodalBackbone,
    UTILITY_CONTEXT_ORDER,
    pack_utility_context_masks,
    sample_history_subset,
)


def tiny_config() -> CausalBackboneConfig:
    return CausalBackboneConfig(
        text_dim=12,
        audio_dim=16,
        video_dim=20,
        d_model=32,
        num_heads=4,
        num_layers=2,
        ffn_dim=64,
        num_speakers=8,
        max_turns=64,
        max_relative_turn=16,
        dropout=0.0,
    )


def synthetic_batch(config: CausalBackboneConfig, *, batch: int = 2, length: int = 7) -> dict:
    generator = torch.Generator().manual_seed(20260808)
    return {
        "text_features": torch.randn(batch, length, config.text_dim, generator=generator),
        "audio_features": torch.randn(batch, length, config.audio_dim, generator=generator),
        "video_features": torch.randn(batch, length, config.video_dim, generator=generator),
        "speaker_ids": torch.tensor([[0, 1, 0, 1, 0, 1, 0]] * batch, dtype=torch.long),
        "turn_ids": torch.tensor([[0, 1, 2, 3, 4, 5, 6]] * batch, dtype=torch.long),
        "valid_mask": torch.ones(batch, length, dtype=torch.bool),
        "query_indices": torch.tensor([3, 4], dtype=torch.long),
    }


def test_future_changes_cannot_affect_current_output() -> None:
    config = tiny_config()
    model = CausalMultimodalBackbone(config).eval()
    batch = synthetic_batch(config)
    history = torch.ones_like(batch["valid_mask"])

    with torch.no_grad():
        baseline = model(**batch, history_mask=history).logits
        changed = {name: value.clone() for name, value in batch.items()}
        for row, query in enumerate(batch["query_indices"].tolist()):
            for name in ("text_features", "audio_features", "video_features"):
                changed[name][row, query + 1 :] = 10_000.0 * torch.randn_like(
                    changed[name][row, query + 1 :]
                )
            changed["speaker_ids"][row, query + 1 :] = 7
            changed["turn_ids"][row, query + 1 :] += 20
        actual = model(**changed, history_mask=history).logits

    torch.testing.assert_close(actual, baseline, rtol=0.0, atol=0.0)


def test_values_outside_history_mask_cannot_affect_output() -> None:
    config = tiny_config()
    model = CausalMultimodalBackbone(config).eval()
    batch = synthetic_batch(config)
    history = torch.zeros_like(batch["valid_mask"])
    history[:, 0] = True
    history[:, 2] = True

    with torch.no_grad():
        baseline = model(**batch, history_mask=history).logits
        changed = {name: value.clone() for name, value in batch.items()}
        excluded = ~history
        # Preserve each current query; the hard mask protects all other values.
        excluded.scatter_(1, batch["query_indices"][:, None], False)
        for name in ("text_features", "audio_features", "video_features"):
            changed[name][excluded] = 50_000.0 * torch.randn_like(changed[name][excluded])
        actual = model(**changed, history_mask=history).logits

    torch.testing.assert_close(actual, baseline, rtol=0.0, atol=0.0)


def test_four_context_batch_matches_four_independent_calls() -> None:
    config = tiny_config()
    model = CausalMultimodalBackbone(config).eval()
    batch = synthetic_batch(config)
    valid = batch["valid_mask"]
    masks = torch.zeros(valid.shape[0], 4, valid.shape[1], dtype=torch.bool)
    masks[:, 1, 0] = True
    masks[:, 2, :3] = True
    masks[:, 3, ::2] = True

    with torch.no_grad():
        joint = model.forward_contexts(**batch, context_masks=masks)
        separate = torch.stack(
            [model(**batch, history_mask=masks[:, index]).logits for index in range(4)],
            dim=1,
        )

    assert joint.logits.shape == (valid.shape[0], 4, 7)
    assert joint.probabilities.shape == (valid.shape[0], 4, 7)
    torch.testing.assert_close(joint.logits, separate, rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(
        joint.probabilities.sum(dim=-1),
        torch.ones(valid.shape[0], 4),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_published_configuration_is_strictly_under_two_million_parameters() -> None:
    config_path = ROOT / "configs" / "carma_causal_backbone_v1.json"
    config = CausalBackboneConfig.from_json(config_path)
    model = CausalMultimodalBackbone(config)
    assert model.parameter_count() < 2_000_000
    assert model.parameter_count() < config.parameter_limit


def test_utility_masks_and_random_subset_dropout_contract() -> None:
    base = torch.tensor([[True, True, True, True, True]])
    packed = pack_utility_context_masks(
        s_mask=base & torch.tensor([[True, True, False, False, False]]),
        s_plus_h_mask=base & torch.tensor([[True, True, True, False, False]]),
        t_mask=base & torch.tensor([[True, False, True, True, False]]),
        t_minus_h_mask=base & torch.tensor([[True, False, False, True, False]]),
    )
    assert packed.shape == (1, len(UTILITY_CONTEXT_ORDER), 5)
    assert not packed[:, 0].any()

    dropped = sample_history_subset(
        base,
        valid_mask=torch.ones_like(base),
        turn_ids=torch.arange(5).view(1, 5),
        query_indices=torch.tensor([3]),
        drop_probability=1.0,
    )
    assert not dropped.any()
    kept = sample_history_subset(
        base,
        valid_mask=torch.ones_like(base),
        turn_ids=torch.arange(5).view(1, 5),
        query_indices=torch.tensor([3]),
        drop_probability=0.0,
    )
    assert kept.tolist() == [[True, True, True, False, False]]


def test_cpu_amp_forward_is_finite() -> None:
    if not hasattr(torch, "autocast"):
        pytest.skip("torch.autocast is unavailable")
    config = tiny_config()
    model = CausalMultimodalBackbone(config).eval()
    batch = synthetic_batch(config)
    history = torch.ones_like(batch["valid_mask"])
    try:
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = model(**batch, history_mask=history)
    except RuntimeError as error:
        pytest.skip(f"this CPU build does not support bfloat16 autocast: {error}")
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.probabilities).all()
    assert output.logits.dtype in {torch.bfloat16, torch.float32}
