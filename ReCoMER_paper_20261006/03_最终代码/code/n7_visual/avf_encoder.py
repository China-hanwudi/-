"""Frozen visual-only AVF-MAE++ encoder; official weights and source required."""
import json
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn


def build_encoder(vendor, weights, device):
    sys.path.insert(0, str(vendor))
    from modeling_pretrain_av import PretrainVisionTransformerEncoder
    report = json.loads(Path(weights).with_suffix(".json").read_text())
    cfg = report["args"]
    state = torch.load(weights, map_location="cpu", weights_only=True)
    dim = state["patch_embed.proj.weight"].shape[0]
    depth = max(int(k.split(".")[1]) for k in state if k.startswith("blocks.")) + 1
    assert dim == 512 and depth == 10, (dim, depth)
    assert cfg["attn_type"] == "local_global"
    kwargs = {key: cfg[key] for key in (
        "attn_type", "lg_region_size", "lg_first_attn_type", "lg_third_attn_type",
        "lg_attn_param_sharing_first_third", "lg_attn_param_sharing_all",
        "lg_no_second", "lg_no_third") if key in cfg}
    encoder = PretrainVisionTransformerEncoder(
        img_size=160, patch_size=16, embed_dim=512, depth=depth,
        num_heads=8, qkv_bias=True, mlp_ratio=4, init_values=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    encoder.load_state_dict(state, strict=True)
    # Official source uses a plain sinusoid tensor; persist it in our wrapper.
    position = encoder.pos_embed
    del encoder.pos_embed
    encoder.register_buffer("pos_embed", position, persistent=True)
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


@torch.no_grad()
def encode(encoder, clips):
    mask = torch.zeros(clips.shape[0], encoder.patch_embed.num_patches,
                       device=clips.device, dtype=torch.bool)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        tokens, _ = encoder(clips, mask)
    embedding = tokens.float().mean(1)
    return torch.nn.functional.normalize(embedding, dim=-1)
