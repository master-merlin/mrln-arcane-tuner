"""Tests for krea2 family: vendored transformer + conditioning helpers.

TDD order:
  1. test_krea2_vendor_imports_and_instantiates  — module exists + builds on CPU
  2. test_krea2_vendor_forward_shape             — forward pass produces correct shape
  3. test_krea2_conditioning_helpers             — pack/unpack/prepare_position_ids
"""

import torch
import pytest

# ── Tiny config shared by both model tests ──────────────────────────────────
_TINY_CFG = dict(
    in_channels=64,
    num_layers=2,
    attention_head_dim=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=256,
    timestep_embed_dim=256,
    text_hidden_dim=128,
    num_text_layers=12,
    text_num_attention_heads=4,
    text_num_key_value_heads=4,
    text_intermediate_size=128,
    num_layerwise_text_blocks=1,
    num_refiner_text_blocks=1,
    axes_dims_rope=(32, 48, 48),  # sum=128 == attention_head_dim
    rope_theta=1000.0,
    norm_eps=1e-5,
)


def test_krea2_vendor_imports_and_instantiates():
    from app.engine.models.families.krea2.vendor.transformer_krea2 import Krea2Transformer2DModel

    m = Krea2Transformer2DModel.from_config(_TINY_CFG)
    names = {n for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear)}
    assert any(n.endswith("attn.to_q") for n in names), f"attn.to_q not found. names={names}"
    assert any(n.endswith("ff.gate") for n in names), f"ff.gate not found. names={names}"


def test_krea2_vendor_forward_shape():
    from app.engine.models.families.krea2.vendor.transformer_krea2 import Krea2Transformer2DModel
    from app.engine.models.families.krea2.vendor.krea2_conditioning import prepare_position_ids

    m = Krea2Transformer2DModel.from_config(_TINY_CFG).eval()

    B, img_seq, txt_seq = 1, 16, 7  # 4×4 image grid
    hs = torch.randn(B, img_seq, 64)
    ehs = torch.randn(B, txt_seq, 12, 128)
    ts = torch.tensor([0.5])
    pos = prepare_position_ids(txt_seq, 4, 4, torch.device("cpu"))

    with torch.no_grad():
        out = m(
            hidden_states=hs,
            encoder_hidden_states=ehs,
            timestep=ts,
            position_ids=pos,
            return_dict=False,
        )[0]

    assert out.shape == (B, img_seq, 64), f"unexpected output shape: {out.shape}"
    assert out.isfinite().all(), "output contains NaN or inf"
    assert out.float().std() > 0, "output is degenerate (zero std)"


def test_krea2_conditioning_helpers():
    from app.engine.models.families.krea2.vendor.krea2_conditioning import (
        prepare_position_ids,
        pack_latents,
        unpack_latents,
    )

    # prepare_position_ids
    txt_seq, H, W = 7, 4, 6
    pos = prepare_position_ids(txt_seq, H, W, torch.device("cpu"))
    assert pos.shape == (txt_seq + H * W, 3), f"unexpected pos shape: {pos.shape}"
    # text rows should all be zero
    assert pos[:txt_seq].sum() == 0, "text rows should be all zeros"
    # image rows: t=0, h in [0,H), w in [0,W)
    img_pos = pos[txt_seq:]
    assert img_pos[:, 0].sum() == 0, "image t-axis should be 0"
    assert img_pos[:, 1].max() == H - 1
    assert img_pos[:, 2].max() == W - 1

    # pack_latents / unpack_latents roundtrip
    B, C, Hpx, Wpx = 2, 16, 8, 8
    patch_size = 2
    latents = torch.randn(B, C, Hpx, Wpx)
    packed = pack_latents(latents, patch_size=patch_size)
    assert packed.shape == (B, (Hpx // patch_size) * (Wpx // patch_size), C * patch_size * patch_size)

    # unpack at pixel dimensions (no VAE scale needed since we call the raw unpack)
    unpacked = unpack_latents(packed, Hpx, Wpx, patch_size=patch_size)
    # unpack_latents returns (B, C, 1, H, W) matching Krea2 pipeline convention
    assert unpacked.shape == (B, C, 1, Hpx, Wpx), f"unexpected unpack shape: {unpacked.shape}"
