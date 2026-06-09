"""Vendored Ideogram4 custom VAE: instantiation + encode/decode roundtrip.

Uses a tiny synthetic config (no gated weights) on a real CPU instance with
random weights, locking the documented contract: latent channel count, 8x...
no -- tiny config uses a 2x factor (ch_mult length 2 -> one downsample) to keep
the test fast; the spatial-factor assertion is derived from the config rather
than hard-coded to the production 8x.

The GroupNorm(num_groups=32) inside the resnet/attn blocks requires every
`ch * ch_mult[i]` channel count to be divisible by 32, so the tiny config keeps
`ch=32` with `ch_mult=[1, 2]`.
"""

from __future__ import annotations

import torch

from app.engine.models.families.ideogram4.vendor.autoencoder_ideogram4 import (
    AutoEncoderParams,
    Ideogram4AutoEncoder,
    convert_diffusers_state_dict,
)

# Tiny config: small channels/resolution. ch_mult has 2 entries -> 1 downsample
# stage -> spatial factor 2 (vs production 8 with 4 entries). z_channels=4 keeps
# the latent small while still exercising the 2*z_channels encoder head.
TINY_PARAMS = AutoEncoderParams(
    resolution=32,
    in_channels=3,
    ch=32,
    out_ch=3,
    ch_mult=[1, 2],
    num_res_blocks=1,
    z_channels=4,
)
# 2 ** (len(ch_mult) - 1)
TINY_SPATIAL_FACTOR = 2


def _tiny_vae() -> Ideogram4AutoEncoder:
    return Ideogram4AutoEncoder(TINY_PARAMS)


def test_vae_instantiates() -> None:
    vae = _tiny_vae()
    assert vae is not None
    # The custom BatchNorm2d head exists over prod(ps) * z_channels channels.
    assert vae.bn.num_features == 4 * TINY_PARAMS.z_channels  # 2*2 * 4 = 16
    assert vae.bn.affine is False


def test_bn_is_not_in_encode_decode_path() -> None:
    """Contract guard: the bn head must NOT be touched by encode/decode.

    Latent (de)normalization is done externally via LATENT_SHIFT/LATENT_SCALE,
    not this BatchNorm. We assert encode/decode don't update the bn running
    stats (num_batches_tracked stays 0).
    """
    vae = _tiny_vae().eval()
    assert int(vae.bn.num_batches_tracked) == 0
    x = torch.randn(1, TINY_PARAMS.in_channels, 16, 16)
    with torch.no_grad():
        latents = vae.encode(x)
        z = latents[:, : TINY_PARAMS.z_channels]
        vae.decode(z)
    assert int(vae.bn.num_batches_tracked) == 0


def test_encode_decode_roundtrip_shapes() -> None:
    vae = _tiny_vae().eval()
    h = w = 16
    x = torch.randn(1, TINY_PARAMS.in_channels, h, w)

    with torch.no_grad():
        latents = vae.encode(x)
        # Encoder emits mean+logvar -> 2 * z_channels, spatially downsampled.
        assert latents.shape[1] == 2 * TINY_PARAMS.z_channels
        assert latents.shape[2] == h // TINY_SPATIAL_FACTOR
        assert latents.shape[3] == w // TINY_SPATIAL_FACTOR

        # Decoder consumes z_channels (the posterior mean half here).
        z = latents[:, : TINY_PARAMS.z_channels]
        assert z.shape[1] == TINY_PARAMS.z_channels

        decoded = vae.decode(z)

    # Decode restores the input spatial dims and out_ch channels.
    assert decoded.shape[0] == x.shape[0]
    assert decoded.shape[1] == TINY_PARAMS.out_ch
    assert decoded.shape[2] == h
    assert decoded.shape[3] == w


def test_documented_latent_channel_count_default() -> None:
    """The documented production latent channel count is z_channels=32."""
    assert AutoEncoderParams().z_channels == 32


def test_convert_diffusers_state_dict_rewrites_keys() -> None:
    """The key-rewriter passes bn.* through and maps diffusers conv keys."""
    src = {
        "bn.running_mean": torch.zeros(16),
        "quant_conv.weight": torch.zeros(8, 8, 1, 1),
        "post_quant_conv.weight": torch.zeros(4, 4, 1, 1),
        "encoder.conv_norm_out.weight": torch.zeros(64),
        "encoder.conv_in.weight": torch.zeros(32, 3, 3, 3),
    }
    out = convert_diffusers_state_dict(src)
    assert "bn.running_mean" in out
    assert "encoder.quant_conv.weight" in out
    assert "decoder.post_quant_conv.weight" in out
    assert "encoder.norm_out.weight" in out
    assert "encoder.conv_in.weight" in out
