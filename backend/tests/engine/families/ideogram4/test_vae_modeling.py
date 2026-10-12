"""Ideogram4's VAE is diffusers 0.41.0's ``AutoencoderKLFlux2`` (LANE-150).

Tiny synthetic config (no gated weights), a real CPU instance with random
weights, locking the contract the family relies on: the latent channel count,
the spatial factor derived from the config (2 here: two blocks -> one
downsample; production has four -> 8x), and a ``bn`` head that encode/decode
never touch (the driver normalises with LATENT_SHIFT/LATENT_SCALE).

GroupNorm(num_groups=32) inside the resnet blocks requires every block channel
count to be divisible by 32, hence ``block_out_channels=(32, 64)``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from diffusers import AutoencoderKLFlux2

LATENT_CHANNELS = 4
TINY_SPATIAL_FACTOR = 2  # 2 ** (len(block_out_channels) - 1)


def _tiny_vae() -> AutoencoderKLFlux2:
    return AutoencoderKLFlux2(
        in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2, block_out_channels=(32, 64),
        layers_per_block=1, latent_channels=LATENT_CHANNELS, norm_num_groups=32, sample_size=32,
    ).eval()


def test_vae_instantiates() -> None:
    vae = _tiny_vae()
    # The BatchNorm2d head exists over prod(patch_size) * latent_channels channels.
    assert vae.bn.num_features == 4 * LATENT_CHANNELS  # 2*2 * 4 = 16
    assert vae.bn.affine is False


def test_bn_is_not_in_encode_decode_path() -> None:
    """Contract guard: the bn head must NOT be touched by encode/decode.

    Latent (de)normalization is done externally via LATENT_SHIFT/LATENT_SCALE,
    not this BatchNorm: num_batches_tracked stays 0 (train mode would count).
    """
    vae = _tiny_vae().train()
    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mode()
        vae.decode(z)
    assert int(vae.bn.num_batches_tracked) == 0


def test_encode_decode_roundtrip_shapes() -> None:
    vae = _tiny_vae()
    h = w = 16
    x = torch.randn(1, 3, h, w)
    with torch.no_grad():
        # The posterior mode is the latent_channels mean (NOT 2*latent_channels).
        z = vae.encode(x).latent_dist.mode()
        assert tuple(z.shape) == (1, LATENT_CHANNELS, h // TINY_SPATIAL_FACTOR, w // TINY_SPATIAL_FACTOR)
        decoded = vae.decode(z).sample
    assert tuple(decoded.shape) == (1, 3, h, w)


def test_documented_latent_channel_count() -> None:
    """Production: 32 latent channels (-> 2x2 patchify 128 = the DiT's in_channels)."""
    yaml_path = (
        Path(__file__).resolve().parents[4]
        / "app" / "engine" / "models" / "families" / "ideogram4" / "definitions" / "ideogram4_fp8.yaml"
    )
    arch = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["architecture_params"]
    assert arch["vae._class_name"] == "AutoencoderKLFlux2"
    assert arch["vae.latent_channels"] == 32
    assert arch["transformer.in_channels"] == 32 * 4
