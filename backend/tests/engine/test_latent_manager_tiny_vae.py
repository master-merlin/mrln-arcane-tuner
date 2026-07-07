"""LatentManager encode path for AutoencoderTiny-style VAEs (dreamlite).

``AutoencoderTiny.encode`` returns an ``AutoencoderTinyOutput`` with a
``.latents`` tensor and NO ``latent_dist`` — before the dreamlite family
landed, ``encode_and_cache_batch`` fell through its ``else`` branch and
returned the OUTPUT OBJECT instead of a tensor. These tests pin the
``retrieve_latents``-style ``.latents`` branch (mirroring the diffusers
``retrieve_latents`` helper) including the scalar scaling formula
``z = scaling_factor * (latents - shift_factor)``.
"""

from __future__ import annotations

import torch

from app.engine.components.latents import LatentManager


class _TinyOutput:
    """Mimics diffusers AutoencoderTinyOutput: .latents only, no latent_dist."""

    def __init__(self, latents: torch.Tensor):
        self.latents = latents


class _TinyStubVae:
    """Deterministic AutoencoderTiny-style stub (encode → .latents)."""

    def __init__(self, scaling_factor: float = 1.0, shift_factor: float = 0.0):
        self.dtype = torch.float32
        self.config = type("Cfg", (), {})()
        self.config.scaling_factor = scaling_factor
        self.config.shift_factor = shift_factor
        # 4-ch latent → LatentManager keeps the default 8× spatial downscale
        self.config.latent_channels = 4

    def to(self, *_args, **_kwargs):
        return self

    def encode(self, images: torch.Tensor) -> _TinyOutput:
        b, _c, h, w = images.shape
        # Deterministic pseudo-latents at 8× downscale
        lat = torch.arange(
            b * 4 * (h // 8) * (w // 8), dtype=torch.float32,
        ).reshape(b, 4, h // 8, w // 8)
        return _TinyOutput(lat)


def test_encode_handles_latents_output_without_latent_dist():
    """The .latents branch returns a TENSOR (not the output object)."""
    vae = _TinyStubVae()
    manager = LatentManager(vae, device=torch.device("cpu"))

    images = torch.rand(2, 3, 16, 16) * 2 - 1
    latents = manager.encode_and_cache_batch(images, ids=["a", "b"])

    assert isinstance(latents, torch.Tensor), (
        f"expected a tensor, got {type(latents)}"
    )
    assert latents.shape == (2, 4, 2, 2)
    expected = vae.encode(images.to(torch.float32)).latents
    assert torch.allclose(latents, expected), (
        "scaling 1.0 / shift 0.0 must be the identity on .latents"
    )


def test_encode_latents_branch_applies_scalar_scaling():
    """z = scaling_factor * (latents - shift_factor) — shared with the
    standard AutoencoderKL branch so decode (z / scale + shift) inverts it."""
    vae = _TinyStubVae(scaling_factor=0.5, shift_factor=0.25)
    manager = LatentManager(vae, device=torch.device("cpu"))

    images = torch.rand(1, 3, 16, 16) * 2 - 1
    latents = manager.encode_and_cache_batch(images, ids=["a"])

    raw = vae.encode(images.to(torch.float32)).latents
    assert torch.allclose(latents, 0.5 * (raw - 0.25))


def test_encode_real_autoencoder_tiny_roundtrip():
    """Integration: a real (tiny-config) diffusers AutoencoderTiny encodes to
    a [B, 4, H/8, W/8] tensor through the manager."""
    from diffusers import AutoencoderTiny

    torch.manual_seed(0)
    vae = AutoencoderTiny(
        encoder_block_out_channels=(8, 8, 8, 8),
        decoder_block_out_channels=(8, 8, 8, 8),
        num_encoder_blocks=(1, 1, 1, 1),
        num_decoder_blocks=(1, 1, 1, 1),
    ).eval()

    manager = LatentManager(vae, device=torch.device("cpu"))
    images = torch.rand(1, 3, 32, 32) * 2 - 1
    latents = manager.encode_and_cache_batch(images, ids=["img0"])

    assert isinstance(latents, torch.Tensor)
    assert latents.shape == (1, 4, 4, 4)
    assert latents.isfinite().all()
    with torch.no_grad():
        expected = vae.encode(images).latents  # scaling 1.0, shift 0.0
    assert torch.allclose(latents, expected)
