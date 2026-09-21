"""minimax_h3 pixel adapter (plan row 2.0).

The H3 video VAE speaks ImageNet-normalised RGB over a ``[0, 1]`` base range
(diffusers ``autoencoder_kl_minimax_h3.py:512-515``); the engine's pixel
contract is ``[-1, 1]`` (``components/video.py`` produces it and
``components/latents.py`` hands it to ``vae.encode`` unchanged). No per-family
pixel seam exists, so the family wraps the visual VAE ONCE at the loader
boundary with ``H3PixelAdaptedVAE``. These tests use a stub inner VAE that
records what reaches it, so every assertion is on observable pixel values.
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn

from app.engine.components.latents import LatentManager
from app.engine.models.families.minimax_h3.pixel_adapter import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    H3PixelAdaptedVAE,
)

MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1)


class _RecordingVAE(nn.Module):
    """Inner stub: records ``encode`` input, returns a canned ``decode``."""

    def __init__(self, decode_value: torch.Tensor | None = None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))  # gives the module a dtype/device
        self.config = types.SimpleNamespace(
            latents_mean=(0.5, -0.25, 2.0), latents_std=(2.0, 0.5, 4.0), latent_channels=3,
        )
        self.spatial_compression_ratio = 16
        self.tiling_calls: list[str] = []
        self.encoded: torch.Tensor | None = None
        self._decode_value = decode_value

    def encode(self, x, return_dict=True):
        self.encoded = x.detach().clone()
        return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mode=lambda: x))

    def decode(self, z, return_dict=True):
        sample = self._decode_value if self._decode_value is not None else z
        return types.SimpleNamespace(sample=sample)

    def enable_tiling(self, *a, **k):
        self.tiling_calls.append("enable")

    def disable_tiling(self):
        self.tiling_calls.append("disable")


def _frame(value: float) -> torch.Tensor:
    return torch.full((1, 3, 1, 4, 4), value)


# ── 1. encode: [-1, 1] pixels reach the inner encoder ImageNet-normalised ──


def test_encode_maps_minus_one_one_pixels_to_imagenet_space():
    inner = _RecordingVAE()
    vae = H3PixelAdaptedVAE(inner)

    vae.encode(_frame(1.0))  # pure white
    expected_white = (1.0 - MEAN) / STD
    assert torch.allclose(inner.encoded, expected_white.expand_as(inner.encoded), atol=1e-6), (
        f"white reached the encoder as {inner.encoded[0, :, 0, 0, 0].tolist()}, "
        f"expected {expected_white.flatten().tolist()}"
    )
    assert round(inner.encoded[0, 0, 0, 0, 0].item(), 4) == 2.2489  # R channel, the row's number

    vae.encode(_frame(-1.0))  # pure black
    expected_black = -MEAN / STD
    assert torch.allclose(inner.encoded, expected_black.expand_as(inner.encoded), atol=1e-6)


# ── 2. decode: the inverse, including mid-gray that the clamp cannot mask ──


def test_decode_maps_imagenet_space_back_to_minus_one_one():
    cases = {
        "black": ((-MEAN / STD), -1.0),
        "white": (((1.0 - MEAN) / STD), 1.0),
        "mid-gray": (((0.5 - MEAN) / STD), 0.0),
    }
    for name, (inner_value, expected) in cases.items():
        inner = _RecordingVAE(decode_value=inner_value.expand(1, 3, 1, 4, 4).clone())
        out = H3PixelAdaptedVAE(inner).decode(torch.zeros(1, 3, 1, 1, 1)).sample
        assert out.shape == (1, 3, 1, 4, 4)
        assert torch.allclose(out, torch.full_like(out, expected), atol=1e-5), (
            f"{name}: decoded to {out[0, :, 0, 0, 0].tolist()}, expected {expected} per channel"
        )


# ── 3. encode → decode through an identity inner VAE round-trips ────────


def test_identity_inner_vae_round_trips_pixels():
    inner = _RecordingVAE()  # decode(z) returns z: the identity autoencoder
    vae = H3PixelAdaptedVAE(inner)
    torch.manual_seed(0)
    pixels = torch.rand(2, 3, 2, 4, 4) * 2 - 1
    latents = vae.encode(pixels).latent_dist.mode()
    out = vae.decode(latents).sample
    assert torch.allclose(out, pixels, atol=1e-6), f"max |Δ| = {(out - pixels).abs().max().item():.3e}"


# ── 4. the loader hands over the wrapper, and it looks like an ordinary VAE ──


def test_loader_wraps_the_visual_vae_once_at_the_boundary(monkeypatch):
    from app.engine.models.families.minimax_h3.loader import MiniMaxH3Loader
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    definition = ModelRegistry._definitions["minimax-h3-t2va"]

    loader = MiniMaxH3Loader(torch.device("cpu"))
    specs = {s.key: s for s in loader.get_component_manifest(definition)}
    inner = _RecordingVAE()
    # No weights: stub the path resolution and from_pretrained, keep the
    # base's own post-load dispatch (the seam under test) intact.
    monkeypatch.setattr(loader, "_resolve_component_path", lambda *a, **k: "stub-path")
    monkeypatch.setattr(loader, "_load_component", lambda *a, **k: inner)

    vae = loader._load_single_spec(specs["vae"], definition, "root", torch.float32, "cpu")
    assert isinstance(vae, H3PixelAdaptedVAE), type(vae).__name__
    assert vae.inner is inner
    assert not isinstance(vae.inner, H3PixelAdaptedVAE), "wrapped twice"

    # The audio VAE is untouched.
    assert specs["audio_vae"].post_load_hook is None

    # It looks like an ordinary VAE to LatentManager: same config stats, and
    # denormalize on the wrapper == denormalize on the inner (positive control).
    assert vae.config.latents_mean == inner.config.latents_mean
    assert vae.config is inner.config
    z = torch.randn(1, 3, 2, 2, 2)
    assert torch.equal(
        LatentManager.denormalize_latents(z, vae), LatentManager.denormalize_latents(z, inner)
    )
    assert vae.dtype == inner.scale.dtype and vae.device == inner.scale.device
    assert vae.spatial_compression_ratio == 16  # attribute passthrough
    vae.enable_tiling()
    vae.disable_tiling()
    assert inner.tiling_calls == ["enable", "disable"]
