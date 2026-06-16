"""Per-channel latent normalization (encode) ↔ denormalization (decode).

Regression for the LTX-2 "samples are pure noise" bug: the LTX-2 VAE stores
``latents_mean``/``latents_std`` as module BUFFERS (``vae.latents_mean``), not
config entries, with ``latents_std ≈ 0.15``. The old detection only checked
``vae.config`` → LTX-2 latents were cached UN-normalized while the model works
in normalized space, so decoded samples were ~6.7× off → noise. encode and the
sampler's decode now share these helpers so they're guaranteed inverse.
"""

from __future__ import annotations

import torch

from app.engine.components.latents import LatentManager


class _Cfg:
    pass


def _vae(*, config_stats=False, buffer_stats=False, channels=4, std=0.15):
    vae = type("FakeVae", (), {})()
    cfg = _Cfg()
    if config_stats:
        cfg.latents_mean = [0.0] * channels
        cfg.latents_std = [std] * channels
    vae.config = cfg
    if buffer_stats:
        vae.latents_mean = torch.zeros(channels)
        vae.latents_std = torch.full((channels,), std)
    return vae


def test_resolve_stats_from_config():
    assert LatentManager._resolve_norm_stats(_vae(config_stats=True)) is not None


def test_resolve_stats_from_buffer_ltx2():
    # LTX-2: stats on the module (buffers), NOT config — the case the old
    # config-only check missed.
    vae = _vae(buffer_stats=True)
    assert LatentManager._resolve_norm_stats(vae) is not None


def test_resolve_stats_none_for_image_vae():
    # config present but no stats, no buffers → scalar-scaling VAE.
    assert LatentManager._resolve_norm_stats(_vae()) is None


def test_normalize_denormalize_roundtrip_5d():
    vae = _vae(buffer_stats=True, channels=4)
    z = torch.randn(1, 4, 2, 3, 3)  # [B, C, F, H, W]
    norm = LatentManager.normalize_latents(z, vae)
    back = LatentManager.denormalize_latents(norm, vae)
    assert torch.allclose(back, z, atol=1e-5)


def test_normalize_denormalize_roundtrip_4d():
    vae = _vae(config_stats=True, channels=4)
    z = torch.randn(2, 4, 8, 8)  # [B, C, H, W]
    back = LatentManager.denormalize_latents(LatentManager.normalize_latents(z, vae), vae)
    assert torch.allclose(back, z, atol=1e-5)


def test_normalize_scales_up_by_inverse_std():
    # std≈0.15 → normalize divides by ~0.15 → ~6.7× larger. The skipped scale
    # that turned decoded samples into noise.
    vae = _vae(buffer_stats=True, channels=4, std=0.15)
    z = torch.randn(1, 4, 2, 3, 3)
    norm = LatentManager.normalize_latents(z, vae)
    assert norm.std() > z.std() * 3.0


def test_image_vae_normalize_and_denormalize_are_identity():
    vae = _vae()  # no stats anywhere
    z = torch.randn(1, 4, 2, 3, 3)
    assert torch.equal(LatentManager.normalize_latents(z, vae), z)
    assert torch.equal(LatentManager.denormalize_latents(z, vae), z)


def test_per_channel_stats_applied_independently():
    vae = type("FakeVae", (), {})()
    vae.config = _Cfg()
    vae.latents_mean = torch.tensor([1.0, -2.0])
    vae.latents_std = torch.tensor([0.5, 4.0])
    z = torch.zeros(1, 2, 1, 1, 1)
    norm = LatentManager.normalize_latents(z, vae)
    # channel 0: (0 - 1)/0.5 = -2 ; channel 1: (0 - (-2))/4 = 0.5
    assert torch.allclose(norm[0, 0], torch.tensor(-2.0))
    assert torch.allclose(norm[0, 1], torch.tensor(0.5))
