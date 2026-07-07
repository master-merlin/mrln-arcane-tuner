"""LatentManager support for the HunyuanVideo VAE (Kandinsky 5.0's VAE).

``AutoencoderKLHunyuanVideo`` consumes 5D ``[B, C, F, H, W]`` input and
compresses temporally 4x / spatially 8x with a SCALAR scaling factor
(0.476986, no per-channel latents_mean/std) — the LatentManager must detect it
as a video VAE and report temporal 4.
"""

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from app.engine.components.latents import LatentManager


class FakeHunyuanVideoVAE(nn.Module):
    """Stand-in for AutoencoderKLHunyuanVideo (class NAME drives detection)."""

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self._dtype = dtype
        self.config = MagicMock()
        self.config.scaling_factor = 0.476986
        self.config.shift_factor = None
        self.config.latents_mean = None
        self.config.latents_std = None
        self.spatial_compression_ratio = 8

    @property
    def dtype(self):
        return self._dtype

    def encode(self, x):
        b, c, f, h, w = x.shape
        lf = (f - 1) // 4 + 1
        out = torch.randn(b, 16, lf, h // 8, w // 8, dtype=self._dtype)
        result = MagicMock()
        result.latent_dist.sample.return_value = out
        return result


FakeHunyuanVideoVAE.__name__ = "AutoencoderKLHunyuanVideo"
FakeHunyuanVideoVAE.__qualname__ = "AutoencoderKLHunyuanVideo"


def test_hunyuan_video_vae_detected_as_video():
    lm = LatentManager(FakeHunyuanVideoVAE(), device="cpu")
    assert lm._is_video_vae() is True


def test_hunyuan_video_vae_temporal_downscale_is_4():
    lm = LatentManager(FakeHunyuanVideoVAE(), device="cpu")
    assert lm.temporal_downscale() == 4


def test_hunyuan_video_vae_spatial_and_scaling():
    lm = LatentManager(FakeHunyuanVideoVAE(), device="cpu")
    assert lm.spatial_downscale == 8
    assert lm.scaling_factor == 0.476986
    # No per-channel norm stats — the SCALAR scaling_factor path applies.
    assert lm._has_latent_norm_stats() is False


def test_hunyuan_video_clip_encodes_5d_with_scalar_scaling():
    """A 9-frame clip → 3 latent frames, scaled by 0.476986 (sample * sf)."""
    lm = LatentManager(FakeHunyuanVideoVAE(), device="cpu")
    clip = torch.randn(1, 3, 9, 64, 64)
    latents = lm.encode_and_cache_batch(clip, ids=["clip0"])
    assert latents.ndim == 5
    assert latents.shape == (1, 16, 3, 8, 8)


def test_latent_frames_formula_for_kandinsky_defaults():
    # 121 frames (K5 default) → 31 latent frames; 17 → 5.
    assert LatentManager.latent_frames(121, 4) == 31
    assert LatentManager.latent_frames(17, 4) == 5
