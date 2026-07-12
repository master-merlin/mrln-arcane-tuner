"""WAN 2.2 TI2V-5B sampler noise shape + saver arch string.

The shared ``WanVideoSamplerBase`` hardcodes wan21/wan22's VAE constants
(16 channels, spatial÷8) — WRONG for the 5B's new higher-compression VAE (48
channels, spatial÷16). Pins the two overrides that fix it, plus the saver's
fixed (not mode-suffixed) architecture string, since ``mode: both`` has no
single t2v/i2v label to embed the way wan21/wan22 do.
"""

from __future__ import annotations

import torch

from app.engine.models.families.wan22_ti2v_5b.sampler import Wan22Ti2v5bSampler


def test_initial_noise_shape_uses_48_channels_and_spatial_16():
    sampler = object.__new__(Wan22Ti2v5bSampler)
    sampler.device = torch.device("cpu")
    sampler.config = {}
    gen = torch.Generator().manual_seed(0)

    noise = sampler._create_initial_noise(width=640, height=704, generator=gen)

    # F for the base default 17-frame preview: (17-1)/4 + 1 = 5.
    assert noise.shape == (1, 48, 5, 704 // 16, 640 // 16)
    assert noise.dtype == torch.float32


def test_output_fps_is_24_not_wan21_wan22_16():
    assert Wan22Ti2v5bSampler.output_fps == 24.0


def test_saver_arch_string_is_fixed_ti2v_5b():
    from app.engine.models.families.wan22_ti2v_5b.saver import _ARCH

    assert _ARCH == "wan2.2-ti2v-5b"
