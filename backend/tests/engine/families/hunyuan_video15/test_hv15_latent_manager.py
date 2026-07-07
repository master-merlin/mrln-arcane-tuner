"""LatentManager registration for the HunyuanVideo 1.5 VAE.

``AutoencoderKLHunyuanVideo15`` must be recognized as a 5D video VAE
(``_is_video_vae`` → True, matched by CLASS NAME so the heavy diffusers class
is never imported) with temporal downscale 4 and the ``(F-1)//4+1`` latent
frame math. The scalar 1.03682 scaling factor resolves from the VAE config
(multiply-on-encode path — no latents_mean/std stats).
"""

from types import SimpleNamespace

import pytest

from app.engine.components.latents import LatentManager


class AutoencoderKLHunyuanVideo15:  # noqa: N801 — matched by NAME, not import
    """Name-matched stand-in (LatentManager walks the MRO by class name)."""

    config = SimpleNamespace(scaling_factor=1.03682, latent_channels=32)
    spatial_compression_ratio = 16


class _Subclassed(AutoencoderKLHunyuanVideo15):
    """MRO walk must also match subclasses."""


def _manager(vae) -> LatentManager:
    return LatentManager(vae, device="cpu")


def test_hv15_vae_is_video_vae():
    assert _manager(AutoencoderKLHunyuanVideo15())._is_video_vae() is True


def test_hv15_vae_subclass_is_video_vae():
    assert _manager(_Subclassed())._is_video_vae() is True


def test_hv15_temporal_downscale_is_4():
    assert _manager(AutoencoderKLHunyuanVideo15()).temporal_downscale() == 4


def test_hv15_arch_override_beats_inference():
    mgr = LatentManager(
        AutoencoderKLHunyuanVideo15(),
        device="cpu",
        arch_params={"vae_temporal_downsample": 8},
    )
    assert mgr.temporal_downscale() == 8


def test_hv15_spatial_downscale_from_vae_ratio():
    # spatial_compression_ratio=16 is trusted over the 8x image default.
    assert _manager(AutoencoderKLHunyuanVideo15()).spatial_downscale == 16


def test_hv15_scaling_factor_from_config():
    assert _manager(AutoencoderKLHunyuanVideo15()).scaling_factor == pytest.approx(
        1.03682
    )


@pytest.mark.parametrize(
    ("frames", "expected"),
    [(1, 1), (5, 2), (17, 5), (121, 31)],
)
def test_hv15_latent_frames_math(frames, expected):
    # (F - 1) // 4 + 1 — one latent frame per group of four plus the lead.
    assert LatentManager.latent_frames(frames, 4) == expected
