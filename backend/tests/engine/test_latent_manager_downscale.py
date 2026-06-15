"""LatentManager spatial-downscale resolution.

Regression for the spurious ``latent_spatial_mismatch`` warning on LTX-2: the
manager defaulted ``spatial_downscale`` to 8 (no family sets
``vae_downsample_factor``), but LTX-2's VAE compresses 32× — so its shape
validator false-alarmed on every still-image latent (96x168 expected vs 24x42
actual).  Video VAEs expose their true ratio via ``spatial_compression_ratio``;
the manager now trusts it.  Image VAEs (SDXL / Flux) don't expose it → 8×.
"""

from app.engine.components.latents import LatentManager


class _VAE:
    """Minimal VAE stand-in; sets ``spatial_compression_ratio`` only when given."""

    def __init__(self, spatial_compression_ratio=None) -> None:
        if spatial_compression_ratio is not None:
            self.spatial_compression_ratio = spatial_compression_ratio


def test_spatial_downscale_reads_vae_compression_ratio():
    lm = LatentManager(_VAE(spatial_compression_ratio=32), device="cpu")
    assert lm.spatial_downscale == 32


def test_spatial_downscale_defaults_to_8_without_ratio():
    lm = LatentManager(_VAE(), device="cpu")
    assert lm.spatial_downscale == 8


def test_arch_override_beats_vae_ratio():
    lm = LatentManager(
        _VAE(spatial_compression_ratio=32),
        device="cpu",
        arch_params={"vae_downsample_factor": 16},
    )
    assert lm.spatial_downscale == 16
