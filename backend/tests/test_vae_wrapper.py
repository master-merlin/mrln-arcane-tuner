"""Tests for VAE normalization — architecture-agnostic.

Both FLUX families now use native diffusers VAEs (AutoencoderKL,
AutoencoderKLFlux2) with built-in scaling.  These tests validate the
scaling/shift math without importing any deleted wrapper classes.
"""

import torch


class TestVAEScalarNormalization:
    """Validate (z - shift) * scale and its inverse."""

    def test_scale_shift_roundtrip(self):
        """Apply scale+shift then invert — should recover original."""
        z = torch.randn(1, 32, 64, 64)
        scaling_factor = 0.3611
        shift_factor = 0.1159

        scaled = (z - shift_factor) * scaling_factor
        recovered = scaled / scaling_factor + shift_factor

        assert torch.allclose(z, recovered, atol=1e-6)

    def test_identity_when_factors_are_trivial(self):
        """shift=0, scale=1 should be identity."""
        z = torch.randn(1, 128, 32, 32)
        scaled = (z - 0.0) * 1.0
        assert torch.equal(z, scaled)

    def test_shift_moves_mean(self):
        """Shift should translate the distribution center."""
        z = torch.ones(1, 32, 16, 16) * 5.0
        shift = 5.0
        scale = 1.0
        result = (z - shift) * scale
        assert torch.allclose(result.mean(), torch.tensor(0.0), atol=1e-6)

    def test_scale_changes_variance(self):
        """Scaling factor should linearly change the standard deviation."""
        torch.manual_seed(42)
        z = torch.randn(1, 32, 64, 64)
        factor = 0.5
        scaled = z * factor
        assert abs(scaled.std().item() - z.std().item() * factor) < 0.01


class TestPixelUnshuffle:
    """Validate pixel-unshuffle packing used by FLUX VAEs."""

    def test_32ch_to_128ch_shape(self):
        """32-channel 64x64 → 128-channel 32x32 via pixel unshuffle."""
        from einops import rearrange

        z = torch.randn(1, 32, 64, 64)
        packed = rearrange(z, "b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2)
        assert packed.shape == (1, 128, 32, 32)

    def test_pixel_unshuffle_roundtrip(self):
        """Unshuffle then reshuffle should recover input exactly."""
        from einops import rearrange

        z = torch.randn(1, 32, 64, 64)
        packed = rearrange(z, "b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2)
        unpacked = rearrange(packed, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=2, p2=2, c=32)
        assert torch.allclose(z, unpacked)

    def test_different_spatial_sizes(self):
        """Unshuffle works for non-square latents."""
        from einops import rearrange

        z = torch.randn(1, 32, 128, 64)
        packed = rearrange(z, "b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2)
        assert packed.shape == (1, 128, 64, 32)
