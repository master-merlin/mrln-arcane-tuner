"""
Tests for NoiseInterpolation — noise scheduling factory.

Covers: linear, ddpm, cosine modes, edge cases, and error paths.
"""
import math

import pytest
import torch

from app.engine.strategies.noise_interpolation import NoiseInterpolation


# ── Factory Construction ─────────────────────────────────────────────────


class TestNoiseInterpolationInit:
    """Tests for NoiseInterpolation construction and validation."""

    def test_linear_mode_no_scheduler(self):
        """Linear mode should work without a scheduler."""
        interp = NoiseInterpolation("linear")
        assert interp.mode == "linear"
        assert interp.scheduler is None

    def test_cosine_mode_no_scheduler(self):
        """Cosine mode should work without a scheduler."""
        interp = NoiseInterpolation("cosine")
        assert interp.mode == "cosine"

    def test_ddpm_requires_scheduler(self):
        """DDPM mode should raise if no scheduler is provided."""
        with pytest.raises(ValueError, match="requires a scheduler"):
            NoiseInterpolation("ddpm")

    def test_ddpm_with_scheduler(self):
        """DDPM mode should accept a scheduler."""
        scheduler = type("FakeScheduler", (), {
            "alphas_cumprod": torch.linspace(1.0, 0.001, 1000)
        })()
        interp = NoiseInterpolation("ddpm", scheduler=scheduler)
        assert interp.mode == "ddpm"
        assert interp.scheduler is scheduler

    def test_unknown_mode_raises(self):
        """Unknown modes should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown noise interpolation"):
            NoiseInterpolation("unknown_mode")


# ── Linear Mode ──────────────────────────────────────────────────────────


class TestLinearInterpolation:
    """Tests for linear (rectified-flow) noise interpolation."""

    def test_t0_returns_clean_signal(self):
        """At t=0, noisy = (1-0)*x + 0*noise = x."""
        interp = NoiseInterpolation("linear")
        latents = torch.randn(2, 4, 8, 8)
        noise = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.0, 0.0])

        result = interp.add_noise(latents, noise, t)
        assert torch.allclose(result, latents, atol=1e-6)

    def test_t1000_returns_pure_noise(self):
        """At t=1000, noisy = (1-1)*x + 1*noise = noise."""
        interp = NoiseInterpolation("linear")
        latents = torch.randn(2, 4, 8, 8)
        noise = torch.randn(2, 4, 8, 8)
        t = torch.tensor([1000.0, 1000.0])

        result = interp.add_noise(latents, noise, t)
        assert torch.allclose(result, noise, atol=1e-6)

    def test_t500_midpoint_blend(self):
        """At t=500, noisy = 0.5*x + 0.5*noise."""
        interp = NoiseInterpolation("linear")
        latents = torch.ones(1, 4, 4, 4)
        noise = torch.zeros(1, 4, 4, 4)
        t = torch.tensor([500.0])

        result = interp.add_noise(latents, noise, t)
        expected = 0.5 * latents + 0.5 * noise
        assert torch.allclose(result, expected, atol=1e-6)

    def test_different_timesteps_per_sample(self):
        """Each sample should use its own timestep."""
        interp = NoiseInterpolation("linear")
        latents = torch.ones(3, 4, 4, 4)
        noise = torch.zeros(3, 4, 4, 4)
        t = torch.tensor([0.0, 500.0, 1000.0])

        result = interp.add_noise(latents, noise, t)
        # Sample 0: t=0 → latents
        assert torch.allclose(result[0], latents[0], atol=1e-6)
        # Sample 1: t=500 → 0.5*latents
        assert torch.allclose(result[1], 0.5 * latents[1], atol=1e-6)
        # Sample 2: t=1000 → noise
        assert torch.allclose(result[2], noise[2], atol=1e-6)

    def test_output_shape_matches_input(self):
        """Output shape should match input latents shape."""
        interp = NoiseInterpolation("linear")
        latents = torch.randn(4, 16, 32, 32)
        noise = torch.randn(4, 16, 32, 32)
        t = torch.tensor([100.0, 200.0, 300.0, 400.0])

        result = interp.add_noise(latents, noise, t)
        assert result.shape == latents.shape


# ── Cosine Mode ──────────────────────────────────────────────────────────


class TestCosineInterpolation:
    """Tests for cosine noise interpolation."""

    def test_t0_returns_clean_signal(self):
        """At t=0, cos(0)=1, sin(0)=0 → result = x."""
        interp = NoiseInterpolation("cosine")
        latents = torch.randn(2, 4, 8, 8)
        noise = torch.randn(2, 4, 8, 8)
        t = torch.tensor([0.0, 0.0])

        result = interp.add_noise(latents, noise, t)
        assert torch.allclose(result, latents, atol=1e-6)

    def test_t1000_returns_pure_noise(self):
        """At t=1000, cos(π/2)=0, sin(π/2)=1 → result = noise."""
        interp = NoiseInterpolation("cosine")
        latents = torch.randn(2, 4, 8, 8)
        noise = torch.randn(2, 4, 8, 8)
        t = torch.tensor([1000.0, 1000.0])

        result = interp.add_noise(latents, noise, t)
        assert torch.allclose(result, noise, atol=1e-4)

    def test_t500_midpoint(self):
        """At t=500, cos(π/4) ≈ sin(π/4) ≈ 0.7071."""
        interp = NoiseInterpolation("cosine")
        latents = torch.ones(1, 4, 4, 4)
        noise = torch.ones(1, 4, 4, 4) * 2.0
        t = torch.tensor([500.0])

        result = interp.add_noise(latents, noise, t)
        angle = 0.5 * (math.pi / 2.0)
        expected = math.cos(angle) * latents + math.sin(angle) * noise
        assert torch.allclose(result, expected, atol=1e-5)


# ── DDPM Mode ────────────────────────────────────────────────────────────


class TestDDPMInterpolation:
    """Tests for DDPM noise interpolation."""

    @pytest.fixture
    def ddpm_interp(self):
        """Create a DDPM interpolation with a mock scheduler."""
        scheduler = type("FakeScheduler", (), {
            "alphas_cumprod": torch.linspace(0.9999, 0.001, 1000)
        })()
        return NoiseInterpolation("ddpm", scheduler=scheduler)

    def test_output_shape(self, ddpm_interp):
        """DDPM output should match input shape."""
        latents = torch.randn(2, 4, 8, 8)
        noise = torch.randn(2, 4, 8, 8)
        t = torch.tensor([100, 900])

        result = ddpm_interp.add_noise(latents, noise, t)
        assert result.shape == latents.shape

    def test_early_timestep_mostly_signal(self, ddpm_interp):
        """At early timesteps (low noise), result should be close to latents."""
        latents = torch.ones(1, 4, 4, 4)
        noise = torch.zeros(1, 4, 4, 4)
        t = torch.tensor([0])  # Earliest timestep

        result = ddpm_interp.add_noise(latents, noise, t)
        # alphas_cumprod[0] ≈ 0.9999 → √0.9999 ≈ 1.0
        assert torch.allclose(result, latents, atol=0.01)

    def test_late_timestep_mostly_noise(self, ddpm_interp):
        """At late timesteps (high noise), result should be close to noise."""
        latents = torch.zeros(1, 4, 4, 4)
        noise = torch.ones(1, 4, 4, 4)
        t = torch.tensor([999])  # Latest timestep

        result = ddpm_interp.add_noise(latents, noise, t)
        # alphas_cumprod[999] ≈ 0.001 → √(1-0.001) ≈ 1.0
        assert torch.allclose(result, noise, atol=0.05)
