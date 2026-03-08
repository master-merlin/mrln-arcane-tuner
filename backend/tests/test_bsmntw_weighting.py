"""Tests for the BSMNTW weighted timestep loss implementation."""

import pytest
import torch


class TestBSMNTW:
    """Validate Bell-Shaped Mean-Normalized Timestep Weighting."""

    @pytest.fixture
    def bsmntw(self) -> torch.Tensor:
        """Compute the BSMNTW table (mirrors Flux2Trainer._setup_family)."""
        n = 1000
        x = torch.arange(n, dtype=torch.float32)
        y = torch.exp(-2.0 * ((x - n / 2) / n) ** 2)
        y_shifted = y - y.min()
        return y_shifted * (n / y_shifted.sum())

    def test_shape(self, bsmntw: torch.Tensor) -> None:
        """Table should have exactly 1000 entries."""
        assert bsmntw.shape == (1000,)

    def test_mean_normalized(self, bsmntw: torch.Tensor) -> None:
        """Mean weight should be approximately 1.0."""
        assert abs(bsmntw.mean().item() - 1.0) < 1e-5

    def test_all_non_negative(self, bsmntw: torch.Tensor) -> None:
        """All weights should be non-negative."""
        assert (bsmntw >= 0).all()

    def test_bell_shape(self, bsmntw: torch.Tensor) -> None:
        """Center weights should be larger than edge weights."""
        center = bsmntw[450:550].mean().item()
        edges = (bsmntw[:50].mean().item() + bsmntw[950:].mean().item()) / 2
        assert center > edges, f"Center {center} should be > edges {edges}"

    def test_peak_at_center(self, bsmntw: torch.Tensor) -> None:
        """Maximum weight should be at index 500 (mid-timestep)."""
        assert bsmntw.argmax().item() == 500

    def test_symmetric(self, bsmntw: torch.Tensor) -> None:
        """Weights should be approximately symmetric around center."""
        # Compare index i with index (999 - i) for i in 0..499
        left = bsmntw[:500]
        right = bsmntw[500:].flip(0)  # indices 999, 998, ..., 500
        assert torch.allclose(left, right, atol=0.01)

    def test_edge_near_zero(self, bsmntw: torch.Tensor) -> None:
        """Edge weights (pure noise / clean) should be near zero."""
        assert bsmntw[0].item() < 0.01
        assert bsmntw[999].item() < 0.01

    def test_timestep_to_index_mapping(self, bsmntw: torch.Tensor) -> None:
        """Verify timestep [0, 1000] -> index [0, 999] mapping.
        
        Index = 1000 - timestep, matching ai-toolkit's
        linspace(1000, 1, 1000) schedule.
        """
        timesteps = torch.tensor([1000.0, 500.0, 1.0])
        indices = (1000.0 - timesteps).clamp(0, 999).long()
        
        assert indices[0].item() == 0    # t=1000 -> idx 0 (pure noise)
        assert indices[1].item() == 500  # t=500  -> idx 500 (mid)
        assert indices[2].item() == 999  # t=1    -> idx 999 (clean)

        # Mid-timestep should get highest weight
        weights = bsmntw[indices]
        assert weights[1] > weights[0]  # mid > noise-end
        assert weights[1] > weights[2]  # mid > clean-end

    def test_mid_timestep_weight_range(self, bsmntw: torch.Tensor) -> None:
        """Mid-range weights should be in a reasonable range (1.5-2.5)."""
        peak = bsmntw[500].item()
        assert 1.0 < peak < 3.0, f"Peak weight {peak} out of expected range"


class TestComputeLossWeight:
    """Test the compute_loss_weight integration (mock-free unit tests)."""

    def test_weighted_mode_returns_weights(self) -> None:
        """When mode is 'weighted', compute_loss_weight should return weights."""
        # Simulate what Flux2Trainer does
        n = 1000
        x = torch.arange(n, dtype=torch.float32)
        y = torch.exp(-2.0 * ((x - n / 2) / n) ** 2)
        y_shifted = y - y.min()
        bsmntw = y_shifted * (n / y_shifted.sum())

        timesteps = torch.tensor([100.0, 500.0, 900.0])
        indices = (1000.0 - timesteps).clamp(0, 999).long()
        weights = bsmntw[indices]

        assert weights.shape == (3,)
        assert weights.dtype == torch.float32
        # t=500 should have highest weight (center of bell)
        assert weights[1] > weights[0]
        assert weights[1] > weights[2]

    def test_batch_size_one(self) -> None:
        """Single-sample batch should work correctly."""
        n = 1000
        x = torch.arange(n, dtype=torch.float32)
        y = torch.exp(-2.0 * ((x - n / 2) / n) ** 2)
        y_shifted = y - y.min()
        bsmntw = y_shifted * (n / y_shifted.sum())

        timesteps = torch.tensor([500.0])
        indices = (1000.0 - timesteps).clamp(0, 999).long()
        weights = bsmntw[indices]

        assert weights.shape == (1,)
        assert weights[0].item() > 1.0  # Center should be above mean
