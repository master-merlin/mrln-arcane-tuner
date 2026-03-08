"""
Tests for bucketing logic and latent caching.
Phase 3: Data Pipeline — Bucketing & Latents.
"""

import os
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from app.engine.components.bucketing import BucketManager
from app.engine.components.latents import LatentManager


# ── Bucketing Tests ──────────────────────────────────────────────────────


class TestBucketGeneration:
    """Tests for bucket generation and assignment."""

    def test_default_generates_sdxl_buckets(self):
        """Default BucketManager generates SDXL-standard buckets for 1024."""
        bm = BucketManager(base_resolutions=1024)
        assert len(bm.buckets) > 0
        # Must include the standard square bucket
        squares = [b for b in bm.buckets if b["width"] == 1024 and b["height"] == 1024]
        assert len(squares) == 1

    def test_all_buckets_divisible_by_32(self):
        """Every bucket dimension must be divisible by 32."""
        bm = BucketManager(base_resolutions=1024)
        for b in bm.buckets:
            assert b["width"] % 32 == 0, f"width {b['width']} not divisible by 32"
            assert b["height"] % 32 == 0, f"height {b['height']} not divisible by 32"

    def test_multi_resolution_buckets(self):
        """Multi-resolution generates scaled buckets for each base."""
        bm = BucketManager(base_resolutions=[512, 1024])
        has_512_square = any(b["width"] == 512 and b["height"] == 512 for b in bm.buckets)
        has_1024_square = any(b["width"] == 1024 and b["height"] == 1024 for b in bm.buckets)
        assert has_512_square
        assert has_1024_square

    def test_scaled_buckets_divisibility(self):
        """Buckets for non-1024 base resolutions also respect divisibility."""
        bm = BucketManager(base_resolutions=768, divisibility=32)
        for b in bm.buckets:
            assert b["width"] % 32 == 0
            assert b["height"] % 32 == 0


class TestBucketAssignment:
    """Tests for bucket assignment logic."""

    def test_square_image_gets_square_bucket(self):
        """A square image should get the square bucket."""
        bm = BucketManager(base_resolutions=1024)
        bucket = bm.get_bucket(1024, 1024)
        assert bucket["width"] == 1024
        assert bucket["height"] == 1024

    def test_landscape_image_gets_landscape_bucket(self):
        """A wide image should be assigned a landscape bucket."""
        bm = BucketManager(base_resolutions=1024)
        bucket = bm.get_bucket(1920, 1080)
        assert bucket["width"] > bucket["height"], f"Expected landscape, got {bucket['width']}x{bucket['height']}"

    def test_portrait_image_gets_portrait_bucket(self):
        """A tall image should be assigned a portrait bucket."""
        bm = BucketManager(base_resolutions=1024)
        bucket = bm.get_bucket(768, 1280)
        assert bucket["height"] > bucket["width"], f"Expected portrait, got {bucket['width']}x{bucket['height']}"

    def test_extreme_panoramic_aspect_ratio(self):
        """Very wide image should still get a valid bucket."""
        bm = BucketManager(base_resolutions=1024)
        bucket = bm.get_bucket(4000, 500)
        assert bucket["width"] > 0 and bucket["height"] > 0
        assert bucket["width"] % 32 == 0

    def test_very_small_image(self):
        """A small image should still map to a valid bucket."""
        bm = BucketManager(base_resolutions=1024)
        bucket = bm.get_bucket(256, 256)
        assert bucket["width"] > 0 and bucket["height"] > 0

    def test_fallback_when_no_buckets(self):
        """get_bucket returns square fallback if buckets list is empty."""
        bm = BucketManager(base_resolutions=1024)
        bm.buckets = []  # Force empty
        bucket = bm.get_bucket(800, 600)
        assert bucket["width"] == 1024
        assert bucket["height"] == 1024

    def test_tie_prefers_higher_resolution(self):
        """When two buckets have equal crop, prefer the higher-resolution one."""
        bm = BucketManager(base_resolutions=[512, 1024])
        bucket = bm.get_bucket(1024, 1024)
        # For a 1024x1024 image, the 1024 square is a perfect match — should pick it over 512 square
        assert bucket["width"] == 1024


class TestBucketDistribution:
    """Tests for bucket distribution tracking and logging."""

    def test_distribution_tracking(self):
        """get_bucket increments distribution counter for assigned bucket."""
        bm = BucketManager(base_resolutions=1024)
        bm.get_bucket(1024, 1024)
        bm.get_bucket(1024, 1024)
        bm.get_bucket(1920, 1080)
        
        assert bm._distribution["1024x1024"] == 2
        total = sum(bm._distribution.values())
        assert total == 3

    def test_reset_distribution(self):
        """reset_distribution clears all counters."""
        bm = BucketManager(base_resolutions=1024)
        bm.get_bucket(1024, 1024)
        assert sum(bm._distribution.values()) > 0
        bm.reset_distribution()
        assert sum(bm._distribution.values()) == 0

    def test_log_distribution_runs(self):
        """log_distribution doesn't crash (smoke test)."""
        bm = BucketManager(base_resolutions=1024)
        bm.get_bucket(1920, 1080)
        bm.get_bucket(768, 1280)
        bm.log_distribution()  # Should not raise


# ── Latent Tests ─────────────────────────────────────────────────────────


class MockVAE(nn.Module):
    """Minimal VAE mock that takes [B,3,H,W] → [B,4,H//8,W//8]."""
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self._dtype = dtype
        self.config = MagicMock()
        self.config.scaling_factor = 0.18215
        self.config.shift_factor = None
        self.config.latents_mean = None
        self.config.latents_std = None
    
    @property
    def dtype(self):
        return self._dtype
    
    def encode(self, x):
        b, c, h, w = x.shape
        out = torch.randn(b, 4, h // 8, w // 8, dtype=self._dtype)
        result = MagicMock()
        result.latent_dist.sample.return_value = out
        return result


class MockFluxVAE(nn.Module):
    """Mock Flux/BFL VAE that returns raw tensor."""
    def __init__(self, dtype=torch.float16):
        super().__init__()
        self._dtype = dtype
    
    @property
    def dtype(self):
        return self._dtype

    def encode(self, x):
        b, c, h, w = x.shape
        return torch.randn(b, 16, h // 8, w // 8, dtype=self._dtype)


class TestLatentManager:
    """Tests for LatentManager encoding and caching."""

    def test_encode_diffusers_vae(self):
        """Diffusers VAE output (latent_dist) is correctly handled."""
        vae = MockVAE()
        lm = LatentManager(vae, device="cpu")
        images = torch.randn(2, 3, 512, 512)
        latents = lm.encode_and_cache_batch(images, ["a", "b"])
        
        assert latents.shape == (2, 4, 64, 64)
        assert latents.dtype == torch.float32

    def test_encode_flux_vae(self):
        """Flux/BFL VAE that returns raw tensor is handled."""
        vae = MockFluxVAE()
        lm = LatentManager(vae, device="cpu")
        images = torch.randn(2, 3, 512, 512)
        latents = lm.encode_and_cache_batch(images, ["a", "b"])
        
        assert latents.shape == (2, 16, 64, 64)
        assert latents.dtype == torch.float16

    def test_scaling_factor_from_config(self):
        """Scaling factor is read from VAE config."""
        vae = MockVAE()
        lm = LatentManager(vae, device="cpu")
        assert lm.scaling_factor == 0.18215

    def test_scaling_factor_fallback(self):
        """If no config, scaling factor defaults to 1.0."""
        vae = MockFluxVAE()
        lm = LatentManager(vae, device="cpu")
        assert lm.scaling_factor == 1.0

    def test_cache_roundtrip(self, tmp_path):
        """Encode → save → load cycle produces identical latents."""
        vae = MockVAE()
        cache_dir = str(tmp_path / "cache")
        lm = LatentManager(vae, device="cpu", cache_dir=cache_dir)

        images = torch.randn(2, 3, 256, 256)
        original = lm.encode_and_cache_batch(images, ["img1", "img2"])
        
        # Verify files created
        assert os.path.exists(os.path.join(cache_dir, "img1.safetensors"))
        assert os.path.exists(os.path.join(cache_dir, "img2.safetensors"))
        
        # Reload
        loaded = lm.load_cached_latents(["img1", "img2"])
        assert loaded is not None
        assert torch.allclose(original.cpu(), loaded.cpu())

    def test_cache_returns_none_on_missing(self, tmp_path):
        """load_cached_latents returns None if any file is missing."""
        vae = MockVAE()
        cache_dir = str(tmp_path / "cache")
        lm = LatentManager(vae, device="cpu", cache_dir=cache_dir)
        
        result = lm.load_cached_latents(["nonexistent_img"])
        assert result is None

    def test_cache_per_item_dirs(self, tmp_path):
        """Per-item cache_dirs work correctly."""
        vae = MockVAE()
        lm = LatentManager(vae, device="cpu")
        
        dir_a = str(tmp_path / "dir_a")
        dir_b = str(tmp_path / "dir_b")
        
        images = torch.randn(2, 3, 256, 256)
        lm.encode_and_cache_batch(images, ["a", "b"], cache_dirs=[dir_a, dir_b])
        
        assert os.path.exists(os.path.join(dir_a, "a.safetensors"))
        assert os.path.exists(os.path.join(dir_b, "b.safetensors"))

    def test_resolve_cache_dir(self):
        """Static helper builds the expected cache path."""
        result = LatentManager.resolve_cache_dir("/data/my_dataset", "sdxl_base", "1.0.0", "1024x1024")
        expected = os.path.join("/data/my_dataset", ".cache", "sdxl_base", "1.0.0", "latents", "original", "1024x1024")
        assert result == expected

    def test_mirror_dir(self, tmp_path):
        """Mirror directory receives a copy of cached latents."""
        vae = MockVAE()
        cache_dir = str(tmp_path / "primary")
        mirror = str(tmp_path / "mirror")
        lm = LatentManager(vae, device="cpu", cache_dir=cache_dir)
        
        images = torch.randn(1, 3, 256, 256)
        lm.encode_and_cache_batch(images, ["test"], mirror_dir=mirror)
        
        assert os.path.exists(os.path.join(cache_dir, "test.safetensors"))
        assert os.path.exists(os.path.join(mirror, "test.safetensors"))
