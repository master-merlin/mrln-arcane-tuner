"""Tests for LatentManager content-addressed filenames and cache coverage."""

import os
import tempfile

import pytest
import torch

from app.engine.components.latents import LatentManager


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_image(tmp_dir):
    """Create a small dummy image file and return its path."""
    path = os.path.join(tmp_dir, "photo_001.jpg")
    content = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # fake JPEG header + padding
    with open(path, "wb") as f:
        f.write(content)
    return path


@pytest.fixture
def sample_image_altered(tmp_dir):
    """Create a different image with the SAME filename as sample_image."""
    path = os.path.join(tmp_dir, "photo_001.jpg")
    content = b"\xff\xd8\xff\xe0" + b"\x01" * 100  # different content
    with open(path, "wb") as f:
        f.write(content)
    return path


# ── hash_source_file ─────────────────────────────────────────────────────

class TestHashSourceFile:
    def test_deterministic(self, sample_image):
        h1 = LatentManager.hash_source_file(sample_image)
        h2 = LatentManager.hash_source_file(sample_image)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_dir):
        path_a = os.path.join(tmp_dir, "a.jpg")
        path_b = os.path.join(tmp_dir, "b.jpg")
        with open(path_a, "wb") as f:
            f.write(b"content_a")
        with open(path_b, "wb") as f:
            f.write(b"content_b")
        assert LatentManager.hash_source_file(path_a) != LatentManager.hash_source_file(path_b)

    def test_returns_hex_string(self, sample_image):
        h = LatentManager.hash_source_file(sample_image)
        assert len(h) == 64  # SHA-256 hex = 64 chars
        int(h, 16)  # Should not raise — valid hex


# ── latent_filename ──────────────────────────────────────────────────────

class TestLatentFilename:
    def test_format(self, sample_image):
        fname = LatentManager.latent_filename("photo_001", sample_image)
        assert fname.endswith(".safetensors")
        assert fname.startswith("photo_001_")
        # Last 16 hex chars before .safetensors is the hash
        stem = fname.replace(".safetensors", "")
        # Hash is the last segment after the final underscore
        hash_part = stem.rsplit("_", 1)[-1]
        assert len(hash_part) == 16
        int(hash_part, 16)  # Valid hex

    def test_deterministic(self, sample_image):
        f1 = LatentManager.latent_filename("id", sample_image)
        f2 = LatentManager.latent_filename("id", sample_image)
        assert f1 == f2

    def test_different_source_different_name(self, tmp_dir):
        p1 = os.path.join(tmp_dir, "a.jpg")
        p2 = os.path.join(tmp_dir, "b.jpg")
        with open(p1, "wb") as f:
            f.write(b"aaa")
        with open(p2, "wb") as f:
            f.write(b"bbb")
        f1 = LatentManager.latent_filename("same_id", p1)
        f2 = LatentManager.latent_filename("same_id", p2)
        assert f1 != f2

    def test_replaced_image_invalidates_cache(self, tmp_dir):
        """Replacing image content (same name) must produce a different filename."""
        path = os.path.join(tmp_dir, "photo.jpg")
        with open(path, "wb") as f:
            f.write(b"original pixels")
        fname_before = LatentManager.latent_filename("photo", path)

        with open(path, "wb") as f:
            f.write(b"edited pixels")
        fname_after = LatentManager.latent_filename("photo", path)

        assert fname_before != fname_after


# ── check_cache_coverage with source_paths ───────────────────────────────

class TestCheckCacheCoverageContentAddressed:
    def test_coverage_with_source_paths(self, tmp_dir, sample_image):
        """Uses content-addressed filenames when source_paths provided."""
        cache_dir = os.path.join(tmp_dir, "cache")
        os.makedirs(cache_dir)

        # Pre-create the expected cache file with content-addressed name
        fname = LatentManager.latent_filename("photo_001", sample_image)
        dummy_tensor = torch.zeros(4, 8, 8)
        from safetensors.torch import save_file
        save_file({"latents": dummy_tensor}, os.path.join(cache_dir, fname))

        # Create a mock LatentManager (no VAE needed for coverage check)
        mgr = LatentManager.__new__(LatentManager)
        mgr.cache_dir = None

        cached, missing, missing_ids = mgr.check_cache_coverage(
            ids=["photo_001"],
            cache_dirs=[cache_dir],
            source_paths=[sample_image],
        )
        assert cached == 1
        assert missing == 0

    def test_legacy_fallback_without_source_paths(self, tmp_dir):
        """Without source_paths, falls back to bare {id}.safetensors."""
        cache_dir = os.path.join(tmp_dir, "cache")
        os.makedirs(cache_dir)
        from safetensors.torch import save_file
        save_file({"latents": torch.zeros(4, 8, 8)}, os.path.join(cache_dir, "img1.safetensors"))

        mgr = LatentManager.__new__(LatentManager)
        mgr.cache_dir = None

        cached, missing, _ = mgr.check_cache_coverage(
            ids=["img1"],
            cache_dirs=[cache_dir],
        )
        assert cached == 1
        assert missing == 0


# ── resolve_cache_dir ────────────────────────────────────────────────────

class TestResolveCacheDir:
    def test_includes_latents_segment(self):
        result = LatentManager.resolve_cache_dir("/data/ds", "flux-dev", "v1", "1024x1024")
        assert "latents" in result
        parts = result.replace("\\", "/").split("/")
        assert parts[-1] == "1024x1024"
        assert parts[-2] == "original"
        assert parts[-3] == "latents"

    def test_variant_segment(self):
        result = LatentManager.resolve_cache_dir("/data/ds", "flux-dev", "v1", "1024x1024", "masked")
        parts = result.replace("\\", "/").split("/")
        assert parts[-2] == "masked"
        assert parts[-3] == "latents"
