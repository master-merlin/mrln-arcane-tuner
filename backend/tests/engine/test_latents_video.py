"""Tests for LatentManager's true 5D video path + image regression guard."""

import os

import torch
import torch.nn as nn
from unittest.mock import MagicMock

from app.engine.components.latents import LatentManager


# ── Fake 5D video VAE ────────────────────────────────────────────────────


class FakeWanVAE(nn.Module):
    """A stand-in for AutoencoderKLWan.

    Spatial downscale 8×, temporal downscale 4× (latent_f = (F-1)/4 + 1).
    Class NAME matters: _is_video_vae / temporal_downscale match by name.
    """

    __name__ = "AutoencoderKLWan"  # not used by MRO walk but harmless

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self._dtype = dtype
        self.config = MagicMock()
        self.config.scaling_factor = 1.0
        self.config.shift_factor = None
        self.config.latents_mean = None
        self.config.latents_std = None

    @property
    def dtype(self):
        return self._dtype

    def encode(self, x):
        # x is 5D [B, C, F, H, W]; compress spatially 8×, temporally 4×.
        b, c, f, h, w = x.shape
        lf = (f - 1) // 4 + 1
        out = torch.randn(b, 16, lf, h // 8, w // 8, dtype=self._dtype)
        result = MagicMock()
        result.latent_dist.sample.return_value = out
        return result


# Make the class NAME match the detector (MRO walk uses type().__mro__ names).
FakeWanVAE.__name__ = "AutoencoderKLWan"
FakeWanVAE.__qualname__ = "AutoencoderKLWan"


class MockImageVAE(nn.Module):
    """Minimal still-image VAE [B,3,H,W] → [B,4,H//8,W//8]."""

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


# ── Detection + temporal-downscale formula ───────────────────────────────


class TestVideoVaeDetection:
    def test_wan_detected_as_video_vae(self):
        lm = LatentManager(FakeWanVAE(), device="cpu")
        assert lm._is_video_vae() is True

    def test_image_vae_not_video(self):
        lm = LatentManager(MockImageVAE(), device="cpu")
        assert lm._is_video_vae() is False

    def test_temporal_downscale_wan_is_4(self):
        lm = LatentManager(FakeWanVAE(), device="cpu")
        assert lm.temporal_downscale() == 4

    def test_temporal_downscale_arch_override(self):
        lm = LatentManager(
            MockImageVAE(),
            device="cpu",
            arch_params={"vae_temporal_downsample": 8},
        )
        assert lm.temporal_downscale() == 8

    def test_latent_frames_formula(self):
        # (F-1)/4 + 1
        assert LatentManager.latent_frames(13, 4) == 4  # (12/4)+1
        assert LatentManager.latent_frames(1, 4) == 1
        assert LatentManager.latent_frames(17, 4) == 5
        # (F-1)/8 + 1 (LTX)
        assert LatentManager.latent_frames(17, 8) == 3
        # temporal=1 → identity
        assert LatentManager.latent_frames(9, 1) == 9


# ── 5D encode + cache ────────────────────────────────────────────────────


class TestVaeColocation:
    """encode_and_cache_batch must co-locate the VAE with the compute device.

    The VAE is offloaded to CPU after pre-caching (low_vram=True), but a
    train-loop cache MISS still routes through encode_and_cache_batch with the
    input already on the device. Without the JIT move the CUDA input meets CPU
    weights → "Input type (CUDABFloat16Type) and weight type (CPUBFloat16Type)
    should be the same" (the G1 rerun crash). Mirrors the audio-VAE / connector
    co-location pattern.
    """

    def test_encode_colocates_offloaded_vae(self):
        vae = FakeWanVAE()
        seen: list = []
        real_to = vae.to

        def _record_to(*args, **kwargs):
            if args:
                seen.append(args[0])
            elif "device" in kwargs:
                seen.append(kwargs["device"])
            return real_to(*args, **kwargs)

        vae.to = _record_to  # type: ignore[method-assign]

        lm = LatentManager(vae, device="cpu")
        lm.encode_and_cache_batch(torch.randn(1, 3, 13, 64, 64), ["clip0"])

        assert "cpu" in [str(d) for d in seen], (
            f"VAE not moved to the compute device before encode; .to() saw {seen}"
        )


class TestVideo5DEncode:
    def test_video_latent_stays_5d(self):
        """A genuine 5D video input produces a 5D cached latent (no squeeze)."""
        lm = LatentManager(FakeWanVAE(), device="cpu")
        # [B, C, F, H, W] = [1, 3, 13, 64, 64]
        clip = torch.randn(1, 3, 13, 64, 64)
        latents = lm.encode_and_cache_batch(clip, ["clip0"])
        # latent_f = (13-1)/4 + 1 = 4; spatial 64/8 = 8.
        assert latents.ndim == 5
        assert latents.shape == (1, 16, 4, 8, 8)

    def test_still_through_video_vae_stays_4d(self):
        """A 4D still fed to a video VAE keeps the cached latent 4D."""
        lm = LatentManager(FakeWanVAE(), device="cpu")
        still = torch.randn(2, 3, 64, 64)
        latents = lm.encode_and_cache_batch(still, ["a", "b"])
        assert latents.ndim == 4
        assert latents.shape == (2, 16, 8, 8)

    def test_validate_shape_5d_accepts_correct(self, caplog):
        lm = LatentManager(FakeWanVAE(), device="cpu")
        latents = torch.randn(1, 16, 4, 8, 8)  # correct for F=13, 64x64
        input_shape = torch.Size([1, 3, 13, 64, 64])
        # Should not log a mismatch warning.
        lm._validate_shape(latents, input_shape)  # no exception

    def test_video_cache_dir_carries_frames_and_fps(self, tmp_path):
        """The cache directory encodes the temporal slice for videos."""
        res_str = "512x512x13f16.0"
        cache_dir = LatentManager.resolve_cache_dir(
            str(tmp_path),
            "wan",
            "1.0.0",
            res_str,
            "original",
        )
        assert res_str in cache_dir.replace("\\", "/")

    def test_video_trim_in_filename_hash(self, tmp_path):
        """Two trims of the same source file produce distinct filenames."""
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"fake-video-bytes-fake-video-bytes")
        a = LatentManager.latent_filename("clip", str(src), extra_key="t0.0-1.0")
        b = LatentManager.latent_filename("clip", str(src), extra_key="t1.0-2.0")
        assert a != b
        assert a.startswith("clip_") and a.endswith(".safetensors")

    def test_video_latent_cache_roundtrip(self, tmp_path):
        """5D latent survives save → load with a trim extra_key."""
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"abcdef-clip-bytes")
        cache_dir = str(tmp_path / "cache")
        lm = LatentManager(FakeWanVAE(), device="cpu")

        clip = torch.randn(1, 3, 13, 64, 64)
        original = lm.encode_and_cache_batch(
            clip,
            ["clip"],
            cache_dirs=[cache_dir],
            source_paths=[str(src)],
            extra_keys=["t0.0-1.0"],
        )
        loaded = lm.load_cached_latents(
            ["clip"],
            cache_dirs=[cache_dir],
            source_paths=[str(src)],
            extra_keys=["t0.0-1.0"],
        )
        assert loaded is not None
        assert loaded.ndim == 5
        assert torch.allclose(original.cpu(), loaded.cpu())


# ── Image regression guard (byte-identical cache path) ───────────────────


class TestImageCachePathUnchanged:
    def test_image_filename_byte_identical_to_legacy(self, tmp_path):
        """An image item's cache filename is identical with/without the new
        extra_key parameter (empty extra_key == legacy source-bytes-only hash).
        """
        src = tmp_path / "photo.jpg"
        src.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 200)

        # Legacy = hash of source bytes only.
        import hashlib

        legacy_hash = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
        expected_legacy = f"photo_001_{legacy_hash}.safetensors"

        # New code path with empty extra_key must match byte-for-byte.
        produced = LatentManager.latent_filename("photo_001", str(src))
        produced_explicit_empty = LatentManager.latent_filename(
            "photo_001",
            str(src),
            extra_key="",
        )
        assert produced == expected_legacy
        assert produced_explicit_empty == expected_legacy

    def test_image_cache_dir_byte_identical(self):
        """Image cache directory (no temporal suffix) is unchanged."""
        produced = LatentManager.resolve_cache_dir(
            "/data/ds",
            "sdxl",
            "1.0.0",
            "1024x1024",
            "original",
        )
        expected = os.path.join(
            "/data/ds",
            ".cache",
            "sdxl",
            "1.0.0",
            "latents",
            "original",
            "1024x1024",
        )
        assert produced == expected

    def test_image_latent_save_load_unchanged(self, tmp_path):
        """A still image round-trips with bare {id}.safetensors (no source_paths)."""
        cache_dir = str(tmp_path / "cache")
        lm = LatentManager(MockImageVAE(), device="cpu")
        images = torch.randn(2, 3, 256, 256)
        original = lm.encode_and_cache_batch(
            images, ["img1", "img2"], cache_dirs=[cache_dir, cache_dir]
        )
        assert os.path.exists(os.path.join(cache_dir, "img1.safetensors"))
        loaded = lm.load_cached_latents(
            ["img1", "img2"], cache_dirs=[cache_dir, cache_dir]
        )
        assert torch.allclose(original.cpu(), loaded.cpu())


class TestSliceLatentWindow:
    def test_slices_frame_axis_4d(self):
        # [C, f, h, w] per-item latent → window of 2 frames at start 1.
        latent = torch.arange(4 * 5 * 2 * 2, dtype=torch.float32).reshape(4, 5, 2, 2)
        out = LatentManager.slice_latent_window(latent, window_frames=2, start=1)
        assert out.shape == (4, 2, 2, 2)
        assert torch.equal(out, latent[:, 1:3, :, :])

    def test_slices_frame_axis_5d(self):
        # [B, C, f, h, w] batched latent → frame axis is dim 2.
        latent = torch.randn(3, 4, 6, 2, 2)
        out = LatentManager.slice_latent_window(latent, window_frames=3, start=2)
        assert out.shape == (3, 4, 3, 2, 2)
        assert torch.equal(out, latent[:, :, 2:5, :, :])


class TestLoadCachedLatentWindows:
    def _cache_full_clip(self, tmp_path, frames):
        """Encode + cache a fake full-clip latent; return (lm, cache_dir, src)."""
        cache_dir = str(tmp_path / "cache")
        lm = LatentManager(FakeWanVAE(), device="cpu")
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"full-clip-bytes")
        # FakeWanVAE: latent_f = (F-1)//4 + 1. F=33 → 9 latent frames.
        clip = torch.randn(1, 3, frames, 64, 64)
        lm.encode_and_cache_batch(
            clip,
            ["clip"],
            cache_dirs=[cache_dir],
            source_paths=[str(src)],
            extra_keys=["t0.0-None-slideF33"],
        )
        return lm, cache_dir, str(src)

    def test_loads_and_slices_to_window(self, tmp_path):
        lm, cache_dir, src = self._cache_full_clip(tmp_path, frames=33)  # 9 latent frames
        gen = torch.Generator().manual_seed(0)
        out = lm.load_cached_latent_windows(
            ["clip"], [cache_dir], source_paths=[src],
            extra_keys=["t0.0-None-slideF33"], window_frames=3, generator=gen,
        )
        assert out is not None
        # [B, C, window, h, w]
        assert out.shape == (1, 16, 3, 8, 8)

    def test_returns_none_on_miss(self, tmp_path):
        lm, cache_dir, src = self._cache_full_clip(tmp_path, frames=33)
        out = lm.load_cached_latent_windows(
            ["clip"], [cache_dir], source_paths=[src],
            extra_keys=["t0.0-None-slideF999"], window_frames=3,  # wrong key → miss
        )
        assert out is None

    def test_window_within_bounds_over_many_draws(self, tmp_path):
        lm, cache_dir, src = self._cache_full_clip(tmp_path, frames=33)  # 9 latent frames
        full = lm.load_cached_latents(
            ["clip"], [cache_dir], source_paths=[src], extra_keys=["t0.0-None-slideF33"]
        )[0]  # [C, 9, h, w]
        for seed in range(20):
            gen = torch.Generator().manual_seed(seed)
            out = lm.load_cached_latent_windows(
                ["clip"], [cache_dir], source_paths=[src],
                extra_keys=["t0.0-None-slideF33"], window_frames=4, generator=gen,
            )[0]  # [C, 4, h, w]
            assert out.shape[1] == 4
            matched = any(
                torch.equal(out, full[:, s : s + 4, :, :]) for s in range(9 - 4 + 1)
            )
            assert matched, f"window not a contiguous slice (seed {seed})"
