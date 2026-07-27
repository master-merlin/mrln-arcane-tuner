"""Task W2.T2 — atomic latent/TE cache writes + corrupt-file fallback.

Two defects compose into a poison pill:

* ``_save_to_disk`` / ``TextEmbeddingCache.save`` wrote the ``.safetensors``
  cache file NON-atomically via a raw ``save_file``. A kill/crash mid-write
  leaves a truncated file at the FINAL path.
* The load sites caught only ``(OSError, KeyError)``. In the installed
  safetensors 0.8.0, ``SafetensorError`` subclasses ``Exception`` DIRECTLY
  (NOT ``OSError``), so a truncated file raised straight through the train
  loop and killed the job — and since ``os.path.exists`` coverage checks
  count the truncated file as "cached", every subsequent run crashed at the
  same step.

These tests pin: (1) a truncated cache file degrades to a MISS (``None``),
never a raise, on every load site; (2) a write that fails mid-serialization
never leaves a corrupted/partial file at the final path.
"""

from __future__ import annotations

import os

import pytest
import torch
from safetensors.torch import save_file

from app.engine.components.latents import LatentManager
from app.engine.components.text_embeddings import TextEmbeddingCache


def _bare_latent_manager() -> LatentManager:
    """A LatentManager with no VAE — sufficient for load/save-path tests."""
    lm = LatentManager.__new__(LatentManager)
    lm.cache_dir = None
    lm.device = "cpu"
    return lm


# ── Truncated cache file → MISS, not raise ────────────────────────────────


class TestTruncatedCacheFileFallsBackToMiss:
    def test_load_cached_latents_returns_none_on_truncated_file(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        # Truncated/garbage .safetensors (not a valid header) at the legacy
        # bare filename the load path will look for.
        (cache / "photo.safetensors").write_bytes(b"\x00" * 64)

        lm = _bare_latent_manager()
        out = lm.load_cached_latents(["photo"], [str(cache)])

        assert out is None  # miss → caller re-encodes and overwrites

    def test_load_cached_latent_windows_returns_none_on_truncated_file(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "clip.safetensors").write_bytes(b"\x00" * 64)

        lm = _bare_latent_manager()
        out = lm.load_cached_latent_windows(["clip"], [str(cache)], window_frames=2)

        assert out is None

    def test_text_embedding_cache_load_returns_none_on_truncated_file(self, tmp_path):
        """Already broad-catch (``except Exception``) prior to this task —
        pinned here as a regression lock, not a fix.
        """
        cache_dir = tmp_path / "te_cache"
        cache_dir.mkdir()
        fname = TextEmbeddingCache.caption_to_filename("a caption", "src")
        (cache_dir / fname).write_bytes(b"\x00" * 64)

        out = TextEmbeddingCache.load("a caption", str(cache_dir), "src")

        assert out is None


# ── Interrupted write must not corrupt the final path ─────────────────────


class TestInterruptedWriteDoesNotCorruptFinalPath:
    def test_latent_save_to_disk_interrupted_leaves_previous_file_intact(
        self, tmp_path, monkeypatch
    ):
        cache = tmp_path / "cache"
        cache.mkdir()
        path = cache / "img.safetensors"
        save_file({"latents": torch.zeros(2, 2)}, str(path))
        original_bytes = path.read_bytes()

        def _boom(*_args, **_kwargs):
            raise RuntimeError("simulated crash mid-serialize")

        monkeypatch.setattr("app.engine.utils.safe_save.save_file", _boom)

        lm = _bare_latent_manager()
        latents = torch.ones(1, 2, 2)  # batch of one item: "img"
        with pytest.raises(RuntimeError):
            lm._save_to_disk(latents, ["img"], cache_dirs=[str(cache)])

        # Final path untouched — no partial/corrupt overwrite — and no
        # leftover tmp file.
        assert path.read_bytes() == original_bytes
        assert not os.path.exists(str(path) + ".tmp")

    def test_latent_save_to_disk_interrupted_leaves_no_file_when_none_existed(
        self, tmp_path, monkeypatch
    ):
        cache = tmp_path / "cache"
        cache.mkdir()
        path = cache / "img.safetensors"

        def _boom(*_args, **_kwargs):
            raise RuntimeError("simulated crash mid-serialize")

        monkeypatch.setattr("app.engine.utils.safe_save.save_file", _boom)

        lm = _bare_latent_manager()
        latents = torch.ones(1, 2, 2)
        with pytest.raises(RuntimeError):
            lm._save_to_disk(latents, ["img"], cache_dirs=[str(cache)])

        assert not path.exists()
        assert not os.path.exists(str(path) + ".tmp")

    def test_text_embedding_save_interrupted_leaves_previous_file_intact(
        self, tmp_path, monkeypatch
    ):
        cache_dir = tmp_path / "te_cache"
        cache_dir.mkdir()
        caption = "a caption"
        fname = TextEmbeddingCache.caption_to_filename(caption, "src")
        path = cache_dir / fname
        save_file({"emb": torch.zeros(4)}, str(path))
        original_bytes = path.read_bytes()

        def _boom(*_args, **_kwargs):
            raise RuntimeError("simulated crash mid-serialize")

        monkeypatch.setattr("app.engine.utils.safe_save.save_file", _boom)

        with pytest.raises(RuntimeError):
            TextEmbeddingCache.save(caption, torch.ones(4), str(cache_dir), "src")

        assert path.read_bytes() == original_bytes
        assert not os.path.exists(str(path) + ".tmp")
