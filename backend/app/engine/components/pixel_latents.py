"""Pixel-passthrough LatentManager for pixel-space families (no VAE).

Extracted from ``families/hidream_o1/trainer.py`` so every pixel-space
family (hidream_o1, prx_pixel, ...) shares one implementation instead of
forking the bypass logic.

The base training loop calls ``latent_manager.encode_and_cache_batch()``
unconditionally on cache miss, which raises when ``vae=None``. Installing
this passthrough (via the family trainer's ``_configure_managers``
override) makes the base loop receive the raw pixel tensor untouched —
"latents" simply ARE the ``[B, C, H, W]`` pixel batch.
"""

from __future__ import annotations

import torch


class PixelPassthroughLatentManager:
    """Drop-in LatentManager replacement for pixel-space families.

    Behaviour:
    - ``load_cached_latents``: always returns ``None`` (cache miss path).
      Pixel-space families do not cache latents — the model processes
      pixels live.
    - ``encode_and_cache_batch``: returns the pixel tensor as-is. The
      base loop stores this as ``latents``; families that pull pixels from
      ``batch["images"]`` directly (hidream_o1) ignore the value, families
      that train on the loop's latents (prx_pixel) consume it verbatim.
    - ``check_cache_coverage``: reports all items as cached so
      ``_pre_cache_latents`` is a no-op.
    - ``latent_filename`` / ``_validate_shape``: delegated to a stub so
      ``_build_cache_manifest`` and similar helpers don't crash.
    """

    def load_cached_latents(
        self,
        ids: list[str],
        cache_dirs: list[str] | None = None,
        source_paths: list[str] | None = None,
        extra_keys: list[str] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        """Always report a cache miss — pixel-space has no latent cache.

        Must return None so the base loop falls into ``encode_and_cache_batch``,
        which returns the actual 4D ``batch["images"]`` tensor. The base loop
        later does ``latents.shape[1]`` for noise-offset shaping — that only
        works on the 4D image tensor, not on a length-N sentinel.

        ``device``/``dtype`` accepted (unused) for drop-in signature parity
        with ``LatentManager.load_cached_latents`` — the base training loop
        calls this polymorphically and passes them unconditionally.
        """
        return None

    def encode_and_cache_batch(
        self,
        image_batch: torch.Tensor,
        ids: list[str],
        cache_dirs: list[str] | None = None,
        source_paths: list[str] | None = None,
        extra_keys: list[str] | None = None,
    ) -> torch.Tensor:
        """Return pixel values unchanged — no VAE encoding needed."""
        return image_batch

    def check_cache_coverage(
        self,
        ids: list[str],
        cache_dirs: list[str],
        source_paths: list[str] | None = None,
        extra_keys: list[str] | None = None,
    ) -> tuple[int, int, list[str]]:
        """Report all items as cached so pre-cache step is skipped."""
        n = len(ids)
        return n, 0, []

    @staticmethod
    def latent_filename(img_id: str, source_path: str) -> str:
        """Stub — pixel-space families don't write latent files."""
        return f"{img_id}.safetensors"
