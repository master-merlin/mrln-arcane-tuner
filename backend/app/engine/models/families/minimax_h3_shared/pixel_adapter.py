"""MiniMax-H3 pixel adapter: the ``[-1, 1]`` <-> ImageNet-space VAE seam.

The H3 VAEs (diffusers ``AutoencoderKLMiniMaxH3`` for video and its image
sibling) take ImageNet-normalised RGB over a ``[0, 1]`` base range and return
the same space: ``encode`` expects ``(pixel01 - mean) / std`` and ``decode``
emits it, so a caller applies ``sample * std + mean`` and clamps to ``[0, 1]``
(diffusers ``autoencoder_kl_minimax_h3.py:512-515``). The engine's contract is
``[-1, 1]`` everywhere (``components/video.py`` produces it,
``components/latents.py`` hands it to ``vae.encode`` unchanged) and NO
per-family pixel seam exists, so the conversion lives at the family boundary:
a loader wraps the visual VAE ONCE and every consumer (``LatentManager``, the
sampler, a PSNR check) sees an ordinary ``[-1, 1]`` VAE. The statistics are
broadcast over any trailing dims, so an image ``(B, 3, H, W)`` and a video
``(B, 3, T, H, W)`` tensor take the same path.

``PIXEL_ADAPTER_VERSION`` is an input of a family's latent-cache key: a change
to this conversion must invalidate every cached latent. Each family reads it
through its own ``pixel_adapter`` module.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

PIXEL_ADAPTER_VERSION = 1

# ImageNet statistics the H3 VAE was trained against (diffusers
# ``autoencoder_kl_minimax_h3.py:514``).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class H3PixelAdaptedVAE(nn.Module):
    """Wrap an H3 visual VAE so it speaks the engine's ``[-1, 1]`` pixels.

    ``encode(x)``: ``inner.encode(((x + 1) / 2 - mean) / std)``.
    ``decode(z)``: ``((inner.decode(z).sample * std + mean).clamp(0, 1)) * 2 - 1``.

    Everything else (``config``, ``dtype``, ``device``, ``to()``,
    ``latents_mean`` / ``latents_std``, ``enable_tiling`` / ``disable_tiling``,
    ``spatial_compression_ratio``, ...) is the inner VAE's, so
    ``LatentManager.normalize_latents`` / ``denormalize_latents`` and the
    sampler see an ordinary VAE.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        if isinstance(inner, H3PixelAdaptedVAE):
            raise TypeError("H3PixelAdaptedVAE wraps the visual VAE once; got a wrapper")
        self.inner = inner

    # -- pixel conversion ------------------------------------------------

    def _stats(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (1, 3) + (1,) * (like.ndim - 2)
        mean = torch.tensor(IMAGENET_MEAN, device=like.device, dtype=like.dtype).view(shape)
        std = torch.tensor(IMAGENET_STD, device=like.device, dtype=like.dtype).view(shape)
        return mean, std

    def encode(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        mean, std = self._stats(x)
        return self.inner.encode(((x + 1.0) / 2.0 - mean) / std, *args, **kwargs)

    def decode(self, z: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        out = self.inner.decode(z, *args, **kwargs)
        sample = out.sample if hasattr(out, "sample") else out[0]
        mean, std = self._stats(sample)
        pixels01 = (sample * std + mean).clamp(0.0, 1.0)
        pixels = pixels01 * 2.0 - 1.0
        if hasattr(out, "sample"):
            out.sample = pixels
            return out
        return (pixels,) + tuple(out[1:])

    # -- an ordinary VAE to every consumer --------------------------------

    @property
    def config(self) -> Any:
        return self.inner.config

    @property
    def dtype(self) -> torch.dtype:
        return getattr(self.inner, "dtype", None) or next(self.inner.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return getattr(self.inner, "device", None) or next(self.inner.parameters()).device

    @property
    def latents_mean(self) -> Any:
        return getattr(self.inner, "latents_mean", None)

    @property
    def latents_std(self) -> Any:
        return getattr(self.inner, "latents_std", None)

    def enable_tiling(self, *args: Any, **kwargs: Any) -> None:
        self.inner.enable_tiling(*args, **kwargs)

    def disable_tiling(self) -> None:
        self.inner.disable_tiling()

    def __getattr__(self, name: str) -> Any:
        # nn.Module resolves submodules/parameters/buffers first; anything
        # else (spatial_compression_ratio, use_tiling, ...) is the inner's.
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self._modules.get("inner")
            if inner is None:
                raise
            return getattr(inner, name)
