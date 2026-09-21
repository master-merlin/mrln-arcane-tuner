"""H3 pixel adapter — the family's ``[-1, 1]`` <-> ImageNet-space seam (plan row 2.0).

The H3 video VAE (diffusers ``AutoencoderKLMiniMaxH3``) takes ImageNet-normalised
RGB over a ``[0, 1]`` base range and returns the same space: ``encode`` expects
``(pixel01 - mean) / std`` and ``decode`` emits it, so a caller applies
``sample * std + mean`` and clamps to ``[0, 1]`` (diffusers
``autoencoder_kl_minimax_h3.py:512-515``). The engine's contract is ``[-1, 1]``
everywhere (``components/video.py`` produces it, ``components/latents.py`` hands
it to ``vae.encode`` unchanged) and NO per-family pixel seam exists, so the
conversion lives here, at the family boundary: the loader wraps the visual VAE
ONCE (``loader.py``, ``post_load_hook``) and every consumer — ``LatentManager``,
the sampler, GATE-2's PSNR — sees an ordinary ``[-1, 1]`` VAE. The audio VAE is
untouched.

``PIXEL_ADAPTER_VERSION`` is an input of the latent-cache key (row 2.3, D10): a
change to this conversion invalidates every cached video latent.
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
    """Wrap the H3 visual VAE so it speaks the engine's ``[-1, 1]`` pixels.

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
