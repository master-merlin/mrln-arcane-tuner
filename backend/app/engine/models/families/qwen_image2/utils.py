"""Qwen-Image 2.1 latent arithmetic.

The 2.1 transformer consumes latents UNPATCHED (``patch_size: 1`` in
``transformer/config.json``), so packing is a plain spatial flatten
``[B, C, H, W] -> [B, H*W, C]`` -- unlike ``qwen_image`` (2512), which packs
2x2 patches. The spatial compression is upstream's constant, not a value
derived from the VAE config.

Evidence: diffusers ``6256aa7666`` ``pipeline_qwenimage21.py`` --
``vae_scale_factor = 16`` and ``_pack_latents`` / ``_unpack_latents``, kept in
``vendor/pipeline_qwenimage21.py`` and used as the oracle by
``tests/test_qwen_image2_definition.py``.
"""

from __future__ import annotations

from torch import Tensor

# One latent token covers a 16x16 pixel tile (upstream QwenImage21PipelineHelpers).
VAE_SCALE_FACTOR = 16


def latent_grid(height: int, width: int) -> tuple[int, int]:
    """Latent ``(h, w)`` for a pixel ``height`` x ``width``, upstream's rounding.

    ``prepare_latents`` rounds each side down to a multiple of
    ``2 * VAE_SCALE_FACTOR`` pixels before dividing, and so does this.
    """
    return (
        2 * (int(height) // (VAE_SCALE_FACTOR * 2)),
        2 * (int(width) // (VAE_SCALE_FACTOR * 2)),
    )


def pack_latents(latents: Tensor) -> Tensor:
    """``[B, C, H, W] -> [B, H*W, C]`` (a plain spatial flatten, no patching)."""
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).transpose(1, 2)


def unpack_latents(packed: Tensor, height: int, width: int) -> Tensor:
    """``[B, H*W, C] -> [B, C, H, W]`` for a LATENT ``height`` x ``width``."""
    b, _, c = packed.shape
    return packed.transpose(1, 2).reshape(b, c, height, width)
