"""Vendored MiniMax-H3-Image transformer + VAE (diffusers PR #15230 @ 875373ac). See README.md.

Both classes are re-exported LAZILY (PEP 562): importing this package never imports torch or
diffusers, so it can never raise at import (ARCHITECTURE D1); a failure surfaces at first use.
"""

__all__ = ["MiniMaxH3ImageTransformer2DModel", "AutoencoderKLMiniMaxH3Image"]


def __getattr__(name: str):
    if name == "MiniMaxH3ImageTransformer2DModel":
        from .transformer_minimax_h3_image import MiniMaxH3ImageTransformer2DModel

        return MiniMaxH3ImageTransformer2DModel
    if name == "AutoencoderKLMiniMaxH3Image":
        from .autoencoder_kl_minimax_h3_image import AutoencoderKLMiniMaxH3Image

        return AutoencoderKLMiniMaxH3Image
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
