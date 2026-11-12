"""Vendored MiniMax-H3-Image transformer + VAE (diffusers PR #15230 @ 875373ac). See README.md.

Both classes are re-exported LAZILY (PEP 562): importing this package never imports torch or
diffusers, so it can never raise at import (ARCHITECTURE D1); a failure surfaces at first use.

The checkpoint's native ``transformer/config.json`` names its class ``MiniMaxH3DiTModel``, and
the definition carries that harvested name as ``transformer._class_name``. The name resolves here
to the diffusers port the loader builds from ``transformer_diffusers/``, so whatever looks the
definition's class up by that name (the VRAM estimator's kept-module inventory) finds the model
that is trained. The native config keys the port shares (widths, depth, heads, patch size) carry
over; the rest take the port's defaults, which are the checkpoint's values.
"""

__all__ = ["MiniMaxH3ImageTransformer2DModel", "AutoencoderKLMiniMaxH3Image"]

#: The checkpoint's own class names, mapped to the vendored class that implements each.
NATIVE_CLASS_NAMES = {"MiniMaxH3DiTModel": "MiniMaxH3ImageTransformer2DModel"}


def __getattr__(name: str):
    name = NATIVE_CLASS_NAMES.get(name, name)
    if name == "MiniMaxH3ImageTransformer2DModel":
        from .transformer_minimax_h3_image import MiniMaxH3ImageTransformer2DModel

        return MiniMaxH3ImageTransformer2DModel
    if name == "AutoencoderKLMiniMaxH3Image":
        from .autoencoder_kl_minimax_h3_image import AutoencoderKLMiniMaxH3Image

        return AutoencoderKLMiniMaxH3Image
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
