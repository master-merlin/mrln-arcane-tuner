"""Vendored FLUX.3 transformer (diffusers PR #15046 @ ec288418). See README.md for upstream + licence.

``Flux3Transformer2DModel`` is re-exported LAZILY (PEP 562): importing this package never imports
diffusers, so it can never raise at import (ARCHITECTURE D1); a failure surfaces at first use.
"""

__all__ = ["Flux3Transformer2DModel"]


def __getattr__(name: str):
    if name == "Flux3Transformer2DModel":
        from .transformer_flux3 import Flux3Transformer2DModel

        return Flux3Transformer2DModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
