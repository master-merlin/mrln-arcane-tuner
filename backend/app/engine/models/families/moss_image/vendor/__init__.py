"""Vendored MOSS-Image transformer + scheduler (OpenMOSS-Team/MOSS-Image-Base @ 1ceede80). See README.md.

Both classes are re-exported LAZILY (PEP 562): importing this package never imports torch or
diffusers, so it can never raise at import (ARCHITECTURE D1); a failure surfaces at first use.
"""

__all__ = ["MOSSImageTransformer", "MOSSImageScheduler"]


def __getattr__(name: str):
    if name == "MOSSImageTransformer":
        from .transformer_moss_image import MOSSImageTransformer

        return MOSSImageTransformer
    if name == "MOSSImageScheduler":
        from .scheduler_moss_image import MOSSImageScheduler

        return MOSSImageScheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
