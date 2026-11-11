"""H3 pixel adapter for the video family (plan row 2.0).

The wrapper itself (``[-1, 1]`` <-> ImageNet space) is shared with the image
family and lives in ``minimax_h3_shared.pixel_adapter``; the loader wraps the
visual VAE ONCE (``loader.py``, ``post_load_hook``) and every consumer sees an
ordinary ``[-1, 1]`` VAE. The audio VAE is untouched.

``PIXEL_ADAPTER_VERSION`` is an input of this family's latent-cache key (row
2.3, D10), read from THIS module at call time
(``MiniMaxH3Driver.latent_cache_fingerprint``): a change to the conversion
invalidates every cached video latent.
"""

from __future__ import annotations

from app.engine.models.families.minimax_h3_shared.pixel_adapter import (
    IMAGENET_MEAN as IMAGENET_MEAN,
    IMAGENET_STD as IMAGENET_STD,
    PIXEL_ADAPTER_VERSION as PIXEL_ADAPTER_VERSION,
    H3PixelAdaptedVAE as H3PixelAdaptedVAE,
)

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "PIXEL_ADAPTER_VERSION", "H3PixelAdaptedVAE"]
