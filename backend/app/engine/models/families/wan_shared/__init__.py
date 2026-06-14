"""Shared WAN building blocks (no ``family.py`` — invisible to the registry).

This package holds the reusable pieces both WAN 2.1 (phase B3) and WAN 2.2
(a later phase) depend on:

- :mod:`text_encoding` — UMT5-XXL prompt encoding compatible with
  ``TextEmbeddingCache`` (returns a raw ``[B, L, D]`` tensor).
- :mod:`vae_utils` — Wan-VAE ``latents_mean`` / ``latents_std`` channel
  normalization + the ``4n+1`` frame-count rule helpers.
- :mod:`i2v_conditioning` — first-frame VAE latent + 4-channel temporal mask
  builder producing the 36-channel I2V transformer input.
- :mod:`driver_base` — :class:`WanDriverBase`: flow-match ``add_noise`` lerp,
  ``forward_pass`` calling the WAN transformer, and the LoRA-target fallback.
- :mod:`sampler_base` — :class:`WanVideoSamplerBase`: a strictly-fp32
  FlowMatchEuler denoise loop producing a :class:`SampleArtifact` at 16 fps.

Deliberately contains NO ``family.py`` so ``ModelRegistry.discover_families``
never registers ``wan_shared`` as a family.
"""

from app.engine.models.families.wan_shared.driver_base import WanDriverBase
from app.engine.models.families.wan_shared.i2v_conditioning import (
    build_i2v_conditioning,
    build_temporal_mask,
)
from app.engine.models.families.wan_shared.sampler_base import WanVideoSamplerBase
from app.engine.models.families.wan_shared.vae_utils import (
    assert_frame_rule,
    denormalize_wan_latents,
    is_valid_frame_count,
    latent_frames_4x,
    normalize_wan_latents,
    wan_latent_stats,
)

__all__ = [
    "WanDriverBase",
    "WanVideoSamplerBase",
    "build_i2v_conditioning",
    "build_temporal_mask",
    "assert_frame_rule",
    "denormalize_wan_latents",
    "is_valid_frame_count",
    "latent_frames_4x",
    "normalize_wan_latents",
    "wan_latent_stats",
]
