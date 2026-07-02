"""WAN 2.1 sampler — subclasses :class:`WanVideoSamplerBase`.

The fp32 FlowMatchEuler denoise trajectory, ``SampleArtifact`` (16 fps mp4)
output, and the resolution-dependent shift (3.0 at 480p, 5.0 at 720p) all live
in the shared base — WAN 2.1 needs no overrides at all.
"""

from __future__ import annotations

from app.engine.models.families.wan_shared.sampler_base import WanVideoSamplerBase


class Wan21Sampler(WanVideoSamplerBase):
    """WAN 2.1 video sampler (T2V / I2V)."""
