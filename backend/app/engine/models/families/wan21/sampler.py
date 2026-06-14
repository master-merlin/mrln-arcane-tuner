"""WAN 2.1 sampler — subclasses :class:`WanVideoSamplerBase`.

The fp32 FlowMatchEuler denoise trajectory + ``SampleArtifact`` (16 fps mp4)
output all live in the shared base. WAN 2.1 only picks the resolution shift
(3.0 at 480p, 5.0 at 720p) from the requested sample height.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.engine.models.families.wan_shared.sampler_base import WanVideoSamplerBase

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.engine.core.pipeline import GenericTrainingPipeline


class Wan21Sampler(WanVideoSamplerBase):
    """WAN 2.1 video sampler (T2V / I2V)."""

    def __init__(self, pipeline: GenericTrainingPipeline) -> None:
        super().__init__(pipeline)
        # Resolution-dependent shift: 5.0 for >=720p, else 3.0.
        try:
            res = int((pipeline.config.get("resolutions") or [480])[0])
        except (TypeError, ValueError, IndexError):
            res = 480
        self.shift = 5.0 if res >= 720 else 3.0
