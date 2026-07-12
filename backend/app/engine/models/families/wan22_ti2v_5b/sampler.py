"""WAN 2.2 TI2V-5B sampler — subclasses :class:`WanVideoSamplerBase`.

The fp32 FlowMatchEuler trajectory, ``SampleArtifact`` output, and CFG combine
all live in the shared base and need no changes. Two things DO need overriding
because the base hardcodes WAN 2.1/2.2-A14B's VAE constants
(``WAN_VAE_SPATIAL=8``, 16 latent channels) which are WRONG for the 5B's new
higher-compression VAE (spatial 16, ``z_dim=48``):

- :meth:`_create_initial_noise` — 48 channels, spatial÷16 (not 16 channels,÷8).
- ``output_fps`` — 24 (the TI2V-5B model-card native fps), not WAN 2.1/2.2's 16.

Like ``wan21``/``wan22``, in-training preview sampling runs the plain T2V path
(no first-frame conditioning) even on an i2v-configured run — the base
``denoise()`` already passes ``encoder_hidden_states_image=None`` and a scalar
per-batch timestep unconditionally, which is exactly TI2V-5B's t2v forward
contract, so no override is needed there.
"""

from __future__ import annotations

import torch
from torch import Tensor

from app.engine.models.families.wan_shared.sampler_base import (
    WAN_VAE_TEMPORAL,
    WanVideoSamplerBase,
)

# TI2V-5B's new high-compression VAE (vs wan21/wan22's spatial=8, z_dim=16).
_TI2V5B_VAE_SPATIAL = 16
_TI2V5B_Z_DIM = 48


class Wan22Ti2v5bSampler(WanVideoSamplerBase):
    """WAN 2.2 TI2V-5B video sampler (T2V preview path)."""

    output_fps: float = 24.0

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator
    ) -> Tensor:
        """Create a 5D noise latent ``[1, 48, F, H/16, W/16]`` for the clip."""
        num_frames = self._effective_sample_frames(17, "4n+1")
        latent_f = (num_frames - 1) // WAN_VAE_TEMPORAL + 1
        lat_h = height // _TI2V5B_VAE_SPATIAL
        lat_w = width // _TI2V5B_VAE_SPATIAL
        shape = (1, _TI2V5B_Z_DIM, latent_f, lat_h, lat_w)
        return torch.randn(
            shape, generator=generator, device=self.device, dtype=torch.float32
        )
