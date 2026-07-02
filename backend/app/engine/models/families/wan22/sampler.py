"""WAN 2.2 sampler — two-stage dual-expert fp32 FlowMatchEuler denoise.

WAN 2.2 samples in two stages along the denoise trajectory: the **high-noise
expert** drives steps where the current sigma (timestep fraction) ``>= boundary``
and the **low-noise expert** drives steps below it. The trajectory (sigma math +
latent accumulation) stays in **fp32 with NO autocast** — only the transformer
forward may run in the model dtype. This is the shared autocast-collapse contract
(:func:`assert_no_autocast_collapse`), reused via the wan_shared base's
:meth:`euler_integrate` / :meth:`build_denoise`.

In ``swap`` mode the experts are migrated across CPU/GPU via the router's swap
helper at most once per sample pass (a single boundary crossing per descending
schedule), so a full clip costs one swap, not one per step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from app.engine.models.families.wan_shared.sampler_base import (
    WAN_FLOWMATCH_SCALE,
    WanVideoSamplerBase,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.engine.core.pipeline import GenericTrainingPipeline


class Wan22Sampler(WanVideoSamplerBase):
    """WAN 2.2 dual-expert video sampler (T2V / I2V)."""

    def __init__(self, pipeline: GenericTrainingPipeline) -> None:
        super().__init__(pipeline)
        # Resolution-dependent shift: 5.0 for >=720p, else 3.0.
        try:
            res = int((pipeline.config.get("resolutions") or [480])[0])
        except (TypeError, ValueError, IndexError):
            res = 480
        self.shift = 5.0 if res >= 720 else 3.0
        driver = getattr(pipeline, "driver", None)
        self.boundary: float = float(getattr(driver, "boundary", 0.875))

    # ── Two-stage dual-expert denoise (fp32 trajectory) ───────────────────

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Any:
        """fp32 FlowMatchEuler denoise that switches expert at the boundary.

        Each Euler step uses the high expert while ``sigma >= boundary`` and the
        low expert below it. The sigma math + latent accumulation run in fp32
        (via the shared :meth:`euler_integrate`); only the per-step transformer
        forward uses the model dtype.
        """
        driver = self.pipeline.driver
        high = getattr(driver, "transformer_high", None) or driver.get_primary_model()
        low = getattr(driver, "transformer_low", None) or high

        # Ensure both experts are on-device for the pass (resident); in swap mode
        # place_experts_for_start already staged them — to() here is idempotent.
        for m in (high, low):
            if m is not None:
                self._ensure_transformer_on_device(m)

        # Both experts are MIXED-dtype under mixed-precision training (bf16 bulk
        # + fp32 scale_shift_table/time_embedder), so a single input cast can't
        # satisfy every layer (``next(parameters()).dtype`` is just whichever is
        # first). Run the forward in the SAME autocast regime as training (per-op
        # casting); the Euler trajectory stays fp32 OUTSIDE the autocast (in
        # ``euler_integrate``), preserving the no-collapse contract.
        autocast_dtype = getattr(self.pipeline, "autocast_dtype", None) or torch.bfloat16
        device_type = torch.device(self.device).type
        sigmas = self._build_sigmas(num_steps).to(self.device)
        text = prompt_embedding

        # Classifier-free guidance (see WanVideoSamplerBase.denoise): contrast a
        # conditional vs UNCONDITIONAL velocity when guidance_scale > 1,
        # v = v_u + scale * (v_c - v_u). Both branches at a given step share the
        # SAME sigma → the SAME boundary-based expert selection, so the uncond
        # forward always routes through the expert the cond forward used.
        # guidance_scale <= 1 keeps the single conditional forward.
        cfg_on = guidance_scale is not None and float(guidance_scale) > 1.0
        text_uncond = None
        if cfg_on:
            neg_text = str(self.config.get("sample_negative_prompt", "") or "")
            text_uncond = self.encode_prompt(neg_text)

        def _forward(x: Tensor, sigma: Tensor, cond: Any) -> Tensor:
            # sigma is the [0,1] timestep fraction; pick the expert by boundary
            # (boundary is the same [0,1] fraction → matches the router's
            # boundary*1000 split in training). The transformer itself is
            # conditioned on the RAW [0,1000] timestep (sigma*1000), like the
            # diffusers WanPipeline — feeding the bare sigma caused pure noise.
            expert = high if float(sigma) >= self.boundary else low
            t = (sigma * WAN_FLOWMATCH_SCALE).reshape(1).expand(x.shape[0])
            with torch.no_grad(), torch.autocast(
                device_type=device_type, dtype=autocast_dtype
            ):
                out = expert(
                    hidden_states=x,
                    timestep=t,
                    encoder_hidden_states=cond,
                    encoder_hidden_states_image=None,
                    return_dict=False,
                )
            return out[0] if isinstance(out, tuple) else out

        def _velocity(x: Tensor, sigma: Tensor) -> Tensor:
            v_c = _forward(x, sigma, text)
            if not cfg_on:
                return v_c
            # Combine in fp32; the uncond forward shares the cond regime and
            # (same sigma) the same expert.
            v_c = v_c.to(torch.float32)
            v_u = _forward(x, sigma, text_uncond).to(torch.float32)
            return v_u + guidance_scale * (v_c - v_u)

        # euler_integrate forces the trajectory to fp32 (no autocast wrapper).
        return self.euler_integrate(noise, sigmas, _velocity)
