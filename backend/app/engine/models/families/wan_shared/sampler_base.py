"""Shared WAN video sampler base — strictly-fp32 FlowMatchEuler denoise.

:class:`WanVideoSamplerBase` produces short video clips during training as
:class:`SampleArtifact` objects (mp4 at 16 fps). The denoise TRAJECTORY runs in
fp32 with NO autocast around the loop — only the transformer forward may use the
model dtype. Wrapping the loop in ``autocast(bf16)`` collapses multi-step
sampling toward the conditional mean even when training is correct; this is THE
sampler-collapse gotcha that :func:`assert_no_autocast_collapse` guards.

The fp32 Euler integration is factored into :meth:`euler_integrate`, a pure
function over ``(x0, sigmas, velocity_fn)``. The precision-contract test drives
THIS REAL method with a ``LinearVelocityFakeTransformer`` velocity field and
compares the endpoint to the fp64 analytic solution — so the test exercises the
exact code path training/sampling uses, not a copy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog
import torch
from PIL import Image
from torch import Tensor

from app.engine.core.sampling import GenericSamplingPipeline, SampleArtifact
from app.engine.models.families.wan_shared.vae_utils import (
    WAN_VAE_SPATIAL,
    WAN_VAE_TEMPORAL,
    denormalize_wan_latents,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.engine.core.pipeline import GenericTrainingPipeline

logger = structlog.get_logger(__name__)

WAN_NATIVE_FPS = 16.0


class WanVideoSamplerBase(GenericSamplingPipeline):
    """Flow-match Euler video sampler (fp32 trajectory) for WAN families."""

    # Resolution shift for the FlowMatchEuler sigma schedule (3.0 @480p,
    # 5.0 @720p). Subclasses/definitions may override.
    shift: float = 3.0
    output_fps: float = WAN_NATIVE_FPS

    def __init__(self, pipeline: GenericTrainingPipeline) -> None:
        super().__init__(pipeline)
        self._scheduler = None

    # ── fp32 Euler integration core (the precision-critical path) ──────────

    @staticmethod
    def euler_integrate(
        x0: Tensor,
        sigmas: Tensor,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        """Integrate ``dx/dσ = v(x, σ)`` with forward Euler in fp32.

        The trajectory (sigma math + latent accumulation) is forced to fp32.
        ``velocity_fn(x, sigma)`` may internally run a bf16 transformer, but its
        result is cast back to fp32 before accumulation so the loop never
        accumulates in reduced precision (the autocast-collapse guard).

        Args:
            x0: Initial latent (any dtype; promoted to fp32 internally).
            sigmas: 1-D descending schedule (e.g. 1 → 0), ``steps + 1`` values.
            velocity_fn: ``(x_fp32, sigma_scalar_fp32) -> velocity`` callable.

        Returns:
            The fp32 endpoint after stepping through ``sigmas``.
        """
        x = x0.to(torch.float32)
        s = sigmas.to(torch.float32)
        for i in range(len(s) - 1):
            dt = (s[i + 1] - s[i]).to(torch.float32)
            v = velocity_fn(x, s[i]).to(torch.float32)
            x = x + dt * v
        return x

    def build_denoise(
        self, transformer: Any, **forward_kwargs: Any
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Return a ``denoise(x0, sigmas)`` closure over a transformer.

        Used by the precision-contract test: pass a
        ``LinearVelocityFakeTransformer`` and the returned callable runs the
        REAL :meth:`euler_integrate` fp32 loop, calling the transformer for the
        velocity at each step (forward may be model-dtype; accumulation is fp32).
        """

        def _velocity(x: Tensor, sigma: Tensor) -> Tensor:
            # The transformer forward may run in its own dtype; the integrator
            # casts the result back to fp32. We deliberately do NOT wrap this in
            # autocast — the trajectory stays fp32.
            t = sigma.reshape(1).to(x.dtype)
            out = transformer(x, t, **forward_kwargs)
            return out[0] if isinstance(out, tuple) else out

        def _denoise(x0: Tensor, sigmas: Tensor) -> Tensor:
            return self.euler_integrate(x0, sigmas, _velocity)

        return _denoise

    # ── Scheduler / sigma schedule ─────────────────────────────────────────

    def _build_sigmas(self, num_steps: int) -> Tensor:
        """Resolution-shifted FlowMatchEuler sigma schedule, descending 1 → 0.

        Uses the shared ``shifted_sigmas`` helper. The shift is resolved from
        config (``model_shift_fixed``, injected by the WAN contract) if present,
        otherwise falls back to the class-level ``self.shift`` attribute (3.0 @
        480p, 5.0 @ 720p). Computed in fp32.
        """
        from app.engine.strategies.sigma_schedule import shifted_sigmas

        shift = float(self.config.get("model_shift_fixed") or self.shift)
        return shifted_sigmas(num_steps, shift)

    # ── GenericSamplingPipeline hooks ──────────────────────────────────────

    def encode_prompt(self, prompt: str) -> Any:
        """Encode a prompt to ``[1, L, D]`` UMT5 embeddings via the trainer's
        CACHED ``encode_text``.

        Sampling runs after ``run_trainer`` offloads the UMT5 encoder
        (``driver.text_encoder`` → ``None``), so going through the driver
        directly would call ``None(...)`` → "'NoneType' object is not callable".
        The trainer's ``encode_text`` serves from the warm text cache instead
        (the expanded sample prompts are pre-cached by ``WanTextCacheMixin``), so
        it survives the offload. Caching off → the trainer falls back to the
        still-resident driver encoder.
        """
        trainer = self.pipeline
        dtype = next(trainer.driver.get_primary_model().parameters()).dtype
        out = trainer.encode_text([prompt], dtype)
        emb = out.embeddings if hasattr(out, "embeddings") else out
        return emb.to(self.device, dtype=dtype)

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator
    ) -> Tensor:
        """Create a 5D noise latent ``[1, 16, F, H/8, W/8]`` for the clip.

        Frame count is read from config (``sample_num_frames``, default 17 →
        ``4n+1``), compressed temporally by 4. Noise is fp32.
        """
        num_frames = int(self.config.get("sample_num_frames", 17))
        latent_f = (num_frames - 1) // WAN_VAE_TEMPORAL + 1
        lat_h = height // WAN_VAE_SPATIAL
        lat_w = width // WAN_VAE_SPATIAL
        shape = (1, 16, latent_f, lat_h, lat_w)
        return torch.randn(
            shape, generator=generator, device=self.device, dtype=torch.float32
        )

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Any:
        """Full fp32 FlowMatchEuler denoise loop using the trained transformer.

        The transformer forward may run in model dtype; the sigma math and
        latent accumulation stay fp32 (no autocast around the loop).
        """
        transformer = self.pipeline.driver.get_primary_model()
        self._ensure_transformer_on_device(transformer)

        # The WAN transformer is a MIXED-dtype module under mixed-precision
        # training: most weights are bf16 but precision-sensitive params
        # (scale_shift_table, time_embedder, norms) stay fp32, so
        # ``next(parameters()).dtype`` is just whichever param is first (fp32
        # scale_shift_table) and NO single input cast satisfies every layer —
        # casting inputs to fp32 fed the bf16 patch_embedding and crashed
        # ("Input type float vs bias BFloat16"). Run the forward in the SAME
        # autocast regime as training (per-op casting handles the mixed module).
        # The Euler trajectory stays fp32 OUTSIDE the autocast (in
        # ``euler_integrate``), so the no-collapse contract still holds.
        autocast_dtype = getattr(self.pipeline, "autocast_dtype", None) or torch.bfloat16
        device_type = torch.device(self.device).type

        sigmas = self._build_sigmas(num_steps).to(self.device)
        text = prompt_embedding

        def _velocity(x: Tensor, sigma: Tensor) -> Tensor:
            # Sigma is already in [0, 1] (training passes t/1000; WAN consumes
            # [0, 1]). Inputs stay in their natural dtype — autocast casts per-op
            # to match the training forward — and euler_integrate upcasts the
            # result to fp32 before accumulation (no autocast around the loop).
            t = sigma.reshape(1).expand(x.shape[0])
            with torch.no_grad(), torch.autocast(
                device_type=device_type, dtype=autocast_dtype
            ):
                out = transformer(
                    hidden_states=x,
                    timestep=t,
                    encoder_hidden_states=text,
                    encoder_hidden_states_image=None,
                    return_dict=False,
                )
            pred = out[0] if isinstance(out, tuple) else out
            return pred

        latents = self.euler_integrate(noise, sigmas, _velocity)
        return latents

    def decode_latents(self, latents: Any) -> Image.Image | SampleArtifact:
        """VAE-decode the fp32 latent clip → a :class:`SampleArtifact` (mp4)."""
        vae = self.pipeline.driver.vae
        latents = denormalize_wan_latents(latents, vae)
        with torch.no_grad():
            decoded = vae.decode(latents.to(vae.dtype), return_dict=False)[0]

        # decoded: [B, C, F, H, W] in ~[-1, 1]; emit the first clip as
        # [C, F, H, W] (the SampleArtifact canonical layout).
        clip = decoded[0].float().clamp(-1.0, 1.0)
        return SampleArtifact(frames=clip, fps=self.output_fps)
