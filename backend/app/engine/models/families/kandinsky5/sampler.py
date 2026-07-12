"""Kandinsky 5.0 sampler — strictly-fp32 flow-match Euler, channels-last.

Produces short mp4 previews (24 fps) during training as
:class:`SampleArtifact` objects. Replicates the upstream
``Kandinsky5T2VPipeline`` denoise loop:

- **fp32 trajectory, NO autocast**: sigma math + latent accumulation run in
  fp32; only the transformer forward runs in the model's own dtype (inputs
  cast per-call, exactly like the pipeline's ``.to(dtype)``) — the
  autocast-collapse gotcha guard.
- **Inline dual-forward CFG** when ``guidance_scale > 1.0`` (T2V default 5.0):
  ``v = v_uncond + gs * (v_cond - v_uncond)``, with the pipeline's DEFAULT
  negative prompt injected when none is configured.
- **Channels-last state incl. cond concat**: the latent state carries the
  ``visual_cond``/mask channels (like the pipeline's ``prepare_latents``); the
  Euler update writes only ``[..., :C]`` — and, for an image-conditioned I2V
  denoise, only frames ``1:`` (frame 0 stays the image latent).
- Plain static-shift sigma schedule (``shifted_sigmas``, shift 5.0 from the
  repo scheduler config) — no dynamic mu/sigmas, matching the pipeline's
  plain ``set_timesteps``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog
import torch
from PIL import Image
from torch import Tensor

from app.engine.core.sampling import GenericSamplingPipeline, SampleArtifact
from app.engine.core.text_encoding import TextEncoderOutput

from .driver import (
    FLOWMATCH_SCALE,
    Kandinsky5Driver,
    get_scale_factor,
    resolve_negative_prompt,
    to_channels_first,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.engine.core.pipeline import GenericTrainingPipeline

logger = structlog.get_logger(__name__)

KANDINSKY5_NATIVE_FPS = 24.0
# Repo scheduler config shift (FlowMatchEulerDiscreteScheduler, shift=5.0).
KANDINSKY5_DEFAULT_SHIFT = 5.0


class Kandinsky5Sampler(GenericSamplingPipeline):
    """Flow-match Euler video sampler for Kandinsky 5.0 (fp32 trajectory)."""

    output_fps: float = KANDINSKY5_NATIVE_FPS

    def __init__(self, pipeline: GenericTrainingPipeline) -> None:
        super().__init__(pipeline)

    # ── fp32 Euler integration core (the precision-critical path) ──────────

    def euler_integrate(
        self,
        x0: Tensor,
        sigmas: Tensor,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
        *,
        num_channels: int | None = None,
        start_frame: int = 0,
    ) -> Tensor:
        """Integrate ``dx/dσ = v(x, σ)`` with forward Euler in fp32.

        The state may carry EXTRA trailing channels (the visual_cond concat) —
        ``num_channels`` restricts the update to ``[..., :num_channels]``
        (the pipeline's ``latents[..., :num_channels_latents] = step(...)``),
        and ``start_frame`` skips leading frames (the I2V pipeline updates
        ``latents[:, 1:]`` only). ``velocity_fn`` sees the FULL state and must
        return a ``num_channels``-wide velocity.

        The trajectory (sigma math + accumulation) is forced to fp32;
        ``velocity_fn`` may internally run the model dtype but its result is
        upcast before accumulation (the autocast-collapse guard). Emits the
        per-step ``Sampling {i}/{N}`` status when a JobLogWriter is attached.
        """
        x = x0.to(torch.float32)
        s = sigmas.to(torch.float32)
        total_steps = len(s) - 1
        c = num_channels if num_channels is not None else x.shape[-1]
        for i in range(total_steps):
            if getattr(self, "_log_writer", None):
                self._log_writer.status(f"Sampling {i + 1}/{total_steps}")
            dt = (s[i + 1] - s[i]).to(torch.float32)
            v = velocity_fn(x, s[i]).to(torch.float32)
            if start_frame > 0 or num_channels is not None:
                upd = x[:, start_frame:, ..., :c] + dt * v[:, start_frame:]
                x = x.clone()
                x[:, start_frame:, ..., :c] = upd
            else:
                x = x + dt * v
        return x

    def build_denoise(
        self, transformer: Any, **forward_kwargs: Any
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """``denoise(x0, sigmas)`` closure for the precision-contract test —
        drives the REAL :meth:`euler_integrate` fp32 loop."""

        def _velocity(x: Tensor, sigma: Tensor) -> Tensor:
            t = sigma.reshape(1).to(x.dtype)
            out = transformer(x, t, **forward_kwargs)
            return out[0] if isinstance(out, tuple) else out

        def _denoise(x0: Tensor, sigmas: Tensor) -> Tensor:
            return self.euler_integrate(x0, sigmas, _velocity)

        return _denoise

    # ── Sigma schedule ─────────────────────────────────────────────────────

    def _build_sigmas(self, num_steps: int) -> Tensor:
        """Static-shift FlowMatchEuler schedule, descending 1 → 0 (fp32).

        Shift resolution: ``model_shift_fixed`` (injected from the definition's
        ``scheduler.flow_shift`` by the video contract) → arch
        ``scheduler.shift`` → 5.0 (the shipped checkpoints' value). No dynamic
        mu — the Kandinsky pipelines call plain ``set_timesteps(N)``.
        """
        from app.engine.strategies.sigma_schedule import shifted_sigmas

        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        shift = float(
            self.config.get("model_shift_fixed")
            or arch.get("scheduler.shift")
            or KANDINSKY5_DEFAULT_SHIFT
        )
        return shifted_sigmas(num_steps, shift)

    # ── GenericSamplingPipeline hooks ──────────────────────────────────────

    def encode_prompt(self, prompt: str) -> TextEncoderOutput:
        """Encode via the trainer's CACHED ``encode_text`` (both TEs are
        offloaded by sample time; the pre-cache warmed every sample prompt and
        the CFG negative)."""
        trainer = self.pipeline
        out = trainer.encode_text([prompt], torch.float32)
        if not isinstance(out, TextEncoderOutput):  # pragma: no cover - guard
            raise TypeError("Kandinsky5 encode_text must return TextEncoderOutput")
        return out

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator
    ) -> Tensor:
        """Channels-LAST 5D noise ``[1, F_lat, H/8, W/8, C]`` (fp32).

        Frame count from ``sample_num_frames`` (default 17 → 4n+1),
        compressed temporally by 4 (HunyuanVideo VAE).
        """
        driver: Kandinsky5Driver = self.pipeline.driver
        # Per-prompt num_frames override (None = run default; 1 = still).
        num_frames = self._effective_sample_frames(17, "4n+1")
        latent_f = (num_frames - 1) // 4 + 1
        lat_h = height // driver.vae_spatial
        lat_w = width // driver.vae_spatial
        shape = (1, latent_f, lat_h, lat_w, driver.in_visual_dim)
        return torch.randn(
            shape, generator=generator, device=self.device, dtype=torch.float32
        )

    def _conditioning_image_latent(self) -> Tensor | None:
        """Optional I2V preview conditioning latent (channels-last
        ``[1, 1, H_lat, W_lat, C]``).

        Default ``None`` — previews on I2V definitions run unconditioned
        (zero cond + zero mask, which the visual_cond checkpoint reads as "no
        image"). Wiring a real preview image through the sample-prompt config
        is a UI follow-up; ``denoise`` already handles a returned latent
        (frame-0 pin + mask + frame-0 step skip).
        """
        return None

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: TextEncoderOutput,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Tensor:
        """Full fp32 flow-match Euler denoise with inline dual-forward CFG.

        Replicates the pipeline's regime: inputs are cast to the transformer's
        own dtype per forward (NO autocast anywhere); the trajectory stays
        fp32 in :meth:`euler_integrate`.
        """
        driver: Kandinsky5Driver = self.pipeline.driver
        transformer = driver.get_primary_model()
        if isinstance(transformer, torch.nn.Module):
            self._ensure_transformer_on_device(transformer)
        model_dtype = self._model_dtype(transformer)

        num_channels = driver.in_visual_dim
        x = noise.to(torch.float32)
        start_frame = 0

        image_latent = self._conditioning_image_latent()
        if image_latent is not None:
            image_latent = image_latent.to(x.device, torch.float32)
            x = torch.cat([image_latent, x[:, 1:]], dim=1)  # frame 0 = image
            start_frame = 1

        if driver.visual_cond:
            visual_cond = torch.zeros_like(x)
            cond_mask = x.new_zeros(*x.shape[:-1], 1)
            if image_latent is not None:
                visual_cond[:, :1] = image_latent
                cond_mask[:, :1] = 1.0
            x = torch.cat([x, visual_cond, cond_mask], dim=-1)

        _, f, h, w, _ = x.shape
        visual_rope_pos = Kandinsky5Driver.build_visual_rope_pos(f, h, w, x.device)
        scale_factor = get_scale_factor(
            h * driver.vae_spatial, w * driver.vae_spatial
        )

        # CFG: contrast against the (possibly pipeline-default) negative.
        gs = float(guidance_scale) if guidance_scale is not None else None
        cfg_on = gs is not None and gs > 1.0
        text_uncond: TextEncoderOutput | None = None
        if cfg_on:
            text_uncond = self.encode_prompt(resolve_negative_prompt(self.config))

        def _forward(
            x_full: Tensor, sigma: Tensor, teo: TextEncoderOutput
        ) -> Tensor:
            # Trajectory steps in sigma ∈ [0,1]; the transformer is conditioned
            # on the RAW [0,1000] timestep (pipeline scale — the pure-noise
            # gotcha). Inputs cast to the model dtype, NO autocast.
            t = (sigma * FLOWMATCH_SCALE).reshape(1)
            cu = teo.require_attention_mask()
            text_rope_pos = Kandinsky5Driver.build_text_rope_pos(cu, x_full.device)
            with torch.no_grad():
                out = transformer(
                    hidden_states=x_full.to(model_dtype),
                    encoder_hidden_states=teo.embeddings.to(
                        x_full.device, model_dtype
                    ),
                    timestep=t.to(device=x_full.device, dtype=model_dtype),
                    pooled_projections=teo.require_pooled().to(
                        x_full.device, model_dtype
                    ),
                    visual_rope_pos=visual_rope_pos,
                    text_rope_pos=text_rope_pos,
                    scale_factor=scale_factor,
                    sparse_params=None,
                    return_dict=False,
                )
            return out[0] if isinstance(out, (tuple, list)) else out

        def _velocity(x_full: Tensor, sigma: Tensor) -> Tensor:
            v_c = _forward(x_full, sigma, prompt_embedding)
            if not cfg_on:
                return v_c
            # Combine in fp32 (matches euler_integrate's accumulation regime).
            v_c = v_c.to(torch.float32)
            v_u = _forward(x_full, sigma, text_uncond).to(torch.float32)
            return v_u + gs * (v_c - v_u)

        sigmas = self._build_sigmas(num_steps).to(x.device)
        latents = self.euler_integrate(
            x,
            sigmas,
            _velocity,
            num_channels=num_channels,
            start_frame=start_frame,
        )
        # Drop the cond concat channels — decode sees the C latent channels.
        return latents[..., :num_channels]

    def decode_latents(self, latents: Tensor) -> Image.Image | SampleArtifact:
        """Channels-last latents → HunyuanVideo VAE decode → mp4 artifact.

        ``[1, F, H, W, C]`` → ``[1, C, F, H, W]``, divided by the scalar
        scaling factor 0.476986 (the pipeline's ``video / scaling_factor``),
        then decoded and clamped to the ``SampleArtifact`` canonical
        ``[C, F, H, W]`` float layout at 24 fps.
        """
        driver: Kandinsky5Driver = self.pipeline.driver
        vae = driver.vae
        video = to_channels_first(latents.to(torch.float32))
        scaling = float(getattr(vae.config, "scaling_factor", 0.476986))
        video = video / scaling
        with torch.no_grad():
            decoded = vae.decode(video.to(vae.dtype), return_dict=False)[0]

        clip = decoded[0].float().clamp(-1.0, 1.0)  # [C, F, H, W]
        return SampleArtifact(frames=clip, fps=self.output_fps)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _model_dtype(transformer: Any) -> torch.dtype:
        """The transformer's dtype (pipeline uses ``self.transformer.dtype``);
        falls back to a parameter probe / fp32 for test fakes."""
        dtype = getattr(transformer, "dtype", None)
        if isinstance(dtype, torch.dtype):
            return dtype
        try:
            return next(transformer.parameters()).dtype
        except (AttributeError, StopIteration):
            return torch.float32
