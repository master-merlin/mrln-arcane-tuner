"""MOSS-Image preview sampler (LANE-165 spec scenario 5).

Euler on the flow ``x_t = (1 - t) * x0 + t * noise`` (velocity ``noise - x0``) with REAL
classifier-free guidance ``v = v_uncond + g * (v_cond - v_uncond)``. The schedule is a uniform grid on
[1, 0] shifted by the YAML's ``scheduler.shift`` (5.0); the transformer takes the RAW ``sigma * 1000``
timestep. As the official pipeline does, the NEGATIVE branch skips block 22 (Evidence: research
§sampler).

Precision contract: noise, the CFG combination and the Euler update are fp32, the latent is cast to the
transformer's dtype per forward, and there is NO ``torch.autocast`` anywhere (autocast around an N-step
flow-match loop collapses sampling to the conditional mean). Latents are BatchNorm-un-normalised and
unpatchified through the ``flux2`` helpers and decoded by ``AutoencoderKLFlux2``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
import torch
from PIL import Image
from torch import Tensor

from app.engine.core.sampling import GenericSamplingPipeline


if TYPE_CHECKING:
    from .trainer import MossImageTrainer

logger = structlog.get_logger(__name__)

# Fallbacks only; the definition's ``defaults`` are the source (model card: 50 steps, CFG 3.0).
_DEFAULT_RESOLUTION = 1024
_DEFAULT_STEPS = 50
_DEFAULT_GUIDANCE = 3.0
_DEFAULT_SHIFT = 5.0
_VAE_SCALE = 8  # AutoencoderKLFlux2 spatial compression
_PATCH = 2
_LATENT_CHANNELS = 32
# The official pipeline's negative-branch skip (research §sampler).
UNCOND_SKIP_LAYERS = (22,)


class MossImageSampler(GenericSamplingPipeline):
    """Euler + real CFG on the shifted grid, block 22 skipped on the negative branch."""

    pipeline: MossImageTrainer

    def _sample_prompt_fallbacks(self) -> dict[str, Any]:
        """The definition's inference defaults for an unset sample-prompt field (spec scenario 11).

        Resolved once by ``_get_sample_prompts`` before any sampling; a value the user set (20 / 3.5
        included) is never replaced, so the logged guidance is the executed one.
        """
        defaults = getattr(self.pipeline.definition, "defaults", {}) or {}
        resolution = int(defaults.get("resolution", _DEFAULT_RESOLUTION))
        return {
            "width": resolution,
            "height": resolution,
            "num_inference_steps": int(defaults.get("num_inference_steps", _DEFAULT_STEPS)),
            "guidance_scale": float(defaults.get("guidance_scale", _DEFAULT_GUIDANCE)),
        }

    def _flow_shift(self) -> float:
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        return float(arch.get("scheduler.shift", _DEFAULT_SHIFT))

    # ── Text ─────────────────────────────────────────────────────────────

    def encode_prompt(self, prompt: str) -> dict[str, Any]:
        """Cond and empty-prompt uncond through the trainer's cache-aware encoder."""
        trainer = self.pipeline
        dtype = next(trainer.transformer.parameters()).dtype
        cond = trainer.encode_text([prompt], dtype=dtype)
        uncond = trainer.encode_text([""], dtype=dtype)
        return {"cond": self._to_device(cond), "uncond": self._to_device(uncond)}

    def _to_device(self, out: Any) -> Any:
        """``TextEncoderOutput`` is a plain dataclass (no ``.to``): move its tensor fields explicitly."""
        from dataclasses import replace  # noqa: PLC0415

        return replace(
            out,
            embeddings=out.embeddings.to(self.device),
            attention_mask=None if out.attention_mask is None else out.attention_mask.to(self.device),
            pooled=None if out.pooled is None else out.pooled.to(self.device),
        )

    # ── Latents ──────────────────────────────────────────────────────────

    def _create_initial_noise(self, width: int, height: int, generator: torch.Generator) -> Tensor:
        """Spatial fp32 noise ``(1, 128, H/16, W/16)`` (the patchified latent grid)."""
        grid_h = height // _VAE_SCALE // _PATCH
        grid_w = width // _VAE_SCALE // _PATCH
        shape = (1, _LATENT_CHANNELS * _PATCH * _PATCH, grid_h, grid_w)
        return torch.randn(shape, generator=generator, device=self.device, dtype=torch.float32)

    def _velocity(self, transformer: Any, x: Tensor, ts: Tensor, text: Any, skip: tuple[int, ...] | None) -> Tensor:
        out = transformer(
            hidden_states=x,
            timestep=ts,
            encoder_hidden_states=text.embeddings,
            encoder_attention_mask=text.attention_mask,
            return_dict=False,
            skip_layers=list(skip) if skip else None,
        )
        return (out[0] if isinstance(out, tuple) else out).to(torch.float32)

    def _inference_sigmas(self, num_steps: int) -> Tensor:
        """The official schedule: the vendored scheduler's shifted, training-grid-snapped sigmas (+ trailing 0).

        The scheduler's ``shift`` is the YAML's inference shift; its ``training_shift`` stays at the
        upstream default, independent of training timestep sampling (which goes through the shared
        ``TimestepSampler`` and reads no ``training_shift``). Snapping can merge
        steps, so the returned length (not ``num_steps``) drives the loop.
        """
        from .vendor.scheduler_moss_image import MOSSImageScheduler

        scheduler = MOSSImageScheduler(shift=self._flow_shift())
        scheduler.set_timesteps(num_steps)
        return scheduler.sigmas.to(dtype=torch.float32)

    def denoise(
        self, noise: Tensor, prompt_embedding: Any, num_steps: int, guidance_scale: float, seed: int,
    ) -> Tensor:
        """Euler + CFG; returns BN-un-normalised, unpatchified latents ``(1, 32, H/8, W/8)``."""
        from app.engine.models.families.flux2.utils import bn_denormalize, unpatchify_latents

        transformer = self.pipeline.transformer
        dtype = next(transformer.parameters()).dtype
        cond = prompt_embedding["cond"]
        uncond = prompt_embedding["uncond"]
        do_cfg = guidance_scale > 1.0

        sigmas = self._inference_sigmas(num_steps)
        latents = noise.to(device=self.device, dtype=torch.float32)

        self._ensure_transformer_on_device(transformer)
        with torch.no_grad():
            for step_i in range(len(sigmas) - 1):
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Sampling {step_i + 1}/{num_steps}")
                sigma, sigma_next = sigmas[step_i], sigmas[step_i + 1]
                # The transformer takes the raw 0-1000 value (spec scenario 2).
                ts = (sigma * 1000.0).expand(latents.shape[0]).to(device=self.device, dtype=torch.float32)
                xin = latents.to(dtype=dtype)
                v = self._velocity(transformer, xin, ts, cond, None)
                if do_cfg:
                    v_un = self._velocity(transformer, xin, ts, uncond, UNCOND_SKIP_LAYERS)
                    v = v_un + guidance_scale * (v - v_un)
                latents = latents + (sigma_next - sigma).to(self.device) * v

        return unpatchify_latents(bn_denormalize(latents, self.pipeline.vae))

    def decode_latents(self, latents: Any) -> Image.Image:
        vae = self.pipeline.vae
        with torch.no_grad():
            image = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return Image.fromarray((image[0] * 255).round().astype("uint8"))
