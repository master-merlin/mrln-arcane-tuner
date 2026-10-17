"""FLUX.3 Image Klein preview sampler (LANE-144 spec D6).

Euler on the flow ``x_t = (1 - t) * x0 + t * noise`` (velocity ``noise - x0``) with REAL classifier-free
guidance ``v = v_uncond + g * (v_cond - v_uncond)`` (the base model has no guidance embedding; the
unconditional branch is the empty prompt encoded at the prompt's own window). The schedule is the
checkpoint's FIXED shift ``a = 6.93`` applied to a uniform grid on [1, 0]; the transformer receives
the shifted value in ``[0, 1]`` (it multiplies by 1000 itself).

Precision contract: the latent trajectory stays fp32 and is cast to the transformer's dtype per forward;
there is NO ``torch.autocast`` anywhere (autocast around an N-step flow-match loop collapses sampling
to the conditional mean). Latents are BatchNorm-un-normalised and unpatchified through the ``flux2``
helpers and decoded by ``AutoencoderKLFlux2``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
import torch
from PIL import Image
from torch import Tensor

from app.engine.core.sampling import GenericSamplingPipeline

from .driver import make_img_ids
from .trainer import DEFAULT_FLOW_SHIFT, fixed_shift

if TYPE_CHECKING:
    from .trainer import Flux3Trainer

logger = structlog.get_logger(__name__)

# Fallbacks only; the definition's ``defaults`` are the source (BFL reference: 50 steps, CFG 3.5, 1024).
_DEFAULT_RESOLUTION = 1024
_DEFAULT_STEPS = 50
_DEFAULT_GUIDANCE = 3.5
_VAE_SCALE = 8  # AutoencoderKLFlux2 spatial compression
_PATCH = 2
_LATENT_CHANNELS = 32


class Flux3Sampler(GenericSamplingPipeline):
    """Euler + real CFG on the fixed 6.93 shift."""

    pipeline: Flux3Trainer

    def _sample_single(self, prompt_cfg: dict[str, Any], step: int) -> Any:
        """Fill the native preview defaults (definition-sourced) before the generic flow."""
        cfg = dict(prompt_cfg)
        defaults = getattr(self.pipeline.definition, "defaults", {}) or {}
        resolution = int(defaults.get("resolution", _DEFAULT_RESOLUTION))
        fill = {
            "width": resolution,
            "height": resolution,
            "num_inference_steps": int(defaults.get("num_inference_steps", _DEFAULT_STEPS)),
            "guidance_scale": float(defaults.get("guidance_scale", _DEFAULT_GUIDANCE)),
        }
        for key, value in fill.items():
            if cfg.get(key) in (None, 0):
                cfg[key] = value
        return super()._sample_single(cfg, step)

    def _flow_shift(self) -> float:
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        return float(arch.get("scheduler.shift", DEFAULT_FLOW_SHIFT))

    # ── Text ─────────────────────────────────────────────────────────────

    def encode_prompt(self, prompt: str) -> dict[str, Tensor]:
        """Cond and empty-prompt uncond through the trainer's cache-aware encoder (same window)."""
        trainer = self.pipeline
        dtype = next(trainer.transformer.parameters()).dtype
        cond = trainer.encode_text([prompt], dtype=dtype)
        uncond = trainer.encode_text([""], dtype=dtype)
        return {"cond": cond.to(self.device), "uncond": uncond.to(self.device)}

    # ── Latents ──────────────────────────────────────────────────────────

    def _create_initial_noise(self, width: int, height: int, generator: torch.Generator) -> Tensor:
        """Packed fp32 noise ``(1, (H/16) * (W/16), 128)``; stores the image ids on the driver."""
        grid_h = height // _VAE_SCALE // _PATCH
        grid_w = width // _VAE_SCALE // _PATCH
        shape = (1, _LATENT_CHANNELS * _PATCH * _PATCH, grid_h, grid_w)
        noise = torch.randn(shape, generator=generator, device=self.device, dtype=torch.float32)
        self.pipeline.driver._current_img_ids = make_img_ids(grid_h, grid_w, self.device)
        return noise.reshape(1, shape[1], grid_h * grid_w).permute(0, 2, 1)

    def denoise(
        self, noise: Tensor, prompt_embedding: Any, num_steps: int, guidance_scale: float, seed: int,
    ) -> Tensor:
        """Euler + CFG; returns BN-un-normalised, unpatchified latents ``(1, 32, H/8, W/8)``."""
        from app.engine.models.families.flux2.utils import bn_denormalize, unpatchify_latents

        driver = self.pipeline.driver
        transformer = self.pipeline.transformer
        dtype = next(transformer.parameters()).dtype
        cond = prompt_embedding["cond"]
        uncond = prompt_embedding["uncond"]
        do_cfg = guidance_scale > 1.0

        img_ids = driver._current_img_ids
        grid_w = int(img_ids[:, 2].max().item() * 2) + 1
        grid_h = int(img_ids[:, 1].max().item() * 2) + 1

        sigmas = fixed_shift(torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32), self._flow_shift())
        latents = noise.to(device=self.device, dtype=torch.float32)

        self._ensure_transformer_on_device(transformer)
        with torch.no_grad():
            for step_i in range(num_steps):
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Sampling {step_i + 1}/{num_steps}")
                sigma, sigma_next = sigmas[step_i], sigmas[step_i + 1]
                # driver.forward_pass takes the pipeline's [0, 1000] scale and divides by 1000 once.
                ts = (sigma * 1000.0).expand(latents.shape[0]).to(device=self.device, dtype=dtype)
                xin = latents.to(dtype=dtype)
                v = driver.forward_pass(xin, ts, cond.to(dtype=dtype), {}).to(torch.float32)
                if do_cfg:
                    v_un = driver.forward_pass(xin, ts, uncond.to(dtype=dtype), {}).to(torch.float32)
                    v = v_un + guidance_scale * (v - v_un)
                latents = latents + (sigma_next - sigma).to(self.device) * v

        spatial = latents.permute(0, 2, 1).reshape(latents.shape[0], latents.shape[2], grid_h, grid_w)
        return unpatchify_latents(bn_denormalize(spatial, self.pipeline.vae))

    def decode_latents(self, latents: Any) -> Image.Image:
        vae = self.pipeline.vae
        with torch.no_grad():
            image = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return Image.fromarray((image[0] * 255).round().astype("uint8"))
