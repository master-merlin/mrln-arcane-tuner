"""SDXL-specific sampling implementation for generating images during training.

Implements the DDIMScheduler-based denoising loop for epsilon prediction,
dual CLIP prompt encoding via the trainer's pipeline, and VAE decoding
to PIL images.

The sampler operates on the training model directly (no copy) in eval mode
with EMA weights swapped in (if EMA is enabled).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
import torch
from PIL import Image
from torch import Tensor

from app.engine.core.sampling import GenericSamplingPipeline

if TYPE_CHECKING:
    from .trainer import SDXLTrainer

logger = structlog.get_logger(__name__)

# Native in-training-preview defaults (StableDiffusionXLPipeline.__call__:
# 50 steps, guidance 5.0, 1024²). Definition-sourced via sdxl_base_1.0.yaml's
# `defaults`; these constants are the fallback used only when a definition
# omits the key. The generic base's 20 steps / 3.5 guidance are off-
# distribution for SDXL (its native pipeline uses 50 / 5.0).
_SDXL_DEFAULT_RESOLUTION: int = 1024
_SDXL_DEFAULT_STEPS: int = 50
_SDXL_DEFAULT_GUIDANCE: float = 5.0


class SDXLSampler(GenericSamplingPipeline):
    """SDXL family sampler — DDIM denoising with Classifier-Free Guidance.

    Generates sample images using the SDXL UNet, dual CLIP encoders,
    and VAE. Uses a clean DDIMScheduler for inference (separate from the
    training DDPMScheduler).

    The ``encode_prompt`` method eagerly encodes both the positive prompt
    AND an empty-string negative prompt in one call, so that text encoders
    can be safely offloaded/unloaded afterwards — matching the pattern
    used by Flux2 and Z-Image samplers.
    """

    pipeline: SDXLTrainer  # Narrow type for IDE support

    def __init__(self, pipeline: SDXLTrainer) -> None:
        super().__init__(pipeline)
        # Create a separate DDIM scheduler for sampling (faster than DDPM)
        from diffusers import DDIMScheduler

        # Read scheduler params from the trainer to stay aligned with training
        train_sched = pipeline.scheduler
        pred_type = getattr(
            train_sched.config, "prediction_type", "epsilon"
        )
        beta_start = getattr(train_sched.config, "beta_start", 0.00085)
        beta_end = getattr(train_sched.config, "beta_end", 0.012)

        self._inference_scheduler = DDIMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type=pred_type,
            clip_sample=False,
            set_alpha_to_one=False,
        )

    # ── Native sample defaults (W3-1; ovis/boogu precedent) ──────────────

    def _sample_single(self, prompt_cfg: dict[str, Any], step: int) -> Any:
        """Fill SDXL's native preview defaults before the generic flow.

        Sources 50 steps / 5.0 guidance / 1024² from the definition's
        ``defaults`` (constants are fallback only) — the diffusers SDXL
        pipeline's own ``__call__`` defaults, versus the generic base's
        off-distribution 20 steps / 3.5 guidance. Explicit per-prompt values
        always win (fill only when unset/0).
        """
        cfg = dict(prompt_cfg)
        defaults = getattr(self.pipeline.definition, "defaults", {}) or {}
        resolution = int(defaults.get("resolution", _SDXL_DEFAULT_RESOLUTION))
        fill = {
            "width": resolution,
            "height": resolution,
            "num_inference_steps": int(
                defaults.get("num_inference_steps", _SDXL_DEFAULT_STEPS),
            ),
            "guidance_scale": float(
                defaults.get("guidance_scale", _SDXL_DEFAULT_GUIDANCE),
            ),
        }
        for key, value in fill.items():
            if cfg.get(key) in (None, 0):
                cfg[key] = value
        return super()._sample_single(cfg, step)

    # ── Abstract Hook Implementations ────────────────────────────────────

    def encode_prompt(self, prompt: str) -> dict[str, Any]:
        """Encode positive AND negative (empty) prompts for CFG.

        Both are encoded in the same call while the text encoders are on
        the GPU, so that pre-cached embeddings remain available even after
        text encoder offload.

        Returns:
            Dict with ``cond``, ``uncond``, ``pooled_cond``,
            ``pooled_uncond`` tensors.
        """
        # Use the loaded UNet's dtype, not the training-time
        # autocast_dtype (the two can disagree -- e.g. mixed_precision
        # defaults to fp16 while the UNet may be loaded in bf16).
        model_dtype = next(self.pipeline.unet.parameters()).dtype

        # Positive prompt
        cond = self.pipeline.encode_text([prompt], dtype=model_dtype)
        pooled_cond = self.pipeline._pooled_embeds.clone()

        # Negative / unconditional prompt (empty string)
        uncond = self.pipeline.encode_text([""], dtype=model_dtype)
        pooled_uncond = self.pipeline._pooled_embeds.clone()

        return {
            "cond": cond,
            "uncond": uncond,
            "pooled_cond": pooled_cond,
            "pooled_uncond": pooled_uncond,
        }

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator
    ) -> Tensor:
        """Create noise in SDXL VAE latent space.

        Shape: [1, 4, H/8, W/8] (standard SDXL VAE has 4 latent channels).

        Args:
            width: Output image width in pixels.
            height: Output image height in pixels.
            generator: Seeded random generator.

        Returns:
            Noise tensor on ``self.device`` in float32.
        """
        latent_h = height // 8
        latent_w = width // 8
        return torch.randn(
            (1, 4, latent_h, latent_w),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Tensor:
        """DDIM denoising loop with classifier-free guidance.

        When ``guidance_scale > 1.0``, runs a single batched UNet forward
        pass with concatenated unconditional + conditional inputs, then
        applies the CFG formula. This matches the diffusers SDXL pipeline.

        Args:
            noise: Initial noise [1, 4, H/8, W/8].
            prompt_embedding: Dict with ``cond``, ``uncond``,
                ``pooled_cond``, ``pooled_uncond`` tensors.
            num_steps: Number of denoising steps.
            guidance_scale: CFG scale (> 1 enables classifier-free guidance).
            seed: Random seed (unused — seeding happens in noise gen).

        Returns:
            Denoised latent tensor [1, 4, H/8, W/8].
        """
        scheduler = self._inference_scheduler
        scheduler.set_timesteps(num_steps, device=self.device)

        model_dtype = next(self.pipeline.unet.parameters()).dtype
        use_amp = getattr(self.pipeline, "use_amp", True)

        # Unpack pre-cached embeddings
        cond_embeds = prompt_embedding["cond"].to(
            dtype=model_dtype, device=self.device
        )
        uncond_embeds = prompt_embedding["uncond"].to(
            dtype=model_dtype, device=self.device
        )
        pooled_cond = prompt_embedding["pooled_cond"].to(
            dtype=model_dtype, device=self.device
        )
        pooled_uncond = prompt_embedding["pooled_uncond"].to(
            dtype=model_dtype, device=self.device
        )

        do_cfg = guidance_scale > 1.0

        # Keep latents in fp32 for numerical stability in scheduler steps
        latents = noise.to(dtype=torch.float32)

        # Build SDXL added conditions
        _, _, latent_h, latent_w = noise.shape
        target_h = latent_h * 8
        target_w = latent_w * 8

        # time_ids: (orig_h, orig_w, crop_top, crop_left, target_h, target_w)
        time_ids = torch.tensor(
            [[target_h, target_w, 0, 0, target_h, target_w]],
            dtype=model_dtype,
            device=self.device,
        )

        # Prepare CFG-concatenated inputs (uncond first, then cond — diffusers convention)
        if do_cfg:
            text_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0)
            pooled_embeds = torch.cat([pooled_uncond, pooled_cond], dim=0)
            time_ids_batch = torch.cat([time_ids, time_ids], dim=0)
        else:
            text_embeds = cond_embeds
            pooled_embeds = pooled_cond
            time_ids_batch = time_ids

        added_cond_kwargs = {
            "text_embeds": pooled_embeds,
            "time_ids": time_ids_batch,
        }

        self.logger.debug(
            "denoising_start",
            num_steps=num_steps,
            latent_shape=list(latents.shape),
            text_shape=list(text_embeds.shape),
            guidance_scale=guidance_scale,
            do_cfg=do_cfg,
            model_dtype=str(model_dtype),
        )

        total_steps = len(scheduler.timesteps)
        for step_i, t in enumerate(scheduler.timesteps, 1):
            if getattr(self, "_log_writer", None):
                self._log_writer.status(f"Sampling {step_i}/{total_steps}")
            t_batch = t.unsqueeze(0).to(self.device)

            # Expand latents for CFG: [uncond_input, cond_input]
            if do_cfg:
                latent_input = torch.cat([latents, latents], dim=0)
                t_batch = torch.cat([t_batch, t_batch], dim=0)
            else:
                latent_input = latents

            # UNet forward under autocast
            with torch.autocast("cuda", dtype=model_dtype, enabled=use_amp):
                noise_pred = self.pipeline.unet(
                    latent_input.to(model_dtype),
                    t_batch,
                    encoder_hidden_states=text_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

            # Apply CFG formula
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # DDIM step in fp32 (outside autocast for numerical stability)
            latents = scheduler.step(
                noise_pred.float(), t, latents, return_dict=False
            )[0]

        return latents

    def decode_latents(self, latents: Any) -> Image.Image:
        """VAE-decode latent tensor to PIL image.

        Reads the scaling factor from VAE config (matching the encoding
        path in ``LatentManager``). When the VAE has ``force_upcast: true``
        (standard for SDXL), latents are cast to fp32 — NOT the VAE's
        parameter dtype which may be fp16.

        Args:
            latents: Denoised latent [1, 4, H/8, W/8].

        Returns:
            PIL Image in RGB mode.
        """
        vae = self.pipeline.vae
        vae_device = next(vae.parameters()).device

        # Read scaling factor from VAE config (matches LatentManager encoding)
        scaling_factor = getattr(vae.config, "scaling_factor", 0.13025)

        # force_upcast: the SDXL VAE is extremely sensitive to half-precision
        # artifacts. When force_upcast is true, always decode in fp32.
        force_upcast = getattr(vae.config, "force_upcast", False)
        if force_upcast:
            decode_dtype = torch.float32
        else:
            decode_dtype = next(vae.parameters()).dtype

        # Move VAE to compute device if needed
        if vae_device != self.device:
            vae.to(self.device)

        # Undo VAE scaling applied during latent encoding
        # Encoding: z = scaling_factor * sample
        # Decoding: sample = z / scaling_factor
        scaled_latents = latents.to(dtype=decode_dtype) / scaling_factor

        # Decode with autocast DISABLED — the SDXL VAE needs full fp32
        # precision; running under autocast causes washed-out grey artifacts.
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            if force_upcast:
                vae.to(dtype=torch.float32)
            decoded = vae.decode(scaled_latents, return_dict=False)

        # Move VAE back to save VRAM
        if vae_device != self.device:
            vae.to(vae_device)
            torch.cuda.empty_cache()

        # Post-process: tensor → PIL Image
        if isinstance(decoded, (tuple, list)):
            image_tensor = decoded[0]
        else:
            image_tensor = decoded

        image_tensor = image_tensor.clamp(-1, 1)
        image_tensor = (image_tensor + 1.0) / 2.0  # → [0, 1]
        image_tensor = image_tensor.squeeze(0)  # [3, H, W]
        image_tensor = image_tensor.permute(1, 2, 0)  # [H, W, 3]
        image_np = image_tensor.cpu().float().numpy()
        image_np = (image_np * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(image_np, mode="RGB")
