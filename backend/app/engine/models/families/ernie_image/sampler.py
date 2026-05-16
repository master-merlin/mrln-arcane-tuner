"""ERNIE-Image sampler -- mirrors ``ErnieImagePipeline.__call__``.

Differences from the standard diffusers pipeline (training preview):
*  Uses already-loaded components (no pipeline re-instantiation).
*  Skips the optional ``pe`` Prompt Enhancer; the user-supplied prompt
   is fed verbatim to the text encoder.
*  Uses cached training-time text embeddings when available (one
   encode per unique prompt across the whole training session).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
import torch
from PIL import Image
from torch import Tensor

from app.engine.core.sampling import GenericSamplingPipeline

if TYPE_CHECKING:
    from .trainer import ErnieImageTrainer

logger = structlog.get_logger(__name__)


# VAE downscales by a factor of 8 (3 stride-2 levels); the pipeline's
# 2x2 patchify adds another factor of 2 → combined model-input scale 16.
VAE_SPATIAL_DOWNSCALE = 8


class ErnieImageSampler(GenericSamplingPipeline):
    """ERNIE-Image sampler -- flow-matching Euler with optional CFG."""

    pipeline: ErnieImageTrainer

    def __init__(self, pipeline: ErnieImageTrainer) -> None:
        super().__init__(pipeline)
        self._scheduler = None

    # ── Lazy scheduler ───────────────────────────────────────────────────

    def _get_scheduler(self):
        if self._scheduler is not None:
            return self._scheduler
        from diffusers import FlowMatchEulerDiscreteScheduler

        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        self._scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=int(arch.get("scheduler.num_train_timesteps", 1000)),
            shift=float(arch.get("scheduler.shift", 4.0)),
        )
        return self._scheduler

    # ── Text encoding ────────────────────────────────────────────────────

    def encode_prompt(self, prompt: str) -> dict[str, Any]:
        """Encode positive + negative prompt for optional CFG.

        Delegates to the trainer's cache-aware ``encode_text``.  Returns
        already-padded ``(text_bth, attention_mask)`` pairs for both
        the conditional and unconditional paths.
        """
        dtype = self.pipeline.autocast_dtype
        cond_emb, cond_mask = self.pipeline.encode_text([prompt], dtype=dtype)
        uncond_emb, uncond_mask = self.pipeline.encode_text([""], dtype=dtype)
        return {
            "cond_emb": cond_emb,
            "cond_mask": cond_mask,
            "uncond_emb": uncond_emb,
            "uncond_mask": uncond_mask,
        }

    # ── Latent helpers ───────────────────────────────────────────────────

    def _resolve_vae_latent_channels(self) -> int:
        """Return the VAE's latent channel count (32 for ``AutoencoderKLFlux2``)."""
        vae = self.pipeline.vae
        z = getattr(getattr(vae, "config", None), "latent_channels", None) or 32
        return int(z)

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator,
    ) -> Tensor:
        """Random noise in the model's **patchified** input space.

        Combined scale factor = VAE downscale (8) × patchify (2) = 16,
        so the model sees a ``H/16 × W/16`` grid with
        ``in_channels = 4 × vae_latent_channels`` channels.
        """
        transformer = self.pipeline.transformer
        in_channels = int(getattr(transformer.config, "in_channels", 128))
        latent_h = height // (VAE_SPATIAL_DOWNSCALE * 2)
        latent_w = width // (VAE_SPATIAL_DOWNSCALE * 2)

        return torch.randn(
            (1, in_channels, latent_h, latent_w),
            generator=generator,
            device=self.device,
            dtype=self.pipeline.autocast_dtype,
        )

    # ── Core sampling methods ────────────────────────────────────────────

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Tensor:
        """Flow-matching Euler denoising loop with optional CFG.

        Returns the un-denormalized patched latents
        ``[1, in_channels, H/16, W/16]``; BN-denormalize + unpatchify +
        VAE decode happen in :meth:`decode_latents`.
        """
        device = self.device
        transformer = self.pipeline.transformer
        scheduler = self._get_scheduler()
        dtype = self.pipeline.autocast_dtype

        cond_emb = prompt_embedding["cond_emb"]
        cond_mask = prompt_embedding["cond_mask"]
        uncond_emb = prompt_embedding["uncond_emb"]
        uncond_mask = prompt_embedding["uncond_mask"]
        do_cfg = guidance_scale > 1.0

        cond_lens = cond_mask.sum(dim=1).to(dtype=torch.long, device=device)
        uncond_lens = uncond_mask.sum(dim=1).to(dtype=torch.long, device=device)

        latents = noise.to(device=device, dtype=dtype)

        # Linear sigma schedule matching the official pipeline.
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1)
        scheduler.set_timesteps(sigmas=sigmas[:-1], device=device)
        timesteps = scheduler.timesteps

        transformer.to(device)
        with torch.no_grad():
            total_steps = len(timesteps)
            for step_i, t in enumerate(timesteps, 1):
                print(f"[STATUS:Sampling {step_i}/{total_steps}]", flush=True)
                t_batch = torch.full(
                    (latents.shape[0],), t.item(),
                    device=device, dtype=dtype,
                )

                # Conditional pass — transformer expects t in [0, 1].
                pred_cond = transformer(
                    hidden_states=latents,
                    timestep=t_batch / 1000.0,
                    text_bth=cond_emb,
                    text_lens=cond_lens,
                    return_dict=False,
                )[0]

                if do_cfg:
                    pred_uncond = transformer(
                        hidden_states=latents,
                        timestep=t_batch / 1000.0,
                        text_bth=uncond_emb,
                        text_lens=uncond_lens,
                        return_dict=False,
                    )[0]
                    pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                else:
                    pred = pred_cond

                latents = scheduler.step(pred, t, latents, return_dict=False)[0]

        return latents

    def decode_latents(self, latents: Any) -> Image.Image:
        """BN-denormalize → unpatchify → VAE decode → PIL.

        Mirrors the post-denoise tail of ``ErnieImagePipeline.__call__``.
        """
        from app.engine.models.families.ernie_image.utils import (
            bn_denormalize,
            unpatchify_latents,
        )

        vae = self.pipeline.vae

        latents = bn_denormalize(latents, vae)
        latents = unpatchify_latents(latents)

        with torch.no_grad():
            images = vae.decode(latents.to(vae.dtype), return_dict=False)[0]

        images = (images.clamp(-1, 1) + 1.0) / 2.0
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        arr = (images[0] * 255).round().astype("uint8")
        return Image.fromarray(arr)
