"""Qwen-Image sampling — matches ``QwenImagePipeline`` from diffusers.

Key differences from the previous (broken) sampler:

*  **Prompt template** — wraps prompts in the Qwen-Image system/user/assistant
   chat format and drops the first 34 system tokens, exactly like the diffusers
   reference ``_get_qwen_prompt_embeds``.
*  **Packed latent space** — operates in ``[B, num_patches, C*4]`` space via
   ``_pack_latents`` / ``_unpack_latents``, matching the reference pipeline.
*  **``txt_seq_lens``** — derived from attention mask (actual valid token count),
   not the full padded length.
*  **VAE decode normalization** — ``z / (1/std) + mean`` with 5D
   ``[B, C, 1, H, W]`` input, taking ``[:, :, 0]`` from the output.
*  **guidance=None** — ``guidance_embeds: false`` in model config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import torch
from PIL import Image
from torch import Tensor

from app.engine.core.sampling import GenericSamplingPipeline

if TYPE_CHECKING:
    from .trainer import QwenImageTrainer

logger = structlog.get_logger(__name__)


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Empirical mu schedule — copied from QwenImagePipeline."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


class QwenImageSampler(GenericSamplingPipeline):
    """Qwen-Image sampler — matches ``QwenImagePipeline`` from diffusers."""

    pipeline: QwenImageTrainer

    def __init__(self, pipeline: QwenImageTrainer) -> None:
        super().__init__(pipeline)
        self._scheduler = None

    # ── Lazy scheduler ───────────────────────────────────────────────────

    def _get_scheduler(self):
        if self._scheduler is not None:
            return self._scheduler
        from diffusers import FlowMatchEulerDiscreteScheduler

        # Match config from YAML definition
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        self._scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=int(arch.get("scheduler.num_train_timesteps", 1000)),
            use_dynamic_shifting=True,
            base_shift=float(arch.get("scheduler.base_shift", 0.5)),
            max_shift=float(arch.get("scheduler.max_shift", 0.9)),
            base_image_seq_len=int(arch.get("scheduler.base_image_seq_len", 256)),
            max_image_seq_len=int(arch.get("scheduler.max_image_seq_len", 8192)),
        )
        return self._scheduler

    # ── Text encoding ────────────────────────────────────────────────────

    def encode_prompt(self, prompt: str) -> dict[str, Any]:
        """Encode prompt via the trainer's cache-aware ``encode_text()``.

        Delegates all caching and TE management to the trainer.
        Returns dict with ``embeds`` [1, L, D] and ``mask`` [1, L].
        """
        embeds, mask = self.pipeline.encode_text(
            [prompt],
            dtype=next(self.pipeline.transformer.parameters()).dtype,
        )
        return {"embeds": embeds, "mask": mask}

    # ── Latent packing (matches QwenImagePipeline exactly) ───────────────

    @staticmethod
    def _pack_latents(
        latents: Tensor, batch_size: int, num_channels: int,
        height: int, width: int,
    ) -> Tensor:
        """[B, C, H, W] → [B, (H/2)*(W/2), C*4]."""
        latents = latents.view(
            batch_size, num_channels, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(
            batch_size, (height // 2) * (width // 2), num_channels * 4
        )

    @staticmethod
    def _unpack_latents(
        latents: Tensor, height: int, width: int, vae_scale_factor: int,
    ) -> Tensor:
        """[B, num_patches, C*4] → [B, C, 1, H, W]."""
        batch_size, num_patches, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(
            batch_size, height // 2, width // 2, channels // 4, 2, 2
        )
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch_size, channels // 4, 1, height, width)

    # ── Core sampling methods ────────────────────────────────────────────

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator
    ) -> Tensor:
        """Create packed noise [B, num_patches, C*4] for QwenImage.

        Also stores metadata needed by ``denoise()`` and
        ``decode_latents()``.
        """
        device = self.device
        transformer = self.pipeline.transformer
        vae = self.pipeline.vae
        # Use the loaded transformer's dtype, not training-time
        # autocast_dtype (which defaults to fp16 while the transformer
        # may be loaded in bf16).
        dtype = next(transformer.parameters()).dtype

        # VAE scale factor
        vae_sf = (
            2 ** len(vae.temperal_downsample)
            if hasattr(vae, "temperal_downsample")
            else 8
        )

        # Latent dims (accounting for VAE compression + packing)
        lat_h = 2 * (height // (vae_sf * 2))
        lat_w = 2 * (width // (vae_sf * 2))
        num_channels = transformer.config.in_channels // 4

        # Generate noise in 5D [B,C,1,H,W] → pack to [B, num_patches, C*4]
        noise_shape = (1, 1, num_channels, lat_h, lat_w)
        latents = torch.randn(
            noise_shape, generator=generator, device=device, dtype=dtype
        )
        latents = self._pack_latents(latents, 1, num_channels, lat_h, lat_w)

        # Store metadata for denoise/decode
        self._lat_h = lat_h
        self._lat_w = lat_w
        self._vae_sf = vae_sf
        self._sample_height = height
        self._sample_width = width

        return latents

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Any:
        """Flow-matching Euler denoising matching QwenImagePipeline.__call__.

        Returns a dict with ``latents`` (5D tensor ready for VAE decode)
        and geometry metadata for ``decode_latents()``.
        """
        device = self.device
        transformer = self.pipeline.transformer
        scheduler = self._get_scheduler()
        dtype = next(transformer.parameters()).dtype

        height = self._sample_height
        width = self._sample_width
        lat_h = self._lat_h
        lat_w = self._lat_w
        vae_sf = self._vae_sf

        # Text embeddings
        prompt_embeds = prompt_embedding["embeds"]
        prompt_mask = prompt_embedding["mask"]

        # Use pre-created noise
        latents = noise.to(device=device, dtype=dtype)

        # img_shapes for RoPE
        img_shapes = [[(1, lat_h // 2, lat_w // 2)]]

        # txt_seq_lens from attention mask (actual valid token count)
        txt_seq_lens = prompt_mask.sum(dim=1).tolist()

        # Timestep schedule
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        image_seq_len = latents.shape[1]

        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        mu = _calculate_shift(
            image_seq_len,
            int(arch.get("scheduler.base_image_seq_len", 256)),
            int(arch.get("scheduler.max_image_seq_len", 4096)),
            float(arch.get("scheduler.base_shift", 0.5)),
            float(arch.get("scheduler.max_shift", 1.15)),
        )
        scheduler.set_timesteps(
            num_inference_steps=num_steps, device=device,
            sigmas=sigmas, mu=mu,
        )
        timesteps = scheduler.timesteps

        # guidance=None (guidance_embeds: false)
        guidance = None

        # Denoising loop
        self._ensure_transformer_on_device(transformer)
        scheduler.set_begin_index(0)
        with torch.no_grad():
            total_steps = len(timesteps)
            for step_i, t in enumerate(timesteps, 1):
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Sampling {step_i}/{total_steps}")
                ts = t.expand(latents.shape[0]).to(dtype)

                noise_pred = transformer(
                    hidden_states=latents,
                    timestep=ts / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]

                latents = scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

        # Unpack: [B, num_patches, C*4] → [B, C, 1, H, W]
        latents = self._unpack_latents(latents, height, width, vae_sf)

        # Return latents + metadata for decode_latents
        return {"latents": latents, "height": height, "width": width}

    def decode_latents(self, latents_bundle: Any) -> Image.Image:
        """VAE-decode latent tensor to PIL image.

        Handles QwenImage-specific 5D normalization and frame extraction.

        Args:
            latents_bundle: Dict with ``latents`` [B, C, 1, H, W].

        Returns:
            PIL Image in RGB mode.
        """
        vae = self.pipeline.vae
        latents = latents_bundle["latents"]
        latents = latents.to(vae.dtype)

        # VAE normalization (reference formula)
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0 / torch.tensor(vae.config.latents_std)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents / latents_std + latents_mean

        # VAE decode (5D input → take frame 0)
        with torch.no_grad():
            decoded = vae.decode(latents, return_dict=False)[0]

        # [B, C, T, H, W] → [B, C, H, W] (take frame 0)
        if decoded.ndim == 5:
            image_tensor = decoded[:, :, 0]
        else:
            image_tensor = decoded

        # Post-process to PIL
        image_tensor = image_tensor.clamp(-1, 1)
        image_tensor = (image_tensor + 1.0) / 2.0
        image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)
        image_np = image_tensor.cpu().float().numpy()
        image_np = (image_np * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(image_np, mode="RGB")
