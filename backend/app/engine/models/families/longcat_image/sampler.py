"""LongCat-Image sampling — matches ``LongCatImagePipeline`` from diffusers.

Replicated pipeline semantics:

*  **Packed latent space** — ``[B, (H/2)*(W/2), C*4]`` via ``_pack_latents`` /
   ``_unpack_latents`` (Flux-style 2×2 pixel-unshuffle).
*  **RoPE ids** — text ids ``(0, i, i)`` then image ids at modality 1 offset
   by ``tokenizer_max_length`` (512, 512) — pipeline ``prepare_pos_ids``.
*  **Timesteps** — ``sigmas = linspace(1, 1/steps)``; dynamic shift with
   ``mu = calculate_shift(seq_len, 256, 4096, 0.5, 1.15)``; the transformer
   receives ``t / 1000``.
*  **CFG** — enabled when ``guidance_scale > 1`` with a negative prompt,
   plus cfg_renorm (pipeline default ``enable_cfg_renorm=True``,
   ``cfg_renorm_min=0.0``).
*  **Precision contract** — fp32 latent trajectory, ``torch.no_grad()``,
   NO ``torch.autocast`` around the transformer forward (autocast-collapse
   gotcha).
*  **VAE decode** — ``latents / scaling_factor + shift_factor``.

Deliberately NOT replicated: the pipeline's prompt-rewrite step
(``rewire_prompt`` — a Qwen2.5-VL ``generate`` call).  Training previews
must reflect the trained caption distribution, not a rewritten one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import torch
from PIL import Image
from torch import Tensor

from app.engine.core.sampling import GenericSamplingPipeline
from .driver import TOKENIZER_MAX_LENGTH, LongCatImageDriver

if TYPE_CHECKING:
    from .trainer import LongCatImageTrainer

logger = structlog.get_logger(__name__)


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Empirical mu schedule — copied from LongCatImagePipeline."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _cfg_renorm(
    noise_pred: Tensor, noise_pred_text: Tensor, cfg_renorm_min: float = 0.0,
) -> Tensor:
    """CFG renormalization — copied from LongCatImagePipeline.__call__.

    Rescales the CFG-combined prediction so its per-token norm never exceeds
    the conditional prediction's norm (``scale`` clamped to ``max=1.0``).
    """
    cond_norm = torch.norm(noise_pred_text, dim=-1, keepdim=True)
    noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
    scale = (cond_norm / (noise_norm + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
    return noise_pred * scale


class LongCatImageSampler(GenericSamplingPipeline):
    """LongCat-Image sampler — matches ``LongCatImagePipeline`` from diffusers."""

    pipeline: LongCatImageTrainer

    def __init__(self, pipeline: LongCatImageTrainer) -> None:
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
            use_dynamic_shifting=bool(arch.get("scheduler.use_dynamic_shifting", True)),
            base_shift=float(arch.get("scheduler.base_shift", 0.5)),
            max_shift=float(arch.get("scheduler.max_shift", 1.15)),
            base_image_seq_len=int(arch.get("scheduler.base_image_seq_len", 256)),
            max_image_seq_len=int(arch.get("scheduler.max_image_seq_len", 4096)),
        )
        return self._scheduler

    def _compute_mu(self, image_seq_len: int) -> float:
        """Resolution-derived mu (pipeline ``calculate_shift`` defaults)."""
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        return _calculate_shift(
            image_seq_len,
            int(arch.get("scheduler.base_image_seq_len", 256)),
            int(arch.get("scheduler.max_image_seq_len", 4096)),
            float(arch.get("scheduler.base_shift", 0.5)),
            float(arch.get("scheduler.max_shift", 1.15)),
        )

    def _max_length(self) -> int:
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        return int(arch.get("te.max_length", TOKENIZER_MAX_LENGTH))

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

    # ── Latent packing (matches LongCatImagePipeline exactly) ────────────

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
        """[B, num_patches, C*4] → [B, C, H, W] (pipeline-verbatim, 4D)."""
        batch_size, num_patches, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(
            batch_size, height // 2, width // 2, channels // 4, 2, 2
        )
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch_size, channels // 4, height, width)

    # ── Core sampling methods ────────────────────────────────────────────

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator
    ) -> Tensor:
        """Create packed fp32 noise ``[1, (h/2)*(w/2), C*4]``.

        Matches the pipeline's ``prepare_latents`` (randn in unpacked space,
        then ``_pack_latents``); kept fp32 for the trajectory contract.
        Also stores geometry metadata for ``denoise()``/``decode_latents()``.
        """
        vae = self.pipeline.vae
        if hasattr(vae.config, "block_out_channels"):
            vae_sf = 2 ** (len(vae.config.block_out_channels) - 1)
        else:
            vae_sf = 8

        lat_h = 2 * (height // (vae_sf * 2))
        lat_w = 2 * (width // (vae_sf * 2))
        num_channels = self.pipeline.transformer.config.in_channels // 4

        latents = torch.randn(
            (1, num_channels, lat_h, lat_w),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        latents = self._pack_latents(latents, 1, num_channels, lat_h, lat_w)

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
        """Flow-matching Euler denoising matching LongCatImagePipeline.__call__.

        fp32 latent trajectory; cond + uncond forwards when guidance_scale > 1
        (with cfg_renorm); returns a dict with unpacked 4D ``latents`` ready
        for VAE decode plus geometry metadata.
        """
        device = self.device
        transformer = self.pipeline.transformer
        scheduler = self._get_scheduler()
        dtype = next(transformer.parameters()).dtype

        height = self._sample_height
        width = self._sample_width
        vae_sf = self._vae_sf
        max_length = self._max_length()

        # Model-boundary dtype: embeds may come from an fp32 cache while the
        # transformer is bf16 — align them (the trajectory itself stays fp32).
        prompt_embeds = prompt_embedding["embeds"].to(device=device, dtype=dtype)

        # fp32 trajectory (model inputs are cast per step)
        latents = noise.to(device=device, dtype=torch.float32)

        # Negative prompt for CFG (pipeline: do_classifier_free_guidance
        # when guidance_scale > 1; negative_prompt defaults to "")
        do_cfg = guidance_scale > 1.0
        if do_cfg:
            negative = self.config.get("sample_negative_prompt", "") or ""
            neg_result = self.encode_prompt(negative)
            negative_embeds = neg_result["embeds"].to(device=device, dtype=dtype)
        else:
            negative_embeds = None

        # RoPE ids — text window then image window offset by (512, 512)
        txt_ids = LongCatImageDriver._prepare_text_ids(
            prompt_embeds.shape[1], device,
        )
        if do_cfg:
            neg_txt_ids = LongCatImageDriver._prepare_text_ids(
                negative_embeds.shape[1], device,
            )
        img_ids = LongCatImageDriver._prepare_image_ids(
            self._lat_h // 2, self._lat_w // 2, max_length, device,
        )

        # Timestep schedule: sigmas linspace(1, 1/steps) + dynamic-shift mu
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        image_seq_len = latents.shape[1]
        mu = self._compute_mu(image_seq_len)
        scheduler.set_timesteps(
            num_inference_steps=num_steps, device=device,
            sigmas=sigmas, mu=mu,
        )
        timesteps = scheduler.timesteps

        # Denoising loop
        self._ensure_transformer_on_device(transformer)
        with torch.no_grad():
            total_steps = len(timesteps)
            for step_i, t in enumerate(timesteps, 1):
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Sampling {step_i}/{total_steps}")

                timestep = t.expand(latents.shape[0]).to(dtype)
                latent_input = latents.to(dtype)

                noise_pred_text = transformer(
                    hidden_states=latent_input,
                    timestep=timestep / 1000,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0].float()

                if do_cfg:
                    noise_pred_uncond = transformer(
                        hidden_states=latent_input,
                        timestep=timestep / 1000,
                        encoder_hidden_states=negative_embeds,
                        txt_ids=neg_txt_ids,
                        img_ids=img_ids,
                        return_dict=False,
                    )[0].float()
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    # Pipeline default: enable_cfg_renorm=True, cfg_renorm_min=0.0
                    noise_pred = _cfg_renorm(
                        noise_pred, noise_pred_text, cfg_renorm_min=0.0,
                    )
                else:
                    noise_pred = noise_pred_text

                latents = scheduler.step(
                    noise_pred.to(torch.float32), t, latents, return_dict=False
                )[0]

        # Unpack: [B, num_patches, C*4] → [B, C, H, W]
        latents = self._unpack_latents(latents, height, width, vae_sf)

        return {"latents": latents, "height": height, "width": width}

    def decode_latents(self, latents_bundle: Any) -> Image.Image:
        """VAE-decode latent tensor to PIL image.

        Reference formula: ``latents / scaling_factor + shift_factor``.
        """
        vae = self.pipeline.vae
        latents = latents_bundle["latents"]
        vae_dtype = getattr(vae, "dtype", None) or next(vae.parameters()).dtype

        scaling_factor = getattr(vae.config, "scaling_factor", 1.0) or 1.0
        shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0

        with torch.no_grad():
            scaled = latents.to(dtype=vae_dtype) / scaling_factor + shift_factor
            decoded = vae.decode(scaled, return_dict=False)

        image_tensor = decoded[0] if isinstance(decoded, (tuple, list)) else decoded

        image_tensor = image_tensor.clamp(-1, 1)
        image_tensor = (image_tensor + 1.0) / 2.0
        image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)
        image_np = image_tensor.cpu().float().numpy()
        image_np = (image_np * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(image_np, mode="RGB")
