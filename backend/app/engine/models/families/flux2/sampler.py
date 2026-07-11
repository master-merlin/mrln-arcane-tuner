"""FLUX.2 Klein sampler — implements ``Flux2KleinPipeline`` behavior.

The official Klein pipeline (``Flux2KleinPipeline``, diffusers >=0.37)
differs from ``Flux2Pipeline`` in several critical ways:

*  **guidance=None** — Klein passes ``guidance=None`` to the transformer
   (no guidance embedder used).
*  **Classifier-free guidance** — Klein uses standard CFG: two-pass
   (cond + uncond) with ``neg + scale * (cond - neg)``.
*  **Qwen3 text encoder** — uses ``_get_qwen3_prompt_embeds`` with
   ``apply_chat_template(enable_thinking=False)`` and ``attention_mask``.

This sampler reproduces that behavior using the trainer's already-loaded
components (transformer, VAE, text encoder, scheduler).
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
    from .trainer import Flux2Trainer

logger = structlog.get_logger(__name__)

# Native in-training-preview defaults (FLUX.2 / Klein pipeline __call__:
# 50 steps, guidance 4.0, 1024²). Definition-sourced via each YAML's
# `defaults`; these constants are the fallback used only when a definition
# omits the key. The generic base's 20 steps / 3.5 guidance are off-
# distribution for FLUX.2 (its native pipeline uses 50 / 4.0).
_FLUX2_DEFAULT_RESOLUTION: int = 1024
_FLUX2_DEFAULT_STEPS: int = 50
_FLUX2_DEFAULT_GUIDANCE: float = 4.0


def _compute_mu(image_seq_len: int, num_steps: int) -> float:
    """Empirical mu schedule matching ``Flux2KleinPipeline``."""
    mu = 0.5 + 0.5 * np.log2(image_seq_len / 256)
    return mu


class Flux2Sampler(GenericSamplingPipeline):
    """FLUX.2 Klein sampler — Euler + CFG matching KleinPipeline."""

    pipeline: Flux2Trainer

    def __init__(self, pipeline: Flux2Trainer) -> None:
        super().__init__(pipeline)
        self._scheduler = None

    # ── Native sample defaults (W3-1; ovis/boogu precedent) ──────────────

    def _sample_single(self, prompt_cfg: dict[str, Any], step: int) -> Any:
        """Fill FLUX.2's native preview defaults before the generic flow.

        Sources 50 steps / 4.0 guidance / 1024² from the definition's
        ``defaults`` (constants are fallback only) — the FLUX.2/Klein
        pipeline's own ``__call__`` defaults, versus the generic base's
        off-distribution 20 steps / 3.5 guidance. Explicit per-prompt values
        always win (fill only when unset/0).
        """
        cfg = dict(prompt_cfg)
        defaults = getattr(self.pipeline.definition, "defaults", {}) or {}
        resolution = int(defaults.get("resolution", _FLUX2_DEFAULT_RESOLUTION))
        fill = {
            "width": resolution,
            "height": resolution,
            "num_inference_steps": int(
                defaults.get("num_inference_steps", _FLUX2_DEFAULT_STEPS),
            ),
            "guidance_scale": float(
                defaults.get("guidance_scale", _FLUX2_DEFAULT_GUIDANCE),
            ),
        }
        for key, value in fill.items():
            if cfg.get(key) in (None, 0):
                cfg[key] = value
        return super()._sample_single(cfg, step)

    # ── Lazy scheduler ───────────────────────────────────────────────────

    def _get_scheduler(self):
        if self._scheduler is not None:
            return self._scheduler
        from diffusers import FlowMatchEulerDiscreteScheduler

        self._scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=1.15,
            base_image_seq_len=256,
            max_image_seq_len=4096,
        )
        return self._scheduler

    # ── Text encoding (Qwen3, same as KleinPipeline) ────────────────────

    def encode_prompt(self, prompt: str) -> dict[str, Any]:
        """Encode prompt AND empty negative prompt for CFG.

        Returns dict with ``cond`` and ``uncond`` embeddings + text_ids.
        """
        trainer = self.pipeline
        device = self.device
        dtype = next(trainer.transformer.parameters()).dtype

        # Positive prompt
        cond_embeds = trainer.encode_text([prompt], dtype=dtype)
        cond_embeds = cond_embeds.to(device=device, dtype=dtype)
        cond_text_ids = self._prepare_text_ids(cond_embeds).to(device)

        # Negative prompt (empty string) for CFG
        uncond_embeds = trainer.encode_text([""], dtype=dtype)
        uncond_embeds = uncond_embeds.to(device=device, dtype=dtype)
        uncond_text_ids = self._prepare_text_ids(uncond_embeds).to(device)

        return {
            "cond": cond_embeds,
            "cond_ids": cond_text_ids,
            "uncond": uncond_embeds,
            "uncond_ids": uncond_text_ids,
        }

    # ── Latent helpers (match KleinPipeline exactly) ─────────────────────

    @staticmethod
    def _prepare_text_ids(x: Tensor) -> Tensor:
        """Create text position IDs: [B, L, 4]."""
        B, L, _ = x.shape
        out = []
        for _ in range(B):
            t = torch.arange(1)
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)  # noqa: E741
            coords = torch.cartesian_prod(t, h, w, l)
            out.append(coords)
        return torch.stack(out)

    @staticmethod
    def _prepare_latent_ids(latents: Tensor) -> Tensor:
        """Create latent position IDs: [B, H*W, 4]."""
        B, _, H, W = latents.shape
        t = torch.arange(1)
        h = torch.arange(H)
        w = torch.arange(W)
        l = torch.arange(1)  # noqa: E741
        ids = torch.cartesian_prod(t, h, w, l)
        return ids.unsqueeze(0).expand(B, -1, -1)

    @staticmethod
    def _pack_latents(latents: Tensor) -> Tensor:
        """[B, C, H, W] -> [B, H*W, C]."""
        B, C, H, W = latents.shape
        return latents.reshape(B, C, H * W).permute(0, 2, 1)

    @staticmethod
    def _unpack_latents(x: Tensor, x_ids: Tensor) -> Tensor:
        """Scatter-unpack: [B, H*W, C] -> [B, C, H, W]."""
        out_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1
            flat = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat.unsqueeze(1).expand(-1, ch), data)
            out = out.view(h, w, ch).permute(2, 0, 1)
            out_list.append(out)
        return torch.stack(out_list, dim=0)

    @staticmethod
    def _unpatchify(latents: Tensor) -> Tensor:
        """Undo 2x2 patchification."""
        B, C, H, W = latents.shape
        latents = latents.reshape(B, C // 4, 2, 2, H, W)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(B, C // 4, H * 2, W * 2)

    # ── Core sampling methods ────────────────────────────────────────────

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator
    ) -> Tensor:
        """Create packed noise [1, L, 128] for Klein-style transformer.

        VAE downscale = 8, patch factor = 2 → grid is (H/16) × (W/16).
        Channels: 32 VAE × 4 (2×2 patch) = 128.
        """
        device = self.device
        # VAE spatial compression: 8x downscale
        vae_sf = 8
        lat_h = height // vae_sf
        lat_w = width // vae_sf
        num_channels = 32  # Klein VAE latent channels

        # Create noise in patchified space [B, C*4, H/2, W/2]
        pH, pW = lat_h // 2, lat_w // 2
        shape = (1, num_channels * 4, pH, pW)
        latents = torch.randn(shape, generator=generator, device=device)

        # Prepare latent IDs [B, H*W, 4] — inline _prepare_latent_ids
        t = torch.arange(1)
        h = torch.arange(pH)
        w = torch.arange(pW)
        l = torch.arange(1)  # noqa: E741
        ids = torch.cartesian_prod(t, h, w, l)
        self._latent_ids = ids.unsqueeze(0).expand(1, -1, -1).to(device)

        # Pack to [B, L, C] — inline _pack_latents
        B, C, H, W = latents.shape
        return latents.reshape(B, C, H * W).permute(0, 2, 1)

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Any:
        """Klein-style denoising loop: Euler + CFG (guidance=None).

        Returns BN-denormalised unpatchified latent tensor for
        ``decode_latents()`` to VAE-decode.
        """
        device = self.device
        transformer = self.pipeline.transformer
        vae = self.pipeline.vae
        scheduler = self._get_scheduler()
        dtype = next(transformer.parameters()).dtype

        # Unpack text embeddings
        cond = prompt_embedding["cond"]
        cond_ids = prompt_embedding["cond_ids"]
        uncond = prompt_embedding["uncond"]
        uncond_ids = prompt_embedding["uncond_ids"]

        do_cfg = guidance_scale > 1.0

        # Use noise created by _create_initial_noise
        latents = noise.to(device=device, dtype=dtype)
        latent_ids = self._latent_ids.to(device)

        # Timestep schedule (dynamic shifting)
        image_seq_len = latents.shape[1]
        mu = _compute_mu(image_seq_len, num_steps)
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        scheduler.set_timesteps(
            num_inference_steps=num_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps = scheduler.timesteps

        # Denoising loop
        self._ensure_transformer_on_device(transformer)
        with torch.no_grad():
            total_steps = len(timesteps)
            for step_i, t in enumerate(timesteps, 1):
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Sampling {step_i}/{total_steps}")
                ts = t.expand(latents.shape[0]).to(dtype)
                latent_input = latents.to(dtype)

                # Conditional forward pass (guidance=None for Klein)
                noise_pred = transformer(
                    hidden_states=latent_input,
                    timestep=ts / 1000,
                    guidance=None,
                    encoder_hidden_states=cond,
                    txt_ids=cond_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]

                # CFG: unconditional forward pass + combine
                if do_cfg:
                    neg_pred = transformer(
                        hidden_states=latent_input,
                        timestep=ts / 1000,
                        guidance=None,
                        encoder_hidden_states=uncond,
                        txt_ids=uncond_ids,
                        img_ids=latent_ids,
                        return_dict=False,
                    )[0]
                    noise_pred = neg_pred + guidance_scale * (
                        noise_pred - neg_pred
                    )

                # Euler step
                latents = scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

        # Post-process: unpack latents (VAE decode happens in decode_latents)
        latents = self._unpack_latents(latents, latent_ids)

        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
            latents.device, latents.dtype
        )
        bn_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = latents * bn_std + bn_mean

        latents = self._unpatchify(latents)

        return latents

    def decode_latents(self, latents: Any) -> Image.Image:
        """VAE-decode BN-denormalised latent tensor to PIL image.

        Args:
            latents: Unpatchified latent tensor [1, C, H, W].

        Returns:
            PIL Image in RGB mode.
        """
        vae = self.pipeline.vae
        with torch.no_grad():
            image = vae.decode(latents.to(vae.dtype), return_dict=False)[0]

        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image[0] * 255).round().astype("uint8")
        return Image.fromarray(image)
