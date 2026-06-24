"""Krea2Sampler — in-training preview sampler for Krea-2 (Raw + Turbo).

Implements classifier-free guidance flow-matching Euler denoising for Krea-2-Raw
(28-step, CFG) and the few-step distilled path for Krea-2-Turbo (8-step,
guidance_scale=0 → single cond pass).

Correctness invariants enforced here:
1. TIMESTEP: driver.forward_pass receives [0, 1000] timesteps (it divides
   internally by 1000 before the transformer). Never pass ts/1000 again.
2. CFG convention: velocity = cond + guidance_scale*(cond - uncond), only
   when guidance_scale > 0. guidance_scale==0 → single cond pass (Turbo).
3. Distilled mu: is_distilled → fixed mu=1.15. Else → resolution-derived via
   _calculate_shift with Krea defaults.
4. NO autocast around the DiT forward (autocast-collapse gotcha). Mirror
   qwen_image: torch.no_grad(), native dtype, fp32 trajectory.
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
    from .trainer import Krea2Trainer

logger = structlog.get_logger(__name__)

# Fixed mu for distilled (Turbo) checkpoints.
_DISTILLED_MU: float = 1.15

# Krea-2 scheduler defaults (can be overridden via architecture_params YAML).
_KREA2_BASE_SHIFT: float = 0.5
_KREA2_MAX_SHIFT: float = 1.15
_KREA2_BASE_IMAGE_SEQ_LEN: int = 256
_KREA2_MAX_IMAGE_SEQ_LEN: int = 6400


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = _KREA2_BASE_IMAGE_SEQ_LEN,
    max_seq_len: int = _KREA2_MAX_IMAGE_SEQ_LEN,
    base_shift: float = _KREA2_BASE_SHIFT,
    max_shift: float = _KREA2_MAX_SHIFT,
) -> float:
    """Empirical mu schedule — linear interpolation over seq_len."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


class Krea2Sampler(GenericSamplingPipeline):
    """Krea-2 flow-matching sampler — Raw (CFG) + Turbo (distilled, no CFG).

    Structural template: qwen_image/sampler.py
    CFG cond/uncond pattern: ltx2/sampler.py
    Transformer call: reuses driver.forward_pass (DRY — packing/position_ids
    are already implemented and tested there).
    """

    pipeline: "Krea2Trainer"

    def __init__(self, pipeline: "Krea2Trainer") -> None:
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
            base_shift=float(arch.get("scheduler.base_shift", _KREA2_BASE_SHIFT)),
            max_shift=float(arch.get("scheduler.max_shift", _KREA2_MAX_SHIFT)),
            base_image_seq_len=int(
                arch.get("scheduler.base_image_seq_len", _KREA2_BASE_IMAGE_SEQ_LEN)
            ),
            max_image_seq_len=int(
                arch.get("scheduler.max_image_seq_len", _KREA2_MAX_IMAGE_SEQ_LEN)
            ),
        )
        return self._scheduler

    # ── Distilled mu branch ──────────────────────────────────────────────

    def _is_distilled(self) -> bool:
        """Read is_distilled from definition defaults (set per YAML)."""
        defn = self.pipeline.definition
        defaults = getattr(defn, "defaults", {}) or {}
        return bool(defaults.get("is_distilled", False))

    def _compute_mu(self, image_seq_len: int) -> float:
        """Compute flow-matching mu per correctness invariant 3.

        Distilled (Turbo) → fixed 1.15.
        Raw → resolution-derived via _calculate_shift with Krea defaults,
        overrideable via architecture_params YAML.
        """
        if self._is_distilled():
            return _DISTILLED_MU
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        return _calculate_shift(
            image_seq_len,
            base_seq_len=int(arch.get("scheduler.base_image_seq_len", _KREA2_BASE_IMAGE_SEQ_LEN)),
            max_seq_len=int(arch.get("scheduler.max_image_seq_len", _KREA2_MAX_IMAGE_SEQ_LEN)),
            base_shift=float(arch.get("scheduler.base_shift", _KREA2_BASE_SHIFT)),
            max_shift=float(arch.get("scheduler.max_shift", _KREA2_MAX_SHIFT)),
        )

    # ── Text encoding ────────────────────────────────────────────────────

    def encode_prompt(self, prompt: str) -> dict[str, Any]:
        """Encode prompt via trainer's cache-aware encode_text().

        Delegates to pipeline.encode_text (NOT driver.encode_text) so the
        cached path is used when the TE is offloaded after pre-caching.

        Returns dict with ``embeds`` [1, L, num_layers, dim] (4-D) and
        ``mask`` [1, L].
        """
        trainer = self.pipeline
        dtype = next(trainer.transformer.parameters()).dtype
        embeds, mask = trainer.encode_text([prompt], dtype=dtype)
        return {"embeds": embeds, "mask": mask}

    # ── Initial noise ────────────────────────────────────────────────────

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator
    ) -> Tensor:
        """Create noise [B, C, lat_h, lat_w] for Krea-2.

        Krea-2 operates in [B, C, H, W] space (4-D, NOT packed like qwen_image).
        The driver.forward_pass packs internally before feeding the transformer.
        C=16 (latent channels; in_channels=64 = 16 * patch_size^2 after packing).
        """
        device = self.device
        transformer = self.pipeline.transformer
        dtype = next(transformer.parameters()).dtype

        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        vae_sf = int(arch.get("vae.vae_scale_factor", 8))
        lat_channels = int(arch.get("vae.latent_channels", 16))

        lat_h = height // vae_sf
        lat_w = width // vae_sf

        # Store for denoise/decode
        self._lat_h = lat_h
        self._lat_w = lat_w
        self._vae_sf = vae_sf
        self._sample_height = height
        self._sample_width = width

        noise = torch.randn(
            (1, lat_channels, lat_h, lat_w),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        return noise

    # ── Denoise loop ─────────────────────────────────────────────────────

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Any:
        """Flow-matching Euler denoising for Krea-2.

        Precision invariants (binding):
        - Trajectory runs in fp32 (no autocast around forward).
        - driver.forward_pass receives raw [0,1000] timesteps (divides by 1000 internally).
        - CFG: velocity = cond + guidance_scale*(cond - uncond), only when guidance_scale > 0.
        - Distilled mu: fixed 1.15; else resolution-derived.

        Returns dict with ``latents`` [B, C, 1, lat_h, lat_w] and geometry metadata.
        """
        device = self.device
        driver = self.pipeline.driver
        transformer = self.pipeline.transformer
        scheduler = self._get_scheduler()

        dtype = next(transformer.parameters()).dtype

        prompt_embeds = prompt_embedding["embeds"]
        prompt_mask = prompt_embedding["mask"]

        # Latents start as fp32 trajectory (no autocast)
        latents = noise.to(device=device, dtype=torch.float32)

        # Packed image_seq_len for mu calculation.
        # Krea-2 packs [B, C, H, W] → [B, (H/p)*(W/p), C*p*p] inside driver.
        patch_size = int(
            getattr(self.pipeline.definition, "architecture_params", {}).get(
                "transformer.patch_size", 2
            )
        )
        lat_h = self._lat_h
        lat_w = self._lat_w
        img_seq_len = (lat_h // patch_size) * (lat_w // patch_size)

        mu = self._compute_mu(img_seq_len)

        # Timestep schedule: descending sigmas [1.0 → 1/num_steps]
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        scheduler.set_timesteps(
            num_inference_steps=num_steps,
            sigmas=sigmas,
            mu=mu,
            device=device,
        )
        timesteps = scheduler.timesteps

        # CFG: encode negative prompt when guidance_scale > 0
        cfg_on = float(guidance_scale) > 0.0
        uncond_embeds = None
        uncond_mask = None
        if cfg_on:
            neg_text = str(self.config.get("sample_negative_prompt", "") or "")
            neg_embedding = self.encode_prompt(neg_text)
            uncond_embeds = neg_embedding["embeds"]
            uncond_mask = neg_embedding["mask"]

        # Move transformer to device (respects block-swap if active)
        self._ensure_transformer_on_device(transformer)

        scheduler.set_begin_index(0)

        # Denoise loop — fp32 trajectory, no autocast (invariant 4).
        # driver.forward_pass receives raw [0,1000] timesteps (invariant 1).
        with torch.no_grad():
            total_steps = len(timesteps)
            for step_i, t in enumerate(timesteps, 1):
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Sampling {step_i}/{total_steps}")

                # Cast latents to model dtype for the forward; keep fp32 trajectory.
                xin = latents.to(dtype=dtype)

                # Raw [0, 1000] timestep for driver.forward_pass.
                # scheduler.timesteps are already in [0, 1000] scale.
                ts = t.expand(xin.shape[0]).to(dtype=dtype)

                # Conditional velocity
                v_cond = driver.forward_pass(
                    noisy_input=xin,
                    timesteps=ts,
                    text_embeddings=(
                        prompt_embeds.to(dtype=dtype),
                        prompt_mask,
                    ),
                    batch={},
                ).to(torch.float32)

                if cfg_on:
                    # Unconditional velocity
                    v_uncond = driver.forward_pass(
                        noisy_input=xin,
                        timesteps=ts,
                        text_embeddings=(
                            uncond_embeds.to(dtype=dtype),
                            uncond_mask,
                        ),
                        batch={},
                    ).to(torch.float32)

                    # Krea-2 CFG convention: velocity = cond + gs*(cond - uncond)
                    noise_pred = v_cond + guidance_scale * (v_cond - v_uncond)
                else:
                    noise_pred = v_cond

                # Scheduler step advances the fp32 trajectory
                latents = scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0].to(torch.float32)

        # Expand to 5-D [B, C, 1, H, W] for VAE decode (matches qwen_image convention)
        latents_5d = latents.unsqueeze(2)

        return {
            "latents": latents_5d,
            "height": self._sample_height,
            "width": self._sample_width,
        }

    # ── VAE decode ───────────────────────────────────────────────────────

    def decode_latents(self, latents_bundle: Any) -> Image.Image:
        """VAE-decode latent tensor to PIL image.

        Mirrors qwen_image: same AutoencoderKLQwenImage VAE with 5-D normalization.
        latents_bundle: dict with ``latents`` [B, C, 1, H, W].
        """
        vae = self.pipeline.vae
        latents = latents_bundle["latents"]
        latents = latents.to(vae.dtype)

        # VAE normalization (reference formula, same as qwen_image)
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

        # VAE decode (5-D input → take frame 0)
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
