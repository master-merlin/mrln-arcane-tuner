"""OvisImageSampler — in-training preview sampler for Ovis-Image.

Replicates ``OvisImagePipeline.__call__`` semantics:

Correctness invariants enforced here:
1. TIMESTEP: ``driver.forward_pass`` receives raw [0, 1000] timesteps (it
   divides internally by 1000 before the transformer). Never pass ts/1000
   again.
2. CFG per the pipeline: gate ``guidance_scale > 1``; combine
   ``velocity = neg + guidance_scale * (pos - neg)``.
3. mu/shift: ``sigmas = linspace(1, 1/num_steps, num_steps)`` and
   ``mu = calculate_shift(image_seq_len, 256, 4096, 0.5, 1.15)`` with the
   checkpoint's dynamic-shifting FlowMatchEulerDiscreteScheduler.
4. NO autocast around the DiT forward (autocast-collapse gotcha):
   ``torch.no_grad()``, native model dtype, fp32 latent trajectory.
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
    from .trainer import OvisImageTrainer

logger = structlog.get_logger(__name__)

# Ovis scheduler defaults (checkpoint scheduler_config.json; overrideable
# via architecture_params YAML).
_OVIS_SHIFT: float = 3.0
_OVIS_BASE_SHIFT: float = 0.5
_OVIS_MAX_SHIFT: float = 1.15
_OVIS_BASE_IMAGE_SEQ_LEN: int = 256
_OVIS_MAX_IMAGE_SEQ_LEN: int = 4096


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = _OVIS_BASE_IMAGE_SEQ_LEN,
    max_seq_len: int = _OVIS_MAX_IMAGE_SEQ_LEN,
    base_shift: float = _OVIS_BASE_SHIFT,
    max_shift: float = _OVIS_MAX_SHIFT,
) -> float:
    """Empirical mu schedule — copied from the diffusers Ovis pipeline."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _combine_cfg(pos: Tensor, neg: Tensor, guidance_scale: float) -> Tensor:
    """Ovis CFG combine: ``neg + guidance_scale * (pos - neg)``.

    Matches ``OvisImagePipeline.__call__``
    (``noise_pred = neg_noise_pred + guidance_scale * (noise_pred -
    neg_noise_pred)``) — NOT the krea2 ``cond + g*(cond - uncond)`` form.
    """
    return neg + guidance_scale * (pos - neg)


class OvisImageSampler(GenericSamplingPipeline):
    """Ovis-Image flow-matching sampler with true CFG.

    Structural template: krea2/sampler.py.
    Transformer call: reuses ``driver.forward_pass`` (DRY — packing,
    img_ids/txt_ids, and the /1000 timestep scale are already implemented
    and tested there).
    """

    pipeline: "OvisImageTrainer"

    def __init__(self, pipeline: "OvisImageTrainer") -> None:
        super().__init__(pipeline)
        self._scheduler = None

    # ── Lazy scheduler ───────────────────────────────────────────────────

    def _get_scheduler(self):
        if self._scheduler is not None:
            return self._scheduler
        from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: PLC0415

        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        self._scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=int(
                arch.get("scheduler.num_train_timesteps", 1000),
            ),
            shift=float(arch.get("scheduler.shift", _OVIS_SHIFT)),
            use_dynamic_shifting=bool(
                arch.get("scheduler.use_dynamic_shifting", True),
            ),
            base_shift=float(arch.get("scheduler.base_shift", _OVIS_BASE_SHIFT)),
            max_shift=float(arch.get("scheduler.max_shift", _OVIS_MAX_SHIFT)),
            base_image_seq_len=int(
                arch.get("scheduler.base_image_seq_len", _OVIS_BASE_IMAGE_SEQ_LEN),
            ),
            max_image_seq_len=int(
                arch.get("scheduler.max_image_seq_len", _OVIS_MAX_IMAGE_SEQ_LEN),
            ),
        )
        return self._scheduler

    # ── mu (dynamic shift) ───────────────────────────────────────────────

    def _compute_mu(self, image_seq_len: int) -> float:
        """Resolution-derived mu via the pipeline's calculate_shift."""
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        return _calculate_shift(
            image_seq_len,
            base_seq_len=int(
                arch.get("scheduler.base_image_seq_len", _OVIS_BASE_IMAGE_SEQ_LEN),
            ),
            max_seq_len=int(
                arch.get("scheduler.max_image_seq_len", _OVIS_MAX_IMAGE_SEQ_LEN),
            ),
            base_shift=float(arch.get("scheduler.base_shift", _OVIS_BASE_SHIFT)),
            max_shift=float(arch.get("scheduler.max_shift", _OVIS_MAX_SHIFT)),
        )

    # ── Text encoding ────────────────────────────────────────────────────

    def encode_prompt(self, prompt: str) -> dict[str, Any]:
        """Encode prompt via the trainer's cache-aware ``encode_text()``.

        Delegates to ``pipeline.encode_text`` (NOT ``driver.encode_text``)
        so the cached path is used when the TE is offloaded after
        pre-caching. Returns dict with ``embeds`` [1, L, D] and ``mask``
        [1, L].
        """
        trainer = self.pipeline
        dtype = next(trainer.transformer.parameters()).dtype
        embeds, mask = trainer.encode_text([prompt], dtype=dtype)
        return {"embeds": embeds, "mask": mask}

    # ── Initial noise ────────────────────────────────────────────────────

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator
    ) -> Tensor:
        """Create noise [1, C, lat_h, lat_w] in unpacked latent space.

        The driver packs 2×2 internally, so latent dims must be even —
        mirrored from ``OvisImagePipeline.prepare_latents``:
        ``lat = 2 * (px // (vae_scale_factor * 2))``.
        """
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        vae_sf = int(arch.get("vae.vae_scale_factor", 8))
        lat_channels = int(arch.get("vae.latent_channels", 16))

        lat_h = 2 * (height // (vae_sf * 2))
        lat_w = 2 * (width // (vae_sf * 2))

        # Store for denoise/decode
        self._sample_height = height
        self._sample_width = width

        return torch.randn(
            (1, lat_channels, lat_h, lat_w),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )

    # ── Denoise loop ─────────────────────────────────────────────────────

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Tensor:
        """Flow-matching Euler denoising matching OvisImagePipeline.__call__.

        Precision invariants (binding):
        - Trajectory runs in fp32 (no autocast around the forward).
        - driver.forward_pass receives raw [0,1000] timesteps.
        - CFG only when guidance_scale > 1; combine neg + g*(pos - neg).

        Returns:
            fp32 latents ``[1, C, lat_h, lat_w]``.
        """
        device = self.device
        driver = self.pipeline.driver
        transformer = self.pipeline.transformer
        scheduler = self._get_scheduler()

        dtype = next(transformer.parameters()).dtype

        prompt_embeds = prompt_embedding["embeds"]
        prompt_mask = prompt_embedding["mask"]

        # fp32 trajectory (no autocast)
        latents = noise.to(device=device, dtype=torch.float32)

        # Packed image_seq_len for mu (driver packs 2×2 internally).
        img_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        mu = self._compute_mu(img_seq_len)

        # Timestep schedule: descending sigmas [1.0 → 1/num_steps] + mu.
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        scheduler.set_timesteps(
            num_inference_steps=num_steps,
            sigmas=sigmas,
            mu=mu,
            device=device,
        )
        timesteps = scheduler.timesteps

        # CFG per OvisImagePipeline: gate at guidance_scale > 1.
        cfg_on = float(guidance_scale) > 1.0
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

        # Denoise loop — fp32 trajectory, no autocast (invariant 4);
        # driver.forward_pass receives raw [0,1000] timesteps (invariant 1).
        with torch.no_grad():
            total_steps = len(timesteps)
            for step_i, t in enumerate(timesteps, 1):
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Sampling {step_i}/{total_steps}")

                # Cast latents to model dtype for the forward; keep fp32
                # trajectory outside.
                xin = latents.to(dtype=dtype)

                # Raw [0, 1000] timestep for driver.forward_pass.
                ts = t.expand(xin.shape[0]).to(dtype=dtype)

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
                    v_uncond = driver.forward_pass(
                        noisy_input=xin,
                        timesteps=ts,
                        text_embeddings=(
                            uncond_embeds.to(dtype=dtype),
                            uncond_mask,
                        ),
                        batch={},
                    ).to(torch.float32)

                    noise_pred = _combine_cfg(v_cond, v_uncond, guidance_scale)
                else:
                    noise_pred = v_cond

                # Scheduler step advances the fp32 trajectory
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[
                    0
                ].to(torch.float32)

        return latents

    # ── VAE decode ───────────────────────────────────────────────────────

    def decode_latents(self, latents: Any) -> Image.Image:
        """Decode latent tensor to a PIL image.

        Matches the pipeline:
        ``latents / scaling_factor + shift_factor`` → ``vae.decode``.
        """
        vae = self.pipeline.vae
        vae_dtype = next(vae.parameters()).dtype

        vae.to(self.device)
        with torch.no_grad():
            scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
            shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
            scaled = latents.to(dtype=vae_dtype) / scaling_factor + shift_factor
            decoded = vae.decode(scaled, return_dict=False)

        if isinstance(decoded, (tuple, list)):
            image_tensor = decoded[0]
        else:
            image_tensor = decoded

        image_tensor = image_tensor.clamp(-1, 1)
        image_tensor = (image_tensor + 1.0) / 2.0
        image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)
        image_np = image_tensor.cpu().float().numpy()
        image_np = (image_np * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(image_np, mode="RGB")
