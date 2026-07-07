"""PRXPixelSampler — in-training preview sampler for pixel-space PRX.

Replicates ``PRXPixelPipeline.__call__`` semantics:

Correctness invariants enforced here:
1. INITIAL NOISE: pixel-space ``randn(1, 3, H, W) × noise_scale`` (2.0) —
   ``prepare_latents`` scales the Gaussian start because the model was
   trained against non-unit noise. NO VAE downscale (vae_scale_factor 1).
2. X0 → VELOCITY: the model predicts the CLEAN image. Per step the
   prediction is converted before the scheduler:
   ``v = (latents - x0_cfg) / clamp(t/1000, min=0.05)`` — with CFG applied
   to the x0 PREDICTION first (pipeline-verbatim order).
3. TIMESTEP: ``driver.forward_pass`` receives raw [0, 1000] timesteps (the
   shared adapter divides by 1000 before the transformer — PRX convention).
   The scheduler side stays raw.
4. SCHEDULER: checkpoint ``FlowMatchEulerDiscreteScheduler`` with STATIC
   shift 3.0 — plain ``set_timesteps(num_steps)``; NO mu, NO dynamic
   shifting, NO custom sigmas.
5. CFG per the pipeline: gate ``guidance_scale > 1.0``; combine
   ``uncond + guidance_scale * (cond - uncond)`` on the x0 preds.
6. NO autocast around the DiT forward (autocast-collapse gotcha):
   ``torch.no_grad()``, native model dtype, fp32 latent trajectory. Cached
   TE embeddings are cast to the model dtype at the forward boundary.
7. NO VAE DECODE: the denoised output IS the image in [-1, 1] —
   ``decode_latents`` is a pure clamp → [0, 255] postprocess (reuses the
   pixel-space no-decode convention from hidream_o1).
8. Native resolution: sample-prompt entries default to the definition's
   1024×1024 / 28 steps / guidance 4.0 when unset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
import torch
from PIL import Image
from torch import Tensor

from app.engine.core.sampling import GenericSamplingPipeline
from app.engine.models.families.prx_pixel.driver import x0_to_velocity

if TYPE_CHECKING:
    from .trainer import PRXPixelTrainer

logger = structlog.get_logger(__name__)

# PRXPixel pipeline / checkpoint defaults.
_PRX_PIXEL_SHIFT: float = 3.0
_PRX_PIXEL_DEFAULT_RESOLUTION: int = 1024
_PRX_PIXEL_DEFAULT_STEPS: int = 28
_PRX_PIXEL_DEFAULT_GUIDANCE: float = 4.0
_PRX_PIXEL_NOISE_SCALE: float = 2.0
_PRX_PIXEL_T_FLOOR: float = 0.05


def _combine_cfg(pos: Tensor, neg: Tensor, guidance_scale: float) -> Tensor:
    """PRX CFG combine: ``uncond + guidance_scale * (cond - uncond)``.

    Matches ``PRXPixelPipeline.__call__`` — applied to the x0 PREDICTIONS
    (``noise_pred = noise_uncond + guidance_scale * (noise_text -
    noise_uncond)`` happens BEFORE the x0→velocity conversion).
    """
    return neg + guidance_scale * (pos - neg)


class PRXPixelSampler(GenericSamplingPipeline):
    """PRXPixel flow-matching sampler with x0 conversion and true CFG.

    Structural template: prx/sampler.py (latent sibling).
    Transformer call: reuses ``driver.forward_pass`` (DRY — the normalized
    timestep convention lives in prx_shared's adapter, already tested).
    """

    pipeline: "PRXPixelTrainer"

    def __init__(self, pipeline: "PRXPixelTrainer") -> None:
        super().__init__(pipeline)
        self._scheduler = None

    # ── Lazy scheduler ───────────────────────────────────────────────────

    def _get_scheduler(self):
        """Checkpoint scheduler: static shift 3.0, NO dynamic shifting."""
        if self._scheduler is not None:
            return self._scheduler
        from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: PLC0415

        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        self._scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=int(
                arch.get("scheduler.num_train_timesteps", 1000),
            ),
            shift=float(arch.get("scheduler.shift", _PRX_PIXEL_SHIFT)),
            use_dynamic_shifting=bool(
                arch.get("scheduler.use_dynamic_shifting", False),
            ),
        )
        return self._scheduler

    # ── Native-resolution defaults (invariant 8) ─────────────────────────

    def _sample_single(self, prompt_cfg: dict[str, Any], step: int) -> Image.Image:
        """Fill PRXPixel-native defaults before the generic sampling flow.

        Pipeline ``__call__`` defaults: 1024×1024, 28 steps, guidance 4.0
        (sourced from the definition's ``defaults`` when present).
        """
        cfg = dict(prompt_cfg)
        defaults = getattr(self.pipeline.definition, "defaults", {}) or {}
        resolution = int(
            defaults.get("resolution", _PRX_PIXEL_DEFAULT_RESOLUTION),
        )
        fill = {
            "width": resolution,
            "height": resolution,
            "num_inference_steps": int(
                defaults.get("num_inference_steps", _PRX_PIXEL_DEFAULT_STEPS),
            ),
            "guidance_scale": float(
                defaults.get("guidance_scale", _PRX_PIXEL_DEFAULT_GUIDANCE),
            ),
        }
        for key, value in fill.items():
            if cfg.get(key) in (None, 0):
                cfg[key] = value
        return super()._sample_single(cfg, step)

    # ── Text encoding ────────────────────────────────────────────────────

    def encode_prompt(self, prompt: str) -> dict[str, Any]:
        """Encode prompt via the trainer's cache-aware ``encode_text()``.

        Delegates to ``pipeline.encode_text`` (NOT ``driver.encode_text``)
        so the cached path is used when the TE is offloaded after
        pre-caching. Returns dict with ``embeds`` [1, L, D] and BOOL
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
        """Create fp32 noise ``[1, 3, H, W] × noise_scale`` in PIXEL space.

        Mirrors ``PRXPixelPipeline.prepare_latents``: full-resolution randn
        (vae_scale_factor is 1 — no downscale) multiplied by the pipeline's
        ``noise_scale`` (2.0). H/W must divide the transformer's patch_size
        (16 for the checkpoint), which the 1024-native defaults satisfy.
        """
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        in_channels = int(arch.get("transformer.in_channels", 3))
        noise_scale = float(
            arch.get("pipeline.noise_scale", _PRX_PIXEL_NOISE_SCALE),
        )

        noise = torch.randn(
            (1, in_channels, height, width),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        return noise * noise_scale

    # ── Denoise loop ─────────────────────────────────────────────────────

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Tensor:
        """x0-prediction Euler denoising matching PRXPixelPipeline.__call__.

        Precision invariants (binding):
        - Trajectory runs in fp32 (no autocast around the forward).
        - driver.forward_pass receives raw [0,1000] timesteps (the shared
          adapter normalizes — never divide here too).
        - Plain ``set_timesteps(num_steps)`` — no mu / sigmas.
        - CFG only when guidance_scale > 1; combine on the x0 predictions,
          THEN convert to velocity with the 0.05 t-floor.

        Returns:
            fp32 pixels ``[1, 3, H, W]`` in ``[-1, 1]`` (the image itself).
        """
        device = self.device
        driver = self.pipeline.driver
        transformer = self.pipeline.transformer
        scheduler = self._get_scheduler()

        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        num_train_timesteps = int(
            arch.get("scheduler.num_train_timesteps", 1000),
        )
        t_floor = float(
            arch.get("pipeline.velocity_t_floor", _PRX_PIXEL_T_FLOOR),
        )

        dtype = next(transformer.parameters()).dtype

        # Model-boundary dtype: embeds may come from an fp32 cache while
        # the transformer is bf16 — align them (trajectory stays fp32).
        prompt_embeds = prompt_embedding["embeds"].to(device=device, dtype=dtype)
        prompt_mask = prompt_embedding["mask"].to(device=device)

        # fp32 trajectory (no autocast)
        latents = noise.to(device=device, dtype=torch.float32)

        # Timestep schedule: plain set_timesteps — the checkpoint's static
        # shift 3.0 is applied inside the scheduler. NO mu (invariant 4).
        scheduler.set_timesteps(num_steps, device=device)
        timesteps = scheduler.timesteps

        # CFG per PRXPixelPipeline: gate at guidance_scale > 1.0.
        cfg_on = float(guidance_scale) > 1.0
        uncond_embeds = None
        uncond_mask = None
        if cfg_on:
            neg_text = str(self.config.get("sample_negative_prompt", "") or "")
            neg_embedding = self.encode_prompt(neg_text)
            uncond_embeds = neg_embedding["embeds"].to(
                device=device,
                dtype=dtype,
            )
            uncond_mask = neg_embedding["mask"].to(device=device)

        # Move transformer to device (respects block-swap if active)
        self._ensure_transformer_on_device(transformer)

        scheduler.set_begin_index(0)

        # Denoise loop — fp32 trajectory, no autocast (invariant 6);
        # driver.forward_pass receives raw [0,1000] timesteps (invariant 3).
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

                x0_cond = driver.forward_pass(
                    noisy_input=xin,
                    timesteps=ts,
                    text_embeddings=(prompt_embeds, prompt_mask),
                    batch={},
                ).to(torch.float32)

                if cfg_on:
                    x0_uncond = driver.forward_pass(
                        noisy_input=xin,
                        timesteps=ts,
                        text_embeddings=(uncond_embeds, uncond_mask),
                        batch={},
                    ).to(torch.float32)

                    x0_pred = _combine_cfg(x0_cond, x0_uncond, guidance_scale)
                else:
                    x0_pred = x0_cond

                # x0 → velocity with the pipeline's t-floor (invariant 2):
                # v = (latents - x0_pred) / clamp(t/1000, min=0.05).
                velocity = x0_to_velocity(
                    latents,
                    x0_pred,
                    t,
                    num_train_timesteps=num_train_timesteps,
                    t_floor=t_floor,
                )

                # Scheduler step advances the fp32 trajectory (raw t)
                latents = scheduler.step(velocity, t, latents, return_dict=False)[
                    0
                ].to(torch.float32)

        return latents

    # ── Pixel postprocess (NO VAE) ───────────────────────────────────────

    def decode_latents(self, latents: Any) -> Image.Image:
        """Convert the denoised pixel tensor to a PIL image — NO VAE.

        Pipeline step 8: the output IS the image in ``[-1, 1]``; postprocess
        is clamp → ``(x + 1) / 2`` → uint8 (the same math the latent
        families apply AFTER their VAE decode).
        """
        image_tensor = latents.to(torch.float32).clamp(-1, 1)
        image_tensor = (image_tensor + 1.0) / 2.0
        image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)
        image_np = image_tensor.cpu().float().numpy()
        image_np = (image_np * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(image_np, mode="RGB")
