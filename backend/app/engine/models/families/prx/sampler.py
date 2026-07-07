"""PRXSampler — in-training preview sampler for latent PRX.

Replicates ``PRXPipeline.__call__`` semantics:

Correctness invariants enforced here:
1. TIMESTEP: ``driver.forward_pass`` receives raw [0, 1000] timesteps (the
   shared adapter divides by 1000 before the transformer — PRX convention).
   The scheduler side stays raw.
2. SCHEDULER: checkpoint ``FlowMatchEulerDiscreteScheduler`` with STATIC
   shift 3.0 — plain ``set_timesteps(num_steps)``; NO mu, NO dynamic
   shifting, NO custom sigmas (unlike ovis/longcat/krea2).
3. CFG per the pipeline: gate ``guidance_scale > 1.0``; combine
   ``uncond + guidance_scale * (cond - uncond)``.
4. NO autocast around the DiT forward (autocast-collapse gotcha):
   ``torch.no_grad()``, native model dtype, fp32 latent trajectory. Cached
   TE embeddings are cast to the model dtype at the forward boundary.
5. Native resolution: sample-prompt entries default to the definition's
   512×512 / 28 steps / guidance 4.0 when unset (the generic base assumes
   1024 — off-distribution for this 512-native checkpoint).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
import torch
from PIL import Image
from torch import Tensor

from app.engine.core.sampling import GenericSamplingPipeline

if TYPE_CHECKING:
    from .trainer import PRXTrainer

logger = structlog.get_logger(__name__)

# PRX pipeline / checkpoint defaults.
_PRX_SHIFT: float = 3.0
_PRX_DEFAULT_RESOLUTION: int = 512
_PRX_DEFAULT_STEPS: int = 28
_PRX_DEFAULT_GUIDANCE: float = 4.0


def _combine_cfg(pos: Tensor, neg: Tensor, guidance_scale: float) -> Tensor:
    """PRX CFG combine: ``uncond + guidance_scale * (cond - uncond)``.

    Matches ``PRXPipeline.__call__``
    (``noise_pred = noise_uncond + guidance_scale * (noise_text -
    noise_uncond)``).
    """
    return neg + guidance_scale * (pos - neg)


class PRXSampler(GenericSamplingPipeline):
    """PRX flow-matching sampler with true CFG.

    Structural template: ovis_image/sampler.py.
    Transformer call: reuses ``driver.forward_pass`` (DRY — the normalized
    timestep convention lives in prx_shared's adapter, already tested).
    """

    pipeline: "PRXTrainer"

    def __init__(self, pipeline: "PRXTrainer") -> None:
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
            shift=float(arch.get("scheduler.shift", _PRX_SHIFT)),
            use_dynamic_shifting=bool(
                arch.get("scheduler.use_dynamic_shifting", False),
            ),
        )
        return self._scheduler

    # ── Native-resolution defaults (invariant 5) ─────────────────────────

    def _sample_single(
        self, prompt_cfg: dict[str, Any], step: int
    ) -> Image.Image:
        """Fill PRX-native defaults before the generic sampling flow.

        The generic base defaults unset width/height to 1024 and steps to
        20 — off-distribution for this 512-native checkpoint. Pipeline
        ``__call__`` defaults: 512×512, 28 steps, guidance 4.0 (sourced
        from the definition's ``defaults`` when present).
        """
        cfg = dict(prompt_cfg)
        defaults = getattr(self.pipeline.definition, "defaults", {}) or {}
        resolution = int(defaults.get("resolution", _PRX_DEFAULT_RESOLUTION))
        fill = {
            "width": resolution,
            "height": resolution,
            "num_inference_steps": int(
                defaults.get("num_inference_steps", _PRX_DEFAULT_STEPS),
            ),
            "guidance_scale": float(
                defaults.get("guidance_scale", _PRX_DEFAULT_GUIDANCE),
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
        """Create fp32 noise ``[1, C, h/8, w/8]`` in unpacked latent space.

        Mirrors ``PRXPipeline.prepare_latents`` (plain ``randn`` at
        ``pixels // vae_scale_factor``; the transformer patchifies
        internally, so H/W must divide ``vae_scale_factor * patch_size``
        — 16 for the sft checkpoint, which 512 satisfies).
        """
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        vae_sf = int(arch.get("vae.vae_scale_factor", 8))
        lat_channels = int(arch.get("vae.latent_channels", 16))

        return torch.randn(
            (1, lat_channels, height // vae_sf, width // vae_sf),
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
        """Flow-matching Euler denoising matching PRXPipeline.__call__.

        Precision invariants (binding):
        - Trajectory runs in fp32 (no autocast around the forward).
        - driver.forward_pass receives raw [0,1000] timesteps (the shared
          adapter normalizes — never divide here too).
        - Plain ``set_timesteps(num_steps)`` — no mu / sigmas.
        - CFG only when guidance_scale > 1; combine uncond + g*(cond-uncond).

        Returns:
            fp32 latents ``[1, C, lat_h, lat_w]``.
        """
        device = self.device
        driver = self.pipeline.driver
        transformer = self.pipeline.transformer
        scheduler = self._get_scheduler()

        dtype = next(transformer.parameters()).dtype

        # Model-boundary dtype: embeds may come from an fp32 cache while
        # the transformer is bf16 — align them (trajectory stays fp32).
        prompt_embeds = prompt_embedding["embeds"].to(device=device, dtype=dtype)
        prompt_mask = prompt_embedding["mask"].to(device=device)

        # fp32 trajectory (no autocast)
        latents = noise.to(device=device, dtype=torch.float32)

        # Timestep schedule: plain set_timesteps — the checkpoint's static
        # shift 3.0 is applied inside the scheduler. NO mu (invariant 2).
        scheduler.set_timesteps(num_steps, device=device)
        timesteps = scheduler.timesteps

        # CFG per PRXPipeline: gate at guidance_scale > 1.0.
        cfg_on = float(guidance_scale) > 1.0
        uncond_embeds = None
        uncond_mask = None
        if cfg_on:
            neg_text = str(self.config.get("sample_negative_prompt", "") or "")
            neg_embedding = self.encode_prompt(neg_text)
            uncond_embeds = neg_embedding["embeds"].to(
                device=device, dtype=dtype,
            )
            uncond_mask = neg_embedding["mask"].to(device=device)

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
                    text_embeddings=(prompt_embeds, prompt_mask),
                    batch={},
                ).to(torch.float32)

                if cfg_on:
                    v_uncond = driver.forward_pass(
                        noisy_input=xin,
                        timesteps=ts,
                        text_embeddings=(uncond_embeds, uncond_mask),
                        batch={},
                    ).to(torch.float32)

                    noise_pred = _combine_cfg(v_cond, v_uncond, guidance_scale)
                else:
                    noise_pred = v_cond

                # Scheduler step advances the fp32 trajectory (raw t)
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
