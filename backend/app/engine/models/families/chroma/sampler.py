"""ChromaSampler — in-training preview sampler for Chroma.

Replicates ``ChromaPipeline.__call__`` semantics (diffusers 0.39,
``venv/Lib/site-packages/diffusers/pipelines/chroma/pipeline_chroma.py``):

Correctness invariants enforced here:
1. TIMESTEP: ``driver.forward_pass`` receives raw ``[0, 1000]`` timesteps (it
   divides internally by 1000 before the transformer). Never pass ts/1000
   again.
2. CFG is REAL (not guidance-distilled) per the pipeline: gate
   ``guidance_scale > 1`` (``do_classifier_free_guidance`` property, lines
   626-627); combine ``velocity = neg + guidance_scale * (pos - neg)``
   (lines 920-933) — two full transformer forward passes per step.
3. Scheduler: a REAL ``FlowMatchEulerDiscreteScheduler`` instantiated from
   the definition's ``architecture_params`` (NOT hand-rolled shift math) —
   this correctly branches on the checkpoint's OWN config rather than
   assuming Flux/Ovis-style dynamic shifting. Chroma1-HD ships
   ``use_dynamic_shifting=false`` + a static ``shift=3.0`` (the pipeline
   ALWAYS computes + passes ``mu`` regardless, lines 833-846 — the
   scheduler silently ignores it when not dynamic, see
   ``scheduling_flow_match_euler_discrete.py`` lines 309/348-351); Base
   ships an even sparser config (only ``num_train_timesteps`` +
   ``use_beta_sigmas=true`` — everything else at the scheduler class's own
   defaults, i.e. static ``shift=1.0`` == no shift + Karras-style beta
   resampling of the sigma schedule). Letting the real scheduler class
   handle ``use_beta_sigmas``/``use_karras_sigmas``/``use_exponential_sigmas``
   plus dynamic-vs-static shift avoids re-deriving that branching logic by
   hand and keeps both definitions correct from one code path.
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
from .utils_chroma_scheduler import build_scheduler, calculate_shift

if TYPE_CHECKING:
    from .trainer import ChromaTrainer

logger = structlog.get_logger(__name__)

# ``ChromaPipeline.__call__`` native defaults (pipeline_chroma.py signature,
# lines 649-651). NOTE the docstring text says "guidance_scale ... defaults
# to 3.5" (line 692) but the ACTUAL Python default is 5.0 (line 651) — the
# code default wins; the docstring is stale/copy-pasted from Flux.
_CHROMA_DEFAULT_RESOLUTION: int = 1024
_CHROMA_DEFAULT_STEPS: int = 35
_CHROMA_DEFAULT_GUIDANCE: float = 5.0


def _combine_cfg(pos: Tensor, neg: Tensor, guidance_scale: float) -> Tensor:
    """Chroma CFG combine: ``neg + guidance_scale * (pos - neg)``.

    Matches ``ChromaPipeline.__call__`` line 933: ``noise_pred =
    neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)``.
    """
    return neg + guidance_scale * (pos - neg)


class ChromaSampler(GenericSamplingPipeline):
    """Chroma flow-matching sampler with true (non-distilled) CFG."""

    pipeline: "ChromaTrainer"

    def __init__(self, pipeline: "ChromaTrainer") -> None:
        super().__init__(pipeline)
        self._scheduler = None

    # ── Native sample defaults ───────────────────────────────────────────

    def _sample_single(self, prompt_cfg: dict[str, Any], step: int) -> Image.Image:
        """Fill Chroma-native defaults before the generic sampling flow.

        Pipeline ``__call__`` defaults: 35 steps, guidance 5.0 (sourced from
        the definition's ``defaults`` when present).
        """
        cfg = dict(prompt_cfg)
        defaults = getattr(self.pipeline.definition, "defaults", {}) or {}
        resolution = int(defaults.get("resolution", _CHROMA_DEFAULT_RESOLUTION))
        fill = {
            "width": resolution,
            "height": resolution,
            "num_inference_steps": int(
                defaults.get("num_inference_steps", _CHROMA_DEFAULT_STEPS),
            ),
            "guidance_scale": float(
                defaults.get("guidance_scale", _CHROMA_DEFAULT_GUIDANCE),
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
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        self._scheduler = build_scheduler(arch)
        return self._scheduler

    def _compute_mu(self, image_seq_len: int) -> float:
        """Resolution-derived mu via the pipeline's own ``calculate_shift``."""
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        return calculate_shift(
            image_seq_len,
            base_seq_len=int(arch.get("scheduler.base_image_seq_len", 256)),
            max_seq_len=int(arch.get("scheduler.max_image_seq_len", 4096)),
            base_shift=float(arch.get("scheduler.base_shift", 0.5)),
            max_shift=float(arch.get("scheduler.max_shift", 1.15)),
        )

    # ── Text encoding ────────────────────────────────────────────────────

    def encode_prompt(self, prompt: str) -> dict[str, Any]:
        """Encode prompt via the trainer's cache-aware ``encode_text()``.

        Returns dict with ``embeds`` [1, L, 4096] and ``mask`` [1, L]
        (Chroma's modified padding mask — see ``driver.encode_text``).
        """
        trainer = self.pipeline
        dtype = next(trainer.transformer.parameters()).dtype
        embeds, mask = trainer.encode_text([prompt], dtype=dtype)
        return {"embeds": embeds, "mask": mask}

    # ── Initial noise ────────────────────────────────────────────────────

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator,
    ) -> Tensor:
        """Create noise [1, 16, lat_h, lat_w] in unpacked latent space.

        Mirrors ``ChromaPipeline.prepare_latents``: ``lat = 2 * (px //
        (vae_scale_factor * 2))``.
        """
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        vae_sf = int(arch.get("vae.vae_scale_factor", 8))
        lat_channels = int(arch.get("vae.latent_channels", 16))

        lat_h = 2 * (height // (vae_sf * 2))
        lat_w = 2 * (width // (vae_sf * 2))

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
        """Flow-matching Euler denoising matching ChromaPipeline.__call__.

        Precision invariants (binding):
        - Trajectory runs in fp32 (no autocast around the forward).
        - driver.forward_pass receives raw [0,1000] timesteps.
        - CFG only when guidance_scale > 1; combine neg + g*(pos - neg).

        Returns:
            fp32 latents ``[1, 16, lat_h, lat_w]``.
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
        # ChromaPipeline ALWAYS passes mu (line 831-846); the scheduler
        # silently ignores it when use_dynamic_shifting is False.
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        scheduler.set_timesteps(
            num_inference_steps=num_steps,
            sigmas=sigmas,
            mu=mu,
            device=device,
        )
        timesteps = scheduler.timesteps

        # CFG per ChromaPipeline: gate at guidance_scale > 1.
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

        with torch.no_grad():
            total_steps = len(timesteps)
            for step_i, t in enumerate(timesteps, 1):
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Sampling {step_i}/{total_steps}")

                xin = latents.to(dtype=dtype)
                ts = t.expand(xin.shape[0]).to(dtype=dtype)

                v_cond = driver.forward_pass(
                    noisy_input=xin,
                    timesteps=ts,
                    text_embeddings=(
                        prompt_embeds.to(dtype=dtype),
                        prompt_mask.to(dtype=dtype),
                    ),
                    batch={},
                ).to(torch.float32)

                if cfg_on:
                    v_uncond = driver.forward_pass(
                        noisy_input=xin,
                        timesteps=ts,
                        text_embeddings=(
                            uncond_embeds.to(dtype=dtype),
                            uncond_mask.to(dtype=dtype),
                        ),
                        batch={},
                    ).to(torch.float32)

                    noise_pred = _combine_cfg(v_cond, v_uncond, guidance_scale)
                else:
                    noise_pred = v_cond

                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[
                    0
                ].to(torch.float32)

        return latents

    # ── VAE decode ───────────────────────────────────────────────────────

    def decode_latents(self, latents: Any) -> Image.Image:
        """Decode latent tensor to a PIL image.

        Matches the pipeline (lines 965-967):
        ``latents = (latents / scaling_factor) + shift_factor`` →
        ``vae.decode``.
        """
        vae = self.pipeline.vae
        vae_dtype = next(vae.parameters()).dtype

        # ChromaPipeline._unpack_latents operates on the PACKED sequence;
        # our `latents` here are already unpacked spatial [1, 16, h, w]
        # (driver.forward_pass packs/unpacks internally). No re-pack needed.

        vae.to(self.device)
        with torch.no_grad():
            scaling_factor = getattr(vae.config, "scaling_factor", 0.3611)
            shift_factor = getattr(vae.config, "shift_factor", 0.1159) or 0.0
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
