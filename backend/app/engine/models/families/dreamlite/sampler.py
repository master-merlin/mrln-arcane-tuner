"""DreamLiteSampler — in-training preview sampler for DreamLite.

Replicates ``DreamLitePipeline.__call__`` (base, CFG) and
``DreamLiteMobilePipeline.__call__`` (mobile, CFG-distilled) semantics:

Correctness invariants enforced here:
1. TIMESTEP: ``driver.forward_pass`` receives RAW [0, 1000] timesteps —
   the DreamLite U-Net's sinusoidal ``time_proj`` consumes raw timesteps
   (there is NO /1000 anywhere in this family, unlike the DiT families).
2. CFG per the base pipeline: cond + uncond stacked into ONE batched
   forward per step (``[uncond, cond]``, uncond at index 0), combined as
   ``uncond + guidance_scale * (cond - uncond)``. Mobile (``is_distilled``
   definitions) or ``guidance_scale <= 0`` → single un-batched pass and
   the negative prompt is never encoded (CFG was distilled away).
3. mu/shift replicate the pipeline AT RUNTIME: ``sigmas = linspace(1,
   1/num_steps, num_steps)`` and ``mu = calculate_shift(image_seq_len,
   base_image_seq_len, max_image_seq_len, base_shift, max_shift)`` where
   the shift constants come from the CHECKPOINT scheduler config
   (0.5 / **1.15** / 256 / 4096 — the pipeline reads scheduler.config, so
   the calculate_shift signature default 1.16 never applies) and
   ``image_seq_len = lat_h * lat_w // 4``.
4. NO torch.autocast around the UNet forward (autocast-collapse gotcha):
   ``torch.no_grad()``, native model dtype, fp32 latent trajectory.
5. Cached TE embeddings are cast to the MODEL dtype before the forward
   (fp32 cache vs bf16 model is a real crash — Wave 1 lesson).
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
    from .trainer import DreamLiteTrainer

logger = structlog.get_logger(__name__)

# DreamLite scheduler defaults (checkpoint scheduler_config.json;
# overrideable via architecture_params YAML).
_DREAMLITE_SHIFT: float = 3.0
_DREAMLITE_BASE_SHIFT: float = 0.5
_DREAMLITE_MAX_SHIFT: float = 1.15
_DREAMLITE_BASE_IMAGE_SEQ_LEN: int = 256
_DREAMLITE_MAX_IMAGE_SEQ_LEN: int = 4096


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = _DREAMLITE_BASE_IMAGE_SEQ_LEN,
    max_seq_len: int = _DREAMLITE_MAX_IMAGE_SEQ_LEN,
    base_shift: float = _DREAMLITE_BASE_SHIFT,
    max_shift: float = _DREAMLITE_MAX_SHIFT,
) -> float:
    """Empirical mu schedule — copied from the diffusers DreamLite pipeline."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _combine_cfg(cond: Tensor, uncond: Tensor, guidance_scale: float) -> Tensor:
    """DreamLite CFG combine: ``uncond + guidance_scale * (cond - uncond)``.

    Matches ``DreamLitePipeline.__call__``
    (``noise_pred_uncond + guidance_scale * (noise_pred_cond -
    noise_pred_uncond)``) — standard CFG, NOT krea2's
    ``cond + g*(cond - uncond)`` form.
    """
    return uncond + guidance_scale * (cond - uncond)


class DreamLiteSampler(GenericSamplingPipeline):
    """DreamLite flow-matching sampler — Base (batched CFG) + Mobile (none).

    Structural template: krea2/ovis_image samplers.
    UNet call: reuses ``driver.forward_pass`` (DRY — the width concat,
    time_ids, raw-timestep handling, and prediction slice are already
    implemented and contract-tested there). CFG runs as ONE batched driver
    call per step, exactly like the pipeline's single UNet invocation.
    """

    pipeline: "DreamLiteTrainer"

    def __init__(self, pipeline: "DreamLiteTrainer") -> None:
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
            shift=float(arch.get("scheduler.shift", _DREAMLITE_SHIFT)),
            use_dynamic_shifting=bool(
                arch.get("scheduler.use_dynamic_shifting", True),
            ),
            base_shift=float(
                arch.get("scheduler.base_shift", _DREAMLITE_BASE_SHIFT),
            ),
            max_shift=float(
                arch.get("scheduler.max_shift", _DREAMLITE_MAX_SHIFT),
            ),
            base_image_seq_len=int(
                arch.get(
                    "scheduler.base_image_seq_len",
                    _DREAMLITE_BASE_IMAGE_SEQ_LEN,
                ),
            ),
            max_image_seq_len=int(
                arch.get(
                    "scheduler.max_image_seq_len",
                    _DREAMLITE_MAX_IMAGE_SEQ_LEN,
                ),
            ),
        )
        return self._scheduler

    # ── mu (dynamic shift) ───────────────────────────────────────────────

    def _compute_mu(self, image_seq_len: int) -> float:
        """Resolution-derived mu via the pipeline's calculate_shift, fed the
        checkpoint scheduler config (0.5 / 1.15 / 256 / 4096)."""
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        return _calculate_shift(
            image_seq_len,
            base_seq_len=int(
                arch.get(
                    "scheduler.base_image_seq_len",
                    _DREAMLITE_BASE_IMAGE_SEQ_LEN,
                ),
            ),
            max_seq_len=int(
                arch.get(
                    "scheduler.max_image_seq_len",
                    _DREAMLITE_MAX_IMAGE_SEQ_LEN,
                ),
            ),
            base_shift=float(
                arch.get("scheduler.base_shift", _DREAMLITE_BASE_SHIFT),
            ),
            max_shift=float(
                arch.get("scheduler.max_shift", _DREAMLITE_MAX_SHIFT),
            ),
        )

    # ── CFG gating ───────────────────────────────────────────────────────

    def _is_distilled(self) -> bool:
        """Read is_distilled from definition defaults (set per YAML)."""
        defaults = getattr(self.pipeline.definition, "defaults", {}) or {}
        return bool(defaults.get("is_distilled", False))

    # ── Text encoding ────────────────────────────────────────────────────

    def encode_prompt(self, prompt: str) -> dict[str, Any]:
        """Encode a POSITIVE prompt via the trainer's cache-aware
        ``encode_text()`` (which applies the pipeline's ``"[Generate]: "``
        prefix internally).

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
        """Create noise [1, 4, lat_h, lat_w] — ``prepare_latents`` formula
        (``lat = px // vae_scale_factor``; AutoencoderTiny → 8×)."""
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        vae_sf = int(arch.get("vae.vae_scale_factor", 8))
        lat_channels = int(arch.get("vae.latent_channels", 4))

        lat_h = height // vae_sf
        lat_w = width // vae_sf

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
        """Flow-matching Euler denoising matching DreamLitePipeline.__call__.

        Precision invariants (binding):
        - Trajectory runs in fp32 (no autocast around the forward).
        - driver.forward_pass receives RAW [0,1000] timesteps.
        - TE embeddings are cast to the model dtype before the forward.
        - CFG: distilled definitions and guidance_scale <= 0 → single pass;
          otherwise ONE batched [uncond, cond] forward per step, combined
          uncond + g*(cond - uncond).

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

        # Pipeline: image_seq_len = lat_h * lat_w // 4.
        img_seq_len = (latents.shape[2] * latents.shape[3]) // 4
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

        # CFG gate: mobile is CFG-distilled — guidance is IGNORED there
        # (DreamLiteMobilePipeline warns + ignores); base gates on gs > 0.
        cfg_on = float(guidance_scale) > 0.0 and not self._is_distilled()
        if cfg_on:
            # Negative prompt encodes RAW (no "[Generate]: " prefix) —
            # pipeline: prompts=[negative_prompt, "[Generate]: "+prompt].
            neg_text = str(self.config.get("sample_negative_prompt", "") or "")
            uncond_embeds, uncond_mask = self.pipeline.encode_uncond_text(
                [neg_text], dtype=dtype,
            )
            # Stack [uncond, cond] once — reused every step (fixed-length
            # embeddings, so torch.cat is safe).
            batched_embeds = torch.cat(
                [uncond_embeds.to(dtype=dtype), prompt_embeds.to(dtype=dtype)],
                dim=0,
            )
            batched_mask = torch.cat([uncond_mask, prompt_mask], dim=0)

        # Move the UNet to device (respects block-swap if active)
        self._ensure_transformer_on_device(transformer)

        scheduler.set_begin_index(0)

        # Denoise loop — fp32 trajectory, no autocast (invariant 4);
        # driver.forward_pass receives RAW [0,1000] timesteps (invariant 1).
        with torch.no_grad():
            total_steps = len(timesteps)
            for step_i, t in enumerate(timesteps, 1):
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Sampling {step_i}/{total_steps}")

                # Cast latents to model dtype for the forward; keep fp32
                # trajectory outside.
                xin = latents.to(dtype=dtype)

                if cfg_on:
                    # ONE batched forward per step, like the pipeline's
                    # torch.cat([latents] * 2) UNet call. Raw timesteps.
                    xin2 = torch.cat([xin, xin], dim=0)
                    ts = t.expand(xin2.shape[0]).to(dtype=dtype)
                    v = driver.forward_pass(
                        noisy_input=xin2,
                        timesteps=ts,
                        text_embeddings=(batched_embeds, batched_mask),
                        batch={},
                    ).to(torch.float32)
                    v_uncond, v_cond = v.chunk(2, dim=0)
                    noise_pred = _combine_cfg(v_cond, v_uncond, guidance_scale)
                else:
                    ts = t.expand(xin.shape[0]).to(dtype=dtype)
                    noise_pred = driver.forward_pass(
                        noisy_input=xin,
                        timesteps=ts,
                        text_embeddings=(
                            prompt_embeds.to(dtype=dtype),
                            prompt_mask,
                        ),
                        batch={},
                    ).to(torch.float32)

                # Scheduler step advances the fp32 trajectory
                latents = scheduler.step(
                    noise_pred, t, latents, return_dict=False,
                )[0].to(torch.float32)

        return latents

    # ── VAE decode ───────────────────────────────────────────────────────

    def decode_latents(self, latents: Any) -> Image.Image:
        """Decode latent tensor to a PIL image.

        Matches the pipeline: ``latents / scaling_factor + shift_factor``
        → ``vae.decode`` (AutoencoderTiny outputs the diffusers [-1, 1]
        convention — DecoderTiny rescales internally).
        """
        vae = self.pipeline.vae
        vae_dtype = getattr(vae, "dtype", None) or next(vae.parameters()).dtype

        vae.to(self.device)
        with torch.no_grad():
            scaling_factor = getattr(vae.config, "scaling_factor", 1.0) or 1.0
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
