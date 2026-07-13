"""Lumina2Sampler — in-training preview sampler for Lumina-Image-2.0.

Replicates ``Lumina2Pipeline.__call__`` semantics (diffusers 0.39,
``venv/Lib/site-packages/diffusers/pipelines/lumina2/pipeline_lumina2.py``).

Correctness invariants enforced here:

1. TIMESTEP: ``driver.forward_pass`` receives the RAW ``[0, 1000]`` timestep
   (it reverses + negates internally — see ``driver.py`` module docstring
   §3, the family's #1 silent-LoRA-killer risk). Never pre-reverse here.
2. CFG is TRUE and gated ``guidance_scale > 1`` (pipeline default 4.0,
   ``do_classifier_free_guidance`` property). The negative/uncond prompt is
   encoded WITHOUT the system-prompt prefix — ``trainer.encode_negative_
   text``, NOT ``trainer.encode_text`` (``driver.py`` module docstring §1).
3. ``cfg_normalization=True`` and ``cfg_trunc_ratio=1.0`` are PINNED at the
   pipeline's own defaults (not surfaced as knobs — see ``driver.py``
   module docstring §4): every step runs CFG (trunc_ratio=1.0 never
   truncates) and the combined velocity is renormalized to the conditional
   prediction's ``dim=-1`` norm (``pipeline_lumina2.py`` lines 749-752).
   Because ``driver.forward_pass`` already returns the NEGATED velocity
   (module docstring §3), combining + renormalizing the (already negated)
   ``v_cond``/``v_uncond`` here with the SAME formula the pipeline applies
   pre-negation is mathematically identical (``torch.norm`` is sign-
   invariant, and negation distributes linearly through the combine) — no
   second negation is needed in this sampler.
4. Scheduler: a REAL ``FlowMatchEulerDiscreteScheduler`` built from the
   definition's ``architecture_params``. The live checkpoint's
   ``scheduler/scheduler_config.json`` is STATIC-shift
   (``use_dynamic_shifting=false``, ``shift=6.0`` — independently
   cross-checked against ComfyUI's own ``Lumina2`` supported-model entry,
   ``sampling_settings={"shift": 6.0}``), so ``mu`` is computed (matching
   the pipeline's unconditional computation) but the scheduler ignores it
   (chroma1-hd precedent). NOTE: the real pipeline computes its ``mu`` input
   from ``latents.shape[1]`` — the LATENT CHANNEL COUNT (16), not a spatial
   sequence length (``pipeline_lumina2.py`` line 696, immediately after an
   UNPACKED ``prepare_latents`` — this looks like an upstream copy-paste
   quirk from the packed-latent Flux lineage). Since ``use_dynamic_shifting``
   is False for the shipped checkpoint this value is provably inert; this
   sampler instead computes ``mu`` from the actual patchified sequence
   length (``(H/2)*(W/2)``, matching ovis_image/chroma's convention) so the
   number is meaningful if a future NetaYume/Neta-Lumina definition ever
   ships a dynamic-shifting scheduler config.
5. NO autocast around the DiT forward (autocast-collapse gotcha):
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
    from .trainer import Lumina2Trainer

logger = structlog.get_logger(__name__)

# ``Lumina2Pipeline.__call__`` native defaults (pipeline_lumina2.py
# signature, lines 526-550).
_LUMINA2_DEFAULT_RESOLUTION: int = 1024
_LUMINA2_DEFAULT_STEPS: int = 30
_LUMINA2_DEFAULT_GUIDANCE: float = 4.0

# Scheduler defaults (live checkpoint's scheduler/scheduler_config.json).
_SHIFT: float = 6.0
_BASE_SHIFT: float = 0.5
_MAX_SHIFT: float = 1.15
_BASE_IMAGE_SEQ_LEN: int = 256
_MAX_IMAGE_SEQ_LEN: int = 4096


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = _BASE_IMAGE_SEQ_LEN,
    max_seq_len: int = _MAX_IMAGE_SEQ_LEN,
    base_shift: float = _BASE_SHIFT,
    max_shift: float = _MAX_SHIFT,
) -> float:
    """``calculate_shift`` — copied from ``pipeline_lumina2.py`` lines 64-74
    (itself ``# Copied from diffusers.pipelines.flux.pipeline_flux.
    calculate_shift``)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _combine_cfg(pos: Tensor, neg: Tensor, guidance_scale: float) -> Tensor:
    """Lumina2 CFG combine + normalization (``pipeline_lumina2.py`` lines
    747-752): ``velocity = neg + g*(pos - neg)``, then renormalize the
    combined ``dim=-1`` norm back to the conditional prediction's norm.

    ``pos``/``neg`` here are the driver's ALREADY-NEGATED velocity outputs
    (module docstring point 3) — see module docstring for why applying the
    identical formula to the negated values is exact.
    """
    noise_pred = neg + guidance_scale * (pos - neg)
    cond_norm = torch.norm(pos, dim=-1, keepdim=True)
    noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
    return noise_pred * (cond_norm / noise_norm)


class Lumina2Sampler(GenericSamplingPipeline):
    """Lumina-Image-2.0 flow-matching sampler with true, normalized CFG."""

    pipeline: "Lumina2Trainer"

    def __init__(self, pipeline: "Lumina2Trainer") -> None:
        super().__init__(pipeline)
        self._scheduler = None

    # ── Native sample defaults ───────────────────────────────────────────

    def _sample_single(self, prompt_cfg: dict[str, Any], step: int) -> Image.Image:
        """Fill Lumina2-native defaults before the generic sampling flow.

        Pipeline ``__call__`` defaults: 30 steps, guidance 4.0 (sourced from
        the definition's ``defaults`` when present).
        """
        cfg = dict(prompt_cfg)
        defaults = getattr(self.pipeline.definition, "defaults", {}) or {}
        resolution = int(defaults.get("resolution", _LUMINA2_DEFAULT_RESOLUTION))
        fill = {
            "width": resolution,
            "height": resolution,
            "num_inference_steps": int(
                defaults.get("num_inference_steps", _LUMINA2_DEFAULT_STEPS),
            ),
            "guidance_scale": float(
                defaults.get("guidance_scale", _LUMINA2_DEFAULT_GUIDANCE),
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
        from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: PLC0415

        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        self._scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=int(
                arch.get("scheduler.num_train_timesteps", 1000),
            ),
            shift=float(arch.get("scheduler.shift", _SHIFT)),
            use_dynamic_shifting=bool(
                arch.get("scheduler.use_dynamic_shifting", False),
            ),
            base_shift=float(arch.get("scheduler.base_shift", _BASE_SHIFT)),
            max_shift=float(arch.get("scheduler.max_shift", _MAX_SHIFT)),
            base_image_seq_len=int(
                arch.get("scheduler.base_image_seq_len", _BASE_IMAGE_SEQ_LEN),
            ),
            max_image_seq_len=int(
                arch.get("scheduler.max_image_seq_len", _MAX_IMAGE_SEQ_LEN),
            ),
        )
        return self._scheduler

    def _compute_mu(self, image_seq_len: int) -> float:
        """Resolution-derived mu — see module docstring point 4 for why this
        uses the patchified sequence length rather than mirroring the
        pipeline's (inert, under the shipped static-shift config) ``latents.
        shape[1]`` channel-count quirk."""
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        return _calculate_shift(
            image_seq_len,
            base_seq_len=int(
                arch.get("scheduler.base_image_seq_len", _BASE_IMAGE_SEQ_LEN),
            ),
            max_seq_len=int(
                arch.get("scheduler.max_image_seq_len", _MAX_IMAGE_SEQ_LEN),
            ),
            base_shift=float(arch.get("scheduler.base_shift", _BASE_SHIFT)),
            max_shift=float(arch.get("scheduler.max_shift", _MAX_SHIFT)),
        )

    # ── Text encoding ────────────────────────────────────────────────────

    def encode_prompt(self, prompt: str) -> dict[str, Any]:
        """Encode the POSITIVE prompt via the trainer's cache-aware
        ``encode_text()`` (system prompt applied). Returns dict with
        ``embeds`` [1, L, 2304] and ``mask`` [1, L].
        """
        trainer = self.pipeline
        dtype = next(trainer.transformer.parameters()).dtype
        embeds, mask = trainer.encode_text([prompt], dtype=dtype)
        return {"embeds": embeds, "mask": mask}

    def _encode_negative_prompt(self, prompt: str) -> dict[str, Any]:
        """Encode the CFG negative/uncond prompt WITHOUT the system prompt
        (``trainer.encode_negative_text`` — see ``driver.py`` module
        docstring §1)."""
        trainer = self.pipeline
        dtype = next(trainer.transformer.parameters()).dtype
        embeds, mask = trainer.encode_negative_text([prompt], dtype=dtype)
        return {"embeds": embeds, "mask": mask}

    # ── Initial noise ────────────────────────────────────────────────────

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator,
    ) -> Tensor:
        """Create noise [1, 16, lat_h, lat_w] in unpacked latent space.

        Mirrors ``Lumina2Pipeline.prepare_latents``: ``lat = 2 * (px //
        (vae_scale_factor * 2))`` with ``vae_scale_factor = 8`` (hardcoded
        in ``Lumina2Pipeline.__init__``, NOT derived from the VAE config).
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
        """Flow-matching Euler denoising matching Lumina2Pipeline.__call__.

        Precision invariants (binding):
        - Trajectory runs in fp32 (no autocast around the forward).
        - driver.forward_pass receives raw [0,1000] timesteps (reverses +
          negates internally).
        - CFG only when guidance_scale > 1; combine + renormalize.

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

        # Patchified image_seq_len for mu — see module docstring point 4.
        img_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        mu = self._compute_mu(img_seq_len)

        # Timestep schedule: descending sigmas [1.0 -> 1/num_steps] + mu.
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        scheduler.set_timesteps(
            num_inference_steps=num_steps,
            sigmas=sigmas,
            mu=mu,
            device=device,
        )
        timesteps = scheduler.timesteps

        # CFG per Lumina2Pipeline: gate at guidance_scale > 1.
        cfg_on = float(guidance_scale) > 1.0
        uncond_embeds = None
        uncond_mask = None
        if cfg_on:
            neg_text = str(self.config.get("sample_negative_prompt", "") or "")
            neg_embedding = self._encode_negative_prompt(neg_text)
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

                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[
                    0
                ].to(torch.float32)

        return latents

    # ── VAE decode ───────────────────────────────────────────────────────

    def decode_latents(self, latents: Any) -> Image.Image:
        """Decode latent tensor to a PIL image.

        Matches the pipeline: ``latents / scaling_factor + shift_factor`` ->
        ``vae.decode``.
        """
        vae = self.pipeline.vae
        vae_dtype = next(vae.parameters()).dtype

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
