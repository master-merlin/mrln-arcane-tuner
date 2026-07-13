"""OmniGen2Sampler — in-training preview sampler for OmniGen2.

Replicates ``OmniGen2Pipeline.__call__``/``processing()`` semantics
(``omnigen2/pipelines/omnigen2/pipeline_omnigen2.py`` at the pinned
``vendor/REVISION``; the pipeline itself is NOT vendored — only its
denoise math is mirrored here, driven through ``driver.forward_pass``).

Correctness invariants:

1. TIMESTEP: ``driver.forward_pass`` receives the scheduler's RAW ``[0, 1)``
   ``t`` (pipeline ``predict`` L758) — the transformer's own
   ``timestep_scale: 1000`` config multiplies internally. Never scale here.
2. Scheduler: the LOADER-provided VENDORED ``FlowMatchEulerDiscreteScheduler``
   (``driver.scheduler``) — never a fresh/stock instance (stock diffusers'
   same-named class walks time the OPPOSITE direction; see the vendored
   module's header). ``set_timesteps(num_steps, device,
   num_tokens=latents.shape[-2] * latents.shape[-1])`` mirrors
   ``retrieve_timesteps`` (pipeline L630-636 — num_tokens is the LATENT
   pixel count, not the patch count).
3. Guidance mapping (pipeline L672-723; ``cfg_range`` pinned at its
   ``(0, 1.0)`` default — guidance applies at every step):
   - Our single ``guidance_scale`` knob == ``text_guidance_scale``
     (pipeline default 4.0).
   - ``image_guidance_scale``: :meth:`_resolve_image_guidance` — 1.0 on
     this (T2I) sampler, mirroring the pipeline's forced
     ``image_guidance_scale = 1`` when no input images (L562-563). The
     Edit subclass resolves a real value.
   - ``text_g > 1 and img_g > 1`` -> 3-pass:
     ``pred = uncond + img_g*(ref - uncond) + text_g*(cond - ref)``
     where cond=(positive text, ref), ref=(NEGATIVE text, ref),
     uncond=(NEGATIVE text, no ref) — L672-706.
   - ``text_g > 1`` only -> 2-pass:
     ``pred = uncond + text_g*(cond - uncond)`` with uncond=(NEGATIVE
     text, NO ref) — L707-723. Note the uncond drops the ref image too.
   - otherwise -> single pass (no branch fires upstream; a lone
     ``img_g > 1`` is ignored exactly as upstream ignores it).
4. NEGATIVE prompt: encoded through the trainer's SAME ``encode_text``
   path — the pipeline chat-templates ``negative_prompt`` identically to
   positives (L413-418, default ``""``), so ``sample_negative_prompt``
   (default ``""``) is honored, NOT ignored (contrast boogu).
5. NO autocast around the DiT forward (autocast-collapse gotcha):
   ``torch.no_grad()``, native model dtype, fp32 latent trajectory.
6. Decode: ``latents / scaling_factor`` then ``+ shift_factor`` ->
   ``vae.decode`` (pipeline L739-744).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
import torch
from PIL import Image
from torch import Tensor

from app.engine.core.sampling import GenericSamplingPipeline

if TYPE_CHECKING:
    from .trainer import OmniGen2Trainer

logger = structlog.get_logger(__name__)

# ``OmniGen2Pipeline.__call__`` native defaults (signature L484-486) — the
# fallback when the definition's ``defaults`` dict is missing a key (the
# shipped YAML carries edit-tuned values instead, from example_edit.sh).
_DEFAULT_RESOLUTION = 1024
_DEFAULT_STEPS = 28
_DEFAULT_TEXT_GUIDANCE = 4.0


class OmniGen2Sampler(GenericSamplingPipeline):
    """OmniGen2 flow-matching preview sampler (vendored-scheduler loop)."""

    pipeline: "OmniGen2Trainer"

    # ── Hooks the Edit subclass overrides ────────────────────────────────

    def _forward_batch(self) -> dict[str, Any]:
        """Extra ``batch`` dict for the CONDITIONAL (and ref-CFG) forward.

        ``{}`` here (pure T2I). ``OmniGen2EditSampler`` overrides this to
        feed the clean control latents. The UNCONDITIONAL forward always
        uses ``{}`` (the pipeline's uncond branch passes
        ``ref_image_hidden_states=None`` — L702/L721).
        """
        return {}

    def _resolve_image_guidance(self) -> float:
        """``image_guidance_scale`` — 1.0 on the T2I sampler (pipeline
        L562-563 forces 1 with no input images)."""
        return 1.0

    # ── Native sample defaults ───────────────────────────────────────────

    def _sample_single(self, prompt_cfg: dict[str, Any], step: int) -> Any:
        cfg = dict(prompt_cfg)
        defaults = getattr(self.pipeline.definition, "defaults", {}) or {}
        resolution = int(defaults.get("resolution", _DEFAULT_RESOLUTION))
        fill = {
            "width": resolution,
            "height": resolution,
            "num_inference_steps": int(
                defaults.get("num_inference_steps", _DEFAULT_STEPS),
            ),
            "guidance_scale": float(
                defaults.get("guidance_scale", _DEFAULT_TEXT_GUIDANCE),
            ),
        }
        for key, value in fill.items():
            if cfg.get(key) in (None, 0):
                cfg[key] = value
        return super()._sample_single(cfg, step)

    # ── Text encoding ────────────────────────────────────────────────────

    def encode_prompt(self, prompt: str) -> dict[str, Any]:
        """Encode via the trainer's cache-aware ``encode_text()`` (chat
        template applied — positives and the negative go through the SAME
        path, module docstring §4). Returns ``embeds`` [1, L, 2048] +
        ``mask`` [1, L]."""
        trainer = self.pipeline
        dtype = next(trainer.transformer.parameters()).dtype
        embeds, mask = trainer.encode_text([prompt], dtype=dtype)
        return {"embeds": embeds, "mask": mask}

    # ── Initial noise ────────────────────────────────────────────────────

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator,
    ) -> Tensor:
        """``prepare_latents`` (pipeline L188-215): plain ``px //
        vae_scale_factor`` latent grid — the transformer's own
        ``patch_size`` packing happens internally."""
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
        """Forward-Euler denoise mirroring ``OmniGen2Pipeline.processing()``
        (module docstring §2-3). ``guidance_scale`` == text_guidance_scale.
        """
        device = self.device
        driver = self.pipeline.driver
        transformer = self.pipeline.transformer

        scheduler = driver.scheduler
        if scheduler is None:
            raise RuntimeError(
                "omnigen2 sampler: driver.scheduler is None — the "
                "LOADER-provided vendored scheduler must be assigned "
                "(assign_components()) before sampling; a fresh/stock "
                "scheduler has the OPPOSITE time direction."
            )

        dtype = next(transformer.parameters()).dtype

        prompt_embeds = prompt_embedding["embeds"]
        prompt_mask = prompt_embedding["mask"]

        # fp32 trajectory (no autocast) — precision contract.
        latents = noise.to(device=device, dtype=torch.float32)

        # retrieve_timesteps -> set_timesteps(..., num_tokens=lat_H*lat_W)
        # (pipeline L630-636).
        num_tokens = latents.shape[-2] * latents.shape[-1]
        scheduler.set_timesteps(
            num_inference_steps=num_steps, device=device, num_tokens=num_tokens,
        )
        timesteps = scheduler.timesteps

        text_g = float(guidance_scale)
        cond_batch = self._forward_batch()
        has_ref = bool(cond_batch.get("control_latents"))
        img_g = float(self._resolve_image_guidance()) if has_ref else 1.0

        # Negative embeds needed whenever ANY CFG branch fires.
        uncond_embeds = None
        uncond_mask = None
        if text_g > 1.0:
            neg_text = str(self.config.get("sample_negative_prompt", "") or "")
            neg_embedding = self.encode_prompt(neg_text)
            uncond_embeds = neg_embedding["embeds"]
            uncond_mask = neg_embedding["mask"]

        self._ensure_transformer_on_device(transformer)

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
                    text_embeddings=(prompt_embeds.to(dtype=dtype), prompt_mask),
                    batch=cond_batch,
                ).to(torch.float32)

                if text_g > 1.0 and img_g > 1.0:
                    # 3-pass (pipeline L672-706): ref = (neg text, WITH ref),
                    # uncond = (neg text, NO ref).
                    v_ref = driver.forward_pass(
                        noisy_input=xin,
                        timesteps=ts,
                        text_embeddings=(uncond_embeds.to(dtype=dtype), uncond_mask),
                        batch=cond_batch,
                    ).to(torch.float32)
                    v_uncond = driver.forward_pass(
                        noisy_input=xin,
                        timesteps=ts,
                        text_embeddings=(uncond_embeds.to(dtype=dtype), uncond_mask),
                        batch={},
                    ).to(torch.float32)
                    noise_pred = (
                        v_uncond
                        + img_g * (v_ref - v_uncond)
                        + text_g * (v_cond - v_ref)
                    )
                elif text_g > 1.0:
                    # 2-pass (pipeline L707-723): uncond drops text AND ref.
                    v_uncond = driver.forward_pass(
                        noisy_input=xin,
                        timesteps=ts,
                        text_embeddings=(uncond_embeds.to(dtype=dtype), uncond_mask),
                        batch={},
                    ).to(torch.float32)
                    noise_pred = v_uncond + text_g * (v_cond - v_uncond)
                else:
                    noise_pred = v_cond

                latents = scheduler.step(
                    noise_pred, t, latents, return_dict=False,
                )[0].to(torch.float32)

        return latents

    # ── VAE decode ───────────────────────────────────────────────────────

    def decode_latents(self, latents: Tensor) -> Image.Image:
        """``latents / scaling_factor + shift_factor`` -> ``vae.decode``
        (pipeline L739-744, in that order).

        Device placement is owned by the base ``_sample_single``'s
        ``_ensure_on_gpu(["vae"])`` bracket — no ``vae.to(...)`` here.
        """
        vae = self.pipeline.vae
        vae_dtype = next(vae.parameters()).dtype

        with torch.no_grad():
            scaling_factor = getattr(vae.config, "scaling_factor", 1.0) or 1.0
            shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
            scaled = latents.to(dtype=vae_dtype) / scaling_factor + shift_factor
            decoded = vae.decode(scaled, return_dict=False)

        image_tensor = decoded[0] if isinstance(decoded, (tuple, list)) else decoded

        image_tensor = image_tensor.clamp(-1, 1)
        image_tensor = (image_tensor + 1.0) / 2.0
        image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)
        image_np = image_tensor.cpu().float().numpy()
        image_np = (image_np * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(image_np, mode="RGB")
