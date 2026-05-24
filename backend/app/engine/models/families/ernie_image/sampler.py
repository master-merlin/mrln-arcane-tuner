"""ERNIE-Image sampler -- mirrors ``ErnieImagePipeline.__call__``.

Differences from the standard diffusers pipeline (training preview):
*  Uses already-loaded components (no pipeline re-instantiation).
*  Skips the optional ``pe`` Prompt Enhancer; the user-supplied prompt
   is fed verbatim to the text encoder.
*  Uses cached training-time text embeddings when available (one
   encode per unique prompt across the whole training session).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
import torch
from PIL import Image
from torch import Tensor

from app.engine.core.sampling import GenericSamplingPipeline

if TYPE_CHECKING:
    from .trainer import ErnieImageTrainer

logger = structlog.get_logger(__name__)


# VAE downscales by a factor of 8 (3 stride-2 levels); the pipeline's
# 2x2 patchify adds another factor of 2 → combined model-input scale 16.
VAE_SPATIAL_DOWNSCALE = 8


class ErnieImageSampler(GenericSamplingPipeline):
    """ERNIE-Image sampler -- flow-matching Euler with optional CFG."""

    pipeline: ErnieImageTrainer

    def __init__(self, pipeline: ErnieImageTrainer) -> None:
        super().__init__(pipeline)
        self._scheduler = None

    # ── Lazy scheduler ───────────────────────────────────────────────────

    def _get_scheduler(self):
        if self._scheduler is not None:
            return self._scheduler
        from diffusers import FlowMatchEulerDiscreteScheduler

        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        # FlowMatchEulerDiscreteScheduler.set_timesteps applies the
        # ``shift * s / (1 + (shift - 1) * s)`` transform UNCONDITIONALLY
        # when ``use_dynamic_shifting`` is False — passing explicit sigmas
        # only bypasses the *generation* of sigmas, not the shift transform.
        # ERNIE-Image's pretrained scheduler ships with ``shift=4.0``;
        # defaulting to 1.0 produces a flat schedule that visibly distorts
        # samples (audited against the official pipeline output).
        self._scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=int(arch.get("scheduler.num_train_timesteps", 1000)),
            shift=float(arch.get("scheduler.shift", 4.0)),
            use_dynamic_shifting=bool(arch.get("scheduler.use_dynamic_shifting", False)),
            base_shift=float(arch.get("scheduler.base_shift", 0.5)),
            max_shift=float(arch.get("scheduler.max_shift", 1.15)),
            base_image_seq_len=int(arch.get("scheduler.base_image_seq_len", 256)),
            max_image_seq_len=int(arch.get("scheduler.max_image_seq_len", 4096)),
            shift_terminal=arch.get("scheduler.shift_terminal", None),
            invert_sigmas=bool(arch.get("scheduler.invert_sigmas", False)),
            use_karras_sigmas=bool(arch.get("scheduler.use_karras_sigmas", False)),
            use_exponential_sigmas=bool(arch.get("scheduler.use_exponential_sigmas", False)),
            use_beta_sigmas=bool(arch.get("scheduler.use_beta_sigmas", False)),
        )
        return self._scheduler

    # ── Text encoding ────────────────────────────────────────────────────

    def _model_dtype(self) -> torch.dtype:
        """Return the actual transformer parameter dtype.

        Match the official ``ErnieImagePipeline`` which uses
        ``self.transformer.dtype``; do NOT use
        ``pipeline.autocast_dtype`` (a training-config knob that
        defaults to fp16 and silently mismatches the bf16-loaded
        transformer at sample time, causing PyTorch to repromote
        dtypes per-op and accumulate precision drift over the
        denoising loop).
        """
        return next(self.pipeline.transformer.parameters()).dtype

    def encode_prompt(self, prompt: str) -> dict[str, Any]:
        """Encode positive + negative prompt for optional CFG.

        Delegates to the trainer's cache-aware ``encode_text``.  Returns
        ``(text_bth, attention_mask)`` pairs for both the conditional and
        unconditional paths; ``denoise()`` re-pads them to a common
        ``Tmax`` so the CFG batched forward sees matched sequence
        lengths (otherwise the per-token rope positions diverge between
        the cond and uncond passes).
        """
        dtype = self._model_dtype()
        cond_emb, cond_mask = self.pipeline.encode_text([prompt], dtype=dtype)
        uncond_emb, uncond_mask = self.pipeline.encode_text([""], dtype=dtype)
        return {
            "cond_emb": cond_emb,
            "cond_mask": cond_mask,
            "uncond_emb": uncond_emb,
            "uncond_mask": uncond_mask,
        }

    # ── Latent helpers ───────────────────────────────────────────────────

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator,
    ) -> Tensor:
        """Random noise in the model's **patchified** input space.

        Combined scale factor = VAE downscale (8) × patchify (2) = 16,
        so the model sees a ``H/16 × W/16`` grid with
        ``in_channels = 4 × vae_latent_channels`` channels.
        """
        transformer = self.pipeline.transformer
        in_channels = int(getattr(transformer.config, "in_channels", 128))
        latent_h = height // (VAE_SPATIAL_DOWNSCALE * 2)
        latent_w = width // (VAE_SPATIAL_DOWNSCALE * 2)

        return torch.randn(
            (1, in_channels, latent_h, latent_w),
            generator=generator,
            device=self.device,
            dtype=self._model_dtype(),
        )

    # ── Core sampling methods ────────────────────────────────────────────

    @staticmethod
    def _pad_to_common_length(
        a_emb: Tensor, a_lens: Tensor, b_emb: Tensor, b_lens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Right-pad ``a_emb`` and ``b_emb`` so they share ``Tmax``.

        Concatenated CFG forward needs matched sequence lengths so the
        transformer's per-token position encoding lines up across the
        two halves of the batch.  Returns ``(text_bth, text_lens)`` ready
        for a single transformer call on batch size 2.
        """
        t_max = int(max(a_emb.shape[1], b_emb.shape[1]))
        device = a_emb.device
        dtype = a_emb.dtype
        feat_dim = a_emb.shape[-1]

        def _right_pad(emb: Tensor) -> Tensor:
            if emb.shape[1] == t_max:
                return emb
            pad = torch.zeros(
                (emb.shape[0], t_max - emb.shape[1], feat_dim),
                device=device, dtype=dtype,
            )
            return torch.cat([emb, pad], dim=1)

        text_bth = torch.cat([_right_pad(a_emb), _right_pad(b_emb)], dim=0)
        text_lens = torch.cat([a_lens, b_lens], dim=0).to(
            dtype=torch.long, device=device,
        )
        return text_bth, text_lens

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Tensor:
        """Flow-matching Euler denoising loop with optional CFG.

        Mirrors ``ErnieImagePipeline.__call__`` step-for-step:

        * Linear sigma schedule ``linspace(1, 0, N+1)[:-1]`` →
          timesteps in ``[0, num_train_timesteps]``.
        * Single batched forward when CFG is on: concatenate
          ``[uncond, cond]`` latents and text along the batch dim, then
          ``chunk(2)`` the velocity prediction.
        * Timestep is passed verbatim (no ``/ 1000``) because the
          transformer's ``Timesteps`` embedding consumes the raw value;
          the pretrained checkpoint expects the full ``[0, 1000]``
          range.

        Returns the un-denormalized patched latents
        ``[1, in_channels, H/16, W/16]``; BN-denormalize + unpatchify +
        VAE decode happen in :meth:`decode_latents`.
        """
        device = self.device
        transformer = self.pipeline.transformer
        scheduler = self._get_scheduler()
        dtype = self._model_dtype()

        cond_emb = prompt_embedding["cond_emb"]
        cond_mask = prompt_embedding["cond_mask"]
        uncond_emb = prompt_embedding["uncond_emb"]
        uncond_mask = prompt_embedding["uncond_mask"]
        do_cfg = guidance_scale > 1.0

        cond_lens = cond_mask.sum(dim=1).to(dtype=torch.long, device=device)
        uncond_lens = uncond_mask.sum(dim=1).to(dtype=torch.long, device=device)

        latents = noise.to(device=device, dtype=dtype)
        batch_size = latents.shape[0]

        # Build the (possibly concatenated) text tensors once — outside the
        # denoising loop, since the prompts don't change per step.
        if do_cfg:
            text_bth, text_lens = self._pad_to_common_length(
                uncond_emb.to(device=device, dtype=dtype), uncond_lens,
                cond_emb.to(device=device, dtype=dtype), cond_lens,
            )
        else:
            text_bth = cond_emb.to(device=device, dtype=dtype)
            text_lens = cond_lens

        # Linear sigma schedule -- matches ErnieImagePipeline line ~316.
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1)
        scheduler.set_timesteps(sigmas=sigmas[:-1], device=device)
        timesteps = scheduler.timesteps

        self._ensure_transformer_on_device(transformer)
        with torch.no_grad():
            total_steps = len(timesteps)
            for step_i, t in enumerate(timesteps, 1):
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Sampling {step_i}/{total_steps}")

                if do_cfg:
                    latent_model_input = torch.cat([latents, latents], dim=0)
                    t_batch = torch.full(
                        (batch_size * 2,), t.item(),
                        device=device, dtype=dtype,
                    )
                else:
                    latent_model_input = latents
                    t_batch = torch.full(
                        (batch_size,), t.item(),
                        device=device, dtype=dtype,
                    )

                pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=t_batch,
                    text_bth=text_bth,
                    text_lens=text_lens,
                    return_dict=False,
                )[0]

                if do_cfg:
                    pred_uncond, pred_cond = pred.chunk(2, dim=0)
                    pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                latents = scheduler.step(pred, t, latents, return_dict=False)[0]

        return latents

    def decode_latents(self, latents: Any) -> Image.Image:
        """BN-denormalize → unpatchify → VAE decode → PIL.

        Mirrors the post-denoise tail of ``ErnieImagePipeline.__call__``.
        """
        from app.engine.models.families.ernie_image.utils import (
            bn_denormalize,
            unpatchify_latents,
        )

        vae = self.pipeline.vae

        latents = bn_denormalize(latents, vae)
        latents = unpatchify_latents(latents)

        with torch.no_grad():
            images = vae.decode(latents.to(vae.dtype), return_dict=False)[0]

        images = (images.clamp(-1, 1) + 1.0) / 2.0
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        arr = (images[0] * 255).round().astype("uint8")
        return Image.fromarray(arr)
