"""NucleusImageSampler — in-training preview sampler for Nucleus-Image.

Replicates ``NucleusMoEImagePipeline.__call__`` semantics (diffusers 0.39,
``venv/Lib/site-packages/diffusers/pipelines/nucleusmoe_image/
pipeline_nucleusmoe_image.py``, read in full).

Correctness invariants enforced here:

1. TIMESTEP: ``driver.forward_pass`` receives the RAW ``[0, 1000]`` timestep
   (it normalizes to ``[0,1]`` WITHOUT reversing, then negates the raw model
   output internally — see ``driver.py`` module docstring §4, the family's
   #1 silent-LoRA-killer risk, and the INVERSE mistake from ``lumina2``).
   Never pre-reverse here.
2. CFG is TRUE and gated ``guidance_scale > 1`` (pipeline default 4.0,
   ``do_cfg = guidance_scale > 1``, ``pipeline_nucleusmoe_image.py`` line
   497). The negative/uncond prompt uses the SAME ``trainer.encode_text`` as
   the positive prompt — Nucleus applies the identical system-prompt chat
   template to both (unlike ``lumina2``'s asymmetry — see ``driver.py``
   module docstring §1). When no negative prompt is configured, the pipeline
   falls back to an empty string (``negative_prompt = [""] * batch_size``,
   line 500) — this sampler does the same.
3. CFG combine + renormalize (``pipeline_nucleusmoe_image.py`` lines
   594-599)::

       comb_pred = neg + guidance_scale * (pos - neg)
       cond_norm = norm(pos, dim=-1)
       noise_norm = norm(comb_pred, dim=-1)
       noise_pred = comb_pred * (cond_norm / noise_norm)
       noise_pred = -noise_pred   # final negation, AFTER the combine

   CRITICAL GROUPING FACT: the pipeline performs this combine entirely in
   PACKED space — ``prepare_latents`` packs to ``[B, seq, C*p*p]`` (line
   356 calls ``_pack_latents``) and the ONLY unpack happens after the
   whole loop (line 627), so the loop's ``noise_pred`` is ``[B, seq, 64]``
   and its ``torch.norm(..., dim=-1)`` is a PER-SPATIAL-TOKEN norm over
   the 64 packed channel values (16 latent channels x 2x2 patch). This
   sampler's ``driver.forward_pass`` returns UNPACKED ``[B, 16, H, W]``
   tensors, where a naive ``dim=-1`` norm would be over WIDTH — the wrong
   grouping (that is ``lumina2``'s convention, whose pipeline does CFG
   unpacked; copying it here was a real reviewed-and-fixed bug).
   ``_combine_cfg`` therefore norms over the per-token
   ``(C, patch_h, patch_w)`` element group of the unpacked tensor —
   exactly the same 64-element set as the packed token vector
   (``_pack_latents`` is a pure per-token permutation, and the Frobenius
   norm is permutation-invariant), pinned by
   ``test_nucleus_image_sampler.py``.

   Because ``driver.forward_pass`` ALREADY returns the per-call NEGATED
   velocity (module docstring §4 in ``driver.py``), combining +
   renormalizing the (already negated) ``v_cond``/``v_uncond`` here with the
   SAME formula the pipeline applies pre-negation is mathematically
   identical to negating the pipeline's real (combine-then-negate) result —
   ``torch.norm`` is sign-invariant and negation distributes linearly
   through the affine combine (identical derivation to ``lumina2``'s
   sampler, see that module's docstring point 3). No second negation is
   needed in this sampler.
4. Scheduler: a REAL ``FlowMatchEulerDiscreteScheduler`` built from the
   definition's ``architecture_params``. The live checkpoint's
   ``scheduler/scheduler_config.json`` is STATIC-shift
   (``use_dynamic_shifting=false``, ``shift=1.0``, ``time_shift_type=
   "exponential"``) — fetched directly from the HF repo, 2026-07-13. ``mu``
   is still computed (matching the pipeline's own unconditional
   computation, ``calculate_shift`` — copied verbatim from
   ``pipeline_flux.calculate_shift`` per the pipeline's own comment) and
   passed to ``set_timesteps``, but the scheduler ignores it since
   ``use_dynamic_shifting`` is False for the shipped checkpoint
   (chroma1-hd/lumina2 precedent).
5. Packed (2x2-patchified) latents: this sampler creates and iterates
   UNPACKED ``[1, 16, H, W]`` noise, delegating the pack/unpack to
   ``driver.forward_pass`` each step (module docstring §5 in ``driver.py``)
   — the same code path used at training time (DRY, no duplicated packing
   math). VAE decode re-adds the frame dim (``AutoencoderKLQwenImage`` is a
   3-D causal VAE, ``[B, C, 1, H, W]`` — the SAME class/normalization
   formula ``qwen_image``'s sampler already uses, since it is literally the
   same VAE).
6. NO autocast around the DiT forward (autocast-collapse gotcha):
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
    from .trainer import NucleusImageTrainer

logger = structlog.get_logger(__name__)

# ``NucleusMoEImagePipeline.__call__`` native defaults
# (pipeline_nucleusmoe_image.py signature lines 381-403; default_sample_size
# = 128, vae_scale_factor = 8 -> 128*8 = 1024).
_NUCLEUS_DEFAULT_RESOLUTION: int = 1024
_NUCLEUS_DEFAULT_STEPS: int = 50
_NUCLEUS_DEFAULT_GUIDANCE: float = 4.0

# Scheduler defaults (live checkpoint's scheduler/scheduler_config.json,
# fetched 2026-07-13 — NOTE shift=1.0, NOT lumina2's 6.0).
_SHIFT: float = 1.0
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
    """``calculate_shift`` — copied from ``pipeline_nucleusmoe_image.py``
    lines 58-69 (itself ``# Copied from diffusers.pipelines.flux.
    pipeline_flux.calculate_shift``)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _combine_cfg(
    pos: Tensor,
    neg: Tensor,
    guidance_scale: float,
    patch_size: int = 2,
) -> Tensor:
    """Nucleus CFG combine + normalization (``pipeline_nucleusmoe_image.py``
    lines 594-597): ``velocity = neg + g*(pos - neg)``, then renormalize the
    combined prediction back to the conditional prediction's PER-PACKED-TOKEN
    norm.

    GROUPING (module docstring point 3): the real pipeline runs this in
    PACKED space (``[B, seq, C*p*p]``), so its ``torch.norm(..., dim=-1)``
    is per spatial token over the ``C*p*p`` (= 64 for the real checkpoint)
    packed channel values. ``pos``/``neg`` here are UNPACKED
    ``[B, C, H, W]`` driver outputs, so this function norms over each
    token's ``(C, patch_h, patch_w)`` element group — the identical
    64-element set (``_pack_latents`` is a pure per-token permutation;
    the Frobenius norm is permutation-invariant). A naive ``dim=-1`` norm
    on the unpacked tensor would norm over WIDTH — wrong grouping (that is
    lumina2's convention, whose pipeline does CFG unpacked).

    ``pos``/``neg`` here are the driver's ALREADY-NEGATED velocity outputs
    (module docstring point 3) — see module docstring for why applying the
    identical formula to the negated values is exact.
    """
    noise_pred = neg + guidance_scale * (pos - neg)

    B, C, H, W = pos.shape
    p = patch_size
    # [B, C, H, W] -> [B, C, H/p, p, W/p, p]; per-token groups are dims
    # (1, 3, 5) = (C, patch_h, patch_w) at each (H/p, W/p) location — the
    # exact element set of one packed token vector.
    pos_tokens = pos.reshape(B, C, H // p, p, W // p, p)
    comb_tokens = noise_pred.reshape(B, C, H // p, p, W // p, p)
    # vector_norm (L2) over the token group — identical to the packed-space
    # torch.norm(x, dim=-1) 2-norm; torch.norm itself rejects >2-dim tuples.
    cond_norm = torch.linalg.vector_norm(pos_tokens, dim=(1, 3, 5), keepdim=True)
    noise_norm = torch.linalg.vector_norm(comb_tokens, dim=(1, 3, 5), keepdim=True)
    scaled = comb_tokens * (cond_norm / noise_norm)
    return scaled.reshape(B, C, H, W)


class NucleusImageSampler(GenericSamplingPipeline):
    """Nucleus-Image flow-matching sampler with true, normalized CFG."""

    pipeline: "NucleusImageTrainer"

    def __init__(self, pipeline: "NucleusImageTrainer") -> None:
        super().__init__(pipeline)
        self._scheduler = None

    # ── Native sample defaults ───────────────────────────────────────────

    def _sample_single(self, prompt_cfg: dict[str, Any], step: int) -> Image.Image:
        """Fill Nucleus-native defaults before the generic sampling flow.

        Pipeline ``__call__`` defaults: 50 steps, guidance 4.0, resolution
        1024 (sourced from the definition's ``defaults`` when present).
        """
        cfg = dict(prompt_cfg)
        defaults = getattr(self.pipeline.definition, "defaults", {}) or {}
        resolution = int(defaults.get("resolution", _NUCLEUS_DEFAULT_RESOLUTION))
        fill = {
            "width": resolution,
            "height": resolution,
            "num_inference_steps": int(
                defaults.get("num_inference_steps", _NUCLEUS_DEFAULT_STEPS),
            ),
            "guidance_scale": float(
                defaults.get("guidance_scale", _NUCLEUS_DEFAULT_GUIDANCE),
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
        """Resolution-derived mu — inert under the shipped static-shift
        config (module docstring point 4)."""
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
        ``encode_text()``. Returns dict with ``embeds`` [1, L, D] and
        ``mask`` [1, L]."""
        trainer = self.pipeline
        dtype = next(trainer.transformer.parameters()).dtype
        embeds, mask = trainer.encode_text([prompt], dtype=dtype)
        return {"embeds": embeds, "mask": mask}

    def _encode_negative_prompt(self, prompt: str) -> dict[str, Any]:
        """Encode the CFG negative/uncond prompt — SAME ``encode_text`` as
        the positive path (module docstring point 2; no asymmetry for this
        family, unlike ``lumina2``)."""
        trainer = self.pipeline
        dtype = next(trainer.transformer.parameters()).dtype
        text = prompt if prompt else ""
        embeds, mask = trainer.encode_text([text], dtype=dtype)
        return {"embeds": embeds, "mask": mask}

    # ── Initial noise ────────────────────────────────────────────────────

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator,
    ) -> Tensor:
        """Create noise [1, 16, lat_h, lat_w] in unpacked latent space.

        Mirrors ``NucleusMoEImagePipeline.prepare_latents``: latent dims
        rounded to a multiple of ``patch_size`` after dividing by the VAE
        scale factor (8, derived from ``2 ** len(vae.temperal_downsample)``,
        3 downsample stages).
        """
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        vae_sf = int(arch.get("vae.vae_scale_factor", 8))
        lat_channels = int(arch.get("vae.latent_channels", 16))
        patch_size = int(arch.get("transformer.patch_size", 2))

        lat_h = patch_size * (height // (vae_sf * patch_size))
        lat_w = patch_size * (width // (vae_sf * patch_size))

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
        """Flow-matching Euler denoising matching
        ``NucleusMoEImagePipeline.__call__``.

        Precision invariants (binding):
        - Trajectory runs in fp32 (no autocast around the forward).
        - driver.forward_pass receives raw [0,1000] timesteps (normalizes,
          does NOT reverse, negates internally).
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
        patch_size = int(
            (getattr(self.pipeline.definition, "architecture_params", {}) or {})
            .get("transformer.patch_size", 2),
        )
        img_seq_len = (latents.shape[2] // patch_size) * (latents.shape[3] // patch_size)
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

        # CFG per NucleusMoEImagePipeline: gate at guidance_scale > 1.
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

                    noise_pred = _combine_cfg(
                        v_cond, v_uncond, guidance_scale, patch_size=patch_size,
                    )
                else:
                    noise_pred = v_cond

                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[
                    0
                ].to(torch.float32)

        return latents

    # ── VAE decode ───────────────────────────────────────────────────────

    def decode_latents(self, latents: Any) -> Image.Image:
        """Decode latent tensor to a PIL image.

        ``AutoencoderKLQwenImage`` is a 3-D causal VAE — re-add the frame
        dim (``[B, C, H, W] -> [B, C, 1, H, W]``) and apply the SAME
        ``latents_mean``/``latents_std`` normalization ``qwen_image``'s
        sampler uses (identical VAE class), then take frame 0 of the
        decoded output.
        """
        vae = self.pipeline.vae
        vae_dtype = next(vae.parameters()).dtype

        vae.to(self.device)

        latents_5d = latents.to(dtype=vae_dtype).unsqueeze(2)

        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latents_5d.device, latents_5d.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(vae.config.latents_std)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latents_5d.device, latents_5d.dtype)
        )
        latents_5d = latents_5d / latents_std + latents_mean

        with torch.no_grad():
            decoded = vae.decode(latents_5d, return_dict=False)

        image_tensor = decoded[0] if isinstance(decoded, (tuple, list)) else decoded
        if image_tensor.ndim == 5:
            image_tensor = image_tensor[:, :, 0]

        image_tensor = image_tensor.clamp(-1, 1)
        image_tensor = (image_tensor + 1.0) / 2.0
        image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)
        image_np = image_tensor.cpu().float().numpy()
        image_np = (image_np * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(image_np, mode="RGB")
