"""BooguImageSampler — in-training preview sampler for Boogu-Image (Base +
Turbo), Task 6.

Mirrors ``BooguImagePipeline.processing()`` (Base, vendored-scheduler CFG
loop) and ``BooguImageTurboPipeline.processing()`` (Turbo, DMD few-step
loop) — see the upstream clone at
``.agent/workdir/sdd-boogu/upstream/boogu/pipelines/boogu/`` and
``task-6-report.md`` for the full file:line evidence trail. Reuses
``driver.forward_pass`` (list-tensor adapter, freqs_cis, raw-``[0,1)``
timestep contract — all proven in Tasks 4/5) and the driver's
``encode_text`` (Qwen3-VL, per-caption system-prompt selection) rather than
duplicating any of that logic.

## Base loop (``pipeline_boogu.py:3243`` ``processing()``)

- Scheduler: the LOADER-provided vendored ``FlowMatchEulerDiscreteScheduler``
  (``driver.scheduler``) — NEVER a fresh/stock instance (it carries the
  checkpoint's ``do_shift``/``time_shift_version``/``seq_len`` config).
  ``set_timesteps(num_inference_steps, device, num_tokens=H*W)`` mirrors
  ``retrieve_timesteps`` -> ``scheduler.set_timesteps(..., num_tokens=
  latents.shape[-2] * latents.shape[-1])`` (:3277-3283).
- Guidance: with ``control_inputs: 0`` (pure T2I, both shipped definitions),
  ``_get_task_type_by_ref_latents`` (:3001-3011) always returns ``"t2i"``
  (``ref_latents`` is always empty), so the branch that ALWAYS executes is
  the final ``elif text_guidance_scale > 1.0:`` (:3615-3649) — Lumina-style
  scale-1 guidance:
  ``model_pred = model_pred + (text_guidance_scale - 1) * (model_pred -
  model_pred_drop_all)``, where ``model_pred_drop_all`` is predicted with
  ``negative_instruction_embeds`` (upstream's ``encode_instruction`` default
  negative instruction is ``""``, i.e. ``driver.encode_text("")`` — the DROP
  system prompt per driver.py's ``_select_system_prompt``) and
  ``ref_image_hidden_states=None``. Gate: ``text_guidance_scale > 1.0`` ->
  ON (two forwards); ``== 1.0`` -> OFF (one forward, no negative encode).
  ``cfg_range`` defaults to ``(0.0, 1.0)`` (:2682) i.e. guidance applies at
  EVERY step by default — the shipped definitions carry no override, so this
  sampler applies it uniformly across the whole walk (no windowing knob).
- Step: ``scheduler.step(model_pred, t, latents, return_dict=False)[0]``
  (:3651-3653) — plain forward Euler, already proven exact by the driver's
  perfect-velocity round-trip test; reused verbatim here via the loader
  scheduler.
- Decode: ``latents / scaling_factor`` THEN ``+ shift_factor`` (:3682-3685,
  in that order) -> ``vae.decode(latents)``.

## Turbo loop (``pipeline_boogu_turbo.py``)

- Sigma ladder: ``linspace(conditioning_sigma, 1.0, steps+1)[:-1]``
  (``_build_dmd_student_sigmas``, :43-72), default ``conditioning_sigma =
  0.001`` (``__call__`` default, :128).
- Predict step (``_predict_dmd_student_step``, :74-98): velocity forward at
  the current ``(latents, sigma)``, then
  ``x0_hat = latents + (1 - sigma) * model_pred`` — the one-step "clean
  estimate" (algebraically: ``x0_hat == x0`` exactly when ``model_pred``
  is the true ``x0 - noise`` velocity for the ``(noise, x0)`` pair that
  produced the current ``latents`` via Boogu's own
  ``x_t = (1-t)*noise + t*x0`` lerp — see driver.py's time-convention
  derivation; the DMD sigma plays the same role as the driver's ``t``).
- Renoise (``_renoise_dmd_latents``, :100-118, skipped on the LAST step):
  draws FRESH noise (independent of whatever produced the current latents)
  and re-lerps the ``x0_hat`` estimate to the NEXT sigma:
  ``latents = (1 - sigma_next) * fresh_noise + sigma_next * x0_hat`` — same
  lerp form as ``add_noise``, with ``sigma_next`` playing the role of ``t``.
- Hard assert (:163-171): DMD student inference requires
  ``text_guidance_scale == image_guidance_scale == 1.0`` and
  ``empty_instruction_guidance_scale == 0.0`` (no CFG at all). This sampler
  only exposes a single ``guidance_scale`` (text_gs equivalent — the
  image/empty terms are TI2I-only and this family is pure T2I,
  ``control_inputs: 0``), so it asserts ``guidance_scale == 1.0``.
- Decode: IDENTICAL order/formula to the Base loop's tail (:211-218).

## Precision contract (binding, house-wide gotchas)

fp32 latent trajectory throughout; ``torch.no_grad()``; NO
``torch.autocast`` anywhere (autocast-collapse gotcha — collapses N-step
sampling to the conditional mean); cached text embeddings are cast to the
model's own dtype immediately before each forward (fp32 cache vs bf16 model
crash otherwise, mirroring ``driver.forward_pass``'s own dtype handling).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
import torch
from PIL import Image
from torch import Tensor

from app.engine.core.sampling import GenericSamplingPipeline

if TYPE_CHECKING:
    from .trainer import BooguImageTrainer

logger = structlog.get_logger(__name__)

# Native sample defaults, definition-sourced (`defaults:` block in
# base.yaml / turbo.yaml). These constants are ONLY the fallback used when a
# definition's own `defaults` dict is missing the key entirely — the
# shipped YAMLs always carry them (task-2-brief.md).
_DEFAULT_RESOLUTION = 1024
_BASE_DEFAULT_STEPS = 50
_BASE_DEFAULT_GUIDANCE = 4.0
_TURBO_DEFAULT_STEPS = 4
_TURBO_DEFAULT_GUIDANCE = 1.0

# pipeline_boogu_turbo.py:128 (__call__'s dmd_conditioning_sigma default).
_DMD_CONDITIONING_SIGMA = 0.001

# Empty-string CFG negative -- driver.encode_text("") routes through the
# DROP system prompt (driver.py _select_system_prompt). Do NOT substitute a
# user-configurable `sample_negative_prompt` here: the base checkpoint's
# learned unconditional anchor lives specifically under the DROP prompt for
# an EMPTY caption (Task-5 review Finding 2), not under some other text.
_CFG_NEGATIVE_CAPTION = ""


def _build_dmd_sigmas(
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    conditioning_sigma: float,
) -> Tensor:
    """DMD sigma ladder — ``pipeline_boogu_turbo.py:66-72`` verbatim (the
    no-custom-timesteps branch; this sampler never passes custom
    timesteps)."""
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1 for DMD student inference.")
    return torch.linspace(
        conditioning_sigma, 1.0, num_steps + 1, device=device, dtype=dtype,
    )[:-1]


class BooguImageSampler(GenericSamplingPipeline):
    """Boogu-Image sampler — Base (vendored-scheduler CFG loop) dispatches
    to :meth:`_denoise_base`, Turbo (DMD few-step) to :meth:`_denoise_turbo`,
    selected by the definition's ``defaults.is_distilled`` flag (same knob
    ``BooguImageDriver``/krea2 precedent uses to distinguish Raw/Turbo)."""

    pipeline: "BooguImageTrainer"

    def __init__(self, pipeline: "BooguImageTrainer") -> None:
        super().__init__(pipeline)

    # ── Base/Turbo dispatch ──────────────────────────────────────────────

    def _is_turbo(self) -> bool:
        defn = self.pipeline.definition
        defaults = getattr(defn, "defaults", {}) or {}
        return bool(defaults.get("is_distilled", False))

    # ── Native sample defaults (ovis F-lesson: generic 20/3.5 defaults are
    #    off-distribution) ───────────────────────────────────────────────

    def _sample_single(self, prompt_cfg: dict[str, Any], step: int) -> Any:
        cfg = dict(prompt_cfg)
        defn = self.pipeline.definition
        defaults = getattr(defn, "defaults", {}) or {}
        turbo = self._is_turbo()

        resolution = int(defaults.get("resolution", _DEFAULT_RESOLUTION))
        default_steps = _TURBO_DEFAULT_STEPS if turbo else _BASE_DEFAULT_STEPS
        default_guidance = (
            _TURBO_DEFAULT_GUIDANCE if turbo else _BASE_DEFAULT_GUIDANCE
        )
        fill = {
            "width": resolution,
            "height": resolution,
            "num_inference_steps": int(
                defaults.get("num_inference_steps", default_steps),
            ),
            "guidance_scale": float(
                defaults.get("guidance_scale", default_guidance),
            ),
        }
        for key, value in fill.items():
            if cfg.get(key) in (None, 0):
                cfg[key] = value
        return super()._sample_single(cfg, step)

    # ── Text encoding ────────────────────────────────────────────────────

    def encode_prompt(self, prompt: str) -> dict[str, Any]:
        """Encode via the trainer's cache-aware ``encode_text()`` (NOT
        ``driver.encode_text`` directly) so the TE-cache/offload path is
        used, matching krea2/ovis. Returns ``embeds`` [1, L, 4096] and
        ``mask`` [1, L]."""
        trainer = self.pipeline
        dtype = next(trainer.transformer.parameters()).dtype
        embeds, mask = trainer.encode_text([prompt], dtype=dtype)
        return {"embeds": embeds, "mask": mask}

    # ── Initial noise ────────────────────────────────────────────────────

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator,
    ) -> Tensor:
        """``prepare_latents`` (``pipeline_boogu.py:836-874``): plain
        ``height // vae_scale_factor`` / ``width // vae_scale_factor``
        (no 2x2 pre-packing at the latent-grid level — the transformer's
        own ``patch_size`` packing happens internally, contract 4/5)."""
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        vae_sf = int(arch.get("vae.vae_scale_factor", 8))
        lat_channels = int(arch.get("vae.latent_channels", 16))

        lat_h = height // vae_sf
        lat_w = width // vae_sf

        return torch.randn(
            (1, lat_channels, lat_h, lat_w),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )

    # ── Denoise dispatch ─────────────────────────────────────────────────

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Tensor:
        if self._is_turbo():
            return self._denoise_turbo(
                noise, prompt_embedding, num_steps, guidance_scale, seed,
            )
        return self._denoise_base(
            noise, prompt_embedding, num_steps, guidance_scale, seed,
        )

    # ── Base denoise loop (vendored scheduler, Lumina-style CFG) ─────────

    def _denoise_base(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Tensor:
        """Mirrors ``BooguImagePipeline.processing()`` for pure T2I
        (``control_inputs: 0`` -> the final ``elif text_guidance_scale >
        1.0:`` branch, ``pipeline_boogu.py:3615-3649``). See module
        docstring for the full derivation."""
        device = self.device
        driver = self.pipeline.driver
        transformer = self.pipeline.transformer

        scheduler = driver.scheduler
        if scheduler is None:
            raise RuntimeError(
                "boogu_image sampler: driver.scheduler is None — the "
                "LOADER-provided vendored scheduler must be assigned "
                "(assign_components()) before sampling; a fresh/stock "
                "scheduler would silently drop the checkpoint's shift "
                "config."
            )

        dtype = next(transformer.parameters()).dtype

        prompt_embeds = prompt_embedding["embeds"]
        prompt_mask = prompt_embedding["mask"]

        # fp32 trajectory (no autocast) — precision contract.
        latents = noise.to(device=device, dtype=torch.float32)

        # retrieve_timesteps -> scheduler.set_timesteps(..., num_tokens=
        # latents.shape[-2] * latents.shape[-1]) (pipeline_boogu.py:3277-3283).
        num_tokens = latents.shape[-2] * latents.shape[-1]
        scheduler.set_timesteps(
            num_inference_steps=num_steps, device=device, num_tokens=num_tokens,
        )
        timesteps = scheduler.timesteps

        # Gate: text_gs > 1.0 -> ON (two forwards); == 1.0 -> OFF.
        cfg_on = float(guidance_scale) > 1.0
        uncond_embeds = None
        uncond_mask = None
        if cfg_on:
            # driver.encode_text("") -> the DROP system prompt (Task-5 fix;
            # do NOT hand-roll a different negative encode). Encoded once,
            # reused across all steps.
            neg_embedding = self.encode_prompt(_CFG_NEGATIVE_CAPTION)
            uncond_embeds = neg_embedding["embeds"]
            uncond_mask = neg_embedding["mask"]

        self._ensure_transformer_on_device(transformer)

        with torch.no_grad():
            total_steps = len(timesteps)
            for step_i, t in enumerate(timesteps, 1):
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Sampling {step_i}/{total_steps}")

                # Cast latents to model dtype for the forward; fp32
                # trajectory stays outside.
                xin = latents.to(dtype=dtype)

                # Raw [0, 1) timestep (contract 2 — no /1000 or *1000).
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

                    # Lumina-style scale-1 guidance (pipeline_boogu.py:3649).
                    noise_pred = v_cond + (guidance_scale - 1.0) * (v_cond - v_uncond)
                else:
                    noise_pred = v_cond

                # Loader-provided scheduler's own forward-Euler step.
                latents = scheduler.step(
                    noise_pred, t, latents, return_dict=False,
                )[0].to(torch.float32)

        return latents

    # ── Turbo denoise loop (DMD few-step) ─────────────────────────────────

    def _denoise_turbo(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Tensor:
        """Mirrors ``BooguImageTurboPipeline.processing()``'s DMD branch
        (``pipeline_boogu_turbo.py:144-218``). See module docstring for the
        renoise-formula derivation."""
        if float(guidance_scale) != 1.0:
            raise ValueError(
                "boogu_image turbo sampler requires guidance_scale == 1.0 "
                "(DMD student inference has no CFG) — mirrors "
                "pipeline_boogu_turbo.py's hard assert "
                "(text_guidance_scale == image_guidance_scale == 1.0, "
                "empty_instruction_guidance_scale == 0.0). "
                f"Got guidance_scale={guidance_scale!r}."
            )

        device = self.device
        driver = self.pipeline.driver
        transformer = self.pipeline.transformer
        dtype = next(transformer.parameters()).dtype

        prompt_embeds = prompt_embedding["embeds"]
        prompt_mask = prompt_embedding["mask"]

        latents = noise.to(device=device, dtype=torch.float32)

        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        conditioning_sigma = float(
            arch.get("turbo.dmd_conditioning_sigma", _DMD_CONDITIONING_SIGMA),
        )
        sigmas = _build_dmd_sigmas(
            num_steps, device, torch.float32, conditioning_sigma,
        )

        self._ensure_transformer_on_device(transformer)

        # The house denoise() interface receives only `seed` (not the
        # generator used for _create_initial_noise) — a fresh generator is
        # deterministically re-derived from it for the renoise draws
        # (documented deviation from upstream's SINGLE shared generator
        # threaded through both initial latents and renoise —
        # pipeline_boogu_turbo.py:137/:178/:204 — which is not plumbed
        # through this interface).
        #
        # The `+ 1` is LOAD-BEARING, do not "simplify" it away: the base
        # _sample_single already seeded a generator with `seed` and drew the
        # initial noise as its FIRST randn of this exact latent shape
        # (sampling.py:573-574). Re-seeding with the SAME seed here would
        # make the first renoise draw bitwise IDENTICAL to the initial
        # latents — violating the DMD renoise contract of independent noise
        # (upstream's shared generator yields distinct sequential draws).
        # Pinned by TestTurboRenoiseMath::
        # test_first_renoise_draw_decorrelated_from_initial_noise.
        generator = torch.Generator(device=device).manual_seed(seed + 1)

        with torch.no_grad():
            sigma_list = sigmas.tolist()
            total_steps = len(sigma_list)
            for step_i, sigma in enumerate(sigma_list):
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Sampling {step_i + 1}/{total_steps}")

                xin = latents.to(dtype=dtype)
                ts = torch.full(
                    (xin.shape[0],), sigma, device=device, dtype=dtype,
                )

                v = driver.forward_pass(
                    noisy_input=xin,
                    timesteps=ts,
                    text_embeddings=(
                        prompt_embeds.to(dtype=dtype),
                        prompt_mask,
                    ),
                    batch={},
                ).to(torch.float32)

                # _predict_dmd_student_step (pipeline_boogu_turbo.py:74-98).
                x0_hat = latents + (1.0 - sigma) * v

                if step_i < total_steps - 1:
                    sigma_next = sigma_list[step_i + 1]
                    fresh_noise = torch.randn(
                        latents.shape, generator=generator,
                        device=device, dtype=torch.float32,
                    )
                    # _renoise_dmd_latents (pipeline_boogu_turbo.py:100-118).
                    latents = (
                        (1.0 - sigma_next) * fresh_noise + sigma_next * x0_hat
                    )
                else:
                    latents = x0_hat

        return latents

    # ── VAE decode ───────────────────────────────────────────────────────

    def decode_latents(self, latents: Tensor) -> Image.Image:
        """``latents / scaling_factor + shift_factor`` -> ``vae.decode``
        (``pipeline_boogu.py:3681-3686`` / ``pipeline_boogu_turbo.py:
        211-218`` — identical order in both loops' decode tail).

        Device placement is owned by the base ``_sample_single`` (its Phase 3
        ``_ensure_on_gpu(["vae"])`` / ``_offload_to_cpu`` bracket) — no
        ``vae.to(...)`` here, so offload bookkeeping has exactly one owner.
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
