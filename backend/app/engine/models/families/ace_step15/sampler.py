"""ACE-Step 1.5 sampler — strictly-fp32 flow-match Euler audio previews.

Produces in-training preview audio as :class:`AudioSampleArtifact` objects
(persisted as ``sample_XX_stepNNNNNN.wav`` by the base
``GenericSamplingPipeline`` — see ``core/sampling.py``). Replicates
``AceStepPipeline.__call__``'s denoise loop (diffusers 0.39.0) for the
turbo (guidance-distilled, no CFG) checkpoint this family ships by default:

- **fp32 trajectory, NO autocast**: the Euler accumulation runs in fp32; only
  the transformer forward runs in the model's own dtype (inputs cast
  per-call) — the same autocast-collapse guard every other family's sampler
  uses.
- **``context_latents`` constant**: same text2music default (silence +
  all-ones mask) as training — see ``driver._build_context_latents``.
- **Per-prompt lyrics/duration_s seam**: :meth:`_sample_single` stashes the
  full prompt config BEFORE ``encode_prompt`` runs (the base class sets
  ``_active_prompt_cfg`` too late for that call) so lyrics reach the
  condition encoder.
- **CFG (base/sft only, NOT the shipped turbo default)**: real APG (Adaptive
  Projected Guidance — https://huggingface.co/papers/2410.02416), NOT plain
  ``v_uncond + gs*(v_cond - v_uncond)``. Verified against the pipeline's own
  denoise loop (``pipeline_ace_step.py`` lines ~1156-1193, diffusers 0.39.0,
  read in full for task C2 / `ACE-Step/acestep-v15-xl-base-diffusers`'s
  recon): base/SFT checkpoints run
  ``diffusers.guiders.adaptive_projected_guidance.normalized_guidance(
  pred_cond=v_cond, pred_uncond=v_uncond, guidance_scale=guidance_scale-1.0,
  momentum_buffer=MomentumBuffer(momentum=-0.75), eta=0.0,
  norm_threshold=2.5, use_original_formulation=True, norm_dim=(1,))`` — a
  *stateful* (momentum carries across denoise steps), norm-clamped, direction-
  projected guidance blend, not a plain linear one. Both ``normalized_guidance``
  and ``MomentumBuffer`` are diffusers-native (already pinned, zero new
  vendoring) — reused verbatim rather than re-derived, so this driver can
  never silently drift from upstream's own formula. Against the model's
  LEARNED ``null_condition_emb`` (never a re-encoded empty string — the
  pipeline's own comment: that's out-of-distribution). Turbo (this family's
  shipped default checkpoint, ``definitions/base.yaml``) never exercises this
  path: ``guidance_scale`` defaults to 1.0 and ``is_turbo`` disables CFG
  outright, matching upstream's own guidance-distillation behavior — pinned
  by ``test_denoise_turbo_ignores_cfg`` (unchanged regression pin). The XL
  base checkpoint (``definitions/xl_base.yaml``, task C2) is the first
  shipped definition to actually exercise this branch, with
  ``guidance_scale: 7.0`` matching the base model card's documented default.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog
import torch
from diffusers.guiders.adaptive_projected_guidance import (
    MomentumBuffer,
    normalized_guidance,
)
from torch import Tensor

from app.engine.core.sampling import AudioSampleArtifact, GenericSamplingPipeline

from .driver import AceStep15Driver

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.engine.core.pipeline import GenericTrainingPipeline

logger = structlog.get_logger(__name__)

# Repo scheduler config default (turbo checkpoint's recommended shift; see
# recon report §7 / the diffusers pipeline's `shift: float = 3.0` default).
ACE_STEP15_DEFAULT_SHIFT = 3.0

# APG (Adaptive Projected Guidance) constants — byte-identical to the values
# hard-coded in `AceStepPipeline.__call__`'s denoise loop (diffusers 0.39.0,
# `pipeline_ace_step.py` lines ~1159/1184-1193): these are NOT exposed as
# pipeline kwargs upstream either, so there is nothing to make user-facing
# here — matching upstream means matching these exact constants.
ACE_STEP15_APG_MOMENTUM = -0.75
ACE_STEP15_APG_ETA = 0.0
ACE_STEP15_APG_NORM_THRESHOLD = 2.5
ACE_STEP15_APG_NORM_DIM: tuple[int, ...] = (1,)


class AceStep15Sampler(GenericSamplingPipeline):
    """Flow-match Euler audio sampler for ACE-Step 1.5 (fp32 trajectory)."""

    def __init__(self, pipeline: "GenericTrainingPipeline") -> None:
        super().__init__(pipeline)

    # ── fp32 Euler integration core (the precision-critical path) ──────────

    def euler_integrate(
        self,
        x0: Tensor,
        sigmas: Tensor,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        """Integrate ``dx/dsigma = v(x, sigma)`` with forward Euler in fp32.

        Mirrors ``AceStepPipeline``'s scheduler-driven loop: the pipeline's
        ``FlowMatchEulerDiscreteScheduler.step`` for THIS shift-schedule
        reduces to plain ``x = x + (sigma_next - sigma) * v`` (num_train_
        timesteps=1, shift=1.0 on the scheduler object itself — the pipeline
        applies its OWN shift to the sigma values before handing them to the
        scheduler, so the scheduler's internal step is unshifted linear
        Euler). No extra channels/frame pinning here (unlike video families —
        ACE-Step's "what's given" side-channel is the SEPARATE
        ``context_latents`` argument, not part of the trajectory state).
        """
        x = x0.to(torch.float32)
        s = sigmas.to(torch.float32)
        total_steps = len(s) - 1
        for i in range(total_steps):
            if getattr(self, "_log_writer", None):
                self._log_writer.status(f"Sampling {i + 1}/{total_steps}")
            dt = (s[i + 1] - s[i]).to(torch.float32)
            v = velocity_fn(x, s[i]).to(torch.float32)
            x = x + dt * v
        return x

    def _build_sigmas(self, num_steps: int) -> Tensor:
        """Shifted linear schedule, descending 1 -> 0 (fp32) — byte-identical
        to ``AceStepPipeline._get_timestep_schedule`` before its trailing
        ``[:-1]`` truncation (which the scheduler re-appends internally as a
        terminal 0 — net result is the SAME full descending array this
        returns, ready for :meth:`euler_integrate`'s ``len(sigmas) - 1``
        interval convention)."""
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        shift = float(
            self.config.get("model_shift_fixed")
            or arch.get("scheduler.shift")
            or ACE_STEP15_DEFAULT_SHIFT
        )
        t = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)
        if shift != 1.0:
            t = shift * t / (1 + (shift - 1) * t)
        return t

    # ── GenericSamplingPipeline hooks ──────────────────────────────────────

    def _sample_single(
        self, prompt_cfg: dict[str, Any], step: int
    ) -> AudioSampleArtifact:
        """Stash the full prompt config BEFORE the base class's
        ``encode_prompt(prompt)`` call — it only forwards the prompt STRING,
        but :meth:`encode_prompt` needs ``lyrics`` from the same config. The
        base class sets ``_active_prompt_cfg`` too, but only AFTER
        ``encode_prompt`` already ran (see ``core/sampling.py::_sample_single``).
        """
        self._active_prompt_cfg = prompt_cfg
        return super()._sample_single(prompt_cfg, step)

    def encode_prompt(self, prompt: str) -> tuple[Tensor, Tensor]:
        """Encode via the trainer's CACHED ``encode_text`` (both the TE and
        the condition encoder are offloaded by sample time; the pre-cache
        warmed every sample prompt's (caption, lyrics) pair)."""
        cfg = getattr(self, "_active_prompt_cfg", None) or {}
        lyrics = cfg.get("lyrics") or ""
        trainer = self.pipeline
        out = trainer.encode_text([prompt], torch.float32, batch={"lyrics": [lyrics]})
        if not (isinstance(out, tuple) and len(out) == 2):
            raise TypeError(  # pragma: no cover - defensive
                "AceStep15 encode_text must return (encoder_hidden_states, "
                "encoder_attention_mask)"
            )
        return out

    def _effective_sample_duration_s(self, default: float) -> float:
        """Per-prompt ``duration_s`` override (``SamplePromptConfig.duration_s``),
        the audio analogue of ``_effective_sample_frames``."""
        cfg = getattr(self, "_active_prompt_cfg", None) or {}
        per_prompt = cfg.get("duration_s")
        return float(per_prompt) if per_prompt else float(default)

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator
    ) -> Tensor:
        """``[1, T_lat, D]`` fp32 noise (width/height are unused — audio has
        no spatial dims; the base class's signature still passes them)."""
        driver: AceStep15Driver = self.pipeline.driver
        duration_s = self._effective_sample_duration_s(
            self.config.get("duration_s", 30.0)
        )
        latent_length = max(math.ceil(duration_s * driver.latents_per_second), 1)
        shape = (1, latent_length, driver.audio_acoustic_hidden_dim)
        return torch.randn(
            shape, generator=generator, device=self.device, dtype=torch.float32
        )

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: tuple[Tensor, Tensor],
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Tensor:
        """Full fp32 flow-match Euler denoise (see module docstring for the
        turbo-default / optional-CFG split)."""
        driver: AceStep15Driver = self.pipeline.driver
        transformer = driver.get_primary_model()
        if isinstance(transformer, torch.nn.Module):
            self._ensure_transformer_on_device(transformer)
        model_dtype = self._model_dtype(transformer)

        x = noise.to(torch.float32)
        b, t_len, _ = x.shape
        context_latents = driver._build_context_latents(
            b, t_len, x.device, model_dtype
        )

        is_turbo = bool(getattr(transformer.config, "is_turbo", False))
        gs = float(guidance_scale) if guidance_scale is not None else None
        cfg_on = (not is_turbo) and gs is not None and gs > 1.0

        encoder_hidden_states, _mask = prompt_embedding
        null_emb = None
        # Momentum is STATEFUL across denoise steps (upstream instantiates
        # ONE MomentumBuffer before the loop, not per-step) — see module
        # docstring's "CFG (base/sft only...)" section.
        momentum_buffer = MomentumBuffer(momentum=ACE_STEP15_APG_MOMENTUM) if cfg_on else None
        if cfg_on:
            null_emb = driver.condition_encoder.null_condition_emb.to(
                x.device, model_dtype
            ).expand_as(encoder_hidden_states)

        def _forward(x_full: Tensor, t01: Tensor, cond: Tensor) -> Tensor:
            t_tensor = t01.reshape(1).expand(b).to(model_dtype)
            with torch.no_grad():
                out = transformer(
                    hidden_states=x_full.to(model_dtype),
                    timestep=t_tensor,
                    timestep_r=t_tensor,
                    encoder_hidden_states=cond.to(x_full.device, model_dtype),
                    context_latents=context_latents,
                    return_dict=False,
                )
            return out[0] if isinstance(out, (tuple, list)) else out

        def _velocity(x_full: Tensor, sigma: Tensor) -> Tensor:
            v_c = _forward(x_full, sigma, encoder_hidden_states)
            if not cfg_on:
                return v_c
            v_c = v_c.to(torch.float32)
            v_u = _forward(x_full, sigma, null_emb).to(torch.float32)
            # Real APG (not a plain linear CFG blend) — byte-identical call
            # shape to `AceStepPipeline.__call__`'s own `normalized_guidance`
            # invocation (module docstring). `guidance_scale - 1.0` is
            # upstream's own offset convention for `use_original_formulation
            # =True` (the "original formulation" CFG paper's `pred_cond +
            # (w-1)*update`, not the diffusers-native `pred_uncond + w*update`
            # convention).
            return normalized_guidance(
                pred_cond=v_c,
                pred_uncond=v_u,
                guidance_scale=gs - 1.0,
                momentum_buffer=momentum_buffer,
                eta=ACE_STEP15_APG_ETA,
                norm_threshold=ACE_STEP15_APG_NORM_THRESHOLD,
                use_original_formulation=True,
                norm_dim=ACE_STEP15_APG_NORM_DIM,
            )

        sigmas = self._build_sigmas(num_steps).to(x.device)
        return self.euler_integrate(x, sigmas, _velocity)

    def decode_latents(self, latents: Tensor) -> AudioSampleArtifact:
        """``[1, T, D]`` latents -> Oobleck VAE decode -> WAV artifact.

        Replicates the pipeline's exact two-stage output normalization
        (hard anti-clip, then rescale to a consistent -1 dBFS peak) — without
        it, decoded previews are inconsistently loud relative to the
        reference model's inference output.

        C3 DEPENDENCY — jobs-API visibility: the returned artifact is
        persisted by ``GenericSamplingPipeline._persist_artifact`` as
        ``samples/sample_XX_stepNNNNNN.wav``, but the jobs API does NOT
        surface it yet: ``app/api/training/job_routes.py``'s
        ``_SAMPLE_EXTENSIONS`` has no ``.wav`` entry, so ``list_job_samples``
        silently skips every audio preview (the file IS on disk and correct).
        Task C3 (frontend audio sample tile) owns extending
        ``_SAMPLE_EXTENSIONS`` + the ``get_sample_image`` content-type
        handling — deliberately NOT patched from this branch.
        """
        driver: AceStep15Driver = self.pipeline.driver
        vae = driver.vae
        audio_latents = latents.to(torch.float32).transpose(1, 2)  # [B,T,D]->[B,D,T]
        with torch.no_grad():
            audio = vae.decode(audio_latents.to(vae.dtype), return_dict=False)[0]
        audio = audio.float()

        peak = audio.abs().amax(dim=[1, 2], keepdim=True)
        if torch.any(peak > 1.0):
            audio = audio / peak.clamp(min=1.0)
        target_amp = 10.0 ** (-1.0 / 20.0)  # -1 dBFS
        peak = audio.abs().amax(dim=[1, 2], keepdim=True).clamp(min=1e-6)
        audio = audio * (target_amp / peak)

        waveform = audio[0].clamp(-1.0, 1.0)  # [C, T]
        sample_rate = int(getattr(vae.config, "sampling_rate", 48000))
        return AudioSampleArtifact(waveform=waveform, sample_rate=sample_rate)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _model_dtype(transformer: Any) -> torch.dtype:
        dtype = getattr(transformer, "dtype", None)
        if isinstance(dtype, torch.dtype):
            return dtype
        try:
            return next(transformer.parameters()).dtype
        except (AttributeError, StopIteration):
            return torch.float32
