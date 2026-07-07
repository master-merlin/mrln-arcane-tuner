"""HunyuanVideo 1.5 sampler — strictly-fp32 FlowMatchEuler denoise.

Produces short clips during training as :class:`SampleArtifact` mp4s. The
denoise TRAJECTORY runs in fp32 with NO autocast around the loop — only the
transformer forward runs under the training autocast regime. Wrapping the loop
in ``autocast(bf16)`` collapses multi-step sampling toward the conditional
mean even when training is correct (the sampler-collapse gotcha the precision
contract pins).

CFG equivalence with the upstream guider
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The upstream ``HunyuanVideo15Pipeline`` has NO ``guidance_scale`` argument —
the scale lives in the repo's ``guider`` (``ClassifierFreeGuidance``,
``guidance_scale=6.0``, ``use_original_formulation=False``), whose
``prepare_inputs`` splits cond/uncond into SEQUENTIAL single forwards and
combines::

    pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

This sampler implements the byte-equivalent classic dual-forward CFG (two
forwards per step, fp32 combine) reading ``guidance_scale`` from the sample
config — the guider component itself is EXCLUDED from the loader manifest.
The equivalence is pinned by ``test_hv15_sampler_cfg.py``.

Sigma schedule
~~~~~~~~~~~~~~
The upstream pipeline passes ``sigmas = linspace(1.0, 0.0, N + 1)[:-1]`` (no
``mu``) into ``FlowMatchEulerDiscreteScheduler.set_timesteps``, which — with
the repo scheduler config (``use_dynamic_shifting=false, shift=5.0``) —
applies the static shift ``σ' = s·σ / (1 + (s−1)·σ)``. The shared
``shifted_sigmas`` helper reproduces exactly that.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog
import torch
from PIL import Image
from torch import Tensor

from app.engine.core.sampling import GenericSamplingPipeline, SampleArtifact
from app.engine.models.families.hunyuan_video15.driver import (
    build_model_input,
    build_t2v_cond_and_mask,
    zero_image_embeds,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.engine.core.pipeline import GenericTrainingPipeline

logger = structlog.get_logger(__name__)

HV15_NATIVE_FPS = 24.0
HV15_VAE_SPATIAL = 16
HV15_VAE_TEMPORAL = 4
HV15_LATENT_CHANNELS = 32

# The hv15 time embedder consumes the RAW FlowMatchEuler value (sinusoidal, no
# internal /1000); the trajectory steps in sigma ∈ [0, 1] and conditions the
# transformer on sigma * 1000 — the same scale training feeds.
HV15_FLOWMATCH_SCALE = 1000.0

# Repo scheduler config (verified hub scheduler_config.json): static shift.
HV15_DEFAULT_SHIFT = 5.0


class Hv15Sampler(GenericSamplingPipeline):
    """Flow-match Euler video sampler (fp32 trajectory) for hv15.

    Previews are always the T2V input contract (zero cond/mask channels + zero
    ``image_embeds``) — also for i2v runs, mirroring the WAN samplers.
    """

    def __init__(self, pipeline: GenericTrainingPipeline) -> None:
        super().__init__(pipeline)
        arch = (
            getattr(getattr(pipeline, "definition", None), "architecture_params", {})
            or {}
        )
        try:
            self.output_fps = float(arch.get("video.native_fps", HV15_NATIVE_FPS))
        except (TypeError, ValueError):
            self.output_fps = HV15_NATIVE_FPS

    # ── fp32 Euler integration core (the precision-critical path) ──────────

    def euler_integrate(
        self,
        x0: Tensor,
        sigmas: Tensor,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        """Integrate ``dx/dσ = v(x, σ)`` with forward Euler in fp32.

        The trajectory (sigma math + latent accumulation) is forced to fp32;
        ``velocity_fn`` may internally run a bf16 transformer but its result is
        cast back to fp32 before accumulation. Emits the per-step
        ``Sampling {i}/{N}`` status (1-based, byte-identical to the image
        families' format) through the JobLogWriter when one is attached.
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

    def build_denoise(
        self, transformer: Any, **forward_kwargs: Any
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Return a ``denoise(x0, sigmas)`` closure over a transformer.

        Used by the precision-contract test: a fake linear-velocity transformer
        runs through the REAL :meth:`euler_integrate` fp32 loop.
        """

        def _velocity(x: Tensor, sigma: Tensor) -> Tensor:
            t = sigma.reshape(1).to(x.dtype)
            out = transformer(x, t, **forward_kwargs)
            return out[0] if isinstance(out, tuple) else out

        def _denoise(x0: Tensor, sigmas: Tensor) -> Tensor:
            return self.euler_integrate(x0, sigmas, _velocity)

        return _denoise

    # ── Scheduler / sigma schedule ─────────────────────────────────────────

    def _build_sigmas(self, num_steps: int) -> Tensor:
        """Static-shift FlowMatchEuler sigma schedule, descending 1 → 0.

        Shift resolution: ``model_shift_fixed`` from config (injected by the
        video contract from the definition's ``scheduler.flow_shift``) →
        else the repo default 5.0.
        """
        from app.engine.strategies.sigma_schedule import shifted_sigmas

        shift = float(self.config.get("model_shift_fixed") or HV15_DEFAULT_SHIFT)
        return shifted_sigmas(num_steps, shift)

    # ── GenericSamplingPipeline hooks ──────────────────────────────────────

    def encode_prompt(self, prompt: str) -> Any:
        """Encode a prompt via the trainer's CACHED ``encode_text``.

        Sampling runs after the dual TE is offloaded — the trainer serves the
        ``(emb, mask, emb2, mask2)`` 4-tuple from the warm cache. The
        embeddings are cast to the MODEL dtype (contract: cached TE embeds are
        stored in the encode-time dtype and must be re-cast for the forward);
        the int64 masks are only moved to the device.
        """
        trainer = self.pipeline
        dtype = next(trainer.driver.get_primary_model().parameters()).dtype
        emb, mask, emb2, mask2 = trainer.encode_text([prompt], dtype)
        return (
            emb.to(self.device, dtype=dtype),
            mask.to(self.device),
            emb2.to(self.device, dtype=dtype),
            mask2.to(self.device),
        )

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator
    ) -> Tensor:
        """5D fp32 noise ``[1, 32, (F-1)/4+1, H/16, W/16]`` for the clip."""
        num_frames = int(self.config.get("sample_num_frames", 17))
        latent_f = (num_frames - 1) // HV15_VAE_TEMPORAL + 1
        lat_h = height // HV15_VAE_SPATIAL
        lat_w = width // HV15_VAE_SPATIAL
        shape = (1, HV15_LATENT_CHANNELS, latent_f, lat_h, lat_w)
        return torch.randn(
            shape, generator=generator, device=self.device, dtype=torch.float32
        )

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Any:
        """Full fp32 FlowMatchEuler denoise with REAL dual-forward CFG.

        ``guidance_scale > 1`` contrasts the conditional velocity against an
        unconditional (empty/negative-prompt) one, combined in fp32 as
        ``v = v_u + s * (v_c - v_u)`` — equivalent to the upstream guider's
        ``pred_uncond + gs * (pred_cond - pred_uncond)``. ``<= 1`` keeps the
        single conditional forward.

        The transformer forward runs in the SAME autocast regime as training
        (the hv15 transformer is mixed-dtype under mixed-precision training);
        the Euler trajectory stays fp32 OUTSIDE the autocast.
        """
        transformer = self.pipeline.driver.get_primary_model()
        self._ensure_transformer_on_device(transformer)

        autocast_dtype = (
            getattr(self.pipeline, "autocast_dtype", None) or torch.bfloat16
        )
        device_type = torch.device(self.device).type

        sigmas = self._build_sigmas(num_steps).to(self.device)
        text = prompt_embedding

        gs = float(guidance_scale) if guidance_scale is not None else None
        cfg_on = gs is not None and gs > 1.0
        text_uncond = None
        if cfg_on:
            neg_text = str(self.config.get("sample_negative_prompt", "") or "")
            text_uncond = self.encode_prompt(neg_text)

        def _forward(x: Tensor, sigma: Tensor, cond: Any) -> Tensor:
            emb, mask, emb2, mask2 = cond
            # T2V input contract: zero cond/mask channels + zero image_embeds
            # (the transformer detects the all-zero image stream).
            cond_lat, mask_c = build_t2v_cond_and_mask(x)
            hidden = build_model_input(x, cond_lat, mask_c)
            image_embeds = zero_image_embeds(x.shape[0], x.device, x.dtype)
            # RAW [0, 1000] timestep — the scale training feeds; the bare
            # sigma would make the frozen time embedder read t≈0.
            t = (sigma * HV15_FLOWMATCH_SCALE).reshape(1).expand(x.shape[0])
            with torch.no_grad(), torch.autocast(
                device_type=device_type, dtype=autocast_dtype
            ):
                out = transformer(
                    hidden_states=hidden,
                    timestep=t,
                    encoder_hidden_states=emb,
                    encoder_attention_mask=mask,
                    encoder_hidden_states_2=emb2,
                    encoder_attention_mask_2=mask2,
                    image_embeds=image_embeds,
                    return_dict=False,
                )
            return out[0] if isinstance(out, tuple) else out

        def _velocity(x: Tensor, sigma: Tensor) -> Tensor:
            v_c = _forward(x, sigma, text)
            if not cfg_on:
                return v_c
            v_c = v_c.to(torch.float32)
            v_u = _forward(x, sigma, text_uncond).to(torch.float32)
            return v_u + gs * (v_c - v_u)

        return self.euler_integrate(noise, sigmas, _velocity)

    def decode_latents(self, latents: Any) -> Image.Image | SampleArtifact:
        """VAE-decode the fp32 latent clip → a :class:`SampleArtifact` (mp4).

        The hv15 VAE uses a SCALAR ``scaling_factor`` (1.03682, no
        latents_mean/std): encode multiplied by it, so decode DIVIDES —
        byte-matching the upstream pipeline's
        ``latents.to(vae.dtype) / vae.config.scaling_factor``.
        """
        vae = self.pipeline.driver.vae
        scaling = float(vae.config.scaling_factor)
        with torch.no_grad():
            decoded = vae.decode(
                latents.to(vae.dtype) / scaling, return_dict=False
            )[0]

        # decoded: [B, C, F, H, W] in ~[-1, 1]; emit the first clip [C, F, H, W].
        clip = decoded[0].float().clamp(-1.0, 1.0)
        return SampleArtifact(frames=clip, fps=self.output_fps)
