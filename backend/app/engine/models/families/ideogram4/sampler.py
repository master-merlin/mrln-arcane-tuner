"""Ideogram 4 in-training sampler.

Reproduces the upstream ``Ideogram4Pipeline`` denoise path using the trainer's
already-loaded components (vendored DiT, custom Ideogram4 VAE, Qwen3-VL text
encoder). The denoise reuses the driver's validated :meth:`forward_pass` so the
text+image sequence packing (position_ids / segment_ids / indicator) is
identical to training.

Family specifics that shape this sampler:

* **Custom flow-match scheduler (ported, NOT diffusers).** Upstream Ideogram 4
  does **not** use ``FlowMatchEulerDiscreteScheduler``; it uses a
  resolution-aware **logit-normal** noise schedule
  (``scheduler.LogitNormalSchedule`` / ``get_schedule_for_resolution``) over a
  linear ``[0, 1]`` step grid (``make_step_intervals``), with a plain Euler
  flow-match update ``z = z + v * (s - t)`` walking ``t: 1 -> 0``. Forcing
  diffusers here would change the sigma spacing materially, so the upstream
  schedule + step are ported verbatim in :class:`_Ideogram4FlowSchedule`. The
  ``mu``/``std`` defaults match the fp8 repo's ``V4_DEFAULT_20`` preset
  (``mu=0.0``, ``std=1.75``); ``mu`` is the schedule mean at the reference
  512x512 resolution and shifts with the image area.

* **Asymmetric CFG.** Both branches go through the SAME DiT via
  ``driver.forward_pass``: the conditional branch sends the real Qwen3-VL text
  features; the unconditional branch sends a ZEROED text-feature tensor of the
  same shape (so the packing/positions/indicator are identical and only
  ``llm_features`` differ). This mirrors upstream, which runs the negative
  branch with ``neg_llm_features = zeros``. The CFG combine
  ``uncond + scale*(cond - uncond)`` is done in fp32.

* **Normalized flow space.** Training normalizes image latents
  (``utils.normalize_latents``) but keeps noise raw (Task 5), so the model's
  flow space IS the normalized latent space. The sampler starts from raw noise
  in sequence space ``[1, S, 128]``, runs the Euler loop (the model predicts
  velocity in normalized space), then ``denormalize_latents`` ->
  ``unpatchify_from_seq`` -> VAE ``decode`` to pixels. This matches upstream
  ``pipeline_ideogram4.py::_decode`` (``z = z*scale + shift``, unpatchify, then
  ``autoencoder.decoder(z)``).

* **Timestep convention.** The schedule produces timesteps in the shared
  trainer ``[0, 1000]`` convention; they are passed STRAIGHT to
  ``driver.forward_pass``, which divides by ``NUM_TRAIN_TIMESTEPS=1000`` to the
  ``[0, 1]`` the DiT's scalar embedder wants. The sampler does NOT pre-divide
  (mirrors microsoft_lens; guards the prior x1000 embedder bug).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import structlog
import torch
from PIL import Image
from torch import Tensor

from app.engine.core.sampling import GenericSamplingPipeline
from app.engine.models.families.ideogram4 import utils

if TYPE_CHECKING:
    from .trainer import IdeogramV4Trainer

logger = structlog.get_logger(__name__)

# Trainer/driver timestep convention: the driver's forward_pass divides the
# incoming timestep by this to reach the DiT's [0, 1] flow value. The sampler
# therefore emits timesteps in [0, 1000] (raw schedule value * this).
NUM_TRAIN_TIMESTEPS = 1000.0

# fp8-repo V4_DEFAULT_20 preset (sampler_configs.py): mu=0.0, std=1.75.
DEFAULT_SCHEDULE_MU = 0.0
DEFAULT_SCHEDULE_STD = 1.75


# ── Module-level pure helpers (unit-tested) ──────────────────────────────────


def zeroed_like_text(feats: Tensor) -> Tensor:
    """Zeroed text-feature tensor of the same shape (asymmetric-CFG uncond).

    Upstream runs the negative branch with ``neg_llm_features = zeros`` of the
    text feature shape; we reuse the SAME DiT/packing and only zero the text
    features, so positions/indicator/segment_ids stay identical to the cond
    branch.
    """
    return torch.zeros_like(feats)


def combine_asymmetric_cfg(
    cond: Tensor, uncond: Tensor, guidance_scale: float,
) -> Tensor:
    """``uncond + scale*(cond - uncond)`` in fp32 (asymmetric CFG combine)."""
    u = uncond.float()
    return u + guidance_scale * (cond.float() - u)


# ── Ported upstream logit-normal flow schedule ───────────────────────────────


class _Ideogram4FlowSchedule:
    """Resolution-aware logit-normal schedule + linear Euler step (ported).

    Ports ``ideogram4.scheduler.LogitNormalSchedule`` +
    ``get_schedule_for_resolution`` + ``make_step_intervals`` and the upstream
    ``__call__`` Euler update. The schedule maps a linear ``u in [0, 1]`` grid
    to a flow time ``t in (~0, ~1)`` via a logit-normal warp whose mean grows
    with the image area (``known_mean + 0.5*log(area/512^2)``).

    Loop walks ``t: ~1 -> ~0`` over ``num_steps`` Euler steps; each step uses
    ``delta = t(u_i) - t(u_{i+1})`` and updates ``z = z + v * delta``.
    """

    # Logit-SNR clamps (upstream LogitNormalSchedule defaults).
    LOGSNR_MIN = -15.0
    LOGSNR_MAX = 18.0

    def __init__(
        self,
        num_steps: int,
        height: int,
        width: int,
        mu: float = DEFAULT_SCHEDULE_MU,
        std: float = DEFAULT_SCHEDULE_STD,
    ) -> None:
        self.num_steps = num_steps
        # Stashed so the sampler's cache key can detect a resolution change
        # (mean depends on height/width — see below).
        self.height = height
        self.width = width
        # Resolution-aware mean: known_mean (== mu) at 512x512, +0.5*log(area
        # ratio) elsewhere (get_schedule_for_resolution).
        num_pixels = height * width
        known_pixels = 512 * 512
        self.mean = mu + 0.5 * math.log(num_pixels / known_pixels)
        self.std = std
        # Linear step grid u_0..u_num_steps over [0, 1] (make_step_intervals).
        self._intervals = torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float64)

    def _warp(self, u: torch.Tensor) -> torch.Tensor:
        """Logit-normal warp of a linear ``u in [0, 1]`` to flow time (float64).

        Mirrors ``LogitNormalSchedule.__call__``: ``t = 1 - sigmoid(mean +
        std*ndtri(u))``, clamped to the logit-SNR-implied ``[t_min, t_max]``.
        """
        u = u.to(torch.float64)
        z = torch.special.ndtri(u)
        y = self.mean + self.std * z
        t_ = 1.0 - torch.special.expit(y)
        t_min = 1.0 / (1 + math.exp(0.5 * self.LOGSNR_MAX))
        t_max = 1.0 / (1 + math.exp(0.5 * self.LOGSNR_MIN))
        return t_.clamp(t_min, t_max)

    def flow_times(self) -> list[float]:
        """Per-grid-point flow times ``t(u_i)`` for ``i in 0..num_steps``."""
        warped = self._warp(self._intervals)
        return [float(v) for v in warped]


class IdeogramV4Sampler(GenericSamplingPipeline):
    """Ideogram 4 sampler — ported logit-normal flow-match + asymmetric CFG."""

    pipeline: IdeogramV4Trainer

    def __init__(self, pipeline: IdeogramV4Trainer) -> None:
        super().__init__(pipeline)
        self._scheduler: _Ideogram4FlowSchedule | None = None
        # Post-patchify latent grid, stashed by _create_initial_noise and
        # consumed by denoise (img_shapes via the driver batch) + unpatchify.
        self._latent_h = 0
        self._latent_w = 0
        # Pixel resolution, stashed for the resolution-aware schedule.
        self._height = 0
        self._width = 0

    # ── Lazy scheduler (ported upstream logit-normal schedule) ───────────

    def _get_scheduler(self, num_steps: int) -> _Ideogram4FlowSchedule:
        # Resolution-dependent, so rebuild if the stashed resolution changed.
        if (
            self._scheduler is None
            or self._scheduler.num_steps != num_steps
            or (self._scheduler.height, self._scheduler.width)
            != (self._height, self._width)
        ):
            self._scheduler = _Ideogram4FlowSchedule(
                num_steps=num_steps,
                height=self._height,
                width=self._width,
            )
        return self._scheduler

    # ── Text encoding ────────────────────────────────────────────────────

    def encode_prompt(self, prompt: str) -> dict[str, Any]:
        """Encode the prompt's conditional text features for the cond branch.

        Calls the driver directly (not the trainer's cached path) so sample
        prompts never pollute the training text cache. The unconditional
        branch is the zeroed cond features, built at denoise time
        (asymmetric CFG through the same DiT), so only the cond pair is
        encoded here.
        """
        driver = self.pipeline.driver

        # f32 features — the denoise loop's precision contract (see denoise):
        # the bf16 DiT handles its own casting, exactly like the upstream
        # pipeline which feeds float32 llm_features.
        cond = driver.encode_text([prompt], torch.float32)
        return {
            "cond": (
                cond.embeddings.to(self.device),
                cond.attention_mask.to(self.device),
            ),
        }

    # ── Noise ────────────────────────────────────────────────────────────

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator
    ) -> Tensor:
        """Create raw sequence-space noise ``[1, S, 128]`` for the resolution.

        Raw randn ``[1, 32, h, w]`` -> ``patchify_to_seq`` -> ``[1, S, 128]``.
        Noise stays UN-normalized (training keeps noise raw; only image latents
        carry the latent-norm). The post-patchify grid
        ``(h//8//2, w//8//2)`` is stashed for the driver's packing and the
        final unpatchify.
        """
        self._height = height
        self._width = width
        lat_h = height // utils.VAE_SPATIAL_DOWNSCALE
        lat_w = width // utils.VAE_SPATIAL_DOWNSCALE
        self._latent_h = lat_h // utils.PATCH_FACTOR
        self._latent_w = lat_w // utils.PATCH_FACTOR

        noise = torch.randn(
            (1, utils.VAE_LATENT_CHANNELS, lat_h, lat_w),
            generator=generator,
            device=self.device,
        )
        return utils.patchify_to_seq(noise)

    # ── Denoise ──────────────────────────────────────────────────────────

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Tensor:
        """Euler flow-match loop with asymmetric CFG (ported from upstream).

        Returns a denormalized, unpatchified latent ``[1, 32, H, W]`` ready for
        :meth:`decode_latents`.

        The DiT forward runs WITHOUT autocast and the whole trajectory
        (latents, timesteps, text features, CFG combine, Euler update) stays in
        fp32 — see the precision contract in the loop body; autocast around
        this family's DiT collapses sampling to the conditional mean (GPU
        ablation, 2026-06-10). The model predicts velocity in NORMALIZED
        latent space (training normalizes image latents, keeps noise raw), so
        the trajectory stays in normalized space; ``denormalize_latents`` is
        applied once after the loop, before unpatchify/decode (matches upstream
        ``_decode``).
        """
        device = self.device
        driver = self.pipeline.driver
        transformer = self.pipeline.transformer
        scheduler = self._get_scheduler(num_steps)

        # ── PRECISION CONTRACT (GPU-validated, 2026-06-10) ──
        # NO autocast, f32 trajectory/timesteps/features — mirroring upstream's
        # loop exactly. A precision ablation (6 full denoise loops, same seed)
        # against an upstream-equal f32 reference proved:
        #   bf16 features:            cos(z_final, ref) = 1.0000  (harmless)
        #   bf16 z + bf16 timesteps:  cos = 0.97                  (harmless)
        #   torch.autocast(bf16):     cos = 0.32 -> flat/blank image (FATAL)
        # The vendored DiT keeps deliberate f32 islands (1e4-scaled t-sinusoids,
        # RoPE phases over positions offset by 65536, adaln modulation); autocast
        # force-downcasts those ops and collapses 20-step sampling to the
        # conditional mean. The bf16 model handles its own input casting (the
        # upstream pipeline feeds f32 into the same weights). ai-toolkit likewise
        # samples WITHOUT autocast. Training is unaffected by this contract: the
        # train-time autocast forward is a single (non-compounding) pass and is
        # validated end-to-end (LoRA renders in ComfyUI).
        cond_feats, cond_mask = prompt_embedding["cond"]
        cond_feats = cond_feats.float()
        do_cfg = guidance_scale > 1.0
        uncond_feats = zeroed_like_text(cond_feats)

        latents = noise.to(device=device, dtype=torch.float32)
        batch = {"latent_h": self._latent_h, "latent_w": self._latent_w}

        # Per-grid-point flow times t(u_i) in [~0, ~1]; the loop walks the
        # interval grid downward (i = num_steps-1 .. 0) exactly like upstream.
        times = scheduler.flow_times()  # length num_steps + 1

        self._ensure_transformer_on_device(transformer)
        with torch.no_grad():
            for loop_idx, i in enumerate(range(num_steps - 1, -1, -1), 1):
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Sampling {loop_idx}/{num_steps}")

                t_val = times[i + 1]
                s_val = times[i]
                delta = s_val - t_val
                # Feed the RAW flow time ``t_val`` to the DiT — exactly as
                # upstream ``pipeline_ideogram4.py::__call__`` does
                # (``t = schedule(step_intervals[i+1])`` passed straight to the
                # transformer's ``t=``). This matches the family's proven
                # convention: t=0 is noise, t=1 is data, velocity = data - noise
                # (driver.forward_pass is byte-exact vs upstream at raw ``t``).
                # The loop starts at ``t_val ~ 0`` (z is noise) and walks toward
                # ``t_val ~ 1`` (data), so the DiT always sees the CURRENT flow
                # position. forward_pass divides by NUM_TRAIN_TIMESTEPS to reach
                # the DiT's [0, 1]; emit the [0,1000] value here — do NOT
                # pre-divide (x1000 embedder guard), do NOT extra-x1000, and do
                # NOT invert to ``1 - t_val`` (that told the DiT "data" while z
                # was still noise — an inversion that contradicts forward_pass).
                ts = torch.full(
                    (latents.shape[0],),
                    t_val * NUM_TRAIN_TIMESTEPS,
                    device=device,
                    dtype=torch.float32,
                )

                # Direct forward — NO autocast (see precision contract above).
                cond_pred = driver.forward_pass(
                    latents, ts, (cond_feats, cond_mask), batch,
                )
                if do_cfg:
                    uncond_pred = driver.forward_pass(
                        latents, ts, (uncond_feats, cond_mask), batch,
                    )

                # CFG combine + Euler step in fp32.
                if do_cfg:
                    v = combine_asymmetric_cfg(cond_pred, uncond_pred, guidance_scale)
                else:
                    v = cond_pred.float()

                latents = latents + v * delta

        latents = utils.denormalize_latents(latents.float())
        return utils.unpatchify_from_seq(latents, self._latent_h, self._latent_w)

    # ── Decode ───────────────────────────────────────────────────────────

    def decode_latents(self, latents: Any) -> Image.Image:
        """VAE-decode an unpatchified latent ``[1, 32, H, W]`` to a PIL image.

        Uses the VAE's ``decode`` wrapper (delegates to ``decoder``), matching
        upstream ``_decode``'s ``autoencoder.decoder(z)``. Image is clamped to
        ``[-1, 1]`` then mapped to ``[0, 255]`` (upstream range).
        """
        vae = self.pipeline.vae
        with torch.no_grad():
            image = vae.decode(latents.to(vae.dtype))

        image = image.float().clamp(-1.0, 1.0)
        image = (image + 1.0) * 127.5
        image = image.round().to(torch.uint8)
        image = image.permute(0, 2, 3, 1).cpu().numpy()
        return Image.fromarray(image[0])
