"""Microsoft Lens in-training sampler.

Reproduces the ``LensPipeline`` denoise path using the trainer's
already-loaded components (DiT, FLUX.2 VAE, GPT-OSS text encoder, scheduler).

Family specifics that shape this sampler:

* **Sequence-space latents** ``[B, S, 128]`` (``S = latent_h * latent_w``,
  ``128 = 32 VAE channels * 2*2 patchify``). The denoise loop reuses the
  driver's validated :meth:`forward_pass` so the 4-layer text-feature split,
  ``encoder_hidden_states_mask`` and ``img_shapes`` are handled identically to
  training.
* **Flow-matching** with ``FlowMatchEulerDiscreteScheduler`` configured exactly
  as ``microsoft/Lens-Base`` ships it (``shift=3.0``, dynamic shifting,
  exponential time-shift). The DiT consumes timesteps in ``[0, 1]``, so the
  scheduler's ``[0, 1000]`` timesteps are divided by 1000 before each forward.
* **Standard CFG** (two-pass): Lens-Base is a non-distilled base model, so we
  run a conditional and an unconditional pass and combine with
  ``uncond + scale * (cond - uncond)``.
* **BN (de)normalization** in sequence space with the VAE's running stats,
  applied after the loop, before unpatchifying for VAE decode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import torch
from PIL import Image
from torch import Tensor

from app.engine.core.sampling import GenericSamplingPipeline
from app.engine.models.families.microsoft_lens import utils

if TYPE_CHECKING:
    from .trainer import MicrosoftLensTrainer

logger = structlog.get_logger(__name__)

# VAE spatial compression (8x) and the 2x2 latent patchify factor. Together
# they map a pixel resolution to the post-patchify latent grid: a side of
# ``px`` pixels becomes ``px // (VAE_SPATIAL_DOWNSCALE * PATCH_FACTOR)`` tokens.
VAE_SPATIAL_DOWNSCALE = 8
PATCH_FACTOR = 2
VAE_LATENT_CHANNELS = 32


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Linear ``mu`` interpolation for dynamic shifting (diffusers convention).

    Matches ``Lens-Base``'s scheduler config (base/max image_seq_len 256/4096,
    base/max shift 0.5/1.15): ``mu`` rises linearly from ``base_shift`` at
    ``base_seq_len`` to ``max_shift`` at ``max_seq_len``.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


class MicrosoftLensSampler(GenericSamplingPipeline):
    """Microsoft Lens sampler — Euler flow-matching + two-pass CFG."""

    pipeline: MicrosoftLensTrainer

    # ``encode_prompt`` below calls ``driver.encode_text`` directly every
    # round (not the trainer's cached path), so the base sampler must
    # bracket the text encoder to GPU/CPU around it.
    needs_live_te = True

    def __init__(self, pipeline: MicrosoftLensTrainer) -> None:
        super().__init__(pipeline)
        self._scheduler = None
        # Post-patchify latent grid, stashed by _create_initial_noise and
        # consumed by denoise (img_shapes) + unpatchify.
        self._latent_h = 0
        self._latent_w = 0

    # ── Lazy scheduler (definition-sourced; matches Lens-Base) ───────────

    def _arch(self) -> dict:
        defn = getattr(self.pipeline, "definition", None)
        return getattr(defn, "architecture_params", {}) or {}

    def _get_scheduler(self):
        """Build the sampling scheduler from the definition's ``scheduler.*``
        architecture_params (W3-1 plumbing).

        The values are Lens-Base's own ``scheduler_config.json`` (shipped in
        ``lens_base.yaml``: shift 3.0, dynamic shifting, exponential time-shift,
        1000 train timesteps, base/max shift 0.5/1.15, base/max image_seq_len
        256/4096). The constant fallbacks below are byte-identical and used
        only if a definition omits a key.
        """
        if self._scheduler is None:
            from diffusers import FlowMatchEulerDiscreteScheduler

            arch = self._arch()
            self._scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=int(
                    arch.get("scheduler.num_train_timesteps", 1000),
                ),
                shift=float(arch.get("scheduler.shift", 3.0)),
                use_dynamic_shifting=bool(
                    arch.get("scheduler.use_dynamic_shifting", True),
                ),
                base_shift=float(arch.get("scheduler.base_shift", 0.5)),
                max_shift=float(arch.get("scheduler.max_shift", 1.15)),
                base_image_seq_len=int(
                    arch.get("scheduler.base_image_seq_len", 256),
                ),
                max_image_seq_len=int(
                    arch.get("scheduler.max_image_seq_len", 4096),
                ),
                time_shift_type=str(
                    arch.get("scheduler.time_shift_type", "exponential"),
                ),
            )
        return self._scheduler

    # ── Text encoding ────────────────────────────────────────────────────

    def encode_prompt(self, prompt: str) -> dict[str, Any]:
        """Encode the prompt for CFG.

        Calls the driver directly (not the trainer's cached path) so sample
        prompts never pollute the training text cache. The base sampler has
        already moved the text encoder to the sampling device.

        The empty-string unconditional embedding is encoded LAZILY in
        ``denoise()`` — only when ``guidance_scale > 1`` actually engages
        the two-pass CFG combine — instead of unconditionally here, which
        used to pay a full extra text-encoder forward every sampling round
        even when CFG was off (``guidance_scale <= 1``, e.g. distilled/
        turbo-style sampling with no negative pass at all).
        """
        driver = self.pipeline.driver
        dtype = next(self.pipeline.transformer.parameters()).dtype

        cond = driver.encode_text([prompt], dtype)
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
        """Create sequence-space noise ``[1, S, 128]`` for the given resolution.

        VAE downscale = 8, patch factor = 2 → grid is ``(H/16) x (W/16)``.
        Channels: 32 VAE × 4 (2×2 patch) = 128.
        """
        lat_h = height // VAE_SPATIAL_DOWNSCALE
        lat_w = width // VAE_SPATIAL_DOWNSCALE
        self._latent_h = lat_h // PATCH_FACTOR
        self._latent_w = lat_w // PATCH_FACTOR

        noise = torch.randn(
            (1, VAE_LATENT_CHANNELS, lat_h, lat_w),
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
        """Euler flow-matching loop with two-pass CFG.

        Returns a BN-denormalised, unpatchified latent ``[1, 32, H, W]`` ready
        for :meth:`decode_latents`.

        The DiT forward runs under the **same autocast as training**
        (``pipeline.autocast_dtype``/``use_amp``) and the CFG combine + Euler
        step run in fp32. This matters once the LoRA is non-trivial: the model
        learns its velocities under autocast, so sampling the trained adapter
        in a different precision regime (or accumulating Euler steps in bf16)
        diverges into noise — even though the near-zero step-0 adapter and the
        robust base tolerate it, and the same saved weights render correctly in
        ComfyUI (which runs the model in a consistent dtype).
        """
        device = self.device
        driver = self.pipeline.driver
        transformer = self.pipeline.transformer
        vae = self.pipeline.vae
        dtype = next(transformer.parameters()).dtype
        scheduler = self._get_scheduler()

        # Mirror the training forward's precision regime exactly.
        amp_dtype = getattr(self.pipeline, "autocast_dtype", dtype)
        use_amp = getattr(self.pipeline, "use_amp", device.type == "cuda")

        cond_stacked, cond_mask = prompt_embedding["cond"]
        do_cfg = guidance_scale > 1.0
        uncond_stacked = uncond_mask = None
        if do_cfg:
            # T10 moved this encode out of encode_prompt() (Phase 1, inside
            # _sample_single's needs_live_te bracket) so a CFG-off round never
            # pays for it. But denoise() runs in Phase 2, AFTER Phase 1 has
            # already offloaded the text encoder back to CPU — a live driver
            # forward here needs its OWN bracket, or encode_text moves its
            # cuda inputs against a CPU-resident module and raises a device
            # mismatch on every CFG-on round (the default: guidance_scale=3.5).
            te_moved = self._ensure_on_gpu(["text_encoder"])
            uncond = driver.encode_text([""], dtype)
            uncond_stacked = uncond.embeddings.to(device)
            uncond_mask = uncond.attention_mask.to(device)
            self._offload_to_cpu(te_moved)

        latents = noise.to(device=device, dtype=dtype)
        batch = {"latent_h": self._latent_h, "latent_w": self._latent_w}

        # Timestep schedule (dynamic shifting on the image sequence length).
        # Definition-sourced base/max shift + seq-len (W3-1) — byte-identical
        # fallbacks keep the pipeline's calculate_shift math unchanged.
        arch = self._arch()
        image_seq_len = latents.shape[1]
        mu = _calculate_shift(
            image_seq_len,
            base_seq_len=int(arch.get("scheduler.base_image_seq_len", 256)),
            max_seq_len=int(arch.get("scheduler.max_image_seq_len", 4096)),
            base_shift=float(arch.get("scheduler.base_shift", 0.5)),
            max_shift=float(arch.get("scheduler.max_shift", 1.15)),
        )
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        scheduler.set_timesteps(
            num_inference_steps=num_steps, device=device, sigmas=sigmas, mu=mu,
        )
        timesteps = scheduler.timesteps

        self._ensure_transformer_on_device(transformer)
        with torch.no_grad():
            total = len(timesteps)
            for i, t in enumerate(timesteps, 1):
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Sampling {i}/{total}")
                # Pass the scheduler's [0, 1000] timestep in the shared trainer
                # convention; driver.forward_pass divides to the DiT's [0, 1].
                ts = t.expand(latents.shape[0]).to(device=device, dtype=dtype)

                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp,
                ):
                    cond_pred = driver.forward_pass(
                        latents, ts, (cond_stacked, cond_mask), batch,
                    )
                    if do_cfg:
                        uncond_pred = driver.forward_pass(
                            latents, ts, (uncond_stacked, uncond_mask), batch,
                        )

                # CFG combine + Euler step in fp32 (outside autocast) for
                # numerical stability across the denoising trajectory.
                if do_cfg:
                    pred = uncond_pred.float() + guidance_scale * (
                        cond_pred.float() - uncond_pred.float()
                    )
                else:
                    pred = cond_pred.float()

                latents = scheduler.step(
                    pred, t, latents.float(), return_dict=False,
                )[0].to(dtype)

        latents = utils.bn_denormalize_seq(latents, vae)
        return utils.unpatchify_from_seq(latents, self._latent_h, self._latent_w)

    # ── Decode ───────────────────────────────────────────────────────────

    def decode_latents(self, latents: Any) -> Image.Image:
        """VAE-decode an unpatchified latent ``[1, 32, H, W]`` to a PIL image."""
        vae = self.pipeline.vae
        with torch.no_grad():
            image = vae.decode(latents.to(vae.dtype), return_dict=False)[0]

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return Image.fromarray((image[0] * 255).round().astype("uint8"))
