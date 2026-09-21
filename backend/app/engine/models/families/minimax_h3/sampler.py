"""MiniMax-H3 sampler (plan row 2.10) — previews during training.

Production caller (DECISION-67 (a)): ``MiniMaxH3Trainer._create_sampler()``
returns this when ``sample_every_n_steps > 0``; ``pipeline_train.py`` then
calls ``should_sample(step)`` / ``generate_samples(step)`` every step.

The reverse loop runs on the INSTALLED ``diffusers.MiniMaxH3Scheduler`` in
H3's clock (``t = 1 − σ``, ``t = 1`` clean, data-ward velocity — the same
convention the trainer's ``add_noise`` / ``compute_target`` pin), CFG = 1,
with the trajectory held in fp32 and the DiT call entered with autocast
OFF (memory ``autocast-sampler-collapse-gotcha``: an autocast wrapper around
the DiT forward once collapsed sampling to the conditional mean). Audio is
co-denoised on its own clock — ``σ_a = remap(σ_v, shift_v → shift_a)`` — the
one ``u``, two clocks rule of research §3.2. The inference-only
audio-velocity scaling of research §3.8 is deliberately NOT ported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
import torch
from torch import Tensor

from app.engine.components.latents import LatentManager
from app.engine.core.sampling import GenericSamplingPipeline, SampleArtifact

if TYPE_CHECKING:
    from .trainer import MiniMaxH3Trainer

logger = structlog.get_logger(__name__)

_DEFAULT_SAMPLE_FRAMES = 107  # 17·6 + 5 — the definitions' documented default


def latent_frames_for(pixel_frames: int) -> int:
    """``17n + 5`` pixel frames → ``5n + 2`` latent frames (the visual VAE
    chunks 17 frames into 5 behind a 5-frame head; research §3.3)."""
    n, rem = divmod(int(pixel_frames) - 5, 17)
    if int(pixel_frames) < 5 or rem:
        raise ValueError(f"{pixel_frames} frames is not a legal H3 length (17n+5)")
    return 5 * n + 2


class MiniMaxH3Sampler(GenericSamplingPipeline):
    """fp32 reverse loop on the diffusers scheduler, video + stereo audio."""

    pipeline: MiniMaxH3Trainer
    needs_live_te = False  # prompts are served from the trainer's text cache

    def __init__(self, pipeline: MiniMaxH3Trainer) -> None:
        super().__init__(pipeline)
        self._latent_grid: tuple[int, int, int] | None = None
        self._sample_frames: int | None = None
        self._last_audio_latents: Tensor | None = None

    # ── Definition-derived facts ────────────────────────────────────────

    def _arch(self) -> dict[str, Any]:
        return dict(getattr(self.pipeline.definition, "architecture_params", {}) or {})

    def _model_dtype(self) -> torch.dtype:
        try:
            return next(self.pipeline.transformer.parameters()).dtype
        except (AttributeError, StopIteration, TypeError):
            return torch.float32

    def _audio_latents_for(self, pixel_frames: int) -> int:
        from .packing import audio_latent_num_frames  # the trainer's count (build_batch_extra)

        arch = self._arch()
        fps = float(arch.get("video.frame_rate", 24.0) or 24.0)
        rate = float(arch.get("audio.latent_rate", 40) or 40)
        return audio_latent_num_frames(pixel_frames, fps, rate)

    # ── Abstract hooks ──────────────────────────────────────────────────

    def encode_prompt(self, prompt: str) -> Any:
        """The trainer's CACHED text path: the 63 GB Qwen3-VL was released
        after pre-caching, and the sample prompts were warmed into the cache
        (`_pre_cache_text_embeddings`), so this never touches the encoder."""
        return self.pipeline.encode_text([prompt], self._model_dtype())

    def _create_initial_noise(self, width: int, height: int, generator: torch.Generator) -> Tensor:
        driver = self.pipeline.driver
        arch = self._arch()
        frames = self._effective_sample_frames(_DEFAULT_SAMPLE_FRAMES, driver.frame_rule())
        driver.assert_clip_frames(frames)  # the row-2.7 give-up, never padded
        vae_spatial = int(arch.get("video.vae_spatial", 16) or 16)
        channels = int(arch.get("transformer.in_channels", 24) or 24)
        lat_f = latent_frames_for(frames)
        lat_h, lat_w = max(height // vae_spatial, 1), max(width // vae_spatial, 1)
        self._latent_grid = (lat_f, lat_h, lat_w)
        self._sample_frames = frames
        return torch.randn(
            (1, channels, lat_f, lat_h, lat_w), generator=generator, device=self.device, dtype=torch.float32
        )

    @staticmethod
    def _advance(scheduler: Any, x: Tensor, velocity: Tensor, timestep: Tensor) -> Tensor:
        """ONE Euler step through the diffusers scheduler, in fp32: the
        precision-contract surface — the prediction and the sample are
        float32 when they reach ``scheduler.step``."""
        return scheduler.step(
            velocity.to(torch.float32), timestep, x.to(torch.float32), return_dict=False
        )[0]

    def _schedulers(self, num_steps: int) -> tuple[Any, Any]:
        """The video scheduler on ``video.sigma_shift`` and the audio one on
        the SAME ``u`` grid remapped to ``audio.sigma_shift`` (one draw, two
        clocks — exactly what the driver's ``audio_timestep`` derives)."""
        from diffusers import MiniMaxH3Scheduler

        from .schedule import remap_sigma

        settings = self.pipeline.driver._require_settings("sampler")
        video = MiniMaxH3Scheduler(shift=float(settings.sigma_shift_video))
        video.set_timesteps(int(num_steps) + 1, device=self.device)
        audio = MiniMaxH3Scheduler(shift=float(settings.sigma_shift_audio))
        audio.set_timesteps(
            sigmas=remap_sigma(video.sigmas.cpu(), float(settings.sigma_shift_video), float(settings.sigma_shift_audio)),
            device=self.device,
        )
        return video, audio

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Tensor:
        """The reverse loop: CFG = 1 (``guidance_scale`` is accepted for the
        base's signature and ignored — no unconditional branch), the joint
        forward through the driver (one packed sequence, ``[t_v, t_a]``),
        both streams stepped by their own scheduler in fp32."""
        driver = self.pipeline.driver
        driver._require_settings("sampler")  # refuses an unconfigured driver by name
        model_dtype = self._model_dtype()
        sched_v, sched_a = self._schedulers(num_steps)
        x_v = noise.to(device=self.device, dtype=torch.float32)
        # The audio arm is NOT gated on `train_audio` (plan row 3.5): that
        # switch zeroes the audio loss, the rows stay packed (row 3.2), so
        # the preview denoises the sequence the run trains on.
        arch = self._arch()
        frames = self._sample_frames or _DEFAULT_SAMPLE_FRAMES
        channels = int(arch.get("audio_vae.latent_channels", 32) or 32)
        gen = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
        x_a: Tensor | None = torch.randn(
            (x_v.shape[0], 2, channels, self._audio_latents_for(frames)),
            generator=gen, device=self.device, dtype=torch.float32,
        )

        total = int(sched_v.timesteps.numel())
        for i in range(total):
            if getattr(self.pipeline, "_log_writer", None):
                self.pipeline._log_writer.status(f"Sampling {i + 1}/{total}")
            t_v = sched_v.timesteps[i]
            batch: dict[str, Any] = {}
            if x_a is not None:
                batch["audio_noisy"] = x_a.to(model_dtype)
            # The DiT sees autocast OFF on every device type, whatever
            # context the caller entered — the fp32 trajectory contract.
            with (
                torch.no_grad(),
                torch.autocast("cuda", enabled=False),
                torch.autocast("cpu", enabled=False),
            ):
                v_v, v_a = driver.forward_pass(
                    x_v.to(model_dtype), t_v.reshape(1), prompt_embedding, batch, cfg_augment=False
                )
            x_v = self._advance(sched_v, x_v, v_v, t_v)
            if x_a is not None and v_a is not None:
                x_a = self._advance(sched_a, x_a, v_a, sched_a.timesteps[i])

        self._last_audio_latents = x_a
        return x_v

    def decode_latents(self, latents: Any) -> SampleArtifact:
        """Video through the pixel-adapted VAE (pixels back in ``[-1, 1]``),
        audio through the stereo helper — best-effort: an audio-only failure
        degrades to a silent clip, logged and counted, never a lost sample."""
        components = getattr(self.pipeline, "components", {}) or {}
        vae = components.get("vae") or getattr(self.pipeline, "vae", None)
        if vae is None:
            raise RuntimeError("minimax_h3 sampler: no visual VAE to decode with")
        with torch.no_grad():
            raw = LatentManager.denormalize_latents(latents.to(torch.float32), vae)
            decoded = vae.decode(raw.to(vae.dtype), return_dict=False)[0]
        frames = decoded[0].float().clamp(-1.0, 1.0)  # [C, F, H, W]

        audio = None
        audio_latents = self._last_audio_latents
        audio_vae = components.get("audio_vae")
        if audio_latents is not None and audio_vae is not None:
            try:
                from . import audio_latents as al

                arch = self._arch()
                moved = self._ensure_on_gpu(["audio_vae"])
                try:
                    with torch.no_grad():
                        wave = al.decode_stereo(audio_vae, audio_latents.to(audio_vae.dtype))
                finally:
                    self._offload_to_cpu(moved)
                audio = (wave[0].detach().float().cpu().clamp_(-1.0, 1.0), int(arch.get("audio.sampling_rate", 32000)))
            except Exception as exc:  # noqa: BLE001 - best-effort, surfaced
                count = getattr(self, "_audio_decode_failures", 0) + 1
                self._audio_decode_failures = count
                logger.warning(
                    "minimax_h3_audio_decode_failed",
                    error=str(exc),
                    occurrence=count,
                    hint="audio dropped — sample saved as a silent clip",
                )
                audio = None

        fps = float(self.config.get("sample_fps", self._arch().get("video.frame_rate", 24.0)) or 24.0)
        return SampleArtifact(frames=frames, audio=audio, fps=fps)
