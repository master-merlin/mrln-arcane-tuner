"""LTX 2.3 sampler — fp32 flow-match denoise → video (+ optional audio) clip.

Reproduces the ``LTX2Pipeline`` denoise behaviour using the trainer's
already-loaded components.  Two precision contracts are honoured:

1. **Flow-match [0, 1000] scale.** The transformer sees ``t / 1000``; the Euler
   sigma math runs in fp32.
2. **No autocast collapse.** The denoise TRAJECTORY (sigma math + latent
   accumulation) runs in fp32 with NO autocast around the loop — only the
   transformer forward may use the model dtype.  This is the surface proven by
   ``assert_no_autocast_collapse``: :meth:`build_denoise` returns a closure that
   integrates a known velocity field in fp32.

Audio: when the run trained the audio stream, the audio latents are decoded via
the ``LTX2Vocoder`` (lazy) and returned in the :class:`SampleArtifact`'s
``audio`` field; otherwise ``audio=None`` and a silent mp4 is written.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog
import torch
from torch import Tensor

from app.engine.core.sampling import GenericSamplingPipeline, SampleArtifact

if TYPE_CHECKING:
    from .trainer import Ltx2Trainer

logger = structlog.get_logger(__name__)


class Ltx2Sampler(GenericSamplingPipeline):
    """LTX 2.3 sampler — fp32 Euler flow-match, video + optional audio."""

    pipeline: "Ltx2Trainer"

    def __init__(self, pipeline: "Ltx2Trainer") -> None:
        super().__init__(pipeline)

    # ── fp32 Euler integration (the precision-contract surface) ──────────

    @staticmethod
    def euler_denoise(
        transformer: Callable[..., Tensor],
        x0: Tensor,
        sigmas: Tensor,
    ) -> Tensor:
        """Integrate ``dx/dσ = v(x)`` with forward Euler, entirely in fp32.

        ``sigmas`` is a 1-D descending schedule (e.g. 1 → 0).  The latent
        accumulation and the sigma deltas are computed in fp32 with NO autocast
        wrapper.  ``transformer`` supplies the velocity ``v(x)`` (in real
        sampling this is the DiT forward; in the precision test it is the
        ``LinearVelocityFakeTransformer``).

        Returning the fp32 trajectory endpoint is what lets
        ``assert_no_autocast_collapse`` pin the contract: a bf16/autocast loop
        would drift past the tight tolerance.
        """
        x = x0.to(torch.float32)
        s = sigmas.to(torch.float32)
        for i in range(len(s) - 1):
            dt = (s[i + 1] - s[i])
            v = transformer(x).to(torch.float32)
            x = x + dt * v
        return x

    def build_denoise(
        self, transformer: Callable[..., Tensor],
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Return a ``(x0, sigmas) -> endpoint`` closure for the contract test.

        Binds a velocity-providing ``transformer`` to :meth:`euler_denoise`.
        The precision test passes a ``LinearVelocityFakeTransformer`` so the
        endpoint has a known fp32 analytic reference.
        """
        def _denoise(x0: Tensor, sigmas: Tensor) -> Tensor:
            return self.euler_denoise(transformer, x0, sigmas)

        return _denoise

    # ── Abstract hooks ───────────────────────────────────────────────────

    def encode_prompt(self, prompt: str) -> Any:
        """Encode the prompt via the trainer's CACHED text path.

        Must call ``trainer.encode_text`` (NOT ``trainer.driver.encode_text``):
        the 12B Gemma3 is offloaded after pre-caching, so the driver's encoder
        is ``None`` at sample time — calling it crashed sampling with
        "'NoneType' object is not callable". The sample prompts are warmed into
        ``trainer.text_cache`` during ``_pre_cache_text_embeddings``, so the
        cached path serves them without reloading the encoder. (Same pattern as
        flux1/flux2/sdxl samplers.)
        """
        trainer = self.pipeline
        dtype = next(trainer.transformer.parameters()).dtype
        return trainer.encode_text([prompt], dtype)

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator,
    ) -> Tensor:
        """Create packed 5D video noise for the LTX-2 transformer.

        Spatial 32× downscale + temporal 8× → latent grid
        ``(F-1)/8 + 1`` frames × ``H/32`` × ``W/32``, then packed by patch size.
        """
        driver = self.pipeline.driver
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        vae_sp = int(arch.get("video.vae_spatial", 32))
        vae_t = int(arch.get("video.vae_temporal", 8))
        num_channels = int(arch.get("transformer.in_channels", 128))
        num_frames = int(self.config.get("sample_num_frames", 25))  # 8n+1

        lat_h = height // vae_sp
        lat_w = width // vae_sp
        lat_f = (num_frames - 1) // vae_t + 1

        latents = torch.randn(
            (1, num_channels, lat_f, lat_h, lat_w),
            generator=generator, device=self.device,
        )
        return driver.prepare_latents(latents)

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Any:
        """Run the fp32 flow-match denoise loop with the real DiT forward.

        Builds a descending sigma schedule, then integrates in fp32 via
        :meth:`euler_denoise`.  Only the transformer forward uses the model
        dtype; the trajectory accumulation stays fp32 (no autocast wrapper).
        """
        driver = self.pipeline.driver
        transformer = self.pipeline.transformer
        model_dtype = next(transformer.parameters()).dtype
        # No autocast here (fp32 trajectory), so the conditioning must already
        # be in the transformer's dtype.
        video_emb = driver._video_embeddings(prompt_embedding).to(model_dtype)
        f, h, w = driver._latent_grid()

        sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)

        # Step the schedule in fp32; the per-step timestep is the sigma on the
        # [0, 1000] flow-match scale, passed RAW to the transformer (NOT ÷1000 —
        # only add_noise normalizes). Video-only: a single zero audio token +
        # isolate_modalities, mirroring forward_pass.
        x = noise.to(torch.float32)
        for i in range(len(sigmas) - 1):
            dt = sigmas[i + 1] - sigmas[i]
            t_val = float(sigmas[i]) * 1000.0
            xin = x.to(model_dtype)
            ts = x.new_ones(x.shape[0]) * t_val  # fp32 [0, 1000]
            audio_h, audio_emb = driver._dummy_audio_inputs(xin, video_emb)
            with torch.no_grad():
                out = transformer(
                    hidden_states=xin,
                    audio_hidden_states=audio_h,
                    encoder_hidden_states=video_emb,
                    audio_encoder_hidden_states=audio_emb,
                    timestep=ts,
                    sigma=ts,
                    num_frames=f,
                    height=h,
                    width=w,
                    fps=driver.frame_rate,
                    audio_num_frames=1,
                    isolate_modalities=True,
                    return_dict=False,
                )
            v = (out[0] if isinstance(out, (tuple, list)) else out).to(torch.float32)
            x = x + dt * v

        return x

    def decode_latents(self, latents: Any) -> SampleArtifact:
        """Decode packed latents → video frames (+ optional audio waveform).

        Returns a :class:`SampleArtifact` so the base persists an mp4.  Audio is
        present only when the run trained the audio stream and a vocoder is
        loaded; otherwise ``audio=None`` (silent clip).
        """
        driver = self.pipeline.driver
        vae = self.pipeline.vae
        frames = self._decode_video(vae, driver, latents)

        audio = None
        if driver.train_audio and driver.vocoder is not None:
            audio = self._decode_audio(driver)

        fps = float(self.config.get("sample_fps", driver.frame_rate))
        return SampleArtifact(frames=frames, audio=audio, fps=fps)

    # ── Decode helpers ───────────────────────────────────────────────────

    def _decode_video(self, vae, driver, latents: Tensor) -> Tensor:
        """Unpack + VAE-decode latents → ``[C, F, H, W]`` float in [-1, 1]."""
        shape = driver._latent_shape
        if shape is not None:
            f, h, w = shape
            unpacked = self._unpack_latents(
                latents, f, h, w, driver.patch_size, driver.patch_size_t,
            )
        else:
            unpacked = latents
        with torch.no_grad():
            decoded = vae.decode(unpacked.to(vae.dtype), return_dict=False)[0]
        # [B, C, F, H, W] → [C, F, H, W]
        return decoded[0].float().clamp(-1.0, 1.0)

    def _decode_audio(self, driver) -> Tensor | None:
        """Vocoder-decode audio latents → waveform (best-effort, lazy)."""
        audio_latents = getattr(driver, "_last_audio_latents", None)
        if audio_latents is None:
            return None
        with torch.no_grad():
            wav = driver.vocoder(audio_latents)
        return wav.float() if isinstance(wav, Tensor) else None

    @staticmethod
    def _unpack_latents(
        latents: Tensor,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ) -> Tensor:
        """Inverse of ``_pack_latents`` — [B, L, D] → [B, C, F, H, W]."""
        batch_size = latents.shape[0]
        latents = latents.reshape(
            batch_size, num_frames, height, width, -1,
            patch_size_t, patch_size, patch_size,
        )
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7)
        latents = latents.reshape(
            batch_size, -1,
            num_frames * patch_size_t,
            height * patch_size,
            width * patch_size,
        )
        return latents
