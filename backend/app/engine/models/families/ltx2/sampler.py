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

from app.engine.components.latents import LatentManager
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

    def _audio_enabled(self) -> bool:
        """True when this run should denoise + decode a joint audio stream."""
        driver = self.pipeline.driver
        return (
            bool(getattr(driver, "train_audio", False))
            and getattr(driver, "audio_vae", None) is not None
            and getattr(driver, "vocoder", None) is not None
        )

    def _audio_num_frames(self) -> int:
        """Latent audio length for the sampled clip's duration.

        Mirrors ``LTX2Pipeline``: ``audio_latents_per_second = sample_rate /
        mel_hop / temporal_compression`` (16000/160/4 = 25), times the clip
        duration ``num_frames / fps``. Falls back to sane LTX-2 defaults when the
        audio VAE doesn't expose the ratios.
        """
        driver = self.pipeline.driver
        audio_vae = driver.audio_vae
        cfg = getattr(audio_vae, "config", None)
        sr = int(getattr(cfg, "sample_rate", driver.audio_sampling_rate) or 16000)
        hop = int(getattr(cfg, "mel_hop_length", 160) or 160)
        temporal = int(getattr(audio_vae, "temporal_compression_ratio", 4) or 4)
        num_frames = int(self.config.get("sample_num_frames", 25))
        fps = float(self.config.get("sample_fps", driver.frame_rate) or driver.frame_rate or 24.0)
        duration_s = num_frames / fps if fps > 0 else 0.0
        per_s = sr / hop / temporal if (hop and temporal) else 25.0
        return max(int(round(duration_s * per_s)), 1)

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

        Audio-on: a packed audio-latent stream is co-denoised on the SAME sigma
        schedule with ``isolate_modalities=False`` (the joint forward returns
        ``(video, audio)``), and the final audio latents are stashed on the
        driver for :meth:`_decode_audio`. Audio-off (or no vocoder): a single
        isolated zero audio token, mirroring the video-only training forward.
        """
        driver = self.pipeline.driver
        transformer = self.pipeline.transformer
        model_dtype = next(transformer.parameters()).dtype
        # No autocast here (fp32 trajectory), so the conditioning must already
        # be in the transformer's dtype.
        video_emb = driver._video_embeddings(prompt_embedding).to(model_dtype)
        f, h, w = driver._latent_grid()

        audio_on = self._audio_enabled()
        driver._last_audio_latents = None
        audio_x = None
        audio_emb = None
        audio_len = 1
        if audio_on:
            audio_len = self._audio_num_frames()
            gen = torch.Generator(device=self.device).manual_seed(seed + 1)
            audio_x = torch.randn(
                noise.shape[0], audio_len, driver.audio_in_channels,
                generator=gen, device=self.device, dtype=torch.float32,
            )
            audio_emb = driver._audio_embeddings(prompt_embedding, video_emb).to(model_dtype)

        # Classifier-free guidance: when guidance_scale > 1 we contrast the
        # conditional velocity against an UNCONDITIONAL one (empty/negative
        # prompt): v = v_u + scale * (v_c - v_u). Without it the preview is
        # effectively CFG=1 and massively under-shows the LoRA vs ComfyUI. The
        # negative prompt is warmed into the text cache by the trainer (the 12B
        # TE is offloaded by sample time); default "" is the standard uncond.
        cfg_on = guidance_scale is not None and float(guidance_scale) > 1.0
        video_emb_uncond = None
        audio_emb_uncond = None
        if cfg_on:
            neg_text = str(self.config.get("sample_negative_prompt", "") or "")
            neg_embedding = self.encode_prompt(neg_text)
            video_emb_uncond = driver._video_embeddings(neg_embedding).to(model_dtype)
            if audio_on:
                audio_emb_uncond = driver._audio_embeddings(
                    neg_embedding, video_emb_uncond
                ).to(model_dtype)

        from app.engine.strategies.sigma_schedule import shifted_sigmas
        shift = float(
            self.config.get("model_shift_fixed")
            or self.config.get("model_shift_max_shift")
            or 1.0
        )
        sigmas = shifted_sigmas(num_steps, shift, device=self.device)

        # One DiT forward at the current step → (v_video_fp32, v_audio|None).
        # Defined once (no loop-variable capture); called once per guidance
        # branch. Audio-on feeds the live audio latents + matching audio text
        # emb; audio-off uses the isolated dummy audio stream.
        def _velocity(
            xin: Tensor,
            ts: Tensor,
            audio_cur: Tensor | None,
            vemb: Tensor,
            aemb: Any,
        ) -> tuple[Tensor, Tensor | None]:
            if audio_on:
                a_in = audio_cur.to(model_dtype)
                a_emb = aemb
            else:
                a_in, a_emb = driver._dummy_audio_inputs(xin, vemb)
            with torch.no_grad():
                out = transformer(
                    hidden_states=xin,
                    audio_hidden_states=a_in,
                    encoder_hidden_states=vemb,
                    audio_encoder_hidden_states=a_emb,
                    timestep=ts,
                    sigma=ts,
                    num_frames=f,
                    height=h,
                    width=w,
                    fps=driver.frame_rate,
                    audio_num_frames=audio_len if audio_on else 1,
                    isolate_modalities=not audio_on,
                    return_dict=False,
                )
            if audio_on:
                return out[0].to(torch.float32), out[1].to(torch.float32)
            v = (out[0] if isinstance(out, (tuple, list)) else out).to(torch.float32)
            return v, None

        # Step the schedule in fp32; the per-step timestep is the sigma on the
        # [0, 1000] flow-match scale, passed RAW to the transformer (NOT ÷1000 —
        # only add_noise normalizes).
        x = noise.to(torch.float32)
        total_steps = len(sigmas) - 1
        for i in range(total_steps):
            # Per-step progress for the UI (byte-identical to the image
            # families' format — job_log.jsonl → LogTailer parses it).
            if getattr(self, "_log_writer", None):
                self._log_writer.status(f"Sampling {i + 1}/{total_steps}")
            dt = sigmas[i + 1] - sigmas[i]
            t_val = float(sigmas[i]) * 1000.0
            xin = x.to(model_dtype)
            ts = x.new_ones(x.shape[0]) * t_val  # fp32 [0, 1000]

            v_video, v_audio = _velocity(xin, ts, audio_x, video_emb, audio_emb)
            if cfg_on:
                v_video_u, v_audio_u = _velocity(
                    xin, ts, audio_x, video_emb_uncond, audio_emb_uncond
                )
                v_video = v_video_u + guidance_scale * (v_video - v_video_u)
                if audio_on:
                    v_audio = v_audio_u + guidance_scale * (v_audio - v_audio_u)

            x = x + dt * v_video
            if audio_on:
                audio_x = audio_x + dt * v_audio

        if audio_on:
            driver._last_audio_latents = audio_x

        return x

    def decode_latents(self, latents: Any) -> SampleArtifact:
        """Decode packed latents → video frames (+ optional audio waveform).

        Returns a :class:`SampleArtifact` so the base persists an mp4.  Audio is
        present only when the run trained the audio stream and a vocoder is
        loaded; otherwise ``audio=None`` (silent clip). Audio decode is
        best-effort: any failure degrades to a silent clip (the video sample is
        never lost to an audio-only error).
        """
        driver = self.pipeline.driver
        vae = self.pipeline.vae
        frames = self._decode_video(vae, driver, latents)

        audio = None
        if self._audio_enabled():
            try:
                audio = self._decode_audio(driver)
            except Exception as e:  # noqa: BLE001
                logger.warning("ltx2_audio_decode_failed", error=str(e))
                audio = None

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
        # The model emits latents in the VAE's NORMALIZED space ((z-mean)/std);
        # the decoder expects raw-scale latents. LTX-2's std≈0.15 per channel, so
        # skipping this fed ~6.7× too-large values to the VAE → pure-noise
        # samples. Denormalize (per-channel, on the unpacked [B,C,F,H,W] tensor,
        # exact inverse of LatentManager.normalize_latents used at encode time).
        unpacked = LatentManager.denormalize_latents(unpacked, vae)
        with torch.no_grad():
            decoded = vae.decode(unpacked.to(vae.dtype), return_dict=False)[0]
        # [B, C, F, H, W] → [C, F, H, W]
        return decoded[0].float().clamp(-1.0, 1.0)

    def _decode_audio(self, driver) -> tuple[Tensor, int] | None:
        """Audio latents → waveform: denormalize → unpack → audio VAE → vocoder.

        Returns ``(waveform, sample_rate)`` at the VOCODER's output rate (LTX-2
        upsamples the 16 kHz mel domain to ~24 kHz) so the mp4 muxer tags the
        AAC stream correctly. The audio VAE + vocoder are phased onto the GPU for
        the decode and pushed back to CPU afterwards (they were offloaded after
        pre-caching). Returns ``None`` when there are no audio latents.
        """
        from .audio_mel import denormalize_audio_latents, unpack_audio_latents

        audio_latents = getattr(driver, "_last_audio_latents", None)
        if audio_latents is None:
            return None
        audio_vae = driver.audio_vae
        vocoder = driver.vocoder
        if audio_vae is None or vocoder is None:
            return None

        cfg = getattr(audio_vae, "config", None)
        num_mel_bins = int(getattr(cfg, "mel_bins", 64) or 64)
        mel_comp = int(getattr(audio_vae, "mel_compression_ratio", 4) or 4)
        latent_mel_bins = max(num_mel_bins // mel_comp, 1)
        out_sr = int(
            getattr(getattr(vocoder, "config", None), "output_sampling_rate", 24000) or 24000
        )

        moved = self._ensure_on_gpu(["audio_vae", "vocoder"])
        try:
            lat = audio_latents.to(self.device)
            mean = getattr(audio_vae, "latents_mean", None)
            std = getattr(audio_vae, "latents_std", None)
            if mean is not None and std is not None:
                lat = denormalize_audio_latents(lat, mean, std)
            # Packed [B, L, C*M] → spectrogram latent [B, C, L, M].
            lat = unpack_audio_latents(lat, latent_mel_bins)
            with torch.no_grad():
                lat = lat.to(audio_vae.dtype)
                mel = audio_vae.decode(lat, return_dict=False)[0]
                wav = vocoder(mel)
        finally:
            self._offload_to_cpu(moved)

        if not isinstance(wav, Tensor):
            return None
        wav = wav.detach().float().cpu()
        if wav.ndim == 3:  # [B, C, N] → [C, N]
            wav = wav[0]
        return wav.clamp_(-1.0, 1.0), out_sr

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
