"""LTX 2.3 model driver — family-specific training behavior.

Implements ``IModelDriver`` for the LTX 2.3 joint audio + video DiT.

Flow-match contract (the silent-failure guard)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LTX-2 trains with flow matching on the ``[0, 1000]`` FlowMatchEuler scale:

    add_noise:  noisy = (t/1000) * noise + (1 - t/1000) * latents
    target:     v = noise - latents              (t-independent)
    forward:    transformer sees timestep = t / 1000  (normalized)

``add_noise`` is the contract surface proven by
``assert_flowmatch_timestep_contract`` — it must NOT apply an extra ×1000.

Joint audio + video
~~~~~~~~~~~~~~~~~~~~~
When ``train_audio`` is on, the SAME timestep ``t`` drives both modalities and
the loss is::

    loss = video_fm_loss + audio_weight * masked_audio_fm_loss

Audio is OPTIONAL: with ``train_audio=False`` the audio stream is omitted
entirely (no audio VAE / vocoder loaded, audio LoRA modules excluded).  With it
on, clips lacking audio carry ``audio_mask=0`` and their audio loss is forced to
zero (see :mod:`.audio`).
"""

from __future__ import annotations

from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver
from app.engine.core.text_encoding import TextEncoderOutput

from .audio import masked_audio_loss

logger = structlog.get_logger(__name__)

# Flow-match timestep scale (FlowMatchEulerDiscreteScheduler lives in [0, 1000]).
_FLOWMATCH_SCALE = 1000.0

# Video-stream LoRA targets — verified against ``LTX2VideoTransformer3DModel``'s
# real module tree (transformer_blocks.N.{attn1,attn2,ff}).
_VIDEO_LORA_TARGETS = (
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
    "ff.net.0.proj", "ff.net.2",
)

# Audio-stream + cross-modal LoRA targets — only when train_audio.  These use
# the ``audio_`` prefixed module names (audio_attn1/2, audio_ff) plus the two
# cross-attention bridges (audio_to_video_attn, video_to_audio_attn).  PEFT
# suffix matching keeps these disjoint from the video targets because the audio
# modules are ``audio_attn*`` / ``audio_ff`` (underscore, not a ``.`` boundary).
_AUDIO_LORA_TARGETS = (
    "audio_attn1.to_q", "audio_attn1.to_k", "audio_attn1.to_v",
    "audio_attn1.to_out.0",
    "audio_attn2.to_q", "audio_attn2.to_k", "audio_attn2.to_v",
    "audio_attn2.to_out.0",
    "audio_ff.net.0.proj", "audio_ff.net.2",
    "audio_to_video_attn.to_q", "audio_to_video_attn.to_k",
    "audio_to_video_attn.to_v", "audio_to_video_attn.to_out.0",
    "video_to_audio_attn.to_q", "video_to_audio_attn.to_k",
    "video_to_audio_attn.to_v", "video_to_audio_attn.to_out.0",
)


class Ltx2Driver(IModelDriver):
    """LTX 2.3 driver (joint audio + video)."""

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Assigned by assign_components()
        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.tokenizer: Any = None
        self.connectors: nn.Module | None = None
        self.audio_vae: nn.Module | None = None
        self.vocoder: nn.Module | None = None
        self._components: dict[str, Any] = {}

        # Architecture params (populated in assign_components).
        self.train_audio: bool = False
        self.audio_weight: float = 1.0
        self.patch_size: int = 1
        self.patch_size_t: int = 1
        self.te_max_length: int = 256
        self.frame_rate: float = 24.0
        # Transformer interface dims (for the joint forward's audio stream +
        # the connector→caption-projection contract).
        self.audio_in_channels: int = 128
        self.caption_channels: int = 3840
        self.audio_sampling_rate: int = 16000
        self._audio_mel = None  # lazily-built AudioMelExtractor (audio-on runs)
        self._latent_shape: tuple[int, int, int] | None = None  # (F, H, W)

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded LTX 2.3 components and cache architecture params."""
        self._components = components
        self.transformer = components["unet"]
        self.vae = components["vae"]
        self.text_encoder = components.get("text_encoder")
        self.tokenizer = components.get("tokenizer")
        self.connectors = components.get("connectors")
        self.audio_vae = components.get("audio_vae")
        self.vocoder = components.get("vocoder")

        arch = getattr(self.definition, "architecture_params", {}) or {}
        # Audio is gated by the run config; the definition exposes whether the
        # model *has* an audio stream, the trainer flips train_audio on per-run.
        self.train_audio = bool(arch.get("has_audio", False)) and (
            self.audio_vae is not None or arch.get("force_train_audio", False)
        )
        self.audio_weight = float(arch.get("audio.loss_weight", 1.0))
        self.te_max_length = int(arch.get("te.max_length", 256))
        self.frame_rate = float(arch.get("video.frame_rate", 24.0))
        self.audio_in_channels = int(arch.get("transformer.audio_in_channels", 128))
        self.caption_channels = int(arch.get("transformer.caption_channels", 3840))
        self.audio_sampling_rate = int(arch.get("audio.sampling_rate", 16000))

        # Patch sizes come from the transformer config (default 1×1 for LTX-2).
        cfg = getattr(self.transformer, "config", None)
        if cfg is not None:
            self.patch_size = int(getattr(cfg, "patch_size", 1))
            self.patch_size_t = int(getattr(cfg, "patch_size_t", 1))
        else:
            self.patch_size = int(arch.get("transformer.patch_size", 1))
            self.patch_size_t = int(arch.get("transformer.patch_size_t", 1))

        self.logger.info(
            "ltx2_config",
            train_audio=self.train_audio,
            audio_weight=self.audio_weight,
            patch_size=self.patch_size,
            patch_size_t=self.patch_size_t,
            frame_rate=self.frame_rate,
        )

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.transformer

    def get_text_encoders(self) -> dict[str, nn.Module]:
        result: dict[str, nn.Module] = {}
        if self.text_encoder is not None:
            result["text_encoder"] = self.text_encoder
        return result

    def get_lora_targets(self) -> list[str]:
        """LoRA targets — video stream always; audio + cross-modal when on.

        Returns trailing-suffix names for PEFT ``key.endswith(f".{target}")``
        matching.  When ``train_audio`` is False the audio sub-stream modules
        are excluded so no audio LoRA is created for an audio-off run.
        """
        targets = list(_VIDEO_LORA_TARGETS)
        if self.train_audio:
            targets.extend(_AUDIO_LORA_TARGETS)
        return targets

    def init_scheduler(self) -> Any:
        """LTX-2 uses flow matching — no external scheduler at train time."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """LTX-2 loads in bf16 (bf16 autocast, no GradScaler)."""
        return torch.bfloat16

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder (Gemma3) LoRA not supported for LTX-2."""
        return []

    # --- Phase 2: Text Encoding ---

    def encode_text(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Encode captions: Gemma3 hidden states → connectors → text emb pair.

        Runs the Gemma3 text encoder with ``output_hidden_states=True``, then
        feeds the per-layer hidden states through ``LTX2TextConnectors`` which
        returns ``(video_text_emb, audio_text_emb, ...)``.  The video embedding
        rides in ``embeddings`` (cached as ``te1``); the audio text embedding
        rides in the ``pooled`` slot (the ``te2`` half of the post-connector
        pair).  When audio is off, only the video embedding is consumed
        downstream and ``pooled`` stays ``None``.
        """
        input_ids, attention_mask = self._tokenize(captions)

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            # Stack per-layer hidden states → LTX2TextConnectors expects shape
            # ``(B, L, caption_channels, num_layers)`` (num_layers LAST), so we
            # stack on the trailing axis — NOT dim=0, which would feed the
            # connector a transposed tensor.
            hidden = torch.stack(outputs.hidden_states, dim=-1)
            video_emb, audio_emb = self._run_connectors(hidden, attention_mask)

        return TextEncoderOutput(
            embeddings=video_emb.to(dtype=dtype),
            attention_mask=attention_mask,
            pooled=audio_emb.to(dtype=dtype) if audio_emb is not None else None,
        )

    def _tokenize(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch of captions with the Gemma3 tokenizer."""
        enc = self.tokenizer(
            captions,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.te_max_length,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        return input_ids, attention_mask

    def _run_connectors(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run ``LTX2TextConnectors`` → (video_text_emb, audio_text_emb)."""
        if self.connectors is None:
            # No connectors (e.g. a fake / video-only smoke path) — pass the
            # last hidden state through as the video embedding.
            return hidden_states[-1], None
        # The connectors are a SECOND text-encoding stage. The pipeline's TE
        # GPU-move only relocates ``get_text_encoders()`` (the Gemma3 — the
        # connectors deliberately stay OUT of that dict so they aren't quantized
        # or LoRA'd as a text encoder), so the connectors can still be CPU-
        # resident here while the Gemma3 hidden states are on the GPU → device
        # mismatch in ``text_proj_in``. Co-locate just-in-time (a no-op once
        # resident) so the projection always runs on the hidden states' device.
        self.connectors.to(hidden_states.device)
        out = self.connectors(
            text_encoder_hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        # forward returns a tuple (video_text_emb, audio_text_emb, ...).
        video_emb = out[0]
        audio_emb = out[1] if len(out) > 1 else None
        return video_emb, audio_emb

    # --- Phase 5: Training Loop Hooks ---

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Flow-match noising on the [0, 1000] scale.

        ``noisy = (t/1000) * noise + (1 - t/1000) * latents``.  ``t`` is a
        FlowMatchEuler timestep in ``[0, 1000]``; we normalize by the scale
        here (NOT an extra ×1000).  Proven by
        ``assert_flowmatch_timestep_contract``.
        """
        frac = timesteps / _FLOWMATCH_SCALE
        while frac.ndim < latents.ndim:
            frac = frac.unsqueeze(-1)
        return frac * noise + (1.0 - frac) * latents

    def compute_target(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Flow-matching velocity target ``noise - latents`` (t-independent)."""
        return noise - latents

    def prepare_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Pack 5D video latents ``[B, C, F, H, W]`` → token sequence ``[B, L, D]``.

        Mirrors ``LTX2Pipeline._pack_latents`` exactly: patchify by
        ``patch_size_t`` (temporal) and ``patch_size`` (spatial), permute, and
        collapse the patch dims into the channel dim.  Records the unpacked
        ``(F, H, W)`` so the sampler can unpack symmetrically.
        """
        packed = self._pack_latents(latents, self.patch_size, self.patch_size_t)
        # Record the post-patch latent grid (F, H, W) so forward_pass can supply
        # RoPE coords. A still image arrives 4D ([B, C, H, W]) → implicit F=1
        # (the latent cache squeezes the dummy frame dim for stills); record it
        # too so single-image runs get correct coordinates.
        if latents.ndim == 5:
            _, _, f, h, w = latents.shape
        else:
            _, _, h, w = latents.shape
            f = 1
        self._latent_shape = (
            f // self.patch_size_t,
            h // self.patch_size,
            w // self.patch_size,
        )
        return packed

    @staticmethod
    def _pack_latents(
        latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1,
    ) -> torch.Tensor:
        """Verbatim port of ``LTX2Pipeline._pack_latents`` (diffusers 0.38.0).

        [B, C, F, H, W] → [B, F/p_t * H/p * W/p, C * p_t * p * p].
        Handles the 4D still-image case (F implicitly 1) by unsqueezing.
        """
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)  # [B, C, 1, H, W]
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_f = num_frames // patch_size_t
        post_h = height // patch_size
        post_w = width // patch_size
        latents = latents.reshape(
            batch_size, -1, post_f, patch_size_t,
            post_h, patch_size, post_w, patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """``LTX2VideoTransformer3DModel`` forward → video velocity prediction.

        The real model is a JOINT audio+video transformer: ``audio_hidden_states``
        and ``audio_encoder_hidden_states`` are required, the timestep is the raw
        flow-match ``[0, 1000]`` value (NOT normalized — only ``add_noise``
        divides by the scale), RoPE needs the latent ``num_frames/height/width``
        + ``fps``, and it returns a ``(video, audio)`` tuple.

        VIDEO-ONLY (``train_audio`` off, or a clip with no audio latents): feed a
        single zero audio token and set ``isolate_modalities=True`` so the audio
        stream cannot influence the video prediction; keep only the video output.

        AUDIO-ON (``train_audio`` and ``batch["audio_clean"]`` present): noise the
        clean audio latents on the SAME timestep, run the joint forward with
        ``isolate_modalities=False``, and stash the audio prediction/target/mask
        into ``batch`` for :meth:`Ltx2Trainer._compute_step_loss` to combine.
        """
        video_emb = self._video_embeddings(text_embeddings)
        f, h, w = self._latent_grid()
        fps = self._batch_frame_rate(batch)

        audio_clean = batch.get("audio_clean") if self.train_audio else None

        if audio_clean is None:
            # Video-only: minimal isolated dummy audio.
            audio_h, audio_emb = self._dummy_audio_inputs(noisy_input, video_emb)
            output = self.transformer(
                hidden_states=noisy_input,
                audio_hidden_states=audio_h,
                encoder_hidden_states=video_emb,
                audio_encoder_hidden_states=audio_emb,
                timestep=timesteps,  # raw [0, 1000] flow-match scale
                sigma=timesteps,  # LTX-2.3 prompt modulation (harmless when unused)
                num_frames=f,
                height=h,
                width=w,
                fps=fps,
                audio_num_frames=1,
                isolate_modalities=True,  # no audio↔video leakage
                return_dict=False,
            )
            return output[0] if isinstance(output, (tuple, list)) else output

        # Joint audio + video: noise the audio stream on the shared timestep.
        audio_clean = audio_clean.to(device=noisy_input.device, dtype=noisy_input.dtype)
        audio_emb = self._audio_embeddings(text_embeddings, video_emb)
        audio_noise = torch.randn_like(audio_clean)
        audio_noisy = self.add_noise(audio_clean, audio_noise, timesteps)
        audio_target = self.compute_target(audio_clean, audio_noise, timesteps)

        output = self.transformer(
            hidden_states=noisy_input,
            audio_hidden_states=audio_noisy,
            encoder_hidden_states=video_emb,
            audio_encoder_hidden_states=audio_emb,
            timestep=timesteps,
            sigma=timesteps,
            num_frames=f,
            height=h,
            width=w,
            fps=fps,
            audio_num_frames=audio_noisy.shape[1],
            isolate_modalities=False,  # joint: audio↔video cross-attention on
            return_dict=False,
        )
        video_pred, audio_pred = output[0], output[1]
        # Hand the audio terms to the trainer's joint-loss step (reads batch[...]).
        batch["audio_pred"] = audio_pred
        batch["audio_target"] = audio_target
        if batch.get("audio_mask") is None:
            batch["audio_mask"] = torch.ones(
                audio_clean.shape[0], device=audio_clean.device,
            )
        return video_pred

    @staticmethod
    def _audio_embeddings(text_embeddings: Any, video_emb: torch.Tensor) -> torch.Tensor:
        """Audio text embedding (connector's 2nd output, cached in ``pooled``).

        Falls back to a zero tensor shaped like the video embedding when absent
        so a malformed cache degrades to "no audio prompt" rather than crashing.
        """
        pooled = getattr(text_embeddings, "pooled", None)
        return pooled if pooled is not None else torch.zeros_like(video_emb)

    def encode_audio_clean(
        self, waveform: torch.Tensor, sample_rate: int,
    ) -> torch.Tensor:
        """Waveform ``[B, C, N]`` → clean (packed + normalized) audio latents.

        The flow-match TARGET for the audio stream.  Builds the log-mel
        spectrogram with the exact LTX-2 transform, VAE-encodes, packs, and
        normalizes (see :mod:`.audio_mel`).  Used by the data/caching layer.
        """
        from .audio_mel import AudioMelExtractor, encode_clean_audio_latents

        if self._audio_mel is None:
            self._audio_mel = AudioMelExtractor(
                sample_rate=self.audio_sampling_rate,
            ).to(self.device)
        mel = self._audio_mel.waveform_to_mel(
            waveform.to(self.device), sample_rate,
        )
        # The mel transform emits fp32; the audio VAE may be bf16/fp16. Match its
        # dtype so ``audio_vae.encode`` doesn't hit a CUDA mat-mul dtype mismatch.
        vae_dtype = getattr(self.audio_vae, "dtype", None)
        if vae_dtype is not None:
            mel = mel.to(dtype=vae_dtype)
        return encode_clean_audio_latents(self.audio_vae, mel)

    def _latent_grid(self) -> tuple[int, int, int]:
        """Post-patch latent ``(F, H, W)`` recorded by :meth:`prepare_latents`.

        Used to build the transformer's RoPE coordinates.  ``prepare_latents``
        runs before every forward (training) / before denoise (sampling), so
        the shape reflects the current batch / sample resolution.
        """
        if self._latent_shape is None:
            raise RuntimeError(
                "prepare_latents() must run before forward_pass() — it records "
                "the latent (F, H, W) the transformer needs for RoPE coords."
            )
        return self._latent_shape

    def _dummy_audio_inputs(
        self, hidden_ref: torch.Tensor, video_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Minimal zero audio stream for video-only training/sampling.

        The joint transformer requires ``audio_hidden_states`` (``[B, n, audio_in_channels]``
        → ``audio_proj_in``) and ``audio_encoder_hidden_states``
        (``[B, l, caption_channels]`` → ``audio_caption_projection``).  With
        ``isolate_modalities=True`` the audio stream is decoupled from video, so
        a single zero token in each suffices (cheap; output discarded).
        """
        b = hidden_ref.shape[0]
        audio_h = torch.zeros(
            b, 1, self.audio_in_channels,
            dtype=hidden_ref.dtype, device=hidden_ref.device,
        )
        audio_emb = torch.zeros(
            b, 1, self.caption_channels,
            dtype=video_emb.dtype, device=hidden_ref.device,
        )
        return audio_h, audio_emb

    def compute_loss(
        self,
        video_pred: torch.Tensor,
        video_target: torch.Tensor,
        batch: dict[str, Any],
        *,
        audio_pred: torch.Tensor | None = None,
        audio_target: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Joint flow-match loss ``video_fm + audio_weight * masked_audio_fm``.

        The video term is a plain MSE over the full batch (video loss always
        flows, for every item — images included).  The audio term is masked:
        absent-audio and image items (mask=0) contribute ZERO, computed via
        :func:`.audio.masked_audio_loss`.  When ``train_audio`` is off (no audio
        tensors supplied) the audio term is omitted and the result is the video
        loss alone.

        Audio noise/target share the same timestep as video (generated by the
        trainer in the same step), so this function only combines the two.
        """
        video_loss = torch.nn.functional.mse_loss(
            video_pred.float(), video_target.float(),
        )

        if not self.train_audio or audio_pred is None or audio_target is None:
            return video_loss

        if audio_mask is None:
            audio_mask = torch.ones(
                audio_pred.shape[0], device=audio_pred.device,
            )
        a_loss = masked_audio_loss(audio_pred, audio_target, audio_mask)
        return video_loss + self.audio_weight * a_loss

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self):
        from .saver import Ltx2Saver

        return Ltx2Saver()

    def get_save_metadata(self) -> dict[str, str]:
        return {"modelspec.architecture": "ltx-2.3"}

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """LTX-2 block topology: a single stack of transformer_blocks."""
        topology: list[dict[str, Any]] = []
        model = self.get_primary_model()
        if model is not None:
            blocks = getattr(model, "transformer_blocks", None)
            if blocks is not None:
                topology.append({
                    "name": "transformer_blocks",
                    "attr_path": "transformer_blocks",
                    "count": len(blocks),
                    "approx_vram_mb": 480,
                })
        return topology

    # --- Internal helpers ---

    @staticmethod
    def _video_embeddings(text_embeddings: Any) -> torch.Tensor:
        """Extract the video text embedding from a TextEncoderOutput or tensor."""
        if hasattr(text_embeddings, "embeddings"):
            return text_embeddings.embeddings
        return text_embeddings

    def _batch_frame_rate(self, batch: dict[str, Any]) -> float:
        """Resolve the per-item fps for frame_rate conditioning.

        Uses the batch's ``target_fps`` (set by the video data pipeline) when
        present, else the definition default.
        """
        fps = batch.get("target_fps") if isinstance(batch, dict) else None
        if fps is None:
            return self.frame_rate
        if isinstance(fps, (list, tuple)) and fps:
            return float(fps[0])
        try:
            return float(fps)
        except (TypeError, ValueError):
            return self.frame_rate
