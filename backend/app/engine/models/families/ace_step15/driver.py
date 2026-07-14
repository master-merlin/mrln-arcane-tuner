"""ACE-Step 1.5 model driver — family-specific training behavior.

NO VENDORING: diffusers 0.39.0 (already pinned) ships the full ACE-Step 1.5
model tree natively — ``AceStepTransformer1DModel``
(``diffusers.models.transformers.ace_step_transformer``),
``AceStepConditionEncoder`` (+ ``AceStepLyricEncoder``/``AceStepTimbreEncoder``,
``diffusers.pipelines.ace_step.modeling_ace_step``), and ``AutoencoderOobleck``
(``diffusers.models.autoencoders.autoencoder_oobleck``) — verified by import
(``from diffusers import AceStepPipeline, AceStepTransformer1DModel,
AutoencoderOobleck``) and by reading the installed
``diffusers/pipelines/ace_step/pipeline_ace_step.py`` directly. This driver
replicates that pipeline's math for the TRAINING path (flow-match forward +
condition encoding); no flash-attn/triton dependency exists anywhere in the
chain (SDPA via diffusers' attention dispatch).

Faithfully replicated from ``AceStepPipeline`` (diffusers 0.39.0, read at
``venv/Lib/site-packages/diffusers/pipelines/ace_step/pipeline_ace_step.py``):

Flow-match contract (raw ``[0, 1]``, NOT ``[0, 1000]`` at the DiT boundary)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``AceStepTransformer1DModel.forward(timestep=..., timestep_r=...)`` expects a
RAW ``[0, 1]`` scalar per sample (``t_curr_tensor`` in the pipeline's denoise
loop). This driver does NOT override ``sample_timesteps``/``add_noise`` — the
framework defaults (``TimestepSampler.sample_scaled`` → raw ``[0, 1000]``,
``NoiseInterpolation._linear`` → divides by 1000 for the blend) already
produce the correct ``(1-t)*latents + t*noise`` result; :meth:`forward_pass`
divides the RAW ``[0, 1000]`` timestep by 1000 itself right before the DiT
call (the single place the scale conversion happens — mirrors the
WAN/Kandinsky "pure-noise-LoRA" gotcha guard). ``timestep`` and ``timestep_r``
are set to the SAME value: all shipped ACE-Step 1.5 checkpoints train with
``use_meanflow=False``, which collapses the MeanFlow dual-timestep pair to
``r == t`` (see recon report §3).

Condition encoding (text + lyric + timbre -> one cross-attn sequence)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Captions go through the FULL Qwen3 text encoder (contextual hidden states);
lyrics go through ONLY its token-EMBEDDING layer (``get_input_embeddings()``)
— the lyric encoder inside ``AceStepConditionEncoder`` does the contextual
encoding. Both are formatted with the exact SFT prompt/lyric templates the
checkpoint was trained on (:func:`format_condition_text`).

Context latents (the DiT's "what's already given" side-channel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ACE-Step 1.5 is a masked-generation architecture: the DiT always receives a
SEPARATE ``context_latents`` argument (``cat([src_latents, chunk_mask],
dim=-1)``) describing what audio content is already fixed vs. what to
generate. For plain text2music (this family's only trained task — no
repaint/cover/audio-to-audio), ``src_latents`` is the model's own LEARNED
``condition_encoder.silence_latent`` tiled to the batch's latent length and
``chunk_mask`` is all-ones ("generate everything") — a CONSTANT independent of
the real training audio, exactly matching the pipeline's ``task_type
="text2music"``, no-``src_audio`` default path.

genre_ratio (CFG-dropout training mechanic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``AceStepConditionEncoder.null_condition_emb`` is a learned parameter the
upstream checkpoints were trained with at a 0.15 per-step drop probability
(confirmed by the diffusers source's own docstring: "trained with
cfg_ratio=0.15"). :meth:`forward_pass` reproduces this: per-sample Bernoulli
draw at ``self.genre_ratio`` (wired from the run config's ``genre_ratio``
field by the trainer) replaces the WHOLE condition sequence with the
broadcast null embedding.
"""

from __future__ import annotations

import math
from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver

logger = structlog.get_logger(__name__)

# LoRA target SUFFIXES — diffusers-native naming (verified by instantiating
# AceStepTransformer1DModel and listing nn.Linear modules): each transformer
# layer has `self_attn.{to_q,to_k,to_v,to_out.0}` AND
# `cross_attn.{to_q,to_k,to_v,to_out.0}` — a plain suffix match hits BOTH
# attention blocks (recon's "attention_type": "both" preset) with no risk of
# bleeding into `mlp.*`/`time_embed*`/`condition_embedder` (different names).
ACE_STEP15_LORA_SUFFIXES: list[str] = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
]

FLOWMATCH_SCALE = 1000.0

# Default vocal language for lyrics formatting (matches the pipeline default).
DEFAULT_VOCAL_LANGUAGE = "en"


def format_condition_text(
    prompt: str,
    lyrics: str,
    *,
    vocal_language: str = DEFAULT_VOCAL_LANGUAGE,
    audio_duration: float = 60.0,
) -> tuple[str, str]:
    """Format (caption, lyrics) into the SFT prompt templates (module-level;
    unit-tested; shared by the driver and the trainer's TE pre-cache so the
    cache key can't drift from what the driver would encode).

    Byte-identical to ``AceStepPipeline._format_prompt`` /
    ``_build_metadata_string`` (diffusers 0.39.0) — see the module docstring.
    """
    from diffusers.pipelines.ace_step.pipeline_ace_step import (
        DEFAULT_DIT_INSTRUCTION,
        SFT_GEN_PROMPT,
    )

    instruction = DEFAULT_DIT_INSTRUCTION
    if not instruction.endswith(":"):
        instruction = instruction + ":"

    dur_str = f"{int(audio_duration)} seconds" if audio_duration and audio_duration > 0 else "30 seconds"
    metas_str = f"- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: {dur_str}\n"

    formatted_text = SFT_GEN_PROMPT.format(instruction, prompt, metas_str)
    formatted_lyrics = f"# Languages\n{vocal_language}\n\n# Lyric\n{lyrics}<|endoftext|>"
    return formatted_text, formatted_lyrics


def tile_to_length(tensor: torch.Tensor, length: int) -> torch.Tensor:
    """Tile/crop a ``[1, T0, D]`` tensor along dim 1 to exactly ``length``.

    Shared helper for the silence-latent reuse (both the ``src_latents``
    context-channel default AND the "no reference audio" timbre default read
    the SAME ``condition_encoder.silence_latent`` buffer — see the pipeline's
    ``prepare_src_latents``/``__call__`` no-reference-audio branches).
    """
    t0 = tensor.shape[1]
    if t0 >= length:
        return tensor[:, :length, :]
    repeats = (length + t0 - 1) // t0
    return tensor.repeat(1, repeats, 1)[:, :length, :]


class AceStep15Driver(IModelDriver):
    """ACE-Step 1.5 driver (text2music DiT LoRA)."""

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Assigned by assign_components()
        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.tokenizer: Any = None
        self.condition_encoder: nn.Module | None = None
        # Driver-owned stashes of the condition encoder's two tiny buffers.
        # The shared pipeline pops text encoders — including the condition
        # encoder, which get_text_encoders() deliberately exposes — from
        # `components` after embedding caching, and prepare_for_training's
        # alias re-sync then re-runs assign_components() with that reduced
        # dict. forward_pass needs these buffers on EVERY step (context
        # latents + genre-drop null), so they must outlive the module.
        self._silence_latent: torch.Tensor | None = None
        self._null_condition_emb: torch.Tensor | None = None
        self._components: dict[str, Any] = {}

        arch = getattr(definition, "architecture_params", {}) or {}
        self.audio_acoustic_hidden_dim: int = int(
            arch.get("audio.acoustic_hidden_dim", 64)
        )
        # latents_per_second is re-derived from the loaded VAE in
        # assign_components() (authoritative); this is only the pre-load
        # fallback (matches the recon-derived default: 48000 / 1920 = 25.0).
        self.latents_per_second: float = float(arch.get("audio.latent_hz", 25.0))
        self.max_text_length: int = int(arch.get("audio.max_text_length", 256))
        self.max_lyric_length: int = int(arch.get("audio.max_lyric_length", 2048))
        # 30s of timbre-reference context (matches the pipeline's
        # `prepare_reference_audio_latents` fixed window) — recomputed from
        # latents_per_second once the VAE is assigned.
        self.timbre_fix_frame: int = math.ceil(30 * self.latents_per_second)

        # Wired by the trainer (`_setup_family`) from the run config — the
        # driver itself never reads `self.config` (it doesn't receive one;
        # only definition + device, per the house IModelDriver contract).
        self.genre_ratio: float = 0.15

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        self._components = components
        self.transformer = components.get("unet")
        self.vae = components.get("vae")
        self.text_encoder = components.get("text_encoder")
        self.tokenizer = components.get("tokenizer")
        self.condition_encoder = components.get("condition_encoder")
        if self.condition_encoder is not None:
            # Stash the two tiny buffers driver-side (CPU clones) so the
            # per-step forward keeps working after the pipeline pops the
            # 0.61B encoder module from the components dict (see __init__).
            self._silence_latent = (
                self.condition_encoder.silence_latent.detach().to("cpu").clone()
            )
            self._null_condition_emb = (
                self.condition_encoder.null_condition_emb.detach().to("cpu").clone()
            )

        cfg = getattr(self.transformer, "config", None)
        if cfg is not None:
            self.audio_acoustic_hidden_dim = int(
                getattr(cfg, "audio_acoustic_hidden_dim", self.audio_acoustic_hidden_dim)
            )

        vae_cfg = getattr(self.vae, "config", None)
        if vae_cfg is not None:
            sample_rate = float(getattr(vae_cfg, "sampling_rate", 48000))
            ratios = getattr(vae_cfg, "downsampling_ratios", None) or (1920,)
            downsample = 1
            for r in ratios:
                downsample *= int(r)
            if downsample > 0:
                self.latents_per_second = sample_rate / float(downsample)
        self.timbre_fix_frame = math.ceil(30 * self.latents_per_second)

        self.logger.info(
            "ace_step15_config",
            audio_acoustic_hidden_dim=self.audio_acoustic_hidden_dim,
            latents_per_second=self.latents_per_second,
            timbre_fix_frame=self.timbre_fix_frame,
        )

    @property
    def silence_latent(self) -> torch.Tensor:
        """The condition encoder's silence latent (driver-owned stash)."""
        if self._silence_latent is None:
            raise RuntimeError(
                "silence_latent unavailable — no condition_encoder was ever "
                "assigned to this driver"
            )
        return self._silence_latent

    @property
    def null_condition_emb(self) -> torch.Tensor:
        """The learned null-condition embedding (driver-owned stash)."""
        if self._null_condition_emb is None:
            raise RuntimeError(
                "null_condition_emb unavailable — no condition_encoder was "
                "ever assigned to this driver"
            )
        return self._null_condition_emb

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.transformer

    def get_text_encoders(self) -> dict[str, nn.Module]:
        """The Qwen3 text encoder AND the condition encoder.

        Both are frozen (LoRA trains the DiT only, per upstream's
        ``target_modules_str: "q_proj k_proj v_proj o_proj"`` — see recon
        report §2) and both are needed to produce the FINAL cached
        conditioning sequence, so treating ``condition_encoder`` as a "text
        encoder" here gets it freeze/offload/phased-sampling handling for
        free from the shared pipeline (``_freeze_all`` /
        ``_offload_text_encoders`` / the sampler's phased GPU management all
        key off this dict).
        """
        result: dict[str, nn.Module] = {}
        if self.text_encoder is not None:
            result["text_encoder"] = self.text_encoder
        if self.condition_encoder is not None:
            result["condition_encoder"] = self.condition_encoder
        return result

    def get_lora_targets(self) -> list[str]:
        entries = list(getattr(self.definition, "lora_targetable_modules", None) or [])
        return entries or list(ACE_STEP15_LORA_SUFFIXES)

    def get_te_lora_targets(self) -> list[str]:
        """Text/condition encoder LoRA not supported — upstream trains DiT-only."""
        return []

    def init_scheduler(self) -> Any:
        """Flow matching — no external training scheduler (see module docstring)."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """Transformer + TE + condition encoder load in bf16 (VAE stays fp32)."""
        return torch.bfloat16

    # --- Phase 2: Text/Condition Encoding ---

    def encode_condition(
        self,
        prompts: list[str],
        lyrics: list[str],
        dtype: torch.dtype,
        audio_duration: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode (caption, lyrics) pairs -> the DiT's cross-attn conditioning.

        Replicates ``AceStepPipeline.encode_prompt`` + the
        ``condition_encoder(...)`` call from ``__call__`` (steps 1 + 4), using
        the model's own learned silence latent for the timbre-reference input
        (no per-item reference audio for plain text2music LoRA training —
        matches the pipeline's "no reference_audio" default path).

        Returns:
            ``(encoder_hidden_states [B, L, D], encoder_attention_mask [B, L])``.
        """
        batch_size = len(prompts)
        dur = audio_duration if audio_duration and audio_duration > 0 else 30.0

        all_text_strs: list[str] = []
        all_lyric_strs: list[str] = []
        for i in range(batch_size):
            text_str, lyric_str = format_condition_text(
                prompts[i], lyrics[i] if i < len(lyrics) else "", audio_duration=dur
            )
            all_text_strs.append(text_str)
            all_lyric_strs.append(lyric_str)

        text_inputs = self.tokenizer(
            all_text_strs,
            padding="longest",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        text_attention_mask = text_inputs.attention_mask.to(self.device).bool()

        lyric_inputs = self.tokenizer(
            all_lyric_strs,
            padding="longest",
            truncation=True,
            max_length=self.max_lyric_length,
            return_tensors="pt",
        )
        lyric_input_ids = lyric_inputs.input_ids.to(self.device)
        lyric_attention_mask = lyric_inputs.attention_mask.to(self.device).bool()

        with torch.no_grad():
            text_hidden_states = self.text_encoder(
                input_ids=text_input_ids
            ).last_hidden_state
            embed_layer = self.text_encoder.get_input_embeddings()
            lyric_hidden_states = embed_layer(lyric_input_ids)

            silence_latent = self.silence_latent.to(
                device=self.device, dtype=dtype
            )
            refer_audio_acoustic = tile_to_length(
                silence_latent, self.timbre_fix_frame
            ).expand(batch_size, -1, -1).contiguous()
            refer_audio_order_mask = torch.arange(
                batch_size, device=self.device, dtype=torch.long
            )

            encoder_hidden_states, encoder_attention_mask = self.condition_encoder(
                text_hidden_states=text_hidden_states.to(dtype),
                text_attention_mask=text_attention_mask,
                lyric_hidden_states=lyric_hidden_states.to(dtype),
                lyric_attention_mask=lyric_attention_mask,
                refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic,
                refer_audio_order_mask=refer_audio_order_mask,
            )
        return encoder_hidden_states, encoder_attention_mask

    def encode_text(self, captions: list[str], dtype: torch.dtype) -> Any:
        """IModelDriver contract entry point (no lyrics) — used only when the
        trainer calls the driver directly without per-item lyrics (e.g. a
        generic caller that doesn't know about the audio seam). The real
        training/sampling path calls :meth:`encode_condition` directly with
        per-item lyrics via the trainer/sampler overrides.
        """
        return self.encode_condition(captions, [""] * len(captions), dtype)

    # --- Phase 5: Training Loop Hooks ---

    def prepare_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """VAE-native channels-first ``[B, D, T]`` -> transformer ``[B, T, D]``.

        Mirrors the pipeline's own transpose at the VAE boundary
        (``ref_latents.transpose(1, 2)`` / ``audio_latents = xt.transpose(1,
        2)``) — our latent cache stays VAE-native (channels-first), the
        transformer layout is built here at the training-loop boundary,
        exactly the Kandinsky5 channels-last precedent for a different-layout
        DiT.
        """
        if latents.ndim != 3:
            raise ValueError(
                "AceStep15 prepare_latents expects a 3D [B, D, T] VAE latent, "
                f"got {tuple(latents.shape)}"
            )
        return latents.transpose(1, 2).contiguous()

    def _build_context_latents(
        self, batch_size: int, latent_length: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """``cat([src_latents, chunk_mask], dim=-1)`` — the text2music default
        (silence-latent src + all-ones "generate everything" mask). See the
        module docstring's "Context latents" section."""
        silence_latent = self.silence_latent.to(device=device, dtype=dtype)
        src_latents = tile_to_length(silence_latent, latent_length).expand(
            batch_size, -1, -1
        )
        chunk_mask = torch.ones(
            batch_size, latent_length, self.audio_acoustic_hidden_dim,
            device=device, dtype=dtype,
        )
        return torch.cat([src_latents, chunk_mask], dim=-1)

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """``AceStepTransformer1DModel`` forward -> velocity prediction.

        - ``hidden_states``: ``[B, T, D]`` (post ``prepare_latents`` transpose).
        - ``timestep``/``timestep_r``: RAW ``[0, 1000]`` scaled DOWN to ``[0,
          1]`` here (the one place the /1000 happens — see module docstring).
        - ``context_latents``: the text2music constant (silence + ones mask).
        - genre_ratio: per-sample Bernoulli drop of the WHOLE condition
          sequence to the learned null embedding (see module docstring).
        """
        if not (isinstance(text_embeddings, tuple) and len(text_embeddings) == 2):
            raise TypeError(
                "AceStep15 forward_pass requires a (encoder_hidden_states, "
                f"encoder_attention_mask) tuple, got {type(text_embeddings).__name__}"
            )
        encoder_hidden_states, _encoder_attention_mask = text_embeddings

        dtype = self.transformer.dtype
        device = noisy_input.device
        b, t, _ = noisy_input.shape

        t01 = (timesteps.to(torch.float32) / FLOWMATCH_SCALE).to(dtype=dtype, device=device)

        encoder_hidden_states = encoder_hidden_states.to(device=device, dtype=dtype)
        genre_ratio = float(getattr(self, "genre_ratio", 0.0) or 0.0)
        if genre_ratio > 0.0 and self.transformer.training:
            drop_mask = torch.rand(b, device=device) < genre_ratio
            if bool(drop_mask.any()):
                null_emb = self.null_condition_emb.to(
                    device=device, dtype=dtype
                )
                null_expanded = null_emb.expand_as(encoder_hidden_states)
                encoder_hidden_states = torch.where(
                    drop_mask.view(-1, 1, 1), null_expanded, encoder_hidden_states
                )

        context_latents = self._build_context_latents(b, t, device, dtype)

        output = self.transformer(
            hidden_states=noisy_input.to(dtype),
            timestep=t01,
            timestep_r=t01,
            encoder_hidden_states=encoder_hidden_states,
            context_latents=context_latents,
            return_dict=False,
        )
        return output[0] if isinstance(output, (tuple, list)) else output

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self) -> Any:
        from .saver import AceStep15Saver

        return AceStep15Saver()

    def get_save_metadata(self) -> dict[str, str]:
        return {"modelspec.architecture": "ace-step15.dit"}
