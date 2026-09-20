"""MiniMax-H3 model driver — Task 6: non-training surface.

Implements the ``IModelDriver`` methods that do NOT require the real training
forward pass: component wiring, the curated LoRA target list, block topology,
and loading dtype — all sourced from the ``ModelDefinition`` (Task 4), the
single source of truth. Text encoding (the Qwen3-VL layer-50 tap, plan row
2.1) and the flow-match convention (row 1.2) are real; what still belongs to
the packed joint audio+video forward and LoRA saving lands in later PR1 rows
and raises ``NotImplementedError`` naming it explicitly — per
the "failure is never silent" invariant, a job that somehow reaches those
methods must fail loudly, not silently train on wrong data or produce a
plausible-looking empty/None default.

``init_scheduler`` — DO NOT CHANGE without reading this
---------------------------------------------------------
``init_scheduler`` returns ``None`` and MUST keep doing exactly that. H3
trains with flow matching (like every sibling family here: ltx2, wan21,
ace_step15, zimage, flux2) — no external scheduler object at train time — so
``return None`` is not a stub, it is the correct permanent answer.

This is also a structural trap: ``app.engine.core.hook_dispatch.
TRIVIAL_BODIES`` recognizes ONLY the exact normalized body ``"return None"``
for ``init_scheduler`` as the trivial/no-op baseline. Any other body —
including one that raises ``NotImplementedError`` — is a *meaningful
override* per ``driver_meaningfully_overrides``, which silently enrolls
``minimax_h3`` into the reviewed auto-delegation allowlist
(``AUTODELEGATED_FAMILY_HOOKS`` in
``tests/engine/test_hook_wiring_meta.py``) and trips
``test_autodelegated_family_hook_set_is_exactly_expected``. The same guard
covers every hook in the derived ``CLOBBER_HOOKS`` set (``add_noise``,
``build_batch_extra``, ``compute_target``, ``get_te_cache``,
``sample_timesteps``, ``set_te_cache``, ``init_scheduler``). PR1 row 1.2
overrides ``add_noise`` / ``compute_target`` / ``sample_timesteps`` here AND
delegates each explicitly from ``MiniMaxH3Trainer`` in the same commit
(plan ordering rule 1), so the family never relies on auto-delegation and
the reviewed allowlist stays unchanged. Any further CLOBBER hook this driver
grows must land with its trainer delegation the same way.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver

from .packing import H3PackedLayout
from .settings import H3EffectiveSettings


@dataclass(frozen=True)
class H3StepLoss:
    """The three numbers one training step reports (plan row 2.4)."""

    loss: torch.Tensor
    loss_video: torch.Tensor
    loss_audio: torch.Tensor


def _lands_in_pr1(what: str) -> NotImplementedError:
    return NotImplementedError(
        f"minimax_h3 {what} lands in PR1; PR0 (Task 6) ships the "
        "non-training driver surface only."
    )


class MiniMaxH3Driver(IModelDriver):
    """MiniMax-H3 driver — non-training surface (Task 6).

    Handles:
    - Component wiring (tokenizer/processor, Qwen3-VL text encoder, visual
      VAE, audio VAE, diffusers transformer) per the loader manifest (Task 5).
    - LoRA target list and block topology, both read VERBATIM from the
      definition (Task 4) so the YAML stays the single source of truth —
      pinned by ``test_definition_ships_curated_target_list_matching_driver``.
    - bf16 loading dtype (every definition's ``detected_precision`` is bf16
      throughout).

    Text encoding, the joint audio+video forward pass, and LoRA saving raise
    ``NotImplementedError`` naming PR1 (see module docstring).
    """

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device

        # Assigned by assign_components() — component keys per
        # MiniMaxH3Loader.get_component_manifest (Task 5).
        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.audio_vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.tokenizer: Any = None
        self._components: dict[str, Any] = {}
        # The ONE resolved settings object (row 1.0); the trainer applies it
        # once at setup. Every consumer that needs it refuses to run without.
        self.settings: H3EffectiveSettings | None = None
        self._layouts: dict[tuple[Any, ...], H3PackedLayout] = {}

    def apply_settings(self, settings: H3EffectiveSettings) -> None:
        self.settings = settings

    def _require_settings(self, what: str) -> H3EffectiveSettings:
        if self.settings is None:
            raise RuntimeError(
                f"minimax_h3 {what}: H3EffectiveSettings not applied — the trainer "
                "must call driver.apply_settings(resolve_h3_settings(...)) at setup"
            )
        return self.settings

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded MiniMax-H3 components into driver state.

        Five components per the loader manifest (Task 5): ``tokenizer``
        (processor, never moved to device), ``text_encoder`` (Qwen3-VL-32B,
        cached then unloaded before the DiT loads), ``vae`` (visual),
        ``audio_vae`` (MONO — run once per stereo channel), ``transformer``
        (diffusers' class; ``transformer`` or ``transformer_ref`` subfolder per
        definition).
        """
        self._components = components
        self.transformer = components.get("transformer")
        self.vae = components.get("vae")
        self.audio_vae = components.get("audio_vae")
        self.text_encoder = components.get("text_encoder")
        self.tokenizer = components.get("tokenizer")

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.transformer

    def get_text_encoders(self) -> dict[str, nn.Module]:
        """Return the single Qwen3-VL text encoder."""
        result: dict[str, nn.Module] = {}
        if self.text_encoder is not None:
            result["text_encoder"] = self.text_encoder
        return result

    def get_lora_targets(self) -> list[str]:
        """Return the definition's curated target list VERBATIM.

        The YAML (Task 4) is the single source of truth — this method must
        never compute, filter, or otherwise derive its own list, or the two
        could drift silently (the nucleus_image contract). Pinned by
        ``test_definition_ships_curated_target_list_matching_driver``.
        """
        return list(self.definition.lora_targetable_modules)

    def init_scheduler(self) -> Any:
        # ``None`` (not a raise) deliberately: H3 trains with flow matching,
        # like every sibling family here — this IS the correct answer, not a
        # placeholder. See the module docstring for why this exact body must
        # never change (the TRIVIAL_BODIES / auto-delegation guard trap).
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """H3 checkpoints are bf16 throughout (every definition's
        ``detected_precision``: text_encoder/vae/unet all ``torch.bfloat16``)."""
        return torch.bfloat16

    # --- Phase 2: Text Encoding (plan row 2.1) ---
    #
    # Evidence for the tap: MiniMax-H3 README "The H3-Encoder uses the full
    # pretrained weights of Qwen3-VL-32B and provides the hidden states from
    # its 50th layer" (research §4); ai-toolkit `text_encoder.py` "the
    # **unnormalized** hidden_states[50] … hidden_states[0] is the embedding
    # output". transformers 5.14.1 (installed; probed on a tiny Qwen3-VL,
    # `.agent/workdir/minimax-pr1/hs_probe.py`): `hidden_states` has
    # `num_hidden_layers + 1` entries, `[0]` == the input embeddings, only
    # `[-1]` carries the final RMSNorm — so `[50]` on the 64-layer stack is
    # decoder layer 49's raw output, exactly the conditioning H3 was trained on.
    # Presentation: raw tokens, no chat template, no special tokens
    # (ai-toolkit, same file); the caption is capped at `te.max_length`.

    def _tokenizer_only(self) -> Any:
        """The tokenizer under the ``AutoProcessor`` (or a bare tokenizer)."""
        tok = self.tokenizer
        return getattr(tok, "tokenizer", tok)

    def _tap_index(self) -> int:
        return int((self.definition.architecture_params or {})["te.hidden_state_tap_index"])

    def _prompt_max_tokens(self) -> int:
        return int((self.definition.architecture_params or {})["te.max_length"])

    def tokenizer_fingerprint(self) -> str:
        """SHA-256 over the tokenizer class + its full vocabulary — a
        different vocabulary yields different ids for the same caption, hence
        a different embedding; the cache key must carry it."""
        cached = getattr(self, "_tokenizer_fp", None)
        if cached is not None:
            return cached
        tok = self._tokenizer_only()
        vocab = tok.get_vocab()
        digest = hashlib.sha256()
        digest.update(type(tok).__name__.encode("utf-8"))
        for token, idx in sorted(vocab.items(), key=lambda kv: (kv[1], kv[0])):
            digest.update(f"{idx}:{token}\n".encode("utf-8"))
        self._tokenizer_fp = digest.hexdigest()
        return self._tokenizer_fp

    def te_cache_scope(self) -> str:
        """Path segment the disk cache lives under: tap index + tokenizer."""
        return f"tap{self._tap_index()}-{self.tokenizer_fingerprint()[:16]}"

    def te_cache_key(self, prompt: str) -> str:
        """The string a cached embedding is keyed on — every input the
        embedding is a function of: definition, tap index, tokenizer, prompt.
        Human-readable; ``TextEmbeddingCache`` hashes it for the filename."""
        return (
            f"minimax_h3|{self.definition.id}|tap={self._tap_index()}"
            f"|tok={self.tokenizer_fingerprint()[:16]}|{prompt}"
        )

    def _encode_one(self, caption: str) -> torch.Tensor:
        """``[L, D]`` layer-tap hidden states for ONE caption, no padding."""
        tok = self._tokenizer_only()
        ids = list(tok(caption, add_special_tokens=False)["input_ids"])[: self._prompt_max_tokens()]
        if not ids:
            # Empty (dropout / unconditional) prompt: one pad token keeps the
            # sequence non-degenerate (ai-toolkit's fallback).
            ids = [getattr(tok, "pad_token_id", None) or 0]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        te = self.text_encoder
        inner = getattr(te, "model", te)  # skip the LM head — dead weight here
        tap = self._tap_index()
        with torch.no_grad():
            out = inner(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = out.hidden_states
        if tap >= len(hidden_states):
            raise ValueError(
                f"te.hidden_state_tap_index={tap} but the encoder returned "
                f"{len(hidden_states)} hidden states (num_hidden_layers + 1)"
            )
        return hidden_states[tap][0]

    def encode_text(self, captions: list[str], dtype: torch.dtype) -> Any:
        """Qwen3-VL layer-tap conditioning, batched by right-padding.

        Each caption is encoded ALONE (no padding inside the encoder, so the
        result is byte-identical to the single-prompt reference path) and the
        batch is assembled by zero-padding to the longest — ``attention_mask``
        marks the real rows.
        """
        from app.engine.core.text_encoding import TextEncoderOutput

        if self.text_encoder is None or self.tokenizer is None:
            raise RuntimeError(
                "minimax_h3 encode_text: text_encoder/tokenizer not assigned — "
                "the encoder was released or assign_components() never ran"
            )
        rows = [self._encode_one(cap) for cap in captions]
        max_len = max(r.shape[0] for r in rows)
        emb = rows[0].new_zeros((len(rows), max_len, rows[0].shape[-1]))
        mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=rows[0].device)
        for i, r in enumerate(rows):
            emb[i, : r.shape[0]] = r
            mask[i, : r.shape[0]] = 1
        return TextEncoderOutput(embeddings=emb.to(dtype=dtype), attention_mask=mask)

    # --- Text-encoder lifecycle: released BEFORE the DiT loads ---

    def text_encoder_weight_bytes(self) -> int:
        te = self.text_encoder
        if te is None or not hasattr(te, "parameters"):
            return 0
        return sum(p.numel() * p.element_size() for p in te.parameters())

    def release_text_encoders(self) -> None:
        """Drop EVERY reference the driver holds: the attribute
        ``get_text_encoders()`` reads AND the entry in the shared component
        dict (the base offload pops the trainer's copy — the same dict — but
        the driver owns component state, so it drops its own view too)."""
        self.text_encoder = None
        self._components.pop("text_encoder", None)

    def assert_text_encoder_released(self) -> None:
        """The 63 GB encoder and the 62 GB DiT never coexist: refuse the DiT
        load while the encoder is still held anywhere the driver can see."""
        if self.text_encoder is not None or "text_encoder" in self._components:
            gb = self.text_encoder_weight_bytes() / 1e9
            raise RuntimeError(
                f"text encoder still resident ({gb:.2f} GB) at DiT load — "
                "release it (cache the embeddings) before materialising the transformer"
            )

    # --- Latent-cache identity (plan row 2.3) ---

    def latent_cache_fingerprint(self) -> str:
        """Short hash of EVERY input the RAW VAE latent depends on — and no
        other (D10): the pixel convention (`PIXEL_ADAPTER_VERSION`, read at
        call time), the visual VAE's class and its config. NOT the packed
        layout version and NOT the sigma shifts: those shape the sequence at
        train time, and keying on them would re-encode a dataset on every
        settings change. The frame rule already lives in the resolution
        segment of the cache path (`tgt_f`).
        """
        from . import pixel_adapter

        vae = self.vae
        if vae is None:
            raise RuntimeError(
                "minimax_h3 latent_cache_fingerprint: no visual VAE assigned — "
                "the cache key cannot be derived without the encoder identity"
            )
        inner = getattr(vae, "inner", vae)
        config = getattr(inner, "config", None)
        if config is None:
            config_dict: dict[str, Any] = {}
        elif hasattr(config, "items"):  # diffusers FrozenDict or a plain dict
            config_dict = dict(config.items())
        else:
            config_dict = dict(vars(config))
        payload = json.dumps(
            {
                "pixel_adapter": pixel_adapter.PIXEL_ADAPTER_VERSION,
                "vae_class": type(inner).__name__,
                "vae_config": config_dict,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]

    # --- Frame rule (plan row 2.7) ---
    #
    # The visual VAE chunks 17 pixel frames into 5 latent frames behind a
    # 5-frame head, so a legal clip length is ``17n+5`` (research §3.3) — read
    # from the definition, never a literal (RULE-21), and snapped through the
    # object every video family shares (`video_contract.snap_frames`) so the
    # family's answer and the temporal bucketing ladder can never disagree.
    # Ingestion (`pipeline_data.prepare_data`) is where a too-short clip is
    # skipped; these are the family-side give-ups for callers that hold a
    # frame count of their own (previews, the sampler's `num_frames`).

    def frame_rule(self) -> str:
        rule = self.definition.architecture_params.get("video.frame_rule")
        if not rule:
            raise ValueError(
                f"minimax_h3 {self.definition.id}: architecture_params['video.frame_rule'] is missing"
            )
        return str(rule)

    def _frame_floor(self) -> int:
        from app.engine.components.bucketing import BucketManager

        parsed = BucketManager._parse_frame_step(self.frame_rule())
        if parsed is None:
            raise ValueError(f"minimax_h3: unparseable video.frame_rule {self.frame_rule()!r}")
        return parsed[1]

    def frame_ladder(self, max_frames: int) -> list[int]:
        from app.engine.components.bucketing import BucketManager

        return BucketManager.frame_ladder(int(max_frames), self.frame_rule())

    def snap_num_frames(self, num_frames: int) -> int:
        """Snap DOWN to the nearest legal length (``97 → 90``); at or below the
        floor the floor itself (there is no legal value under it)."""
        from app.engine.core.video_contract import snap_frames

        return snap_frames(int(num_frames), self.frame_rule())

    def assert_clip_frames(self, num_frames: int) -> None:
        """Refuse a clip shorter than the floor — loudly, never padded."""
        floor = self._frame_floor()
        if int(num_frames) < floor:
            raise ValueError(
                f"clip has {int(num_frames)} frames; the smallest legal H3 length is "
                f"{floor} ({self.frame_rule()})"
            )

    # --- Phase 4: Precision, LoRA Targets & Layer Manifest ---

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported — Qwen3-VL-32B stays frozen.

        This is a definitive architectural answer, not a PR1 placeholder:
        ``family.py`` deliberately does NOT override ``supports_train_te``,
        so it inherits the ``latent_diffusion`` archetype's ``False``
        default — the ~66.7 GB TE (66,714,780,128 bytes bf16 on disk, not
        the ~48 GB previously stated here — see the definitions'
        text-encoder comment) must never train.
        """
        return []

    # --- Phase 5: Training Loop Hooks ---
    #
    # The INVERTED flow-match contract (concept §5.1, closed oracle against
    # diffusers 0.40.0 `scheduling_minimax_h3.py`: `:170-171` t = 1 − σ,
    # `:225` x_t = t·x₀ + (1−t)·noise, `:273` x̂₀ = x_t + σ·v):
    #     x₀ = x_t + σ·v   ⇒   v = x₀ − noise   (unique; no scale, no sign freedom)
    # Research §3.1: ai-toolkit `t_v = 1.0 − sigma_v`, `return -noise_pred`;
    # diffusion-pipe `t_v = 1.0 − sigma_v`, `-video_out` — the same contract.
    # These three are CLOBBER hooks: `MiniMaxH3Trainer` delegates each one
    # explicitly (ordering rule 1) so `test_autodelegated_family_hook_set_is_
    # exactly_expected` stays byte-identical.

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        config: dict[str, Any],
        latents: torch.Tensor | None = None,
        progress: float = 0.0,
    ) -> torch.Tensor:
        """Draw ONE ``u`` per item, push it through the VIDEO shift and return
        ``t_v = 1 − σ_v`` in ``[0, 1]`` (the audio clock is derived from σ_v
        in the forward pass through ``H3SigmaSchedule`` — never a second draw).

        Family default draw is ``uniform`` — with the definition's shift on top
        it reproduces the inference grid (diffusion-pipe's recommendation,
        research §4); a user-set ``timestep_sampling`` wins.
        """
        from app.engine.strategies.timestep_sampling import TimestepSampler  # noqa: PLC0415

        from .schedule import H3SigmaSchedule, sigma_to_t
        from .settings import resolve_h3_settings

        mode = config.get("timestep_sampling", "uniform")
        u = TimestepSampler.sample(
            mode, batch_size, device, config, latents=latents, progress=progress,
        )
        schedule = H3SigmaSchedule.from_settings(resolve_h3_settings(self.definition, config))
        sigma_v, _sigma_a = schedule.draw(u)
        return sigma_to_t(sigma_v).clamp(0.0, 1.0)

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """``x_t = t·x₀ + σ·noise`` with ``σ = 1 − t`` — H3's clock (``t = 1``
        clean), identical to ``MiniMaxH3Scheduler.scale_noise``. Written with
        the noise weight from ``schedule.t_to_sigma`` so the family keeps ONE
        conversion site and both endpoints are exact."""
        from .schedule import t_to_sigma

        t = timesteps.to(device=latents.device, dtype=latents.dtype)
        while t.ndim < latents.ndim:
            t = t.unsqueeze(-1)
        return t * latents + t_to_sigma(t) * noise

    def compute_target(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """The data-ward velocity ``v = x₀ − noise`` — the unique target the
        reference scheduler's ``step`` inverts. The house default
        (``noise − latents``) is the OPPOSITE sign on this family."""
        return latents - noise

    def max_distinct_timesteps(self) -> int:
        """How many DISTINCT timestep values one forward may carry: ``t_v``,
        ``t_a`` and the keyframe ``t_c`` (3) for ``t2v`` / ``both``; the
        reference mode adds clean reference soundtracks (4) — research §3.5."""
        mode = str(self.definition.architecture_params.get("mode", "t2v"))
        return 4 if mode == "reference" else 3

    def assert_timestep_cardinality(self, timesteps: torch.Tensor) -> None:
        """The DRIVER boundary check before ``packed_forward``: the transformer
        embeds whatever ``timestep`` arrives (`transformer_minimax_h3.py:613`),
        so a per-row ``(seq_len,)`` tensor — or ``timestep_indices`` handed
        where ``timestep`` goes — would train silently on garbage. Raises
        ``ValueError`` naming the defect."""
        limit = self.max_distinct_timesteps()
        if timesteps.ndim != 1 or timesteps.numel() > limit:
            raise ValueError(
                f"minimax_h3 expects the DISTINCT timestep set (<= {limit} values), got "
                f"shape {tuple(timesteps.shape)} — per-row timesteps do not belong here"
            )
        if not timesteps.dtype.is_floating_point:
            raise ValueError(
                f"minimax_h3 timesteps must be floating t in [0, 1], got dtype {timesteps.dtype}"
            )
        if bool((timesteps < 0).any()) or bool((timesteps > 1).any()):
            raise ValueError(
                "minimax_h3 timesteps must lie in [0, 1] unscaled on H3's clock (1 = clean), "
                f"got {timesteps.tolist()}"
            )

    # --- Phase 5b: the joint forward + the three-number loss (plan row 2.4) ---

    def audio_timestep(self, t_video: torch.Tensor) -> torch.Tensor:
        """``t_a`` for a given ``t_v``: ONE ``u`` drives both clocks, so the
        audio timestep is the video sigma un-shifted by the video shift and
        re-shifted by the audio shift (research §3.2), never a second draw."""
        from .schedule import H3SigmaSchedule, remap_sigma, sigma_to_t, t_to_sigma

        schedule = H3SigmaSchedule.from_settings(self._require_settings("audio_timestep"))
        sigma_a = remap_sigma(
            t_to_sigma(t_video), schedule.sigma_shift_video, schedule.sigma_shift_audio
        )
        return sigma_to_t(sigma_a).clamp(0.0, 1.0)

    def build_batch_extra(self, items: list[dict]) -> dict[str, Any]:
        """Stack the items' clean audio latents ``(2, C, T)`` into
        ``{"audio_clean": (B, 2, C, T), "audio_mask": (B,)}``. An item without
        audio gets zeros shaped like a present sibling and ``mask = 0``; a
        batch with NO audio (or ``train_audio`` off) returns ``{}`` so the
        forward stays video-only. The trainer (row 2.5) loads the cached
        latents into ``item["audio_latents"]`` before delegating here."""
        present = [item.get("audio_latents") for item in items]
        if not any(a is not None for a in present):
            return {}
        if not self._require_settings("build_batch_extra").train_audio:
            return {}
        ref = next(a for a in present if a is not None)
        stacked = []
        mask = []
        for a in present:
            if a is None or tuple(a.shape) != tuple(ref.shape):
                stacked.append(torch.zeros_like(ref))
                mask.append(0.0)
            else:
                stacked.append(a)
                mask.append(1.0)
        return {
            "audio_clean": torch.stack(stacked, dim=0),
            "audio_mask": torch.tensor(mask, dtype=ref.dtype),
        }

    def _layout_for(self, num_text: int, video: torch.Tensor, audio_latents: int) -> H3PackedLayout:
        from .packing import H3Geometry, build_layout

        key = (num_text, tuple(video.shape[2:]), audio_latents)
        layout = self._layouts.get(key)
        if layout is None:
            layout = build_layout(
                H3Geometry(
                    num_text=num_text,
                    latent_frames=int(video.shape[2]),
                    latent_height=int(video.shape[3]),
                    latent_width=int(video.shape[4]),
                    audio_latents=audio_latents,
                )
            )
            self._layouts[key] = layout
        return layout

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The joint audio+video forward on the packed ``[text | audio | video]``
        sequence — ONE sequence per item, built with that item's TRUE text
        length (the batch's padding rows never enter the transformer) and its
        own distinct timestep set ``[t_v, t_a]``.

        ``noisy_input``: video ``x_t`` ``(B, C, T, H, W)``; ``timesteps``: ``t_v``
        per item ``(B,)``; ``batch["audio_noisy"]``: audio ``x_t`` ``(B, 2, Ca, Ta)``
        (absent → video-only sequence). Returns ``(video_velocity, audio_velocity)``
        in the raw latent shapes; the trainer's step loss (row 2.6) consumes both.
        """
        from .packing import pack_audio, packed_forward, patchify_video, unpack_audio, unpatchify_video

        if self.transformer is None:
            raise RuntimeError("minimax_h3 forward_pass: transformer not assigned")
        emb = getattr(text_embeddings, "embeddings", None)
        mask = getattr(text_embeddings, "attention_mask", None)
        if emb is None:
            emb, mask = text_embeddings  # (embeddings, mask) tuple form
        audio_noisy = batch.get("audio_noisy")
        t_v = timesteps.to(device=noisy_input.device, dtype=torch.float32).reshape(-1)
        if t_v.numel() != noisy_input.shape[0]:
            raise ValueError(
                f"minimax_h3 forward_pass: {t_v.numel()} timesteps for a batch of {noisy_input.shape[0]}"
            )
        t_a = self.audio_timestep(t_v)

        video_out = []
        audio_out = []
        for i in range(noisy_input.shape[0]):
            n_text = int(mask[i].sum()) if mask is not None else int(emb.shape[1])
            text_rows = emb[i : i + 1, :n_text]
            audio_latents = int(audio_noisy.shape[-1]) if audio_noisy is not None else 0
            layout = self._layout_for(n_text, noisy_input, audio_latents)
            video_rows = patchify_video(noisy_input[i : i + 1], layout.patch_size)
            if audio_noisy is not None:
                audio_rows = pack_audio(audio_noisy[i : i + 1])
            else:
                audio_rows = noisy_input.new_zeros((1, 0, int(self.definition.architecture_params["transformer.audio_in_channels"])))
            distinct = torch.stack([t_v[i], t_a[i]])
            self.assert_timestep_cardinality(distinct)
            v_rows, a_rows = packed_forward(
                self.transformer, layout, video_rows, audio_rows, text_rows, distinct
            )
            video_out.append(unpatchify_video(v_rows, layout))
            if audio_noisy is not None:
                audio_out.append(unpack_audio(a_rows, audio_latents))
        video_velocity = torch.cat(video_out, dim=0)
        audio_velocity = torch.cat(audio_out, dim=0) if audio_out else None
        return video_velocity, audio_velocity

    def compute_loss(
        self,
        video_pred: torch.Tensor,
        video_target: torch.Tensor,
        batch: dict[str, Any],
        *,
        audio_pred: torch.Tensor | None = None,
        audio_target: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> H3StepLoss:
        """``loss = loss_video + audio_loss_weight · loss_audio`` — the ONE lever
        between the modalities (settings row 1.0), reported as three numbers.
        The audio term is per-item masked (absent-audio items contribute 0);
        with ``train_audio`` off it is 0 and ``loss == loss_video``. Audio ON
        with no audio tensors is refused, never a silent video-only step."""
        settings = self._require_settings("compute_loss")
        loss_video = torch.nn.functional.mse_loss(video_pred.float(), video_target.float())
        if not settings.train_audio:
            zero = torch.zeros((), dtype=loss_video.dtype, device=loss_video.device)
            return H3StepLoss(loss=loss_video, loss_video=loss_video, loss_audio=zero)
        if audio_pred is None or audio_target is None:
            raise ValueError(
                "minimax_h3 compute_loss: train_audio is on but no audio pred/target "
                "reached the loss — the audio stream was dropped somewhere upstream"
            )
        per_item = (audio_pred.float() - audio_target.float()).pow(2).flatten(1).mean(dim=1)
        if audio_mask is None:
            loss_audio = per_item.mean()
        else:
            m = audio_mask.to(per_item.dtype).reshape(-1)
            loss_audio = (per_item * m).sum() / m.sum().clamp(min=1.0)
        loss = loss_video + float(settings.audio_loss_weight) * loss_audio
        return H3StepLoss(loss=loss, loss_video=loss_video, loss_audio=loss_audio)

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self) -> Any:
        raise _lands_in_pr1("the LoRA saver")

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """Return the definition's block topology verbatim.

        52 blocks total: 50 main ``transformer_blocks`` + 2 NESTED
        ``token_refiner.refiner_blocks`` (a bare ``refiner_blocks`` attr_path
        does not resolve — see the YAML's comment). Read from the definition
        rather than introspecting a loaded model: PR0 never loads real
        weights, and the definition is authoritative regardless.
        """
        return list(self.definition.block_topology)
