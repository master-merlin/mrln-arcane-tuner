"""HunyuanVideo 1.5 driver — flow-match training behavior for the hv15 family.

Mirrors the RESPONSIBILITIES of :mod:`wan_shared.driver_base` (flow-match
``add_noise`` in the raw ``[0, 1000]`` timestep space, transformer
``forward_pass``, LoRA targets, i2v conditioning stash) but is standalone —
the HunyuanVideo 1.5 transformer contract is different enough (65-channel
concat input, dual text streams, mandatory ``image_embeds``) that sharing the
WAN base would obscure both.

Verified against the installed diffusers 0.39 sources
(``pipelines/hunyuan_video1_5/`` + ``models/transformers/
transformer_hunyuan_video15.py``) and the hub checkpoint configs
(``hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_{t2v,i2v}``):

- **65-channel input contract**: the transformer consumes
  ``cat([latents(32), cond_latents(32), mask(1)], dim=1)``. For T2V the cond
  latents + mask are ZEROS (``HunyuanVideo15Pipeline.prepare_cond_latents_and_mask``);
  for I2V the cond carries the first-frame latent at temporal slot 0 (frames
  1: zeroed) and the mask is 1.0 on frame 0
  (``HunyuanVideo15ImageToVideoPipeline.prepare_cond_latents_and_mask``).
- **image_embeds is MANDATORY**: the transformer runs
  ``self.image_embedder(image_embeds)`` unconditionally and detects T2V via
  ``torch.all(image_embeds == 0)`` — so T2V must feed zeros ``(B, 729, 1152)``
  (pipeline lines 719-725) and I2V feeds the Siglip ``last_hidden_state``.
- **Raw [0, 1000] timestep**: ``HunyuanVideo15TimeEmbedding`` is a sinusoidal
  ``Timesteps`` projection over the raw FlowMatchEuler value (no internal
  /1000) — the ``/1000`` lerp lives in the REAL training path's
  ``add_noise`` (the base ``PipelineBaseMixin.add_noise`` → shared
  ``NoiseInterpolation('linear')`` component, the WAN pure-noise gotcha; see
  ``wan_shared/driver_base.py``). Unlike WAN/LTX-2/Kandinsky5, hv15 has NO
  driver-level ``add_noise`` override: i2v conditioning never touches which
  frames get noised — it is carried entirely by the separate cond/mask
  channels built in :func:`build_model_input` below, so every frame
  (including frame 0) is noised uniformly and the base component's generic
  lerp is exactly correct as-is (proven equivalent + pinned in
  ``test_hv15_addnoise_wiring.py``).
- **Dual TE**: Qwen2.5-VL chat-template encoding (``hidden_states[-3]``, crop
  the first 108 template tokens) + a ByT5 glyph channel fed from QUOTED
  substrings of the prompt (zero embeddings + zero mask when the prompt has no
  quotes — the common training-caption case).
"""

from __future__ import annotations

import re
from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver

logger = structlog.get_logger(__name__)

# ── Channel / token layout constants (hub checkpoint config) ───────────────
NOISE_CHANNELS = 32
COND_CHANNELS = 32
MASK_CHANNELS = 1
IN_CHANNELS = NOISE_CHANNELS + COND_CHANNELS + MASK_CHANNELS  # 65

VISION_NUM_SEMANTIC_TOKENS = 729  # Siglip (384/14)^2 patches
IMAGE_EMBED_DIM = 1152

# ── Qwen2.5-VL text-encoding constants (pipeline lines 226-238) ────────────
QWEN_TOKENIZER_MAX_LENGTH = 1000
QWEN_CROP_START = 108  # chat-template prefix tokens cropped from emb + mask
QWEN_HIDDEN_LAYERS_TO_SKIP = 2  # hidden_states[-(skip+1)] == hidden_states[-3]
BYT5_TOKENIZER_MAX_LENGTH = 256

# Byte-identical to ``HunyuanVideo15Pipeline.system_message`` — the line
# continuations preserve the upstream 9-space runs, and a test pins equality
# against the installed pipeline's default.
# fmt: off
HV15_SYSTEM_MESSAGE = "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video."
# fmt: on

# ── LoRA targets ────────────────────────────────────────────────────────────
# Per-block suffixes; PEFT list matching is ``endswith``-based, and the token
# refiner (``context_embedder.token_refiner.refiner_blocks.N.attn.to_q`` …)
# would ALSO match bare suffixes — so :func:`hv15_lora_target_paths` expands
# them to FULL ``transformer_blocks.{i}.*`` paths (refiner/embedders excluded
# by construction).
HV15_BLOCK_LORA_SUFFIXES: list[str] = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
]

HV15_NUM_LAYERS_DEFAULT = 54


def hv15_lora_target_paths(num_layers: int = HV15_NUM_LAYERS_DEFAULT) -> list[str]:
    """Full-path LoRA targets ``transformer_blocks.{i}.{suffix}``.

    Full paths (not bare suffixes) so the 2-layer token refiner — whose blocks
    carry the same ``attn.to_q``/``ff.net.*`` module names — is never wrapped.
    """
    return [
        f"transformer_blocks.{i}.{suffix}"
        for i in range(int(num_layers))
        for suffix in HV15_BLOCK_LORA_SUFFIXES
    ]


# ── Glyph (ByT5) text extraction — replicates pipeline extract_glyph_texts ──
_GLYPH_PATTERN = r"\"(.*?)\"|“(.*?)”"


def extract_glyph_text(prompt: str) -> str | None:
    """Extract quoted substrings and format the ByT5 glyph prompt.

    Byte-replicates ``pipelines.hunyuan_video1_5.extract_glyph_texts``:
    straight or curly double-quoted spans, de-duplicated (order-preserving)
    when more than one, formatted ``Text "x". Text "y". `` — or ``None`` when
    the prompt contains no quoted text (→ zero embeddings downstream).
    """
    matches = re.findall(_GLYPH_PATTERN, prompt)
    result = [m[0] or m[1] for m in matches]
    result = list(dict.fromkeys(result)) if len(result) > 1 else result
    if result:
        return ". ".join([f'Text "{text}"' for text in result]) + ". "
    return None


# ── Weight-free 65-channel input builders (unit-tested with tiny tensors) ──
def build_t2v_cond_and_mask(
    noisy_latents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """T2V conditioning: zero cond latents ``[B,32,F,H,W]`` + zero mask ``[B,1,F,H,W]``."""
    if noisy_latents.ndim != 5:
        raise ValueError(
            f"noisy_latents must be 5D [B, C, F, H, W], got {tuple(noisy_latents.shape)}"
        )
    b, c, f, h, w = noisy_latents.shape
    cond = torch.zeros(
        (b, c, f, h, w), device=noisy_latents.device, dtype=noisy_latents.dtype
    )
    mask = torch.zeros(
        (b, MASK_CHANNELS, f, h, w),
        device=noisy_latents.device,
        dtype=noisy_latents.dtype,
    )
    return cond, mask


def build_i2v_cond_and_mask(
    noisy_latents: torch.Tensor,
    first_frame_latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """I2V conditioning: first-frame latent at slot 0 + frame-0 mask.

    Mirrors ``HunyuanVideo15ImageToVideoPipeline.prepare_cond_latents_and_mask``:
    the cond tensor carries the (VAE-encoded, scaled) first-frame latent in
    temporal slot 0 with frames ``1:`` zeroed; the 1-channel mask is 1.0 on
    frame 0 and 0.0 elsewhere.

    Args:
        noisy_latents: ``[B, 32, F, H, W]`` — defines the output shapes.
        first_frame_latent: ``[B, 32, 1, H, W]`` (or ``[B, 32, H, W]``).
    """
    if noisy_latents.ndim != 5:
        raise ValueError(
            f"noisy_latents must be 5D [B, C, F, H, W], got {tuple(noisy_latents.shape)}"
        )
    if noisy_latents.shape[1] != NOISE_CHANNELS:
        raise ValueError(
            f"noisy_latents must have {NOISE_CHANNELS} channels, "
            f"got {noisy_latents.shape[1]}"
        )
    if first_frame_latent.ndim == 4:
        first_frame_latent = first_frame_latent.unsqueeze(2)
    if first_frame_latent.shape[2] != 1:
        first_frame_latent = first_frame_latent[:, :, :1, :, :]

    b, c, f, h, w = noisy_latents.shape
    cond = torch.zeros(
        (b, c, f, h, w), device=noisy_latents.device, dtype=noisy_latents.dtype
    )
    cond[:, :, 0, :, :] = first_frame_latent[:, :, 0, :, :].to(
        device=noisy_latents.device, dtype=noisy_latents.dtype
    )
    mask = torch.zeros(
        (b, MASK_CHANNELS, f, h, w),
        device=noisy_latents.device,
        dtype=noisy_latents.dtype,
    )
    mask[:, :, 0, :, :] = 1.0
    return cond, mask


def build_model_input(
    noisy_latents: torch.Tensor,
    cond_latents: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """``cat([latents(32), cond(32), mask(1)], dim=1)`` → 65-channel input."""
    out = torch.cat([noisy_latents, cond_latents, mask], dim=1)
    if out.shape[1] != IN_CHANNELS:
        raise ValueError(
            f"model input must have {IN_CHANNELS} channels, got {out.shape[1]}"
        )
    return out


def zero_image_embeds(
    batch: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """T2V ``image_embeds``: zeros ``(B, 729, 1152)`` — the transformer detects
    the all-zero stream (``torch.all(image_embeds == 0)``) and masks it out."""
    return torch.zeros(
        (batch, VISION_NUM_SEMANTIC_TOKENS, IMAGE_EMBED_DIM),
        device=device,
        dtype=dtype,
    )


# ── Text encoding (module-level so it unit-tests with fake TEs) ────────────
def encode_qwen_prompt(
    text_encoder: nn.Module,
    tokenizer: Any,
    captions: list[str],
    device: torch.device,
    dtype: torch.dtype,
    *,
    max_length: int = QWEN_TOKENIZER_MAX_LENGTH,
    crop_start: int = QWEN_CROP_START,
    system_message: str = HV15_SYSTEM_MESSAGE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen2.5-VL prompt encoding — replicates ``_get_mllm_prompt_embeds``.

    Chat template (system message + user prompt, generation prompt appended),
    padded/truncated to ``max_length + crop_start``; hidden layer
    ``hidden_states[-3]`` (skip 2); the first ``crop_start`` template tokens
    are cropped from BOTH the embeddings and the attention mask.

    Returns ``(emb [B, max_length, D], mask [B, max_length])``.
    """
    template = [
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": p if p else " "},
        ]
        for p in captions
    ]
    text_inputs = tokenizer.apply_chat_template(
        template,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        padding="max_length",
        max_length=max_length + crop_start,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device=device)
    attention_mask = text_inputs.attention_mask.to(device=device)

    with torch.no_grad():
        hidden = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states[-(QWEN_HIDDEN_LAYERS_TO_SKIP + 1)]

    if crop_start is not None and crop_start > 0:
        hidden = hidden[:, crop_start:]
        attention_mask = attention_mask[:, crop_start:]

    return hidden.to(dtype=dtype), attention_mask


def encode_byt5_prompt(
    text_encoder_2: nn.Module,
    tokenizer_2: Any,
    captions: list[str],
    device: torch.device,
    dtype: torch.dtype,
    *,
    max_length: int = BYT5_TOKENIZER_MAX_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ByT5 glyph-channel encoding — replicates ``_get_byt5_prompt_embeds``.

    Quoted substrings are extracted per caption; a caption WITHOUT quoted text
    (the common training case) yields ZERO embeddings ``(1, 256, d_model)`` and
    a ZERO int64 mask — cached like any other embedding (cheap, and the
    transformer's valid-token reordering drops the masked tokens).

    Returns ``(emb [B, max_length, d_model], mask [B, max_length])``.
    """
    d_model = int(text_encoder_2.config.d_model)
    embeds_list: list[torch.Tensor] = []
    masks_list: list[torch.Tensor] = []

    for caption in captions:
        glyph_text = extract_glyph_text(caption)
        if glyph_text is None:
            embeds_list.append(
                torch.zeros((1, max_length, d_model), device=device, dtype=dtype)
            )
            masks_list.append(
                torch.zeros((1, max_length), device=device, dtype=torch.int64)
            )
            continue
        tokens = tokenizer_2(
            glyph_text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            emb = text_encoder_2(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask.float(),
            )[0]
        embeds_list.append(emb.to(device=device, dtype=dtype))
        masks_list.append(tokens.attention_mask.to(device=device))

    return torch.cat(embeds_list, dim=0), torch.cat(masks_list, dim=0)


class Hv15Driver(IModelDriver):
    """HunyuanVideo 1.5 family driver (480p T2V / I2V)."""

    # Batch keys for i2v extras (populated by ``attach_conditioning`` /
    # the trainer's ``build_batch_extra``).
    BATCH_FIRST_FRAME_LATENT = "hv15_first_frame_latent"
    BATCH_IMAGE_EMBED = "hv15_image_embed"

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.text_encoder_2: nn.Module | None = None
        self.tokenizer: Any = None
        self.tokenizer_2: Any = None
        self.image_encoder: nn.Module | None = None
        self.feature_extractor: Any = None
        self._components: dict[str, Any] = {}
        self._warned_missing_image_embed = False

        arch = getattr(definition, "architecture_params", {}) or {}
        self.mode: str = str(arch.get("mode", "t2v")).lower()
        self.is_i2v: bool = self.mode == "i2v"
        self.num_layers: int = int(
            arch.get("transformer.num_layers", HV15_NUM_LAYERS_DEFAULT)
        )
        self.te_max_length: int = int(arch.get("te.max_length", QWEN_TOKENIZER_MAX_LENGTH))
        self.te2_max_length: int = int(arch.get("te2.max_length", BYT5_TOKENIZER_MAX_LENGTH))

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        self._components = components
        self.transformer = components.get("unet")
        self.vae = components.get("vae")
        self.text_encoder = components.get("text_encoder")
        self.text_encoder_2 = components.get("text_encoder_2")
        self.tokenizer = components.get("tokenizer")
        self.tokenizer_2 = components.get("tokenizer_2")
        self.image_encoder = components.get("image_encoder")
        self.feature_extractor = components.get("feature_extractor")

        self.logger.info(
            "hv15_config",
            mode=self.mode,
            is_i2v=self.is_i2v,
            num_layers=self.num_layers,
            has_image_encoder=self.image_encoder is not None,
        )

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.transformer

    def get_text_encoders(self) -> dict[str, nn.Module]:
        result: dict[str, nn.Module] = {}
        if self.text_encoder is not None:
            result["text_encoder"] = self.text_encoder
        if self.text_encoder_2 is not None:
            result["text_encoder_2"] = self.text_encoder_2
        return result

    def get_lora_targets(self) -> list[str]:
        """Full-path block targets — definition enrichment overrides at runtime."""
        definition_targets = getattr(self.definition, "lora_targetable_modules", None)
        if definition_targets:
            self.logger.info(
                "lora_targets_from_definition", count=len(definition_targets)
            )
            return list(definition_targets)
        targets = hv15_lora_target_paths(self.num_layers)
        self.logger.info("lora_targets_full_path_defaults", count=len(targets))
        return targets

    def init_scheduler(self) -> Any:
        """Flow matching — no external training scheduler."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """Transformer + TEs load in bf16 (VAE dtype comes from the loader)."""
        return torch.bfloat16

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported for hv15 (frozen dual TE)."""
        return []

    # --- Phase 2: Text Encoding ---

    def encode_text(
        self, captions: list[str], dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dual-TE encode → ``(emb, mask, emb2, mask2)``.

        - ``emb``  ``[B, 1000, 3584]`` Qwen2.5-VL hidden_states[-3], crop 108.
        - ``mask`` ``[B, 1000]`` int64.
        - ``emb2`` ``[B, 256, 1472]`` ByT5 glyph embeddings (zeros w/o quotes).
        - ``mask2`` ``[B, 256]`` int64 (zeros w/o quotes).
        """
        emb, mask = encode_qwen_prompt(
            self.text_encoder,
            self.tokenizer,
            captions,
            self.device,
            dtype,
            max_length=self.te_max_length,
        )
        emb2, mask2 = encode_byt5_prompt(
            self.text_encoder_2,
            self.tokenizer_2,
            captions,
            self.device,
            dtype,
            max_length=self.te2_max_length,
        )
        return emb, mask, emb2, mask2

    # --- Phase 5: Training Loop Hooks ---

    def prepare_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Lift a 4D still latent to a 1-frame 5D clip (WAN precedent).

        Stills in a mixed stills+video run are cached 4D ``[B, C, H, W]``; the
        transformer unpacks ``b, c, f, h, w`` so they need a frame axis.
        """
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)
        return latents

    def attach_conditioning(self, batch: dict[str, Any], latents: torch.Tensor) -> None:
        """I2V: stash the clean first-frame latent (the clip's own frame 0).

        T2V is a no-op. The forward builds the ``[cond(32), mask(1)]`` channels
        from this stash. NOTE: the upstream pipeline encodes the conditioning
        image with ``sample_mode="argmax"``; training reuses the cached
        (sampled) latent's frame 0 instead — the WAN i2v precedent.

        F=1 STILL GUARD: a single still on an i2v run trains as t2v — there is no
        frame to predict beyond a conditioning frame, so stashing (and later
        leaking, via the ``cond`` channels with ``mask=1`` on the only frame) the
        still's own clean latent as the answer would be a degenerate,
        zero-information step (hv15 computes the loss uniformly over all frames
        with no frame-0 exclusion, so it never crashes — worse than ltx2/k5's
        loud NaN). Skip the stash; :meth:`forward_pass` takes the
        zeroed-conditioning t2v path when F=1. (WAN parity — commit 11173c2c; the
        ltx2/k5 ``_i2v_conditioning_engaged`` F>1 gate.)
        """
        if not self.is_i2v:
            return
        if self.BATCH_FIRST_FRAME_LATENT in batch:
            return
        lat = latents if latents.ndim == 5 else latents.unsqueeze(2)
        if lat.shape[2] <= 1:
            return
        batch[self.BATCH_FIRST_FRAME_LATENT] = lat[:, :, :1, :, :].detach().clone()

    # NOTE: no ``add_noise`` override here (dead-dispatch audit finding,
    # A0-4). The real training loop calls the TRAINER's ``self.add_noise``
    # (MRO-resolved, ``pipeline_train.py``), which ``Hv15Trainer`` does not
    # override — it resolves to ``PipelineBaseMixin.add_noise`` →
    # ``NoiseInterpolation('linear')``. A driver-level override used to live
    # here with the SAME formula (``t*noise + (1-t)*latents``, raw [0,1000]
    # scale) — proven algebraically identical to the base component and
    # never reached by the real path, so it was deleted rather than wired
    # (see ``test_hv15_addnoise_wiring.py``). Unlike WAN/LTX-2/Kandinsky5,
    # hv15's i2v conditioning never needs an add_noise special case — it is
    # carried entirely by the separate cond/mask channels concatenated in
    # ``build_model_input`` (all frames, including frame 0, are noised
    # uniformly).

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """hv15 transformer forward — velocity over the 32 noise channels.

        Builds the 65-channel input HERE (never folded into the latent the
        trainer noises): T2V → zero cond/mask + zero ``image_embeds``; I2V →
        first-frame cond/mask from the batch stash + the cached Siglip
        ``last_hidden_state`` (falls back to zeros with a warning if the
        Siglip cache is missing — the transformer then reads the image stream
        as inactive, the cond channels still condition).

        The timestep is passed RAW ``[0, 1000]`` — the hv15 time embedder is a
        sinusoidal ``Timesteps`` over the raw value (dividing here too would
        make the frozen embedder read every step as t≈0 → pure-noise samples).
        """
        emb, mask, emb2, mask2 = self._unpack_text(text_embeddings)

        if noisy_input.ndim != 5:
            raise ValueError(
                f"hv15 forward_pass expects 5D latents, got {tuple(noisy_input.shape)}"
            )

        if self.is_i2v and noisy_input.shape[2] > 1:
            first_frame = batch.get(self.BATCH_FIRST_FRAME_LATENT)
            if first_frame is None:
                raise ValueError(
                    "I2V forward_pass requires batch["
                    f"'{self.BATCH_FIRST_FRAME_LATENT}'] (first-frame latent)."
                )
            cond, mask_c = build_i2v_cond_and_mask(noisy_input, first_frame)
            image_embeds = batch.get(self.BATCH_IMAGE_EMBED)
            if image_embeds is None:
                if not self._warned_missing_image_embed:
                    self.logger.warning(
                        "hv15_i2v_missing_image_embed_fallback_zeros",
                        hint="Siglip embedding cache missing — image stream inactive",
                    )
                    self._warned_missing_image_embed = True
                image_embeds = zero_image_embeds(
                    noisy_input.shape[0], noisy_input.device, noisy_input.dtype
                )
            else:
                image_embeds = image_embeds.to(
                    device=noisy_input.device, dtype=noisy_input.dtype
                )
        else:
            # T2V, OR an F=1 STILL on an i2v run → train as t2v. There is no frame
            # to predict beyond a conditioning frame, so the cond(32) + mask(1)
            # channels are ZEROED (``mask=0`` ⇒ "denoise this frame", zero cond ⇒
            # "no reference") and ``image_embeds`` is the all-zero stream the
            # transformer detects (``torch.all(image_embeds == 0)``) and masks
            # out. This keeps the still's own clean latent from being handed back
            # as the answer — hv15's uniform (no frame-0-excluded) loss would
            # otherwise let the i2v path drive the loss to ~0 by copying it (a
            # degenerate, answer-leaked step). WAN parity (commit 11173c2c);
            # ltx2/k5's F>1 _i2v_conditioning_engaged gate.
            cond, mask_c = build_t2v_cond_and_mask(noisy_input)
            image_embeds = zero_image_embeds(
                noisy_input.shape[0], noisy_input.device, noisy_input.dtype
            )

        hidden_states = build_model_input(noisy_input, cond, mask_c)

        output = self.transformer(
            hidden_states=hidden_states,
            timestep=timesteps,
            encoder_hidden_states=emb,
            encoder_attention_mask=mask,
            encoder_hidden_states_2=emb2,
            encoder_attention_mask_2=mask2,
            image_embeds=image_embeds,
            return_dict=False,
        )
        return output[0] if isinstance(output, tuple) else output

    @staticmethod
    def _unpack_text(
        text_embeddings: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack the documented 4-tuple ``(emb, mask, emb2, mask2)``."""
        if isinstance(text_embeddings, tuple) and len(text_embeddings) == 4:
            return text_embeddings
        raise ValueError(
            "hv15 forward_pass requires the (emb, mask, emb2, mask2) 4-tuple "
            f"from encode_text, got {type(text_embeddings)}"
        )

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self) -> Any:
        from app.engine.models.families.hunyuan_video15.saver import Hv15Saver

        return Hv15Saver(mode=self.mode)

    def get_block_topology(self) -> list[dict[str, Any]]:
        """Single ``transformer_blocks`` stack for the VRAM-management UI."""
        topology: list[dict[str, Any]] = []
        model = self.get_primary_model()
        if model is not None:
            blocks = getattr(model, "transformer_blocks", None)
            if blocks is not None:
                topology.append(
                    {
                        "name": "transformer_blocks",
                        "attr_path": "transformer_blocks",
                        "count": len(blocks),
                        "approx_vram_mb": 320,
                    }
                )
        return topology
