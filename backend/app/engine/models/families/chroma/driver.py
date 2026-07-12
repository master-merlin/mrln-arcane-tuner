"""ChromaDriver — family-specific training behavior for Chroma.

Implements ``IModelDriver`` for lodestones' Chroma (Chroma1-Base /
Chroma1-HD): an 8.9B FLUX.1-schnell-derived DiT with CLIP removed (T5-XXL
only) and a distilled "pruned" adaLN modulation approach. Unlike its
schnell lineage, Chroma is NOT guidance-distilled — it uses REAL
classifier-free guidance (two full forward passes), confirmed from
``ChromaPipeline.__call__`` (diffusers 0.39,
``venv/Lib/site-packages/diffusers/pipelines/chroma/pipeline_chroma.py``):
``do_classifier_free_guidance = guidance_scale > 1`` (property, line 626-627)
and the denoise loop runs a COND + UNCOND transformer forward per step,
combining ``noise_pred = neg_noise_pred + guidance_scale * (noise_pred -
neg_noise_pred)`` (lines 920-933). The transformer itself
(``ChromaTransformer2DModel.forward``,
``venv/Lib/site-packages/diffusers/models/transformers/transformer_chroma.py``
lines 477-490) takes NO ``guidance`` kwarg at all (unlike
``FluxTransformer2DModel``) — the distilled-guidance approximator
(``self.distilled_guidance_layer``) only replaces Flux's per-block adaLN
modulation MLPs, it does NOT take an external guidance-scale input.

Chroma specifics mirrored here:
- Text encoder: T5-XXL ONLY (no CLIP, no pooled projections — the
  transformer config's ``pooled_projection_dim``/``guidance_embeds`` keys
  are VESTIGIAL leftovers from the Flux-schnell lineage; the pipeline's
  ``encode_prompt``/``forward`` never reference them — see YAML comments).
- Attention-mask "padding foot-gun": Chroma requires exactly ONE padding
  token to remain UNMASKED (``pipeline_chroma.py`` lines 249-252):
  ``mask_indices <= seq_lengths`` (note ``<=``, not ``<``) — one position
  PAST the real token count survives the mask. This differs from a naive
  padding mask and is a documented Chroma-specific quality requirement
  (see https://huggingface.co/lodestones/Chroma#tldr-masking-t5-padding-tokens).
  This modified mask (NOT the raw tokenizer mask used for the T5 forward
  itself) is what the TRANSFORMER consumes via joint attention
  (``ChromaTransformerBlock``/``ChromaSingleTransformerBlock`` multiply
  ``attention_mask[:, None, None, :] * attention_mask[:, None, :, None]``
  into the attention op) — extended with an all-ones mask for the image
  tokens (``pipeline_chroma.py`` ``_prepare_attention_mask``, lines
  599-615). We replicate BOTH masks and cache the modified one alongside
  the T5 embeddings (TE caching requirement).
- Transformer forward: packs latents 2×2 (reuses ``flux1.utils``, byte-
  identical to Flux's own packing — Chroma's blocks import
  ``FluxAttention``/``FluxAttnProcessor`` directly from
  ``transformer_flux.py``). ``txt_ids``/``img_ids`` are Flux-convention
  ALL-ZERO / grid ids (``pipeline_chroma.py`` line 324: ``text_ids =
  torch.zeros(prompt_embeds.shape[1], 3)`` — NOT Ovis's arange trick).
- Timestep scale: the driver passes ``timesteps / 1000.0`` — the
  transformer multiplies by 1000 internally
  (``transformer_chroma.py`` line 528: ``timestep = timestep.to(...) *
  1000``). Exactly once, never twice (flow-match timestep-scale gotcha).
"""

from __future__ import annotations

from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver
from app.engine.core.text_encoding import TextEncoderOutput


logger = structlog.get_logger(__name__)

# ``ChromaPipeline._get_t5_prompt_embeds`` / ``encode_prompt`` default.
_DEFAULT_T5_MAX_LENGTH = 512


class ChromaDriver(IModelDriver):
    """Chroma family driver (Chroma1-Base / Chroma1-HD).

    Handles:
    - Single T5-XXL text encoder (no CLIP anywhere)
    - Chroma's padding-survives-one-token attention mask
    - Flux-style packed flow-matching forward pass (no guidance kwarg)
    - Double + single stream LoRA targets (Flux-identical module names)
    """

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Assigned by assign_components()
        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.tokenizer: Any = None
        self._components: dict[str, Any] = {}

        # Architecture params
        arch = getattr(definition, "architecture_params", {}) or {}
        self.te_t5_max_length: int = int(
            arch.get("te.t5_max_length", _DEFAULT_T5_MAX_LENGTH),
        )

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded Chroma components and cache architecture params."""
        self._components = components
        self.transformer = components["unet"]
        self.vae = components.get("vae")
        self.text_encoder = components.get("text_encoder")
        self.tokenizer = components.get("tokenizer")

        self.logger.info(
            "chroma_config",
            t5_max_len=self.te_t5_max_length,
        )

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.transformer

    def get_text_encoders(self) -> dict[str, nn.Module]:
        """Return the single T5 text encoder — Chroma has no CLIP."""
        result: dict[str, nn.Module] = {}
        if self.text_encoder is not None:
            result["text_encoder"] = self.text_encoder
        return result

    def get_lora_targets(self) -> list[str]:
        """Chroma LoRA targets — diffusers FluxTransformerBlock modules.

        Chroma's blocks (``ChromaTransformerBlock``/
        ``ChromaSingleTransformerBlock``) reuse ``FluxAttention``/
        ``FluxAttnProcessor`` verbatim (``transformer_chroma.py`` line 33:
        ``from .transformer_flux import FluxAttention, FluxAttnProcessor``),
        so the module surface is byte-identical to flux1's. Mirrors
        ``Flux1Driver.get_lora_targets`` (closest lineage) rather than
        ovis_image's curated/excluded surface — Chroma's top-level
        ``proj_out`` collision (``transformer_chroma.py`` line 472) is left
        untargeted-but-uncurated exactly like FLUX.1's own top-level
        ``proj_out`` (``transformer_flux.py`` line 631), for consistency
        with the family this architecture is directly derived from.
        """
        return [
            # Double stream (ChromaTransformerBlock)
            "to_q", "to_k", "to_v", "to_out.0",
            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
            "ff.net.0.proj", "ff.net.2",
            "ff_context.net.0.proj", "ff_context.net.2",
            # Single stream (ChromaSingleTransformerBlock)
            "proj_mlp", "proj_out",
        ]

    def init_scheduler(self) -> Any:
        """Chroma uses flow matching — no external scheduler."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """Chroma loads in bf16."""
        return torch.bfloat16

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported for Chroma — T5 stays frozen."""
        return []

    def get_layer_manifest(self) -> Any:
        """Chroma layer manifest with joint + single transformer blocks."""
        from app.engine.core.layer_manifest import (
            BlockInfo,
            ModelLayerManifest,
        )

        blocks: list[BlockInfo] = []
        model = self.get_primary_model()
        if model is not None:
            joint = getattr(model, "transformer_blocks", None)
            if joint is not None:
                for i, block in enumerate(joint):
                    blocks.append(BlockInfo(
                        name=f"transformer_blocks.{i}",
                        block_type="joint",
                        param_count=sum(p.numel() for p in block.parameters()),
                        depth_index=i,
                    ))
            single = getattr(model, "single_transformer_blocks", None)
            if single is not None:
                offset = len(blocks)
                for i, block in enumerate(single):
                    blocks.append(BlockInfo(
                        name=f"single_transformer_blocks.{i}",
                        block_type="single",
                        param_count=sum(p.numel() for p in block.parameters()),
                        depth_index=offset + i,
                    ))

        return ModelLayerManifest(
            transformer_blocks=blocks,
            lora_targets=self.get_lora_targets(),
            te_lora_targets=self.get_te_lora_targets(),
        )

    # --- Phase 2: Text Encoding ---

    def encode_text(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Encode captions replicating ``ChromaPipeline._get_t5_prompt_embeds``
        (diffusers 0.39, ``pipeline_chroma.py`` lines 209-263).

        1. Tokenize ``padding="max_length"``, ``max_length=te_t5_max_length``
           (512), ``truncation=True``.
        2. Forward T5 WITH the RAW tokenizer attention mask (Chroma diverges
           from FLUX here: "unlike FLUX, Chroma uses the attention mask when
           generating the T5 embedding" — pipeline comment, line 240).
        3. Build a SEPARATE, modified mask for the transformer: all tokens
           strictly beyond ``seq_lengths`` are masked, but position
           ``seq_lengths`` itself (one PAST the last real token) stays
           unmasked — ``mask_indices <= seq_lengths`` (line 252), the
           documented Chroma padding foot-gun. This is the mask returned
           here (NOT the raw tokenizer mask), so caching + forward_pass see
           the exact mask the transformer consumes.

        Returns:
            ``TextEncoderOutput`` with T5 embeddings ``[B, L, 4096]`` and the
            modified transformer-ready ``attention_mask`` ``[B, L]``.
        """
        device = self.device
        batch_size = len(captions)

        text_inputs = self.tokenizer(
            captions,
            padding="max_length",
            max_length=self.te_t5_max_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        tokenizer_mask = text_inputs.attention_mask
        tokenizer_mask_device = tokenizer_mask.to(device)

        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                output_hidden_states=False,
                attention_mask=tokenizer_mask_device,
            )[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # Chroma padding mask: one padding token past the real content
        # survives unmasked (`<=`, not `<`) — see module + method docstring.
        seq_lengths = tokenizer_mask_device.sum(dim=1)
        mask_indices = (
            torch.arange(tokenizer_mask_device.size(1), device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        attention_mask = (mask_indices <= seq_lengths.unsqueeze(1)).to(
            dtype=dtype, device=device,
        )

        return TextEncoderOutput(embeddings=prompt_embeds, attention_mask=attention_mask)

    # --- Phase 5: Training Loop Hooks ---

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """ChromaTransformer2DModel forward: predict velocity.

        Mirrors ``ChromaPipeline.__call__``'s per-step transformer call
        (lines 909-918): packs latents, builds all-zero txt_ids + Flux-grid
        img_ids, extends the (already Chroma-modified) text attention mask
        with an all-ones image-token mask
        (``_prepare_attention_mask``, lines 599-615), and passes NO
        ``guidance`` kwarg — the transformer's ``forward`` signature has none.

        Args:
            noisy_input: Noisy latents ``[B, 16, H, W]``.
            timesteps: Flow-matching timesteps in ``[0, 1000]`` scale.
            text_embeddings: ``(embeddings, attention_mask)`` tuple (the
                trainer's ``encode_text`` contract) or a plain embeddings
                tensor (mask-less fallback).
            batch: Full batch dict (unused; interface compat).

        Returns:
            Velocity prediction ``[B, 16, H, W]``.
        """
        from app.engine.models.families.flux1.utils import (
            pack_latents,
            unpack_latents,
        )

        if isinstance(text_embeddings, tuple):
            enc_hs, txt_mask = text_embeddings
        else:
            enc_hs, txt_mask = text_embeddings, None

        B, C, H, W = noisy_input.shape

        packed, img_ids = pack_latents(noisy_input)
        img_ids = img_ids.to(device=noisy_input.device, dtype=enc_hs.dtype)

        # Flux/Chroma convention: txt_ids are all-zero (NOT Ovis's arange).
        txt_seq_len = enc_hs.shape[1]
        txt_ids = torch.zeros(
            txt_seq_len, 3, device=noisy_input.device, dtype=enc_hs.dtype,
        )

        # Extend the (already-modified) text mask with all-ones image tokens.
        full_attention_mask = None
        if txt_mask is not None:
            image_seq_len = packed.shape[1]
            ones_img = torch.ones(
                B, image_seq_len, device=noisy_input.device, dtype=txt_mask.dtype,
            )
            full_attention_mask = torch.cat([txt_mask, ones_img], dim=1)

        # Scale timesteps exactly once: [0, 1000] → [0, 1]. The transformer
        # multiplies by 1000 internally (transformer_chroma.py line 528).
        model_timesteps = timesteps / 1000.0

        model = self.get_primary_model()
        output = model(
            hidden_states=packed,
            encoder_hidden_states=enc_hs,
            timestep=model_timesteps,
            txt_ids=txt_ids,
            img_ids=img_ids,
            attention_mask=full_attention_mask,
            return_dict=False,
        )
        pred = output[0] if isinstance(output, tuple) else output

        return unpack_latents(pred, H, W)

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self):
        """Return the Chroma ai-toolkit-format LoRA saver."""
        from app.engine.models.families.chroma.saver import ChromaSaver

        return ChromaSaver()

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """Chroma block topology: double (joint) + single stream blocks."""
        topology = []
        model = self.get_primary_model()
        if model is not None:
            joint = getattr(model, "transformer_blocks", None)
            if joint is not None:
                topology.append({
                    "name": "double_blocks",
                    "attr_path": "transformer_blocks",
                    "count": len(joint),
                    "approx_vram_mb": 640,
                })
            single = getattr(model, "single_transformer_blocks", None)
            if single is not None:
                topology.append({
                    "name": "single_blocks",
                    "attr_path": "single_transformer_blocks",
                    "count": len(single),
                    "approx_vram_mb": 320,
                })
        return topology
