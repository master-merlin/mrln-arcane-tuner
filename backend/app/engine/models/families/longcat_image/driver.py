"""LongCat-Image model driver — family-specific training behavior.

Implements ``IModelDriver`` for LongCat-Image (Flux-like double+single DiT).

Text encoding replicates ``LongCatImagePipeline._encode_prompt`` exactly:
quotation-aware tokenization (``split_quotation``: quoted spans are tokenized
per character for glyph rendering), middle segment padded to
``tokenizer_max_length`` (512), wrapped in the captioning-expert
prefix/suffix chat template, ``hidden_states[-1]`` with the prefix/suffix
rows sliced off.  The reference pipeline's prompt-REWRITE step
(``rewire_prompt`` — a Qwen2.5-VL ``generate`` call) is deliberately NOT
replicated: training captions are ground truth.
"""

from __future__ import annotations

import re
from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver
from app.engine.core.text_encoding import TextEncoderOutput


logger = structlog.get_logger(__name__)

# Qwen2.5-VL context window used by LongCatImagePipeline (tokenizer_max_length).
TOKENIZER_MAX_LENGTH = 512

# Prompt template literals from ``LongCatImagePipeline.__init__``.
PROMPT_TEMPLATE_PREFIX = (
    "<|im_start|>system\nAs an image captioning expert, generate a "
    "descriptive text prompt based on an image content, suitable for "
    "input to a text-to-image model.<|im_end|>\n<|im_start|>user\n"
)
PROMPT_TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def split_quotation(prompt: str, quote_pairs=None) -> list[tuple[str, bool]]:
    """Split *prompt* on single/double quote pairs (pipeline-verbatim).

    Returns ``[(segment, is_quoted), ...]``.  Quoted segments are later
    tokenized per character so the model renders the exact glyphs.
    Word-internal apostrophes (``it's``) are protected from matching.
    """
    word_internal_quote_pattern = re.compile(r"[a-zA-Z]+'[a-zA-Z]+")
    matches_word_internal = word_internal_quote_pattern.findall(prompt)
    mapping_word_internal: list[list[str]] = []

    for i, word_src in enumerate(set(matches_word_internal)):
        word_tgt = "longcat_$##$_longcat" * (i + 1)
        prompt = prompt.replace(word_src, word_tgt)
        mapping_word_internal.append([word_src, word_tgt])

    if quote_pairs is None:
        quote_pairs = [("'", "'"), ('"', '"'), ("‘", "’"), ("“", "”")]
    pattern = "|".join(
        re.escape(q1) + r"[^" + re.escape(q1 + q2) + r"]*?" + re.escape(q2)
        for q1, q2 in quote_pairs
    )
    parts = re.split(f"({pattern})", prompt)

    result: list[tuple[str, bool]] = []
    for part in parts:
        for word_src, word_tgt in mapping_word_internal:
            part = part.replace(word_tgt, word_src)
        if re.match(pattern, part):
            if len(part):
                result.append((part, True))
        else:
            if len(part):
                result.append((part, False))
    return result


class LongCatImageDriver(IModelDriver):
    """LongCat-Image family driver.

    Handles:
    - Single TE assignment (Qwen2.5-VL in text-only mode — same class
      as qwen_image)
    - Double + single stream LoRA target patterns (Flux-like split)
    - Flow-matching scheduler (None)
    - Packed-latent forward pass (2×2 patchify → 3-axis RoPE ids)
    """

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Assigned by assign_components()
        self.model: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.tokenizer: Any = None
        self._components: dict[str, Any] = {}

        # Architecture params
        self.max_length: int = TOKENIZER_MAX_LENGTH

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded LongCat-Image components and cache architecture params."""
        self._components = components
        self.model = components["unet"]
        self.vae = components["vae"]
        self.text_encoder = components.get(
            "text_encoder", getattr(self, "text_encoder", None),
        )
        self.tokenizer = components.get(
            "tokenizer", getattr(self, "tokenizer", None),
        )

        arch = getattr(self.definition, "architecture_params", {}) or {}
        self.max_length = int(arch.get("te.max_length", TOKENIZER_MAX_LENGTH))

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.model

    def get_text_encoders(self) -> dict[str, nn.Module]:
        result: dict[str, nn.Module] = {}
        if self.text_encoder is not None:
            result["text_encoder"] = self.text_encoder
        return result

    def get_lora_targets(self) -> list[str]:
        """LongCat-Image LoRA targets — double + single stream projections.

        Same double/single precedent as flux1: the ``proj_out`` suffix also
        matches the transformer's top-level output projection (intentional —
        identical to the flux1 family's behavior).
        """
        definition_targets = getattr(
            self.definition, "lora_targetable_modules", None,
        )
        if definition_targets and len(definition_targets) > 0:
            self.logger.info(
                "lora_targets_from_definition",
                count=len(definition_targets),
            )
            return definition_targets

        self.logger.info("lora_targets_pattern_defaults")
        return [
            # Double stream (LongCatImageTransformerBlock)
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
            "attn.add_q_proj", "attn.add_k_proj",
            "attn.add_v_proj", "attn.to_add_out",
            "ff.net.0.proj", "ff.net.2",
            "ff_context.net.0.proj", "ff_context.net.2",
            # Single stream (LongCatImageSingleTransformerBlock) — attn.to_q/k/v
            # patterns above also match here (pre_only attention, no to_out)
            "proj_mlp", "proj_out",
        ]

    def init_scheduler(self) -> Any:
        """LongCat-Image uses flow matching — no external scheduler."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """LongCat-Image loads in bf16."""
        return torch.bfloat16

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported for LongCat-Image."""
        return []

    def get_layer_manifest(self) -> Any:
        """Layer manifest with double (joint) + single transformer blocks."""
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
        """Encode captions via Qwen2.5-VL, LongCat template semantics."""
        return self.encode_longcat(captions, dtype)

    def encode_longcat(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Replicate ``LongCatImagePipeline._encode_prompt``.

        1. Quotation-aware tokenization: quoted spans per character.
        2. Pad the caption segment to ``max_length`` (padding="max_length" —
           the reference passes NO attention mask to the transformer, so
           padded rows intentionally participate downstream).
        3. Wrap with the always-attended prefix/suffix template tokens.
        4. TE forward with ``output_hidden_states=True`` → ``hidden_states[-1]``.
        5. Slice off the prefix/suffix rows → ``[B, max_length, D]``.

        Returns:
            ``TextEncoderOutput`` with ``embeddings [B, max_length, D]`` and
            ``attention_mask [B, max_length]`` (the caption-segment mask —
            kept for cache/metadata; the transformer forward does not use it).
        """
        batch_all_tokens: list[list[int]] = []
        for caption in captions:
            all_tokens: list[int] = []
            for segment, quoted in split_quotation(caption):
                if quoted:
                    for sub_word in segment:
                        tokens = self.tokenizer(
                            sub_word, add_special_tokens=False,
                        )["input_ids"]
                        all_tokens.extend(tokens)
                else:
                    tokens = self.tokenizer(
                        segment, add_special_tokens=False,
                    )["input_ids"]
                    all_tokens.extend(tokens)

            if len(all_tokens) > self.max_length:
                self.logger.warning(
                    "caption_truncated",
                    max_length=self.max_length,
                    token_count=len(all_tokens),
                )
                all_tokens = all_tokens[: self.max_length]
            batch_all_tokens.append(all_tokens)

        text_tokens_and_mask = self.tokenizer.pad(
            {"input_ids": batch_all_tokens},
            max_length=self.max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        prefix_tokens = self.tokenizer(
            PROMPT_TEMPLATE_PREFIX, add_special_tokens=False,
        )["input_ids"]
        suffix_tokens = self.tokenizer(
            PROMPT_TEMPLATE_SUFFIX, add_special_tokens=False,
        )["input_ids"]
        prefix_len = len(prefix_tokens)
        suffix_len = len(suffix_tokens)

        ids_dtype = text_tokens_and_mask.input_ids.dtype
        mask_dtype = text_tokens_and_mask.attention_mask.dtype
        batch_size = text_tokens_and_mask.input_ids.size(0)

        prefix_batch = torch.tensor(prefix_tokens, dtype=ids_dtype).unsqueeze(0).expand(batch_size, -1)
        suffix_batch = torch.tensor(suffix_tokens, dtype=ids_dtype).unsqueeze(0).expand(batch_size, -1)
        prefix_mask = torch.ones(batch_size, prefix_len, dtype=mask_dtype)
        suffix_mask = torch.ones(batch_size, suffix_len, dtype=mask_dtype)

        input_ids = torch.cat(
            (prefix_batch, text_tokens_and_mask.input_ids, suffix_batch), dim=-1,
        ).to(self.device)
        attention_mask = torch.cat(
            (prefix_mask, text_tokens_and_mask.attention_mask, suffix_mask), dim=-1,
        ).to(self.device)

        with torch.no_grad():
            text_output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        prompt_embeds = text_output.hidden_states[-1].detach()
        prompt_embeds = prompt_embeds[:, prefix_len:-suffix_len, :]

        return TextEncoderOutput(
            embeddings=prompt_embeds.to(dtype=dtype),
            attention_mask=text_tokens_and_mask.attention_mask.to(self.device),
        )

    # --- Positional ids (pipeline ``prepare_pos_ids``) ---

    @staticmethod
    def _prepare_text_ids(num_token: int, device: torch.device) -> torch.Tensor:
        """Text RoPE ids — modality 0, rows ``(0, i, i)``."""
        pos_ids = torch.zeros(num_token, 3, device=device)
        pos_ids[:, 1] = torch.arange(num_token, device=device)
        pos_ids[:, 2] = torch.arange(num_token, device=device)
        return pos_ids

    @staticmethod
    def _prepare_image_ids(
        height: int, width: int, start: int, device: torch.device,
    ) -> torch.Tensor:
        """Image RoPE ids — modality 1, offset by the text window ``start``."""
        pos_ids = torch.zeros(height, width, 3, device=device)
        pos_ids[..., 0] = 1
        pos_ids[..., 1] += torch.arange(height, device=device)[:, None] + start
        pos_ids[..., 2] += torch.arange(width, device=device)[None, :] + start
        return pos_ids.reshape(height * width, 3)

    # --- Phase 5: Training Loop Hooks ---

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """LongCatImageTransformer2DModel forward with pack/unpack.

        Divides the raw ``[0, 1000]`` timesteps by 1000 exactly ONCE — the
        transformer multiplies ×1000 internally before the time embedding
        (the pipeline passes ``timestep / 1000`` the same way).

        Args:
            noisy_input: Noisy latents ``[B, C, H, W]``.
            timesteps: Scaled timesteps ``[0, 1000]``.
            text_embeddings: ``(embeddings, attention_mask)`` tuple or tensor.
            batch: Full batch dict.

        Returns:
            Model prediction ``[B, C, H, W]``.
        """
        if isinstance(text_embeddings, tuple):
            enc_hs, _enc_mask = text_embeddings
        else:
            enc_hs = text_embeddings

        B, C, H, W = noisy_input.shape
        model = self.get_primary_model()

        # Pack: [B, C, H, W] → [B, (H/2)*(W/2), C*4] (pipeline _pack_latents)
        x = noisy_input.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, (H // 2) * (W // 2), C * 4)

        model_timesteps = timesteps / 1000.0

        # RoPE ids: text window then image window offset by tokenizer_max_length
        # (the pipeline starts image ids at (512, 512) regardless of prompt).
        txt_ids = self._prepare_text_ids(enc_hs.shape[1], self.device)
        img_ids = self._prepare_image_ids(
            H // 2, W // 2, self.max_length, self.device,
        )

        output = model(
            hidden_states=x,
            encoder_hidden_states=enc_hs,
            timestep=model_timesteps,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )
        pred = output[0] if isinstance(output, tuple) else output

        # Unpack: [B, (H/2)*(W/2), C*4] → [B, C, H, W]
        pred = pred.view(B, H // 2, W // 2, C, 2, 2)
        pred = pred.permute(0, 3, 1, 4, 2, 5)
        return pred.reshape(B, C, H, W)

    # compute_target — inherited default: flow-match velocity ``noise - latents``
    # (matches FlowMatchEulerDiscreteScheduler.step in the reference pipeline).

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self):
        """Return LongCat-Image ai-toolkit-format LoRA saver."""
        from app.engine.models.families.longcat_image.saver import (
            LongCatImageSaver,
        )

        return LongCatImageSaver()

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """LongCat-Image block topology: double (joint) + single stream."""
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
