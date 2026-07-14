"""Qwen-Image model driver — family-specific training behavior.

Implements ``IModelDriver`` for Qwen-Image (2512).
Phase 1 scope: loading-related methods only.
"""

from __future__ import annotations

import hashlib
from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver
from app.engine.core.text_encoding import TextEncoderOutput


logger = structlog.get_logger(__name__)

# From QwenImageTrainer — Qwen2.5-VL context window.  Matches the trainer's
# production value (``QwenImageTrainer.TOKENIZER_MAX_LENGTH``); the trainer also
# syncs ``driver.max_length`` in ``_assign_components`` so the delegated encode
# uses the run's exact context window.
TOKENIZER_MAX_LENGTH = 1024

# Prompt template (from QwenImagePipeline.__init__)
PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_DROP_IDX = 34  # system preamble tokens to drop


# Fingerprint of the pre-encode prompt transformation (the system-prompt chat
# template + the number of system-preamble tokens dropped), hashed so any future
# edit to either changes the fingerprint automatically. Exposed publicly so
# ``QwenImageTrainer`` can version its disk-cache key (``_TE_TEMPLATE_ID`` in
# trainer.py — the boogu_image precedent) WITHOUT reaching across the module
# boundary to reconstruct the template itself. A stale-cache hazard: the raw
# caption alone was previously the disk key, so a template edit would silently
# serve embeddings encoded under the OLD template (the dreamlite poisoned-cache
# incident class).
_TE_TEMPLATE_FINGERPRINT = hashlib.sha256(
    f"{PROMPT_TEMPLATE}\x00{PROMPT_TEMPLATE_DROP_IDX}".encode("utf-8")
).hexdigest()[:16]


def te_template_fingerprint() -> str:
    """Public fingerprint of this driver's pre-encode prompt transformation.

    Used by ``QwenImageTrainer`` to version its disk-cache key so a change to
    ``PROMPT_TEMPLATE`` / ``PROMPT_TEMPLATE_DROP_IDX`` produces fresh on-disk
    filenames instead of silently reusing embeddings encoded under the old
    template.
    """
    return _TE_TEMPLATE_FINGERPRINT


class QwenImageDriver(IModelDriver):
    """Qwen-Image family driver (2512).

    Handles:
    - Single TE assignment (Qwen2.5-VL in text-only mode)
    - Joint attention + MLP LoRA targets
    - Flow-matching scheduler (None)
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
        """Wire loaded Qwen-Image components."""
        self._components = components
        self.model = components["unet"]
        self.vae = components["vae"]
        self.text_encoder = components.get(
            "text_encoder", getattr(self, "text_encoder", None),
        )
        self.tokenizer = components.get(
            "tokenizer", getattr(self, "tokenizer", None),
        )

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
        """Qwen-Image LoRA targets — joint attention + MLP."""
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
            # Joint attention projections
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
            # Cross-attention
            "attn.add_q_proj", "attn.add_k_proj",
            "attn.add_v_proj", "attn.to_add_out",
            # MLP layers
            "img_mlp.net.0.proj", "img_mlp.net.2",
            "txt_mlp.net.0.proj", "txt_mlp.net.2",
            # Modulation layers (adaptive normalization)
            "img_mod.1", "txt_mod.1",
        ]

    def init_scheduler(self) -> Any:
        """Qwen-Image uses flow matching — no external scheduler."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """Qwen-Image loads in bf16."""
        return torch.bfloat16

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported for Qwen-Image."""
        return []

    def get_layer_manifest(self) -> Any:
        """Qwen-Image layer manifest with joint transformer blocks."""
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

        return ModelLayerManifest(
            transformer_blocks=blocks,
            lora_targets=self.get_lora_targets(),
            te_lora_targets=self.get_te_lora_targets(),
        )

    # --- Phase 2: Text Encoding ---

    def encode_text(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Encode captions via Qwen2.5-VL text-only mode."""
        return self.encode_qwen_vl(captions, dtype)

    def encode_qwen_vl(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Qwen2.5-VL text-only encoding with template + mask.

        1. Wrap each caption in system/user/assistant template
        2. Tokenize with ``max_length = 1024 + 34``
        3. Extract non-padding tokens via masked hidden
        4. Drop first 34 tokens (system preamble)
        5. Re-pad to max actual length in batch

        Args:
            captions: Batch of caption strings.
            dtype: Target dtype.

        Returns:
            ``TextEncoderOutput`` with embeddings ``[B, L, D]``
            and ``attention_mask [B, L]``.
        """
        # 1. Wrap in template
        txt = [PROMPT_TEMPLATE.format(cap) for cap in captions]

        # 2. Tokenize
        max_len = self.max_length + PROMPT_TEMPLATE_DROP_IDX
        text_inputs = self.tokenizer(
            txt, max_length=max_len, padding=True,
            truncation=True, return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attn_mask = text_inputs.attention_mask.to(self.device)

        # 3. Forward through text encoder
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states[-1]

        # 4. Extract non-padding tokens
        split_hs = self._extract_masked_hidden(hidden_states, attn_mask)

        # 5. Drop first 34 tokens (system preamble), re-pad
        split_hs = [e[PROMPT_TEMPLATE_DROP_IDX:] for e in split_hs]
        attn_mask_list = [
            torch.ones(e.size(0), dtype=torch.long, device=self.device)
            for e in split_hs
        ]
        max_seq_len = max(e.size(0) for e in split_hs)
        prompt_embeds = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
            for u in split_hs
        ])
        encoder_attn_mask = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
            for u in attn_mask_list
        ])

        return TextEncoderOutput(
            embeddings=prompt_embeds.to(dtype=dtype),
            attention_mask=encoder_attn_mask,
        )

    @staticmethod
    def _extract_masked_hidden(
        hidden_states: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Extract only non-padding tokens per batch element."""
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)

    # --- Phase 5: Training Loop Hooks ---

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """QwenImageTransformer2DModel forward pass with patchify/unpatchify.

        Args:
            noisy_input: Noisy latents ``[B, C, H, W]``.
            timesteps: Scaled timesteps ``[0, 1000]``.
            text_embeddings: ``(embeddings, attention_mask)`` tuple or tensor.
            batch: Full batch dict.

        Returns:
            Model prediction ``[B, C, H, W]``.
        """
        if isinstance(text_embeddings, tuple):
            enc_hs, enc_mask = text_embeddings
        else:
            enc_hs = text_embeddings
            enc_mask = None

        B, C, H, W = noisy_input.shape
        model = self.get_primary_model()
        patch_size = getattr(model.config, "patch_size", 2)

        # Patchify: [B, C, H, W] → [B, (H/p)*(W/p), C*p*p]
        pH = H // patch_size
        pW = W // patch_size
        x = noisy_input.reshape(B, C, pH, patch_size, pW, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, pH * pW, C * patch_size * patch_size)

        model_timesteps = timesteps / 1000.0
        img_shapes = [(1, pH, pW)] * B

        # diffusers 0.39 removed txt_seq_lens from the transformer forward —
        # encoder_hidden_states_mask alone carries the valid-token lengths.
        output = model(
            hidden_states=x,
            encoder_hidden_states=enc_hs,
            encoder_hidden_states_mask=enc_mask,
            timestep=model_timesteps,
            img_shapes=img_shapes,
            return_dict=False,
        )

        pred = output[0] if isinstance(output, tuple) else output

        # Unpatchify: [B, pH*pW, out_C*p*p] → [B, out_C, H, W]
        out_channels = getattr(model.config, "out_channels", C)
        pred = pred.reshape(B, pH, pW, out_channels, patch_size, patch_size)
        pred = pred.permute(0, 3, 1, 4, 2, 5)
        pred = pred.reshape(B, out_channels, H, W)

        return pred

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self):
        """Return Qwen-Image ai-toolkit-format LoRA saver."""
        from app.engine.models.families.qwen_image.saver import QwenImageSaver

        return QwenImageSaver()

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """Qwen-Image block topology: single block type."""
        topology = []
        model = self.get_primary_model()
        if model is not None:
            blocks = getattr(model, "transformer_blocks", None)
            if blocks is not None:
                topology.append({
                    "name": "transformer_blocks",
                    "attr_path": "transformer_blocks",
                    "count": len(blocks),
                    "approx_vram_mb": 400,
                })
        return topology



