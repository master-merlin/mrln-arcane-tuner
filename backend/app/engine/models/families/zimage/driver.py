"""Z-Image model driver — family-specific training behavior.

Implements ``IModelDriver`` for Z-Image (S3-DiT).
Phase 1 scope: loading-related methods only.
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


class ZImageDriver(IModelDriver):
    """Z-Image family driver (S3-DiT, single-stream).

    Handles:
    - Single TE assignment (AutoModelForCausalLM)
    - S3-DiT attention + feed-forward LoRA targets
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
        self.max_length: int = 512

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded Z-Image components and cache architecture params."""
        self._components = components
        self.model = components["unet"]
        self.vae = components["vae"]
        self.text_encoder = components.get(
            "text_encoder", getattr(self, "text_encoder", None),
        )
        self.tokenizer = components.get(
            "tokenizer", getattr(self, "tokenizer", None),
        )

        # Architecture params
        arch = getattr(self.definition, "architecture_params", {}) or {}
        self.max_length = int(arch.get("te.max_length", 512))

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.model

    def get_text_encoders(self) -> dict[str, nn.Module]:
        result: dict[str, nn.Module] = {}
        if self.text_encoder is not None:
            result["text_encoder"] = self.text_encoder
        return result

    def release_text_encoders(self) -> None:
        """Null the attr get_text_encoders() reads."""
        self.text_encoder = None

    def get_lora_targets(self) -> list[str]:
        """Z-Image S3-DiT LoRA targets."""
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
            "attention.to_q", "attention.to_k",
            "attention.to_v", "attention.to_out.0",
            "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
        ]

    def init_scheduler(self) -> Any:
        """Z-Image uses flow matching — no external scheduler."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """Z-Image loads in bf16."""
        return torch.bfloat16

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported for Z-Image."""
        return []

    def get_layer_manifest(self) -> Any:
        """Z-Image layer manifest with S3-DiT transformer blocks."""
        from app.engine.core.layer_manifest import (
            BlockInfo,
            ModelLayerManifest,
        )

        blocks: list[BlockInfo] = []
        model = self.get_primary_model()
        if model is not None:
            # S3-DiT blocks
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
        """Encode captions via variable-length Qwen3 encoding."""
        return self.encode_variable_length(captions, dtype)

    def encode_variable_length(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Qwen3 variable-length encoding (thinking mode).

        Uses hidden_states[-2] and returns only non-padding tokens
        per sample, producing variable-length tensors.

        Args:
            captions: Batch of caption strings.
            dtype: Target dtype.

        Returns:
            ``TextEncoderOutput`` with ``embeddings`` as
            ``list[Tensor]`` — ``[Li, D]`` per sample.
        """
        # 1. Apply Qwen3 chat template
        templated: list[str] = []
        for cap in captions:
            messages = [{"role": "user", "content": cap}]
            txt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            templated.append(txt)

        # 2. Tokenize
        text_inputs = self.tokenizer(
            templated, padding="max_length", max_length=self.max_length,
            truncation=True, return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attn_mask = text_inputs.attention_mask.to(self.device).bool()

        # 3. Forward — use hidden_states[-2] (second-to-last)
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
        hidden = outputs.hidden_states[-2]

        # 4. Extract non-padding tokens per sample
        embeddings_list: list[torch.Tensor] = []
        for i in range(len(hidden)):
            embeddings_list.append(hidden[i][attn_mask[i]].to(dtype=dtype))

        return TextEncoderOutput(embeddings=embeddings_list)

    # --- Phase 5: Training Loop Hooks ---

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """ZImageTransformer2DModel forward pass.

        Z-Image uses inverted timestep convention and per-sample list API.

        Args:
            noisy_input: Noisy latents ``[B, C, H, W]``.
            timesteps: Scaled timesteps ``[0, 1000]``.
            text_embeddings: List of per-sample embeddings ``[Li, D]``.
            batch: Full batch dict.

        Returns:
            Model prediction ``[B, C, H, W]``.
        """
        # Z-Image: t=1 → clean, t=0 → noise (inverted)
        model_timesteps = (1000.0 - timesteps) / 1000.0

        # Per-sample tensors with frame dim: [C, H, W] → [C, 1, H, W]
        x_list = [noisy_input[i].unsqueeze(1) for i in range(noisy_input.shape[0])]
        cap_list = text_embeddings

        model = self.get_primary_model()
        output = model(
            x=x_list,
            t=model_timesteps,
            cap_feats=cap_list,
            return_dict=False,
        )

        sample_list = output[0] if isinstance(output, tuple) else output
        # [C, 1, H, W] → [C, H, W], then stack to [B, C, H, W]
        return torch.stack([s.squeeze(1) for s in sample_list], dim=0)

    def compute_target(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Z-Image velocity: ``latents - noise`` (inverted convention)."""
        return latents - noise

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self):
        """Return Z-Image ai-toolkit-format LoRA saver."""
        from app.engine.models.families.zimage.saver import ZImageSaver

        return ZImageSaver()

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """Z-Image block topology: single block type."""
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



