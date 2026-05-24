"""HiDream-O1 model driver — pixel-space Unified Transformer.

Unique among families: no separate VAE, no separate text encoder.
The unified model handles tokenization, text encoding, and image
generation in one pass. ``get_text_encoders`` returns an empty dict
and ``get_vae`` returns ``None`` — the base ``GenericTrainingPipeline``
is patched in Task 10 to tolerate this.

Architecture details (from spike_notes.md Task 3a):
- Model class: ``Qwen3VLForConditionalGeneration`` (Saganaki22's customized
  version with ``x_embedder``, ``final_layer2``, ``t_embedder1`` heads).
- Forward kwargs: ``input_ids``, ``position_ids``, ``vinputs`` (noisy
  patches), ``timestep``, ``token_types``, ``use_flash_attn``, ``use_sage_attn``.
- Output: ``Qwen3VLModelOutputWithPast.x_pred`` (per-patch x0 prediction).
"""

from __future__ import annotations

from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver, IModelSaver


class HiDreamO1Driver(IModelDriver):
    """Pixel-space unified transformer — single component, no VAE/TE."""

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        self.model: nn.Module | None = None
        self._components: dict[str, Any] = {}

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire the loaded unified model into the driver."""
        self._components = components
        self.model = components["unet"]
        self.logger.info(
            "hidream_o1.driver.components_assigned",
            keys=list(components.keys()),
        )

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        if self.model is None:
            raise RuntimeError(
                "HiDreamO1Driver has no model yet — call assign_components first.",
            )
        return self.model

    def get_text_encoders(self) -> dict[str, nn.Module]:
        """Intentional: no separate text encoder in this architecture."""
        return {}

    def get_vae(self) -> nn.Module | None:
        """Intentional: pixel-space model, no VAE."""
        return None

    def get_lora_targets(self) -> list[str]:
        """HiDream-O1 LoRA targets — all linear/attention modules."""
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
            "to_q", "to_k", "to_v", "to_out",
            "q_proj", "k_proj", "v_proj",
            "fc1", "fc2",
        ]

    def init_scheduler(self) -> Any:
        """HiDream-O1 uses flow matching — no external scheduler."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """HiDream-O1 loads in bf16."""
        return torch.bfloat16

    def encode_text(
        self,
        captions: list[str],
        dtype: torch.dtype,
    ) -> Any:
        """Encode text via the unified model (not a separate TE).

        HiDream-O1 has no standalone text encoder. The unified model
        handles text encoding internally during forward pass.
        This method is a placeholder for pipeline compatibility.
        """
        raise NotImplementedError(
            "HiDream-O1 has no standalone text encoder — "
            "text encoding is part of the unified forward pass.",
        )

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported (no separate TE)."""
        return []

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Forward pass through the unified HiDream-O1 model.

        This is a placeholder. The actual implementation will depend
        on the Qwen3VLForConditionalGeneration API from the vendor code.
        """
        raise NotImplementedError(
            "HiDreamO1Driver.forward_pass is intentionally not implemented. "
            "Use HiDreamO1Trainer.compute_loss(batch) instead — the recipe "
            "calls the model with custom kwargs (vinputs, timestep, "
            "token_types) that don't match the generic forward_pass shape.",
        )

    def get_saver(self) -> IModelSaver:
        """Return HiDream-O1 LoRA saver (ComfyUI-compatible kohya format)."""
        # Lazy import keeps the heavy safetensors import off the registry-
        # discovery path.
        from .saver import HiDreamO1Saver

        save_dtype = getattr(self.definition, "save_dtype", None) or "bf16"
        return HiDreamO1Saver(save_dtype=save_dtype)
