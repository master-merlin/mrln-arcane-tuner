"""BooguImageDriver — family-specific training behavior for Boogu-Image.

Task 2 scope (this file): a MINIMAL concrete ``IModelDriver`` subclass —
required so the suite-wide ``backend/tests/test_resolve_capabilities.py``
guard (which imports ``families.<family>.driver`` and constructs the driver
weight-free for every registered definition — see that module's docstring:
"Driver text-encoder LoRA introspection ... is *weight-free*: every driver
returns a hardcoded list and its ``__init__`` takes only
``(definition, device)`` without touching weights") stays green the moment
this family's two YAML definitions are registered.

Only the pieces that are (a) exercised by that guard (``__init__`` +
``get_te_lora_targets``) and (b) low-risk boilerplate common to every family
(component wiring, ``get_lora_targets`` reading the curated definition list,
``resolve_loading_dtype``) are real. Everything touching the actual forward
pass, Qwen3-VL text encoding, the vendored scheduler, and the saver is
genuine design work deferred to the tasks that own it (Tasks 3-7 — see
``.agent/workdir/sdd-boogu/task-2-brief.md``) and raises
``NotImplementedError`` with a pointer to that brief.
"""

from __future__ import annotations

from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver, IModelSaver
from app.engine.core.text_encoding import TextEncoderOutput


logger = structlog.get_logger(__name__)

_NOT_IMPLEMENTED_NOTE = (
    " — lands in a later task, see .agent/workdir/sdd-boogu/task-2-brief.md"
)


class BooguImageDriver(IModelDriver):
    """Boogu-Image family driver (Base / Turbo share the same transformer
    geometry — only the checkpoint repo and native sample defaults differ)."""

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Assigned by assign_components()
        self.model: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.processor: Any = None
        self._components: dict[str, Any] = {}

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded Boogu-Image components into driver state."""
        self._components = components
        self.model = components.get("unet")
        self.vae = components.get("vae")
        self.text_encoder = components.get("text_encoder")
        self.processor = components.get("processor")

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
        """Boogu-Image LoRA targets — the curated definition list.

        The definition YAML MUST ship a non-empty curated
        ``lora_targetable_modules`` (guarded by
        ``test_lora_target_lists_shipped.py`` and pinned exactly by
        ``app/engine/tests/test_boogu_image_definitions.py``); this driver
        trusts it and only falls back to a conservative attention-only
        pattern if a caller somehow constructs a definition without one
        (should never happen for a shipped YAML).
        """
        definition_targets = getattr(
            self.definition, "lora_targetable_modules", None,
        )
        if definition_targets:
            self.logger.info(
                "lora_targets_from_definition", count=len(definition_targets),
            )
            return definition_targets

        self.logger.warning("lora_targets_pattern_fallback_no_curated_list")
        return ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"]

    def init_scheduler(self) -> Any:
        raise NotImplementedError("boogu_image scheduler wiring" + _NOT_IMPLEMENTED_NOTE)

    def resolve_loading_dtype(self) -> torch.dtype:
        """Boogu-Image loads in bf16 (mllm + transformer + VAE all bf16)."""
        return torch.bfloat16

    # --- Phase 2: Text Encoding ---

    def encode_text(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        raise NotImplementedError(
            "boogu_image Qwen3-VL text encoding" + _NOT_IMPLEMENTED_NOTE
        )

    # --- Phase 4: Precision, LoRA Targets & Layer Manifest ---

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder (Qwen3-VL mllm) LoRA not supported for Boogu-Image."""
        return []

    # --- Phase 5: Training Loop Hooks ---

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        raise NotImplementedError("boogu_image forward pass" + _NOT_IMPLEMENTED_NOTE)

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self) -> IModelSaver:
        raise NotImplementedError("boogu_image saver" + _NOT_IMPLEMENTED_NOTE)
