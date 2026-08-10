"""MiniMax-H3 model driver — PR0 SCAFFOLD STUB.

Task 6 owns the real driver surface: LoRA targets over the packed single-
stream attention/FFN blocks, the ``t = 1 - sigma`` / ``v = x0 - noise``
flow-match contract (INVERTED vs every other family here — see
``family.py``'s module docstring and the spec's PR1 notes), the joint
audio+video forward through the packed ``[text | conditions | audio |
video]`` sequence, and the AdaLN-excluding target list.

This stub exists ONLY so the registry-wide trainer/driver resolution guard
(``tests/engine/test_hook_wiring_meta.py::test_every_family_resolves_a_trainer_and_driver``)
has an honest, real ``IModelDriver`` subclass to resolve ``minimax_h3``'s
trainer against. Every method below is a placeholder that raises loudly
rather than returning a plausible-looking default — per the "failure is
never silent" invariant, a job that somehow reaches this driver must fail
with a clear "lands in PR1" message, not silently train on wrong data.

Keep this file small. Do not grow it ahead of Task 6.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver

_NOT_YET = (
    "minimax_h3 driver surface lands in PR1 (Task 6); PR0 ships the scaffold only."
)


class MiniMaxH3Driver(IModelDriver):
    """MiniMax-H3 driver — PR0 STUB. Full surface lands with Task 6."""

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        raise NotImplementedError(_NOT_YET)

    def get_components(self) -> dict[str, Any]:
        raise NotImplementedError(_NOT_YET)

    def get_primary_model(self) -> nn.Module:
        raise NotImplementedError(_NOT_YET)

    def get_text_encoders(self) -> dict[str, nn.Module]:
        raise NotImplementedError(_NOT_YET)

    def get_lora_targets(self) -> list[str]:
        raise NotImplementedError(_NOT_YET)

    def init_scheduler(self) -> Any:
        # ``None`` (not a raise) deliberately: this is IModelDriver's own
        # trivial baseline for the hook (see hook_dispatch.TRIVIAL_BODIES),
        # so the meta-guard does not mistake the stub for a real override
        # of the auto-delegating ``init_scheduler`` clobber hook. H3 trains
        # with flow matching (like every other family here), so this is
        # also the value Task 6 is expected to keep.
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        raise NotImplementedError(_NOT_YET)

    # --- Phase 2: Text Encoding ---

    def encode_text(self, captions: list[str], dtype: torch.dtype) -> Any:
        raise NotImplementedError(_NOT_YET)

    # --- Phase 4: Precision, LoRA Targets & Layer Manifest ---

    def get_te_lora_targets(self) -> list[str]:
        raise NotImplementedError(_NOT_YET)

    # --- Phase 5: Training Loop Hooks ---

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        raise NotImplementedError(_NOT_YET)

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self) -> Any:
        raise NotImplementedError(_NOT_YET)
