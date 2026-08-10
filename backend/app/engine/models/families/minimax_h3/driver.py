"""MiniMax-H3 model driver — Task 6: non-training surface.

Implements the ``IModelDriver`` methods that do NOT require the real training
forward pass: component wiring, the curated LoRA target list, block topology,
and loading dtype — all sourced from the ``ModelDefinition`` (Task 4), the
single source of truth. Everything that belongs to the packed joint
audio+video forward, the ``t = 1 - sigma`` / ``v = x0 - noise`` INVERTED
flow-match contract (see ``family.py``'s module docstring), and LoRA saving
lands in PR1 and raises ``NotImplementedError`` naming it explicitly — per
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
``sample_timesteps``, ``set_te_cache``, ``init_scheduler``) — this driver
deliberately does not override ANY of them, even to raise, so they stay
"dead but harmless" (unreachable: ``forward_pass`` — and the trainer's
``_setup_family``, even earlier — already raise first) instead of tripping
the same trap.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver


def _lands_in_pr1(what: str) -> NotImplementedError:
    return NotImplementedError(
        f"minimax_h3 {what} lands in PR1; PR0 (Task 6) ships the "
        "non-training driver surface only."
    )


class MiniMaxH3Driver(IModelDriver):
    """MiniMax-H3 driver — non-training surface (Task 6).

    Handles:
    - Component wiring (tokenizer/processor, Qwen3-VL text encoder, visual
      VAE, audio VAE, vendored transformer) per the loader manifest (Task 5).
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

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded MiniMax-H3 components into driver state.

        Five components per the loader manifest (Task 5): ``tokenizer``
        (processor, never moved to device), ``text_encoder`` (Qwen3-VL-32B,
        cached then unloaded before the DiT loads), ``vae`` (visual),
        ``audio_vae`` (MONO — run once per stereo channel), ``transformer``
        (vendored; ``transformer`` or ``transformer_ref`` subfolder per
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

    # --- Phase 2: Text Encoding ---

    def encode_text(self, captions: list[str], dtype: torch.dtype) -> Any:
        raise _lands_in_pr1(
            "text encoding (Qwen3-VL, hidden_state_tap_index=50)"
        )

    # --- Phase 4: Precision, LoRA Targets & Layer Manifest ---

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported — Qwen3-VL-32B stays frozen.

        This is a definitive architectural answer, not a PR1 placeholder:
        ``family.py`` deliberately does NOT override ``supports_train_te``,
        so it inherits the ``latent_diffusion`` archetype's ``False``
        default — the 48 GB TE must never train.
        """
        return []

    # --- Phase 5: Training Loop Hooks ---

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        raise _lands_in_pr1(
            "the joint audio+video forward pass (packed [text | conditions "
            "| audio | video] sequence, inverted flow-match contract)"
        )

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
