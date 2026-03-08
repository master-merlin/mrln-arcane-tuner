"""Targeted layer training — selective LoRA adapter freezing.

Allows users to specify which modules keep their LoRA adapters trainable.
All LoRA adapters on non-matching modules are frozen.

**Critical invariant:** Base model parameters are NEVER touched.
They are frozen by PEFT and must stay frozen.  This module ONLY
controls which LoRA adapters (``lora_A`` / ``lora_B``) are trainable.

The patterns originate from ``get_trainable_layer_names()`` which lists
base-model parameter names *before* PEFT wrapping.  After PEFT, names
gain a ``base_model.model.`` prefix and LoRA adapters add
``lora_A.weight`` / ``lora_B.weight`` suffixes.

For LoRA parameters, the parent module path is extracted and matched
against the user's patterns.

Usage::

    manager = TargetedLayerManager([".*attention.*", ".*ff\\.linear.*"])
    manager.apply(model)   # Freezes LoRA adapters except on attention + ff
"""

from __future__ import annotations

import re

import structlog
import torch.nn as nn

logger = structlog.get_logger(__name__)

# PEFT prefix injected by ``get_peft_model()``
_PEFT_PREFIX = "base_model.model."

# Suffixes that identify LoRA adapter parameters
_LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight",
                  ".lora_A.default.weight", ".lora_B.default.weight",
                  ".lora_embedding_A", ".lora_embedding_B")


def _strip_peft_prefix(name: str) -> str:
    """Remove the ``base_model.model.`` prefix added by PEFT."""
    if name.startswith(_PEFT_PREFIX):
        return name[len(_PEFT_PREFIX):]
    return name


def _get_lora_parent_path(name: str) -> str | None:
    """Extract the parent module path from a LoRA parameter name.

    E.g. ``base_model.model.blocks.0.attn.to_q.lora_A.weight``
    → ``blocks.0.attn.to_q``

    Returns None if this is not a LoRA parameter.
    """
    stripped = _strip_peft_prefix(name)
    for suffix in _LORA_SUFFIXES:
        if stripped.endswith(suffix):
            parent = stripped[: -len(suffix)]
            return parent.rstrip(".")
    return None


class TargetedLayerManager:
    """Selective LoRA adapter freezing based on regex patterns.

    Only controls which LoRA adapters remain trainable.
    Base model parameters are NEVER modified.

    Args:
        target_patterns: Regex strings matched against module names.
            Only LoRA adapters on matching modules keep
            ``requires_grad=True``.  If ``None`` or empty, ``apply()``
            is a no-op.
    """

    def __init__(self, target_patterns: list[str] | None = None) -> None:
        self.target_patterns = target_patterns or []

    def apply(self, module: nn.Module) -> None:
        """Selectively freeze LoRA adapters on non-matching modules.

        Base model parameters are NEVER touched — they stay exactly
        as PEFT left them (frozen).

        Args:
            module: The PEFT-wrapped ``nn.Module``.
        """
        if not self.target_patterns:
            return

        compiled = [re.compile(p) for p in self.target_patterns]

        lora_matched = 0
        lora_frozen = 0
        base_skipped = 0

        for name, param in module.named_parameters():
            # Check if this is a LoRA adapter parameter
            parent_path = _get_lora_parent_path(name)

            if parent_path is None:
                # Base model parameter — DO NOT TOUCH.
                # PEFT has already set requires_grad correctly.
                base_skipped += 1
                continue

            # LoRA param: match the parent module path against patterns.
            # Patterns were generated from base param names like
            # "blocks.0.attn.to_q.weight", so also try with ".weight".
            parent_with_weight = parent_path + ".weight"
            if any(p.search(parent_path) or p.search(parent_with_weight)
                   for p in compiled):
                param.requires_grad_(True)
                lora_matched += 1
            else:
                param.requires_grad_(False)
                lora_frozen += 1

        logger.info(
            "targeted_training_applied",
            patterns=len(self.target_patterns),
            lora_trainable=lora_matched,
            lora_frozen=lora_frozen,
            base_params_untouched=base_skipped,
        )

        if lora_matched == 0:
            logger.error(
                "targeted_training_zero_lora_matched",
                patterns=self.target_patterns,
                hint="No LoRA adapters matched the target patterns. "
                     "Training will fail with an empty optimizer. "
                     "Check that patterns match base model module names.",
            )
