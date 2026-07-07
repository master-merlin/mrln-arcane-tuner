"""Shared PRX LoRA target patterns — consumed by every PRX-architecture family.

The PRX transformer (``diffusers.PRXTransformer2DModel``) uses FUSED
attention projections — unlike every other family in this repo there is no
``to_q``/``to_k``/``to_v`` split:

- ``blocks.{i}.attention.img_qkv_proj``  (fused image QKV, hidden → 3×hidden)
- ``blocks.{i}.attention.txt_kv_proj``   (fused text KV,  hidden → 2×hidden)
- ``blocks.{i}.attention.to_out.0``      (attention output projection)
- ``blocks.{i}.gate_proj / up_proj / down_proj``  (gated-GELU MLP)

``modulation.lin`` (zero-init adaLN) is deliberately NOT targeted.

None of these suffixes collide with the transformer's top-level Linears
(``img_in``, ``txt_in``, ``time_in.*``, ``final_layer.*``), so no PEFT
``exclude_modules`` entry is needed — pinned by the verification helper
below and the family test suites. This holds for the pixel-space variant
too: its bottleneck ``img_in`` becomes ``img_in.0`` / ``img_in.1``, which
still matches no target suffix.

This module is family-agnostic: nothing here hardcodes a family name, so
both the latent ``prx`` family and the future pixel-space sibling consume
the same list.
"""

from __future__ import annotations

import torch.nn as nn

# Per-block LoRA target suffixes (PEFT suffix-matching semantics).
PRX_BLOCK_LORA_TARGETS: tuple[str, ...] = (
    "attention.img_qkv_proj",
    "attention.txt_kv_proj",
    "attention.to_out.0",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Number of LoRA-wrapped modules each PRXBlock contributes.
PRX_TARGETS_PER_BLOCK: int = len(PRX_BLOCK_LORA_TARGETS)


def get_prx_lora_targets() -> list[str]:
    """Return a fresh (mutable) copy of the PRX per-block target patterns."""
    return list(PRX_BLOCK_LORA_TARGETS)


def matching_linear_modules(
    model: nn.Module,
    targets: list[str] | tuple[str, ...] | None = None,
) -> dict[str, list[str]]:
    """Map each target pattern to the ``nn.Linear`` module names it matches.

    Mirrors PEFT's suffix-matching (``name == target`` or
    ``name.endswith("." + target)``). Used by tests to verify every pattern
    matches ≥1 real module and that no top-level module is swept in.
    """
    patterns = list(targets) if targets is not None else list(PRX_BLOCK_LORA_TARGETS)
    linear_names = [
        name for name, module in model.named_modules() if isinstance(module, nn.Linear)
    ]
    return {
        pattern: [
            name
            for name in linear_names
            if name == pattern or name.endswith("." + pattern)
        ]
        for pattern in patterns
    }
