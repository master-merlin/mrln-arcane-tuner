"""Custom LoRA Linear wrapper for HiDream-O1.

Mirrors Saganaki22's ``HiDreamO1LoRALinear`` (MIT). We don't use peft
because:
1. The ComfyUI ecosystem expects kohya-style key prefixes
   (``diffusion_model.<key>.{lora_down,lora_up,alpha}``), not peft-native
   (``base_model.model....lora_A/B.weight``).
2. peft's adapter abstraction adds machinery we don't need; a 50-line
   wrapper is clearer for a single-family use case.

This file is OURS, not vendored — we own the code style and conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = structlog.get_logger(__name__)


# Saganaki22's "aitoolkit" preset: exclude any module whose qualified
# name contains these substrings. Matches ai-toolkit's documented recipe.
LORA_EXCLUDED_SUBSTRINGS: tuple[str, ...] = ("lm_head", "patch_embed", "visual")


class HiDreamO1LoRALinear(nn.Module):
    """Wraps an ``nn.Linear`` (or linear-like) with a rank-r LoRA adapter.

    Frozen base + trainable ``lora_down``, ``lora_up``. Output:
    ``base(x) + (x @ down.T) @ up.T * (alpha / rank)``.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        lora_key: str,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base = base
        self.lora_key = lora_key
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.rank)
        self.dropout: nn.Module = (
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        for p in self.base.parameters():
            p.requires_grad = False

        dev = getattr(base, "weight", None)
        dev = dev.device if dev is not None else torch.device("cpu")
        self.lora_down = nn.Parameter(
            torch.empty(self.rank, base.in_features, device=dev, dtype=torch.float32),
        )
        self.lora_up = nn.Parameter(
            torch.zeros(base.out_features, self.rank, device=dev, dtype=torch.float32),
        )
        nn.init.kaiming_uniform_(self.lora_down, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_dtype = base_out.dtype if torch.is_floating_point(base_out) else x.dtype
        x_lora = self.dropout(x).to(lora_dtype)
        down = self.lora_down.to(device=x_lora.device, dtype=lora_dtype)
        up = self.lora_up.to(device=x_lora.device, dtype=lora_dtype)
        lora_out = F.linear(F.linear(x_lora, down), up)
        return base_out + lora_out.to(base_out.dtype) * self.scaling

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        yield self.lora_down
        yield self.lora_up


@dataclass
class LoRAInjectionResult:
    layers: list[HiDreamO1LoRALinear]
    skipped: list[str]


def _is_linear_like(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.Linear)
        or module.__class__.__name__.endswith("Linear")
        or (
            hasattr(module, "in_features")
            and hasattr(module, "out_features")
            and hasattr(module, "weight")
        )
    )


def _excluded(name: str) -> bool:
    return any(part in name for part in LORA_EXCLUDED_SUBSTRINGS)


def _normalize_lora_key(module_name: str) -> str:
    """Strip ``model.model.`` / ``model.`` prefix for the LoRA key."""
    for prefix in ("model.model.", "model."):
        if module_name.startswith(prefix):
            return module_name[len(prefix):]
    return module_name


def inject_lora_layers(
    root: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> LoRAInjectionResult:
    """Walk ``root`` and replace linear-like modules with LoRA-wrapped versions.

    Uses Saganaki22's "aitoolkit" preset: every linear-like layer EXCEPT
    those whose qualified name contains ``lm_head``, ``patch_embed``, or
    ``visual`` substrings.

    Returns the injected wrapper list (for parameter collection) and the
    list of names skipped because they failed the ``_is_linear_like`` /
    ``in_features``/``out_features`` check.
    """
    layers: list[HiDreamO1LoRALinear] = []
    skipped: list[str] = []

    candidates = []
    for name, module in list(root.named_modules()):
        if _excluded(name):
            continue
        if not _is_linear_like(module):
            continue
        if not (hasattr(module, "in_features") and hasattr(module, "out_features")):
            skipped.append(name)
            continue
        candidates.append((name, module))

    for full_name, base in candidates:
        parts = full_name.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf = parts[-1]

        wrapper = HiDreamO1LoRALinear(
            base,
            lora_key=_normalize_lora_key(full_name),
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        setattr(parent, leaf, wrapper)
        layers.append(wrapper)

    logger.info(
        "hidream_o1.lora.injected",
        injected=len(layers),
        skipped=len(skipped),
        rank=rank,
        alpha=alpha,
    )
    return LoRAInjectionResult(layers=layers, skipped=skipped)


def lora_parameters(layers: list[HiDreamO1LoRALinear]) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for layer in layers:
        params.extend(layer.trainable_parameters())
    return params
