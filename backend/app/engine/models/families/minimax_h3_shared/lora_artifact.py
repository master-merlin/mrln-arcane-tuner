"""PEFT adapters -> the native-layout MiniMax-H3 LoRA artifact (the shared core).

The artifact every H3 producer emits and ComfyUI consumes, and what diffusers
converts back (``loaders/lora_conversion_utils.py``): native module names
under a ``diffusion_model.`` prefix, ``lora_A`` / ``lora_B``, PEFT's scaling
FOLDED into ``lora_B``, no ``.alpha`` key, so a loader that applies scale 1
reproduces the trained delta exactly.

The per-tensor transforms (``build_artifact``):
  * ``q`` / ``k`` / ``v`` - three independent rank-r adapters on the separate
    projections become ONE fused ``attn.qkv_proj`` adapter of rank 3r:
    ``A_fused = [A_q; A_k; A_v]`` (3r x hidden) and ``B_fused``
    block-diagonal (3*inner x 3r), exact, no approximation; the consumer
    chunks ``lora_B`` in thirds and hands every third the whole ``lora_A``,
    so each third sees only its own rank columns (the zero blocks kill the
    others).
  * ``fc1`` - a fused SwiGLU input projection whose diffusers rows are
    ``[value; gate]`` and native rows ``[gate; value]``: a permutation of
    OUTPUT rows, so it touches ``lora_B`` alone.
  * ``rename`` - everything else is a name change.

Which diffusers module maps to which native module, and with which transform,
is model-specific: each family passes its own ``native_module_for``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ARTIFACT_PREFIX = "diffusion_model."

_QKV_SLOT = {"q": 0, "k": 1, "v": 2}

# diffusers module path -> (native module path, transform)
NativeModuleMap = Callable[[str], tuple[str, str]]


@dataclass(frozen=True)
class Adapter:
    """One PEFT LoRA pair on a diffusers module, with the scaling PEFT applies
    in its forward (``alpha / r``, or ``alpha / sqrt(r)`` under rsLoRA - read
    off the layer, never recomputed)."""

    lora_A: Any
    lora_B: Any
    scaling: float


def adapters_from_peft_model(model: Any, adapter_name: str = "default", *, owner: str) -> dict[str, Adapter]:
    """Every LoRA pair on a PEFT-wrapped transformer, keyed by the diffusers
    module path (``base_model.model.`` stripped), with each layer's own
    ``scaling``. Incomplete pairs and layers without a scaling are refused,
    the message led by ``owner``."""
    import torch
    from peft import get_peft_model_state_dict

    scalings: dict[str, float] = {}
    for name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and adapter_name in scaling and hasattr(module, "lora_A"):
            scalings[name.removeprefix("base_model.model.")] = float(scaling[adapter_name])

    pairs: dict[str, dict[str, Any]] = {}
    for key, value in get_peft_model_state_dict(model, adapter_name=adapter_name).items():
        if not isinstance(value, torch.Tensor):
            continue
        clean = key.removeprefix("base_model.model.")
        for suffix, slot in ((".lora_A.weight", "A"), (".lora_B.weight", "B")):
            if clean.endswith(suffix):
                pairs.setdefault(clean[: -len(suffix)], {})[slot] = value.detach()
                break

    adapters: dict[str, Adapter] = {}
    for module, pair in pairs.items():
        if "A" not in pair or "B" not in pair:
            raise ValueError(f"{owner}: incomplete LoRA pair on {module!r}")
        if module not in scalings:
            raise ValueError(f"{owner}: no PEFT scaling found for {module!r}")
        adapters[module] = Adapter(lora_A=pair["A"], lora_B=pair["B"], scaling=scalings[module])
    return adapters


def build_artifact(adapters: dict[str, Adapter], native_module_for: NativeModuleMap) -> dict[str, Any]:
    """The artifact tensors (float32; the saver casts) in the native layout.
    A projection missing from a fused triple (an adaptive rebuild narrowed
    the targets) contributes a zero block, so the fused delta stays exact."""
    import torch

    out: dict[str, Any] = {}
    fused: dict[str, dict[int, tuple[Any, Any]]] = {}
    for module, adapter in adapters.items():
        native, transform = native_module_for(module)
        a = adapter.lora_A.detach().float()
        b = adapter.lora_B.detach().float() * float(adapter.scaling)  # the fold
        if transform in _QKV_SLOT:
            fused.setdefault(native, {})[_QKV_SLOT[transform]] = (a, b)
            continue
        if transform == "fc1":
            half = b.shape[0] // 2
            b = torch.cat([b[half:], b[:half]], dim=0)  # [value; gate] -> [gate; value]
        elif transform != "rename":
            raise ValueError(f"unknown LoRA transform {transform!r} for {module!r}")
        out[f"{ARTIFACT_PREFIX}{native}.lora_A.weight"] = a.contiguous()
        out[f"{ARTIFACT_PREFIX}{native}.lora_B.weight"] = b.contiguous()

    for native, slots in fused.items():
        any_a, any_b = next(iter(slots.values()))
        inner, hidden = any_b.shape[0], any_a.shape[1]
        total_rank = sum(slots[i][0].shape[0] for i in range(3) if i in slots)
        a_fused = torch.cat([slots[i][0] for i in range(3) if i in slots], dim=0)
        b_fused = torch.zeros((3 * inner, total_rank), dtype=any_b.dtype)
        col = 0
        for i in range(3):
            if i not in slots:
                continue
            a_i, b_i = slots[i]
            r_i = a_i.shape[0]
            b_fused[i * inner : (i + 1) * inner, col : col + r_i] = b_i
            col += r_i
        assert a_fused.shape == (total_rank, hidden)
        out[f"{ARTIFACT_PREFIX}{native}.lora_A.weight"] = a_fused.contiguous()
        out[f"{ARTIFACT_PREFIX}{native}.lora_B.weight"] = b_fused.contiguous()
    return out
