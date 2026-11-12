"""MiniMax-H3-Image: the diffusers module -> native module map of the LoRA file.

The checkpoint ships its transformer twice (``MiniMaxAI/MiniMax-H3-Image`` at ``00dd081e``):
``transformer_diffusers/`` under the vendored diffusers class's names, which PEFT trains
(``transformer_blocks.N.attn.to_q``), and ``transformer/`` under the model's native names,
which ComfyUI's repack and the rest of the ecosystem load (``blocks.N.attn.qkv_proj``). The LoRA
file is written in the native layout, so this module relates the two, module by module:

==========================================  ========================================  ==========
diffusers module (PEFT)                     native module                             transform
==========================================  ========================================  ==========
``transformer_blocks.N.attn.to_q|k|v``      ``blocks.N.attn.qkv_proj``                q / k / v
``transformer_blocks.N.attn.to_out.0``      ``blocks.N.attn.out_proj``                rename
``transformer_blocks.N.ff.net.0.proj``      ``blocks.N.mlp.fc1``                      fc1
``transformer_blocks.N.ff.net.2``           ``blocks.N.mlp.fc2``                      rename
``transformer_blocks.N.adaln_proj.linear``  ``blocks.N.adaln_proj.linear``            rename
``token_refiner.refiner_blocks.N.<same>``   ``token_refiner.blocks.N.<same>``         (as above)
==========================================  ========================================  ==========

A refiner block has no AdaLN projection. The per-step output heads (``pdd_proj_out``) and the
stem projections are not LoRA targets of this family and map to nothing.

The tensor work (fusing q/k/v, the SwiGLU half swap, the ``diffusion_model.`` prefix and the
scale folded into ``lora_B``) is the shared H3 core, ``minimax_h3_shared.lora_artifact``.
"""

from __future__ import annotations

import re
from typing import Any

from app.engine.models.families.minimax_h3_shared import lora_artifact as _core
from app.engine.models.families.minimax_h3_shared.lora_artifact import (
    ARTIFACT_PREFIX as ARTIFACT_PREFIX,
    Adapter as Adapter,
)

# Frozen once shipped: a reader may branch on it.
ARTIFACT_LAYOUT_VERSION = 1

_OWNER = "minimax_h3_image lora_keys"

_BLOCK = re.compile(r"^(?:(?P<refiner>token_refiner\.refiner_blocks)|transformer_blocks)\.(?P<n>\d+)\.(?P<rest>.+)$")

# diffusers block-relative module -> (native block-relative module, transform)
_BLOCK_MODULES: dict[str, tuple[str, str]] = {
    "attn.to_q": ("attn.qkv_proj", "q"),
    "attn.to_k": ("attn.qkv_proj", "k"),
    "attn.to_v": ("attn.qkv_proj", "v"),
    "attn.to_out.0": ("attn.out_proj", "rename"),
    "ff.net.0.proj": ("mlp.fc1", "fc1"),
    "ff.net.2": ("mlp.fc2", "rename"),
}
# Only the main blocks carry an AdaLN modulation table.
_MAIN_BLOCK_ONLY: dict[str, tuple[str, str]] = {
    "adaln_proj.linear": ("adaln_proj.linear", "rename"),
}


def native_module_for(diffusers_module: str) -> tuple[str, str]:
    """A diffusers LoRA module path -> ``(native module path, transform)``, the transform
    one of ``"q"`` / ``"k"`` / ``"v"`` (a slot of the fused ``qkv_proj``), ``"fc1"`` (the
    SwiGLU half swap) or ``"rename"``. Raises ``KeyError`` on a module no rule covers: a
    name is never invented."""
    match = _BLOCK.match(diffusers_module)
    if match is not None:
        refiner = match.group("refiner") is not None
        rules = _BLOCK_MODULES if refiner else {**_BLOCK_MODULES, **_MAIN_BLOCK_ONLY}
        rule = rules.get(match.group("rest"))
        if rule is not None:
            native_rest, transform = rule
            stack = "token_refiner.blocks" if refiner else "blocks"
            return f"{stack}.{match.group('n')}.{native_rest}", transform
    raise KeyError(f"no MiniMax-H3-Image LoRA rule maps diffusers module {diffusers_module!r}")


def adapters_from_peft_model(model: Any, adapter_name: str = "default") -> dict[str, Adapter]:
    """Every LoRA pair on the PEFT-wrapped transformer, keyed by diffusers module path, with
    each layer's own scaling (the shared core)."""
    return _core.adapters_from_peft_model(model, adapter_name, owner=_OWNER)


def build_artifact(adapters: dict[str, Adapter]) -> dict[str, Any]:
    """The LoRA file's tensors (float32; the saver casts) in the native layout, through the
    shared core with this family's :func:`native_module_for`."""
    return _core.build_artifact(adapters, native_module_for)


# ── The other direction: a native-layout file onto the training modules ──────

_NATIVE = re.compile(
    r"^(?:(?P<refiner>token_refiner\.blocks)|blocks)\.(?P<n>\d+)\."
    r"(?P<rest>attn\.qkv_proj|attn\.out_proj|mlp\.fc1|mlp\.fc2|adaln_proj\.linear)$"
)
# native block-relative module -> the diffusers block-relative modules it feeds
_NATIVE_TO_DIFFUSERS: dict[str, tuple[str, ...]] = {
    "attn.qkv_proj": ("attn.to_q", "attn.to_k", "attn.to_v"),
    "attn.out_proj": ("attn.to_out.0",),
    "mlp.fc1": ("ff.net.0.proj",),
    "mlp.fc2": ("ff.net.2",),
    "adaln_proj.linear": ("adaln_proj.linear",),
}


def training_modules_for(native_module: str) -> tuple[str, ...]:
    """A native module path -> the diffusers module paths PEFT trains for it (q, k, v for the
    fused ``qkv_proj``). Raises ``KeyError`` on a module no rule covers, a refiner block's
    AdaLN projection included (a refiner block has none)."""
    match = _NATIVE.match(native_module)
    if match is not None and not (match.group("refiner") and match.group("rest") == "adaln_proj.linear"):
        stack = "token_refiner.refiner_blocks" if match.group("refiner") else "transformer_blocks"
        return tuple(f"{stack}.{match.group('n')}.{rest}" for rest in _NATIVE_TO_DIFFUSERS[match.group("rest")])
    raise KeyError(f"no MiniMax-H3-Image LoRA rule maps native module {native_module!r}")


def training_adapter_from_native(tensors: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """A native-layout LoRA file (``diffusion_model.<native>.lora_A/B.weight``, an optional
    ``.alpha``) -> ``{diffusers module: (lora_A, lora_B)}`` with the scale folded into ``lora_B``,
    the key map the shared training adapter (``core/training_adapter.py``) loads through.

    The inverse of :func:`build_artifact`, exactly: the fused ``qkv_proj`` delta ``B @ A`` is
    split by ``B``'s rows in thirds (q, k, v), each third keeping the whole ``A``, so each
    projection's delta is its third of the fused one; ``fc1``'s rows go back from the native
    ``[gate; value]`` to diffusers' ``[value; gate]``. A key no rule covers raises."""
    import torch

    pairs: dict[str, dict[str, Any]] = {}
    for key, value in tensors.items():
        if not key.startswith(ARTIFACT_PREFIX):
            raise KeyError(f"{_OWNER}: {key!r} is not under {ARTIFACT_PREFIX!r}")
        body = key[len(ARTIFACT_PREFIX):]
        for suffix, slot in ((".lora_A.weight", "A"), (".lora_B.weight", "B"), (".alpha", "alpha")):
            if body.endswith(suffix):
                pairs.setdefault(body[: -len(suffix)], {})[slot] = value
                break
        else:
            raise KeyError(f"{_OWNER}: {key!r} is no LoRA tensor")

    out: dict[str, tuple[Any, Any]] = {}
    for native, pair in sorted(pairs.items()):
        if "A" not in pair or "B" not in pair:
            raise ValueError(f"{_OWNER}: incomplete LoRA pair on {native!r}")
        a = pair["A"].float()
        b = pair["B"].float()
        if "alpha" in pair:
            b = b * (float(pair["alpha"]) / a.shape[0])
        modules = training_modules_for(native)
        rest = native.rsplit(".", 2)[-2:]
        if len(modules) == 3:
            if b.shape[0] % 3:
                raise ValueError(f"{_OWNER}: {native} lora_B has {b.shape[0]} rows, not three equal thirds")
            for module, third in zip(modules, torch.chunk(b, 3, dim=0)):
                out[module] = (a.clone(), third.contiguous())
        elif rest == ["mlp", "fc1"]:
            half = b.shape[0] // 2
            out[modules[0]] = (a, torch.cat([b[half:], b[:half]], dim=0))  # [gate; value] -> [value; gate]
        else:
            out[modules[0]] = (a, b)
    return out
