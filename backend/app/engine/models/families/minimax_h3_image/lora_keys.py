"""MiniMax-H3-Image: the diffusers module -> native module map of the LoRA file.

The checkpoint ships its transformer twice (``MiniMaxAI/MiniMax-H3-Image`` at ``fbde899f``):
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
