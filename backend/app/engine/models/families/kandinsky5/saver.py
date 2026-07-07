"""Kandinsky 5.0 LoRA saver + upstream-mixin key mapping.

Saved key format (ai-toolkit / ComfyUI convention, via ``GenericLoRASaver``)::

    diffusion_model.visual_transformer_blocks.{i}.{module}.lora_A/B.weight

Kandinsky 5.0 is the rare family with a REAL upstream LoRA mixin:
``diffusers.loaders.KandinskyLoraLoaderMixin`` (SD3-style, native diffusers
PEFT keys)::

    transformer.visual_transformer_blocks.{i}.{module}.lora_A/B.weight

The two formats differ ONLY by prefix, so :func:`to_diffusers_lora` /
:func:`from_diffusers_lora` provide a lossless bidirectional mapping —
``to_diffusers_lora(sd)`` produces a dict that
``Kandinsky5T2VPipeline.load_lora_weights`` (which routes into
``transformer.load_lora_adapter`` with ``prefix="transformer"``) accepts
directly. Pinned by ``test_kandinsky5_lora_portability.py``.
"""

from __future__ import annotations

import torch

from app.engine.core.pipeline.saver_base import GenericLoRASaver

# Our on-disk prefix (ai-toolkit / ComfyUI).
AI_TOOLKIT_PREFIX = "diffusion_model."
# The KandinskyLoraLoaderMixin native prefix (== mixin.transformer_name + ".").
DIFFUSERS_PREFIX = "transformer."


class Kandinsky5Saver(GenericLoRASaver):
    """Saves Kandinsky 5.0 LoRA weights as ai-toolkit-format safetensors."""

    architecture_name = "kandinsky5"


def to_diffusers_lora(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Our saved format → KandinskyLoraLoaderMixin native format.

    ``diffusion_model.{path}.lora_A/B.weight`` →
    ``transformer.{path}.lora_A/B.weight`` — directly loadable through
    ``Kandinsky5{T2V,I2V}Pipeline.load_lora_weights`` /
    ``Kandinsky5Transformer3DModel.load_lora_adapter``.

    Keys without our prefix are passed through unchanged (defensive).
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(AI_TOOLKIT_PREFIX):
            key = DIFFUSERS_PREFIX + key[len(AI_TOOLKIT_PREFIX) :]
        out[key] = value
    return out


def from_diffusers_lora(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """KandinskyLoraLoaderMixin native format → our saved format (inverse)."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(DIFFUSERS_PREFIX):
            key = AI_TOOLKIT_PREFIX + key[len(DIFFUSERS_PREFIX) :]
        out[key] = value
    return out
