"""FLUX.3 Image Klein LoRA saver (LANE-144 spec D5) -- BFL-native keys, ComfyUI-loadable.

Layout (the inverse of diffusers' ``_convert_non_diffusers_flux3_lora_to_diffusers``, PR #15046
``ec288418`` ``lora_conversion_utils.py:2554-2571``, which only renames module paths):

- every key is ``diffusion_model.<native module>.lora_A.weight`` / ``.lora_B.weight`` -- exactly two
  tensors per module, PEFT's suffix convention, no ``.alpha`` tensor, no qkv fusion;
- the PEFT scale ``alpha / rank`` is FOLDED into ``lora_B`` and ``ss_network_alpha`` records the rank,
  so the file describes itself and its delta at strength 1 equals PEFT's (the ``minimax_h3`` precedent,
  ``minimax_h3/lora_keys.py`` ``build_artifact``);
- metadata comes from the one helper (``app.engine.utils.lora_metadata``).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import structlog
import torch
from peft.utils import get_peft_model_state_dict

from app.engine.core.interfaces import ModelSaver
from app.engine.utils.lora_metadata import kohya_config_metadata, trigger_metadata
from app.engine.utils.safe_save import safe_save_file

logger = structlog.get_logger(__name__)

ARCHITECTURE = "flux3"

# diffusers module path -> BFL-native: the exact inverse of the converter's renames (source above).
# Prefix renames first, then ONE leaf rename; ``txt_mode_blocks.`` is unchanged in both directions.
_PREFIX_RENAMES = (
    ("mode_blocks.", "content_mode_blocks."),
    ("single_transformer_blocks.", "single_blocks."),
)
_LEAF_RENAMES = (
    (".attn.to_q", ".q_proj"),
    (".attn.to_k", ".k_proj"),
    (".attn.to_v", ".v_proj"),
    (".attn.to_out.0", ".attn_out"),
    (".ff.linear_in", ".mlp_in"),
    (".ff.linear_out", ".mlp_out"),
)

_SAVE_DTYPES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def diffusers_to_native_module(name: str) -> str:
    """One diffusers module path -> its BFL-native path; an unmapped LoRA module RAISES."""
    native = name
    # ``txt_mode_blocks.`` ends in ``mode_blocks.``: rename only a path that STARTS with it.
    for old, new in _PREFIX_RENAMES:
        if native.startswith(old):
            native = new + native[len(old):]
            break
    for old, new in _LEAF_RENAMES:
        if native.endswith(old):
            return native[: -len(old)] + new
    raise ValueError(f"flux3 saver: no native name for LoRA module {name!r}")


def build_native_artifact(
    peft_sd: dict[str, torch.Tensor], rank: int, alpha: float, use_rslora: bool = False,
) -> dict[str, torch.Tensor]:
    """PEFT state dict (diffusers names) -> native-key tensors with ``alpha / rank`` folded into B."""
    scale = alpha / math.sqrt(rank) if use_rslora else alpha / rank
    artifact: dict[str, torch.Tensor] = {}
    for key, value in peft_sd.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean = key.replace("base_model.model.", "").removeprefix("transformer.")
        module, sep, suffix = clean.rpartition(".lora_")
        if not sep or suffix not in ("A.weight", "B.weight"):
            continue  # not an adapter tensor (a stray trainable base parameter)
        native = diffusers_to_native_module(module)
        tensor = value.detach()
        if suffix == "B.weight":
            tensor = tensor.float() * scale
        artifact[f"diffusion_model.{native}.lora_{suffix}"] = tensor
    return artifact


class Flux3Saver(ModelSaver):
    """Write the PEFT adapters as a ``diffusion_model.``-prefixed native-named safetensors file.

    A save that cannot produce the artifact RAISES (the manager logs it and fails a final save) --
    never a silent no-file success.
    """

    def save(
        self, components: dict[str, Any], path: Path, metadata: dict[str, Any] | None = None,
    ) -> None:
        model = components.get("unet")
        if model is None:
            raise ValueError("flux3 saver: no 'unet' component to save")
        if not hasattr(model, "peft_config"):
            raise ValueError("flux3 saver: the transformer is not a PEFT model -- nothing to extract")

        peft_cfg = next(iter(model.peft_config.values()))
        rank = int(getattr(peft_cfg, "r", 0) or 0)
        if rank <= 0:
            raise ValueError("flux3 saver: the PEFT config carries no rank")
        alpha = float(getattr(peft_cfg, "lora_alpha", rank))
        artifact = build_native_artifact(
            get_peft_model_state_dict(model), rank, alpha, bool(getattr(peft_cfg, "use_rslora", False)),
        )
        if not artifact:
            raise ValueError("flux3 saver: the PEFT model carries no LoRA adapters")

        config = components.get("config")
        if not isinstance(config, dict):
            config = {}
        save_metadata: dict[str, str] = {
            "format": "pt",
            "software": '{"name": "Arcane Tuner"}',
            "version": "1.0",
            "modelspec.architecture": ARCHITECTURE,
        }
        # The scale is folded into lora_B, so the file's effective alpha IS its rank.
        save_metadata.update(kohya_config_metadata(config, rank=rank, alpha=float(rank)))
        save_metadata.update(trigger_metadata(config))
        if metadata:
            save_metadata.update({k: str(v) for k, v in metadata.items()})

        dtype = _SAVE_DTYPES.get(str(config.get("save_precision", "bf16")).lower(), torch.bfloat16)
        tensors = {k: v.to(dtype).contiguous() for k, v in artifact.items()}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_save_file(tensors, str(path), metadata=save_metadata)
        logger.info("flux3_save_lora", path=str(path), num_tensors=len(tensors), rank=rank,
                    alpha=alpha, save_dtype=str(dtype))
