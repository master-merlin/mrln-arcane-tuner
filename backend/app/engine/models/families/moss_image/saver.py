"""MOSS-Image LoRA saver (LANE-165 spec scenario 7) -- kohya layout, ComfyUI-core loadable.

Layout (Evidence: research §LoRA keys, the layout ComfyUI core has loaded since 2026-10-16):

- every key is ``diffusion_model.layers.N.proj.<q|k|v|o|ffn_up_gate|ffn_down>.lora_A.weight`` /
  ``.lora_B.weight`` -- the transformer's own module paths, exactly two tensors per target
  (504 for 42 blocks x 6 projections), no ``.alpha`` tensor, q/k/v never fused;
- the PEFT scale ``alpha / rank`` (rsLoRA: ``alpha / sqrt(rank)``) is FOLDED into ``lora_B`` and
  ``ss_network_alpha`` records the rank, so ``B @ A`` on disk is the effective training delta (the
  flux3 convention, ``flux3/saver.py``);
- metadata comes from the one helper (``app.engine.utils.lora_metadata``, ECOSYSTEM §5 #7).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import structlog
import torch
from peft.utils import get_peft_model_state_dict

from app.engine.core.interfaces import ModelSaver
from app.engine.utils.lora_metadata import kohya_config_metadata, trigger_metadata
from app.engine.utils.safe_save import safe_save_file

logger = structlog.get_logger(__name__)

ARCHITECTURE = "moss_image"

_MODULE = re.compile(r"layers\.\d+\.proj\.(?:q|k|v|o|ffn_up_gate|ffn_down)")

_SAVE_DTYPES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def build_artifact(
    peft_sd: dict[str, torch.Tensor], rank: int, alpha: float, use_rslora: bool = False,
) -> dict[str, torch.Tensor]:
    """PEFT state dict -> ``diffusion_model.``-prefixed tensors with the scale folded into B."""
    scale = alpha / math.sqrt(rank) if use_rslora else alpha / rank
    artifact: dict[str, torch.Tensor] = {}
    for key, value in peft_sd.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean = key.replace("base_model.model.", "").removeprefix("transformer.")
        module, sep, suffix = clean.rpartition(".lora_")
        if not sep or suffix not in ("A.weight", "B.weight"):
            continue  # not an adapter tensor (a stray trainable base parameter)
        if not _MODULE.fullmatch(module):
            raise ValueError(f"moss_image saver: unexpected LoRA module {module!r}")
        tensor = value.detach()
        if suffix == "B.weight":
            tensor = tensor.float() * scale
        artifact[f"diffusion_model.{module}.lora_{suffix}"] = tensor
    return artifact


class MossImageSaver(ModelSaver):
    """Write the PEFT adapters as a kohya-layout safetensors file.

    A save that cannot produce the artifact RAISES -- never a silent no-file success.
    """

    def save(
        self, components: dict[str, Any], path: Path, metadata: dict[str, Any] | None = None,
    ) -> None:
        model = components.get("unet")
        if model is None:
            raise ValueError("moss_image saver: no 'unet' component to save")
        if not hasattr(model, "peft_config"):
            raise ValueError("moss_image saver: the transformer is not a PEFT model -- nothing to extract")

        peft_cfg = next(iter(model.peft_config.values()))
        rank = int(getattr(peft_cfg, "r", 0) or 0)
        if rank <= 0:
            raise ValueError("moss_image saver: the PEFT config carries no rank")
        alpha = float(getattr(peft_cfg, "lora_alpha", rank))
        artifact = build_artifact(
            get_peft_model_state_dict(model), rank, alpha, bool(getattr(peft_cfg, "use_rslora", False)),
        )
        if not artifact:
            raise ValueError("moss_image saver: the PEFT model carries no LoRA adapters")

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
        logger.info("moss_image_save_lora", path=str(path), num_tensors=len(tensors), rank=rank,
                    alpha=alpha, save_dtype=str(dtype))
