"""HiDream-O1 LoRA saver — ComfyUI-loader-compatible artifact format.

Writes:
- ``<output_name>.safetensors`` with kohya-style keys:
    ``diffusion_model.<lora_key>.lora_down.weight``  (rank, in_features)
    ``diffusion_model.<lora_key>.lora_up.weight``    (out_features, rank)
    ``diffusion_model.<lora_key>.alpha``             (scalar)
- ``hidream_o1_lora_config.json`` sidecar with metadata.

Format matches Saganaki22's ``training/lora.py:lora_state_dict()`` (MIT),
which is the convention ComfyUI's native HiDream-O1 LoRA loader expects
(verified against Kijai-published reference LoRAs).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
import torch
import torch.nn as nn
from safetensors.torch import save_file

from app.engine.core.interfaces import IModelSaver

from .lora_wrapper import HiDreamO1LoRALinear

logger = structlog.get_logger(__name__)


SAVE_DTYPES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def _resolve_dtype(name: str | torch.dtype | None) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    if name is None:
        return torch.bfloat16
    return SAVE_DTYPES.get(str(name).lower(), torch.bfloat16)


def _collect_lora_layers(model: nn.Module) -> list[HiDreamO1LoRALinear]:
    """Walk ``model`` and return all ``HiDreamO1LoRALinear`` wrappers."""
    return [m for m in model.modules() if isinstance(m, HiDreamO1LoRALinear)]


def _build_state_dict(
    layers: list[HiDreamO1LoRALinear],
    *,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Construct the kohya-style state dict from LoRA wrappers."""
    state: dict[str, torch.Tensor] = {}
    for layer in layers:
        key = f"diffusion_model.{layer.lora_key}"
        state[f"{key}.lora_down.weight"] = (
            layer.lora_down.detach().to("cpu", dtype=dtype).contiguous()
        )
        state[f"{key}.lora_up.weight"] = (
            layer.lora_up.detach().to("cpu", dtype=dtype).contiguous()
        )
        state[f"{key}.alpha"] = torch.tensor(float(layer.alpha), dtype=dtype)
    return state


class HiDreamO1Saver(IModelSaver):
    """Save HiDream-O1 LoRA weights in ComfyUI-compatible kohya format."""

    def __init__(self, save_dtype: str | torch.dtype = "bf16"):
        self.save_dtype: torch.dtype = _resolve_dtype(save_dtype)
        self.logger = structlog.get_logger(self.__class__.__name__)

    def save(  # type: ignore[override]
        self,
        model: nn.Module,
        out_dir: str,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save the LoRA weights + sidecar.

        Args:
            model: The full model (with ``HiDreamO1LoRALinear`` wrappers
                in place).
            out_dir: Output directory. Will be created if missing.
            name: Filename stem (e.g. ``"my_lora"`` -> ``"my_lora.safetensors"``).
            metadata: Optional metadata fields written to the sidecar JSON.

        Returns:
            Path to the output directory (handy for caller logging).

        Raises:
            RuntimeError: if no ``HiDreamO1LoRALinear`` wrappers were found in
                the model — saving would produce an empty file.
        """
        layers = _collect_lora_layers(model)
        if not layers:
            raise RuntimeError(
                "HiDreamO1Saver.save: no HiDreamO1LoRALinear wrappers found "
                "in the model — refusing to write an empty LoRA file. "
                "Did inject_lora_layers run before saving?",
            )

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        state = _build_state_dict(layers, dtype=self.save_dtype)
        weights_path = out_path / f"{name}.safetensors"
        save_file(state, str(weights_path))

        sidecar = self._build_sidecar(layers, metadata or {})
        sidecar_path = out_path / "hidream_o1_lora_config.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

        self.logger.info(
            "hidream_o1.saver.saved",
            out_dir=str(out_path),
            name=name,
            layers=len(layers),
            state_keys=len(state),
            dtype=str(self.save_dtype),
        )
        return out_path

    def _build_sidecar(
        self,
        layers: list[HiDreamO1LoRALinear],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Compose the JSON sidecar from input metadata + observed layer info."""
        # Infer rank/alpha from layers if not supplied
        rank = metadata.get("rank")
        if rank is None and layers:
            rank = layers[0].rank
        alpha = metadata.get("alpha")
        if alpha is None and layers:
            alpha = layers[0].alpha

        return {
            "base_model": metadata.get("base_model", "HiDream-ai/HiDream-O1-Image"),
            "vendor_revision": metadata.get("vendor_revision"),
            "rank": rank,
            "alpha": alpha,
            "target_preset": metadata.get("target_preset", "aitoolkit"),
            "excluded_modules": list(metadata.get("excluded_modules", [])) or [
                "lm_head", "patch_embed", "visual",
            ],
            "save_dtype": str(self.save_dtype),
            "recipe": {
                "noise_scale": metadata.get("noise_scale", 8.0),
                "timestep_type": metadata.get("timestep_type", "linear"),
                "max_loss": metadata.get("max_loss", 1.0),
            },
            "layer_count": len(layers),
            "key_format": "diffusion_model.<lora_key>.{lora_down.weight,lora_up.weight,alpha}",
        }
