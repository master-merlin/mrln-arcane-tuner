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

from app.engine.core.interfaces import IModelSaver
from app.engine.utils.lora_metadata import trigger_metadata
from app.engine.utils.safe_save import safe_save_file

from .lora_wrapper import HiDreamO1LoRALinear
from .vendor.pipeline import NOISE_SCALE as NOISE_SCALE_DEFAULT

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

    # ── IModelSaver interface (base-conforming, called by CheckpointManager) ──

    def save(
        self,
        components: dict[str, Any],
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Base-interface conforming ``save``.

        ``CheckpointManager.save_checkpoint`` calls
        ``saver.save(components_dict, dist_path, metadata=metadata)``
        where ``dist_path`` is a ``Path`` to the ``.safetensors`` file.

        Honors ``config["save_precision"]`` (fp16/bf16/fp32) by switching
        ``self.save_dtype`` for this call so different runs can pick their
        artifact precision via training settings.
        """
        path = Path(path)
        model = components.get("unet")
        if model is None:
            self.logger.error("hidream_o1.saver.no_unet_in_components")
            return

        config = components.get("config") or {}

        # Per-call save_dtype override from training config
        prev_dtype = self.save_dtype
        if isinstance(config, dict) and config.get("save_precision"):
            self.save_dtype = _resolve_dtype(config.get("save_precision"))

        merged_meta: dict[str, Any] = {
            "noise_scale": float(config.get("noise_scale", NOISE_SCALE_DEFAULT)),
            "timestep_type": config.get("timestep_type", "linear"),
            "max_loss": float(config.get("max_loss", 1.0)),
            "config": config if isinstance(config, dict) else None,
        }
        if metadata:
            merged_meta.update(metadata)

        try:
            self._save_lora(
                model=model,
                out_dir=str(path.parent),
                name=path.stem,
                metadata=merged_meta,
            )
        finally:
            self.save_dtype = prev_dtype

    # ── Direct API (used by tests and external callers) ───────────────────

    def save_lora(
        self,
        model: nn.Module,
        out_dir: str,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Public direct API for saving LoRA weights + sidecar.

        Args:
            model: The full model (with ``HiDreamO1LoRALinear`` wrappers).
            out_dir: Output directory. Will be created if missing.
            name: Filename stem (e.g. ``"my_lora"`` → ``"my_lora.safetensors"``).
            metadata: Optional metadata fields written to the sidecar JSON.

        Returns:
            Path to the output directory (handy for caller logging).
        """
        return self._save_lora(model, out_dir, name, metadata)

    def _save_lora(
        self,
        model: nn.Module,
        out_dir: str,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Internal implementation — shared by ``save`` and ``save_lora``.

        Raises:
            RuntimeError: if no ``HiDreamO1LoRALinear`` wrappers were found in
                the model — saving would produce an empty file.
        """
        layers = _collect_lora_layers(model)
        if not layers:
            raise RuntimeError(
                "HiDreamO1Saver._save_lora: no HiDreamO1LoRALinear wrappers found "
                "in the model — refusing to write an empty LoRA file. "
                "Did inject_lora_layers run before saving?",
            )

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        state = _build_state_dict(layers, dtype=self.save_dtype)
        weights_path = out_path / f"{name}.safetensors"

        header_metadata = self._build_header_metadata(layers, metadata or {})
        safe_save_file(state, str(weights_path), metadata=header_metadata)

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
            header_meta_keys=len(header_metadata),
        )
        return out_path

    def _build_header_metadata(
        self,
        layers: list[HiDreamO1LoRALinear],
        metadata: dict[str, Any],
    ) -> dict[str, str]:
        """Build Kohya-compatible ss_* header for the .safetensors file.

        Mirrors ``GenericLoRASaver``'s metadata so ComfyUI / lora-inspector /
        Civitai can read rank, alpha, optimizer, learning rate, seed, etc.
        directly from the header. Rank/alpha are read off the wrappers
        because we don't use PEFT — there's no ``peft_config`` to query.
        """
        rank = int(metadata.get("rank") or (layers[0].rank if layers else 0))
        alpha = float(metadata.get("alpha") or (layers[0].alpha if layers else 0.0))

        header: dict[str, str] = {
            "format": "pt",
            "software": '{"name": "Arcane Tuner"}',
            "version": "1.0",
            "ss_network_dim": str(rank),
            "ss_network_alpha": str(alpha),
            "modelspec.architecture": "hidream_o1",
        }

        config = metadata.get("config")
        if isinstance(config, dict):
            _MAP_STR = {
                "optimizer_type": "ss_optimizer",
                "lr_scheduler": "ss_lr_scheduler",
                "mixed_precision": "ss_mixed_precision",
                "lora_name": "ss_output_name",
                "definition_id": "ss_sd_model_name",
                "model_family": "ss_base_model_version",
                "timestep_sampling": "ss_timestep_sampling",
            }
            _MAP_NUM = {
                "learning_rate": "ss_learning_rate",
                "max_train_steps": "ss_steps",
                "train_batch_size": "ss_batch_size_per_device",
                "gradient_accumulation_steps": "ss_gradient_accumulation_steps",
                "noise_offset": "ss_noise_offset",
                "min_snr_gamma": "ss_min_snr_gamma",
                "lr_warmup_steps": "ss_warmup_steps",
                "weight_decay": "ss_weight_decay",
                "seed": "ss_seed",
            }
            for cfg_key, ss_key in _MAP_STR.items():
                val = config.get(cfg_key)
                if val is not None and str(val).strip():
                    header[ss_key] = str(val)
            for cfg_key, ss_key in _MAP_NUM.items():
                val = config.get(cfg_key)
                if val is not None:
                    header[ss_key] = str(val)
            header.update(trigger_metadata(config))

            resolutions = config.get("resolutions")
            if resolutions and isinstance(resolutions, list):
                r = resolutions[0]
                header["ss_resolution"] = f"({r},{r})"

            datasets = config.get("datasets")
            if datasets and isinstance(datasets, list):
                header["ss_num_train_images"] = str(
                    sum(d.get("num_repeats", 1) for d in datasets if isinstance(d, dict))
                )

        # HiDream-O1 recipe fields — useful when inspecting why a LoRA behaves
        # a certain way at inference (noise_scale and max_loss are non-default).
        for k in ("noise_scale", "timestep_type", "max_loss", "vendor_revision"):
            if k in metadata and metadata[k] is not None:
                header[f"ss_{k}"] = str(metadata[k])

        # Caller-provided step/job_id from CheckpointManager
        for k in ("step", "ss_session_id"):
            if k in metadata and metadata[k] is not None:
                header[k if k.startswith("ss_") else f"ss_{k}"] = str(metadata[k])

        return header

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
