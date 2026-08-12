"""Microsoft Lens LoRA saver.

Produces Kohya / ComfyUI ``lora_unet_*`` keys (via ``convert_peft_to_kohya``)
with mandatory ``ss_*`` metadata. Modeled on ``SDXLSaver``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import structlog
import torch
from peft import get_peft_model_state_dict

from app.engine.core.interfaces import ModelSaver
from app.engine.utils.lora_conversion import convert_peft_to_kohya
from app.engine.utils.lora_metadata import trigger_metadata
from app.engine.utils.safe_save import safe_save_file

logger = structlog.get_logger(__name__)

_MAP = {
    "optimizer_type": "ss_optimizer",
    "lr_scheduler": "ss_lr_scheduler",
    "mixed_precision": "ss_mixed_precision",
    "lora_name": "ss_output_name",
    "definition_id": "ss_sd_model_name",
    "timestep_sampling": "ss_timestep_sampling",
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


class MicrosoftLensSaver(ModelSaver):
    """Save Lens LoRA weights as Kohya/ComfyUI-compatible safetensors."""

    def save(
        self,
        components: dict[str, Any],
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        unet = components.get("unet")
        if not unet or not hasattr(unet, "peft_config"):
            logger.error("microsoft_lens_save_no_peft_model")
            return

        peft_sd = get_peft_model_state_dict(unet)
        combined = {f"lora_unet.{k}": v for k, v in peft_sd.items()}
        if not combined:
            logger.warning("no_lora_weights_found")
            return

        config = components.get("config", {}) or {}
        peft_cfg = next(iter(unet.peft_config.values()), None)
        rank = int(getattr(peft_cfg, "r", config.get("network_rank", 16)))
        alpha = float(getattr(peft_cfg, "lora_alpha", config.get("network_alpha", rank)))

        final_dict = convert_peft_to_kohya(
            combined, model_type="microsoft_lens", alpha=alpha,
        )

        save_metadata = {
            "format": "pt",
            "software": '{"name": "Arcane Tuner"}',
            "version": "1.0",
            "ss_network_module": "networks.lora",
            "ss_network_dim": str(rank),
            "ss_network_alpha": str(alpha),
            "ss_base_model_version": "microsoft_lens",
            "modelspec.architecture": "microsoft_lens",
        }
        if isinstance(config, dict):
            for cfg_key, ss_key in _MAP.items():
                val = config.get(cfg_key)
                if val is not None and str(val).strip():
                    save_metadata[ss_key] = str(val)
            save_metadata.update(trigger_metadata(config))
            resolutions = config.get("resolutions")
            if isinstance(resolutions, list) and resolutions:
                save_metadata["ss_resolution"] = f"({resolutions[0]},{resolutions[0]})"
            if peft_cfg is not None:
                tm = getattr(peft_cfg, "target_modules", None)
                if tm:
                    save_metadata["ss_network_args"] = json.dumps({
                        "target_modules": sorted(tm) if not isinstance(tm, str) else [tm],
                    })
        if metadata:
            save_metadata.update({k: str(v) for k, v in metadata.items()})

        try:
            save_prec = str(config.get("save_precision", "bf16")).lower()
            save_dtype = (
                torch.float16 if save_prec == "fp16"
                else torch.bfloat16 if save_prec == "bf16"
                else torch.float32
            )
            for k in final_dict:
                final_dict[k] = final_dict[k].to(save_dtype)

            dir_part = os.path.dirname(str(path))
            if dir_part:
                os.makedirs(dir_part, exist_ok=True)
            safe_save_file(final_dict, str(path), metadata=save_metadata)
            if os.path.exists(str(path)):
                size_mb = os.path.getsize(str(path)) / (1024 * 1024)
                logger.info("lora_saved", path=str(path), size_mb=f"{size_mb:.2f}MB")
        except (OSError, ValueError, RuntimeError) as e:
            # Log with full context, then re-raise — a swallowed save
            # failure here previously let a training job "succeed" while
            # writing no LoRA file. The caller (CheckpointManager) decides
            # whether a save failure at this point is fatal (final save)
            # or should be logged-and-continued (periodic checkpoint).
            logger.error("save_failed", path=str(path), error=str(e))
            raise
