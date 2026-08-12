"""WAN 2.2 TI2V-5B LoRA saver — PEFT diffusers keys → ComfyUI ``diffusion_model.*``.

Single dense transformer → a single output file (mirrors ``Wan21Saver``, NOT
``Wan22Saver``'s two-expert ``_high_noise``/``_low_noise`` split — there is only
one transformer here). The diffusers → ComfyUI key mapping is IDENTICAL to
``wan21``/``wan22`` (same ``WanTransformerBlock`` module layout — only the
per-block hidden size / layer count differ, which the key mapping doesn't care
about), so it's reused verbatim from :mod:`wan_shared.saver_base`.

Metadata records ``modelspec.architecture: wan2.2-ti2v-5b`` (not a
``{mode}``-suffixed string like wan21/wan22 — this checkpoint's ``mode`` is
``both``, chosen per training run via ``video_mode``, not baked into the
definition).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog
import torch
from peft import get_peft_model_state_dict

from app.engine.core.interfaces import ModelSaver
from app.engine.models.families.wan_shared.saver_base import (
    _convert_diffusers_to_comfy,
)
from app.engine.utils.lora_metadata import trigger_metadata
from app.engine.utils.safe_save import safe_save_file

logger = structlog.get_logger(__name__)

_ARCH = "wan2.2-ti2v-5b"


class Wan22Ti2v5bSaver(ModelSaver):
    """Save WAN 2.2 TI2V-5B LoRA weights in ComfyUI-format safetensors."""

    def save(
        self,
        components: dict[str, Any],
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        model = components.get("unet")
        if model is None:
            logger.error("wan22_ti2v_5b_save_no_model")
            return
        if not hasattr(model, "peft_config"):
            logger.warning("transformer_not_peft_model")
            return

        # 1. Extract PEFT state dict → clean diffusers keys.
        peft_sd = get_peft_model_state_dict(model)
        diffusers_sd: dict[str, torch.Tensor] = {}
        for key, value in peft_sd.items():
            if not isinstance(value, torch.Tensor):
                continue
            if "lora_A" not in key and "lora_B" not in key:
                continue
            clean = key.replace("base_model.model.", "")
            diffusers_sd[clean] = value

        if not diffusers_sd:
            logger.warning("no_lora_weights_found_to_save")
            return

        # 2. Convert diffusers → ComfyUI keys.
        final_dict = _convert_diffusers_to_comfy(diffusers_sd)
        if not final_dict:
            logger.warning("wan22_ti2v_5b_no_keys_after_conversion")
            return

        # 3. Rank / alpha from PEFT config.
        config = components.get("config", {}) or {}
        rank, alpha = 16, 16.0
        peft_cfg = next(iter(model.peft_config.values()), None)
        if peft_cfg:
            rank = int(getattr(peft_cfg, "r", 16))
            alpha = float(getattr(peft_cfg, "lora_alpha", rank))

        save_metadata = {
            "format": "pt",
            "software": '{"name": "Arcane Tuner"}',
            "version": "1.0",
            "ss_network_dim": str(rank),
            "ss_network_alpha": str(alpha),
            "modelspec.architecture": _ARCH,
        }

        if isinstance(config, dict):
            _MAP = {
                "optimizer_type": "ss_optimizer",
                "lr_scheduler": "ss_lr_scheduler",
                "mixed_precision": "ss_mixed_precision",
                "lora_name": "ss_output_name",
                "definition_id": "ss_sd_model_name",
                "model_family": "ss_base_model_version",
                "learning_rate": "ss_learning_rate",
                "max_train_steps": "ss_steps",
                "timestep_sampling": "ss_timestep_sampling",
                "video_mode": "ss_wan_video_mode",
            }
            for cfg_key, ss_key in _MAP.items():
                val = config.get(cfg_key)
                if val is not None and str(val).strip():
                    save_metadata[ss_key] = str(val)
            save_metadata.update(trigger_metadata(config))

        if metadata:
            save_metadata.update({k: str(v) for k, v in metadata.items()})

        # 4. Save precision (default bf16).
        save_prec = (
            config.get("save_precision", "bf16").lower()
            if isinstance(config, dict)
            else "bf16"
        )
        save_dtype = (
            torch.float16
            if save_prec == "fp16"
            else torch.bfloat16
            if save_prec == "bf16"
            else torch.float32
        )
        for k in final_dict:
            final_dict[k] = final_dict[k].to(save_dtype)

        logger.info(
            "wan22_ti2v_5b_save_lora",
            path=str(path),
            arch=_ARCH,
            num_tensors=len(final_dict),
            save_dtype=str(save_dtype),
        )

        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        safe_save_file(final_dict, str(path), metadata=save_metadata)
        logger.info("wan22_ti2v_5b_save_complete", path=str(path))
