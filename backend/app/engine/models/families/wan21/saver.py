"""WAN 2.1 LoRA saver — PEFT diffusers keys → ComfyUI ``diffusion_model.*``.

ComfyUI / kohya consume WAN LoRAs in the non-diffusers naming that diffusers'
``_convert_non_diffusers_wan_lora_to_diffusers`` reverses::

    diffusion_model.blocks.{i}.self_attn.{q,k,v,o}.lora_{down,up}.weight
    diffusion_model.blocks.{i}.cross_attn.{q,k,v,o}.lora_{down,up}.weight
    diffusion_model.blocks.{i}.cross_attn.{k_img,v_img}.lora_{down,up}.weight   # i2v
    diffusion_model.blocks.{i}.ffn.{0,2}.lora_{down,up}.weight

This saver maps the PEFT-extracted diffusers keys to that format:

    blocks.{i}.attn1.{to_q,to_k,to_v,to_out.0}  → self_attn.{q,k,v,o}
    blocks.{i}.attn2.{to_q,to_k,to_v,to_out.0}  → cross_attn.{q,k,v,o}
    blocks.{i}.attn2.{add_k_proj,add_v_proj}    → cross_attn.{k_img,v_img}   # i2v
    blocks.{i}.ffn.{net.0.proj,net.2}           → ffn.{0,2}
    lora_A → lora_down , lora_B → lora_up

Metadata records ``modelspec.architecture: wan2.1-t2v`` / ``wan2.1-i2v``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import structlog
import torch
from peft import get_peft_model_state_dict

from app.engine.core.interfaces import ModelSaver
from app.engine.utils.safe_save import safe_save_file

logger = structlog.get_logger(__name__)

# Suffix maps (diffusers module-suffix → ComfyUI sub-name).
_SELF_ATTN = {"to_q": "q", "to_k": "k", "to_v": "v", "to_out.0": "o"}
_CROSS_ATTN = {
    "to_q": "q",
    "to_k": "k",
    "to_v": "v",
    "to_out.0": "o",
    "add_k_proj": "k_img",
    "add_v_proj": "v_img",
}
_FFN = {"net.0.proj": "ffn.0", "net.2": "ffn.2"}


def _convert_diffusers_to_comfy(
    diffusers_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map diffusers PEFT LoRA keys → ComfyUI ``diffusion_model.blocks.*`` keys."""
    out: dict[str, torch.Tensor] = {}
    unconverted: list[str] = []

    for key, value in diffusers_sd.items():
        m = re.match(r"blocks\.(\d+)\.(.+)\.(lora_[AB])\.weight$", key)
        if not m:
            unconverted.append(key)
            continue
        block_idx, module, ab = m.group(1), m.group(2), m.group(3)
        ud = "lora_down" if ab == "lora_A" else "lora_up"

        comfy_module: str | None = None
        if module.startswith("attn1."):
            comfy_module = _map_suffix(module[len("attn1.") :], _SELF_ATTN, "self_attn")
        elif module.startswith("attn2."):
            comfy_module = _map_suffix(
                module[len("attn2.") :], _CROSS_ATTN, "cross_attn"
            )
        elif module.startswith("ffn."):
            sub = _FFN.get(module[len("ffn.") :])
            comfy_module = sub  # already "ffn.0" / "ffn.2"

        if comfy_module is None:
            unconverted.append(key)
            continue

        out_key = f"diffusion_model.blocks.{block_idx}.{comfy_module}.{ud}.weight"
        out[out_key] = value

    if unconverted:
        logger.warning("wan21_saver_unconverted_keys", keys=unconverted)

    return out


def _map_suffix(suffix: str, table: dict[str, str], prefix: str) -> str | None:
    sub = table.get(suffix)
    if sub is None:
        return None
    return f"{prefix}.{sub}"


class Wan21Saver(ModelSaver):
    """Save WAN 2.1 LoRA weights in ComfyUI-format safetensors."""

    def __init__(self, mode: str = "t2v") -> None:
        self.mode = str(mode).lower() if mode else "t2v"

    def save(
        self,
        components: dict[str, Any],
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        model = components.get("unet")
        if model is None:
            logger.error("wan21_save_no_model")
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
            logger.warning("wan21_no_keys_after_conversion")
            return

        # 3. Rank / alpha from PEFT config.
        config = components.get("config", {}) or {}
        rank, alpha = 16, 16.0
        peft_cfg = next(iter(model.peft_config.values()), None)
        if peft_cfg:
            rank = int(getattr(peft_cfg, "r", 16))
            alpha = float(getattr(peft_cfg, "lora_alpha", rank))

        arch = f"wan2.1-{self.mode}"
        save_metadata = {
            "format": "pt",
            "software": '{"name": "Arcane Tuner"}',
            "version": "1.0",
            "ss_network_dim": str(rank),
            "ss_network_alpha": str(alpha),
            "modelspec.architecture": arch,
        }

        if isinstance(config, dict):
            _MAP = {
                "optimizer_type": "ss_optimizer",
                "lr_scheduler": "ss_lr_scheduler",
                "mixed_precision": "ss_mixed_precision",
                "lora_name": "ss_output_name",
                "definition_id": "ss_sd_model_name",
                "model_family": "ss_base_model_version",
                "global_triggerword": "ss_training_comment",
                "learning_rate": "ss_learning_rate",
                "max_train_steps": "ss_steps",
                "timestep_sampling": "ss_timestep_sampling",
            }
            for cfg_key, ss_key in _MAP.items():
                val = config.get(cfg_key)
                if val is not None and str(val).strip():
                    save_metadata[ss_key] = str(val)

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
            "wan21_save_lora",
            path=str(path),
            arch=arch,
            num_tensors=len(final_dict),
            save_dtype=str(save_dtype),
        )

        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        safe_save_file(final_dict, str(path), metadata=save_metadata)
        logger.info("wan21_save_complete", path=str(path))
