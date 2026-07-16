"""WAN 2.2 dual-expert LoRA saver — writes TWO ComfyUI-format files.

WAN 2.2 A14B trains two experts, so a single run produces TWO LoRA files
following the ComfyUI / community WAN 2.2 naming:

    {stem}_high_noise.safetensors   (the high-noise expert, diffusers transformer)
    {stem}_low_noise.safetensors    (the low-noise expert,  diffusers transformer_2)

Each file uses the SAME ``diffusion_model.blocks.*`` key mapping as WAN 2.1
(``self_attn`` / ``cross_attn`` / ``ffn.{0,2}`` + ``lora_{down,up}``) — reused
directly from :mod:`wan_shared.saver_base` so the two families never drift.
Per-file metadata records ``modelspec.architecture: wan2.2-{t2v,i2v}-{high,low}``.

The saver is handed both PEFT models via the components dict
(``unet_high`` / ``unet_low``) plus the single output path; it derives the two
filenames from that path's stem.
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
from app.engine.utils.safe_save import safe_save_file

logger = structlog.get_logger(__name__)


class Wan22Saver(ModelSaver):
    """Save WAN 2.2 dual-expert LoRA weights as two ComfyUI-format files."""

    # ``modelspec.architecture`` family prefix — ``{ARCH_FAMILY}-{mode}-{expert}``.
    # Subclasses (e.g. Bernini-R's dual saver, which is 100%-stock wan2.2-arch
    # weights) override only this for provenance; the tensor keys stay identical.
    ARCH_FAMILY = "wan2.2"

    def __init__(self, mode: str = "t2v") -> None:
        self.mode = str(mode).lower() if mode else "t2v"

    # ── Public API ────────────────────────────────────────────────────────

    def save(
        self,
        components: dict[str, Any],
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write ``{stem}_high_noise`` + ``{stem}_low_noise`` safetensors."""
        high = components.get("unet_high")
        low = components.get("unet_low")
        if high is None and low is None:
            # Fall back to a single "unet" (e.g. resume-only paths) → high slot.
            high = components.get("unet")

        config = components.get("config", {}) or {}
        path = Path(path)
        stem = path.stem
        suffix = path.suffix or ".safetensors"
        parent = path.parent
        os.makedirs(parent, exist_ok=True)

        wrote_any = False
        for expert, model in (("high", high), ("low", low)):
            if model is None:
                continue
            out_path = parent / f"{stem}_{expert}_noise{suffix}"
            if self._save_one(model, out_path, expert, config, metadata):
                wrote_any = True

        if not wrote_any:
            logger.warning("wan22_save_no_experts")

    # ── One expert → one file ─────────────────────────────────────────────

    def _save_one(
        self,
        model: Any,
        out_path: Path,
        expert: str,
        config: dict[str, Any],
        metadata: dict[str, Any] | None,
    ) -> bool:
        if not hasattr(model, "peft_config"):
            logger.warning("wan22_expert_not_peft_model", expert=expert)
            return False

        # 1. PEFT state dict → clean diffusers keys.
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
            logger.warning("wan22_expert_no_lora_weights", expert=expert)
            return False

        # 2. Convert diffusers → ComfyUI keys (reuse the WAN 2.1 mapping).
        final_dict = _convert_diffusers_to_comfy(diffusers_sd)
        if not final_dict:
            logger.warning("wan22_expert_no_keys_after_conversion", expert=expert)
            return False

        # 3. Rank / alpha from PEFT config.
        rank, alpha = 16, 16.0
        peft_cfg = next(iter(model.peft_config.values()), None)
        if peft_cfg:
            rank = int(getattr(peft_cfg, "r", 16))
            alpha = float(getattr(peft_cfg, "lora_alpha", rank))

        arch = f"{self.ARCH_FAMILY}-{self.mode}-{expert}"
        save_metadata: dict[str, str] = {
            "format": "pt",
            "software": '{"name": "Arcane Tuner"}',
            "version": "1.0",
            "ss_network_dim": str(rank),
            "ss_network_alpha": str(alpha),
            "modelspec.architecture": arch,
            "wan22_expert": expert,
            "wan22_noise_level": "high" if expert == "high" else "low",
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
                "expert_switch_interval": "ss_wan22_switch_interval",
                "expert_swap_mode": "ss_wan22_swap_mode",
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
            "wan22_save_expert_lora",
            path=str(out_path),
            expert=expert,
            arch=arch,
            num_tensors=len(final_dict),
            save_dtype=str(save_dtype),
        )
        safe_save_file(final_dict, str(out_path), metadata=save_metadata)
        logger.info("wan22_save_expert_complete", path=str(out_path), expert=expert)
        return True
