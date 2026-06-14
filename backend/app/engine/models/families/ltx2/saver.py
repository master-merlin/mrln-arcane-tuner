"""LTX 2.3 LoRA saver — export PEFT weights to ComfyUI ``diffusion_model.*``.

LTX-2's diffusers transformer keys map 1:1 to the ComfyUI single-file layout:
the diffusers module path (``transformer_blocks.N.attn1.to_q`` …) is prefixed
with ``diffusion_model.`` and written to a single safetensors file.  No QKV
fusion is needed (LTX-2 uses separate ``to_q``/``to_k``/``to_v``).

Audio-stream LoRA keys (``audio_attn*``, ``audio_ff*``, the cross-modal
``audio_to_video_attn`` / ``video_to_audio_attn`` bridges) are present in the
PEFT state dict ONLY when the run trained audio, so they are written ONLY then —
no extra gating is required beyond what PEFT already produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
import torch
from peft import get_peft_model_state_dict

from app.engine.core.interfaces import ModelSaver
from app.engine.utils.safe_save import safe_save_file

logger = structlog.get_logger(__name__)

# Substrings that identify audio-stream / cross-modal LoRA keys.  Used only for
# diagnostics (counting how many audio keys were written).
_AUDIO_KEY_MARKERS = (
    "audio_attn", "audio_ff",
    "audio_to_video_attn", "video_to_audio_attn",
)


class Ltx2Saver(ModelSaver):
    """Save LTX 2.3 LoRA weights in ComfyUI ``diffusion_model.*`` format."""

    def save(
        self,
        components: dict[str, Any],
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Extract PEFT LoRA weights, prefix to ComfyUI keys, and save.

        Args:
            components: Dict containing ``unet`` (the PEFT-wrapped transformer)
                and optionally ``config`` with save settings.
            path: Output file path.
            metadata: Additional safetensors-header metadata.
        """
        model = components.get("unet")
        if model is None:
            logger.error("ltx2_save_no_model")
            return
        if not hasattr(model, "peft_config"):
            logger.warning("transformer_not_peft_model")
            return

        # 1. Extract PEFT state dict → clean diffusers LoRA keys.
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

        # 2. Prefix to ComfyUI keys (1:1 rename, no fusion).
        final_dict: dict[str, torch.Tensor] = {}
        audio_keys = 0
        for key, value in diffusers_sd.items():
            out_key = key if key.startswith("diffusion_model.") else f"diffusion_model.{key}"
            final_dict[out_key] = value
            if any(marker in key for marker in _AUDIO_KEY_MARKERS):
                audio_keys += 1

        # 3. Rank / alpha from PEFT config.
        config = components.get("config", {}) or {}
        rank, alpha = 16, 16.0
        peft_cfg = next(iter(model.peft_config.values()), None)
        if peft_cfg:
            rank = int(getattr(peft_cfg, "r", 16))
            alpha = float(getattr(peft_cfg, "lora_alpha", rank))

        # 4. Metadata.
        save_metadata = {
            "format": "pt",
            "software": '{"name": "Arcane Tuner"}',
            "version": "1.0",
            "ss_network_dim": str(rank),
            "ss_network_alpha": str(alpha),
            "modelspec.architecture": "ltx-2.3",
        }
        save_metadata.update(self._config_metadata(config))
        if metadata:
            save_metadata.update({k: str(v) for k, v in metadata.items()})

        # 5. Save dtype (LTX-2 default: bf16).
        save_prec = str(config.get("save_precision", "bf16")).lower()
        save_dtype = (
            torch.float16 if save_prec == "fp16"
            else torch.bfloat16 if save_prec == "bf16"
            else torch.float32
        )
        for k in final_dict:
            final_dict[k] = final_dict[k].to(save_dtype)

        logger.info(
            "ltx2_save_lora",
            path=str(path),
            num_tensors=len(final_dict),
            audio_lora_keys=audio_keys,
            save_dtype=str(save_dtype),
        )

        # 6. Write.
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_save_file(final_dict, str(path), metadata=save_metadata)
        logger.info("ltx2_save_complete", path=str(path))

    @staticmethod
    def _config_metadata(config: dict[str, Any]) -> dict[str, str]:
        """Map training config → Kohya ``ss_*`` metadata keys."""
        if not isinstance(config, dict) or not config:
            return {}
        mapping = {
            "optimizer_type": "ss_optimizer",
            "lr_scheduler": "ss_lr_scheduler",
            "mixed_precision": "ss_mixed_precision",
            "lora_name": "ss_output_name",
            "definition_id": "ss_sd_model_name",
            "model_family": "ss_base_model_version",
            "learning_rate": "ss_learning_rate",
            "max_train_steps": "ss_steps",
            "train_batch_size": "ss_batch_size_per_device",
            "seed": "ss_seed",
        }
        out: dict[str, str] = {}
        for cfg_key, ss_key in mapping.items():
            val = config.get(cfg_key)
            if val is not None and str(val).strip():
                out[ss_key] = str(val)
        return out
