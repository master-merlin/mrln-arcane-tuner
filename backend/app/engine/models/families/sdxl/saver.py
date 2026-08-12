"""SDXL LoRA saver.

Extracts PEFT adapter weights, converts to Kohya-compatible format via
``convert_peft_to_kohya``, and saves as safetensors with standard metadata
for ComfyUI / A1111 compatibility.
"""

import os
import torch
import structlog
from pathlib import Path
from peft import get_peft_model_state_dict
from app.engine.utils.lora_metadata import trigger_metadata
from app.engine.utils.safe_save import safe_save_file

from app.engine.core.interfaces import ModelSaver
from app.engine.utils.lora_conversion import convert_peft_to_kohya

logger = structlog.get_logger(__name__)

class SDXLSaver(ModelSaver):
    """Save SDXL LoRA weights as Kohya-compatible safetensors."""

    def save(
        self,
        components: dict[str, any],
        path: Path,
        metadata: dict[str, any] | None = None,
    ) -> None:
        """Extract LoRA weights, convert to Kohya format, and save.

        Args:
            components: Dict with 'unet', optional 'text_encoder_1'/'text_encoder_2',
                        and 'config' for precision settings.
            path: Output safetensors file path.
            metadata: Optional extra metadata to embed in the file.
        """
        logger.info("saving_lora", path=str(path))
        
        unet = components.get("unet")
        te1 = components.get("text_encoder_1")
        te2 = components.get("text_encoder_2")
        
        if not unet:
            logger.error("unet_missing_for_save")
            return

        # 1. Get Peft state dicts
        unet_state_dict = get_peft_model_state_dict(unet)
        
        te1_state_dict = {}
        if te1 and hasattr(te1, "peft_config"):
            te1_state_dict = get_peft_model_state_dict(te1)
            
        te2_state_dict = {}
        if te2 and hasattr(te2, "peft_config"):
            te2_state_dict = get_peft_model_state_dict(te2)

        # 2. Combine with prefixes
        combined_dict = {}
        for k, v in unet_state_dict.items():
            combined_dict[f"lora_unet.{k}"] = v
        for k, v in te1_state_dict.items():
            combined_dict[f"lora_te1.{k}"] = v
        for k, v in te2_state_dict.items():
            combined_dict[f"lora_te2.{k}"] = v

        if not combined_dict:
            logger.warning("no_lora_weights_found")
            return

        # 3. Handle Alpha and Rank — extract from PEFT config (ground truth)
        config = components.get("config", {})
        rank = 16  # fallback
        alpha = 16.0
        if unet and hasattr(unet, "peft_config"):
            # peft_config is a dict of adapter_name -> LoraConfig
            peft_cfg = next(iter(unet.peft_config.values()), None)
            if peft_cfg:
                rank = int(getattr(peft_cfg, "r", 16))
                alpha = float(getattr(peft_cfg, "lora_alpha", rank))
                logger.debug("rank_from_peft", rank=rank, alpha=alpha)
        else:
            rank = int(config.get("network_rank", 16))
            alpha = float(config.get("network_alpha", rank))

        # 4. Convert and Save
        try:
            final_dict = convert_peft_to_kohya(combined_dict, model_type="sdxl", alpha=alpha)
            
            # Metadata — enrich with Kohya-compatible ss_ keys
            peft_cfg_sdxl = next(iter(unet.peft_config.values()), None) if hasattr(unet, "peft_config") else None
            save_metadata = {
                "format": "pt",
                "ss_network_module": "networks.lora",
                "ss_network_dim": str(rank),
                "ss_network_alpha": str(alpha),
                "ss_base_model_version": "sdxl_1.0",
                "modelspec.architecture": "sdxl",
                "software": '{"name": "Arcane Tuner"}',
                "version": "1.0",
            }

            # Map training config → Kohya ss_ metadata keys
            if config and isinstance(config, dict):
                _MAP = {
                    "optimizer_type": "ss_optimizer",
                    "lr_scheduler": "ss_lr_scheduler",
                    "mixed_precision": "ss_mixed_precision",
                    "lora_name": "ss_output_name",
                    "definition_id": "ss_sd_model_name",
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
                for cfg_key, ss_key in _MAP.items():
                    val = config.get(cfg_key)
                    if val is not None and str(val).strip():
                        save_metadata[ss_key] = str(val)
                save_metadata.update(trigger_metadata(config))

                resolutions = config.get("resolutions")
                if resolutions and isinstance(resolutions, list):
                    r = resolutions[0]
                    save_metadata["ss_resolution"] = f"({r},{r})"

                if peft_cfg_sdxl:
                    tm = getattr(peft_cfg_sdxl, "target_modules", None)
                    if tm:
                        import json as _json
                        save_metadata["ss_network_args"] = _json.dumps({
                            "target_modules": sorted(tm) if not isinstance(tm, str) else [tm],
                        })

            if metadata:
                save_metadata.update({k: str(v) for k, v in metadata.items()})

            # Determine precision
            save_prec = config.get("save_precision")
            if not save_prec:
                save_prec = config.get("mixed_precision", "fp16")
                logger.debug("save_precision_not_set_falling_back_to_mixed", mixed=save_prec)
            
            save_prec = save_prec.lower()
            save_dtype = torch.float16 if save_prec == "fp16" else (torch.bfloat16 if save_prec == "bf16" else torch.float32)
            
            logger.info("lora_save_precision_selected", requested=save_prec, dtype=str(save_dtype))
            
            for k in final_dict:
                final_dict[k] = final_dict[k].to(save_dtype)
                
            os.makedirs(os.path.dirname(path), exist_ok=True)
            safe_save_file(final_dict, path, metadata=save_metadata)
            
            # Log file size
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                logger.info("lora_saved_successfully", path=str(path), size_mb=f"{size_mb:.2f}MB", keys_count=len(final_dict))

            
        except (OSError, ValueError, RuntimeError) as e:
            # Log with full context, then re-raise — a swallowed save
            # failure here previously let a training job "succeed" while
            # writing no LoRA file. The caller (CheckpointManager) decides
            # whether a save failure at this point is fatal (final save)
            # or should be logged-and-continued (periodic checkpoint).
            logger.error("lora_save_failed", path=str(path), error=str(e))
            raise
