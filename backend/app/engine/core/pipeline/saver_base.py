"""Generic LoRA saver — shared base for ai-toolkit-format families.

Extracts PEFT adapter weights, strips internal prefixes, adds the
``diffusion_model.`` key prefix, and saves as safetensors with standard
metadata.  Families that use raw PEFT format (lora_A/B keys, no Kohya
conversion) subclass this and set ``architecture_name``.

Families with custom conversion (SDXL → Kohya, Flux2 → BFL) should
keep their own ``ModelSaver`` subclass.
"""

import os
from pathlib import Path
from typing import Any

import structlog
import torch
from peft import get_peft_model_state_dict
from app.engine.utils.safe_save import safe_save_file

from app.engine.core.interfaces import ModelSaver

logger = structlog.get_logger(__name__)


class GenericLoRASaver(ModelSaver):
    """Save LoRA weights in ai-toolkit-compatible format.

    Subclasses MUST set ``architecture_name`` (used in ``modelspec.architecture``
    metadata and structured log events).

    Output key format::

        diffusion_model.{diffusers_module}.lora_A.weight
        diffusion_model.{diffusers_module}.lora_B.weight
    """

    architecture_name: str = ""
    """Override in subclass, e.g. ``"flux1"``, ``"qwen_image"``, ``"zimage"``."""

    key_prefix: str = "diffusion_model."
    """Key namespace for the exported LoRA modules.

    The house default ``"diffusion_model."`` is the ai-toolkit/BFL convention
    ComfyUI maps for families whose ComfyUI-internal module names equal their
    diffusers names (qwen_image, kandinsky5, ernie_image, ...). Flux-architecture
    families whose ComfyUI implementation uses BFL-native names
    (``double_blocks.*``/``single_blocks.*``) instead of diffusers
    ``transformer_blocks.*`` MUST override this to ``"transformer."`` — that is
    the diffusers/PEFT/SimpleTuner prefix ComfyUI's ``flux_to_diffusers``
    ``key_map`` registers (``comfy/lora.py::model_lora_keys_unet``). Emitting
    ``diffusion_model.`` with diffusers module names matches NOTHING in the
    Flux key_map and yields a silent zero-effect LoRA. See
    ``ovis_image/saver.py``.
    """

    def save(
        self,
        components: dict[str, Any],
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Extract PEFT LoRA weights and save as safetensors.

        Args:
            components: Dict with ``unet`` (PEFT-wrapped model) and
                optional ``config`` with save settings.
            path: Output file path.
            metadata: Optional extra metadata for the safetensors header.
        """
        arch = self.architecture_name or self.__class__.__name__

        unet = components.get("unet")
        if not unet:
            logger.error(f"{arch}_save_no_model")
            return

        if not hasattr(unet, "peft_config"):
            logger.warning("transformer_not_peft_model")
            return

        # 1. Extract PEFT state dict
        peft_sd = get_peft_model_state_dict(unet)

        # 2. Filter non-LoRA keys, strip prefixes, add diffusion_model.
        final_dict: dict[str, torch.Tensor] = {}
        for key, value in peft_sd.items():
            if not isinstance(value, torch.Tensor):
                continue
            if "lora_A" not in key and "lora_B" not in key:
                continue

            clean = key.replace("base_model.model.", "")
            # Strip any house prefix already present, then apply this saver's
            # configured namespace (default ``diffusion_model.``; Flux-family
            # savers override ``key_prefix`` to ``transformer.``).
            for _known in ("diffusion_model.", "transformer."):
                if clean.startswith(_known):
                    clean = clean[len(_known):]
                    break
            clean = f"{self.key_prefix}{clean}"
            final_dict[clean] = value

        if not final_dict:
            logger.warning("no_lora_weights_found_to_save")
            return

        # 3. Extract rank/alpha from PEFT config
        config = components.get("config", {})
        rank = 16
        alpha = 16.0
        peft_cfg = next(iter(unet.peft_config.values()), None)
        if peft_cfg:
            rank = int(getattr(peft_cfg, "r", 16))
            alpha = float(getattr(peft_cfg, "lora_alpha", rank))

        # 4. Metadata — enrich with Kohya-compatible ss_ keys for
        #    third-party tool compatibility (lora-inspector, Civitai, ComfyUI, etc.)
        save_metadata = {
            "format": "pt",
            "software": '{"name": "Arcane Tuner"}',
            "version": "1.0",
            "ss_network_dim": str(rank),
            "ss_network_alpha": str(alpha),
            "modelspec.architecture": arch,
        }

        # Map training config → Kohya ss_ metadata keys
        if config and isinstance(config, dict):
            _MAP_STR = {
                "optimizer_type": "ss_optimizer",
                "lr_scheduler": "ss_lr_scheduler",
                "mixed_precision": "ss_mixed_precision",
                "lora_name": "ss_output_name",
                "definition_id": "ss_sd_model_name",
                "model_family": "ss_base_model_version",
                "global_triggerword": "ss_training_comment",
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
                    save_metadata[ss_key] = str(val)

            for cfg_key, ss_key in _MAP_NUM.items():
                val = config.get(cfg_key)
                if val is not None:
                    save_metadata[ss_key] = str(val)

            # Resolution — store as "(W,H)" like Kohya
            resolutions = config.get("resolutions")
            if resolutions and isinstance(resolutions, list):
                r = resolutions[0]
                save_metadata["ss_resolution"] = f"({r},{r})"

            # Target modules — from PEFT config
            if peft_cfg:
                tm = getattr(peft_cfg, "target_modules", None)
                if tm:
                    import json as _json
                    save_metadata["ss_network_args"] = _json.dumps({
                        "target_modules": sorted(tm) if not isinstance(tm, str) else [tm],
                    })

            # Dataset info
            datasets = config.get("datasets")
            if datasets and isinstance(datasets, list):
                save_metadata["ss_num_train_images"] = str(
                    sum(d.get("num_repeats", 1) for d in datasets if isinstance(d, dict))
                )

        if metadata:
            save_metadata.update({k: str(v) for k, v in metadata.items()})

        # 5. Save precision (default: bf16)
        save_prec = (
            config.get("save_precision", "bf16").lower()
            if isinstance(config, dict) else "bf16"
        )
        save_dtype = (
            torch.float16 if save_prec == "fp16"
            else torch.bfloat16 if save_prec == "bf16"
            else torch.float32
        )
        for k in final_dict:
            final_dict[k] = final_dict[k].to(save_dtype)

        logger.info(
            f"{arch}_lora_weights_prepared",
            total_keys=len(final_dict),
            save_dtype=str(save_dtype),
            lora_a_keys=sum(1 for k in final_dict if "lora_A" in k),
            lora_b_keys=sum(1 for k in final_dict if "lora_B" in k),
        )

        # 6. Save
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            safe_save_file(final_dict, str(path), metadata=save_metadata)

            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                logger.info("lora_saved", path=str(path), size_mb=f"{size_mb:.2f}MB")
        except (OSError, ValueError, RuntimeError) as e:
            # Log with full context, then RE-RAISE — a swallowed write failure
            # here previously let a training job "succeed" while writing no LoRA
            # file. The caller (CheckpointManager) decides whether this is fatal
            # (final save → fail the job) or log-and-continue (periodic).
            logger.error("save_failed", path=str(path), error=str(e))
            raise
