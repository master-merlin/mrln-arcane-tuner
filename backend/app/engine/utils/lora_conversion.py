"""PEFT → Kohya LoRA state-dict converter.

Converts PEFT adapter state dicts to Kohya-SS / ComfyUI compatible
naming conventions, including alpha injection for proper weight scaling.
"""

from __future__ import annotations

import structlog
import torch

logger = structlog.get_logger(__name__)


def convert_peft_to_kohya(
    state_dict: dict[str, torch.Tensor],
    model_type: str = "sdxl",
    alpha: float | None = None,
) -> dict[str, torch.Tensor]:
    """Convert a PEFT state dict into Kohya-SS / ComfyUI compatible format.

    Maps ``lora_A`` / ``lora_B`` keys to ``lora_down`` / ``lora_up`` and
    replaces dots with underscores in module paths.  Injects per-module
    ``.alpha`` tensors when *alpha* is provided.

    Args:
        state_dict: Raw PEFT state dict with ``lora_unet.`` / ``lora_te*`` prefixes.
        model_type: Model family identifier (currently informational).
        alpha: Alpha scaling value to inject for each LoRA module.

    Returns:
        New state dict with Kohya-compatible key names.
    """
    new_sd = {}
    modules_seen = set()
    
    for key, value in state_dict.items():
        try:
            if not isinstance(value, torch.Tensor):
                continue
                
            # 1. Determine Prefix & Strip PEFT-specific prefixes
            prefix = ""
            clean = key
            if "lora_unet" in key or "diffusion_model" in key:
                prefix = "lora_unet"
                clean = key.replace("lora_unet.", "").replace("base_model.model.", "").replace("model.diffusion_model.", "")
            elif "lora_te1" in key:
                prefix = "lora_te1"
                clean = key.replace("lora_te1.", "").replace("base_model.model.", "")
            elif "lora_te2" in key:
                prefix = "lora_te2"
                clean = key.replace("lora_te2.", "").replace("base_model.model.", "")
            else:
                # Unknown key, pass through
                new_sd[key] = value
                continue

            # 2. Extract Suffix and Core Module Path
            # We map LoRA A/B to down/up and ensure weights have the proper .weight suffix
            suffix = ""
            module_path = ""
            
            if ".lora_A.weight" in clean:
                suffix = "lora_down.weight"
                module_path = clean.replace(".lora_A.weight", "")
            elif ".lora_B.weight" in clean:
                suffix = "lora_up.weight"
                module_path = clean.replace(".lora_B.weight", "")
            elif ".lora_down.weight" in clean:
                suffix = "lora_down.weight"
                module_path = clean.replace(".lora_down.weight", "")
            elif ".lora_up.weight" in clean:
                suffix = "lora_up.weight"
                module_path = clean.replace(".lora_up.weight", "")
            elif ".alpha" in clean:
                suffix = "alpha"
                module_path = clean.replace(".alpha", "")
            else:
                # Treat other trailing parts as part of the suffix
                parts = clean.split(".")
                suffix = parts[-1]
                module_path = ".".join(parts[:-1])

            # 3. Assemble the Kohya-style Module Name
            # The core convention is prefix_module_name_dots_replaced_with_underscores.suffix
            module_name = module_path.replace(".", "_")
            
            # 4. Final Key Placement
            final_key = f"{prefix}_{module_name}.{suffix}"
            new_sd[final_key] = value
            modules_seen.add(f"{prefix}_{module_name}")

        except (KeyError, AttributeError, TypeError) as e:
            logger.error("lora_key_conversion_failed", key=key, error=str(e))
            continue

    # 5. Alpha Injection
    # Crucial for ComfyUI/Kohya: every module needs a .alpha key to scale its weights correctly.
    if alpha is not None:
        for module_prefix in modules_seen:
            alpha_key = f"{module_prefix}.alpha"
            if alpha_key not in new_sd:
                # Scale constants are typically stored as float32
                new_sd[alpha_key] = torch.tensor(alpha, dtype=torch.float32)

    return new_sd
