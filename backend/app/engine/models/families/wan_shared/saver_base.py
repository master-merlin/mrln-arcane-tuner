"""Shared WAN LoRA key mapping — PEFT diffusers keys → ComfyUI ``diffusion_model.*``.

ComfyUI / kohya consume WAN LoRAs in the non-diffusers naming that diffusers'
``_convert_non_diffusers_wan_lora_to_diffusers`` reverses::

    diffusion_model.blocks.{i}.self_attn.{q,k,v,o}.lora_{down,up}.weight
    diffusion_model.blocks.{i}.cross_attn.{q,k,v,o}.lora_{down,up}.weight
    diffusion_model.blocks.{i}.cross_attn.{k_img,v_img}.lora_{down,up}.weight   # i2v
    diffusion_model.blocks.{i}.ffn.{0,2}.lora_{down,up}.weight

:func:`_convert_diffusers_to_comfy` maps the PEFT-extracted diffusers keys to
that format:

    blocks.{i}.attn1.{to_q,to_k,to_v,to_out.0}  → self_attn.{q,k,v,o}
    blocks.{i}.attn2.{to_q,to_k,to_v,to_out.0}  → cross_attn.{q,k,v,o}
    blocks.{i}.attn2.{add_k_proj,add_v_proj}    → cross_attn.{k_img,v_img}   # i2v
    blocks.{i}.ffn.{net.0.proj,net.2}           → ffn.{0,2}
    lora_A → lora_down , lora_B → lora_up

The transformer block layout (and therefore the key mapping) is identical
between WAN 2.1 and WAN 2.2 (each 2.2 expert is the same transformer shape as
2.1), so both families' savers share this module — wan22 converts each expert
separately through the same function.
"""

from __future__ import annotations

import re

import structlog
import torch

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
        logger.warning("wan_saver_unconverted_keys", keys=unconverted)

    return out


def _map_suffix(suffix: str, table: dict[str, str], prefix: str) -> str | None:
    sub = table.get(suffix)
    if sub is None:
        return None
    return f"{prefix}.{sub}"
