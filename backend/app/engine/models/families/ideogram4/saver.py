"""Ideogram 4 LoRA saver -- generic ComfyUI / ai-toolkit safetensors format.

Training runs on diffusers' ``Ideogram4Transformer2DModel``, whose attention has
split ``to_q`` / ``to_k`` / ``to_v`` / ``to_out.0`` Linears; the RELEASED file
layout (ARCHITECTURE D2, the user's LANE-150 interview answer Q3) is the ported
model's fused ``attention.qkv`` / ``attention.o``. The saver re-fuses exactly:

    qkv.lora_A = cat([A_q, A_k, A_v], dim=0)          (3r x H)
    qkv.lora_B = block_diag(B_q, B_k, B_v)            (3H x 3r)

so ``qkv.lora_B @ qkv.lora_A`` is the vertical stack of the three split deltas
(the zero off-diagonal blocks kill every cross term). The fused key carries rank
3r where a pre-LANE-150 file (one shared A) carried rank r; diffusers' converter
reads the rank from the tensor shape and chunks ``lora_B`` in thirds
(``_convert_non_diffusers_ideogram4_lora_to_diffusers``,
lora_conversion_utils.py:2993, :3029), so both load.
"""
from __future__ import annotations

import re

import torch

from app.engine.core.pipeline.saver_base import GenericLoRASaver

_SPLIT_QKV = re.compile(r"^(?P<base>.*\.attention\.)to_(?P<proj>[qkv])\.lora_(?P<ab>[AB])\.weight$")
_SPLIT_OUT = re.compile(r"^(?P<base>.*\.attention\.)to_out\.0\.(?P<rest>lora_[AB]\.weight)$")


class IdeogramV4Saver(GenericLoRASaver):
    """Save Ideogram 4 LoRA weights in the released fused-key format."""

    architecture_name = "ideogram4"

    def remap_lora_state_dict(
        self, state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        split: dict[str, dict[str, torch.Tensor]] = {}
        for key, value in state_dict.items():
            m = _SPLIT_QKV.match(key)
            if m:
                split.setdefault(m["base"], {})[m["proj"] + m["ab"]] = value
                continue
            m = _SPLIT_OUT.match(key)
            if m:
                out[f"{m['base']}o.{m['rest']}"] = value
                continue
            out[key] = value
        for base, parts in split.items():
            # A projection a custom target list left untrained has a zero delta;
            # zero A/B blocks of the trained ones' rank write exactly that, so the
            # fused file stays the trained function (to_q/k/v are all H x H).
            ref_a = next(parts[p + "A"] for p in "qkv" if p + "A" in parts)
            ref_b = next(parts[p + "B"] for p in "qkv" if p + "B" in parts)
            a = [parts.get(p + "A", torch.zeros_like(ref_a)) for p in "qkv"]
            b = [parts.get(p + "B", torch.zeros_like(ref_b)) for p in "qkv"]
            out[f"{base}qkv.lora_A.weight"] = torch.cat(a, dim=0)
            out[f"{base}qkv.lora_B.weight"] = torch.block_diag(*b)
        return out
