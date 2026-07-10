# Vendored from huggingface/diffusers (local pinned release 0.39.0)
# Source: src/diffusers/loaders/lora_conversion_utils.py :: _convert_non_diffusers_lumina2_lora_to_diffusers
# vendored for boogu_image family — local diffusers 0.39.0
#
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTE: This is diffusers' PRIVATE helper (leading underscore, no public API
# guarantee). Vendored verbatim (byte-faithful body, only `re`/`torch` imports
# added since the original lives inside a larger module that already imports
# them) for insulation against upstream diffusers changes. It converts a
# non-diffusers ("ComfyUI-style") Lumina2 LoRA state dict — which BooguImage's
# architecture is derived from — into diffusers' native key naming. It is not
# used by this task; a later boogu_image task (LoRA save/load) will consume
# it, most likely after adapting the hardcoded qkv split sizes ([2304, 768,
# 768], diffusers' stock Lumina2 dims) to BooguImage's real dims (to_q=3360,
# to_k=to_v=840 with kv_heads=7).

import re

import torch


def _convert_non_diffusers_lumina2_lora_to_diffusers(state_dict):
    # Remove "diffusion_model." prefix from keys.
    state_dict = {k[len("diffusion_model.") :]: v for k, v in state_dict.items()}
    converted_state_dict = {}

    def get_num_layers(keys, pattern):
        layers = set()
        for key in keys:
            match = re.search(pattern, key)
            if match:
                layers.add(int(match.group(1)))
        return len(layers)

    def process_block(prefix, index, convert_norm):
        # Process attention qkv: pop lora_A and lora_B weights.
        lora_down = state_dict.pop(f"{prefix}.{index}.attention.qkv.lora_A.weight")
        lora_up = state_dict.pop(f"{prefix}.{index}.attention.qkv.lora_B.weight")
        for attn_key in ["to_q", "to_k", "to_v"]:
            converted_state_dict[f"{prefix}.{index}.attn.{attn_key}.lora_A.weight"] = lora_down
        for attn_key, weight in zip(["to_q", "to_k", "to_v"], torch.split(lora_up, [2304, 768, 768], dim=0)):
            converted_state_dict[f"{prefix}.{index}.attn.{attn_key}.lora_B.weight"] = weight

        # Process attention out weights.
        converted_state_dict[f"{prefix}.{index}.attn.to_out.0.lora_A.weight"] = state_dict.pop(
            f"{prefix}.{index}.attention.out.lora_A.weight"
        )
        converted_state_dict[f"{prefix}.{index}.attn.to_out.0.lora_B.weight"] = state_dict.pop(
            f"{prefix}.{index}.attention.out.lora_B.weight"
        )

        # Process feed-forward weights for layers 1, 2, and 3.
        for layer in range(1, 4):
            converted_state_dict[f"{prefix}.{index}.feed_forward.linear_{layer}.lora_A.weight"] = state_dict.pop(
                f"{prefix}.{index}.feed_forward.w{layer}.lora_A.weight"
            )
            converted_state_dict[f"{prefix}.{index}.feed_forward.linear_{layer}.lora_B.weight"] = state_dict.pop(
                f"{prefix}.{index}.feed_forward.w{layer}.lora_B.weight"
            )

        if convert_norm:
            converted_state_dict[f"{prefix}.{index}.norm1.linear.lora_A.weight"] = state_dict.pop(
                f"{prefix}.{index}.adaLN_modulation.1.lora_A.weight"
            )
            converted_state_dict[f"{prefix}.{index}.norm1.linear.lora_B.weight"] = state_dict.pop(
                f"{prefix}.{index}.adaLN_modulation.1.lora_B.weight"
            )

    noise_refiner_pattern = r"noise_refiner\.(\d+)\."
    num_noise_refiner_layers = get_num_layers(state_dict.keys(), noise_refiner_pattern)
    for i in range(num_noise_refiner_layers):
        process_block("noise_refiner", i, convert_norm=True)

    context_refiner_pattern = r"context_refiner\.(\d+)\."
    num_context_refiner_layers = get_num_layers(state_dict.keys(), context_refiner_pattern)
    for i in range(num_context_refiner_layers):
        process_block("context_refiner", i, convert_norm=False)

    core_transformer_pattern = r"layers\.(\d+)\."
    num_core_transformer_layers = get_num_layers(state_dict.keys(), core_transformer_pattern)
    for i in range(num_core_transformer_layers):
        process_block("layers", i, convert_norm=True)

    if len(state_dict) > 0:
        raise ValueError(f"`state_dict` should be empty at this point but has {state_dict.keys()=}")

    for key in list(converted_state_dict.keys()):
        converted_state_dict[f"transformer.{key}"] = converted_state_dict.pop(key)

    return converted_state_dict
