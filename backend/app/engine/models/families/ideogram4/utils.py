"""Ideogram 4 family utilities: fp8 dequant, patchify, latent-norm, chat render.

Layer-index map: Qwen3-VL has 36 transformer layers; we extract hidden states
from POST-LAYER indices [0,3,6,9,12,15,18,21,24,27,30,33,35] (the upstream
``QWEN3_VL_ACTIVATION_LAYERS``) and concat on the feature dim (13 multi-scale
slices). These are the outputs of decoder layer ``k``. Because HF
``output_hidden_states`` prepends the embedding output at ``hidden_states[0]``,
the post-layer-``k`` activation is read at HF ``hidden_states[k+1]`` -- the
driver applies that ``+1`` shift (mirrors ``microsoft_lens
lens_layers_to_hf_indices``).
"""
from __future__ import annotations

import torch

# 13 Qwen3-VL hidden-state indices (see module docstring).
QWEN3VL_SELECTED_LAYERS: tuple[int, ...] = (
    0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35,
)

# VAE 8x spatial compression, 2x2 latent patchify, 32 base channels.
VAE_SPATIAL_DOWNSCALE = 8
PATCH_FACTOR = 2
VAE_LATENT_CHANNELS = 32


def dequantize_fp8_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Dequantize fp8 weights using per-output-channel float32 scales.

    Weights stored as ``X.weight`` (fp8) paired with ``X.weight_scale``
    (float32, one entry per output channel) become ``weight.to(float32) *
    scale[:, None]``. The ``.weight_scale`` keys are dropped. Tensors without
    a matching scale pass through untouched.
    """
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.endswith(".weight_scale"):
            continue
        scale_key = key + "_scale"
        if scale_key in state_dict:
            scale = state_dict[scale_key].to(torch.float32)
            deq = tensor.to(torch.float32) * scale.reshape(-1, *([1] * (tensor.ndim - 1)))
            out[key] = deq
        else:
            out[key] = tensor
    return out


def patchify_to_seq(latents: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B, (H/2)*(W/2), C*4] via 2x2 space-to-depth."""
    b, c, h, w = latents.shape
    p = PATCH_FACTOR
    x = latents.reshape(b, c, h // p, p, w // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // p) * (w // p), c * p * p)
    return x


def unpatchify_from_seq(seq: torch.Tensor, lat_h: int, lat_w: int) -> torch.Tensor:
    """Inverse of :func:`patchify_to_seq` given the post-patchify grid."""
    b, s, d = seq.shape
    p = PATCH_FACTOR
    c = d // (p * p)
    x = seq.reshape(b, lat_h, lat_w, c, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, c, lat_h * p, lat_w * p)
    return x


def render_chat_prompt(caption: str, tokenizer) -> str:
    """Render a caption through the Qwen3 chat template (text-only)."""
    messages = [{"role": "user", "content": caption}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
