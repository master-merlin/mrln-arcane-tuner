# Vendored from huggingface/diffusers @ 6a71b6e332abae01a05d36133003e5370ca1d0a8 (0.39.0.dev0)
# Source: src/diffusers/pipelines/krea2/pipeline_krea2.py
#
# Extracts only the standalone conditioning helpers from Krea2Pipeline:
#   - get_text_hidden_states  (refactored from pipeline method → module fn)
#   - prepare_position_ids    (static → module fn, unchanged logic)
#   - pack_latents            (from _pack_latents, patch_size param)
#   - unpack_latents          (from _unpack_latents, patch_size param,
#                              no vae_scale_factor baked in — caller passes pixel dims)
#
# None of these require a real Qwen3VL / tokenizer at import time.
# get_text_hidden_states is ported faithfully; unit testing requires a live
# text encoder and is deferred to Phase 2.
#
# Copyright 2026 Krea AI and The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    # Imported for type hints only — no hard runtime dependency on transformers here.
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# ── Constants (match Krea2Pipeline.__init__) ──────────────────────────────────
_PROMPT_TEMPLATE_PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
_PROMPT_TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
_PROMPT_TEMPLATE_ENCODE_START_IDX = 34      # system-prefix token count to drop
_PROMPT_TEMPLATE_ENCODE_NUM_SUFFIX_TOKENS = 5  # assistant-suffix token count


def prepare_position_ids(
    text_seq_len: int,
    grid_height: int,
    grid_width: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the ``(text_seq_len + grid_height * grid_width, 3)`` rotary coordinates
    for the combined [text | image] sequence.

    Text tokens sit at the origin ``(0, 0, 0)``; image tokens carry their
    ``(0, h, w)`` latent-grid coordinates.

    Args:
        text_seq_len: Number of text tokens (after dropping the system prefix).
        grid_height:  Latent grid height in patches (image_height // patch_size).
        grid_width:   Latent grid width in patches  (image_width  // patch_size).
        device:       Target device for the returned tensor.

    Returns:
        Tensor of shape ``(text_seq_len + grid_height * grid_width, 3)``.
    """
    text_ids = torch.zeros(text_seq_len, 3, device=device)
    image_ids = torch.zeros(grid_height, grid_width, 3, device=device)
    image_ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
    image_ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
    image_ids = image_ids.reshape(grid_height * grid_width, 3)
    return torch.cat([text_ids, image_ids], dim=0)


def pack_latents(
    latents: torch.Tensor,
    patch_size: int = 2,
) -> torch.Tensor:
    """Pack ``(B, C, H, W)`` latents into ``(B, (H/p)*(W/p), C*p*p)`` patches.

    Args:
        latents:    Float tensor of shape ``(B, C, H, W)``.
        patch_size: Side length of each square patch (default 2 for Krea-2).

    Returns:
        Packed tensor of shape ``(B, num_patches, C * patch_size ** 2)``.
    """
    B, C, H, W = latents.shape
    p = patch_size
    latents = latents.view(B, C, H // p, p, W // p, p)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(B, (H // p) * (W // p), C * p * p)
    return latents


def unpack_latents(
    latents: torch.Tensor,
    height: int,
    width: int,
    patch_size: int = 2,
) -> torch.Tensor:
    """Unpack ``(B, num_patches, C*p*p)`` back to ``(B, C, 1, H, W)``.

    The extra ``1`` on the frame dimension matches the Krea-2 / Qwen-Image VAE
    convention (5-D latents ``B C F H W``).

    Args:
        latents:    Packed tensor of shape ``(B, num_patches, channels_packed)``.
        height:     Latent grid height in pixels (NOT patch units).
        width:      Latent grid width in pixels (NOT patch units).
        patch_size: Side length of each square patch (default 2 for Krea-2).

    Returns:
        Tensor of shape ``(B, C, 1, H, W)`` where ``C = channels_packed / p**2``.
    """
    B, _, channels_packed = latents.shape
    p = patch_size
    h = height // p   # grid height in patches
    w = width // p    # grid width in patches
    C = channels_packed // (p * p)

    latents = latents.view(B, h, w, C, p, p)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(B, C, 1, height, width)
    return latents


def get_text_hidden_states(
    tokenizer: "PreTrainedTokenizerBase",
    text_encoder: "PreTrainedModel",
    prompt: str | list[str],
    select_layers: tuple[int, ...] | list[int],
    max_sequence_length: int = 512,
    device: torch.device | None = None,
    prefix_idx: int = _PROMPT_TEMPLATE_ENCODE_START_IDX,
    num_suffix_tokens: int = _PROMPT_TEMPLATE_ENCODE_NUM_SUFFIX_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise *prompt* into the fixed-length Krea-2 layout and tap the
    selected encoder hidden states.

    Krea-2 pads the prompt in the middle of the chat template::

        [prefix | prompt | PAD ... | suffix]

    The suffix tokens therefore sit *downstream* of the padding.  To match
    training-time mRoPE positions, cumulative-valid-token positions are built
    explicitly (padding does not consume a position).

    After the encoder forward, the first ``prefix_idx`` rows (system prefix)
    are dropped so the returned tensors start at the user prompt.

    Args:
        tokenizer:           Qwen2 tokenizer paired with the text encoder.
        text_encoder:        Qwen3VLModel with ``output_hidden_states=True`` support.
        prompt:              A single string or a list of strings.
        select_layers:       Indices into ``hidden_states`` tuple (0 = embedding output)
                             to stack; length must equal ``transformer.config.num_text_layers``.
        max_sequence_length: Total prompt token budget (default 512).
        device:              Device to move tensors onto (default: text_encoder device).
        prefix_idx:          Number of system-prefix tokens to drop (default 34).
        num_suffix_tokens:   Number of assistant-suffix tokens appended after padding (default 5).

    Returns:
        A ``(hidden_states, attention_mask)`` tuple:
        - ``hidden_states``:  ``(B, text_seq_len, len(select_layers), text_hidden_dim)``
        - ``attention_mask``: ``(B, text_seq_len)`` bool, True for valid tokens.
    """
    if device is None:
        # Infer from the text encoder's first parameter.
        device = next(text_encoder.parameters()).device

    if isinstance(prompt, str):
        prompt = [prompt]

    text = [_PROMPT_TEMPLATE_PREFIX + p for p in prompt]
    text_tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_sequence_length + prefix_idx - num_suffix_tokens,
        return_tensors="pt",
    ).to(device)

    suffix_tokens = tokenizer(
        [_PROMPT_TEMPLATE_SUFFIX] * len(text),
        return_tensors="pt",
    ).to(device)

    input_ids = torch.cat([text_tokens.input_ids, suffix_tokens.input_ids], dim=1)
    attention_mask = torch.cat([text_tokens.attention_mask, suffix_tokens.attention_mask], dim=1).bool()

    # Build cumulative-valid-token position IDs (see docstring for why).
    position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
    # Expand to 3 mRoPE axes (T/H/W are identical for text).
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
    )

    hidden_states = torch.stack(
        [outputs.hidden_states[i] for i in select_layers],
        dim=2,
    )  # (B, full_seq, num_layers, dim)

    # Drop the system prefix rows.
    hidden_states = hidden_states[:, prefix_idx:]
    attention_mask = attention_mask[:, prefix_idx:]
    return hidden_states, attention_mask
