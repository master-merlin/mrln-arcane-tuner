"""UMT5-XXL prompt encoding for WAN families.

WAN 2.1 / 2.2 use a UMT5-XXL encoder (``transformers.UMT5EncoderModel``) with a
max sequence length of ~512. The encoder is run with an attention mask and the
last hidden state is returned. Padding positions are zeroed (masked) so the
transformer's cross-attention ignores them, matching the diffusers
``WanPipeline._get_t5_prompt_embeds`` behavior.

The result is a single ``[B, L, D]`` tensor, which is compatible with the
``TextEmbeddingCache`` single-tensor cache path (``te1``) used by the
latent-diffusion trainers. This module is the one place the WAN families
(present and future) share their text-encode logic.
"""

from __future__ import annotations

from typing import Any

import torch

from app.engine.core.text_encoding import TextEncoderOutput

WAN_TE_MAX_LENGTH = 512


def encode_umt5(
    text_encoder: Any,
    tokenizer: Any,
    captions: list[str],
    device: torch.device,
    dtype: torch.dtype,
    *,
    max_length: int = WAN_TE_MAX_LENGTH,
) -> TextEncoderOutput:
    """Encode captions with UMT5-XXL → ``TextEncoderOutput`` (``[B, L, D]``).

    Padding positions are masked to zero so the WAN transformer's
    cross-attention ignores them. The attention mask is also returned (handy
    for callers that want it) but the primary output is the masked
    ``embeddings`` tensor — a single tensor compatible with the te1 cache.

    Args:
        text_encoder: A ``UMT5EncoderModel`` (or compatible) instance.
        tokenizer: The matching tokenizer (``AutoTokenizer``).
        captions: Batch of caption strings.
        device: Target device for tokenization tensors.
        dtype: Target dtype for the returned embeddings.
        max_length: Token cap (default 512).

    Returns:
        ``TextEncoderOutput`` with ``embeddings [B, L, D]`` (padding zeroed)
        and ``attention_mask [B, L]``.
    """
    text_inputs = tokenizer(
        captions,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    hidden = outputs.last_hidden_state

    # Zero out padding positions so cross-attention ignores them. Broadcast the
    # [B, L] mask over the feature dim.
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    hidden = hidden * mask

    return TextEncoderOutput(
        embeddings=hidden.to(dtype=dtype),
        attention_mask=attention_mask,
    )
