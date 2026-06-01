"""Microsoft Lens latent packing + text-feature utilities.

Mirrors the math in the vendored ``lens/pipeline.py``:

* Latents live in sequence space ``[B, S, 128]`` (S = latent_h * latent_w,
  128 = 32 VAE channels * 2 * 2 patchify), channel order ``(c, p1, p2)``.
* BN-normalize over the 128-ch dim with the FLUX.2 VAE's running stats.
* Lens wraps prompts in a chat template, takes 4 hidden layers, and drops
  the first 97 tokens (txt_offset).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from einops import rearrange
from torch import Tensor

# Lens chat-template constants (verbatim from lens/pipeline.py).
CHAT_SYSTEM = (
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background."
)
CHAT_ASSISTANT_THINKING = "Need to generate one image according to the description."
DEFAULT_TXT_OFFSET = 97
DEFAULT_SELECTED_LAYERS: Tuple[int, ...] = (5, 11, 17, 23)


def patchify_to_seq(latents: Tensor) -> Tensor:
    """``[B, C, H, W]`` -> ``[B, (H/2)(W/2), C*4]`` (channel order ``c p1 p2``)."""
    return rearrange(
        latents, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=2, p2=2,
    )


def unpatchify_from_seq(seq: Tensor, latent_h: int, latent_w: int) -> Tensor:
    """Inverse of :func:`patchify_to_seq`. ``[B, S, C*4]`` -> ``[B, C, H, W]``."""
    return rearrange(
        seq, "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
        h=latent_h, w=latent_w, p1=2, p2=2,
    )


def lens_layers_to_hf_indices(selected: Sequence[int]) -> List[int]:
    """Lens post-layer indices -> stock HF ``output_hidden_states`` indices.

    HF records ``hidden_states[0]`` = embeddings and ``hidden_states[k+1]`` =
    output of decoder layer ``k``. Lens captures the output of layer ``k`` for
    ``k`` in ``selected``, so the HF index is ``k + 1``.
    """
    return [int(k) + 1 for k in selected]


def _bn_eps(vae: object) -> float:
    return float(getattr(vae.bn, "eps", 1e-5))


def bn_normalize_seq(seq: Tensor, vae: object) -> Tensor:
    """Normalize sequence-space latents ``[B, S, 128]`` with VAE BN stats."""
    mean = vae.bn.running_mean.view(1, 1, -1).to(seq.device, seq.dtype)
    std = torch.sqrt(
        vae.bn.running_var.view(1, 1, -1).to(seq.device, seq.dtype) + _bn_eps(vae)
    )
    return (seq - mean) / std


def bn_denormalize_seq(seq: Tensor, vae: object) -> Tensor:
    """Inverse of :func:`bn_normalize_seq`."""
    mean = vae.bn.running_mean.view(1, 1, -1).to(seq.device, seq.dtype)
    std = torch.sqrt(
        vae.bn.running_var.view(1, 1, -1).to(seq.device, seq.dtype) + _bn_eps(vae)
    )
    return seq * std + mean


def drop_txt_offset(
    features: List[Tensor], mask: Tensor, offset: int = DEFAULT_TXT_OFFSET,
) -> Tuple[List[Tensor], Tensor]:
    """Drop the fixed chat-template prefix (first ``offset`` tokens).

    Mirrors ``LensPipeline._get_text_embeddings``: when the sequence is
    longer than ``offset`` we slice ``[:, offset:, :]``; otherwise we return
    zero-length features + mask.
    """
    seq_len = features[0].shape[1]
    if seq_len > offset:
        sliced = [f[:, offset:, :].contiguous() for f in features]
        return sliced, mask[:, offset:].contiguous()
    empty = [f[:, :0, :].contiguous() for f in features]
    return empty, mask[:, :0].contiguous()


def render_chat_prompt(prompt: str, tokenizer: object) -> str:
    """Render one prompt through Lens's chat template (no tokenization)."""
    conversation = [
        {"role": "system", "content": CHAT_SYSTEM, "thinking": None},
        {"role": "user", "content": prompt, "thinking": None},
        {"role": "assistant", "thinking": CHAT_ASSISTANT_THINKING, "content": ""},
    ]
    text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False,
    )
    return text.split("<|return|>")[0]
