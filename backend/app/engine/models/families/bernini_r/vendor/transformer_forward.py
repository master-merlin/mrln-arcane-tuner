# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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
# Adapted from bernini/models/transformer_wan.py @
# d111e3546f40407ff5326259c87e2a0e1416e583 (see ./REVISION) for the MRLN Arcane
# Tuner. Upstream is itself "Adapted from
# diffusers/models/transformers/transformer_wan.py for Bernini inference:
# variable-length attention with cu_seqlens, optional Ulysses sequence parallel,
# and latents that are patch-embedded by the caller."
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT THIS VENDOR IS (and is NOT)
# ─────────────────────────────────────────────────────────────────────────────
# Bernini-R adds ZERO new weight modules — its checkpoints are 100%-stock Wan
# and load into diffusers 0.39 ``WanTransformer3DModel`` verbatim (so LoRA keys
# stay wan-canonical). The ONLY architectural novelty is data-side: each latent
# stream is patch-embedded SEPARATELY with a per-stream ``source_id`` rotary
# phase, the streams are TOKEN-concatenated ``[cond..., target]``, run through
# ONE full-bidirectional attention window, and the velocity is read back for the
# TARGET tokens only.
#
# We therefore vendor only that assembly (a "forward adapter"), operating on a
# stock ``WanTransformer3DModel`` instance and calling its OWN submodules
# (``patch_embedding``, ``condition_embedder``, ``blocks``, ``norm_out``,
# ``proj_out``). This keeps parity with stock diffusers exact and avoids
# re-hosting any trained weights.
#
# ─────────────────────────────────────────────────────────────────────────────
# MRLN-PATCH divergences from upstream bernini/models/transformer_wan.py
# ─────────────────────────────────────────────────────────────────────────────
#  * flash-attn 2.8.3 varlen (``varlen_attention`` + ``cu_seqlens_*``) is NOT
#    ported. Upstream packs ``[total_tokens, heads, head_dim]`` and slices
#    per-sample attention windows with ``cu_seqlens``. For v1 (single sample per
#    forward, one contiguous ``[cond, target]`` sequence) that window is exactly
#    the whole packed sequence, so we run the STOCK block's batched SDPA over
#    ``[B, seq, ...]`` — full bidirectional attention spanning cond+target,
#    which is what the per-SAMPLE (not per-STREAM) upstream ``cu_seqlens`` encode.
#  * Ulysses sequence-parallel plumbing (``prepare_inputs_for_sp``,
#    ``gather_seq_scatter_heads``, ``get_parallel_state`` …) is NOT ported —
#    single-GPU only, where every upstream collective is a no-op.
#  * Upstream's ``WanRotaryPosEmbed`` emits COMPLEX freqs (``use_real=False``)
#    and applies ``source_id`` via a complex multiply ``freqs = freqs *
#    freqs_visual_id``. Stock diffusers 0.39's ``WanRotaryPosEmbed`` emits REAL
#    ``(cos, sin)`` tables (``use_real=True, repeat_interleave_real=True``)
#    consumed by ``WanAttnProcessor``. :func:`source_id_rope` reproduces the
#    identical rotation in the real domain (``angle_total = angle_base +
#    angle_id``) so the stock blocks/weights apply it unchanged. This is pinned
#    against the upstream complex formula by the forward-parity test.
#  * The caller-side ``patch_vae_latent`` (patch-embed + rope, returned to the
#    caller) is preserved as :func:`patch_vae_latent`, minus the ``cu_seqlens``
#    bookkeeping.

from __future__ import annotations

from typing import Any

import torch
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput

# Rotary base. Stock diffusers ``WanRotaryPosEmbed`` is constructed with the
# default ``theta=10000.0`` (``transformer_wan.py``: ``WanRotaryPosEmbed(
# attention_head_dim, patch_size, rope_max_seq_len)`` — no theta override), and
# upstream Bernini's ``WanRotaryPosEmbed`` defaults ``theta=10000.0`` too, so the
# base rope and the source_id phase share the same theta. The stock module does
# not expose theta as an attribute; this constant mirrors both.
BERNINI_ROPE_THETA: float = 10000.0


def source_id_rope(
    model: Any, latent: torch.Tensor, source_id: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stock ``(freqs_cos, freqs_sin)`` rope for ``latent`` with the ``source_id``
    rotary phase multiplied in (upstream ``use_src_id_rotary_emb``).

    Returns the two real tables ``[1, T*H*W, 1, head_dim]`` the stock
    ``WanAttnProcessor`` consumes (it reads ``freqs_cos[..., 0::2]`` and
    ``freqs_sin[..., 1::2]``). ``source_id`` may be fractional (upstream
    interpolates references into the trained id range).
    """
    freqs_cos, freqs_sin = model.rope(
        latent
    )  # [1, seq, 1, head_dim], repeat-interleaved

    # MRLN-PATCH: upstream multiplies a COMPLEX id-phase into COMPLEX base freqs
    # (``freqs = freqs * freqs_visual_id``). source_id=0 → phase exp(i*0)=1+0j is
    # the identity, so the no-condition / target-token path stays byte-identical
    # to stock diffusers. Short-circuit it (also avoids needless work).
    if float(source_id) == 0.0:
        return freqs_cos, freqs_sin

    head_dim = int(model.config.attention_head_dim)
    device = freqs_cos.device

    # Complex id-phase per rotary pair: exp(i * source_id * w_k), k in [0,head/2).
    # Computed in float64 for phase accuracy, then folded into the real tables.
    id_freqs = get_1d_rotary_pos_embed(
        head_dim,
        torch.tensor([float(source_id)], dtype=torch.float64, device=device),
        BERNINI_ROPE_THETA,
        use_real=False,
        repeat_interleave_real=False,
        freqs_dtype=torch.float64,
    )  # complex [1, head_dim // 2]
    id_cos = (
        id_freqs.real.to(freqs_cos.dtype)
        .repeat_interleave(2, dim=1)
        .view(1, 1, 1, head_dim)
    )
    id_sin = (
        id_freqs.imag.to(freqs_sin.dtype)
        .repeat_interleave(2, dim=1)
        .view(1, 1, 1, head_dim)
    )

    # Real-domain complex multiply: (cos_b + i sin_b)(cos_id + i sin_id). The
    # repeat-interleaved layout makes this elementwise-correct for the pairs the
    # stock apply_rotary_emb reads (cos[0::2], sin[1::2]).
    new_cos = freqs_cos * id_cos - freqs_sin * id_sin
    new_sin = freqs_sin * id_cos + freqs_cos * id_sin
    return new_cos, new_sin


def patch_vae_latent(
    model: Any, latent: torch.Tensor, source_id: float
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Patch-embed a VAE latent ``[B,C,T,H,W]`` into ``[B, T*H*W, inner]`` tokens,
    with its ``source_id`` rope — the caller assembles the token sequence.

    Port of upstream ``WanTransformer3DModel.patch_vae_latent`` (rope first, then
    the stock ``patch_embedding`` conv + flatten/transpose).
    """
    rope = source_id_rope(model, latent, source_id)
    hidden_states = model.patch_embedding(latent).flatten(2).transpose(1, 2)
    return hidden_states, rope


def bernini_packed_forward(
    model: Any,
    cond_latents: list[torch.Tensor],
    cond_source_ids: list[float],
    target_latent: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    *,
    target_source_id: float = 0.0,
    return_dict: bool = False,
):
    """Bernini-R packed forward over a stock ``WanTransformer3DModel``.

    Assembles ``[cond..., target]`` token streams (upstream
    ``torch.cat(cond_lats + [noisy_latent], dim=1)`` — condition tokens FIRST,
    target LAST), runs the stock blocks as ONE full-bidirectional attention
    window, and returns the velocity for the TARGET tokens only, unpatchified to
    ``[B, out_ch, F, H, W]``.

    Args:
        model: a stock diffusers ``WanTransformer3DModel`` (weights loaded).
        cond_latents: clean condition-video latents ``[B,C,T,H,W]`` (may be []).
        cond_source_ids: matching ``source_id`` per condition stream (e.g. v2v →
            ``[1.0]``). Same length as ``cond_latents``.
        target_latent: the noisy target latent ``[B,C,T,H,W]`` (``source_id=0``).
        timestep: raw ``[0,1000]`` per-sample scalar(s) — shared by ALL tokens,
            including the clean condition tokens (upstream ``timestep =
            t.expand(1)``; NO per-token t=0 trick).
        encoder_hidden_states: UMT5 text features ``[B, L, text_dim]``.
        target_source_id: source_id for the target stream (0.0; kept explicit for
            symmetry with degenerate cases).
        return_dict: wrap the output in ``Transformer2DModelOutput`` if True.

    Returns:
        Velocity ``[B, out_ch, F, H, W]`` for the target stream, or a
        ``Transformer2DModelOutput`` / 1-tuple per ``return_dict``.
    """
    if len(cond_latents) != len(cond_source_ids):
        raise ValueError(
            f"cond_latents ({len(cond_latents)}) and cond_source_ids "
            f"({len(cond_source_ids)}) length mismatch."
        )

    p_t, p_h, p_w = model.config.patch_size

    # 1. Patch-embed each stream separately, with its source_id rope. Order is
    #    condition streams first, target last (upstream cat ordering).
    streams = list(cond_latents) + [target_latent]
    source_ids = list(cond_source_ids) + [target_source_id]

    token_parts: list[torch.Tensor] = []
    cos_parts: list[torch.Tensor] = []
    sin_parts: list[torch.Tensor] = []
    for latent, sid in zip(streams, source_ids):
        tokens, (cos, sin) = patch_vae_latent(model, latent, sid)
        token_parts.append(tokens)
        cos_parts.append(cos)
        sin_parts.append(sin)

    hidden_states = torch.cat(token_parts, dim=1)  # [B, total_tokens, inner]
    rotary_emb = (torch.cat(cos_parts, dim=1), torch.cat(sin_parts, dim=1))

    # 2. Condition embedding — single raw timestep, shared over every token.
    temb, timestep_proj, enc_hs, _ = model.condition_embedder(
        timestep, encoder_hidden_states, None, timestep_seq_len=None
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))  # [B, 6, inner]

    # 3. Stock transformer blocks. attn1 (self) spans the WHOLE packed sequence →
    #    full bidirectional attention over cond+target (recon risk #1); attn2 is
    #    text cross-attention. MRLN-PATCH: upstream's packed cu_seqlens/Ulysses
    #    forward is replaced by the stock block loop (batched SDPA).
    #
    # MRLN-PATCH: this packed loop BYPASSES ``WanTransformer3DModel.forward``,
    # which is the file where diffusers implements the gradient-checkpointing
    # gate. A naive ``for block in model.blocks`` therefore made
    # ``enable_gradient_checkpointing()`` silently INERT on the packed path —
    # harmless for the 1.3B (~13.6 GB peak), fatal for the 14B (40 blocks × dim
    # 5120 × ~13.8k packed tokens retained ~80-100 GB of activations → 189.9 GiB
    # OOM at the first training forward on a 95.6 GiB card). We re-implement the
    # STOCK gate verbatim (diffusers 0.39 transformer_wan.py:694-701): route each
    # block through the model's OWN ``_gradient_checkpointing_func`` (installed by
    # ``enable_gradient_checkpointing`` — modeling_utils.py:285-313) iff grad is
    # enabled AND checkpointing is on. The ``torch.is_grad_enabled()`` half keeps
    # sampling (which runs under ``torch.no_grad()``) on the plain, non-recompute
    # path, so no-grad numerics are byte-unchanged.
    grad_ckpt_func = getattr(model, "_gradient_checkpointing_func", None)
    if (
        torch.is_grad_enabled()
        and getattr(model, "gradient_checkpointing", False)
        and grad_ckpt_func is not None
    ):
        for block in model.blocks:
            hidden_states = grad_ckpt_func(
                block, hidden_states, enc_hs, timestep_proj, rotary_emb
            )
    else:
        for block in model.blocks:
            hidden_states = block(hidden_states, enc_hs, timestep_proj, rotary_emb)

    # 4. Output norm + projection (stock final head; temb is the per-sample
    #    [B, inner] embedding, broadcast over all tokens).
    shift, scale = (model.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
    hidden_states = (
        model.norm_out(hidden_states.float()) * (1 + scale) + shift
    ).type_as(hidden_states)
    hidden_states = model.proj_out(hidden_states)

    # 5. Slice the TARGET tokens (the tail of the packed sequence) and unpatchify.
    cond_total = sum(part.shape[1] for part in token_parts[:-1])
    b = target_latent.shape[0]
    ppf = target_latent.shape[2] // p_t
    pph = target_latent.shape[3] // p_h
    ppw = target_latent.shape[4] // p_w

    target_tokens = hidden_states[:, cond_total:, :]
    target_tokens = target_tokens.reshape(b, ppf, pph, ppw, p_t, p_h, p_w, -1)
    target_tokens = target_tokens.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = (
        target_tokens.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    )  # [B, out_ch, F, H, W]

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
