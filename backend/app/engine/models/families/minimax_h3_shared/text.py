"""The MiniMax-H3 text path, shared by the video and image models.

Evidence for the tap: MiniMax-H3 README "The H3-Encoder uses the full
pretrained weights of Qwen3-VL-32B and provides the hidden states from its
50th layer"; ai-toolkit ``text_encoder.py`` "the **unnormalized**
hidden_states[50] ... hidden_states[0] is the embedding output". In
transformers 5.14.1 ``hidden_states`` has ``num_hidden_layers + 1`` entries,
``[0]`` is the input embeddings and only ``[-1]`` carries the final RMSNorm,
so ``[50]`` on the 64-layer stack is decoder layer 49's raw output.
Presentation: raw tokens, no chat template, no special tokens; the caption is
capped at the definition's ``te.max_length``. The tap index and the cap are
read from each family's definition and passed in here.

The cache key carries every input the embedding is a function of; its leading
family prefix is passed in by the caller, so two families never share a cache
entry and an existing family's key stays what it always was.

The encoder release drops every reference WITHOUT a device-to-host copy: the
module is gone right after, so the copy buys nothing, and on the 63 GB
encoder it crashed the process natively (an access violation inside
``Module.to("cpu")``, no Python exception).
"""

from __future__ import annotations

import gc
import hashlib
from typing import Any

import torch


def tokenizer_only(tokenizer: Any) -> Any:
    """The tokenizer under an ``AutoProcessor`` (or a bare tokenizer)."""
    return getattr(tokenizer, "tokenizer", tokenizer)


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """SHA-256 over the tokenizer class + its full vocabulary. A different
    vocabulary yields different ids for the same caption, hence a different
    embedding; the cache key must carry it."""
    tok = tokenizer_only(tokenizer)
    vocab = tok.get_vocab()
    digest = hashlib.sha256()
    digest.update(type(tok).__name__.encode("utf-8"))
    for token, idx in sorted(vocab.items(), key=lambda kv: (kv[1], kv[0])):
        digest.update(f"{idx}:{token}\n".encode("utf-8"))
    return digest.hexdigest()


def te_cache_scope(tap: int, fingerprint: str) -> str:
    """Path segment the disk cache lives under: tap index + tokenizer."""
    return f"tap{int(tap)}-{fingerprint[:16]}"


def te_cache_key(prefix: str, definition_id: str, tap: int, fingerprint: str, prompt: str) -> str:
    """The string a cached embedding is keyed on: the family ``prefix``
    (e.g. ``"<family>|"``), the definition, the tap index, the tokenizer and
    the prompt. Human-readable; ``TextEmbeddingCache`` hashes it for the
    filename."""
    return f"{prefix}{definition_id}|tap={int(tap)}|tok={fingerprint[:16]}|{prompt}"


def encode_caption(
    tokenizer: Any,
    text_encoder: Any,
    caption: str,
    *,
    tap: int,
    max_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """``[L, D]`` layer-tap hidden states for ONE caption, no padding."""
    tok = tokenizer_only(tokenizer)
    ids = list(tok(caption, add_special_tokens=False)["input_ids"])[: int(max_tokens)]
    if not ids:
        # Empty (dropout / unconditional) prompt: one pad token keeps the
        # sequence non-degenerate (ai-toolkit's fallback).
        ids = [getattr(tok, "pad_token_id", None) or 0]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    inner = getattr(text_encoder, "model", text_encoder)  # skip the LM head: dead weight here
    with torch.no_grad():
        out = inner(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = out.hidden_states
    if tap >= len(hidden_states):
        raise ValueError(
            f"te.hidden_state_tap_index={tap} but the encoder returned "
            f"{len(hidden_states)} hidden states (num_hidden_layers + 1)"
        )
    return hidden_states[tap][0]


def pad_rows(rows: list[torch.Tensor], dtype: torch.dtype) -> Any:
    """Per-caption ``[L_i, D]`` rows -> one right-zero-padded batch with an
    ``attention_mask`` marking the real rows."""
    from app.engine.core.text_encoding import TextEncoderOutput

    max_len = max(r.shape[0] for r in rows)
    emb = rows[0].new_zeros((len(rows), max_len, rows[0].shape[-1]))
    mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=rows[0].device)
    for i, r in enumerate(rows):
        emb[i, : r.shape[0]] = r
        mask[i, : r.shape[0]] = 1
    return TextEncoderOutput(embeddings=emb.to(dtype=dtype), attention_mask=mask)


def encode_captions(
    tokenizer: Any,
    text_encoder: Any,
    captions: list[str],
    dtype: torch.dtype,
    *,
    tap: int,
    max_tokens: int,
    device: torch.device,
) -> Any:
    """The layer-tap conditioning, batched by right-padding. Each caption is
    encoded ALONE (no padding inside the encoder, so the result is
    byte-identical to the single-prompt reference path)."""
    rows = [
        encode_caption(tokenizer, text_encoder, cap, tap=tap, max_tokens=max_tokens, device=device)
        for cap in captions
    ]
    return pad_rows(rows, dtype)


def module_weight_bytes(module: Any) -> int:
    """Bytes of a module's parameters (0 for none or a non-module)."""
    if module is None or not hasattr(module, "parameters"):
        return 0
    return sum(p.numel() * p.element_size() for p in module.parameters())


def host_peak_bytes() -> int | None:
    """Peak host RSS of this process (the encoder's loading peak); a missing
    attribute (non-Windows lacks ``peak_wset``) degrades to the current RSS,
    never to a raise."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", None) or info.rss)
    except Exception:  # noqa: BLE001 - a metric, never a failure
        return None


def release_text_encoder(trainer: Any) -> None:
    """Release the trainer's text encoder UNCONDITIONALLY, without a host copy.

    VRAM is freed by dropping EVERY reference (the trainer's component dict,
    the driver's attribute and dict through ``driver.release_text_encoders()``,
    the trainer alias) before ``empty_cache()``. Logs the weight bytes and the
    loading peaks after the cleanup, so ``cuda_allocated_after_bytes`` says
    whether dropping the references actually freed the device. Idempotent
    through ``trainer._te_unloaded``."""
    if trainer._te_unloaded:
        return
    weight_bytes = trainer.driver.text_encoder_weight_bytes()
    trainer.components.pop("text_encoder", None)
    trainer.driver.release_text_encoders()
    if not isinstance(getattr(type(trainer), "text_encoder", None), property):
        trainer.text_encoder = None
    trainer._te_unloaded = True
    gc.collect()
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.empty_cache()
    trainer.logger.info(
        "text_encoder released",
        weight_bytes=weight_bytes,
        host_peak_bytes=host_peak_bytes(),
        cuda_peak_bytes=torch.cuda.max_memory_allocated() if cuda else None,
        cuda_allocated_after_bytes=torch.cuda.memory_allocated() if cuda else None,
    )
