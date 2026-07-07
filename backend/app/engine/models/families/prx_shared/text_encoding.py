"""Shared PRX prompt encoding — replicates ``PRXPipeline.encode_prompt``.

Semantics (pipeline-verbatim, ``pipeline_prx.py``):
1. DeepFloyd-style text cleaning via the pipeline's own
   ``TextPreprocessor.clean_text`` (imported, not duplicated) — the default
   path; ``basic_clean`` (ftfy + HTML unescape) when cleaning is skipped.
2. Tokenize with ``padding="max_length"``,
   ``max_length = tokenizer.model_max_length`` (256 for the sft checkpoint),
   ``truncation=True``.
3. Embeddings = ``text_encoder(input_ids, attention_mask,
   output_hidden_states=True)["last_hidden_state"]`` — no layer tapping,
   no zero-masking, no slicing.
4. The attention mask is returned as a BOOLEAN tensor (the transformer's
   ``attention_mask`` input expects bool semantics).

Family-agnostic: both the latent ``prx`` family and the future pixel-space
sibling share the identical T5Gemma prompt contract.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

_preprocessor = None


def _get_preprocessor():
    """Lazily build (and cache) the diffusers PRX ``TextPreprocessor``."""
    global _preprocessor
    if _preprocessor is None:
        from diffusers.pipelines.prx.pipeline_prx import (  # noqa: PLC0415
            TextPreprocessor,
        )

        _preprocessor = TextPreprocessor()
    return _preprocessor


def encode_prx_text(
    tokenizer: Any,
    text_encoder: nn.Module,
    captions: list[str],
    device: torch.device,
    max_length: int | None = None,
    clean_text: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode captions exactly as ``PRXPipeline._encode_prompt_standard``.

    Args:
        tokenizer: The checkpoint tokenizer (GemmaTokenizerFast for the sft
            checkpoint; resolved via AutoTokenizer).
        text_encoder: ``T5GemmaEncoder`` instance.
        captions: Raw caption strings.
        device: Target device for input ids / outputs.
        max_length: Tokenizer max length; defaults to
            ``tokenizer.model_max_length`` (the pipeline's default).
        clean_text: Apply the full DeepFloyd cleaning pipeline (pipeline
            default); ``False`` uses the light ``basic_clean`` path.

    Returns:
        ``(embeddings [B, L, hidden], attention_mask bool [B, L])``.
    """
    pre = _get_preprocessor()
    clean_fn = pre.clean_text if clean_text else pre.basic_clean
    cleaned = [clean_fn(text) for text in captions]

    max_len = max_length or tokenizer.model_max_length
    tokens = tokenizer(
        cleaned,
        padding="max_length",
        max_length=max_len,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].bool().to(device)

    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    embeddings = outputs["last_hidden_state"]

    return embeddings, attention_mask
