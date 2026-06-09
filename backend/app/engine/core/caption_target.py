# backend/app/engine/core/caption_target.py
"""Resolve a model definition to its caption tokenizer + usable token budget.

The tokenizer is a *family* property (all flux1 definitions share T5; all SDXL
definitions share CLIP), so we map by family and read the max length from the
definition's ``architecture_params``. Families we do not tokenize precisely
(e.g. flux2's Qwen text encoder) fall back to a length heuristic — the count is
approximate but never errors and needs no model download.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.engine.models.registry import registry


@dataclass(frozen=True)
class CaptionTarget:
    family: str
    tokenizer_kind: str            # "clip" | "t5" | "heuristic"
    tokenizer_id: str | None       # HF repo id, or None for heuristic
    raw_max_length: int            # the encoder's max sequence length
    usable_limit: int              # raw minus reserved special tokens


def resolve_caption_target(definition_id: str) -> CaptionTarget:
    """Map a definition id to a :class:`CaptionTarget`.

    Raises ``ValueError`` if the definition id is unknown.
    """
    defn = registry.get_definition(definition_id)
    if defn is None:
        raise ValueError(f"unknown definition id: {definition_id!r}")

    arch = defn.architecture_params or {}
    family = defn.family

    if family == "sdxl":
        raw = int(arch.get("te.max_position_embeddings", 77))
        return CaptionTarget(family, "clip", "openai/clip-vit-large-patch14", raw, max(raw - 2, 1))

    if family == "flux1":
        raw = int(arch.get("te.t5_max_length", 256))
        return CaptionTarget(family, "t5", "google/t5-v1_1-xxl", raw, max(raw - 1, 1))

    # Fallback: flux2 (Qwen TE), microsoft_lens, and any future family.
    raw = int(
        arch.get("te.max_length")
        or arch.get("te.t5_max_length")
        or arch.get("te.max_position_embeddings")
        or 77
    )
    return CaptionTarget(family, "heuristic", None, raw, raw)
