# backend/app/core/llm/caption_refine.py
"""Caption refinement over an OpenAI-compatible chat client.

Two layers:
  * ``REFINE_PRESETS`` — the legacy, model-agnostic preset prompts (kept for the
    plain ``refine_caption(..., preset=...)`` path and its callers/tests).
  * ``build_refine_system_prompt`` — a MODEL-AWARE prompt built from the target
    model's :class:`CaptionTarget`. It picks caption *style* (booru tagging vs
    natural-language captioning) and bakes in the model's usable token budget,
    so each variant is refined for what that model actually understands instead
    of a one-size-fits-all tag rewrite. The ``preset`` becomes a secondary
    *operation* (standardize / synonym_merge) phrased for the chosen style.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.engine.core.caption_target import CaptionTarget

REFINE_PRESETS: dict[str, str] = {
    "standardize": (
        "You are a dataset caption editor. Rewrite the caption using a consistent, "
        "comma-separated tag style with a single space after each comma. Preserve every "
        "concept and all meaning; do NOT add, remove, or invent concepts. "
        "Return ONLY the rewritten caption with no preamble."
    ),
    "synonym_merge": (
        "You are a dataset caption editor. Collapse synonymous tags to a single canonical "
        "term and remove duplicate tags. Preserve all distinct concepts and meaning; do NOT "
        "add or invent concepts. Return ONLY the rewritten caption with no preamble."
    ),
}

# Caption styles. "tags" = booru-style comma list (CLIP-era models like SDXL);
# "natural_language" = descriptive prose (T5 / large-context models like Flux).
STYLE_TAGS = "tags"
STYLE_NATURAL = "natural_language"

# Per-style phrasing of each operation preset, appended to the style guidance.
_OPERATION_INTENT: dict[str, dict[str, str]] = {
    "standardize": {
        STYLE_TAGS: "Normalise to a consistent comma-separated tag style (one space after each comma).",
        STYLE_NATURAL: "Polish into clean, well-formed prose with consistent phrasing and punctuation.",
    },
    "synonym_merge": {
        STYLE_TAGS: "Collapse synonymous tags to one canonical term and drop duplicate tags.",
        STYLE_NATURAL: "Remove redundant or repeated descriptions so each idea appears only once.",
    },
}

_PRESERVE = (
    "Preserve every concept and all meaning; do NOT add, remove, or invent concepts. "
)
_RETURN_ONLY = "Return ONLY the rewritten caption with no preamble."


def caption_style_for(target: CaptionTarget) -> str:
    """Auto-derive the caption style from the model's text encoder.

    CLIP text encoders (SDXL) have tiny budgets and were trained on booru-style
    tag soups → tagging. T5 / large-context encoders (Flux, etc.) understand
    natural language → descriptive captioning.
    """
    if target.tokenizer_kind == "clip" or target.family == "sdxl":
        return STYLE_TAGS
    return STYLE_NATURAL


def build_refine_system_prompt(
    target: CaptionTarget, preset: str, style: str = "auto"
) -> str:
    """Build a model-aware refine system prompt.

    ``style`` is ``"auto"`` (derive from the model), ``"tags"``, or
    ``"natural_language"`` (explicit user override). The model's usable token
    budget is baked in so the LLM targets the right length for that encoder.

    When the target's family has a structured :class:`CaptionFormat`, delegates
    to ``fmt.build_refine_prompt`` so structured captions (e.g. Ideogram 4 JSON)
    get a schema-preserving instruction instead of a plain prose/tag prompt.
    """
    from app.core.captioning.formats import (
        get_caption_format,
    )  # lazy – avoids circular import

    fmt = get_caption_format(target.family)
    if fmt.is_structured:
        structured = fmt.build_refine_prompt(target, {})
        if structured:
            return structured

    effective = (
        style if style in (STYLE_TAGS, STYLE_NATURAL) else caption_style_for(target)
    )
    budget = target.usable_limit
    operation = _OPERATION_INTENT.get(preset, {}).get(effective, "")

    if effective == STYLE_TAGS:
        base = (
            "You are a dataset caption editor for a text-to-image model whose "
            f"{target.tokenizer_kind.upper()} text encoder has a hard limit of about {budget} tokens. "
            "Rewrite the caption as a concise, comma-separated list of booru-style tags ordered "
            f"from most to least important, fitting comfortably within {budget} tokens. "
        )
    else:
        base = (
            "You are a dataset caption editor for a text-to-image model with strong natural-language "
            f"understanding (token budget about {budget} tokens). Rewrite the caption as a fluent, "
            "descriptive natural-language caption written in complete sentences (not tags), making full "
            f"use of the available {budget}-token budget without padding or repetition. "
        )

    op = f"{operation} " if operation else ""
    return f"{base}{op}{_PRESERVE}{_RETURN_ONLY}"


async def refine_caption(
    client: Any,
    model: str,
    text: str,
    preset: str,
    extra_system: str = "",
    system_prompt: str | None = None,
) -> str:
    """Refine one caption.

    When ``system_prompt`` is provided (a model-aware prompt from
    :func:`build_refine_system_prompt`) it is used verbatim and ``preset`` is
    ignored. Otherwise the legacy ``REFINE_PRESETS[preset]`` is used (raises
    ``KeyError`` on an unknown preset). ``extra_system`` is appended either way.
    """
    system = system_prompt if system_prompt is not None else REFINE_PRESETS[preset]
    if extra_system:
        system = f"{system}\n{extra_system}"
    result = await client.chat(model, system, text)
    return result.strip()
