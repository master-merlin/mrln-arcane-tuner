# backend/app/core/llm/caption_refine.py
"""Preset-driven caption refinement over an OpenAI-compatible chat client."""

from __future__ import annotations

from typing import Any

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


async def refine_caption(
    client: Any,
    model: str,
    text: str,
    preset: str,
    extra_system: str = "",
) -> str:
    """Refine one caption with a preset system prompt. Raises KeyError on unknown preset."""
    system = REFINE_PRESETS[preset]
    if extra_system:
        system = f"{system}\n{extra_system}"
    result = await client.chat(model, system, text)
    return result.strip()
