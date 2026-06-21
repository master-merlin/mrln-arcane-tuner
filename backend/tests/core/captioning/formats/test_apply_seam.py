"""Unit tests for apply_generation_seam (formats/__init__.py).

Four scenarios:
1. Structured format + non-api model -> system_prompt + overrides set, no response_format.
2. Structured format + api-* model   -> response_format also set.
3. Plain format                      -> params completely unchanged.
4. A generic template system_prompt IS overridden by the structured contract.
"""

from __future__ import annotations

from app.core.captioning.formats import (
    apply_generation_seam,
    Ideogram4Format,
    PlainFormat,
)


def test_structured_non_api_sets_prompt_and_overrides():
    fmt = Ideogram4Format()
    params: dict = {"caption_instructions": "Be concise."}
    apply_generation_seam(params, fmt, "qwen3-vl-8B-Instruct")

    # system_prompt is built from build_generation_prompt with caption_instructions
    assert "system_prompt" in params
    assert "Be concise." in params["system_prompt"]
    # generation_overrides merged: a ceiling only, NO forced min-token floor
    # (a floor makes a model that finished the JSON spiral into trailing garbage)
    assert params.get("max_tokens") == 4096
    assert "min_new_tokens" not in params
    # non-api model -> no response_format
    assert "response_format" not in params


def test_structured_api_model_sets_response_format():
    fmt = Ideogram4Format()
    params: dict = {}
    apply_generation_seam(params, fmt, "api-openai-gpt4o")

    assert "system_prompt" in params
    assert "min_new_tokens" not in params
    assert params["response_format"] == {"type": "json_object"}


def test_plain_format_params_unchanged():
    fmt = PlainFormat()
    params_before = {"foo": "bar", "caption_instructions": "x"}
    params = params_before.copy()
    apply_generation_seam(params, fmt, "api-openai-gpt4o")
    assert params == params_before


def test_contract_leads_template_and_extra_folded_as_additional_guidance():
    """The format's JSON contract is the authoritative system prompt and must
    LEAD. The user's template system prompt AND the Additional-instructions
    field are both preserved, appended after the contract as additional
    guidance. A stray user_prompt is dropped so it cannot fight the contract."""
    fmt = Ideogram4Format()
    params: dict = {
        "system_prompt": "No brand names.",
        "user_prompt": "Describe this image in detail.",
        "caption_instructions": "Focus on the car.",
    }
    apply_generation_seam(params, fmt, "qwen3-vl-8B-Instruct")
    sp = params["system_prompt"]
    # Contract present and leading (before the appended guidance).
    assert "compositional_deconstruction" in sp
    assert sp.index("compositional_deconstruction") < sp.index("No brand names.")
    # Both the template and the explicit field survive as additional guidance.
    assert "No brand names." in sp
    assert "Focus on the car." in sp
    assert "ADDITIONAL INSTRUCTIONS" in sp
    assert "user_prompt" not in params
