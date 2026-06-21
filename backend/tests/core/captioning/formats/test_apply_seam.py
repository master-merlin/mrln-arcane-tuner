"""Unit tests for apply_generation_seam (formats/__init__.py).

Four scenarios:
1. Structured format + non-api model -> system_prompt + overrides set, no response_format.
2. Structured format + api-* model   -> response_format also set.
3. Plain format                      -> params completely unchanged.
4. Existing system_prompt not overwritten.
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
    # generation_overrides merged
    assert params.get("min_new_tokens") == 3072
    assert params.get("max_tokens") == 4096
    # non-api model -> no response_format
    assert "response_format" not in params


def test_structured_api_model_sets_response_format():
    fmt = Ideogram4Format()
    params: dict = {}
    apply_generation_seam(params, fmt, "api-openai-gpt4o")

    assert "system_prompt" in params
    assert params.get("min_new_tokens") == 3072
    assert params["response_format"] == {"type": "json_object"}


def test_plain_format_params_unchanged():
    fmt = PlainFormat()
    params_before = {"foo": "bar", "caption_instructions": "x"}
    params = params_before.copy()
    apply_generation_seam(params, fmt, "api-openai-gpt4o")
    assert params == params_before


def test_existing_system_prompt_not_overwritten():
    fmt = Ideogram4Format()
    original_prompt = "my custom prompt"
    params: dict = {"system_prompt": original_prompt}
    apply_generation_seam(params, fmt, "qwen3-vl-8B-Instruct")
    assert params["system_prompt"] == original_prompt
