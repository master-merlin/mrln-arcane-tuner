# backend/tests/test_caption_refine.py
import asyncio

from app.core.llm.caption_refine import (
    REFINE_PRESETS,
    build_refine_system_prompt,
    caption_style_for,
    refine_caption,
)
from app.engine.core.caption_target import CaptionTarget

_SDXL = CaptionTarget("sdxl", "clip", "openai/clip-vit-large-patch14", 77, 75)
_FLUX1 = CaptionTarget("flux1", "t5", "google/t5-v1_1-xxl", 256, 255)


class _FakeClient:
    def __init__(self):
        self.calls = []
    async def chat(self, model, system, user, options=None):
        self.calls.append((model, system, user))
        return f"  refined::{user}  "


def test_presets_exist():
    assert "standardize" in REFINE_PRESETS
    assert "synonym_merge" in REFINE_PRESETS


def test_refine_uses_preset_system_prompt_and_strips():
    c = _FakeClient()
    out = asyncio.run(refine_caption(c, "m", "cat, cat, dog", "standardize"))
    assert out == "refined::cat, cat, dog"  # stripped
    model, system, user = c.calls[0]
    assert system == REFINE_PRESETS["standardize"]
    assert user == "cat, cat, dog"


def test_unknown_preset_raises():
    c = _FakeClient()
    try:
        asyncio.run(refine_caption(c, "m", "x", "nope"))
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_extra_system_is_appended():
    c = _FakeClient()
    asyncio.run(refine_caption(c, "m", "x", "standardize", extra_system="Budget: 512 T5 tokens."))
    _, system, _ = c.calls[0]
    assert system.endswith("Budget: 512 T5 tokens.")
    assert system.startswith(REFINE_PRESETS["standardize"])


# --- model-aware prompt building -------------------------------------------------

def test_caption_style_auto_derives_tags_for_clip_and_prose_for_t5():
    assert caption_style_for(_SDXL) == "tags"
    assert caption_style_for(_FLUX1) == "natural_language"


def test_build_prompt_sdxl_is_tag_style_and_token_aware():
    p = build_refine_system_prompt(_SDXL, "standardize")
    assert "tag" in p.lower()
    assert "75" in p                       # usable token budget surfaced
    assert "sentence" not in p.lower()     # not prose


def test_build_prompt_flux1_is_natural_language_and_token_aware():
    p = build_refine_system_prompt(_FLUX1, "standardize")
    assert "natural-language" in p.lower() or "sentence" in p.lower()
    assert "255" in p                      # usable token budget surfaced


def test_build_prompt_style_override_wins_over_auto():
    # Force tags on a T5 model, and prose on an SDXL model.
    forced_tags = build_refine_system_prompt(_FLUX1, "standardize", style="tags")
    assert "tag" in forced_tags.lower()
    forced_prose = build_refine_system_prompt(_SDXL, "standardize", style="natural_language")
    assert "natural-language" in forced_prose.lower() or "sentence" in forced_prose.lower()


def test_build_prompt_reflects_operation_preset():
    merge = build_refine_system_prompt(_FLUX1, "synonym_merge")
    standardize = build_refine_system_prompt(_FLUX1, "standardize")
    assert merge != standardize
    assert "redundant" in merge.lower() or "duplicate" in merge.lower()


def test_refine_caption_system_prompt_override_bypasses_presets():
    c = _FakeClient()
    out = asyncio.run(refine_caption(c, "m", "a, b", "standardize", system_prompt="CUSTOM SYS"))
    assert out == "refined::a, b"
    _, system, _ = c.calls[0]
    assert system == "CUSTOM SYS"          # the built prompt is used verbatim, not REFINE_PRESETS
