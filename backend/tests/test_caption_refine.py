# backend/tests/test_caption_refine.py
import asyncio

from app.core.llm.caption_refine import REFINE_PRESETS, refine_caption


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
