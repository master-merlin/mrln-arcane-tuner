# backend/tests/core/llm/test_caption_refine_format.py
from app.core.llm.caption_refine import build_refine_system_prompt
from app.engine.core.caption_target import CaptionTarget


def _ideogram_target():
    return CaptionTarget("ideogram4", "heuristic", None, 2048, 2048)


def _flux_target():
    return CaptionTarget("flux1", "t5", "google/t5-v1_1-xxl", 256, 255)


def test_structured_family_gets_schema_preserving_prompt():
    p = build_refine_system_prompt(_ideogram_target(), "standardize", "auto")
    assert "JSON" in p
    assert "schema" in p.lower()


def test_flat_family_keeps_legacy_prompt():
    p = build_refine_system_prompt(_flux_target(), "standardize", "auto")
    assert "JSON" not in p
    assert "natural-language" in p.lower()
