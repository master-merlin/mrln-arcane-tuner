# backend/tests/test_caption_variants.py
"""Unit tests for caption variant storage + resolver (tmp dirs, real file I/O)."""

from app.core.captioning import caption_variants as cv


def test_write_and_read_variant(tmp_path):
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "img1", "a flux caption")
    assert cv.read_variant(ds, "flux1-schnell", "img1") == "a flux caption"
    assert cv.has_variant(ds, "flux1-schnell", "img1") is True


def test_read_missing_variant_is_none(tmp_path):
    assert cv.read_variant(str(tmp_path), "flux1-schnell", "nope") is None
    assert cv.has_variant(str(tmp_path), "flux1-schnell", "nope") is False


def test_resolve_prefers_variant_then_general(tmp_path):
    ds = str(tmp_path)
    (tmp_path / "img1.txt").write_text("general caption", encoding="utf-8")
    assert cv.resolve_caption(ds, "img1", "flux1-schnell") == "general caption"
    cv.write_variant(ds, "flux1-schnell", "img1", "variant caption")
    assert cv.resolve_caption(ds, "img1", "flux1-schnell") == "variant caption"
    assert cv.resolve_caption(ds, "img1", None) == "general caption"


def test_resolve_missing_everything_is_empty(tmp_path):
    assert cv.resolve_caption(str(tmp_path), "ghost", "flux1-schnell") == ""


def test_list_variant_definition_ids(tmp_path):
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "a", "x")
    cv.write_variant(ds, "sdxl_base_1.0", "a", "y")
    assert set(cv.list_variant_definition_ids(ds)) == {"flux1-schnell", "sdxl_base_1.0"}
