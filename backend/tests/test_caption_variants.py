# backend/tests/test_caption_variants.py
"""Unit tests for caption variant storage + resolver (tmp dirs, real file I/O)."""

import os

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


def test_masked_variant_write_read_has_roundtrip(tmp_path):
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "img1", "masked variant", masked=True)
    assert cv.read_variant(ds, "flux1-schnell", "img1", masked=True) == "masked variant"
    assert cv.has_variant(ds, "flux1-schnell", "img1", masked=True) is True
    # masked and original axes are independent
    assert cv.read_variant(ds, "flux1-schnell", "img1") is None
    assert cv.has_variant(ds, "flux1-schnell", "img1") is False


def test_masked_variant_path_nests_masked_segment(tmp_path):
    ds = str(tmp_path)
    p = cv.variant_path(ds, "flux1-schnell", "img1", masked=True)
    assert p.endswith(os.path.join("captions", "flux1-schnell", "masked", "img1.txt"))


def test_resolve_masked_precedence(tmp_path):
    ds = str(tmp_path)
    # rung 4: nothing → ''
    assert cv.resolve_caption(ds, "img1", "flux1-schnell", masked=True) == ""
    # rung 3: original general caption is the fallback when no masked caption
    (tmp_path / "img1.txt").write_text("original caption", encoding="utf-8")
    assert cv.resolve_caption(ds, "img1", "flux1-schnell", masked=True) == "original caption"
    # rung 2: dedicated masked caption wins over original
    masked_dir = tmp_path / "masked"
    masked_dir.mkdir()
    (masked_dir / "img1.txt").write_text("masked caption", encoding="utf-8")
    assert cv.resolve_caption(ds, "img1", "flux1-schnell", masked=True) == "masked caption"
    # rung 1: masked variant wins over everything
    cv.write_variant(ds, "flux1-schnell", "img1", "masked variant", masked=True)
    assert cv.resolve_caption(ds, "img1", "flux1-schnell", masked=True) == "masked variant"


def test_resolve_original_axis_unaffected_by_masked_files(tmp_path):
    ds = str(tmp_path)
    (tmp_path / "img1.txt").write_text("original caption", encoding="utf-8")
    masked_dir = tmp_path / "masked"
    masked_dir.mkdir()
    (masked_dir / "img1.txt").write_text("masked caption", encoding="utf-8")
    # masked=False must never read masked/ — still resolves to the general caption
    assert cv.resolve_caption(ds, "img1", "flux1-schnell") == "original caption"
