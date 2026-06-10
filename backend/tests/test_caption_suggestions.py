# backend/tests/test_caption_suggestions.py
import os

from app.core.captioning import caption_suggestions as sg
from app.core.captioning import caption_variants as cv


def test_write_read_list_suggestion(tmp_path):
    ds = str(tmp_path)
    sg.write_suggestion(ds, "flux1-schnell", "img1", "suggested caption")
    assert sg.read_suggestion(ds, "flux1-schnell", "img1") == "suggested caption"
    assert sg.list_suggestion_stems(ds, "flux1-schnell") == ["img1"]


def test_accept_promotes_suggestion_to_variant_and_clears(tmp_path):
    ds = str(tmp_path)
    sg.write_suggestion(ds, "flux1-schnell", "img1", "new variant")
    sg.accept_suggestion(ds, "flux1-schnell", "img1")
    assert cv.read_variant(ds, "flux1-schnell", "img1") == "new variant"
    assert sg.read_suggestion(ds, "flux1-schnell", "img1") is None  # consumed


def test_accept_snapshots_existing_variant_to_bak(tmp_path):
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "img1", "old variant")
    sg.write_suggestion(ds, "flux1-schnell", "img1", "new variant")
    sg.accept_suggestion(ds, "flux1-schnell", "img1")
    assert cv.read_variant(ds, "flux1-schnell", "img1") == "new variant"
    bak = cv.variant_path(ds, "flux1-schnell", "img1") + ".bak"
    assert os.path.exists(bak)
    with open(bak, encoding="utf-8") as f:
        assert f.read() == "old variant"


def test_reject_deletes_suggestion(tmp_path):
    ds = str(tmp_path)
    sg.write_suggestion(ds, "flux1-schnell", "img1", "x")
    sg.reject_suggestion(ds, "flux1-schnell", "img1")
    assert sg.read_suggestion(ds, "flux1-schnell", "img1") is None


def test_masked_suggestion_roundtrip_and_path(tmp_path):
    ds = str(tmp_path)
    sg.write_suggestion(ds, "flux1-schnell", "img1", "masked suggestion", masked=True)
    assert sg.read_suggestion(ds, "flux1-schnell", "img1", masked=True) == "masked suggestion"
    assert sg.list_suggestion_stems(ds, "flux1-schnell", masked=True) == ["img1"]
    p = sg.suggestion_path(ds, "flux1-schnell", "img1", masked=True)
    assert p.endswith(os.path.join("suggestions", "flux1-schnell", "masked", "img1.txt"))


def test_masked_accept_promotes_to_masked_variant_and_snapshots(tmp_path):
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "img1", "old masked variant", masked=True)
    sg.write_suggestion(ds, "flux1-schnell", "img1", "new masked variant", masked=True)
    sg.accept_suggestion(ds, "flux1-schnell", "img1", masked=True)
    assert cv.read_variant(ds, "flux1-schnell", "img1", masked=True) == "new masked variant"
    assert sg.read_suggestion(ds, "flux1-schnell", "img1", masked=True) is None
    bak = cv.variant_path(ds, "flux1-schnell", "img1", masked=True) + ".bak"
    assert os.path.exists(bak)
    with open(bak, encoding="utf-8") as f:
        assert f.read() == "old masked variant"
    # original axis untouched
    assert cv.read_variant(ds, "flux1-schnell", "img1") is None


def test_masked_and_original_suggestion_listings_are_isolated(tmp_path):
    ds = str(tmp_path)
    sg.write_suggestion(ds, "flux1-schnell", "orig", "o")
    sg.write_suggestion(ds, "flux1-schnell", "msk", "m", masked=True)
    assert sg.list_suggestion_stems(ds, "flux1-schnell") == ["orig"]
    assert sg.list_suggestion_stems(ds, "flux1-schnell", masked=True) == ["msk"]


def test_masked_reject_deletes_only_masked(tmp_path):
    ds = str(tmp_path)
    sg.write_suggestion(ds, "flux1-schnell", "img1", "o")
    sg.write_suggestion(ds, "flux1-schnell", "img1", "m", masked=True)
    sg.reject_suggestion(ds, "flux1-schnell", "img1", masked=True)
    assert sg.read_suggestion(ds, "flux1-schnell", "img1", masked=True) is None
    assert sg.read_suggestion(ds, "flux1-schnell", "img1") == "o"
