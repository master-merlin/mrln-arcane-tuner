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
