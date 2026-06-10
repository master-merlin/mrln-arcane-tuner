"""Unit tests for the masked axis of select_training_caption."""

from app.core.captioning import caption_variants as cv
from app.engine.core.pipeline.caption_selection import select_training_caption


def _item(ds, masked_caption="masked cap"):
    it = {
        "path": f"{ds}/img1.png",
        "caption": "general cap",
        "dataset_path": ds,
    }
    if masked_caption is not None:
        it["masked_caption"] = masked_caption
    return it


def test_masked_variant_overrides_everything(tmp_path):
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "img1", "masked variant", masked=True)
    assert select_training_caption(_item(ds), "flux1-schnell", False, masked=True) == "masked variant"


def test_masked_uses_masked_caption_when_no_variant(tmp_path):
    ds = str(tmp_path)
    assert select_training_caption(_item(ds), "flux1-schnell", False, masked=True) == "masked cap"


def test_masked_falls_back_to_original_when_no_masked_caption(tmp_path):
    ds = str(tmp_path)
    assert select_training_caption(_item(ds, masked_caption=None), "flux1-schnell", False, masked=True) == "general cap"


def test_masked_use_general_ignores_variant(tmp_path):
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "img1", "masked variant", masked=True)
    assert select_training_caption(_item(ds), "flux1-schnell", True, masked=True) == "masked cap"


def test_original_axis_unchanged(tmp_path):
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "img1", "orig variant")
    assert select_training_caption(_item(ds), "flux1-schnell", False) == "orig variant"
    assert select_training_caption(_item(ds), "flux1-schnell", True) == "general cap"
