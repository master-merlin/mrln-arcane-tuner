from app.engine.core.pipeline.caption_selection import select_training_caption
from app.core.captioning import caption_variants as cv


def test_returns_general_when_no_definition():
    item = {"caption": "general cap", "path": "/x/img1.png", "dataset_path": "/x"}
    assert select_training_caption(item, None, use_general=False) == "general cap"


def test_returns_general_when_use_general_override(tmp_path):
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "img1", "variant cap")
    item = {"caption": "general cap", "path": f"{ds}/img1.png", "dataset_path": ds}
    assert select_training_caption(item, "flux1-schnell", use_general=True) == "general cap"


def test_returns_variant_when_present(tmp_path):
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "img1", "variant cap")
    item = {"caption": "general cap", "path": f"{ds}/img1.png", "dataset_path": ds}
    assert select_training_caption(item, "flux1-schnell", use_general=False) == "variant cap"


def test_falls_back_to_general_when_no_variant(tmp_path):
    item = {"caption": "general cap", "path": f"{tmp_path}/img1.png", "dataset_path": str(tmp_path)}
    assert select_training_caption(item, "flux1-schnell", use_general=False) == "general cap"


def test_falls_back_when_dataset_path_missing():
    item = {"caption": "general cap", "path": "/x/img1.png"}  # no dataset_path
    assert select_training_caption(item, "flux1-schnell", use_general=False) == "general cap"


def test_never_raises_on_bad_input():
    assert select_training_caption({}, "flux1-schnell", use_general=False) == ""
    assert select_training_caption({"caption": None}, "d", use_general=False) == ""
