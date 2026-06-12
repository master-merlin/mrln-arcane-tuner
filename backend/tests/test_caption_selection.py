from app.core.captioning import caption_variants as cv
from app.engine.core.pipeline.caption_selection import select_training_caption


def test_returns_general_when_no_definition():
    item = {"caption": "general cap", "path": "/x/img1.png", "dataset_path": "/x"}
    assert select_training_caption(item, None) == "general cap"


def test_model_aware_off_ignores_variant(tmp_path):
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "img1", "variant cap")
    item = {
        "caption": "general cap", "path": f"{ds}/img1.png", "dataset_path": ds,
        "use_model_aware_captions": False,
    }
    assert select_training_caption(item, "flux1-schnell") == "general cap"


def test_returns_variant_when_present(tmp_path):
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "img1", "variant cap")
    item = {"caption": "general cap", "path": f"{ds}/img1.png", "dataset_path": ds}
    assert select_training_caption(item, "flux1-schnell") == "variant cap"


def test_falls_back_to_general_when_no_variant(tmp_path):
    item = {"caption": "general cap", "path": f"{tmp_path}/img1.png", "dataset_path": str(tmp_path)}
    assert select_training_caption(item, "flux1-schnell") == "general cap"


def test_falls_back_when_dataset_path_missing():
    item = {"caption": "general cap", "path": "/x/img1.png"}  # no dataset_path
    assert select_training_caption(item, "flux1-schnell") == "general cap"


def test_never_raises_on_bad_input():
    assert select_training_caption({}, "flux1-schnell") == ""
    assert select_training_caption({"caption": None}, "d") == ""


def test_use_captions_off_returns_empty_even_with_variant(tmp_path):
    # Captions off: no file caption, no variant lookup. Trigger/prefix are
    # assembled later in _get_batch, so '' here = trigger-word-only training.
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "img1", "variant cap")
    item = {
        "caption": "general cap", "path": f"{ds}/img1.png", "dataset_path": ds,
        "use_captions": False,
    }
    assert select_training_caption(item, "flux1-schnell") == ""


def test_use_captions_off_empty_on_masked_axis_too(tmp_path):
    ds = str(tmp_path)
    item = {
        "caption": "general cap", "masked_caption": "masked cap",
        "path": f"{ds}/img1.png", "dataset_path": ds,
        "use_captions": False,
    }
    assert select_training_caption(item, "flux1-schnell", masked=True) == ""


def test_model_aware_off_masked_keeps_masked_fallback(tmp_path):
    # MA off on the masked axis: skip the variant but keep the
    # masked_caption → caption fallback chain.
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "img1", "masked variant", masked=True)
    item = {
        "caption": "general cap", "masked_caption": "masked cap",
        "path": f"{ds}/img1.png", "dataset_path": ds,
        "use_model_aware_captions": False,
    }
    assert select_training_caption(item, "flux1-schnell", masked=True) == "masked cap"


def test_missing_flags_default_to_captions_on_and_model_aware_on(tmp_path):
    # Old job configs carry neither flag — behavior must equal today's:
    # variant preferred over general.
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "img1", "variant cap")
    item = {"caption": "general cap", "path": f"{ds}/img1.png", "dataset_path": ds}
    assert select_training_caption(item, "flux1-schnell") == "variant cap"
