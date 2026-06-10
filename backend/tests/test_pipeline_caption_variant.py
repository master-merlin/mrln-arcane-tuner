"""Focused tests for the per-step trained caption resolving per-definition
variants in PipelineDataMixin._select_variant (the general/non-masked path).

Mirrors the TE pre-cache path (_build_caption_hints): a variant overrides the
general caption, with a defensive fallback to item["caption"] when there is no
definition / no variant / use_general / any error.
"""

from types import SimpleNamespace

from app.core.captioning import caption_variants as cv
from app.engine.core.pipeline.pipeline_data import PipelineDataMixin


def _make_mixin(def_id, use_general):
    """Build a bare mixin instance carrying just definition + config."""
    inst = object.__new__(PipelineDataMixin)
    inst.definition = SimpleNamespace(id=def_id) if def_id is not None else None
    inst.config = {"use_general_captions": use_general}
    return inst


def _item(ds, name="img1"):
    return {
        "path": f"{ds}/{name}.png",
        "caption": "general cap",
        "dataset_path": ds,
        "cache_dir": f"{ds}/cache",
        "has_masked": False,
    }


def test_select_variant_uses_variant_when_present(tmp_path):
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "img1", "variant cap")
    inst = _make_mixin("flux1-schnell", use_general=False)

    path, cap, cache_dir = inst._select_variant(_item(ds))

    assert cap == "variant cap"
    assert path == f"{ds}/img1.png"
    assert cache_dir == f"{ds}/cache"


def test_select_variant_falls_back_to_general_when_no_variant(tmp_path):
    ds = str(tmp_path)
    inst = _make_mixin("flux1-schnell", use_general=False)

    _, cap, _ = inst._select_variant(_item(ds))

    assert cap == "general cap"


def test_select_variant_use_general_override_ignores_variant(tmp_path):
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "img1", "variant cap")
    inst = _make_mixin("flux1-schnell", use_general=True)

    _, cap, _ = inst._select_variant(_item(ds))

    assert cap == "general cap"


def test_select_variant_no_definition_returns_general(tmp_path):
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "img1", "variant cap")
    inst = _make_mixin(None, use_general=False)

    _, cap, _ = inst._select_variant(_item(ds))

    assert cap == "general cap"


def _masked_item(ds, name="img1", masked_caption="masked cap"):
    """Item that always takes the masked branch (original_weight=0 → random>=0 true)."""
    it = {
        "path": f"{ds}/{name}.png",
        "caption": "general cap",
        "dataset_path": ds,
        "cache_dir": f"{ds}/cache",
        "has_masked": True,
        "masked_path": f"{ds}/masked/{name}.jpg",
        "masked_cache_dir": f"{ds}/cache_masked",
        "original_weight": 0.0,
    }
    if masked_caption is not None:
        it["masked_caption"] = masked_caption
        it["has_masked_caption"] = True
    return it


def test_select_variant_uses_masked_variant_when_present(tmp_path):
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "img1", "masked variant", masked=True)
    inst = _make_mixin("flux1-schnell", use_general=False)
    path, cap, cache_dir = inst._select_variant(_masked_item(ds))
    assert cap == "masked variant"
    assert path == f"{ds}/masked/img1.jpg"
    assert cache_dir == f"{ds}/cache_masked"


def test_select_variant_masked_uses_masked_caption_when_no_variant(tmp_path):
    ds = str(tmp_path)
    inst = _make_mixin("flux1-schnell", use_general=False)
    _, cap, _ = inst._select_variant(_masked_item(ds))
    assert cap == "masked cap"


def test_masked_no_caption_falls_back_to_original(tmp_path):
    # Pins the spec §4 behavior change: no masked caption + no masked variant → original caption.
    ds = str(tmp_path)
    inst = _make_mixin("flux1-schnell", use_general=False)
    _, cap, _ = inst._select_variant(_masked_item(ds, masked_caption=None))
    assert cap == "general cap"


def test_select_variant_masked_use_general_ignores_variant(tmp_path):
    ds = str(tmp_path)
    cv.write_variant(ds, "flux1-schnell", "img1", "masked variant", masked=True)
    inst = _make_mixin("flux1-schnell", use_general=True)
    _, cap, _ = inst._select_variant(_masked_item(ds))
    assert cap == "masked cap"
