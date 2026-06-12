"""Schema contract for the per-dataset caption toggles (Training UI renders
the 'Captions' inline group + depends_on grey-out straight from this schema)."""

from app.engine.models.base import DatasetItem


def test_defaults_are_on():
    ds = DatasetItem(dataset_name="x")
    assert ds.use_captions is True
    assert ds.use_model_aware_captions is True


def test_schema_metadata_drives_the_ui():
    props = DatasetItem.model_json_schema()["properties"]
    uc = props["use_captions"]
    ma = props["use_model_aware_captions"]
    assert uc["inline_group"] == "caption_toggles"
    assert ma["inline_group"] == "caption_toggles"
    # Disabling captions disables (greys out + cascades off) model-aware in the UI.
    assert ma["depends_on"] == "use_captions"


def test_old_configs_without_flags_deserialize_to_defaults():
    ds = DatasetItem.model_validate({"dataset_name": "x", "num_repeats": 2})
    assert ds.use_captions is True
    assert ds.use_model_aware_captions is True
