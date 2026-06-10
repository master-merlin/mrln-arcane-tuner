"""The mass-caption write seam routes to a per-definition variant when given a definition_id."""

from unittest.mock import MagicMock, patch

from app.core.captioning import caption_batch
from app.core.captioning import caption_variants as cv


def test_write_caption_variant_path(tmp_path):
    ds_path = str(tmp_path)
    fake_ds = MagicMock()
    fake_ds.path = ds_path
    with patch("app.core.dataset_manager.dataset_manager") as dm:
        dm.get_dataset.return_value = fake_ds
        caption_batch._write_caption(
            "myds", "img1.png", "a variant caption", "original", definition_id="flux1-schnell"
        )
    assert cv.read_variant(ds_path, "flux1-schnell", "img1") == "a variant caption"


def test_write_caption_general_unchanged_when_no_definition():
    with patch("app.core.dataset_manager.dataset_manager") as dm:
        caption_batch._write_caption("myds", "img1.png", "general", "original", definition_id=None)
    dm.save_caption.assert_called_once()
    args = dm.save_caption.call_args[0]
    assert args[0] == "myds"
    assert args[1] == "img1.txt"
