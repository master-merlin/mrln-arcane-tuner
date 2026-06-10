# backend/tests/test_caption_refine_batch.py
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.captioning import caption_refine_batch as crb
from app.core.captioning import caption_suggestions as sg


@patch.object(crb, "task_manager")
@patch.object(crb, "dataset_manager")
@patch.object(crb.caption_refine, "refine_caption", new_callable=AsyncMock)
def test_refine_batch_writes_suggestions_and_completes(mock_refine, mock_dm, mock_tm, tmp_path):
    mock_refine.return_value = "refined cap"
    ds = MagicMock()
    ds.path = str(tmp_path)
    mock_dm.get_dataset.return_value = ds
    mock_tm.is_cancelled.return_value = False
    (tmp_path / "img1.txt").write_text("general caption", encoding="utf-8")

    crb.run_caption_refine_batch(
        "t1",
        dataset_name="ds",
        image_rel_paths=["img1.png"],
        definition_id="flux1-schnell",
        preset="standardize",
        model="qwen2.5:7b-instruct",
        base_url="http://test",
    )

    assert sg.read_suggestion(str(tmp_path), "flux1-schnell", "img1") == "refined cap"
    # source caption passed to refine was the general caption
    assert mock_refine.await_args.args[2] == "general caption"
    mock_tm.complete.assert_called_once_with("t1")


@patch.object(crb, "task_manager")
@patch.object(crb, "dataset_manager")
@patch.object(crb.caption_refine, "refine_caption", new_callable=AsyncMock)
def test_refine_batch_unknown_dataset_fails(mock_refine, mock_dm, mock_tm, tmp_path):
    mock_dm.get_dataset.return_value = None
    crb.run_caption_refine_batch("t2", dataset_name="ghost", image_rel_paths=["a.png"],
                                 definition_id="d", preset="standardize", model="m", base_url="http://test")
    mock_tm.fail.assert_called_once()
    mock_tm.complete.assert_not_called()
