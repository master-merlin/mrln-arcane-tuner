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
    mock_tm._loop = None  # no event loop in test → suggestion.written emission is a no-op
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
    # The worker no longer decides "done" by reaching the end; it hands its
    # tally to finish_batch, which derives the terminal state (LANE-52).
    mock_tm.finish_batch.assert_called_once_with("t1", ok=1, failed=0, error=None)


@patch.object(crb, "task_manager")
@patch.object(crb, "dataset_manager")
@patch.object(crb.caption_refine, "refine_caption", new_callable=AsyncMock)
def test_refine_batch_masked_writes_masked_suggestion(mock_refine, mock_dm, mock_tm, tmp_path):
    mock_refine.return_value = "refined masked cap"
    ds = MagicMock()
    ds.path = str(tmp_path)
    mock_dm.get_dataset.return_value = ds
    mock_tm.is_cancelled.return_value = False
    mock_tm._loop = None  # no event loop in test → suggestion.written emission is a no-op
    # masked source caption present
    masked_dir = tmp_path / "masked"
    masked_dir.mkdir()
    (masked_dir / "img1.txt").write_text("masked source caption", encoding="utf-8")

    crb.run_caption_refine_batch(
        "t1", dataset_name="ds", image_rel_paths=["img1.png"],
        definition_id="flux1-schnell", preset="standardize",
        model="qwen2.5:7b-instruct", base_url="http://test", target="masked",
    )

    # suggestion written on the MASKED axis
    assert sg.read_suggestion(str(tmp_path), "flux1-schnell", "img1", masked=True) == "refined masked cap"
    assert sg.read_suggestion(str(tmp_path), "flux1-schnell", "img1") is None
    # source passed to refine was the masked caption
    assert mock_refine.await_args.args[2] == "masked source caption"
    # The worker no longer decides "done" by reaching the end; it hands its
    # tally to finish_batch, which derives the terminal state (LANE-52).
    mock_tm.finish_batch.assert_called_once_with("t1", ok=1, failed=0, error=None)


@patch.object(crb, "_emit_suggestion_written")
@patch.object(crb, "task_manager")
@patch.object(crb, "dataset_manager")
@patch.object(crb.caption_refine, "refine_caption", new_callable=AsyncMock)
def test_refine_batch_emits_suggestion_written(mock_refine, mock_dm, mock_tm, mock_emit, tmp_path):
    """Each written suggestion broadcasts a suggestion.written event so the
    frontend review updates live (no re-navigation needed)."""
    mock_refine.return_value = "refined cap"
    ds = MagicMock()
    ds.path = str(tmp_path)
    mock_dm.get_dataset.return_value = ds
    mock_tm.is_cancelled.return_value = False
    (tmp_path / "img1.txt").write_text("general caption", encoding="utf-8")

    crb.run_caption_refine_batch(
        "t1", dataset_name="ds", image_rel_paths=["img1.png"],
        definition_id="flux1-schnell", preset="standardize",
        model="qwen2.5:7b-instruct", base_url="http://test",
    )

    mock_emit.assert_called_once()
    kw = mock_emit.call_args.kwargs
    assert kw["dataset_name"] == "ds"
    assert kw["stem"] == "img1"
    assert kw["definition_id"] == "flux1-schnell"
    assert kw["target"] == "original"
    assert kw["suggestion"] == "refined cap"


@patch.object(crb, "resolve_caption_target")
@patch.object(crb, "task_manager")
@patch.object(crb, "dataset_manager")
@patch.object(crb.caption_refine, "refine_caption", new_callable=AsyncMock)
def test_refine_batch_passes_model_aware_system_prompt(mock_refine, mock_dm, mock_tm, mock_target, tmp_path):
    """The batch resolves the definition's caption target and refines with a
    model-aware (style + token-budget) system prompt, not a fixed preset."""
    from app.engine.core.caption_target import CaptionTarget

    mock_target.return_value = CaptionTarget("sdxl", "clip", "x", 77, 75)
    mock_refine.return_value = "refined"
    ds = MagicMock()
    ds.path = str(tmp_path)
    mock_dm.get_dataset.return_value = ds
    mock_tm.is_cancelled.return_value = False
    mock_tm._loop = None
    (tmp_path / "img1.txt").write_text("a, b", encoding="utf-8")

    crb.run_caption_refine_batch(
        "t1", dataset_name="ds", image_rel_paths=["img1.png"],
        definition_id="sdxl_base_1.0", preset="standardize",
        model="m", base_url="http://test", style="auto",
    )

    sp = mock_refine.await_args.kwargs["system_prompt"]
    assert sp is not None
    assert "tag" in sp.lower()   # SDXL → tagging style
    assert "75" in sp            # usable token budget baked in


@patch.object(crb, "_emit_variant_written")
@patch.object(crb, "task_manager")
@patch.object(crb, "dataset_manager")
@patch.object(crb.caption_refine, "refine_caption", new_callable=AsyncMock)
def test_refine_batch_auto_accept_promotes_variant(mock_refine, mock_dm, mock_tm, mock_emit, tmp_path):
    """With auto_accept, each refined caption is promoted straight to the live
    variant (no pending suggestion left to review) and a variant.written event
    fires instead of suggestion.written."""
    from app.core.captioning import caption_variants as cv

    mock_refine.return_value = "refined cap"
    ds = MagicMock()
    ds.path = str(tmp_path)
    mock_dm.get_dataset.return_value = ds
    mock_tm.is_cancelled.return_value = False
    mock_tm._loop = None
    (tmp_path / "img1.txt").write_text("general caption", encoding="utf-8")

    crb.run_caption_refine_batch(
        "t1", dataset_name="ds", image_rel_paths=["img1.png"],
        definition_id="flux1-schnell", preset="standardize",
        model="m", base_url="http://test", auto_accept=True,
    )

    # promoted to the live variant, no pending suggestion remains
    assert cv.read_variant(str(tmp_path), "flux1-schnell", "img1") == "refined cap"
    assert sg.read_suggestion(str(tmp_path), "flux1-schnell", "img1") is None
    mock_emit.assert_called_once()
    assert mock_emit.call_args.kwargs["definition_id"] == "flux1-schnell"
    assert mock_emit.call_args.kwargs["stem"] == "img1"
    # The worker no longer decides "done" by reaching the end; it hands its
    # tally to finish_batch, which derives the terminal state (LANE-52).
    mock_tm.finish_batch.assert_called_once_with("t1", ok=1, failed=0, error=None)


@patch.object(crb, "resolve_caption_target", side_effect=ValueError("unknown"))
@patch.object(crb, "task_manager")
@patch.object(crb, "dataset_manager")
@patch.object(crb.caption_refine, "refine_caption", new_callable=AsyncMock)
def test_refine_batch_falls_back_when_target_unresolvable(mock_refine, mock_dm, mock_tm, mock_target, tmp_path):
    """If the definition can't be resolved to a caption target, refine still
    runs with the legacy preset (system_prompt=None) rather than crashing."""
    mock_refine.return_value = "refined"
    ds = MagicMock()
    ds.path = str(tmp_path)
    mock_dm.get_dataset.return_value = ds
    mock_tm.is_cancelled.return_value = False
    mock_tm._loop = None
    (tmp_path / "img1.txt").write_text("a, b", encoding="utf-8")

    crb.run_caption_refine_batch(
        "t1", dataset_name="ds", image_rel_paths=["img1.png"],
        definition_id="ghost-def", preset="standardize", model="m", base_url="http://test",
    )

    assert mock_refine.await_args.kwargs["system_prompt"] is None
    # The worker no longer decides "done" by reaching the end; it hands its
    # tally to finish_batch, which derives the terminal state (LANE-52).
    mock_tm.finish_batch.assert_called_once_with("t1", ok=1, failed=0, error=None)


@patch.object(crb, "task_manager")
@patch.object(crb, "dataset_manager")
@patch.object(crb.caption_refine, "refine_caption", new_callable=AsyncMock)
def test_refine_batch_unknown_dataset_fails(mock_refine, mock_dm, mock_tm, tmp_path):
    mock_dm.get_dataset.return_value = None
    crb.run_caption_refine_batch("t2", dataset_name="ghost", image_rel_paths=["a.png"],
                                 definition_id="d", preset="standardize", model="m", base_url="http://test")
    mock_tm.fail.assert_called_once()
    mock_tm.complete.assert_not_called()
    mock_tm.finish_batch.assert_not_called()
