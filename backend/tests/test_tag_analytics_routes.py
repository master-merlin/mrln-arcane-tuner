"""E2E test for the tag-analytics route."""

from unittest.mock import patch

_MODULE = "app.api.dataset.analysis_routes"


@patch(f"{_MODULE}.dataset_manager")
@patch(f"{_MODULE}.asyncio.to_thread")
def test_tag_analytics_success(mock_to_thread, mock_dm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync

    mock_dm.get_dataset.return_value = object()  # truthy (dataset exists)
    mock_dm.get_dataset_pairs.return_value = [
        {"media_file": "a.png", "caption_content": "cat, dog"},
        {"media_file": "b.png", "caption_content": "cat, day, night"},
    ]

    resp = client.get("/api/datasets/myds/tag-analytics?top_n=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_images"] == 2
    freq = {t["tag"]: t["count"] for t in body["top_tags"]}
    assert freq["cat"] == 2
    assert "labels" in body["cooccurrence"] and "matrix" in body["cooccurrence"]
    # default rules include day/night → one contradiction on b.png
    assert any(set([c["a"], c["b"]]) == {"day", "night"} for c in body["contradictions"])


@patch("app.core.llm.caption_refine.caption_style_for", return_value="natural_language")
@patch("app.engine.core.caption_target.resolve_caption_target")
@patch("app.core.captioning.caption_variants.resolve_caption")
@patch(f"{_MODULE}.dataset_manager")
@patch(f"{_MODULE}.asyncio.to_thread")
def test_tag_analytics_model_aware_uses_variants_and_prose(
    mock_to_thread, mock_dm, mock_resolve, mock_target, mock_style, client
):
    """With definition_id, captions come from the variant resolver and the style
    is derived from the model (here natural_language -> prose)."""
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync

    ds = type("D", (), {"path": "/ds"})()
    mock_dm.get_dataset.return_value = ds
    mock_dm.get_dataset_pairs.return_value = [
        {"media_file": "A1.png", "caption_content": "generic one"},
        {"media_file": "B2.png", "caption_content": "generic two"},
    ]
    variants = {
        "A1": "a red sports car on a road",
        "B2": "a blue sports car on a road",
    }
    mock_resolve.side_effect = lambda path, stem, defid, masked: variants[stem]

    resp = client.get("/api/datasets/myds/tag-analytics?definition_id=sdxl_base_1.0")
    assert resp.status_code == 200
    body = resp.json()
    assert body["style"] == "prose"
    freq = {t["tag"]: t["count"] for t in body["top_tags"]}
    assert freq.get("sports car") == 2     # phrase from the VARIANT captions
    assert "generic one" not in freq       # general captions were NOT used
    mock_resolve.assert_any_call("/ds", "A1", "sdxl_base_1.0", False)


@patch("app.core.llm.caption_refine.caption_style_for", return_value="tags")
@patch("app.engine.core.caption_target.resolve_caption_target")
@patch("app.core.captioning.caption_variants.resolve_caption")
@patch(f"{_MODULE}.dataset_manager")
@patch(f"{_MODULE}.asyncio.to_thread")
def test_tag_analytics_model_tags_but_prose_content_falls_back_to_prose(
    mock_to_thread, mock_dm, mock_resolve, mock_target, mock_style, client
):
    """A tag-style model (SDXL) whose variant captions are actually prose must
    NOT be tag-split into one giant tag — it falls back to prose analysis."""
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync

    ds = type("D", (), {"path": "/ds"})()
    mock_dm.get_dataset.return_value = ds
    mock_dm.get_dataset_pairs.return_value = [
        {"media_file": "A1.png", "caption_content": "x"},
        {"media_file": "B2.png", "caption_content": "y"},
    ]
    variants = {
        "A1": "a red sports car parked on a long empty road",
        "B2": "a blue sports car parked on a wide open road",
    }
    mock_resolve.side_effect = lambda path, stem, defid, masked: variants[stem]

    body = client.get("/api/datasets/d/tag-analytics?definition_id=sdxl_base_1.0").json()
    assert body["style"] == "prose"   # reconciled away from the model's "tags"
    freq = {t["tag"]: t["count"] for t in body["top_tags"]}
    assert freq.get("sports car") == 2


@patch(f"{_MODULE}.dataset_manager")
@patch(f"{_MODULE}.asyncio.to_thread")
def test_tag_analytics_unknown_dataset_404(mock_to_thread, mock_dm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_dm.get_dataset.return_value = None

    resp = client.get("/api/datasets/ghost/tag-analytics")
    assert resp.status_code == 404
