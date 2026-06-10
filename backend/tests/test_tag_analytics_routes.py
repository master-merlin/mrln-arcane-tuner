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


@patch(f"{_MODULE}.dataset_manager")
@patch(f"{_MODULE}.asyncio.to_thread")
def test_tag_analytics_unknown_dataset_404(mock_to_thread, mock_dm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_dm.get_dataset.return_value = None

    resp = client.get("/api/datasets/ghost/tag-analytics")
    assert resp.status_code == 404
