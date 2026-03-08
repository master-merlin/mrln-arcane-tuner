"""
E2E tests for api/dataset/adjustment_routes.py — adjust, batch, color match, histogram, cube export.
"""

import json
from unittest.mock import patch


_ADJ_MODULE = "app.api.dataset.adjustment_routes"


# ── Single Adjustment ────────────────────────────────────────────────────


@patch(f"{_ADJ_MODULE}.dataset_manager")
@patch(f"{_ADJ_MODULE}.asyncio.to_thread")
def test_adjust_media_success(mock_to_thread, mock_dm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync

    response = client.post("/api/datasets/myds/adjust", json={
        "path": "img.png",
        "hue_shift": 0.0,
        "saturation": 1.0,
        "contrast": 1.0,
    })
    assert response.status_code == 200
    assert response.json()["status"] == "adjusted"


@patch(f"{_ADJ_MODULE}.dataset_manager")
@patch(f"{_ADJ_MODULE}.asyncio.to_thread")
def test_adjust_media_not_found(mock_to_thread, mock_dm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_dm.apply_adjustments.side_effect = FileNotFoundError("not found")

    response = client.post("/api/datasets/ghost/adjust", json={
        "path": "img.png",
        "hue_shift": 0.0,
        "saturation": 1.0,
        "contrast": 1.0,
    })
    assert response.status_code == 404


@patch(f"{_ADJ_MODULE}.dataset_manager")
@patch(f"{_ADJ_MODULE}.asyncio.to_thread")
def test_adjust_media_bad_request(mock_to_thread, mock_dm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_dm.apply_adjustments.side_effect = ValueError("bad params")

    response = client.post("/api/datasets/myds/adjust", json={
        "path": "img.png",
        "hue_shift": 0.0,
        "saturation": 1.0,
        "contrast": 1.0,
    })
    assert response.status_code == 400


# ── Batch Adjustment (SSE) ────────────────────────────────────────────────


@patch(f"{_ADJ_MODULE}.dataset_manager")
@patch(f"{_ADJ_MODULE}.asyncio.to_thread")
def test_adjust_batch_returns_sse(mock_to_thread, mock_dm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync

    response = client.post("/api/datasets/myds/adjust-batch", json={
        "paths": ["a.png", "b.png"],
        "hue_shift": 0.0,
        "saturation": 1.0,
        "contrast": 1.0,
    })
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    # Check that SSE events are present
    lines = [line for line in response.text.strip().split("\n") if line.startswith("data:")]
    assert len(lines) >= 3  # 2 files + 1 done event
    last_event = json.loads(lines[-1].replace("data: ", ""))
    assert last_event.get("done") is True


# ── Histogram ────────────────────────────────────────────────────────────


@patch(f"{_ADJ_MODULE}.dataset_manager")
@patch(f"{_ADJ_MODULE}.asyncio.to_thread")
def test_get_histogram_not_found(mock_to_thread, mock_dm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_dm.get_dataset.return_value = None

    response = client.get("/api/datasets/ghost/histogram?image_path=img.png")
    assert response.status_code == 404
