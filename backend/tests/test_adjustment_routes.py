"""
E2E tests for api/dataset/adjustment_routes.py — color match, histogram, cube export.

The single/batch adjustment routes (/adjust, /adjust-batch) were removed
(W5.T10) — frontend-verified orphans, confirmed via a repo-wide grep for
AdjustmentRequest/BatchAdjustmentRequest/adjustMedia/adjust_media/
applyAdjustment across frontend/src (zero hits). Their dedicated tests are
removed with them.
"""

from unittest.mock import patch


_ADJ_MODULE = "app.api.dataset.adjustment_routes"


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
