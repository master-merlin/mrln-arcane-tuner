"""Overlay routes broadcast entity.changed with entity='overlay'.

The frontend OverlayStore reconciles overlay state off these events.
Emissions happen directly inside the async route handlers (no
``run_coroutine_threadsafe`` needed — we're already on the loop).

We exercise the two destructive endpoints (``delete_overlay`` and
``commit_overlay``) because they're cheap to set up: just an overlay
PNG on disk plus a stub Dataset. The creator (``render_pipeline``) is
covered implicitly — its emission uses the same helper and shape as
``delete``/``commit`` — and is skipped here to avoid pulling in the
full image-processing pipeline and its GPU/model deps just for an
emission check.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dataset_manager import Dataset, dataset_manager
from app.main import app


@pytest.fixture
def overlay_dataset(tmp_path):
    """Seed dataset_manager with a Dataset whose dataset_root has an
    overlay PNG + the source image, so commit/delete have real files
    to act on. Cleans up the registry entry on exit.
    """
    ds_id = f"ds-overlay-emit-{time.time_ns()}"
    ds_name = f"overlay-emit-{ds_id}"
    rel_path = "subdir/image.png"
    ds_root = tmp_path / ds_id
    (ds_root / "subdir").mkdir(parents=True, exist_ok=True)
    (ds_root / "overlays").mkdir(exist_ok=True)

    # Tiny real-ish PNG headers so any imager wouldn't choke if it
    # peeked — but the routes themselves don't read the bytes.
    png_sig = b"\x89PNG\r\n\x1a\n"
    (ds_root / rel_path).write_bytes(png_sig)
    (ds_root / "overlays" / "image.png").write_bytes(png_sig)

    ds = Dataset(
        id=ds_id,
        name=ds_name,
        path=str(ds_root),
        created_at=time.time(),
        multimedia_count=1,
        media_metadata={
            rel_path: {
                "width": 512,
                "height": 512,
                "aspect_ratio": 1.0,
                "orientation": "square",
                "size_bytes": len(png_sig),
                "has_caption": False,
                "is_video": False,
                "enabled": True,
                "has_overlay": True,
                "overlay_hash": "deadbeef",
                "overlay_dimensions": [512, 512],
            }
        },
    )

    saved = dataset_manager.datasets.get(ds_name)
    dataset_manager.datasets[ds_name] = ds
    try:
        yield ds, rel_path
    finally:
        if saved is None:
            dataset_manager.datasets.pop(ds_name, None)
        else:
            dataset_manager.datasets[ds_name] = saved


@pytest.mark.asyncio
async def test_delete_overlay_broadcasts_deleted(overlay_dataset):
    """DELETE /api/datasets/{name}/overlay/{image_path:path} emits
    entity.changed:deleted (entity='overlay', id='{ds}/{path}', payload=None).
    """
    ds, rel_path = overlay_dataset
    mock_broadcast = AsyncMock()

    transport = ASGITransport(app=app)
    with patch(
        "app.api.dataset.overlay_routes.event_manager.broadcast",
        mock_broadcast,
    ):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.delete(
                f"/api/datasets/{ds.name}/overlay/{rel_path}",
            )

    assert resp.status_code == 200, resp.text

    entity_calls = [
        c for c in mock_broadcast.await_args_list
        if c.args and c.args[0] == "entity.changed"
    ]
    deletes = [
        c for c in entity_calls
        if c.args[1]["op"] == "deleted" and c.args[1]["entity"] == "overlay"
    ]
    assert len(deletes) == 1, (
        f"expected one overlay deleted event, got {len(deletes)} from {entity_calls}"
    )
    env = deletes[0].args[1]
    assert env["id"] == f"{ds.name}/{rel_path}"
    assert env["payload"] is None

    # Overlay PNG should be gone on disk.
    assert not (Path(ds.path) / "overlays" / "image.png").exists()


@pytest.mark.asyncio
async def test_commit_overlay_broadcasts_deleted(overlay_dataset):
    """POST /api/datasets/{name}/overlay/commit emits entity.changed:deleted
    (commit flattens the overlay into the original, removing the overlay
    file — from the OverlayStore's perspective the overlay is gone).
    """
    ds, rel_path = overlay_dataset
    mock_broadcast = AsyncMock()

    transport = ASGITransport(app=app)
    with patch(
        "app.api.dataset.overlay_routes.event_manager.broadcast",
        mock_broadcast,
    ):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                f"/api/datasets/{ds.name}/overlay/commit",
                json={"image_path": rel_path},
            )

    assert resp.status_code == 200, resp.text

    entity_calls = [
        c for c in mock_broadcast.await_args_list
        if c.args and c.args[0] == "entity.changed"
    ]
    overlay_deletes = [
        c for c in entity_calls
        if c.args[1]["op"] == "deleted" and c.args[1]["entity"] == "overlay"
    ]
    assert len(overlay_deletes) == 1, (
        f"expected one overlay deleted event, got {len(overlay_deletes)} from {entity_calls}"
    )
    env = overlay_deletes[0].args[1]
    assert env["id"] == f"{ds.name}/{rel_path}"
    assert env["payload"] is None

    # Overlay PNG should be gone (flattened into the original).
    assert not (Path(ds.path) / "overlays" / "image.png").exists()
