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

import asyncio
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


@pytest.mark.asyncio
async def test_commit_overlay_does_not_revert_concurrent_overlay_dims_race(
    overlay_dataset, monkeypatch,
):
    """Regression: ``commit_overlay`` used to read
    ``dataset.media_metadata[key]["overlay_dimensions"]`` via a plain,
    unlocked dict access to seed the width/height it writes through the
    atomic ``update_media_flags``. If a concurrent request (e.g. another
    overlay render, or a mask delete — precisely T14's own docstring
    scenario) updated ``overlay_dimensions`` in the window between that
    stale read and commit's own ``update_media_flags`` call, the OLD
    value would still get written under the lock — silently reverting the
    concurrent request's newer state.

    This fires a same-shaped concurrent update (via the same atomic
    ``update_media_flags`` a real racer would use) at the exact point the
    OLD code's plain read preceded: the ``_size_if_exists`` ``to_thread``
    hop, the only ``await`` between the old unlocked read and commit's
    lock call. Asserts the FINAL width/height reflect the concurrent
    update's dimensions, not the ones present when the request started —
    provable only if the derivation happens atomically with the write,
    not via an earlier unlocked read.
    """
    ds, rel_path = overlay_dataset
    assert ds.media_metadata[rel_path]["overlay_dimensions"] == [512, 512]

    real_to_thread = asyncio.to_thread
    fired = {"done": False}

    async def _to_thread_with_race(func, *args, **kwargs):
        if not fired["done"] and getattr(func, "__name__", "") == "_size_if_exists":
            fired["done"] = True
            # Simulate a second, concurrent request (e.g. another overlay
            # render) updating this item's overlay dimensions via the SAME
            # atomic path a real racer would use — landing strictly
            # between commit_overlay's old stale read and its own
            # update_media_flags call.
            dataset_manager.update_media_flags(
                ds.name, rel_path, overlay_dimensions=[1024, 768],
            )
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _to_thread_with_race)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            f"/api/datasets/{ds.name}/overlay/commit",
            json={"image_path": rel_path},
        )

    assert resp.status_code == 200, resp.text
    assert fired["done"], "race hook never fired — test setup is stale"

    meta = ds.media_metadata[rel_path]
    assert meta["width"] == 1024, (
        f"width must reflect the CONCURRENT update (1024), not a value "
        f"read before it landed; got {meta.get('width')}"
    )
    assert meta["height"] == 768, (
        f"height must reflect the CONCURRENT update (768), not a value "
        f"read before it landed; got {meta.get('height')}"
    )


@pytest.mark.asyncio
async def test_render_pipeline_replace_recipe_uses_original_source(overlay_dataset, tmp_path):
    """When replace_recipe=true, the render must source from the original
    image — not from an existing overlay PNG. Regression for the
    dimension-chain bug: a recipe with upscale produces a 2x overlay; a
    subsequent recipe with only WB (replace_recipe=true) must produce a
    1x overlay matching the original's dimensions, not a 2x one chained
    from the previous render.
    """
    from PIL import Image

    ds, rel_path = overlay_dataset

    # Replace the fixture's tiny PNG header with a real 100×80 image so PIL
    # can actually open it during render.
    src = Path(ds.path) / rel_path
    Image.new("RGB", (100, 80), color=(128, 64, 32)).save(src, format="PNG")

    # Pre-seed an overlay PNG at 200×160 to simulate a prior upscale chain.
    overlay = Path(ds.path) / "overlays" / "image.png"
    Image.new("RGB", (200, 160), color=(64, 32, 16)).save(overlay, format="PNG")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            f"/api/datasets/{ds.name}/render-pipeline",
            json={
                "image_path": rel_path,
                "blocks": [{"type": "white_balance", "enabled": True, "params": {"temperature": 6500, "tint": 0}}],
                "tile_size": 512,
                "tile_pad": 32,
                "replace_recipe": True,
            },
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Original dimensions (100×80), NOT chained from the pre-seeded 200×160 overlay.
    assert body["dimensions"] == [100, 80], (
        f"replace_recipe=true should source from original; got dims {body['dimensions']}"
    )
