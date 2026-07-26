import time

import pytest

from app.core.tasks.task_manager import task_manager
from app.core.image_processing import pipeline_batch

PATHS = ["a.png", "b.png"]
BLOCKS = [{"type": "white_balance", "enabled": True, "params": {"temperature": 6500, "tint": 0}}]


@pytest.fixture(autouse=True)
def no_loop():
    task_manager.set_loop(None)


def test_pipeline_all_items_completes(monkeypatch):
    rendered = []
    monkeypatch.setattr(
        pipeline_batch, "_render_one",
        lambda name, path, blocks, ts, tp, rr: rendered.append((path, ts, tp, rr)),
    )
    t = task_manager.create(type="adjust_batch", title="x", total=2, dataset_name="ds")
    pipeline_batch.run_pipeline_batch(
        t.id, dataset_name="ds", image_paths=PATHS, blocks=BLOCKS,
        tile_size=512, tile_pad=32, replace_recipe=False,
    )
    assert [r[0] for r in rendered] == ["a.png", "b.png"]
    assert rendered[0] == ("a.png", 512, 32, False)
    task = task_manager.get(t.id)
    assert task.status.value == "completed"
    assert task.ok == 2
    assert task.failed == 0


def test_pipeline_item_error_continues(monkeypatch):
    def render(name, path, blocks, ts, tp, rr):
        if path == "a.png":
            raise RuntimeError("GPU OOM")
    monkeypatch.setattr(pipeline_batch, "_render_one", render)
    t = task_manager.create(type="adjust_batch", title="x", total=2, dataset_name="ds")
    pipeline_batch.run_pipeline_batch(
        t.id, dataset_name="ds", image_paths=PATHS, blocks=BLOCKS,
        tile_size=512, tile_pad=32, replace_recipe=False,
    )
    task = task_manager.get(t.id)
    assert task.status.value == "completed"
    assert task.ok == 1
    assert task.failed == 1


def test_pipeline_cancel_midway(monkeypatch):
    rendered = []
    t = task_manager.create(type="adjust_batch", title="x", total=2, dataset_name="ds")

    def render_then_cancel(name, path, blocks, ts, tp, rr):
        rendered.append(path)
        task_manager.cancel(t.id)
    monkeypatch.setattr(pipeline_batch, "_render_one", render_then_cancel)

    pipeline_batch.run_pipeline_batch(
        t.id, dataset_name="ds", image_paths=PATHS, blocks=BLOCKS,
        tile_size=512, tile_pad=32, replace_recipe=False,
    )
    assert rendered == ["a.png"]
    assert task_manager.get(t.id).status.value == "cancelled"


def test_pipeline_progress_advances(monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline_batch, "_render_one", lambda *a: None)
    orig_update = task_manager.update

    def spy(task_id, **kw):
        if "current" in kw:
            seen.append(kw["current"])
        return orig_update(task_id, **kw)
    monkeypatch.setattr(task_manager, "update", spy)

    t = task_manager.create(type="adjust_batch", title="x", total=2, dataset_name="ds")
    pipeline_batch.run_pipeline_batch(
        t.id, dataset_name="ds", image_paths=PATHS, blocks=BLOCKS,
        tile_size=512, tile_pad=32, replace_recipe=False,
    )
    assert seen == [1, 2]


def test_render_one_writes_overlay_and_sets_metadata(monkeypatch, tmp_path):
    """Integration: real _render_one with a CPU block (white_balance) on a tiny
    image. No GPU/model deps."""
    from PIL import Image
    from app.core.dataset_manager import Dataset, dataset_manager

    ds_id = f"ds-pb-{time.time_ns()}"
    ds_name = f"pb-{ds_id}"
    rel = "img.png"
    ds_root = tmp_path / ds_id
    ds_root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color=(120, 80, 40)).save(ds_root / rel, format="PNG")

    ds = Dataset(
        id=ds_id, name=ds_name, path=str(ds_root), created_at=time.time(),
        media_metadata={rel: {"width": 64, "height": 48, "has_overlay": False}},
    )
    dataset_manager.datasets[ds_name] = ds
    monkeypatch.setattr(dataset_manager, "_persist_media_item", lambda dataset, path: None)
    try:
        pipeline_batch._render_one(ds_name, rel, BLOCKS, 512, 32, False)
        assert (ds_root / "overlays" / "img.png").exists()
        assert (ds_root / "overlays.json").exists()
        assert ds.media_metadata[rel]["has_overlay"] is True
        assert ds.media_metadata[rel]["overlay_dimensions"] == [64, 48]
    finally:
        dataset_manager.datasets.pop(ds_name, None)


def test_render_pipeline_batch_route_enqueues(monkeypatch, tmp_path):
    """Route now resolves the dataset + validates every image_path (traversal
    guard) before enqueueing — monkeypatch get_dataset so 'ds1' resolves to a
    real tmp-rooted dataset instead of the unregistered name this test used
    pre-fix (which relied on the route never looking the dataset up)."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.api.dataset import overlay_routes

    monkeypatch.setattr(
        overlay_routes.dataset_manager,
        "get_dataset",
        lambda name: type(
            "_FakeDataset", (), {"path": str(tmp_path), "media_metadata": {}}
        )(),
    )

    captured = {}

    def fake_enqueue(task_id, worker_fn, *, lane="gpu"):
        captured["task_id"] = task_id
        captured["lane"] = lane
    monkeypatch.setattr(overlay_routes.task_manager, "enqueue", fake_enqueue)

    client = TestClient(app)
    resp = client.post(
        "/api/datasets/ds1/render-pipeline/batch",
        json={"image_paths": ["a.png", "b.png"],
              "blocks": [{"type": "contrast", "enabled": True, "params": {"factor": 1.1}}]},
    )
    assert resp.status_code == 200
    assert resp.json()["task_id"] == captured["task_id"]
    assert captured["lane"] == "gpu"


def test_render_pipeline_task_route_enqueues(monkeypatch, tmp_path):
    """Same dataset-resolution note as test_render_pipeline_batch_route_enqueues."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.api.dataset import overlay_routes

    monkeypatch.setattr(
        overlay_routes.dataset_manager,
        "get_dataset",
        lambda name: type(
            "_FakeDataset", (), {"path": str(tmp_path), "media_metadata": {}}
        )(),
    )

    captured = {}

    def fake_enqueue(task_id, worker_fn, *, lane="gpu"):
        captured["task_id"] = task_id
        captured["lane"] = lane
    monkeypatch.setattr(overlay_routes.task_manager, "enqueue", fake_enqueue)

    created = {}
    orig_create = overlay_routes.task_manager.create

    def spy_create(**kw):
        created.update(kw)
        return orig_create(**kw)
    monkeypatch.setattr(overlay_routes.task_manager, "create", spy_create)

    client = TestClient(app)
    resp = client.post(
        "/api/datasets/ds1/render-pipeline/task",
        json={"image_path": "a.png",
              "blocks": [{"type": "upscale", "enabled": True, "params": {}}]},
    )
    assert resp.status_code == 200
    assert resp.json()["task_id"] == captured["task_id"]
    assert captured["lane"] == "gpu"
    assert created["total"] == 1
    assert created["type"] == "render_task"
