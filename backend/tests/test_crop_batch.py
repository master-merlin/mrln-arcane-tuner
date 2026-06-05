import pytest

from app.core.tasks.task_manager import task_manager
from app.core.dataset import crop_batch

ITEMS = [
    {"path": "a.png", "target_width": 512, "target_height": 512},
    {"path": "b.png", "target_width": 768, "target_height": 512},
]


@pytest.fixture(autouse=True)
def no_loop():
    task_manager.set_loop(None)


def test_crop_all_items_completes(monkeypatch):
    cropped = []
    monkeypatch.setattr(
        crop_batch, "_crop",
        lambda name, path, tw, th, origin: cropped.append((path, tw, th, origin)),
    )
    t = task_manager.create(type="crop_batch", title="x", total=2, dataset_name="ds")
    crop_batch.run_crop_batch(t.id, dataset_name="ds", items=ITEMS, origin="top")

    assert cropped == [("a.png", 512, 512, "top"), ("b.png", 768, 512, "top")]
    task = task_manager.get(t.id)
    assert task.status.value == "completed"
    assert task.ok == 2
    assert task.failed == 0


def test_crop_item_error_continues(monkeypatch):
    def crop(name, path, tw, th, origin):
        if path == "a.png":
            raise RuntimeError("bad image")
    monkeypatch.setattr(crop_batch, "_crop", crop)

    t = task_manager.create(type="crop_batch", title="x", total=2, dataset_name="ds")
    crop_batch.run_crop_batch(t.id, dataset_name="ds", items=ITEMS, origin="center")

    task = task_manager.get(t.id)
    assert task.status.value == "completed"
    assert task.ok == 1
    assert task.failed == 1


def test_crop_cancel_midway(monkeypatch):
    cropped = []

    t = task_manager.create(type="crop_batch", title="x", total=2, dataset_name="ds")

    def crop_then_cancel(name, path, tw, th, origin):
        cropped.append(path)
        task_manager.cancel(t.id)
    monkeypatch.setattr(crop_batch, "_crop", crop_then_cancel)

    crop_batch.run_crop_batch(t.id, dataset_name="ds", items=ITEMS, origin="center")

    assert cropped == ["a.png"]
    assert task_manager.get(t.id).status.value == "cancelled"


def test_crop_progress_advances(monkeypatch):
    seen = []
    monkeypatch.setattr(crop_batch, "_crop", lambda *a: None)
    orig_update = task_manager.update

    def spy(task_id, **kw):
        if "current" in kw:
            seen.append(kw["current"])
        return orig_update(task_id, **kw)
    monkeypatch.setattr(task_manager, "update", spy)

    t = task_manager.create(type="crop_batch", title="x", total=2, dataset_name="ds")
    crop_batch.run_crop_batch(t.id, dataset_name="ds", items=ITEMS, origin="center")

    assert seen == [1, 2]
