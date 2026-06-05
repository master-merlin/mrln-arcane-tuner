import pytest

from app.core.tasks.task_manager import task_manager
from app.core.masking import mask_generate_batch


class StubService:
    def generate_mask(self, image_path, model_id, params):
        return f"mask::{image_path}"


@pytest.fixture(autouse=True)
def no_loop():
    task_manager.set_loop(None)


def test_generate_writes_and_unloads(monkeypatch):
    saved, unloaded = [], {"v": False}
    monkeypatch.setattr(mask_generate_batch, "_get_service", lambda: StubService())
    monkeypatch.setattr(mask_generate_batch, "_full_path", lambda ds, rel: f"/full/{rel}")
    monkeypatch.setattr(mask_generate_batch, "_save_mask",
                        lambda ds, rel, mask: saved.append((rel, mask)))
    monkeypatch.setattr(mask_generate_batch, "_unload",
                        lambda: unloaded.__setitem__("v", True))

    t = task_manager.create(type="mask_generate_batch", title="x", total=2, dataset_name="ds")
    mask_generate_batch.run_mask_generate_batch(
        t.id, dataset_name="ds", image_rel_paths=["a.png", "b.png"],
        model_id="rembg", params={},
    )

    assert [s[0] for s in saved] == ["a.png", "b.png"]
    assert saved[0][1] == "mask::/full/a.png"
    task = task_manager.get(t.id)
    assert task.status.value == "completed"
    assert task.ok == 2
    assert unloaded["v"] is True


def test_generate_cancel_midway(monkeypatch):
    saved, unloaded = [], {"v": False}
    monkeypatch.setattr(mask_generate_batch, "_get_service", lambda: StubService())
    monkeypatch.setattr(mask_generate_batch, "_full_path", lambda ds, rel: f"/full/{rel}")
    monkeypatch.setattr(mask_generate_batch, "_unload",
                        lambda: unloaded.__setitem__("v", True))

    t = task_manager.create(type="mask_generate_batch", title="x", total=3, dataset_name="ds")

    def save_then_cancel(ds, rel, mask):
        saved.append(rel)
        task_manager.cancel(t.id)
    monkeypatch.setattr(mask_generate_batch, "_save_mask", save_then_cancel)

    mask_generate_batch.run_mask_generate_batch(
        t.id, dataset_name="ds", image_rel_paths=["a.png", "b.png", "c.png"],
        model_id="rembg", params={},
    )

    assert saved == ["a.png"]
    assert task_manager.get(t.id).status.value == "cancelled"
    assert unloaded["v"] is True


def test_generate_item_error_continues(monkeypatch):
    monkeypatch.setattr(mask_generate_batch, "_get_service", lambda: StubService())
    monkeypatch.setattr(mask_generate_batch, "_full_path", lambda ds, rel: f"/full/{rel}")
    monkeypatch.setattr(mask_generate_batch, "_unload", lambda: None)

    def save(ds, rel, mask):
        if rel == "bad.png":
            raise RuntimeError("disk full")
    monkeypatch.setattr(mask_generate_batch, "_save_mask", save)

    t = task_manager.create(type="mask_generate_batch", title="x", total=2, dataset_name="ds")
    mask_generate_batch.run_mask_generate_batch(
        t.id, dataset_name="ds", image_rel_paths=["bad.png", "good.png"],
        model_id="rembg", params={},
    )

    task = task_manager.get(t.id)
    assert task.status.value == "completed"
    assert task.ok == 1
    assert task.failed == 1


def test_generate_reconciles_mask_count_when_saved(monkeypatch):
    from app.core.dataset_manager import dataset_manager
    calls = []
    monkeypatch.setattr(mask_generate_batch, "_get_service", lambda: StubService())
    monkeypatch.setattr(mask_generate_batch, "_full_path", lambda ds, rel: f"/full/{rel}")
    monkeypatch.setattr(mask_generate_batch, "_save_mask", lambda ds, rel, mask: None)
    monkeypatch.setattr(mask_generate_batch, "_unload", lambda: None)
    monkeypatch.setattr(dataset_manager, "reconcile_mask_count", lambda name: calls.append(name))

    t = task_manager.create(type="mask_generate_batch", title="x", total=1, dataset_name="ds")
    mask_generate_batch.run_mask_generate_batch(
        t.id, dataset_name="ds", image_rel_paths=["a.png"], model_id="rembg", params={},
    )
    assert calls == ["ds"]


def test_generate_skips_reconcile_when_nothing_saved(monkeypatch):
    from app.core.dataset_manager import dataset_manager
    calls = []
    monkeypatch.setattr(mask_generate_batch, "_get_service", lambda: StubService())
    monkeypatch.setattr(mask_generate_batch, "_full_path", lambda ds, rel: f"/full/{rel}")
    monkeypatch.setattr(mask_generate_batch, "_save_mask",
                        lambda ds, rel, mask: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(mask_generate_batch, "_unload", lambda: None)
    monkeypatch.setattr(dataset_manager, "reconcile_mask_count", lambda name: calls.append(name))

    t = task_manager.create(type="mask_generate_batch", title="x", total=1, dataset_name="ds")
    mask_generate_batch.run_mask_generate_batch(
        t.id, dataset_name="ds", image_rel_paths=["a.png"], model_id="rembg", params={},
    )
    assert calls == []  # every item failed → mask_count unchanged, no reconcile


def test_reconcile_mask_count_recomputes(monkeypatch):
    from app.core.dataset_manager import dataset_manager, Dataset
    monkeypatch.setattr(dataset_manager, "_persist_dataset", lambda ds: None)
    monkeypatch.setattr(dataset_manager, "_loop", None)  # no broadcast in tests
    ds = Dataset(
        id="i1", name="rds", path="/tmp/rds", created_at=0.0, mask_count=0,
        media_metadata={"a.png": {"has_mask": True}, "b.png": {"has_mask": True}, "c.png": {}},
    )
    dataset_manager.datasets["rds"] = ds
    try:
        dataset_manager.reconcile_mask_count("rds")
        assert ds.mask_count == 2
    finally:
        dataset_manager.datasets.pop("rds", None)
