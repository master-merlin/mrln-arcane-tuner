import pytest

from app.core.tasks.task_manager import task_manager
from app.core.dataset import rescan_batch


@pytest.fixture(autouse=True)
def no_loop():
    task_manager.set_loop(None)


def test_count_multimedia_sums(monkeypatch):
    monkeypatch.setattr(rescan_batch, "_dataset_file_count",
                        lambda name: {"a": 3, "b": 5}.get(name, 0))
    assert rescan_batch.count_multimedia(["a", "b"]) == 8
    assert rescan_batch.count_multimedia([]) == 0


def test_run_rescan_single_completes(monkeypatch):
    scanned = []
    monkeypatch.setattr(rescan_batch, "_dataset_file_count", lambda name: 2)
    monkeypatch.setattr(rescan_batch, "_unload", lambda: None)

    def fake_scan(name, force_full, progress_cb):
        scanned.append((name, force_full))
        progress_cb(1, 2, "a.png")
        progress_cb(2, 2, "b.png")
    monkeypatch.setattr(rescan_batch, "_scan", fake_scan)

    t = task_manager.create(type="rescan_batch", title="x", total=2, dataset_name="ds")
    rescan_batch.run_rescan_batch(t.id, dataset_names=["ds"], force_full=False, total=2)

    assert scanned == [("ds", False)]
    task = task_manager.get(t.id)
    assert task.status.value == "completed"
    assert task.current == 2
    assert task.ok == 2


def test_run_rescan_library_global_counter(monkeypatch):
    monkeypatch.setattr(rescan_batch, "_dataset_file_count", lambda name: 2)
    monkeypatch.setattr(rescan_batch, "_unload", lambda: None)

    def fake_scan(name, force_full, progress_cb):
        progress_cb(1, 2, "x.png")
        progress_cb(2, 2, "y.png")
    monkeypatch.setattr(rescan_batch, "_scan", fake_scan)

    currents, items = [], []
    real_update = task_manager.update

    def rec(task_id, *, current=None, item=None, ok=None, failed=None):
        if current is not None:
            currents.append(current)
        if item is not None:
            items.append(item)
        real_update(task_id, current=current, item=item, ok=ok, failed=failed)
    monkeypatch.setattr(task_manager, "update", rec)

    t = task_manager.create(type="rescan_batch", title="x", total=4, dataset_name=None)
    rescan_batch.run_rescan_batch(t.id, dataset_names=["d1", "d2"], force_full=True, total=4)

    assert 3 in currents and 4 in currents          # counter spans both datasets
    assert any("d2 → " in i for i in items)     # current_item carries dataset name
    assert task_manager.get(t.id).current == 4
    assert task_manager.get(t.id).status.value == "completed"


def test_run_rescan_cancel_between_datasets(monkeypatch):
    monkeypatch.setattr(rescan_batch, "_dataset_file_count", lambda name: 1)
    monkeypatch.setattr(rescan_batch, "_unload", lambda: None)
    scanned = []
    t = task_manager.create(type="rescan_batch", title="x", total=2, dataset_name=None)

    def fake_scan(name, force_full, progress_cb):
        scanned.append(name)
        task_manager.cancel(t.id)                   # cancel after first dataset
    monkeypatch.setattr(rescan_batch, "_scan", fake_scan)

    rescan_batch.run_rescan_batch(t.id, dataset_names=["d1", "d2"], force_full=False, total=2)

    assert scanned == ["d1"]                         # second dataset skipped
    assert task_manager.get(t.id).status.value == "cancelled"


def test_run_rescan_dataset_error_continues(monkeypatch):
    monkeypatch.setattr(rescan_batch, "_dataset_file_count", lambda name: 1)
    monkeypatch.setattr(rescan_batch, "_unload", lambda: None)
    scanned = []

    def fake_scan(name, force_full, progress_cb):
        scanned.append(name)
        if name == "bad":
            raise RuntimeError("boom")
    monkeypatch.setattr(rescan_batch, "_scan", fake_scan)

    t = task_manager.create(type="rescan_batch", title="x", total=2, dataset_name=None)
    rescan_batch.run_rescan_batch(t.id, dataset_names=["bad", "good"], force_full=False, total=2)

    assert scanned == ["bad", "good"]               # continued past the failure
    task = task_manager.get(t.id)
    assert task.status.value == "completed"
    assert task.failed == 1
    assert task.ok == 1                             # only "good" counted ok


def test_run_rescan_unloads_in_finally(monkeypatch):
    unloaded = {"v": False}
    monkeypatch.setattr(rescan_batch, "_dataset_file_count", lambda name: 1)
    monkeypatch.setattr(rescan_batch, "_unload", lambda: unloaded.__setitem__("v", True))
    monkeypatch.setattr(rescan_batch, "_scan", lambda name, force_full, progress_cb: None)

    t = task_manager.create(type="rescan_batch", title="x", total=1, dataset_name="ds")
    rescan_batch.run_rescan_batch(t.id, dataset_names=["ds"], force_full=False, total=1)

    assert unloaded["v"] is True
