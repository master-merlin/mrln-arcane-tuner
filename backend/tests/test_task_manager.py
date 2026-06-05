import threading as _threading

import pytest
from app.core.tasks.task_manager import TaskManager
from app.core.tasks.task import TaskStatus


@pytest.fixture
def tm():
    m = TaskManager()
    m.set_loop(None)  # no loop in tests → broadcasts become no-ops
    return m


def test_create_then_lifecycle(tm):
    t = tm.create(type="caption_batch", title="Captioning · ds", total=3, dataset_name="ds")
    assert t.status == TaskStatus.PENDING
    assert tm.get(t.id) is t and t.total == 3 and t.current == 0

    tm.start(t.id)
    assert t.status == TaskStatus.RUNNING and t.started_at is not None

    tm.update(t.id, current=1, item="a.png", ok=1)
    assert t.current == 1 and t.current_item == "a.png" and t.ok == 1

    tm.complete(t.id)
    assert t.status == TaskStatus.COMPLETED and t.finished_at is not None


def test_fail_sets_error(tm):
    t = tm.create(type="caption_batch", title="x", total=1)
    tm.start(t.id)
    tm.fail(t.id, "boom")
    assert t.status == TaskStatus.FAILED and t.error == "boom" and t.finished_at is not None


def test_cancel_flag(tm):
    t = tm.create(type="caption_batch", title="x", total=5)
    tm.start(t.id)
    assert tm.is_cancelled(t.id) is False
    tm.cancel(t.id)
    assert tm.is_cancelled(t.id) is True


def test_list_is_insertion_ordered(tm):
    a = tm.create(type="caption_batch", title="a", total=1)
    b = tm.create(type="caption_batch", title="b", total=1)
    assert [t.id for t in tm.list()] == [a.id, b.id]


def test_lane_runs_sequentially(tm):
    order: list[str] = []
    gate = _threading.Event()

    def worker_a(task_id):
        order.append("a-start")
        gate.wait(2)
        order.append("a-end")

    def worker_b(task_id):
        order.append("b-start")

    a = tm.create(type="caption_batch", title="a", total=1)
    b = tm.create(type="caption_batch", title="b", total=1)
    tm.enqueue(a.id, worker_a, lane="gpu")
    tm.enqueue(b.id, worker_b, lane="gpu")

    # b must NOT start until a finishes.
    _threading.Event().wait(0.1)
    assert order == ["a-start"]
    assert tm.get(b.id).status.value == "pending"

    gate.set()
    tm.join_lane("gpu", timeout=2)
    assert order == ["a-start", "a-end", "b-start"]
    assert tm.get(a.id).status.value == "completed"
    assert tm.get(b.id).status.value == "completed"


def test_finish_emits_dataset_invalidated_when_scoped(tm, monkeypatch):
    """A dataset-scoped task that finishes broadcasts dataset.invalidated so
    clients reconcile (caption batch writes/renames files)."""
    calls: list[str] = []
    monkeypatch.setattr(tm, "_emit_dataset_invalidated", lambda name: calls.append(name))
    t = tm.create(type="caption_batch", title="x", total=1, dataset_name="ds")
    tm.start(t.id)
    tm.complete(t.id)
    assert calls == ["ds"]


def test_finish_emits_dataset_invalidated_on_failure(tm, monkeypatch):
    """Partial runs change files too — failure still triggers reconcile."""
    calls: list[str] = []
    monkeypatch.setattr(tm, "_emit_dataset_invalidated", lambda name: calls.append(name))
    t = tm.create(type="caption_batch", title="x", total=1, dataset_name="ds")
    tm.start(t.id)
    tm.fail(t.id, "boom")
    assert calls == ["ds"]


def test_finish_no_invalidate_without_dataset(tm, monkeypatch):
    """Non-dataset tasks must not broadcast a dataset signal."""
    calls: list[str] = []
    monkeypatch.setattr(tm, "_emit_dataset_invalidated", lambda name: calls.append(name))
    t = tm.create(type="export", title="x", total=1)  # no dataset_name
    tm.start(t.id)
    tm.complete(t.id)
    assert calls == []


def test_cancel_pending_never_runs(tm):
    ran: list[str] = []
    gate = _threading.Event()

    def slow(task_id):
        gate.wait(2)

    def should_skip(task_id):
        ran.append("skipped-ran")

    a = tm.create(type="caption_batch", title="a", total=1)
    b = tm.create(type="caption_batch", title="b", total=1)
    tm.enqueue(a.id, slow, lane="gpu")
    tm.enqueue(b.id, should_skip, lane="gpu")

    tm.cancel(b.id)            # cancel while still pending
    gate.set()
    tm.join_lane("gpu", timeout=2)

    assert ran == []                                   # worker never invoked
    assert tm.get(b.id).status.value == "cancelled"
