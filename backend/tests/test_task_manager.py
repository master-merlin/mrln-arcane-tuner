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
