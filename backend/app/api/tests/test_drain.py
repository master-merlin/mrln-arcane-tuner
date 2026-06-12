import pytest

from app.core.drain import DrainActive, is_draining, set_draining


@pytest.fixture(autouse=True)
def _reset_drain():
    set_draining(False)
    yield
    set_draining(False)


def test_starts_not_draining():
    assert is_draining() is False


def test_set_draining_toggles():
    set_draining(True)
    assert is_draining() is True
    set_draining(False)
    assert is_draining() is False


def test_drain_active_is_runtime_error():
    assert issubclass(DrainActive, RuntimeError)


def test_enqueue_blocked_on_gpu_lane_while_draining():
    from app.core.tasks.task_manager import TaskManager

    tm = TaskManager()
    t = tm.create(type="demo", title="demo")
    set_draining(True)
    with pytest.raises(DrainActive):
        tm.enqueue(t.id, lambda _id: None, lane="gpu")


def test_enqueue_allowed_on_background_lane_while_draining():
    from app.core.tasks.task_manager import TaskManager

    tm = TaskManager()
    t = tm.create(type="demo", title="demo")
    set_draining(True)
    tm.enqueue(t.id, lambda _id: None, lane="background")
    tm.join_lane("background", timeout=2.0)


def test_enqueue_allowed_on_gpu_lane_when_not_draining():
    from app.core.tasks.task_manager import TaskManager

    tm = TaskManager()
    t = tm.create(type="demo", title="demo")
    tm.enqueue(t.id, lambda _id: None, lane="gpu")
    tm.join_lane("gpu", timeout=2.0)
