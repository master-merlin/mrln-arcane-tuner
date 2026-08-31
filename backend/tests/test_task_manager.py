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


def test_finish_cancelled_marks_task_cancelled(tm):
    """Public finish_cancelled() (P2c / B-CLEAN-10) must behave identically to
    the old private _finish(task_id, TaskStatus.CANCELLED) call that the 7
    batch workers used to reach into directly."""
    t = tm.create(type="caption_batch", title="x", total=5)
    tm.start(t.id)
    tm.finish_cancelled(t.id)
    assert t.status == TaskStatus.CANCELLED and t.finished_at is not None


def test_finish_cancelled_emits_dataset_invalidated_when_scoped(tm, monkeypatch):
    """Same dataset-reconcile broadcast as complete()/fail() — cancelled runs
    can still have written/renamed files before the cancel was observed."""
    calls: list[str] = []
    monkeypatch.setattr(tm, "_emit_dataset_invalidated", lambda name: calls.append(name))
    t = tm.create(type="caption_batch", title="x", total=1, dataset_name="ds")
    tm.start(t.id)
    tm.finish_cancelled(t.id)
    assert calls == ["ds"]


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


def test_create_defaults_user_visible_true(tm):
    t = tm.create(type="x", title="x")
    assert t.user_visible is True
    assert "user_visible" in t.model_dump()


def test_create_silent_task(tm):
    t = tm.create(type="x", title="x", user_visible=False)
    assert t.user_visible is False


def test_terminal_records_pruned_beyond_max(tm):
    """Unbounded growth guard (W4.T5): _tasks/_cancels/_last_emit only ever
    grew before this — a long session of batch ops (caption batches, rescan
    batches, etc.) would leak memory and bloat every GET /tasks poll forever.
    Terminal (finished) records must be capped at the newest 500."""
    ids: list[str] = []
    for i in range(600):
        t = tm.create(type="caption_batch", title=f"t{i}", total=1)
        tm.start(t.id)
        tm.complete(t.id)
        ids.append(t.id)

    assert len(tm.list()) <= 500
    assert len(tm._cancels) <= 500
    assert len(tm._last_emit) <= 500

    # Oldest pruned, newest retained.
    assert tm.get(ids[0]) is None
    assert ids[0] not in tm._cancels
    assert tm.get(ids[-1]) is not None
    assert ids[-1] in tm._cancels


def test_terminal_pruning_never_drops_active_tasks(tm):
    """A still-RUNNING/PENDING task must never be pruned, even once 500+
    OTHER tasks have finished after it."""
    active = tm.create(type="caption_batch", title="active", total=1)
    tm.start(active.id)

    for i in range(600):
        t = tm.create(type="caption_batch", title=f"t{i}", total=1)
        tm.start(t.id)
        tm.complete(t.id)

    assert tm.get(active.id) is not None
    assert tm.get(active.id).status == TaskStatus.RUNNING


class TestMaintenanceNeverBlocksUserWork:
    """A task the user cannot see must never hold the lane the user's work runs
    on (ARCHITECTURE D10, "never block a shared lane").

    Measured on the live server during UAT round 4 (LANE-52): the boot-time
    `cache_stats_warmup` (`user_visible=False`) ran 532 s on `lane="background"`
    and a user-initiated caption-refine POSTed two minutes later sat `pending`
    with `started_at=None` until the warm-up's exact finishing microsecond. One
    FIFO thread per lane plus a silent tenant on that lane is starvation by
    construction, and no amount of care inside either worker prevents it.

    The rule is structural rather than per-callsite: `enqueue` routes any
    `user_visible=False` task onto the lane's `:maintenance` sibling, so a
    future silent task cannot be dropped onto a user lane by accident. Silent
    tasks still serialise against EACH OTHER - that is the point, not the same
    bug moved: unbounded latency between two housekeeping sweeps costs a user
    nothing, while unbounded latency in front of a user's own request is the
    defect.
    """

    def test_a_silent_task_does_not_delay_a_visible_one(self, tm):
        started: list[str] = []
        release = _threading.Event()

        def slow_maintenance(task_id):
            started.append("maintenance")
            # Longer than the poll window below, so "the user's task ran" can
            # only mean it overtook the sweep, never that the sweep ended first.
            release.wait(30)

        def user_work(task_id):
            started.append("user")

        warm = tm.create(type="cache_stats_warmup", title="Cache stats",
                         user_visible=False)
        refine = tm.create(type="caption_refine_batch", title="Refine", total=1)
        tm.enqueue(warm.id, slow_maintenance, lane="background")
        tm.enqueue(refine.id, user_work, lane="background")

        # The user's task must reach a terminal state while the sweep runs on.
        for _ in range(40):
            if "user" in started:
                break
            _threading.Event().wait(0.05)

        assert started[:1] == ["maintenance"], "the sweep never started"
        assert "user" in started, (
            "the user's task waited behind an invisible maintenance sweep"
        )
        assert tm.get(refine.id).status == TaskStatus.COMPLETED
        assert tm.get(warm.id).status == TaskStatus.RUNNING
        release.set()
        tm.join_lane("background:maintenance", timeout=5)

    def test_two_silent_tasks_still_serialise(self, tm):
        """The fix must not turn maintenance into unbounded fan-out: two silent
        sweeps share the maintenance lane and run one at a time."""
        order: list[str] = []
        gate = _threading.Event()

        def first(task_id):
            order.append("first-start")
            gate.wait(5)
            order.append("first-end")

        def second(task_id):
            order.append("second-start")

        a = tm.create(type="cache_stats_warmup", title="a", user_visible=False)
        b = tm.create(type="cache_stats_warmup", title="b", user_visible=False)
        tm.enqueue(a.id, first, lane="background")
        tm.enqueue(b.id, second, lane="background")

        _threading.Event().wait(0.3)
        assert order == ["first-start"]
        gate.set()
        tm.join_lane("background:maintenance", timeout=5)
        assert order == ["first-start", "first-end", "second-start"]

    def test_a_silent_gpu_task_is_still_refused_while_draining(self, tm, monkeypatch):
        """Lane derivation happens AFTER the drain check, so a silent GPU task
        cannot sneak past the drain gate by landing on `gpu:maintenance`."""
        import sys

        from app.core.drain import DrainActive
        # `app.core.tasks.task_manager` as an ATTRIBUTE of the package is the
        # singleton instance (re-exported by __init__), not the module — go
        # through sys.modules or monkeypatch retargets the wrong object.
        tm_mod = sys.modules["app.core.tasks.task_manager"]
        monkeypatch.setattr(tm_mod, "is_draining", lambda: True)

        silent = tm.create(type="cache_stats_warmup", title="x", user_visible=False)
        with pytest.raises(DrainActive):
            tm.enqueue(silent.id, lambda _id: None, lane="gpu")


class TestQueuedTaskSaysWhyItWaits:
    """`pending` with no reason is indistinguishable from broken - which is
    exactly how the user read the nine-minute wait in UAT round 4 (LANE-52).
    `queue_position` counts the tasks that must finish on this task's lane
    before it starts; 0 means "next, or already running"."""

    def test_position_counts_the_tasks_ahead(self, tm):
        gate = _threading.Event()

        a = tm.create(type="caption_batch", title="a")
        b = tm.create(type="caption_batch", title="b")
        c = tm.create(type="caption_batch", title="c")
        tm.enqueue(a.id, lambda _id: gate.wait(5), lane="gpu")
        for _ in range(100):
            if tm.get(a.id).status == TaskStatus.RUNNING:
                break
            _threading.Event().wait(0.05)
        tm.enqueue(b.id, lambda _id: None, lane="gpu")
        tm.enqueue(c.id, lambda _id: None, lane="gpu")

        assert tm.get(a.id).queue_position == 0, "a running task is not waiting"
        assert tm.get(b.id).queue_position == 1, "b waits on the running task"
        assert tm.get(c.id).queue_position == 2, "c waits on the running task and on b"

        gate.set()
        tm.join_lane("gpu", timeout=5)
        assert tm.get(c.id).queue_position == 0

    def test_position_shrinks_as_the_lane_drains(self, tm):
        first_gate = _threading.Event()
        second_gate = _threading.Event()

        a = tm.create(type="caption_batch", title="a")
        b = tm.create(type="caption_batch", title="b")
        c = tm.create(type="caption_batch", title="c")
        tm.enqueue(a.id, lambda _id: first_gate.wait(5), lane="gpu")
        tm.enqueue(b.id, lambda _id: second_gate.wait(5), lane="gpu")
        tm.enqueue(c.id, lambda _id: None, lane="gpu")

        first_gate.set()
        for _ in range(100):
            if tm.get(c.id).queue_position == 1:
                break
            _threading.Event().wait(0.05)
        assert tm.get(c.id).queue_position == 1, (
            "the queue drained but the waiting task still advertises its old place"
        )
        second_gate.set()
        tm.join_lane("gpu", timeout=5)

    def test_the_task_names_the_lane_it_waits_on(self, tm):
        t = tm.create(type="cache_stats_warmup", title="x", user_visible=False)
        tm.enqueue(t.id, lambda _id: None, lane="background")
        tm.join_lane("background:maintenance", timeout=5)
        assert tm.get(t.id).lane == "background:maintenance"

    def test_position_is_broadcast_when_it_changes(self, tm):
        """The Task Center learns its place from the `task_update` frame, not
        from a poll - so a shrinking position that is never broadcast is a
        number nobody sees."""
        sent: list[tuple[str, int]] = []
        tm._broadcast = lambda task, throttle=False: sent.append(  # type: ignore[method-assign]
            (task.id, task.queue_position))

        gate = _threading.Event()
        a = tm.create(type="caption_batch", title="a")
        b = tm.create(type="caption_batch", title="b")
        tm.enqueue(a.id, lambda _id: gate.wait(5), lane="gpu")
        for _ in range(100):
            if tm.get(a.id).status == TaskStatus.RUNNING:
                break
            _threading.Event().wait(0.05)
        sent.clear()
        tm.enqueue(b.id, lambda _id: None, lane="gpu")

        assert (b.id, 1) in sent, "the queued task never announced its place"
        gate.set()
        tm.join_lane("gpu", timeout=5)
