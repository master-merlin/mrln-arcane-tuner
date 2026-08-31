# backend/tests/test_task_batch_terminal_state.py
"""LANE-52 — a counted batch's terminal state comes from its own tally.

Defect (UAT round 4, measured on the live server): a `caption_refine_batch`
whose ONE item raised `httpx.ReadTimeout` was recorded

    status="completed", current=1, ok=0, failed=1, error=None

because the worker's per-item `except` swallowed the exception into a counter
and the epilogue called `task_manager.complete(task_id)` unconditionally. The
Task Center showed a finished task, nothing had changed on disk, and no reason
was given anywhere — ARCHITECTURE D10 "failure never silent".

These tests assert on the OBSERVABLE task record (a real TaskManager, not a
mock of the thing under test), and every adopter is checked, because the
unconditional-complete shape was copy-pasted across five workers.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.core.captioning import caption_refine_batch as crb
from app.core.tasks.task import TaskStatus
from app.core.tasks.task_manager import TaskManager


@pytest.fixture
def tm():
    m = TaskManager()
    m.set_loop(None)  # no loop in tests → broadcasts are no-ops
    return m


# ── the shared rule ───────────────────────────────────────────────────────


def test_no_failures_completes_with_no_error(tm):
    t = tm.create(type="caption_refine_batch", title="x", total=2)
    tm.finish_batch(t.id, ok=2, failed=0)
    assert t.status == TaskStatus.COMPLETED
    assert t.error is None


def test_every_item_failed_is_failed_not_completed(tm):
    """The defect itself: ok == 0 and failed > 0 must never read as success."""
    t = tm.create(type="caption_refine_batch", title="x", total=1)
    tm.finish_batch(t.id, ok=0, failed=1, error="ReadTimeout")
    assert t.status == TaskStatus.FAILED
    assert t.error is not None
    assert "1 of 1 items failed" in t.error
    assert "ReadTimeout" in t.error  # the reason reaches the record


def test_partial_failure_completes_but_still_states_the_reason(tm):
    """Real work landed, so the batch completed — but the failure is not
    silent: the summary rides on `error`, which the Task Center renders on
    presence rather than on status."""
    t = tm.create(type="caption_refine_batch", title="x", total=3)
    tm.finish_batch(t.id, ok=2, failed=1, error="boom")
    assert t.status == TaskStatus.COMPLETED
    assert t.error == "1 of 3 items failed (last: boom)"


def test_all_failed_without_a_reason_still_fails_with_a_count(tm):
    """A caller that has no exception text must still not produce a silent
    success; the count alone is a reason."""
    t = tm.create(type="caption_refine_batch", title="x", total=4)
    tm.finish_batch(t.id, ok=0, failed=4)
    assert t.status == TaskStatus.FAILED
    assert t.error == "4 of 4 items failed"


def test_unknown_task_id_is_a_noop(tm):
    tm.finish_batch("nope", ok=0, failed=1, error="x")  # must not raise


# ── the reported defect, end to end through the real worker ───────────────


@patch.object(crb, "dataset_manager")
@patch.object(crb.caption_refine, "refine_caption", new_callable=AsyncMock)
def test_refine_batch_reports_failed_when_every_item_raises(mock_refine, mock_dm, tmp_path, tm):
    """The live reproduction: one image, the LLM call times out. Before
    LANE-52 this task ended `completed / ok=0 / failed=1 / error=None`."""
    import httpx

    mock_refine.side_effect = httpx.ReadTimeout("")
    ds = MagicMock()
    ds.path = str(tmp_path)
    mock_dm.get_dataset.return_value = ds
    (tmp_path / "img1.txt").write_text("general caption", encoding="utf-8")

    task = tm.create(type="caption_refine_batch", title="Refine", total=1)
    with patch.object(crb, "task_manager", tm):
        crb.run_caption_refine_batch(
            task.id,
            dataset_name="ds",
            image_rel_paths=["img1.png"],
            definition_id="flux1-schnell",
            preset="standardize",
            model="qwen2.5:7b-instruct",
            base_url="http://test",
        )

    assert task.status == TaskStatus.FAILED
    assert task.ok == 0 and task.failed == 1
    # httpx.ReadTimeout("") stringifies to "" — the class name is the fallback,
    # so the record never says "failed for no reason".
    assert task.error is not None and "ReadTimeout" in task.error


@patch.object(crb, "dataset_manager")
@patch.object(crb.caption_refine, "refine_caption", new_callable=AsyncMock)
def test_refine_batch_still_completes_when_items_succeed(mock_refine, mock_dm, tmp_path, tm):
    """Positive control — without it a worker that failed everything would
    also pass the test above."""
    mock_refine.return_value = "refined cap"
    ds = MagicMock()
    ds.path = str(tmp_path)
    mock_dm.get_dataset.return_value = ds
    (tmp_path / "img1.txt").write_text("general caption", encoding="utf-8")

    task = tm.create(type="caption_refine_batch", title="Refine", total=1)
    with patch.object(crb, "task_manager", tm):
        crb.run_caption_refine_batch(
            task.id, dataset_name="ds", image_rel_paths=["img1.png"],
            definition_id="flux1-schnell", preset="standardize",
            model="m", base_url="http://test",
        )

    assert task.status == TaskStatus.COMPLETED
    assert task.ok == 1 and task.failed == 0 and task.error is None


# ── no adopter may go back to the unconditional epilogue ──────────────────


ADOPTERS = [
    "app/core/captioning/caption_batch.py",
    "app/core/captioning/caption_refine_batch.py",
    "app/core/image_processing/pipeline_batch.py",
    "app/core/masking/mask_generate_batch.py",
    "app/core/video/split_batch.py",
]


@pytest.mark.parametrize("rel", ADOPTERS)
def test_counted_batch_workers_do_not_complete_unconditionally(rel):
    """Structural pin. Each of these workers keeps an `ok`/`failed` tally, and
    each used to end in a bare `task_manager.complete(...)` that ignored it.
    A tallying worker must hand the tally to `finish_batch`; `complete()` on
    its own is exactly the shape that made an all-failed batch look done.

    Positive control: the file must contain the tally, so a worker renamed or
    emptied does not pass by having neither call.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]  # backend/ — never CWD
    src = (root / rel).read_text(encoding="utf-8")
    assert "failed += 1" in src, f"{rel}: expected a per-item failure tally"
    assert "task_manager.finish_batch(" in src, (
        f"{rel}: a counted batch must finalize through finish_batch"
    )
    assert "task_manager.complete(" not in src, (
        f"{rel}: bare complete() ignores the failed counter (LANE-52)"
    )
