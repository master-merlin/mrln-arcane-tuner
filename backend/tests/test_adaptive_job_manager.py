"""adapt-event routing + rebuild restart orchestration (spec §6).

The trainer talks to the backend by appending JSON lines to ``job_log.jsonl``;
``LogTailer`` reads them and ``JobManager._dispatch_log_entry`` routes them.
These tests pin the ``adapt`` branch and the rebuild handoff it feeds:

**The trainer's rebuild exit is an ordinary ``exit(0)``** — the pending
``rebuild_request`` is the ONLY thing that tells a rebuild apart from a
genuine completion, so both directions need proof (a rebuild must relaunch,
an ordinary completion must NOT be swallowed).

Fixture style mirrors ``test_job_manager.py``: a real ``JobManager`` with a
mock event loop and the DB/queue side effects patched out.
"""

import asyncio
import json
import os
import re
import time

import pytest
from unittest.mock import MagicMock, patch

from app.core.job import JobStatus
from app.core.job_manager import JobManager


def _make_config(**overrides) -> dict:
    """Minimal training config (same shape as test_job_manager.py's helper)."""
    defaults = {
        "output_dir": "outputs",
        "lora_name": "test_lora",
        "definition_id": "flux/dev",
    }
    defaults.update(overrides)
    return defaults


@pytest.fixture
def jm_job():
    """``(manager, job, broadcasts)`` for a RUNNING job.

    Broadcasts are captured rather than awaited (the dispatcher hands them to
    the loop from the tailer thread), and the DB / queue side effects are
    patched out so the assertions are about the orchestration only.

    ``_stop_tailer`` is deliberately NOT stubbed: it is the seam that releases
    the log file before the restart rotates it, and stubbing it hides exactly
    the failure ``test_rebuild_restart_rotates_the_log_and_relaunches_once``
    exists to catch. With no tailer registered it is a no-op anyway.
    """
    mgr = JobManager()
    broadcasts: list[tuple[str, dict]] = []
    with patch("app.core.job_manager.event_manager") as mock_em, \
            patch("app.core.job_manager.asyncio.run_coroutine_threadsafe"), \
            patch.object(mgr, "_persist_status"), \
            patch.object(mgr, "_persist_config"), \
            patch.object(mgr, "schedule_advance_queue"):
        mock_em.broadcast.side_effect = (
            lambda topic, payload: broadcasts.append((topic, payload))
        )
        mgr.set_loop(MagicMock(spec=asyncio.AbstractEventLoop))
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING
        yield mgr, job, broadcasts


def _adapt(data: dict, t: float = 1.0) -> dict:
    return {"type": "adapt", "t": t, "data": data}


def _exit(code: int, error: str | None = None, t: float = 2.0) -> dict:
    payload: dict = {"code": code}
    if error is not None:
        payload["error"] = error
    return {"type": "exit", "t": t, "data": payload}


_NARROW = {
    "step": 100, "event_index": 0, "kind": "narrow", "active_count": 5,
    "total_count": 8, "hot_count": 3, "frozen_this_event": 3,
    "reactivated_this_event": 0, "active_param_pct": 62.5,
    "earliest_active_block": 2, "top_modules": [],
}

_REBUILD = {
    "step": 100, "event_index": 1, "kind": "rebuild_request", "active_count": 1,
    "total_count": 8, "hot_count": 1, "frozen_this_event": 0,
    "reactivated_this_event": 0, "active_param_pct": 12.5,
    "earliest_active_block": 0, "top_modules": [],
    "checkpoint_dir": "checkpoint-000100",
    "keep_patterns": [r"^blocks\.0\.to_q$"],
    "rebuild_count": 1,
}


# ── adapt dispatch ───────────────────────────────────────────────────────


def test_adapt_event_appends_log_and_broadcasts(jm_job):
    """The frozen wire format the Jobs screen parses: a ``job_log`` message
    whose body is ``{"adapt": <event>}``, mirrored into ``job.logs`` so a
    reconnect re-hydrates the events it missed."""
    mgr, job, broadcasts = jm_job
    mgr._dispatch_log_entry(job.id, _adapt(_NARROW))

    expected = json.dumps({"adapt": _NARROW})
    assert job.logs == [expected]
    assert ("job_log", {
        "job_id": job.id, "message": expected, "timestamp": 1.0,
    }) in broadcasts


def test_adapt_log_buffer_is_bounded(jm_job):
    """An unbounded buffer is a real leak on a long run — the adapt branch
    caps ``job.logs`` exactly like the neighbouring ``log`` branch."""
    mgr, job, _ = jm_job
    job.logs = [f"line-{i}" for i in range(1000)]
    mgr._dispatch_log_entry(job.id, _adapt(_NARROW))
    assert len(job.logs) == 1000
    assert job.logs[0] == "line-1"   # oldest dropped, not the newest


def test_non_rebuild_adapt_event_records_no_pending(jm_job):
    """Only ``rebuild_request`` arms the interception — a narrow/probe event
    must never make the next clean exit relaunch the job."""
    mgr, job, _ = jm_job
    mgr._dispatch_log_entry(job.id, _adapt(_NARROW))
    assert job.id not in mgr._pending_rebuilds


# ── rebuild handoff ──────────────────────────────────────────────────────


def test_rebuild_request_sets_pending_and_exit_zero_restarts(jm_job):
    mgr, job, _ = jm_job
    calls: list[tuple] = []
    with patch.object(mgr, "_restart_for_rebuild",
                      side_effect=lambda jid, d: calls.append((jid, d))):
        mgr._dispatch_log_entry(job.id, _adapt(_REBUILD))
        assert mgr._pending_rebuilds[job.id] == _REBUILD
        mgr._dispatch_log_entry(job.id, _exit(0))

    assert calls == [(job.id, _REBUILD)]
    assert job.status != JobStatus.COMPLETED     # a rebuild is not a finished run
    assert job.id not in mgr._pending_rebuilds   # consumed, never left behind


def test_exit_without_pending_completes_normally(jm_job):
    """Negative proof: the rebuild interception must not swallow ordinary
    completions."""
    mgr, job, _ = jm_job
    with patch.object(mgr, "_restart_for_rebuild") as restart:
        mgr._dispatch_log_entry(job.id, _exit(0))
    restart.assert_not_called()
    assert job.status == JobStatus.COMPLETED


def test_exit_nonzero_with_pending_fails_normally(jm_job):
    """A crash AFTER the rebuild announcement is an ordinary failure — and the
    pending entry must not survive to hijack a later run of the same job."""
    mgr, job, _ = jm_job
    with patch.object(mgr, "_restart_for_rebuild") as restart:
        mgr._dispatch_log_entry(job.id, _adapt(_REBUILD))
        mgr._dispatch_log_entry(job.id, _exit(1, "boom"))
    restart.assert_not_called()
    assert job.status == JobStatus.FAILED
    assert job.error == "boom"
    assert job.id not in mgr._pending_rebuilds


def test_user_stop_wins_over_pending_rebuild(jm_job):
    """A deliberate user stop beats a rebuild the trainer announced first —
    otherwise Stop would silently relaunch the run.

    Only the RELAUNCH is this task's to suppress; how the pre-existing exit
    handler labels a stopped job's clean exit is left exactly as it was.
    """
    mgr, job, _ = jm_job
    with patch.object(mgr, "_restart_for_rebuild") as restart:
        mgr._dispatch_log_entry(job.id, _adapt(_REBUILD))
        job.status = JobStatus.STOPPED          # stop_job ran before the exit line
        mgr._dispatch_log_entry(job.id, _exit(0))
    restart.assert_not_called()
    assert job.finished_at is not None          # ended, not relaunched
    assert job.id not in mgr._pending_rebuilds


# ── restart ──────────────────────────────────────────────────────────────


def _run_with_checkpoint(mgr, tmp_path, name="checkpoint-000100"):
    run = tmp_path / "run"
    ckpt = run / name
    ckpt.mkdir(parents=True)
    (ckpt / "training_state.json").write_text("{}", encoding="utf-8")
    return str(run), str(ckpt)


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    """Poll *predicate* until true or *timeout*.

    The relaunch is deliberately handed to a worker thread (it must not run on
    the tailer's dispatch thread — see
    ``test_rebuild_restart_rotates_the_log_and_relaunches_once``), so tests
    wait for it instead of asserting synchronously.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_restart_for_rebuild_updates_config_and_relaunches(jm_job, tmp_path):
    mgr, job, _ = jm_job
    run_dir, ckpt = _run_with_checkpoint(mgr, tmp_path)
    relaunched: list[tuple] = []
    with patch.object(mgr, "_get_job_output_dir", return_value=run_dir), \
            patch.object(mgr, "restart_job",
                         side_effect=lambda jid, fresh: relaunched.append((jid, fresh))):
        mgr._restart_for_rebuild(job.id, _REBUILD)
        assert _wait_until(lambda: bool(relaunched))

    assert job.config["targeted_layers"] == [r"^blocks\.0\.to_q$"]
    # The RESOLVED absolute path, not the basename that arrived over the log.
    assert job.config["resume_from_checkpoint"] == os.path.abspath(ckpt)
    assert job.config["use_cached_latents"] is True
    assert job.config["use_cached_embeddings"] is True
    assert relaunched == [(job.id, False)]
    assert job.status != JobStatus.COMPLETED


def test_restart_for_rebuild_persists_the_narrowed_config(jm_job, tmp_path):
    """The relaunched process reads its config from the DB — an in-memory-only
    update would restart the run with the OLD (wide) targeted_layers."""
    mgr, job, _ = jm_job
    run_dir, ckpt = _run_with_checkpoint(mgr, tmp_path)
    with patch.object(mgr, "_get_job_output_dir", return_value=run_dir), \
            patch.object(mgr, "restart_job"), \
            patch.object(mgr, "_persist_config") as persist:
        mgr._restart_for_rebuild(job.id, _REBUILD)
    persist.assert_called_once()
    persisted = persist.call_args[0][1]
    assert persisted["targeted_layers"] == [r"^blocks\.0\.to_q$"]
    assert persisted["resume_from_checkpoint"] == os.path.abspath(ckpt)


@pytest.mark.parametrize("bad_dir", [
    "..\\..\\evil",                 # windows traversal
    "../../evil",                   # posix traversal
    "checkpoint-000100/../../evil",  # traversal behind a valid-looking prefix
    "/etc/passwd",                  # absolute path
    "",                             # empty
])
def test_restart_for_rebuild_rejects_escaping_checkpoint_dir(jm_job, tmp_path, bad_dir):
    """``checkpoint_dir`` arrives in a log file — it must never escape the run
    dir, and a rejected value must FAIL the job (never fall through to a
    silent 'completed')."""
    mgr, job, _ = jm_job
    run_dir, _ckpt = _run_with_checkpoint(mgr, tmp_path)
    with patch.object(mgr, "_get_job_output_dir", return_value=run_dir), \
            patch.object(mgr, "restart_job") as restart:
        mgr._restart_for_rebuild(job.id, {**_REBUILD, "checkpoint_dir": bad_dir})
    restart.assert_not_called()
    assert job.status == JobStatus.FAILED
    assert "rebuild" in (job.error or "").lower()


def test_restart_for_rebuild_rejects_sibling_dir_sharing_the_run_prefix(jm_job, tmp_path):
    """Containment is resolve + commonpath, never ``startswith``: a sibling run
    whose name merely EXTENDS this run's (``…/run2`` vs ``…/run``) is OUTSIDE,
    yet passes a prefix test.

    The anchored name regex would reject this value first, so it is relaxed
    here on purpose — otherwise the containment guard behind it is never
    exercised and could silently be a prefix test. The sibling checkpoint is
    fully resumable, so only containment can reject it.
    """
    mgr, job, _ = jm_job
    run_dir, _ckpt = _run_with_checkpoint(mgr, tmp_path)
    sibling = tmp_path / "run2" / "checkpoint-000100"
    sibling.mkdir(parents=True)
    (sibling / "training_state.json").write_text("{}", encoding="utf-8")
    escaping = os.path.join("..", "run2", "checkpoint-000100")
    with patch.object(mgr, "_get_job_output_dir", return_value=run_dir), \
            patch("app.core.job_manager._RESUMABLE_DIR_RE", re.compile(r"^.*$")), \
            patch.object(mgr, "restart_job") as restart:
        mgr._restart_for_rebuild(job.id, {**_REBUILD, "checkpoint_dir": escaping})
    restart.assert_not_called()
    assert job.status == JobStatus.FAILED


def test_restart_for_rebuild_rejects_unresumable_checkpoint(jm_job, tmp_path):
    mgr, job, _ = jm_job
    run = tmp_path / "run"
    (run / "checkpoint-000100").mkdir(parents=True)   # no training_state.json
    with patch.object(mgr, "_get_job_output_dir", return_value=str(run)), \
            patch.object(mgr, "restart_job") as restart:
        mgr._restart_for_rebuild(job.id, _REBUILD)
    restart.assert_not_called()
    assert job.status == JobStatus.FAILED


def test_restart_for_rebuild_rejects_empty_keep_patterns(jm_job, tmp_path):
    """An empty ``targeted_layers`` means "train EVERYTHING" downstream — the
    exact inverse of a narrowing rebuild. Reject rather than relaunch wide."""
    mgr, job, _ = jm_job
    run_dir, _ckpt = _run_with_checkpoint(mgr, tmp_path)
    with patch.object(mgr, "_get_job_output_dir", return_value=run_dir), \
            patch.object(mgr, "restart_job") as restart:
        mgr._restart_for_rebuild(job.id, {**_REBUILD, "keep_patterns": []})
    restart.assert_not_called()
    assert job.status == JobStatus.FAILED


def test_rejected_rebuild_leaves_no_pending_entry(jm_job, tmp_path):
    """End-to-end through the dispatcher: a rejected handoff fails the job and
    leaves nothing behind that a later run could trip over."""
    mgr, job, _ = jm_job
    run_dir, _ckpt = _run_with_checkpoint(mgr, tmp_path)
    with patch.object(mgr, "_get_job_output_dir", return_value=run_dir), \
            patch.object(mgr, "restart_job") as restart:
        mgr._dispatch_log_entry(
            job.id, _adapt({**_REBUILD, "checkpoint_dir": "..\\..\\evil"}),
        )
        mgr._dispatch_log_entry(job.id, _exit(0))
    restart.assert_not_called()
    assert job.status == JobStatus.FAILED
    assert job.id not in mgr._pending_rebuilds


def test_rebuild_restart_rotates_the_log_and_relaunches_once(jm_job, tmp_path):
    """End-to-end over a REAL ``LogTailer``: the relaunch must happen only
    after the tailer has released ``job_log.jsonl``.

    The tailer holds that file open for its whole polling loop, and
    ``restart_job`` rotates it — a rename Windows refuses while it is open.
    ``_reset_job_log_state`` swallows that failure, so a relaunch driven from
    the tailer's own dispatch thread leaves the previous run's
    ``rebuild_request`` + ``exit(0)`` lines in place for the next tailer to
    replay: the job would relaunch forever. Asserted on the bytes on disk (the
    old log is gone, a rotated copy exists, the offset was reset) plus exactly
    ONE launch.
    """
    from app.core.log_tailer import LogTailer, LOG_FILENAME

    mgr, job, _ = jm_job
    run_dir, ckpt = _run_with_checkpoint(mgr, tmp_path)
    log_path = os.path.join(run_dir, LOG_FILENAME)
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(_adapt(_REBUILD)) + "\n")
        fh.write(json.dumps(_exit(0)) + "\n")

    launched: list[str] = []
    tailer = LogTailer(job.id, log_path, mgr._dispatch_log_entry, poll_interval=0.02)
    mgr._tailers[job.id] = tailer
    try:
        with patch.object(mgr, "_get_job_output_dir", return_value=run_dir), \
                patch.object(mgr, "start_job", side_effect=launched.append):
            tailer.start()
            assert _wait_until(lambda: bool(launched)), "restart never happened"
            # Give a (buggy) replay loop room to fire a second launch.
            time.sleep(0.3)
        # Asserted BEFORE the cleanup stop() below: a second stop() force-saves
        # the offset again and would recreate the file this checks is gone.
        assert launched == [job.id]                       # exactly one relaunch
        assert not os.path.exists(log_path)               # rotated, not left behind
        assert [p for p in os.listdir(run_dir) if p.startswith("job_log.")]
        assert not os.path.exists(log_path + ".offset")   # next tailer starts at 0
        assert job.config["resume_from_checkpoint"] == os.path.abspath(ckpt)
    finally:
        tailer.stop()
