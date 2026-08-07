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
import logging
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


def test_malformed_adapt_payload_is_surfaced_not_dropped(jm_job, caplog):
    """A non-dict payload can only mean the trainer's event contract changed —
    and one of these events carries the rebuild handoff. Warn; never swallow."""
    mgr, job, broadcasts = jm_job
    with caplog.at_level(logging.WARNING, logger="app.core.job_manager"):
        mgr._dispatch_log_entry(job.id, _adapt("not-a-dict"))
    assert job.logs == []
    assert not broadcasts
    assert "adapt_entry_malformed" in caplog.text


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


def test_second_rebuild_in_the_same_job_relaunches_from_the_new_checkpoint(
    jm_job, tmp_path,
):
    """rebuild → relaunch → rebuild again, on the SAME job record.

    This is the path where a leaked pending entry would actually bite: the
    first handoff must be fully consumed before the second arrives, and the
    second must resume from ITS checkpoint, not the first one's.
    """
    mgr, job, _ = jm_job
    run_dir, first = _run_with_checkpoint(mgr, tmp_path)
    second = os.path.join(run_dir, "checkpoint-000200")
    os.makedirs(second)
    with open(os.path.join(second, "training_state.json"), "w", encoding="utf-8") as fh:
        fh.write("{}")

    relaunches: list[str] = []

    def fake_restart(job_id, fresh):
        relaunches.append(job_id)
        job.status = JobStatus.RUNNING        # a real relaunch re-enters RUNNING

    with patch.object(mgr, "_get_job_output_dir", return_value=run_dir), \
            patch.object(mgr, "restart_job", side_effect=fake_restart):
        mgr._dispatch_log_entry(job.id, _adapt(_REBUILD))
        mgr._dispatch_log_entry(job.id, _exit(0))
        assert _wait_until(lambda: len(relaunches) == 1)
        assert job.config["resume_from_checkpoint"] == os.path.abspath(first)

        mgr._dispatch_log_entry(job.id, _adapt(
            {**_REBUILD, "checkpoint_dir": "checkpoint-000200",
             "keep_patterns": [r"^blocks\.1\.to_k$"], "rebuild_count": 2},
        ))
        mgr._dispatch_log_entry(job.id, _exit(0))
        assert _wait_until(lambda: len(relaunches) == 2)

    assert job.config["resume_from_checkpoint"] == os.path.abspath(second)
    assert job.config["targeted_layers"] == [r"^blocks\.1\.to_k$"]
    assert job.id not in mgr._pending_rebuilds
    assert mgr._rebuild_restarts[job.id] == 2


def test_rebuild_relaunches_are_capped_per_session(jm_job, tmp_path):
    """A backend-side budget, independent of the controller's own cap.

    The controller's counter lives in the trainer and is NOT consulted when
    the backend is replaying its own stale log lines — the runaway this bounds
    emits nothing new. Past the cap the job must FAIL with the reason, not
    relaunch again.
    """
    mgr, job, _ = jm_job
    run_dir, _ckpt = _run_with_checkpoint(mgr, tmp_path)
    relaunches: list[str] = []

    def fake_restart(job_id, fresh):
        relaunches.append(job_id)
        job.status = JobStatus.RUNNING

    with patch.object(mgr, "_get_job_output_dir", return_value=run_dir), \
            patch.object(mgr, "restart_job", side_effect=fake_restart):
        for i in range(mgr._MAX_REBUILD_RESTARTS):
            mgr._dispatch_log_entry(job.id, _adapt(_REBUILD))
            mgr._dispatch_log_entry(job.id, _exit(0))
            assert _wait_until(lambda n=i: len(relaunches) == n + 1), \
                f"relaunch {i + 1} within the cap never happened"
        # One more than the budget allows.
        mgr._dispatch_log_entry(job.id, _adapt(_REBUILD))
        mgr._dispatch_log_entry(job.id, _exit(0))

    assert len(relaunches) == mgr._MAX_REBUILD_RESTARTS   # no further relaunch
    assert job.status == JobStatus.FAILED
    assert "cap" in (job.error or "").lower()
    assert job.id not in mgr._pending_rebuilds


def test_rebuild_budget_is_cleared_by_a_fresh_restart(jm_job, tmp_path):
    """A restart-from-zero is a new session — it must not inherit a spent
    budget (mirrors how ``_auto_resume_state`` is managed)."""
    mgr, job, _ = jm_job
    mgr._rebuild_restarts[job.id] = mgr._MAX_REBUILD_RESTARTS
    job.status = JobStatus.FAILED
    with patch.object(mgr, "_get_job_output_dir", return_value=str(tmp_path / "run")), \
            patch.object(mgr, "_reconcile_active_jobs"), \
            patch.object(mgr, "start_job"):
        mgr.restart_job(job.id, fresh=True)
    assert job.id not in mgr._rebuild_restarts


def test_relaunch_failure_before_launch_fails_the_job_loudly(jm_job, tmp_path):
    """``restart_job`` raising BEFORE ``start_job`` used to leave the record
    STOPPED with no error and no broadcast — a run that vanished for no stated
    reason. It must end FAILED, with the reason, and broadcast."""
    mgr, job, broadcasts = jm_job
    run_dir, _ckpt = _run_with_checkpoint(mgr, tmp_path)
    with patch.object(mgr, "_get_job_output_dir", return_value=run_dir), \
            patch.object(mgr, "restart_job", side_effect=RuntimeError("no launcher")):
        mgr._restart_for_rebuild(job.id, _REBUILD)
        assert _wait_until(lambda: job.status == JobStatus.FAILED)
    assert "no launcher" in (job.error or "")
    assert any(topic == "job_update" for topic, _ in broadcasts)


def test_relaunch_stands_down_when_the_user_intervened(jm_job, tmp_path):
    """The sibling ``_schedule_auto_resume._fire`` re-checks before firing;
    match it — a job someone else took over is no longer ours to relaunch."""
    mgr, job, _ = jm_job
    with patch.object(mgr, "restart_job") as restart:
        job.status = JobStatus.RUNNING     # e.g. a manual relaunch got there first
        mgr._run_rebuild_restart(job.id)
    restart.assert_not_called()
    assert job.status == JobStatus.RUNNING


# ── the user's own targeted_layers survive a rebuild ─────────────────────


_USER_PATTERNS = [r"^blocks\.\d+\.to_q$"]


def _rebuild_once(mgr, job, run_dir, data=None):
    """Drive one narrowing rebuild through ``_restart_for_rebuild``."""
    with patch.object(mgr, "_get_job_output_dir", return_value=run_dir), \
            patch.object(mgr, "restart_job"):
        mgr._restart_for_rebuild(job.id, data or _REBUILD)


def _fresh_restart(mgr, job, run_dir):
    job.status = JobStatus.FAILED
    with patch.object(mgr, "_get_job_output_dir", return_value=run_dir), \
            patch.object(mgr, "_reconcile_active_jobs"), \
            patch.object(mgr, "_persist_config") as persist, \
            patch.object(mgr, "start_job"):
        mgr.restart_job(job.id, fresh=True)
    return persist


def test_fresh_restart_restores_the_users_targeted_layers(jm_job, tmp_path):
    """A rebuild overwrites ``targeted_layers`` on the PERSISTED record. The
    narrowed set is this run's machine-derived state, not user intent — a
    restart-from-zero that inherits it would constrain a full run to the
    previous run's final keep-set forever, and the original selection would be
    unrecoverable from the record."""
    mgr, job, _ = jm_job
    job.config["targeted_layers"] = list(_USER_PATTERNS)
    run_dir, _ckpt = _run_with_checkpoint(mgr, tmp_path)

    _rebuild_once(mgr, job, run_dir)
    assert job.config["targeted_layers"] == [r"^blocks\.0\.to_q$"]  # narrowed

    persist = _fresh_restart(mgr, job, run_dir)
    assert job.config["targeted_layers"] == _USER_PATTERNS
    assert "pre_adaptive_targeted_layers" not in job.config
    assert persist.call_args[0][1]["targeted_layers"] == _USER_PATTERNS


def test_fresh_restart_after_a_rebuild_of_an_untargeted_run_clears_the_field(
    jm_job, tmp_path,
):
    """The user selected nothing, so "restore the original" means the field is
    GONE — leaving the narrowed list would silently target a run the user asked
    to train whole."""
    mgr, job, _ = jm_job
    assert "targeted_layers" not in job.config
    run_dir, _ckpt = _run_with_checkpoint(mgr, tmp_path)

    _rebuild_once(mgr, job, run_dir)
    _fresh_restart(mgr, job, run_dir)
    assert "targeted_layers" not in job.config
    assert "pre_adaptive_targeted_layers" not in job.config


def test_only_the_first_rebuild_stashes_the_original_selection(jm_job, tmp_path):
    """Segment two must not stash segment one's already-narrowed set — that is
    how a multi-rebuild run loses the original just as thoroughly, one layer of
    indirection later."""
    mgr, job, _ = jm_job
    job.config["targeted_layers"] = list(_USER_PATTERNS)
    run_dir, _ckpt = _run_with_checkpoint(mgr, tmp_path)
    second = os.path.join(run_dir, "checkpoint-000200")
    os.makedirs(second)
    with open(os.path.join(second, "training_state.json"), "w", encoding="utf-8") as fh:
        fh.write("{}")

    _rebuild_once(mgr, job, run_dir)
    _rebuild_once(mgr, job, run_dir, {
        **_REBUILD, "checkpoint_dir": "checkpoint-000200",
        "keep_patterns": [r"^blocks\.1\.to_k$"], "rebuild_count": 2,
    })
    assert job.config["targeted_layers"] == [r"^blocks\.1\.to_k$"]

    _fresh_restart(mgr, job, run_dir)
    assert job.config["targeted_layers"] == _USER_PATTERNS


# ── pending-rebuild entries never outlive their run ──────────────────────


def _kill_via_watchdog(mgr, job):
    """Drive the real PID watchdog through an exit-message-less death.

    ``time.sleep`` is neutralized so the watchdog's poll cadence does not make
    the test wait; everything else — the thread, the finalize block, the status
    transition — is the production path.
    """
    with patch("app.core.job_manager.time.sleep"), \
            patch.object(mgr, "_resolve_worker_pid", return_value=4242), \
            patch.object(mgr, "_trainer_tree_dead", return_value=True), \
            patch.object(mgr, "_terminate_process_tree", return_value=[]):
        mgr._start_pid_watchdog(job.id, 4242)
        assert _wait_until(lambda: job.status == JobStatus.STOPPED), \
            "the watchdog never finalized the job"


def test_hard_killed_trainer_does_not_hijack_the_next_run(jm_job, tmp_path):
    """The trainer announces a rebuild and then dies WITHOUT an exit line (TDR,
    hard kill) — the window between the two covers VRAM-peak capture, loss
    history and teardown. The watchdog finalizes the job, so nothing pops the
    pending entry, and the NEXT run of the same job would be relaunched from a
    stale checkpoint the moment it exits cleanly."""
    mgr, job, _ = jm_job
    job.pid = 4242
    mgr._dispatch_log_entry(job.id, _adapt(_REBUILD))
    assert job.id in mgr._pending_rebuilds

    _kill_via_watchdog(mgr, job)
    assert job.id not in mgr._pending_rebuilds

    # The user restarts; that run finishes for real.
    with patch.object(mgr, "_restart_for_rebuild") as restart:
        job.status = JobStatus.RUNNING
        mgr._dispatch_log_entry(job.id, _exit(0))
    restart.assert_not_called()
    assert job.status == JobStatus.COMPLETED


def test_stop_clears_a_pending_rebuild(jm_job):
    """A deliberate Stop between the announcement and the exit line ends the
    run — the exit handler's STOPPED guard suppresses THIS relaunch, but the
    entry itself must not survive to meet a later clean exit."""
    mgr, job, _ = jm_job
    job.pid = 4242
    mgr._dispatch_log_entry(job.id, _adapt(_REBUILD))
    with patch.object(mgr, "_terminate_process_tree", return_value=[]):
        mgr.stop_job(job.id)
    assert job.id not in mgr._pending_rebuilds


def test_phantom_reconcile_clears_a_pending_rebuild(jm_job):
    """The queue's self-heal is a fourth way a run ends with no exit line: it
    demotes a RUNNING job whose process is gone straight to FAILED."""
    mgr, job, _ = jm_job
    job.pid = 4242
    mgr._dispatch_log_entry(job.id, _adapt(_REBUILD))
    with patch.object(mgr, "_is_trainer_process", return_value=False):
        assert mgr._reconcile_active_jobs() == [job.id]
    assert job.id not in mgr._pending_rebuilds


def test_restart_clears_a_pending_rebuild(jm_job, tmp_path):
    """A manual relaunch of a job whose rebuild handoff was never consumed
    starts a NEW run; the old announcement is not part of it."""
    mgr, job, _ = jm_job
    mgr._dispatch_log_entry(job.id, _adapt(_REBUILD))
    job.status = JobStatus.FAILED
    with patch.object(mgr, "_get_job_output_dir", return_value=str(tmp_path / "run")), \
            patch.object(mgr, "_reconcile_active_jobs"), \
            patch.object(mgr, "start_job"):
        mgr.restart_job(job.id, fresh=False)
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

    The rename-while-open failure is Windows-specific, so on a POSIX runner
    this test passes for a weaker reason (the rotation would have succeeded
    either way). It still pins the ordering everywhere; the relaunch cap in
    ``test_rebuild_relaunches_are_capped_per_session`` is the platform-neutral
    bound on the same runaway.
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
