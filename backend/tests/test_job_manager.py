"""
Tests for the JobManager lifecycle orchestrator.

Covers: CRUD operations, status transitions, signal-based pause/resume/stop,
output directory resolution, event broadcasting, and error handling.
"""
import os
import asyncio
import signal

import pytest
from unittest.mock import MagicMock, patch

from app.core.job import Job, JobStatus
from app.core.job_manager import JobManager


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_config(**overrides) -> dict:
    """Build a minimal training config."""
    defaults = {
        "output_dir": "outputs",
        "lora_name": "test_lora",
        "definition_id": "flux/dev",
    }
    defaults.update(overrides)
    return defaults


# ── CRUD Operations ──────────────────────────────────────────────────────


class TestJobCRUD:
    """Tests for create, list, get, delete operations."""

    def test_create_job_returns_pending_job(self):
        """create_job should return a Job with PENDING status."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())

        assert isinstance(job, Job)
        assert job.status == JobStatus.PENDING
        assert job.plugin_id == "flux/dev"
        # create_job injects job_id into the config dict for downstream use
        assert job.config["job_id"] == job.id
        expected_config = _make_config()
        expected_config["job_id"] = job.id
        assert job.config == expected_config

    def test_create_job_registers_in_jobs_dict(self):
        """Created job should be retrievable via get_job."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())

        assert mgr.get_job(job.id) is job

    def test_list_jobs_returns_newest_first(self):
        """list_jobs should return jobs sorted by creation time, newest first."""
        mgr = JobManager()
        j1 = mgr.create_job("flux/dev", _make_config())
        j1.created_at = 1000.0
        j2 = mgr.create_job("sdxl/base", _make_config())
        j2.created_at = 2000.0

        listed = mgr.list_jobs()
        assert listed[0].id == j2.id
        assert listed[1].id == j1.id

    def test_list_jobs_empty(self):
        """list_jobs on empty manager returns empty list."""
        mgr = JobManager()
        assert mgr.list_jobs() == []

    def test_get_job_not_found(self):
        """get_job returns None for unknown ID."""
        mgr = JobManager()
        assert mgr.get_job("nonexistent") is None

    def test_delete_job_removes_entry(self):
        """delete_job should remove job from registry."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        mgr.delete_job(job.id)

        assert mgr.get_job(job.id) is None

    def test_delete_nonexistent_job_is_noop(self):
        """delete_job for unknown ID should not raise."""
        mgr = JobManager()
        mgr.delete_job("nonexistent")  # Should not raise


# ── Set Loop ─────────────────────────────────────────────────────────────


class TestSetLoop:
    """Tests for event loop registration."""

    def test_set_loop_stores_loop(self):
        """set_loop should store the event loop reference."""
        mgr = JobManager()
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        mgr.set_loop(mock_loop)

        assert mgr._loop is mock_loop


# ── Start Job ────────────────────────────────────────────────────────────


class TestStartJob:
    """Tests for start_job precondition checks."""

    def test_start_job_not_found_raises(self):
        """start_job for unknown ID should raise ValueError."""
        mgr = JobManager()
        with pytest.raises(ValueError, match="Job not found"):
            mgr.start_job("nonexistent")

    def test_start_already_running_raises(self):
        """start_job on a RUNNING job should raise ValueError."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING

        with pytest.raises(ValueError, match="already running"):
            mgr.start_job(job.id)

    @patch("app.core.job_manager.plugin_manager")
    def test_start_job_missing_plugin_raises(self, mock_pm):
        """start_job with unregistered plugin should raise ValueError."""
        mock_pm.get_plugin.return_value = None
        mgr = JobManager()
        job = mgr.create_job("unknown_plugin", _make_config())

        with pytest.raises(ValueError, match="not found"):
            mgr.start_job(job.id)


# ── Stop Job ─────────────────────────────────────────────────────────────


class TestStopJob:
    """Tests for stop_job."""

    def test_stop_not_found_raises(self):
        """stop_job for unknown ID should raise ValueError."""
        mgr = JobManager()
        with pytest.raises(ValueError, match="Job not found"):
            mgr.stop_job("nonexistent")

    @patch("app.core.job_manager.os.kill")
    def test_stop_running_job_sends_sigterm(self, mock_kill):
        """stop_job on RUNNING job should call os.kill with SIGTERM."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING
        job.pid = 12345

        mgr.stop_job(job.id)

        mock_kill.assert_called_once_with(12345, signal.SIGTERM)
        assert job.status == JobStatus.STOPPED
        assert job.finished_at is not None

    @patch("app.core.job_manager.os.kill", side_effect=ProcessLookupError)
    def test_stop_missing_process_sets_failed(self, mock_kill):
        """stop_job when process already exited should set FAILED."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING
        job.pid = 99999

        mgr.stop_job(job.id)

        assert job.status == JobStatus.FAILED
        assert job.error == "Process not found"

    def test_stop_non_running_job_is_noop(self):
        """stop_job on PENDING job without PID should not crash."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        # job.status is PENDING, job.pid is None — no kill should be attempted
        mgr.stop_job(job.id)
        # Status should remain PENDING since the condition wasn't met
        assert job.status == JobStatus.PENDING

    @patch("app.core.job_manager.os.kill")
    def test_stop_paused_job_sends_sigterm(self, mock_kill):
        """stop_job on PAUSED job should call os.kill with SIGTERM."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.PAUSED
        job.pid = 12345

        mgr.stop_job(job.id)

        mock_kill.assert_called_once_with(12345, signal.SIGTERM)
        assert job.status == JobStatus.STOPPED
        assert job.finished_at is not None


# ── Output Directory Resolution ──────────────────────────────────────────


class TestOutputDirResolution:
    """Tests for _get_job_output_dir."""

    def test_basic_output_dir(self):
        """Standard config should produce {output_dir}/{lora_name}_{model}."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())

        result = mgr._get_job_output_dir(job)
        assert result == os.path.join("outputs", "test_lora_dev")

    def test_output_dir_with_colons_in_definition(self):
        """Colons in definition_id should be replaced with underscores."""
        mgr = JobManager()
        job = mgr.create_job("sdxl/base", _make_config(definition_id="sdxl/base:v1.0"))

        result = mgr._get_job_output_dir(job)
        assert ":" not in result
        assert "base_v1.0" in result

    def test_output_dir_default_values(self):
        """Missing config keys should fall back to defaults."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", {})

        result = mgr._get_job_output_dir(job)
        assert result == os.path.join("outputs", "untitled_")


# ── Pause / Resume ───────────────────────────────────────────────────────


class TestPauseResume:
    """Tests for pause_job and resume_job signal dispatching."""

    @patch("app.engine.components.signal_manager.TrainingSignalManager")
    def test_pause_sends_signal_and_updates_status(self, mock_signal_cls):
        """pause_job should send 'pause' signal and set PAUSED status."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING

        mgr.pause_job(job.id)

        mock_signal_cls.send_signal.assert_called_once()
        args = mock_signal_cls.send_signal.call_args
        assert args[0][1] == "pause"
        assert job.status == JobStatus.PAUSED
        assert job.paused_at is not None

    def test_pause_non_running_raises(self):
        """pause_job on non-RUNNING job should raise ValueError."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        # PENDING status
        with pytest.raises(ValueError, match="Cannot pause"):
            mgr.pause_job(job.id)

    @patch("app.engine.components.signal_manager.TrainingSignalManager")
    def test_resume_sends_signal_and_updates_status(self, mock_signal_cls):
        """resume_job should send 'resume' signal and set RUNNING status."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.PAUSED

        mgr.resume_job(job.id)

        mock_signal_cls.send_signal.assert_called_once()
        args = mock_signal_cls.send_signal.call_args
        assert args[0][1] == "resume"
        assert job.status == JobStatus.RUNNING
        assert job.paused_at is None

    def test_resume_non_paused_raises(self):
        """resume_job on non-PAUSED job should raise ValueError."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        with pytest.raises(ValueError, match="Cannot resume"):
            mgr.resume_job(job.id)

    def test_pause_not_found_raises(self):
        """pause_job for unknown ID should raise ValueError."""
        mgr = JobManager()
        with pytest.raises(ValueError, match="Job not found"):
            mgr.pause_job("nonexistent")

    def test_resume_not_found_raises(self):
        """resume_job for unknown ID should raise ValueError."""
        mgr = JobManager()
        with pytest.raises(ValueError, match="Job not found"):
            mgr.resume_job("nonexistent")


# ── Soft Stop ────────────────────────────────────────────────────────────


class TestSoftStop:
    """Tests for soft_stop_job."""

    @patch("app.engine.components.signal_manager.TrainingSignalManager")
    def test_soft_stop_sends_signal(self, mock_signal_cls):
        """soft_stop_job should send 'soft_stop' signal."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING

        mgr.soft_stop_job(job.id)

        mock_signal_cls.send_signal.assert_called_once()
        args = mock_signal_cls.send_signal.call_args
        assert args[0][1] == "soft_stop"

    def test_soft_stop_pending_raises(self):
        """soft_stop_job on PENDING job should raise ValueError."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        with pytest.raises(ValueError, match="Cannot soft stop"):
            mgr.soft_stop_job(job.id)

    def test_soft_stop_not_found_raises(self):
        """soft_stop_job for unknown ID should raise ValueError."""
        mgr = JobManager()
        with pytest.raises(ValueError, match="Job not found"):
            mgr.soft_stop_job("nonexistent")

    @patch("app.engine.components.signal_manager.TrainingSignalManager")
    def test_soft_stop_paused_sends_signal(self, mock_signal_cls):
        """soft_stop_job on PAUSED job should send 'soft_stop' signal."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.PAUSED

        mgr.soft_stop_job(job.id)

        mock_signal_cls.send_signal.assert_called_once()
        args = mock_signal_cls.send_signal.call_args
        assert args[0][1] == "soft_stop"

    @patch("app.engine.components.signal_manager.TrainingSignalManager")
    def test_soft_stop_does_not_change_status(self, mock_signal_cls):
        """soft_stop_job should NOT change job status (process handles exit)."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING

        mgr.soft_stop_job(job.id)
        # Status should remain RUNNING — the process will exit on its own
        assert job.status == JobStatus.RUNNING


# ── Restart Job ──────────────────────────────────────────────────────────


class TestRestartJob:
    """Tests for restart_job."""

    def test_restart_not_found_raises(self):
        """restart_job for unknown ID should raise ValueError."""
        mgr = JobManager()
        with pytest.raises(ValueError, match="Job not found"):
            mgr.restart_job("nonexistent")

    def test_restart_running_raises(self):
        """restart_job on RUNNING job should raise ValueError."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING

        with pytest.raises(ValueError, match="Cannot restart"):
            mgr.restart_job(job.id)

    def test_restart_pending_raises(self):
        """restart_job on PENDING job should raise ValueError."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())

        with pytest.raises(ValueError, match="Cannot restart"):
            mgr.restart_job(job.id)

    @patch.object(JobManager, "start_job")
    def test_restart_resets_state_and_relaunches(self, mock_start):
        """restart_job should reset fields and call start_job."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.COMPLETED
        job.error = "some error"
        job.pid = 999
        job.started_at = 1000.0
        job.finished_at = 2000.0
        job.logs = ["line1", "line2"]

        mgr.restart_job(job.id)

        assert job.status == JobStatus.PENDING
        assert job.error is None
        assert job.pid is None
        assert job.started_at is None
        assert job.finished_at is None
        assert job.logs == []
        mock_start.assert_called_once_with(job.id)

    @patch.object(JobManager, "start_job")
    def test_restart_queues_when_another_job_running(self, mock_start):
        """Restarting from the archive while a job is running must queue (not
        launch concurrently — the GPU runs one job at a time)."""
        mgr = JobManager()
        running = mgr.create_job("flux/dev", _make_config(lora_name="busy"))
        running.status = JobStatus.RUNNING
        archived = mgr.create_job("flux/dev", _make_config(lora_name="done"))
        archived.status = JobStatus.COMPLETED

        with patch.object(mgr, "_reset_job_log_state"), patch.object(mgr, "_persist_status"):
            mgr.restart_job(archived.id)

        assert archived.status == JobStatus.PENDING
        mock_start.assert_not_called()

    @patch.object(JobManager, "start_job")
    def test_restart_queues_when_another_job_paused(self, mock_start):
        """A paused job still holds the GPU, so a restart must queue behind it."""
        mgr = JobManager()
        paused = mgr.create_job("flux/dev", _make_config(lora_name="paused"))
        paused.status = JobStatus.PAUSED
        archived = mgr.create_job("flux/dev", _make_config(lora_name="done"))
        archived.status = JobStatus.STOPPED

        with patch.object(mgr, "_reset_job_log_state"), patch.object(mgr, "_persist_status"):
            mgr.restart_job(archived.id)

        assert archived.status == JobStatus.PENDING
        mock_start.assert_not_called()

    @patch.object(JobManager, "start_job")
    def test_restart_starts_immediately_when_idle(self, mock_start):
        """With nothing running, a restart launches right away."""
        mgr = JobManager()
        archived = mgr.create_job("flux/dev", _make_config(lora_name="done"))
        archived.status = JobStatus.STOPPED

        with patch.object(mgr, "_reset_job_log_state"), patch.object(mgr, "_persist_status"):
            mgr.restart_job(archived.id)

        mock_start.assert_called_once_with(archived.id)

    @patch.object(JobManager, "start_job")
    def test_restart_queued_appends_behind_pending(self, mock_start):
        """A queued restart gets a priority after existing pending jobs."""
        mgr = JobManager()
        running = mgr.create_job("flux/dev", _make_config(lora_name="busy"))
        running.status = JobStatus.RUNNING
        p1 = mgr.create_job("flux/dev", _make_config(lora_name="p1"))
        p1.status = JobStatus.PENDING
        p1.priority = 0
        archived = mgr.create_job("flux/dev", _make_config(lora_name="done"))
        archived.status = JobStatus.COMPLETED

        with patch.object(mgr, "_reset_job_log_state"), patch.object(mgr, "_persist_status"):
            mgr.restart_job(archived.id)

        assert archived.priority > p1.priority

    @patch.object(JobManager, "start_job")
    def test_restart_rotates_log_and_drops_offset(self, mock_start, tmp_path):
        """Regression: restart must clear stale log + offset so the new
        tailer can't re-dispatch the previous run's exit message.

        Without this, when the backend restarts and the user clicks
        "Continue" on a failed job, the new LogTailer loads the stale
        on-disk offset (which can point pre-exit from a buggy prior
        run), re-reads the old exit line, and immediately marks the
        restarted job FAILED with the old error.
        """
        mgr = JobManager()
        cfg = _make_config(output_dir=str(tmp_path), lora_name="lora")
        job = mgr.create_job("flux/dev", cfg)
        job.status = JobStatus.FAILED
        job.error = "old EMA crash"

        # Recreate the on-disk artefacts of the previous failed run:
        # the trainer's job_log.jsonl (with a stale exit message) and
        # the tailer's persisted offset.
        out_dir = mgr._get_job_output_dir(job)
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "job_log.jsonl")
        offset_path = log_path + ".offset"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write('{"t":1.0,"type":"exit","data":{"code":1,"error":"old EMA crash"}}\n')
        with open(offset_path, "w", encoding="utf-8") as f:
            f.write("0")  # stale pre-exit offset

        mgr.restart_job(job.id)

        assert not os.path.exists(log_path), (
            "active log_path must be cleared so the new trainer starts fresh"
        )
        assert not os.path.exists(offset_path), (
            "stale offset must be removed so the new tailer starts at 0"
        )
        # Rotation preserves the previous log for forensics
        rotated = [
            n for n in os.listdir(out_dir)
            if n.startswith("job_log.") and n.endswith(".jsonl") and n != "job_log.jsonl"
        ]
        assert rotated, "previous log should be rotated, not deleted"


# ── Event Broadcasting ───────────────────────────────────────────────────


class TestEventBroadcasting:
    """Tests for event broadcasting during lifecycle transitions."""

    @patch("app.core.job_manager.event_manager")
    @patch("app.engine.components.signal_manager.TrainingSignalManager")
    def test_pause_broadcasts_when_loop_set(self, mock_signal_cls, mock_em):
        """pause_job should schedule a broadcast if event loop is set."""
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        mgr = JobManager()
        mgr.set_loop(mock_loop)
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING

        with patch("app.core.job_manager.asyncio.run_coroutine_threadsafe") as mock_rct:
            mgr.pause_job(job.id)
            mock_rct.assert_called_once()

    @patch("app.engine.components.signal_manager.TrainingSignalManager")
    def test_pause_skips_broadcast_when_no_loop(self, mock_signal_cls):
        """pause_job should NOT broadcast if no event loop is set."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING

        with patch("app.core.job_manager.asyncio.run_coroutine_threadsafe") as mock_rct:
            mgr.pause_job(job.id)
            mock_rct.assert_not_called()

    @patch("app.core.job_manager.event_manager")
    @patch("app.engine.components.signal_manager.TrainingSignalManager")
    def test_resume_broadcasts_when_loop_set(self, mock_signal_cls, mock_em):
        """resume_job should schedule a broadcast if event loop is set."""
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        mgr = JobManager()
        mgr.set_loop(mock_loop)
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.PAUSED

        with patch("app.core.job_manager.asyncio.run_coroutine_threadsafe") as mock_rct:
            mgr.resume_job(job.id)
            mock_rct.assert_called_once()

    @patch("app.engine.components.signal_manager.TrainingSignalManager")
    def test_soft_stop_does_not_broadcast(self, mock_signal_cls):
        """soft_stop_job should NOT broadcast (process handles its own exit)."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING

        with patch("app.core.job_manager.asyncio.run_coroutine_threadsafe") as mock_rct:
            mgr.soft_stop_job(job.id)
            mock_rct.assert_not_called()


# ── Load From DB ─────────────────────────────────────────────────────────


class TestLoadFromDB:
    """Tests for load_from_db hydration on startup."""

    @patch("app.core.db.repositories.job_repo.JobHistoryRepository")
    def test_load_populates_jobs(self, MockRepo):
        """load_from_db should hydrate _jobs from DB rows."""
        MockRepo.return_value.list_recent.return_value = [
            {
                "id": "db-job-1",
                "definition_id": "flux/dev",
                "config": {"lora_name": "test"},
                "status": "completed",
                "created_at": 1000.0,
                "started_at": 1001.0,
                "finished_at": 1500.0,
                "error": None,
            },
        ]
        mgr = JobManager()
        mgr.load_from_db()

        job = mgr.get_job("db-job-1")
        assert job is not None
        assert job.status == JobStatus.COMPLETED
        # load_from_db resolves all rows to the single "standard" plugin
        assert job.plugin_id == "standard"
        assert job.config == {"lora_name": "test"}

    @patch("app.core.db.repositories.job_repo.JobHistoryRepository")
    def test_running_demoted_to_stopped(self, MockRepo):
        """Jobs that were running at shutdown should be marked stopped."""
        MockRepo.return_value.list_recent.return_value = [
            {
                "id": "db-job-run",
                "definition_id": "flux/dev",
                "config": {},
                "status": "running",
                "created_at": 1000.0,
                "started_at": 1001.0,
                "finished_at": None,
                "error": None,
            },
        ]
        mgr = JobManager()
        mgr.load_from_db()

        job = mgr.get_job("db-job-run")
        assert job is not None
        assert job.status == JobStatus.STOPPED

    @patch("app.core.db.repositories.job_repo.JobHistoryRepository")
    def test_paused_queued_for_relaunch(self, MockRepo):
        """Paused jobs whose process is dead are loaded as PENDING and queued for relaunch."""
        MockRepo.return_value.list_recent.return_value = [
            {
                "id": "db-job-paused",
                "definition_id": "sdxl/base",
                "config": {},
                "status": "paused",
                "created_at": 1000.0,
                "started_at": 1001.0,
                "finished_at": None,
                "error": None,
            },
        ]
        mgr = JobManager()
        mgr.load_from_db()

        job = mgr.get_job("db-job-paused")
        assert job is not None
        # Dead-process paused jobs are demoted to PENDING then queued for re-launch
        assert job.status == JobStatus.PENDING
        assert any(r["id"] == "db-job-paused" for r in mgr._recovery_jobs)

    @patch("app.core.db.repositories.job_repo.JobHistoryRepository")
    def test_existing_jobs_not_overwritten(self, MockRepo):
        """Live in-memory jobs should not be replaced by DB rows."""
        MockRepo.return_value.list_recent.return_value = [
            {
                "id": "live-job",
                "definition_id": "flux/dev",
                "config": {"from": "db"},
                "status": "completed",
                "created_at": 1000.0,
                "started_at": None,
                "finished_at": None,
                "error": None,
            },
        ]
        mgr = JobManager()
        live_job = mgr.create_job("flux/dev", {"from": "memory"})
        live_job.id = "live-job"  # Force same ID
        mgr._jobs["live-job"] = live_job

        mgr.load_from_db()

        # Should still be the in-memory version (create_job adds a job_id key)
        cfg = mgr.get_job("live-job").config
        assert cfg["from"] == "memory"
        assert "job_id" in cfg  # injected by create_job, not the DB row

    @patch("app.core.db.repositories.job_repo.JobHistoryRepository")
    def test_load_handles_exception_gracefully(self, MockRepo):
        """load_from_db should not crash if the DB is unavailable."""
        MockRepo.return_value.list_recent.side_effect = Exception("DB locked")
        mgr = JobManager()
        mgr.load_from_db()  # Should not raise

        assert mgr.list_jobs() == []


class TestFreshRestart:
    """restart_job(fresh=True) deletes the run's output folder first."""

    def test_delete_job_output_dir_removes_run_folder(self, tmp_path):
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config(output_dir=str(tmp_path), lora_name="run1"))
        run_dir = tmp_path / "run1_dev"
        run_dir.mkdir()
        (run_dir / "loss_history.json").write_text("[]", encoding="utf-8")

        mgr._delete_job_output_dir(job)
        assert not run_dir.exists()

    def test_delete_job_output_dir_absent_is_noop(self, tmp_path):
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config(output_dir=str(tmp_path), lora_name="ghost"))
        mgr._delete_job_output_dir(job)  # folder never created -> must not raise

    def test_restart_fresh_deletes_then_relaunches(self, tmp_path):
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config(output_dir=str(tmp_path), lora_name="run2"))
        job.status = JobStatus.COMPLETED
        run_dir = tmp_path / "run2_dev"
        run_dir.mkdir()
        (run_dir / "x.txt").write_text("data", encoding="utf-8")

        with patch.object(mgr, "start_job") as mock_start, \
                patch.object(mgr, "_stop_tailer"), \
                patch.object(mgr, "_reset_job_log_state"), \
                patch.object(mgr, "_persist_status"):
            mgr.restart_job(job.id, fresh=True)

        assert not run_dir.exists()
        mock_start.assert_called_once_with(job.id)

    def test_restart_without_fresh_keeps_output(self, tmp_path):
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config(output_dir=str(tmp_path), lora_name="run3"))
        job.status = JobStatus.FAILED
        run_dir = tmp_path / "run3_dev"
        run_dir.mkdir()
        (run_dir / "x.txt").write_text("data", encoding="utf-8")

        with patch.object(mgr, "start_job"), \
                patch.object(mgr, "_stop_tailer"), \
                patch.object(mgr, "_reset_job_log_state"), \
                patch.object(mgr, "_persist_status"):
            mgr.restart_job(job.id, fresh=False)

        assert run_dir.exists()


class TestReorderPending:
    """reorder_pending swaps pending run order via in-memory priority."""

    def _mk_pending(self, mgr, n):
        jobs = []
        for i in range(n):
            j = mgr.create_job("flux/dev", _make_config(lora_name=f"j{i}"))
            j.created_at = 1000.0 + i  # ascending FIFO
            jobs.append(j)
        return jobs

    def _order(self, jobs):
        return [j.id for j in sorted(jobs, key=lambda j: (j.priority, j.created_at))]

    def test_move_up_runs_sooner(self):
        mgr = JobManager()
        a, b, c = self._mk_pending(mgr, 3)
        mgr.reorder_pending(c.id, "up")  # a, b, c -> a, c, b
        assert self._order([a, b, c]) == [a.id, c.id, b.id]

    def test_move_down_runs_later(self):
        mgr = JobManager()
        a, b, c = self._mk_pending(mgr, 3)
        mgr.reorder_pending(a.id, "down")  # a, b, c -> b, a, c
        assert self._order([a, b, c]) == [b.id, a.id, c.id]

    def test_move_up_at_top_is_noop(self):
        mgr = JobManager()
        a, b = self._mk_pending(mgr, 2)
        mgr.reorder_pending(a.id, "up")
        assert self._order([a, b]) == [a.id, b.id]

    def test_invalid_direction_raises(self):
        mgr = JobManager()
        (a,) = self._mk_pending(mgr, 1)
        with pytest.raises(ValueError):
            mgr.reorder_pending(a.id, "sideways")

    def test_unknown_job_raises(self):
        mgr = JobManager()
        self._mk_pending(mgr, 1)
        with pytest.raises(ValueError):
            mgr.reorder_pending("ghost", "up")


class TestStartJobClearsStaleSignal:
    """Regression: a fresh launch must not inherit a leftover pause signal.

    Output dirs are keyed by lora_name + definition_id, so a new job can reuse
    a directory whose previous occupant left an unconsumed pause/soft_stop
    signal. Without clearing it, the new trainer reads the stale signal on its
    first check and blocks in the pause loop (status "Training", no GPU load,
    no steps) — the exact idle-training bug observed in the field.
    """

    def _mock_plugin(self):
        plugin = MagicMock()
        # A process object with no `.pid` attr makes start_job skip the
        # LogTailer + PID watchdog, keeping the test free of background threads.
        plugin.start_training.return_value = MagicMock(spec=[])
        return plugin

    @patch("app.core.job_manager.plugin_manager")
    def test_start_job_removes_stale_signal(self, mock_pm, tmp_path):
        mock_pm.get_plugin.return_value = self._mock_plugin()
        mgr = JobManager()
        job = mgr.create_job("standard", _make_config(output_dir=str(tmp_path), lora_name="reuse"))
        out_dir = mgr._get_job_output_dir(job)
        os.makedirs(out_dir, exist_ok=True)
        sig = os.path.join(out_dir, "signal.json")
        with open(sig, "w", encoding="utf-8") as f:
            f.write('{"action": "pause"}')

        mgr.start_job(job.id)

        assert not os.path.exists(sig), "stale pause signal must be cleared on launch"

    @patch("app.core.job_manager.plugin_manager")
    def test_start_job_preserves_signal_when_disabled(self, mock_pm, tmp_path):
        """relaunch_paused passes clear_stale_signal=False to keep its
        intentional pause-before-launch signal."""
        mock_pm.get_plugin.return_value = self._mock_plugin()
        mgr = JobManager()
        job = mgr.create_job("standard", _make_config(output_dir=str(tmp_path), lora_name="recover"))
        out_dir = mgr._get_job_output_dir(job)
        os.makedirs(out_dir, exist_ok=True)
        sig = os.path.join(out_dir, "signal.json")
        with open(sig, "w", encoding="utf-8") as f:
            f.write('{"action": "pause"}')

        mgr.start_job(job.id, clear_stale_signal=False)

        assert os.path.exists(sig), "intentional pause signal must survive recovery launch"


class TestAutoQueue:
    """Backend-owned queue advancement.

    The queue must drain unattended (no browser open), so advancing to the
    next pending job is the backend's job — triggered on every terminal
    transition, at startup, and when the auto-queue toggle is switched on.
    Gated by a persisted ``jobs.auto_queue`` setting and the single-GPU rule
    (never run two jobs at once).
    """

    def _pending(self, mgr, lora, priority, created_at):
        j = mgr.create_job("flux/dev", _make_config(lora_name=lora))
        j.status = JobStatus.PENDING
        j.priority = priority
        j.created_at = created_at
        return j

    def test_advance_starts_next_pending_when_idle(self):
        mgr = JobManager()
        j = self._pending(mgr, "a", 0, 1.0)
        with patch.object(mgr, "_auto_queue_enabled", return_value=True), \
                patch.object(mgr, "start_job") as ms:
            mgr.advance_queue()
        ms.assert_called_once_with(j.id)

    def test_advance_noop_when_auto_queue_disabled(self):
        mgr = JobManager()
        self._pending(mgr, "a", 0, 1.0)
        with patch.object(mgr, "_auto_queue_enabled", return_value=False), \
                patch.object(mgr, "start_job") as ms:
            mgr.advance_queue()
        ms.assert_not_called()

    def test_advance_noop_when_a_job_is_running(self):
        mgr = JobManager()
        running = mgr.create_job("flux/dev", _make_config(lora_name="busy"))
        running.status = JobStatus.RUNNING
        self._pending(mgr, "next", 0, 2.0)
        with patch.object(mgr, "_auto_queue_enabled", return_value=True), \
                patch.object(mgr, "start_job") as ms:
            mgr.advance_queue()
        ms.assert_not_called()

    def test_advance_noop_when_a_job_is_paused(self):
        """A paused job still holds the GPU/VRAM — don't start another."""
        mgr = JobManager()
        paused = mgr.create_job("flux/dev", _make_config(lora_name="paused"))
        paused.status = JobStatus.PAUSED
        self._pending(mgr, "next", 0, 2.0)
        with patch.object(mgr, "_auto_queue_enabled", return_value=True), \
                patch.object(mgr, "start_job") as ms:
            mgr.advance_queue()
        ms.assert_not_called()

    def test_advance_picks_priority_then_fifo(self):
        mgr = JobManager()
        # created earliest but higher priority number => should NOT run first
        self._pending(mgr, "late", 5, 1.0)
        winner = self._pending(mgr, "winner", 0, 3.0)
        self._pending(mgr, "mid", 0, 4.0)  # same priority, later created
        with patch.object(mgr, "_auto_queue_enabled", return_value=True), \
                patch.object(mgr, "start_job") as ms:
            mgr.advance_queue()
        ms.assert_called_once_with(winner.id)

    def test_advance_noop_when_no_pending(self):
        mgr = JobManager()
        done = mgr.create_job("flux/dev", _make_config(lora_name="done"))
        done.status = JobStatus.COMPLETED
        with patch.object(mgr, "_auto_queue_enabled", return_value=True), \
                patch.object(mgr, "start_job") as ms:
            mgr.advance_queue()
        ms.assert_not_called()

    def test_advance_noop_when_launch_already_in_flight(self):
        """The in-flight guard prevents two terminal events from each
        launching a job in the window before status flips to RUNNING."""
        mgr = JobManager()
        self._pending(mgr, "a", 0, 1.0)
        mgr._starting = True
        with patch.object(mgr, "_auto_queue_enabled", return_value=True), \
                patch.object(mgr, "start_job") as ms:
            mgr.advance_queue()
        ms.assert_not_called()

    def test_advance_skips_failed_start_and_tries_next(self):
        """If launching the next job fails, the queue must not stall — skip it
        and try the following pending job."""
        mgr = JobManager()
        j1 = self._pending(mgr, "bad", 0, 1.0)
        j2 = self._pending(mgr, "good", 1, 2.0)
        started: list[str] = []

        def fake_start(job_id, **kw):
            job = mgr.get_job(job_id)
            if job_id == j1.id:
                job.status = JobStatus.FAILED  # mimic start_job's failure path
                raise ValueError("boom")
            job.status = JobStatus.RUNNING
            started.append(job_id)

        with patch.object(mgr, "_auto_queue_enabled", return_value=True), \
                patch.object(mgr, "start_job", side_effect=fake_start):
            mgr.advance_queue()
        assert started == [j2.id]

    def test_exit_completed_schedules_advance(self):
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING
        with patch.object(mgr, "schedule_advance_queue") as ms, \
                patch.object(mgr, "_persist_status"), patch.object(mgr, "_stop_tailer"):
            mgr._handle_exit_message(job.id, {"code": 0})
        assert job.status == JobStatus.COMPLETED
        ms.assert_called_once()

    def test_exit_failed_schedules_advance(self):
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING
        with patch.object(mgr, "schedule_advance_queue") as ms, \
                patch.object(mgr, "_persist_status"), patch.object(mgr, "_stop_tailer"):
            mgr._handle_exit_message(job.id, {"code": 1, "error": "kaboom"})
        assert job.status == JobStatus.FAILED
        ms.assert_called_once()

    def test_exit_user_stopped_does_not_advance(self):
        """A user hard-stop (status already STOPPED) is an intentional
        intervention — do NOT auto-start the next job."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.STOPPED  # set by stop_job before the process exits
        with patch.object(mgr, "schedule_advance_queue") as ms, \
                patch.object(mgr, "_persist_status"), patch.object(mgr, "_stop_tailer"):
            mgr._handle_exit_message(job.id, {"code": 143})
        assert job.status == JobStatus.STOPPED
        ms.assert_not_called()


class TestPriorityPersistence:
    """Manual pending-queue order (priority) must survive a backend restart.

    Priority is the primary run-order key; without persistence a restart resets
    everything to 0 and the queue reverts to FIFO-by-created_at, silently losing
    the user's arrangement.
    """

    def _pending(self, mgr, lora, created_at):
        j = mgr.create_job("flux/dev", _make_config(lora_name=lora))
        j.status = JobStatus.PENDING
        j.created_at = created_at
        return j

    @patch("app.core.db.repositories.job_repo.JobHistoryRepository")
    def test_reorder_persists_every_pending_priority(self, MockRepo):
        mgr = JobManager()
        a = self._pending(mgr, "a", 1.0)
        b = self._pending(mgr, "b", 2.0)
        c = self._pending(mgr, "c", 3.0)
        MockRepo.return_value.set_priority.reset_mock()

        mgr.reorder_pending(c.id, "up")  # a,b,c -> a,c,b

        # In-memory order reflects the move
        assert a.priority == 0 and c.priority == 1 and b.priority == 2
        # …and each new priority was persisted (id -> priority)
        persisted = {
            call.args[0]: call.args[1]
            for call in MockRepo.return_value.set_priority.call_args_list
        }
        assert persisted == {a.id: 0, c.id: 1, b.id: 2}

    @patch("app.core.db.repositories.job_repo.JobHistoryRepository")
    def test_load_from_db_hydrates_priority(self, MockRepo):
        MockRepo.return_value.list_recent.return_value = [
            {
                "id": "j-pri", "definition_id": "flux/dev", "config": {},
                "status": "pending", "created_at": 1.0, "started_at": None,
                "finished_at": None, "error": None, "priority": 7,
            },
        ]
        mgr = JobManager()
        mgr.load_from_db()
        assert mgr.get_job("j-pri").priority == 7

    @patch("app.core.db.repositories.job_repo.JobHistoryRepository")
    def test_load_from_db_priority_defaults_zero(self, MockRepo):
        MockRepo.return_value.list_recent.return_value = [
            {
                "id": "j-nopri", "definition_id": "flux/dev", "config": {},
                "status": "pending", "created_at": 1.0, "started_at": None,
                "finished_at": None, "error": None,  # no priority key
            },
        ]
        mgr = JobManager()
        mgr.load_from_db()
        assert mgr.get_job("j-nopri").priority == 0

    @patch.object(JobManager, "start_job")
    def test_restart_queued_persists_priority(self, mock_start):
        mgr = JobManager()
        running = mgr.create_job("flux/dev", _make_config(lora_name="busy"))
        running.status = JobStatus.RUNNING
        archived = mgr.create_job("flux/dev", _make_config(lora_name="done"))
        archived.status = JobStatus.COMPLETED

        with patch.object(mgr, "_reset_job_log_state"), \
                patch.object(mgr, "_persist_status") as mp:
            mgr.restart_job(archived.id)

        pending_persist = [c for c in mp.call_args_list if c.args[1:2] == ("pending",)]
        assert pending_persist, "restart must persist the pending status"
        assert pending_persist[0].kwargs.get("priority") == archived.priority


class TestSignalPauseReconcile:
    """Trainer-side pause/resume log events must drive the live job status, so
    a signal-paused run shows PAUSED (and offers Resume) regardless of how the
    pause arose."""

    def test_paused_event_sets_status_paused(self):
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING
        mgr._reconcile_signal_pause(job.id, '{"event": "training_paused_by_signal"}')
        assert job.status == JobStatus.PAUSED
        assert job.paused_at is not None

    def test_resumed_event_sets_status_running(self):
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.PAUSED
        job.paused_at = 123.0
        mgr._reconcile_signal_pause(job.id, '{"event": "training_resumed_by_signal"}')
        assert job.status == JobStatus.RUNNING
        assert job.paused_at is None

    def test_does_not_override_terminal_state(self):
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.STOPPED
        mgr._reconcile_signal_pause(job.id, '{"event": "training_paused_by_signal"}')
        assert job.status == JobStatus.STOPPED

    def test_unrelated_log_is_noop(self):
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING
        mgr._reconcile_signal_pause(job.id, '{"event": "sampling_complete"}')
        assert job.status == JobStatus.RUNNING

