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

