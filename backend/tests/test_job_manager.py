"""
Tests for the JobManager lifecycle orchestrator.

Covers: CRUD operations, status transitions, signal-based pause/resume/stop,
output directory resolution, event broadcasting, and error handling.
"""
import os
import asyncio

import pytest
from unittest.mock import MagicMock, patch

from app.core.job import Job, JobStatus
from app.core.job_manager import JobManager, JobConflictError
from app.engine.core.definitions import ModelDefinition
from app.engine.models.registry import registry


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

    def test_delete_running_job_without_force_raises_conflict(self):
        """delete_job on a RUNNING job with force=False must refuse — the job
        survives and its trainer subprocess is left alone (no GPU zombie)."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING
        job.pid = 12345

        with pytest.raises(JobConflictError):
            mgr.delete_job(job.id)

        assert mgr.get_job(job.id) is not None
        assert mgr.get_job(job.id).status == JobStatus.RUNNING

    def test_delete_paused_job_without_force_raises_conflict(self):
        """A PAUSED job still holds VRAM — same guard as RUNNING."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.PAUSED
        job.pid = 12345

        with pytest.raises(JobConflictError):
            mgr.delete_job(job.id)

        assert mgr.get_job(job.id) is not None

    @patch.object(JobManager, "_terminate_process_tree", return_value=[12345, 67890])
    def test_delete_running_job_with_force_kills_tree_first(self, mock_tree):
        """force=True kills the process tree (the exact zombie stop_job kills)
        BEFORE removing the job, and drops any auto-resume bookkeeping —
        otherwise an unowned trainer keeps holding VRAM and the single-GPU
        guard sees the GPU as free, letting auto-queue double-launch."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING
        job.pid = 12345
        mgr._auto_resume_state[job.id] = {"total": 1}

        mgr.delete_job(job.id, force=True)

        mock_tree.assert_called_once_with(12345)
        assert mgr.get_job(job.id) is None
        assert job.id not in mgr._auto_resume_state

    @patch.object(JobManager, "_terminate_process_tree")
    def test_delete_pending_job_needs_no_force(self, mock_tree):
        """A PENDING (not yet launched) job has no process to kill — delete
        must succeed without force and must not attempt a kill."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        mgr.delete_job(job.id)
        assert mgr.get_job(job.id) is None
        mock_tree.assert_not_called()


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

    @patch.object(JobManager, "_start_pid_watchdog")
    @patch("app.core.job_manager.LogTailer")
    @patch("app.core.job_manager.plugin_manager")
    def test_preflight_marks_job_preparing_without_persisting_running(
        self, mock_pm, _mock_tailer, _mock_watchdog, tmp_path,
    ):
        """During the base-model pre-fetch the job must show RUNNING + a
        'downloading' label (not idle 'pending') so the queue card and top bar
        reflect the real work in progress. The preparing state is in-memory
        ONLY — 'running' is NOT persisted at pre-fetch time, so a restart
        mid-download cleanly resumes from the persisted 'pending' row rather
        than being marked stopped/stranded."""
        mgr = JobManager()
        job = mgr.create_job("standard", _make_config())

        proc = MagicMock()
        proc.pid = 4242
        mock_pm.get_plugin.return_value.start_training.return_value = proc

        persisted = []
        seen = {}

        def fake_preflight(j):
            seen["status"] = j.status
            seen["label"] = j.status_label
            seen["persisted_running"] = any(s == "running" for _, s in persisted)

        with patch.object(mgr, "_preflight_download", side_effect=fake_preflight), \
             patch.object(
                 mgr, "_persist_status",
                 side_effect=lambda jid, status, **kw: persisted.append((jid, status)),
             ), \
             patch.object(mgr, "_get_job_output_dir", return_value=str(tmp_path)), \
             patch("app.engine.components.signal_manager.TrainingSignalManager"):
            mgr.start_job(job.id)

        # The job was actively "preparing" while the model downloaded.
        assert seen["status"] == JobStatus.RUNNING
        assert seen["label"] and "download" in seen["label"].lower()
        # …but 'running' was NOT persisted at that point (DB still 'pending').
        assert seen["persisted_running"] is False

    @patch.object(JobManager, "_start_pid_watchdog")
    @patch("app.core.job_manager.LogTailer")
    @patch("app.core.job_manager.event_manager")
    @patch("app.core.job_manager.plugin_manager")
    def test_preflight_broadcasts_preparing_state(
        self, mock_pm, _mock_em, _mock_tailer, _mock_watchdog, tmp_path,
    ):
        """The preparing transition must broadcast BEFORE the download begins,
        so the frontend flips the card to RUNNING + 'downloading' immediately
        rather than only after the (possibly multi-minute) fetch completes."""
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        mgr = JobManager()
        mgr.set_loop(mock_loop)
        job = mgr.create_job("standard", _make_config())

        proc = MagicMock()
        proc.pid = 4242
        mock_pm.get_plugin.return_value.start_training.return_value = proc

        seen = {}
        with patch("app.core.job_manager.asyncio.run_coroutine_threadsafe") as mock_rct:
            def fake_preflight(j):
                seen["broadcasts_before_download"] = mock_rct.call_count

            with patch.object(mgr, "_preflight_download", side_effect=fake_preflight), \
                 patch.object(mgr, "_persist_status"), \
                 patch.object(mgr, "_get_job_output_dir", return_value=str(tmp_path)), \
                 patch("app.engine.components.signal_manager.TrainingSignalManager"):
                mgr.start_job(job.id)

        assert seen["broadcasts_before_download"] >= 1


# ── Stop Job ─────────────────────────────────────────────────────────────


class TestStopJob:
    """Tests for stop_job."""

    def test_stop_not_found_raises(self):
        """stop_job for unknown ID should raise ValueError."""
        mgr = JobManager()
        with pytest.raises(ValueError, match="Job not found"):
            mgr.stop_job("nonexistent")

    @patch.object(JobManager, "_terminate_process_tree", return_value=[12345, 67890])
    def test_stop_running_job_kills_process_tree(self, mock_tree):
        """stop_job on a RUNNING job kills the whole tree (launcher + worker)."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING
        job.pid = 12345

        mgr.stop_job(job.id)

        mock_tree.assert_called_once_with(12345)
        assert job.status == JobStatus.STOPPED
        assert job.finished_at is not None
        assert job.pid is None  # cleared so a stale watchdog can't re-find it

    @patch.object(JobManager, "_terminate_process_tree", return_value=[])
    def test_stop_already_dead_job_marks_stopped(self, mock_tree):
        """stop_job on a job whose process already exited (stale 'running')
        still resolves to STOPPED — one click clears the stuck card."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING
        job.pid = 99999  # already gone → tree-kill finds nothing

        mgr.stop_job(job.id)

        mock_tree.assert_called_once_with(99999)
        assert job.status == JobStatus.STOPPED
        assert job.error is None  # a user stop is not a failure

    @patch.object(JobManager, "_terminate_process_tree")
    def test_stop_non_running_job_is_noop(self, mock_tree):
        """stop_job on a PENDING job should not kill anything or change state."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        # job.status is PENDING — no kill should be attempted
        mgr.stop_job(job.id)
        mock_tree.assert_not_called()
        assert job.status == JobStatus.PENDING

    @patch.object(JobManager, "_terminate_process_tree", return_value=[12345])
    def test_stop_paused_job_kills_process_tree(self, mock_tree):
        """stop_job on a PAUSED job kills the tree too."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.PAUSED
        job.pid = 12345

        mgr.stop_job(job.id)
        mock_tree.assert_called_once_with(12345)
        assert job.status == JobStatus.STOPPED
        assert job.finished_at is not None

    @patch.object(JobManager, "schedule_advance_queue")
    @patch.object(JobManager, "_terminate_process_tree", return_value=[12345])
    def test_stop_advances_the_queue(self, mock_tree, mock_advance):
        """Stopping the running job frees the GPU, so the queue must advance —
        otherwise a pending job strands until a manual Start (the recurring
        auto-queue complaint). Mirrors the natural-exit / watchdog paths.
        """
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING
        job.pid = 12345

        mgr.stop_job(job.id)

        mock_advance.assert_called_once()

    @patch.object(JobManager, "schedule_advance_queue")
    @patch.object(JobManager, "_terminate_process_tree")
    def test_stop_noop_does_not_advance(self, mock_tree, mock_advance):
        """A no-op stop (job not RUNNING/PAUSED) returns early and must not
        touch the queue."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())  # PENDING
        mgr.stop_job(job.id)
        mock_advance.assert_not_called()


class TestProcessTreeLifecycle:
    """Tree-kill + worker-resolution + death-detection (the zombie / stale-UI fixes)."""

    def test_terminate_process_tree_kills_parent_and_descendants(self):
        parent = MagicMock()
        parent.pid = 100
        child = MagicMock()
        child.pid = 200
        parent.children.return_value = [child]

        with patch("psutil.Process", return_value=parent) as mock_proc, \
             patch("psutil.wait_procs", return_value=([], [])) as mock_wait:
            killed = JobManager._terminate_process_tree(100)

        mock_proc.assert_called_once_with(100)
        parent.children.assert_called_once_with(recursive=True)
        # Both the worker child AND the launcher are terminated.
        child.terminate.assert_called_once()
        parent.terminate.assert_called_once()
        mock_wait.assert_called_once()
        assert set(killed) == {100, 200}

    def test_terminate_process_tree_kills_stragglers(self):
        parent = MagicMock()
        parent.pid = 100
        parent.children.return_value = []
        with patch("psutil.Process", return_value=parent), \
             patch("psutil.wait_procs", return_value=([], [parent])):
            JobManager._terminate_process_tree(100)
        parent.terminate.assert_called_once()
        parent.kill.assert_called_once()  # SIGTERM didn't take → SIGKILL

    def test_terminate_process_tree_noop_for_none(self):
        assert JobManager._terminate_process_tree(None) == []
        assert JobManager._terminate_process_tree(0) == []

    def test_resolve_worker_pid_returns_trainer_child(self):
        parent = MagicMock()
        parent.pid = 100
        worker = MagicMock()
        worker.pid = 200
        unrelated = MagicMock()
        unrelated.pid = 300
        parent.children.return_value = [unrelated, worker]

        def is_trainer(pid, job_id):
            return pid == 200  # only the worker matches this job

        with patch("psutil.Process", return_value=parent), \
             patch.object(JobManager, "_is_trainer_process", side_effect=is_trainer):
            assert JobManager._resolve_worker_pid(100, "job-x") == 200

    def test_resolve_worker_pid_falls_back_to_launcher(self):
        parent = MagicMock()
        parent.pid = 100
        parent.children.return_value = []
        with patch("psutil.Process", return_value=parent), \
             patch.object(JobManager, "_is_trainer_process", return_value=False):
            # No trainer child (POSIX / single-process) → watch the launcher itself.
            assert JobManager._resolve_worker_pid(100, "job-x") == 100

    def test_trainer_tree_dead_launcher_gone(self):
        mgr = JobManager()
        with patch.object(JobManager, "_is_pid_alive", return_value=False):
            assert mgr._trainer_tree_dead(100, 200) is True

    def test_trainer_tree_dead_worker_killed_launcher_lingers(self):
        """THE stale-UI case: launcher alive, worker hard-killed → dead."""
        mgr = JobManager()
        alive = {100: True, 200: False}
        with patch.object(JobManager, "_is_pid_alive", side_effect=lambda p: alive.get(p, False)):
            assert mgr._trainer_tree_dead(100, 200) is True

    def test_trainer_tree_alive_when_both_running(self):
        mgr = JobManager()
        with patch.object(JobManager, "_is_pid_alive", return_value=True):
            assert mgr._trainer_tree_dead(100, 200) is False

    def test_trainer_tree_single_process_tracks_launcher(self):
        """Single-process layout (worker_pid == launcher): rely on launcher liveness."""
        mgr = JobManager()
        with patch.object(JobManager, "_is_pid_alive", return_value=True):
            assert mgr._trainer_tree_dead(100, 100) is False


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
    def test_restart_fresh_clears_stale_resume_from_checkpoint(self, mock_start):
        """A from-zero (fresh) restart must strip resume_from_checkpoint.

        If the job was previously continued from a checkpoint, its config
        still carries resume_from_checkpoint pointing into the output dir that
        the fresh restart just deleted — the trainer would then crash in
        _resume_if_needed with FileNotFoundError. Fresh means from scratch.
        """
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.STOPPED
        job.config["resume_from_checkpoint"] = r"D:\outputs\run\final"

        persisted: dict = {}
        with patch.object(mgr, "_delete_job_output_dir"), patch.object(
            mgr, "_reset_job_log_state"
        ), patch.object(mgr, "_persist_status"), patch.object(
            mgr, "_persist_config", side_effect=lambda jid, cfg: persisted.update(cfg)
        ):
            mgr.restart_job(job.id, fresh=True)

        assert "resume_from_checkpoint" not in job.config, (
            "fresh restart left the stale resume path in the job config"
        )
        assert "resume_from_checkpoint" not in persisted, (
            "the cleared config was not persisted"
        )
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
        MockRepo.return_value.list_by_statuses.return_value = []
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
        MockRepo.return_value.list_by_statuses.return_value = []
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
        MockRepo.return_value.list_by_statuses.return_value = []
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
        MockRepo.return_value.list_by_statuses.return_value = []
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
        MockRepo.return_value.list_by_statuses.return_value = []
        MockRepo.return_value.list_recent.side_effect = Exception("DB locked")
        mgr = JobManager()
        mgr.load_from_db()  # Should not raise

        assert mgr.list_jobs() == []


class TestLoadFromDbActiveStatusHydration:
    """T4: restart hydration must never strand an active job outside the
    ``list_recent`` 200-row recency window.

    Uses a REAL, isolated DatabaseEngine/JobHistoryRepository (not mocked) so
    the actual ``ORDER BY created_at DESC LIMIT 200`` semantics are exercised
    — the bug is specifically about that SQL window dropping an old row, so a
    mock-only test wouldn't prove the fix.
    """

    @pytest.fixture()
    def isolated_db(self, tmp_path):
        """Swap the DatabaseEngine singleton for a private tmp-file DB for the
        duration of one test, so this test's rows never mix with the shared
        session-scoped test DB (``conftest.py::_isolate_test_db``) that other
        tests' ``create_job`` calls also write into."""
        from app.core.db.engine import DatabaseEngine

        old_instance = DatabaseEngine._instance
        engine = DatabaseEngine(db_path=str(tmp_path / "t4_isolated.db"))
        engine.initialize()
        DatabaseEngine._instance = engine
        yield engine
        engine.close()
        DatabaseEngine._instance = old_instance

    def test_old_pending_job_survives_the_200_row_recency_window(self, isolated_db):
        from app.core.db.repositories.job_repo import JobHistoryRepository

        repo = JobHistoryRepository()

        old_pending_id = "old-pending-job"
        repo.create(
            {
                "id": old_pending_id,
                "lora_name": "old",
                "definition_id": "flux/dev",
                "status": "pending",
                "config": {},
                "created_at": 1.0,  # the oldest row in the table
            }
        )

        # 201 terminal rows, ALL newer than the pending job — guarantees a
        # plain `list_recent(limit=200)` window excludes it (only 200 slots,
        # 201 newer candidates).
        for i in range(201):
            repo.create(
                {
                    "id": f"terminal-{i}",
                    "lora_name": f"t{i}",
                    "definition_id": "flux/dev",
                    "status": "completed",
                    "config": {},
                    "created_at": 1000.0 + i,
                }
            )

        mgr = JobManager()
        mgr.load_from_db()

        hydrated = mgr.get_job(old_pending_id)
        assert hydrated is not None, (
            "a pending job older than the 200-row recency window must still "
            "be hydrated — otherwise it's invisible to the queue forever"
        )
        assert hydrated.status == JobStatus.PENDING


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


class TestStartJobLaunchFailureMarksFailed:
    """Any exception during launch must reset the job to FAILED (T3).

    Before the fix, only ``except (OSError, ValueError, RuntimeError)``
    reset the job — a KeyError/TypeError from a bad config, or an OSError
    raised by ``clear_stale_signal`` (which sat OUTSIDE the try), left the
    job stuck RUNNING with ``pid=None``. ``_reconcile_active_jobs``
    deliberately skips pid-less jobs ("possibly mid-launch"), so that phantom
    blocks the single-GPU queue until a backend restart.
    """

    def _mgr_with_job(self, tmp_path):
        mgr = JobManager()
        job = mgr.create_job(
            "standard", _make_config(output_dir=str(tmp_path), lora_name="boom")
        )
        return mgr, job

    @patch("app.core.job_manager.plugin_manager")
    def test_oserror_from_start_training_marks_failed(self, mock_pm, tmp_path):
        """Regression guard: the pre-existing OSError handling must still
        mark FAILED, persist, and re-raise exactly as before."""
        mgr, job = self._mgr_with_job(tmp_path)
        mock_pm.get_plugin.return_value.start_training.side_effect = OSError("boom")

        with pytest.raises(OSError):
            mgr.start_job(job.id)

        assert job.status == JobStatus.FAILED
        assert job.error == "boom"

    @patch("app.core.job_manager.plugin_manager")
    def test_keyerror_from_start_training_marks_failed(self, mock_pm, tmp_path):
        """A non-OSError/ValueError/RuntimeError exception (e.g. a bad-config
        KeyError) must ALSO reset the job to FAILED instead of stranding it
        RUNNING with pid=None."""
        mgr, job = self._mgr_with_job(tmp_path)
        mock_pm.get_plugin.return_value.start_training.side_effect = KeyError("boom")

        with pytest.raises(KeyError):
            mgr.start_job(job.id)

        assert job.status == JobStatus.FAILED
        assert job.pid is None

    @patch("app.core.job_manager.plugin_manager")
    def test_typeerror_from_start_training_marks_failed(self, mock_pm, tmp_path):
        mgr, job = self._mgr_with_job(tmp_path)
        mock_pm.get_plugin.return_value.start_training.side_effect = TypeError("boom")

        with pytest.raises(TypeError):
            mgr.start_job(job.id)

        assert job.status == JobStatus.FAILED

    @patch("app.core.job_manager.plugin_manager")
    def test_launch_failure_does_not_block_the_single_gpu_guard(self, mock_pm, tmp_path):
        """After a launch failure the job must NOT be counted as
        RUNNING/PAUSED by the single-GPU guard — otherwise the queue is
        wedged until a manual restart."""
        mgr, job = self._mgr_with_job(tmp_path)
        mock_pm.get_plugin.return_value.start_training.side_effect = KeyError("boom")

        with pytest.raises(KeyError):
            mgr.start_job(job.id)

        # A fresh pending job must be claimable — proves the failed launch
        # isn't still occupying the single-GPU slot.
        other = mgr.create_job(
            "standard", _make_config(output_dir=str(tmp_path), lora_name="next")
        )
        assert mgr._claim_next_pending() == other.id

    @patch("app.core.job_manager.plugin_manager")
    def test_clear_stale_signal_oserror_marks_failed(self, mock_pm, tmp_path):
        """``clear_stale_signal`` now runs INSIDE the guarded try — an OSError
        clearing a stale signal file must also reset the job to FAILED
        instead of leaving it RUNNING with no trainer ever launched."""
        mgr, job = self._mgr_with_job(tmp_path)

        with patch(
            "app.engine.components.signal_manager.TrainingSignalManager"
        ) as mock_sig_mgr:
            mock_sig_mgr.return_value.clear_signal.side_effect = OSError("disk full")

            with pytest.raises(OSError):
                mgr.start_job(job.id)  # clear_stale_signal=True by default

        assert job.status == JobStatus.FAILED
        mock_pm.get_plugin.return_value.start_training.assert_not_called()


class TestStartJobPreflightDownload:
    """start_job pre-fetches the base model in-process before launching.

    The trainer runs in a detached subprocess where the download-progress WS
    bridge is a no-op, so the top-bar indicator never updates for the base-model
    download. Pre-fetching in the API process (where the WS loop is captured)
    fixes that; the subprocess then loads from the warm cache.
    """

    def _mock_plugin(self):
        plugin = MagicMock()
        # No `.pid` → start_job skips the LogTailer + PID watchdog (no threads).
        plugin.start_training.return_value = MagicMock(spec=[])
        return plugin

    def _fake_definition(self):
        """spec=ModelDefinition mock with a real registered family so
        _apply_video_contract -> resolve_capabilities(definition.family)
        doesn't blow up on a MagicMock family name. discover_families() is
        idempotent (guarded by ModelRegistry._discovered) so this is safe to
        call regardless of what ran before this test in the session."""
        registry.discover_families()
        fake_def = MagicMock(spec=ModelDefinition)
        fake_def.family = "sdxl"
        fake_def.architecture_params = {}
        fake_def.control_inputs = 0
        fake_def.defaults = {}
        return fake_def

    @patch("app.engine.utils.model_utils.ModelPathResolver.ensure_definition_cached")
    @patch("app.engine.models.registry.registry.get_definition")
    @patch("app.core.job_manager.plugin_manager")
    def test_start_job_prefetches_model(self, mock_pm, mock_get_def, mock_prefetch, tmp_path):
        mock_pm.get_plugin.return_value = self._mock_plugin()
        fake_def = self._fake_definition()
        mock_get_def.return_value = fake_def
        mgr = JobManager()
        job = mgr.create_job(
            "flux/dev", _make_config(output_dir=str(tmp_path), lora_name="pf"),
        )
        # create_job's own video-contract validation (_apply_video_contract)
        # also resolves the definition; reset so the assertion below isolates
        # start_job's preflight call, which is what this test targets.
        mock_get_def.reset_mock()

        mgr.start_job(job.id)

        mock_get_def.assert_called_once_with("flux/dev")
        mock_prefetch.assert_called_once_with(fake_def)

    @patch("app.engine.utils.model_utils.ModelPathResolver.ensure_definition_cached")
    @patch("app.engine.models.registry.registry.get_definition")
    @patch("app.core.job_manager.plugin_manager")
    def test_start_job_skips_prefetch_on_recovery(self, mock_pm, mock_get_def, mock_prefetch, tmp_path):
        mock_pm.get_plugin.return_value = self._mock_plugin()
        mock_get_def.return_value = self._fake_definition()
        mgr = JobManager()
        job = mgr.create_job(
            "flux/dev", _make_config(output_dir=str(tmp_path), lora_name="recpf"),
        )

        mgr.start_job(job.id, preflight=False)

        mock_prefetch.assert_not_called()

    @patch("app.engine.utils.model_utils.ModelPathResolver.ensure_definition_cached")
    @patch("app.engine.models.registry.registry.get_definition")
    @patch("app.core.job_manager.plugin_manager")
    def test_prefetch_failure_does_not_block_launch(self, mock_pm, mock_get_def, mock_prefetch, tmp_path):
        """A pre-fetch error is best-effort: log and launch anyway (the trainer
        surfaces the real error via the job log as before)."""
        plugin = self._mock_plugin()
        mock_pm.get_plugin.return_value = plugin
        mock_get_def.return_value = self._fake_definition()
        mock_prefetch.side_effect = RuntimeError("hub down")
        mgr = JobManager()
        job = mgr.create_job(
            "flux/dev", _make_config(output_dir=str(tmp_path), lora_name="pferr"),
        )

        mgr.start_job(job.id)

        plugin.start_training.assert_called_once()

    @patch("app.engine.utils.model_utils.ModelPathResolver.ensure_definition_cached")
    @patch("app.engine.models.registry.registry.get_definition")
    @patch("app.core.job_manager.plugin_manager")
    def test_prefetch_failure_updates_status_label_and_broadcasts(
        self, mock_pm, mock_get_def, mock_prefetch, tmp_path,
    ):
        """A pre-fetch failure (e.g. the HF stall guard exhausting its
        retries) must not leave the UI stuck on 'Downloading base model…'
        forever — the label flips to a failure message and broadcasts
        immediately, even though the launch itself proceeds (best-effort:
        the trainer's OWN guarded resolve either succeeds from a partial-but-
        resumable cache or fails the job visibly via the job log)."""
        plugin = self._mock_plugin()
        mock_pm.get_plugin.return_value = plugin
        mock_get_def.return_value = self._fake_definition()
        mock_prefetch.side_effect = RuntimeError(
            "HF download stalled: no progress for 180s on attempt 3/3 "
            "for 'org/repo' — check network/proxy",
        )
        mgr = JobManager()
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        mgr.set_loop(mock_loop)
        job = mgr.create_job(
            "flux/dev", _make_config(output_dir=str(tmp_path), lora_name="pflabel"),
        )

        with patch("app.core.job_manager.asyncio.run_coroutine_threadsafe") as mock_rct:
            mgr.start_job(job.id)

        assert job.status_label == "Base model download failed — trainer will retry"
        # Broadcast on the "Downloading base model…" transition AND again on
        # the failure-label transition — not just the first one.
        assert mock_rct.call_count >= 2
        plugin.start_training.assert_called_once()


class TestStartJobPreflightBroadcastFailureMarksFailed:
    """T3 completion: the preflight status-flip + broadcast sits BEFORE the
    guarded try in start_job (job.status = RUNNING, then an unguarded
    run_coroutine_threadsafe broadcast). A RuntimeError from that broadcast
    (e.g. a closed/stopped event loop during shutdown) or a job.model_dump()
    serialization error previously escaped uncaught, leaving the job
    phantom-RUNNING with pid=None. _reconcile_active_jobs deliberately skips
    pid-less jobs ("possibly mid-launch"), so that phantom wedges the
    single-GPU queue until a backend restart.
    """

    def _mock_plugin(self):
        plugin = MagicMock()
        # No `.pid` → start_job would skip the LogTailer + PID watchdog if it
        # ever got that far (it must not, in this scenario).
        plugin.start_training.return_value = MagicMock(spec=[])
        return plugin

    def _fake_definition(self):
        """spec=ModelDefinition mock with a real registered family — see
        TestStartJobPreflightDownload._fake_definition for why."""
        registry.discover_families()
        fake_def = MagicMock(spec=ModelDefinition)
        fake_def.family = "sdxl"
        fake_def.architecture_params = {}
        fake_def.control_inputs = 0
        fake_def.defaults = {}
        return fake_def

    @patch("app.engine.utils.model_utils.ModelPathResolver.ensure_definition_cached")
    @patch("app.engine.models.registry.registry.get_definition")
    @patch("app.core.job_manager.plugin_manager")
    def test_preflight_broadcast_failure_marks_failed_not_running(
        self, mock_pm, mock_get_def, mock_prefetch, tmp_path,
    ):
        mock_pm.get_plugin.return_value = self._mock_plugin()
        mock_get_def.return_value = self._fake_definition()

        mgr = JobManager()
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        mgr.set_loop(mock_loop)
        job = mgr.create_job(
            "flux/dev", _make_config(output_dir=str(tmp_path), lora_name="pfbcast"),
        )
        # A second pending job — proves below that the failed launch isn't
        # still occupying the single-GPU slot.
        other = mgr.create_job(
            "flux/dev", _make_config(output_dir=str(tmp_path), lora_name="pfbcast-2"),
        )

        with patch(
            "app.core.job_manager.asyncio.run_coroutine_threadsafe",
        ) as mock_rct:
            mock_rct.side_effect = RuntimeError("Event loop is closed")
            with pytest.raises(RuntimeError):
                mgr.start_job(job.id)

        assert job.status == JobStatus.FAILED
        assert job.pid is None
        mock_pm.get_plugin.return_value.start_training.assert_not_called()

        assert mgr._claim_next_pending() == other.id


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


class TestAutoResumeOnGpuFault:
    """A transient GPU device fault (TDR / ``GpuRcReset`` → ``cudaErrorUnknown``)
    kills the trainer process but leaves valid checkpoints. Rather than stranding
    the run as FAILED, the manager auto-resumes it from the latest resumable
    checkpoint (bounded so a deterministic bug can't crash-loop forever)."""

    def _job_with_checkpoint(self, mgr, tmp_path, step=2750):
        job = mgr.create_job("krea2/raw", _make_config())
        job.status = JobStatus.RUNNING
        run = tmp_path / "run"
        ckpt = run / f"checkpoint-{step:06d}"
        ckpt.mkdir(parents=True)
        (ckpt / "training_state.json").write_text("{}")
        return job, run

    def _patches(self, mgr, run):
        return (
            patch.object(mgr, "_get_job_output_dir", return_value=str(run)),
            patch.object(mgr, "_schedule_auto_resume"),
            patch.object(mgr, "schedule_advance_queue"),
            patch.object(mgr, "_persist_status"),
            patch.object(mgr, "_stop_tailer"),
        )

    def test_cuda_unknown_error_schedules_auto_resume(self, tmp_path):
        mgr = JobManager()
        job, run = self._job_with_checkpoint(mgr, tmp_path)
        p_dir, p_sched, p_adv, p_ps, p_st = self._patches(mgr, run)
        with p_dir, p_sched as sched, p_adv as adv, p_ps, p_st:
            mgr._handle_exit_message(
                job.id,
                {"code": 1, "error": "CUDA error: unknown error\ncudaErrorUnknown"},
            )
        sched.assert_called_once_with(job.id, "checkpoint-002750")
        adv.assert_not_called()  # GPU reserved for the imminent resume
        assert job.status == JobStatus.FAILED

    def test_illegal_memory_access_schedules_auto_resume(self, tmp_path):
        """``cudaErrorIllegalAddress`` is the other face of post-TDR context
        death (seen 2026-07-07: 1033 nvlddmkm GpuRcReset events, then the
        trainer died in ``loss.backward()`` with this string)."""
        mgr = JobManager()
        job, run = self._job_with_checkpoint(mgr, tmp_path)
        p_dir, p_sched, p_adv, p_ps, p_st = self._patches(mgr, run)
        with p_dir, p_sched as sched, p_adv as adv, p_ps, p_st:
            mgr._handle_exit_message(
                job.id,
                {
                    "code": 1,
                    "error": "CUDA error: an illegal memory access was "
                    "encountered\nSearch for `cudaErrorIllegalAddress' in ...",
                },
            )
        sched.assert_called_once_with(job.id, "checkpoint-002750")
        adv.assert_not_called()
        assert job.status == JobStatus.FAILED

    def test_non_gpu_error_does_not_auto_resume(self, tmp_path):
        mgr = JobManager()
        job, run = self._job_with_checkpoint(mgr, tmp_path)
        p_dir, p_sched, p_adv, p_ps, p_st = self._patches(mgr, run)
        with p_dir, p_sched as sched, p_adv as adv, p_ps, p_st:
            mgr._handle_exit_message(
                job.id, {"code": 1, "error": "ValueError: bad tensor shape"}
            )
        sched.assert_not_called()
        adv.assert_called_once()

    def test_oom_does_not_auto_resume(self, tmp_path):
        """OOM will just OOM again on resume — not a transient fault."""
        mgr = JobManager()
        job, run = self._job_with_checkpoint(mgr, tmp_path)
        p_dir, p_sched, p_adv, p_ps, p_st = self._patches(mgr, run)
        with p_dir, p_sched as sched, p_adv as adv, p_ps, p_st:
            mgr._handle_exit_message(
                job.id, {"code": 1, "error": "CUDA out of memory. Tried to allocate 4 GiB"}
            )
        sched.assert_not_called()
        adv.assert_called_once()

    def test_no_resumable_checkpoint_does_not_auto_resume(self, tmp_path):
        mgr = JobManager()
        job = mgr.create_job("krea2/raw", _make_config())
        job.status = JobStatus.RUNNING
        empty = tmp_path / "empty"
        empty.mkdir()
        p_dir, p_sched, p_adv, p_ps, p_st = self._patches(mgr, empty)
        with p_dir, p_sched as sched, p_adv as adv, p_ps, p_st:
            mgr._handle_exit_message(
                job.id, {"code": 1, "error": "CUDA error: unknown error"}
            )
        sched.assert_not_called()
        adv.assert_called_once()

    def test_stall_budget_gives_up_after_repeated_no_progress(self, tmp_path):
        """Crashing at the SAME checkpoint step (no forward progress) exhausts
        the stall budget and falls back to a plain FAILED + queue advance."""
        mgr = JobManager()
        job, run = self._job_with_checkpoint(mgr, tmp_path, step=2750)
        p_dir, p_sched, p_adv, p_ps, p_st = self._patches(mgr, run)
        err = {"code": 1, "error": "cudaErrorUnknown"}
        with p_dir, p_sched as sched, p_adv as adv, p_ps, p_st:
            # Each crash re-enters from RUNNING (a real resume would set RUNNING).
            for _ in range(mgr._AUTO_RESUME_MAX_STALL + 1):
                job.status = JobStatus.RUNNING
                mgr._handle_exit_message(job.id, err)
        # Scheduled while stall budget remained, then gave up on the last crash.
        assert sched.call_count == mgr._AUTO_RESUME_MAX_STALL
        assert adv.call_count == 1

    def test_progress_resets_stall_budget(self, tmp_path):
        """A resume that reaches a NEW checkpoint resets the stall counter, so a
        flaky-but-progressing run keeps auto-resuming."""
        mgr = JobManager()
        job, run = self._job_with_checkpoint(mgr, tmp_path, step=2750)
        p_dir, p_sched, p_adv, p_ps, p_st = self._patches(mgr, run)
        err = {"code": 1, "error": "cudaErrorUnknown"}
        with p_dir, p_sched as sched, p_adv as adv, p_ps, p_st:
            job.status = JobStatus.RUNNING
            mgr._handle_exit_message(job.id, err)  # crash @2750 → resume
            # Simulate forward progress: a newer checkpoint appears.
            newck = run / "checkpoint-003000"
            newck.mkdir()
            (newck / "training_state.json").write_text("{}")
            job.status = JobStatus.RUNNING
            mgr._handle_exit_message(job.id, err)  # crash @3000 → resume again
        assert sched.call_count == 2
        assert sched.call_args_list[-1][0] == (job.id, "checkpoint-003000")
        adv.assert_not_called()

    def test_disabled_setting_skips_auto_resume(self, tmp_path):
        mgr = JobManager()
        job, run = self._job_with_checkpoint(mgr, tmp_path)
        p_dir, p_sched, p_adv, p_ps, p_st = self._patches(mgr, run)
        with p_dir, p_sched as sched, p_adv as adv, p_ps, p_st, \
                patch.object(mgr, "_auto_resume_enabled", return_value=False):
            mgr._handle_exit_message(
                job.id, {"code": 1, "error": "CUDA error: unknown error"}
            )
        sched.assert_not_called()
        adv.assert_called_once()


class _SyncTimer:
    """Drop-in for ``threading.Timer`` that runs the callback inline on
    ``start()`` (no real thread, no cooldown wall-clock) so the REAL ``_fire``
    body executes deterministically under test."""

    def __init__(self, interval, function, *args, **kwargs):
        self.function = function
        self.daemon = False

    def start(self):
        self.function()


class TestAutoResumeFire:
    """Exercise the ``_fire`` callback scheduled by ``_schedule_auto_resume``
    itself (the existing TestAutoResumeOnGpuFault suite mocks
    ``_schedule_auto_resume`` wholesale, so ``_fire``'s user-intervention cancel
    guard and its resume-failure → queue-advance fallback never run).

    ``threading.Timer`` is replaced with a synchronous stand-in so the real
    ``_fire`` runs inline; ``resume_from_checkpoint`` / ``schedule_advance_queue``
    are patched only to OBSERVE the seam (and to inject a failure), never to
    stub ``_fire`` internals."""

    def _failed_job(self, mgr):
        job = mgr.create_job("krea2/raw", _make_config())
        job.status = JobStatus.FAILED  # terminal → auto-resume eligible
        return job

    def test_fire_cancels_when_user_intervened(self):
        """If the job left the auto-resume-eligible (terminal) state during the
        cooldown — user stop/relaunch — ``_fire`` stands down: no resume."""
        mgr = JobManager()
        job = self._failed_job(mgr)
        job.status = JobStatus.STOPPED  # user intervened while cooling down
        with patch("app.core.job_manager.threading.Timer", _SyncTimer), \
                patch.object(mgr, "resume_from_checkpoint") as resume, \
                patch.object(mgr, "schedule_advance_queue") as adv:
            mgr._schedule_auto_resume(job.id, "checkpoint-002750")
        resume.assert_not_called()
        adv.assert_not_called()  # the intervening path owns the queue now

    def test_fire_happy_path_attempts_resume(self):
        """Still terminal at fire time → ``_fire`` relaunches from checkpoint."""
        mgr = JobManager()
        job = self._failed_job(mgr)
        with patch("app.core.job_manager.threading.Timer", _SyncTimer), \
                patch.object(mgr, "resume_from_checkpoint") as resume, \
                patch.object(mgr, "schedule_advance_queue") as adv:
            mgr._schedule_auto_resume(job.id, "checkpoint-002750")
        resume.assert_called_once_with(job.id, "checkpoint-002750")
        adv.assert_not_called()  # the resumed run reclaims the GPU

    def test_fire_resume_failure_falls_back_to_advance(self):
        """If the resume itself blows up, ``_fire`` must not strand the queue —
        it falls back to ``schedule_advance_queue``."""
        mgr = JobManager()
        job = self._failed_job(mgr)
        with patch("app.core.job_manager.threading.Timer", _SyncTimer), \
                patch.object(
                    mgr, "resume_from_checkpoint",
                    side_effect=RuntimeError("relaunch exploded"),
                ) as resume, \
                patch.object(mgr, "schedule_advance_queue") as adv:
            mgr._schedule_auto_resume(job.id, "checkpoint-002750")
        resume.assert_called_once()
        adv.assert_called_once()

    def test_fire_noop_when_job_deleted_during_cooldown(self):
        """Job removed during the cooldown → ``_fire`` finds nothing and exits
        without resuming or advancing."""
        mgr = JobManager()
        job = self._failed_job(mgr)
        job_id = job.id
        mgr.delete_job(job_id)
        with patch("app.core.job_manager.threading.Timer", _SyncTimer), \
                patch.object(mgr, "resume_from_checkpoint") as resume, \
                patch.object(mgr, "schedule_advance_queue") as adv:
            mgr._schedule_auto_resume(job_id, "checkpoint-002750")
        resume.assert_not_called()
        adv.assert_not_called()


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
        MockRepo.return_value.list_by_statuses.return_value = []
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
        MockRepo.return_value.list_by_statuses.return_value = []
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


class TestStaleActiveReconciliation:
    """A job stuck RUNNING/PAUSED whose trainer process is gone (dead, or a
    reused PID now running something else) must not permanently block the
    single-GPU queue. The guard treats any RUNNING/PAUSED job as "GPU busy", so
    one phantom wedges every start until a *container* restart resets in-memory
    state. Reconciling liveness on the advance/restart paths lets the queue
    self-heal: the phantom is marked FAILED and stops blocking.
    """

    # ── process-identity check (defeats PID reuse) ───────────────────────

    def test_is_trainer_process_false_for_no_pid(self):
        mgr = JobManager()
        assert mgr._is_trainer_process(None, "job-1") is False
        assert mgr._is_trainer_process(0, "job-1") is False

    @patch("psutil.Process")
    def test_is_trainer_process_true_when_cmdline_matches(self, MockProc):
        # The trainer is launched as `run_trainer.py --config '{... job_id ...}'`,
        # so both markers appear in its cmdline.
        MockProc.return_value.cmdline.return_value = [
            "/usr/bin/python", "-u", "/app/backend/run_trainer.py",
            "--definition_id", "flux/dev", "--config", '{"job_id": "job-xyz"}',
        ]
        mgr = JobManager()
        assert mgr._is_trainer_process(4321, "job-xyz") is True

    @patch("psutil.Process")
    def test_is_trainer_process_false_for_reused_pid(self, MockProc):
        # A recycled PID now running an unrelated process — must NOT be mistaken
        # for our trainer.
        MockProc.return_value.cmdline.return_value = ["/bin/bash", "-c", "sleep 1"]
        mgr = JobManager()
        assert mgr._is_trainer_process(4321, "job-xyz") is False

    @patch("psutil.Process")
    def test_is_trainer_process_false_for_other_job_id(self, MockProc):
        # Our trainer, but for a different job — not THIS job's process.
        MockProc.return_value.cmdline.return_value = [
            "python", "run_trainer.py", "--config", '{"job_id": "other-job"}',
        ]
        mgr = JobManager()
        assert mgr._is_trainer_process(4321, "job-xyz") is False

    @patch("psutil.Process", side_effect=Exception("NoSuchProcess"))
    def test_is_trainer_process_false_when_pid_dead(self, _MockProc):
        mgr = JobManager()
        assert mgr._is_trainer_process(4321, "job-xyz") is False

    # ── reconciliation ───────────────────────────────────────────────────

    def test_reconcile_demotes_phantom_running_job(self):
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING
        job.pid = 5555  # set, but not a live trainer (mocked below)
        with patch.object(mgr, "_is_trainer_process", return_value=False), \
                patch.object(mgr, "_persist_status"), patch.object(mgr, "_stop_tailer"):
            reconciled = mgr._reconcile_active_jobs()
        assert reconciled == [job.id]
        assert job.status == JobStatus.FAILED
        assert job.pid is None
        assert job.finished_at is not None

    def test_reconcile_demotes_phantom_paused_job(self):
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.PAUSED
        job.pid = 5555
        with patch.object(mgr, "_is_trainer_process", return_value=False), \
                patch.object(mgr, "_persist_status"), patch.object(mgr, "_stop_tailer"):
            mgr._reconcile_active_jobs()
        assert job.status == JobStatus.FAILED

    def test_reconcile_keeps_live_trainer(self):
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING
        job.pid = 5555
        with patch.object(mgr, "_is_trainer_process", return_value=True), \
                patch.object(mgr, "_persist_status"), patch.object(mgr, "_stop_tailer"):
            reconciled = mgr._reconcile_active_jobs()
        assert reconciled == []
        assert job.status == JobStatus.RUNNING

    def test_reconcile_ignores_pidless_job(self):
        """A RUNNING job with no PID may be mid-launch — don't demote it (avoids
        racing start_job's status→pid window). Only PID-bearing phantoms are
        reconciled at runtime."""
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config())
        job.status = JobStatus.RUNNING
        job.pid = None
        with patch.object(mgr, "_is_trainer_process", return_value=False) as chk, \
                patch.object(mgr, "_persist_status"), patch.object(mgr, "_stop_tailer"):
            reconciled = mgr._reconcile_active_jobs()
        assert reconciled == []
        assert job.status == JobStatus.RUNNING
        chk.assert_not_called()  # short-circuits before the (slow) psutil probe

    # ── self-heal at the decision points ─────────────────────────────────

    def test_advance_queue_self_heals_phantom_then_starts_pending(self):
        """The wedge fix: a phantom RUNNING job is reconciled at advance time so
        the pending job starts — no container restart needed."""
        mgr = JobManager()
        phantom = mgr.create_job("flux/dev", _make_config(lora_name="phantom"))
        phantom.status = JobStatus.RUNNING
        phantom.pid = 5555
        pending = mgr.create_job("flux/dev", _make_config(lora_name="next"))
        pending.status = JobStatus.PENDING
        with patch.object(mgr, "_auto_queue_enabled", return_value=True), \
                patch.object(mgr, "_is_trainer_process", return_value=False), \
                patch.object(mgr, "_persist_status"), patch.object(mgr, "_stop_tailer"), \
                patch.object(mgr, "start_job") as ms:
            mgr.advance_queue()
        assert phantom.status == JobStatus.FAILED
        ms.assert_called_once_with(pending.id)

    def test_advance_queue_keeps_live_running_job_blocking(self):
        """Reconciliation must NOT demote a genuinely-running trainer — the
        single-GPU guard still blocks."""
        mgr = JobManager()
        running = mgr.create_job("flux/dev", _make_config(lora_name="live"))
        running.status = JobStatus.RUNNING
        running.pid = 5555
        pending = mgr.create_job("flux/dev", _make_config(lora_name="next"))
        pending.status = JobStatus.PENDING
        with patch.object(mgr, "_auto_queue_enabled", return_value=True), \
                patch.object(mgr, "_is_trainer_process", return_value=True), \
                patch.object(mgr, "start_job") as ms:
            mgr.advance_queue()
        assert running.status == JobStatus.RUNNING
        ms.assert_not_called()

    @patch.object(JobManager, "start_job")
    def test_restart_self_heals_phantom_and_launches(self, mock_start):
        """Restarting a failed job while a phantom blocks the GPU must reconcile
        the phantom and launch — not stay queued behind a dead process."""
        mgr = JobManager()
        phantom = mgr.create_job("flux/dev", _make_config(lora_name="phantom"))
        phantom.status = JobStatus.RUNNING
        phantom.pid = 5555
        failed = mgr.create_job("flux/dev", _make_config(lora_name="done"))
        failed.status = JobStatus.FAILED
        with patch.object(mgr, "_is_trainer_process", return_value=False), \
                patch.object(mgr, "_reset_job_log_state"), \
                patch.object(mgr, "_persist_status"), patch.object(mgr, "_stop_tailer"):
            mgr.restart_job(failed.id)
        assert phantom.status == JobStatus.FAILED
        mock_start.assert_called_once_with(failed.id)

    # ── startup hydration uses the identity check ────────────────────────

    @patch("app.core.db.repositories.job_repo.JobHistoryRepository")
    def test_load_demotes_running_with_reused_pid(self, MockRepo):
        """A running job whose stored PID is alive but is NOT our trainer (PID
        reuse / an unrelated orphan) must be demoted, not kept RUNNING."""
        MockRepo.return_value.list_by_statuses.return_value = []
        MockRepo.return_value.list_recent.return_value = [{
            "id": "db-reused", "definition_id": "flux/dev", "config": {},
            "status": "running", "created_at": 1.0, "started_at": 1.0,
            "finished_at": None, "error": None, "pid": 4321,
        }]
        mgr = JobManager()
        with patch.object(mgr, "_is_trainer_process", return_value=False):
            mgr.load_from_db()
        assert mgr.get_job("db-reused").status == JobStatus.STOPPED

    @patch("app.core.db.repositories.job_repo.JobHistoryRepository")
    def test_load_keeps_running_when_pid_is_our_trainer(self, MockRepo):
        """An orphaned-but-alive trainer (our run_trainer for this job) is kept
        RUNNING and queued for re-attach — training survives a server restart."""
        MockRepo.return_value.list_by_statuses.return_value = []
        MockRepo.return_value.list_recent.return_value = [{
            "id": "db-live", "definition_id": "flux/dev", "config": {},
            "status": "running", "created_at": 1.0, "started_at": 1.0,
            "finished_at": None, "error": None, "pid": 4321,
        }]
        mgr = JobManager()
        with patch.object(mgr, "_is_trainer_process", return_value=True):
            mgr.load_from_db()
        assert mgr.get_job("db-live").status == JobStatus.RUNNING
        assert any(r["id"] == "db-live" for r in mgr._recovery_jobs)


class TestResumeFromCheckpoint:
    """resume_from_checkpoint reuses the SAME job, setting resume_from_checkpoint."""

    def _make_resumable(self, mgr, tmp_path, lora="resume_run"):
        job = mgr.create_job("flux/dev", _make_config(output_dir=str(tmp_path), lora_name=lora))
        job.status = JobStatus.STOPPED
        ckpt = tmp_path / f"{lora}_dev" / "checkpoint-000500"
        ckpt.mkdir(parents=True)
        (ckpt / "training_state.json").write_text("{}", encoding="utf-8")
        return job, ckpt

    def test_resume_sets_config_and_relaunches_same_job(self, tmp_path):
        mgr = JobManager()
        job, ckpt = self._make_resumable(mgr, tmp_path)
        with patch.object(mgr, "start_job") as mock_start, \
                patch.object(mgr, "_stop_tailer"), \
                patch.object(mgr, "_reset_job_log_state"), \
                patch.object(mgr, "_persist_status"), \
                patch.object(mgr, "_persist_config") as mock_pc:
            mgr.resume_from_checkpoint(job.id, "checkpoint-000500")

        assert os.path.normpath(job.config["resume_from_checkpoint"]) == os.path.normpath(str(ckpt))
        assert job.config["use_cached_latents"] is True
        assert job.config["use_cached_embeddings"] is True
        assert job.status == JobStatus.PENDING
        mock_start.assert_called_once_with(job.id)
        mock_pc.assert_called_once()
        # No new queue item — the same record is reused.
        assert len(mgr.list_jobs()) == 1

    def test_resume_rejects_non_resumable_dir(self, tmp_path):
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config(output_dir=str(tmp_path), lora_name="r2"))
        job.status = JobStatus.STOPPED
        # Folder exists but has no training_state.json.
        (tmp_path / "r2_dev" / "checkpoint-000100").mkdir(parents=True)
        with pytest.raises(ValueError, match="not resumable"):
            mgr.resume_from_checkpoint(job.id, "checkpoint-000100")

    def test_resume_rejects_bad_dir_name(self, tmp_path):
        mgr = JobManager()
        job = mgr.create_job("flux/dev", _make_config(output_dir=str(tmp_path), lora_name="r3"))
        job.status = JobStatus.STOPPED
        with pytest.raises(ValueError, match="Invalid checkpoint"):
            mgr.resume_from_checkpoint(job.id, "../secrets")

    def test_resume_unknown_job_raises(self):
        mgr = JobManager()
        with pytest.raises(ValueError, match="not found"):
            mgr.resume_from_checkpoint("nope", "checkpoint-000500")

    def test_resume_running_job_raises(self, tmp_path):
        mgr = JobManager()
        job, _ = self._make_resumable(mgr, tmp_path, lora="r4")
        job.status = JobStatus.RUNNING
        with pytest.raises(ValueError, match="Cannot resume"):
            mgr.resume_from_checkpoint(job.id, "checkpoint-000500")

