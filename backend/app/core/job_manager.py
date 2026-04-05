"""Training job lifecycle manager.

Orchestrates job creation, subprocess launching, log streaming,
pause/resume/stop signals, and event broadcasting.  Runs as a
singleton ``job_manager`` instance.
"""

from __future__ import annotations

import asyncio
import os
import signal
import threading
import time

import structlog

from typing import Any

from app.core.events import event_manager
from app.core.job import Job, JobStatus
from app.core.plugin_manager import plugin_manager

logger = structlog.get_logger(__name__)


class JobManager:
    """Manages the lifecycle of training jobs (create → run → finish)."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        # Reference to the main event loop for scheduling async broadcasts from threads
        self._loop: asyncio.AbstractEventLoop | None = None
        # Jobs needing post-startup recovery (re-launch paused, re-attach alive)
        self._recovery_jobs: list[dict] = []

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Store the main event loop for cross-thread broadcasts."""
        self._loop = loop

    def load_from_db(self) -> None:
        """Hydrate the in-memory job registry from the SQLite job_history table.

        Called once at startup so previously-created jobs survive backend
        restarts.  Jobs that were ``running`` or ``paused`` at shutdown are
        demoted to ``stopped`` because the training subprocess is no longer
        alive.
        """
        try:
            from app.core.db.repositories.job_repo import JobHistoryRepository

            repo = JobHistoryRepository()
            rows = repo.list_recent(limit=200, include_active=True)

            loaded = 0
            with self._lock:
                for row in rows:
                    if row["id"] in self._jobs:
                        continue  # Don't overwrite live jobs

                    status_str = row.get("status", "pending")
                    stored_pid = row.get("pid")

                    # For running/paused jobs, check if the subprocess survived
                    if status_str in ("running", "paused"):
                        if self._is_pid_alive(stored_pid):
                            # Process survived the restart — keep status
                            logger.info(
                                "job_process_still_alive",
                                job_id=row["id"],
                                pid=stored_pid,
                                status=status_str,
                            )
                            # We can't re-attach stdout, but the process is alive.
                            # Store in _orphaned_jobs for post-load re-attachment.
                            self._recovery_jobs.append({
                                "id": row["id"],
                                "status": status_str,
                                "pid": stored_pid,
                            })
                        else:
                            # Process is dead
                            if status_str == "paused":
                                # Paused jobs should be re-launched and re-paused
                                logger.info(
                                    "paused_job_process_dead_will_relaunch",
                                    job_id=row["id"],
                                    pid=stored_pid,
                                )
                                self._recovery_jobs.append({
                                    "id": row["id"],
                                    "status": "relaunch_paused",
                                })
                                status_str = "pending"  # Load as pending, will re-launch later
                            else:
                                # Running job whose process died → stopped
                                status_str = "stopped"
                                try:
                                    repo.update_status(row["id"], status="stopped", error="Interrupted by system restart")
                                except Exception as e:
                                    logger.warning("failed_to_update_interrupted_job_status", error=str(e))

                    # Resolve plugin_id: DB stores definition_id which may
                    # be the HF model path (legacy) or 'standard' (correct).
                    raw_def_id = row.get("definition_id", "")
                    resolved_plugin_id = raw_def_id if raw_def_id in ("standard",) else "standard"

                    self._jobs[row["id"]] = Job(
                        id=row["id"],
                        plugin_id=resolved_plugin_id,
                        config=row.get("config") or {},
                        status=JobStatus(status_str),
                        created_at=row.get("created_at", 0),
                        started_at=row.get("started_at"),
                        finished_at=row.get("finished_at"),
                        error=row.get("error"),
                        pid=stored_pid,
                    )
                    loaded += 1

            logger.info("jobs_loaded_from_db", count=loaded)
        except Exception as e:
            logger.warning("jobs_load_from_db_failed", error=str(e))

    def recover_jobs(self) -> None:
        """Post-startup recovery for jobs whose state needs action.

        Must be called AFTER ``load_from_db()`` and ``set_loop()`` so that
        the event loop and plugin manager are ready.

        Handles:
        - ``relaunch_paused``: Process died while paused → re-launch then
          immediately write a pause signal file so the trainer pauses
          after loading the latest checkpoint.
        - ``running`` / ``paused`` with live PID: Process survived but we
          can't re-attach stdout.  Status is kept as-is; real-time logs
          are lost but the training continues.
        """
        from app.engine.components.signal_manager import TrainingSignalManager

        for entry in self._recovery_jobs:
            job_id = entry["id"]
            job = self.get_job(job_id)
            if not job:
                continue

            if entry["status"] == "relaunch_paused":
                try:
                    # Write pause signal BEFORE launching so the trainer
                    # sees it on first check after startup.
                    output_dir = self._get_job_output_dir(job)
                    TrainingSignalManager.send_signal(output_dir, "pause")

                    logger.info("relaunching_paused_job", job_id=job_id)
                    self.start_job(job_id)

                    # After start_job the status is RUNNING; set it to PAUSED
                    # because the trainer will block on the pause signal.
                    with self._lock:
                        job.status = JobStatus.PAUSED
                        job.paused_at = time.time()
                    self._persist_status(job_id, "paused")

                    if self._loop:
                        asyncio.run_coroutine_threadsafe(
                            event_manager.broadcast("job_update", job.model_dump()),
                            self._loop,
                        )
                except Exception as e:
                    logger.error("relaunch_paused_job_failed", job_id=job_id, error=str(e))
                    with self._lock:
                        job.status = JobStatus.STOPPED
                        job.error = f"Failed to re-launch after restart: {e}"
                    self._persist_status(job_id, "stopped", error=job.error)

            elif entry["status"] in ("running", "paused"):
                # Process alive but we can't re-attach stdout.
                logger.info(
                    "orphaned_job_kept",
                    job_id=job_id,
                    status=entry["status"],
                    pid=entry.get("pid"),
                    note="Real-time log streaming unavailable until next restart",
                )

        self._recovery_jobs.clear()

    def create_job(self, plugin_id: str, config: dict[str, Any]) -> Job:
        """Create a new pending job and register it."""
        job = Job.create(plugin_id, config)
        job.config["job_id"] = job.id
        with self._lock:
            self._jobs[job.id] = job

        try:
            from app.core.db.repositories.job_repo import JobHistoryRepository
            repo = JobHistoryRepository()
            
            payload = {
                "id": job.id,
                "project_id": config.get("project_id"),
                "lora_name": config.get("lora_name", ""),
                "definition_id": config.get("definition_id") or plugin_id,
                "status": "pending",
                "config": job.config,
                "created_at": job.created_at,
            }
            
            datasets = config.get("datasets", [])
            if datasets:
                payload["datasets_config"] = [
                    {
                        "dataset_name": ds.get("dataset_name", ""),
                        "dataset_id": ds.get("dataset_id"),
                        "num_repeats": ds.get("num_repeats", 1),
                        "masking_enabled": ds.get("masking_enabled", False),
                        "caption_dropout": float(ds.get("caption_dropout_rate", 0)),
                    }
                    for ds in datasets
                ]
            repo.create(payload)
        except Exception as e:
            logger.warning("jobs_create_db_failed", error=str(e))

        return job

    def list_jobs(self) -> list[Job]:
        """Return all jobs sorted by creation time (newest first)."""
        with self._lock:
            return sorted(list(self._jobs.values()), key=lambda x: x.created_at, reverse=True)

    def get_job(self, job_id: str) -> Job | None:
        """Look up a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    # ── DB persistence helpers ───────────────────────────────────────

    def _persist_status(self, job_id: str, status: str, **kwargs) -> None:
        """Sync a status change to the database (fire-and-forget)."""
        try:
            from app.core.db.repositories.job_repo import JobHistoryRepository
            JobHistoryRepository().update_status(job_id, status=status, **kwargs)
        except Exception as e:
            logger.warning("persist_status_failed", job_id=job_id, error=str(e))

    def _persist_delete(self, job_id: str) -> None:
        """Remove a job record from the database."""
        try:
            from app.core.db.repositories.job_repo import JobHistoryRepository
            JobHistoryRepository().delete(job_id)
        except Exception as e:
            logger.warning("persist_delete_failed", job_id=job_id, error=str(e))

    @staticmethod
    def _is_pid_alive(pid: int | None) -> bool:
        """Check if a process with the given PID is still running."""
        if pid is None or pid <= 0:
            return False
        try:
            if os.name == "nt":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                # PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                handle = kernel32.OpenProcess(0x1000, False, pid)
                if handle:
                    kernel32.CloseHandle(handle)
                    return True
                return False
            else:
                os.kill(pid, 0)
                return True
        except (OSError, PermissionError):
            return False

    def delete_job(self, job_id: str) -> None:
        """Remove a job from the registry and the database."""
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
        self._persist_delete(job_id)

    def start_job(self, job_id: str) -> None:
        """Launch the training subprocess for a job."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError("Job not found")

        if job.status == JobStatus.RUNNING:
            raise ValueError("Job already running")

        plugin = plugin_manager.get_plugin(job.plugin_id)
        if not plugin:
            raise ValueError(f"Plugin {job.plugin_id} not found")

        try:
            process = plugin.start_training(job.config)

            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            self._persist_status(job_id, "running", started_at=job.started_at)

            # Broadcast start
            if self._loop:
                asyncio.run_coroutine_threadsafe(
                    event_manager.broadcast("job_update", job.model_dump()),
                    self._loop
                )

            if hasattr(process, 'pid'):
                job.pid = process.pid
                self._persist_status(job_id, "running", pid=process.pid)

                loop = self._loop

                def log_listener(proc, job_obj, loop):
                    """Stream subprocess stdout to logs and WebSocket."""
                    thread_logger = structlog.get_logger(__name__)
                    thread_logger.debug("log_listener_started", job_id=job_obj.id)

                    for line in iter(proc.stdout.readline, ""):
                        if not line:
                            break
                        clean_line = line.strip()
                        if clean_line:
                            # Parse [CACHE_READY:["ds1","ds2"]] — update dataset has_cache flags
                            if clean_line.startswith("[CACHE_READY:") and clean_line.endswith("]"):
                                try:
                                    import json as _json
                                    from app.core.dataset_manager import dataset_manager as dm
                                    payload = clean_line[len("[CACHE_READY:"):-1]
                                    ds_names = _json.loads(payload)
                                    dm.set_loop(loop)
                                    dm.mark_cache_created(ds_names)
                                except Exception as e:
                                    thread_logger.warning("cache_ready_parse_error", error=str(e))
                                continue

                            # Parse [STATUS:label] markers from trainer
                            if clean_line.startswith("[STATUS:") and clean_line.endswith("]"):
                                label = clean_line[8:-1]
                                with self._lock:
                                    job_obj.status_label = label
                                asyncio.run_coroutine_threadsafe(
                                    event_manager.broadcast("job_update", job_obj.model_dump()),
                                    loop
                                )
                                continue

                            # Parse [WARNING:message] markers from trainer
                            if clean_line.startswith("[WARNING:") and clean_line.endswith("]"):
                                warning_msg = clean_line[9:-1]
                                with self._lock:
                                    job_obj.warnings.append(warning_msg)
                                asyncio.run_coroutine_threadsafe(
                                    event_manager.broadcast("job_warning", {
                                        "job_id": job_obj.id,
                                        "message": warning_msg,
                                        "timestamp": time.time()
                                    }),
                                    loop
                                )
                                continue

                            # Bridge to main server log
                            thread_logger.info("job_log", job_id=job_obj.id, message=clean_line)

                            # Broadcast to WebSocket
                            asyncio.run_coroutine_threadsafe(
                                event_manager.broadcast("job_log", {
                                    "job_id": job_obj.id,
                                    "message": clean_line,
                                    "timestamp": time.time()
                                }),
                                loop
                            )

                            with self._lock:
                                job_obj.logs.append(clean_line)
                                if len(job_obj.logs) > 1000:
                                    job_obj.logs.pop(0)

                    # Process finished
                    proc.stdout.close()
                    if proc.stderr:
                        proc.stderr.close()
                    proc.wait()
                    with self._lock:
                        job_obj.finished_at = time.time()
                        job_obj.pid = None  # Clear PID reference
                        if proc.returncode == 0:
                            job_obj.status = JobStatus.COMPLETED
                            self._persist_status(job_obj.id, "completed", finished_at=job_obj.finished_at)
                        elif job_obj.status != JobStatus.STOPPED:
                            job_obj.status = JobStatus.FAILED
                            job_obj.error = f"Process exited with code {proc.returncode}"
                            self._persist_status(job_obj.id, "failed", finished_at=job_obj.finished_at, error=job_obj.error)

                        asyncio.run_coroutine_threadsafe(
                            event_manager.broadcast("job_update", job_obj.model_dump()),
                            loop
                        )

                thread = threading.Thread(target=log_listener, args=(process, job, loop), daemon=True)
                thread.start()

        except (OSError, ValueError, RuntimeError) as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            self._persist_status(job_id, "failed", error=job.error)
            raise

    def stop_job(self, job_id: str) -> None:
        """Send SIGTERM to the training subprocess."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError("Job not found")

        if job.status in (JobStatus.RUNNING, JobStatus.PAUSED) and job.pid:
            try:
                os.kill(job.pid, signal.SIGTERM)
                job.status = JobStatus.STOPPED
                job.finished_at = time.time()
                self._persist_status(job_id, "stopped", finished_at=job.finished_at)
            except ProcessLookupError:
                job.status = JobStatus.FAILED
                job.error = "Process not found"
                self._persist_status(job_id, "failed", error=job.error)
            except OSError as e:
                job.error = str(e)

    def _get_job_output_dir(self, job: Job) -> str:
        """Resolve the output directory for a job from its config.

        Must match the path logic in the trainers:
        All families: {output_root}/{lora_name}_{definition_id}
        """
        output_dir = job.config.get("output_dir", "outputs")
        lora_name = job.config.get("lora_name", "untitled")
        definition_id = job.config.get("definition_id", "")
        model_part = definition_id.split("/")[-1].replace(":", "_")
        run_name = f"{lora_name}_{model_part}"
        return os.path.join(output_dir, run_name)

    def pause_job(self, job_id: str) -> None:
        """Send pause signal to a running training job."""
        from app.engine.components.signal_manager import TrainingSignalManager

        job = self.get_job(job_id)
        if not job:
            raise ValueError("Job not found")
        if job.status != JobStatus.RUNNING:
            raise ValueError(f"Cannot pause job in state {job.status}")

        output_dir = self._get_job_output_dir(job)
        TrainingSignalManager.send_signal(output_dir, "pause")
        job.status = JobStatus.PAUSED
        job.paused_at = time.time()
        self._persist_status(job_id, "paused")

        if self._loop:
            asyncio.run_coroutine_threadsafe(
                event_manager.broadcast("job_update", job.model_dump()),
                self._loop,
            )

    def resume_job(self, job_id: str) -> None:
        """Send resume signal to a paused training job."""
        from app.engine.components.signal_manager import TrainingSignalManager

        job = self.get_job(job_id)
        if not job:
            raise ValueError("Job not found")
        if job.status != JobStatus.PAUSED:
            raise ValueError(f"Cannot resume job in state {job.status}")

        output_dir = self._get_job_output_dir(job)
        TrainingSignalManager.send_signal(output_dir, "resume")
        job.status = JobStatus.RUNNING
        job.paused_at = None
        self._persist_status(job_id, "running")

        if self._loop:
            asyncio.run_coroutine_threadsafe(
                event_manager.broadcast("job_update", job.model_dump()),
                self._loop,
            )

    def soft_stop_job(self, job_id: str) -> None:
        """Send soft stop signal — trainer saves checkpoint then exits."""
        from app.engine.components.signal_manager import TrainingSignalManager

        job = self.get_job(job_id)
        if not job:
            raise ValueError("Job not found")
        if job.status not in (JobStatus.RUNNING, JobStatus.PAUSED):
            raise ValueError(f"Cannot soft stop job in state {job.status}")

        output_dir = self._get_job_output_dir(job)
        TrainingSignalManager.send_signal(output_dir, "soft_stop")
        # Don't change status yet — the process will exit and log_listener will update

    def restart_job(self, job_id: str) -> None:
        """Reset a finished/failed job and re-launch it."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError("Job not found")

        if job.status in [JobStatus.RUNNING, JobStatus.PENDING]:
            raise ValueError(f"Cannot restart job in state {job.status}")

        logger.info("restarting_job", job_id=job_id)

        with self._lock:
            job.status = JobStatus.PENDING
            job.error = None
            job.pid = None
            job.started_at = None
            job.finished_at = None
            job.logs = []
            job.warnings = []
            job.status_label = None
            job.paused_at = None
        self._persist_status(job_id, "pending", error=None)

        self.start_job(job_id)

    # ── Sampling Pause ──────────────────────────────────────────────────

    SAMPLING_PAUSED_FILENAME = "sampling_paused"

    def _sampling_flag_path(self, job: Job) -> str:
        """Return the path to the sampling_paused flag file for a job."""
        output_dir = self._get_job_output_dir(job)
        return os.path.join(output_dir, self.SAMPLING_PAUSED_FILENAME)

    def pause_sampling(self, job_id: str) -> None:
        """Create a flag file that tells the trainer to skip sampling."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError("Job not found")

        flag_path = self._sampling_flag_path(job)
        os.makedirs(os.path.dirname(flag_path), exist_ok=True)
        with open(flag_path, "w") as f:
            f.write("")
        logger.info("sampling_paused", job_id=job_id, flag_path=flag_path)

    def resume_sampling(self, job_id: str) -> None:
        """Remove the sampling_paused flag file to re-enable sampling."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError("Job not found")

        flag_path = self._sampling_flag_path(job)
        try:
            if os.path.exists(flag_path):
                os.remove(flag_path)
        except OSError:
            pass
        logger.info("sampling_resumed", job_id=job_id)

    def is_sampling_paused(self, job_id: str) -> bool:
        """Check if sampling is paused for a job."""
        job = self.get_job(job_id)
        if not job:
            return False
        return os.path.exists(self._sampling_flag_path(job))

    # ── Sampling Cadence Override ────────────────────────────────────────

    SAMPLING_CADENCE_FILENAME = "sampling_cadence"

    def _sampling_cadence_path(self, job: Job) -> str:
        """Return the path to the sampling_cadence override file for a job."""
        output_dir = self._get_job_output_dir(job)
        return os.path.join(output_dir, self.SAMPLING_CADENCE_FILENAME)

    def set_sampling_cadence(self, job_id: str, interval: int) -> None:
        """Write a cadence override file so the trainer changes its sampling interval at runtime."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError("Job not found")

        cadence_path = self._sampling_cadence_path(job)
        os.makedirs(os.path.dirname(cadence_path), exist_ok=True)
        with open(cadence_path, "w") as f:
            f.write(str(interval))
        logger.info("sampling_cadence_set", job_id=job_id, interval=interval)

    def get_sampling_cadence(self, job_id: str) -> int | None:
        """Read the cadence override file. Returns None if no override is active."""
        job = self.get_job(job_id)
        if not job:
            return None

        cadence_path = self._sampling_cadence_path(job)
        if not os.path.exists(cadence_path):
            return None
        try:
            with open(cadence_path) as f:
                return int(f.read().strip())
        except (ValueError, OSError):
            return None


# Global instance
job_manager = JobManager()
