"""Training job lifecycle manager.

Orchestrates job creation, subprocess launching, log streaming,
pause/resume/stop signals, and event broadcasting.  Runs as a
singleton ``job_manager`` instance.
"""

from __future__ import annotations

import asyncio
import json as _json
import os
import signal
import threading
import time

import structlog

from typing import Any

from app.core.events import event_manager
from app.core.job import Job, JobStatus
from app.core.log_tailer import LogTailer, LOG_FILENAME
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
        # Active log tailers keyed by job_id
        self._tailers: dict[str, LogTailer] = {}

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
                            # Store for post-load re-attachment of LogTailer + watchdog
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
        - ``running`` / ``paused`` with live PID: Process survived — re-attach
          a LogTailer so we resume streaming and start a PID watchdog.
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
                    # Keep the pause signal we just wrote — don't let start_job
                    # clear it, that's how the trainer knows to pause after
                    # loading the latest checkpoint.
                    self.start_job(job_id, clear_stale_signal=False)

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
                # Process alive — re-attach LogTailer + PID watchdog
                pid = entry.get("pid")
                output_dir = self._get_job_output_dir(job)
                log_path = os.path.join(output_dir, LOG_FILENAME)

                logger.info(
                    "reattaching_orphaned_job",
                    job_id=job_id,
                    status=entry["status"],
                    pid=pid,
                    log_path=log_path,
                )

                # Re-attach LogTailer (resumes from saved offset)
                tailer = LogTailer(job_id, log_path, self._dispatch_log_entry)
                tailer.start()
                self._tailers[job_id] = tailer

                # Start PID watchdog
                self._start_pid_watchdog(job_id, pid)

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

        loop = self._loop
        if loop is not None:
            from app.core.events import emit_entity_change
            asyncio.run_coroutine_threadsafe(
                emit_entity_change(
                    event_manager.broadcast,
                    entity="job",
                    op="created",
                    id=job.id,
                    payload=job.model_dump(),
                ),
                loop,
            )

        return job

    def list_jobs(self) -> list[Job]:
        """Return all jobs sorted by creation time (newest first)."""
        with self._lock:
            return sorted(list(self._jobs.values()), key=lambda x: x.created_at, reverse=True)

    def reorder_pending(self, job_id: str, direction: str) -> None:
        """Move a pending job up/down in the run queue.

        Reassigns the in-memory ``priority`` of all pending jobs to reflect the
        new order (lower priority = runs sooner). Edge moves are no-ops. Not
        persisted — order reverts to FIFO-by-created_at on a server restart.
        """
        if direction not in ("up", "down"):
            raise ValueError(f"Invalid direction: {direction}")
        with self._lock:
            pending = sorted(
                (j for j in self._jobs.values() if j.status == JobStatus.PENDING),
                key=lambda j: (j.priority, j.created_at),
            )
            idx = next((i for i, j in enumerate(pending) if j.id == job_id), -1)
            if idx == -1:
                raise ValueError("Pending job not found")
            swap = idx - 1 if direction == "up" else idx + 1
            if swap < 0 or swap >= len(pending):
                return  # already at an edge
            pending[idx], pending[swap] = pending[swap], pending[idx]
            for order, j in enumerate(pending):
                j.priority = order

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
            import psutil
            return psutil.pid_exists(pid)
        except Exception:
            return False

    def delete_job(self, job_id: str) -> None:
        """Remove a job from the registry and the database. Broadcasts entity.changed."""
        self._stop_tailer(job_id)
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
        self._persist_delete(job_id)

        loop = self._loop
        if loop is not None:
            from app.core.events import emit_entity_change
            asyncio.run_coroutine_threadsafe(
                emit_entity_change(
                    event_manager.broadcast,
                    entity="job",
                    op="deleted",
                    id=job_id,
                ),
                loop,
            )

    # ── Log Tailer Dispatcher ────────────────────────────────────────

    def _dispatch_log_entry(self, job_id: str, entry: dict[str, Any]) -> None:
        """Route a parsed log entry from ``LogTailer`` to the appropriate handler.

        Called from the tailer's background thread for each JSON line
        read from ``job_log.jsonl``.
        """
        msg_type = entry.get("type", "")
        data = entry.get("data")
        timestamp = entry.get("t", time.time())

        job = self.get_job(job_id)
        if not job:
            return

        loop = self._loop
        if not loop:
            return

        if msg_type == "log":
            with self._lock:
                job.logs.append(str(data))
                if len(job.logs) > 1000:
                    job.logs.pop(0)
            # Reflect a trainer-side signal pause/resume in the live job status.
            # The trainer pauses itself whenever it reads a pause signal — which
            # may NOT have come from an explicit pause_job() (e.g. a leftover
            # signal). Without this, the job would sit as RUNNING/"Training"
            # with no GPU load and the user would have no way to resume it.
            self._reconcile_signal_pause(job_id, str(data))
            asyncio.run_coroutine_threadsafe(
                event_manager.broadcast("job_log", {
                    "job_id": job_id,
                    "message": str(data),
                    "timestamp": timestamp,
                }),
                loop,
            )

        elif msg_type == "status":
            with self._lock:
                job.status_label = str(data)
            asyncio.run_coroutine_threadsafe(
                event_manager.broadcast("job_update", job.model_dump()),
                loop,
            )

        elif msg_type == "step":
            # Forward step metrics to WebSocket for real-time chart updates
            asyncio.run_coroutine_threadsafe(
                event_manager.broadcast("job_log", {
                    "job_id": job_id,
                    "message": _json.dumps(data) if isinstance(data, dict) else str(data),
                    "timestamp": timestamp,
                }),
                loop,
            )

        elif msg_type == "warning":
            with self._lock:
                job.warnings.append(str(data))
            asyncio.run_coroutine_threadsafe(
                event_manager.broadcast("job_warning", {
                    "job_id": job_id,
                    "message": str(data),
                    "timestamp": timestamp,
                }),
                loop,
            )

        elif msg_type == "cache_ready":
            self._handle_cache_ready(data, loop)

        # (pause/resume reconciliation handled inside the "log" branch above)

        elif msg_type == "exit":
            self._handle_exit_message(job_id, data)

    def _reconcile_signal_pause(self, job_id: str, message: str) -> None:
        """Sync job status with trainer-side pause/resume signal log events.

        The trainer logs ``training_paused_by_signal`` when it blocks on a pause
        signal and ``training_resumed_by_signal`` when it continues. We mirror
        that into ``job.status`` so the live status is authoritative regardless
        of how the pause arose (explicit pause_job, crash-recovery, or a stale
        leftover signal) — and so the user can always Resume a paused run.

        Only flips between RUNNING and PAUSED; never overrides a terminal state
        (stopped/completed/failed) that may have raced in.
        """
        if "training_paused_by_signal" in message:
            target = JobStatus.PAUSED
        elif "training_resumed_by_signal" in message:
            target = JobStatus.RUNNING
        else:
            return

        job = self.get_job(job_id)
        if not job:
            return
        with self._lock:
            if job.status == target or job.status not in (JobStatus.RUNNING, JobStatus.PAUSED):
                return
            job.status = target
            job.paused_at = time.time() if target == JobStatus.PAUSED else None

        self._persist_status(job_id, "paused" if target == JobStatus.PAUSED else "running")
        loop = self._loop
        if loop:
            asyncio.run_coroutine_threadsafe(
                event_manager.broadcast("job_update", job.model_dump()),
                loop,
            )

    def _handle_cache_ready(self, ds_names: Any, loop: asyncio.AbstractEventLoop) -> None:
        """Mark datasets as cache-bearing after trainer reports readiness."""
        try:
            from app.core.dataset_manager import dataset_manager as dm
            if isinstance(ds_names, list):
                dm.set_loop(loop)
                dm.mark_cache_created(ds_names)
        except Exception as e:
            logger.warning("cache_ready_handle_error", error=str(e))

    def _handle_exit_message(self, job_id: str, data: Any) -> None:
        """Process an exit message from the trainer subprocess."""
        job = self.get_job(job_id)
        if not job:
            return

        code = data.get("code", 1) if isinstance(data, dict) else 1
        error = data.get("error") if isinstance(data, dict) else None

        # Diagnostic: if the trainer reported exit but its PID or any
        # descendants are still alive, training work is still running
        # (e.g. orphaned DataLoader workers, async CUDA cleanup) and we
        # are about to remove the job from the active queue while the
        # GPU is still busy.  Log loudly so the next repro pinpoints the
        # leaking subsystem.
        pid = job.pid
        if pid:
            try:
                import psutil
                if psutil.pid_exists(pid):
                    proc = psutil.Process(pid)
                    children = proc.children(recursive=True)
                    if proc.is_running() or children:
                        logger.warning(
                            "exit_message_while_process_alive",
                            job_id=job_id,
                            pid=pid,
                            exit_code=code,
                            parent_running=proc.is_running(),
                            child_pids=[c.pid for c in children],
                        )
            except Exception:
                pass

        with self._lock:
            job.finished_at = time.time()
            job.pid = None
            if code == 0:
                job.status = JobStatus.COMPLETED
            elif job.status != JobStatus.STOPPED:
                job.status = JobStatus.FAILED
                job.error = error or f"Process exited with code {code}"

        final_status = "completed" if code == 0 else (
            "stopped" if job.status == JobStatus.STOPPED else "failed"
        )
        kwargs: dict[str, Any] = {"finished_at": job.finished_at}
        if job.error:
            kwargs["error"] = job.error
        self._persist_status(job_id, final_status, **kwargs)

        self._stop_tailer(job_id)

        if self._loop:
            asyncio.run_coroutine_threadsafe(
                event_manager.broadcast("job_update", job.model_dump()),
                self._loop,
            )

    # ── PID Watchdog ─────────────────────────────────────────────────

    def _start_pid_watchdog(self, job_id: str, pid: int) -> None:
        """Poll a PID periodically; finalize the job if the process dies.

        This is a safety net for cases where the trainer exits without
        writing an ``exit`` message to the log file (e.g. OOM kill, segfault).
        """

        def _watch() -> None:
            while True:
                time.sleep(5)
                if not self._is_pid_alive(pid):
                    job = self.get_job(job_id)
                    if not job:
                        break
                    if job.status in (JobStatus.RUNNING, JobStatus.PAUSED):
                        # Give the tailer a moment to process any final lines
                        time.sleep(2)
                        # Re-check: the exit message handler may have already updated
                        job = self.get_job(job_id)
                        if job and job.status in (JobStatus.RUNNING, JobStatus.PAUSED):
                            logger.warning(
                                "pid_watchdog_process_died",
                                job_id=job_id,
                                pid=pid,
                            )
                            with self._lock:
                                job.status = JobStatus.STOPPED
                                job.finished_at = time.time()
                                job.pid = None
                                job.error = "Process exited unexpectedly (detected by watchdog)"
                            self._persist_status(
                                job_id, "stopped",
                                finished_at=job.finished_at,
                                error=job.error,
                            )
                            self._stop_tailer(job_id)
                            if self._loop:
                                asyncio.run_coroutine_threadsafe(
                                    event_manager.broadcast("job_update", job.model_dump()),
                                    self._loop,
                                )
                    break

        thread = threading.Thread(
            target=_watch,
            daemon=True,
            name=f"pid_watchdog_{job_id[:8]}",
        )
        thread.start()

    # ── Tailer Management ────────────────────────────────────────────

    def _stop_tailer(self, job_id: str) -> None:
        """Stop and remove the LogTailer for a job (if active)."""
        tailer = self._tailers.pop(job_id, None)
        if tailer:
            tailer.stop()

    # ── Job Control ──────────────────────────────────────────────────

    def start_job(self, job_id: str, clear_stale_signal: bool = True) -> None:
        """Launch the training subprocess for a job.

        Args:
            job_id: The job to launch.
            clear_stale_signal: When true (the default), remove any leftover
                ``signal.json`` from the job's output directory before
                launching. Output dirs are keyed by ``lora_name`` +
                ``definition_id``, so a new job can reuse a directory whose
                previous occupant left an unconsumed pause/soft_stop signal —
                the fresh trainer would then read it on its first check and
                block in the pause loop (status "Training", no GPU load, no
                steps). The crash-recovery ``relaunch_paused`` path passes
                ``False`` because it *intentionally* writes a pause signal
                immediately before calling start_job.
        """
        job = self.get_job(job_id)
        if not job:
            raise ValueError("Job not found")

        if job.status == JobStatus.RUNNING:
            raise ValueError("Job already running")

        plugin = plugin_manager.get_plugin(job.plugin_id)
        if not plugin:
            raise ValueError(f"Plugin {job.plugin_id} not found")

        if clear_stale_signal:
            from app.engine.components.signal_manager import TrainingSignalManager
            TrainingSignalManager(self._get_job_output_dir(job)).clear_signal()

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

                # Start file-based log tailing
                output_dir = self._get_job_output_dir(job)
                log_path = os.path.join(output_dir, LOG_FILENAME)

                tailer = LogTailer(job_id, log_path, self._dispatch_log_entry)
                tailer.start()
                self._tailers[job_id] = tailer

                # Start PID watchdog as safety net
                self._start_pid_watchdog(job_id, process.pid)

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
                self._stop_tailer(job_id)
            except ProcessLookupError:
                job.status = JobStatus.FAILED
                job.error = "Process not found"
                self._persist_status(job_id, "failed", error=job.error)
                self._stop_tailer(job_id)
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

    def _delete_job_output_dir(self, job: Job) -> None:
        """Delete a run's output folder (for a fresh restart). No-op if absent."""
        import shutil

        output_dir = self._get_job_output_dir(job)
        if output_dir and os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
            logger.info("deleted_job_output_dir", job_id=job.id, output_dir=output_dir)

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
        # Don't change status yet — the process will exit and the exit handler will update

    def restart_job(self, job_id: str, fresh: bool = False) -> None:
        """Reset a finished/failed job and re-launch it.

        When ``fresh`` is true, the run's output folder is deleted first so the
        restart starts from a clean slate (no prior checkpoints/samples/logs);
        otherwise the existing output is reused.
        """
        job = self.get_job(job_id)
        if not job:
            raise ValueError("Job not found")

        if job.status in [JobStatus.RUNNING, JobStatus.PENDING]:
            raise ValueError(f"Cannot restart job in state {job.status}")

        logger.info("restarting_job", job_id=job_id, fresh=fresh)

        if fresh:
            self._delete_job_output_dir(job)

        self._stop_tailer(job_id)
        self._reset_job_log_state(job)

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
            # Append behind any jobs already pending — the restart shouldn't
            # jump the queue just because its created_at is from the original
            # run (priority sorts ahead of created_at).
            max_priority = max(
                (j.priority for j in self._jobs.values()
                 if j.status == JobStatus.PENDING and j.id != job_id),
                default=-1,
            )
            job.priority = max_priority + 1
            # The GPU runs one job at a time. If another job is already
            # training (or paused, holding VRAM), leave this restart queued.
            gpu_busy = any(
                j.status in (JobStatus.RUNNING, JobStatus.PAUSED)
                for j in self._jobs.values()
            )
        self._persist_status(job_id, "pending", error=None)

        if gpu_busy:
            # Stay queued; auto-queue (or a manual Start) launches it when the
            # GPU frees up. Broadcast so the UI moves it from Archive into the
            # pending queue rather than (wrongly) starting it concurrently.
            logger.info("restart_queued_behind_active_job", job_id=job_id)
            if self._loop:
                asyncio.run_coroutine_threadsafe(
                    event_manager.broadcast("job_update", job.model_dump()),
                    self._loop,
                )
            return

        self.start_job(job_id)

    def _reset_job_log_state(self, job: Job) -> None:
        """Rotate the job's log file + drop its tailer offset.

        The trainer writes ``job_log.jsonl`` in append mode, and the
        tailer persists its byte offset in ``job_log.jsonl.offset``.
        On restart of a finished/failed job, those files contain the
        previous run's terminal ``exit`` message — if the next tailer
        starts from a stale or pre-exit offset (which happens across a
        backend restart, where the in-memory tailer state is gone but
        the on-disk offset survives) it re-dispatches that exit and
        immediately marks the restarted job FAILED with the prior
        run's error. Rotate the log (so the previous run's logs stay
        available for forensics) and delete the offset so the new
        tailer starts at byte 0 of an empty file.
        """
        output_dir = self._get_job_output_dir(job)
        log_path = os.path.join(output_dir, LOG_FILENAME)
        offset_path = log_path + ".offset"

        if os.path.exists(log_path):
            ts = time.strftime("%Y%m%d-%H%M%S")
            rotated = os.path.join(output_dir, f"job_log.{ts}.jsonl")
            try:
                os.replace(log_path, rotated)
                logger.info(
                    "job_log_rotated_for_restart",
                    job_id=job.id, rotated_to=rotated,
                )
            except OSError as e:
                logger.warning(
                    "job_log_rotate_failed",
                    job_id=job.id, error=str(e),
                )

        if os.path.exists(offset_path):
            try:
                os.remove(offset_path)
            except OSError as e:
                logger.warning(
                    "job_log_offset_remove_failed",
                    job_id=job.id, error=str(e),
                )

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
