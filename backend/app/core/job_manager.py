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
from app.core.naming import model_part_from_definition_id
from app.core.log_tailer import LogTailer, LOG_FILENAME
from app.core.plugin_manager import plugin_manager

logger = structlog.get_logger(__name__)


def _parse_persisted_log_lines(log_path: str, limit: int = 1000) -> list[str]:
    """Reconstruct a job's display log tail from its persisted ``job_log.jsonl``.

    Lets a stopped/failed job (or one whose in-memory buffer was lost to a
    backend restart) still show its tail. Returns the last *limit* renderable
    lines: ``log`` data verbatim, ``warning``/``exit`` flagged inline. Per-step
    metric entries are skipped — parity with the in-memory buffer, which only
    stores ``log`` lines.
    """
    if not os.path.isfile(log_path):
        return []
    out: list[str] = []
    try:
        with open(log_path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = _json.loads(raw)
                except ValueError:
                    continue
                msg_type = entry.get("type")
                data = entry.get("data")
                if msg_type == "log":
                    out.append(str(data))
                elif msg_type == "warning":
                    out.append(f"[WARNING] {data}")
                elif msg_type == "exit" and isinstance(data, dict) and data.get("code"):
                    err = data.get("error") or f"exited with code {data.get('code')}"
                    out.append(f"[EXIT] {err}")
    except OSError:
        pass
    return out[-limit:]


_MIRROR_LEVEL = {"log": "info", "warning": "warning"}


def _trainer_msg_for_server_log(msg_type: str, data: Any) -> tuple[str, str] | None:
    """Map a trainer job-log entry to a ``(level, text)`` pair for the SERVER log.

    Returns ``None`` for entry types that would only flood the server log
    (per-step metrics, UI status labels, cache-ready, clean exits). The trainer
    is a separate process, so mirroring these into the main logger is what makes
    them visible in the Server log viewer (and its download) and persisted in
    ``server.log``.
    """
    if msg_type in _MIRROR_LEVEL:
        return (_MIRROR_LEVEL[msg_type], str(data))
    if msg_type == "exit" and isinstance(data, dict) and data.get("code"):
        err = data.get("error") or f"exited with code {data.get('code')}"
        return ("error", f"trainer exited: {err}")
    return None


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
        # Guard: a queue-advance launch is in flight (status hasn't flipped to
        # RUNNING yet). Prevents two terminal events from each starting a job.
        self._starting: bool = False

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

                    # For running/paused jobs, check if OUR trainer subprocess
                    # survived. Identity-match (not bare pid_exists) so a reused
                    # PID — or an unrelated orphan — can't masquerade as a live
                    # trainer and keep the job RUNNING, wedging the queue.
                    if status_str in ("running", "paused"):
                        if self._is_trainer_process(stored_pid, row["id"]):
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
                        priority=row.get("priority") or 0,
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
                    # loading the latest checkpoint. preflight=False: a resumed
                    # run is already cached, and recovery may run on the event
                    # loop where a blocking pre-fetch download would be unsafe.
                    self.start_job(
                        job_id, clear_stale_signal=False, preflight=False,
                    )

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
                "priority": job.priority,
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

        Reassigns the ``priority`` of all pending jobs to reflect the new order
        (lower priority = runs sooner) and persists it, so the arrangement
        survives a backend restart (priority is the primary run-order key;
        ``created_at`` is only the FIFO tiebreaker). Edge moves are no-ops.
        """
        if direction not in ("up", "down"):
            raise ValueError(f"Invalid direction: {direction}")
        new_priorities: list[tuple[str, int]] = []
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
                new_priorities.append((j.id, order))

        # Persist outside the lock (disk I/O) so the manual order survives a
        # restart instead of reverting to FIFO-by-created_at.
        for jid, priority in new_priorities:
            self._persist_priority(jid, priority)

    def get_job(self, job_id: str) -> Job | None:
        """Look up a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    # States whose config may be edited: pending (the edit changes what the job
    # will run) or any terminal state (edits the historical record). Running and
    # paused jobs are locked — their config is in flight.
    _CONFIG_EDITABLE_STATES = {"pending", "completed", "failed", "stopped"}

    def update_job_config(self, job_id: str, new_config: dict[str, Any]) -> dict[str, Any]:
        """Edit a job's stored training config.

        For a pending job the in-memory copy is updated too, so the upcoming
        run (``start_job`` reads ``job.config``) uses the edited values. Running
        and paused jobs are rejected. Returns the refreshed DB row.
        """
        from app.core.db.repositories.job_repo import JobHistoryRepository
        repo = JobHistoryRepository()

        job = self.get_job(job_id)
        if job is not None:
            status = job.status.value if hasattr(job.status, "value") else str(job.status)
        else:
            row = repo.get_by_id(job_id)
            if not row:
                raise ValueError("Job not found")
            status = str(row.get("status"))

        if status not in self._CONFIG_EDITABLE_STATES:
            raise ValueError(f"Cannot edit the config of a {status} job")

        # Keep the self-reference the rest of the pipeline relies on.
        new_config = dict(new_config)
        new_config["job_id"] = job_id

        if job is not None:
            with self._lock:
                job.config = new_config

        repo.update_config(job_id, new_config)

        loop = self._loop
        if loop is not None:
            from app.core.events import emit_entity_change
            asyncio.run_coroutine_threadsafe(
                emit_entity_change(
                    event_manager.broadcast,
                    entity="job",
                    op="updated",
                    id=job_id,
                ),
                loop,
            )

        return repo.get_by_id(job_id) or {}

    # ── DB persistence helpers ───────────────────────────────────────

    def _persist_status(self, job_id: str, status: str, **kwargs) -> None:
        """Sync a status change to the database (fire-and-forget)."""
        try:
            from app.core.db.repositories.job_repo import JobHistoryRepository
            JobHistoryRepository().update_status(job_id, status=status, **kwargs)
        except Exception as e:
            logger.warning("persist_status_failed", job_id=job_id, error=str(e))

    def _persist_priority(self, job_id: str, priority: int) -> None:
        """Sync a pending job's run-queue priority to the database."""
        try:
            from app.core.db.repositories.job_repo import JobHistoryRepository
            JobHistoryRepository().set_priority(job_id, priority)
        except Exception as e:
            logger.warning("persist_priority_failed", job_id=job_id, error=str(e))

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

    @staticmethod
    def _is_trainer_process(pid: int | None, job_id: str) -> bool:
        """True iff *pid* is a live process that is THIS job's trainer subprocess.

        Stronger than a bare ``pid_exists``: the trainer is launched as
        ``run_trainer.py --config '{… "job_id": <id> …}'`` (see
        ``StandardPlugin.start_training``), so both ``run_trainer`` and the job
        id appear verbatim in its cmdline. Matching on them defeats PID reuse —
        a recycled PID now running an unrelated process won't carry our markers,
        so a dead trainer can't be mistaken for a live one and wedge the
        single-GPU queue across a restart.
        """
        if not pid or pid <= 0:
            return False
        try:
            import psutil
            cmdline = " ".join(psutil.Process(int(pid)).cmdline())
        except Exception:
            return False
        return "run_trainer" in cmdline and job_id in cmdline

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

        # Mirror trainer (side-process) messages into the MAIN server log so
        # they appear in the Server log viewer + its download and persist in
        # server.log. Independent of the broadcast loop below: even with no WS
        # clients, the file write still happens. Noise types (step/status) are
        # filtered out by _trainer_msg_for_server_log.
        mirror = _trainer_msg_for_server_log(msg_type, data)
        if mirror:
            level, text = mirror
            getattr(logger, level)(
                "trainer_message", job_id=job_id, source="lora-worker", msg=text,
            )

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

        # A run that ended on its own (completed or failed) frees the GPU —
        # advance the queue. A user hard-stop leaves status STOPPED here and is
        # intentionally skipped: stopping is a deliberate intervention.
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            self.schedule_advance_queue()

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
                            # An unexpected death still frees the GPU — don't
                            # let one crashed run stall the whole overnight
                            # queue.
                            self.schedule_advance_queue()
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

    def _preflight_download(self, job: Job) -> None:
        """Download the job's base model in-process before launching the trainer.

        Best-effort progress UX only — runs the model resolve through the
        WS-emitting ``_resolve_hf`` path so the top-bar download indicator
        updates (the detached trainer subprocess can't emit those events). The
        subprocess then loads from the warm HF cache. Called from ``start_job``
        on a worker thread (off the event loop), so the blocking download is
        safe and the WS emits schedule onto the captured loop.

        Failures are swallowed: the pre-fetch must never block or fail a launch
        that the trainer could otherwise complete — the trainer re-resolves the
        model and surfaces any real error through the job log as before.
        """
        try:
            from app.engine.models.registry import registry
            from app.engine.utils.model_utils import ModelPathResolver

            definition_id = job.config.get("definition_id") or job.plugin_id
            definition = registry.get_definition(definition_id)
            if definition is None:
                return
            ModelPathResolver.ensure_definition_cached(definition)
        except Exception as e:
            logger.warning(
                "preflight_download_failed", job_id=job.id, error=str(e),
            )

    def start_job(
        self, job_id: str, clear_stale_signal: bool = True, preflight: bool = True,
    ) -> None:
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
            preflight: When true (the default), download the base model in this
                (API) process before launching, so its progress reaches the
                top-bar download indicator — the trainer subprocess can't emit
                those WS events. The crash-recovery ``relaunch_paused`` path
                passes ``False``: a resumed run is already cached, and recovery
                may run on the event loop where a blocking download is unsafe.
        """
        job = self.get_job(job_id)
        if not job:
            raise ValueError("Job not found")

        if job.status == JobStatus.RUNNING:
            raise ValueError("Job already running")

        plugin = plugin_manager.get_plugin(job.plugin_id)
        if not plugin:
            raise ValueError(f"Plugin {job.plugin_id} not found")

        if preflight:
            # Reflect the prerequisite base-model download in the job's LIVE
            # status so the queue card + top bar show real work in progress
            # instead of an idle "pending" while a large model fetches. This is
            # an in-memory transition, broadcast immediately. We deliberately do
            # NOT persist "running" here: the download has no trainer/PID yet, so
            # leaving the DB row "pending" means a restart mid-download cleanly
            # resumes from pending — vs. a stranded "running" row that recovery
            # would mark stopped. ``_preflight_download`` then emits the HF
            # download progress onto the top-bar indicator.
            job.status = JobStatus.RUNNING
            job.status_label = "Downloading base model…"
            if job.started_at is None:
                job.started_at = time.time()
            if self._loop:
                asyncio.run_coroutine_threadsafe(
                    event_manager.broadcast("job_update", job.model_dump()),
                    self._loop,
                )
            self._preflight_download(job)

        if clear_stale_signal:
            from app.engine.components.signal_manager import TrainingSignalManager
            TrainingSignalManager(self._get_job_output_dir(job)).clear_signal()

        try:
            process = plugin.start_training(job.config)

            job.status = JobStatus.RUNNING
            if job.started_at is None:
                job.started_at = time.time()
            # Base model is cached now; hand the status line back to the trainer
            # (it emits its own "Loading"/"Training"/… labels). Clearing avoids a
            # stale "Downloading base model…" lingering until the first trainer
            # status message arrives.
            job.status_label = None
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

    # ── Auto-Queue (backend-owned queue advancement) ─────────────────

    @staticmethod
    def _auto_queue_enabled() -> bool:
        """Read the persisted ``jobs.auto_queue`` preference (default off).

        Server-side so the queue advances unattended — independent of whether
        any browser/Jobs tab is open. Cheap: settings are cached in-memory by
        the settings manager.
        """
        try:
            from app.core.settings_manager import get_settings_manager
            mod = get_settings_manager().get_module_settings("jobs")
            return bool(mod.get("auto_queue", False))
        except Exception as e:
            logger.warning("auto_queue_setting_read_failed", error=str(e))
            return False

    def _claim_next_pending(self) -> str | None:
        """Atomically pick the next pending job to launch, or None.

        Returns None (taking no action) when a launch is already in flight, a
        job is already RUNNING/PAUSED (single-GPU rule — PAUSED still holds
        VRAM), or there are no pending jobs. On success, sets the in-flight
        guard so a concurrent caller can't also launch.
        """
        with self._lock:
            if self._starting:
                return None
            if any(
                j.status in (JobStatus.RUNNING, JobStatus.PAUSED)
                for j in self._jobs.values()
            ):
                return None
            pending = sorted(
                (j for j in self._jobs.values() if j.status == JobStatus.PENDING),
                key=lambda j: (j.priority, j.created_at),
            )
            if not pending:
                return None
            self._starting = True
            return pending[0].id

    def _reconcile_active_jobs(self) -> list[str]:
        """Demote any RUNNING/PAUSED job whose trainer process is gone to FAILED.

        The single-GPU guard treats every RUNNING/PAUSED job as "GPU busy" and
        refuses to launch the next one. If a trainer dies without the exit
        handler or PID watchdog catching it — e.g. an orphaned subprocess left
        alive after a server restart, or a PID that was reused — the job is
        stuck "active" forever and blocks every start until a *container*
        restart clears the in-memory state. Re-checking real liveness here lets
        the queue self-heal: a phantom is marked FAILED and stops blocking.

        Only PID-bearing phantoms are reconciled. A RUNNING job with no PID may
        be mid-launch (``start_job`` sets status before PID), so we leave it
        alone to avoid racing that window. Returns the reconciled job ids.
        """
        with self._lock:
            active = [
                j for j in self._jobs.values()
                if j.status in (JobStatus.RUNNING, JobStatus.PAUSED)
            ]
        reconciled: list[str] = []
        for job in active:
            if job.pid is None:
                continue  # possibly mid-launch — trust the status
            if self._is_trainer_process(job.pid, job.id):
                continue  # genuinely our live trainer
            stale_pid = job.pid
            with self._lock:
                job.status = JobStatus.FAILED
                job.finished_at = time.time()
                job.pid = None
                job.error = job.error or (
                    "Training process is no longer running (recovered)."
                )
            self._persist_status(
                job.id, "failed", finished_at=job.finished_at, error=job.error,
            )
            self._stop_tailer(job.id)
            reconciled.append(job.id)
            logger.warning(
                "reconciled_stale_active_job", job_id=job.id, stale_pid=stale_pid,
            )
            if self._loop:
                asyncio.run_coroutine_threadsafe(
                    event_manager.broadcast("job_update", job.model_dump()),
                    self._loop,
                )
        return reconciled

    def advance_queue(self) -> None:
        """Start the next pending job if auto-queue is on and the GPU is idle.

        The single source of truth for queue advancement. Called on every
        terminal transition (job exit, watchdog death), at startup after
        recovery, and when the auto-queue toggle is switched on. Safe to call
        from any thread. A no-op when auto-queue is disabled, a job is already
        RUNNING/PAUSED, a launch is in flight, or no pending jobs remain.

        If launching a job fails, the queue does not stall: the failed job is
        skipped (``start_job`` marks it FAILED) and the next pending job is
        tried.
        """
        if not self._auto_queue_enabled():
            return
        # Self-heal first: drop any phantom "active" job (dead/orphaned trainer)
        # so it stops blocking the single-GPU guard before we pick what to run.
        self._reconcile_active_jobs()
        while True:
            next_id = self._claim_next_pending()
            if next_id is None:
                return
            try:
                logger.info("auto_queue_advancing", job_id=next_id)
                self.start_job(next_id)
                return  # one job is now launching/running — done
            except Exception as e:
                # start_job already marked the job FAILED; drop the guard and
                # try the next pending job on the loop's next pass.
                logger.error("auto_queue_start_failed", job_id=next_id, error=str(e))
            finally:
                with self._lock:
                    self._starting = False

    def schedule_advance_queue(self) -> None:
        """Run ``advance_queue`` on a short-lived daemon thread.

        Terminal-transition callers run on the log-tailer or watchdog threads;
        launching the next job there would block them on a multi-second model
        load. Offloading keeps those threads responsive. The in-flight guard +
        single-GPU check make concurrent invocations safe.
        """
        threading.Thread(
            target=self.advance_queue,
            daemon=True,
            name="advance_queue",
        ).start()

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

    def read_persisted_logs(self, job: Job, limit: int = 1000) -> list[str]:
        """Read a job's log tail from its persisted ``job_log.jsonl`` on disk.

        The source of truth for a finished job's logs: survives the in-memory
        buffer being cleared and backend restarts.
        """
        log_path = os.path.join(self._get_job_output_dir(job), LOG_FILENAME)
        return _parse_persisted_log_lines(log_path, limit=limit)

    def _get_job_output_dir(self, job: Job) -> str:
        """Resolve the output directory for a job from its config.

        Must match the path logic in the trainers:
        All families: {output_root}/{lora_name}_{definition_id}
        """
        output_dir = job.config.get("output_dir", "outputs")
        lora_name = job.config.get("lora_name", "untitled")
        definition_id = job.config.get("definition_id", "")
        model_part = model_part_from_definition_id(definition_id)
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

        # Self-heal any phantom "active" job (dead/orphaned trainer) so a stale
        # RUNNING/PAUSED entry can't make this restart queue behind a process
        # that no longer exists.
        self._reconcile_active_jobs()

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
        # Persist the new pending status AND the recomputed priority so a
        # restart-queued run keeps its place behind existing pending jobs even
        # across a backend restart.
        self._persist_status(job_id, "pending", error=None, priority=job.priority)

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
