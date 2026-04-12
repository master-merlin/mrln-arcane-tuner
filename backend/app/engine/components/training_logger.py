
import json
import os
import time
from typing import Any
import structlog

logger = structlog.get_logger(__name__)

# Buffer size before flushing metrics to SQLite
_METRICS_FLUSH_INTERVAL = 50


class TrainingLogger:
    """
    Standardized logger for training progress.
    Logs to structlog (backend) and accumulates loss history for persistence.
    """
    def __init__(
        self,
        max_steps: int,
        output_dir: str | None = None,
        elapsed_offset: float = 0.0,
        step_offset: int = 0,
        signal_manager: object | None = None,
        log_writer: object | None = None,
    ):
        self.max_steps = max_steps
        self.start_time = time.time()
        self.last_step_time = self.start_time
        self.output_dir = output_dir
        self.elapsed_offset = elapsed_offset
        self.step_offset = step_offset
        self.signal_manager = signal_manager
        self.log_writer = log_writer
        self._loss_history: list[dict] = []
        self._last_paused_elapsed: float = 0.0
        # Separate training start from logger creation so prep time
        # (model loading, latent/TE caching) doesn't pollute ETA.
        self._training_start_time: float | None = None
        self._training_paused_at_start: float = 0.0
        # Save time tracking — excludes save duration from step_time / ETA
        self._save_start: float = 0.0
        self._save_times: list[float] = []  # rolling buffer (last 3)
        self._total_save_time: float = 0.0  # cumulative save time this session

        # DB metrics buffer — flushed every N steps
        self._metrics_buffer: list[dict] = []
        self._job_id: str | None = None  # set externally by pipeline

    def pause_step_timer(self) -> None:
        """Call before a checkpoint save to exclude its duration from step_time."""
        self._save_start = time.time()

    def resume_step_timer(self) -> None:
        """Call after a checkpoint save to resume normal step timing."""
        if self._save_start > 0:
            save_duration = time.time() - self._save_start
            # Shift last_step_time forward so next log_step sees clean step_time
            self.last_step_time += save_duration
            self._total_save_time += save_duration
            # Rolling buffer of last 3 save durations
            self._save_times.append(save_duration)
            if len(self._save_times) > 3:
                self._save_times.pop(0)
            self._save_start = 0.0

    @property
    def avg_save_time(self) -> float:
        """Average save duration over last 3 saves (0.0 if no saves yet)."""
        return sum(self._save_times) / len(self._save_times) if self._save_times else 0.0

    def _paused_elapsed(self) -> float:
        """Total time spent paused (from signal manager)."""
        if self.signal_manager and hasattr(self.signal_manager, 'paused_elapsed'):
            return self.signal_manager.paused_elapsed
        return 0.0

    def get_total_elapsed(self) -> float:
        """Return total elapsed time including prior sessions, excluding pauses."""
        return (time.time() - self.start_time) - self._paused_elapsed() + self.elapsed_offset

    def log_step(
        self,
        step: int,
        loss: float,
        lr: float = 0.0,
        extra: dict[str, Any] | None = None
    ):
        """Log a single training step and accumulate loss history."""
        import math
        
        # Sanitize NaN/Inf to prevent invalid JSON breaking the frontend
        def _safe(v: float) -> float:
            return 0.0 if (math.isnan(v) or math.isinf(v)) else v
        
        current_time = time.time()

        # Adjust last_step_time if we just came out of a pause
        current_paused = self._paused_elapsed()
        if current_paused > self._last_paused_elapsed:
            pause_delta = current_paused - self._last_paused_elapsed
            self.last_step_time += pause_delta
            self._last_paused_elapsed = current_paused

        # Record when actual training started (first log_step call)
        # so that prep time doesn't inflate the ETA.
        if self._training_start_time is None:
            self._training_start_time = current_time
            self._training_paused_at_start = current_paused

        step_time = current_time - self.last_step_time
        self.last_step_time = current_time
        
        session_elapsed = (current_time - self.start_time) - current_paused
        total_elapsed = session_elapsed + self.elapsed_offset
        # ETA: use only training time (excludes prep AND save time) for avg step time
        training_elapsed = (current_time - self._training_start_time) - (
            current_paused - self._training_paused_at_start
        ) - self._total_save_time
        session_steps = (step + 1) - self.step_offset
        avg_time_per_step = training_elapsed / session_steps if session_steps > 0 else 0
        remaining_steps = self.max_steps - (step + 1)
        eta = avg_time_per_step * remaining_steps
        # Add projected save overhead for remaining checkpoint saves
        if self._save_times:
            save_every = 0
            # _save_every is set externally by the training loop
            if hasattr(self, '_save_every') and self._save_every > 0:
                save_every = self._save_every
            if save_every > 0:
                remaining_saves = remaining_steps // save_every
                eta += self.avg_save_time * remaining_saves
        
        progress = int(((step + 1) / self.max_steps) * 100)
        progress = min(100, max(0, progress))
        
        safe_loss = _safe(float(loss))
        safe_lr = _safe(float(lr)) if lr is not None else 0.0
        
        log_data = {
            "loss": safe_loss,
            "step": step + 1,
            "progress": progress,
            "learning_rate": safe_lr,
            "status": "training",
            "step_time": round(step_time, 3),
            "elapsed": int(total_elapsed),
            "eta": int(eta)
        }
        
        if extra:
            for k, v in extra.items():
                if isinstance(v, float):
                    extra[k] = _safe(v)
            log_data.update(extra)
            
        # Backend structured log
        logger.info("step_progress", **log_data)

        # File-based IPC: emit step metrics to job_log.jsonl
        if self.log_writer and hasattr(self.log_writer, "step"):
            self.log_writer.step(log_data)

        # Accumulate loss history
        self._loss_history.append({
            "step": step + 1,
            "loss": round(safe_loss, 6),
            "lr": safe_lr,
            "elapsed": int(total_elapsed),
            "timestamp": int(current_time),
        })

        # Buffer metrics for DB flush
        if self._job_id:
            metric = {
                "step": step + 1,
                "loss": round(safe_loss, 6),
                "lr": safe_lr,
                "grad_norm": extra.get("grad_norm") if extra else None,
                "timestep_mean": extra.get("timestep_mean") if extra else None,
                "epoch": extra.get("epoch") if extra else None,
            }
            self._metrics_buffer.append(metric)
            if len(self._metrics_buffer) >= _METRICS_FLUSH_INTERVAL:
                self.flush_metrics()

    def save_loss_history(self, output_dir: str | None = None):
        """
        Persist accumulated loss history to JSON file.
        Path: {output_dir}/loss_history.json
        """
        target_dir = output_dir or self.output_dir
        if not target_dir or not self._loss_history:
            return

        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, "loss_history.json")
        
        try:
            with open(path, "w") as f:
                json.dump(self._loss_history, f, indent=2)
            logger.debug("loss_history_saved", path=path, entries=len(self._loss_history))
        except OSError as e:
            logger.warning("loss_history_save_failed", path=path, error=str(e))

        # Flush remaining metrics to DB
        self.flush_metrics()

    def flush_metrics(self) -> None:
        """Flush buffered step metrics to SQLite."""
        if not self._metrics_buffer or not self._job_id:
            return
        try:
            from app.core.db.repositories.metrics_repo import MetricsRepository
            repo = MetricsRepository()
            repo.batch_insert(self._job_id, self._metrics_buffer)
            self._metrics_buffer.clear()
        except Exception as e:
            logger.warning("metrics_flush_failed", error=str(e))

    @property
    def loss_history(self) -> list[dict]:
        """Return accumulated loss history entries."""
        return self._loss_history
