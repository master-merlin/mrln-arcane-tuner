"""JobHistoryRepository — CRUD for the ``job_history`` + ``job_datasets`` tables.

Provides persistent training run records with full config snapshots,
metrics summaries, and dataset lineage tracking.
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog

from app.core.db.engine import get_db

logger = structlog.get_logger(__name__)


class JobHistoryRepository:
    """Persistent training run history."""

    # ── Reads ────────────────────────────────────────────────────────

    def get_by_id(self, job_id: str) -> dict[str, Any] | None:
        conn = get_db().connection()
        row = conn.execute(
            "SELECT * FROM job_history WHERE id = ?", (job_id,)
        ).fetchone()
        return self._from_row(row) if row else None

    def list_recent(
        self, limit: int = 50, offset: int = 0,
        definition_id: str | None = None,
        status: str | None = None,
        lora_name: str | None = None,
        project_id: str | None = None,
        include_active: bool = False,
    ) -> list[dict[str, Any]]:
        """Paginated job history with optional filters.

        By default, excludes active statuses (pending/running/paused) so
        the archive only shows terminal jobs.  Pass ``include_active=True``
        or an explicit ``status`` to override.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if status:
            clauses.append("status = ?")
            params.append(status)
        elif not include_active:
            # Only show terminal states in history by default
            clauses.append("status NOT IN ('pending', 'running', 'paused')")

        if definition_id:
            clauses.append("definition_id = ?")
            params.append(definition_id)
        if lora_name:
            clauses.append("lora_name LIKE ?")
            params.append(f"%{lora_name}%")
        if project_id:
            clauses.append("project_id = ?")
            params.append(project_id)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT * FROM job_history {where}
            ORDER BY created_at DESC LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        conn = get_db().connection()
        rows = conn.execute(sql, params).fetchall()
        return [self._from_row(r) for r in rows]

    def get_by_lora_name(self, lora_name: str) -> list[dict[str, Any]]:
        conn = get_db().connection()
        rows = conn.execute(
            "SELECT * FROM job_history WHERE lora_name = ? ORDER BY created_at DESC",
            (lora_name,),
        ).fetchall()
        return [self._from_row(r) for r in rows]

    def get_by_dataset(self, dataset_id: str) -> list[dict[str, Any]]:
        """Find all jobs that used a specific dataset."""
        conn = get_db().connection()
        rows = conn.execute("""
            SELECT jh.* FROM job_history jh
            INNER JOIN job_datasets jd ON jd.job_id = jh.id
            WHERE jd.dataset_id = ?
            ORDER BY jh.created_at DESC
        """, (dataset_id,)).fetchall()
        return [self._from_row(r) for r in rows]

    def get_datasets_for_job(self, job_id: str) -> list[dict[str, Any]]:
        """Get dataset linkage records for a job."""
        conn = get_db().connection()
        rows = conn.execute(
            "SELECT * FROM job_datasets WHERE job_id = ?", (job_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self, project_id: str | None = None) -> dict[str, Any]:
        """Aggregate training statistics for the stats modal.

        Read-only. ``project_id`` narrows every aggregate to one project
        (``None`` = global). Legacy ``definition_id = 'standard'`` placeholder
        repair is handled once by the v17 migration, NOT here.
        """
        conn = get_db().connection()

        # Filter fragments: appended to a WHERE clause that always exists
        # (`WHERE 1=1`) so every query has one insertion point. `jflt` is the
        # variant for queries joined through the job_history alias `j`.
        flt = "" if project_id is None else "AND project_id = ?"
        jflt = "" if project_id is None else "AND j.project_id = ?"
        args: tuple[str, ...] = () if project_id is None else (project_id,)

        # ── Core counts ──────────────────────────────────────────
        totals = conn.execute(f"""
            SELECT
                COUNT(*)                                              AS total_jobs,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
                SUM(CASE WHEN status = 'failed'    THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN status = 'stopped'   THEN 1 ELSE 0 END) AS stopped,
                SUM(CASE WHEN status = 'running'   THEN 1 ELSE 0 END) AS running,
                SUM(CASE WHEN status = 'paused'    THEN 1 ELSE 0 END) AS paused,
                COALESCE(SUM(completed_steps), 0)                      AS total_steps,
                COALESCE(SUM(duration_seconds), 0)                     AS total_runtime_sec,
                COALESCE(SUM(training_seconds), 0)                     AS total_training_sec
            FROM job_history WHERE 1=1 {flt}
        """, args).fetchone()

        total_jobs = totals["total_jobs"] or 0
        completed  = totals["completed"] or 0

        # ── Averages (completed only) ────────────────────────────
        avgs = conn.execute(f"""
            SELECT
                AVG(completed_steps) AS avg_steps,
                AVG(avg_loss)        AS avg_loss,
                AVG(min_loss)        AS avg_min_loss,
                AVG(avg_step_time)   AS avg_step_time_sec,
                AVG(duration_seconds) AS avg_runtime_sec
            FROM job_history WHERE status = 'completed' {flt}
        """, args).fetchone()

        # ── Model family breakdown ───────────────────────────────
        families = conn.execute(f"""
            SELECT definition_id, COUNT(*) AS count
            FROM job_history WHERE 1=1 {flt}
            GROUP BY definition_id
            ORDER BY count DESC
        """, args).fetchall()

        # ── Optimizer breakdown ──────────────────────────────────
        optimizers = conn.execute(f"""
            SELECT optimizer_type, COUNT(*) AS count
            FROM job_history
            WHERE optimizer_type IS NOT NULL {flt}
            GROUP BY optimizer_type
            ORDER BY count DESC
        """, args).fetchall()

        # ── Dataset usage ────────────────────────────────────────
        dataset_stats = conn.execute(f"""
            SELECT COUNT(DISTINCT jd.dataset_name) AS unique_datasets
            FROM job_datasets jd JOIN job_history j ON jd.job_id = j.id
            WHERE 1=1 {jflt}
        """, args).fetchone()

        # ── Most recent job ──────────────────────────────────────
        last_job = conn.execute(f"""
            SELECT lora_name, definition_id, status, created_at
            FROM job_history WHERE 1=1 {flt}
            ORDER BY created_at DESC LIMIT 1
        """, args).fetchone()

        return {
            "total_jobs": total_jobs,
            "completed": completed,
            "failed": totals["failed"] or 0,
            "stopped": totals["stopped"] or 0,
            "running": totals["running"] or 0,
            "paused": totals["paused"] or 0,
            "success_rate": round(completed / total_jobs * 100, 1) if total_jobs > 0 else 0,
            "total_steps": totals["total_steps"],
            "total_runtime_sec": round(totals["total_runtime_sec"], 1),
            "total_training_sec": round(totals["total_training_sec"], 1),
            "avg_steps": round(avgs["avg_steps"] or 0),
            "avg_loss": round(avgs["avg_loss"] or 0, 6),
            "avg_min_loss": round(avgs["avg_min_loss"] or 0, 6),
            "avg_step_time_sec": round(avgs["avg_step_time_sec"] or 0, 3),
            "avg_runtime_sec": round(avgs["avg_runtime_sec"] or 0, 1),
            "model_families": [
                {"id": r["definition_id"], "count": r["count"]}
                for r in families
            ],
            "optimizers": [
                {"name": r["optimizer_type"], "count": r["count"]}
                for r in optimizers
            ],
            "unique_datasets": dataset_stats["unique_datasets"] or 0,
            "last_job": dict(last_job) if last_job else None,
        }

    # ── Writes ───────────────────────────────────────────────────────

    def create(self, data: dict[str, Any]) -> None:
        """Insert a new job history record."""
        data = self._prepare(data)
        cols = [k for k in data if k != "datasets_config"]
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO job_history ({', '.join(cols)}) VALUES ({placeholders})"

        datasets_config = data.pop("datasets_config", None)

        with get_db().write() as conn:
            conn.execute(sql, [data[c] for c in cols])

            # Insert job_datasets linkage
            if datasets_config:
                for ds_cfg in datasets_config:
                    conn.execute("""
                        INSERT INTO job_datasets
                        (job_id, dataset_id, dataset_name, dataset_version,
                         num_repeats, masking_enabled, caption_dropout)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        data["id"],
                        ds_cfg.get("dataset_id"),
                        ds_cfg.get("dataset_name", ""),
                        ds_cfg.get("dataset_version", "1.0.0"),
                        ds_cfg.get("num_repeats", 1),
                        int(ds_cfg.get("masking_enabled", False)),
                        ds_cfg.get("caption_dropout", 0.0),
                    ))

    def update_status(self, job_id: str, status: str, **kwargs) -> None:
        """Update job status and optional fields."""
        updates = {"status": status}
        updates.update(kwargs)
        self._update(job_id, updates)

    def complete(self, job_id: str, **kwargs) -> None:
        """Mark a job as completed with final metrics."""
        updates = {
            "status": "completed",
            "finished_at": time.time(),
        }
        updates.update(kwargs)
        self._update(job_id, updates)

    def fail(self, job_id: str, error: str, **kwargs) -> None:
        """Mark a job as failed."""
        updates = {
            "status": "failed",
            "finished_at": time.time(),
            "error": error,
        }
        updates.update(kwargs)
        self._update(job_id, updates)

    def update_progress(self, job_id: str, completed_steps: int, **kwargs) -> None:
        """Update step progress (called on each checkpoint)."""
        updates = {"completed_steps": completed_steps}
        updates.update(kwargs)
        self._update(job_id, updates)

    def set_priority(self, job_id: str, priority: int) -> None:
        """Persist a pending job's run-queue priority (no status change)."""
        self._update(job_id, {"priority": int(priority)})

    def update_config(self, job_id: str, config: dict[str, Any]) -> None:
        """Replace a job's stored config and refresh the config-derived display
        columns (lora_name / definition_id / project_id) so the queue + history
        labels stay consistent with the edited config. Run-result columns
        (loss, vram, step times, …) are left untouched."""
        updates: dict[str, Any] = {"config": config}
        for col in ("lora_name", "definition_id", "project_id"):
            if col in config:
                updates[col] = config[col]
        self._update(job_id, updates)

    def get_config_for_rerun(self, job_id: str) -> dict[str, Any] | None:
        """Extract config dict from a job for re-submission."""
        job = self.get_by_id(job_id)
        if not job:
            return None
        config = job.get("config")
        if isinstance(config, str):
            config = json.loads(config)
        return config

    def delete(self, job_id: str) -> None:
        """Delete a job and its related records from the database."""
        with get_db().write() as conn:
            conn.execute("DELETE FROM job_datasets WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM job_history WHERE id = ?", (job_id,))

    # ── Internal ─────────────────────────────────────────────────────

    def _update(self, job_id: str, updates: dict[str, Any]) -> None:
        updates = self._prepare(updates)
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [job_id]
        with get_db().write() as conn:
            conn.execute(
                f"UPDATE job_history SET {set_clause} WHERE id = ?", values
            )

    @staticmethod
    def _prepare(data: dict[str, Any]) -> dict[str, Any]:
        """Coerce types for SQLite."""
        data = dict(data)
        # JSON-encode complex fields
        for key in ("config", "datasets_used", "loss_history",
                     "targeted_layers", "tags"):
            if key in data and not isinstance(data.get(key), (str, type(None))):
                data[key] = json.dumps(data[key])
        # Bool → int
        for key in ("ema_enabled",):
            if key in data:
                data[key] = int(bool(data[key]))
        return data

    @staticmethod
    def _from_row(row) -> dict[str, Any]:
        """Convert DB row to dict, parsing JSON fields."""
        d = dict(row)
        for key in ("config", "datasets_used", "loss_history",
                     "targeted_layers", "tags"):
            if d.get(key) and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        d["ema_enabled"] = bool(d.get("ema_enabled", 0))
        return d
