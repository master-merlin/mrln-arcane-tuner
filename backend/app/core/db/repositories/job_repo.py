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
