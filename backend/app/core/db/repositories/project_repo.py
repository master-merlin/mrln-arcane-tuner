"""ProjectRepository — CRUD for the ``projects`` table.

Manages project lifecycle and M:N dataset associations via
``project_datasets`` join table.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import structlog

from app.core.db.engine import get_db

logger = structlog.get_logger(__name__)


class ProjectRepository:
    """Project lifecycle and dataset association."""

    # ── Reads ────────────────────────────────────────────────────────

    def list_all(self) -> list[dict[str, Any]]:
        """List all projects ordered by creation date (newest first)."""
        conn = get_db().connection()
        rows = conn.execute(
            "SELECT * FROM projects ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_by_id(self, project_id: str) -> dict[str, Any] | None:
        conn = get_db().connection()
        row = conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_by_name(self, name: str) -> dict[str, Any] | None:
        conn = get_db().connection()
        row = conn.execute(
            "SELECT * FROM projects WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    # ── Writes ───────────────────────────────────────────────────────

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create a new project. Returns created row."""
        now = time.time()
        project_id = data.get("id", str(uuid.uuid4()))
        name = data["name"]
        description = data.get("description", "")
        color = data.get("color", "#6366f1")

        with get_db().write() as conn:
            conn.execute(
                "INSERT INTO projects (id, name, description, color, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (project_id, name, description, color, now, now),
            )

        logger.info("project_created", project_id=project_id, name=name)
        return self.get_by_id(project_id)  # type: ignore[return-value]

    def update(self, project_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update project fields."""
        updates = dict(updates)
        updates["updated_at"] = time.time()
        updates.pop("id", None)
        updates.pop("created_at", None)

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [project_id]

        with get_db().write() as conn:
            conn.execute(
                f"UPDATE projects SET {set_clause} WHERE id = ?", values
            )

        return self.get_by_id(project_id)

    def delete(self, project_id: str) -> None:
        """Delete a project (cascades to templates, preferences, etc.)."""
        with get_db().write() as conn:
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        logger.info("project_deleted", project_id=project_id)

    # ── Dataset associations ─────────────────────────────────────────

    def get_datasets(self, project_id: str) -> list[dict[str, Any]]:
        """Get datasets associated with a project."""
        conn = get_db().connection()
        rows = conn.execute(
            "SELECT d.* FROM datasets d "
            "INNER JOIN project_datasets pd ON d.id = pd.dataset_id "
            "WHERE pd.project_id = ? "
            "ORDER BY d.name",
            (project_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def add_dataset(self, project_id: str, dataset_id: str) -> None:
        """Associate a dataset with a project."""
        with get_db().write() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO project_datasets "
                "(project_id, dataset_id, added_at) VALUES (?, ?, ?)",
                (project_id, dataset_id, time.time()),
            )

    def remove_dataset(self, project_id: str, dataset_id: str) -> bool:
        """Remove a dataset association from a project.

        Returns whether the association actually existed (and was removed).
        """
        with get_db().write() as conn:
            cursor = conn.execute(
                "DELETE FROM project_datasets "
                "WHERE project_id = ? AND dataset_id = ?",
                (project_id, dataset_id),
            )
            return cursor.rowcount > 0

    # ── Stats ────────────────────────────────────────────────────────

    def get_stats(self, project_id: str) -> dict[str, int]:
        """Get template and dataset counts for a project."""
        conn = get_db().connection()
        stats: dict[str, int] = {}

        for table in ("captioning_templates", "masking_templates", "training_templates"):
            row = conn.execute(
                f"SELECT COUNT(*) as c FROM {table} WHERE project_id = ?",
                (project_id,),
            ).fetchone()
            stats[table] = row["c"]

        row = conn.execute(
            "SELECT COUNT(*) as c FROM project_datasets WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        stats["datasets"] = row["c"]

        row = conn.execute(
            "SELECT COUNT(*) as c FROM job_history WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        stats["jobs"] = row["c"]

        return stats
