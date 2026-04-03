"""SavedConceptRepository — CRUD for ``saved_concepts``.

Saved concepts are masking reference prompts/points that can be
scoped globally (project_id = NULL) or per-project. Users can
move concepts between scopes.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import structlog

from app.core.db.engine import get_db

logger = structlog.get_logger(__name__)


class SavedConceptRepository:
    """Saved masking concepts with flexible scope."""

    TABLE = "saved_concepts"

    # ── Reads ────────────────────────────────────────────────────────

    def list_for_project(
        self, project_id: str | None = None
    ) -> list[dict[str, Any]]:
        """List concepts visible to a project (global + project-specific)."""
        conn = get_db().connection()

        if project_id:
            rows = conn.execute(
                f"SELECT * FROM {self.TABLE} "
                "WHERE project_id IS NULL OR project_id = ? "
                "ORDER BY project_id NULLS FIRST, name",
                (project_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT * FROM {self.TABLE} "
                "WHERE project_id IS NULL ORDER BY name"
            ).fetchall()

        return [self._from_row(r) for r in rows]

    def get_by_id(self, concept_id: str) -> dict[str, Any] | None:
        conn = get_db().connection()
        row = conn.execute(
            f"SELECT * FROM {self.TABLE} WHERE id = ?", (concept_id,)
        ).fetchone()
        return self._from_row(row) if row else None

    # ── Writes ───────────────────────────────────────────────────────

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        data = dict(data)
        data.setdefault("id", f"concept_{uuid.uuid4().hex[:12]}")
        data.setdefault("created_at", now)
        data.setdefault("updated_at", now)
        data.setdefault("model_id", "sam3")

        if "points" in data and not isinstance(data["points"], str):
            data["points"] = json.dumps(data["points"])

        cols = list(data.keys())
        placeholders = ", ".join("?" for _ in cols)

        with get_db().write() as conn:
            conn.execute(
                f"INSERT INTO {self.TABLE} ({', '.join(cols)}) VALUES ({placeholders})",
                [data[c] for c in cols],
            )

        return self.get_by_id(data["id"])  # type: ignore[return-value]

    def update(self, concept_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        updates = dict(updates)
        updates["updated_at"] = time.time()
        updates.pop("id", None)
        updates.pop("created_at", None)

        if "points" in updates and not isinstance(updates["points"], str):
            updates["points"] = json.dumps(updates["points"])

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [concept_id]

        with get_db().write() as conn:
            conn.execute(
                f"UPDATE {self.TABLE} SET {set_clause} WHERE id = ?", values
            )

        return self.get_by_id(concept_id)

    def delete(self, concept_id: str) -> None:
        with get_db().write() as conn:
            conn.execute(
                f"DELETE FROM {self.TABLE} WHERE id = ?", (concept_id,)
            )

    def move_scope(self, concept_id: str, new_project_id: str | None) -> dict[str, Any] | None:
        """Move a concept between global ↔ project scope."""
        return self.update(concept_id, {"project_id": new_project_id})

    # ── Internal ─────────────────────────────────────────────────────

    @staticmethod
    def _from_row(row: Any) -> dict[str, Any]:
        d = dict(row)
        if d.get("points") and isinstance(d["points"], str):
            try:
                d["points"] = json.loads(d["points"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d
