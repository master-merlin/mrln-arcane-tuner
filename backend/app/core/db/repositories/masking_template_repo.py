"""MaskingTemplateRepository — CRUD for ``masking_templates``.

Same project-scoped pattern as CaptioningTemplateRepository.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import structlog

from app.core.db.engine import get_db

logger = structlog.get_logger(__name__)


class MaskingTemplateRepository:
    """Domain-specific masking template storage."""

    TABLE = "masking_templates"

    # ── Reads ────────────────────────────────────────────────────────

    def list_for_project(
        self,
        model_id: str,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List templates merging General + project scope."""
        conn = get_db().connection()

        if project_id:
            rows = conn.execute(
                f"SELECT * FROM {self.TABLE} "
                "WHERE model_id = ? AND (project_id IS NULL OR project_id = ?) "
                "ORDER BY project_id NULLS FIRST, name",
                (model_id, project_id),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT * FROM {self.TABLE} "
                "WHERE model_id = ? AND project_id IS NULL "
                "ORDER BY name",
                (model_id,),
            ).fetchall()

        return [self._from_row(r) for r in rows]

    def list_general(self, model_id: str) -> list[dict[str, Any]]:
        """List only General (global) templates."""
        conn = get_db().connection()
        rows = conn.execute(
            f"SELECT * FROM {self.TABLE} "
            "WHERE model_id = ? AND project_id IS NULL "
            "ORDER BY name",
            (model_id,),
        ).fetchall()
        return [self._from_row(r) for r in rows]

    def get_by_id(self, template_id: str) -> dict[str, Any] | None:
        conn = get_db().connection()
        row = conn.execute(
            f"SELECT * FROM {self.TABLE} WHERE id = ?", (template_id,)
        ).fetchone()
        return self._from_row(row) if row else None

    # ── Writes ───────────────────────────────────────────────────────

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create a new masking template."""
        now = time.time()
        data = dict(data)
        data.setdefault("id", f"mask_{uuid.uuid4().hex[:12]}")
        data.setdefault("created_at", now)
        data.setdefault("updated_at", now)
        data.setdefault("used_count", 0)
        data.setdefault("is_default", False)
        data.setdefault("readonly", False)

        data = self._prepare(data)
        cols = list(data.keys())
        placeholders = ", ".join("?" for _ in cols)

        with get_db().write() as conn:
            conn.execute(
                f"INSERT INTO {self.TABLE} ({', '.join(cols)}) VALUES ({placeholders})",
                [data[c] for c in cols],
            )

        logger.info("masking_template_created", id=data["id"], name=data.get("name"))
        return self.get_by_id(data["id"])  # type: ignore[return-value]

    def update(self, template_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update specific fields."""
        updates = dict(updates)
        updates["updated_at"] = time.time()
        for key in ("id", "created_at", "model_id"):
            updates.pop(key, None)
        updates = self._prepare(updates)

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [template_id]

        with get_db().write() as conn:
            conn.execute(
                f"UPDATE {self.TABLE} SET {set_clause} WHERE id = ?", values
            )

        return self.get_by_id(template_id)

    def delete(self, template_id: str) -> None:
        """Delete a template (prevents deleting readonly defaults)."""
        with get_db().write() as conn:
            conn.execute(
                f"DELETE FROM {self.TABLE} WHERE id = ? AND readonly = 0",
                (template_id,),
            )

    def increment_usage(self, template_id: str) -> None:
        with get_db().write() as conn:
            conn.execute(
                f"UPDATE {self.TABLE} "
                "SET used_count = used_count + 1, last_used_at = ? "
                "WHERE id = ?",
                (time.time(), template_id),
            )

    # ── Branch ───────────────────────────────────────────────────────

    def branch(
        self, template_id: str, target_project_id: str, new_name: str | None = None
    ) -> dict[str, Any]:
        """Branch a General template into a project scope."""
        source = self.get_by_id(template_id)
        if not source:
            raise ValueError(f"Template {template_id} not found")

        branch_data = {
            "model_id": source["model_id"],
            "project_id": target_project_id,
            "name": new_name or f"{source['name']} (Project)",
            "config": source.get("config", {}),
            "branched_from": template_id,
        }
        return self.create(branch_data)

    # ── Internal ─────────────────────────────────────────────────────

    @staticmethod
    def _prepare(data: dict[str, Any]) -> dict[str, Any]:
        data = dict(data)
        if "config" in data and not isinstance(data.get("config"), (str, type(None))):
            data["config"] = json.dumps(data["config"])
        for key in ("is_default", "readonly"):
            if key in data:
                data[key] = int(bool(data[key]))
        return data

    @staticmethod
    def _from_row(row: Any) -> dict[str, Any]:
        d = dict(row)
        if d.get("config") and isinstance(d["config"], str):
            try:
                d["config"] = json.loads(d["config"])
            except (json.JSONDecodeError, TypeError):
                pass
        d["is_default"] = bool(d.get("is_default", 0))
        d["readonly"] = bool(d.get("readonly", 0))
        return d
