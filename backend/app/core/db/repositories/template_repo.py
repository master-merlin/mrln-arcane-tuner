"""TemplateRepository — unified CRUD for the ``templates`` table.

Manages training, captioning, and masking templates in a single table
with a ``category`` discriminator column.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import structlog

from app.core.db.engine import get_db

logger = structlog.get_logger(__name__)


class TemplateRepository:
    """Unified template storage across all categories."""

    VALID_CATEGORIES = ("training", "captioning", "masking")

    # ── Reads ────────────────────────────────────────────────────────

    def list_by_category(
        self, category: str, definition_id: str | None = None,
        model_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List templates filtered by category and optional target."""
        clauses = ["category = ?"]
        params: list[Any] = [category]

        if definition_id:
            clauses.append("definition_id = ?")
            params.append(definition_id)
        if model_id:
            clauses.append("model_id = ?")
            params.append(model_id)

        where = " AND ".join(clauses)
        conn = get_db().connection()
        rows = conn.execute(
            f"SELECT * FROM templates WHERE {where} ORDER BY name", params
        ).fetchall()
        return [self._from_row(r) for r in rows]

    def get_by_id(self, template_id: str) -> dict[str, Any] | None:
        conn = get_db().connection()
        row = conn.execute(
            "SELECT * FROM templates WHERE id = ?", (template_id,)
        ).fetchone()
        return self._from_row(row) if row else None

    def get_by_name(
        self, category: str, name: str
    ) -> dict[str, Any] | None:
        conn = get_db().connection()
        row = conn.execute(
            "SELECT * FROM templates WHERE category = ? AND name = ?",
            (category, name),
        ).fetchone()
        return self._from_row(row) if row else None

    # ── Writes ───────────────────────────────────────────────────────

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create a new template. Returns created row."""
        data = dict(data)
        data.setdefault("id", str(uuid.uuid4()))
        data.setdefault("created_at", time.time())
        data.setdefault("updated_at", time.time())
        data.setdefault("used_count", 0)
        data.setdefault("is_default", False)
        data.setdefault("readonly", False)

        data = self._prepare(data)
        cols = list(data.keys())
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT OR REPLACE INTO templates ({', '.join(cols)}) VALUES ({placeholders})"

        with get_db().write() as conn:
            conn.execute(sql, [data[c] for c in cols])

        return self.get_by_id(data["id"])

    def update(self, template_id: str, updates: dict[str, Any]) -> None:
        """Update specific fields on a template."""
        updates = dict(updates)
        updates["updated_at"] = time.time()
        updates = self._prepare(updates)
        # Don't update immutable fields
        for key in ("id", "category", "created_at"):
            updates.pop(key, None)

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [template_id]
        with get_db().write() as conn:
            conn.execute(
                f"UPDATE templates SET {set_clause} WHERE id = ?", values
            )

    def delete(self, template_id: str) -> None:
        with get_db().write() as conn:
            conn.execute(
                "DELETE FROM templates WHERE id = ?", (template_id,)
            )

    def increment_usage(self, template_id: str) -> None:
        """Track template popularity."""
        with get_db().write() as conn:
            conn.execute("""
                UPDATE templates
                SET used_count = used_count + 1, last_used_at = ?
                WHERE id = ?
            """, (time.time(), template_id))

    # ── Bulk import (migration) ──────────────────────────────────────

    def bulk_import(
        self, category: str, templates: list[dict[str, Any]]
    ) -> int:
        """Import templates from legacy settings format."""
        count = 0
        with get_db().write() as conn:
            for tpl in templates:
                data = dict(tpl)
                data["category"] = category
                data.setdefault("id", str(uuid.uuid4()))
                data.setdefault("created_at", time.time())
                data.setdefault("updated_at", time.time())
                data.setdefault("used_count", 0)

                data = self._prepare(data)
                cols = list(data.keys())
                placeholders = ", ".join("?" for _ in cols)
                conn.execute(
                    f"INSERT OR REPLACE INTO templates ({', '.join(cols)}) "
                    f"VALUES ({placeholders})",
                    [data[c] for c in cols],
                )
                count += 1

        logger.info("templates_bulk_imported", category=category, count=count)
        return count

    # ── Internal ─────────────────────────────────────────────────────

    @staticmethod
    def _prepare(data: dict[str, Any]) -> dict[str, Any]:
        data = dict(data)
        # JSON-encode config
        for key in ("config",):
            if key in data and not isinstance(data.get(key), (str, type(None))):
                data[key] = json.dumps(data[key])
        # Booleans → int
        for key in ("is_default", "readonly"):
            if key in data:
                data[key] = int(bool(data[key]))
        return data

    @staticmethod
    def _from_row(row) -> dict[str, Any]:
        d = dict(row)
        # Parse JSON config
        if d.get("config") and isinstance(d["config"], str):
            try:
                d["config"] = json.loads(d["config"])
            except (json.JSONDecodeError, TypeError):
                pass
        d["is_default"] = bool(d.get("is_default", 0))
        d["readonly"] = bool(d.get("readonly", 0))
        return d
