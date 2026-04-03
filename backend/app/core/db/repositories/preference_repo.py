"""PreferenceRepository — CRUD for ``project_preferences``.

Stores per-project (or General) active selections: which captioning
model is selected, which template is active, qwen3 variant, etc.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import structlog

from app.core.db.engine import get_db

logger = structlog.get_logger(__name__)


class PreferenceRepository:
    """Active selections per project (or General)."""

    TABLE = "project_preferences"

    def get(self, project_id: str | None = None) -> dict[str, Any]:
        """Get preferences for a project (or General if None).

        Returns a dict with active selections. Creates a default row
        if none exists.
        """
        conn = get_db().connection()

        if project_id:
            row = conn.execute(
                f"SELECT * FROM {self.TABLE} WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        else:
            row = conn.execute(
                f"SELECT * FROM {self.TABLE} WHERE project_id IS NULL"
            ).fetchone()

        if row:
            d = dict(row)
            if isinstance(d.get("training_selections"), str):
                try:
                    d["training_selections"] = json.loads(d["training_selections"])
                except (json.JSONDecodeError, TypeError):
                    d["training_selections"] = {}
            return d

        # Auto-create default preferences
        return self._create_default(project_id)

    def upsert(self, project_id: str | None, updates: dict[str, Any]) -> dict[str, Any]:
        """Update or create preferences for a project."""
        existing = self.get(project_id)
        updates = dict(updates)

        # Don't allow changing the project_id or id
        updates.pop("project_id", None)
        updates.pop("id", None)

        # JSON-encode training_selections if dict
        if "training_selections" in updates and isinstance(updates["training_selections"], dict):
            updates["training_selections"] = json.dumps(updates["training_selections"])

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [existing["id"]]

        with get_db().write() as conn:
            conn.execute(
                f"UPDATE {self.TABLE} SET {set_clause} WHERE id = ?", values
            )

        return self.get(project_id)

    # ── Internal ─────────────────────────────────────────────────────

    def _create_default(self, project_id: str | None) -> dict[str, Any]:
        """Create default preferences row."""
        pref_id = str(uuid.uuid4())
        with get_db().write() as conn:
            conn.execute(
                f"INSERT INTO {self.TABLE} "
                "(id, project_id, selected_caption_model, qwen3_variant, "
                " selected_mask_model, training_selections) "
                "VALUES (?, ?, 'florence-2', '4B-Instruct', 'sam3', '{}')",
                (pref_id, project_id),
            )

        return self.get(project_id)
