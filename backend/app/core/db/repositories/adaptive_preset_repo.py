"""AdaptivePresetRepository — CRUD for ``adaptive_preset_templates``.

Same project-scoped pattern as MaskingTemplateRepository, minus ``model_id``:
an adaptive preset is a bag of layer-targeting knobs, not a model-scoped
template, so the same preset applies to every definition.

Three FACTORY presets are seeded readonly. Editing one is a client-side
branch (create a user preset carrying ``branched_from``), never a write to
the factory row — that is what keeps the shipped presets meaningful as a
reference point across upgrades.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import structlog

from app.core.db.engine import get_db

logger = structlog.get_logger(__name__)

TABLE = "adaptive_preset_templates"

# Fixed ids so seeding is an INSERT OR IGNORE against a stable key — a random
# id per run would re-seed duplicates on every startup.
FACTORY_ID_PREFIX = "factory-"


def _factory_rows() -> list[tuple[str, str, str]]:
    """``(id, display_name, config_json)`` for each shipped factory preset.

    ``preset`` records provenance (``factory:<name>``) alongside the knob
    values, so a job config carrying this dict stays self-describing even
    after the template row it came from is renamed or branched away.
    """
    from app.engine.models.adaptive import FACTORY_PRESETS

    return [
        (
            f"{FACTORY_ID_PREFIX}{name}",
            name.capitalize(),
            json.dumps({**knobs, "preset": f"factory:{name}"}),
        )
        for name, knobs in FACTORY_PRESETS.items()
    ]


def seed_factory_presets_with_conn(conn) -> None:
    """Seed the factory presets on an EXISTING write connection.

    Takes a connection rather than opening its own so it can run inside the
    schema migration that creates the table (``get_db().write()`` is not
    reentrant — nesting it inside the migration's write context would
    deadlock on the engine's write lock).

    ``INSERT OR IGNORE`` on fixed ids: re-running must never duplicate a row
    nor overwrite one, or a live selection would silently revert.
    """
    now = time.time()
    for preset_id, name, config in _factory_rows():
        conn.execute(
            f"INSERT OR IGNORE INTO {TABLE} "
            "(id, project_id, name, is_default, readonly, config, "
            " created_at, updated_at, used_count) "
            "VALUES (?, NULL, ?, 0, 1, ?, ?, ?, 0)",
            (preset_id, name, config, now, now),
        )


class AdaptivePresetRepository:
    """Domain-specific adaptive preset template storage."""

    TABLE = TABLE

    # ── Seeding ──────────────────────────────────────────────────────

    def seed_factory_presets(self) -> None:
        """Idempotently ensure the three readonly factory presets exist."""
        with get_db().write() as conn:
            seed_factory_presets_with_conn(conn)

    # ── Reads ────────────────────────────────────────────────────────

    def list_for_project(
        self,
        _scope_id: str | None = None,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List presets merging General + project scope.

        ``_scope_id`` is accepted and ignored: the shared template helpers
        (import plan, duplicate-name check) call every domain repo as
        ``list_for_project(scope, project_id)`` — captioning/masking pass a
        ``model_id`` and training a ``definition_id``, but an adaptive preset
        is not scoped to either.
        """
        conn = get_db().connection()
        if project_id:
            where = "(project_id IS NULL OR project_id = ?)"
            params: list[Any] = [project_id]
        else:
            where = "project_id IS NULL"
            params = []

        rows = conn.execute(
            f"SELECT * FROM {self.TABLE} WHERE {where} "
            "ORDER BY project_id NULLS FIRST, name",
            params,
        ).fetchall()

        return [self._from_row(r) for r in rows]

    def list_general(self) -> list[dict[str, Any]]:
        """List only General (global) presets — includes the factory three."""
        conn = get_db().connection()
        rows = conn.execute(
            f"SELECT * FROM {self.TABLE} WHERE project_id IS NULL ORDER BY name"
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
        """Create a new adaptive preset."""
        now = time.time()
        data = dict(data)
        data.setdefault("id", f"adaptive_{uuid.uuid4().hex[:12]}")
        data.setdefault("created_at", now)
        data.setdefault("updated_at", now)
        data.setdefault("used_count", 0)
        data.setdefault("is_default", False)
        # Never inherited from a create request: only seeding may mint a
        # readonly row, otherwise a client could forge an undeletable preset.
        data["readonly"] = False

        data = self._prepare(data)
        cols = list(data.keys())
        placeholders = ", ".join("?" for _ in cols)

        with get_db().write() as conn:
            conn.execute(
                f"INSERT INTO {self.TABLE} ({', '.join(cols)}) VALUES ({placeholders})",
                [data[c] for c in cols],
            )

        logger.info("adaptive_preset_created", id=data["id"], name=data.get("name"))
        return self.get_by_id(data["id"])  # type: ignore[return-value]

    def update(self, template_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update specific fields."""
        updates = dict(updates)
        updates["updated_at"] = time.time()
        # ``model_id``/``definition_id`` are other domains' scoping keys and
        # have no column here — the shared PUT request model spans all four
        # domains, so they must be dropped rather than reaching the SET clause
        # as an unknown column (a 500 on an otherwise valid rename).
        # ``readonly`` is never client-settable: only seeding mints one.
        for key in ("id", "created_at", "readonly", "model_id", "definition_id"):
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
        """Delete a preset (the ``readonly = 0`` scope keeps the factory three
        alive even if a caller reaches the repo without the route guard)."""
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
        """Branch a General preset into a project scope.

        The copy is always editable (``create`` forces ``readonly = False``) —
        a readonly branch would dead-end the edit flow it exists to enable.
        """
        source = self.get_by_id(template_id)
        if not source:
            raise ValueError(f"Template {template_id} not found")

        return self.create({
            "project_id": target_project_id,
            "name": new_name or f"{source['name']} (Project)",
            "config": source.get("config", {}),
            "branched_from": template_id,
        })

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
