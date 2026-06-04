"""DefinitionStatsRepository — CRUD for the ``definition_stats`` table.

A local, fully-recomputable cache of per-``definition_id`` calibration
coefficients used by the estimation wall. The ``stats`` column is a JSON
blob so the metric set can evolve without schema churn (mirrors the
JSON-column idiom used by ``job_history.config``).
"""

from __future__ import annotations

import json
import time
from typing import Any

from app.core.db.engine import get_db


class DefinitionStatsRepository:
    """Persistent per-definition estimation coefficients."""

    # ── Reads ────────────────────────────────────────────────────────

    def get(self, definition_id: str) -> dict[str, Any] | None:
        conn = get_db().connection()
        row = conn.execute(
            "SELECT * FROM definition_stats WHERE definition_id = ?",
            (definition_id,),
        ).fetchone()
        return self._from_row(row) if row else None

    def get_all(self) -> list[dict[str, Any]]:
        conn = get_db().connection()
        rows = conn.execute(
            "SELECT * FROM definition_stats ORDER BY definition_id"
        ).fetchall()
        return [self._from_row(r) for r in rows]

    # ── Writes ───────────────────────────────────────────────────────

    def upsert(self, definition_id: str, run_count: int,
               stats: dict[str, Any], source_version: int = 1) -> None:
        """Insert or replace the stats row for a definition."""
        with get_db().write() as conn:
            conn.execute(
                """
                INSERT INTO definition_stats
                    (definition_id, run_count, stats, updated_at, source_version)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(definition_id) DO UPDATE SET
                    run_count      = excluded.run_count,
                    stats          = excluded.stats,
                    updated_at     = excluded.updated_at,
                    source_version = excluded.source_version
                """,
                (
                    definition_id,
                    int(run_count),
                    json.dumps(stats),
                    time.time(),
                    int(source_version),
                ),
            )

    def delete(self, definition_id: str) -> None:
        with get_db().write() as conn:
            conn.execute(
                "DELETE FROM definition_stats WHERE definition_id = ?",
                (definition_id,),
            )

    # ── Internal ─────────────────────────────────────────────────────

    @staticmethod
    def _from_row(row) -> dict[str, Any]:
        d = dict(row)
        raw = d.get("stats")
        if isinstance(raw, str):
            try:
                d["stats"] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                d["stats"] = {}
        return d
