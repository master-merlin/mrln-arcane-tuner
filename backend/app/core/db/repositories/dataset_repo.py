"""DatasetRepository — CRUD for the ``datasets`` table.

Maps between the ``Dataset`` Pydantic model and SQLite rows.
All media-item operations are delegated to ``MediaItemRepository``.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from app.core.db.engine import get_db

logger = structlog.get_logger(__name__)


class DatasetRepository:
    """Thin wrapper around the ``datasets`` table."""

    # Fields that map 1:1 between Dataset model and DB columns.
    _COLUMNS = [
        "id", "name", "path", "description", "created_at",
        "last_scanned_at", "file_count", "total_size_bytes",
        "multimedia_count", "caption_count", "mask_count",
        "caption_coverage", "missing", "preview_image",
        "majority_ar", "harmonization_score", "classifier",
        "version", "has_cache", "source_type", "license", "updated_at",
    ]

    # ── Reads ────────────────────────────────────────────────────────

    def get_all(self) -> list[dict[str, Any]]:
        """Return all datasets as dicts (without media_metadata)."""
        conn = get_db().connection()
        rows = conn.execute("SELECT * FROM datasets ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    def get_by_name(self, name: str) -> dict[str, Any] | None:
        """Lookup a dataset by name."""
        conn = get_db().connection()
        row = conn.execute(
            "SELECT * FROM datasets WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def get_by_id(self, dataset_id: str) -> dict[str, Any] | None:
        """Lookup a dataset by ID."""
        conn = get_db().connection()
        row = conn.execute(
            "SELECT * FROM datasets WHERE id = ?", (dataset_id,)
        ).fetchone()
        return dict(row) if row else None

    # ── Writes ───────────────────────────────────────────────────────

    def upsert(self, data: dict[str, Any]) -> None:
        """Insert or update a dataset row.

        ``data`` should be a dict with keys matching ``_COLUMNS``.
        Boolean fields are coerced to int for SQLite.
        """
        data = dict(data)  # defensive copy
        data.setdefault("updated_at", time.time())

        # Coerce booleans → int for SQLite
        for key in ("caption_coverage", "missing", "has_cache"):
            if key in data:
                data[key] = int(bool(data[key]))

        cols = [c for c in self._COLUMNS if c in data]
        placeholders = ", ".join("?" for _ in cols)
        updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "id")

        sql = f"""
            INSERT INTO datasets ({', '.join(cols)})
            VALUES ({placeholders})
            ON CONFLICT(id) DO UPDATE SET {updates}
        """
        with get_db().write() as conn:
            conn.execute(sql, [data[c] for c in cols])

    def delete(self, dataset_id: str) -> None:
        """Delete a dataset and its media items (cascaded)."""
        with get_db().write() as conn:
            conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))

    def delete_by_name(self, name: str) -> None:
        """Delete a dataset by name."""
        with get_db().write() as conn:
            conn.execute("DELETE FROM datasets WHERE name = ?", (name,))

    # ── Bulk import (migration) ──────────────────────────────────────

    def bulk_import(self, datasets: list[dict[str, Any]]) -> int:
        """Import multiple datasets in a single transaction.

        Returns the number of rows inserted.
        """
        count = 0
        with get_db().write() as conn:
            for data in datasets:
                data = dict(data)
                data.setdefault("updated_at", time.time())
                for key in ("caption_coverage", "missing", "has_cache"):
                    if key in data:
                        data[key] = int(bool(data[key]))

                cols = [c for c in self._COLUMNS if c in data]
                placeholders = ", ".join("?" for _ in cols)
                sql = f"""
                    INSERT OR REPLACE INTO datasets ({', '.join(cols)})
                    VALUES ({placeholders})
                """
                conn.execute(sql, [data[c] for c in cols])
                count += 1

        logger.info("datasets_bulk_imported", count=count)
        return count
