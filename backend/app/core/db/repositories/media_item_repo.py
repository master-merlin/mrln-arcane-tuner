"""MediaItemRepository — CRUD for the ``media_items`` table.

Each row represents a single image/video inside a dataset.
Supports bulk upsert for scan results and single-row update for
crop/mask operations (the main performance win over JSON).
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog

from app.core.db.engine import get_db

logger = structlog.get_logger(__name__)


class MediaItemRepository:
    """Thin wrapper around the ``media_items`` table."""

    _COLUMNS = [
        "dataset_id", "rel_path", "width", "height", "aspect_ratio",
        "orientation", "size_bytes", "solid_hash", "is_majority_ar",
        "target_width", "target_height", "has_mask", "has_masked",
        "has_masked_caption", "mask_info", "has_caption",
        "is_video", "frame_count", "tags", "notes", "quality_score",
        "added_at", "has_overlay", "enabled",
        "control_count", "control_info",
    ]

    # ── Reads ────────────────────────────────────────────────────────

    def get_by_dataset(self, dataset_id: str) -> list[dict[str, Any]]:
        """Return all media items for a dataset."""
        conn = get_db().connection()
        rows = conn.execute(
            "SELECT * FROM media_items WHERE dataset_id = ? ORDER BY rel_path",
            (dataset_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_by_path(self, dataset_id: str, rel_path: str) -> dict[str, Any] | None:
        """Lookup a single media item by dataset + relative path."""
        conn = get_db().connection()
        row = conn.execute(
            "SELECT * FROM media_items WHERE dataset_id = ? AND rel_path = ?",
            (dataset_id, rel_path),
        ).fetchone()
        return dict(row) if row else None

    def count_by_dataset(self, dataset_id: str) -> int:
        """Count media items in a dataset."""
        conn = get_db().connection()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM media_items WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
        return row["cnt"]

    def find_similar(
        self, dataset_id: str, solid_hash: str
    ) -> list[dict[str, Any]]:
        """Find items with the same perceptual hash (duplicate detection)."""
        conn = get_db().connection()
        rows = conn.execute(
            "SELECT * FROM media_items WHERE dataset_id = ? AND solid_hash = ?",
            (dataset_id, solid_hash),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Writes ───────────────────────────────────────────────────────

    def upsert(self, data: dict[str, Any]) -> None:
        """Insert or update a single media item.

        Keyed on (dataset_id, rel_path).
        """
        data = self._prepare(data)
        cols = [c for c in self._COLUMNS if c in data]
        placeholders = ", ".join("?" for _ in cols)
        updates = ", ".join(
            f"{c}=excluded.{c}" for c in cols
            if c not in ("dataset_id", "rel_path")
        )

        sql = f"""
            INSERT INTO media_items ({', '.join(cols)})
            VALUES ({placeholders})
            ON CONFLICT(dataset_id, rel_path) DO UPDATE SET {updates}
        """
        with get_db().write() as conn:
            conn.execute(sql, [data[c] for c in cols])

    def update(self, dataset_id: str, rel_path: str, updates: dict[str, Any]) -> None:
        """Update specific fields on a single media item (opens its own write context)."""
        with get_db().write() as conn:
            self.update_with_conn(conn, dataset_id, rel_path, updates)

    def update_with_conn(
        self, conn, dataset_id: str, rel_path: str, updates: dict[str, Any]
    ) -> None:
        """Update specific fields using an external connection (shared transaction).

        This is the fast path for crop/mask operations — one row
        instead of rewriting the entire dataset.
        """
        updates = self._prepare(updates)
        settable = {k: v for k, v in updates.items()
                    if k in self._COLUMNS and k not in ("dataset_id", "rel_path")}
        if not settable:
            return

        set_clause = ", ".join(f"{k} = ?" for k in settable)
        values = list(settable.values()) + [dataset_id, rel_path]

        conn.execute(
            f"UPDATE media_items SET {set_clause} "
            "WHERE dataset_id = ? AND rel_path = ?",
            values,
        )

    def delete(self, dataset_id: str, rel_path: str) -> None:
        """Remove a single media item (opens its own write context)."""
        with get_db().write() as conn:
            self.delete_with_conn(conn, dataset_id, rel_path)

    def delete_with_conn(self, conn, dataset_id: str, rel_path: str) -> None:
        """Remove a single media item using an external connection."""
        conn.execute(
            "DELETE FROM media_items WHERE dataset_id = ? AND rel_path = ?",
            (dataset_id, rel_path),
        )

    def delete_by_dataset(self, dataset_id: str) -> int:
        """Remove all media items for a dataset. Returns count deleted."""
        with get_db().write() as conn:
            cursor = conn.execute(
                "DELETE FROM media_items WHERE dataset_id = ?", (dataset_id,)
            )
            return cursor.rowcount

    def prune_missing_with_conn(
        self, conn, dataset_id: str, keep_rel_paths
    ) -> int:
        """Delete rows whose ``rel_path`` is not in ``keep_rel_paths``.

        Makes the table mirror the authoritative in-memory metadata after a
        scan/harmonize: files that were renamed, converted or removed leave
        behind ghost rows (with stale hashes/thumbnails) that otherwise persist
        across restarts and resurface in Dataset Analysis as near-duplicates
        against filenames no longer on disk. Computes the diff in Python to
        avoid the SQLite ``IN (...)`` parameter limit. Returns rows deleted.
        """
        keep = set(keep_rel_paths)
        existing = conn.execute(
            "SELECT rel_path FROM media_items WHERE dataset_id = ?", (dataset_id,)
        ).fetchall()
        stale = [r["rel_path"] for r in existing if r["rel_path"] not in keep]
        for rel_path in stale:
            conn.execute(
                "DELETE FROM media_items WHERE dataset_id = ? AND rel_path = ?",
                (dataset_id, rel_path),
            )
        return len(stale)

    # ── Bulk operations ──────────────────────────────────────────────

    def bulk_upsert(self, dataset_id: str, items: list[dict[str, Any]]) -> int:
        """Insert or update many media items in one transaction."""
        with get_db().write() as conn:
            return self.bulk_upsert_with_conn(conn, dataset_id, items)

    def bulk_upsert_with_conn(
        self, conn, dataset_id: str, items: list[dict[str, Any]]
    ) -> int:
        """Insert or update many media items using an external connection.

        Returns number of rows affected.
        """
        count = 0
        for raw in items:
            data = self._prepare(raw)
            data["dataset_id"] = dataset_id
            cols = [c for c in self._COLUMNS if c in data]
            placeholders = ", ".join("?" for _ in cols)
            updates = ", ".join(
                f"{c}=excluded.{c}" for c in cols
                if c not in ("dataset_id", "rel_path")
            )
            sql = f"""
                INSERT INTO media_items ({', '.join(cols)})
                VALUES ({placeholders})
                ON CONFLICT(dataset_id, rel_path) DO UPDATE SET {updates}
            """
            conn.execute(sql, [data[c] for c in cols])
            count += 1

        return count

    def bulk_import_from_metadata(
        self, dataset_id: str, media_metadata: dict[str, dict[str, Any]]
    ) -> int:
        """Import from the legacy ``media_metadata`` dict format.

        Keys are relative paths, values are the per-item metadata dicts.
        """
        items = []
        for rel_path, meta in media_metadata.items():
            item = dict(meta)
            item["rel_path"] = rel_path
            item["dataset_id"] = dataset_id
            items.append(item)
        return self.bulk_upsert(dataset_id, items)

    # ── Conversion to legacy format ──────────────────────────────────

    def to_metadata_dict(self, dataset_id: str) -> dict[str, dict[str, Any]]:
        """Convert DB rows back to the legacy ``media_metadata`` format.

        Returns ``{rel_path: {field: value, ...}, ...}`` for backward
        compat with the existing API layer.
        """
        items = self.get_by_dataset(dataset_id)
        result: dict[str, dict[str, Any]] = {}
        for item in items:
            rel_path = item.pop("rel_path")
            item.pop("id", None)
            item.pop("dataset_id", None)
            # Convert int booleans back
            for key in ("is_majority_ar", "has_caption", "is_video",
                        "has_mask", "has_masked", "has_masked_caption",
                        "has_overlay", "enabled"):
                if key in item:
                    item[key] = bool(item[key])
            # Parse JSON fields
            for key in ("mask_info", "control_info", "tags", "notes"):
                if item.get(key) and isinstance(item[key], str):
                    try:
                        item[key] = json.loads(item[key])
                    except (json.JSONDecodeError, TypeError):
                        pass
            result[rel_path] = item
        return result

    # ── Internal ─────────────────────────────────────────────────────

    @staticmethod
    def _prepare(data: dict[str, Any]) -> dict[str, Any]:
        """Coerce types for SQLite storage."""
        data = dict(data)  # defensive copy
        # Booleans → int
        for key in ("is_majority_ar", "has_caption", "is_video",
                    "has_mask", "has_masked", "has_masked_caption",
                    "has_overlay", "enabled"):
            if key in data:
                data[key] = int(bool(data[key]))
        # Dicts/lists → JSON strings
        for key in ("mask_info", "control_info", "tags", "notes"):
            if key in data and not isinstance(data.get(key), (str, type(None))):
                data[key] = json.dumps(data[key])
        # Default added_at
        data.setdefault("added_at", time.time())
        return data
