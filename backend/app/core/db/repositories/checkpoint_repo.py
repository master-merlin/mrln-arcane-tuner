"""CheckpointRepository — CRUD for the ``checkpoints`` table.

Tracks individual checkpoint saves within a training job.
"""

from __future__ import annotations

import time
from typing import Any

from app.core.db.engine import get_db


class CheckpointRepository:
    """Checkpoint tracking per training job."""

    # Every column of the ``checkpoints`` table except the autoincrement
    # ``id`` PK. Allowlist filter on dict-key column interpolation — the same
    # guard as MediaItemRepository/_COLUMNS (media_item_repo.py) — so an
    # unexpected dict key can never reach raw f-string SQL as a column name.
    _COLUMNS = [
        "job_id",
        "step",
        "path",
        "lora_file",
        "lora_size_bytes",
        "created_at",
        "loss_at_step",
        "lr_at_step",
        "is_final",
        "is_deleted",
    ]

    def add(self, data: dict[str, Any]) -> None:
        """Record a checkpoint save event."""
        data = dict(data)
        data.setdefault("created_at", time.time())
        data.setdefault("is_final", 0)
        data.setdefault("is_deleted", 0)
        for key in ("is_final", "is_deleted"):
            data[key] = int(bool(data[key]))

        cols = [c for c in self._COLUMNS if c in data]
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO checkpoints ({', '.join(cols)}) VALUES ({placeholders})"
        with get_db().write() as conn:
            conn.execute(sql, [data[c] for c in cols])

    def list_by_job(self, job_id: str) -> list[dict[str, Any]]:
        conn = get_db().connection()
        rows = conn.execute(
            "SELECT * FROM checkpoints WHERE job_id = ? ORDER BY step",
            (job_id,),
        ).fetchall()
        return [self._from_row(r) for r in rows]

    def mark_deleted(self, checkpoint_id: int) -> None:
        with get_db().write() as conn:
            conn.execute(
                "UPDATE checkpoints SET is_deleted = 1 WHERE id = ?",
                (checkpoint_id,),
            )

    def get_by_path(self, path: str) -> dict[str, Any] | None:
        conn = get_db().connection()
        row = conn.execute(
            "SELECT * FROM checkpoints WHERE path = ?", (path,)
        ).fetchone()
        return self._from_row(row) if row else None

    @staticmethod
    def _from_row(row) -> dict[str, Any]:
        d = dict(row)
        d["is_final"] = bool(d.get("is_final", 0))
        d["is_deleted"] = bool(d.get("is_deleted", 0))
        return d
