"""SampleImageRepository — CRUD for the ``sample_images`` table.

Tracks generated sample images during training for visual inspection.
"""

from __future__ import annotations

import time
from typing import Any

from app.core.db.engine import get_db


class SampleImageRepository:
    """Training sample image tracking."""

    def add(self, data: dict[str, Any]) -> None:
        """Record a generated sample image."""
        data = dict(data)
        data.setdefault("created_at", time.time())
        cols = ["job_id", "step", "prompt", "seed", "path",
                "width", "height", "created_at"]
        vals = [data.get(c) for c in cols]
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO sample_images ({', '.join(cols)}) VALUES ({placeholders})"
        with get_db().write() as conn:
            conn.execute(sql, vals)

    def list_by_job(self, job_id: str) -> list[dict[str, Any]]:
        conn = get_db().connection()
        rows = conn.execute(
            "SELECT * FROM sample_images WHERE job_id = ? ORDER BY step",
            (job_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_by_step(self, job_id: str, step: int) -> list[dict[str, Any]]:
        conn = get_db().connection()
        rows = conn.execute(
            "SELECT * FROM sample_images WHERE job_id = ? AND step = ?",
            (job_id, step),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_by_job(self, job_id: str) -> int:
        with get_db().write() as conn:
            cursor = conn.execute(
                "DELETE FROM sample_images WHERE job_id = ?", (job_id,)
            )
            return cursor.rowcount
