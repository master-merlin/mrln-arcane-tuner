"""MetricsRepository — bulk insert/query for the ``step_metrics`` table.

Stores per-step training metrics (loss, lr, grad norm) for post-hoc
analysis and loss curve charting.
"""

from __future__ import annotations

import time
from typing import Any

from app.core.db.engine import get_db


class MetricsRepository:
    """Per-step training metrics storage."""

    def batch_insert(self, job_id: str, metrics: list[dict[str, Any]]) -> int:
        """Bulk insert step metrics in one transaction.

        Each dict should have at least ``step`` and ``loss``.
        """
        now = time.time()
        count = 0
        with get_db().write() as conn:
            for m in metrics:
                conn.execute("""
                    INSERT INTO step_metrics
                    (job_id, step, loss, lr, grad_norm, timestep_mean, epoch, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    job_id,
                    m.get("step", 0),
                    m.get("loss"),
                    m.get("lr"),
                    m.get("grad_norm"),
                    m.get("timestep_mean"),
                    m.get("epoch"),
                    now,
                ))
                count += 1
        return count

    def get_loss_curve(self, job_id: str) -> list[dict[str, Any]]:
        """Return step, loss, lr for charting."""
        conn = get_db().connection()
        rows = conn.execute(
            "SELECT step, loss, lr, grad_norm, timestep_mean, epoch "
            "FROM step_metrics WHERE job_id = ? ORDER BY step",
            (job_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_summary(self, job_id: str) -> dict[str, Any]:
        """Aggregate metrics for a job."""
        conn = get_db().connection()
        row = conn.execute("""
            SELECT
                COUNT(*) as total_steps,
                AVG(loss) as avg_loss,
                MIN(loss) as min_loss,
                MAX(loss) as max_loss,
                MIN(lr) as min_lr,
                MAX(lr) as max_lr
            FROM step_metrics WHERE job_id = ?
        """, (job_id,)).fetchone()
        return dict(row) if row else {}

    def prune(self, job_id: str, keep_every_n: int = 10) -> int:
        """Thin old metrics data, keeping every Nth step.

        Returns number of rows deleted.
        """
        with get_db().write() as conn:
            cursor = conn.execute("""
                DELETE FROM step_metrics
                WHERE job_id = ? AND step % ? != 0
            """, (job_id, keep_every_n))
            return cursor.rowcount

    def delete_by_job(self, job_id: str) -> int:
        """Remove all metrics for a job."""
        with get_db().write() as conn:
            cursor = conn.execute(
                "DELETE FROM step_metrics WHERE job_id = ?", (job_id,)
            )
            return cursor.rowcount
