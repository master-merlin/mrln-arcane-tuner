"""Backfill / reconcile run costs from disk, then rebuild definition stats.

Seeds the estimation wall from whatever local history exists. For every
completed ``job_history`` row it recovers missing cost fields from the run's
output directory (``training_log.json`` for size/elapsed, a directory walk
for total footprint), persists them back, and finally recomputes the
per-definition calibration coefficients.

VRAM peaks cannot be recovered for legacy runs (per-step VRAM was never
persisted to the DB), so only runs trained after this feature shipped
contribute VRAM calibration. Time, output size, and disk footprint are
recoverable for any run that still has its output directory.
"""

from __future__ import annotations

import json
import os
from typing import Any

import structlog

from app.core.db.engine import get_db
from app.core.stats import definition_stats_service

logger = structlog.get_logger(__name__)


def run_backfill() -> dict[str, Any]:
    """Reconcile disk → DB, then recompute all definition stats.

    Returns a summary suitable for an API response.
    """
    conn = get_db().connection()
    rows = conn.execute(
        "SELECT * FROM job_history WHERE status = 'completed'"
    ).fetchall()

    rows_updated = 0
    fields_recovered = 0
    for row in rows:
        updates = _recover_row(row)
        if updates:
            _persist(row["id"], updates)
            rows_updated += 1
            fields_recovered += len(updates)

    stats_summary = definition_stats_service.recompute(None)

    result = {
        "completed_runs": len(rows),
        "rows_updated": rows_updated,
        "fields_recovered": fields_recovered,
        "definitions": stats_summary,
    }
    logger.info("stats_backfill_complete", **{k: v for k, v in result.items()
                                             if k != "definitions"})
    return result


# ── Recovery ────────────────────────────────────────────────────────────


def _recover_row(row) -> dict[str, Any]:
    """Compute updates for a single run from its on-disk artifacts."""
    keys = row.keys()
    out_dir = row["output_dir"] if "output_dir" in keys else None
    if not out_dir or not os.path.isdir(out_dir):
        return {}

    updates: dict[str, Any] = {}
    tlog = _read_training_log(out_dir)

    # Final LoRA artifact (path + size)
    have_size = _val(row, "final_lora_size_bytes")
    have_file = _val(row, "final_lora_file")
    if not have_size or not have_file:
        path, size = _recover_final_lora(row, out_dir, tlog)
        if size and not have_size:
            updates["final_lora_size_bytes"] = size
        if path and not have_file:
            updates["final_lora_file"] = path

    # LoRA on-disk flag: refreshed on EVERY reconcile pass (unlike the
    # fields above, not gated behind "missing") so it self-heals if the
    # file is deleted outside the app after the job completed. Only set
    # when a final_lora_file is actually known (this pass or already on
    # the row) — mirrors get_stats' own `final_lora_file IS NOT NULL` gate.
    final_file = updates.get("final_lora_file") or have_file
    if final_file:
        on_disk = 1 if os.path.isfile(final_file) else 0
        if on_disk != (_val(row, "lora_on_disk") or 0):
            updates["lora_on_disk"] = on_disk

    # Total on-disk footprint
    if "total_run_bytes" in keys and not _val(row, "total_run_bytes"):
        total = _dir_size(out_dir)
        if total:
            updates["total_run_bytes"] = total

    # Average step time (legacy rows that never recorded it)
    if not _val(row, "avg_step_time"):
        ast = _recover_avg_step_time(row, tlog)
        if ast:
            updates["avg_step_time"] = ast

    return updates


def _recover_final_lora(row, out_dir: str, tlog: dict | None) -> tuple[str | None, int | None]:
    """Locate the run's final LoRA. Returns (path, size); path is None when
    only a logged size survives (the file itself is gone)."""
    # Prefer an explicit final file path
    final_file = row["final_lora_file"] if "final_lora_file" in row.keys() else None
    if final_file and os.path.isfile(final_file):
        try:
            return final_file, os.path.getsize(final_file)
        except OSError:
            pass
    # training_log records the saved filename + MB size
    if tlog:
        name = tlog.get("lora_filename")
        if name and name not in ("(not yet saved)",):
            path = os.path.join(out_dir, name)
            if os.path.isfile(path):
                try:
                    return path, os.path.getsize(path)
                except OSError:
                    pass
        mb = tlog.get("lora_file_size_mb")
        if mb:
            return None, int(float(mb) * 1024 * 1024)
    # Fallback: largest *_final*.safetensors, else largest .safetensors
    return _largest_safetensors(out_dir)


def _recover_avg_step_time(row, tlog: dict | None) -> float | None:
    if not tlog:
        return None
    elapsed = tlog.get("elapsed_seconds")
    steps = _val(row, "completed_steps") or tlog.get("step")
    if elapsed and steps and steps > 0:
        return float(elapsed) / float(steps)
    return None


# ── Disk helpers ────────────────────────────────────────────────────────


def _read_training_log(out_dir: str) -> dict | None:
    path = os.path.join(out_dir, "training_log.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _largest_safetensors(out_dir: str) -> tuple[str | None, int | None]:
    best: tuple[str | None, int] = (None, 0)
    final_best: tuple[str | None, int] = (None, 0)
    for root, _dirs, files in os.walk(out_dir):
        for f in files:
            if not f.endswith(".safetensors"):
                continue
            path = os.path.join(root, f)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size > best[1]:
                best = (path, size)
            if "final" in f.lower() and size > final_best[1]:
                final_best = (path, size)
    chosen = final_best if final_best[0] else best
    return chosen if chosen[0] else (None, None)


def _dir_size(out_dir: str) -> int | None:
    total = 0
    for root, _dirs, files in os.walk(out_dir):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                continue
    return total or None


def _val(row, key: str):
    return row[key] if key in row.keys() else None


def _persist(job_id: str, updates: dict[str, Any]) -> None:
    # Allowlist filter on dict-key column interpolation — same guard as
    # MediaItemRepository/_COLUMNS (media_item_repo.py) and
    # JobHistoryRepository/_COLUMNS (job_repo.py, which owns the canonical
    # job_history column list — reused here rather than duplicated).
    from app.core.db.repositories.job_repo import JobHistoryRepository

    updates = {k: v for k, v in updates.items() if k in JobHistoryRepository._COLUMNS}
    if not updates:
        return
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [job_id]
    with get_db().write() as conn:
        conn.execute(
            f"UPDATE job_history SET {set_clause} WHERE id = ?", values
        )
