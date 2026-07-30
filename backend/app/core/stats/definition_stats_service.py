"""Per-definition statistics service — the brain of the estimation wall.

Turns completed ``job_history`` rows into per-``definition_id`` calibration
coefficients (median ``actual / cost_model`` over local runs), and combines
those coefficients with the config-driven cost model to produce precise,
data-calibrated estimates (wall time, output size, throughput, disk
footprint, VRAM).

Design: ``estimate = cost_model(config) × calibration_coeff[definition_id]``.
With zero local runs the coefficients fall back to defaults that reproduce
the legacy heuristics, and ``stats_available`` is ``False`` so the UI can
prompt the user to backfill from history.
"""

from __future__ import annotations

import json
from statistics import median
from typing import Any

import structlog

from app.core.db.repositories.definition_stats_repo import DefinitionStatsRepository
from app.engine.utils import cost_model

logger = structlog.get_logger(__name__)

STATS_SOURCE_VERSION = 1

# Analytic VRAMReport fields we calibrate per-component (+ the caching peak).
VRAM_COMPONENTS = (
    "model_weights_mb",
    "lora_adapters_mb",
    "optimizer_states_mb",
    "gradients_mb",
    "activations_mb",
    "overhead_mb",
    "caching_peak_mb",
)


# ── Recompute (history → coefficients) ──────────────────────────────────


def recompute(definition_id: str | None = None) -> dict[str, Any]:
    """Rebuild ``definition_stats`` from completed ``job_history`` rows.

    Pass a ``definition_id`` to refresh one definition (cheap, used on each
    job completion), or ``None`` to refresh every definition that has runs.
    Recompute-from-scratch keeps the cache self-correcting — no incremental
    drift. Returns a summary ``{definition_id: run_count}``.
    """
    from app.core.db.engine import get_db

    conn = get_db().connection()
    if definition_id:
        rows = conn.execute(
            "SELECT * FROM job_history WHERE status = 'completed' AND definition_id = ?",
            (definition_id,),
        ).fetchall()
        grouped: dict[str, list] = {definition_id: list(rows)}
    else:
        rows = conn.execute(
            "SELECT * FROM job_history WHERE status = 'completed'"
        ).fetchall()
        grouped = {}
        for r in rows:
            grouped.setdefault(r["definition_id"], []).append(r)

    repo = DefinitionStatsRepository()
    summary: dict[str, int] = {}
    for def_id, def_rows in grouped.items():
        if not def_id:
            continue
        stats = _aggregate(def_id, def_rows)
        repo.upsert(def_id, len(def_rows), stats, STATS_SOURCE_VERSION)
        summary[def_id] = len(def_rows)

    logger.info("definition_stats_recomputed", definitions=len(summary),
                scope=definition_id or "all")
    return summary


def _row_config(row) -> dict[str, Any]:
    """Effective config for a run: config JSON snapshot, gaps filled from columns."""
    cfg: dict[str, Any] = {}
    raw = row["config"] if "config" in row.keys() else None
    if raw:
        try:
            cfg = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (json.JSONDecodeError, TypeError):
            cfg = {}
    # Column fallbacks (only used by cost_model when the JSON key is absent)
    for col, key in (
        ("network_rank", "network_rank"),
        ("batch_size", "train_batch_size"),
        ("grad_accum", "gradient_accumulation_steps"),
        ("quantization", "quantization"),
        ("mixed_precision", "mixed_precision"),
        ("total_steps", "max_train_steps"),
    ):
        if cfg.get(key) in (None, "") and col in row.keys() and row[col] is not None:
            cfg[key] = row[col]
    return cfg


def _aggregate(definition_id: str, rows: list) -> dict[str, Any]:
    """Compute median calibration coefficients across a definition's runs."""
    time_samples: list[float] = []
    bpr_samples: list[float] = []
    disk_samples: list[float] = []
    # Per-component VRAM calibration ratios (measured ÷ analytic).
    vram_samples: dict[str, list[float]] = {f: [] for f in VRAM_COMPONENTS}

    defn = _get_definition(definition_id)

    for row in rows:
        cfg = _row_config(row)
        keys = row.keys()

        avg_step_time = row["avg_step_time"] if "avg_step_time" in keys else None
        t = cost_model.normalize_time(avg_step_time or 0, cfg)
        if t is not None:
            time_samples.append(t)

        lora_bytes = row["final_lora_size_bytes"] if "final_lora_size_bytes" in keys else None
        bpr = cost_model.normalize_bytes_per_rank(lora_bytes or 0, cfg)
        if bpr is not None:
            bpr_samples.append(bpr)

        total_bytes = row["total_run_bytes"] if "total_run_bytes" in keys else None
        d = cost_model.normalize_disk_overhead(total_bytes or 0, lora_bytes or 0)
        if d is not None:
            disk_samples.append(d)

        # Per-component VRAM calibration: read the measured breakdown from the
        # run's training_log.json, compare against the analytic breakdown for
        # this run's config. Requires the definition (registry) to be loaded.
        if defn is not None:
            out_dir = row["output_dir"] if "output_dir" in keys else None
            measured = _read_vram_measured(out_dir)
            if measured:
                analytic = _analytic_vram(defn, cfg)
                if analytic:
                    for field in VRAM_COMPONENTS:
                        a = analytic.get(field, 0) or 0
                        m = measured.get(field)
                        if m and a > 0:
                            vram_samples[field].append(m / a)

    def _agg(samples: list[float]) -> dict[str, Any] | None:
        if not samples:
            return None
        return {"value": float(median(samples)), "samples": len(samples)}

    stats: dict[str, Any] = {}
    if (t := _agg(time_samples)):
        stats["time_coeff"] = t
    if (b := _agg(bpr_samples)):
        stats["bytes_per_rank"] = b
    if (d := _agg(disk_samples)):
        stats["disk_overhead_ratio"] = d
    vram: dict[str, Any] = {}
    for field, samples in vram_samples.items():
        if (agg := _agg(samples)):
            vram[field] = agg
    if vram:
        # Stamp the analytic-formula version these ratios were derived against.
        # A coefficient is measured ÷ analytic, so a formula change silently
        # invalidates it — see VRAM_FORMULA_VERSION. Non-dict value, so the
        # readers' `isinstance(entry, dict)` filter skips it naturally.
        from app.engine.utils.vram_estimator import VRAM_FORMULA_VERSION

        vram["formula_version"] = VRAM_FORMULA_VERSION
        stats["vram"] = vram
    return stats


# ── Estimate (config + coefficients → numbers) ──────────────────────────


def estimate(definition_id: str, config: dict[str, Any]) -> dict[str, Any]:
    """Produce a full calibrated estimate payload for the wall."""
    row = DefinitionStatsRepository().get(definition_id)
    stats: dict[str, Any] = row["stats"] if row else {}
    run_count = int(row["run_count"]) if row else 0

    def _coeff(key: str, default: float) -> tuple[float, int, bool]:
        entry = stats.get(key)
        if isinstance(entry, dict) and entry.get("value") is not None:
            return float(entry["value"]), int(entry.get("samples", 0)), True
        return default, 0, False

    time_coeff, time_n, time_cal = _coeff("time_coeff", cost_model.DEFAULT_TIME_COEFF)
    bpr, bpr_n, bpr_cal = _coeff("bytes_per_rank", cost_model.DEFAULT_BYTES_PER_RANK)
    disk_ratio, disk_n, disk_cal = _coeff(
        "disk_overhead_ratio", cost_model.DEFAULT_DISK_OVERHEAD_RATIO
    )

    sec_per_step = cost_model.step_time_cost(config) * time_coeff
    wall_seconds = cost_model.wall_time_seconds(config, time_coeff)
    output_bytes = cost_model.lora_bytes(config, bpr)
    disk_bytes = cost_model.disk_footprint_bytes(config, bpr, disk_ratio)
    steps_per_sec = (1.0 / sec_per_step) if sec_per_step > 0 else 0.0

    vram = _vram_estimate(definition_id, config, stats)

    return {
        "definition_id": definition_id,
        "stats_available": run_count > 0,
        "samples": run_count,
        "updated_at": row["updated_at"] if row else None,
        "wall_time": {
            "seconds": round(wall_seconds),
            "display": _fmt_duration(wall_seconds),
            "samples": time_n,
            "calibrated": time_cal,
        },
        "output_size": {
            "bytes": round(output_bytes),
            "display": _fmt_bytes(output_bytes),
            "samples": bpr_n,
            "calibrated": bpr_cal,
        },
        "throughput": {
            "steps_per_sec": round(steps_per_sec, 3),
            "display": _fmt_throughput(steps_per_sec),
            "samples": time_n,
            "calibrated": time_cal,
        },
        "disk_footprint": {
            "bytes": round(disk_bytes),
            "display": _fmt_bytes(disk_bytes),
            "samples": disk_n,
            "calibrated": disk_cal,
        },
        "vram": vram,
    }


def get(definition_id: str) -> dict[str, Any]:
    """Raw stats + freshness for the wall hint."""
    row = DefinitionStatsRepository().get(definition_id)
    if not row:
        return {"definition_id": definition_id, "stats_available": False,
                "run_count": 0, "stats": {}, "updated_at": None}
    return {
        "definition_id": definition_id,
        "stats_available": int(row["run_count"]) > 0,
        "run_count": row["run_count"],
        "stats": row["stats"],
        "updated_at": row["updated_at"],
    }


# ── VRAM helpers ────────────────────────────────────────────────────────


def _get_definition(definition_id: str):
    try:
        from app.engine.models.registry import registry
        return registry._definitions.get(definition_id)
    except Exception:
        return None


def _analytic_vram(defn, config: dict[str, Any]) -> dict[str, Any] | None:
    """Uncalibrated analytic VRAM report (for computing calibration ratios)."""
    try:
        from app.engine.utils.vram_estimator import VRAMEstimator
        return VRAMEstimator.estimate(defn, config).to_dict()
    except Exception:
        return None


def _read_vram_measured(out_dir: str | None) -> dict[str, Any] | None:
    """Load the per-component measured VRAM breakdown from training_log.json."""
    if not out_dir:
        return None
    import json
    import os
    path = os.path.join(out_dir, "training_log.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    measured = data.get("vram_measured")
    return measured if isinstance(measured, dict) else None


def _vram_estimate(definition_id: str, config: dict[str, Any],
                   stats: dict[str, Any]) -> dict[str, Any] | None:
    """Calibrated VRAM report (analytic × learned per-component multipliers)."""
    defn = _get_definition(definition_id)
    if defn is None:
        return None
    from app.engine.utils.vram_estimator import VRAM_FORMULA_VERSION, VRAMEstimator

    # Build a per-component calibration dict from learned coefficients.
    calibration: dict[str, float] = {}
    vram_stats = stats.get("vram") or {}
    # Coefficients are ``measured ÷ analytic``. When the analytic formula has
    # changed since they were derived, applying them skews the estimate by the
    # formula delta — in the under-estimating direction if the correction shrank
    # a term, which _sane_calibration's plausibility band cannot detect. Drop
    # the whole block; the next completed run for this definition (or a stats
    # backfill) re-derives it against the current formula.
    if vram_stats.get("formula_version") != VRAM_FORMULA_VERSION:
        if vram_stats:
            logger.info(
                "vram_calibration_stale",
                definition_id=definition_id,
                stamped=vram_stats.get("formula_version"),
                current=VRAM_FORMULA_VERSION,
            )
        vram_stats = {}
    for field, entry in vram_stats.items():
        if isinstance(entry, dict) and entry.get("value") is not None:
            calibration[field] = float(entry["value"])
    try:
        report = VRAMEstimator.estimate(defn, config, calibration=calibration or None)
        out = report.to_dict()
        out["calibrated"] = bool(calibration)
        out["calibrated_components"] = sorted(calibration.keys())
        return out
    except Exception as e:
        logger.warning("vram_estimate_failed", definition_id=definition_id, error=str(e))
        return None


# ── Formatting ──────────────────────────────────────────────────────────


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = round((seconds % 3600) / 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m"
    return f"{seconds}s"


def _fmt_bytes(num: float) -> str:
    num = max(0.0, float(num))
    mb = num / (1024 * 1024)
    if mb >= 1024:
        return f"~{mb / 1024:.1f} GB"
    return f"~{round(mb)} MB"


def _fmt_throughput(steps_per_sec: float) -> str:
    if steps_per_sec <= 0:
        return "—"
    if steps_per_sec >= 1:
        return f"{steps_per_sec:.2f} it/s"
    return f"{1.0 / steps_per_sec:.1f} s/it"
