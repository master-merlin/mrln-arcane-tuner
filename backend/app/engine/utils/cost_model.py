"""Config-driven cost model for the estimation wall.

Predicts the *relative* cost of a training run from its config knobs. The
absolute scale (which is hardware- and model-specific) is supplied separately
as a per-``definition_id`` calibration coefficient learned from local runs::

    estimate = cost_model(config) × calibration_coeff[definition_id]

Each cost function is normalised so the **reference config** — batch 1,
accumulation 1, 1024px, gradient-checkpointing on, no quantization, frozen
text encoder, fp16 save — returns ``1.0``. That makes the zero-data fallback
coefficients reproduce the legacy hardcoded heuristics exactly:

- wall time: ``1.2 s`` per step  →  ``DEFAULT_TIME_COEFF``
- output:    ``14 MB`` per rank  →  ``DEFAULT_BYTES_PER_RANK``

All functions are pure (no torch / GPU / DB) so they are trivially testable.
Config keys accept both the canonical ``BaseTrainingConfig`` names and the
``job_history`` column aliases (e.g. ``train_batch_size`` / ``batch_size``).
"""

from __future__ import annotations

from typing import Any

# Reference config anchors (cost == 1.0 here)
REF_RESOLUTION = 1024

# Zero-data fallbacks — chosen so the reference config matches the legacy
# client-side heuristics the wall used before calibration data existed.
DEFAULT_TIME_COEFF = 1.2  # seconds per step at the reference config
DEFAULT_BYTES_PER_RANK = 14 * 1024 * 1024  # ~14 MB per LoRA rank (fp16)
DEFAULT_DISK_OVERHEAD_RATIO = 6.0  # total run bytes ÷ final LoRA bytes (guess)

# Relative compute multipliers (vs the reference config).
# Quantized base weights add dequant overhead per forward/backward; the
# per-definition coefficient absorbs hardware specifics, so keep these mild.
_QUANT_TIME_FACTOR: dict[str, float] = {
    "nf4": 1.10,
    "int4": 1.10,
    "int5": 1.08,
    "int6": 1.08,
    "int7": 1.05,
    "int8": 1.05,
    "qint4": 1.10,
    "qint8": 1.05,
}

# save_precision → bytes-per-element multiplier (vs fp16/bf16 baseline)
_SAVE_PRECISION_FACTOR: dict[str, float] = {
    "fp32": 2.0,
    "float32": 2.0,
    "fp16": 1.0,
    "bf16": 1.0,
    "float16": 1.0,
    "bfloat16": 1.0,
}


# ── Config accessors (tolerant of both naming schemes) ──────────────────


def _first(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        v = config.get(k)
        if v is not None:
            return v
    return default


def _num(value: Any, default: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    return n if n > 0 else default


def resolution_of(config: dict[str, Any]) -> int:
    """Representative training resolution (the largest bucket)."""
    res = _first(config, "resolutions")
    if isinstance(res, (list, tuple)) and res:
        try:
            return int(max(int(r) for r in res))
        except (TypeError, ValueError):
            pass
    return int(_num(_first(config, "resolution", "width", default=REF_RESOLUTION), REF_RESOLUTION))


def batch_of(config: dict[str, Any]) -> float:
    return _num(_first(config, "train_batch_size", "batch_size", default=1), 1)


def accum_of(config: dict[str, Any]) -> float:
    return _num(_first(config, "gradient_accumulation_steps", "grad_accum", default=1), 1)


def steps_of(config: dict[str, Any]) -> int:
    return int(_num(_first(config, "max_train_steps", "total_steps", default=1000), 1000))


def rank_of(config: dict[str, Any]) -> float:
    return _num(
        _first(config, "network_rank", "lora_rank", "rank", "lora_dim", "network_dim", default=16),
        16,
    )


# ── Cost functions (all normalised to 1.0 at the reference config) ──────


def step_time_cost(config: dict[str, Any]) -> float:
    """Relative wall-time cost of a *single* step (reference == 1.0).

    Scales with the in-flight work per optimizer step: micro-batch size,
    gradient-accumulation passes, and pixel area. Gradient checkpointing,
    base quantization, and TE training apply mild secondary multipliers.
    """
    cost = batch_of(config) * accum_of(config)
    res = resolution_of(config)
    cost *= (res / REF_RESOLUTION) ** 2

    # Gradient checkpointing recomputes the forward pass → reference (on).
    # Turning it OFF is faster.
    gc_on = bool(_first(config, "gradient_checkpointing", default=True))
    if not gc_on:
        cost *= 0.75

    quant = str(_first(config, "quantization", default="none") or "none")
    cost *= _QUANT_TIME_FACTOR.get(quant, 1.0)

    if bool(_first(config, "train_text_encoder", default=False)):
        cost *= 1.25

    return max(cost, 1e-6)


def wall_time_seconds(config: dict[str, Any], time_coeff: float) -> float:
    """Total estimated wall time = per-step cost × coeff × step count."""
    return step_time_cost(config) * float(time_coeff) * steps_of(config)


def save_precision_factor(config: dict[str, Any]) -> float:
    prec = str(_first(config, "save_precision", default="fp16") or "fp16").lower()
    return _SAVE_PRECISION_FACTOR.get(prec, 1.0)


def lora_bytes(config: dict[str, Any], bytes_per_rank: float) -> float:
    """Estimated final LoRA file size in bytes."""
    return float(bytes_per_rank) * rank_of(config) * save_precision_factor(config)


def disk_footprint_bytes(config: dict[str, Any], bytes_per_rank: float,
                         disk_overhead_ratio: float) -> float:
    """Estimated total on-disk footprint of a run (LoRA + working set)."""
    return lora_bytes(config, bytes_per_rank) * float(disk_overhead_ratio)


# ── Per-run normalisation (used when aggregating history into coeffs) ────


def normalize_time(avg_step_time: float, config: dict[str, Any]) -> float | None:
    """Per-run time coefficient: seconds-per-step ÷ relative step cost."""
    if not avg_step_time or avg_step_time <= 0:
        return None
    cost = step_time_cost(config)
    return avg_step_time / cost if cost > 0 else None


def normalize_bytes_per_rank(final_lora_size_bytes: float, config: dict[str, Any]) -> float | None:
    """Per-run bytes-per-rank, factoring out rank and save precision."""
    if not final_lora_size_bytes or final_lora_size_bytes <= 0:
        return None
    rank = rank_of(config)
    prec = save_precision_factor(config)
    denom = rank * prec
    return final_lora_size_bytes / denom if denom > 0 else None


def normalize_disk_overhead(total_run_bytes: float, final_lora_size_bytes: float) -> float | None:
    """Per-run total-disk ÷ final-LoRA ratio."""
    if not total_run_bytes or not final_lora_size_bytes or final_lora_size_bytes <= 0:
        return None
    ratio = total_run_bytes / final_lora_size_bytes
    return ratio if ratio >= 1.0 else None
