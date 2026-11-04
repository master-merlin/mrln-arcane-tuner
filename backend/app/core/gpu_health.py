"""One reading of a GPU's thermal, clock, power and throttle state, through NVML.

It tells two things apart that look alike from outside. A field this card doesn't
offer is None and the rest of the reading stands. A device that stopped answering
makes the whole reading ``{"readable": False, "error": ...}``, so a lost card is
never shown as a card with some fields missing.

NVML reads create no CUDA context and don't load the card. This module must stay
free of torch: it is called beside a running training loop.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Try to import NVML — gracefully degrade if not available (e.g. no NVIDIA GPU)
# ---------------------------------------------------------------------------
try:
    import pynvml  # provided by nvidia-ml-py

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False
    logger.warning("nvml_unavailable", hint="GPU health readings disabled")

_REASON_NAMES = {
    0x1: "idle",
    0x2: "applications_clocks",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal",
    0x40: "hw_thermal",
    0x80: "hw_power_brake",
    0x100: "display_clock",
}

_NS_PER_MS = 1_000_000
_MW_PER_W = 1000


def read_gpu_health(index: int) -> dict[str, Any] | None:
    """Read card ``index``. None without NVML; never raises."""
    if not _NVML_AVAILABLE:
        return None
    try:
        return _read_fields(pynvml.nvmlDeviceGetHandleByIndex(index))
    except Exception as exc:
        return {"readable": False, "error": _error_text(exc)}


def _read_fields(handle: Any) -> dict[str, Any]:
    reasons = _optional(pynvml.nvmlDeviceGetCurrentClocksEventReasons, handle)
    return {
        "temperature_c": _optional(
            pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU,
        ),
        "temperature_margin_c": _as_int(
            _optional(pynvml.nvmlDeviceGetMarginTemperature, handle),
        ),
        "fan_percent": _optional(pynvml.nvmlDeviceGetFanSpeed, handle),
        "clock_graphics_mhz": _optional(
            pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_GRAPHICS,
        ),
        "clock_memory_mhz": _optional(
            pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_MEM,
        ),
        "power_draw_w": _watts(_optional(pynvml.nvmlDeviceGetPowerUsage, handle)),
        "power_limit_w": _watts(_optional(pynvml.nvmlDeviceGetEnforcedPowerLimit, handle)),
        "throttle_raw": None if reasons is None else hex(reasons),
        "throttle_reasons": None if reasons is None else _reason_names(reasons),
        "thermal_throttle_ms": _violation_ms(handle, pynvml.NVML_PERF_POLICY_THERMAL),
        "power_throttle_ms": _violation_ms(handle, pynvml.NVML_PERF_POLICY_POWER),
        "readable": True,
    }


def _optional(query: Any, *args: Any) -> Any:
    """A query's answer, or None if this card doesn't offer it.

    Every other NVML error propagates: it means the device failed.
    """
    try:
        return query(*args)
    except (
        pynvml.NVMLError_NotSupported,
        pynvml.NVMLError_NoPermission,
        pynvml.NVMLError_FunctionNotFound,
    ):
        return None


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _watts(milliwatts: int | None) -> float | None:
    return None if milliwatts is None else milliwatts / _MW_PER_W


def _violation_ms(handle: Any, policy: int) -> float | None:
    status = _optional(pynvml.nvmlDeviceGetViolationStatus, handle, policy)
    return None if status is None else status.violationTime / _NS_PER_MS


def _reason_names(mask: int) -> list[str]:
    names = []
    for position in range(mask.bit_length()):
        bit = 1 << position
        if mask & bit:
            names.append(_REASON_NAMES.get(bit, f"unknown_{bit:#x}"))
    return names


def _error_text(exc: Exception) -> str:
    # The binding's __str__ may call back into NVML for a code it doesn't name.
    try:
        return str(exc) or type(exc).__name__
    except Exception:
        return type(exc).__name__
