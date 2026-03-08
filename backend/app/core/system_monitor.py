"""System and GPU monitoring service.

Provides live GPU metrics (VRAM, temperature, power, utilization) via
``nvidia-ml-py`` (NVML bindings) and system metrics (RAM, CPU) via ``psutil``.
Designed for both snapshot REST endpoints and periodic WebSocket broadcasting.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from typing import Any

import psutil
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Try to import NVML — gracefully degrade if not available (e.g. no NVIDIA GPU)
# ---------------------------------------------------------------------------
try:
    import pynvml  # provided by nvidia-ml-py

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
    logger.info("nvml_initialised", driver=pynvml.nvmlSystemGetDriverVersion())
except Exception:
    _NVML_AVAILABLE = False
    logger.warning("nvml_unavailable", hint="GPU monitoring disabled")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GPUStatus:
    """Snapshot of a single GPU's state."""

    index: int
    name: str
    vram_used_mb: int
    vram_total_mb: int
    vram_percent: float
    temperature_c: int
    power_draw_w: float
    power_limit_w: float
    gpu_utilization: int          # %
    memory_utilization: int       # %
    clock_graphics_mhz: int
    clock_memory_mhz: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SystemStatus:
    """Snapshot of CPU / RAM state."""

    ram_used_mb: int
    ram_total_mb: int
    ram_percent: float
    cpu_percent: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MonitorSnapshot:
    """Combined GPU + system snapshot."""

    gpus: list[GPUStatus] = field(default_factory=list)
    system: SystemStatus | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpus": [g.to_dict() for g in self.gpus],
            "system": self.system.to_dict() if self.system else None,
        }


# ---------------------------------------------------------------------------
# Monitor service
# ---------------------------------------------------------------------------

class SystemMonitor:
    """Live GPU and system health monitoring.

    Usage::

        monitor = SystemMonitor()
        snapshot = monitor.snapshot()      # one-shot
        async for snap in monitor.stream(interval_s=2):
            ...  # continuous streaming
    """

    # ── Snapshot ──────────────────────────────────────────────────────────

    def snapshot(self) -> MonitorSnapshot:
        """Return a combined GPU + system health snapshot."""
        return MonitorSnapshot(
            gpus=self._gpu_snapshot(),
            system=self._system_snapshot(),
        )

    # ── Streaming ────────────────────────────────────────────────────────

    async def stream(self, interval_s: float = 2.0):
        """Async generator yielding snapshots at *interval_s* cadence."""
        while True:
            yield self.snapshot()
            await asyncio.sleep(interval_s)

    # ── Private ──────────────────────────────────────────────────────────

    @staticmethod
    def _gpu_snapshot() -> list[GPUStatus]:
        """Query all NVIDIA GPUs via NVML."""
        if not _NVML_AVAILABLE:
            return []

        gpus: list[GPUStatus] = []
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU,
                    )
                except pynvml.NVMLError:
                    temp = -1

                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
                except pynvml.NVMLError:
                    power = 0.0

                try:
                    power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
                except pynvml.NVMLError:
                    power_limit = 0.0

                try:
                    clock_gfx = pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_GRAPHICS,
                    )
                except pynvml.NVMLError:
                    clock_gfx = 0

                try:
                    clock_mem = pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_MEM,
                    )
                except pynvml.NVMLError:
                    clock_mem = 0

                vram_used = mem.used // (1024 * 1024)
                vram_total = mem.total // (1024 * 1024)

                gpus.append(GPUStatus(
                    index=i,
                    name=name,
                    vram_used_mb=vram_used,
                    vram_total_mb=vram_total,
                    vram_percent=round(vram_used / vram_total * 100, 1) if vram_total else 0,
                    temperature_c=temp,
                    power_draw_w=round(power, 1),
                    power_limit_w=round(power_limit, 1),
                    gpu_utilization=util.gpu,
                    memory_utilization=util.memory,
                    clock_graphics_mhz=clock_gfx,
                    clock_memory_mhz=clock_mem,
                ))
        except pynvml.NVMLError as e:
            logger.error("gpu_snapshot_failed", error=str(e))

        return gpus

    @staticmethod
    def _system_snapshot() -> SystemStatus:
        """Query system RAM and CPU via psutil."""
        mem = psutil.virtual_memory()
        return SystemStatus(
            ram_used_mb=int(mem.used / (1024 * 1024)),
            ram_total_mb=int(mem.total / (1024 * 1024)),
            ram_percent=round(mem.percent, 1),
            cpu_percent=psutil.cpu_percent(interval=None),
        )


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
system_monitor = SystemMonitor()
