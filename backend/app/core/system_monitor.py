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
class GpuProcess:
    """A single process holding VRAM on a GPU."""

    pid: int
    name: str
    used_mb: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    vram_free_mb: int = 0          # total − used (device-wide)
    processes: list[GpuProcess] = field(default_factory=list)  # top VRAM holders

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
                vram_free = mem.free // (1024 * 1024)

                gpus.append(GPUStatus(
                    index=i,
                    name=name,
                    vram_used_mb=vram_used,
                    vram_total_mb=vram_total,
                    vram_free_mb=vram_free,
                    vram_percent=round(vram_used / vram_total * 100, 1) if vram_total else 0,
                    temperature_c=temp,
                    power_draw_w=round(power, 1),
                    power_limit_w=round(power_limit, 1),
                    gpu_utilization=util.gpu,
                    memory_utilization=util.memory,
                    clock_graphics_mhz=clock_gfx,
                    clock_memory_mhz=clock_mem,
                    processes=SystemMonitor._gpu_processes(handle),
                ))
        except pynvml.NVMLError as e:
            logger.error("gpu_snapshot_failed", error=str(e))

        return gpus

    @staticmethod
    def _gpu_processes(handle, top_n: int = 8) -> list[GpuProcess]:
        """Top VRAM-holding processes on a GPU (compute + graphics).

        Lets the System Monitor answer "who's using my VRAM" (e.g. ComfyUI),
        which is exactly the external usage the estimate-wall fit check budgets
        around. Best-effort: NVML API names differ across driver/pynvml
        versions, and per-process memory may be unavailable on some setups.
        """
        if not _NVML_AVAILABLE:
            return []

        # Merge compute + graphics processes by pid (a pid may appear in both).
        by_pid: dict[int, int] = {}
        for variant in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses_v3",
            "nvmlDeviceGetGraphicsRunningProcesses_v2",
            "nvmlDeviceGetGraphicsRunningProcesses",
        ):
            fn = getattr(pynvml, variant, None)
            if fn is None:
                continue
            try:
                for p in fn(handle):
                    used = getattr(p, "usedGpuMemory", None)
                    if not used:
                        continue
                    mb = int(used) // (1024 * 1024)
                    by_pid[p.pid] = max(by_pid.get(p.pid, 0), mb)
            except pynvml.NVMLError:
                continue
            # First working compute fn + first working graphics fn is enough;
            # but iterating all is cheap and the max() merge is idempotent.

        # Windows WDDM: NVML reports the process list but no per-process memory
        # (usedGpuMemory is None). Fall back to the Windows performance counter
        # ("GPU Process Memory \ Dedicated Usage"), which is what Task Manager
        # uses and the only source of per-process VRAM bytes on Windows.
        if not by_pid:
            by_pid = SystemMonitor._windows_gpu_proc_mem()

        if not by_pid:
            return []

        try:
            import psutil
        except Exception:
            psutil = None

        procs = [
            GpuProcess(pid=pid, name=SystemMonitor._proc_name(psutil, pid), used_mb=mb)
            for pid, mb in by_pid.items()
            if mb > 0
        ]
        procs.sort(key=lambda gp: gp.used_mb, reverse=True)
        return procs[:top_n]

    @staticmethod
    def _proc_name(psutil_mod, pid: int) -> str:
        """Resolve a pid to a process name (best-effort)."""
        if psutil_mod is None:
            return f"pid {pid}"
        try:
            return psutil_mod.Process(pid).name()
        except Exception:
            return f"pid {pid}"

    # Cache the (costly-ish) Windows perf-counter read so the 2 Hz metrics
    # stream doesn't re-query PDH every tick. (pid -> MB, monotonic timestamp).
    _win_proc_cache: tuple[float, dict[int, int]] = (0.0, {})
    _WIN_PROC_TTL = 3.0  # seconds

    @staticmethod
    def _windows_gpu_proc_mem() -> dict[int, int]:
        """Per-process dedicated GPU memory (MB) on Windows via PDH.

        Reads ``\\GPU Process Memory(*)\\Dedicated Usage`` using
        ``PdhAddEnglishCounterW`` so it works regardless of the OS display
        language. Returns ``{pid: mb}``; empty on non-Windows or any failure.
        """
        import sys
        import time

        if not sys.platform.startswith("win"):
            return {}

        now = time.monotonic()
        ts, cached = SystemMonitor._win_proc_cache
        if cached and (now - ts) < SystemMonitor._WIN_PROC_TTL:
            return cached

        result: dict[int, int] = {}
        try:
            import ctypes
            from ctypes import wintypes

            pdh = ctypes.WinDLL("pdh.dll")
            PDH_FMT_LARGE = 0x00000400

            class _Value(ctypes.Structure):
                _fields_ = [("CStatus", wintypes.DWORD),
                            ("largeValue", ctypes.c_longlong)]

            class _Item(ctypes.Structure):
                _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _Value)]

            query = wintypes.HANDLE()
            if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
                return {}
            try:
                counter = wintypes.HANDLE()
                path = r"\GPU Process Memory(*)\Dedicated Usage"
                if pdh.PdhAddEnglishCounterW(query, path, 0, ctypes.byref(counter)) != 0:
                    return {}
                if pdh.PdhCollectQueryData(query) != 0:
                    return {}

                size = wintypes.DWORD(0)
                count = wintypes.DWORD(0)
                # First call sizes the buffer (returns PDH_MORE_DATA).
                pdh.PdhGetFormattedCounterArrayW(
                    counter, PDH_FMT_LARGE,
                    ctypes.byref(size), ctypes.byref(count), None,
                )
                if size.value == 0:
                    return {}
                buf = ctypes.create_string_buffer(size.value)
                rc = pdh.PdhGetFormattedCounterArrayW(
                    counter, PDH_FMT_LARGE,
                    ctypes.byref(size), ctypes.byref(count),
                    ctypes.cast(buf, ctypes.POINTER(_Item)),
                )
                if rc != 0:
                    return {}
                items = ctypes.cast(
                    buf, ctypes.POINTER(_Item * count.value)
                ).contents
                for it in items:
                    name = it.szName or ""
                    val = it.FmtValue.largeValue
                    if val <= 0 or not name.startswith("pid_"):
                        continue
                    try:
                        pid = int(name.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                    # Sum across a pid's instances (multiple adapters/segments).
                    result[pid] = result.get(pid, 0) + int(val) // (1024 * 1024)
            finally:
                pdh.PdhCloseQuery(query)
        except Exception as e:
            logger.debug("windows_gpu_proc_mem_failed", error=str(e))
            return {}

        SystemMonitor._win_proc_cache = (now, result)
        return result

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
