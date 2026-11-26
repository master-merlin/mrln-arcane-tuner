"""The card reading, a run's telemetry file and the limits that stop a run.

The card is read through NVML (``gpu_health``), which starts no process per poll and loads no
torch. The telemetry file is opened, appended and closed for every row: a file a writer holds
open can list as empty on Windows, so a reader mid-run would see nothing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from app.core import gpu_health
from app.core.sampler_compare.plan import Limits

CARD_INDEX = 0  # the card a comparison watches: the first, as NVML (and nvidia-smi) number them
COLUMNS = ("unix_time,elapsed_s,step,step_time_s,vram_mib,temperature_c,power_w,power_limit_w,"
           "available_ram_mib,rss_mib,note")
MIB_PER_GIB = 1024
_BYTES_PER_MIB = 2**20


@dataclass(frozen=True)
class Card:
    vram_mib: int
    temperature_c: float
    power_w: float | None
    power_limit_w: float | None


def vram_used_mib(index: int) -> int | None:
    """Card ``index``'s memory in use, through NVML (``gpu_health`` reads none); None without one."""
    if not gpu_health.nvml_ready():
        return None
    nvml = gpu_health.pynvml
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
        try:  # the count nvidia-smi shows: without the driver's reserve
            info = nvml.nvmlDeviceGetMemoryInfo(handle, version=nvml.nvmlMemory_v2)
        except nvml.NVMLError:
            info = nvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.used) // _BYTES_PER_MIB
    except Exception:  # noqa: BLE001 - a card that stopped answering is no reading, never a crash
        return None


def read_card(index: int, health=gpu_health.read_gpu_health, vram=vram_used_mib) -> Card | None:
    """Card ``index``'s reading, or None: no NVML, a card that stopped answering, or no
    temperature or memory reading."""
    reading = health(index)
    if not reading or not reading.get("readable") or reading.get("temperature_c") is None:
        return None
    used = vram(index)
    if used is None:
        return None
    return Card(used, float(reading["temperature_c"]), reading.get("power_draw_w"), reading.get("power_limit_w"))


@dataclass
class Sample:
    elapsed_s: float
    step: int
    step_time_s: float | None
    card: Card | None
    available_ram_mib: float
    rss_mib: float
    note: str = ""


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


class TelemetryFile:
    """A CSV with a header, one row per sample, every row closed on disk."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8", newline="") as out:
            out.write(COLUMNS + "\n")

    def row(self, sample: Sample) -> None:
        card = sample.card
        note = sample.note or ("card unreadable" if card is None else "")
        cells = [
            f"{time.time():.0f}", _cell(round(sample.elapsed_s)), _cell(sample.step), _cell(sample.step_time_s),
            _cell(card and card.vram_mib), _cell(card and card.temperature_c), _cell(card and card.power_w),
            _cell(card and card.power_limit_w), _cell(round(sample.available_ram_mib)), _cell(round(sample.rss_mib)),
            note.replace(",", ";").replace("\n", " "),
        ]
        with open(self.path, "a", encoding="utf-8", newline="") as out:
            out.write(",".join(cells) + "\n")


def breach(sample: Sample, limits: Limits, quiet_s: float) -> str | None:
    """Why this sample stops the run, or None. A value at its limit does not stop it."""
    card = sample.card
    if card is None:
        return "the card can't be read"
    if card.temperature_c > limits.temperature_c:
        return f"the card reads {card.temperature_c:g} C, over {limits.temperature_c:g} C"
    if card.vram_mib > limits.vram_mib:
        return f"VRAM {card.vram_mib} MiB, over {limits.vram_mib} MiB"
    ram_gib = sample.available_ram_mib / MIB_PER_GIB
    if ram_gib < limits.free_ram_gib:
        return f"host available RAM {ram_gib:.1f} GiB, under {limits.free_ram_gib:g} GiB"
    if quiet_s > limits.stall_minutes * 60:
        return f"no log output for {limits.stall_minutes:g} min"
    if sample.elapsed_s > limits.run_minutes * 60:
        return f"over the {limits.run_minutes:g} min limit"
    return None


def _higher(current, value):
    return value if current is None else max(current, value)


def _rounded(value, digits: int):
    return None if value is None else round(value, digits)


@dataclass
class Peaks:
    """A run's running extremes, and the sum and count its mean step time comes from."""

    temperature_c: float | None = None
    vram_mib: int | None = None
    power_w: float | None = None
    min_ram_gib: float | None = None
    rss_gib: float | None = None
    step_time_sum_s: float = 0.0
    step_time_count: int = 0

    def add(self, sample: Sample) -> None:
        card = sample.card
        if card is not None:
            self.temperature_c = _higher(self.temperature_c, card.temperature_c)
            self.vram_mib = _higher(self.vram_mib, card.vram_mib)
            if card.power_w is not None:
                self.power_w = _higher(self.power_w, card.power_w)
        ram_gib = sample.available_ram_mib / MIB_PER_GIB
        self.min_ram_gib = ram_gib if self.min_ram_gib is None else min(self.min_ram_gib, ram_gib)
        self.rss_gib = _higher(self.rss_gib, sample.rss_mib / MIB_PER_GIB)
        if sample.step_time_s is not None:
            self.step_time_sum_s += sample.step_time_s
            self.step_time_count += 1

    def summary(self) -> dict:
        count = self.step_time_count
        return {
            "peak_temperature_c": _rounded(self.temperature_c, 1),
            "peak_vram_mib": self.vram_mib,
            "peak_power_w": _rounded(self.power_w, 1),
            "min_available_ram_gib": _rounded(self.min_ram_gib, 2),
            "peak_rss_gib": _rounded(self.rss_gib, 2),
            "mean_step_time_s": round(self.step_time_sum_s / count, 3) if count else None,
        }
