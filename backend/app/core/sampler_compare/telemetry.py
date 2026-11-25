"""The card reading, a run's telemetry file and the limits that stop a run.

The telemetry file is opened, appended and closed for every row: a file a writer holds open can
list as empty on Windows, so a reader mid-run would see nothing.
"""
from __future__ import annotations

import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.core.sampler_compare.plan import Limits

QUERY = "memory.used,temperature.gpu,power.draw,power.limit"
READ_TIMEOUT_S = 30
COLUMNS = ("unix_time,elapsed_s,step,step_time_s,vram_mib,temperature_c,power_w,power_limit_w,"
           "available_ram_mib,rss_mib,note")
MIB_PER_GIB = 1024


@dataclass(frozen=True)
class Card:
    vram_mib: int
    temperature_c: float
    power_w: float | None
    power_limit_w: float | None


def _optional(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None  # "[N/A]": a field this card does not offer


def parse_card(line: str) -> Card | None:
    """One ``nvidia-smi`` line of ``QUERY``; None for anything that is not a reading."""
    fields = [f.strip() for f in line.split(",")]
    if len(fields) != 4:
        return None
    try:
        vram, temperature = int(float(fields[0])), float(fields[1])
    except ValueError:
        return None
    return Card(vram, temperature, _optional(fields[2]), _optional(fields[3]))


def read_card(run=subprocess.run) -> Card | None:
    """The first card's reading (a host with two cards: the first line), or None."""
    try:
        done = run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
                   capture_output=True, text=True, timeout=READ_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = (done.stdout or "").strip().splitlines()
    return parse_card(lines[0]) if lines else None


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


@dataclass
class Peaks:
    temperature_c: list[float] = field(default_factory=list)
    vram_mib: list[int] = field(default_factory=list)
    power_w: list[float] = field(default_factory=list)
    ram_gib: list[float] = field(default_factory=list)
    rss_gib: list[float] = field(default_factory=list)
    step_time_s: list[float] = field(default_factory=list)

    def add(self, sample: Sample) -> None:
        if sample.card is not None:
            self.temperature_c.append(sample.card.temperature_c)
            self.vram_mib.append(sample.card.vram_mib)
            if sample.card.power_w is not None:
                self.power_w.append(sample.card.power_w)
        self.ram_gib.append(sample.available_ram_mib / MIB_PER_GIB)
        self.rss_gib.append(sample.rss_mib / MIB_PER_GIB)
        if sample.step_time_s is not None:
            self.step_time_s.append(sample.step_time_s)

    def summary(self) -> dict:
        def top(values, digits=2):
            return round(max(values), digits) if values else None

        return {
            "peak_temperature_c": top(self.temperature_c, 1),
            "peak_vram_mib": max(self.vram_mib) if self.vram_mib else None,
            "peak_power_w": top(self.power_w, 1),
            "min_available_ram_gib": round(min(self.ram_gib), 2) if self.ram_gib else None,
            "peak_rss_gib": top(self.rss_gib),
            "mean_step_time_s": round(statistics.fmean(self.step_time_s), 3) if self.step_time_s else None,
        }
