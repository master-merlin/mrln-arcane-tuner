"""The card's state in the job log, from a thread beside the training loop.

A stalled loop writes nothing, and the stall is what has to be on record, so this
runs on its own. It reads through NVML only: no torch, no CUDA and no random stream,
so it can't change what trains. Keep this module's whole import chain free of torch.

The rules of its locks:

- An NVML call holds no lock, so a hung card holds up nobody.
- ``_gate`` makes "are we stopped?" and the write one step, and ``stop`` passes through
  it. So once ``stop`` has passed, no line can follow, not even from a reading that was
  in flight.
- ``_values_lock`` guards only the recorded values, so ``summary`` never waits for a
  write. It is taken inside ``_gate`` and never the other way round.
"""

from __future__ import annotations

import copy
import os
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from app.core import gpu_health
from app.core.gpu_health import read_gpu_health

INTERVAL_S = 10.0
# How long ``stop`` waits for a write in progress, beyond its own timeout. A write that
# takes longer means a stuck disk, on which the next line of the job would wait as well.
WRITE_WAIT_S = 5.0

_LINE_TYPE = "gpu_health"
_ENV = "CUDA_VISIBLE_DEVICES"


def visible_device(value: str | None) -> int | str | None:
    """The card a job uses, from the value of ``CUDA_VISIBLE_DEVICES``.

    A job uses the first visible device. Gives NVML's index, as an int, or a UUID to look
    up, or None if the job is shown no card. NVML ignores the variable, and CUDA stops at
    the first entry it can't read, so an entry that isn't a device means no card.
    """
    if value is None:
        return 0
    first = value.split(",")[0].strip()
    if first.startswith(("GPU-", "MIG-")):
        return first
    return int(first) if first.isascii() and first.isdigit() else None


def _nvml_index_for_uuid(uuid: str) -> int:
    """NVML's index of the one whole card whose UUID starts with ``uuid``.

    CUDA takes a prefix of a UUID. A MIG instance is no whole card and has no index here.
    Raises if it names no single card: a reading of another card would be a false record.
    """
    import pynvml

    matches = [
        index
        for index in range(pynvml.nvmlDeviceGetCount())
        if pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(index)).startswith(uuid)
    ]
    if len(matches) != 1:
        raise LookupError(f"{_ENV} names {uuid}: {len(matches)} cards match")
    return matches[0]


def _describe(exc: Exception) -> str:
    # The NVML binding's __str__ can itself call NVML, and fail.
    try:
        return f"{type(exc).__name__}: {exc}"
    except Exception:
        return type(exc).__name__


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{round(seconds, 1):g} s"
    if seconds < 3600:
        return f"{round(seconds / 60, 1):g} min"
    hours, rest = divmod(round(seconds / 60), 60)
    return f"{hours} h {rest} min"


def _known(value: Any, unit: str) -> str:
    return "not known" if value is None else f"{value:g}{unit}"


def _growth(name: str, milliseconds: float | None) -> str:
    if milliseconds is None:
        return f"{name} not known"
    if 0 < milliseconds < 1:
        return f"{name} grew by under 1 ms"
    if milliseconds < 1000:
        return f"{name} grew by {round(milliseconds)} ms"
    return f"{name} grew by {_duration(milliseconds / 1000)}"


def _last_good_text(reading: dict[str, Any]) -> str:
    draw, limit = reading.get("power_draw_w"), reading.get("power_limit_w")
    power = (
        "power not known" if draw is None
        else f"{draw:g} W of {limit:g} W" if limit is not None
        else f"{draw:g} W"
    )
    reasons = reading.get("throttle_reasons")
    throttle = (
        "throttle reasons not known" if reasons is None
        else f"throttle reasons {', '.join(reasons)}" if reasons
        else "no throttle reasons"
    )
    return (
        f"The last good reading: {_known(reading.get('temperature_c'), ' C')}, "
        f"margin {_known(reading.get('temperature_margin_c'), ' C')}, "
        f"fan {_known(reading.get('fan_percent'), '%')}, {power}, {throttle}."
    )


def _unreadable_text(summary: dict[str, Any]) -> str:
    first = summary.get("first_unreadable_after_seconds")
    total = _duration(summary.get("unreadable_seconds") or 0.0)
    error = summary.get("last_error")
    said = "" if error is None else f": {' '.join(str(error).split())}"
    since = "" if first is None else f", first {_duration(first)} into the record"
    if summary["unreadable_at_end"]:
        return f"The card was unreadable at the end of the job{since}, {total} in all{said}."
    return f"The card was unreadable for {total}{since}, and came back{said}."


def format_summary(summary: dict[str, Any], *, ended: bool = True) -> str:
    """The job's ``gpu_health_summary`` line, for a person, from ``summary()``'s values.

    ``ended`` is what ``stop`` returned: False means the recorder was still waiting.
    """
    last_good = summary["last_good"]
    sentences: list[str] = []
    if last_good is not None:
        record = summary.get("record_seconds")
        lasted = "" if record is None else f"recorded for {_duration(record)}; "
        sentences.append(
            f"{lasted}peak temperature {_known(summary['peak_temperature_c'], ' C')}, "
            f"lowest margin {_known(summary['lowest_margin_c'], ' C')} to the card's limit, "
            f"peak fan {_known(summary['peak_fan_percent'], '%')}; "
            f"{_growth('thermal throttling', summary['thermal_throttle_growth_ms'])}, "
            f"{_growth('power throttling', summary['power_throttle_growth_ms'])}.",
        )
        sentences.append(_last_good_text(last_good))
    elif summary["was_unreadable"]:
        sentences.append("no good reading of the card was recorded.")
    elif ended:
        sentences.append("no reading of the card was recorded: no NVML, or no card for this job.")
    else:
        sentences.append("no reading of the card was recorded.")
    if summary["was_unreadable"]:
        sentences.append(_unreadable_text(summary))
    if not ended:
        age = summary.get("last_reading_age_seconds")
        last = "" if age is None else f"; the last reading came {_duration(age)} before the end"
        sentences.append(
            f"A read of the card or a write to the job log had not returned when the job ended{last}.",
        )
    failures = summary["write_failures"]
    if failures:
        noun = "reading" if failures == 1 else "readings"
        sentences.append(f"{failures} {noun} could not be written to the job log.")
    return "gpu_health_summary: " + " ".join(sentences)


class _CounterGrowth:
    """The growth of a counter that normally only rises.

    The last value stays through a stretch with no reading, so what the counter gained
    meanwhile still counts. A value below the last one (a driver reset) becomes the new
    baseline and adds nothing: this can understate the growth and never overstates it.
    """

    def __init__(self) -> None:
        self.total: float | None = None
        self._last: float | None = None

    def add(self, value: float | None) -> None:
        if value is None:
            return
        if self._last is None:
            self.total = 0.0
        elif value > self._last:
            self.total += value - self._last
        self._last = value


def _higher(current: float | None, value: float | None) -> float | None:
    if value is None:
        return current
    return value if current is None or value > current else current


def _lower(current: float | None, value: float | None) -> float | None:
    if value is None:
        return current
    return value if current is None or value < current else current


class GpuHealthRecorder:
    """Writes one ``gpu_health`` line per reading and keeps the job's extremes.

    A readable card writes a line at each reading, the first at start. An unreadable one
    writes one line when it goes so, and the next readable reading ends the stretch.

    Without NVML, or for a job shown no card, nothing starts and ``summary`` holds no
    reading. No failure of the reader or the writer reaches the caller or ends the thread.
    """

    def __init__(
        self,
        writer: Any,
        *,
        reader: Callable[[int], dict[str, Any] | None] = read_gpu_health,
        interval_s: float = INTERVAL_S,
    ) -> None:
        self._writer = writer
        self._reader = reader
        self._interval_s = interval_s
        self._device = visible_device(os.environ.get(_ENV))
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._gate = threading.Lock()
        self._values_lock = threading.Lock()

        self._readable: bool | None = None
        self._was_unreadable = False
        self._last_error: str | None = None
        self._last_good: dict[str, Any] | None = None
        self._peak_temperature: float | None = None
        self._lowest_margin: float | None = None
        self._peak_fan: float | None = None
        self._thermal_growth = _CounterGrowth()
        self._power_growth = _CounterGrowth()
        self._write_failures = 0
        self._clock = time.monotonic
        self._first_reading_at: float | None = None
        self._last_reading_at: float | None = None
        self._first_unreadable_at: float | None = None
        self._unreadable_since: float | None = None
        self._unreadable_before = 0.0

    # ── Public API ────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start the thread, and return whether it started. Makes no NVML call."""
        if self._thread is not None or self._stop_event.is_set():
            return False
        # Only what is already known. To find out is an NVML call, which can hang, so it
        # is for the thread to do: not on the job's own.
        if gpu_health.nvml_ruled_out() or self._device is None:
            return False
        thread = threading.Thread(target=self._run, name="gpu-health-recorder", daemon=True)
        try:
            thread.start()
        except RuntimeError:
            return False  # out of threads: the job goes on with no record of the card
        self._thread = thread
        return True

    def stop(self, timeout: float = 2.0) -> bool:
        """Stop, and return whether the thread has ended and no line can follow.

        ``timeout`` bounds the wait for a thread inside an NVML call, which stays behind
        as a daemon and writes nothing when its reading arrives. A write in progress is
        waited for up to ``WRITE_WAIT_S`` more, so that a return leaves no line to come;
        only a write stuck for longer can still land, and then this returns False.
        """
        deadline = time.monotonic() + timeout
        self._stop_event.set()
        if self._gate.acquire(timeout=WRITE_WAIT_S + max(0.0, timeout)):
            self._gate.release()
        if self._thread is None:
            return True
        self._thread.join(max(0.0, deadline - time.monotonic()))
        return not self._thread.is_alive()

    def summary(self) -> dict[str, Any]:
        """What was recorded, for the job's summary line. Makes no NVML call.

        Waits for no write and no reading, so a dead card or a stuck disk can't hold it.
        A value the card never gave is None, and so is the growth of a counter it never gave.
        """
        with self._values_lock:
            now = self._clock()
            unreadable = self._unreadable_before
            if self._unreadable_since is not None:
                unreadable += now - self._unreadable_since
            first, last, went = (
                self._first_reading_at, self._last_reading_at, self._first_unreadable_at,
            )
            return {
                "record_seconds": None if first is None else now - first,
                "last_reading_age_seconds": None if last is None else now - last,
                "first_unreadable_after_seconds": None if went is None else went - first,
                "unreadable_seconds": unreadable,
                "peak_temperature_c": self._peak_temperature,
                "lowest_margin_c": self._lowest_margin,
                "peak_fan_percent": self._peak_fan,
                "thermal_throttle_growth_ms": self._thermal_growth.total,
                "power_throttle_growth_ms": self._power_growth.total,
                "last_good": copy.deepcopy(self._last_good),
                "was_unreadable": self._was_unreadable,
                "unreadable_at_end": self._readable is False,
                "last_error": self._last_error,
                "write_failures": self._write_failures,
            }

    # ── The thread ────────────────────────────────────────────────────

    def _run(self) -> None:
        while True:
            reading = self._read()
            if reading is None or not self._take(reading):
                return
            if self._stop_event.wait(self._interval_s):
                return

    def _read(self) -> dict[str, Any] | None:
        """One reading, holding no lock. None means there is no NVML."""
        try:
            if isinstance(self._device, str):
                if not gpu_health.nvml_ready():
                    return None
                self._device = _nvml_index_for_uuid(self._device)
            return self._reader(self._device)
        except Exception as exc:
            return {"readable": False, "error": _describe(exc)}

    def _take(self, reading: dict[str, Any]) -> bool:
        """Record a reading and write its line, unless stopped. Returns whether to go on."""
        with self._gate:
            if self._stop_event.is_set():
                return False
            if self._record(reading):
                self._write(reading)
        return True

    def _record(self, reading: dict[str, Any]) -> bool:
        """Keep what the reading says, and return whether it gets a line."""
        readable = reading.get("readable") is True
        with self._values_lock:
            now = self._clock()
            if self._first_reading_at is None:
                self._first_reading_at = now
            self._last_reading_at = now
            state_changed = readable != self._readable
            self._readable = readable
            if not readable:
                self._was_unreadable = True
                self._last_error = reading.get("error")
                if self._first_unreadable_at is None:
                    self._first_unreadable_at = now
                if self._unreadable_since is None:
                    self._unreadable_since = now
                return state_changed
            if self._unreadable_since is not None:
                self._unreadable_before += now - self._unreadable_since
                self._unreadable_since = None
            self._last_good = copy.deepcopy(reading)
            self._peak_temperature = _higher(self._peak_temperature, reading.get("temperature_c"))
            self._lowest_margin = _lower(self._lowest_margin, reading.get("temperature_margin_c"))
            self._peak_fan = _higher(self._peak_fan, reading.get("fan_percent"))
            self._thermal_growth.add(reading.get("thermal_throttle_ms"))
            self._power_growth.add(reading.get("power_throttle_ms"))
            return True

    def _write(self, reading: dict[str, Any]) -> None:
        """Costs the line on failure and nothing more: the values are already recorded."""
        try:
            self._writer.emit(_LINE_TYPE, reading)
        except Exception as exc:
            with self._values_lock:
                self._write_failures += 1
                first_failure = self._write_failures == 1
            if first_failure:
                # safety-net print: the job log is what just failed, and logging would
                # route back into it; stderr reaches trainer_stdout.log.
                print(
                    f"WARNING: gpu_health line not written: {_describe(exc)}",
                    file=sys.stderr,
                    flush=True,
                )
