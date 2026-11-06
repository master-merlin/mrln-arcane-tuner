"""The card's state in the job log, from a thread beside the training loop.

A stalled loop writes nothing, and the stall is what has to be on record, so this
runs on its own. It reads through NVML only: no torch, no CUDA and no random stream,
so it can't change what trains. Keep this module's whole import chain free of torch.

The rules of its locks:

- An NVML call holds no lock, so a hung card holds up nobody.
- ``_gate`` makes "are we stopped?" and the write one step, and ``stop`` passes through
  it. So once ``stop`` has passed, no line can follow, not even from a reading that was
  in flight.
- ``_values_lock`` guards only the recorded values and which card is the job's, so
  ``summary`` never waits for a write. It is taken inside ``_gate`` and never the other
  way round.

Which card is the job's: CUDA and NVML number the cards each in their own order, so a
CUDA ordinal is never taken for NVML's index. NVML finds the card by its UUID or its place
on the PCI bus, and where it can't tell, nothing is read until the trainer names the card.
"""

from __future__ import annotations

import copy
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from app.core import gpu_health
from app.core.gpu_health import read_gpu_health
from app.engine.components.job_log_writer import note

INTERVAL_S = 10.0
# How long ``stop`` waits for a write in progress, beyond its own timeout. A write that
# takes longer means a stuck disk, which the writer then waits on for a bound only.
WRITE_WAIT_S = 5.0

_LINE_TYPE = "gpu_health"
_ENV = "CUDA_VISIBLE_DEVICES"
_ORDER_ENV = "CUDA_DEVICE_ORDER"
_NOT_KNOWN = object()  # a reading of no card: which one is the job's isn't known yet


def visible_device(value: str | None) -> int | str | None:
    """The card a job uses, as CUDA names it, from the value of ``CUDA_VISIBLE_DEVICES``.

    A job uses the first visible device. Gives its CUDA ordinal, as an int, which is not
    NVML's index (see ``_find_card``), or a UUID to look up, or None if the job is shown
    no card. CUDA stops at the first entry it can't read, so an entry that isn't a device
    means no card.
    """
    if value is None:
        return 0
    first = value.split(",")[0].strip()
    if first.startswith(("GPU-", "MIG-")):
        return first
    return int(first) if first.isascii() and first.isdigit() else None


def _pci_place(pynvml: Any, handle: Any) -> tuple[int, int, int]:
    info = pynvml.nvmlDeviceGetPciInfo(handle)
    return (int(info.domain), int(info.bus), int(info.device))


def _bus_id(place: tuple[int, int, int]) -> str:
    domain, bus, device = place
    return f"{domain:08X}:{bus:02X}:{device:02X}.0"


def _find_card(
    wanted: int | str | tuple[int, int, int], pci_order: bool,
) -> tuple[int, str] | None:
    """NVML's index of the card ``wanted`` names, and its PCI bus id. Makes NVML calls.

    ``wanted`` is a CUDA ordinal, a UUID or the prefix CUDA takes of one, or the place on
    the PCI bus (domain, bus, device) the trainer named. An ordinal is a place in CUDA's
    order, which is the PCI bus order with ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` and otherwise
    CUDA's guess at the fastest card first, which NVML can't know: of several cards in that
    order, this gives None and no card is guessed. A MIG instance is no whole card and has
    no index here. Raises if ``wanted`` names no single card: a reading of another card
    would be a false record.
    """
    import pynvml

    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    if isinstance(wanted, tuple):
        named = f"the trainer names {_bus_id(wanted)}"
        matches = [i for i, h in enumerate(handles) if _pci_place(pynvml, h) == wanted]
    elif isinstance(wanted, str):
        named = f"{_ENV} names {wanted}"
        matches = [
            i for i, h in enumerate(handles) if pynvml.nvmlDeviceGetUUID(h).startswith(wanted)
        ]
    elif pci_order:
        named = f"CUDA device {wanted} in PCI bus order"
        by_bus = sorted(range(len(handles)), key=lambda i: _pci_place(pynvml, handles[i]))
        matches = by_bus[wanted:wanted + 1]
    elif len(handles) > 1:
        return None
    else:
        named = f"CUDA device {wanted}"
        matches = list(range(len(handles)))[wanted:wanted + 1]
    if len(matches) != 1:
        raise LookupError(f"{named}: {len(matches)} cards match")
    index = matches[0]
    return index, _bus_id(_pci_place(pynvml, handles[index]))


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
    if last_good is None and summary.get("card_not_known"):
        sentences.append(
            "no reading of the card was recorded: which of the cards this job trains on "
            "was not known.",
        )
    elif last_good is not None:
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
        # Which card is the job's, as CUDA or the trainer names it, and NVML's index and bus
        # id of it once found. Both under ``_values_lock``.
        self._wanted: int | str | tuple[int, int, int] | None = visible_device(
            os.environ.get(_ENV),
        )
        self._pci_order = os.environ.get(_ORDER_ENV) == "PCI_BUS_ID"
        self._card: tuple[int, str] | None = None
        self._card_not_known = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._wake = threading.Event()  # the end of an interval: stopped, or a card named
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
        # The change to unreadable whose line was lost: the next unreadable reading owes it.
        self._change_unwritten = False
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
        if gpu_health.nvml_ruled_out() or self._wanted is None:
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
        self._wake.set()
        if self._gate.acquire(timeout=WRITE_WAIT_S + max(0.0, timeout)):
            self._gate.release()
        if self._thread is None:
            return True
        self._thread.join(max(0.0, deadline - time.monotonic()))
        return not self._thread.is_alive()

    def bind_device(self, pci_domain: int, pci_bus: int, pci_device: int) -> None:
        """Name the card the job trains on, by its place on the PCI bus. Makes no NVML call.

        The trainer knows it once torch has chosen the card, and what it names wins over
        what the environment gave. The next reading is of that card, and comes at once.
        """
        with self._values_lock:
            self._wanted = (int(pci_domain), int(pci_bus), int(pci_device))
            self._card = None
        self._wake.set()

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
                "card_not_known": self._card_not_known,
            }

    # ── The thread ────────────────────────────────────────────────────

    def _run(self) -> None:
        while True:
            reading = self._read()
            if reading is None:
                return
            if reading is not _NOT_KNOWN and not self._take(reading):
                return
            # A card not known yet waits for the trainer to name it, with no NVML call.
            self._wake.wait(None if reading is _NOT_KNOWN else self._interval_s)
            self._wake.clear()
            if self._stop_event.is_set():
                return

    def _read(self) -> Any:
        """One reading, holding no lock, with the bus id of its card.

        None means there is no NVML, and ``_NOT_KNOWN`` that which card is the job's
        isn't known yet.
        """
        try:
            if not gpu_health.nvml_ready():
                return None
            with self._values_lock:
                wanted, card = self._wanted, self._card
            if card is None:
                card = _find_card(wanted, self._pci_order)
                with self._values_lock:
                    self._card_not_known = card is None
                    if card is not None and self._wanted == wanted:  # not named anew
                        self._card = card
                if card is None:
                    return _NOT_KNOWN
            reading = self._reader(card[0])
        except Exception as exc:
            return {"readable": False, "error": _describe(exc)}
        return None if reading is None else {**reading, "pci_bus_id": card[1]}

    def _take(self, reading: dict[str, Any]) -> bool:
        """Record a reading and write its line, unless stopped. Returns whether to go on."""
        with self._gate:
            if self._stop_event.is_set():
                return False
            if self._record(reading) or self._change_unwritten:
                written = self._write(reading)
                self._change_unwritten = not written and reading.get("readable") is not True
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

    def _write(self, reading: dict[str, Any]) -> bool:
        """Whether the line was written. A failure costs the line and nothing more: the
        values are already recorded. The writer says so by returning False, or raises."""
        try:
            if self._writer.emit(_LINE_TYPE, reading) is not False:
                return True
            error = getattr(self._writer, "last_error", None) or "the job log refused it"
        except Exception as exc:
            error = _describe(exc)
        with self._values_lock:
            self._write_failures += 1
            first_failure = self._write_failures == 1
        if first_failure:
            # The job log is what just failed, and logging would route back into it;
            # stderr reaches trainer_stdout.log. Best-effort: this runs inside the gate
            # that stop waits on.
            note(f"WARNING: gpu_health line not written: {error}")
        return False
