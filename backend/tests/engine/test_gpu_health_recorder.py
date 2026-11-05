"""The GPU health recorder beside the training loop, and the writer it shares.

Nothing here races on a clock: readers and writers are scripted, and a test waits
on an event. A wait that can only time out on correct code is short, so it costs
time and can't fail a test falsely; a wait for something that must happen is long.
"""

from __future__ import annotations

import ast
import atexit
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.core import gpu_health
from app.core.gpu_health import read_gpu_health
from app.engine.components import gpu_health_recorder, job_log_writer
from app.engine.components.gpu_health_recorder import (
    INTERVAL_S,
    GpuHealthRecorder,
    format_summary,
    visible_device,
)
from app.engine.components.job_log_writer import JobLogWriter

BACKEND = Path(__file__).resolve().parents[2]
RECORDER_SOURCE = Path(gpu_health_recorder.__file__)
REAL_NVML = gpu_health.nvml_ready()

MUST_HAPPEN_S = 30.0  # only spent when a test is about to fail


def good(**fields: Any) -> dict[str, Any]:
    return {
        "temperature_c": 60,
        "temperature_margin_c": 20,
        "fan_percent": 40,
        "thermal_throttle_ms": 0.0,
        "power_throttle_ms": 0.0,
        "throttle_reasons": [],
        "readable": True,
        **fields,
    }


def bad(error: str = "GPU is lost") -> dict[str, Any]:
    return {"readable": False, "error": error}


class FakeWriter:
    def __init__(self) -> None:
        self.lines: list[tuple[str, Any]] = []
        self._changed = threading.Condition()

    def emit(self, msg_type: str, data: Any) -> None:
        with self._changed:
            self.lines.append((msg_type, data))
            self._changed.notify_all()

    def wait_for(self, count: int) -> bool:
        with self._changed:
            return self._changed.wait_for(
                lambda: len(self.lines) >= count, MUST_HAPPEN_S,
            )

    def wait_for_briefly(self, count: int) -> bool:
        """For a line that must not come: a short wait, which only correct code spends."""
        with self._changed:
            return self._changed.wait_for(lambda: len(self.lines) >= count, 0.3)

    def temperatures(self) -> list[Any]:
        return [data.get("temperature_c") for _, data in self.lines]

    def readable(self) -> list[bool]:
        return [data["readable"] for _, data in self.lines]


class ScriptedReader:
    """Gives its readings in turn, then None, which ends the thread."""

    def __init__(self, *readings: Any) -> None:
        self.readings = list(readings)
        self.calls: list[int] = []
        self.threads: list[threading.Thread] = []

    def __call__(self, index: int) -> Any:
        self.calls.append(index)
        self.threads.append(threading.current_thread())
        if not self.readings:
            return None
        reading = self.readings.pop(0)
        if isinstance(reading, Exception):
            raise reading
        return reading


class EndlessReader:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _index: int) -> dict[str, Any]:
        self.calls += 1
        return good()


class HangingReader:
    """One good reading of 60 degrees, then the next call hangs until released."""

    def __init__(self) -> None:
        self.hung = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def __call__(self, _index: int) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            return good(temperature_c=60)
        self.hung.set()
        self.release.wait(MUST_HAPPEN_S * 4)
        return good(temperature_c=99)


class StallingWriter(FakeWriter):
    """A write that takes as long as the test says, its line landing at the end."""

    def __init__(self) -> None:
        super().__init__()
        self.in_write = threading.Event()
        self.release = threading.Event()

    def emit(self, msg_type: str, data: Any) -> None:
        self.in_write.set()
        self.release.wait(MUST_HAPPEN_S * 4)
        super().emit(msg_type, data)


def stop_in_thread(recorder: GpuHealthRecorder, timeout: float) -> queue.Queue:
    """Call ``stop`` on the side, so a stop that never returns fails the test.

    The queue gets (returned, seconds, lines at the moment it returned).
    """
    results: queue.Queue = queue.Queue()

    def call() -> None:
        started = time.monotonic()
        returned = recorder.stop(timeout=timeout)
        results.put(
            (returned, time.monotonic() - started, len(recorder._writer.lines)),
        )

    threading.Thread(target=call, daemon=True).start()
    return results


def finish(recorder: GpuHealthRecorder) -> None:
    """Wait for the thread to end by itself; the assertions after it are then final.

    It must not stop the recorder: a stop would cut a scripted reader short.
    """
    if recorder._thread is not None:
        recorder._thread.join(MUST_HAPPEN_S)
    assert recorder.running is False


@pytest.fixture(autouse=True)
def _card_is_there(monkeypatch: pytest.MonkeyPatch) -> None:
    # These tests run on machines with no NVML: the readers are fakes, and so is the one
    # card that NVML lists, unless a test lists others.
    monkeypatch.setattr(gpu_health, "_NVML_AVAILABLE", True)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.setitem(sys.modules, "pynvml", _FakeNvml("GPU-0000"))


def recorder_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "gpu-health-recorder"]


# ── Lines over time ─────────────────────────────────────────────────────


def test_the_lines_reach_the_real_job_log_at_the_interval(tmp_path: Path) -> None:
    writer = JobLogWriter(str(tmp_path))
    reader = ScriptedReader(good(), good(), good())
    recorder = GpuHealthRecorder(writer, reader=reader, interval_s=0.02)
    assert recorder.start() is True
    finish(recorder)
    writer.close()

    entries = [json.loads(line) for line in (tmp_path / "job_log.jsonl").open()]
    assert [entry["type"] for entry in entries] == ["gpu_health"] * 3
    assert entries[0]["data"]["temperature_c"] == 60
    gaps = [b["t"] - a["t"] for a, b in zip(entries, entries[1:])]
    assert all(gap >= 0.015 for gap in gaps), gaps  # a floor can't race


def test_the_interval_is_ten_seconds() -> None:
    assert INTERVAL_S == 10


def test_the_first_reading_comes_at_start_not_after_an_interval() -> None:
    writer = FakeWriter()
    recorder = GpuHealthRecorder(writer, reader=EndlessReader(), interval_s=3600)
    recorder.start()
    assert writer.wait_for(1)
    # The thread is asleep in a one-hour wait now: stop wakes it.
    assert recorder.stop(timeout=MUST_HAPPEN_S) is True
    assert len(writer.lines) == 1


def test_the_lines_keep_coming_while_the_main_thread_is_blocked() -> None:
    writer = FakeWriter()
    recorder = GpuHealthRecorder(writer, reader=EndlessReader(), interval_s=0.001)
    recorder.start()
    # The main thread does nothing but wait here, as a stalled loop would.
    assert writer.wait_for(5)
    recorder.stop()


def test_start_reads_nothing_on_the_calling_thread() -> None:
    reader = HangingReader()
    reader.calls = 1  # the first call already hangs
    recorder = GpuHealthRecorder(FakeWriter(), reader=reader, interval_s=3600)
    # A hung card must not hold up the job that is starting.
    assert recorder.start() is True
    assert reader.hung.wait(MUST_HAPPEN_S)
    recorder.stop(timeout=0.0)
    reader.release.set()
    finish(recorder)


# ── Unreadable ──────────────────────────────────────────────────────────


def test_one_line_at_each_change_of_readable_state() -> None:
    writer = FakeWriter()
    reader = ScriptedReader(
        good(temperature_c=1),
        bad("GPU is lost"),
        bad("Unknown Error"),  # another error is no change of state
        bad("GPU is lost"),
        good(temperature_c=2),
        good(temperature_c=3),
        bad(),
        bad(),
        good(temperature_c=4),
    )
    recorder = GpuHealthRecorder(writer, reader=reader, interval_s=0.0)
    recorder.start()
    finish(recorder)

    assert writer.readable() == [True, False, True, True, False, True]
    assert writer.temperatures() == [1, None, 2, 3, None, 4]
    assert writer.lines[1] == (
        "gpu_health", {**bad("GPU is lost"), "pci_bus_id": "00000000:01:00.0"},
    )


def test_a_card_unreadable_from_the_start_writes_one_line() -> None:
    writer = FakeWriter()
    recorder = GpuHealthRecorder(
        writer, reader=ScriptedReader(bad(), bad(), bad()), interval_s=0.0,
    )
    recorder.start()
    finish(recorder)
    assert writer.readable() == [False]


def test_a_reader_that_raises_counts_as_unreadable_and_the_thread_goes_on() -> None:
    writer = FakeWriter()
    reader = ScriptedReader(good(temperature_c=1), RuntimeError("boom"), good(temperature_c=2))
    recorder = GpuHealthRecorder(writer, reader=reader, interval_s=0.0)
    recorder.start()
    finish(recorder)

    assert writer.temperatures() == [1, None, 2]
    assert "boom" in writer.lines[1][1]["error"]
    assert recorder.summary()["was_unreadable"] is True


# ── Without NVML, without a card ────────────────────────────────────────


def test_without_nvml_no_thread_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_health, "_NVML_AVAILABLE", False)
    writer = FakeWriter()
    reader = ScriptedReader(good())
    recorder = GpuHealthRecorder(writer, reader=reader)

    assert recorder.start() is False
    assert recorder_threads() == []
    assert recorder.running is False
    assert reader.calls == []
    assert writer.lines == []
    assert recorder.stop() is True
    assert recorder.summary()["last_good"] is None


@pytest.mark.parametrize("value", ["", "-1", "-1,0", "NoDevFiles"])
def test_a_job_shown_no_card_starts_no_thread(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    recorder = GpuHealthRecorder(FakeWriter(), reader=ScriptedReader(good()))
    assert recorder.start() is False
    assert recorder_threads() == []


def test_none_from_the_reader_ends_the_thread_with_no_line() -> None:
    writer = FakeWriter()
    recorder = GpuHealthRecorder(writer, reader=ScriptedReader(), interval_s=0.0)
    recorder.start()
    finish(recorder)
    assert writer.lines == []


def test_a_thread_that_cannot_start_does_not_reach_the_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(_self: threading.Thread) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    recorder = GpuHealthRecorder(FakeWriter(), reader=EndlessReader())
    assert recorder.start() is False
    assert recorder.stop() is True


# ── Stop ────────────────────────────────────────────────────────────────


def test_stop_ends_the_thread_and_stopping_again_is_harmless() -> None:
    recorder = GpuHealthRecorder(FakeWriter(), reader=EndlessReader(), interval_s=0.001)
    recorder.start()
    assert recorder.running is True
    assert recorder.stop(timeout=MUST_HAPPEN_S) is True
    assert recorder.running is False
    assert recorder_threads() == []
    assert recorder.stop() is True


def test_stop_before_start_is_harmless_and_it_never_starts_after() -> None:
    recorder = GpuHealthRecorder(FakeWriter(), reader=EndlessReader())
    assert recorder.stop() is True
    assert recorder.start() is False
    assert recorder_threads() == []


def test_a_second_start_makes_no_second_thread() -> None:
    recorder = GpuHealthRecorder(FakeWriter(), reader=EndlessReader(), interval_s=3600)
    assert recorder.start() is True
    assert recorder.start() is False
    assert len(recorder_threads()) == 1
    recorder.stop(timeout=MUST_HAPPEN_S)


def test_a_hung_reader_bounds_stop_and_its_late_reading_writes_nothing() -> None:
    writer = FakeWriter()
    reader = HangingReader()
    recorder = GpuHealthRecorder(writer, reader=reader, interval_s=0.0)
    recorder.start()
    assert reader.hung.wait(MUST_HAPPEN_S)

    results = stop_in_thread(recorder, 2.0)
    try:
        returned, seconds, lines_at_stop = results.get(timeout=MUST_HAPPEN_S)
    finally:
        reader.release.set()  # also lets the thread go if stop never returned
    assert returned is False  # the thread is still inside the call
    assert 1.9 <= seconds < 2.5
    assert lines_at_stop == 1

    # The reading that was in flight arrives now.
    finish(recorder)
    assert writer.temperatures() == [60]
    assert recorder.summary()["peak_temperature_c"] == 60


def test_stop_waits_for_a_write_in_progress_past_its_timeout() -> None:
    writer = StallingWriter()
    recorder = GpuHealthRecorder(writer, reader=EndlessReader(), interval_s=3600)
    recorder.start()
    assert writer.in_write.wait(MUST_HAPPEN_S)

    results = stop_in_thread(recorder, 0.01)
    with pytest.raises(queue.Empty):
        results.get(timeout=0.2)  # 20 times its timeout: it holds for the write

    writer.release.set()
    returned, _seconds, lines_at_stop = results.get(timeout=MUST_HAPPEN_S)
    assert returned is True
    assert lines_at_stop == 1  # the line was down before stop returned


class _FiresOnFirstCheck(threading.Event):
    """Runs a hook in the gap right after the recorder's first look at the flag."""

    def __init__(self, hook: Any) -> None:
        super().__init__()
        self.hook = hook

    def is_set(self) -> bool:
        value = super().is_set()
        if threading.current_thread().name == "gpu-health-recorder":
            hook, self.hook = self.hook, None
            if hook is not None:
                hook()
        return value


def test_stop_cannot_return_between_the_check_and_the_write() -> None:
    writer = FakeWriter()
    recorder = GpuHealthRecorder(writer, reader=EndlessReader(), interval_s=3600)
    results: list[queue.Queue] = []
    returned_early: queue.Queue = queue.Queue()

    def stop_in_the_gap() -> None:
        # The recorder has seen "not stopped" and hasn't written. Its stop has a
        # timeout far shorter than this gap, so only the check and the write being
        # one step can hold it back.
        stopping = stop_in_thread(recorder, 0.01)
        results.append(stopping)
        try:
            returned_early.put(stopping.get(timeout=0.3))
        except queue.Empty:
            pass

    recorder._stop_event = _FiresOnFirstCheck(stop_in_the_gap)
    recorder.start()
    assert writer.wait_for(1)

    assert returned_early.empty(), "stop returned before the write"
    _returned, _seconds, lines_at_stop = results[0].get(timeout=MUST_HAPPEN_S)
    assert lines_at_stop == 1
    finish(recorder)
    assert len(writer.lines) == 1


def test_a_write_hung_for_good_cannot_hold_stop_or_the_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_health_recorder, "WRITE_WAIT_S", 0.05)
    writer = StallingWriter()
    recorder = GpuHealthRecorder(writer, reader=EndlessReader(), interval_s=3600)
    recorder.start()
    assert writer.in_write.wait(MUST_HAPPEN_S)

    try:
        returned, _seconds, lines_at_stop = stop_in_thread(recorder, 0.01).get(
            timeout=MUST_HAPPEN_S,
        )
        assert returned is False
        assert lines_at_stop == 0

        summaries: queue.Queue = queue.Queue()
        threading.Thread(
            target=lambda: summaries.put(recorder.summary()), daemon=True,
        ).start()
        assert summaries.get(timeout=MUST_HAPPEN_S)["peak_temperature_c"] == 60
    finally:
        writer.release.set()
    finish(recorder)


# ── A card whose NVML init hangs ────────────────────────────────────────


class _HangingInit:
    """Enough of pynvml for ``nvmlInit`` to hang until released, then fail."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.threads: list[threading.Thread] = []

    def nvmlInit(self) -> None:  # noqa: N802
        self.threads.append(threading.current_thread())
        self.entered.set()
        self.release.wait(MUST_HAPPEN_S * 4)
        raise OSError("no library")


@pytest.fixture
def hanging_init(monkeypatch: pytest.MonkeyPatch) -> Any:
    nvml = _HangingInit()
    monkeypatch.setattr(gpu_health, "pynvml", nvml)
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(gpu_health, "_NVML_AVAILABLE", None)
    yield nvml
    nvml.release.set()


def _returns_soon(call: Any) -> Any:
    """``call``'s result, from the side, so that a hang fails the test."""
    results: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: results.put(call()), daemon=True).start()
    return results.get(timeout=MUST_HAPPEN_S)


def test_a_hanging_init_runs_on_the_recorders_thread_and_holds_up_nothing(
    hanging_init: _HangingInit,
) -> None:
    writer = FakeWriter()
    recorder = GpuHealthRecorder(writer, interval_s=0.0)  # the real reader

    assert _returns_soon(recorder.start) is True
    assert hanging_init.entered.wait(MUST_HAPPEN_S)
    assert [thread.name for thread in hanging_init.threads] == ["gpu-health-recorder"]

    assert _returns_soon(lambda: recorder.stop(timeout=0.01)) is False
    summary = _returns_soon(recorder.summary)
    assert summary["last_good"] is None
    assert summary["was_unreadable"] is False

    # The init fails now, after the stop, and that writes no line.
    hanging_init.release.set()
    finish(recorder)
    assert writer.lines == []


def test_a_uuid_lookup_waits_for_the_init_on_the_same_thread(
    hanging_init: _HangingInit, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaaa")
    writer = FakeWriter()
    recorder = GpuHealthRecorder(writer, interval_s=0.0)
    assert _returns_soon(recorder.start) is True
    assert hanging_init.entered.wait(MUST_HAPPEN_S)
    assert recorder.running is True
    hanging_init.release.set()
    finish(recorder)  # the init failed: no NVML, no lookup, and the thread ended
    assert writer.lines == []


def test_a_reader_not_yet_tried_is_no_reason_to_stay_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only a known absence keeps the thread down: anything else takes a call to learn.
    monkeypatch.setattr(gpu_health, "_NVML_AVAILABLE", None)
    recorder = GpuHealthRecorder(FakeWriter(), reader=EndlessReader(), interval_s=3600)
    assert recorder.start() is True
    assert recorder.stop(timeout=MUST_HAPPEN_S) is True


def test_the_recorder_reads_no_private_flag_of_the_reader() -> None:
    assert "_NVML_AVAILABLE" not in RECORDER_SOURCE.read_text(encoding="utf-8")


# ── How long the record lasted ──────────────────────────────────────────


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _ClockedReader:
    """Each reading comes with a time: the clock is set before it is handed over."""

    def __init__(self, clock: _Clock, *timed: tuple[float, dict[str, Any]]) -> None:
        self.clock = clock
        self.timed = list(timed)

    def __call__(self, _index: int) -> dict[str, Any] | None:
        if not self.timed:
            return None
        self.clock.now, reading = self.timed.pop(0)
        return reading


def _timed_summary(end: float, *timed: tuple[float, dict[str, Any]]) -> dict[str, Any]:
    clock = _Clock()
    recorder = GpuHealthRecorder(
        FakeWriter(), reader=_ClockedReader(clock, *timed), interval_s=0.0,
    )
    recorder._clock = clock
    recorder.start()
    finish(recorder)
    clock.now = end
    return recorder.summary()


def test_the_record_knows_how_long_it_lasted() -> None:
    summary = _timed_summary(1090.0, (1000.0, good()), (1080.0, good()))
    assert summary["record_seconds"] == 90.0
    assert summary["last_reading_age_seconds"] == 10.0
    assert summary["first_unreadable_after_seconds"] is None
    assert summary["unreadable_seconds"] == 0.0


def test_the_record_knows_when_the_card_went_and_for_how_long() -> None:
    summary = _timed_summary(
        1300.0,
        (1000.0, good()),
        (1100.0, bad()),
        (1120.0, bad()),
        (1150.0, good()),
        (1250.0, bad()),
    )
    assert summary["first_unreadable_after_seconds"] == 100.0
    assert summary["unreadable_seconds"] == 100.0  # 50 and 50 up to the end
    assert summary["unreadable_at_end"] is True


def test_a_record_with_no_reading_has_no_times() -> None:
    recorder = GpuHealthRecorder(FakeWriter(), reader=ScriptedReader(), interval_s=0.0)
    recorder.start()
    finish(recorder)
    summary = recorder.summary()
    assert summary["record_seconds"] is None
    assert summary["last_reading_age_seconds"] is None


# ── The summary line ────────────────────────────────────────────────────


def _summary_of(**values: Any) -> dict[str, Any]:
    return {
        "peak_temperature_c": 78,
        "lowest_margin_c": 12,
        "peak_fan_percent": 62,
        "thermal_throttle_growth_ms": 3.0,
        "power_throttle_growth_ms": 192_000.0,
        "last_good": good(
            temperature_c=71, temperature_margin_c=19, fan_percent=55,
            power_draw_w=412.5, power_limit_w=600.0, clock_graphics_mhz=2430,
            clock_memory_mhz=10000, throttle_reasons=["sw_power_cap"],
        ),
        "was_unreadable": False,
        "unreadable_at_end": False,
        "last_error": None,
        "write_failures": 0,
        "record_seconds": 8040.0,
        "last_reading_age_seconds": 4.0,
        "first_unreadable_after_seconds": None,
        "unreadable_seconds": 0.0,
        **values,
    }


def test_the_line_says_it_all_in_words_and_units() -> None:
    line = format_summary(_summary_of())
    assert line.startswith("gpu_health_summary: ")
    assert "\n" not in line
    assert "recorded for 2 h 14 min" in line
    assert "peak temperature 78 C" in line
    assert "lowest margin 12 C" in line
    assert "peak fan 62%" in line
    assert "thermal throttling grew by 3 ms" in line
    assert "power throttling grew by 3.2 min" in line
    assert "last good reading: 71 C" in line
    assert "margin 19 C" in line
    assert "fan 55%" in line
    assert "412.5 W of 600 W" in line
    assert "sw_power_cap" in line
    assert "unreadable" not in line
    assert "None" not in line


def test_the_line_says_what_is_not_known_rather_than_none() -> None:
    line = format_summary(
        _summary_of(
            lowest_margin_c=None,
            peak_fan_percent=None,
            thermal_throttle_growth_ms=None,
            power_throttle_growth_ms=None,
            last_good=good(
                temperature_margin_c=None, fan_percent=None, power_draw_w=None,
                power_limit_w=None, throttle_reasons=None,
            ),
        ),
    )
    assert "lowest margin not known" in line
    assert "peak fan not known" in line
    assert "thermal throttling not known" in line
    assert "power throttling not known" in line
    assert "None" not in line


def test_zero_reads_as_zero_and_not_as_unknown() -> None:
    line = format_summary(
        _summary_of(
            lowest_margin_c=0, peak_fan_percent=0, thermal_throttle_growth_ms=0.0,
        ),
    )
    assert "lowest margin 0 C" in line
    assert "peak fan 0%" in line
    assert "thermal throttling grew by 0 ms" in line


def test_the_line_of_a_card_lost_to_the_end() -> None:
    line = format_summary(
        _summary_of(
            was_unreadable=True,
            unreadable_at_end=True,
            last_error="GPU is lost",
            first_unreadable_after_seconds=3600.0,
            unreadable_seconds=120.0,
        ),
    )
    assert "unreadable at the end of the job" in line
    assert "first 1 h 0 min into the record" in line
    assert "2 min in all" in line
    assert "GPU is lost" in line
    assert "last good reading: 71 C" in line


def test_the_line_of_a_card_that_came_back() -> None:
    line = format_summary(
        _summary_of(
            was_unreadable=True,
            last_error="Unknown Error",
            first_unreadable_after_seconds=30.0,
            unreadable_seconds=20.0,
        ),
    )
    assert "unreadable for 20 s" in line
    assert "came back" in line
    assert "at the end of the job" not in line


def test_the_line_of_a_card_never_read() -> None:
    line = format_summary(
        _summary_of(
            peak_temperature_c=None, lowest_margin_c=None, peak_fan_percent=None,
            thermal_throttle_growth_ms=None, power_throttle_growth_ms=None,
            last_good=None, record_seconds=None, last_reading_age_seconds=None,
        ),
    )
    assert line.startswith("gpu_health_summary: no reading")
    assert "None" not in line


def test_the_line_of_a_card_unreadable_from_the_start() -> None:
    line = format_summary(
        _summary_of(
            peak_temperature_c=None, lowest_margin_c=None, peak_fan_percent=None,
            thermal_throttle_growth_ms=None, power_throttle_growth_ms=None,
            last_good=None, was_unreadable=True, unreadable_at_end=True,
            last_error="GPU is lost", first_unreadable_after_seconds=0.0,
            unreadable_seconds=60.0,
        ),
    )
    assert "no good reading" in line
    assert "GPU is lost" in line
    assert "None" not in line


def test_the_line_says_when_the_recorder_was_still_waiting() -> None:
    line = format_summary(_summary_of(last_reading_age_seconds=34.0), ended=False)
    assert "had not returned" in line
    assert "the last reading came 34 s before the end" in line
    assert "had not returned" not in format_summary(_summary_of())


def test_the_line_counts_the_readings_it_could_not_write() -> None:
    assert "3 readings could not be written" in format_summary(_summary_of(write_failures=3))
    assert "could not be written" not in format_summary(_summary_of())


def test_an_error_text_stays_on_the_one_line() -> None:
    line = format_summary(
        _summary_of(was_unreadable=True, unreadable_at_end=True, last_error="a\r\nb"),
    )
    assert "\n" not in line
    assert "\r" not in line


# ── A writer that fails ─────────────────────────────────────────────────


class FailingWriter(FakeWriter):
    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    def emit(self, msg_type: str, data: Any) -> None:
        if self.failures > 0:
            self.failures -= 1
            raise OSError("No space left on device")
        super().emit(msg_type, data)


def test_a_writer_that_raises_costs_the_line_and_not_the_thread(
    capsys: pytest.CaptureFixture[str],
) -> None:
    writer = FailingWriter(failures=3)
    reader = ScriptedReader(
        good(temperature_c=90), good(temperature_c=50), good(temperature_c=51),
        good(temperature_c=52), good(temperature_c=53),
    )
    recorder = GpuHealthRecorder(writer, reader=reader, interval_s=0.0)
    recorder.start()
    finish(recorder)  # also: nothing reached this thread

    assert writer.temperatures() == [52, 53]
    summary = recorder.summary()
    assert summary["peak_temperature_c"] == 90  # it was recorded all the same
    assert summary["write_failures"] == 3
    # A person finds it in the trainer's stderr, once, not once per interval.
    assert capsys.readouterr().err.count("No space left on device") == 1


def test_a_closed_real_writer_is_no_failure_at_all(tmp_path: Path) -> None:
    writer = JobLogWriter(str(tmp_path))
    writer.close()
    recorder = GpuHealthRecorder(writer, reader=ScriptedReader(good()), interval_s=0.0)
    recorder.start()
    finish(recorder)
    assert (tmp_path / "job_log.jsonl").read_text() == ""
    assert recorder.summary()["write_failures"] == 0


# ── The recorded values ─────────────────────────────────────────────────


def record(*readings: Any) -> dict[str, Any]:
    recorder = GpuHealthRecorder(
        FakeWriter(), reader=ScriptedReader(*readings), interval_s=0.0,
    )
    recorder.start()
    finish(recorder)
    return recorder.summary()


def test_the_peaks_and_the_lowest_margin() -> None:
    summary = record(
        good(temperature_c=60, temperature_margin_c=10, fan_percent=30),
        good(temperature_c=71, temperature_margin_c=0, fan_percent=0),
        good(temperature_c=65, temperature_margin_c=5, fan_percent=55),
        good(temperature_c=None, temperature_margin_c=None, fan_percent=None),
    )
    assert summary["peak_temperature_c"] == 71
    assert summary["lowest_margin_c"] == 0  # zero is a value, not a missing one
    assert summary["peak_fan_percent"] == 55
    assert summary["unreadable_at_end"] is False
    assert summary["was_unreadable"] is False
    assert summary["last_error"] is None


def test_a_margin_past_the_limit_is_the_lowest() -> None:
    summary = record(good(temperature_margin_c=3), good(temperature_margin_c=-2))
    assert summary["lowest_margin_c"] == -2


def test_counter_growth_since_the_job_started() -> None:
    summary = record(
        good(thermal_throttle_ms=2.0, power_throttle_ms=100.0),
        good(thermal_throttle_ms=5.0, power_throttle_ms=100.0),
    )
    assert summary["thermal_throttle_growth_ms"] == 3.0
    assert summary["power_throttle_growth_ms"] == 0.0


def test_counter_growth_survives_an_unreadable_stretch() -> None:
    summary = record(
        good(thermal_throttle_ms=2.0),
        bad(),
        bad(),
        good(thermal_throttle_ms=5.0),
        bad(),
    )
    assert summary["thermal_throttle_growth_ms"] == 3.0
    assert summary["unreadable_at_end"] is True


def test_a_counter_going_backwards_gives_no_negative_growth() -> None:
    summary = record(
        good(thermal_throttle_ms=100.0),
        good(thermal_throttle_ms=110.0),
        good(thermal_throttle_ms=4.0),  # a driver reset: 4 is the new baseline
        good(thermal_throttle_ms=6.0),
    )
    assert summary["thermal_throttle_growth_ms"] == 12.0


def test_a_counter_missing_for_a_while_keeps_its_baseline() -> None:
    summary = record(
        good(power_throttle_ms=None),
        good(power_throttle_ms=7.0),
        good(power_throttle_ms=None),
        good(power_throttle_ms=9.0),
    )
    assert summary["power_throttle_growth_ms"] == 2.0


def test_values_the_card_never_offered_stay_none() -> None:
    summary = record(
        good(
            temperature_margin_c=None, fan_percent=None,
            thermal_throttle_ms=None, power_throttle_ms=None,
        ),
    )
    assert summary["lowest_margin_c"] is None
    assert summary["peak_fan_percent"] is None
    assert summary["thermal_throttle_growth_ms"] is None
    assert summary["power_throttle_growth_ms"] is None
    assert summary["peak_temperature_c"] == 60


def test_the_last_good_reading_outlives_the_card() -> None:
    summary = record(good(temperature_c=70), bad("GPU is lost"), bad("GPU is lost"))
    assert summary["last_good"]["temperature_c"] == 70
    assert summary["was_unreadable"] is True
    assert summary["unreadable_at_end"] is True
    assert summary["last_error"] == "GPU is lost"


def test_a_card_that_came_back_was_unreadable_but_is_not_at_the_end() -> None:
    summary = record(good(temperature_c=70), bad(), good(temperature_c=71))
    assert summary["last_good"]["temperature_c"] == 71
    assert summary["was_unreadable"] is True
    assert summary["unreadable_at_end"] is False


def test_a_card_never_read_has_no_good_reading() -> None:
    summary = record(bad(), bad())
    assert summary["last_good"] is None
    assert summary["peak_temperature_c"] is None
    assert summary["thermal_throttle_growth_ms"] is None
    assert summary["was_unreadable"] is True


def test_the_summary_makes_no_call_and_is_the_callers_own_copy() -> None:
    reader = ScriptedReader(good(throttle_reasons=["hw_thermal"]))
    recorder = GpuHealthRecorder(FakeWriter(), reader=reader, interval_s=0.0)
    recorder.start()
    finish(recorder)
    calls = len(reader.calls)

    first = recorder.summary()
    first["last_good"]["throttle_reasons"].append("scribbled")
    assert recorder.summary()["last_good"]["throttle_reasons"] == ["hw_thermal"]
    assert len(reader.calls) == calls


# ── The device index ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 0),
        ("0", 0),
        ("1", 1),
        ("1,0", 1),
        (" 2 , 0 ", 2),
        ("GPU-d09a7952-6fc1", "GPU-d09a7952-6fc1"),
        ("GPU-d09a,0", "GPU-d09a"),
        ("MIG-abc", "MIG-abc"),
        ("", None),
        ("  ", None),
        ("-1", None),
        ("-1,0", None),
        ("abc", None),
        ("1.5", None),
        (",0", None),
    ],
)
def test_visible_device(value: str | None, expected: Any) -> None:
    assert visible_device(value) == expected


class _FakeNvml:
    """Enough of pynvml to look a card up by its UUID or its place on the PCI bus.

    ``buses`` gives each NVML index its bus number; by default they rise with the index.
    """

    def __init__(self, *uuids: str, buses: tuple[int, ...] | None = None) -> None:
        self.uuids = uuids
        self.buses = buses or tuple(range(1, len(uuids) + 1))
        self.threads: list[threading.Thread] = []

    def nvmlDeviceGetCount(self) -> int:  # noqa: N802
        self.threads.append(threading.current_thread())
        return len(self.uuids)

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:  # noqa: N802
        return index

    def nvmlDeviceGetUUID(self, handle: int) -> str:  # noqa: N802
        return self.uuids[handle]

    def nvmlDeviceGetPciInfo(self, handle: int) -> Any:  # noqa: N802
        return types.SimpleNamespace(domain=0, bus=self.buses[handle], device=0)


def _cards(monkeypatch: pytest.MonkeyPatch, *uuids: str, **kwargs: Any) -> _FakeNvml:
    nvml = _FakeNvml(*uuids, **kwargs)
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    return nvml


@pytest.mark.parametrize("value", [None, "0"])
def test_the_one_card_there_is_the_jobs_card(
    monkeypatch: pytest.MonkeyPatch, value: str | None,
) -> None:
    _cards(monkeypatch, "GPU-aaaa", buses=(0x41,))
    if value is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    reader = ScriptedReader(good())
    writer = FakeWriter()
    recorder = GpuHealthRecorder(writer, reader=reader, interval_s=0.0)
    recorder.start()
    finish(recorder)
    assert set(reader.calls) == {0}
    assert writer.lines[0][1]["pci_bus_id"] == "00000000:41:00.0"  # which card it was


@pytest.mark.parametrize(
    ("value", "index"), [(None, 2), ("0", 2), ("1", 0), ("2,0", 1)],
)
def test_in_pci_bus_order_cudas_ordinal_is_the_cards_place_on_the_bus(
    monkeypatch: pytest.MonkeyPatch, value: str | None, index: int,
) -> None:
    # NVML's own order need not be the bus order: here bus 0x01 is NVML's index 2.
    _cards(monkeypatch, "GPU-a", "GPU-b", "GPU-c", buses=(0x21, 0x41, 0x01))
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if value is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    reader = ScriptedReader(good())
    recorder = GpuHealthRecorder(FakeWriter(), reader=reader, interval_s=0.0)
    recorder.start()
    finish(recorder)
    assert set(reader.calls) == {index}


@pytest.mark.parametrize("value", [None, "0", "1"])
def test_in_cudas_own_order_of_several_cards_no_card_is_guessed_until_the_trainer_says(
    monkeypatch: pytest.MonkeyPatch, value: str | None,
) -> None:
    # CUDA puts its guess at the fastest card first, and NVML knows nothing of that guess.
    _cards(monkeypatch, "GPU-a", "GPU-b", buses=(0x01, 0x41))
    if value is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    writer = FakeWriter()
    reader = ScriptedReader(good(), good())
    recorder = GpuHealthRecorder(writer, reader=reader, interval_s=0.0)
    assert recorder.start() is True
    try:
        assert not writer.wait_for_briefly(1)
        assert reader.calls == []

        recorder.bind_device(0, 0x41, 0)  # what torch says of the card it trains on
        finish(recorder)
    finally:
        recorder.stop(timeout=MUST_HAPPEN_S)
    assert set(reader.calls) == {1}
    assert len(writer.lines) == 2
    assert {data["pci_bus_id"] for _, data in writer.lines} == {"00000000:41:00.0"}


def test_a_job_whose_card_was_never_named_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    _cards(monkeypatch, "GPU-a", "GPU-b")
    recorder = GpuHealthRecorder(FakeWriter(), reader=ScriptedReader(good()), interval_s=0.0)
    recorder.start()
    deadline = time.monotonic() + MUST_HAPPEN_S
    while not recorder.summary().get("card_not_known") and time.monotonic() < deadline:
        time.sleep(0.01)
    assert recorder.stop(timeout=MUST_HAPPEN_S) is True
    line = format_summary(recorder.summary())
    assert "which of the cards this job trains on was not known" in line
    assert "no NVML" not in line


def test_the_trainers_card_wins_over_one_found_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cards(monkeypatch, "GPU-a", "GPU-b", buses=(0x01, 0x41))
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    writer = FakeWriter()
    reader = EndlessReader()
    recorder = GpuHealthRecorder(writer, reader=reader, interval_s=3600)
    recorder.start()
    try:
        assert writer.wait_for(1)
        recorder.bind_device(0, 0x41, 0)
        assert writer.wait_for(2)  # the bind wakes it: no wait for the next interval
    finally:
        assert recorder.stop(timeout=MUST_HAPPEN_S) is True
    assert [data["pci_bus_id"] for _, data in writer.lines[:2]] == [
        "00000000:01:00.0", "00000000:41:00.0",
    ]


def test_a_bound_card_that_nvml_does_not_have_is_unreadable_and_none_is_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cards(monkeypatch, "GPU-a", "GPU-b")
    writer = FakeWriter()
    calls: list[int] = []
    recorder = GpuHealthRecorder(writer, reader=calls.append, interval_s=3600)
    recorder.start()
    recorder.bind_device(0, 0x99, 0)
    try:
        assert writer.wait_for(1)
    finally:
        assert recorder.stop(timeout=MUST_HAPPEN_S) is True
    assert calls == []
    assert writer.readable() == [False]
    assert "00000000:99:00.0" in writer.lines[0][1]["error"]


@pytest.mark.parametrize(
    "value", ["GPU-bbbb-1111", "GPU-bbbb", "GPU-bbbb,0"],
)
def test_a_uuid_becomes_nvmls_index_by_full_name_or_prefix(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    nvml = _FakeNvml("GPU-aaaa-1111", "GPU-bbbb-1111")
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    reader = ScriptedReader(good())
    recorder = GpuHealthRecorder(FakeWriter(), reader=reader, interval_s=0.0)
    recorder.start()
    finish(recorder)
    assert set(reader.calls) == {1}
    # The lookup is an NVML call, so it too belongs on the recorder's thread.
    assert nvml.threads
    assert threading.current_thread() not in nvml.threads


@pytest.mark.parametrize(
    "value", ["GPU-cccc", "GPU-", "MIG-aaaa"],  # no such card; two cards; not a whole card
)
def test_a_uuid_naming_no_single_card_is_unreadable_and_no_card_is_guessed(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", _FakeNvml("GPU-aaaa-1111", "GPU-bbbb-1111"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    writer = FakeWriter()
    calls: list[int] = []
    recorder = GpuHealthRecorder(writer, reader=calls.append, interval_s=3600)
    assert recorder.start() is True
    assert writer.wait_for(1)
    assert recorder.stop(timeout=MUST_HAPPEN_S) is True

    assert calls == []
    assert writer.readable() == [False]
    assert value in writer.lines[0][1]["error"]
    assert recorder.summary()["was_unreadable"] is True


# ── It can't change training ────────────────────────────────────────────


def _imported_roots(source: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_the_module_imports_nothing_that_touches_the_training_state() -> None:
    imported = _imported_roots(RECORDER_SOURCE)
    assert not imported & {"torch", "numpy", "random"}, imported


def _run_python(code: str) -> str:
    # Unset, so that the job's device is 0 and only NVML decides whether it starts. Neither
    # script imports torch, and the first one proves it.
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=BACKEND, env=env, capture_output=True, text=True, timeout=120, check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip().splitlines()[-1]


def test_its_real_import_chain_and_a_run_pull_no_torch_or_numpy() -> None:
    last_line = _run_python(
        """
import json, sys, threading
from app.engine.components.gpu_health_recorder import GpuHealthRecorder
from app.core import gpu_health
gpu_health._NVML_AVAILABLE = True
class W:
    got = threading.Event()
    def emit(self, *_): self.got.set()
r = GpuHealthRecorder(W(), reader=lambda _i: {"readable": True}, interval_s=0.0)
r.start()
assert W.got.wait(30)
r.stop()
r.summary()
print(json.dumps(sorted(m for m in ("torch", "numpy") if m in sys.modules)))
""",
    )
    assert json.loads(last_line) == []


def test_with_no_pynvml_at_all_the_real_chain_starts_nothing() -> None:
    last_line = _run_python(
        """
import json, sys, threading
sys.modules["pynvml"] = None  # import pynvml now raises ImportError
from app.engine.components.gpu_health_recorder import GpuHealthRecorder
class W:
    lines = []
    def emit(self, *args): self.lines.append(args)
w = W()
r = GpuHealthRecorder(w)
started = r.start()
print(json.dumps([started, threading.active_count(), r.stop(), w.lines]))
""",
    )
    assert json.loads(last_line) == [False, 1, True, []]


@pytest.mark.parametrize("use_real_card", [False, True])
def test_the_random_states_are_the_same_after_running(
    monkeypatch: pytest.MonkeyPatch, use_real_card: bool,
) -> None:
    torch = pytest.importorskip("torch")
    if use_real_card and not REAL_NVML:
        pytest.skip("no NVML on this machine")

    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    python_before = random.getstate()

    writer = FakeWriter()
    reader = read_gpu_health if use_real_card else EndlessReader()
    recorder = GpuHealthRecorder(writer, reader=reader, interval_s=0.001)
    recorder.start()
    assert writer.wait_for(5)
    assert recorder.stop(timeout=MUST_HAPPEN_S) is True

    assert torch.equal(torch.get_rng_state(), torch_before)
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert random.getstate() == python_before


# ── The writer, now written from two threads ────────────────────────────


def _hammer(writer: JobLogWriter, threads: int, lines: int) -> None:
    start = threading.Barrier(threads)

    def emit_all(thread: int) -> None:
        start.wait(MUST_HAPPEN_S)
        for line in range(lines):
            # Lengths on both sides of the buffer's size, so flushes land anywhere.
            writer.emit("step", {"id": [thread, line], "pad": "x" * (line * 37 % 9000)})

    workers = [threading.Thread(target=emit_all, args=(i,)) for i in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()


def test_emit_from_8_threads_gives_only_whole_parseable_lines(tmp_path: Path) -> None:
    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        writer = JobLogWriter(str(tmp_path))
        _hammer(writer, threads=8, lines=500)
        writer.close()
    finally:
        sys.setswitchinterval(switch_interval)

    text = (tmp_path / "job_log.jsonl").read_text(encoding="utf-8")
    assert text.endswith("\n")
    ids = [tuple(json.loads(line)["data"]["id"]) for line in text.splitlines()]
    assert sorted(ids) == [(t, n) for t in range(8) for n in range(500)]


class _FileThatWritesInTwoParts:
    """A file whose write isn't atomic, as nothing promises that it is.

    CPython's text layer happens to be nearly atomic under the GIL, which is why the
    test above can pass with no lock at all. This one takes that luck away: each
    write stops halfway until a second writer gets in or a short wait runs out.
    """

    closed = False

    def __init__(self) -> None:
        self.parts: list[str] = []
        self._inside = 0
        self._guard = threading.Lock()
        self.overlapped = threading.Event()

    def write(self, text: str) -> None:
        with self._guard:
            self._inside += 1
            if self._inside > 1:
                self.overlapped.set()
        middle = len(text) // 2
        self.parts.append(text[:middle])
        self.overlapped.wait(0.02)
        self.parts.append(text[middle:])
        with self._guard:
            self._inside -= 1

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def test_emit_holds_its_lock_across_the_whole_write(tmp_path: Path) -> None:
    writer = JobLogWriter(str(tmp_path))
    writer._file.close()
    fake = _FileThatWritesInTwoParts()
    writer._file = fake

    _hammer(writer, threads=8, lines=3)

    assert not fake.overlapped.is_set()
    lines = "".join(fake.parts).splitlines()
    assert len(lines) == 24
    for line in lines:
        json.loads(line)


class _FileThatLogsWhileWriting(_FileThatWritesInTwoParts):
    """A finaliser or a warning can log on the thread that is mid-write."""

    def __init__(self, writer: JobLogWriter) -> None:
        super().__init__()
        self.writer = writer

    def write(self, text: str) -> None:
        writer, self.writer = self.writer, None
        if writer is not None:
            writer.emit("log", "from inside a write")
        self.parts.append(text)


def test_emit_from_inside_a_write_on_the_same_thread_does_not_deadlock(
    tmp_path: Path,
) -> None:
    writer = JobLogWriter(str(tmp_path))
    # If this deadlocks, the close at exit would wait on the same lock for ever.
    atexit.unregister(writer.close)
    writer._file.close()
    fake = _FileThatLogsWhileWriting(writer)
    writer._file = fake

    worker = threading.Thread(target=writer.emit, args=("log", "outer"), daemon=True)
    worker.start()
    worker.join(MUST_HAPPEN_S)
    assert not worker.is_alive(), "emit deadlocked on its own lock"
    assert len(fake.parts) == 2


def test_nothing_can_be_written_between_the_exit_line_and_the_close(
    tmp_path: Path,
) -> None:
    writer = JobLogWriter(str(tmp_path))
    close = writer.close
    late: list[threading.Thread] = []

    def another_thread_writes_first() -> None:
        # The exit line is down and the file still open: the worst moment for the
        # recorder to write. It has this long to get in, and must not.
        thread = threading.Thread(target=writer.emit, args=("gpu_health", {"readable": True}))
        thread.start()
        thread.join(0.2)
        late.append(thread)
        close()

    writer.close = another_thread_writes_first
    writer.exit(0)
    writer.close = close
    late[0].join(MUST_HAPPEN_S)

    lines = (tmp_path / "job_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["type"] for line in lines] == ["exit"]


class _FileThatStallsOnAReading(_FileThatWritesInTwoParts):
    """A disk that stalls on the recorder's line, until released."""

    def __init__(self) -> None:
        super().__init__()
        self.in_write = threading.Event()
        self.release = threading.Event()

    def write(self, text: str) -> None:
        if '"gpu_health"' in text:
            self.in_write.set()
            self.release.wait(MUST_HAPPEN_S * 4)
        self.parts.append(text)


def _writer_on(tmp_path: Path, fake: Any) -> JobLogWriter:
    writer = JobLogWriter(str(tmp_path))
    atexit.unregister(writer.close)
    writer._file.close()
    writer._file = fake
    return writer


def test_a_stuck_write_holds_another_threads_line_for_one_bound_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(job_log_writer, "LOCK_WAIT_S", 0.5, raising=False)
    fake = _FileThatStallsOnAReading()
    writer = _writer_on(tmp_path, fake)
    stuck = threading.Thread(
        target=writer.emit, args=("gpu_health", {"readable": True}), daemon=True,
    )
    stuck.start()
    assert fake.in_write.wait(MUST_HAPPEN_S)
    try:
        results: queue.Queue = queue.Queue()

        def training_thread() -> None:
            started = time.monotonic()
            written = [writer.step({"step": n}) for n in range(4)] + [writer.log("next")]
            results.put((written, time.monotonic() - started))

        threading.Thread(target=training_thread, daemon=True).start()
        # Waiting on the stuck write for good would leave this queue empty.
        written, seconds = results.get(timeout=10.0)
        assert written == [False] * 5  # each line lost, and each one says so
        assert seconds < 1.5  # one bound of 0.5 s, and no more waiting while it is stuck
    finally:
        fake.release.set()
    stuck.join(MUST_HAPPEN_S)

    # The disk is back: the next line gets through, and waits as before.
    assert writer.log("back") is True
    assert writer.stuck is False
    assert '"back"' in fake.parts[-1]


def test_a_stuck_write_bounds_the_exit_line_and_the_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(job_log_writer, "LOCK_WAIT_S", 0.1, raising=False)
    fake = _FileThatStallsOnAReading()
    writer = _writer_on(tmp_path, fake)
    threading.Thread(
        target=writer.emit, args=("gpu_health", {"readable": True}), daemon=True,
    ).start()
    assert fake.in_write.wait(MUST_HAPPEN_S)
    try:
        assert _returns_within(5.0, lambda: writer.exit(0)) is None
        assert _returns_within(5.0, writer.close) is None
        assert not any('"exit"' in part for part in fake.parts)
    finally:
        fake.release.set()


def _returns_within(seconds: float, call: Any) -> Any:
    results: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: results.put(call()), daemon=True).start()
    return results.get(timeout=seconds)


def test_a_line_is_written_and_says_so(tmp_path: Path) -> None:
    writer = JobLogWriter(str(tmp_path))
    assert writer.log("one") is True
    writer.close()


def test_emit_after_close_writes_nothing_and_does_not_raise(tmp_path: Path) -> None:
    writer = JobLogWriter(str(tmp_path))
    writer.close()
    writer.emit("log", "late")
    assert (tmp_path / "job_log.jsonl").read_text(encoding="utf-8") == ""


# ── A disk error, through the real writer ───────────────────────────────


class _FileThatFailsOn(_FileThatWritesInTwoParts):
    """A disk that is full for the writes it is told, and fine for the rest."""

    def __init__(self, *failing: int) -> None:
        super().__init__()
        self.failing = set(failing)
        self.writes = 0

    def write(self, text: str) -> None:
        self.writes += 1
        if self.writes in self.failing:
            raise OSError(28, "No space left on device")
        self.parts.append(text)

    def lines(self) -> list[dict[str, Any]]:
        return [json.loads(part)["data"] for part in self.parts]


def test_a_disk_error_in_the_real_writer_is_counted_and_told_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    fake = _FileThatFailsOn(1, 2)
    writer = _writer_on(tmp_path, fake)
    reader = ScriptedReader(
        good(temperature_c=90), good(temperature_c=50), good(temperature_c=51),
    )
    recorder = GpuHealthRecorder(writer, reader=reader, interval_s=0.0)
    recorder.start()
    finish(recorder)

    assert [line["temperature_c"] for line in fake.lines()] == [51]
    summary = recorder.summary()
    assert summary["peak_temperature_c"] == 90  # recorded all the same
    assert summary["write_failures"] == 2
    assert capsys.readouterr().err.count("No space left on device") == 1


def test_a_lost_change_to_unreadable_is_written_at_the_next_reading(tmp_path: Path) -> None:
    fake = _FileThatFailsOn(2)  # the line saying the card went
    writer = _writer_on(tmp_path, fake)
    reader = ScriptedReader(good(), bad("lost once"), bad("lost twice"), bad("lost thrice"))
    recorder = GpuHealthRecorder(writer, reader=reader, interval_s=0.0)
    recorder.start()
    finish(recorder)

    # The change is on record once the disk takes it, and only once.
    assert [line["readable"] for line in fake.lines()] == [True, False]
    assert fake.lines()[1]["error"] == "lost twice"
    assert recorder.summary()["write_failures"] == 1
