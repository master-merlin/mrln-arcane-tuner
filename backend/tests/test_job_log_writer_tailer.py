"""Writer-to-tailer smoke test for the file-based IPC channel.

Verifies that `JobLogWriter` -> `LogTailer` round-trips the message
types this codebase relies on (`status`, `warning`, `cache_ready`,
`log`) with correct `type` and `data` fields. This is the post-IPC-
migration regression guard: if the channel breaks, sampling progress
and caching progress messages stop reaching the UI.
"""
from __future__ import annotations

import json
import os
import threading
import time

import pytest

from app.engine.components.job_log_writer import JobLogWriter
from app.core.log_tailer import LogTailer


@pytest.fixture
def writer_and_tailer(tmp_path):
    """Yield (writer, tailer, collected_entries) wired against a tmp dir.

    The tailer dispatches every parsed entry into a list. Tests inspect
    that list to verify what came through the channel.
    """
    collected: list[tuple[str, dict]] = []
    dispatch_event = threading.Event()

    def dispatcher(job_id: str, entry: dict) -> None:
        collected.append((job_id, entry))
        dispatch_event.set()

    writer = JobLogWriter(str(tmp_path))
    tailer = LogTailer(
        job_id="test-job",
        log_path=writer.log_path,
        dispatcher=dispatcher,
        poll_interval=0.05,
    )
    tailer.start()

    try:
        yield writer, tailer, collected, dispatch_event
    finally:
        tailer.stop()
        writer.close()


def _wait_for_n_entries(collected: list, n: int, timeout: float = 2.0) -> bool:
    """Spin-wait until `collected` has `n` items or `timeout` elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if len(collected) >= n:
            return True
        time.sleep(0.05)
    return False


def test_status_message_roundtrip(writer_and_tailer):
    writer, _tailer, collected, _ev = writer_and_tailer

    writer.status("Sampling 3/10")

    assert _wait_for_n_entries(collected, 1), (
        f"Tailer did not receive entry within timeout; got {collected}"
    )
    job_id, entry = collected[0]
    assert job_id == "test-job"
    assert entry["type"] == "status"
    assert entry["data"] == "Sampling 3/10"


def test_warning_message_roundtrip(writer_and_tailer):
    writer, _tailer, collected, _ev = writer_and_tailer

    writer.warning("Sampling skipped - nvfp4 quant unsupported")

    assert _wait_for_n_entries(collected, 1)
    _, entry = collected[0]
    assert entry["type"] == "warning"
    assert entry["data"] == "Sampling skipped - nvfp4 quant unsupported"


def test_cache_ready_message_roundtrip(writer_and_tailer):
    writer, _tailer, collected, _ev = writer_and_tailer

    writer.emit("cache_ready", ["dataset_a", "dataset_b"])

    assert _wait_for_n_entries(collected, 1)
    _, entry = collected[0]
    assert entry["type"] == "cache_ready"
    assert entry["data"] == ["dataset_a", "dataset_b"]


def test_log_message_roundtrip(writer_and_tailer):
    writer, _tailer, collected, _ev = writer_and_tailer

    writer.log("Trainable params: 12,345,678")

    assert _wait_for_n_entries(collected, 1)
    _, entry = collected[0]
    assert entry["type"] == "log"
    assert entry["data"] == "Trainable params: 12,345,678"


def test_multiple_messages_arrive_in_order(writer_and_tailer):
    writer, _tailer, collected, _ev = writer_and_tailer

    writer.status("Caching Latents (0%)")
    writer.status("Caching Latents (50%)")
    writer.status("Caching Latents (100%)")

    assert _wait_for_n_entries(collected, 3)
    statuses = [e["data"] for _, e in collected if e["type"] == "status"]
    assert statuses == [
        "Caching Latents (0%)",
        "Caching Latents (50%)",
        "Caching Latents (100%)",
    ]


def test_log_exception_writes_full_traceback_lines(tmp_path):
    """A crash's full traceback is forwarded as individual ``log`` entries.

    The trainer is a detached subprocess whose stderr only reaches
    ``trainer_stdout.log`` on disk; the UI reads ``job_log.jsonl``. Without
    this, the operator sees only the one-line exit error and the stack is
    effectively lost.
    """
    writer = JobLogWriter(str(tmp_path))

    def boom():
        raise ValueError("cublas-boom-xyz")

    try:
        boom()
    except ValueError as e:
        writer.log_exception(e)
    writer.close()

    with open(writer.log_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    assert all(e["type"] == "log" for e in entries)
    data = [e["data"] for e in entries]
    joined = "\n".join(data)
    assert "cublas-boom-xyz" in joined          # the message
    assert "ValueError" in joined               # the exception type
    assert "Traceback (most recent call last)" in joined
    assert any("boom" in line for line in data)  # the offending frame, not just the message
    # Each physical traceback line is its own entry so the viewer renders rows.
    assert all("\n" not in line for line in data)


def test_log_exception_no_writer_state_corruption(tmp_path):
    """Forwarding a traceback must not break a subsequent exit message."""
    writer = JobLogWriter(str(tmp_path))
    try:
        raise RuntimeError("x")
    except RuntimeError as e:
        writer.log_exception(e)
    writer.exit(1, error="x")  # must still succeed after log_exception

    with open(writer.log_path, encoding="utf-8") as f:
        types = [json.loads(line)["type"] for line in f if line.strip()]
    assert types[-1] == "exit"


def test_default_poll_interval_is_responsive_for_live_progress(tmp_path):
    """The tailer's default poll cadence bounds how fresh live progress is.

    Sampling/training step messages reach the UI only as fast as the tailer
    drains ``job_log.jsonl``. At the legacy 500 ms default, a fast run's steps
    arrive in coarse 2 Hz bursts, which the user sees as the progress counter
    "jumping" (e.g. 0 → 3 → 11 → 19) rather than counting smoothly. The default
    must stay responsive (≤ 200 ms) so live progress streams near real-time.
    """
    tailer = LogTailer(
        job_id="test-job",
        log_path=str(tmp_path / "job_log.jsonl"),
        dispatcher=lambda _job_id, _entry: None,
    )
    assert tailer.poll_interval == pytest.approx(0.15)
    assert tailer.poll_interval <= 0.2, (
        "Default poll interval must stay responsive (≤200ms) so live "
        "sampling/training progress streams smoothly instead of in bursts."
    )


def test_dispatcher_stopping_tailer_advances_offset_past_dispatched_line(tmp_path):
    # Regression: the exit-message handler in JobManager calls
    # _stop_tailer from inside the dispatched callback. Before the fix,
    # tailer.stop() would Thread.join(self) and raise, killing the
    # polling loop before its offset write — so on the next start
    # (after a restart) the same exit line was re-dispatched, marking
    # the restarted job FAILED and detaching the backend from the live
    # trainer subprocess.
    collected: list[dict] = []

    writer = JobLogWriter(str(tmp_path))
    writer.exit(code=1, error="EMA device mismatch")  # writes + closes

    final_size = os.path.getsize(writer.log_path)

    tailer_holder: dict[str, LogTailer] = {}

    def dispatcher(job_id: str, entry: dict) -> None:
        collected.append(entry)
        if entry.get("type") == "exit":
            # Mirror JobManager._handle_exit_message: stop from within
            # the tailer thread. Must not raise, and must persist offset.
            tailer_holder["t"].stop()

    tailer = LogTailer(
        job_id="test-job",
        log_path=writer.log_path,
        dispatcher=dispatcher,
        poll_interval=0.05,
    )
    tailer_holder["t"] = tailer
    tailer.start()

    deadline = time.time() + 2.0
    while time.time() < deadline and not collected:
        time.sleep(0.05)
    # Give the polling loop a moment to advance offset post-dispatch
    time.sleep(0.2)

    assert collected and collected[0]["type"] == "exit"
    with open(tailer.offset_path, "r", encoding="utf-8") as f:
        persisted_offset = int(f.read().strip())
    assert persisted_offset == final_size, (
        f"Offset should be at EOF after dispatching exit; "
        f"persisted={persisted_offset}, file_size={final_size}. "
        f"A stale offset re-dispatches the old exit on restart."
    )
