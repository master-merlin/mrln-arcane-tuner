"""Writer-to-tailer smoke test for the file-based IPC channel.

Verifies that `JobLogWriter` -> `LogTailer` round-trips the message
types this codebase relies on (`status`, `warning`, `cache_ready`,
`log`) with correct `type` and `data` fields. This is the post-IPC-
migration regression guard: if the channel breaks, sampling progress
and caching progress messages stop reaching the UI.
"""
from __future__ import annotations

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
