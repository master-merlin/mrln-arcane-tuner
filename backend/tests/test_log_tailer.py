"""LogTailer offset-persistence — batched + atomic (W4.T8).

Before this, ``_save_offset`` did a full open/write/close per DISPATCHED
line — at high-rate per-step trainer logging that's constant disk churn
alongside training I/O. The write was also non-atomic: a crash mid-write
left a partial int, ``_load_offset`` failed ``int()`` and reset to 0,
re-dispatching the entire log on reattach.

The 0.15s poll cadence (test_default_poll_interval_is_responsive_for_live_progress
in test_job_log_writer_tailer.py) is load-bearing and untouched here — only
the offset-persistence cadence is batched, not dispatch itself.
"""

from __future__ import annotations

import json
import os
import time

from app.core.log_tailer import LogTailer


def _make_tailer(tmp_path, dispatcher=None, poll_interval: float = 0.02) -> LogTailer:
    log_path = tmp_path / "job_log.jsonl"
    log_path.write_text("")
    return LogTailer(
        job_id="job",
        log_path=str(log_path),
        dispatcher=dispatcher or (lambda _j, _e: None),
        poll_interval=poll_interval,
    )


def test_offset_persisted_in_batches_not_per_line(tmp_path, monkeypatch):
    """Dispatching 100 lines must write the offset file far fewer than 100
    times — batched by line-count/interval, plus one final force-save on
    stop()."""
    collected: list[dict] = []
    tailer = _make_tailer(tmp_path, dispatcher=lambda _j, e: collected.append(e))

    offset_opens = {"count": 0}
    real_open = open

    def counting_open(path, *args, **kwargs):
        p = str(path)
        if p in (tailer.offset_path, tailer.offset_path + ".tmp"):
            offset_opens["count"] += 1
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", counting_open)

    tailer.start()
    try:
        with real_open(tailer.log_path, "a", encoding="utf-8") as f:
            for i in range(100):
                f.write(json.dumps({"type": "log", "data": f"line-{i}"}) + "\n")
            f.flush()

        deadline = time.time() + 5.0
        while time.time() < deadline and len(collected) < 100:
            time.sleep(0.02)
    finally:
        tailer.stop()

    assert len(collected) == 100, f"only {len(collected)}/100 lines dispatched"
    # ~5 line-count-triggered saves (every 20 lines) + 1 forced save on stop()
    # — nowhere near a per-line write.
    assert offset_opens["count"] <= 10, (
        f"offset file opened {offset_opens['count']} times for 100 dispatched "
        f"lines — persistence must be batched, not per-line"
    )

    with open(tailer.offset_path, encoding="utf-8") as f:
        persisted = int(f.read().strip())
    assert persisted == os.path.getsize(tailer.log_path), (
        "stop() must force-persist the true final offset even if the last "
        "batch of lines hadn't hit the batching threshold"
    )
    assert not os.path.exists(tailer.offset_path + ".tmp")


def test_load_offset_resets_on_garbage_content(tmp_path):
    """A corrupted/partial offset file (e.g. from a crash mid-write) must
    reset to 0 instead of raising — never crash the backend on restart."""
    log_path = tmp_path / "job_log.jsonl"
    log_path.write_text("")
    offset_path = str(log_path) + ".offset"
    with open(offset_path, "wb") as f:
        f.write(b"12\x00")

    tailer = LogTailer(
        job_id="job",
        log_path=str(log_path),
        dispatcher=lambda _j, _e: None,
    )
    assert tailer._offset == 0


def test_save_offset_is_atomic_tmp_replace(tmp_path):
    """Persisting writes a sibling .tmp file then os.replace()s it onto the
    real offset path — no .tmp left lingering, real file holds the value."""
    tailer = _make_tailer(tmp_path)
    tailer._offset = 42
    tailer._save_offset(force=True)

    with open(tailer.offset_path, encoding="utf-8") as f:
        assert f.read().strip() == "42"
    assert not os.path.exists(tailer.offset_path + ".tmp")
