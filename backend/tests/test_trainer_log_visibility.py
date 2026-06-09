"""Trainer (side-process) log visibility.

Two gaps this covers:

1. A stopped/failed job must retain its log tail. The trainer's full log
   persists in ``job_log.jsonl`` on the volume, so we can reconstruct the tail
   from disk when the in-memory buffer is gone (job ended / backend restart).

2. Trainer messages must reach the MAIN server log so they show in the Server
   log viewer and its download — the trainer is a separate process whose
   messages otherwise live only in the job-scoped buffer.
"""
from app.engine.components.job_log_writer import JobLogWriter
from app.core.job_manager import (
    _parse_persisted_log_lines,
    _trainer_msg_for_server_log,
)


# ── Fix 1: persist the job tail ───────────────────────────────────────────

def test_parse_persisted_log_lines_reconstructs_tail(tmp_path):
    w = JobLogWriter(str(tmp_path))
    w.log("Loading Model")
    w.warning("low VRAM")
    w.exit(1, error="CUDA error: CUBLAS_STATUS_INVALID_VALUE")  # writes + closes

    lines = _parse_persisted_log_lines(str(tmp_path / "job_log.jsonl"))
    joined = "\n".join(lines)
    assert "Loading Model" in joined
    assert "low VRAM" in joined
    assert "CUBLAS_STATUS_INVALID_VALUE" in joined


def test_parse_persisted_log_lines_missing_file(tmp_path):
    assert _parse_persisted_log_lines(str(tmp_path / "nope.jsonl")) == []


def test_parse_persisted_log_lines_respects_limit(tmp_path):
    w = JobLogWriter(str(tmp_path))
    for i in range(50):
        w.log(f"line {i}")
    w.close()
    lines = _parse_persisted_log_lines(str(tmp_path / "job_log.jsonl"), limit=10)
    assert len(lines) == 10
    assert "line 49" in lines[-1]


def test_parse_persisted_log_lines_skips_metric_noise(tmp_path):
    """Per-step metrics aren't part of the textual tail (parity with the
    in-memory buffer, which only stores 'log' entries)."""
    w = JobLogWriter(str(tmp_path))
    w.step({"loss": 0.1})
    w.log("real line")
    w.close()
    lines = _parse_persisted_log_lines(str(tmp_path / "job_log.jsonl"))
    assert lines == ["real line"]


# ── Fix 2: mirror trainer messages into the server log ────────────────────

def test_mirror_log_warning_exit():
    lvl, text = _trainer_msg_for_server_log("log", "hello")
    assert lvl == "info" and "hello" in text

    lvl, text = _trainer_msg_for_server_log("warning", "careful")
    assert lvl == "warning" and "careful" in text

    lvl, text = _trainer_msg_for_server_log("exit", {"code": 1, "error": "boom"})
    assert lvl == "error" and "boom" in text


def test_mirror_skips_noise_and_clean_exit():
    assert _trainer_msg_for_server_log("step", {"loss": 1}) is None
    assert _trainer_msg_for_server_log("status", "Training") is None
    assert _trainer_msg_for_server_log("cache_ready", ["d"]) is None
    assert _trainer_msg_for_server_log("exit", {"code": 0}) is None
