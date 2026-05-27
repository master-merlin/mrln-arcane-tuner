"""Tests for WSProgressTqdm + with_progress.

We don't actually exercise huggingface_hub here — instead we instantiate
WSProgressTqdm directly with known total/desc and verify the throttled
progress payloads it submits via `schedule_emit_from_thread`.
"""
from unittest.mock import patch
import pytest
from app.api.events.download_progress import (
    WSProgressTqdm, with_progress, DownloadProgress,
)


def _capture_emits():
    """Patch the thread-safe emitter so we capture payloads instead of
    scheduling them on a (nonexistent) loop."""
    captured: list[DownloadProgress] = []
    patcher = patch(
        "app.api.events.download_progress.schedule_emit_from_thread",
        side_effect=lambda p: captured.append(p),
    )
    return captured, patcher


def test_starting_emit_on_init():
    captured, p = _capture_emits()
    with p:
        WSProgressTqdm(total=100, source="hf", model_id="m1", category="caption")
    assert len(captured) == 1
    assert captured[0].status == "starting"
    assert captured[0].total_bytes == 100
    assert captured[0].percent == 0


def test_update_emits_downloading_with_throttle():
    captured, p = _capture_emits()
    with p:
        bar = WSProgressTqdm(total=100, source="hf", model_id="m1", category="caption")
        captured.clear()  # drop the 'starting' emit
        # Fast successive updates — only one should pass per throttle window
        bar.update(1); bar.update(1); bar.update(1)
        # Big jump should always emit
        bar.update(50)
    statuses = [e.status for e in captured]
    assert "downloading" in statuses
    assert any(e.percent and e.percent >= 50 for e in captured if e.status == "downloading")


def test_close_emits_complete():
    captured, p = _capture_emits()
    with p:
        bar = WSProgressTqdm(total=100, source="hf", model_id="m1", category="caption")
        bar.update(100)
        bar.close()
    assert any(e.status == "complete" and e.percent == 100 for e in captured)


def test_unknown_total_emits_none_percent():
    captured, p = _capture_emits()
    with p:
        bar = WSProgressTqdm(total=None, source="hf", model_id="m1", category="caption")
        bar.update(50)
        bar.close()
    starts = [e for e in captured if e.status == "starting"]
    assert starts[0].percent is None
    assert starts[0].total_bytes is None
    completes = [e for e in captured if e.status == "complete"]
    assert completes[0].percent is None


def test_with_progress_emits_starting_and_complete_on_clean_exit():
    captured, p = _capture_emits()
    with p:
        with with_progress(model_id="facebook/sam3", category="mask"):
            pass
    statuses = [e.status for e in captured]
    assert statuses == ["starting", "complete"]


def test_with_progress_emits_error_on_exception():
    captured, p = _capture_emits()
    with p:
        with pytest.raises(RuntimeError, match="boom"):
            with with_progress(model_id="facebook/sam3", category="mask"):
                raise RuntimeError("boom")
    statuses = [e.status for e in captured]
    assert statuses[0] == "starting"
    assert statuses[-1] == "error"
    assert any("boom" in (e.error or "") for e in captured)
