"""The metrics stream reports faults once, and never feeds the log->WS loop.

``_stream_metrics`` ended in ``except Exception: pass``. A silent swallow
(ARCHITECTURE D10 invariant 4) made "the client left" and "snapshot() is
broken" indistinguishable from outside the process.

Un-silencing it is the delicate part, and this file exists because of that
rather than because of the swallow: a log record emitted from a WebSocket
handler is mirrored to every connected client, each mirror writes a frame,
each frame logs a trace on ``uvicorn.error``, and each trace is mirrored again.
That loop is a RECORDED lesson — 114 MB / 1.19M lines with one client. So the
tests assert both halves: exactly one record, and emitted inside
``ws_send_scope`` so the mirror skips it.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from app.api.websocket import _stream_metrics


class FakeWebSocket:
    def __init__(self, fail_on_send: BaseException | None = None):
        self.client = "testclient:12345"
        self.sent: list[str] = []
        self._fail_on_send = fail_on_send

    async def send_text(self, msg: str) -> None:
        if self._fail_on_send is not None:
            raise self._fail_on_send
        self.sent.append(msg)


def _records(caplog, level: int) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == level]


@pytest.mark.asyncio
async def test_a_broken_snapshot_logs_exactly_one_warning(caplog, monkeypatch):
    """The count is the assertion.

    A handler that logs per iteration would still 'log the error', and would
    also rebuild the flood this codebase already paid for once.
    """
    from app.core import system_monitor as sm

    def _boom():
        raise ValueError("nvml exploded")

    monkeypatch.setattr(sm.system_monitor, "snapshot", _boom)

    with caplog.at_level(logging.DEBUG):
        await _stream_metrics(FakeWebSocket(), interval_s=0.01)

    warnings = [r for r in _records(caplog, logging.WARNING)
                if "ws_metrics_stream_failed" in r.getMessage()]
    assert len(warnings) == 1, (
        f"expected exactly one WARNING, got {len(warnings)}: "
        f"{[r.getMessage() for r in warnings]}"
    )
    text = warnings[0].getMessage()
    assert "nvml exploded" in text, "the cause must be in the record"
    assert "testclient" in text, "the connection must be in the record"


@pytest.mark.asyncio
async def test_the_failure_record_is_emitted_inside_ws_send_scope(monkeypatch):
    """The half that stops the loop reopening.

    Asserted on the ContextVar the mirror actually consults, not on the
    presence of a ``with`` block in the source — the question is whether the
    handler would skip this record, and that is what the flag decides.
    """
    from app.core import logger as logger_mod
    from app.core import system_monitor as sm

    seen: list[bool] = []
    original = logger_mod.get_logger("app.api.websocket").warning

    def _spy(*args, **kwargs):
        seen.append(logger_mod._in_ws_log.get())
        return original(*args, **kwargs)

    monkeypatch.setattr(sm.system_monitor, "snapshot", lambda: 1 / 0)

    import app.api.websocket as ws_mod

    monkeypatch.setattr(ws_mod.logger, "warning", _spy)
    await _stream_metrics(FakeWebSocket(), interval_s=0.01)

    assert seen == [True], (
        "the failure WARNING was emitted OUTSIDE ws_send_scope; it will be "
        "mirrored to every client and can reopen the log->WS->log loop"
    )


@pytest.mark.asyncio
async def test_cancellation_propagates_and_is_not_logged_as_a_fault(caplog):
    """`unsubscribe_metrics` cancels this task; that is a normal exit.

    Swallowing CancelledError leaves the canceller awaiting a task that never
    reports finished, and logging it as a fault trains people to ignore the
    warning that matters.
    """
    ws = FakeWebSocket()

    with caplog.at_level(logging.DEBUG):
        task = asyncio.create_task(_stream_metrics(ws, interval_s=5.0))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert not [r for r in _records(caplog, logging.WARNING)
                if "ws_metrics_stream_failed" in r.getMessage()], (
        "cancellation was reported as a fault"
    )


@pytest.mark.asyncio
async def test_a_disconnected_client_is_debug_not_warning(caplog):
    """A client leaving is ordinary; only real faults deserve WARNING."""
    ws = FakeWebSocket(fail_on_send=ConnectionError("client went away"))

    with caplog.at_level(logging.DEBUG):
        await _stream_metrics(ws, interval_s=0.01)

    assert not [r for r in _records(caplog, logging.WARNING)
                if "ws_metrics_stream_failed" in r.getMessage()]
    assert [r for r in _records(caplog, logging.DEBUG)
            if "ws_metrics_stream_closed" in r.getMessage()], (
        "the disconnect is still silent — it should be visible at DEBUG"
    )


@pytest.mark.asyncio
async def test_the_happy_path_still_streams(monkeypatch):
    """Prove the negative: error handling did not break the feature."""
    from app.core import system_monitor as sm

    class _Snap:
        def to_dict(self):
            return {"gpu": []}

    monkeypatch.setattr(sm.system_monitor, "snapshot", lambda: _Snap())

    ws = FakeWebSocket()
    task = asyncio.create_task(_stream_metrics(ws, interval_s=0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert ws.sent, "no metrics were streamed"
    assert "system_metrics" in ws.sent[0]
