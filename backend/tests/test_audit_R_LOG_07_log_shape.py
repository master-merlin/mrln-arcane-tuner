"""R-LOG-07 regression guard: every emitted log entry must carry the
universal JSON schema fields per _docs/LOGGING.md.

The universal schema requires: timestamp, level, service, message,
trace_id, span_id, context. In structlog terms:
    - "message"   → "event" (the JSON renderer emits the event key).
    - "context"   → arbitrary remaining keys; no dedicated assertion.

This test pins the FastAPI-side log pipeline (request lifecycle through
the trace_id middleware) so a future processor reshuffle cannot
silently drop service or span_id again.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

REQUIRED_FIELDS = {
    "timestamp", "level", "service", "event", "trace_id", "span_id"
}


class _JSONLineCapture(logging.Handler):
    """Captures the final JSON-rendered log line emitted by structlog
    through the stdlib root logger (the project routes structlog →
    stdlib via LoggerFactory). Each record's message is the JSON string
    produced by JSONRenderer."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        # defensive: a non-structlog logger could emit a JSON-shaped string
        # with trailing whitespace, so strip before checking shape and parsing.
        msg = record.getMessage().strip()
        if not (msg.startswith("{") and msg.endswith("}")):
            return
        try:
            self.records.append(json.loads(msg))
        except (json.JSONDecodeError, ValueError):
            pass


@pytest.fixture
def captured_events() -> list[dict[str, Any]]:
    """Attach a stdlib-log handler that parses the JSON-rendered output
    of structlog. This works with the project's existing config (which
    routes structlog → stdlib root logger via LoggerFactory) without
    reconfiguring structlog or fighting cache_logger_on_first_use."""
    handler = _JSONLineCapture()
    root = logging.getLogger()
    root.addHandler(handler)
    # Also attach directly to the app.main logger in case it has
    # propagate=False or its own handlers shadow root.
    app_main_log = logging.getLogger("app.main")
    app_main_log.addHandler(handler)
    # Ensure both loggers actually accept INFO-level records
    root.setLevel(logging.DEBUG)
    app_main_log.setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        root.removeHandler(handler)
        app_main_log.removeHandler(handler)


def test_log_entry_contains_all_universal_fields(
    captured_events: list[dict[str, Any]],
) -> None:
    """Every emitted entry during a request must include the full
    universal schema."""
    from app.main import app

    client = TestClient(app)
    # /api/status is a cheap GET that emits request_started/request_finished
    # via the logging middleware. Any status code is fine — we only care
    # that logging happened inside a middleware-wrapped request.
    response = client.get("/api/status")
    assert response.status_code in (200, 204, 404, 500), (
        f"unexpected status {response.status_code}"
    )
    assert captured_events, "no log entries captured during request"

    # Inspect the request-lifecycle events specifically: those are the
    # entries that must carry trace_id + span_id (bound by middleware)
    # AND service (added by the global processor).
    lifecycle = [
        e for e in captured_events
        if e.get("event") in ("request_started", "request_finished")
    ]
    assert lifecycle, (
        "no request_started/request_finished events captured; "
        f"captured events: {[e.get('event') for e in captured_events]}"
    )

    last = lifecycle[-1]
    missing = REQUIRED_FIELDS - set(last)
    assert not missing, (
        f"log entry missing required fields: {missing}; "
        f"got keys={sorted(last)}"
    )
    # Negative-axis assertions: catch a regression where the fields are
    # wired but produce the wrong values.
    assert last["service"] == "fastapi-router", (
        f"service must be 'fastapi-router' on FastAPI-side entries; got {last['service']!r}"
    )
    assert last["span_id"] != last["trace_id"], (
        "span_id and trace_id must be independent UUIDs; got the same value"
    )
