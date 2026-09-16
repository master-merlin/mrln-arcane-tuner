"""
Tests for the logging module.
Covers: setup_logging, config_log_level, WebSocketLogHandler recursion guard,
get_logger, and EndpointFilter.
"""

import logging
from unittest.mock import patch, MagicMock

import pytest

from app.core.logger import (
    setup_logging,
    config_log_level,
    get_logger,
    WebSocketLogHandler,
    _in_ws_log,
)


# ── Helpers ──────────────────────────────────────────────────────────────

_RESTORED_LOGGERS = ["uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"]


@pytest.fixture(autouse=True)
def _clean_logging():
    """Put the loggers back the way this test FOUND them.

    The old version wiped ``root.handlers`` after each test. That also
    discarded the session's ``tests.log`` FileHandler installed by
    ``tests/conftest.py::_isolate_test_logging``, so from this module on
    nothing in the run reached ``tests.log`` (measured 2026-09-02: the log
    stopped at 02:01 in a 73-minute run) and the LANE-63 isolation test went
    red whenever it ran after this file. Snapshot and restore is what "avoid
    cross-contamination" actually requires.
    """
    root = logging.getLogger()
    before = (list(root.handlers), root.level)
    named = {n: (list(logging.getLogger(n).handlers), logging.getLogger(n).propagate)
             for n in _RESTORED_LOGGERS}
    yield
    root.handlers, level = list(before[0]), before[1]
    root.setLevel(level)
    for name, (handlers, propagate) in named.items():
        lg = logging.getLogger(name)
        lg.handlers = list(handlers)
        lg.propagate = propagate


# ── setup_logging ────────────────────────────────────────────────────────


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_adds_console_handler(self):
        """Console StreamHandler should always be present."""
        setup_logging(log_level="INFO", include_file_handler=False)
        root = logging.getLogger()
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, (logging.FileHandler, WebSocketLogHandler))]
        assert len(stream_handlers) >= 1

    def test_adds_websocket_handler(self):
        """WebSocketLogHandler should always be attached."""
        setup_logging(log_level="WARNING", include_file_handler=False)
        root = logging.getLogger()
        ws_handlers = [h for h in root.handlers if isinstance(h, WebSocketLogHandler)]
        assert len(ws_handlers) == 1

    def test_file_handler_skipped_when_disabled(self):
        """No FileHandler when include_file_handler=False."""
        setup_logging(log_level="INFO", include_file_handler=False)
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 0

    def test_uvicorn_propagation_enabled(self):
        """Uvicorn loggers should propagate and have no own handlers."""
        setup_logging(log_level="INFO", include_file_handler=False)
        for name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
            uvlog = logging.getLogger(name)
            assert uvlog.propagate is True
            assert uvlog.handlers == []

    def test_clears_previous_handlers(self):
        """Calling setup_logging twice should not duplicate handlers."""
        setup_logging(log_level="INFO", include_file_handler=False)
        first_count = len(logging.getLogger().handlers)
        setup_logging(log_level="INFO", include_file_handler=False)
        second_count = len(logging.getLogger().handlers)
        assert first_count == second_count


# ── config_log_level ─────────────────────────────────────────────────────


class TestConfigLogLevel:
    """Tests for dynamic log-level switching."""

    def test_sets_root_level(self):
        """Root logger level should match the requested string."""
        config_log_level("DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_sets_uvicorn_level(self):
        """Uvicorn loggers should also adopt the new level."""
        config_log_level("WARNING")
        assert logging.getLogger("uvicorn").level == logging.WARNING

    def test_case_insensitive(self):
        """Level string should be normalised to upper-case."""
        config_log_level("error")
        assert logging.getLogger().level == logging.ERROR

    def test_invalid_level_falls_back_to_info(self):
        """An unknown level string should fall back to INFO."""
        config_log_level("NONEXISTENT")
        assert logging.getLogger().level == logging.INFO


# ── WebSocketLogHandler ──────────────────────────────────────────────────


class TestWebSocketLogHandler:
    """Tests for the WebSocket broadcast handler and its recursion guard."""

    def test_recursion_guard_prevents_reentry(self):
        """If _in_ws_log is already True, emit() must be a no-op."""
        handler = WebSocketLogHandler()
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0, msg="hello",
            args=None, exc_info=None,
        )
        token = _in_ws_log.set(True)
        try:
            with patch("app.core.events.event_manager") as mock_em:
                handler.emit(record)
                mock_em.broadcast.assert_not_called()
        finally:
            _in_ws_log.reset(token)

    def test_filters_websockets_library_logs(self):
        """Records from 'websockets.*' should be silently dropped."""
        handler = WebSocketLogHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        record = logging.LogRecord(
            name="websockets.server", level=logging.DEBUG,
            pathname="", lineno=0, msg="noisy ws debug",
            args=None, exc_info=None,
        )
        with patch("app.core.events.event_manager") as mock_em:
            handler.emit(record)
            mock_em.broadcast.assert_not_called()

    def test_broadcasts_when_loop_available(self):
        """When _log_loop is set and open, emit should schedule a broadcast."""
        # Reset ContextVar — source code has a bug where _in_ws_log is never
        # reset after early-return paths in emit(), which can leak across tests.
        token = _in_ws_log.set(False)

        handler = WebSocketLogHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        record = logging.LogRecord(
            name="test.module", level=logging.INFO,
            pathname="", lineno=0, msg="training complete",
            args=None, exc_info=None,
        )
        mock_loop = MagicMock()
        mock_loop.is_closed.return_value = False

        try:
            with patch("app.core.events.event_manager"), \
                 patch("app.core.logger._log_loop", mock_loop), \
                 patch("app.core.logger.asyncio.run_coroutine_threadsafe") as mock_rcts:
                handler.emit(record)
                # Should have checked loop status and scheduled broadcast
                mock_loop.is_closed.assert_called()
                mock_rcts.assert_called_once()
        finally:
            _in_ws_log.reset(token)


# ── get_logger ───────────────────────────────────────────────────────────


class TestGetLogger:
    """Tests for the get_logger utility."""

    def test_returns_bound_logger(self):
        """get_logger should return a structlog BoundLogger (or proxy)."""
        lg = get_logger("test_module")
        assert lg is not None

    def test_different_names_give_different_loggers(self):
        """Loggers with different names should not be identical objects."""
        a = get_logger("alpha")
        b = get_logger("beta")
        # They're both proxies but bound to different names
        assert a is not b


# ── EndpointFilter ───────────────────────────────────────────────────────


class TestEndpointFilter:
    """Tests for the noisy-endpoint filter added in setup_logging."""

    def test_filters_system_logs_endpoint(self):
        """Log messages mentioning /api/system/logs should be filtered."""
        setup_logging(log_level="INFO", include_file_handler=False)
        root = logging.getLogger()
        # Grab any filter from any handler
        filters = []
        for h in root.handlers:
            filters.extend(h.filters)

        record = logging.LogRecord(
            name="uvicorn.access", level=logging.INFO,
            pathname="", lineno=0,
            msg='GET /api/system/logs HTTP/1.1 200',
            args=None, exc_info=None,
        )
        # At least one filter should reject this
        assert any(not f.filter(record) for f in filters)

    def test_passes_normal_endpoints(self):
        """Normal endpoints should pass the filter."""
        setup_logging(log_level="INFO", include_file_handler=False)
        root = logging.getLogger()
        filters = []
        for h in root.handlers:
            filters.extend(h.filters)

        record = logging.LogRecord(
            name="uvicorn.access", level=logging.INFO,
            pathname="", lineno=0,
            msg='POST /api/datasets/create HTTP/1.1 200',
            args=None, exc_info=None,
        )
        # All filters should pass this
        assert all(f.filter(record) for f in filters)


# ── the cleanup restores, it does not wipe (LANE-63) ─────────────────────


def test_the_session_file_handler_survives_this_module():
    """Runs LAST in this file. Every test above called setup_logging and was
    cleaned up by ``_clean_logging``; the session's tests.log FileHandler
    (``tests/conftest.py::_isolate_test_logging``) must still be on the root
    logger, or the rest of the run logs to nowhere. RED with the old
    ``root.handlers = []`` cleanup."""
    files = [h.baseFilename for h in logging.getLogger().handlers
             if isinstance(h, logging.FileHandler)]
    assert any(f.endswith(".log") and "tests" in f for f in files), files
