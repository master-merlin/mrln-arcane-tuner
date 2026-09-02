import asyncio
import logging
import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import structlog


# Absolute path to backend/server.log, anchored to this file so it is independent
# of the process CWD (whether the backend was launched via start_backend.bat,
# uvicorn from the repo root, an IDE run config, or a restart subprocess).
SERVER_LOG_PATH = Path(__file__).resolve().parents[2] / "server.log"

# The PREVIOUS session's log. `setup_logging` MOVES `server.log` here instead of
# deleting it, and that difference is the whole point (LANE-56, measured
# 2026-09-01): the boot whose log everyone needs is the one that FAILED, and the
# next boot is the user's recovery — so the old code had the recovery destroy
# the evidence of the failure it was recovering from, every single time. One
# generation is kept, not more: the property `server.log` must keep is "this
# session only" (`/api/system/logs` and the Server screen read it and would
# otherwise show a foreign boot's lines as if they were this one's), and a
# bounded history keeps that while making the previous boot answerable.
PREVIOUS_SERVER_LOG_PATH = Path(__file__).resolve().parents[2] / "server.prev.log"


_log_loop = None

def set_logging_loop(loop):
    global _log_loop
    _log_loop = loop


# ContextVar to prevent infinite recursion in logging
_in_ws_log = ContextVar("in_ws_log", default=False)


@contextmanager
def ws_send_scope():
    """Mark the current context as "pushing bytes into a WebSocket".

    Every log record emitted inside this scope is skipped by
    :class:`WebSocketLogHandler` instead of being mirrored to clients.

    This is load-bearing, not belt-and-braces. uvicorn hands its own
    ``uvicorn.error`` logger to the ``websockets`` protocol, which logs one
    ``> TEXT '…' [N bytes]`` DEBUG record for every frame it writes. Mirroring
    such a record emits another frame, which logs another trace, which mirrors
    again — an unbounded log->WS->log loop that fills server.log at event-loop
    speed (observed: 114 MB / 1.19M lines, 100% frame traces, one client).

    The loop used to be impossible by accident: ``broadcast()`` awaited
    ``send_text`` inline, and ``run_coroutine_threadsafe`` propagates the
    context that :meth:`WebSocketLogHandler.emit` had already marked. W4.T9
    moved the send into a per-connection sender task created back in
    ``connect()`` — outside that context — so the guard stopped covering the
    only path that can feed itself. Any code that writes to a socket must wrap
    the write in this scope.
    """
    token = _in_ws_log.set(True)
    try:
        yield
    finally:
        _in_ws_log.reset(token)


class WebSocketLogHandler(logging.Handler):
    def emit(self, record):
        # Recursion Guard: If we are already logging inside this handler (on this context), stop.
        if _in_ws_log.get():
            return

        try:
            # Set context var to True for this block
            token = _in_ws_log.set(True)
            
            # Import here to avoid circular dependency
            from app.core.events import event_manager
            
            # Additional safety: Filter out 'websockets' library logs entirely 
            # as they are noisy and can trigger loops if connection is unstable
            if record.name.startswith("websockets"):
                return

            # Suppress noisy log entries from being broadcast to frontend
            msg = self.format(record)
            if any(skip in msg for skip in (
                '"system_metrics"', '/api/ws', '/media/',
                'request_started', 'request_finished',
                'ping', 'pong',
            )):
                return
            
            if _log_loop and not _log_loop.is_closed():
                 asyncio.run_coroutine_threadsafe(
                    event_manager.broadcast("server_log", {
                        "message": msg,
                        "level": record.levelname,
                        "timestamp": record.created
                    }),
                    _log_loop
                )
        except Exception:
            self.handleError(record)
        finally:
            # Reset context var
            if 'token' in locals():
                _in_ws_log.reset(token)

def config_log_level(level_str: str = "INFO"):
    """
    Updates the log level for both the root logger and structlog.
    """
    level_str = level_str.upper()
    level = getattr(logging, level_str, logging.INFO)

    # 1. Standard Logging
    logging.getLogger().setLevel(level)
    
    # 2. Update specific loggers to ensure they respect the new level
    # especially uvicorn which can be chatty
    for logger_name in ["uvicorn", "uvicorn.access", "fastapi"]:
        logging.getLogger(logger_name).setLevel(level)

    # 3. uvicorn.error is floored at INFO — never DEBUG.
    # uvicorn passes logging.getLogger("uvicorn.error") to the websockets
    # protocol as its logger (uvicorn/protocols/websockets/websockets_impl.py),
    # so this logger's DEBUG stream is one per-frame "> TEXT '…' [N bytes]"
    # trace per WebSocket frame sent or received. That is not an application
    # log: it is unparseable as JSON (_docs/LOGGING.md golden rule 1) and it
    # rides on a logger name that none of the "websockets"-prefixed
    # suppressions below can match. Threshold only — uvicorn's real INFO
    # lifecycle lines, warnings and errors all still surface.
    logging.getLogger("uvicorn.error").setLevel(max(level, logging.INFO))


# Third-party loggers that flood the logs during model downloads. Pinned to a
# WARNING threshold so their INFO/DEBUG (filelock lock acquire/release, urllib3
# connection-pool debug, hf_xet, websockets per-frame "> TEXT" traces) is
# dropped while genuine warnings/errors still surface. Threshold only — no
# message is relabeled.
_NOISY_DOWNLOAD_LOGGERS = ("filelock", "urllib3", "hf_xet", "websockets")


def _quiet_noisy_loggers() -> None:
    """Raise the threshold of known-noisy download loggers to WARNING."""
    for name in _NOISY_DOWNLOAD_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _add_service(_logger, _name, event_dict):
    """R-LOG-07: stamp every FastAPI-side log entry with the canonical
    ``service`` identifier from ``_docs/LOGGING.md`` universal_json_schema.

    The trainer subprocess writes its own JSONL stream via
    :class:`JobLogWriter` with a different envelope shape, and is
    responsible for setting ``service="lora-worker"`` there.
    """
    event_dict.setdefault("service", "fastapi-router")
    return event_dict


def setup_logging(log_level: str = "INFO", include_file_handler: bool = True):
    """
    Configures structlog for JSON output and standard logging integration.
    Starts a fresh server.log for each session, keeping the previous one.
    """
    # ROTATE, never delete (LANE-56). The property this has to keep is "one
    # session per server.log" — the file handler below opens in append mode, so
    # without this step every boot would pile onto the last and the Server
    # screen would show a foreign session's lines. The property it must STOP
    # breaking is that the deleted session is always the interesting one: a
    # replacement server that failed to start logs here, and the very next
    # start is the user's recovery, which used to unlink it.
    if include_file_handler:
        try:
            if SERVER_LOG_PATH.exists():
                os.replace(SERVER_LOG_PATH, PREVIOUS_SERVER_LOG_PATH)
        except OSError:
            # Windows: a previous process that has not fully exited still holds
            # the handle, and then unlink cannot succeed either. Best effort,
            # both ways — a start may never fail because of its own logging.
            try:
                SERVER_LOG_PATH.unlink(missing_ok=True)
            except OSError:
                pass


    # Shared processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _add_service,
        structlog.processors.JSONRenderer()
    ]

    structlog.configure(
        processors=processors,
        # Use StandardLoggerFactory to respect standard logging levels easily
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Use a wrapper class that calls the standard logger
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure Standard Logging
    # We want everything to go to stdout as JSON (12-factor app)
    # File logging is okay but usually docker/systemd handles capture.
    # The original requirement has a file handler. We will keep it but ALSO ensure stdout.
    
    root_logger = logging.getLogger()
    root_logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)

    # 2. File Handler (Legacy requirement persistence)
    # Only attach if requested (Main Process), Worker processes should NOT write to file directly
    if include_file_handler:
        file_handler = logging.FileHandler(SERVER_LOG_PATH, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(file_handler)

    # 3. WebSocket Handler (Global Broadcast)
    ws_handler = WebSocketLogHandler()
    ws_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(ws_handler)

    class EndpointFilter(logging.Filter):
        """Suppress noisy polling, WS keepalive, and media-serving log entries."""
        _SKIP_FRAGMENTS = (
            '/api/system/logs',
            '/api/jobs',
            '/api/ws',
            '/media/',
            '"system_metrics"',
        )
        _SKIP_LOWER = ('ping', 'pong', 'keepalive')

        def filter(self, record: logging.LogRecord) -> bool:
            # Suppress all websockets library frame/keepalive traces
            if record.name.startswith("websockets"):
                return False
            message = record.getMessage()
            if any(frag in message for frag in self._SKIP_FRAGMENTS):
                return False
            msg_lower = message.lower()
            if any(kw in msg_lower for kw in self._SKIP_LOWER):
                return False
            return True

    endpoint_filter = EndpointFilter()
    for h in root_logger.handlers:
        h.addFilter(endpoint_filter)

    # Apply Level
    config_log_level(log_level)

    # Silence chatty download libraries (threshold only — real errors survive).
    _quiet_noisy_loggers()

    # Uvicorn Loggers
    # We want to capture uvicorn logs and format them if possible, but Uvicorn has its own formatters.
    # For now, ensure they propagate to our root logger (which has the handlers) 
    # and disable their default handlers to avoid double logging
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        u_log = logging.getLogger(logger_name)
        u_log.handlers = [] # Remove default handlers
        u_log.propagate = True # Send to root

def get_logger(name: str = None) -> Any:
    """
    Returns a structlog logger.
    """
    return structlog.get_logger(name)
