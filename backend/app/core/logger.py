import asyncio
import logging
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import structlog


# Absolute path to backend/server.log, anchored to this file so it is independent
# of the process CWD (whether the backend was launched via start_backend.bat,
# uvicorn from the repo root, an IDE run config, or a restart subprocess).
SERVER_LOG_PATH = Path(__file__).resolve().parents[2] / "server.log"


_log_loop = None

def set_logging_loop(loop):
    global _log_loop
    _log_loop = loop


# ContextVar to prevent infinite recursion in logging
_in_ws_log = ContextVar("in_ws_log", default=False)

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
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"]:
        logging.getLogger(logger_name).setLevel(level)


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
    ``service`` identifier from ``docs/LOGGING.md`` universal_json_schema.

    The trainer subprocess writes its own JSONL stream via
    :class:`JobLogWriter` with a different envelope shape, and is
    responsible for setting ``service="lora-worker"`` there.
    """
    event_dict.setdefault("service", "fastapi-router")
    return event_dict


def setup_logging(log_level: str = "INFO", include_file_handler: bool = True):
    """
    Configures structlog for JSON output and standard logging integration.
    Resets server.log on startup to ensure a clean log for each session.
    """
    # Reset server.log on startup for clean analysis
    if include_file_handler:
        try:
            SERVER_LOG_PATH.unlink(missing_ok=True)
        except OSError:
            pass  # File may be locked by previous process


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
