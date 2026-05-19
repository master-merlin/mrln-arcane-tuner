"""MRLN Arcane Tuner — FastAPI application entry-point.

Configures logging, CORS, middleware, routers, static-file mounts,
and the application lifespan (startup / shutdown).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import traceback
import uuid
import webbrowser
import asyncio
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.core.logger import setup_logging, get_logger
from app.core.plugin_manager import plugin_manager
from app.core.settings_manager import get_settings_manager

# ── Settings & Logging ───────────────────────────────────────────────────

settings_manager = get_settings_manager()
_app_settings = settings_manager.get_module_settings("application")

LOG_LEVEL: str = _app_settings.get("log_level", "INFO")
BACKEND_PORT: int = _app_settings.get("backend_port", 8000)
FRONTEND_PORT: int = _app_settings.get("frontend_port", 4200)

setup_logging(LOG_LEVEL)
logger = get_logger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hook."""
    loop = asyncio.get_running_loop()

    # Inject event-loop into managers that schedule work from threads
    from app.core.dataset_manager import dataset_manager as dataset_manager_instance
    dataset_manager_instance.set_loop(loop)

    from app.core.job_manager import job_manager
    job_manager.set_loop(loop)
    job_manager.load_from_db()

    from app.core.logger import set_logging_loop
    set_logging_loop(loop)

    # ── Ensure core working directories exist ────────────────────────
    _backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for _dir in ("datasets", "models", os.path.join("models", "upscale"), "outputs"):
        os.makedirs(os.path.join(_backend_root, _dir), exist_ok=True)

    logger.info("starting_api", log_level=LOG_LEVEL)

    # ── Write runtime config for the frontend ────────────────────────
    from app.core.runtime_config import write_runtime_config
    write_runtime_config(BACKEND_PORT, FRONTEND_PORT)

    # Model registry & plugins
    from app.engine.models.registry import registry
    registry.initialize()
    plugin_manager.discover_plugins()

    # Recover jobs whose subprocess died during backend downtime
    job_manager.recover_jobs()

    # ── Auto-start frontend (first cold start only) ──────────────────
    _maybe_start_frontend()

    yield

    logger.info("shutting_down_api")


# ── Frontend Auto-Start ─────────────────────────────────────────────────

_frontend_process: subprocess.Popen | None = None


def _maybe_start_frontend() -> None:
    """Launch the Angular dev server and open browser on first start.

    Conditions:
    - ``start_frontend`` setting is enabled in application settings.
    - This is NOT a restart (``MRLN_RESTART`` env var absent).
    - ``npm`` is available on PATH.
    """
    global _frontend_process  # noqa: PLW0603

    is_restart = os.environ.get("MRLN_RESTART") == "1"
    start_frontend = _app_settings.get("start_frontend", False)

    if not start_frontend:
        logger.debug("frontend_autostart_disabled")
        return

    if is_restart:
        logger.info("frontend_autostart_skipped_restart")
        return

    # Locate npm
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    npm_path = shutil.which(npm_cmd)
    if not npm_path:
        logger.warning("frontend_autostart_npm_not_found",
                       hint="Install Node.js or disable start_frontend in settings.")
        return

    # Resolve the frontend directory (sibling of the backend dir)
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frontend_dir = os.path.join(os.path.dirname(backend_dir), "frontend")

    if not os.path.isdir(frontend_dir):
        logger.warning("frontend_autostart_dir_missing", path=frontend_dir)
        return

    try:
        logger.info("frontend_autostart_launching", cwd=frontend_dir)
        _frontend_process = subprocess.Popen(
            [npm_path, "run", "start"],
            cwd=frontend_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )

        # Open browser after a short delay to let the dev server boot
        def _open_browser() -> None:
            import time
            time.sleep(5)
            url = f"http://localhost:{FRONTEND_PORT}"
            logger.info("frontend_autostart_opening_browser", url=url)
            webbrowser.open(url)

        import threading
        threading.Thread(target=_open_browser, daemon=True).start()

    except OSError as e:
        logger.error("frontend_autostart_failed", error=str(e))


# ── Application ──────────────────────────────────────────────────────────

app = FastAPI(title="MRLN Arcane Tuner API", lifespan=lifespan)


# ── Global Exception Handler ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log unhandled exceptions as CRITICAL with full stack traces."""
    stack_trace = "".join(traceback.format_tb(exc.__traceback__))

    logger.critical(
        "application_crash",
        error=str(exc),
        error_type=type(exc).__name__,
        stack_trace=stack_trace,
        path=request.url.path,
        method=request.method,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "A critical error occurred. Please check server logs.",
        },
    )


# ── Middleware ────────────────────────────────────────────────────────────

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Inject per-request trace ID + span ID and log request lifecycle.

    Per docs/LOGGING.md universal_json_schema (R-LOG-07), every log
    entry must carry both ``trace_id`` (correlates across services —
    sourced from the ``X-Trace-ID`` request header when present) and
    ``span_id`` (unique per request inside this service).
    """
    trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
    span_id = str(uuid.uuid4())
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(trace_id=trace_id, span_id=span_id)

    logger.info("request_started", method=request.method, path=request.url.path)
    response = await call_next(request)
    logger.info("request_finished", status_code=response.status_code)

    response.headers["X-Trace-ID"] = trace_id
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"http://localhost:{FRONTEND_PORT}",
        f"http://127.0.0.1:{FRONTEND_PORT}",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ──────────────────────────────────────────────────────────────

from app.api.websocket import router as ws_router          # noqa: E402
from app.api.training import router as training_router      # noqa: E402
from app.api.filesystem_routes import router as fs_router   # noqa: E402
from app.api.dataset import router as dataset_router  # noqa: E402
from app.api.caption_routes import router as caption_router  # noqa: E402
from app.api.settings_routes import router as settings_router  # noqa: E402
from app.api.masking_routes import router as masking_router  # noqa: E402
from app.api.system_routes import router as system_router    # noqa: E402
from app.api.cache_routes import router as cache_router      # noqa: E402
from app.api.scoring_routes import router as scoring_router  # noqa: E402
from app.api.project_routes import router as project_router  # noqa: E402
from app.api.saved_concept_routes import router as saved_concept_router  # noqa: E402

app.include_router(ws_router, prefix="/api")
app.include_router(training_router, prefix="/api")
app.include_router(fs_router, prefix="/api")
app.include_router(dataset_router, prefix="/api")
app.include_router(caption_router, prefix="/api/captions")
app.include_router(settings_router)
app.include_router(masking_router, prefix="/api")
app.include_router(system_router, prefix="/api")
app.include_router(cache_router, prefix="/api")
app.include_router(scoring_router, prefix="/api")
app.include_router(project_router, prefix="/api")
app.include_router(saved_concept_router, prefix="/api")


# ── Static File Mounts ──────────────────────────────────────────────────

from app.core.dataset_manager import dataset_manager  # noqa: E402

os.makedirs(dataset_manager.default_root, exist_ok=True)
app.mount("/media", StaticFiles(directory=dataset_manager.default_root), name="media")


# ── Root Endpoint ────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Health-check / landing endpoint (frontend served via dev server)."""
    return {
        "message": f"MRLN Arcane Tuner API is running. Access frontend via port {FRONTEND_PORT}.",
        "version": __version__,
    }
