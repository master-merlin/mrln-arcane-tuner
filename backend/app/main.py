"""MRLN Arcane Tuner — FastAPI application entry-point.

Configures logging, CORS, middleware, routers, static-file mounts,
and the application lifespan (startup / shutdown).
"""

from __future__ import annotations

import asyncio
import hmac
import os
import shutil
import subprocess
import traceback
import uuid
import webbrowser
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Form, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.api.schemas.common_schemas import ErrorResponse
from app.core import container_config
from app.core.auth import COOKIE_NAME, LOGIN_HTML, TokenAuthMiddleware
from app.core.logger import setup_logging, get_logger
from app.core.plugin_manager import plugin_manager
from app.core.settings_manager import get_settings_manager

# ── Settings & Logging ───────────────────────────────────────────────────

settings_manager = get_settings_manager()
_app_settings = settings_manager.get_module_settings("application")

LOG_LEVEL: str = _app_settings.get("log_level", "INFO")
BACKEND_PORT: int = container_config.resolve_port(
    _app_settings.get("backend_port", 8000)
)
FRONTEND_PORT: int = _app_settings.get("frontend_port", 4200)

setup_logging(LOG_LEVEL)
logger = get_logger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hook."""
    # FIRST, before anything else starts: refuse to run an unauthenticated
    # server on an address other machines can reach. This is in lifespan, not at
    # import, because nothing imported at startup may raise (ARCHITECTURE D1) —
    # and it is at the TOP of lifespan so the refusal happens before any
    # background task, DB load or socket is set up.
    #
    # BREAKING CHANGE (DECISION-3 (a)): earlier releases documented
    # MRLN_AUTH_TOKEN as "unset = open" and started on 0.0.0.0 regardless. A
    # token-less container that relied on that will now stop with the message
    # below, which names both fixes.
    _refusal = container_config.bind_is_exposed_without_auth()
    if _refusal:
        logger.error("refusing_exposed_bind_without_auth")
        raise RuntimeError(_refusal)

    loop = asyncio.get_running_loop()

    # Inject event-loop into managers that schedule work from threads
    from app.core.dataset_manager import dataset_manager as dataset_manager_instance
    dataset_manager_instance.set_loop(loop)

    from app.core.job_manager import job_manager
    job_manager.set_loop(loop)
    job_manager.load_from_db()

    from app.core.tasks import task_manager
    task_manager.set_loop(loop)

    from app.core.self_update import self_update_service
    self_update_service.set_loop(loop)
    if self_update_service.remote:
        await asyncio.to_thread(self_update_service.probe_availability)
    _update_task = asyncio.create_task(self_update_service.run_periodic_check())

    # Warm the cross-dataset cache-stats aggregation in the background so the
    # datasets KPI is ready by the time the user navigates (silent — hidden from
    # the Task Center). Non-GPU 'background' lane so it never blocks GPU work.
    from app.api.cache_routes import run_cache_stats_refresh
    _warm = task_manager.create(type="cache_stats_warmup", title="Cache stats",
                                user_visible=False)
    task_manager.enqueue(_warm.id, run_cache_stats_refresh, lane="background")

    from app.core.logger import set_logging_loop
    set_logging_loop(loop)

    from app.api.events.download_progress import set_app_loop
    set_app_loop(loop)

    # ── Ensure core working directories exist ────────────────────────
    _backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for _dir in ("datasets", "models", os.path.join("models", "upscale"), "outputs"):
        os.makedirs(os.path.join(_backend_root, _dir), exist_ok=True)

    logger.info("starting_api", log_level=LOG_LEVEL)

    # No runtime-config.json write here. The file is still SHIPPED and still
    # fetched by the SPA at bootstrap — what was retired is the backend
    # rewriting it at runtime, into `frontend/public/` (the SOURCE checkout),
    # which the served build never reads: the container serves
    # `/app/frontend/browser`. Pointless in dev, and a write into a read-only
    # or ephemeral checkout in a container. The keys it wrote are deprecated as
    # URL inputs and influence nothing. See test_runtime_config_writer_retired.

    # Apply Hugging Face auth (env token wins, else the saved Server setting)
    # before the registry initialises — model metadata fetches may need it.
    from app.core.hf_auth import apply_hf_auth
    from app.engine.utils.model_override_manager import ModelOverrideManager
    apply_hf_auth(ModelOverrideManager.get_all().hf_token)

    # Model registry & plugins
    from app.engine.models.registry import registry
    registry.initialize()
    plugin_manager.discover_plugins()

    # Recover jobs whose subprocess died during backend downtime
    job_manager.recover_jobs()

    # If auto-queue is on and the GPU is idle with pending jobs (e.g. a run
    # finished while the backend was down), start draining now — unattended
    # queue progress must not wait for a browser to open the Jobs tab.
    job_manager.schedule_advance_queue()

    # ── Auto-start frontend (first cold start only) ──────────────────
    _maybe_start_frontend()

    yield

    _update_task.cancel()
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
    if container_config.is_container():
        logger.info("frontend_autostart_skipped_container")
        return

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


from app.core.drain import DrainActive  # noqa: E402


@app.exception_handler(DrainActive)
async def _drain_active_handler(request, exc):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


# ── Exception Handlers ───────────────────────────────────────────────────
# All error responses share the standard envelope (docs/API_CONVENTIONS.md):
#   {"detail": <message>, "error_code": <token>, "context": {...}}
# `detail` is preserved verbatim from the raised HTTPException so consumers
# that read structured detail payloads (e.g. the import-dataset 409 conflict
# body) keep working; error_code/context are additive.

# HTTP status → stable machine-readable error code. Unmapped statuses fall back
# to ``HTTP_<status>`` so the field is always populated.
_HTTP_ERROR_CODES: dict[int, str] = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    410: "GONE",
    413: "PAYLOAD_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "UNPROCESSABLE_ENTITY",
    429: "TOO_MANY_REQUESTS",
}


def _error_code_for(status_code: int) -> str:
    return _HTTP_ERROR_CODES.get(status_code, f"HTTP_{status_code}")


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Wrap every ``HTTPException`` (404/409/400/…) in the standard envelope."""
    envelope = ErrorResponse(
        detail=exc.detail,
        error_code=_error_code_for(exc.status_code),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=envelope.model_dump(),
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Map Pydantic/FastAPI request-validation failures into the envelope.

    The per-field error list stays in ``detail`` (matching FastAPI's default
    shape so existing clients still read ``detail``), with a stable
    ``VALIDATION_ERROR`` code.
    """
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            detail=jsonable_encoder(exc.errors()),
            error_code="VALIDATION_ERROR",
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log unhandled exceptions as CRITICAL; never leak tracebacks to clients."""
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
        content=ErrorResponse(
            detail="A critical error occurred. Please check server logs.",
            error_code="INTERNAL_ERROR",
        ).model_dump(),
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


# ── Auth Gate (no-op when no token configured) ───────────────────────────

app.add_middleware(TokenAuthMiddleware, token=container_config.auth_token())


# ── Security headers (CSP) ───────────────────────────────────────────────
# Added LAST on purpose. Starlette's ``add_middleware`` prepends, so the last
# one added is the OUTERMOST — which is what puts the policy on every response,
# including the 401 the auth gate short-circuits with. An error page is still a
# page, and a response that skips the policy is exactly the one an injection
# wants to land in. Registering this before the auth gate would have left that
# 401 uncovered; ``test_csp_is_on_the_auth_gate_401`` pins the ordering rather
# than trusting the reasoning.
from app.api._security_headers import SecurityHeadersMiddleware  # noqa: E402

app.add_middleware(SecurityHeadersMiddleware)


from fastapi.responses import HTMLResponse, RedirectResponse  # noqa: E402


@app.get("/login")
async def login_page():
    """Render the sign-in page (token is submitted via POST)."""
    return HTMLResponse(LOGIN_HTML)


@app.post("/login")
async def login_submit(token: str = Form("")):
    """Validate the access token, set the auth cookie, then redirect home.

    No-op pass-through when auth is disabled (no token configured). The token
    is read from the POST body (never a query string) so it cannot leak into
    access logs, the log stream, browser history, or proxy logs.
    """
    configured = container_config.auth_token()
    if not configured or hmac.compare_digest(token, configured):
        resp = RedirectResponse("/", status_code=302)
        if configured:
            # Secure flag follows deployment intent: in the container the app
            # is always reached over HTTPS at the proxy edge (the backend
            # itself sees plain http), so mark Secure there; keep it off for
            # local-dev http so the cookie is still accepted.
            resp.set_cookie(
                COOKIE_NAME,
                token,
                httponly=True,
                samesite="lax",
                secure=container_config.is_container(),
            )
        return resp
    return HTMLResponse(LOGIN_HTML, status_code=401)


# ── Routers ──────────────────────────────────────────────────────────────

from app.api.websocket import router as ws_router          # noqa: E402
from app.api.training import router as training_router      # noqa: E402
from app.api.filesystem_routes import router as fs_router   # noqa: E402
from app.api.dataset import router as dataset_router  # noqa: E402
from app.api.caption_routes import router as caption_router  # noqa: E402
from app.api.api_provider_routes import router as api_provider_router  # noqa: E402
from app.api.settings_routes import router as settings_router  # noqa: E402
from app.api.masking_routes import router as masking_router  # noqa: E402
from app.api.system_routes import router as system_router    # noqa: E402
from app.api.cache_routes import router as cache_router      # noqa: E402
from app.api.project_routes import router as project_router  # noqa: E402
from app.api.tasks_routes import router as tasks_router  # noqa: E402
from app.api.io_routes import router as io_router  # noqa: E402
from app.api.caption_context_routes import router as caption_context_router  # noqa: E402
from app.api.llm_refine_routes import router as llm_refine_router  # noqa: E402
from app.api.caption_variant_routes import router as caption_variant_router  # noqa: E402

app.include_router(ws_router, prefix="/api")
app.include_router(training_router, prefix="/api")
app.include_router(fs_router, prefix="/api")
app.include_router(dataset_router, prefix="/api")
app.include_router(caption_router, prefix="/api/captions")
app.include_router(api_provider_router, prefix="/api/captions", tags=["captions"])
app.include_router(settings_router)
app.include_router(masking_router, prefix="/api")
app.include_router(system_router, prefix="/api")
app.include_router(cache_router, prefix="/api")
app.include_router(project_router, prefix="/api")
app.include_router(tasks_router, prefix="/api")
app.include_router(io_router, prefix="/api")
app.include_router(caption_context_router)
app.include_router(llm_refine_router)
app.include_router(caption_variant_router, prefix="/api")


# ── Static File Mounts ──────────────────────────────────────────────────

from app.core.dataset_manager import dataset_manager  # noqa: E402


class MediaStaticFiles(StaticFiles):
    """StaticFiles mount that guarantees a ``Vary: Origin`` header and forces
    revalidation on every fetch.

    Browsers key the media cache by URL. The global ``CORSMiddleware`` only
    attaches CORS / ``Vary: Origin`` headers when a request carries an
    ``Origin`` header — so a plain ``<img>`` (no-cors, no Origin) response is
    cached WITHOUT ``Vary: Origin`` and WITHOUT ``Access-Control-Allow-Origin``.
    Chrome may then replay that cached body for a later ``crossorigin`` (cors)
    request, which fails the CORS check and breaks the image — notably the
    editor canvas, which needs pixel access via a crossorigin load.

    Emitting ``Vary: Origin`` on the origin-less responses keeps the cors and
    no-cors variants in separate cache entries so they can never be swapped.
    When an Origin IS present, ``CORSMiddleware`` already adds the header, so
    we only fill the gap for origin-less requests.

    We also stamp ``Cache-Control: no-cache, must-revalidate`` so the browser
    must round-trip back to the server (with the file's ETag) on every load.
    Combined with the frontend's ``?t=lastUpdateTime`` cache-bust, this makes
    post-crop / post-adjust image refresh deterministic: the URL changes when
    the view bumps its timestamp, and even if a stale URL is requested the
    server returns the on-disk bytes after a 304 revalidation — never a
    cached pre-crop body. StaticFiles still sets ``Last-Modified`` / ``ETag``
    from the file mtime, so revalidation is cheap (a 304 when unchanged).
    """

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        has_origin = any(k == b"origin" for k, _ in scope.get("headers", []))
        if not has_origin:
            vary = response.headers.get("vary")
            if not vary:
                response.headers["vary"] = "Origin"
            elif "origin" not in vary.lower():
                response.headers["vary"] = f"{vary}, Origin"
        # Defeat browser HTTP-cache heuristics for media. StaticFiles emits
        # ``Last-Modified`` + ``ETag`` from the file mtime, so revalidation
        # is cheap (304 most of the time). The post-crop bug was browsers
        # serving the pre-crop image from their HTTP cache because the
        # original response had no explicit freshness directive.
        response.headers["cache-control"] = "no-cache, must-revalidate"
        return response


os.makedirs(dataset_manager.default_root, exist_ok=True)
app.mount("/media", MediaStaticFiles(directory=dataset_manager.default_root), name="media")


# ── Frontend SPA (container / production) ────────────────────────────────
# When a built Angular bundle is present, serve it at "/" with SPA fallback
# (unknown non-/api, non-/media paths return index.html for client-side
# routing). All API/WS/media routes are registered earlier, so they win.
_frontend_dist = container_config.frontend_dist_dir()

if _frontend_dist:
    from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

    class SpaStaticFiles(StaticFiles):
        """StaticFiles that falls back to index.html for client-side routes.

        In HTML mode Starlette's ``StaticFiles`` *raises* ``HTTPException(404)``
        for a missing path rather than returning a 404 response, so we catch it
        and serve ``index.html`` to let the Angular router handle the route.
        """

        async def get_response(self, path, scope):
            try:
                response = await super().get_response(path, scope)
            except StarletteHTTPException as exc:
                if exc.status_code == 404:
                    return await super().get_response("index.html", scope)
                raise
            if response.status_code == 404:
                return await super().get_response("index.html", scope)
            return response

    app.mount(
        "/", SpaStaticFiles(directory=_frontend_dist, html=True), name="spa"
    )
    logger.info("frontend_spa_mounted", path=_frontend_dist)
else:
    @app.get("/")
    async def root():
        """Health-check / landing endpoint (frontend served via dev server)."""
        return {
            "message": f"MRLN Arcane Tuner API is running. "
                       f"Access frontend via port {FRONTEND_PORT}.",
            "version": __version__,
        }
