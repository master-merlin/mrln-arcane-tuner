"""
Shared test fixtures for the MRLN Arcane Tuner backend.
"""
import sys
import os
import logging
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

# Add backend directory to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(scope="session", autouse=True)
def _isolate_test_db(tmp_path_factory):
    """Redirect the SQLite singleton to a throwaway DB before any test runs.

    Without this, tests that exercise JobManager or any repository write
    rows into ``backend/app/arcane_tuner.db`` (the production DB). The
    DatabaseEngine singleton is constructed lazily on first ``get_db()``
    call, so we preempt it by installing a pre-initialized instance
    pointed at a tmp file. Every subsequent ``get_db()`` call — including
    those buried inside JobHistoryRepository, CheckpointRepository, etc.
    — sees this test instance.
    """
    from app.core.db.engine import DatabaseEngine

    tmp_db_path = tmp_path_factory.mktemp("db") / "test_arcane_tuner.db"
    test_engine = DatabaseEngine(db_path=str(tmp_db_path))
    test_engine.initialize()

    # If the singleton was somehow already created (e.g. by another
    # session-scope fixture), close it before swapping so we don't leak
    # a connection to the prod DB.
    if DatabaseEngine._instance is not None:
        try:
            DatabaseEngine._instance.close()
        except Exception:
            pass
    DatabaseEngine._instance = test_engine

    yield

    try:
        test_engine.close()
    finally:
        DatabaseEngine._instance = None


@pytest.fixture(scope="session", autouse=True)
def _isolate_test_logging():
    """Redirect all logging to tests/tests.log, keeping server.log clean.

    This fixture runs once per test session and:
    - Calls setup_logging WITHOUT the server.log file handler
    - Removes the WebSocket handler (no event loop in tests)
    - Adds a dedicated FileHandler writing to tests/tests.log
    """
    from app.core.logger import setup_logging

    # Configure structlog + standard logging without server.log or WS
    setup_logging(log_level="DEBUG", include_file_handler=False)

    root = logging.getLogger()
    # Remove any remaining handlers (console, websocket) attached by setup_logging
    root.handlers = []

    # Dedicated test log file — reset on each session
    test_log_path = os.path.join(os.path.dirname(__file__), "tests.log")
    if os.path.exists(test_log_path):
        try:
            os.remove(test_log_path)
        except OSError:
            pass

    handler = logging.FileHandler(test_log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    yield

    # Cleanup after session
    root.handlers = []



@pytest.fixture
def mock_definition():
    """Provides a mock ModelDefinition for testing."""
    definition = MagicMock()
    definition.id = "test_model_id"
    definition.name = "Test Model"
    definition.family = "sdxl"
    definition.source = "/fake/model/path"
    definition.components = {}
    return definition


@pytest.fixture
def mock_config():
    """Provides a standard training config dict for testing."""
    return {
        "lora_name": "test_lora",
        "mixed_precision": "fp16",
        "save_precision": "fp16",
        "output_dir": "./test_outputs",
        "datasets": [{"dataset_name": "test_dataset"}],
        "max_train_steps": 10,
        "train_batch_size": 1,
        "learning_rate": 1e-4,
        "network_rank": 16,
        "network_alpha": 8.0,
        "gradient_accumulation_steps": 1,
        "resolutions": [1024],
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provides a temporary output directory that is auto-cleaned."""
    output = tmp_path / "outputs"
    output.mkdir()
    return str(output)


@pytest.fixture
def client():
    """Shared FastAPI test client — avoids duplicating in every test file."""
    from app.main import app

    return TestClient(app)


@pytest.fixture
def frozen_gpu_snapshot(monkeypatch):
    """Hold the LIVE GPU reading fixed for the duration of one test.

    ``VRAMEstimator.estimate`` queries ``system_monitor.snapshot()`` on every
    call (`app/engine/utils/vram_estimator.py:746`) and copies the device-wide
    figures into the report — ``used_mb`` / ``total_mb`` / ``available_mb`` /
    ``fits``, plus two warning strings that interpolate them. So two estimates
    taken a few milliseconds apart legitimately differ whenever another process
    on the box (ComfyUI, a browser, a training run) allocates or frees VRAM
    between them: measured 2026-08-29 as ``used_mb`` 50851 vs 50731 and
    warnings differing only as "49.7 GB" vs "49.5 GB" (LANES LANE-30).

    A test that compares two estimates for equality is asserting something
    about the MODEL, not about the device, so it must see one snapshot for
    both calls — otherwise the suite's colour depends on what else is open on
    the machine, which is not a gate.

    This takes one REAL reading and replays it, so the assertion still runs
    against this box's actual telemetry (and against the no-GPU path, where
    ``gpus`` is empty, unchanged) — it is not a fabricated device.
    """
    from app.core import system_monitor as sm

    snap = sm.system_monitor.snapshot()  # one real read …
    monkeypatch.setattr(sm.system_monitor, "snapshot", lambda: snap)  # … replayed
    return snap


def pytest_collection_finish(session):
    """Guard against silent collection breakage (audit P0 item 0.6).

    A broken import (e.g. the `__init__.py`-collision that motivated this
    guard) can make pytest silently collect 0 tests and exit green instead
    of loudly failing. Skip the check for deliberately narrow runs (`-k`/
    `-m`/explicit node-ids) where 0 matches can be legitimate.
    """
    if session.config.getoption("-k") or session.config.getoption("-m"):
        return
    if session.config.args and any("::" in a for a in session.config.args):
        return
    if len(session.items) == 0:
        pytest.exit(
            "Collected 0 tests — collection is likely broken (see "
            "backend/tests/conftest.py::pytest_collection_finish).",
            returncode=1,
        )


# --- LANE-57: a REAL LLM endpoint on a real socket -------------------------
# The refine boundary guard (``core/llm/refine_guard.py``) is a network
# predicate. Tests exercise it against a socket, never a mock of the client:
# ``fake_ollama`` answers ``/api/tags`` like Ollama does, ``closed_port`` is a
# port nobody listens on.

class _FakeOllama:
    """Minimal Ollama: ``GET /api/tags`` -> ``{"models": [{"name": ...}]}`` and,
    like the real server, the OpenAI-compatible ``GET /v1/models`` ->
    ``{"data": [{"id": ...}]}`` that the api-* captioning providers list
    through (LANE-65). Mutate ``.models`` per test; ``.hits`` counts every
    request so a test can prove a path made NO probe."""

    def __init__(self) -> None:
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.models: list[str] = []
        self.hits: list[str] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server API
                fake.hits.append(self.path)
                if self.path == "/v1/models":
                    body = json.dumps({"data": [{"id": m} for m in fake.models]}).encode()
                elif self.path == "/api/tags":
                    body = json.dumps({"models": [{"name": m} for m in fake.models]}).encode()
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a, **kw):  # silence the test log
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def fake_ollama():
    """A reachable LLM endpoint with a mutable installed-model list."""
    srv = _FakeOllama()
    try:
        yield srv
    finally:
        srv.close()


@pytest.fixture
def closed_port() -> int:
    """A loopback port that was open a moment ago and is now closed - a
    connection to it is refused, which is what an unvalidated endpoint does."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
