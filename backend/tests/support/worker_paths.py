"""Per-xdist-worker names for the files the suite writes outside ``tmp_path``.

The one rule (LANE-63): a path is suffixed with the worker id ONLY when
``PYTEST_XDIST_WORKER`` is set. Absent (a plain ``pytest backend/tests/test_x.py``,
a debugger, a bisect) the serial name comes back unchanged, so a single-process
run is byte-identical to before xdist existed.
"""
import os
from pathlib import Path

WORKER_ENV = "PYTEST_XDIST_WORKER"


def current_worker() -> str | None:
    """The xdist worker id of this process (``gw0`` …), or ``None`` when serial."""
    return os.environ.get(WORKER_ENV) or None


def worker_suffixed(path: Path, worker: str | None = None) -> Path:
    """``tests.log`` → ``tests-gw3.log`` under worker ``gw3``; unchanged when serial."""
    if worker is None:
        worker = current_worker()
    if not worker:
        return path
    return path.with_name(f"{path.stem}-{worker}{path.suffix}")
