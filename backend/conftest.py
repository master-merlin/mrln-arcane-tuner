"""Root conftest for the backend suite — the earliest code pytest runs per process.

It exists for ONE ordering problem (LANE-63). ``app/main.py`` calls
``setup_logging`` at import time, and ``app/core/logger.py`` fixes
``SERVER_LOG_PATH`` at import time; a test module that does
``from app.main import app`` therefore unlinks and reopens the server log during
COLLECTION, before any fixture — session-scoped or autouse — has run. With
pytest-xdist that happens once per worker, all on the same file, and on this
box it also happens to the log of whatever backend is live.

pytest loads this file before it imports any test module, and xdist sets
``PYTEST_XDIST_WORKER`` in each worker before it builds the worker's config
(``xdist/remote.py`` ``setup_config``: the env assignment precedes
``_prepareconfig``), so here — and only here — the environment can still be
shaped before ``app`` is imported. ``testpaths`` names three roots
(``tests``, ``app/engine/tests``, ``app/api/tests``); this file sits above all
three, which is why it is not in ``tests/conftest.py``.

What it does, per process:

* ``MRLN_SERVER_LOG_PATH`` (ECOSYSTEM §6) → ``tests/server-test.log``, suffixed
  with the worker id under xdist. Assigned UNCONDITIONALLY: xdist workers
  inherit the controller's environment, and the controller (no worker id) has
  already set the un-suffixed name — a ``setdefault`` in the worker would keep
  the controller's value and put every worker back on one file. An operator's
  own ``MRLN_SERVER_LOG_PATH`` is a pointer for the backend, not for pytest,
  and is overridden for the same reason. Serial runs are diverted too — a
  serial ``pytest`` used to wipe the live ``backend/server.log`` — but the NAME
  is the serial one, so the per-worker suffix is the only thing xdist adds.
* ``sys.path`` gets ``backend/`` so ``from tests.support…`` and ``from app…``
  resolve from every root (``--import-mode=importlib`` never adds it itself).
* every test carrying ``@pytest.mark.xdist_group("<name>")`` holds the
  cross-process lock ``_harness/.<name>.lock`` from before its fixtures run
  until after they are torn down (``tests/support/machine_lock.py``). The
  group marker serialises the group inside ONE run; the lock serialises it
  against every other pytest on the box. Marking is the only thing a test
  author does — the pairing lives here, in one place, for all three roots.
  A wait that runs out is an ERROR naming the holder, never a skip.

Everything here is anchored on ``__file__``, never the CWD.
"""
import os
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from tests.support.machine_lock import (  # noqa: E402
    MachineLockTimeout,
    acquire,
    lock_path,
    release,
    resolve_timeout,
)
from tests.support.worker_paths import worker_suffixed  # noqa: E402

SERVER_TEST_LOG = worker_suffixed(_BACKEND / "tests" / "server-test.log")
os.environ["MRLN_SERVER_LOG_PATH"] = str(SERVER_TEST_LOG)

# The SQLite database has the same import-time problem: importing ``app.main``
# constructs the ``DatabaseEngine`` singleton and initialises the file
# (``app/core/db/engine.py`` ``database_initialized``) during COLLECTION,
# before ``tests/conftest.py::_isolate_test_db`` swaps in its tmp engine. With
# no seam set that file is ``backend/app/arcane_tuner.db`` — the live one in
# the main tree — and under xdist N workers initialise ONE WAL database at
# once: measured 2026-09-02, ``-n 16`` never got past it (16 workers, each
# holding the db/-wal/-shm, 9 s CPU apiece, no progress in 25 minutes; ``-n 8``
# squeezed through). ``MRLN_DB_PATH`` is the engine's released env seam, so
# the import-time engine goes to a per-process file next to the logs; the
# session fixture still replaces it with a tmp_path engine for the tests.
IMPORT_TIME_DB = worker_suffixed(_BACKEND / "tests" / "import-time.db")
os.environ["MRLN_DB_PATH"] = str(IMPORT_TIME_DB)

_LOCK_ATTR = "_mrln_machine_lock_path"


def _group_of(item) -> str | None:
    marker = item.get_closest_marker("xdist_group")
    if marker is None:
        return None
    name = marker.args[0] if marker.args else marker.kwargs.get("name")
    return str(name) if name else None


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Acquire BEFORE any fixture (tryfirst: ahead of the runner's setup)."""
    group = _group_of(item)
    if group is None:
        return
    path = lock_path(group)
    try:
        acquire(path, resolve_timeout(None))
    except MachineLockTimeout as exc:
        timeout_msg = str(exc)
    else:
        setattr(item, _LOCK_ATTR, path)
        return
    pytest.fail(f"xdist_group({group!r}) machine lock: {timeout_msg}", pytrace=False)


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    """Release AFTER the fixtures are torn down (trylast), whatever the outcome."""
    path = getattr(item, _LOCK_ATTR, None)
    if path is not None:
        delattr(item, _LOCK_ATTR)
        release(path)
