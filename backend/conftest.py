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

Everything here is anchored on ``__file__``, never the CWD.
"""
import os
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from tests.support.worker_paths import worker_suffixed  # noqa: E402

SERVER_TEST_LOG = worker_suffixed(_BACKEND / "tests" / "server-test.log")
os.environ["MRLN_SERVER_LOG_PATH"] = str(SERVER_TEST_LOG)
