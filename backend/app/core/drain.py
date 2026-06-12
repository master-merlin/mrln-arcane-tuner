"""Global drain flag for update-pending restarts.

Dependency-free so the task manager and the self-update service can both
import it without an import cycle. When draining, new GPU-lane tasks are
rejected while in-flight in-process work runs to completion. Training jobs
are restart-safe subprocesses and are intentionally NOT gated here.
"""

from __future__ import annotations

import threading


class DrainActive(RuntimeError):
    """Raised when new work is rejected because an update restart is pending."""


_draining = threading.Event()


def set_draining(active: bool) -> None:
    if active:
        _draining.set()
    else:
        _draining.clear()


def is_draining() -> bool:
    return _draining.is_set()
