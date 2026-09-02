"""The boot that fails must not have its log deleted by the boot that recovers.

LANE-56, measured 2026-09-01. A UI restart's replacement was ended while it was
still starting; the record in ``restart.log`` said only ``exit_code 1``, and the
child's own log — the one place that could have said how far it got — was
``server.log``, which the user's recovery start then unlinked. Every time this
sequence runs, the evidence is destroyed by the action taken to recover from it,
which is the worst possible ordering.

``setup_logging`` now MOVES the previous session aside instead. Both properties
are asserted here, because keeping only one is how the old behaviour was
justified:

* ``server.log`` still contains THIS session only (``/api/system/logs`` and the
  Server screen read it and would otherwise present a foreign boot's lines);
* the previous session is still readable, at ``server.prev.log``.

The test that used to stand here (``test_precision.py::TestServerLogReset``)
asserted neither: it defined a local ``patched_setup`` that called ``os.remove``
itself and then asserted the file was gone. It never called ``setup_logging``,
so it would have passed against any implementation whatsoever — including one
that did nothing at all.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from app.core import logger as logger_module


@pytest.fixture
def logs(tmp_path, monkeypatch):
    """Point both log paths into a tmp dir and restore logging afterwards."""
    current = tmp_path / "server.log"
    previous = tmp_path / "server.prev.log"
    monkeypatch.setattr(logger_module, "SERVER_LOG_PATH", current)
    monkeypatch.setattr(logger_module, "PREVIOUS_SERVER_LOG_PATH", previous)
    root = logging.getLogger()
    saved = list(root.handlers)
    yield current, previous
    for handler in list(root.handlers):
        if handler not in saved:
            handler.close()
            root.removeHandler(handler)
    root.handlers = saved


def _boot(logs, text: str) -> None:
    """One server session: rotate, then write *text* through the file handler."""
    current, _ = logs
    logger_module.setup_logging("INFO", include_file_handler=True)
    logging.getLogger("boot").info(text)
    for handler in list(logging.getLogger().handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logging.getLogger().removeHandler(handler)
    assert current.exists()


class TestThePreviousBootSurvivesTheNextOne:

    def test_the_failed_boots_log_is_still_readable_after_the_recovery_boot(self, logs):
        """The exact sequence from UAT-5.9, in order."""
        current, previous = logs
        _boot(logs, "REPLACEMENT-THAT-NEVER-CAME-UP")
        _boot(logs, "THE-USERS-RECOVERY-START")

        assert "REPLACEMENT-THAT-NEVER-CAME-UP" in previous.read_text(encoding="utf-8"), \
            "the recovery boot destroyed the log of the boot it was recovering from"
        assert "THE-USERS-RECOVERY-START" in current.read_text(encoding="utf-8")

    def test_server_log_still_holds_this_session_only(self, logs):
        """The property the unlink was there for, kept. Without it the Server
        screen would show a previous boot's lines as if they were this one's."""
        current, _ = logs
        _boot(logs, "FIRST-SESSION")
        _boot(logs, "SECOND-SESSION")

        body = current.read_text(encoding="utf-8")
        assert "SECOND-SESSION" in body
        assert "FIRST-SESSION" not in body, "server.log accumulated two sessions"

    def test_only_one_generation_is_kept(self, logs):
        """Bounded, like every other buffer here: three boots leave two files,
        not three, and the oldest is gone rather than accumulating forever."""
        current, previous = logs
        _boot(logs, "BOOT-ONE")
        _boot(logs, "BOOT-TWO")
        _boot(logs, "BOOT-THREE")

        assert "BOOT-THREE" in current.read_text(encoding="utf-8")
        prev_body = previous.read_text(encoding="utf-8")
        assert "BOOT-TWO" in prev_body
        assert "BOOT-ONE" not in prev_body

    def test_a_first_ever_boot_has_nothing_to_rotate_and_does_not_care(self, logs):
        current, previous = logs
        _boot(logs, "ONLY-BOOT")

        assert "ONLY-BOOT" in current.read_text(encoding="utf-8")
        assert not previous.exists()

    def test_a_worker_without_the_file_handler_rotates_nothing(self, logs):
        """`include_file_handler=False` is the worker/test path (conftest uses
        it). A worker must never move the live server's log out from under it."""
        current, previous = logs
        current.write_text("THE-LIVE-SERVERS-LOG", encoding="utf-8")

        logger_module.setup_logging("INFO", include_file_handler=False)

        assert current.read_text(encoding="utf-8") == "THE-LIVE-SERVERS-LOG"
        assert not previous.exists()


def test_the_two_paths_are_siblings_and_distinct():
    """Anchored on the module file, not the CWD (D10), and not the same file —
    a rotation onto itself would be a delete wearing a different name."""
    assert logger_module.PREVIOUS_SERVER_LOG_PATH != logger_module.SERVER_LOG_PATH
    assert (logger_module.PREVIOUS_SERVER_LOG_PATH.parent
            == logger_module.SERVER_LOG_PATH.parent)
    assert isinstance(logger_module.PREVIOUS_SERVER_LOG_PATH, Path)
