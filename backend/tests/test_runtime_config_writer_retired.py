"""The backend does not write ``runtime-config.json``. Everything else stays.

WHAT WAS WRONG, precisely — and it is NOT "we write keys nobody needs":

The SPA *does* fetch ``/runtime-config.json`` at bootstrap, via Angular's app
initializer, and it *does* parse ``backendPort``/``frontendPort``. What it does
not do is let those values influence any URL — every frontend URL is
origin-relative. The keys are ``@deprecated`` as URL inputs, parsed for
compatibility, and ``load()`` continues on defaults when the file is missing.

The defect was that the backend REWROTE that file at runtime into
``frontend/public/`` — the SOURCE checkout — while the served build is
``frontend/dist/frontend/browser`` (``/app/frontend/browser`` in the container,
via ``MRLN_FRONTEND_DIST``). So every rewrite updated a file the SPA never
loads: pointless in dev, and a runtime write into an ephemeral or read-only
checkout in a container. It was harmless only because the reader tolerates the
file being absent.

So the writer is retired and NOTHING ELSE IS. ``frontend/public/runtime-config.json``
still ships, the keys still exist, the service still parses and validates them.
ARCHITECTURE D2 — deprecate, never drop — and an existing install may have that
file on disk with values set.

HOW THIS GUARD ENUMERATES, and the one thing it cannot see:

Searching for the FILENAME is the fragile shape: a future writer that builds the
path from fragments or a constant would not match. Searching for the FUNCTION
name does not have that hole — any caller must name what it calls, however it
builds its path. Both are checked, because they catch different things: a call
site (the function) and a re-implementation (the filename).

The residual, stated as a limit rather than left to be read as completeness:
neither search sees a re-implementation that assembles the filename from string
fragments (``"runtime-" + "config.json"``). Nothing does that today; if it ever
appears, this guard is silent about it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "backend"
SHIPPED = REPO_ROOT / "frontend" / "public" / "runtime-config.json"
SERVICE = (
    REPO_ROOT / "frontend" / "src" / "app" / "services" / "runtime-config.service.ts"
)

WRITER_FUNCTION = "write_runtime_config"
CONFIG_FILENAME = "runtime-config.json"


def _backend_sources() -> list[Path]:
    """Every backend ``.py`` except caches and this file's own siblings.

    Tests are excluded deliberately: this file necessarily contains both search
    terms, and so would any future test of the same property. Including them
    would make the guard fail on itself.
    """
    out = []
    for path in BACKEND.rglob("*.py"):
        parts = set(path.parts)
        if "__pycache__" in parts or "venv" in parts:
            continue
        if "tests" in parts:
            continue
        out.append(path)
    return out


def _names_the_file_in_code(path: Path) -> bool:
    """True when *path* names the config file outside a ``#`` comment."""
    return any(
        CONFIG_FILENAME in line
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not line.lstrip().startswith("#")
    )


class TestNothingWritesItAnyMore:
    def test_there_are_sources_to_search(self):
        """A search over an empty set proves nothing and looks green.

        The path anchoring here is `__file__`-relative, so it survives a
        worktree; if it ever stops matching the layout, this fails loudly
        instead of quietly finding zero call sites in zero files.
        """
        sources = _backend_sources()
        assert len(sources) > 100, (
            f"only {len(sources)} backend sources found under {BACKEND} — the "
            "enumeration is broken, so the assertions below are vacuous"
        )

    def test_no_call_sites_of_the_writer(self):
        """The robust half: a caller must name the function.

        This catches a call site regardless of how it constructs the path,
        which a filename search cannot.
        """
        offenders = [
            str(p.relative_to(REPO_ROOT))
            for p in _backend_sources()
            if WRITER_FUNCTION in p.read_text(encoding="utf-8", errors="replace")
        ]
        assert not offenders, (
            f"{WRITER_FUNCTION} is called again from {offenders}. The backend "
            "must not write into the frontend SOURCE checkout at runtime — the "
            "served build is a different directory and never reads it."
        )

    def test_no_backend_module_names_the_config_file(self):
        """The other half: catches a re-implementation rather than a call.

        Comment lines are excluded, and that is not a loophole — it is the
        second time in one day this exact trap fired. A guard that forbids
        NAMING a removed feature punishes documenting the removal, and the next
        person deletes the explanation to get green. The removal is explained
        at both former call sites in prose that necessarily says the filename;
        a re-implementation has to put it in a string literal, which is not a
        comment.
        """
        offenders = [
            str(p.relative_to(REPO_ROOT))
            for p in _backend_sources()
            if _names_the_file_in_code(p) and "test_runtime_config" not in p.name
        ]
        assert not offenders, (
            f"backend code references {CONFIG_FILENAME} again: {offenders}. If "
            "this is a READ it may be legitimate — but say so explicitly and "
            "narrow this guard, rather than deleting it."
        )


class TestEverythingElseSurvived:
    """Prove the negative. A guard that only asserts the deletion is satisfied
    by someone deleting the artifact, the keys and the reader too — which would
    break an existing install and violate D2."""

    def test_the_shipped_file_still_exists_and_still_has_both_keys(self):
        assert SHIPPED.exists(), (
            "frontend/public/runtime-config.json was deleted. Retiring the "
            "WRITER does not retire the file: the SPA still fetches it at "
            "bootstrap, and an existing install may have one on disk."
        )
        data = json.loads(SHIPPED.read_text(encoding="utf-8"))
        assert "backendPort" in data and "frontendPort" in data, (
            "the deprecated keys were dropped from the shipped artifact. "
            "ARCHITECTURE D2: deprecate, never drop."
        )

    def test_the_frontend_still_fetches_and_parses_it(self):
        if not SERVICE.exists():
            pytest.skip("runtime-config.service.ts not in this checkout")
        text = SERVICE.read_text(encoding="utf-8")
        assert CONFIG_FILENAME in text, "the SPA no longer fetches the file"
        for key in ("backendPort", "frontendPort"):
            assert key in text, f"the service stopped parsing {key}"

    def test_the_reader_tolerates_the_file_being_absent(self):
        """Why the pointless write was harmless rather than broken.

        The container has been serving a build whose copy nobody rewrote, and
        nothing failed — because ``load()`` catches and continues on defaults.
        If that tolerance is ever removed, retiring the writer stops being safe
        and this test is where that shows up.
        """
        if not SERVICE.exists():
            pytest.skip("runtime-config.service.ts not in this checkout")
        text = SERVICE.read_text(encoding="utf-8")
        assert ".catch(" in text or "catch (" in text, (
            "the runtime-config fetch no longer has a failure path. With no "
            "writer, a missing or stale file must remain non-fatal."
        )


class TestTheseMatchersActuallyFail:
    """Vacuity checks — every assertion above is 'substring absent', which
    passes identically against a tree that never had the thing."""

    def test_the_writer_matcher_notices_a_reintroduced_call(self, tmp_path):
        planted = tmp_path / "sneaky.py"
        planted.write_text(
            "from app.core.runtime_config import write_runtime_config\n",
            encoding="utf-8",
        )
        assert WRITER_FUNCTION in planted.read_text(encoding="utf-8")

    def test_the_filename_matcher_notices_a_reimplementation(self, tmp_path):
        planted = tmp_path / "sneaky.py"
        planted.write_text(
            'P = Path(root) / "frontend" / "public" / "runtime-config.json"\n',
            encoding="utf-8",
        )
        assert _names_the_file_in_code(planted)

    def test_the_filename_matcher_tolerates_the_name_in_a_comment(self, tmp_path):
        """The inverse, and the reason the matcher was narrowed.

        This fired on its first run against my own explanation of the removal
        at both former call sites. Documenting why something is gone must not
        fail the guard that removed it.
        """
        planted = tmp_path / "documented.py"
        planted.write_text(
            "# No runtime-config.json write here; the served build never reads it.\n"
            "x = 1\n",
            encoding="utf-8",
        )
        assert not _names_the_file_in_code(planted)

    def test_the_documented_blind_spot_is_real(self):
        """Named in the docstring, demonstrated here rather than asserted.

        A guard's stated limit should be as checkable as its coverage,
        otherwise "we know about that case" is just a claim.
        """
        assembled = 'name = "runtime-" + "config.json"'
        assert CONFIG_FILENAME not in assembled, (
            "the fragment case is now matchable — tighten the guard and delete "
            "the limit from the docstring"
        )
