"""The restart contract has ONE producer (LANE-56, RULE-21).

The sentinel exit code and the "am I supervised?" question are answered in
``app.core.restart_contract`` and nowhere else. The two supervisor scripts
(``start_backend.bat``, ``entrypoint.sh``) cannot import Python, so they carry
the literal — and ``test_start_backend_bat.py`` / ``test_container_hardening.py``
pin those literals to the constant here. That pin is the wire.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
CONTRACT = BACKEND / "app" / "core" / "restart_contract.py"


def test_the_sentinel_is_75_and_reserved():
    """75 is BSD ``EX_TEMPFAIL``: not 1 (``TerminateProcess`` / a Python error),
    not 2 (usage), not 3 (uvicorn ``STARTUP_FAILURE``, ``uvicorn/config.py:80``),
    not 130/143 (signals). Reserved in ECOSYSTEM §6 2026-09-02; frozen at the
    release that ships it (ARCHITECTURE D2)."""
    from app.core.restart_contract import RESTART_EXIT_CODE

    assert RESTART_EXIT_CODE == 75
    assert RESTART_EXIT_CODE not in (0, 1, 2, 3, 130, 143)


def test_is_supervised_reads_exactly_the_reserved_value(monkeypatch):
    """Strict ``== "1"``: an operator who exported ``MRLN_SUPERVISED=yes`` by
    hand from a bare terminal must NOT be routed to an exit that nothing will
    answer — the fallback (LANE-51/56 launcher) stays the safe default."""
    from app.core import restart_contract

    monkeypatch.setenv("MRLN_SUPERVISED", "1")
    assert restart_contract.is_supervised() is True

    monkeypatch.delenv("MRLN_SUPERVISED")
    assert restart_contract.is_supervised() is False

    for wrong in ("", "0", "true", "yes", " 1"):
        monkeypatch.setenv("MRLN_SUPERVISED", wrong)
        assert restart_contract.is_supervised() is False, repr(wrong)


def test_is_supervised_is_read_at_call_time_not_import_time(monkeypatch):
    """The supervisor sets the variable before launch, but the module may be
    imported by a test or a tool under any environment: the answer must follow
    the environment at the moment of the question."""
    from app.core import restart_contract

    monkeypatch.delenv("MRLN_SUPERVISED", raising=False)
    assert restart_contract.is_supervised() is False
    monkeypatch.setenv("MRLN_SUPERVISED", "1")
    assert restart_contract.is_supervised() is True


def test_the_contract_module_imports_only_the_standard_library():
    """ARCHITECTURE D1: nothing imported at startup may raise, and this module
    sits under ``app.core`` on the restart path of BOTH callers (the route and
    the self-updater). Stdlib only, so it cannot fail for a missing package."""
    tree = ast.parse(CONTRACT.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported, "the scan found nothing — the module changed shape"
    non_stdlib = imported - set(sys.stdlib_module_names)
    assert not non_stdlib, non_stdlib


def test_the_docstring_states_the_whole_contract():
    """The docstring IS the contract the two scripts implement; a supervisor
    author reads it, not the ECOSYSTEM row. Every clause has a reader."""
    from app.core import restart_contract

    doc = restart_contract.__doc__ or ""
    for clause in ("MRLN_SUPERVISED=1", "75", "same console", "MRLN_RESTART=1",
                   "port", "must not loop"):
        assert clause in doc, clause
