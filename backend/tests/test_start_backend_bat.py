"""``start_backend.bat`` relaunches the server in the same terminal (LANE-56).

The restart contract (``app/core/restart_contract.py``): the supervisor sets
``MRLN_SUPERVISED=1``, relaunches on exit code ``RESTART_EXIT_CODE`` in the
same console with ``MRLN_RESTART=1``, re-resolves the port before it, and
passes every other exit code through — a crash must not loop.

Two kinds of test, and the executable one is the guard:

* ``TestTheBatRelaunches`` RUNS the real bat under ``cmd`` with a throwaway
  venv (so ``activate.bat`` and ``venv\\Scripts\\python.exe`` are REAL), a stub
  ``port_resolver.py`` and a stub ``uvicorn`` package whose exit codes the test
  chooses. It asserts on what the second run SAW in its environment and on the
  bat's own exit code. Windows only — ``cmd`` is the thing under test.
* ``TestTheBatText`` pins the shape that makes the loop safe (``setlocal``
  once and above the label; the label below ``activate.bat`` so PATH does not
  grow per pass; the errorlevel captured before it is compared; the literal
  equal to the contract's constant). Cheap, and runs everywhere.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
BAT = BACKEND / "start_backend.bat"

sys.path.insert(0, str(BACKEND))
from app.core.restart_contract import RESTART_EXIT_CODE  # noqa: E402

windows_only = pytest.mark.skipif(os.name != "nt", reason="start_backend.bat runs under cmd")


def _lines() -> list[str]:
    return BAT.read_text(encoding="utf-8").splitlines()


def _index(lines: list[str], needle: str) -> int:
    hits = [i for i, line in enumerate(lines) if needle in line]
    assert hits, f"{needle!r} not found in start_backend.bat"
    return hits[0]


# ── the executable test ───────────────────────────────────────────────────


_STUB_UVICORN_MAIN = r'''
"""Stub uvicorn: record what the bat handed us, exit with the next scripted code."""
import json, os, sys
record = os.environ["STUB_RECORD"]
codes = [int(c) for c in os.environ["STUB_CODES"].split(",")]
runs = []
if os.path.exists(record):
    with open(record, encoding="utf-8") as fh:
        runs = [json.loads(line) for line in fh if line.strip()]
entry = {
    "argv": sys.argv[1:],
    "MRLN_RESTART": os.environ.get("MRLN_RESTART"),
    "MRLN_SUPERVISED": os.environ.get("MRLN_SUPERVISED"),
    "cwd": os.getcwd(),
}
with open(record, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(entry) + "\n")
sys.exit(codes[min(len(runs), len(codes) - 1)])
'''

_STUB_RESOLVER = r'''
"""Stub port_resolver: one line on stdout, and a mark that we were asked."""
import os
with open(os.environ["STUB_RESOLVER_LOG"], "a", encoding="utf-8") as fh:
    fh.write("asked\n")
print(os.environ.get("PORT", "8765"))
'''


class _Harness:
    """A copy of the bat beside a REAL throwaway venv and the two stubs."""

    def __init__(self, tmp_path: Path, bat_text: str | None = None) -> None:
        self.backend = tmp_path / "backend"
        self.backend.mkdir(parents=True)
        self.bat = self.backend / "start_backend.bat"
        # newline="\n": the shipped bat is LF (.gitattributes eol=lf) and cmd
        # re-reads a batch file by byte offset, so the copy under test must be
        # the LF bytes cmd actually executes, not a CRLF twin write_text makes.
        self.bat.write_text(bat_text if bat_text is not None
                            else BAT.read_text(encoding="utf-8"),
                            encoding="utf-8", newline="\n")
        assert b"\r" not in self.bat.read_bytes(), "bat copy must be LF like the shipped file"
        # with_pip=False keeps this under a second; the stubs need no packages.
        # A host that cannot build a venv (embedded Python, AV blocking the
        # python.exe copy) skips by name rather than failing on setup.
        try:
            venv.EnvBuilder(with_pip=False).create(self.backend / "venv")
        except (OSError, subprocess.CalledProcessError) as exc:
            pytest.skip(f"cannot build a throwaway venv here: {exc}")
        assert (self.backend / "venv" / "Scripts" / "python.exe").exists()
        assert (self.backend / "venv" / "Scripts" / "activate.bat").exists()
        (self.backend / "port_resolver.py").write_text(_STUB_RESOLVER, encoding="utf-8")
        pkgs = tmp_path / "pkgs" / "uvicorn"
        pkgs.mkdir(parents=True)
        (pkgs / "__init__.py").write_text("", encoding="utf-8")
        (pkgs / "__main__.py").write_text(_STUB_UVICORN_MAIN, encoding="utf-8")
        self.pythonpath = str(tmp_path / "pkgs")
        self.record = tmp_path / "uvicorn_runs.jsonl"
        self.resolver_log = tmp_path / "resolver_calls.txt"

    def env(self, codes: str) -> dict[str, str]:
        env = dict(os.environ)
        for stale in ("MRLN_RESTART", "MRLN_SUPERVISED", "MRLN_BIND_HOST"):
            env.pop(stale, None)
        env.update({
            "PORT": "8765",
            "PYTHONPATH": self.pythonpath,
            "STUB_RECORD": str(self.record),
            "STUB_CODES": codes,
            "STUB_RESOLVER_LOG": str(self.resolver_log),
        })
        return env

    def run(self, codes: str, *, trailer: str = "") -> subprocess.CompletedProcess:
        """``cmd /c "call bat & <trailer>"`` — the trailer runs in the PARENT
        shell after the bat returns, which is how a leaked variable is seen."""
        command = f'call "{self.bat}"' + (f" & {trailer}" if trailer else "")
        # A raw string, not a list: list2cmdline backslash-escapes the inner
        # quotes and cmd cannot read that. /s = strip the outer quotes, always.
        return subprocess.run(f'cmd /d /s /c "{command}"', capture_output=True,
                              text=True, env=self.env(codes), timeout=120,
                              cwd=str(self.backend.parent))

    def runs(self) -> list[dict]:
        if not self.record.exists():
            return []
        return [json.loads(line) for line in
                self.record.read_text(encoding="utf-8").splitlines() if line.strip()]

    def resolver_calls(self) -> int:
        if not self.resolver_log.exists():
            return 0
        return len(self.resolver_log.read_text(encoding="utf-8").split())


@windows_only
class TestTheBatRelaunches:
    def test_the_sentinel_relaunches_once_in_the_same_shell_with_the_port_re_resolved(
            self, tmp_path):
        h = _Harness(tmp_path)
        proc = h.run(f"{RESTART_EXIT_CODE},0")

        runs = h.runs()
        assert len(runs) == 2, (runs, proc.stdout, proc.stderr)
        assert runs[0]["MRLN_RESTART"] is None, "the first launch is not a restart"
        assert runs[1]["MRLN_RESTART"] == "1", "the relaunch must say it is one (main.py:177)"
        assert runs[0]["MRLN_SUPERVISED"] == "1" and runs[1]["MRLN_SUPERVISED"] == "1"
        assert h.resolver_calls() == 2, "the port is resolved before EVERY launch, never replayed"
        assert "--port 8765" in " ".join(runs[1]["argv"])
        assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
        assert "start_backend: restart requested" in proc.stdout

    def test_any_other_exit_code_is_the_bats_exit_code_and_does_not_loop(self, tmp_path):
        """uvicorn ``STARTUP_FAILURE`` is 3 (``uvicorn/config.py:80``): a server
        that could not start must NOT be started again forever."""
        h = _Harness(tmp_path)
        proc = h.run("3,0")

        assert len(h.runs()) == 1, h.runs()
        assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
        assert "restart requested" not in proc.stdout

    def test_a_relaunch_that_cannot_bind_is_narrated_not_retried(self, tmp_path):
        """75 then 3: the relaunched server failed to start. One line naming the
        likely holder of the port, exit 3 — narration, not a loop (Shepherd cut
        the retry: an unobserved hypothesis is not a branch to ship)."""
        h = _Harness(tmp_path)
        proc = h.run(f"{RESTART_EXIT_CODE},3,0")

        assert len(h.runs()) == 2, h.runs()
        assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
        assert "could not bind" in proc.stdout
        # The supervised path bypasses the launcher, so nothing writes
        # restart.log here: the evidence is the console and server.log.
        assert "server.log" in proc.stdout
        assert "restart.log" not in proc.stdout

    def test_the_supervisor_flag_does_not_leak_into_the_operators_shell(self, tmp_path):
        """``setlocal`` proved by its EFFECT: after the bat returns, the parent
        shell must not carry ``MRLN_SUPERVISED`` — a later bare ``uvicorn`` from
        that window would inherit it, exit 75 and never be relaunched (the
        fallback broken by the new variable). Positive control: the same bat
        without its ``setlocal`` line leaks, so the probe is known to see one."""
        h = _Harness(tmp_path)
        proc = h.run("0", trailer="set MRLN_")
        assert "MRLN_SUPERVISED=" not in proc.stdout, proc.stdout

        text = BAT.read_text(encoding="utf-8")
        assert "\nsetlocal\n" in text
        leaky = _Harness(tmp_path / "leaky", bat_text=text.replace("\nsetlocal\n", "\n", 1))
        control = leaky.run("0", trailer="set MRLN_")
        assert "MRLN_SUPERVISED=1" in control.stdout, control.stdout


# ── the text contract ─────────────────────────────────────────────────────


class TestTheBatText:
    def test_setlocal_once_above_the_first_set_and_above_the_label(self):
        lines = _lines()
        setlocals = [i for i, line in enumerate(lines) if line.strip().lower() == "setlocal"]
        assert len(setlocals) == 1, "one setlocal — a nested one per pass hits cmd's 32-level limit"
        first_set = next(i for i, line in enumerate(lines)
                         if line.strip().lower().startswith("set "))
        assert setlocals[0] < first_set
        assert setlocals[0] < _index(lines, ":launch")

    def test_the_label_sits_below_activate_and_bindhost_and_above_the_resolver(self):
        """A label above ``activate.bat`` grows PATH every pass; a label above
        the resolver would replay the port instead of re-resolving it."""
        lines = _lines()
        activate = _index(lines, "call venv\\Scripts\\activate.bat")
        bindhost = _index(lines, "set MRLN_BIND_HOST=127.0.0.1")
        label = _index(lines, ":launch")
        resolver = _index(lines, "set MRLN_RESOLVED_PORT=")
        assert activate < bindhost < label < resolver

    def test_the_supervisor_flag_is_set_before_uvicorn_runs(self):
        lines = _lines()
        assert _index(lines, "set MRLN_SUPERVISED=1") < _index(lines, "-m uvicorn")

    def test_the_exit_code_is_captured_before_it_is_compared(self):
        """``%errorlevel%`` inside a parenthesised block expands at parse time;
        captured on the line right after uvicorn it is the server's, always."""
        lines = _lines()
        uvicorn = _index(lines, "-m uvicorn")
        assert lines[uvicorn + 1].strip().lower() == "set mrln_exit=%errorlevel%"

    def test_the_compared_literal_is_the_contracts_constant(self):
        """RULE-21: cmd cannot import the constant, so the literal is the wire
        and this pin is what keeps it from drifting."""
        lines = _lines()
        compare = _index(lines, '"%MRLN_EXIT%"==')
        assert f'"{RESTART_EXIT_CODE}"' in lines[compare], lines[compare]

    def test_the_relaunch_sets_the_restart_flag_and_everything_else_passes_through(self):
        lines = _lines()
        compare = _index(lines, f'"%MRLN_EXIT%"=="{RESTART_EXIT_CODE}"')
        block = "\n".join(lines[compare:compare + 6])
        assert "set MRLN_RESTART=1" in block
        assert "goto launch" in block
        assert any("exit /b %MRLN_EXIT%" in line for line in lines[compare:])
