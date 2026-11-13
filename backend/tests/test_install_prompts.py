"""Answering ``n`` at an installer's Y/n prompt stops the step it asks about.

Each test writes the installer's TEXT into a fresh temp folder and runs it
there with ``n`` on stdin. Nothing real can install:

* only the installer is copied - ``torch_stack.py`` and ``requirements.txt``
  are not beside it, so no install step has its inputs;
* the child's PATH is a stub folder plus the Windows system folders. The
  stubs ``python.cmd`` and ``pip.cmd`` append ``<name> <args>`` to a log and
  exit 1, so no real interpreter, pip or network is reachable by name;
* a "venv that exists" is EMPTY placeholder files: if a gate broke and the
  script ran them, they would fail to start and install nothing;
* stdin ends after the ``n``, so a ``pause`` returns at once.

The child sees no ``ANSWER`` (cmd variables are case-insensitive) and no
``MRLN_*`` variable: an inherited ``ANSWER=n`` would make a script that reads
its answer at parse time look gated.

The runner and the assertions take the script text, not the file, so the same
checks can be run on any version of an installer without touching the tree.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
INSTALL_BAT = BACKEND / "install.bat"
INSTALL_PS1 = BACKEND / "install.ps1"

pytestmark = pytest.mark.skipif(
    os.name != "nt", reason="install.bat / install.ps1 run under cmd / Windows PowerShell"
)

_STUB = '@echo %~n0 %*>>"%STUB_LOG%"\r\n@exit /b 1\r\n'
_TIMEOUT_S = 60


@dataclass(frozen=True)
class Run:
    output: str
    returncode: int
    stub_log: str
    venv: Path


def _system_root() -> Path:
    return Path(os.environ.get("SystemRoot", r"C:\Windows"))


def _child_env(stubs: Path, log: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in ("ANSWER", "PATH") and not key.upper().startswith("MRLN_")
    }
    root = _system_root()
    env["PATH"] = os.pathsep.join([str(stubs), str(root / "System32"), str(root)])
    env["STUB_LOG"] = str(log)
    return env


def run_installer(text: str, suffix: str, workdir: Path, *, venv_exists: bool) -> Run:
    """Run installer ``text`` (a ``.bat`` or ``.ps1``) in ``workdir`` with ``n`` on stdin."""
    stubs = workdir / "stubs"
    stubs.mkdir(parents=True)
    for name in ("python", "pip"):
        (stubs / f"{name}.cmd").write_bytes(_STUB.encode("ascii"))
    log = workdir / "stub.log"

    venv = workdir / "venv"
    if venv_exists:
        scripts = venv / "Scripts"
        scripts.mkdir(parents=True)
        # python.exe is what today's installers look for; activate.bat and
        # Activate.ps1 are what older ones looked for.
        for name in ("python.exe", "activate.bat", "Activate.ps1"):
            (scripts / name).write_bytes(b"")

    script = workdir / f"install{suffix}"
    body = text.replace("\r\n", "\n").replace("\n", "\r\n")
    if suffix == ".bat":
        script.write_bytes(body.encode("utf-8"))
        argv = [str(_system_root() / "System32" / "cmd.exe"), "/d", "/c", str(script)]
    else:
        script.write_bytes(body.encode("utf-8-sig"))
        powershell = _system_root() / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        argv = [str(powershell), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]

    done = subprocess.run(
        argv,
        cwd=workdir,
        env=_child_env(stubs, log),
        input=b"n\r\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=_TIMEOUT_S,
        check=False,
    )
    return Run(
        output=done.stdout.decode("utf-8", errors="replace"),
        returncode=done.returncode,
        stub_log=log.read_text(encoding="utf-8", errors="replace") if log.exists() else "",
        venv=venv,
    )


def assert_aborted(run: Run) -> None:
    """``n`` on "Install into it?" stops the install: nothing runs after the prompt."""
    assert "Aborted." in run.output, run.output
    assert run.returncode == 0, run.output
    assert run.stub_log == "", f"ran after the prompt:\n{run.stub_log}"


def assert_venv_skipped(run: Run) -> None:
    """``n`` on "Create a virtual environment?" creates none."""
    assert "Skipping venv" in run.output, run.output
    assert not run.venv.exists(), "a venv folder was created"
    assert "-m venv" not in run.stub_log, f"asked python to create a venv:\n{run.stub_log}"


def _text(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def test_install_bat_n_on_an_existing_venv_aborts(tmp_path):
    assert_aborted(run_installer(_text(INSTALL_BAT), ".bat", tmp_path, venv_exists=True))


def test_install_bat_n_on_create_venv_creates_none(tmp_path):
    assert_venv_skipped(run_installer(_text(INSTALL_BAT), ".bat", tmp_path, venv_exists=False))


def test_install_ps1_n_on_an_existing_venv_aborts(tmp_path):
    assert_aborted(run_installer(_text(INSTALL_PS1), ".ps1", tmp_path, venv_exists=True))


def test_install_ps1_n_on_create_venv_creates_none(tmp_path):
    assert_venv_skipped(run_installer(_text(INSTALL_PS1), ".ps1", tmp_path, venv_exists=False))
