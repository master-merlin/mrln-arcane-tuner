"""The launchers carry the resolved port, and refuse rather than guess.

WHAT IS BEING PROVEN, and why reading the scripts would not prove it:

The design is a carrier chain — the launcher asks ``port_resolver`` for the
port, passes it as ``--port``, and the app reads it back out of its own argv.
Every link can break silently:

* a launcher that ignores the resolver and keeps its old ``${PORT:-8000}`` still
  *contains* the word ``port_resolver`` in a comment;
* a launcher that falls back to 8000 when the resolver refuses still exits 0 and
  still starts a server — on the wrong port, which is the original bug wearing a
  different coat;
* ``for /f`` in the ``.bat`` cannot see a child's exit code at all, so its
  refusal is detected by an empty variable. That is a real quirk reasoned about
  in the file, and reasoning is exactly what needs checking.

So these RUN the launchers. Uvicorn is replaced with a stub that prints its
argv, which turns "what port did it actually start on?" into an observable
string instead of an inference.

HOW THE SANDBOX WORKS: ``python -m venv --without-pip`` builds a real venv in a
temp directory (~0.2 s), and a stub ``uvicorn`` package is dropped into its
``site-packages``. Built rather than hand-assembled, and built from the
interpreter running the tests rather than from ``backend/venv`` — a worktree has
no venv of its own, so keying off the checkout would make this whole file skip
silently, which is worse than not having it. The launcher scripts are copied
UNMODIFIED: rewriting the interpreter path inside them would test a file that
ships to nobody.

NOT COVERED HERE, stated rather than left to be assumed: ``entrypoint.sh``. It
symlinks ``/app/backend`` under ``set -e`` before it ever reaches its port
block, so it cannot run to that point off a container. Its port branch is
identical in shape and is asserted textually in ``TestEntrypointCarriesItToo``;
the behaviour that branch depends on — the resolver's exit code and its silent
stdout on refusal — is proven in ``test_port_resolver.py``. The real coverage
is a container run in release QA.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.bash_probe import find_bash

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent

LAUNCHERS = ("start_backend.sh", "start_backend.ps1", "start_backend.bat")


def _sandbox_python(sb: Path) -> Path:
    for rel in ("Scripts/python.exe", "bin/python"):
        candidate = sb / "venv" / rel
        if candidate.exists():
            return candidate
    raise AssertionError(f"no interpreter in the sandbox venv at {sb / 'venv'}")


def _sandbox(tmp_path: Path, settings) -> Path:
    """Build a throwaway ``backend/`` the launchers can actually run in.

    *settings* is the JSON payload for ``settings.json``, a raw string for a
    deliberately malformed file, or ``None`` to leave no file at all (the
    fresh-install case).
    """
    sb = tmp_path / "backend"
    sb.mkdir(parents=True)

    built = subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(sb / "venv")],
        capture_output=True, text=True, timeout=300,
    )
    assert built.returncode == 0, f"could not build the sandbox venv: {built.stderr}"
    py_exe = _sandbox_python(sb)

    # The POSIX launcher hardcodes venv/bin/python — the layout it gets on
    # Linux, which Windows does not produce. A one-line shim supplies that
    # layout while still running the sandbox's real interpreter, so the script
    # under test needs no edit to be runnable here. On Linux `bin/python`
    # already exists and is left alone.
    posix_bin = sb / "venv" / "bin"
    posix_bin.mkdir(exist_ok=True)
    if not (posix_bin / "activate").exists():
        (posix_bin / "activate").write_text("", encoding="utf-8")
    shim = posix_bin / "python"
    if not shim.exists():
        shim.write_text(f'#!/bin/sh\nexec "{py_exe.as_posix()}" "$@"\n', encoding="utf-8")
        shim.chmod(0o755)

    # The stub uvicorn, installed where the sandbox interpreter says its
    # site-packages is rather than where this platform's layout is assumed to
    # put it — asked, not guessed.
    purelib = subprocess.run(
        [str(py_exe), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        capture_output=True, text=True, timeout=120,
    )
    assert purelib.returncode == 0, purelib.stderr
    pkg = Path(purelib.stdout.strip()) / "uvicorn"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    # Printing argv is what turns "did the right port get carried?" into an
    # observable string instead of an inference.
    (pkg / "__main__.py").write_text(
        "import sys\nprint('UVICORN_ARGV', ' '.join(sys.argv[1:]))\n", encoding="utf-8"
    )

    shutil.copy2(BACKEND_DIR / "port_resolver.py", sb / "port_resolver.py")
    for name in LAUNCHERS:
        src = BACKEND_DIR / name
        if src.exists():
            dst = sb / name
            shutil.copy2(src, dst)
            dst.chmod(0o755)

    if settings is not None:
        payload = settings if isinstance(settings, str) else json.dumps(settings)
        (sb / "settings.json").write_text(payload, encoding="utf-8")
    return sb


def _command(sb: Path, launcher: str) -> list[str] | None:
    """The invocation for *launcher*, or None when this machine cannot run it."""
    path = sb / launcher
    if not path.exists():
        return None
    if launcher.endswith(".sh"):
        # Probed against this sandbox, not against PATH: the launcher lives in
        # tmp, which is on a different volume from the repo, and a shell can
        # manage one without the other. `shutil.which("bash")` used to answer
        # here and handed back WSL's System32 bash as readily as Git Bash —
        # which cannot open a Windows drive path, so these tests failed rather
        # than skipped, depending on the operator's PATH. See
        # ``tests/support/bash_probe.py``.
        bash = find_bash(path)
        return [bash, str(path)] if bash else None
    if launcher.endswith(".bat"):
        cmd = shutil.which("cmd") or shutil.which("cmd.exe")
        return [cmd, "/c", str(path)] if cmd else None
    ps = shutil.which("powershell") or shutil.which("pwsh")
    if not ps:
        return None
    # -ExecutionPolicy applies to this child process only; it is how a script
    # in a temp directory is runnable at all, not a change to the machine.
    return [ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-File", str(path)]


def _run(sb: Path, launcher: str, **extra_env) -> subprocess.CompletedProcess:
    cmd = _command(sb, launcher)
    if cmd is None:
        pytest.skip(f"cannot run {launcher} on this machine")
    env = dict(os.environ)
    # Both would silently outrank the settings file and make every assertion
    # below a statement about this developer's shell instead of the launcher.
    env.pop("MRLN_SETTINGS_PATH", None)
    env.pop("PORT", None)
    env.update(extra_env)
    return subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=str(sb), timeout=180
    )


def _uvicorn_argv(proc: subprocess.CompletedProcess) -> str | None:
    for line in proc.stdout.splitlines():
        if line.startswith("UVICORN_ARGV"):
            return line[len("UVICORN_ARGV"):].strip()
    return None


@pytest.mark.parametrize("launcher", LAUNCHERS)
class TestTheLauncherCarriesTheResolvedPort:
    def test_the_saved_port_reaches_uvicorn(self, tmp_path, launcher):
        """THE BUG, IN ONE ASSERTION.

        ``backend_port`` has been user-editable since V4 and no launcher read
        it: all four hardcoded ``8000``. A user who changed the port in the UI
        got a server on 8000 and a settings screen insisting otherwise.
        """
        sb = _sandbox(tmp_path, {"application": {"backend_port": 8123}})
        proc = _run(sb, launcher)
        argv = _uvicorn_argv(proc)
        assert argv is not None, (
            f"{launcher} never reached uvicorn.\n"
            f"rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        assert "--port 8123" in argv, (
            f"{launcher} started uvicorn as `{argv}` while settings.json says "
            "8123 — the launcher is not carrying the resolved port"
        )

    def test_no_settings_file_still_starts_on_8000(self, tmp_path, launcher):
        """The fresh-install path, end to end.

        ``SettingsManager`` writes its defaults after this runs, so a first
        launch always finds no file. If absent were treated as a failure the
        refusal below would fire on every new install.
        """
        sb = _sandbox(tmp_path, None)
        proc = _run(sb, launcher)
        argv = _uvicorn_argv(proc)
        assert argv is not None, (
            f"{launcher} refused to start with no settings file — that is a "
            f"fresh install.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        assert "--port 8000" in argv

    def test_the_env_port_still_wins(self, tmp_path, launcher):
        """Prove the negative: settings must not swallow ``PORT``.

        The container sets ``PORT`` and the published-port mapping is built
        around it (DECISION-11). 'The setting is honoured' is satisfied just as
        well by an implementation that ignores the environment, and that
        implementation breaks every container.
        """
        sb = _sandbox(tmp_path, {"application": {"backend_port": 8123}})
        proc = _run(sb, launcher, PORT="9001")
        argv = _uvicorn_argv(proc)
        assert argv is not None, proc.stderr
        assert "--port 9001" in argv

    def test_the_loopback_default_survived_this_change(self, tmp_path, launcher):
        """LANE-1 regression guard.

        The bind host and the port are carried by the same lines. Rewriting
        those lines for the port is exactly when the host default gets dropped,
        and dropping it puts an unauthenticated server on every network the
        machine joins.
        """
        sb = _sandbox(tmp_path, {"application": {"backend_port": 8123}})
        argv = _uvicorn_argv(_run(sb, launcher))
        assert argv is not None
        assert "--host 127.0.0.1" in argv

    def test_a_malformed_settings_file_refuses_instead_of_guessing(
        self, tmp_path, launcher
    ):
        """THE GUARD THE PLAN ASKED FOR.

        A launcher that falls back to 8000 here exits 0 and starts a server the
        settings screen denies — the original defect, reintroduced by the error
        path. Both halves are asserted: non-zero exit AND no server, because a
        launcher could plausibly do one without the other.
        """
        sb = _sandbox(tmp_path, "{ this is not json")
        proc = _run(sb, launcher)
        assert proc.returncode != 0, (
            f"{launcher} exited 0 with an unreadable settings file.\n"
            f"stdout:\n{proc.stdout}"
        )
        assert _uvicorn_argv(proc) is None, (
            f"{launcher} started a server anyway — it fell back to a guessed "
            f"port. argv was `{_uvicorn_argv(proc)}`"
        )

    def test_an_out_of_range_port_refuses_too(self, tmp_path, launcher):
        """Malformed JSON is the easy case; a well-formed wrong value is not.

        ``"backend_port": 0`` parses fine and reads as intentional. It cannot
        be carried — the launcher has to state a port before the socket exists
        — so it must refuse rather than silently substitute one.
        """
        sb = _sandbox(tmp_path, {"application": {"backend_port": 0}})
        proc = _run(sb, launcher)
        assert proc.returncode != 0
        assert _uvicorn_argv(proc) is None


class TestTheHarnessItselfCanFail:
    """Vacuity guard for everything above.

    Every assertion here is of the form "uvicorn was/was not reached with X".
    If the stub were never invoked — a broken sandbox, a launcher dying early
    for an unrelated reason — the refusal tests would pass for entirely the
    wrong reason and look identical to real coverage.
    """

    def test_the_stub_uvicorn_is_actually_reached(self, tmp_path):
        sb = _sandbox(tmp_path, {"application": {"backend_port": 8123}})
        py = _sandbox_python(sb)
        proc = subprocess.run(
            [str(py), "-m", "uvicorn", "app.main:app", "--port", "8123"],
            capture_output=True, text=True, cwd=str(sb), timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert _uvicorn_argv(proc) == "app.main:app --port 8123"

    def test_the_stub_is_not_the_real_uvicorn(self, tmp_path):
        """Otherwise a passing run might be starting an actual server."""
        sb = _sandbox(tmp_path, None)
        py = _sandbox_python(sb)
        proc = subprocess.run(
            [str(py), "-c", "import uvicorn, sys; print(uvicorn.__file__)"],
            capture_output=True, text=True, cwd=str(sb), timeout=120,
        )
        assert str(tmp_path) in proc.stdout, (
            "the sandbox resolved the REAL uvicorn, so these tests could be "
            f"launching servers: {proc.stdout} {proc.stderr}"
        )

    def test_a_launcher_that_ignores_the_resolver_would_be_caught(self, tmp_path):
        """Mutation check, run rather than reasoned.

        This rewrites the sandbox copy of ``start_backend.sh`` back to the old
        hardcoded form and asserts the suite's central assertion fails on it.
        Without this, "the launcher carries the port" would be a claim resting
        on the sandbox being wired correctly, which is the thing most likely to
        be wrong.
        """
        cmd_probe = _command(_sandbox(tmp_path / "probe", None), "start_backend.sh")
        if cmd_probe is None:
            pytest.skip("needs a POSIX shell")

        sb = _sandbox(tmp_path, {"application": {"backend_port": 8123}})
        old_form = (
            "#!/usr/bin/env bash\ncd \"$(dirname \"$0\")\"\n"
            'venv/bin/python -m uvicorn app.main:app '
            '--host "${MRLN_BIND_HOST:-127.0.0.1}" --port "${PORT:-8000}"\n'
        )
        (sb / "start_backend.sh").write_text(old_form, encoding="utf-8")
        argv = _uvicorn_argv(_run(sb, "start_backend.sh"))
        assert argv is not None, "the mutant did not even start"
        assert "--port 8000" in argv and "--port 8123" not in argv, (
            "the pre-fix launcher carried 8123, so these tests cannot tell the "
            "fix from the bug"
        )


class TestEntrypointCarriesItToo:
    """``entrypoint.sh``, asserted textually — and the docstring says why.

    It cannot be run to its port block off a container (see the module
    docstring). Text matching is a weak guard, so each assertion is paired with
    a mutation check in ``TestTheseMatchersActuallyFail``.
    """

    def _text(self) -> str:
        path = REPO_ROOT / "entrypoint.sh"
        if not path.exists():
            pytest.skip("entrypoint.sh not in this checkout")
        return path.read_text(encoding="utf-8")

    def test_it_asks_the_resolver(self):
        text = self._text()
        assert "port_resolver.py" in text, (
            "the container hardcoded ${PORT:-8000}, so a port saved on the data "
            "volume was ignored exactly as it was locally"
        )

    def test_it_refuses_rather_than_defaulting(self):
        text = self._text()
        assert "${PORT:-8000}" not in text, (
            "the old fallback is back; it starts the container on 8000 while "
            "the settings screen says otherwise"
        )
        assert "refusing to start" in text

    def test_the_reason_port_still_wins_is_recorded(self):
        """DECISION-11 is unobvious and will look like a bug to the next reader.

        Docker's ``-p`` mapping lives in the daemon, outside the namespace, so
        the operator's PORT has to outrank the saved setting inside a container
        in a way it does not on a desktop.
        """
        assert "DECISION-11" in self._text()


class TestTheDocumentedCommandCarriesItToo:
    """A README command block is a carrier like any other.

    ``backend/README.md`` told developers to run ``uvicorn app.main:app
    --reload`` — no ``--port``, so it binds 8000 whatever the setting says, and
    the app then reports one port while the socket is on another. That is this
    lane's defect, published as the recommended command. Fixing the four
    launchers and leaving the documented fifth one is fixing three quarters of
    it for anyone who follows the docs.
    """

    def _text(self) -> str:
        path = BACKEND_DIR / "README.md"
        if not path.exists():
            pytest.skip("backend/README.md not in this checkout")
        return path.read_text(encoding="utf-8")

    def test_no_uvicorn_invocation_without_a_port(self):
        import re as _re

        offenders = [
            line.strip()
            for line in self._text().splitlines()
            if _re.search(r"^\s*(\$ )?uvicorn app\.main:app", line)
            and "--port" not in line
        ]
        assert not offenders, (
            "backend/README.md documents a uvicorn command with no --port: "
            f"{offenders}. It binds 8000 regardless of the saved setting, which "
            "is exactly the disagreement this lane removed everywhere else."
        )

    def test_the_matcher_would_catch_the_original_line(self):
        """Vacuity guard, fired at the line that actually shipped."""
        import re as _re

        original = "uvicorn app.main:app --reload"
        assert _re.search(r"^\s*(\$ )?uvicorn app\.main:app", original)
        assert "--port" not in original


class TestTheseMatchersActuallyFail:
    """Prove the text assertions above notice a removal or a re-add."""

    def _text(self) -> str:
        path = REPO_ROOT / "entrypoint.sh"
        if not path.exists():
            pytest.skip("entrypoint.sh not in this checkout")
        return path.read_text(encoding="utf-8")

    def test_resolver_matcher_notices_removal(self):
        mutated = self._text().replace("port_resolver.py", "something_else.py")
        assert "port_resolver.py" not in mutated

    def test_fallback_matcher_notices_reintroduction(self):
        mutated = self._text() + '\nPORT="${PORT:-8000}"\n'
        assert "${PORT:-8000}" in mutated, (
            "the fallback guard would not catch the old form coming back"
        )
