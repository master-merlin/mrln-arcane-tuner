"""The port is produced in ONE place, cheaply, and refuses to guess.

WHAT WAS ACTUALLY BROKEN: ``settings.json`` has had a user-editable
``backend_port`` since V4, and not one launcher read it. All four hardcoded
``${PORT:-8000}``, so a user who changed the port in the UI got a server on 8000
and a settings screen insisting otherwise. The fix cannot be "let the shell
parse the JSON" — that is four copies of the rule in three languages, which is
the same second-producer defect the pipeline fix just removed (RULE-21).

So the shell is a CARRIER: it asks this resolver, passes ``--port``, and the app
reads ``--port`` back out of its own argv. These tests are grouped by the three
things that can each silently break that chain:

* ``TestSettingsFileCases`` — the case table, one test per row. The rows differ
  on a distinction that is easy to get backwards and expensive to get wrong:
  an ABSENT file is a first launch (8000, quietly), an UNREADABLE one is a
  misconfiguration (refuse).
* ``TestPrecedence`` — pinned in both directions, because "``PORT`` wins" is
  equally satisfied by an implementation that ignores the setting entirely.
* ``TestImportsInIsolation`` — the resolver stays cheap. Enforced by running it
  where ``app`` does not exist, so a convenience import fails loudly instead of
  quietly adding ~230 ms to every launch.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import port_resolver

BACKEND_DIR = Path(__file__).resolve().parents[1]
RESOLVER_SRC = BACKEND_DIR / "port_resolver.py"


@pytest.fixture(autouse=True)
def _no_ambient_port(monkeypatch):
    """PORT leaks between tests and would mask every settings assertion."""
    monkeypatch.delenv("PORT", raising=False)


def _write_settings(tmp_path: Path, payload, monkeypatch) -> Path:
    """Point the resolver at a settings file of our making.

    Uses ``MRLN_SETTINGS_PATH`` — the same override the container entrypoint
    uses — rather than patching a module global, so what is exercised is the
    real path-resolution branch.
    """
    path = tmp_path / "settings.json"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("MRLN_SETTINGS_PATH", str(path))
    return path


class TestSettingsFileCases:
    """The plan's case table, one test per row."""

    def test_absent_file_is_a_first_launch_not_an_error(self, tmp_path, monkeypatch):
        """THE ROW THAT MATTERS MOST — and that the first draft of the plan
        got wrong.

        ``SettingsManager`` writes its defaults on first construction
        (``settings_manager.py:63``), which happens well after a launcher runs.
        So on a fresh install this resolver ALWAYS finds no file. Refusing here
        would mean a new user's very first launch fails — a worse defect than
        the wrong-port bug this lane exists to fix.
        """
        monkeypatch.setenv("MRLN_SETTINGS_PATH", str(tmp_path / "nope.json"))
        assert port_resolver.resolve_port([]) == 8000

    def test_absent_file_is_silent(self, tmp_path, monkeypatch, capsys):
        """'Silently' is part of the row, so it is asserted, not assumed.

        A warning on every first launch trains users to ignore warnings, which
        costs more than it buys.
        """
        monkeypatch.setenv("MRLN_SETTINGS_PATH", str(tmp_path / "nope.json"))
        port_resolver.resolve_port([])
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""

    def test_a_valid_saved_port_is_honoured(self, tmp_path, monkeypatch):
        """The bug in one line: this is what every launcher used to ignore."""
        _write_settings(tmp_path, {"application": {"backend_port": 8123}}, monkeypatch)
        assert port_resolver.resolve_port([]) == 8123

    def test_unreadable_file_refuses(self, tmp_path, monkeypatch):
        """Exists but cannot be opened.

        Triggered with a DIRECTORY at the settings path rather than a
        permission bit: chmod is advisory-at-best on Windows and this suite
        runs there, so a permissions-based test would silently pass by not
        reproducing the condition. Opening a directory raises ``OSError`` on
        both platforms and — importantly — is not ``FileNotFoundError``, which
        is the exact distinction under test.
        """
        path = tmp_path / "settings.json"
        path.mkdir()
        monkeypatch.setenv("MRLN_SETTINGS_PATH", str(path))
        with pytest.raises(port_resolver.PortResolutionError) as exc:
            port_resolver.resolve_port([])
        assert str(path) in str(exc.value), "the refusal must name the file"

    def test_malformed_json_refuses(self, tmp_path, monkeypatch):
        path = _write_settings(tmp_path, "{ not json at all", monkeypatch)
        with pytest.raises(port_resolver.PortResolutionError) as exc:
            port_resolver.resolve_port([])
        assert str(path) in str(exc.value)

    @pytest.mark.parametrize(
        "value,why",
        [
            ("not-a-number", "a string that is not a number"),
            (None, "null — how a half-migrated file looks"),
            ([8000], "a list"),
            ({"port": 8000}, "an object"),
            (True, "a bool; int(True) is 1, which would be a valid-looking port"),
        ],
    )
    def test_non_integer_refuses(self, tmp_path, monkeypatch, value, why):
        _write_settings(
            tmp_path, {"application": {"backend_port": value}}, monkeypatch
        )
        with pytest.raises(port_resolver.PortResolutionError) as exc:
            port_resolver.resolve_port([])
        assert repr(value) in str(exc.value), f"refusal must name the value ({why})"

    @pytest.mark.parametrize("value", [0, -1, 65536, 99999])
    def test_out_of_range_refuses(self, tmp_path, monkeypatch, value):
        """0 is included deliberately.

        ``--port 0`` means 'let the OS choose', which cannot work through a
        carrier: the launcher has to state the port before the socket exists.
        Accepting it would produce a server on an arbitrary port that the
        settings screen still claims is 0.
        """
        _write_settings(
            tmp_path, {"application": {"backend_port": value}}, monkeypatch
        )
        with pytest.raises(port_resolver.PortResolutionError) as exc:
            port_resolver.resolve_port([])
        assert str(value) in str(exc.value)

    def test_a_numeric_string_is_accepted(self, tmp_path, monkeypatch):
        """Prove the negative: the guard is not simply refusing everything.

        A hand-edited file with ``"backend_port": "8123"`` is a typo, not a
        misconfiguration — the intent is unambiguous.
        """
        _write_settings(
            tmp_path, {"application": {"backend_port": "8123"}}, monkeypatch
        )
        assert port_resolver.resolve_port([]) == 8123

    def test_a_file_without_the_key_defaults_rather_than_refusing(
        self, tmp_path, monkeypatch
    ):
        """Missing is not corrupt.

        A settings file predating the key, or holding only other modules, is
        well-formed. Refusing on it would break every existing install on
        upgrade, which is the same failure shape as refusing on absent.
        """
        _write_settings(tmp_path, {"application": {"log_level": "INFO"}}, monkeypatch)
        assert port_resolver.resolve_port([]) == 8000


class TestPrecedence:
    """Pinned in BOTH directions.

    'The command line wins' is satisfied just as well by an implementation that
    reads nothing else, and that implementation would silently re-break the
    user-editable port. Every rung has a test that it is reachable.
    """

    def test_argv_beats_everything(self, tmp_path, monkeypatch):
        _write_settings(tmp_path, {"application": {"backend_port": 8123}}, monkeypatch)
        monkeypatch.setenv("PORT", "9001")
        assert port_resolver.resolve_port(["--port", "7000"]) == 7000
        assert port_resolver.resolve_port(["--port=7000"]) == 7000

    def test_env_beats_the_setting(self, tmp_path, monkeypatch):
        _write_settings(tmp_path, {"application": {"backend_port": 8123}}, monkeypatch)
        monkeypatch.setenv("PORT", "9001")
        assert port_resolver.resolve_port([]) == 9001

    def test_the_setting_is_reached_when_neither_is_given(self, tmp_path, monkeypatch):
        _write_settings(tmp_path, {"application": {"backend_port": 8123}}, monkeypatch)
        assert port_resolver.resolve_port([]) == 8123

    def test_settings_fallback_beats_the_disk_read(self, tmp_path, monkeypatch):
        """The in-app caller already holds the settings; it must not re-read.

        ``main.py`` passes the loaded ``backend_port`` straight in. If the disk
        read outranked it, an in-process settings change would be ignored until
        restart while the app reported the new value.
        """
        _write_settings(tmp_path, {"application": {"backend_port": 8123}}, monkeypatch)
        assert port_resolver.resolve_port([], settings_fallback=8500) == 8500

    def test_a_junk_env_port_warns_and_falls_through(self, tmp_path, monkeypatch):
        """Tolerated, but never silent (invariant #4).

        The tolerance is deliberate and pinned by the pipeline's own tests: a
        stray environment variable must not take a training run down. The
        warning is what keeps 'tolerated' from becoming 'invisible'.
        """
        _write_settings(tmp_path, {"application": {"backend_port": 8123}}, monkeypatch)
        monkeypatch.setenv("PORT", "not-a-number")
        seen: list[dict] = []
        assert (
            port_resolver.resolve_port([], warn=lambda **kw: seen.append(kw)) == 8123
        )
        assert seen and seen[0].get("value") == "not-a-number"

    def test_an_environ_override_also_steers_the_settings_path(self, tmp_path):
        """One environment, not two.

        ``settings_path()`` used to read ``os.environ`` while ``resolve_port``
        honoured an ``environ`` override, so a caller passing one got the
        ambient settings file — a split-brain of exactly the kind this module
        exists to remove, one level down.
        """
        path = tmp_path / "elsewhere.json"
        path.write_text(
            json.dumps({"application": {"backend_port": 8321}}), encoding="utf-8"
        )
        assert (
            port_resolver.resolve_port([], environ={"MRLN_SETTINGS_PATH": str(path)})
            == 8321
        )

    def test_the_default_warn_channel_actually_writes(self, capsys):
        """Otherwise the test above only proves a callback is invoked.

        A launcher passes no ``warn``, so the default path is the one users
        meet; if it were ``pass`` the assertion above would still be green.
        """
        port_resolver.resolve_port(
            [], environ={"PORT": "junk"}, settings_fallback=8000
        )
        assert "junk" in capsys.readouterr().err


class TestArgvParsing:
    def test_port_without_a_value_does_not_crash(self):
        """Malformed argv is uvicorn's error to report, not ours to die on."""
        assert port_resolver.resolve_port(["--port"], settings_fallback=8123) == 8123

    def test_an_unparseable_argv_port_defers_rather_than_inventing_one(self):
        """The command line is explicit; contradicting it would be worse.

        uvicorn will reject ``--port abc`` itself with a clear message. Picking
        a different number here would start a server on a port nobody asked
        for.
        """
        assert (
            port_resolver.resolve_port(["--port", "abc"], settings_fallback=8123)
            == 8123
        )

    def test_a_later_flag_is_not_mistaken_for_the_port(self):
        assert port_resolver.resolve_port(
            ["--host", "127.0.0.1", "--port", "9100", "--reload"]
        ) == 9100


# ── the cost guard ───────────────────────────────────────────────────────


def _run_isolated(tmp_path: Path, code: str) -> subprocess.CompletedProcess:
    """Run *code* in an interpreter that cannot see ``app`` at all.

    The resolver is COPIED to an empty directory rather than imported from
    ``backend/``: leaving it in place would leave ``app`` importable beside it,
    and then "no app import happened" would be a claim about this particular
    run rather than a property of the module. Here a stray
    ``from app.core… import`` is a hard ``ImportError``.

    ``-E`` blocks ``PYTHONPATH`` at the interpreter rather than by scrubbing the
    environment dict: an inherited entry pointing at ``backend/`` would quietly
    restore exactly what this is trying to remove, and ``-E`` cannot be defeated
    by a variable spelled differently. ``-s`` drops user site-packages.

    NOT ``-I``, which was the first attempt and failed honestly: ``-I`` implies
    ``-E -s`` *and* removes the current directory from ``sys.path``, so the
    sandbox could not find the copied resolver either. The distinction matters —
    it would have been easy to read that ``ModuleNotFoundError`` as the guard
    firing.
    """
    sandbox = tmp_path / "isolated"
    sandbox.mkdir()
    shutil.copy2(RESOLVER_SRC, sandbox / "port_resolver.py")

    env = dict(os.environ)
    env["MRLN_SETTINGS_PATH"] = str(sandbox / "absent.json")
    env.pop("PORT", None)

    return subprocess.run(
        [sys.executable, "-E", "-s", "-c", code],
        capture_output=True,
        text=True,
        cwd=str(sandbox),
        env=env,
        timeout=120,
    )


class TestImportsInIsolation:
    """The resolver must stay outside the application, permanently.

    MEASURED, not assumed (``python -X importtime``, this machine):
    ``app.core.container_config`` costs ~246 ms cumulative — ``app.core.compat``
    238 ms, of which ``structlog`` 152 ms and ``structlog.dev`` 106 ms. Stdlib
    ``json`` is ~14 ms. Four launches' worth of that, to read one integer, is
    what a convenience import would reintroduce.

    Asserted on the OBSERVABLE (it imports and answers where ``app`` does not
    exist), not on a hand-kept list of permitted imports, which would need
    updating every time the stdlib usage changed and would prove less.
    """

    def test_it_imports_and_answers_without_the_app_package(self, tmp_path):
        proc = _run_isolated(
            tmp_path,
            "import port_resolver;"
            "print(port_resolver.resolve_port([]))",
        )
        assert proc.returncode == 0, (
            "the resolver could not be imported without the app package — "
            "something in it now depends on `app`, which puts ~230 ms back on "
            f"every launch.\nstderr:\n{proc.stderr}"
        )
        assert proc.stdout.strip() == "8000"

    def test_the_isolation_is_real_and_this_harness_can_fail(self, tmp_path):
        """Vacuity guard.

        Without this, the test above would pass just as happily in a sandbox
        where ``app`` WAS importable, and would be evidence of nothing. This
        proves the environment genuinely lacks it — so the assertion above is
        about the module, not about the harness.
        """
        proc = _run_isolated(tmp_path, "import app")
        assert proc.returncode != 0, "the sandbox can still see `app`"
        assert "ModuleNotFoundError" in proc.stderr

    def test_no_heavy_module_is_pulled_in(self, tmp_path):
        """Belt to the isolation's braces, and it says WHY each name is here.

        ``app`` being absent already blocks the direct route. This catches the
        indirect one — someone importing ``structlog`` or ``httpx`` straight
        into the resolver, which the sandbox alone would happily allow.
        """
        proc = _run_isolated(
            tmp_path,
            "import sys, port_resolver;"
            "heavy=[m for m in sys.modules "
            "if m.split('.')[0] in {'app','structlog','httpx','torch','fastapi'}];"
            "print(heavy)",
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "[]", (
            f"the resolver pulled in heavy modules: {proc.stdout.strip()}"
        )


class TestTheCliContract:
    """``_main`` is what the shell launchers call, so its output shape is API.

    A launcher captures stdout and passes it to ``--port``. Anything extra on
    stdout — a banner, a warning, a stray newline of prose — becomes part of
    the port and the server fails to start with a confusing error.
    """

    def test_success_prints_only_the_number(self, tmp_path, monkeypatch):
        _write_settings(tmp_path, {"application": {"backend_port": 8123}}, monkeypatch)
        proc = subprocess.run(
            [sys.executable, str(RESOLVER_SRC)],
            capture_output=True,
            text=True,
            env={**os.environ, "MRLN_SETTINGS_PATH": str(tmp_path / "settings.json")},
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "8123"
        assert proc.stdout.count("\n") == 1, "exactly one line, or the shell breaks"

    def test_failure_exits_non_zero_and_explains_on_stderr(self, tmp_path):
        """The launcher's refusal depends entirely on this exit code.

        If a malformed file exited 0 with an empty stdout, the launcher would
        start uvicorn with ``--port ''`` — a confusing crash instead of the
        message naming the file.
        """
        path = tmp_path / "settings.json"
        path.write_text("{ broken", encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(RESOLVER_SRC)],
            capture_output=True,
            text=True,
            env={**os.environ, "MRLN_SETTINGS_PATH": str(path)},
            timeout=120,
        )
        assert proc.returncode != 0
        assert proc.stdout.strip() == "", "no port may be printed on refusal"
        assert str(path) in proc.stderr


class TestTheInAppEntryPointDelegates:
    """One producer (RULE-21), asserted behaviourally.

    ``container_config.resolve_port`` is the in-app door to the same resolver.
    If it ever grows its own copy of the rules, the app and the launchers can
    disagree about the port — which is precisely the bug being fixed, moved one
    layer inward.
    """

    def test_it_honours_the_command_line_the_launcher_carried(self, monkeypatch):
        """The carrier chain's last link.

        The launcher resolves the port and passes ``--port``; this is the app
        reading it back. Without this, the app would recompute — and the whole
        point of the carrier shape is that it does not.
        """
        from app.core import container_config

        monkeypatch.delenv("PORT", raising=False)
        assert container_config.resolve_port(8000, argv=["--port", "8123"]) == 8123

    def test_the_existing_env_and_default_precedence_is_unchanged(self, monkeypatch):
        """Extending, not duplicating, what TASK-3a already pinned."""
        from app.core import container_config

        monkeypatch.setenv("PORT", "9001")
        assert container_config.resolve_port(8123, argv=[]) == 9001
        monkeypatch.delenv("PORT", raising=False)
        assert container_config.resolve_port(8123, argv=[]) == 8123

    def test_it_is_the_same_function_underneath(self):
        """Structural, because the behavioural tests above cannot see a fork.

        A copied implementation could satisfy every assertion in this class and
        still drift the day someone changes one copy.
        """
        src = RESOLVER_SRC.parent / "app" / "core" / "container_config.py"
        body = src.read_text(encoding="utf-8")
        start = body.index("def resolve_port")
        end = body.index("\ndef ", start + 1)
        assert "port_resolver.resolve_port" in body[start:end], (
            "container_config.resolve_port must delegate; a private copy of the "
            "precedence rules is the second producer this lane removed"
        )
