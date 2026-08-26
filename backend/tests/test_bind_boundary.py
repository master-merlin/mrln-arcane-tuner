"""The server refuses to run unauthenticated on an address others can reach.

WHAT THIS CAN AND CANNOT OBSERVE — read before trusting it:

uvicorn binds its socket AFTER the ASGI lifespan runs. Verified in
``uvicorn.server.Server.startup`` (0.52.0): line 107 awaits
``lifespan.startup()``, ``loop.create_server`` is at 125+. So at the moment this
check runs there is no bound socket to inspect, and an earlier version of this
guard was abandoned for exactly that reason.

What the app CAN observe is its own command line and environment. Every launcher
in this repo starts the server as ``python -m uvicorn … --host X``, so ``--host``
is genuinely present in ``sys.argv`` of this process. This therefore checks the
DECLARED bind address, and the declaration is what uvicorn will honour.

The residual gap, stated rather than hidden: a programmatic
``uvicorn.run(host=...)`` leaves nothing in argv, and a socket inherited from a
process manager bypasses ``--host`` entirely. No launcher here does either, and
``test_every_launcher_passes_the_host_through`` is what keeps that true.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.core import container_config

REPO_ROOT = Path(__file__).resolve().parents[2]

LAUNCHERS = {
    "start_backend.ps1": REPO_ROOT / "backend" / "start_backend.ps1",
    "start_backend.sh": REPO_ROOT / "backend" / "start_backend.sh",
    "start_backend.bat": REPO_ROOT / "backend" / "start_backend.bat",
    "entrypoint.sh": REPO_ROOT / "entrypoint.sh",
}


# ── the refusal itself ───────────────────────────────────────────────────


class TestFailClosed:
    """The case that matters is the one where it REFUSES.

    A test that only proves the app starts when everything is fine says nothing
    about a guard — it passes identically against a guard that was deleted.
    """

    @pytest.mark.parametrize(
        "host",
        ["0.0.0.0", "::", "192.168.1.50", "10.0.0.7", "0000:0000::0", "not-an-ip"],
    )
    def test_reachable_host_without_a_token_refuses(self, host, monkeypatch):
        monkeypatch.setenv("MRLN_BIND_HOST", host)
        monkeypatch.setenv("MRLN_AUTH_TOKEN", "")
        reason = container_config.bind_is_exposed_without_auth(argv=[])
        assert reason, f"{host!r} was accepted without a token"
        assert "MRLN_AUTH_TOKEN" in reason, "the message must name the fix"
        assert host in reason, "the message must name the offending host"

    def test_an_unparseable_host_fails_closed(self, monkeypatch):
        """A typo must not silently disable the guard.

        'not-an-ip' is covered above; this states the intent separately so the
        behaviour cannot be 'simplified' into treating unknown as safe.
        """
        monkeypatch.setenv("MRLN_BIND_HOST", "0.0.0.0.0")
        monkeypatch.setenv("MRLN_AUTH_TOKEN", "")
        assert container_config.bind_is_exposed_without_auth(argv=[])

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.53"])
    def test_loopback_without_a_token_still_starts(self, host, monkeypatch):
        """Prove the negative: the guard is not simply refusing everything.

        Without this the parametrized refusals above would pass against a
        function that returned a reason unconditionally.
        """
        monkeypatch.setenv("MRLN_BIND_HOST", host)
        monkeypatch.setenv("MRLN_AUTH_TOKEN", "")
        assert container_config.bind_is_exposed_without_auth(argv=[]) is None

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.50"])
    def test_reachable_host_with_a_token_starts(self, host, monkeypatch):
        """The other half of the negative: a token is what unlocks it."""
        monkeypatch.setenv("MRLN_BIND_HOST", host)
        monkeypatch.setenv("MRLN_AUTH_TOKEN", "a-long-random-string")
        assert container_config.bind_is_exposed_without_auth(argv=[]) is None

    def test_a_whitespace_only_token_does_not_count(self, monkeypatch):
        monkeypatch.setenv("MRLN_BIND_HOST", "0.0.0.0")
        monkeypatch.setenv("MRLN_AUTH_TOKEN", "   ")
        assert container_config.bind_is_exposed_without_auth(argv=[])


class TestTheAppItselfRefusesToStart:
    """End-to-end: the refusal is WIRED, not merely available.

    Everything above tests a function that returns a string. A function that
    returns the right string and is never called protects nothing — this drives
    the real lifespan and asserts the application does not come up.
    """

    def test_startup_aborts_on_a_reachable_bind_without_a_token(self, monkeypatch):
        from fastapi.testclient import TestClient

        from app.main import app

        monkeypatch.setenv("MRLN_BIND_HOST", "0.0.0.0")
        monkeypatch.setenv("MRLN_AUTH_TOKEN", "")

        with pytest.raises(RuntimeError) as exc:  # noqa: PT012 - the raise is the assertion
            with TestClient(app):
                pass

        message = str(exc.value)
        assert "MRLN_AUTH_TOKEN" in message
        assert "0.0.0.0" in message

    def test_startup_completes_on_loopback_without_a_token(self, monkeypatch):
        """Prove the negative: the guard did not break ordinary local use.

        Without this, the test above passes just as well against an app that
        refuses to start under every configuration.

        WHY A PROBE APP AND NOT ``app.main.app`` — this is a real constraint,
        not a shortcut. The refusal case CAN use the real app, because the check
        sits at the top of lifespan and raises before a single manager is
        touched. The success case cannot: running the real lifespan to
        completion also runs its SHUTDOWN on exit, which tears down the
        process-wide dataset/job/task managers that the rest of the suite
        shares. Measured, not assumed — doing it that way turned 9 passing
        export tests into 7 failures with 'coroutine emit_entity_change was
        never awaited'.

        So this drives the same guard through a real lifespan and asserts it
        lets the app come up, while
        ``test_the_real_lifespan_calls_the_guard`` is what ties that pattern
        back to main.py.
        """
        from contextlib import asynccontextmanager

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        monkeypatch.setenv("MRLN_BIND_HOST", "127.0.0.1")
        monkeypatch.setenv("MRLN_AUTH_TOKEN", "")

        @asynccontextmanager
        async def _lifespan(_app: FastAPI):
            refusal = container_config.bind_is_exposed_without_auth()
            if refusal:
                raise RuntimeError(refusal)
            yield

        probe = FastAPI(lifespan=_lifespan)

        @probe.get("/up")
        async def _up():
            return {"up": True}

        with TestClient(probe) as client:
            assert client.get("/up").status_code == 200

    def test_the_real_lifespan_calls_the_guard(self):
        """Tie the probe above back to the shipped app.

        The refusal test already proves main.py consults the guard, but it
        proves it only for the failing branch. This pins that the call is
        present and unconditional, so the guard cannot later be moved behind a
        condition that skips it.
        """
        source = (REPO_ROOT / "backend" / "app" / "main.py").read_text(
            encoding="utf-8"
        )
        assert "bind_is_exposed_without_auth()" in source
        # It must be the first thing lifespan does: a refusal after the managers
        # are wired up leaves half-started global state behind.
        lifespan_at = source.index("async def lifespan")
        guard_at = source.index("bind_is_exposed_without_auth()", lifespan_at)
        loop_at = source.index("asyncio.get_running_loop()", lifespan_at)
        assert guard_at < loop_at, (
            "the bind check must run BEFORE lifespan starts wiring managers"
        )


class TestArgvOutranksEnvironment:
    """`--host` is what uvicorn honours, so it must be what the check reads."""

    def test_argv_host_is_used_even_when_the_env_says_loopback(self, monkeypatch):
        """The gap the env var alone cannot close.

        Somebody runs uvicorn by hand with --host 0.0.0.0 and no
        MRLN_BIND_HOST. Reading only the environment would see the safe default
        and wave through a fully exposed server.
        """
        monkeypatch.setenv("MRLN_BIND_HOST", "127.0.0.1")
        monkeypatch.setenv("MRLN_AUTH_TOKEN", "")
        argv = ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
        assert container_config.bind_is_exposed_without_auth(argv=argv)

    def test_equals_form_is_read_too(self, monkeypatch):
        monkeypatch.setenv("MRLN_AUTH_TOKEN", "")
        monkeypatch.delenv("MRLN_BIND_HOST", raising=False)
        assert container_config.bind_is_exposed_without_auth(
            argv=["uvicorn", "--host=0.0.0.0"]
        )

    def test_a_trailing_host_flag_with_no_value_does_not_crash(self, monkeypatch):
        """Malformed argv must not take the server down at startup."""
        monkeypatch.delenv("MRLN_BIND_HOST", raising=False)
        monkeypatch.setenv("MRLN_AUTH_TOKEN", "")
        assert container_config.bind_host(argv=["uvicorn", "--host"]) == "127.0.0.1"

    def test_default_matches_uvicorn_when_nothing_is_set(self, monkeypatch):
        monkeypatch.delenv("MRLN_BIND_HOST", raising=False)
        assert container_config.bind_host(argv=["uvicorn"]) == "127.0.0.1"


# ── the launchers ────────────────────────────────────────────────────────


class TestEveryLauncherIsCovered:
    """A guard that covers three of four launchers is not a guard.

    The ``.bat`` was missed once already in this programme, so the count is
    asserted rather than the membership: adding a fifth launcher without
    wiring it must fail here.
    """

    def test_there_are_exactly_four_launchers(self):
        found = sorted(
            p.name
            for p in [
                *(REPO_ROOT / "backend").glob("start_backend.*"),
                *(REPO_ROOT.glob("entrypoint.sh")),
            ]
        )
        assert found == [
            "entrypoint.sh",
            "start_backend.bat",
            "start_backend.ps1",
            "start_backend.sh",
        ], (
            f"the set of launchers changed: {found}. Every launcher must pass "
            "--host through from MRLN_BIND_HOST; wire the new one and update "
            "this list."
        )

    @pytest.mark.parametrize("name", sorted(LAUNCHERS))
    def test_launcher_does_not_hardcode_a_wildcard_bind(self, name):
        path = LAUNCHERS[name]
        if not path.exists():
            pytest.skip(f"{name} not in this checkout")
        text = path.read_text(encoding="utf-8")
        # A literal `--host 0.0.0.0` is the defect: it cannot be overridden and
        # it is what put an unauthenticated server on every network.
        assert not re.search(r"--host\s+0\.0\.0\.0", text), (
            f"{name} hardcodes --host 0.0.0.0; it must read MRLN_BIND_HOST"
        )

    @pytest.mark.parametrize("name", sorted(LAUNCHERS))
    def test_launcher_passes_the_bind_host_through(self, name):
        path = LAUNCHERS[name]
        if not path.exists():
            pytest.skip(f"{name} not in this checkout")
        text = path.read_text(encoding="utf-8")
        assert "MRLN_BIND_HOST" in text, (
            f"{name} does not reference MRLN_BIND_HOST, so the lifespan check "
            "and the real bind can disagree — which is the one thing this "
            "design depends on not happening."
        )

    def test_local_launchers_default_to_loopback(self):
        """The safe default is the point; the container is the exception."""
        for name in ("start_backend.ps1", "start_backend.sh", "start_backend.bat"):
            path = LAUNCHERS[name]
            if not path.exists():
                continue
            assert "127.0.0.1" in path.read_text(encoding="utf-8"), (
                f"{name} must default to loopback"
            )

    def test_the_container_default_is_wildcard_and_says_why(self):
        """Not an oversight: loopback inside a container breaks port publishing.

        Pinned WITH its reason so nobody 'fixes' it to 127.0.0.1 and makes
        every container unreachable.
        """
        text = LAUNCHERS["entrypoint.sh"].read_text(encoding="utf-8")
        assert 'MRLN_BIND_HOST:-0.0.0.0' in text
        assert "port publishing" in text, (
            "the reason the container is exempt must live next to the exemption"
        )


class TestTheseMatchersActuallyFail:
    """Vacuity checks for the text assertions above."""

    def test_wildcard_matcher_catches_the_old_line(self):
        old = "venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
        assert re.search(r"--host\s+0\.0\.0\.0", old)

    def test_wildcard_matcher_accepts_the_new_line(self):
        new = '--host "${MRLN_BIND_HOST:-127.0.0.1}" --port "${PORT:-8000}"'
        assert not re.search(r"--host\s+0\.0\.0\.0", new)
