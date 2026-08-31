"""The server process must be able to boot — nothing else in this suite checks.

Every HTTP/WS test here drives the app through Starlette's ``TestClient``, which
calls the ASGI callable in-process. That covers routing, schemas and handlers,
but it never touches the thing an operator actually launches::

    venv/…/python -m uvicorn app.main:app --host … --port …  # start_backend.{ps1,bat,sh}
    python -m uvicorn app.main:app --host … --port …         # entrypoint.sh (container)

So the ASGI *server* is untested by construction, and that is not theoretical:
the venv's ``uvicorn`` was deleted outright. A failed in-place upgrade (pip
cannot replace ``uvicorn.exe`` while the dev server holds it) left pip's
rollback staging behind as ``~vicorn`` / ``~vicorn-0.51.0.dist-info``, and the
sweep that cleared the rest of that debris took the live package with it. The
full gate then reported 5028 passed / 0 failed against a venv that could not
start the server at all. The only symptom was the backend refusing to come up.

What this module pins:

* the ASGI server and its protocol dependencies are installed, at the versions
  ``requirements.txt`` pins;
* the target string the launch scripts actually pass — parsed out of them here
  rather than restated — resolves to a loadable ASGI3 app, and to *this* app;
* uvicorn selects a real WebSocket implementation. With neither ``websockets``
  nor ``wsproto`` installed it resolves ``ws="auto"`` to nothing and answers
  every handshake with "Unsupported upgrade request" — the HTTP API stays
  perfectly healthy while progress, logs and job updates go dark;
* nothing else has quietly vanished from the venv.

Deliberately NOT here: binding a port and serving a request. That would run the
full lifespan (task lanes, plugin scan, model registry) and turn an import-level
smoke test into a slow, side-effecting integration test. The regression this
closes is a missing or unloadable server package, which ``Config.load()``
catches without opening a socket.
"""

from __future__ import annotations

import logging
import re
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import pytest
from packaging.requirements import Requirement

# Anchored on ``__file__``, never the CWD: the suite is run both from
# ``backend/`` and from the repo root (``pytest backend``), and a relative path
# would resolve to nothing in one of them — turning every assertion below into
# a vacuous pass rather than a failure.
_BACKEND = Path(__file__).resolve().parents[1]
_REPO = _BACKEND.parent
_REQUIREMENTS = _BACKEND / "requirements.txt"

_LAUNCHERS = (
    _BACKEND / "start_backend.ps1",
    _BACKEND / "start_backend.bat",
    _BACKEND / "start_backend.sh",
    _REPO / "entrypoint.sh",
)

# ``uvicorn app.main:app …`` and ``python -m uvicorn app.main:app …``
_ASGI_TARGET_RE = re.compile(r"\buvicorn\s+(?:-{1,2}\S+\s+)*([\w.]+:[\w.]+)")
_COMMENT_PREFIXES = ("#", "REM ", "rem ", "::")

# The dependency closure of ``python -m uvicorn app.main:app``: the packages
# whose absence either stops the process from starting or silently strips a
# transport out from under it. ``install-deps.sh`` installs every one of them
# verbatim in the container too — unlike the torch trio, which the image
# deliberately pins to a *different* version than requirements.txt documents
# (see the header of backend/install-deps.sh) — so for these the pinned version
# can be asserted and not merely their presence.
_BOOT_CLOSURE = (
    "uvicorn",           # the ASGI server itself
    "click",             # `python -m uvicorn …` argument parsing
    "h11",               # HTTP/1.1 protocol implementation
    "websockets",        # WebSocket protocol implementation
    "starlette",         # ASGI framework under FastAPI
    "fastapi",
    "python-multipart",  # multipart/form-data — main.py's Form() endpoints
    "anyio",
)


def _applicable_requirements() -> dict[str, Requirement]:
    """``requirements.txt`` as name → requirement, minus the inapplicable ones.

    Environment markers are evaluated for the interpreter running the test, so
    ``pywin32``/``colorama`` drop out on Linux and ``triton`` drops out on
    Windows (where ``triton-windows`` takes its place) instead of reading as
    missing.
    """
    reqs: dict[str, Requirement] = {}
    for raw in _REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        req = Requirement(line)
        if req.marker is not None and not req.marker.evaluate():
            continue
        reqs[req.name] = req
    return reqs


def _launch_targets() -> dict[Path, str]:
    """The ASGI target string each launch script hands to uvicorn."""
    targets: dict[Path, str] = {}
    for script in _LAUNCHERS:
        if not script.exists():
            continue
        for raw in script.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith(_COMMENT_PREFIXES):
                continue
            match = _ASGI_TARGET_RE.search(line)
            if match:
                targets[script] = match.group(1)
                break
    return targets


#: Every launcher must reach uvicorn through the INTERPRETER, never through
#: the `uvicorn` console script. Matches `python -m uvicorn`, `python3 -m
#: uvicorn`, `python.exe -m uvicorn` and any absolute/relative path ending in
#: one of those (`venv/bin/python`, `venv\Scripts\python.exe`).
_PYTHON_DASH_M_UVICORN_RE = re.compile(r"\bpython(?:3(?:\.\d+)?|\.exe)?\s+-m\s+uvicorn\b")


def _launch_invocations() -> dict[Path, str]:
    """The one line in each launch script that actually starts uvicorn."""
    invocations: dict[Path, str] = {}
    for script in _LAUNCHERS:
        if not script.exists():
            continue
        for raw in script.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith(_COMMENT_PREFIXES):
                continue
            if _ASGI_TARGET_RE.search(line):
                invocations[script] = line
                break
    return invocations


def _agreed_target() -> str:
    targets = _launch_targets()
    distinct = set(targets.values())
    assert len(distinct) == 1, f"launch scripts disagree on the ASGI target: {targets}"
    return distinct.pop()


def _load_config(target: str):
    """Do what uvicorn does at startup, minus binding a socket.

    ``log_config=None`` on purpose: the default dict-config runs
    ``logging.config.dictConfig`` and re-attaches uvicorn's own stderr handlers
    to the ``uvicorn*`` loggers, which this suite deliberately owns. What is
    under test is that the server can LOAD the app, not that it can format its
    own logs.
    """
    from uvicorn.config import Config

    config = Config(target, host="0.0.0.0", port=8000, log_config=None)
    try:
        config.load()
    except SystemExit as exc:  # uvicorn sys.exit()s on an unimportable app
        pytest.fail(
            f"uvicorn could not load {target!r} (exit code {exc.code}). This is "
            "exactly what an operator sees as 'the backend will not start'."
        )
    return config


@pytest.fixture(autouse=True)
def _keep_uvicorn_loggers_intact():
    """``uvicorn.Config.__init__`` configures logging as a side effect.

    With ``log_config=None`` it only registers the TRACE level name, but the
    ``uvicorn*`` logger levels are load-bearing elsewhere in this suite —
    ``uvicorn.error`` is the logger uvicorn hands to the ``websockets`` protocol
    for frame traces, and its INFO floor is what stops the WebSocket log mirror
    from feeding itself. Snapshot and restore, so this module can never reach a
    later test through global logging state.
    """
    names = ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi")
    saved = [
        (n, lg.level, list(lg.handlers), lg.propagate)
        for n, lg in ((n, logging.getLogger(n)) for n in names)
    ]
    yield
    for name, level, handlers, propagate in saved:
        restored = logging.getLogger(name)
        restored.setLevel(level)
        restored.handlers = handlers
        restored.propagate = propagate


class TestTheAsgiServerIsInstalled:
    def test_uvicorn_is_importable(self):
        """The one import the whole gate was missing."""
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover - the regression itself
            pytest.fail(
                f"the ASGI server is not installed ({exc}). Nothing else in "
                "this suite imports it, so the gate stays green while the "
                "backend cannot start: pip install -r requirements.txt"
            )
        assert uvicorn.__version__

    @pytest.mark.parametrize("package", _BOOT_CLOSURE)
    def test_boot_dependency_is_installed_at_its_pin(self, package):
        """Presence *and* version — a pinned upgrade that never landed counts.

        The uvicorn incident began as a version skew: requirements.txt said
        0.52.0 while the venv still held 0.51.0, because pip could not replace
        the running executable. Nothing failed, so the mismatch persisted until
        the retry that destroyed the package.
        """
        pinned = _applicable_requirements().get(package)
        assert pinned is not None, (
            f"{package} is in the server's boot path but is no longer pinned in "
            f"{_REQUIREMENTS.name} — either restore the pin or drop it here."
        )
        try:
            installed = distribution(package).version
        except PackageNotFoundError:
            pytest.fail(
                f"{package} is pinned as '{pinned}' but is not installed. The "
                "server process needs it to start or to serve a transport; the "
                "in-process TestClient does not, which is why only this test "
                "notices."
            )
        assert pinned.specifier.contains(installed, prereleases=True), (
            f"{package} {installed} is installed but {_REQUIREMENTS.name} pins "
            f"'{pinned.specifier}'. The venv and the pin have drifted apart — "
            "re-run the install rather than editing this test."
        )


class TestBothLaunchFormsResolve:
    """Both ways of reaching uvicorn must work in this venv.

    CORRECTION (2026-08-31, LANE-48): this class used to say
    "``start_backend.*`` runs the console script". That stopped being true at
    ``6e65e390`` — all four launchers now go through ``python -m uvicorn``
    (see ``TestTheLaunchersAndTheAppAgree::
    test_every_launcher_invokes_uvicorn_through_the_interpreter``, which
    enforces it rather than describing it). The console-script assertions
    below are kept as **venv-integrity** checks, not launcher checks: a
    missing ``uvicorn`` console script is the fingerprint of the failed
    in-place upgrade described in this module's docstring, and it is cheaper
    to notice here than when the server will not come up.
    """

    def test_the_console_script_is_declared_and_loadable(self):
        entry_points = [
            ep
            for ep in distribution("uvicorn").entry_points
            if ep.group == "console_scripts" and ep.name == "uvicorn"
        ]
        assert entry_points, "uvicorn declares no `uvicorn` console script"
        assert callable(entry_points[0].load())

    def test_the_console_script_exists_for_this_interpreter(self):
        """A venv-integrity check, not a launcher check (see the class
        docstring: no launcher calls the console script any more). If the
        venv's launcher is
        gone but a *global* uvicorn is installed — the state a failed in-place
        upgrade leaves behind, since pip cannot overwrite a running
        ``uvicorn.exe`` — PATH silently falls through to the global one, which
        runs against an interpreter that has none of this project's packages.

        Checked against ``sysconfig``'s script directory rather than PATH:
        pytest is invoked as ``venv\\Scripts\\python.exe -m pytest`` without
        activating the venv, so PATH here is not the PATH the launcher gets.
        This asserts about the interpreter running the tests, which is the venv
        the launcher activates.
        """
        import sysconfig

        scripts_dir = Path(sysconfig.get_path("scripts"))
        launchers = [p for p in scripts_dir.glob("uvicorn*") if p.stem == "uvicorn"]
        assert launchers, (
            f"no `uvicorn` launcher in {scripts_dir}. The package can be "
            "installed and importable while its console script is missing — "
            "and then `uvicorn app.main:app` either fails outright or, worse, "
            "silently runs a different interpreter's uvicorn."
        )

    def test_python_dash_m_uvicorn_resolves(self):
        """The container's launch form: ``python -m uvicorn app.main:app``."""
        import importlib.util

        try:
            spec = importlib.util.find_spec("uvicorn.__main__")
        except ModuleNotFoundError as exc:
            pytest.fail(f"`python -m uvicorn` cannot resolve: {exc}")
        assert spec is not None, "uvicorn ships no __main__ module"


class TestTheLaunchersAndTheAppAgree:
    def test_the_launch_scripts_name_one_asgi_target(self):
        """Parsed, not restated: a rename that updates every launcher passes,
        a rename that misses one fails — which is the interesting case."""
        targets = _launch_targets()
        # The three backend launchers are tracked; entrypoint.sh is too, but it
        # is optional here so a partial checkout degrades instead of lying.
        assert len(targets) >= 3, (
            f"only parsed {len(targets)} launch script(s) out of "
            f"{len(_LAUNCHERS)} — the regex or the scripts moved, and this "
            "module would otherwise be asserting nothing at all."
        )
        assert len(set(targets.values())) == 1, (
            f"launch scripts disagree on the ASGI target: {targets}. One of "
            "them starts a different app than the others."
        )

    def test_every_launcher_invokes_uvicorn_through_the_interpreter(self):
        """``python -m uvicorn``, never the bare ``uvicorn`` console script.

        ``6e65e390`` fixed ``start_backend.sh`` calling the console-script
        shim, whose shebang embeds the ABSOLUTE path of the interpreter that
        created the venv: copy, move or rename the checkout and it dies —
        quietly, with an error that names a path nobody recognises. All four
        launchers now carry a comment saying why they use ``-m``, but
        ``_ASGI_TARGET_RE`` accepts BOTH forms, so until this test only the
        prose enforced it and the shim could return with a green gate
        (LESSONS 2026-08-14).

        A positive assertion on the launchers' own bytes, so it needs no
        offender-scan control (CONVENTIONS rule 11) — but it does need to
        prove it read something, hence the count check: a moved or renamed
        launcher must fail here rather than silently shrink the set.
        """
        invocations = _launch_invocations()
        present = [script for script in _LAUNCHERS if script.exists()]

        assert len(invocations) >= 3, (
            f"only parsed {len(invocations)} uvicorn invocation(s) out of "
            f"{len(_LAUNCHERS)} launcher(s) — the scripts or _ASGI_TARGET_RE "
            "moved, and this test would otherwise be asserting nothing."
        )
        assert len(invocations) == len(present), (
            "a launcher exists on disk but no uvicorn invocation was parsed "
            f"from it: {sorted(str(p) for p in set(present) - set(invocations))}"
        )

        for script, line in sorted(invocations.items()):
            assert _PYTHON_DASH_M_UVICORN_RE.search(line), (
                f"{script.name} starts uvicorn as `{line}`. It must invoke it "
                "through the interpreter (`python -m uvicorn`): the `uvicorn` "
                "console script hard-codes the absolute path of the "
                "interpreter that built the venv, so a copied, moved or "
                "renamed checkout fails to start with an error that points "
                "at a path that no longer exists."
            )

    def test_the_target_resolves_to_this_app(self):
        """uvicorn's own import path, on the exact string the scripts pass."""
        config = _load_config(_agreed_target())
        assert config.loaded_app is not None

        # Unwrap uvicorn's middleware to prove the target points at *our* app
        # and not merely at something importable with that name.
        from app.main import app as application

        inner = config.loaded_app
        for _ in range(10):
            if not hasattr(inner, "app"):
                break
            inner = inner.app
        assert inner is application, (
            f"{_agreed_target()!r} loaded {type(inner).__name__}, not the "
            "FastAPI application from app.main."
        )

    def test_the_app_is_detected_as_asgi3(self):
        """An ASGI2 misdetection wraps the app in a compatibility shim and
        breaks every request; uvicorn decides this at load time, silently."""
        assert _load_config(_agreed_target()).interface == "asgi3"


class TestTheServerCanSpeakEveryTransportTheAppUses:
    def test_a_real_websocket_implementation_is_selected(self):
        config = _load_config(_agreed_target())
        assert config.ws_protocol_class is not None, (
            "uvicorn resolved ws='auto' to no implementation, meaning neither "
            "`websockets` nor `wsproto` is installed. The server still starts "
            "and every HTTP route works — it just answers each WebSocket "
            "handshake with 'Unsupported upgrade request', so training "
            "progress, log streaming and job updates go dark with nothing in "
            "the gate or in server.log to explain it."
        )

    def test_an_http_implementation_is_selected(self):
        http_protocol = _load_config(_agreed_target()).http_protocol_class
        assert isinstance(http_protocol, type), (
            f"uvicorn resolved http='auto' to {http_protocol!r}; without an "
            "HTTP protocol class the server cannot answer a single request."
        )

    def test_the_lifespan_runs(self):
        """The startup hook injects the running event loop into the managers
        that schedule work from threads. With lifespan off the app serves
        requests but every cross-thread notification is dropped."""
        from uvicorn.lifespan.off import LifespanOff

        assert _load_config(_agreed_target()).lifespan_class is not LifespanOff


class TestTheVenvStillSatisfiesRequirements:
    def test_every_applicable_requirement_is_installed(self):
        """Presence only — deliberately not versions.

        This is the general form of the uvicorn regression: a package can
        disappear from the venv and every test still passes, because most of
        them only need the handful of libraries their own imports reach.

        Versions are checked for the boot closure above but not here, because
        the container legitimately runs a different torch trio than
        requirements.txt pins (the image bakes 2.11, the local venv runs 2.12.1;
        install-deps.sh filters those lines out of both installs). Asserting
        versions across the whole file would make this test wrong inside the
        image it is shipped in.
        """
        reqs = _applicable_requirements()
        assert len(reqs) > 50, (
            f"only parsed {len(reqs)} requirement(s) from {_REQUIREMENTS} — "
            "the file moved or the parse broke, and this test is asserting "
            "nothing."
        )

        missing = []
        for name in sorted(reqs):
            try:
                distribution(name)
            except PackageNotFoundError:
                missing.append(name)

        assert not missing, (
            f"{len(missing)} pinned package(s) are not installed in this venv: "
            f"{', '.join(missing)}. Re-run backend/install-deps.sh (or pip "
            "install -r requirements.txt) — the suite can pass without them "
            "right up until the code path that needs them runs."
        )
