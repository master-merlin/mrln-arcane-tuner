"""SelfUpdateService._wait_for_idle — bounded drain (W4.T5).

Before this, the PENDING_RESTART drain polled active_task_count() with no
timeout: one worker permanently stuck RUNNING (e.g. a GPU stall) kept the
service draining forever, rejecting new GPU tasks with no operator escape
hatch.

Also here: the frontend rename-swap, and install-deps.sh keeping the installed
torch stack (the last section).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

from app.core.self_update import SelfUpdateService
from tests.support.bash_probe import bash_skip_reason, find_bash


def _make_service() -> SelfUpdateService:
    return SelfUpdateService(app_dir=".", branch="main", remote="")


@pytest.mark.asyncio
async def test_wait_for_idle_returns_true_when_idle_immediately():
    svc = _make_service()
    svc.set_loop(None)  # no loop → _broadcast is a no-op
    assert await svc._wait_for_idle(timeout_s=1.0) is True


@pytest.mark.asyncio
async def test_wait_for_idle_times_out_on_permanently_running_task(monkeypatch):
    svc = _make_service()
    svc.set_loop(None)
    monkeypatch.setattr(svc, "active_task_count", lambda: 1)  # forever busy
    result = await svc._wait_for_idle(timeout_s=0.2)
    assert result is False


@pytest.mark.asyncio
async def test_apply_impl_aborts_drain_on_timeout(monkeypatch):
    """The self-update caller must abort PENDING_RESTART (logging
    self_update_drain_timeout) and re-enable the GPU lane instead of hanging
    or restarting with an in-flight task still running."""
    from app.core.self_update import UpdateState

    svc = _make_service()
    svc.set_loop(None)

    monkeypatch.setattr(svc, "_pull", lambda: True)
    monkeypatch.setattr(svc, "_req_blob", lambda: "same")

    async def _noop_install():
        return None

    async def _noop_build():
        return None

    async def _fake_wait_for_idle():
        return False  # drain never completed — simulates a stuck task

    monkeypatch.setattr(svc, "_install_backend_deps", _noop_install)
    monkeypatch.setattr(svc, "_build_frontend", _noop_build)
    monkeypatch.setattr(svc, "_wait_for_idle", _fake_wait_for_idle)

    restarted = {"called": False}

    async def _fake_restart():
        restarted["called"] = True

    monkeypatch.setattr(svc, "_do_restart", _fake_restart)

    drain_events: list[dict] = []
    monkeypatch.setattr(
        "app.core.self_update.logger.error",
        lambda event, **kw: drain_events.append({"event": event, **kw}),
    )

    draining_calls: list[bool] = []
    monkeypatch.setattr(
        "app.core.self_update.set_draining", lambda v: draining_calls.append(v)
    )

    await svc._apply_impl()

    assert restarted["called"] is False
    assert svc.state == UpdateState.ERROR
    assert any(e["event"] == "self_update_drain_timeout" for e in drain_events)
    # Drain was activated then explicitly released (not left stuck True).
    assert draining_calls == [True, False]


# ── _build_frontend: rename-swap (W5.T10) ─────────────────────────────────
#
# The served frontend dir used to be replaced via rmtree(served) THEN
# copytree(built, served) — a copytree failure (disk full, permission error,
# a crash mid-copy) left `served` either missing entirely or half-written,
# with every page load 404ing/half-loading until the next successful update.
# Building into a sibling `.new` dir first means a copy failure never
# touches the live `served` dir at all; the swap itself is then just two
# near-instant directory renames.


def _write(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


@pytest.mark.asyncio
async def test_build_frontend_swaps_in_new_build(tmp_path, monkeypatch):
    fe = tmp_path / "frontend"
    _write(fe / "dist" / "frontend" / "browser" / "index.html", "NEW BUILD")
    _write(fe / "browser" / "index.html", "OLD BUILD")

    monkeypatch.setattr(
        "app.core.self_update.subprocess.run",
        lambda *a, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    svc = SelfUpdateService(app_dir=str(tmp_path), branch="main", remote="")
    await svc._build_frontend()

    served = fe / "browser"
    assert served.is_dir()
    assert (served / "index.html").read_text() == "NEW BUILD"
    # Temp swap dirs are cleaned up — nothing left behind.
    assert not (fe / "browser.new").exists()
    assert not (fe / "browser.old").exists()


@pytest.mark.asyncio
async def test_build_frontend_copy_failure_leaves_served_dir_untouched(
    tmp_path, monkeypatch
):
    """A copytree failure while staging the new build must NEVER touch the
    live `served` dir — the whole point of building into `.new` first."""
    fe = tmp_path / "frontend"
    _write(fe / "dist" / "frontend" / "browser" / "index.html", "NEW BUILD")
    _write(fe / "browser" / "index.html", "OLD BUILD")

    monkeypatch.setattr(
        "app.core.self_update.subprocess.run",
        lambda *a, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    import shutil

    def _boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copytree", _boom)

    svc = SelfUpdateService(app_dir=str(tmp_path), branch="main", remote="")
    with pytest.raises(OSError, match="disk full"):
        await svc._build_frontend()

    served = fe / "browser"
    assert served.is_dir()
    assert (served / "index.html").read_text() == "OLD BUILD"


# -- _build_frontend: the served dir may be UNRENAMEABLE (found in UAT) -------
#
# The first real self-update from a running container failed with
#
#   self_update_failed  [Errno 18] Invalid cross-device link:
#   '/app/frontend/browser' -> '/app/frontend/browser.old'
#
# Source and target are the same directory, so "cross-device" reads as
# nonsense until you remember what a container's filesystem is. `served` is
# baked into the image by `COPY --from=frontend` and never written afterwards,
# so it still lives in a LOWER overlayfs layer, and overlayfs cannot move a
# whole directory up into the writable layer unless the kernel's
# `redirect_dir` is enabled -- which it is not, by default. Measured in a
# container on 2026-09-06, with a control, because the interesting part is
# the second row:
#
#   os.rename   a directory that came from the image  -> errno 18 EXDEV
#   os.rename   a directory created at runtime        -> OK
#   shutil.move a directory that came from the image  -> OK, tree intact
#
# So it is not "containers cannot rename directories". It is *this* directory,
# *once*: after one successful update `browser` is a runtime directory and
# renames fine, which is why the defect survived every place it is cheap to
# look -- a plain filesystem on Windows, CI, and any second run.


def _explodes_on(path: str, real_rename):
    """os.rename that refuses exactly one path, the way overlayfs does."""

    def _rename(src, dst, *a, **kw):
        if str(src) == path:
            raise OSError(18, "Invalid cross-device link")
        return real_rename(src, dst, *a, **kw)

    return _rename


@pytest.mark.asyncio
async def test_build_frontend_swaps_in_the_new_build_when_served_cannot_be_renamed(
    tmp_path, monkeypatch
):
    fe = tmp_path / "frontend"
    _write(fe / "dist" / "frontend" / "browser" / "index.html", "NEW BUILD")
    _write(fe / "browser" / "index.html", "OLD BUILD")

    monkeypatch.setattr(
        "app.core.self_update.subprocess.run",
        lambda *a, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    # Patched on the `os` module itself, so `shutil.move`'s own internal
    # os.rename hits it too -- that is the whole point: the fix works because
    # shutil.move CATCHES this and copies instead.
    monkeypatch.setattr(os, "rename", _explodes_on(str(fe / "browser"), os.rename))

    svc = SelfUpdateService(app_dir=str(tmp_path), branch="main", remote="")
    await svc._build_frontend()

    served = fe / "browser"
    assert served.is_dir()
    assert (served / "index.html").read_text() == "NEW BUILD"
    assert not (fe / "browser.new").exists()
    assert not (fe / "browser.old").exists()


@pytest.mark.asyncio
async def test_build_frontend_keeps_serving_the_old_build_when_the_fallback_copy_fails(
    tmp_path, monkeypatch
):
    """The `.new` staging exists so a failed swap never breaks the live site.

    That property has to survive on the fallback path too, and it is not
    self-evident there: shutil.move copies the old tree aside and only THEN
    deletes it, so a failure during that copy must leave `served` whole. Pinned
    because the alternative reading -- delete first, copy after -- is exactly
    the bug the rename-swap replaced in W5.T10.
    """
    fe = tmp_path / "frontend"
    _write(fe / "dist" / "frontend" / "browser" / "index.html", "NEW BUILD")
    _write(fe / "browser" / "index.html", "OLD BUILD")

    monkeypatch.setattr(
        "app.core.self_update.subprocess.run",
        lambda *a, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(os, "rename", _explodes_on(str(fe / "browser"), os.rename))

    import shutil

    real_copytree = shutil.copytree
    calls = {"n": 0}

    def _copytree(*a, **kw):
        # 1st call stages the new build; the 2nd is shutil.move's fallback,
        # copying the old build aside. Fail that one.
        calls["n"] += 1
        if calls["n"] >= 2:
            raise OSError("disk full")
        return real_copytree(*a, **kw)

    monkeypatch.setattr(shutil, "copytree", _copytree)

    svc = SelfUpdateService(app_dir=str(tmp_path), branch="main", remote="")
    with pytest.raises(OSError, match="disk full"):
        await svc._build_frontend()

    served = fe / "browser"
    assert served.is_dir()
    assert (served / "index.html").read_text() == "OLD BUILD"


# -- install-deps.sh keeps the installed torch stack (offline, a throwaway venv) --
#
# A self-update runs install-deps.sh inside a running container, whose torch
# stack was baked into the image. The script filters the stack's own pins out of
# requirements.txt, but a NON-stack package can still require another torch in
# its metadata, and pip would then replace the baked stack to satisfy it. So the
# script installs under constraints written from the installed stack: such a
# package fails pip's resolution, which installs nothing, and the script names
# the image to pull.
#
# Everything here is offline: tiny wheels written by hand into a tmp dir, served
# with PIP_NO_INDEX=1 and PIP_FIND_LINKS. The fixture package requires
# torch==9.9.0 and a stub torch 9.9.0 is offered too, so the only thing that can
# stop pip is the constraint. The control run, the same script without a
# constraint, proves the fixture really reaches for the other torch.

BACKEND = Path(__file__).resolve().parents[1]
INSTALL_DEPS = BACKEND / "install-deps.sh"
IMAGE = "mastermerlin/mrln-arcane-tuner"
FIXTURE = "stackneedy"
STACK = ("torch", "torchvision", "torchaudio", "torchao")
_TIMEOUT_S = 600

# One worker holds the sandbox, so the control (which replaces its torch) runs after the rest.
in_sandbox = pytest.mark.xdist_group("install_deps_sandbox")

requires_bash = pytest.mark.skipif(
    bash_skip_reason(INSTALL_DEPS) is not None,
    reason=bash_skip_reason(INSTALL_DEPS) or "",
)


def _cu132() -> dict[str, str]:
    """The default set's stack, each with its local version, read from torch-stack.cfg."""
    spec = importlib.util.spec_from_file_location("torch_stack_for_tests", BACKEND / "torch_stack.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up there
    spec.loader.exec_module(module)
    _default, sets = module.read_stack()
    s = sets["cu132"]
    return {name: f"{getattr(s, name)}+cu132" for name in STACK}


def _wheel(out: Path, name: str, version: str, requires: tuple[str, ...] = ()) -> Path:
    """A minimal pure wheel: METADATA, WHEEL and RECORD, nothing else."""
    dist = f"{name}-{version}.dist-info"
    path = out / f"{name}-{version}-py3-none-any.whl"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n" + "".join(
        f"Requires-Dist: {r}\n" for r in requires
    )
    wheel = "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    record = f"{dist}/METADATA,,\n{dist}/WHEEL,,\n{dist}/RECORD,,\n"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{dist}/METADATA", metadata)
        z.writestr(f"{dist}/WHEEL", wheel)
        z.writestr(f"{dist}/RECORD", record)
    return path


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")


def _bash_tool_dirs(bash: str) -> list[str]:
    """The coreutils next to this bash: a non-login Git Bash does not put them on PATH."""
    here = Path(bash).resolve().parent
    return [str(d) for d in (here, here.parent / "usr" / "bin") if (d / "grep.exe").is_file() or (d / "grep").is_file()]


def _installed(python: Path) -> dict[str, str]:
    done = subprocess.run(
        [str(python), "-m", "pip", "list", "--format=json", "--disable-pip-version-check"],
        capture_output=True, text=True, timeout=_TIMEOUT_S, check=True,
    )
    return {row["name"].lower(): row["version"] for row in json.loads(done.stdout)}


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    """A venv holding the stub cu132 stack, and a find-links dir of every wheel."""
    root = tmp_path_factory.mktemp("install_deps")
    wheels = root / "wheels"
    wheels.mkdir()
    stack = _cu132()
    for name, version in stack.items():
        _wheel(wheels, name, version)
    _wheel(wheels, "torch", "9.9.0")
    _wheel(wheels, FIXTURE, "1.0", requires=("torch==9.9.0",))

    venv = root / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, timeout=_TIMEOUT_S)
    python = _venv_python(venv)
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-index", "--find-links", str(wheels),
         "--disable-pip-version-check", *(f"{n}=={v}" for n, v in stack.items())],
        check=True, capture_output=True, text=True, timeout=_TIMEOUT_S,
    )
    # requirements.txt as the product writes it: the stack's own pins (filtered by the
    # script) and one non-stack package that needs another torch.
    req = root / "requirements.txt"
    req.write_text(
        f"torch=={stack['torch'].split('+')[0]}\n"
        f"torchvision=={stack['torchvision'].split('+')[0]}\n"
        f"{FIXTURE}==1.0\n",
        encoding="utf-8",
    )
    return {"root": root, "wheels": wheels, "venv": venv, "python": python, "req": req, "stack": stack}


def _run(sandbox, script: Path, *, torch_set: str | None = None,
         extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    bash = find_bash(script)
    assert bash, bash_skip_reason(script)
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "MRLN_TORCH_SET", "PIP_LOG")}
    scripts = str(sandbox["python"].parent)
    env["PATH"] = os.pathsep.join([scripts, *_bash_tool_dirs(bash), env.get("PATH", "")])
    env["VIRTUAL_ENV"] = str(sandbox["venv"])
    env["PIP_NO_INDEX"] = "1"
    env["PIP_FIND_LINKS"] = str(sandbox["wheels"])
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    tmp = sandbox["root"] / "tmp"
    tmp.mkdir(exist_ok=True)
    env["TMPDIR"] = str(tmp)
    if torch_set:
        env["MRLN_TORCH_SET"] = torch_set
    env.update(extra_env or {})
    return subprocess.run(
        [bash, str(script), str(sandbox["req"])],
        cwd=sandbox["root"], env=env, capture_output=True, text=True, timeout=_TIMEOUT_S,
    )


@in_sandbox
@requires_bash
def test_install_deps_a_package_needing_another_torch_fails_and_leaves_the_stack(sandbox):
    before = _installed(sandbox["python"])
    proc = _run(sandbox, INSTALL_DEPS)
    after = _installed(sandbox["python"])

    assert proc.returncode != 0, f"install-deps.sh succeeded:\n{proc.stdout}\n{proc.stderr}"
    assert re.search(rf"{re.escape(IMAGE)}:latest(?![-\w])", proc.stderr), (
        f"the failure does not name the image to pull:\n{proc.stderr}"
    )
    for name, version in sandbox["stack"].items():
        assert after.get(name) == version == before.get(name), f"{name}: {before.get(name)} -> {after.get(name)}"
    assert FIXTURE not in after, f"the fixture was installed: {after}"


@in_sandbox
@requires_bash
def test_install_deps_a_non_default_set_names_its_own_image(sandbox):
    proc = _run(sandbox, INSTALL_DEPS, torch_set="cu126")
    assert proc.returncode != 0
    assert f"{IMAGE}:latest-cu126" in proc.stderr, proc.stderr
    assert _installed(sandbox["python"])["torch"] == sandbox["stack"]["torch"]


def _with_probe(tmp_path: Path, probe: str) -> Path:
    """install-deps.sh beside a stand-in torch_stack.py whose `constraints` runs ``probe``."""
    d = tmp_path / "probe"
    d.mkdir()
    shutil.copy(INSTALL_DEPS, d / "install-deps.sh")
    (d / "torch_stack.py").write_text(probe, encoding="utf-8")
    return d / "install-deps.sh"


def _no_torch_code() -> int:
    spec = importlib.util.spec_from_file_location("torch_stack_codes", BACKEND / "torch_stack.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.CONSTRAINTS_NO_TORCH


# Each a probe that did NOT confirm the installed stack. Exit 1 is what any unhandled exception
# gives, so it can never mean "no torch": that has a code of its own.
_PROBE_FAILURES = {
    "raises": "raise RuntimeError('importlib.metadata blew up')\n",
    "exit 1": "import sys\nsys.exit(1)\n",
    "exit 2": "import sys\nsys.exit(2)\n",
    "says 0 but writes nothing": "import sys\nsys.exit(0)\n",
    "writes no torch pin": "import sys\nopen(sys.argv[-1], 'w').write('# nothing\\n')\nsys.exit(0)\n",
}


@in_sandbox
@requires_bash
@pytest.mark.parametrize("failure", list(_PROBE_FAILURES))
def test_install_deps_never_runs_pip_after_a_failed_stack_probe(sandbox, tmp_path, failure):
    """A probe that cannot say what is installed must stop the script: unconstrained, the
    fixture would replace the stack (the control below proves it would)."""
    script = _with_probe(tmp_path, _PROBE_FAILURES[failure])
    pip_log = tmp_path / "pip.log"
    before = _installed(sandbox["python"])
    proc = _run(sandbox, script, extra_env={"PIP_LOG": str(pip_log)})
    after = _installed(sandbox["python"])

    assert proc.returncode != 0, f"install-deps.sh went on after the probe failed:\n{proc.stderr}"
    assert not pip_log.exists(), f"pip ran after the probe failed:\n{proc.stderr}"
    assert after == before
    assert "torch stack" in proc.stderr, proc.stderr


@in_sandbox
@requires_bash
def test_install_deps_in_an_image_requires_the_constraints(sandbox, tmp_path):
    """An image always has its baked stack, so "no torch" there is a fault, never a licence."""
    code = _no_torch_code()
    script = _with_probe(tmp_path, f"import sys\nsys.exit({code})\n")
    pip_log = tmp_path / "pip.log"
    before = _installed(sandbox["python"])
    proc = _run(sandbox, script, torch_set="cu132", extra_env={"PIP_LOG": str(pip_log)})

    assert proc.returncode != 0, proc.stderr
    assert not pip_log.exists(), f"pip ran in an image without its constraints:\n{proc.stderr}"
    assert _installed(sandbox["python"]) == before


@in_sandbox
@requires_bash
def test_install_deps_control_without_the_constraint_the_fixture_replaces_torch(sandbox, tmp_path):
    """The same script, where no torch is confirmed (constraints' own no-torch code), outside an
    image: the unconstrained install, as it always was.

    Runs last in this module: it is the run that breaks the sandbox's stack.
    """
    ctl = tmp_path / "ctl"
    ctl.mkdir()
    shutil.copy(INSTALL_DEPS, ctl / "install-deps.sh")
    (ctl / "torch_stack.py").write_text(
        "import sys\nprint('[torch] no torch is installed: no constraints written', file=sys.stderr)\n"
        f"sys.exit({_no_torch_code()})\n",
        encoding="utf-8",
    )
    proc = _run(sandbox, ctl / "install-deps.sh")
    after = _installed(sandbox["python"])
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert after.get("torch") == "9.9.0", after
    assert after.get(FIXTURE) == "1.0", after
