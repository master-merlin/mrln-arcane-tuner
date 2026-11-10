import asyncio
import os
import subprocess

import pytest

from app.core.self_update import SelfUpdateService, UpdateState


@pytest.fixture
def svc():
    return SelfUpdateService(app_dir="/fake/app", branch="main", remote="https://example/repo.git")


def test_initial_state_is_idle(svc):
    assert svc.state == UpdateState.IDLE


def test_git_status_parses_branch_commit_dirty(svc, monkeypatch):
    def fake_run(args, cwd=None, timeout=None):
        if args[:2] == ["rev-parse", "--abbrev-ref"]:
            return (0, "main\n", "")
        if args[:2] == ["rev-parse", "--short"]:
            return (0, "a1b3c2d\n", "")
        if args[0] == "status":
            return (0, " M backend/app/x.py\n", "")
        return (0, "", "")

    monkeypatch.setattr(svc, "_run_git", fake_run)
    st = svc.git_status()
    assert st["branch"] == "main"
    assert st["commit"] == "a1b3c2d"
    assert st["dirty"] is True
    assert st["is_repo"] is True


def test_git_status_not_a_repo(svc, monkeypatch):
    monkeypatch.setattr(svc, "_run_git", lambda *a, **k: (128, "", "not a git repository"))
    st = svc.git_status()
    assert st["is_repo"] is False


def test_probe_availability_true_when_lsremote_ok(svc, monkeypatch):
    monkeypatch.setattr(svc, "_run_git", lambda *a, **k: (0, "ref\tHEAD\n", ""))
    svc.probe_availability()
    assert svc.available is True


def test_probe_availability_false_when_lsremote_fails(svc, monkeypatch):
    monkeypatch.setattr(svc, "_run_git", lambda *a, **k: (128, "", "Authentication failed"))
    svc.probe_availability()
    assert svc.available is False


def test_active_task_count_counts_running_tasks_only(svc, monkeypatch):
    from app.core import self_update as mod
    from app.core.tasks.task import TaskStatus

    class _Task:
        def __init__(self, status):
            self.status = status

    monkeypatch.setattr(mod, "_list_tasks",
                        lambda: [_Task(TaskStatus.RUNNING), _Task(TaskStatus.PENDING),
                                 _Task(TaskStatus.COMPLETED), _Task(TaskStatus.RUNNING)])
    assert svc.active_task_count() == 2


@pytest.mark.asyncio
async def test_check_reports_behind_count(svc, monkeypatch):
    def fake_run(args, cwd=None, timeout=None):
        if args[0] == "fetch":
            return (0, "", "")
        if args[:2] == ["rev-list", "--count"]:
            return (0, "3\n", "")
        if args[0] == "log":
            return (0, "fix a\nfix b\nfix c\n", "")
        return (0, "", "")

    monkeypatch.setattr(svc, "_run_git", fake_run)
    result = await svc.check()
    assert result["behind"] == 3
    assert result["commits"] == ["fix a", "fix b", "fix c"]
    assert svc.behind == 3


@pytest.mark.asyncio
async def test_apply_runs_pull_build_then_drains_and_restarts(svc, monkeypatch):
    order = []

    def fake_run(args, cwd=None, timeout=None):
        order.append(args[0])
        return (0, "", "")

    async def fake_build():
        order.append("build")

    restarted = {"v": False}

    async def fake_restart():
        order.append("restart")
        restarted["v"] = True

    monkeypatch.setattr(svc, "_run_git", fake_run)
    monkeypatch.setattr(svc, "_build_frontend", fake_build)
    monkeypatch.setattr(svc, "_do_restart", fake_restart)
    monkeypatch.setattr(svc, "active_task_count", lambda: 0)

    await svc._apply_impl()

    assert "fetch" in order and "reset" in order
    assert order.index("build") < order.index("restart")
    assert restarted["v"] is True
    from app.core.drain import is_draining
    assert is_draining() is False


@pytest.mark.asyncio
async def test_apply_reinstalls_deps_when_requirements_changed(svc, monkeypatch):
    # requirements blob differs before vs after the pull → deps reinstalled.
    blobs = iter(["oldsha", "newsha"])
    monkeypatch.setattr(svc, "_req_blob", lambda: next(blobs))
    monkeypatch.setattr(svc, "_run_git", lambda *a, **k: (0, "", ""))

    installed = {"v": False}

    async def fake_install():
        installed["v"] = True

    async def fake_build():
        pass

    async def fake_restart():
        pass

    monkeypatch.setattr(svc, "_install_backend_deps", fake_install)
    monkeypatch.setattr(svc, "_build_frontend", fake_build)
    monkeypatch.setattr(svc, "_do_restart", fake_restart)
    monkeypatch.setattr(svc, "active_task_count", lambda: 0)

    await svc._apply_impl()
    assert installed["v"] is True


@pytest.mark.asyncio
async def test_apply_skips_deps_when_requirements_unchanged(svc, monkeypatch):
    # Same blob before/after → no pip reinstall (fast path).
    monkeypatch.setattr(svc, "_req_blob", lambda: "samesha")
    monkeypatch.setattr(svc, "_run_git", lambda *a, **k: (0, "", ""))

    installed = {"v": False}

    async def fake_install():
        installed["v"] = True

    async def fake_build():
        pass

    async def fake_restart():
        pass

    monkeypatch.setattr(svc, "_install_backend_deps", fake_install)
    monkeypatch.setattr(svc, "_build_frontend", fake_build)
    monkeypatch.setattr(svc, "_do_restart", fake_restart)
    monkeypatch.setattr(svc, "active_task_count", lambda: 0)

    await svc._apply_impl()
    assert installed["v"] is False


@pytest.mark.asyncio
async def test_apply_error_on_pull_failure_no_restart(svc, monkeypatch):
    monkeypatch.setattr(svc, "_run_git", lambda *a, **k: (1, "", "network down"))
    called = {"restart": False}

    async def fake_restart():
        called["restart"] = True

    monkeypatch.setattr(svc, "_do_restart", fake_restart)
    await svc._apply_impl()
    assert svc.state == UpdateState.ERROR
    assert called["restart"] is False


@pytest.mark.asyncio
async def test_check_once_safe_swallows_errors(svc, monkeypatch):
    async def boom():
        raise RuntimeError("network")

    monkeypatch.setattr(svc, "check", boom)
    # Must not raise — periodic loop relies on this.
    await svc.check_once_safe()


@pytest.mark.asyncio
async def test_check_once_safe_runs_check(svc, monkeypatch):
    called = {"v": False}

    async def ok():
        called["v"] = True
        return {"behind": 0, "commits": []}

    monkeypatch.setattr(svc, "check", ok)
    await svc.check_once_safe()
    assert called["v"] is True


@pytest.mark.asyncio
async def test_install_backend_deps_uses_script_when_present(svc, monkeypatch, tmp_path):
    backend = tmp_path / "backend"
    backend.mkdir()
    await asyncio.to_thread((backend / "install-deps.sh").write_text, "#!/usr/bin/env bash\n")
    svc.app_dir = str(tmp_path)

    captured = {}

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    await svc._install_backend_deps()

    assert captured["cmd"] == ["bash", str(backend / "install-deps.sh")]


@pytest.mark.asyncio
async def test_install_backend_deps_fallback_filters_torch_stack(svc, monkeypatch, tmp_path):
    # install-deps.sh absent → the fallback must still exclude the
    # torch-stack lines (torch/torchvision/torchaudio/torchao/triton/
    # triton-windows), matching install-deps.sh's own exclusion — otherwise a
    # plain `pip install -r requirements.txt` would clobber the stack baked into
    # the image's cached Dockerfile layer.
    backend = tmp_path / "backend"
    backend.mkdir()
    await asyncio.to_thread(
        (backend / "requirements.txt").write_text,
        "fastapi==1.0\n"
        "torch==2.15.0\n"
        "torchao==0.18.0\n"
        "torchaudio==2.11.1  # the stack's: declares torch>=2.11\n"
        "torchvision==0.29.2\n"
        "triton-windows==3.9.0.post30; sys_platform == 'win32'\n"
        "triton==3.9.0; sys_platform == 'linux'\n"
        "pydantic==2.0\n",
    )
    svc.app_dir = str(tmp_path)

    captured = {}

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        captured["cmd"] = cmd
        with open(cmd[-1], encoding="utf-8") as f:
            captured["filtered"] = f.read()
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    await svc._install_backend_deps()

    assert captured["cmd"][:4] == [
        "python", "-m", "pip", "install",
    ]
    assert "--break-system-packages" in captured["cmd"]
    filtered = captured["filtered"]
    assert "fastapi==1.0" in filtered
    assert "pydantic==2.0" in filtered
    for pkg_line in ("torch==", "torchao==", "torchvision==", "torchaudio==", "triton-windows==", "triton=="):
        assert pkg_line not in filtered
    # the filtered temp file must not survive the call
    assert not os.path.exists(captured["cmd"][-1])


def _git(cwd, *args):
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
        cwd=cwd, capture_output=True, text=True, timeout=60, check=True,
    )
    return done.stdout.strip()


def _commit_requirements(repo, text, message):
    req = repo / "backend" / "requirements.txt"
    req.parent.mkdir(exist_ok=True)
    req.write_text(text, encoding="utf-8")
    _git(repo, "add", "backend/requirements.txt")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.mark.asyncio
async def test_failed_dep_install_resets_the_checkout_to_the_old_head(monkeypatch, tmp_path):
    """New code on old dependencies is what a failed install would leave; the update undoes the pull."""

    def setup():
        upstream = tmp_path / "upstream"
        upstream.mkdir()
        _git(upstream, "init", "-q", "-b", "main")
        old = _commit_requirements(upstream, "fastapi==1.0\n", "old")
        _git(tmp_path, "clone", "-q", str(upstream), "app")
        new = _commit_requirements(upstream, "fastapi==1.0\nstackneedy==1.0\n", "new")
        return old, new

    old, new = await asyncio.to_thread(setup)
    app = tmp_path / "app"
    svc = SelfUpdateService(app_dir=str(app), branch="main", remote=str(tmp_path / "upstream"))

    async def failing_install():
        raise RuntimeError("backend dep install failed: ResolutionImpossible")

    async def must_not_run():
        raise AssertionError("ran after a failed dependency install")

    monkeypatch.setattr(svc, "_install_backend_deps", failing_install)
    monkeypatch.setattr(svc, "_build_frontend", must_not_run)
    monkeypatch.setattr(svc, "_do_restart", must_not_run)

    await svc._apply_impl()

    head = await asyncio.to_thread(_git, app, "rev-parse", "HEAD")
    assert head == old != new
    assert svc.state == UpdateState.ERROR
    assert "ResolutionImpossible" in svc.error, "the original error is masked"
    assert old[:12] in svc.error, f"the error does not say the checkout was reset: {svc.error}"


@pytest.mark.asyncio
async def test_install_backend_deps_reports_the_tail_of_stderr(svc, monkeypatch, tmp_path):
    """The script's verdict (the image to pull) comes last, after pip's own output."""
    backend = tmp_path / "backend"
    backend.mkdir()
    await asyncio.to_thread((backend / "install-deps.sh").write_text, "#!/usr/bin/env bash\n")
    svc.app_dir = str(tmp_path)
    stderr = "pip noise\n" * 200 + "pull mastermerlin/mrln-arcane-tuner:latest"

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="mastermerlin/mrln-arcane-tuner:latest"):
        await svc._install_backend_deps()


@pytest.mark.asyncio
async def test_install_backend_deps_raises_on_failure(svc, monkeypatch, tmp_path):
    backend = tmp_path / "backend"
    backend.mkdir()
    await asyncio.to_thread((backend / "requirements.txt").write_text, "fastapi==1.0\n")
    svc.app_dir = str(tmp_path)

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="boom"):
        await svc._install_backend_deps()
