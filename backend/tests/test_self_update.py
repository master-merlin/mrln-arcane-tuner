"""SelfUpdateService._wait_for_idle — bounded drain (W4.T5).

Before this, the PENDING_RESTART drain polled active_task_count() with no
timeout: one worker permanently stuck RUNNING (e.g. a GPU stall) kept the
service draining forever, rejecting new GPU tasks with no operator escape
hatch.
"""

from __future__ import annotations

import os
import types

import pytest

from app.core.self_update import SelfUpdateService


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
