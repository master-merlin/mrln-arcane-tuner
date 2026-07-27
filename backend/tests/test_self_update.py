"""SelfUpdateService._wait_for_idle — bounded drain (W4.T5).

Before this, the PENDING_RESTART drain polled active_task_count() with no
timeout: one worker permanently stuck RUNNING (e.g. a GPU stall) kept the
service draining forever, rejecting new GPU tasks with no operator escape
hatch.
"""

from __future__ import annotations

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
