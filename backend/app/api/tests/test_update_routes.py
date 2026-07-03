import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.drain import DrainActive
from app.core.self_update import self_update_service, UpdateState


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    self_update_service.state = UpdateState.IDLE
    self_update_service.available = True
    monkeypatch.setattr(self_update_service, "git_status",
                        lambda: {"is_repo": True, "branch": "main", "commit": "abc1234", "dirty": False})
    monkeypatch.setattr(self_update_service, "active_task_count", lambda: 0)
    yield


def test_status_returns_payload(client):
    r = client.get("/api/system/update/status")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["branch"] == "main"
    assert body["commit"] == "abc1234"
    assert body["state"] == "idle"


def test_check_refused_when_unavailable(client):
    self_update_service.available = False
    assert client.post("/api/system/update/check").status_code == 403


def test_apply_refused_when_unavailable(client):
    self_update_service.available = False
    assert client.post("/api/system/update/apply").status_code == 403


def test_apply_accepted_when_available(client, monkeypatch):
    called = {"v": False}
    monkeypatch.setattr(self_update_service, "apply", lambda: called.__setitem__("v", True))
    r = client.post("/api/system/update/apply")
    assert r.status_code == 200
    assert called["v"] is True


def test_drain_active_maps_to_409(client):
    # Register a throwaway route that raises DrainActive; the global handler
    # must turn it into a 409.
    @app.get("/api/_test/drain-boom")
    async def _boom():
        raise DrainActive("paused")

    r = client.get("/api/_test/drain-boom")
    assert r.status_code == 409
    assert r.json()["detail"] == "paused"
