"""
E2E tests for api/system_routes.py — logs, health, self-update status.
"""

from unittest.mock import patch, MagicMock
from pathlib import Path


_SYS_MODULE = "app.api.system_routes"


def test_get_logs_empty(client):
    """Should return empty when log file doesn't exist."""
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = False
    with patch(f"{_SYS_MODULE}._LOG_FILE", mock_path):
        response = client.get("/api/system/logs")
    assert response.status_code == 200
    assert response.json() == []


def test_clear_logs(client):
    """Should truncate log file."""
    with patch(f"{_SYS_MODULE}.open", create=True) as mock_open:
        mock_file = MagicMock()
        mock_open.return_value.__enter__ = lambda s: mock_file
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        response = client.post("/api/system/logs/clear")
    assert response.status_code == 200
    assert "cleared" in response.json().get("message", "").lower()


def test_get_health(client):
    """Should return a health snapshot with the four KPI-rail fields."""
    response = client.get("/api/system/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["uptime_seconds"] >= 0
    assert isinstance(body["model_count"], int)
    assert isinstance(body["active_jobs"], int)


def test_health_counts_running_and_paused_jobs(client):
    """active_jobs counts RUNNING + PAUSED, not terminal jobs."""
    from app.core.job import JobStatus

    running = MagicMock(status=JobStatus.RUNNING)
    paused = MagicMock(status=JobStatus.PAUSED)
    done = MagicMock(status=JobStatus.COMPLETED)
    failed = MagicMock(status=JobStatus.FAILED)
    with patch(
        "app.core.job_manager.job_manager.list_jobs",
        return_value=[running, paused, done, failed],
    ):
        response = client.get("/api/system/health")
    assert response.status_code == 200
    assert response.json()["active_jobs"] == 2


def test_update_status_full_payload(client):
    """Pin GET /system/update/status — mirrors SelfUpdateService.status_payload()
    verbatim (also the shape broadcast over the update.status WS event)."""
    from app.core.self_update import self_update_service

    with patch.object(
        self_update_service,
        "status_payload",
        return_value={
            "state": "idle",
            "available": True,
            "branch": "main",
            "commit": "abc1234",
            "dirty": False,
            "is_repo": True,
            "behind": 2,
            "active": 0,
            "error": None,
        },
    ):
        response = client.get("/api/system/update/status")
    assert response.status_code == 200
    assert response.json() == {
        "state": "idle",
        "available": True,
        "branch": "main",
        "commit": "abc1234",
        "dirty": False,
        "is_repo": True,
        "behind": 2,
        "active": 0,
        "error": None,
    }


def test_update_check_full_payload(client):
    """Pin POST /system/update/check — {behind, commits}."""
    from app.core.self_update import self_update_service

    async def _fake_check():
        return {"behind": 3, "commits": ["fix: a", "feat: b", "chore: c"]}

    with patch.object(self_update_service, "available", True), patch.object(
        self_update_service, "check", side_effect=_fake_check
    ):
        response = client.post("/api/system/update/check")
    assert response.status_code == 200
    assert response.json() == {
        "behind": 3,
        "commits": ["fix: a", "feat: b", "chore: c"],
    }


def test_update_check_unavailable_403(client):
    from app.core.self_update import self_update_service

    with patch.object(self_update_service, "available", False):
        response = client.post("/api/system/update/check")
    assert response.status_code == 403


def test_system_status_and_gpu_routes_removed(client):
    """B-CLEAN-8: GET /system/status and /system/gpu were orphaned (zero
    frontend callers — live telemetry flows over WebSocket; system.service.ts
    only calls /system/health) and have been removed. Pin that both paths are
    genuinely gone (404), not just failing for some other reason."""
    assert client.get("/api/system/status").status_code == 404
    assert client.get("/api/system/gpu").status_code == 404
