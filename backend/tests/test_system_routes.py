"""
E2E tests for api/system_routes.py — logs, system status, GPU status.
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


@patch("app.core.system_monitor.system_monitor")
def test_get_system_status(mock_mon, client):
    """Should return system snapshot dict."""
    mock_snapshot = MagicMock()
    mock_snapshot.to_dict.return_value = {"cpu_percent": 10.0, "memory_percent": 50.0}
    mock_mon.snapshot.return_value = mock_snapshot
    response = client.get("/api/system/status")
    assert response.status_code == 200
    assert "cpu_percent" in response.json()


@patch("app.core.system_monitor.system_monitor")
def test_get_gpu_status(mock_mon, client):
    """Should return GPU info dict."""
    mock_gpu = MagicMock()
    mock_gpu.to_dict.return_value = {"name": "RTX 4090", "vram_used_mb": 1000}
    mock_snapshot = MagicMock()
    mock_snapshot.gpus = [mock_gpu]
    mock_mon.snapshot.return_value = mock_snapshot
    response = client.get("/api/system/gpu")
    assert response.status_code == 200
    assert "gpus" in response.json()
    assert len(response.json()["gpus"]) == 1
