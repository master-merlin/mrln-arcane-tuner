"""
Tests for the system administration API endpoints.
Covers: /api/system/restart, /api/system/logs, /api/system/logs/clear.
"""


from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path


# ── Restart Endpoint ────────────────────────────────────────────────────


class TestRestartEndpoint:
    """Tests for POST /api/system/restart."""

    def test_restart_returns_message(self, client):
        """Restart endpoint should return a status message."""
        with patch("app.api.system_routes._restart_server_logic"):
            resp = client.post("/api/system/restart")

        assert resp.status_code == 200
        assert "restart" in resp.json()["message"].lower()


# ── Log Endpoints ────────────────────────────────────────────────────────


class TestLogEndpoints:
    """Tests for GET /api/system/logs and POST /api/system/logs/clear."""

    def test_get_logs_returns_list(self, client):
        """GET /api/system/logs should return a list."""
        fake_lines = "line1\nline2\nline3\n"
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        with patch("app.api.system_routes._LOG_FILE", mock_path):
            with patch("builtins.open", mock_open(read_data=fake_lines)):
                resp = client.get("/api/system/logs?lines=10")

        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_logs_missing_file(self, client):
        """GET /api/system/logs should return empty list when server.log is absent."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        with patch("app.api.system_routes._LOG_FILE", mock_path):
            resp = client.get("/api/system/logs")

        assert resp.status_code == 200
        assert resp.json() == []

    def test_clear_logs_truncates(self, client):
        """POST /api/system/logs/clear should return success message."""
        m = mock_open()
        with patch("builtins.open", m):
            resp = client.post("/api/system/logs/clear")

        assert resp.status_code == 200
        assert "cleared" in resp.json()["message"].lower()
