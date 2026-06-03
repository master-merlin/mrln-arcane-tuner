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


# ── GPU snapshot enrichment (free VRAM + per-process) ──────────────────────


class TestGpuVramEnrichment:
    """Free-VRAM serialization + per-process VRAM aggregation."""

    def test_gpu_status_serializes_free_and_processes(self):
        from app.core.system_monitor import GPUStatus, GpuProcess

        g = GPUStatus(
            index=0, name="x", vram_used_mb=100, vram_total_mb=200, vram_percent=50.0,
            temperature_c=40, power_draw_w=1.0, power_limit_w=2.0, gpu_utilization=10,
            memory_utilization=5, clock_graphics_mhz=1000, clock_memory_mhz=2000,
            vram_free_mb=100, processes=[GpuProcess(pid=1, name="comfyui", used_mb=50)],
        )
        d = g.to_dict()
        assert d["vram_free_mb"] == 100
        assert d["processes"][0] == {"pid": 1, "name": "comfyui", "used_mb": 50}

    def test_gpu_processes_merge_sort_and_name(self, monkeypatch):
        from types import SimpleNamespace

        from app.core import system_monitor as sm

        gib = 1024 * 1024 * 1024
        fake = SimpleNamespace(
            NVMLError=Exception,
            nvmlDeviceGetComputeRunningProcesses_v2=lambda h: [
                SimpleNamespace(pid=111, usedGpuMemory=10 * gib),  # 10 GB
                SimpleNamespace(pid=222, usedGpuMemory=2 * gib),   # 2 GB
            ],
            nvmlDeviceGetGraphicsRunningProcesses_v2=lambda h: [
                SimpleNamespace(pid=222, usedGpuMemory=1 * gib),   # dup pid, lower → ignored
            ],
        )
        monkeypatch.setattr(sm, "_NVML_AVAILABLE", True)
        monkeypatch.setattr(sm, "pynvml", fake)
        import psutil
        monkeypatch.setattr(
            psutil, "Process", lambda pid: SimpleNamespace(name=lambda: f"proc{pid}")
        )

        procs = sm.SystemMonitor._gpu_processes(handle=object())
        assert [p.pid for p in procs] == [111, 222]      # sorted desc by used_mb
        assert procs[0].used_mb == 10 * 1024             # MB
        assert procs[1].used_mb == 2 * 1024              # max(2 GB, dup 1 GB)
        assert procs[0].name == "proc111"

    def test_gpu_processes_empty_without_nvml(self, monkeypatch):
        from app.core import system_monitor as sm

        monkeypatch.setattr(sm, "_NVML_AVAILABLE", False)
        assert sm.SystemMonitor._gpu_processes(handle=object()) == []
