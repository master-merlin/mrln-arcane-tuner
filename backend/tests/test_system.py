"""
Tests for the system administration API endpoints.
Covers: /api/system/restart, /api/system/logs, /api/system/logs/clear.
"""

import asyncio
import os
import subprocess
import sys

from unittest.mock import patch, mock_open, AsyncMock, MagicMock
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

    def test_restart_spawn_writes_to_a_file_it_can_never_block_on(self, tmp_path):
        """Two properties at once, and the second was bought with the first.

        The spawn must not inherit our stdio: those handles can point at a dead
        pipe (the IDE terminal that launched the original server crashed), and
        once that pipe's buffer fills, the first console log write blocks while
        HOLDING the logging lock — wedging the entire event loop (2026-07-16
        live incident, e2e3cfc8). But DEVNULL then made a failed restart
        invisible everywhere (LANE-51), so the target must be a real FILE:
        durable, and with no reader that can ever stall it.

        Asserted on the artefact, not on the call: we write through the handle
        the spawn was given and read the bytes back out of the restart log.

        The exit is stubbed at ``schedule_exit`` rather than at ``os._exit``
        (LANE-56). Two reasons, and the first is not a preference: this process
        IS the one under test, so the real exit cannot run here — and the old
        `patch("...system_routes.os._exit")` patched the attribute on the shared
        `os` module, which made the whole suite's survival depend on the stub
        outliving anything that had scheduled a call to it. What the exit
        actually does is proved where it can be: against a real process, in
        ``test_restart_launcher.py::
        test_the_outgoing_server_leaves_the_port_even_when_its_loop_is_blocked``.
        """
        from app.api import system_routes

        log_path = tmp_path / "restart.log"

        def _child_writes(cmd, **kwargs):
            kwargs["stdout"].write("CHILD-SAYS-HELLO\n")
            return MagicMock()

        with (
            patch.object(system_routes.restart_launcher, "RESTART_LOG_PATH", str(log_path)),
            patch("app.api.system_routes.subprocess.Popen",
                  side_effect=_child_writes) as popen,
            patch.object(system_routes.restart_launcher, "schedule_exit") as fake_exit,
            patch("app.api.system_routes.asyncio.sleep", new=AsyncMock()),
        ):
            asyncio.run(system_routes._restart_server_logic())

        kwargs = popen.call_args.kwargs
        assert kwargs.get("stdin") is subprocess.DEVNULL
        assert kwargs.get("stdout") is not subprocess.DEVNULL
        assert kwargs.get("stdout") is kwargs.get("stderr")
        assert kwargs["stdout"].closed, "the parent must drop its copy of the handle"
        # The observable part: what the child wrote is on disk afterwards.
        assert "CHILD-SAYS-HELLO" in log_path.read_text(encoding="utf-8")
        assert fake_exit.call_count == 1, "the outgoing server must still leave"
        assert fake_exit.call_args.args[0] > 0, "the flush window must be bounded"

    def test_restart_hands_off_to_the_launcher_with_a_command_it_accepts(self):
        """The route and the launcher must agree on the command line. Pinned by
        parsing the real argv with the real parser — a wiring mistake here would
        otherwise only show up as a restart that silently never happened."""
        from app.api import system_routes
        from pathlib import Path as _Path

        with (
            patch("app.api.system_routes.subprocess.Popen") as popen,
            patch("app.api.system_routes.os._exit"),
            patch("app.api.system_routes.asyncio.sleep", new=AsyncMock()),
        ):
            asyncio.run(system_routes._restart_server_logic())

        cmd = popen.call_args.args[0]
        assert _Path(cmd[1]).name == "restart_launcher.py"
        assert _Path(cmd[1]).exists()
        opts, child = system_routes.restart_launcher._parse_args(cmd[2:])
        assert opts["old_pid"] == str(os.getpid())
        assert child[0] == sys.executable

    def test_a_previous_restart_failure_is_reported_once(self, tmp_path, caplog):
        """Nobody is alive to report a failed restart when it happens — the next
        server to start is the first listener there is, and it must say so."""
        import json
        from app.api import system_routes

        log_path = tmp_path / "restart.log"
        log_path.write_text(
            json.dumps({"event": "restart_failed", "exit_code": 3,
                        "timestamp": "2026-08-31T18:14:56Z",
                        "message": "the replacement server exited with code 3"}) + "\n",
            encoding="utf-8")

        with patch.object(system_routes.restart_launcher, "RESTART_LOG_PATH", str(log_path)):
            with caplog.at_level("ERROR"):
                first = system_routes.report_pending_restart_failure()
            assert first is not None
            assert "previous_restart_failed" in caplog.text
            assert "exited with code 3" in caplog.text
            # WHEN it failed must survive the hand-over; the log pipeline
            # stamps its own `timestamp` over any it is handed.
            assert "2026-08-31T18:14:56Z" in caplog.text

            caplog.clear()
            assert system_routes.report_pending_restart_failure() is None
            assert "previous_restart_failed" not in caplog.text


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
            index=0,
            name="x",
            vram_used_mb=100,
            vram_total_mb=200,
            vram_percent=50.0,
            temperature_c=40,
            power_draw_w=1.0,
            power_limit_w=2.0,
            gpu_utilization=10,
            memory_utilization=5,
            clock_graphics_mhz=1000,
            clock_memory_mhz=2000,
            vram_free_mb=100,
            processes=[GpuProcess(pid=1, name="comfyui", used_mb=50)],
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
                SimpleNamespace(pid=222, usedGpuMemory=2 * gib),  # 2 GB
            ],
            nvmlDeviceGetGraphicsRunningProcesses_v2=lambda h: [
                SimpleNamespace(
                    pid=222, usedGpuMemory=1 * gib
                ),  # dup pid, lower → ignored
            ],
        )
        monkeypatch.setattr(sm, "_NVML_AVAILABLE", True)
        monkeypatch.setattr(sm, "pynvml", fake)
        import psutil

        monkeypatch.setattr(
            psutil, "Process", lambda pid: SimpleNamespace(name=lambda: f"proc{pid}")
        )

        procs = sm.SystemMonitor._gpu_processes(handle=object())
        assert [p.pid for p in procs] == [111, 222]  # sorted desc by used_mb
        assert procs[0].used_mb == 10 * 1024  # MB
        assert procs[1].used_mb == 2 * 1024  # max(2 GB, dup 1 GB)
        assert procs[0].name == "proc111"

    def test_gpu_processes_empty_without_nvml(self, monkeypatch):
        from app.core import system_monitor as sm

        monkeypatch.setattr(sm, "_NVML_AVAILABLE", False)
        assert sm.SystemMonitor._gpu_processes(handle=object()) == []
