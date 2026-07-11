"""Tests for the HF download stall guard.

``download_with_stall_guard`` runs the actual HF Hub transfer in a CHILD
PROCESS so it can be killed on a stall (Python threads/in-process socket
reads cannot be aborted). These tests exercise the retry/stall state machine
against REAL (but tiny) child processes spawned via injected ``spawn_fn`` /
``probe_bytes_fn`` callables — no mocking of ``subprocess`` itself, no
network, CPU-only, fast (test knobs keep every stall/backoff interval in the
tens-of-milliseconds range).

``TestHfFetchWorkerProtocol`` covers the worker module's own contract
in-process (no subprocess spawn needed there — it's pure function testing of
``run_download`` / ``main``).
"""
from __future__ import annotations

import subprocess
import sys
import time

import pytest

from app.engine.utils.hf_download_guard import download_with_stall_guard


class TestDownloadWithStallGuard:
    def test_stalled_child_killed_and_retried_then_raises(self):
        """A child that sleeps forever and writes nothing is killed after
        ``stall_timeout_s``, retried up to ``max_attempts`` times, and the
        final failure raises a RuntimeError naming the repo + attempts."""
        calls = {"n": 0}

        def spawn():
            calls["n"] += 1
            return subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(999)"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )

        start = time.monotonic()
        with pytest.raises(RuntimeError) as exc_info:
            download_with_stall_guard(
                repo_id="org/stalled-repo",
                stall_timeout_s=0.15,
                max_attempts=3,
                backoff_s=(0.01, 0.01),
                poll_interval_s=0.03,
                spawn_fn=spawn,
                probe_bytes_fn=lambda: 0,
            )
        elapsed = time.monotonic() - start

        assert calls["n"] == 3, "must retry up to max_attempts, no more"
        assert elapsed < 5.0, "wall time must be bounded by the test knobs"
        msg = str(exc_info.value)
        assert "org/stalled-repo" in msg
        assert "3" in msg
        assert "stall" in msg.lower()

    def test_healthy_child_returns_path_no_retries(self, tmp_path):
        """A child that grows a file and exits 0 printing a path is never
        killed and the guard returns that path on the first attempt."""
        blob = tmp_path / "blob.bin"
        calls = {"n": 0}

        def spawn():
            calls["n"] += 1
            code = (
                "import pathlib\n"
                f"pathlib.Path(r'{blob}').write_bytes(b'x' * 1024)\n"
                # safety-net print: stub child's stdout-path protocol (fixture)
                "print('/resolved/healthy-path')\n"
            )
            return subprocess.Popen(
                [sys.executable, "-c", code],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )

        def probe():
            return blob.stat().st_size if blob.exists() else 0

        result = download_with_stall_guard(
            repo_id="org/healthy-repo",
            stall_timeout_s=5.0,
            max_attempts=3,
            poll_interval_s=0.02,
            spawn_fn=spawn,
            probe_bytes_fn=probe,
        )

        assert result == "/resolved/healthy-path"
        assert calls["n"] == 1

    def test_stall_then_success_recovers(self, tmp_path):
        """Attempt 1 stalls (killed); attempt 2 succeeds — the guard
        recovers and returns the second attempt's path."""
        blob = tmp_path / "blob.bin"
        attempts = {"n": 0}

        def spawn():
            attempts["n"] += 1
            if attempts["n"] == 1:
                code = "import time\ntime.sleep(999)\n"
            else:
                code = (
                    "import pathlib\n"
                    f"pathlib.Path(r'{blob}').write_bytes(b'x' * 10)\n"
                    # safety-net print: stub child's stdout-path protocol
                    "print('/resolved/after-retry')\n"
                )
            return subprocess.Popen(
                [sys.executable, "-c", code],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )

        def probe():
            return blob.stat().st_size if blob.exists() else 0

        result = download_with_stall_guard(
            repo_id="org/flaky-repo",
            stall_timeout_s=0.15,
            max_attempts=3,
            backoff_s=(0.01,),
            poll_interval_s=0.03,
            spawn_fn=spawn,
            probe_bytes_fn=probe,
        )

        assert result == "/resolved/after-retry"
        assert attempts["n"] == 2

    def test_slow_but_progressing_child_not_killed(self, tmp_path):
        """A child that writes a byte every poll tick is NOT killed even
        though the total run time exceeds stall_timeout_s — the stall timer
        resets on every byte of growth. Write cadence (0.05s) is kept ~6x
        tighter than the stall timeout (0.3s) so a slow CI scheduler can't
        flake this into a spurious kill."""
        blob = tmp_path / "blob.bin"
        code = (
            "import pathlib, time\n"
            f"p = pathlib.Path(r'{blob}')\n"
            "for _ in range(10):\n"
            "    with open(p, 'ab') as f:\n"
            "        f.write(b'x')\n"
            "    time.sleep(0.05)\n"
            # safety-net print: stub child's stdout-path protocol (fixture)
            "print('/resolved/slow-path')\n"
        )
        calls = {"n": 0}

        def spawn():
            calls["n"] += 1
            return subprocess.Popen(
                [sys.executable, "-c", code],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )

        def probe():
            return blob.stat().st_size if blob.exists() else 0

        # Total child runtime ~= 10 * 0.05s = 0.5s, well past stall_timeout_s
        # (0.3s) IF the timer didn't reset — proving growth-resets-the-timer.
        result = download_with_stall_guard(
            repo_id="org/slow-repo",
            stall_timeout_s=0.3,
            max_attempts=2,
            poll_interval_s=0.02,
            spawn_fn=spawn,
            probe_bytes_fn=probe,
        )

        assert result == "/resolved/slow-path"
        assert calls["n"] == 1, "must not have been killed/retried"

    def test_chatty_child_not_wedged_by_pipe_backpressure(self, tmp_path):
        """REGRESSION (review finding): a child that spews >1MB to stdout
        while healthily growing the cache must NOT be stall-killed.

        Without the guard actively draining the child's pipes, the OS pipe
        buffer (~4-64KB) fills, the child BLOCKS on its next write, disk
        growth stops, and the watchdog kills a perfectly healthy download —
        a chatty child is realistic (tqdm bars if the user env sets
        HF_HUB_DISABLE_PROGRESS_BARS=0, urllib3 retry warnings on exactly the
        flaky networks this guard targets). The resolved path must still
        parse as the last meaningful stdout line after ~1MB of junk."""
        blob = tmp_path / "blob.bin"
        code = (
            "import pathlib\n"
            f"p = pathlib.Path(r'{blob}')\n"
            "junk = 'x' * 1023\n"
            "for _ in range(1024):\n"  # ~1MB of stdout noise
            # safety-net print: stub child's stdout noise (fixture)
            "    print(junk)\n"
            "    with open(p, 'ab') as f:\n"
            "        f.write(b'y')\n"
            # safety-net print: stub child's stdout-path protocol (fixture)
            "print('/resolved/chatty-path')\n"
        )
        calls = {"n": 0}

        def spawn():
            calls["n"] += 1
            return subprocess.Popen(
                [sys.executable, "-c", code],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )

        def probe():
            return blob.stat().st_size if blob.exists() else 0

        result = download_with_stall_guard(
            repo_id="org/chatty-repo",
            stall_timeout_s=0.5,
            max_attempts=2,
            backoff_s=(0.01,),
            poll_interval_s=0.02,
            spawn_fn=spawn,
            probe_bytes_fn=probe,
        )

        assert result == "/resolved/chatty-path"
        assert calls["n"] == 1, "healthy chatty child must not be killed/retried"

    def test_shrinking_cache_is_activity_not_stall(self, tmp_path):
        """REGRESSION (review finding): a SHRINK is activity, not a stall.

        An etag change makes huggingface_hub delete a large partial
        ``*.incomplete`` blob and restart the transfer. With a
        high-water-mark growth check (``cur > last``), the restarted
        transfer would have to re-earn the old high-water byte count before
        the first timer reset — killed at stall_timeout despite full-speed
        progress, potentially eating every attempt. ANY change must reset
        the timer."""
        blob = tmp_path / "blob.bin"
        blob.write_bytes(b"P" * 10240)  # pre-existing large partial
        code = (
            "import pathlib, time\n"
            f"p = pathlib.Path(r'{blob}')\n"
            "p.unlink()\n"  # etag change: partial discarded
            "time.sleep(0.05)\n"
            "for _ in range(8):\n"  # restarted transfer, slow but steady
            "    with open(p, 'ab') as f:\n"
            "        f.write(b'x')\n"
            "    time.sleep(0.05)\n"
            # safety-net print: stub child's stdout-path protocol (fixture)
            "print('/resolved/restarted-path')\n"
        )
        calls = {"n": 0}

        def spawn():
            calls["n"] += 1
            return subprocess.Popen(
                [sys.executable, "-c", code],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )

        def probe():
            return blob.stat().st_size if blob.exists() else 0

        # Child runs ~0.45s; bytes go 10240 -> 0 -> 1..8, never re-earning
        # the old high-water mark. With change-is-activity semantics the
        # timer resets on every sample; with `cur > last` it would be killed
        # at stall_timeout (0.3s).
        result = download_with_stall_guard(
            repo_id="org/etag-restart-repo",
            stall_timeout_s=0.3,
            max_attempts=2,
            backoff_s=(0.01,),
            poll_interval_s=0.02,
            spawn_fn=spawn,
            probe_bytes_fn=probe,
        )

        assert result == "/resolved/restarted-path"
        assert calls["n"] == 1, "shrink-then-grow child must not be killed"

    def test_mtime_touch_resets_stall_timer(self, tmp_path):
        """REGRESSION (review finding, xet mitigation): a same-size mtime
        touch counts as activity. A preallocating writer (potential hf_xet
        pattern) can freeze the directory's apparent byte total early while
        still writing into the file — the default probe therefore includes
        the newest blob mtime in its signature, so those writes still reset
        the stall timer."""
        from app.engine.utils.hf_download_guard import _dir_signature

        repo_dir = tmp_path / "models--org--prealloc"
        blobs = repo_dir / "blobs"
        blobs.mkdir(parents=True)
        blob = blobs / "shard.incomplete"
        blob.write_bytes(b"\0" * 4096)  # preallocated, size never changes
        code = (
            "import os, time\n"
            f"p = r'{blob}'\n"
            "t = os.stat(p).st_mtime\n"
            "for i in range(1, 9):\n"
            "    os.utime(p, (t + i, t + i))\n"  # same size, newer mtime
            "    time.sleep(0.05)\n"
            # safety-net print: stub child's stdout-path protocol (fixture)
            "print('/resolved/prealloc-path')\n"
        )
        calls = {"n": 0}

        def spawn():
            calls["n"] += 1
            return subprocess.Popen(
                [sys.executable, "-c", code],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )

        result = download_with_stall_guard(
            repo_id="org/prealloc-repo",
            stall_timeout_s=0.3,
            max_attempts=2,
            backoff_s=(0.01,),
            poll_interval_s=0.02,
            spawn_fn=spawn,
            probe_bytes_fn=lambda: _dir_signature(str(repo_dir)),
        )

        assert result == "/resolved/prealloc-path"
        assert calls["n"] == 1, "same-size mtime touches must reset the timer"

    def test_nonzero_exit_error_propagated_and_retried(self):
        """A child that exits nonzero with a stderr message is retried up to
        max_attempts; the final RuntimeError includes the stderr detail."""
        calls = {"n": 0}

        def spawn():
            calls["n"] += 1
            code = (
                "import sys\n"
                "sys.stderr.write('boom: repo not found')\n"
                "sys.exit(1)\n"
            )
            return subprocess.Popen(
                [sys.executable, "-c", code],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )

        with pytest.raises(RuntimeError) as exc_info:
            download_with_stall_guard(
                repo_id="org/broken-repo",
                stall_timeout_s=2.0,
                max_attempts=3,
                backoff_s=(0.01, 0.01),
                poll_interval_s=0.02,
                spawn_fn=spawn,
                probe_bytes_fn=lambda: 0,
            )

        assert calls["n"] == 3
        msg = str(exc_info.value)
        assert "boom" in msg
        assert "org/broken-repo" in msg

    def test_env_knobs_used_when_not_passed_explicitly(self, tmp_path, monkeypatch):
        """MRLN_HF_STALL_TIMEOUT_S / MRLN_HF_DOWNLOAD_ATTEMPTS are read at
        CALL time (not import time) when the explicit kwargs are omitted."""
        monkeypatch.setenv("MRLN_HF_STALL_TIMEOUT_S", "0.15")
        monkeypatch.setenv("MRLN_HF_DOWNLOAD_ATTEMPTS", "2")
        calls = {"n": 0}

        def spawn():
            calls["n"] += 1
            return subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(999)"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )

        with pytest.raises(RuntimeError):
            download_with_stall_guard(
                repo_id="org/env-knob-repo",
                backoff_s=(0.01,),
                poll_interval_s=0.03,
                spawn_fn=spawn,
                probe_bytes_fn=lambda: 0,
            )

        assert calls["n"] == 2, "MRLN_HF_DOWNLOAD_ATTEMPTS=2 must be honored"


class TestDefaultSpawnWorker:
    """The guard's REAL default spawn path — ``-m`` module resolution from
    the ``__file__``-derived backend root, the stdin JSON handshake, and the
    child env — exercised offline by handing the real worker an invalid
    payload (``repo_id=None``), which huggingface_hub's argument validation
    rejects before any network I/O."""

    def test_default_spawn_invalid_repo_id_surfaces_error(self):
        from app.engine.utils.hf_download_guard import _make_default_spawn

        spawn = _make_default_spawn(None, None, None)
        proc = spawn()
        try:
            stdout, stderr = proc.communicate(timeout=90)
        finally:
            if proc.poll() is None:
                proc.kill()

        assert proc.returncode != 0, (
            f"worker must exit nonzero on invalid payload; stdout={stdout!r}"
        )
        assert "repo" in (stderr or "").lower(), (
            f"stderr must name the offending key; stderr={stderr!r}"
        )


class TestHfFetchWorkerProtocol:
    """The worker's own contract, tested in-process (no subprocess spawn) —
    fast, offline, deterministic. The real default-spawn plumbing (module
    resolution, cwd bootstrap, stdin handshake, child env) is covered by
    ``TestDefaultSpawnWorker`` above."""

    def test_run_download_snapshot(self, monkeypatch):
        from app.engine.utils import hf_fetch_worker

        monkeypatch.setattr(
            hf_fetch_worker, "snapshot_download",
            lambda **kw: "/cache/snap:" + repr(sorted(kw.items())),
        )
        result = hf_fetch_worker.run_download({"repo_id": "org/repo"})
        assert result.startswith("/cache/snap:")
        assert "'repo_id', 'org/repo'" in result

    def test_run_download_single_file_with_revision(self, monkeypatch):
        from app.engine.utils import hf_fetch_worker

        captured = {}

        def fake_hf_hub_download(**kw):
            captured.update(kw)
            return "/cache/file.safetensors"

        monkeypatch.setattr(hf_fetch_worker, "hf_hub_download", fake_hf_hub_download)
        result = hf_fetch_worker.run_download(
            {"repo_id": "org/repo", "filename": "f.safetensors", "revision": "rev1"},
        )
        assert result == "/cache/file.safetensors"
        assert captured == {
            "repo_id": "org/repo", "filename": "f.safetensors", "revision": "rev1",
        }

    def test_main_prints_path_and_exits_zero(self, monkeypatch, capsys):
        import io

        from app.engine.utils import hf_fetch_worker

        monkeypatch.setattr(
            hf_fetch_worker, "snapshot_download", lambda **kw: "/resolved/path",
        )
        monkeypatch.setattr(
            sys, "stdin", io.StringIO('{"repo_id": "org/repo"}'),
        )
        code = hf_fetch_worker.main()
        assert code == 0
        out = capsys.readouterr().out
        assert out.strip().splitlines()[-1] == "/resolved/path"

    def test_main_reports_error_and_exits_nonzero(self, monkeypatch, capsys):
        import io

        from app.engine.utils import hf_fetch_worker

        def boom(**kw):
            raise RuntimeError("network exploded")

        monkeypatch.setattr(hf_fetch_worker, "snapshot_download", boom)
        monkeypatch.setattr(
            sys, "stdin", io.StringIO('{"repo_id": "org/repo"}'),
        )
        code = hf_fetch_worker.main()
        assert code != 0
        err = capsys.readouterr().err
        assert "network exploded" in err
