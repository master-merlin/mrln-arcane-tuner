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

import itertools
import os
import subprocess
import sys
import time

import pytest

from app.engine.utils.hf_download_guard import download_with_stall_guard

# A child that waits for the test gives up after this much wall time, so a frozen clock or a
# guard that never detects a stall ends the test with a failure instead of a hang.
CHILD_WALL_DEADLINE_S = 30
WATCHDOG_EXIT = 3  # only a child's own give-up exits with it: the guard's kill never gives 3


class StepClock:
    """Virtual time for the guard: it moves only when the test's probe moves it.

    A test that races a real child against a sub-second stall timeout on the wall clock flakes
    whenever the scheduler delays the child (LANE-188). On this clock, whether the timer resets
    depends only on the sequence of probe values.
    """

    def __init__(self) -> None:
        self.t = 0.0
        self.frozen = False

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        if not self.frozen:
            self.t += dt


_END = object()


def scripted_probe(clock: StepClock, steps, measure, done, dt: float):
    """Each call runs the next step in-process (a filesystem change, or ``None`` to only
    measure), advances ``clock`` by ``dt`` and returns ``measure()``.

    The guard's first call takes its baseline, so a script starts with ``None``. After the last
    step the clock is frozen and ``done`` is created: the waiting child then exits.
    """
    it = iter(steps)

    def probe():
        step = next(it, _END)
        if step is _END:
            clock.frozen = True
            done.touch()
        else:
            if step is not None:
                step()
            clock.advance(dt)
        return measure()

    return probe


def waiting_child(done, path: str, before: str = "") -> str:
    """A child that runs ``before``, waits for ``done``, prints ``path`` and exits 0, or gives
    up after ``CHILD_WALL_DEADLINE_S`` with exit ``WATCHDOG_EXIT``."""
    return (
        "import os, sys, time\n"
        f"{before}"
        f"deadline = time.monotonic() + {CHILD_WALL_DEADLINE_S}\n"
        f"while not os.path.exists(r'{done}'):\n"
        "    if time.monotonic() > deadline:\n"
        f"        sys.stderr.write('WATCHDOG: no done after {CHILD_WALL_DEADLINE_S}s')\n"
        f"        sys.exit({WATCHDOG_EXIT})\n"
        "    time.sleep(0.01)\n"
        # safety-net print: stub child's stdout-path protocol (fixture)
        f"print('{path}')\n"
    )


class Children:
    """The guard's ``spawn_fn``: attempt n runs ``code_for(n)``; keeps every child it started."""

    def __init__(self, code_for) -> None:
        self.code_for = code_for
        self.procs: list[subprocess.Popen] = []

    def __call__(self) -> subprocess.Popen:
        proc = subprocess.Popen(
            [sys.executable, "-c", self.code_for(len(self.procs) + 1)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.procs.append(proc)
        return proc

    def kill_all(self) -> None:
        for proc in self.procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def append_byte(path):
    def step():
        with open(path, "ab") as f:
            f.write(b"x")

    return step


def size_of(path):
    return lambda: path.stat().st_size if path.exists() else 0


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
        recovers and returns the second attempt's path.

        On virtual time: attempt 1's probes never change and each moves the
        clock, so it is killed after a fixed number of polls; on attempt 2
        the clock is frozen and its child exits once told to."""
        done = tmp_path / "done"
        clock = StepClock()

        def code_for(attempt):
            if attempt == 1:  # silent: the guard must kill it long before it gives up
                return (
                    "import sys, time\n"
                    f"time.sleep({CHILD_WALL_DEADLINE_S})\n"
                    "sys.stderr.write('WATCHDOG: attempt 1 was never killed')\n"
                    f"sys.exit({WATCHDOG_EXIT})\n"
                )
            return waiting_child(done, "/resolved/after-retry")

        children = Children(code_for)

        def probe():
            if len(children.procs) >= 2:
                clock.frozen = True
                done.touch()
            else:
                clock.advance(0.05)
            return 0

        try:
            result = download_with_stall_guard(
                repo_id="org/flaky-repo",
                stall_timeout_s=0.15,
                max_attempts=3,
                backoff_s=(0.01,),
                poll_interval_s=0.01,
                clock=clock,
                spawn_fn=children,
                probe_bytes_fn=probe,
            )
        finally:
            children.kill_all()

        assert result == "/resolved/after-retry"
        assert len(children.procs) == 2
        assert children.procs[0].returncode != WATCHDOG_EXIT, "attempt 1 must end as a stall kill"

    def test_slow_but_progressing_child_not_killed(self, tmp_path):
        """A download that grows a byte every second poll is NOT killed even
        though its total run time exceeds stall_timeout_s — the stall timer
        resets on every byte of growth.

        On virtual time: each poll is 0.1s, so the gaps (0.2s) stay below the
        timeout (0.3s) and the total (2.1s) is far past it; no scheduling
        delay can stretch a gap."""
        blob, done, clock = tmp_path / "blob.bin", tmp_path / "done", StepClock()
        steps = [None] + [append_byte(blob), None] * 10
        children = Children(lambda attempt: waiting_child(done, "/resolved/slow-path"))
        try:
            result = download_with_stall_guard(
                repo_id="org/slow-repo",
                stall_timeout_s=0.3,
                max_attempts=2,
                poll_interval_s=0.01,
                clock=clock,
                spawn_fn=children,
                probe_bytes_fn=scripted_probe(clock, steps, size_of(blob), done, dt=0.1),
            )
        finally:
            children.kill_all()

        assert result == "/resolved/slow-path"
        assert len(children.procs) == 1, "must not have been killed/retried"

    def test_chatty_child_not_wedged_by_pipe_backpressure(self, tmp_path):
        """REGRESSION (review finding): a child that spews >1MB to stdout
        while healthily growing the cache must NOT be stall-killed.

        Without the guard actively draining the child's pipes, the OS pipe
        buffer (~4-64KB) fills, the child BLOCKS on its next write, disk
        growth stops, and the watchdog kills a perfectly healthy download —
        a chatty child is realistic (tqdm bars if the user env sets
        HF_HUB_DISABLE_PROGRESS_BARS=0, urllib3 retry warnings on exactly the
        flaky networks this guard targets). The resolved path must still
        parse as the last meaningful stdout line after ~1MB of junk.

        The activity stays the child's own: the probe reports progress until
        the child has got past its 1MB of output (``spewed``), or until
        ``CHILD_WALL_DEADLINE_S`` of wall time, after which it reports none
        and the virtual timeout kills the wedged child. A drained child gets
        there in well under a second, so the margin is about 30x."""
        spewed, done, clock = tmp_path / "spewed", tmp_path / "done", StepClock()
        before = (
            "junk = 'x' * 1023\n"
            "for _ in range(1024):\n"  # ~1MB of stdout noise
            # safety-net print: stub child's stdout noise (fixture)
            "    print(junk)\n"
            f"open(r'{spewed}', 'w').close()\n"
        )
        children = Children(lambda attempt: waiting_child(done, "/resolved/chatty-path", before))
        progress = itertools.count()
        deadline = time.monotonic() + CHILD_WALL_DEADLINE_S

        def probe():
            if spewed.exists():
                clock.frozen = True
                done.touch()
                return "spewed"
            clock.advance(0.1)
            # A healthy download grows while it prints; a wedged one stops.
            return next(progress) if time.monotonic() < deadline else "wedged"

        try:
            result = download_with_stall_guard(
                repo_id="org/chatty-repo",
                stall_timeout_s=0.5,
                max_attempts=2,
                backoff_s=(0.01,),
                poll_interval_s=0.01,
                clock=clock,
                spawn_fn=children,
                probe_bytes_fn=probe,
            )
        finally:
            children.kill_all()

        assert result == "/resolved/chatty-path"
        assert len(children.procs) == 1, "healthy chatty child must not be killed/retried"

    def test_shrinking_cache_is_activity_not_stall(self, tmp_path):
        """REGRESSION (review finding): a SHRINK is activity, not a stall.

        An etag change makes huggingface_hub delete a large partial
        ``*.incomplete`` blob and restart the transfer. With a
        high-water-mark growth check (``cur > last``), the restarted
        transfer would have to re-earn the old high-water byte count before
        the first timer reset — killed at stall_timeout despite full-speed
        progress, potentially eating every attempt. ANY change must reset
        the timer."""
        blob, done, clock = tmp_path / "blob.bin", tmp_path / "done", StepClock()
        blob.write_bytes(b"P" * 10240)  # pre-existing large partial
        # The baseline measures the partial (10240); then the etag change
        # discards it, and the restarted transfer grows slow but steady:
        # bytes go 10240 -> 0 -> 1..8, never re-earning the old high-water
        # mark. With change-is-activity semantics the timer resets on every
        # change; with `cur > last` it would be killed at stall_timeout.
        steps = [None, blob.unlink, None] + [append_byte(blob), None] * 8
        children = Children(lambda attempt: waiting_child(done, "/resolved/restarted-path"))
        try:
            result = download_with_stall_guard(
                repo_id="org/etag-restart-repo",
                stall_timeout_s=0.3,
                max_attempts=2,
                backoff_s=(0.01,),
                poll_interval_s=0.01,
                clock=clock,
                spawn_fn=children,
                probe_bytes_fn=scripted_probe(clock, steps, size_of(blob), done, dt=0.1),
            )
        finally:
            children.kill_all()

        assert result == "/resolved/restarted-path"
        assert len(children.procs) == 1, "shrink-then-grow child must not be killed"

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
        done, clock = tmp_path / "done", StepClock()
        t = os.stat(blob).st_mtime

        def touch(i):
            return lambda: os.utime(blob, (t + i, t + i))  # same size, newer mtime

        steps = [None] + [step for i in range(1, 9) for step in (touch(i), None)]
        children = Children(lambda attempt: waiting_child(done, "/resolved/prealloc-path"))
        try:
            result = download_with_stall_guard(
                repo_id="org/prealloc-repo",
                stall_timeout_s=0.3,
                max_attempts=2,
                backoff_s=(0.01,),
                poll_interval_s=0.01,
                clock=clock,
                spawn_fn=children,
                probe_bytes_fn=scripted_probe(
                    clock, steps, lambda: _dir_signature(str(repo_dir)), done, dt=0.1,
                ),
            )
        finally:
            children.kill_all()

        assert result == "/resolved/prealloc-path"
        assert len(children.procs) == 1, "same-size mtime touches must reset the timer"

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

    def test_default_spawn_leaves_no_closed_stdin_pipe(self):
        """Pin for CI run 33687356291 (ubuntu): ``_make_default_spawn`` writes
        the JSON handshake and closes its end of the pipe, but used to leave
        the CLOSED file object on ``proc.stdin``. POSIX ``communicate()``
        flushes a truthy ``stdin`` before draining and a closed TextIOWrapper
        raises ``ValueError: I/O operation on closed file``; Windows' threaded
        path only re-closes it, so the dev box never saw it. The platform-
        independent observable is that ``stdin`` is ``None`` after spawn —
        exactly what Popen holds when stdin is not a pipe, and the state every
        ``communicate``/``wait``/``__exit__`` consumer skips."""
        from app.engine.utils.hf_download_guard import _make_default_spawn

        proc = _make_default_spawn(None, None, None)()
        try:
            assert proc.stdin is None, (
                "spawn must not leave a closed PIPE object on proc.stdin; "
                f"got {proc.stdin!r} (closed={getattr(proc.stdin, 'closed', None)})"
            )
            # The handshake still reached the child: with stdin=None the
            # worker's ``sys.stdin.read()`` saw the payload + EOF and rejected
            # ``repo_id=None`` by name, rather than hanging on an open pipe.
            _stdout, stderr = proc.communicate(timeout=90)
        finally:
            if proc.poll() is None:
                proc.kill()
        assert proc.returncode != 0
        assert "repo" in (stderr or "").lower(), stderr


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
