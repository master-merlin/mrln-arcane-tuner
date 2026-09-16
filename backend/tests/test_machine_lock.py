"""LANE-63 step 4 — the cross-process machine lock behind ``xdist_group``.

Every claim the lock makes is exercised against REAL second processes
(``subprocess``), never a thread pretending to be one, because the property
under test is exactly the one a thread cannot show: two Python processes
that share nothing but a path.

All tests point ``MRLN_MACHINE_LOCK_DIR`` at ``tmp_path`` so they never contend
with (or wrongly recover) a real ``_harness/.gpu.lock`` held by another run.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.support import machine_lock as machine_lock_mod
from tests.support.machine_lock import (
    LOCK_DIR_ENV,
    MachineLockTimeout,
    lock_dir,
    lock_path,
    machine_lock,
    read_holder,
)

BACKEND = Path(__file__).resolve().parents[1]
TESTS = BACKEND / "tests"

# 0x7FFFFFFF: never a Windows pid (they are multiples of 4) and above Linux's
# default pid_max (4194304) — a pid that is dead on every box this runs on.
DEAD_PID = 2**31 - 1


def _child(code: str, lock_dir_: Path, *, timeout: float = 60.0) -> subprocess.Popen:
    env = dict(os.environ, **{LOCK_DIR_ENV: str(lock_dir_)})
    env.pop("PYTEST_XDIST_WORKER", None)
    return subprocess.Popen([sys.executable, "-c", code], cwd=str(BACKEND), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


_HOLD = """
import json, os, time
from pathlib import Path
from tests.support.machine_lock import machine_lock
release = Path(r'{report}').with_suffix('.release')
with machine_lock("{name}", timeout_s={timeout}) as p:
    t0 = time.time()
    # Hold at least {hold}s, then until the parent drops the release file
    # (bounded: a parent that died stops us after 300s).
    time.sleep({hold})
    deadline = time.time() + 300
    while {wait_release} and not release.exists() and time.time() < deadline:
        time.sleep(0.05)
    t1 = time.time()
open(r'{report}', "w").write(json.dumps({{"pid": os.getpid(), "acquired": t0, "released": t1}}))
"""


def _hold_code(report: Path, hold: float, timeout: float = 30.0, name: str = "probe",
               wait_release: bool = False) -> str:
    """A holder that sleeps *hold* seconds; with *wait_release* it then keeps
    the lock until ``<report>.release`` exists — a fixed sleep is not a hold
    a test can rely on when the contender is a child pytest that needs tens
    of seconds to boot on a loaded box (measured: the 6 s holder was gone
    before the child asked, under `-n 8` with two other gates running)."""
    return _HOLD.format(hold=hold, report=str(report), timeout=timeout, name=name,
                        wait_release=wait_release)


def _release(report: Path) -> None:
    report.with_suffix(".release").write_text("go")


# ── path ─────────────────────────────────────────────────────────────────


def test_the_default_directory_is_the_shared_harness_dir_or_the_machine_tempdir(monkeypatch):
    monkeypatch.delenv(LOCK_DIR_ENV, raising=False)
    d = lock_dir()
    repo = BACKEND.parent
    assert d.is_dir()
    if (repo / "_harness").is_dir():
        assert d == repo / "_harness", d
    else:  # a public clone: still one file per machine
        import tempfile

        assert d == Path(tempfile.gettempdir()), d
    assert lock_path("gpu").name == ".gpu.lock"


def test_the_env_override_moves_the_lock(monkeypatch, tmp_path):
    monkeypatch.setenv(LOCK_DIR_ENV, str(tmp_path))
    assert lock_path("gpu") == tmp_path / ".gpu.lock"


# ── two real processes contend ───────────────────────────────────────────


def test_the_loser_waits_and_acquires_after_the_holder_releases(tmp_path):
    a_report, b_report = tmp_path / "a.json", tmp_path / "b.json"
    a = _child(_hold_code(a_report, hold=1.5), tmp_path)
    # Let A take the lock before B starts asking for it.
    deadline = time.time() + 20
    while not (tmp_path / ".probe.lock").exists():
        assert a.poll() is None, a.communicate()[1]
        assert time.time() < deadline, "A never took the lock"
        time.sleep(0.05)
    b = _child(_hold_code(b_report, hold=0.1), tmp_path)
    for proc in (a, b):
        out, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, err[-2000:]
    ra, rb = json.loads(a_report.read_text()), json.loads(b_report.read_text())
    assert rb["acquired"] >= ra["released"], (
        f"B acquired at {rb['acquired']:.3f} while A held until {ra['released']:.3f}")
    assert not (tmp_path / ".probe.lock").exists(), "the lock file outlived both holders"


def test_the_owner_metadata_names_the_holding_process(tmp_path):
    with machine_lock("probe", path=tmp_path / ".probe.lock") as p:
        holder = read_holder(p)
        assert holder is not None
        assert holder.pid == os.getpid()
        assert holder.tree == str(BACKEND.parent)
        assert holder.since.endswith("+00:00")
        assert holder.describe().startswith(f"pid {os.getpid()}")


# ── stale recovery ───────────────────────────────────────────────────────


def test_a_lock_left_by_a_dead_pid_is_recovered(tmp_path):
    p = tmp_path / ".probe.lock"
    p.write_text(json.dumps({"pid": DEAD_PID, "worker": None, "tree": "x",
                             "since": "2026-01-01T00:00:00+00:00"}))
    t0 = time.monotonic()
    with machine_lock("probe", path=p, timeout_s=5.0):
        assert read_holder(p).pid == os.getpid(), "the stale owner was not replaced"
    assert time.monotonic() - t0 < 2.0, "recovery should not wait out the timeout"
    assert not p.exists()


def test_an_unreadable_lock_is_a_writer_in_flight_until_the_grace_runs_out(tmp_path):
    p = tmp_path / ".probe.lock"
    p.write_text("")  # a winner between open() and write()
    with pytest.raises(MachineLockTimeout, match="unreadable owner"):
        with machine_lock("probe", path=p, timeout_s=0.5):
            pass
    assert p.exists(), "a fresh unreadable file was broken"
    old = time.time() - 60
    os.utime(p, (old, old))
    with machine_lock("probe", path=p, timeout_s=0.5):
        assert read_holder(p).pid == os.getpid()


def _live_owner(worker: str = "gwB") -> str:
    """Owner metadata for a LIVE process — this one, so ``psutil`` agrees."""
    return json.dumps({"pid": os.getpid(), "worker": worker, "tree": "another-tree",
                       "since": "2026-09-16T00:00:00+00:00"})


def test_recovery_never_deletes_a_lock_another_waiter_took_in_the_window(tmp_path,
                                                                        monkeypatch):
    """VERIFY 1.01: between judging a lock stale and unlinking it, a second
    waiter can have recovered it and taken it. Unlinking THAT file admits two
    holders at once. The window is forced open here by making the liveness
    probe itself replace the file — exactly what a second waiter does.
    """
    p = tmp_path / ".probe.lock"
    p.write_text(json.dumps({"pid": DEAD_PID, "worker": None, "tree": "x",
                             "since": "2026-01-01T00:00:00+00:00"}))
    live, probes = _live_owner(), []
    real_alive = machine_lock_mod._pid_alive

    def racing_probe(pid: int) -> bool:
        probes.append(pid)
        if len(probes) == 1:
            assert pid == DEAD_PID
            p.write_text(live)  # waiter B recovered it and now holds it
            return False
        return real_alive(pid)

    monkeypatch.setattr(machine_lock_mod, "_pid_alive", racing_probe)
    with pytest.raises(MachineLockTimeout):
        with machine_lock("probe", path=p, timeout_s=0.5):
            pass
    assert p.exists(), "the recovering waiter deleted the live lock of waiter B"
    assert p.read_text() == live, "B's lock was replaced — two holders at once"


def test_recovery_leaves_no_sentinel_behind(tmp_path):
    """The recovery turn is itself a file; a real recovery must not leak it."""
    p = tmp_path / ".probe.lock"
    p.write_text(json.dumps({"pid": DEAD_PID, "worker": None, "tree": "x",
                             "since": "2026-01-01T00:00:00+00:00"}))
    with machine_lock("probe", path=p, timeout_s=5.0):
        pass
    assert not p.exists()
    assert sorted(q.name for q in tmp_path.iterdir()) == [], list(tmp_path.iterdir())


def test_a_live_recovery_turn_keeps_a_second_waiter_out_of_the_unlink(tmp_path):
    """The serialisation itself, seen: while another process holds the recovery
    turn, this waiter may not remove the stale lock — only ONE process is ever
    between a judgement and an unlink, which is what makes the identity check
    conclusive. It waits its bound instead."""
    p = tmp_path / ".probe.lock"
    stale = json.dumps({"pid": DEAD_PID, "worker": None, "tree": "x",
                        "since": "2026-01-01T00:00:00+00:00"})
    p.write_text(stale)
    sentinel = machine_lock_mod.recovery_sentinel(p)
    sentinel.write_text("a recoverer at work")  # fresh: inside the grace
    with pytest.raises(MachineLockTimeout):
        with machine_lock("probe", path=p, timeout_s=0.5):
            pass
    assert p.read_text() == stale, "recovered a lock while another turn was open"
    sentinel.unlink()


def test_a_sentinel_left_by_a_crashed_recoverer_does_not_block_recovery(tmp_path):
    """Prove the negative for the serialisation itself: an orphaned turn older
    than the grace must not make the lock unrecoverable forever."""
    p = tmp_path / ".probe.lock"
    p.write_text(json.dumps({"pid": DEAD_PID, "worker": None, "tree": "x",
                             "since": "2026-01-01T00:00:00+00:00"}))
    sentinel = machine_lock_mod.recovery_sentinel(p)
    sentinel.write_text("crashed recoverer")
    old = time.time() - 10 * machine_lock_mod.UNREADABLE_GRACE_S
    os.utime(sentinel, (old, old))
    with machine_lock("probe", path=p, timeout_s=10.0):
        assert read_holder(p).pid == os.getpid()
    assert not sentinel.exists()


# ── a live holder past the bound is a FAILURE naming it ──────────────────


def test_a_live_holder_past_the_bound_fails_naming_the_holder(tmp_path):
    report = tmp_path / "a.json"
    a = _child(_hold_code(report, hold=0.5, wait_release=True), tmp_path)
    try:
        deadline = time.time() + 20
        while read_holder(tmp_path / ".probe.lock") is None:
            assert a.poll() is None, a.communicate()[1]
            assert time.time() < deadline, "A never took the lock"
            time.sleep(0.05)
        # NOT `a.pid`: on Windows the venv's Scripts\python.exe is a launcher
        # that runs the real interpreter as its child, so Popen.pid is one
        # process up from the os.getpid() the holder wrote. The lock names
        # the process that actually holds it.
        holder_pid = read_holder(tmp_path / ".probe.lock").pid
        assert holder_pid != os.getpid()
        with pytest.raises(MachineLockTimeout) as exc:
            with machine_lock("probe", path=tmp_path / ".probe.lock", timeout_s=0.7):
                pass
        assert f"pid {holder_pid}" in str(exc.value), str(exc.value)
        assert (tmp_path / ".probe.lock").exists(), "the live holder's lock was broken"
        assert read_holder(tmp_path / ".probe.lock").pid == holder_pid
    finally:
        _release(report)
        a.kill()
        a.communicate()
        (tmp_path / ".probe.lock").unlink(missing_ok=True)


def test_release_never_removes_another_processes_lock(tmp_path):
    """After a wrongful recovery race the file may be someone else's by the time
    we release; release checks ownership instead of blindly unlinking."""
    p = tmp_path / ".probe.lock"
    with machine_lock("probe", path=p):
        p.write_text(json.dumps({"pid": DEAD_PID, "worker": "gwZ", "tree": "x",
                                 "since": "2026-01-01T00:00:00+00:00"}))
    assert p.exists(), "released a lock that had been re-owned"
    p.unlink()


# ── the lock is gone after the block raised ──────────────────────────────


def test_the_lock_is_released_when_the_block_raises(tmp_path):
    p = tmp_path / ".probe.lock"
    with pytest.raises(RuntimeError, match="deliberate"):
        with machine_lock("probe", path=p):
            assert p.exists()
            raise RuntimeError("deliberate")
    assert not p.exists()


# ── the pytest pairing in backend/conftest.py ────────────────────────────


@pytest.mark.xdist_group("lockprobe")
def test_a_grouped_test_holds_its_group_lock_while_it_runs():
    """The effect, seen: a bare marker (no fixture named) means this process
    owns `.lockprobe.lock` for the duration. Under MRLN_LOCK_PROBE_RAISE it
    raises so the parent below can watch the lock disappear anyway."""
    p = lock_path("lockprobe")
    assert p.exists(), f"no lock at {p} — the marker/lock pairing is gone"
    holder = read_holder(p)
    assert holder is not None and holder.pid == os.getpid(), holder
    if os.environ.get("MRLN_LOCK_PROBE_RAISE"):
        raise RuntimeError("deliberate")


@pytest.mark.xdist_group("lockprobe")
def test_a_second_grouped_test_of_the_group_takes_the_lock_in_its_turn():
    """Runs in the same child as the probe above (VERIFY 1.02): if the first
    test's teardown failed to release, this one waits out its whole bound
    against its own still-live process and errors."""
    p = lock_path("lockprobe")
    assert p.exists(), f"no lock at {p} — the marker/lock pairing is gone"
    holder = read_holder(p)
    assert holder is not None and holder.pid == os.getpid(), holder


_PROBE_1 = "test_a_grouped_test_holds_its_group_lock_while_it_runs"
_PROBE_2 = "test_a_second_grouped_test_of_the_group_takes_the_lock_in_its_turn"


def _child_pytest(lock_dir_: Path, *extra: str, raise_: bool,
                  env_extra: dict | None = None,
                  nodes: tuple[str, ...] = (_PROBE_1,)) -> subprocess.CompletedProcess:
    env = dict(os.environ, **{LOCK_DIR_ENV: str(lock_dir_)}, **(env_extra or {}))
    # An invented worker id keeps the child's logs off the serial names the
    # parent pytest may be holding open (see test_xdist_worker_isolation.py).
    env["PYTEST_XDIST_WORKER"] = "gwLock"
    if raise_:
        env["MRLN_LOCK_PROBE_RAISE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *extra,
         *(f"{__file__}::{node}" for node in nodes)],
        cwd=str(BACKEND), env=env, capture_output=True, text=True, timeout=300)


def test_a_raising_teardown_hook_still_releases_the_group_lock(tmp_path):
    """VERIFY 1.02: a fixture finaliser that raises makes the runner's own
    teardown hookimpl raise, and pluggy then never calls the impls ordered
    after it. With the release in a plain ``trylast`` hook the lock survived
    the test, and the NEXT test of the group waited out its bound against its
    own still-live process. Observable: the second probe passes, no
    machine-lock message anywhere, no lock file left on disk."""
    proc = _child_pytest(tmp_path, "-p", "tests.support.raising_teardown_plugin",
                         raise_=False, nodes=(_PROBE_1, _PROBE_2),
                         env_extra={"MRLN_MACHINE_LOCK_TIMEOUT": "3"})
    assert "deliberate teardown explosion" in proc.stdout, proc.stdout[-2000:]
    assert "machine lock" not in proc.stdout, (
        "the second grouped test timed out against the first test's unreleased lock\n"
        + proc.stdout[-2000:])
    assert "2 passed" in proc.stdout, proc.stdout[-2000:]
    assert not (tmp_path / ".lockprobe.lock").exists(), (
        "the lock outlived a teardown that raised")


def test_the_group_lock_is_gone_after_a_grouped_test_raised(tmp_path):
    proc = _child_pytest(tmp_path, raise_=True)
    assert proc.returncode != 0, "the deliberate raise did not fail the child"
    assert "deliberate" in proc.stdout
    assert not (tmp_path / ".lockprobe.lock").exists(), "the lock outlived the failed test"


def test_without_the_root_conftest_no_lock_is_taken(tmp_path):
    """Prove the negative: drop backend/conftest.py (`--confcutdir=tests`) and
    the grouped test finds no lock — so it is that file, not xdist, doing it."""
    proc = _child_pytest(tmp_path, f"--confcutdir={TESTS}", raise_=False)
    assert proc.returncode != 0, "passed without the pairing — the check pins nothing"
    assert "the marker/lock pairing is gone" in proc.stdout


def test_a_grouped_test_errors_naming_the_holder_when_the_bound_runs_out(tmp_path):
    p = tmp_path / ".lockprobe.lock"
    report = tmp_path / "a.json"
    a = _child(_hold_code(report, hold=0.5, name="lockprobe", wait_release=True), tmp_path)
    try:
        deadline = time.time() + 20
        while read_holder(p) is None:
            assert a.poll() is None, a.communicate()[1]
            assert time.time() < deadline, "A never took the lock"
            time.sleep(0.05)
        holder_pid = read_holder(p).pid  # see the launcher note above
        proc = _child_pytest(tmp_path, raise_=False,
                             env_extra={"MRLN_MACHINE_LOCK_TIMEOUT": "0.5"})
        assert proc.returncode != 0
        assert "1 error" in proc.stdout or "1 failed" in proc.stdout, proc.stdout[-1500:]
        assert f"pid {holder_pid}" in proc.stdout, proc.stdout[-1500:]
        assert "skipped" not in proc.stdout.splitlines()[-1], "a timeout must never skip"
        assert read_holder(p) is not None and read_holder(p).pid == holder_pid, (
            "the holder was gone before the child asked — the test proved nothing")
    finally:
        _release(report)
        a.kill()
        a.communicate()
        p.unlink(missing_ok=True)
