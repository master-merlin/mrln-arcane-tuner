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
import uuid
from concurrent.futures import ThreadPoolExecutor
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


def test_recovery_leaves_only_the_turn_file_and_that_file_grants_nothing(tmp_path):
    """The recovery turn is itself a file. It is created once and never removed
    (removing it is the window VERIFY 2.02 closed), so the residue of a
    recovery is exactly that one file — and its EXISTENCE must grant nobody
    anything: the very next recovery, with the file lying right there, works."""
    p = tmp_path / ".probe.lock"
    stale = json.dumps({"pid": DEAD_PID, "worker": None, "tree": "x",
                        "since": "2026-01-01T00:00:00+00:00"})
    p.write_text(stale)
    with machine_lock("probe", path=p, timeout_s=5.0):
        pass
    assert not p.exists()
    turn = machine_lock_mod.recovery_sentinel(p)
    assert sorted(q.name for q in tmp_path.iterdir()) == [turn.name], \
        list(tmp_path.iterdir())
    p.write_text(stale)  # a second stale lock, the turn file still present
    with machine_lock("probe", path=p, timeout_s=5.0):
        assert read_holder(p).pid == os.getpid(), "the leftover turn file blocked us"


def test_a_live_recovery_turn_keeps_a_second_waiter_out_of_the_unlink(tmp_path):
    """The serialisation itself, seen: while another PROCESS holds the recovery
    turn, this waiter may not remove the stale lock — only ONE process is ever
    between a judgement and an unlink, which is what makes the identity check
    conclusive. It waits its bound instead."""
    p = tmp_path / ".probe.lock"
    stale = json.dumps({"pid": DEAD_PID, "worker": None, "tree": "x",
                        "since": "2026-01-01T00:00:00+00:00"})
    p.write_text(stale)
    ready, release = tmp_path / "turn.ready", tmp_path / "turn.release"
    child = _turn_holder(p, ready, release)
    try:
        _await(ready, child, "the child never took the recovery turn")
        with pytest.raises(MachineLockTimeout):
            with machine_lock("probe", path=p, timeout_s=0.5):
                pass
        assert p.read_text() == stale, "recovered a lock while another turn was open"
    finally:
        release.write_text("go")
        child.communicate(timeout=60)


_TURN = """
import os, sys, time
from pathlib import Path
from tests.support import machine_lock as m
with m._recovery_turn(Path(r'{lock}')) as mine:
    if not mine:
        sys.exit(7)
    Path(r'{ready}').write_text(str(os.getpid()))
    deadline = time.time() + 120
    while not Path(r'{release}').exists() and time.time() < deadline:
        time.sleep(0.05)
"""


_TURN_CRASH = """
import os, sys
from pathlib import Path
from tests.support import machine_lock as m
with m._recovery_turn(Path(r'{lock}')) as mine:
    if not mine:
        sys.exit(7)
    Path(r'{ready}').write_text(str(os.getpid()))
    os._exit(3)   # a recoverer that dies mid-turn: no finally, no cleanup
"""


def _turn_holder(lock: Path, ready: Path, release: Path) -> subprocess.Popen:
    """A REAL second process holding the recovery turn for *lock*."""
    return _child(_TURN.format(lock=lock, ready=ready, release=release), lock.parent)


def _await(path: Path, proc: subprocess.Popen, what: str, bound: float = 30.0) -> None:
    deadline = time.time() + bound
    while not path.exists():
        assert proc.poll() is None, f"{what}: child died: {proc.communicate()[1][-2000:]}"
        assert time.time() < deadline, what
        time.sleep(0.05)


def test_an_aged_turn_whose_holder_is_alive_is_never_taken_from_it(tmp_path):
    """VERIFY 2.02: the turn may only be granted to ONE process at a time, and
    that must not hinge on how old the turn's file looks. An age check followed
    by an unlink of the PATH hands the turn to a second recoverer (and deletes a
    file it never proved it owned) whenever the holder is merely slow — so the
    inspection and the unlink of the stale lock stop being mutually exclusive.
    Here a live child holds the turn while its file is back-dated far past the
    grace: the turn must stay the child's, and the stale lock must survive.
    """
    p = tmp_path / ".probe.lock"
    stale = json.dumps({"pid": DEAD_PID, "worker": None, "tree": "x",
                        "since": "2026-01-01T00:00:00+00:00"})
    p.write_text(stale)
    ready, release = tmp_path / "turn.ready", tmp_path / "turn.release"
    child = _turn_holder(p, ready, release)
    try:
        _await(ready, child, "the child never took the recovery turn")
        turn = machine_lock_mod.recovery_sentinel(p)
        old = time.time() - 10 * machine_lock_mod.UNREADABLE_GRACE_S
        os.utime(turn, (old, old))  # a live holder that merely LOOKS like a corpse
        with pytest.raises(MachineLockTimeout):
            with machine_lock("probe", path=p, timeout_s=1.0):
                pass
        assert p.read_text() == stale, (
            "recovered the lock while another process held the turn")
        assert turn.exists(), "deleted a turn file this process did not own"
        assert turn.read_text().strip() == ready.read_text().strip(), (
            "the live holder's turn was replaced by ours — two recoverers at once")
    finally:
        release.write_text("go")
        child.communicate(timeout=60)
        assert child.returncode == 0, "the child never held the turn it reported"


def test_a_turn_left_by_a_crashed_recoverer_does_not_block_recovery(tmp_path):
    """Prove the negative for the serialisation itself: a turn orphaned by a
    recoverer that DIED must not make the lock unrecoverable forever. The
    corpse is real here — the child is killed while holding the turn — because
    that, not the age of a file, is what the kernel lock reacts to."""
    p = tmp_path / ".probe.lock"
    p.write_text(json.dumps({"pid": DEAD_PID, "worker": None, "tree": "x",
                             "since": "2026-01-01T00:00:00+00:00"}))
    ready = tmp_path / "turn.ready"
    # os._exit inside the turn: no finally, no unlock, no cleanup — the process
    # simply stops existing, which is the only corpse worth testing. (Killing
    # the Popen would not do: on Windows sys.executable is a launcher and the
    # process that actually holds the turn is its child.)
    child = _child(_TURN_CRASH.format(lock=p, ready=ready), tmp_path)
    _await(ready, child, "the child never took the recovery turn")
    child.communicate(timeout=60)
    assert child.returncode == 3, "the child did not crash inside its turn"
    turn = machine_lock_mod.recovery_sentinel(p)
    assert turn.exists(), "the crash cleared the turn file — nothing was orphaned"
    with machine_lock("probe", path=p, timeout_s=10.0):
        assert read_holder(p).pid == os.getpid()


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
    report = os.environ.get("MRLN_LOCK_PROBE_REPORT")
    if report:  # what THIS child pytest is, the log it writes, and a line in it
        import logging

        from tests.support.worker_paths import current_worker, worker_suffixed

        worker = current_worker()
        logging.getLogger("tests.lockprobe").info("lockprobe ran as %s", worker)
        Path(report).write_text(json.dumps({
            "worker": worker,
            "log": str(worker_suffixed(TESTS / "tests.log")),
        }))
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


def _child_worker_id() -> str:
    """A worker id no other process can be using.

    Invented (never an xdist ``gwN``) so the child's logs stay off the serial
    names the parent pytest may be holding open, and UNIQUE per child because
    the tests below are ungrouped: under ``-n 4`` two of them run at once, and
    one shared id means one ``tests-<id>.log`` that each child truncates on its
    way in (VERIFY 2.03).
    """
    return f"gwLock{uuid.uuid4().hex[:8]}"


def _child_pytest(lock_dir_: Path, *extra: str, raise_: bool,
                  env_extra: dict | None = None,
                  nodes: tuple[str, ...] = (_PROBE_1,),
                  keep_log: bool = False) -> subprocess.CompletedProcess:
    env = dict(os.environ, **{LOCK_DIR_ENV: str(lock_dir_)}, **(env_extra or {}))
    env["PYTEST_XDIST_WORKER"] = worker = _child_worker_id()
    if raise_:
        env["MRLN_LOCK_PROBE_RAISE"] = "1"
    try:
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *extra,
             *(f"{__file__}::{node}" for node in nodes)],
            cwd=str(BACKEND), env=env, capture_output=True, text=True, timeout=300)
    finally:
        if not keep_log:  # a unique id per child means a file per child to clear
            (TESTS / f"tests-{worker}.log").unlink(missing_ok=True)


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


def test_two_child_pytests_never_share_a_worker_identity_or_a_log_file(tmp_path):
    """VERIFY 2.03: the tests that launch a child pytest are ungrouped, so under
    ``-n 4`` two of them run at the same time in different workers. A fixed
    ``PYTEST_XDIST_WORKER`` for every child means one ``tests-<id>.log`` for all
    of them — and the session fixture RESETS that file, so concurrent children
    truncate each other's log. Observable: two children run at once, each names
    itself and the log it wrote; the two must differ and both must survive.
    """
    reports = [tmp_path / "r0.json", tmp_path / "r1.json"]
    dirs = [tmp_path / "d0", tmp_path / "d1"]
    for d in dirs:
        d.mkdir()

    def run(i: int) -> subprocess.CompletedProcess:
        return _child_pytest(dirs[i], raise_=False, keep_log=True,
                             env_extra={"MRLN_LOCK_PROBE_REPORT": str(reports[i])})

    with ThreadPoolExecutor(max_workers=2) as pool:
        procs = [f.result() for f in [pool.submit(run, 0), pool.submit(run, 1)]]
    for proc in procs:
        assert proc.returncode == 0, proc.stdout[-2000:]
    said = [json.loads(r.read_text()) for r in reports]
    logs = [Path(d["log"]) for d in said]
    try:
        assert said[0]["worker"] != said[1]["worker"], (
            f"both children ran as {said[0]['worker']} — one identity, one log")
        assert logs[0] != logs[1], f"both children wrote {logs[0]}"
        for mine, other in ((0, 1), (1, 0)):
            text = logs[mine].read_text(encoding="utf-8")
            assert f"lockprobe ran as {said[mine]['worker']}" in text, (
                f"{logs[mine]} lost its own line — the other child reset it")
            assert said[other]["worker"] not in text, (
                f"{logs[mine]} carries the other child's session")
    finally:
        for log in logs:
            log.unlink(missing_ok=True)


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
