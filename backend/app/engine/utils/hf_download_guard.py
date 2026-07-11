"""Killable, stall-detected, retried HuggingFace Hub downloads.

WHY: ``snapshot_download``/``hf_hub_download`` run in-process with no stall
detection. A wedged socket/SSL/proxy read blocks indefinitely, and Python
threads cannot be killed — a hung in-process download is un-abortable
regardless of ``HF_HUB_DOWNLOAD_TIMEOUT`` edge cases. This module instead
runs the actual transfer in a CHILD PROCESS (``hf_fetch_worker``), so a
stalled attempt CAN be killed outright, watches on-disk cache growth as the
stall signal, and retries a bounded number of times before raising loudly.

TWO CALLERS, ONE PRIMARY PATH: this guard is reached from
``ModelPathResolver._resolve_hf`` in two different processes —
  1. The API process, via ``job_manager._preflight_download`` (best-effort;
     failures there are swallowed, but the job's status_label is updated so
     the UI doesn't linger — see ``job_manager._preflight_download``).
  2. The DETACHED TRAINER subprocess. THIS is the primary path: the
     originally reported stall lived here — the trainer survives backend
     restarts (crash recovery re-attaches to its PID), so a hang inside it is
     invisible and permanent without this guard. The trainer has NO event
     loop and NO WS bridge (``download_progress.schedule_emit_from_thread``
     is a no-op there). This module must not assume either exists — and it
     doesn't: it never calls any WS-emit helper, only the pure
     filesystem-probing helpers (``_repo_cache_dir`` / ``_on_disk_bytes``,
     plain ``os.scandir`` — no asyncio, no loop lookup). The progress-UX
     wrapper (``with_progress`` / ``snapshot_byte_progress``) lives in the
     PARENT (``_resolve_hf``), one layer further out, and is a no-op-safe
     best-effort layer independent of this module.

KNOWN, ACCEPTED REGRESSION: with the transfer happening in a child process,
the parent's ``_capture_per_file`` hook (per-file byte breakdown for the
top-bar's file list) can't observe the child's tqdm — the ``files`` list in
progress payloads is empty for a guarded download. Aggregate bytes/percent
still work correctly because those come from polling the filesystem
(``_on_disk_bytes``), not from tqdm state.

XET FINDING (checked against the installed ``hf_xet`` package + the
``huggingface_hub`` 0.36 source): ``hf_xet``'s in-flight chunks are written to
the SAME ``<blob>.incomplete`` path inside the repo's ``blobs/`` dir as the
classic HTTP downloader —
``huggingface_hub.file_download.xet_get`` receives
``incomplete_path=Path(blob_path + ".incomplete")`` from the exact same
caller as the classic ``http_get`` path (see ``file_download.py`` around the
``_get_metadata_or_catch_error`` call sites). The stall watchdog's on-disk
probe (which scans ``blobs/``) therefore observes xet growth too — no need to
force ``HF_HUB_DISABLE_XET=1``.

Design: ``spawn_fn`` / ``probe_bytes_fn`` / ``poll_interval_s`` / ``clock``
are all injectable with production defaults, so tests exercise the
retry/stall state machine against real (but tiny) child processes instead of
mocking ``subprocess`` — see ``app/engine/tests/test_hf_download_guard.py``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import structlog

logger = structlog.get_logger(__name__)

# ── Config knobs ────────────────────────────────────────────────────────
# Env-overridable, read at CALL time (not import time) so tests can tweak
# them per-test via monkeypatch.setenv without import-order games.

DEFAULT_STALL_TIMEOUT_S = 180.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_S: tuple[float, ...] = (5.0, 15.0)
DEFAULT_POLL_INTERVAL_S = 1.0

_ENV_STALL_TIMEOUT_S = "MRLN_HF_STALL_TIMEOUT_S"
_ENV_MAX_ATTEMPTS = "MRLN_HF_DOWNLOAD_ATTEMPTS"


def _env_stall_timeout_s() -> float:
    try:
        return float(os.environ.get(_ENV_STALL_TIMEOUT_S, DEFAULT_STALL_TIMEOUT_S))
    except (TypeError, ValueError):
        return DEFAULT_STALL_TIMEOUT_S


def _env_max_attempts() -> int:
    try:
        return int(os.environ.get(_ENV_MAX_ATTEMPTS, DEFAULT_MAX_ATTEMPTS))
    except (TypeError, ValueError):
        return DEFAULT_MAX_ATTEMPTS


@dataclass
class _AttemptOutcome:
    ok: bool
    path: Optional[str] = None
    reason: str = ""  # "stalled" | "error" — meaningful only when not ok
    detail: str = ""


def _backend_root() -> str:
    """Backend root dir, derived from ``__file__`` so the worker command
    resolves whether the guard runs in the API process or a detached trainer
    subprocess launched with a different cwd (mirrors
    ``StandardPlugin.start_training``'s own ``backend_root`` derivation)."""
    current_file = os.path.abspath(__file__)
    # up 4: utils -> engine -> app -> backend
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    )


def _default_child_env() -> dict:
    """Child env for the worker subprocess.

    The ``os.environ.setdefault(...)`` calls at the top of ``model_utils.py``
    (and ``hf_fetch_worker.py``) only take effect IN the process that runs
    them — a fresh child does not inherit a parent module's import-time side
    effects, only the environment dict we hand it. So we set the same
    Windows-symlink workaround here too (defense-in-depth: the worker module
    also sets these itself before importing ``huggingface_hub``), plus a
    defensive default download timeout and a quiet stdout (only the resolved
    path should appear there — see ``hf_fetch_worker``'s stdout contract).
    """
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    env.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    return env


def _make_default_spawn(
    repo_id: str, filename: Optional[str], revision: Optional[str],
) -> Callable[[], "subprocess.Popen[str]"]:
    """Build the production ``spawn_fn``: a fresh ``hf_fetch_worker`` child
    per call, so a retried attempt gets a brand-new process."""
    payload = json.dumps({
        "repo_id": repo_id, "filename": filename, "revision": revision,
    })
    backend_root = _backend_root()

    def _spawn() -> "subprocess.Popen[str]":
        cmd = [sys.executable, "-u", "-m", "app.engine.utils.hf_fetch_worker"]
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW
        proc = subprocess.Popen(
            cmd,
            cwd=backend_root,
            env=_default_child_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creationflags,
        )
        try:
            assert proc.stdin is not None
            proc.stdin.write(payload)
            proc.stdin.close()
        except Exception:
            # A failed stdin write shouldn't crash the guard — the worker
            # will fail fast on empty/partial stdin and the normal
            # error-and-retry path below handles it.
            pass
        return proc

    return _spawn


def _make_default_probe(repo_id: str) -> Callable[[], int]:
    """Build the production 'cache-dir prober': on-disk bytes for *repo_id*,
    reused from the download-progress poller (same signal, same directory)."""
    from app.api.events.download_progress import _on_disk_bytes, _repo_cache_dir

    cache_dir = _repo_cache_dir(repo_id)

    def _probe() -> int:
        return _on_disk_bytes(cache_dir)

    return _probe


def _kill(proc: "subprocess.Popen[str]") -> None:
    """Terminate then kill. On Windows both calls invoke ``TerminateProcess``
    (releasing file locks immediately); on POSIX ``terminate`` (SIGTERM) is
    tried first and ``kill`` (SIGKILL) follows only if the process is still
    alive after a short grace wait."""
    for action in (proc.terminate, proc.kill):
        try:
            action()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
            return
        except Exception:
            continue


def _run_one_attempt(
    *,
    spawn: Callable[[], "subprocess.Popen[str]"],
    probe: Callable[[], int],
    stall_timeout_s: float,
    poll_interval_s: float,
    clock: Callable[[], float],
) -> _AttemptOutcome:
    """Spawn one child, watch it to completion or stall-kill it.

    Progress = bytes grew since the last sample, or the child exited (a
    natural exit is handled below, not treated as a stall). The stall timer
    resets on ANY byte growth — metadata/etag phases that write nothing yet
    are covered simply by ``stall_timeout_s`` being generous (default 180s).
    """
    proc = spawn()
    try:
        last_bytes = probe()
    except Exception:
        last_bytes = 0
    last_growth_t = clock()

    try:
        while True:
            if proc.poll() is not None:
                break
            now = clock()
            try:
                cur_bytes = probe()
            except Exception:
                cur_bytes = last_bytes
            if cur_bytes > last_bytes:
                last_bytes = cur_bytes
                last_growth_t = now
            elif now - last_growth_t >= stall_timeout_s:
                _kill(proc)
                try:
                    proc.communicate(timeout=5)
                except Exception:
                    pass
                return _AttemptOutcome(
                    ok=False, reason="stalled", detail="no on-disk growth",
                )
            time.sleep(poll_interval_s)

        try:
            stdout, stderr = proc.communicate(timeout=10)
        except Exception:
            stdout, stderr = "", ""

        if proc.returncode == 0:
            lines = [ln for ln in (stdout or "").splitlines() if ln.strip()]
            if not lines:
                return _AttemptOutcome(
                    ok=False, reason="error",
                    detail="worker exited 0 but printed no path",
                )
            return _AttemptOutcome(ok=True, path=lines[-1].strip())

        detail = (stderr or stdout or "").strip() or f"exit code {proc.returncode}"
        return _AttemptOutcome(ok=False, reason="error", detail=detail)
    finally:
        # Belt-and-braces: never leak a child if something above raised.
        if proc.poll() is None:
            _kill(proc)


def download_with_stall_guard(
    *,
    repo_id: str,
    filename: Optional[str] = None,
    revision: Optional[str] = None,
    stall_timeout_s: Optional[float] = None,
    max_attempts: Optional[int] = None,
    backoff_s: Sequence[float] = DEFAULT_BACKOFF_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    clock: Callable[[], float] = time.monotonic,
    spawn_fn: Optional[Callable[[], "subprocess.Popen[str]"]] = None,
    probe_bytes_fn: Optional[Callable[[], int]] = None,
) -> str:
    """Download *repo_id* (a full snapshot, or a single *filename*) killably.

    Runs the actual ``snapshot_download``/``hf_hub_download`` call in a child
    process (``hf_fetch_worker``) so a wedged transfer CAN be aborted —
    Python threads cannot be killed, so an in-process call is un-abortable.
    Polls on-disk cache growth (``probe_bytes_fn``) as the progress signal; a
    child with no growth for ``stall_timeout_s`` is killed and the attempt
    retried (up to ``max_attempts``, with a short backoff between attempts).
    A killed attempt loses only the in-flight chunk — the next attempt's
    ``snapshot_download``/``hf_hub_download`` re-checks etags and resumes
    from partial blobs (the cache self-heals across attempts).

    ``stall_timeout_s`` / ``max_attempts`` default from the
    ``MRLN_HF_STALL_TIMEOUT_S`` / ``MRLN_HF_DOWNLOAD_ATTEMPTS`` env vars (read
    at CALL time, not import time) when omitted.

    ``spawn_fn`` / ``probe_bytes_fn`` / ``poll_interval_s`` / ``clock`` are
    injectable (production defaults: a real ``hf_fetch_worker`` child +
    on-disk cache-dir probing + 1s polling + ``time.monotonic``) — tests
    supply tiny stub child processes and fake probes to exercise the retry
    and stall-detection state machine quickly and without network.

    Raises ``RuntimeError`` naming the repo, attempts, and (for a stall) the
    timeout once every attempt is exhausted.
    """
    timeout = stall_timeout_s if stall_timeout_s is not None else _env_stall_timeout_s()
    attempts_total = max_attempts if max_attempts is not None else _env_max_attempts()
    if attempts_total < 1:
        attempts_total = 1

    spawn = spawn_fn or _make_default_spawn(repo_id, filename, revision)
    probe = probe_bytes_fn or _make_default_probe(repo_id)

    last_reason = "error"
    last_detail = ""
    for attempt in range(1, attempts_total + 1):
        logger.info(
            "hf_guarded_download_attempt", repo=repo_id, filename=filename,
            attempt=attempt, max_attempts=attempts_total,
        )
        outcome = _run_one_attempt(
            spawn=spawn, probe=probe, stall_timeout_s=timeout,
            poll_interval_s=poll_interval_s, clock=clock,
        )
        if outcome.ok:
            return outcome.path  # type: ignore[return-value]

        last_reason, last_detail = outcome.reason, outcome.detail
        logger.warning(
            "hf_guarded_download_attempt_failed", repo=repo_id, filename=filename,
            attempt=attempt, max_attempts=attempts_total,
            reason=last_reason, detail=last_detail[:300],
        )
        if attempt < attempts_total and backoff_s:
            wait_s = backoff_s[min(attempt - 1, len(backoff_s) - 1)]
            if wait_s:
                time.sleep(wait_s)

    if last_reason == "stalled":
        raise RuntimeError(
            f"HF download stalled: no progress for {timeout:.0f}s on attempt "
            f"{attempts_total}/{attempts_total} for '{repo_id}' — check "
            "network/proxy; partial cache is preserved and resumes on retry"
        )
    raise RuntimeError(
        f"HF download failed for '{repo_id}' after {attempts_total} attempts: "
        f"{last_detail}"
    )
