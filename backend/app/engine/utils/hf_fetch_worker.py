"""Child-process worker for ``hf_download_guard``.

Runs exactly ONE HuggingFace Hub download (snapshot or single file) and
prints the resolved local path as the LAST line of stdout on success (exit
0). Any error is written to stderr with a nonzero exit code. Invoked as::

    sys.executable -m app.engine.utils.hf_fetch_worker

with a JSON payload on stdin: ``{"repo_id": ..., "filename": ..., "revision": ...}``
(``filename``/``revision`` may be ``null``/absent).

This process is the KILLABLE half of the stall guard
(``hf_download_guard.download_with_stall_guard``): the parent watches
on-disk cache growth and kills this process outright on a stall — Python
threads and in-process socket reads cannot be aborted, only a separate OS
process can be. It is deliberately minimal: no event loop, no WS progress
emits. This matters because the guard's PRIMARY caller is the detached
trainer subprocess (``ModelPathResolver._resolve_hf`` there has no app loop
either), so this worker — one process further removed — must not assume an
app loop exists any more than its parent does. Progress UX (the top-bar
download indicator) is entirely the PARENT's job: ``with_progress`` /
``snapshot_byte_progress`` wrap the guard call and poll the filesystem, which
only needs this process's byte-writes on disk, nothing from this process
directly.
"""
from __future__ import annotations

import json
import os
import sys

# Must be set BEFORE the huggingface_hub import below — huggingface_hub.
# constants reads these as module-level constants at IMPORT time, so setting
# them after import is a no-op. This is a fresh child process (no earlier
# huggingface_hub import already happened here), so — unlike the mixed
# ordering in model_utils.py, which only matters for the first import
# anywhere in the long-lived API/trainer process — order is load-bearing here.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402


def run_download(payload: dict) -> str:
    """Perform the HF Hub call described by *payload*. Raises on failure —
    ``main()`` turns any exception into a stderr line + exit 1, which the
    parent guard treats as a retryable attempt failure."""
    repo_id = payload["repo_id"]
    filename = payload.get("filename")
    revision = payload.get("revision")
    # Passed conditionally so a revision-less call keeps the legacy kwargs
    # shape (mirrors ModelPathResolver._resolve_hf).
    rev_kwargs = {"revision": revision} if revision else {}

    if filename:
        return hf_hub_download(repo_id=repo_id, filename=filename, **rev_kwargs)
    return snapshot_download(repo_id=repo_id, **rev_kwargs)


def main() -> int:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
        path = run_download(payload)
    except Exception as e:
        # Any failure (bad JSON, missing repo_id, HF Hub error) is surfaced
        # the same way: stderr + exit 1. The parent guard doesn't distinguish
        # failure sub-kinds, only stalled-vs-errored (via the stall watchdog
        # vs. this exit code), so a single catch-all keeps the contract simple.
        # safety-net print: stderr IS this worker's error channel — the parent
        # guard reads it into the retry/raise message; no logger is configured.
        print(f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 1

    # Contract with the parent guard: the resolved path is the LAST line of
    # stdout. Nothing else should be printed to stdout by this process
    # (HF_HUB_DISABLE_PROGRESS_BARS is set by the parent's child env for
    # exactly this reason).
    # safety-net print: stdout IS the worker→guard result protocol (last line
    # = resolved path); routing through a logger would break the contract.
    print(path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
