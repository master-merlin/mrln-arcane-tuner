"""Filesystem browse routes."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter()

# ── Native folder-picker single-flight (W4.T12) ─────────────────────────
#
# tkinter is not thread-safe across concurrent `tk.Tk()` roots, and the
# dialog thread is pinned until the user closes it — an abandoned dialog (or
# several) would otherwise sit in the shared default executor that every
# other `asyncio.to_thread` call in the app draws from, starving unrelated
# API requests. A dedicated single-worker executor confines the dialog to
# its own thread, and a non-blocking lock rejects a second concurrent
# request with 409 instead of queueing it behind the first (queueing would
# just move the starvation from "shared executor" to "this executor").
_dialog_lock = threading.Lock()
_dialog_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="folder-dialog")


def _ask_for_folder(initial_dir: str, title: str) -> str:
    """Blocking: open the native OS folder-picker dialog and return the choice.

    Uses ``tkinter.filedialog.askdirectory`` which works on Windows, macOS,
    and Linux without additional dependencies. Must only ever run on
    ``_dialog_executor`` (single worker) — never the shared default executor.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    # Bring the dialog to the front on Windows
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(
            initialdir=initial_dir or None,
            title=title,
        )
    finally:
        root.destroy()
    return selected or ""


# ── Response Models ──────────────────────────────────────────────────────

class BrowseEntry(BaseModel):
    """A single directory entry from a filesystem browse listing."""

    name: str
    path: str
    type: str


class BrowseResponse(BaseModel):
    """Directory listing for the frontend folder picker."""

    path: str
    parent: str
    entries: list[BrowseEntry]

# Allowed browsing roots — constrain all browse requests to these trees.
_ALLOWED_ROOTS: list[Path] = [
    Path(__file__).resolve().parents[2],  # backend/
    Path("outputs").resolve(),
]


@router.post("/filesystem/pick-folder")
async def pick_folder(body: dict | None = None):
    """Open a native OS folder-picker dialog and return the chosen path.

    Single-flight (W4.T12): a second request while a dialog is already open
    gets an immediate 409 rather than pinning another thread — an abandoned
    dialog previously left its thread parked forever in the shared default
    executor, and tkinter is not thread-safe across concurrent roots anyway.

    Optional body fields:
        initial_dir: Starting directory for the dialog (default: user home).
        title: Dialog window title (default: "Select Folder").
    """
    if not _dialog_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Folder dialog already open")

    initial_dir = (body or {}).get("initial_dir", "")
    title = (body or {}).get("title", "Select Folder")

    def _run() -> str:
        try:
            return _ask_for_folder(initial_dir, title)
        finally:
            _dialog_lock.release()

    loop = asyncio.get_running_loop()
    path = await loop.run_in_executor(_dialog_executor, _run)
    return {"path": path}


@router.get("/filesystem/browse", response_model=BrowseResponse)
async def browse_filesystem(path: str = "outputs") -> dict:
    """List directories and checkpoint markers at a given path.

    Used by the frontend folder picker for selecting checkpoint directories.
    """
    resolved = Path(path).resolve()

    # Verify the requested path lives under at least one allowed root.
    if not any(
        resolved.is_relative_to(root) for root in _ALLOWED_ROOTS
    ):
        raise HTTPException(
            status_code=403,
            detail="Access denied: path is outside allowed directories.",
        )

    def _scan() -> tuple[bool, bool, list[dict[str, str]]]:
        """Run all blocking filesystem stats in a worker thread.

        Returns (is_dir, permission_denied, entries).
        """
        if not resolved.is_dir():
            return False, False, []
        items: list[dict[str, str]] = []
        try:
            for entry in sorted(resolved.iterdir(), key=lambda e: e.name.lower()):
                if entry.is_dir():
                    is_checkpoint = (entry / "training_state.json").exists()
                    items.append({
                        "name": entry.name,
                        "path": entry.as_posix(),
                        "type": "checkpoint" if is_checkpoint else "directory",
                    })
        except PermissionError:
            return True, True, []
        return True, False, items

    is_dir, permission_denied, entries = await asyncio.to_thread(_scan)
    if not is_dir:
        raise HTTPException(status_code=404, detail=f"Directory not found: {resolved}")
    if permission_denied:
        raise HTTPException(status_code=403, detail=f"Permission denied: {resolved}")

    return {
        "path": resolved.as_posix(),
        "parent": resolved.parent.as_posix(),
        "entries": entries,
    }

