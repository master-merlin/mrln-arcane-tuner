"""Filesystem browse routes."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException


router = APIRouter()

# Allowed browsing roots — constrain all browse requests to these trees.
_ALLOWED_ROOTS: list[Path] = [
    Path(__file__).resolve().parents[2],  # backend/
    Path("outputs").resolve(),
]


@router.post("/filesystem/pick-folder")
async def pick_folder(body: dict | None = None):
    """Open a native OS folder-picker dialog and return the chosen path.

    Uses ``tkinter.filedialog.askdirectory`` which works on Windows,
    macOS, and Linux without additional dependencies.

    Optional body fields:
        initial_dir: Starting directory for the dialog (default: user home).
        title: Dialog window title (default: "Select Folder").
    """
    initial_dir = (body or {}).get("initial_dir", "")
    title = (body or {}).get("title", "Select Folder")

    def _ask() -> str:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        # Bring the dialog to the front on Windows
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            initialdir=initial_dir or None,
            title=title,
        )
        root.destroy()
        return selected or ""

    path = await asyncio.to_thread(_ask)
    return {"path": path}


@router.get("/filesystem/browse")
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

    if not resolved.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {resolved}")

    parent = resolved.parent
    entries: list[dict[str, str]] = []

    try:
        for entry in sorted(resolved.iterdir(), key=lambda e: e.name.lower()):
            if entry.is_dir():
                is_checkpoint = (entry / "training_state.json").exists()
                entries.append({
                    "name": entry.name,
                    "path": entry.as_posix(),
                    "type": "checkpoint" if is_checkpoint else "directory",
                })
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {resolved}")

    return {
        "path": resolved.as_posix(),
        "parent": parent.as_posix(),
        "entries": entries,
    }

