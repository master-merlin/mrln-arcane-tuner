"""Filesystem browse routes."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/filesystem/browse")
async def browse_filesystem(path: str = "outputs") -> dict[str, Any]:
    """List directories and checkpoint markers at a given path.

    Used by the frontend folder picker for selecting checkpoint directories.
    """
    resolved = os.path.abspath(path)

    if not os.path.isdir(resolved):
        raise HTTPException(status_code=404, detail=f"Directory not found: {resolved}")

    parent = os.path.dirname(resolved)
    entries: list[dict[str, str]] = []

    try:
        for entry in sorted(os.scandir(resolved), key=lambda e: e.name.lower()):
            if entry.is_dir():
                is_checkpoint = os.path.exists(
                    os.path.join(entry.path, "training_state.json"),
                )
                entries.append({
                    "name": entry.name,
                    "path": entry.path.replace("\\", "/"),
                    "type": "checkpoint" if is_checkpoint else "directory",
                })
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {resolved}")

    return {
        "path": resolved.replace("\\", "/"),
        "parent": parent.replace("\\", "/"),
        "entries": entries,
    }
