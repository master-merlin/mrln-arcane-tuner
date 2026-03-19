"""Filesystem browse routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.api._path_guard import validate_path_within

router = APIRouter()

# Allowed browsing roots — constrain all browse requests to these trees.
_ALLOWED_ROOTS: list[Path] = [
    Path(__file__).resolve().parents[2],  # backend/
    Path("outputs").resolve(),
]


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
            detail=f"Access denied: path is outside allowed directories.",
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
