"""Checkpoint inspection routes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from app.engine.utils.checkpoint_inspector import inspect_checkpoint

router = APIRouter()

# Allowed roots for checkpoint inspection
_ALLOWED_ROOTS: list[Path] = [
    Path(__file__).resolve().parents[3],  # backend/
    Path("outputs").resolve(),
]


@router.get("/checkpoints/inspect", response_model=dict[str, Any])
async def inspect_checkpoint_route(path: str):
    """Inspect a checkpoint's metadata without loading weights."""
    resolved = Path(path).resolve()

    # Verify the path stays within allowed directories
    if not any(resolved.is_relative_to(root) for root in _ALLOWED_ROOTS):
        raise HTTPException(
            status_code=403,
            detail="Access denied: path is outside allowed directories.",
        )
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {path}")

    return await asyncio.to_thread(inspect_checkpoint, str(resolved))
