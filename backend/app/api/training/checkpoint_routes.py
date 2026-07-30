"""Checkpoint inspection routes."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException

from app.api._path_guard import validate_path_in_allowed_roots
from app.engine.utils.checkpoint_inspector import inspect_checkpoint

router = APIRouter()


@router.get("/checkpoints/inspect", response_model=dict[str, Any])
async def inspect_checkpoint_route(path: str):
    """Inspect a checkpoint's metadata without loading weights."""
    # Shared operator-tool roots (see app/api/_path_guard.ALLOWED_FS_ROOTS) —
    # this module used to carry its own CWD-dependent copy of the list.
    resolved = validate_path_in_allowed_roots(path)
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {path}")

    return await asyncio.to_thread(inspect_checkpoint, str(resolved))
