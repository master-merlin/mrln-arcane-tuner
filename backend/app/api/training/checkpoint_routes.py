"""Checkpoint inspection routes."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter

from app.engine.utils.checkpoint_inspector import inspect_checkpoint

router = APIRouter()


@router.get("/checkpoints/inspect", response_model=dict[str, Any])
async def inspect_checkpoint_route(path: str):
    """Inspect a checkpoint's metadata without loading weights."""
    return await asyncio.to_thread(inspect_checkpoint, path)
