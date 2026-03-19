"""LoRA inspection and resize routes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.schemas.lora_schemas import ResizeLoraRequest

router = APIRouter()

# Allowed roots for LoRA file access
_ALLOWED_ROOTS: list[Path] = [
    Path(__file__).resolve().parents[3],  # backend/
    Path("outputs").resolve(),
]


def _check_lora_path(raw_path: str) -> Path:
    """Resolve and validate a LoRA file path."""
    resolved = Path(raw_path).resolve()
    if not any(resolved.is_relative_to(root) for root in _ALLOWED_ROOTS):
        raise HTTPException(
            status_code=403,
            detail="Access denied: path is outside allowed directories.",
        )
    return resolved


@router.get("/tools/lora/inspect", response_model=dict[str, Any])
async def inspect_lora_file(path: str):
    """Inspect a LoRA safetensors file: metadata, rank, alpha, key structure."""
    from app.engine.utils.lora_tools import inspect_lora

    resolved = _check_lora_path(path)
    try:
        return await asyncio.to_thread(inspect_lora, str(resolved))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"LoRA file not found: {path}")
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/tools/lora/resize", response_model=dict[str, Any])
async def resize_lora_file(request: ResizeLoraRequest):
    """Resize a LoRA's rank using SVD decomposition."""
    import torch
    from app.engine.utils.lora_tools import resize_lora

    _check_lora_path(request.input_path)
    _check_lora_path(request.output_path)

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    save_dtype = dtype_map.get(request.save_dtype) if request.save_dtype else None

    try:
        return await asyncio.to_thread(
            resize_lora,
            request.input_path,
            request.output_path,
            request.new_rank,
            request.new_alpha,
            save_dtype,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"LoRA file not found: {request.input_path}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (RuntimeError, MemoryError) as e:
        raise HTTPException(status_code=500, detail=f"Resize failed: {e}")
