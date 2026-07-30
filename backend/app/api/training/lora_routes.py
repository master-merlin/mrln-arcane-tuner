"""LoRA inspection and resize routes."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException

from app.api._path_guard import validate_path_in_allowed_roots
from app.api.schemas.lora_schemas import ResizeLoraRequest

router = APIRouter()

# Shared operator-tool roots (see app/api/_path_guard.ALLOWED_FS_ROOTS) — this
# module used to carry its own CWD-dependent copy rooted at all of ``backend/``,
# which let the WRITING endpoint (/tools/lora/resize) target the database or a
# venv binary.
_check_lora_path = validate_path_in_allowed_roots


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

    # Pass the RESOLVED paths downstream, not the raw request strings: guarding
    # a path and then handing the unvalidated original to the worker relies on
    # the callee re-deriving the identical resolution.
    in_path = _check_lora_path(request.input_path)
    out_path = _check_lora_path(request.output_path)

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    save_dtype = dtype_map.get(request.save_dtype) if request.save_dtype else None

    try:
        return await asyncio.to_thread(
            resize_lora,
            str(in_path),
            str(out_path),
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
