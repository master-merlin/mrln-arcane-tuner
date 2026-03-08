"""LoRA inspection and resize routes."""

from __future__ import annotations

import asyncio
from typing import Any

import torch
from fastapi import APIRouter, HTTPException

from app.engine.utils.lora_tools import inspect_lora, resize_lora
from app.api.schemas.lora_schemas import ResizeLoraRequest

router = APIRouter()


@router.get("/tools/lora/inspect", response_model=dict[str, Any])
async def inspect_lora_file(path: str):
    """Inspect a LoRA safetensors file: metadata, rank, alpha, key structure."""
    try:
        return await asyncio.to_thread(inspect_lora, path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"LoRA file not found: {path}")
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/tools/lora/resize", response_model=dict[str, Any])
async def resize_lora_file(request: ResizeLoraRequest):
    """Resize a LoRA's rank using SVD decomposition."""
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
