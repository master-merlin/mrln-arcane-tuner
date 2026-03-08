"""LoRA tooling schemas."""

from __future__ import annotations

from pydantic import BaseModel


class ResizeLoraRequest(BaseModel):
    """Request body for LoRA rank resize."""
    input_path: str
    output_path: str
    new_rank: int
    new_alpha: float | None = None
    save_dtype: str | None = None
