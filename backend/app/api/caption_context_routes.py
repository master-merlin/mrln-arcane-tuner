# backend/app/api/caption_context_routes.py
"""Endpoints backing the model-context selector + caption token counter."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from app.api.schemas.caption_context_schemas import (
    DefinitionRef,
    TokenCountRequest,
    TokenCountResponse,
)
from app.core.captioning.tokenizer_service import TokenizerService
from app.core.logger import get_logger
from app.engine.core.caption_target import resolve_caption_target

router = APIRouter(prefix="/api/caption-context", tags=["caption-context"])
logger = get_logger(__name__)


@router.get("/definitions", response_model=list[DefinitionRef])
async def list_definitions() -> list[DefinitionRef]:
    """List registered model definitions (id, family, name) for the selector."""
    from app.engine.models.registry import registry

    out: list[DefinitionRef] = []
    for def_id in registry.list_models():
        defn = registry.get_definition(def_id)
        if defn is not None:
            out.append(DefinitionRef(id=defn.id, family=defn.family, name=defn.name))
    return out


@router.post("/token-count", response_model=TokenCountResponse)
async def token_count(req: TokenCountRequest) -> TokenCountResponse:
    """Count caption tokens for a definition's tokenizer + report the cutoff."""
    try:
        target = resolve_caption_target(req.definition_id)
    except ValueError:
        raise HTTPException(404, f"unknown definition: {req.definition_id}")

    tokens, cutoff = await asyncio.to_thread(
        TokenizerService.get_instance().count_with_cutoff, req.text, target
    )
    return TokenCountResponse(
        tokens=tokens,
        limit=target.usable_limit,
        will_truncate=cutoff is not None,
        cutoff_char_index=cutoff,
    )
