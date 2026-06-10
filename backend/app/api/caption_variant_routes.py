# backend/app/api/caption_variant_routes.py
"""Routes over per-definition caption variants + pending suggestions (accept/reject)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.captioning import caption_suggestions as sg
from app.core.captioning import caption_variants as cv
from app.core.dataset_manager import dataset_manager
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


class SuggestionItem(BaseModel):
    stem: str
    suggestion: str
    current: str


class SuggestionsResponse(BaseModel):
    definition_id: str
    items: list[SuggestionItem]


class AcceptRejectRequest(BaseModel):
    definition_id: str
    stem: str
    masked: bool = False


class AcceptAllRequest(BaseModel):
    definition_id: str
    masked: bool = False


def _ds_path(name: str) -> str:
    ds = dataset_manager.get_dataset(name)
    if ds is None:
        raise HTTPException(404, f"Dataset '{name}' not found.")
    return ds.path


@router.get("/datasets/{name}/caption-variants")
async def list_variants(name: str) -> dict:
    path = _ds_path(name)
    ids = await asyncio.to_thread(cv.list_variant_definition_ids, path)
    return {"definition_ids": ids}


@router.get("/datasets/{name}/caption-suggestions", response_model=SuggestionsResponse)
async def list_suggestions(name: str, definition_id: str, masked: bool = False) -> SuggestionsResponse:
    path = _ds_path(name)

    def _collect() -> list[dict]:
        out: list[dict] = []
        for stem in sg.list_suggestion_stems(path, definition_id, masked):
            out.append({
                "stem": stem,
                "suggestion": sg.read_suggestion(path, definition_id, stem, masked) or "",
                "current": cv.resolve_caption(path, stem, definition_id, masked),
            })
        return out

    items = await asyncio.to_thread(_collect)
    return SuggestionsResponse(definition_id=definition_id, items=items)


@router.post("/datasets/{name}/caption-suggestions/accept")
async def accept(name: str, req: AcceptRejectRequest) -> dict:
    path = _ds_path(name)
    await asyncio.to_thread(sg.accept_suggestion, path, req.definition_id, req.stem, req.masked)
    return {"status": "accepted"}


@router.post("/datasets/{name}/caption-suggestions/reject")
async def reject(name: str, req: AcceptRejectRequest) -> dict:
    path = _ds_path(name)
    await asyncio.to_thread(sg.reject_suggestion, path, req.definition_id, req.stem, req.masked)
    return {"status": "rejected"}


@router.post("/datasets/{name}/caption-suggestions/accept-all")
async def accept_all(name: str, req: AcceptAllRequest) -> dict:
    path = _ds_path(name)

    def _accept_all() -> int:
        stems = sg.list_suggestion_stems(path, req.definition_id, req.masked)
        for stem in stems:
            sg.accept_suggestion(path, req.definition_id, stem, req.masked)
        return len(stems)

    n = await asyncio.to_thread(_accept_all)
    return {"accepted": n}
