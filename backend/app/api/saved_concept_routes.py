"""Saved Concept API — CRUD and scope management for custom masking concepts."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.db.repositories.saved_concept_repo import SavedConceptRepository
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)
repo = SavedConceptRepository()


class SavedConceptCreate(BaseModel):
    name: str
    points: Any = None
    project_id: str | None = None
    model_id: str | None = "sam3"


class SavedConceptUpdate(BaseModel):
    name: str | None = None
    points: Any = None


class ScopeMoveRequest(BaseModel):
    project_id: str | None = None


@router.get("/projects/current/concepts")
@router.get("/projects/{project_id}/concepts")
def list_concepts(project_id: str | None = None):
    """List concepts visible to a project (global + project-specific).
    If project_id is current or not provided, we just list global.
    """
    if project_id == "current":
        project_id = None
    return repo.list_for_project(project_id)


@router.post("/concepts")
def create_concept(request: SavedConceptCreate):
    """Create a new saved concept."""
    try:
        return repo.create(request.model_dump())
    except Exception as e:
        logger.error("create_concept_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/concepts/{concept_id}")
def update_concept(concept_id: str, request: SavedConceptUpdate):
    """Update a saved concept."""
    updates = request.model_dump(exclude_unset=True)
    if not updates:
        return repo.get_by_id(concept_id)
        
    try:
        result = repo.update(concept_id, updates)
        if not result:
            raise HTTPException(status_code=404, detail="Concept not found")
        return result
    except Exception as e:
        logger.error("update_concept_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/concepts/{concept_id}")
def delete_concept(concept_id: str):
    """Delete a saved concept."""
    try:
        repo.delete(concept_id)
        return {"status": "deleted"}
    except Exception as e:
        logger.error("delete_concept_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/concepts/{concept_id}/move")
def move_scope(concept_id: str, request: ScopeMoveRequest):
    """Move a concept between global ↔ project scope."""
    try:
        result = repo.move_scope(concept_id, request.project_id)
        if not result:
            raise HTTPException(status_code=404, detail="Concept not found")
        return result
    except Exception as e:
        logger.error("move_concept_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
