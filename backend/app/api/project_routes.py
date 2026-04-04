"""Project API — full CRUD for projects, dataset associations, and preferences."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.db.repositories.project_repo import ProjectRepository
from app.core.db.repositories.preference_repo import PreferenceRepository
from app.core.logger import get_logger

router = APIRouter(prefix="/projects", tags=["projects"])
logger = get_logger(__name__)

_projects = ProjectRepository()
_prefs = PreferenceRepository()


# ── Schemas ──────────────────────────────────────────────────────────────


class CreateProjectRequest(BaseModel):
    name: str
    description: str = ""
    color: str = "#6366f1"


class UpdateProjectRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    color: str | None = None


class UpdatePreferencesRequest(BaseModel):
    selected_caption_model: str | None = None
    active_caption_template: str | None = None
    qwen3_variant: str | None = None
    selected_mask_model: str | None = None
    active_mask_template: str | None = None
    training_selections: dict[str, Any] | None = None


class DatasetAssociationRequest(BaseModel):
    dataset_id: str


# ── Project CRUD ─────────────────────────────────────────────────────────


@router.get("")
async def list_projects() -> list[dict[str, Any]]:
    """List all projects with stats."""
    projects = _projects.list_all()
    for p in projects:
        p["stats"] = _projects.get_stats(p["id"])
    return projects


@router.post("", status_code=201)
async def create_project(req: CreateProjectRequest) -> dict[str, Any]:
    """Create a new project."""
    if _projects.get_by_name(req.name):
        raise HTTPException(409, f"Project '{req.name}' already exists")
    return _projects.create(req.model_dump())


@router.get("/{project_id}")
async def get_project(project_id: str) -> dict[str, Any]:
    """Get a single project with stats."""
    project = _projects.get_by_id(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    project["stats"] = _projects.get_stats(project_id)
    return project


@router.patch("/{project_id}")
async def update_project(
    project_id: str, req: UpdateProjectRequest
) -> dict[str, Any]:
    """Update project metadata."""
    if not _projects.get_by_id(project_id):
        raise HTTPException(404, "Project not found")
    updates = req.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(400, "No updates provided")
    return _projects.update(project_id, updates)


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: str) -> None:
    """Delete a project (cascades templates, preferences)."""
    if not _projects.get_by_id(project_id):
        raise HTTPException(404, "Project not found")
    _projects.delete(project_id)


# ── Dataset associations ─────────────────────────────────────────────────


@router.get("/{project_id}/datasets")
async def get_project_datasets(project_id: str) -> list[dict[str, Any]]:
    """Get datasets associated with a project."""
    if not _projects.get_by_id(project_id):
        raise HTTPException(404, "Project not found")
    return _projects.get_datasets(project_id)


@router.post("/{project_id}/datasets", status_code=201)
async def add_project_dataset(
    project_id: str, req: DatasetAssociationRequest
) -> dict[str, str]:
    """Associate a dataset with a project."""
    if not _projects.get_by_id(project_id):
        raise HTTPException(404, "Project not found")
    _projects.add_dataset(project_id, req.dataset_id)
    return {"status": "added"}


@router.delete("/{project_id}/datasets/{dataset_id}", status_code=204)
async def remove_project_dataset(project_id: str, dataset_id: str) -> None:
    """Remove a dataset association from a project."""
    _projects.remove_dataset(project_id, dataset_id)


# ── Preferences ──────────────────────────────────────────────────────────


@router.get("/{project_id}/preferences")
async def get_preferences(project_id: str) -> dict[str, Any]:
    """Get preferences for a project."""
    return _prefs.get(project_id if project_id != "general" else None)


@router.put("/{project_id}/preferences")
async def update_preferences(
    project_id: str, req: UpdatePreferencesRequest
) -> dict[str, Any]:
    """Update preferences for a project."""
    pid = project_id if project_id != "general" else None
    updates = req.model_dump(exclude_none=True)
    return _prefs.upsert(pid, updates)
