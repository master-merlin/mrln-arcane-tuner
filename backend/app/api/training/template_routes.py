"""Template CRUD routes — domain-specific with project scoping.

Replaces the legacy unified template routes. Each domain (captioning,
masking, training) has its own repository and scoping rules.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


# ── Request schemas ──────────────────────────────────────────────────────


class CreateCaptioningTemplateRequest(BaseModel):
    model_id: str
    name: str
    project_id: str | None = None
    system_prompt: str = "Describe this image in detail."
    wildcard: str = ""
    config: dict[str, Any] = {}


class CreateMaskingTemplateRequest(BaseModel):
    model_id: str
    name: str
    project_id: str | None = None
    config: dict[str, Any] = {}


class CreateTrainingTemplateRequest(BaseModel):
    definition_id: str
    name: str
    project_id: str | None = None
    config: dict[str, Any] = {}


class UpdateTemplateRequest(BaseModel):
    name: str | None = None
    system_prompt: str | None = None
    wildcard: str | None = None
    config: dict[str, Any] | None = None
    project_id: str | None = None
    definition_id: str | None = None
    model_id: str | None = None


class BranchTemplateRequest(BaseModel):
    target_project_id: str
    new_name: str | None = None


class CreateFromJobRequest(BaseModel):
    job_id: str
    name: str
    project_id: str | None = None


# ── Captioning templates ─────────────────────────────────────────────────


@router.get("/templates/captioning")
async def list_captioning_templates(
    model_id: str | None = None,
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    """List captioning templates (General + project scope)."""
    from app.core.db.repositories.captioning_template_repo import CaptioningTemplateRepository
    repo = CaptioningTemplateRepository()
    return await asyncio.to_thread(repo.list_for_project, model_id, project_id)


@router.post("/templates/captioning", status_code=201)
async def create_captioning_template(
    req: CreateCaptioningTemplateRequest,
) -> dict[str, Any]:
    from app.core.db.repositories.captioning_template_repo import CaptioningTemplateRepository
    repo = CaptioningTemplateRepository()
    return await asyncio.to_thread(repo.create, req.model_dump())


# ── Masking templates ────────────────────────────────────────────────────


@router.get("/templates/masking")
async def list_masking_templates(
    model_id: str | None = None,
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    from app.core.db.repositories.masking_template_repo import MaskingTemplateRepository
    repo = MaskingTemplateRepository()
    return await asyncio.to_thread(repo.list_for_project, model_id, project_id)


@router.post("/templates/masking", status_code=201)
async def create_masking_template(
    req: CreateMaskingTemplateRequest,
) -> dict[str, Any]:
    from app.core.db.repositories.masking_template_repo import MaskingTemplateRepository
    repo = MaskingTemplateRepository()
    return await asyncio.to_thread(repo.create, req.model_dump())


# ── Training templates ───────────────────────────────────────────────────


@router.get("/templates/training")
async def list_training_templates(
    definition_id: str | None = None,
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    from app.core.db.repositories.training_template_repo import TrainingTemplateRepository
    repo = TrainingTemplateRepository()
    return await asyncio.to_thread(repo.list_for_project, definition_id, project_id)


@router.post("/templates/training", status_code=201)
async def create_training_template(
    req: CreateTrainingTemplateRequest,
) -> dict[str, Any]:
    from app.core.db.repositories.training_template_repo import TrainingTemplateRepository
    repo = TrainingTemplateRepository()
    return await asyncio.to_thread(repo.create, req.model_dump())


@router.post("/templates/training/from-job", status_code=201)
async def create_training_template_from_job(
    req: CreateFromJobRequest,
) -> dict[str, Any]:
    """Create a training template from an archived job's config."""
    import json
    from app.core.db.repositories.training_template_repo import TrainingTemplateRepository
    from app.core.db.repositories.job_repo import JobHistoryRepository

    job_repo = JobHistoryRepository()
    job = await asyncio.to_thread(job_repo.get_by_id, req.job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    config = job.get("config", {})
    if isinstance(config, str):
        config = json.loads(config)

    repo = TrainingTemplateRepository()
    return await asyncio.to_thread(
        repo.create_from_job, config, req.name, req.project_id
    )


# ── Shared CRUD (any domain) ────────────────────────────────────────────


def _get_repo(domain: str):
    """Return the correct repository for a template domain."""
    if domain == "captioning":
        from app.core.db.repositories.captioning_template_repo import CaptioningTemplateRepository
        return CaptioningTemplateRepository()
    elif domain == "masking":
        from app.core.db.repositories.masking_template_repo import MaskingTemplateRepository
        return MaskingTemplateRepository()
    elif domain == "training":
        from app.core.db.repositories.training_template_repo import TrainingTemplateRepository
        return TrainingTemplateRepository()
    raise HTTPException(400, f"Unknown template domain: {domain}")


@router.get("/templates/{domain}/{template_id}")
async def get_template(domain: str, template_id: str) -> dict[str, Any]:
    """Get a single template by domain and ID."""
    repo = _get_repo(domain)
    tpl = await asyncio.to_thread(repo.get_by_id, template_id)
    if not tpl:
        raise HTTPException(404, "Template not found")
    return tpl


@router.put("/templates/{domain}/{template_id}")
async def update_template(
    domain: str, template_id: str, req: UpdateTemplateRequest
) -> dict[str, Any]:
    """Update a template."""
    repo = _get_repo(domain)
    existing = await asyncio.to_thread(repo.get_by_id, template_id)
    if not existing:
        raise HTTPException(404, "Template not found")
    if existing.get("readonly"):
        raise HTTPException(403, "Cannot modify a readonly default template")

    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    result = await asyncio.to_thread(repo.update, template_id, updates)
    return result or existing


@router.delete("/templates/{domain}/{template_id}")
async def delete_template(domain: str, template_id: str) -> dict[str, str]:
    """Delete a template."""
    repo = _get_repo(domain)
    existing = await asyncio.to_thread(repo.get_by_id, template_id)
    if not existing:
        raise HTTPException(404, "Template not found")
    if existing.get("readonly"):
        raise HTTPException(403, "Cannot delete a readonly default template")
    await asyncio.to_thread(repo.delete, template_id)
    return {"status": "deleted"}


@router.post("/templates/{domain}/{template_id}/branch")
async def branch_template(
    domain: str, template_id: str, req: BranchTemplateRequest
) -> dict[str, Any]:
    """Branch a General template into a project scope."""
    repo = _get_repo(domain)
    try:
        return await asyncio.to_thread(
            repo.branch, template_id, req.target_project_id, req.new_name
        )
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/templates/{domain}/{template_id}/use")
async def use_template(domain: str, template_id: str) -> dict[str, str]:
    """Increment usage counter."""
    repo = _get_repo(domain)
    await asyncio.to_thread(repo.increment_usage, template_id)
    return {"status": "recorded"}
