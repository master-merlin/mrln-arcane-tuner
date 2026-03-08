"""Template CRUD routes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from app.api.schemas.template_schemas import CreateTemplateRequest, UpdateTemplateRequest

router = APIRouter()


@router.get("/templates")
async def list_templates(
    category: str,
    definition_id: str | None = None,
    model_id: str | None = None,
):
    """List templates by category (training/captioning/masking)."""
    from app.core.db.repositories.template_repo import TemplateRepository
    repo = TemplateRepository()
    return await asyncio.to_thread(
        repo.list_by_category, category, definition_id, model_id
    )


@router.get("/templates/{template_id}")
async def get_template(template_id: str):
    """Return a single template by ID."""
    from app.core.db.repositories.template_repo import TemplateRepository
    repo = TemplateRepository()
    tpl = await asyncio.to_thread(repo.get_by_id, template_id)
    if not tpl:
        raise HTTPException(status_code=404, detail="Template not found")
    return tpl


@router.post("/templates")
async def create_template(request: CreateTemplateRequest):
    """Create a new template."""
    from app.core.db.repositories.template_repo import TemplateRepository
    repo = TemplateRepository()
    return await asyncio.to_thread(repo.create, request.model_dump())


@router.put("/templates/{template_id}")
async def update_template(template_id: str, request: UpdateTemplateRequest):
    """Update an existing template."""
    from app.core.db.repositories.template_repo import TemplateRepository
    repo = TemplateRepository()
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    await asyncio.to_thread(repo.update, template_id, updates)
    return await asyncio.to_thread(repo.get_by_id, template_id)


@router.delete("/templates/{template_id}")
async def delete_template(template_id: str):
    """Delete a template."""
    from app.core.db.repositories.template_repo import TemplateRepository
    repo = TemplateRepository()
    await asyncio.to_thread(repo.delete, template_id)
    return {"status": "deleted"}


@router.post("/templates/{template_id}/use")
async def use_template(template_id: str):
    """Increment usage counter when a template is applied."""
    from app.core.db.repositories.template_repo import TemplateRepository
    repo = TemplateRepository()
    await asyncio.to_thread(repo.increment_usage, template_id)
    return {"status": "recorded"}
