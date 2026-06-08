"""Template CRUD routes — domain-specific with project scoping.

Replaces the legacy unified template routes. Each domain (captioning,
masking, training) has its own repository and scoping rules.
"""

from __future__ import annotations

import asyncio
import io
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


# ── Response schemas ─────────────────────────────────────────────────────
# Each template domain is a fixed table (SELECT *), so the row shape is
# stable. `config` is JSON-parsed into a dict by the repo's `_from_row`,
# but may remain a raw string if parsing fails — hence `dict | str | None`.


class StatusResponse(BaseModel):
    """Simple ``{"status": ...}`` acknowledgement."""

    status: str


class CaptioningTemplate(BaseModel):
    id: str
    project_id: str | None = None
    model_id: str
    name: str
    is_default: bool = False
    readonly: bool = False
    system_prompt: str = "Describe this image in detail."
    # Per-template wildcard substituted into {wildcard} tokens before captioning.
    # MUST be in the response model — without it FastAPI strips the field from
    # every GET/list/update response, so a saved wildcard never reads back and
    # the UI looks like it didn't persist (system_prompt did, hence the asymmetry).
    wildcard: str = ""
    config: dict[str, Any] | str | None = None
    created_at: float
    updated_at: float | None = None
    used_count: int = 0
    last_used_at: float | None = None
    branched_from: str | None = None


class MaskingTemplate(BaseModel):
    id: str
    project_id: str | None = None
    model_id: str
    name: str
    is_default: bool = False
    readonly: bool = False
    config: dict[str, Any] | str | None = None
    created_at: float
    updated_at: float | None = None
    used_count: int = 0
    last_used_at: float | None = None
    branched_from: str | None = None


class TrainingTemplate(BaseModel):
    id: str
    project_id: str | None = None
    definition_id: str
    name: str
    is_default: bool = False
    readonly: bool = False
    config: dict[str, Any] | str | None = None
    created_at: float
    updated_at: float | None = None
    used_count: int = 0
    last_used_at: float | None = None
    branched_from: str | None = None


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


class ExportBundleItem(BaseModel):
    domain: str
    id: str


class ExportBundleRequest(BaseModel):
    items: list[ExportBundleItem]


class ImportEntryResolution(BaseModel):
    action: str = "create"          # "create" | "skip"
    name: str | None = None         # rename override
    install_definition: bool = False
    use_hf_substitution: bool = True


# ── Captioning templates ─────────────────────────────────────────────────


@router.get("/templates/captioning", response_model=list[CaptioningTemplate])
async def list_captioning_templates(
    model_id: str | None = None,
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    """List captioning templates (General + project scope)."""
    from app.core.db.repositories.captioning_template_repo import CaptioningTemplateRepository
    repo = CaptioningTemplateRepository()
    return await asyncio.to_thread(repo.list_for_project, model_id, project_id)


@router.post("/templates/captioning", status_code=201, response_model=CaptioningTemplate)
async def create_captioning_template(
    req: CreateCaptioningTemplateRequest,
) -> dict[str, Any]:
    from app.core.db.repositories.captioning_template_repo import CaptioningTemplateRepository
    repo = CaptioningTemplateRepository()
    return await asyncio.to_thread(repo.create, req.model_dump())


# ── Masking templates ────────────────────────────────────────────────────


@router.get("/templates/masking", response_model=list[MaskingTemplate])
async def list_masking_templates(
    model_id: str | None = None,
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    from app.core.db.repositories.masking_template_repo import MaskingTemplateRepository
    repo = MaskingTemplateRepository()
    return await asyncio.to_thread(repo.list_for_project, model_id, project_id)


@router.post("/templates/masking", status_code=201, response_model=MaskingTemplate)
async def create_masking_template(
    req: CreateMaskingTemplateRequest,
) -> dict[str, Any]:
    from app.core.db.repositories.masking_template_repo import MaskingTemplateRepository
    repo = MaskingTemplateRepository()
    return await asyncio.to_thread(repo.create, req.model_dump())


# ── Training templates ───────────────────────────────────────────────────


@router.get("/templates/training", response_model=list[TrainingTemplate])
async def list_training_templates(
    definition_id: str | None = None,
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    from app.core.db.repositories.training_template_repo import TrainingTemplateRepository
    repo = TrainingTemplateRepository()
    return await asyncio.to_thread(repo.list_for_project, definition_id, project_id)


@router.post("/templates/training", status_code=201, response_model=TrainingTemplate)
async def create_training_template(
    req: CreateTrainingTemplateRequest,
) -> dict[str, Any]:
    from app.core.db.repositories.training_template_repo import TrainingTemplateRepository
    repo = TrainingTemplateRepository()
    return await asyncio.to_thread(repo.create, req.model_dump())


@router.post("/templates/training/from-job", status_code=201, response_model=TrainingTemplate)
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


@router.delete("/templates/{domain}/{template_id}", response_model=StatusResponse)
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


@router.post("/templates/{domain}/{template_id}/use", response_model=StatusResponse)
async def use_template(domain: str, template_id: str) -> dict[str, str]:
    """Increment usage counter."""
    repo = _get_repo(domain)
    await asyncio.to_thread(repo.increment_usage, template_id)
    return {"status": "recorded"}


# ── Export ───────────────────────────────────────────────────────────────


def _safe_filename(name: str | None) -> str:
    # ASCII-only: the Content-Disposition header is encoded as latin-1, so a
    # Unicode letter (CJK/Cyrillic/accented) — which str.isalnum() accepts —
    # would crash the response. Non-ASCII names fall back to "template".
    cleaned = "".join(
        c for c in (name or "")
        if c.isascii() and (c.isalnum() or c in (" ", "-", "_"))
    ).strip()
    return cleaned or "template"


def _export_entry(domain: str, row: dict[str, Any]) -> dict[str, Any]:
    """Build a carried template entry; embed the definition for training rows."""
    from app.core.template import portable

    definition = None
    if domain == "training":
        from app.engine.models.registry import registry

        defn = registry.get_definition(row.get("definition_id") or "")
        definition = defn.model_dump() if defn is not None else None
    return portable.build_template_entry(domain, row, definition)


def _template_zip_response(
    entries: list[dict[str, Any]], filename: str
) -> StreamingResponse:
    from app import __version__ as APP_VERSION
    from app.core.portable.archive import write_manifest_zip
    from app.core.template import portable

    manifest = portable.build_template_manifest(entries, APP_VERSION)
    buf = write_manifest_zip(manifest)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/templates/{domain}/{template_id}/export")
async def export_template(domain: str, template_id: str) -> StreamingResponse:
    """Export a single template as a ``kind='template'`` archive."""
    repo = _get_repo(domain)  # raises 400 for an unknown domain
    row = await asyncio.to_thread(repo.get_by_id, template_id)
    if not row:
        raise HTTPException(404, "Template not found")
    entry = await asyncio.to_thread(_export_entry, domain, row)
    return _template_zip_response(
        [entry], f"{_safe_filename(row.get('name'))}.template.zip"
    )


@router.post("/templates/export")
async def export_templates_bundle(req: ExportBundleRequest) -> StreamingResponse:
    """Export 1..N selected templates (any mix of domains) as one archive."""
    if not req.items:
        raise HTTPException(400, "No templates selected for export.")

    def _collect() -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for item in req.items:
            repo = _get_repo(item.domain)  # raises 400 for an unknown domain
            row = repo.get_by_id(item.id)
            if not row:
                raise HTTPException(404, f"Template not found: {item.domain}/{item.id}")
            entries.append(_export_entry(item.domain, row))
        return entries

    entries = await asyncio.to_thread(_collect)
    if len(entries) == 1:
        filename = f"{_safe_filename(entries[0].get('name'))}.template.zip"
    else:
        filename = "templates-bundle.zip"
    return _template_zip_response(entries, filename)


# ── Import: helpers ──────────────────────────────────────────────────────


def _families_dir():
    from pathlib import Path

    import app.engine.models.registry as _reg_mod

    return Path(_reg_mod.__file__).resolve().parent / "families"


def _find_family_hf_source(family: str, component: str) -> str | None:
    """Return an ``huggingface:`` path for *component* from any present
    definition of the same *family*, or None."""
    from app.core.template import portable
    from app.engine.models.registry import registry

    for did in registry.list_models():
        defn = registry.get_definition(did)
        if defn is None or defn.family != family:
            continue
        comp = defn.components.get(component)
        path = getattr(comp, "path", None)
        if path and not portable.is_local_component_path(path):
            return path
    return None


def _plan_training_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Compute the definition status + local-component substitution plan."""
    from app.core.template import import_service, portable
    from app.engine.models.registry import registry
    from app.engine.utils.model_override_manager import ModelOverrideManager

    definition_id = entry.get("definition_id") or ""
    out: dict[str, Any] = {"definition_id": definition_id, "local_components": []}

    if registry.get_definition(definition_id) is not None:
        out["definition_status"] = "present"
        return out

    carried = entry.get("definition")
    if not isinstance(carried, dict):
        out["definition_status"] = "missing"
        return out

    err = import_service.validate_carried_definition(carried)
    if err is not None:
        out["definition_status"] = "invalid"
        out["definition_error"] = err
        return out

    out["definition_status"] = "installable"
    offline = ModelOverrideManager.is_offline(definition_id)
    family = carried.get("family") or ""
    for comp in portable.scan_local_component_paths(carried):
        hf = None if offline else _find_family_hf_source(family, comp["component"])
        out["local_components"].append(
            {"component": comp["component"], "local_path": comp["path"],
             "hf_substitute": hf}
        )
    return out


def _plan_entry(entry: dict[str, Any], project_id: str | None) -> dict[str, Any]:
    from app.core.template import import_service

    domain = entry["domain"]
    name = entry.get("name") or ""
    item: dict[str, Any] = {
        "domain": domain, "name": name,
        "config_warning": import_service.validate_config(
            domain, entry.get("model_id"), entry.get("config")),
        "duplicate_name": _name_exists(domain, name, project_id),
        "blocker": False,
    }
    if domain in ("captioning", "masking"):
        model_id = entry.get("model_id") or ""
        available = import_service.model_available(domain, model_id)
        item["model_id"] = model_id
        item["model_available"] = available
        item["blocker"] = not available
    else:  # training
        info = _plan_training_entry(entry)
        item.update(info)
        item["blocker"] = info["definition_status"] in ("missing", "invalid")
    return item


def _name_exists(domain: str, name: str, project_id: str | None) -> bool:
    repo = _get_repo(domain)
    rows = repo.list_for_project(None, project_id)
    return any((r.get("name") or "") == name for r in rows)


# ── Import: plan endpoint ────────────────────────────────────────────────


@router.post("/templates/import/plan")
async def plan_template_import(
    file: UploadFile = File(...),
    project_id: str | None = Form(default=None),
) -> dict[str, Any]:
    """Read a template archive and return a per-entry import plan (read-only)."""
    import zipfile

    from app.core.portable.envelope import ManifestError
    from app.core.template import portable

    data = await file.read()

    def _build_plan() -> dict[str, Any]:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                manifest = portable.read_template_manifest(zf)
        except ManifestError as exc:
            raise HTTPException(400, str(exc)) from exc
        except zipfile.BadZipFile as exc:
            raise HTTPException(400, "Archive is not a valid zip file.") from exc
        entries = [
            {**_plan_entry(e, project_id), "index": i}
            for i, e in enumerate(manifest["templates"])
        ]
        return {
            "project_id": project_id,
            "entries": entries,
            "importable_count": sum(1 for e in entries if not e["blocker"]),
        }

    return await asyncio.to_thread(_build_plan)
