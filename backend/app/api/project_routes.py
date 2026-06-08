"""Project API — full CRUD for projects, dataset associations, and preferences."""

from __future__ import annotations

import asyncio
import io
import zipfile
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
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


class DatasetAssociationResponse(BaseModel):
    """Ack for associating a dataset with a project."""

    status: str = "added"


class ExportTemplateRef(BaseModel):
    domain: str
    id: str


class ExportDatasetChoice(BaseModel):
    name: str
    mode: str  # "embed" | "reference" | "exclude"


class ExportProjectRequest(BaseModel):
    templates: list[ExportTemplateRef] = []
    datasets: list[ExportDatasetChoice] = []


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


@router.post("/{project_id}/datasets", status_code=201, response_model=DatasetAssociationResponse)
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


# ── Export ───────────────────────────────────────────────────────────────


@router.post("/{project_id}/export")
async def export_project(project_id: str, req: ExportProjectRequest) -> StreamingResponse:
    """Assemble a kind='project' archive: metadata + preferences + selected
    templates (nested template archives) + datasets (embed/reference/exclude)."""
    from pathlib import Path

    from app import __version__ as APP_VERSION
    from app.api.training.template_routes import export_template_archive_bytes
    from app.core.dataset import portable as dportable
    from app.core.dataset_manager import dataset_manager
    from app.core.portable.archive import write_bundle_zip
    from app.core.project import portable as pportable

    def _build() -> StreamingResponse:
        project = _projects.get_by_id(project_id)
        if not project:
            raise HTTPException(404, "Project not found")
        prefs = _prefs.get(project_id)

        entries: dict[str, bytes] = {}
        seen_arcnames: set[str] = set()
        template_refs: list[dict[str, Any]] = []
        for t in req.templates:
            payload = export_template_archive_bytes(t.domain, t.id)
            if payload is None:
                raise HTTPException(404, f"Template not found: {t.domain}/{t.id}")
            arcname = pportable.unique_arcname(
                f"templates/{t.domain}-{pportable.slugify(t.id)}.zip", seen_arcnames)
            entries[arcname] = payload
            template_refs.append({"domain": t.domain, "archive": arcname})

        dataset_refs: list[dict[str, Any]] = []
        for d in req.datasets:
            if d.mode == "exclude":
                continue
            if d.mode == "reference":
                dataset_refs.append({"mode": "reference", "name": d.name})
                continue
            # embed
            ds = dataset_manager.get_dataset(d.name)
            if ds is None:
                raise HTTPException(404, f"Dataset not found: {d.name}")
            manifest_d = dportable.build_manifest(ds, app_version=APP_VERSION)
            payload = dportable.write_export_zip(Path(ds.path), manifest_d).getvalue()
            arcname = pportable.unique_arcname(
                f"datasets/{pportable.slugify(d.name)}.zip", seen_arcnames)
            entries[arcname] = payload
            dataset_refs.append({"mode": "embed", "name": d.name, "archive": arcname})

        manifest = pportable.build_project_manifest(
            project, prefs, template_refs, dataset_refs, APP_VERSION)
        buf = write_bundle_zip(manifest, entries)
        filename = f"{pportable.slugify(project.get('name'))}.project.zip"
        return StreamingResponse(
            buf, media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    return await asyncio.to_thread(_build)


# ── Import: plan ─────────────────────────────────────────────────────────


@router.post("/import/plan")
async def plan_project_import(file: UploadFile = File(...)) -> dict[str, Any]:
    """Read a project archive and return a dry-run plan (read-only)."""
    from app.api.training.template_routes import plan_template_entries
    from app.core.dataset_manager import dataset_manager
    from app.core.portable.envelope import ManifestError
    from app.core.project import portable as pportable
    from app.core.template import portable as tportable

    data = await file.read()

    def _plan() -> dict[str, Any]:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                manifest = pportable.read_project_manifest(zf)
                proj = manifest["project"]

                templates: list[dict[str, Any]] = []
                for tref in manifest["templates"]:
                    nested = zf.read(tref["archive"])
                    with zipfile.ZipFile(io.BytesIO(nested)) as nzf:
                        tmanifest = tportable.read_template_manifest(nzf)
                    templates.extend(plan_template_entries(tmanifest, None))

                datasets: list[dict[str, Any]] = []
                for dref in manifest["datasets"]:
                    item = {"name": dref["name"], "mode": dref["mode"]}
                    if dref["mode"] == "reference":
                        item["reference_present"] = (
                            dataset_manager.get_dataset(dref["name"]) is not None)
                    elif dref["mode"] == "embed":
                        item["embed_conflict"] = (
                            dataset_manager.get_dataset(dref["name"]) is not None)
                    datasets.append(item)
        except ManifestError as exc:
            raise HTTPException(400, str(exc)) from exc
        except zipfile.BadZipFile as exc:
            raise HTTPException(400, "Archive is not a valid zip file.") from exc

        return {
            "project": {"name": proj.get("name"),
                        "conflict": _projects.get_by_name(proj.get("name")) is not None},
            "templates": templates,
            "datasets": datasets,
        }

    return await asyncio.to_thread(_plan)
