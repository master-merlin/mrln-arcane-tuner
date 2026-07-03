"""Project API — full CRUD for projects, dataset associations, and preferences."""

from __future__ import annotations

import asyncio
import io
import zipfile
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from app.api.training.template_routes import (
    ImportCreatedEntry,
    ImportSkippedEntry,
    TemplatePlanEntry,
)
from app.core.db.repositories.project_repo import ProjectRepository
from app.core.db.repositories.preference_repo import PreferenceRepository
from app.core.events import emit_entity_change, event_manager
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


# ── Response schemas ─────────────────────────────────────────────────────


class ProjectRow(BaseModel):
    """A bare ``projects`` table row (``SELECT *``). No ``stats`` key —
    that's only injected by the list/get-one routes (see ProjectWithStats)."""

    id: str
    name: str
    description: str
    color: str
    created_at: float
    updated_at: float


class ProjectStats(BaseModel):
    """Template/dataset/job counts for a project (``ProjectRepository.get_stats``)."""

    captioning_templates: int
    masking_templates: int
    training_templates: int
    datasets: int
    jobs: int


class ProjectWithStats(ProjectRow):
    """A project row plus its stats — the shape ``list``/``get-one`` return."""

    stats: ProjectStats


class ProjectDatasetRow(BaseModel):
    """A dataset row scoped to a project (raw ``SELECT d.*`` from
    ``datasets``, bypassing the ``Dataset`` domain model). Open model: the
    frontend already treats this as a dynamic bag (``project.service.ts``'s
    ``Dataset`` and ``project-detail.ts``'s ``ProjectDatasetRow`` both declare
    ``[key: string]: unknown``) and the ``datasets`` table has grown via
    several ``ALTER TABLE`` migrations — only the two fields every consumer
    relies on are declared; the rest pass through via ``extra=\"allow\"``."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str


class ProjectPreferencesRow(BaseModel):
    """A ``project_preferences`` row (``training_selections`` JSON-decoded
    by the repo)."""

    id: str
    project_id: str | None = None
    selected_caption_model: str | None = None
    active_caption_template: str | None = None
    qwen3_variant: str | None = None
    selected_mask_model: str | None = None
    active_mask_template: str | None = None
    training_selections: dict[str, Any] = {}


class ProjectImportPlanProjectInfo(BaseModel):
    name: str | None = None
    conflict: bool


class ProjectDatasetPlanItem(BaseModel):
    """One dataset's import-plan entry — mirrors the frontend's
    ``ProjectDatasetPlan`` (project.service.ts) exactly."""

    name: str
    mode: str
    reference_present: bool | None = None
    embed_conflict: bool | None = None


class ProjectImportApplyTemplatesResult(BaseModel):
    created: list[ImportCreatedEntry]
    skipped: list[ImportSkippedEntry]


class ProjectImportApplyResponse(BaseModel):
    project_id: str
    project_name: str
    imported_datasets: list[str]
    linked_references: list[str]
    missing_references: list[str]
    templates: ProjectImportApplyTemplatesResult
    installed_definitions: list[str]


class ProjectImportRollbackResponse(BaseModel):
    status: str
    project_id: str


class ProjectImportPlanResponse(BaseModel):
    project: ProjectImportPlanProjectInfo
    templates: list[TemplatePlanEntry]
    datasets: list[ProjectDatasetPlanItem]


# ── Project CRUD ─────────────────────────────────────────────────────────


@router.get("", response_model=list[ProjectWithStats])
async def list_projects() -> list[dict[str, Any]]:
    """List all projects with stats."""

    def _work() -> list[dict[str, Any]]:
        projects = _projects.list_all()
        for p in projects:
            p["stats"] = _projects.get_stats(p["id"])
        return projects

    return await asyncio.to_thread(_work)


@router.post("", status_code=201, response_model=ProjectRow)
async def create_project(req: CreateProjectRequest) -> dict[str, Any]:
    """Create a new project."""

    def _work() -> dict[str, Any]:
        if _projects.get_by_name(req.name):
            raise HTTPException(409, f"Project '{req.name}' already exists")
        return _projects.create(req.model_dump())

    project = await asyncio.to_thread(_work)
    await emit_entity_change(
        event_manager.broadcast,
        entity="project", op="created", id=project["id"], payload=project,
    )
    return project


@router.get("/{project_id}", response_model=ProjectWithStats)
async def get_project(project_id: str) -> dict[str, Any]:
    """Get a single project with stats."""

    def _work() -> dict[str, Any]:
        project = _projects.get_by_id(project_id)
        if not project:
            raise HTTPException(404, "Project not found")
        project["stats"] = _projects.get_stats(project_id)
        return project

    return await asyncio.to_thread(_work)


@router.patch("/{project_id}", response_model=ProjectRow)
async def update_project(
    project_id: str, req: UpdateProjectRequest
) -> dict[str, Any]:
    """Update project metadata."""

    def _work() -> dict[str, Any]:
        if not _projects.get_by_id(project_id):
            raise HTTPException(404, "Project not found")
        updates = req.model_dump(exclude_none=True)
        if not updates:
            raise HTTPException(400, "No updates provided")
        return _projects.update(project_id, updates)

    project = await asyncio.to_thread(_work)
    await emit_entity_change(
        event_manager.broadcast,
        entity="project", op="updated", id=project_id, payload=project,
    )
    return project


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: str) -> None:
    """Delete a project (cascades templates, preferences)."""

    def _work() -> None:
        if not _projects.get_by_id(project_id):
            raise HTTPException(404, "Project not found")
        _projects.delete(project_id)

    await asyncio.to_thread(_work)
    await emit_entity_change(
        event_manager.broadcast,
        entity="project", op="deleted", id=project_id, payload=None,
    )


# ── Dataset associations ─────────────────────────────────────────────────


@router.get("/{project_id}/datasets", response_model=list[ProjectDatasetRow])
async def get_project_datasets(project_id: str) -> list[dict[str, Any]]:
    """Get datasets associated with a project."""

    def _work() -> list[dict[str, Any]]:
        if not _projects.get_by_id(project_id):
            raise HTTPException(404, "Project not found")
        return _projects.get_datasets(project_id)

    return await asyncio.to_thread(_work)


@router.post("/{project_id}/datasets", status_code=201, response_model=DatasetAssociationResponse)
async def add_project_dataset(
    project_id: str, req: DatasetAssociationRequest
) -> dict[str, str]:
    """Associate a dataset with a project."""

    def _work() -> dict[str, str]:
        if not _projects.get_by_id(project_id):
            raise HTTPException(404, "Project not found")
        _projects.add_dataset(project_id, req.dataset_id)
        return {"status": "added"}

    result = await asyncio.to_thread(_work)
    await _emit_project_membership_updated(project_id)
    return result


@router.delete("/{project_id}/datasets/{dataset_id}", status_code=204)
async def remove_project_dataset(project_id: str, dataset_id: str) -> None:
    """Remove a dataset association from a project."""
    await asyncio.to_thread(_projects.remove_dataset, project_id, dataset_id)
    await _emit_project_membership_updated(project_id)


async def _emit_project_membership_updated(project_id: str) -> None:
    """Broadcast a project `updated` event after a dataset-association change.

    Membership changes (add/remove a dataset) count as project updates —
    the project row itself (name/description/color) is unchanged, but its
    dataset associations are part of its externally-visible state.
    """
    project = await asyncio.to_thread(_projects.get_by_id, project_id)
    if project is not None:
        await emit_entity_change(
            event_manager.broadcast,
            entity="project", op="updated", id=project_id, payload=project,
        )


# ── Preferences ──────────────────────────────────────────────────────────


@router.get("/{project_id}/preferences", response_model=ProjectPreferencesRow)
async def get_preferences(project_id: str) -> dict[str, Any]:
    """Get preferences for a project."""
    pid = project_id if project_id != "general" else None
    return await asyncio.to_thread(_prefs.get, pid)


@router.put("/{project_id}/preferences", response_model=ProjectPreferencesRow)
async def update_preferences(
    project_id: str, req: UpdatePreferencesRequest
) -> dict[str, Any]:
    """Update preferences for a project."""
    pid = project_id if project_id != "general" else None
    updates = req.model_dump(exclude_none=True)
    return await asyncio.to_thread(_prefs.upsert, pid, updates)


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


@router.post("/import/plan", response_model=ProjectImportPlanResponse)
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


# ── Import: apply + rollback ─────────────────────────────────────────────


def _unique_project_name(base: str) -> str:
    candidate = base
    i = 2
    while _projects.get_by_name(candidate) is not None:
        candidate = f"{base} ({i})"
        i += 1
    return candidate


def _uninstall_definition(definition_id: str) -> None:
    """Undo a definition install (mirror of the definitions DELETE route).

    Note: unlike the DELETE route this does NOT cascade a source-override
    removal, because import-time install never writes an override. Revisit if
    definition install ever starts persisting overrides.
    """
    from pathlib import Path

    from app.engine.models.registry import registry

    path = registry._paths.get(definition_id)
    if path:
        p = Path(path)
        if p.exists():
            p.unlink()
    registry._definitions.pop(definition_id, None)
    registry._paths.pop(definition_id, None)


@router.post("/import/apply", response_model=ProjectImportApplyResponse)
async def apply_project_import(
    file: UploadFile = File(...),
    resolutions: str = Form(default="{}"),
) -> dict[str, Any]:
    """Recreate a project from an archive, transactionally (rollback on error).

    Caveat: ``on_conflict='overwrite'`` deletes the existing project before
    creating the new one, so a later failure rolls back the *new* work but the
    *old* project is already gone (mirrors the dataset-import overwrite).
    """
    import json
    import tempfile
    from pathlib import Path

    from app.api.dataset.crud_routes import _import_from_zip_path
    from app.api.training.template_routes import import_template_entries
    from app.core.dataset_manager import dataset_manager
    from app.core.portable.envelope import ManifestError
    from app.core.project import portable as pportable
    from app.core.template import portable as tportable

    data = await file.read()
    try:
        res = json.loads(resolutions or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"Invalid resolutions JSON: {exc}") from exc
    proj_res = res.get("project") or {}
    ds_res = res.get("datasets") or {}
    tpl_res = res.get("templates") or {}

    def _apply() -> dict[str, Any]:
        try:
            outer = zipfile.ZipFile(io.BytesIO(data))
            manifest = pportable.read_project_manifest(outer)
        except ManifestError as exc:
            raise HTTPException(400, str(exc)) from exc
        except zipfile.BadZipFile as exc:
            raise HTTPException(400, "Archive is not a valid zip file.") from exc

        with outer:
            proj = manifest["project"]
            name = proj_res.get("name") or proj.get("name") or "Imported Project"
            on_conflict = proj_res.get("on_conflict")
            if _projects.get_by_name(name) is not None:
                if on_conflict == "overwrite":
                    _projects.delete(_projects.get_by_name(name)["id"])
                elif on_conflict == "rename":
                    name = _unique_project_name(name)
                else:
                    raise HTTPException(
                        409, {"conflict": True, "name": name,
                              "message": f"A project named '{name}' already exists."})

            project_id: str | None = None
            imported_datasets: list[str] = []
            installed_defs: list[str] = []
            try:
                created = _projects.create({
                    "name": name,
                    "description": proj.get("description", ""),
                    "color": proj.get("color") or "#6366f1"})
                project_id = created["id"]

                linked_refs: list[str] = []
                missing_refs: list[str] = []
                for dref in manifest["datasets"]:
                    if dref["mode"] == "reference":
                        ds = dataset_manager.get_dataset(dref["name"])
                        if ds is not None:
                            _projects.add_dataset(project_id, ds.id)
                            linked_refs.append(dref["name"])
                        else:
                            missing_refs.append(dref["name"])
                    elif dref["mode"] == "embed":
                        nested = outer.read(dref["archive"])
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
                        try:
                            tmp.write(nested)
                            tmp.close()
                            oc = (ds_res.get(dref["name"]) or {}).get("on_conflict") or "rename"
                            ds = _import_from_zip_path(Path(tmp.name), oc, None)
                        finally:
                            Path(tmp.name).unlink(missing_ok=True)
                        imported_datasets.append(ds.name)
                        _projects.add_dataset(project_id, ds.id)

                created_t: list[dict[str, Any]] = []
                skipped_t: list[dict[str, Any]] = []
                for i, tref in enumerate(manifest["templates"]):
                    nested = outer.read(tref["archive"])
                    with zipfile.ZipFile(io.BytesIO(nested)) as nzf:
                        tmanifest = tportable.read_template_manifest(nzf)
                    r = import_template_entries(
                        tmanifest, {"0": tpl_res.get(str(i)) or {}}, project_id)
                    created_t.extend(r["created"])
                    skipped_t.extend(r["skipped"])
                    installed_defs.extend(r["installed_definitions"])

                prefs = proj.get("preferences") or {}
                if prefs:
                    _prefs.upsert(project_id, prefs)

                return {
                    "project_id": project_id,
                    "project_name": name,
                    "imported_datasets": imported_datasets,
                    "linked_references": linked_refs,
                    "missing_references": missing_refs,
                    "templates": {"created": created_t, "skipped": skipped_t},
                    "installed_definitions": installed_defs,
                }
            except HTTPException:
                _rollback(project_id, imported_datasets, installed_defs)
                raise
            except Exception as exc:  # noqa: BLE001
                _rollback(project_id, imported_datasets, installed_defs)
                raise HTTPException(500, f"Project import failed and was rolled back: {exc}") from exc

    return await asyncio.to_thread(_apply)


def _rollback(
    project_id: str | None,
    imported_datasets: list[str],
    installed_defs: list[str],
) -> None:
    """Best-effort undo of a project import (order: defs → datasets → project).

    Project delete cascades the ``project_datasets`` association rows, so links
    need no explicit cleanup; only the file-backed artifacts (imported dataset
    folders, newly-installed definition YAMLs) are undone here.
    """
    from app.core.dataset_manager import dataset_manager

    for def_id in installed_defs:
        try:
            _uninstall_definition(def_id)
        except Exception:  # noqa: BLE001, S110
            pass
    for name in imported_datasets:
        try:
            dataset_manager.delete_dataset(name, delete_files=True)
        except Exception:  # noqa: BLE001, S110
            pass
    if project_id:
        try:
            _projects.delete(project_id)
        except Exception:  # noqa: BLE001, S110
            pass


class RollbackImportRequest(BaseModel):
    project_id: str
    imported_datasets: list[str] = []
    installed_definitions: list[str] = []


@router.post("/import/rollback", response_model=ProjectImportRollbackResponse)
async def rollback_project_import(req: RollbackImportRequest) -> dict[str, str]:
    """User-triggered undo of a *successful* import the user decided not to keep
    (e.g. after reviewing soft skips). Reverses defs → datasets → project."""
    await asyncio.to_thread(
        _rollback, req.project_id, req.imported_datasets, req.installed_definitions)
    return {"status": "rolled_back", "project_id": req.project_id}
