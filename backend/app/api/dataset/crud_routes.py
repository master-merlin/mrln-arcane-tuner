"""Dataset CRUD, scanning, upload, pairs, captions, enable/disable, and download routes."""

from __future__ import annotations

import asyncio
import io
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Literal
import zipfile

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask

from app.api._deps import dataset_or_404
from app.api._path_guard import sanitize_filename, validate_path_within
from app.core.dataset_manager import dataset_manager, Dataset
from app.core.logger import get_logger
from app import __version__ as APP_VERSION
from app.core.dataset import portable
from app.api.schemas.dataset_schemas import (
    CreateDatasetRequest,
    UpdateDatasetRequest,
    SetPreviewRequest,
    CaptionRequest,
    ToggleEnabledRequest,
    ImportPathRequest,
    DatasetDeletedResponse,
    MediaPairDeletedResponse,
    UploadResponse,
    CaptionContentResponse,
    CaptionSavedResponse,
    ToggleEnabledResponse,
    EnableAllResponse,
    DatasetPairResponse,
)
from app.api.schemas.common_schemas import TaskEnqueuedResponse
from app.core.dataset.rescan_batch import run_rescan_batch, count_multimedia
from app.core.tasks.task_manager import task_manager

router = APIRouter()
logger = get_logger(__name__)


def get_dataset_or_404(name: str) -> Dataset:
    """Path-operation dependency: resolve a dataset by name or 404."""
    return dataset_or_404(dataset_manager.get_dataset(name))


# ── Dataset CRUD ─────────────────────────────────────────────────────────


@router.get(
    "/datasets",
    response_model=list[Dataset],
    # `media_metadata` is the per-FILE map. On a single dataset it is bounded;
    # on the LIST endpoint it dominates the response and grows with the TOTAL
    # file count across the library, while every other field is O(1) per row —
    # so this payload scales with the wrong quantity, and pays it on every
    # library load. Nothing consumes it from this response: the client reads
    # per-file data from `/datasets/{name}/pairs`, and the cross-dataset MPx
    # histogram is computed server-side by `/datasets/stats/mpx-distribution`
    # for exactly this reason.
    #
    # `{"__all__": {...}}`, NOT `{"media_metadata"}`: against a list response
    # model the bare field-name form silently does nothing and ships the field
    # anyway. The exclusion is serialization-only, so the `excluded_count` and
    # `median_quality_score` computed fields — both derived from
    # `media_metadata` — are still computed and still returned.
    #
    # `GET /datasets/{name}` deliberately still carries it: it is one dataset,
    # so the payload is bounded, and single-dataset consumers may want it.
    response_model_exclude={"__all__": {"media_metadata"}},
)
async def list_datasets():
    """List all registered datasets (without the heavy per-file metadata)."""
    return await asyncio.to_thread(dataset_manager.list_datasets)


@router.get("/datasets/{name}", response_model=Dataset)
async def get_dataset(dataset: Dataset = Depends(get_dataset_or_404)):
    """Return a single dataset by name."""
    return dataset


@router.post("/datasets", response_model=Dataset)
async def create_dataset(request: CreateDatasetRequest):
    """Create a new dataset directory and register it."""
    try:
        logger.info("creating_dataset", dataset_name=request.name)
        return await asyncio.to_thread(
            dataset_manager.create_dataset,
            request.name,
            request.description,
            classifier=request.classifier,
            trigger_word=request.trigger_word,
            tags=request.tags,
            notes=request.notes,
            kind=request.kind,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/datasets/{name}", response_model=Dataset)
async def update_dataset(name: str, request: UpdateDatasetRequest):
    """Update dataset metadata (name, description, classifier, trigger_word, tags, notes)."""
    try:
        logger.info("updating_dataset", old_name=name, new_name=request.name)
        return await asyncio.to_thread(
            dataset_manager.update_dataset,
            name,
            request.name,
            request.description,
            new_classifier=request.classifier,
            new_trigger_word=request.trigger_word,
            new_tags=request.tags,
            new_notes=request.notes,
            new_kind=request.kind,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/datasets/{name}/preview", response_model=Dataset)
async def set_dataset_preview(name: str, request: SetPreviewRequest):
    """Pin the dataset's library cover, or unpin it with ``image_rel_path: null``.

    Without this the cover was whatever the scanner enumerated first, which the
    user had no way to influence. Returns the full dataset so the caller can
    replace its row; the change is also broadcast on the entity channel, so
    every open surface refreshes on its own.
    """
    try:
        logger.info("setting_dataset_preview", dataset_name=name, image=request.image_rel_path)
        return await asyncio.to_thread(
            dataset_manager.set_preview_image, name, request.image_rel_path,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/datasets/{name}", response_model=DatasetDeletedResponse)
async def delete_dataset(name: str, delete_files: bool = False):
    """Unregister a dataset, optionally deleting files on disk."""
    try:
        logger.info("deleting_dataset", dataset_name=name, delete_files=delete_files)
        await asyncio.to_thread(dataset_manager.delete_dataset, name, delete_files)
        return {"status": "deleted", "name": name}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Scanning ─────────────────────────────────────────────────────────────


@router.post("/datasets/{name}/scan", response_model=Dataset)
async def scan_dataset(name: str, force_full: bool = Query(False)):
    """Re-scan a dataset's directory for media and caption changes."""
    try:
        logger.info("scanning_dataset", dataset_name=name, force_full=force_full)
        return await asyncio.to_thread(dataset_manager.scan_dataset, name, force_full)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/datasets/{name}/scan/batch", response_model=TaskEnqueuedResponse)
async def scan_dataset_batch(
    name: str, force_full: bool = Query(False), dataset: Dataset = Depends(get_dataset_or_404),
):
    """Start a backend-owned single-dataset rescan task. Queued on the GPU lane
    (shared with captioning); returns the task id immediately."""
    total = await asyncio.to_thread(count_multimedia, [name])
    task = task_manager.create(
        type="rescan_batch", title=f"Rescan · {name}",
        total=total, dataset_name=name,
    )
    task_manager.enqueue(
        task.id,
        lambda tid: run_rescan_batch(
            tid, dataset_names=[name], force_full=force_full, total=total,
        ),
        lane="gpu",
    )
    return {"task_id": task.id}


@router.post("/datasets/scan-all/batch", response_model=TaskEnqueuedResponse)
async def scan_all_datasets_batch(force_full: bool = Query(False)):
    """Start a backend-owned library-wide rescan task. Runs auto-discovery, then
    queues one file-granular parent task on the GPU lane."""
    names = await asyncio.to_thread(dataset_manager.discover_and_list_dataset_names)
    total = await asyncio.to_thread(count_multimedia, names)
    task = task_manager.create(
        type="rescan_batch", title="Rescan · Library",
        total=total, dataset_name=None,
    )
    task_manager.enqueue(
        task.id,
        lambda tid: run_rescan_batch(
            tid, dataset_names=names, force_full=force_full, total=total,
        ),
        lane="gpu",
    )
    return {"task_id": task.id}


# ── File Upload ──────────────────────────────────────────────────────────


@router.post("/datasets/{name}/upload", response_model=UploadResponse)
async def upload_file(
    name: str,
    file: UploadFile = File(...),
    slot: int = Form(default=0),
    target_stem: str | None = Form(default=None),
    dataset: Dataset = Depends(get_dataset_or_404),
):
    """Upload a file into a dataset directory.

    ``slot=0`` (default) uploads into the dataset root. ``slot=1..3``
    uploads a control image for an existing target: the file is renamed
    to ``{target_stem}{ext}`` inside the slot folder (``control/``,
    ``control_2/``, ``control_3/``) so the stem-pairing convention holds.
    """
    from app.core.dataset.control_helpers import (
        CONTROL_MEDIA_EXTS,
        CONTROL_SLOTS,
    )

    # Sanitize filename to prevent directory traversal via crafted names
    safe_name = sanitize_filename(file.filename or "upload")
    dataset_root = Path(dataset.path)

    if slot == 0:
        save_path = dataset_root / safe_name
        rel_name = safe_name
    else:
        if not 1 <= slot <= len(CONTROL_SLOTS):
            raise HTTPException(
                status_code=400,
                detail=f"slot must be 0..{len(CONTROL_SLOTS)}",
            )
        if not (target_stem or "").strip():
            raise HTTPException(
                status_code=400,
                detail="target_stem is required for control slot uploads",
            )
        target_stem = sanitize_filename(target_stem)
        stems = {
            os.path.splitext(k)[0]
            for k in (dataset.media_metadata or {})
        }
        if target_stem not in stems:
            raise HTTPException(
                status_code=400,
                detail=f"No target image with stem '{target_stem}' in '{name}'",
            )
        ext = Path(safe_name).suffix.lower()
        if ext not in CONTROL_MEDIA_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"Control files must be one of {list(CONTROL_MEDIA_EXTS)}",
            )
        slot_dir = CONTROL_SLOTS[slot - 1]
        (dataset_root / slot_dir).mkdir(parents=True, exist_ok=True)
        # Purge same-stem siblings with other extensions so slot detection
        # (fixed ext priority) stays deterministic after replacement.
        for other_ext in CONTROL_MEDIA_EXTS:
            if other_ext != ext:
                (dataset_root / slot_dir / f"{target_stem}{other_ext}").unlink(
                    missing_ok=True,
                )
        save_path = dataset_root / slot_dir / f"{target_stem}{ext}"
        rel_name = f"{slot_dir}/{target_stem}{ext}"

    logger.info("uploading_file", dataset_name=name, filename=rel_name, slot=slot)

    # Stream to a sibling .tmp file and only os.replace() it onto the final
    # path once the whole body has arrived (mirrors the chunked-read
    # exemplar at crud_routes.py's import_dataset_upload route, and the
    # tmp+replace pattern at dataset/thumbnails.py:88-89). Without this, a
    # client disconnect mid-upload left a truncated file AT the final path —
    # which the next scan indexed as real media — and a re-upload of an
    # existing name truncated the original in place before the copy finished.
    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    try:
        with open(tmp_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)
        os.replace(tmp_path, save_path)
    except OSError as e:
        tmp_path.unlink(missing_ok=True)
        logger.error(
            "upload_failed", dataset_name=name, filename=rel_name, error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))
    except BaseException:
        # Anything else (e.g. the client disconnecting mid-read) still must
        # not leave a truncated file behind — clean up and let it propagate.
        tmp_path.unlink(missing_ok=True)
        raise

    if slot:
        try:
            # Targeted refresh: control flags/dims on the paired media item
            # (no full rescan), emits the media_item entity.changed event.
            await asyncio.to_thread(
                dataset_manager.refresh_control_metadata,
                name,
                target_stem,
            )
        except OSError as e:
            logger.error(
                "upload_failed", dataset_name=name, filename=rel_name, error=str(e)
            )
            raise HTTPException(status_code=500, detail=str(e))

    return {"filename": rel_name, "status": "uploaded"}


# ── Pairs & Media ────────────────────────────────────────────────────────


@router.get("/datasets/{name}/pairs", response_model=list[DatasetPairResponse])
async def get_dataset_pairs(name: str):
    """Return all image-caption pairs for a dataset."""
    try:
        return await asyncio.to_thread(dataset_manager.get_dataset_pairs, name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/datasets/{name}/media")
async def get_dataset_media(
    image_rel_path: str = Query(...), dataset: Dataset = Depends(get_dataset_or_404),
):
    """Serve a media file from a dataset."""
    dataset_root = Path(dataset.path)
    # Validate the resolved path stays inside the dataset directory
    file_path = validate_path_within(dataset_root / image_rel_path, dataset_root)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(file_path))


@router.get("/datasets/{name}/thumbnail")
async def get_dataset_thumbnail(
    image_rel_path: str = Query(...),
    max_edge: int = Query(256),
    dataset: Dataset = Depends(get_dataset_or_404),
):
    """Serve a WebP thumbnail for a dataset image; generates if missing.

    *max_edge* selects the rendition (default 256). It becomes part of the
    cached thumbnail's filename, so it is checked against an allowlist rather
    than a range — an arbitrary integer must never reach a path.
    """
    from app.core.dataset import thumbnails

    if max_edge not in thumbnails.ALLOWED_MAX_EDGES:
        raise HTTPException(
            status_code=400,
            detail=f"max_edge must be one of {list(thumbnails.ALLOWED_MAX_EDGES)}",
        )

    dataset_root = Path(dataset.path)
    # Validate the resolved source path stays inside the dataset directory.
    validate_path_within(dataset_root / image_rel_path, dataset_root)

    thumb_path = await asyncio.to_thread(
        thumbnails.ensure_thumbnail, dataset.path, image_rel_path, max_edge,
    )
    if thumb_path is None:
        raise HTTPException(status_code=404, detail="Thumbnail unavailable")

    etag = f'"{thumb_path.stat().st_mtime_ns}"'
    return FileResponse(
        str(thumb_path),
        media_type="image/webp",
        headers={
            "Cache-Control": "public, max-age=3600",
            "ETag": etag,
        },
    )


@router.delete("/datasets/{name}/pairs/{filename:path}", response_model=MediaPairDeletedResponse)
async def delete_media_pair(name: str, filename: str):
    """Delete a media file and its associated caption."""
    try:
        logger.info("deleting_media_pair", dataset_name=name, filename=filename)
        await asyncio.to_thread(dataset_manager.delete_media_pair, name, filename)
        return {"status": "deleted", "file": filename}
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Captions ─────────────────────────────────────────────────────────────


@router.get("/datasets/{name}/captions/{filename:path}", response_model=CaptionContentResponse)
async def get_caption(name: str, filename: str):
    """Read a caption file's contents."""
    try:
        content = await asyncio.to_thread(dataset_manager.read_caption, name, filename)
        return {"content": content}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/datasets/{name}/captions/{filename:path}", response_model=CaptionSavedResponse)
async def save_caption(name: str, filename: str, request: CaptionRequest):
    """Save or update a caption file."""
    try:
        logger.info("saving_caption", dataset_name=name, filename=filename)
        await asyncio.to_thread(dataset_manager.save_caption, name, filename, request.content)
        return {"status": "saved"}
    except ValueError as e:
        # Matches the GET twin (get_caption) — an unknown dataset is a client
        # error (404), not an INTERNAL_ERROR (was 500).
        raise HTTPException(status_code=404, detail=str(e))


# ── Lyrics (audio sidecar) ───────────────────────────────────────────────
#
# Sibling to the caption endpoints above rather than a caption "variant"
# (see app.core.captioning.caption_variants): lyrics aren't another model's
# rendering of the same caption text, they're a second, independent sidecar
# per audio file (`<stem>.lyrics.txt`, empty/missing allowed). Reuses
# `dataset_manager.read_caption` for reads (filename-driven, media-type
# agnostic) and a dedicated `save_lyrics` for writes (flips `has_lyrics`,
# not `has_caption`, on the matching media entry).


@router.get("/datasets/{name}/lyrics/{filename:path}", response_model=CaptionContentResponse)
async def get_lyrics(name: str, filename: str):
    """Read a lyrics sidecar file's contents."""
    try:
        content = await asyncio.to_thread(dataset_manager.read_caption, name, filename)
        return {"content": content}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/datasets/{name}/lyrics/{filename:path}", response_model=CaptionSavedResponse)
async def save_lyrics(name: str, filename: str, request: CaptionRequest):
    """Save or update a lyrics sidecar file."""
    try:
        logger.info("saving_lyrics", dataset_name=name, filename=filename)
        await asyncio.to_thread(dataset_manager.save_lyrics, name, filename, request.content)
        return {"status": "saved"}
    except ValueError as e:
        # Matches the GET twin (get_lyrics) — an unknown dataset is a client
        # error (404), not an INTERNAL_ERROR (was 500).
        raise HTTPException(status_code=404, detail=str(e))


# ── Image Enable/Disable ────────────────────────────────────────────────


@router.patch(
    "/datasets/{name}/images/{media_file:path}/enabled",
    response_model=ToggleEnabledResponse,
)
async def toggle_image_enabled(name: str, media_file: str, request: ToggleEnabledRequest):
    """Toggle the enabled/disabled state of a single image."""
    try:
        return await asyncio.to_thread(
            dataset_manager.toggle_image_enabled, name, media_file, request.enabled,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/datasets/{name}/images/enable-all", response_model=EnableAllResponse)
async def enable_all_images(name: str):
    """Reset all images in a dataset to enabled."""
    try:
        return await asyncio.to_thread(dataset_manager.enable_all_images, name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Download ─────────────────────────────────────────────────────────────


@router.get("/datasets/{name}/download")
async def download_dataset(name: str, dataset: Dataset = Depends(get_dataset_or_404)):
    """Download a dataset as a zip file (DatasetName_Version.zip).

    W4.T11: streams straight to a temp file (a dataset can be multi-GB of
    video media) instead of building the archive in an in-memory BytesIO —
    mirrors the checkpoint-zip download pattern (job_routes.download_checkpoint_zip).
    """
    dataset_root = Path(dataset.path)
    if not dataset_root.is_dir():
        raise HTTPException(status_code=404, detail="Dataset directory not found on disk")

    zip_filename = f"{dataset.name}_{dataset.version}.zip"
    logger.info("downloading_dataset", dataset_name=name, zip_filename=zip_filename)

    def _build_zip() -> str:
        fd, tmp_path = tempfile.mkstemp(suffix=".zip", prefix="dsdownload_")
        os.close(fd)
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in dataset_root.rglob("*"):
                # Skip derived/cache subdirectories
                if ".cache" in file_path.parts or ".thumbnails" in file_path.parts:
                    continue
                if file_path.is_file():
                    arc_name = file_path.relative_to(dataset_root)
                    zf.write(file_path, arc_name)
        return tmp_path

    tmp_path = await asyncio.to_thread(_build_zip)
    return FileResponse(
        tmp_path,
        media_type="application/zip",
        filename=zip_filename,
        background=BackgroundTask(os.remove, tmp_path),
    )


@router.get("/datasets/{name}/export")
async def export_dataset(name: str, dataset: Dataset = Depends(get_dataset_or_404)):
    """Export a dataset as a portable zip (files + manifest.json metadata)."""
    dataset_root = Path(dataset.path)
    if not dataset_root.is_dir():
        raise HTTPException(status_code=404, detail="Dataset directory not found on disk")

    zip_filename = f"{dataset.name}_{dataset.version}.zip"
    logger.info("exporting_dataset", dataset_name=name, zip_filename=zip_filename)

    def _build() -> io.BytesIO:
        manifest = portable.build_manifest(dataset, app_version=APP_VERSION)
        return portable.write_export_zip(dataset_root, manifest)

    buf = await asyncio.to_thread(_build)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )


def _sanitize_ds_name(name: str) -> str:
    """Folder-safe dataset name.

    The on-disk folder is named after the dataset name, and the frontend serves
    media statically as ``/media/<name>/...`` — so the stored name MUST equal
    its folder basename, or images 404 while captions (loaded via the API) still
    work. Strips everything except alphanumerics, space, hyphen and underscore,
    matching the folder sanitizer in ``DatasetManager.create_dataset``/rename.
    """
    cleaned = "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip()
    return cleaned or f"dataset_{int(time.time())}"


def _resolve_import_name(name: str, on_conflict: str | None, new_name: str | None) -> str:
    """Decide the final dataset name, applying the collision directive.

    The returned name is sanitized so it matches its on-disk folder — the import
    cleanup that renames the folder must apply to the name too, else the name and
    folder diverge and ``/media/<name>`` can't find the files.

    Raises HTTPException(409) when the name is taken and no directive is given.
    """
    base = _sanitize_ds_name(name)
    existing = dataset_manager.get_dataset(base)
    if existing is None:
        return base
    if on_conflict == "overwrite":
        # v1 limitation: the existing dataset is deleted here, before extraction.
        # Callers reach this only after read_manifest() succeeds, so a corrupt
        # archive won't destroy existing data — but a failure during extract or
        # register afterwards leaves neither old nor new. Acceptable for the
        # local export -> delete -> re-import flow; revisit (import-to-temp,
        # swap on success) if this becomes a real risk.
        dataset_manager.delete_dataset(base, delete_files=True)
        return base
    if on_conflict == "rename":
        # Suffix from the resolved base (the user's new_name when given, else
        # "<name> (imported)") — NOT the original name, or a colliding custom
        # rename would be silently discarded. Every candidate is sanitized so the
        # final name stays folder-safe (the parens in "(imported)" are stripped).
        rename_base = (
            _sanitize_ds_name(new_name) if (new_name or "").strip()
            else _sanitize_ds_name(f"{base} (imported)")
        )
        candidate = rename_base
        i = 2
        while dataset_manager.get_dataset(candidate) is not None:
            candidate = _sanitize_ds_name(f"{rename_base} {i}")
            i += 1
        return candidate
    # No directive -> tell the client to prompt.
    raise HTTPException(
        status_code=409,
        detail={"conflict": True, "name": base,
                "message": f"A dataset named '{base}' already exists."},
    )


def _import_from_zip_path(zip_path: Path, on_conflict: str | None, new_name: str | None):
    """Validate, extract, and register a dataset from a zip already on disk.

    The dataset name and its on-disk folder are kept identical (both the
    sanitized ``final_name``) so the static ``/media/<name>`` route resolves — a
    divergence here surfaces as captions present but images blank.
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            manifest = portable.read_manifest(zf)
            source_name = str(manifest.get("dataset", {}).get("name") or "Imported")
            # Already filesystem-safe → folder basename == stored name.
            final_name = _resolve_import_name(source_name, on_conflict, new_name)

            target = Path(dataset_manager.default_root) / final_name
            target.mkdir(parents=True, exist_ok=True)
            try:
                portable.safe_extract(zf, target)
                return dataset_manager.register_imported_dataset(
                    final_name, manifest, path=str(target)
                )
            except Exception:
                # Roll back a half-written import: drop the folder + any row.
                if dataset_manager.get_dataset(final_name) is not None:
                    dataset_manager.delete_dataset(final_name, delete_files=True)
                else:
                    shutil.rmtree(target, ignore_errors=True)
                raise
    except portable.ManifestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Archive is not a valid zip file.") from exc


@router.post("/datasets/import", response_model=Dataset)
async def import_dataset_upload(
    file: UploadFile = File(...),
    on_conflict: Literal["rename", "overwrite"] | None = Form(default=None),
    new_name: str | None = Form(default=None),
):
    """Import a dataset from an uploaded portable zip (multipart)."""
    # Stream the upload to a temp file — never buffer multi-GB archives in RAM.
    suffix = Path(sanitize_filename(file.filename or "import.zip")).suffix or ".zip"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        while chunk := await file.read(1024 * 1024):
            tmp.write(chunk)
        tmp.close()
        logger.info("importing_dataset_upload", filename=file.filename)
        return await asyncio.to_thread(
            _import_from_zip_path, Path(tmp.name), on_conflict, new_name
        )
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@router.post("/datasets/import-path", response_model=Dataset)
async def import_dataset_path(request: ImportPathRequest):
    """Import a dataset from a zip already present on the server filesystem.

    The archive path is intentionally unrestricted: this is a local
    single-user app and the "server path" transport exists precisely so the
    user can point at an export anywhere on their own disk (e.g.
    ``D:/exports/Foo_1.2.0.zip``). Extraction targets are still confined to
    ``default_root`` by ``portable.safe_extract``.
    """
    archive = Path(request.archive_path)
    if not archive.is_file():
        raise HTTPException(status_code=404, detail=f"Archive not found: {request.archive_path}")
    logger.info("importing_dataset_path", archive_path=str(archive))
    return await asyncio.to_thread(
        _import_from_zip_path, archive, request.on_conflict, request.new_name
    )
