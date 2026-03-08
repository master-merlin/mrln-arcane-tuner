"""Dataset CRUD, scanning, upload, pairs, captions, enable/disable, and download routes."""

from __future__ import annotations

import asyncio
import io
import os
import shutil
import zipfile

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse, StreamingResponse

from app.core.dataset_manager import dataset_manager, Dataset
from app.core.logger import get_logger
from app.api.schemas.dataset_schemas import (
    CreateDatasetRequest,
    UpdateDatasetRequest,
    CaptionRequest,
    ToggleEnabledRequest,
)

router = APIRouter()
logger = get_logger(__name__)


# ── Dataset CRUD ─────────────────────────────────────────────────────────


@router.get("/datasets", response_model=list[Dataset])
async def list_datasets():
    """List all registered datasets."""
    return await asyncio.to_thread(dataset_manager.list_datasets)


@router.get("/datasets/{name}", response_model=Dataset)
async def get_dataset(name: str):
    """Return a single dataset by name."""
    dataset = await asyncio.to_thread(dataset_manager.get_dataset, name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
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
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/datasets/{name}", response_model=Dataset)
async def update_dataset(name: str, request: UpdateDatasetRequest):
    """Update dataset metadata (name, description, classifier)."""
    try:
        logger.info("updating_dataset", old_name=name, new_name=request.name)
        return await asyncio.to_thread(
            dataset_manager.update_dataset,
            name,
            request.name,
            request.description,
            new_classifier=request.classifier,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/datasets/{name}")
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


@router.post("/datasets/scan-all", response_model=list[Dataset])
async def scan_all_datasets():
    """Re-scan all registered datasets."""
    logger.info("scanning_all_datasets")
    return await asyncio.to_thread(dataset_manager.scan_all_datasets)


# ── File Upload ──────────────────────────────────────────────────────────


@router.post("/datasets/{name}/upload")
async def upload_file(name: str, file: UploadFile = File(...)):
    """Upload a file into a dataset directory."""
    dataset = await asyncio.to_thread(dataset_manager.get_dataset, name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    save_path = os.path.join(dataset.path, file.filename)
    logger.info("uploading_file", dataset_name=name, filename=file.filename)

    try:
        def save_file():
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        await asyncio.to_thread(save_file)
        return {"filename": file.filename, "status": "uploaded"}
    except OSError as e:
        logger.error("upload_failed", dataset_name=name, filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── Pairs & Media ────────────────────────────────────────────────────────


@router.get("/datasets/{name}/pairs")
async def get_dataset_pairs(name: str):
    """Return all image-caption pairs for a dataset."""
    try:
        return await asyncio.to_thread(dataset_manager.get_dataset_pairs, name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/datasets/{name}/media")
async def get_dataset_media(name: str, image_rel_path: str = Query(...)):
    """Serve a media file from a dataset."""
    dataset = await asyncio.to_thread(dataset_manager.get_dataset, name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = os.path.join(dataset.path, image_rel_path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path)


@router.delete("/datasets/{name}/pairs/{filename:path}")
async def delete_media_pair(name: str, filename: str):
    """Delete a media file and its associated caption."""
    try:
        logger.info("deleting_media_pair", dataset_name=name, filename=filename)
        await asyncio.to_thread(dataset_manager.delete_media_pair, name, filename)
        return {"status": "deleted", "file": filename}
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Captions ─────────────────────────────────────────────────────────────


@router.get("/datasets/{name}/captions/{filename:path}")
async def get_caption(name: str, filename: str):
    """Read a caption file's contents."""
    try:
        content = await asyncio.to_thread(dataset_manager.read_caption, name, filename)
        return {"content": content}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/datasets/{name}/captions/{filename:path}")
async def save_caption(name: str, filename: str, request: CaptionRequest):
    """Save or update a caption file."""
    try:
        logger.info("saving_caption", dataset_name=name, filename=filename)
        await asyncio.to_thread(dataset_manager.save_caption, name, filename, request.content)
        return {"status": "saved"}
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Image Enable/Disable ────────────────────────────────────────────────


@router.patch("/datasets/{name}/images/{media_file:path}/enabled")
async def toggle_image_enabled(name: str, media_file: str, request: ToggleEnabledRequest):
    """Toggle the enabled/disabled state of a single image."""
    try:
        return await asyncio.to_thread(
            dataset_manager.toggle_image_enabled, name, media_file, request.enabled,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/datasets/{name}/images/enable-all")
async def enable_all_images(name: str):
    """Reset all images in a dataset to enabled."""
    try:
        return await asyncio.to_thread(dataset_manager.enable_all_images, name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Download ─────────────────────────────────────────────────────────────


@router.get("/datasets/{name}/download")
async def download_dataset(name: str):
    """Download a dataset as a zip file (DatasetName_Version.zip)."""
    dataset = await asyncio.to_thread(dataset_manager.get_dataset, name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not os.path.isdir(dataset.path):
        raise HTTPException(status_code=404, detail="Dataset directory not found on disk")

    zip_filename = f"{dataset.name}_{dataset.version}.zip"
    logger.info("downloading_dataset", dataset_name=name, zip_filename=zip_filename)

    def _build_zip() -> io.BytesIO:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(dataset.path):
                # Skip .cache subdirectory
                dirs[:] = [d for d in dirs if d != ".cache"]
                for file in files:
                    abs_path = os.path.join(root, file)
                    arc_name = os.path.relpath(abs_path, dataset.path)
                    zf.write(abs_path, arc_name)
        buf.seek(0)
        return buf

    buf = await asyncio.to_thread(_build_zip)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )
