"""Generic import helpers shared across artifact kinds (peek)."""

from __future__ import annotations

import asyncio
import zipfile
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.api._upload_guard import spooled_upload

router = APIRouter(prefix="/import", tags=["import"])


class ArchivePeekResponse(BaseModel):
    """Archive manifest header — lets the client route a dropped archive to
    the correct importer (project/template/dataset). Mirrors
    ``portable.envelope.peek_manifest`` exactly: ``kind`` is required by that
    function (it raises if absent), the versions are best-effort echoes of
    whatever the manifest carried."""

    kind: str
    format_version: Any = None
    app_version: Any = None


@router.post("/peek", response_model=ArchivePeekResponse)
async def peek_archive(file: UploadFile = File(...)) -> dict[str, Any]:
    """Return the archive's manifest header (``kind`` + versions) so the client
    can route it to the correct importer."""
    from app.core.portable.envelope import ManifestError, peek_manifest

    # Stream the upload to a temp file — never buffer a whole archive in RAM
    # just to read its manifest header (archives can embed multi-GB video
    # datasets) — and bound the stream, which reading to EOF did not. The
    # context manager also owns cleanup, including for a client that
    # disconnects mid-upload.
    async with spooled_upload(file, suffix=".zip") as tmp_path:

        def _peek() -> dict[str, Any]:
            try:
                with zipfile.ZipFile(tmp_path) as zf:
                    return peek_manifest(zf)
            except ManifestError as exc:
                raise HTTPException(400, str(exc)) from exc
            except zipfile.BadZipFile as exc:
                raise HTTPException(400, "Archive is not a valid zip file.") from exc

        return await asyncio.to_thread(_peek)
