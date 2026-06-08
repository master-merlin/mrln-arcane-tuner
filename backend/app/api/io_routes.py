"""Generic import helpers shared across artifact kinds (peek)."""

from __future__ import annotations

import asyncio
import io
import zipfile
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(prefix="/import", tags=["import"])


@router.post("/peek")
async def peek_archive(file: UploadFile = File(...)) -> dict[str, Any]:
    """Return the archive's manifest header (``kind`` + versions) so the client
    can route it to the correct importer."""
    from app.core.portable.envelope import ManifestError, peek_manifest

    data = await file.read()

    def _peek() -> dict[str, Any]:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                return peek_manifest(zf)
        except ManifestError as exc:
            raise HTTPException(400, str(exc)) from exc
        except zipfile.BadZipFile as exc:
            raise HTTPException(400, "Archive is not a valid zip file.") from exc

    return await asyncio.to_thread(_peek)
