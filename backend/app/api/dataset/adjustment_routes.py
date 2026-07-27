"""Image adjustment routes — single/batch adjustments, color match, histogram, cube export."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from PIL import Image

from app.api._deps import dataset_or_404
from app.api._path_guard import reject_audio_op, validate_path_within
from app.core.dataset_manager import Dataset, dataset_manager
from app.core.logger import get_logger
from app.api.schemas.adjustment_schemas import (
    ColorMatchRequest,
    CurvePointModel,
    ExportCubeRequest,
    HistogramResponse,
)

router = APIRouter()
logger = get_logger(__name__)


def get_dataset_or_404(name: str) -> Dataset:
    """Path-operation dependency: resolve a dataset by name or 404."""
    return dataset_or_404(dataset_manager.get_dataset(name))


# ── Color Match Preview ─────────────────────────────────────────────────


@router.post("/datasets/{name}/color-match")
async def color_match_preview(
    name: str, request: ColorMatchRequest, dataset: Dataset = Depends(get_dataset_or_404),
):
    """Return a color-matched preview image as JPEG (non-destructive)."""
    from app.core.image_adjustments import apply_color_match

    try:
        dataset_root = Path(dataset.path)
        src_path = validate_path_within(
            dataset_root / request.source_path, dataset_root
        )
        ref_path = validate_path_within(
            dataset_root / request.reference_path, dataset_root
        )

        if not src_path.exists():
            raise HTTPException(status_code=404, detail=f"Source not found: {request.source_path}")
        if not ref_path.exists():
            raise HTTPException(status_code=404, detail=f"Reference not found: {request.reference_path}")
        reject_audio_op(request.source_path, "Color match")
        reject_audio_op(request.reference_path, "Color match")

        def _preview() -> bytes:
            with Image.open(src_path) as src_img, Image.open(ref_path) as ref_img:
                result = apply_color_match(
                    src_img.convert("RGB"),
                    ref_img.convert("RGB"),
                    strength=request.strength,
                    method=request.method,
                )
                buf = io.BytesIO()
                result.save(buf, format="JPEG", quality=92)
                buf.seek(0)
                return buf.getvalue()

        jpeg_bytes = await asyncio.to_thread(_preview)
        return StreamingResponse(io.BytesIO(jpeg_bytes), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Histogram ────────────────────────────────────────────────────────────


@router.get("/datasets/{name}/histogram", response_model=HistogramResponse)
async def get_histogram(
    name: str, image_path: str = Query(...), dataset: Dataset = Depends(get_dataset_or_404),
):
    """Return per-channel histogram data for an image."""
    from app.core.image_adjustments import compute_histogram

    dataset_root = Path(dataset.path)
    full_path = validate_path_within(dataset_root / image_path, dataset_root)
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    reject_audio_op(image_path, "Histogram")

    def _compute():
        from PIL import Image as PILImage
        with PILImage.open(full_path) as img:
            return compute_histogram(img)

    return await asyncio.to_thread(_compute)


# ── Cube LUT Export ──────────────────────────────────────────────────────


@router.post("/datasets/{name}/export-cube")
async def export_cube(name: str, request: ExportCubeRequest):
    """Export current curves configuration as a downloadable .cube file."""
    from app.core.image_adjustments import export_curves_as_cube, CurvePoint

    def _to_points(pts: list[CurvePointModel]) -> list[CurvePoint]:
        return [CurvePoint(x=p.x, y=p.y) for p in pts]

    cube_str = export_curves_as_cube(
        master=_to_points(request.curves.master) or None,
        r=_to_points(request.curves.r) or None,
        g=_to_points(request.curves.g) or None,
        b=_to_points(request.curves.b) or None,
        size=request.size,
    )
    return StreamingResponse(
        io.BytesIO(cube_str.encode("utf-8")),
        media_type="text/plain",
        headers={"Content-Disposition": 'attachment; filename="exported_curves.cube"'},
    )
