"""Image adjustment routes — single/batch adjustments, color match, histogram, cube export."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from PIL import Image

from app.api._deps import dataset_or_404
from app.api._path_guard import reject_audio_op, validate_path_within
from app.core.dataset_manager import Dataset, dataset_manager
from app.core.logger import get_logger
from app.api.schemas.adjustment_schemas import (
    AdjustmentRequest,
    AdjustResponse,
    BatchAdjustmentRequest,
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


# ── Helpers ──────────────────────────────────────────────────────────────


def _build_adjustments_dict(request: AdjustmentRequest | BatchAdjustmentRequest) -> dict:
    """Convert Pydantic request models into the dict format expected by apply_all."""
    adjustments: dict = {}
    if hasattr(request, 'cube_lut') and request.cube_lut:
        adjustments["cube_lut"] = request.cube_lut
        adjustments["cube_lut_strength"] = getattr(request, 'cube_lut_strength', 1.0)
    if hasattr(request, 'curves') and request.curves:
        adjustments["curves"] = {
            "master": [p.model_dump() for p in request.curves.master],
            "r": [p.model_dump() for p in request.curves.r],
            "g": [p.model_dump() for p in request.curves.g],
            "b": [p.model_dump() for p in request.curves.b],
        }
    if request.hue_shift != 0.0:
        adjustments["hue_shift"] = request.hue_shift
    if request.saturation != 1.0:
        adjustments["saturation"] = request.saturation
    if request.contrast != 1.0:
        adjustments["contrast"] = request.contrast
    if request.sharpening:
        adjustments["sharpening"] = {
            "method": request.sharpening.method,
            "params": request.sharpening.params,
        }
    if request.white_balance:
        adjustments["white_balance"] = request.white_balance.model_dump()
    if request.vignette:
        adjustments["vignette"] = request.vignette.model_dump()
    if request.lens_correction:
        adjustments["lens_correction"] = request.lens_correction.model_dump()
    if request.hsl_selective:
        adjustments["hsl_selective"] = {
            k: v.model_dump() for k, v in request.hsl_selective.items()
        }
    return adjustments


# ── Single Image Adjustment ─────────────────────────────────────────────


@router.post("/datasets/{name}/adjust", response_model=AdjustResponse)
async def adjust_media(name: str, request: AdjustmentRequest):
    """Apply image adjustments."""
    try:
        logger.info("adjusting_media", dataset_name=name, path=request.path)
        adjustments = _build_adjustments_dict(request)

        # dataset_manager.apply_adjustments opens `request.path` and overwrites
        # it in place (WRITE primitive) — validate containment before it's
        # ever handed a client-supplied relative path. Skip when the dataset
        # itself can't be resolved; apply_adjustments raises its own
        # ValueError("Dataset not found") in that case, unchanged.
        dataset = await asyncio.to_thread(dataset_manager.get_dataset, name)
        if dataset:
            dataset_root = Path(dataset.path)
            validate_path_within(dataset_root / request.path, dataset_root)

            # Resolve color match reference path to absolute
            if request.color_match:
                validate_path_within(
                    dataset_root / request.color_match.reference_path,
                    dataset_root,
                )
                adjustments["color_match"] = {
                    "reference_path": str(
                        dataset_root / request.color_match.reference_path
                    ),
                    "method": request.color_match.method,
                    "strength": request.color_match.strength,
                }

        await asyncio.to_thread(
            dataset_manager.apply_adjustments,
            name,
            request.path,
            adjustments,
        )
        return {"status": "adjusted", "file": request.path}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Batch Adjustment ────────────────────────────────────────────────────


@router.post("/datasets/{name}/adjust-batch")
async def adjust_media_batch(name: str, request: BatchAdjustmentRequest):
    """Apply adjustments to multiple images. Returns SSE progress stream."""
    adjustments = _build_adjustments_dict(request)
    total = len(request.paths)
    # Resolved once, outside the loop; per-item validation below still
    # 403-guards each path individually (a traversal is reported as a
    # per-item error event — the 200 + SSE stream is already committed by
    # the time any item is processed, so a raised HTTPException can't
    # become the response status the way it does on the single-item route).
    dataset = await asyncio.to_thread(dataset_manager.get_dataset, name)
    dataset_root = Path(dataset.path) if dataset else None

    async def event_stream():
        for idx, path in enumerate(request.paths):
            try:
                if dataset_root is not None:
                    validate_path_within(dataset_root / path, dataset_root)
                await asyncio.to_thread(
                    dataset_manager.apply_adjustments,
                    name,
                    path,
                    adjustments,
                )
                event = {"index": idx, "total": total, "file": path, "status": "ok"}
            except HTTPException as e:
                event = {
                    "index": idx,
                    "total": total,
                    "file": path,
                    "status": "error",
                    "error": e.detail,
                }
            except (ValueError, FileNotFoundError, OSError) as e:
                event = {"index": idx, "total": total, "file": path, "status": "error", "error": str(e)}
            yield f"data: {json.dumps(event)}\n\n"
        yield f"data: {json.dumps({'done': True, 'total': total})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
