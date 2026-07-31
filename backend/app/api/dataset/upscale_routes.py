"""Upscaling routes — model listing and neural upscale with tiled inference."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.api._deps import dataset_or_404
from app.api._path_guard import (
    reject_audio_op,
    safe_remove,
    validate_path_in_allowed_roots,
    validate_path_within,
)
from app.core.dataset_manager import Dataset, dataset_manager
from app.core.logger import get_logger
from app.api.schemas.upscale_schemas import (
    UpscaleListRequest, UpscaleApplyRequest,
    UpscaleListResponse, UpscaleApplyResponse,
)

router = APIRouter()
logger = get_logger(__name__)


def get_dataset_or_404(name: str) -> Dataset:
    """Path-operation dependency: resolve a dataset by name or 404."""
    return dataset_or_404(dataset_manager.get_dataset(name))


_DEFAULT_UPSCALE_FOLDER = (
    Path(__file__).resolve().parents[2] / "engine" / "models" / "upscale"
)
# Also check the legacy path
_LEGACY_UPSCALE_FOLDER = (
    Path(__file__).resolve().parents[3] / "models" / "upscale"
)


@router.post("/upscale/list-models", response_model=UpscaleListResponse)
async def list_upscale_models(request: UpscaleListRequest):
    """Scan a folder for upscale model files (.pth, .safetensors)."""
    folder_str = request.folder.strip() if request.folder else ""
    if not folder_str:
        folder = _DEFAULT_UPSCALE_FOLDER
        if not folder.is_dir():
            folder = _LEGACY_UPSCALE_FOLDER
    else:
        # A client-named folder is an absolute path — contain it to the
        # operator-tool roots. Unguarded, this route was an arbitrary directory
        # lister (returning names, full paths and sizes for every model-suffixed
        # file it found).
        folder = validate_path_in_allowed_roots(folder_str)

    if not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder}")

    model_exts = {".pth", ".safetensors", ".safetensor", ".pt", ".onnx", ".bin"}

    def _scan_models() -> list[dict]:
        items: list[dict] = []
        for f in folder.iterdir():
            if f.is_file() and f.suffix.lower() in model_exts:
                size_mb = f.stat().st_size / (1024 * 1024)
                items.append({
                    "name": f.name,
                    "path": str(f),
                    "size_mb": round(size_mb, 1),
                })
        return items

    models = await asyncio.to_thread(_scan_models)
    models.sort(key=lambda m: m["name"])
    return {"models": models, "folder": str(folder)}


@router.post("/datasets/{name}/upscale", response_model=UpscaleApplyResponse)
async def upscale_media(
    name: str, request: UpscaleApplyRequest, dataset: Dataset = Depends(get_dataset_or_404),
):
    """Upscale an image using a selected model with tiled inference."""
    try:
        import torch  # noqa: F401 — verify availability
        import spandrel  # noqa: F401
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Missing dependencies: pip install spandrel torch"
        )

    # Request-shape validation first: "Upscale is not supported for audio" is a
    # more useful 400 than a 403/404 about some incidental path detail.
    reject_audio_op(request.image_path, "Upscale")

    dataset_root = Path(dataset.path)
    # Containment BEFORE any IO: this route reads ``img_path`` with PIL and then
    # writes the upscaled pixels back over it, so an escaping ``image_path`` was
    # an arbitrary-file OVERWRITE primitive (the exists() check below makes it
    # overwrite-only, which still reaches the database, a definition yaml, a
    # model .safetensors, or the served frontend bundle).
    img_path = validate_path_within(dataset_root / request.image_path, dataset_root)
    # The model is handed to spandrel, which torch.loads ``.pth`` checkpoints —
    # an unrestricted path made that an arbitrary-pickle load sink.
    model_path = validate_path_in_allowed_roots(request.model_path)

    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    def _upscale():
        from PIL import Image as PILImage

        from app.core.image_processing.tiled_inference import (
            cleanup_vram,
            image_to_tensor,
            load_spandrel_model,
            run_tiled_inference,
            tensor_to_image,
        )

        model, scale, device = load_spandrel_model(model_path)

        with PILImage.open(img_path) as img:
            img_rgb = img.convert("RGB")
            w_orig, h_orig = img_rgb.size
            img_tensor = image_to_tensor(img_rgb, device)

        output_tensor = run_tiled_inference(
            model,
            img_tensor,
            tile_size=request.tile_size,
            tile_pad=request.tile_pad,
            scale=scale,
        )
        result_img = tensor_to_image(output_tensor)
        cleanup_vram(model, img_tensor, output_tensor)

        out_w, out_h = result_img.size

        # Post-process rescale if target differs from model native
        effective_scale = scale
        final_w, final_h = out_w, out_h
        target = request.target_scale
        if target > 0 and abs(target - scale) > 0.01:
            final_w = round(w_orig * target)
            final_h = round(h_orig * target)
            resample_map = {
                "lanczos": PILImage.LANCZOS,
                "bicubic": PILImage.BICUBIC,
                "bilinear": PILImage.BILINEAR,
                "nearest": PILImage.NEAREST,
            }
            resample = resample_map.get(request.resize_method, PILImage.LANCZOS)
            result_img = result_img.resize((final_w, final_h), resample)
            effective_scale = target

        result_img.save(str(img_path), quality=95)

        return {"scale": effective_scale, "new_size": [final_w, final_h]}

    try:
        result = await asyncio.to_thread(_upscale)

        # Invalidate masks & masked images — dimensions changed (threaded)
        stem = Path(request.image_path).stem
        for suffix_path in [
            dataset_root / "masks" / f"{stem}.png",
            dataset_root / "masked" / f"{stem}.jpg",
            dataset_root / "masked" / f"{stem}.txt",
        ]:
            await asyncio.to_thread(safe_remove, suffix_path)

        # Update cached metadata so /pairs returns correct dimensions
        lookup_key = request.image_path.replace("\\", "/")
        if lookup_key in dataset.media_metadata:
            def _size_if_exists(p: Path) -> int | None:
                return p.stat().st_size if p.exists() else None

            new_size = await asyncio.to_thread(_size_if_exists, img_path)

            changes: dict[str, Any] = {
                "width": result["new_size"][0],
                "height": result["new_size"][1],
                "has_mask": False,
                "has_masked": False,
                "has_masked_caption": False,
                "mask_info": dataset_manager.REMOVE_FIELD,
            }
            if new_size is not None:
                changes["size_bytes"] = new_size
            # Persist to DB atomically (previously in-memory only — lost on restart)
            await dataset_manager.update_media_flags_async(
                name, request.image_path, **changes,
            )

        # Source pixels were overwritten — refresh thumbnail.
        from app.core.dataset import thumbnails

        await asyncio.to_thread(
            thumbnails.invalidate_thumbnail, dataset.path, request.image_path,
        )
        await asyncio.to_thread(
            thumbnails.ensure_thumbnail, dataset.path, request.image_path,
        )

        # Re-apply the overlay recipe to the upscaled pixels (or drop the stale
        # overlay if it can't be re-rendered) — same reconciliation as crop /
        # adjust. Run in a thread: the re-render is blocking (GPU) work.
        await asyncio.to_thread(
            dataset_manager._reconcile_overlay_after_edit, dataset, request.image_path,
        )

        # Bump patch version for destructive upscale
        await asyncio.to_thread(dataset_manager.bump_dataset_version, name, "patch")
        return {"status": "upscaled", "file": request.image_path, **result}
    except (OSError, RuntimeError, MemoryError) as e:
        raise HTTPException(status_code=500, detail=str(e))
