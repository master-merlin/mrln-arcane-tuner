"""Upscaling routes — model listing and neural upscale with tiled inference."""

from __future__ import annotations

import asyncio
import gc
import os

from fastapi import APIRouter, HTTPException

from app.core.dataset_manager import dataset_manager
from app.core.logger import get_logger
from app.api.schemas.upscale_schemas import UpscaleListRequest, UpscaleApplyRequest

router = APIRouter()
logger = get_logger(__name__)

_DEFAULT_UPSCALE_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "engine", "models", "upscale"
)
# Also check the legacy path
_LEGACY_UPSCALE_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "models", "upscale"
)


@router.post("/upscale/list-models")
async def list_upscale_models(request: UpscaleListRequest):
    """Scan a folder for upscale model files (.pth, .safetensors)."""
    folder = request.folder.strip() if request.folder else ""
    if not folder:
        folder = os.path.normpath(_DEFAULT_UPSCALE_FOLDER)
        if not os.path.isdir(folder):
            folder = os.path.normpath(_LEGACY_UPSCALE_FOLDER)
    if not os.path.isdir(folder):
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder}")

    model_exts = {".pth", ".safetensors", ".safetensor", ".pt", ".onnx", ".bin"}
    models = []
    for f in os.listdir(folder):
        ext = os.path.splitext(f)[1].lower()
        if ext in model_exts:
            full_path = os.path.join(folder, f)
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            models.append({
                "name": f,
                "path": full_path,
                "size_mb": round(size_mb, 1),
            })
    models.sort(key=lambda m: m["name"])
    return {"models": models, "folder": folder}


@router.post("/datasets/{name}/upscale")
async def upscale_media(name: str, request: UpscaleApplyRequest):
    """Upscale an image using a selected model with tiled inference."""
    try:
        import torch
        from spandrel import ModelLoader
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Missing dependencies: pip install spandrel torch"
        )

    dataset = await asyncio.to_thread(dataset_manager.get_dataset, name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    img_path = os.path.join(dataset.path, request.image_path)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")
    if not os.path.exists(request.model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    def _upscale():
        import numpy as np
        from PIL import Image as PILImage

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model via spandrel (supports ESRGAN, RealESRGAN, SwinIR, etc.)
        model = ModelLoader().load_from_file(request.model_path)
        model = model.to(device).eval()

        # Load image
        with PILImage.open(img_path) as img:
            img_rgb = img.convert("RGB")
            img_np = np.array(img_rgb).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        h, w = img_tensor.shape[2], img_tensor.shape[3]
        # Detect model scale — spandrel stores it on the descriptor
        scale = getattr(model, "scale", None)
        if scale is None:
            with torch.inference_mode():
                test_out = model(torch.zeros(1, 3, 8, 8, device=device))
                scale = test_out.shape[2] // 8
            if scale < 1:
                scale = 4
        out_h, out_w = h * scale, w * scale
        output = torch.zeros(1, 3, out_h, out_w, device=device)

        tile = request.tile_size
        pad = request.tile_pad

        # Tiled inference
        with torch.inference_mode():
            for y in range(0, h, tile):
                for x in range(0, w, tile):
                    y1 = max(0, y - pad)
                    x1 = max(0, x - pad)
                    y2 = min(h, y + tile + pad)
                    x2 = min(w, x + tile + pad)

                    tile_in = img_tensor[:, :, y1:y2, x1:x2]
                    tile_out = model(tile_in)

                    # Calculate output region (remove padding)
                    oy1 = (y - y1) * scale
                    ox1 = (x - x1) * scale
                    oy2 = oy1 + min(tile, h - y) * scale
                    ox2 = ox1 + min(tile, w - x) * scale

                    out_y1 = y * scale
                    out_x1 = x * scale
                    out_y2 = out_y1 + min(tile, h - y) * scale
                    out_x2 = out_x1 + min(tile, w - x) * scale

                    output[:, :, out_y1:out_y2, out_x1:out_x2] = tile_out[:, :, oy1:oy2, ox1:ox2]

        result_np = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        result_img = PILImage.fromarray((result_np * 255).astype(np.uint8))

        # Post-process rescale if target differs from model native
        effective_scale = scale
        final_w, final_h = out_w, out_h
        target = request.target_scale
        if target > 0 and abs(target - scale) > 0.01:
            final_w = round(w * target)
            final_h = round(h * target)
            resample_map = {
                "lanczos": PILImage.LANCZOS,
                "bicubic": PILImage.BICUBIC,
                "bilinear": PILImage.BILINEAR,
                "nearest": PILImage.NEAREST,
            }
            resample = resample_map.get(request.resize_method, PILImage.LANCZOS)
            result_img = result_img.resize((final_w, final_h), resample)
            effective_scale = target

        result_img.save(img_path, quality=95)

        # Cleanup VRAM
        del model, img_tensor, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {"scale": effective_scale, "new_size": [final_w, final_h]}

    try:
        result = await asyncio.to_thread(_upscale)
        # Invalidate masks & masked images — dimensions changed
        stem = os.path.splitext(request.image_path)[0]
        for suffix in [
            os.path.join("masks", f"{stem}.png"),
            os.path.join("masked", f"{stem}.jpg"),
            os.path.join("masked", f"{stem}.txt"),
        ]:
            p = os.path.join(dataset.path, suffix)
            if os.path.exists(p):
                os.remove(p)
        # Update cached metadata so /pairs returns correct dimensions
        lookup_key = request.image_path.replace(os.sep, '/')
        if lookup_key in dataset.media_metadata:
            dataset.media_metadata[lookup_key]["width"] = result["new_size"][0]
            dataset.media_metadata[lookup_key]["height"] = result["new_size"][1]
            dataset.media_metadata[lookup_key]["has_mask"] = False
            dataset.media_metadata[lookup_key]["has_masked"] = False
            dataset.media_metadata[lookup_key]["has_masked_caption"] = False
            dataset.media_metadata[lookup_key].pop("mask_info", None)
            real_path = os.path.join(dataset.path, request.image_path)
            if os.path.exists(real_path):
                dataset.media_metadata[lookup_key]["size_bytes"] = os.path.getsize(real_path)
            # Persist to DB (previously in-memory only — lost on restart)
            dataset_manager._persist_media_item(dataset, request.image_path)
        # Bump patch version for destructive upscale
        await asyncio.to_thread(dataset_manager.bump_dataset_version, name, "patch")
        return {"status": "upscaled", "file": request.image_path, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
