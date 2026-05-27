"""Model registry — curated database of pretrained restore & upscale models.

Inspired by cszn/SCUNet's download approach: a flat registry with skip-if-exists
semantics and direct GitHub release URLs.  The registry is intentionally small —
only models that are known to work well with Spandrel's auto-architecture detection.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model Database
# ---------------------------------------------------------------------------

ModelEntry = dict  # {"url": str, "size_mb": float, "description": str}

MODEL_DB: dict[str, dict[str, ModelEntry]] = {
    # ── Restoration / Denoise ─────────────────────────────────────────────
    "restore": {
        "scunet_color_real_gan.pth": {
            "url": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth",
            "size_mb": 68.0,
            "description": "SCUNet Color GAN — real-world denoise, sharper output",
        },
        "scunet_color_real_psnr.pth": {
            "url": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
            "size_mb": 68.0,
            "description": "SCUNet Color PSNR — real-world denoise, smoother output",
        },
        "scunet_color_25.pth": {
            "url": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_25.pth",
            "size_mb": 68.0,
            "description": "SCUNet Color σ=25 — moderate synthetic noise",
        },
        "scunet_color_50.pth": {
            "url": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_50.pth",
            "size_mb": 68.0,
            "description": "SCUNet Color σ=50 — heavy synthetic noise",
        },
    },
    # ── Upscale ───────────────────────────────────────────────────────────
    "upscale": {
        "4x_NMKD-Siax_200k.pth": {
            "url": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Siax_200k.pth",
            "size_mb": 66.8,
            "description": "NMKD-Siax 4x — balanced general-purpose upscale",
        },
        "4x-UltraSharp.pth": {
            "url": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth",
            "size_mb": 66.8,
            "description": "UltraSharp 4x — maximum detail, slightly over-sharpened",
        },
        "RealESRGAN_x2plus.pth": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "size_mb": 64.0,
            "description": "RealESRGAN 2x — official xinntao 2x upscale model",
        },
        "OmniSR_X2_DIV2K.safetensors": {
            "url": "https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X2_DIV2K.safetensors",
            "size_mb": 1.7,
            "description": "OmniSR 2x — lightweight 2x upscale (1.7 MB, fast)",
        },
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_registry(category: str) -> list[dict]:
    """Return all known models for a category with their metadata."""
    entries = MODEL_DB.get(category, {})
    return [
        {"filename": fname, **meta}
        for fname, meta in entries.items()
    ]


def get_download_status(category: str, folder: Path) -> list[dict]:
    """Return registry models with a 'downloaded' flag based on folder contents."""
    entries = MODEL_DB.get(category, {})
    result = []
    for fname, meta in entries.items():
        filepath = folder / fname
        result.append({
            "filename": fname,
            "downloaded": filepath.is_file(),
            "local_size_mb": round(filepath.stat().st_size / (1024 * 1024), 1)
            if filepath.is_file()
            else None,
            **meta,
        })
    return result


async def download_model(
    category: str,
    filename: str,
    target_dir: Path,
    *,
    timeout: float = 300.0,
) -> Path:
    """Download a model from the registry.  Skips if already exists."""
    # Deferred import — app.api.events.download_progress pulls in
    # `event_manager` from app.core.events, which is imported transitively
    # by many modules. Importing at module top here risks circular-import
    # surprises during cold-start ordering.
    from app.api.events.download_progress import (
        DownloadProgress, RateLimiter, emit_download_progress,
    )

    entry = MODEL_DB.get(category, {}).get(filename)
    if not entry:
        msg = f"Unknown model: {category}/{filename}"
        raise ValueError(msg)

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename

    if target_path.is_file():
        logger.info("Model already exists, skipping: %s", target_path)
        return target_path

    url = entry["url"]
    logger.info("Downloading %s → %s", url, target_path)

    rate = RateLimiter()
    base = dict(source="curated", model_id=filename, category=category)

    # Stream download to avoid loading entire model into memory
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0)) or None
                downloaded = 0

                await emit_download_progress(DownloadProgress(
                    **base, status="starting",
                    current_bytes=0, total_bytes=total, percent=0 if total else None,
                ))

                with open(tmp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        f.write(chunk)
                        downloaded += len(chunk)
                        pct = (downloaded * 100 // total) if total else None
                        if rate.allow("downloading", pct):
                            await emit_download_progress(DownloadProgress(
                                **base, status="downloading",
                                current_bytes=downloaded, total_bytes=total,
                                percent=pct,
                            ))
                        if total and downloaded % (5 * 1024 * 1024) < 65536:
                            logger.info(
                                "  %s: %d%% (%d / %d MB)",
                                filename, pct or 0,
                                downloaded // (1024 * 1024),
                                total // (1024 * 1024),
                            )

        # Atomic rename on success
        tmp_path.rename(target_path)
        logger.info("Download complete: %s", target_path)
        await emit_download_progress(DownloadProgress(
            **base, status="complete",
            current_bytes=downloaded, total_bytes=total, percent=100 if total else None,
        ))
        return target_path

    except Exception as exc:
        # Clean up partial download
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        await emit_download_progress(DownloadProgress(
            **base, status="error",
            current_bytes=0, total_bytes=None, percent=None, error=str(exc),
        ))
        raise
