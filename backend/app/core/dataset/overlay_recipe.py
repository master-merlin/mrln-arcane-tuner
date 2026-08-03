"""Re-render an image's overlay from its stored recipe after a destructive edit.

When a base image's pixels change (crop, adjustment, upscale) the previously
rendered overlay PNG becomes stale, but the *recipe* — the non-destructive
pipeline recorded in ``overlays.json`` — is still the user's intent. Re-applying
it to the new base preserves the edit instead of discarding it.

Kept separate from ``media_helpers`` (pure cheap I/O): re-rendering pulls in the
image-processing pipeline (and, for GPU ops like upscale/denoise, model loads),
so the heavy imports live here and only fire when an overlay recipe exists.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger(__name__)


def _overlays_json_path(dataset_path: str) -> str:
    return os.path.join(dataset_path, "overlays.json")


def read_overlay_recipe(dataset_path: str, relative_path: str) -> list[dict] | None:
    """Return the recipe operations for an image, or None if there is none."""
    path = _overlays_json_path(dataset_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    rel = relative_path.replace(os.sep, "/")
    entry = data.get(relative_path) or data.get(rel)
    if not entry:
        return None
    ops = entry.get("operations") or []
    return ops or None


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def rerender_overlay_from_recipe(
    dataset_path: str, relative_path: str,
) -> tuple[tuple[int, int], str, list[dict]] | None:
    """Re-apply the stored recipe to the CURRENT base image and rewrite the
    overlay PNG.

    Returns ``(dimensions, overlay_hash, operations)`` on success, or ``None``
    if the image has no overlay recipe (nothing to do). Raises on render
    failure so the caller can fall back to dropping the stale overlay.
    """
    ops = read_overlay_recipe(dataset_path, relative_path)
    if not ops:
        return None

    from PIL import Image

    from app.core.image_processing.pipeline import PipelineBlock, execute_pipeline

    rel = relative_path.replace(os.sep, "/")
    stem = os.path.splitext(os.path.basename(rel))[0]
    base_path = os.path.join(dataset_path, relative_path)

    with Image.open(base_path) as img:
        img_rgb = img.convert("RGB")

    blocks = [
        PipelineBlock(
            type=op["type"],
            enabled=op.get("enabled", True),
            params=dict(op.get("params", {})),
        )
        for op in ops
    ]
    result = execute_pipeline(img_rgb, blocks)

    overlays_dir = os.path.join(dataset_path, "overlays")
    os.makedirs(overlays_dir, exist_ok=True)
    overlay_path = os.path.join(overlays_dir, f"{stem}.png")
    result.save(overlay_path)

    # Refresh the recipe's timestamp (operations unchanged — same intent,
    # re-applied to the new pixels).
    json_path = _overlays_json_path(dataset_path)
    try:
        # Lock across the read AND the write: this rewrites the whole recipe
        # map, so interleaving with another overlay op on the same dataset
        # silently drops that op's entry. Write goes through the atomic helper
        # so an interrupted rewrite cannot truncate every recipe.
        from app.core.dataset.media_helpers import (
            _overlays_json_lock,
            write_overlays_json,
        )

        with _overlays_json_lock:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            key = relative_path if relative_path in data else rel
            if key in data:
                data[key]["created_at"] = datetime.now(timezone.utc).isoformat()
                data[key]["overlay_file"] = f"overlays/{stem}.png"
                write_overlays_json(json_path, data)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("overlays_json_timestamp_failed", stem=stem, error=str(e))

    logger.info("overlay_rerendered_from_recipe", stem=stem, dimensions=list(result.size))
    return result.size, _file_sha256(overlay_path), ops
