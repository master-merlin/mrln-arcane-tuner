"""Shared media-file helpers used by crop, adjustment, and harmonize operations.

Pure I/O helpers — they touch the filesystem but have no event broadcasting
or persistence logic.
"""

from __future__ import annotations

import os
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def invalidate_mask_files(dataset_path: str, stem: str, reason: str = "edit") -> None:
    """Delete mask, masked image, and masked caption if they exist.

    Called after crop or adjustment operations that change pixel content,
    making the existing mask invalid.

    Args:
        dataset_path: Absolute path to the dataset directory.
        stem: File stem (without extension) of the media file.
        reason: Log context — 'crop' or 'adjustment'.
    """
    masks_dir = os.path.join(dataset_path, "masks")
    masked_dir = os.path.join(dataset_path, "masked")

    files_to_remove = [
        os.path.join(masks_dir, f"{stem}.png"),
        os.path.join(masked_dir, f"{stem}.jpg"),
        os.path.join(masked_dir, f"{stem}.txt"),
    ]

    for p in files_to_remove:
        if os.path.exists(p):
            try:
                os.remove(p)
                logger.warning(
                    f"mask_invalidated_by_{reason}",
                    file=os.path.basename(p),
                    stem=stem,
                )
            except OSError as e:
                logger.warning(
                    "mask_invalidation_failed",
                    file=os.path.basename(p),
                    stem=stem,
                    error=str(e),
                )


def update_metadata_after_edit(
    metadata: dict[str, dict[str, Any]],
    lookup_key: str,
    full_path: str,
    new_dims: tuple[int, int] | None = None,
    dataset_path: str | None = None,
) -> None:
    """Update lightweight metadata fields after a destructive edit.

    Clears the solid hash (content changed), updates size_bytes,
    nullifies mask references (physical files already deleted
    by ``invalidate_mask_files``), and — when ``dataset_path`` is
    supplied — invalidates and regenerates the source's thumbnail
    so the next GET serves a fresh image.

    Args:
        metadata: The dataset's ``media_metadata`` dict.
        lookup_key: Forward-slash key for the media file.
        full_path: Absolute path to the edited file.
        new_dims: Optional ``(width, height)`` if dimensions changed (crop).
            If None, dimensions are left unchanged (adjustment).
        dataset_path: When provided, the source's thumbnail is invalidated
            and regenerated. Callers that mutate source pixels should
            pass this; metadata-only edits may omit it.
    """
    if lookup_key not in metadata:
        return

    entry = metadata[lookup_key]

    # Always update after any edit
    entry["size_bytes"] = os.path.getsize(full_path)
    entry.pop("solid_hash", None)
    entry.pop("quality_score", None)  # re-score on next scan
    entry["has_mask"] = False
    entry["has_masked"] = False
    entry["has_masked_caption"] = False
    entry.pop("mask_info", None)

    # Update dimensions if they changed (crop)
    if new_dims is not None:
        w, h = new_dims
        entry["width"] = w
        entry["height"] = h
        entry["aspect_ratio"] = round(w / h, 5)
        entry["orientation"] = (
            "landscape" if w > h else
            "portrait" if w < h else
            "squared"
        )
        entry["target_width"] = w
        entry["target_height"] = h
        entry["is_majority_ar"] = True

    # Refresh the thumbnail so subsequent GETs serve the new pixels.
    if dataset_path is not None:
        # Local import to avoid a circular import path through scan helpers.
        from app.core.dataset import thumbnails

        thumbnails.invalidate_thumbnail(dataset_path, lookup_key)
        thumbnails.ensure_thumbnail(dataset_path, lookup_key)
