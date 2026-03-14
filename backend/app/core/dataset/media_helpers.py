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
            logger.warning(
                f"mask_invalidated_by_{reason}",
                file=os.path.basename(p),
                stem=stem,
            )
            os.remove(p)


def update_metadata_after_edit(
    metadata: dict[str, dict[str, Any]],
    lookup_key: str,
    full_path: str,
    new_dims: tuple[int, int] | None = None,
) -> None:
    """Update lightweight metadata fields after a destructive edit.

    Clears the solid hash (content changed), updates size_bytes,
    and nullifies mask references (physical files already deleted
    by ``invalidate_mask_files``).

    Args:
        metadata: The dataset's ``media_metadata`` dict.
        lookup_key: Forward-slash key for the media file.
        full_path: Absolute path to the edited file.
        new_dims: Optional ``(width, height)`` if dimensions changed (crop).
            If None, dimensions are left unchanged (adjustment).
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
