"""Shared media-file helpers used by crop, adjustment, and harmonize operations.

Pure I/O helpers — they touch the filesystem but have no event broadcasting
or persistence logic.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

#: Serializes read-modify-write cycles on a dataset's ``overlays.json``.
#: Two overlay operations on the same dataset otherwise interleave read/write
#: and the second one writes back a copy of the recipe map that predates the
#: first — last writer wins, silently.
_overlays_json_lock = threading.Lock()


def write_overlays_json(path: str, data: dict[str, Any]) -> None:
    """Atomically replace an ``overlays.json``.

    The recipe map is the whole overlay history for a dataset, rewritten in full
    on every single-image change. A plain ``open(path, "w")`` truncates it
    before the new bytes land, so an interrupted write leaves a truncated file
    that the next reader fails to parse — losing every recipe, not just the one
    being edited. Same tmp+replace pattern as ``settings_manager``.

    Callers must hold :data:`_overlays_json_lock` across their read AND this
    write; atomicity alone does not make read-modify-write safe.
    """
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except OSError:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def invalidate_overlay_files(dataset_path: str, relative_path: str) -> bool:
    """Delete an image's overlay (rendered PNG + ``overlays.json`` recipe).

    Called after destructive pixel edits (crop, adjustment, upscale): the
    overlay was rendered against the OLD pixels/dimensions, so it is now stale
    and would render misaligned. Returns ``True`` if an overlay actually
    existed (so the caller can emit an ``overlay/deleted`` event and the
    frontend OverlayStore drops the row).

    Args:
        dataset_path: Absolute path to the dataset directory.
        relative_path: Forward-slash-or-OS-sep relative path of the media file
            (the key used in ``overlays.json``).
    """
    rel = relative_path.replace(os.sep, "/")
    stem = os.path.splitext(os.path.basename(rel))[0]
    existed = False

    overlay_path = os.path.join(dataset_path, "overlays", f"{stem}.png")
    if os.path.exists(overlay_path):
        try:
            os.remove(overlay_path)
            existed = True
            logger.warning("overlay_invalidated_by_edit", stem=stem)
        except OSError as e:
            logger.warning("overlay_invalidation_failed", stem=stem, error=str(e))

    # Drop the recipe entry (keyed by the relative image path).
    overlays_json = os.path.join(dataset_path, "overlays.json")
    if os.path.exists(overlays_json):
        try:
            with _overlays_json_lock:
                with open(overlays_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if any(k in data for k in (relative_path, rel)):
                    data.pop(relative_path, None)
                    data.pop(rel, None)
                    existed = True
                    write_overlays_json(overlays_json, data)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("overlays_json_update_failed", stem=stem, error=str(e))

    return existed


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


def refresh_media_metadata_after_change(
    metadata: dict[str, dict[str, Any]],
    lookup_key: str,
    full_path: str,
    new_dims: tuple[int, int] | None = None,
    dataset_path: str | None = None,
) -> None:
    """Refresh lightweight metadata fields after ANY change to a media file.

    Named for "after a change" rather than "after an edit": the helper grew
    out of the crop/adjust edit path, but the same invalidation applies after
    any op that rewrites a file's bytes.

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

    # Control images are NOT invalidated (unlike masks they may be
    # aspect/size-independent, and they're user data we must not destroy).
    # Stamp the pair instead so the health check can flag a target that
    # was pixel-edited after its controls were produced.
    if entry.get("control_info"):
        entry["control_info"]["target_edited_at"] = time.time()

    # Overlay was rendered against the old pixels — drop its metadata too.
    # (The physical overlay file + recipe are removed by
    # ``invalidate_overlay_files``, mirroring the mask invalidation split.)
    entry.pop("has_overlay", None)
    entry.pop("overlay_hash", None)
    entry.pop("overlay_score_stale", None)
    entry.pop("overlay_dimensions", None)

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
