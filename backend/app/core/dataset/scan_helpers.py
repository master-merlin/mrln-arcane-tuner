"""Pure scan-related helpers — metadata extraction and aggregation.

No side-effects, no event broadcasting, no persistence. These functions
are called by ``DatasetManager.scan_dataset`` stages.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any

import cv2
from PIL import Image

from app.core.dataset.geometry import calculate_target_dims


# ── Per-File Extraction ──────────────────────────────────────────────────


def extract_media_dimensions(
    file_path: str, ext: str
) -> tuple[int, int]:
    """Read width/height from an image or video file.

    Returns:
        (width, height) tuple. (0, 0) on failure.
    """
    try:
        if ext in {".mp4", ".gif", ".webm", ".mkv", ".avi"}:
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                return (w, h)
            return (0, 0)
        else:
            with Image.open(file_path) as img:
                return img.size
    except Exception:
        return (0, 0)


def classify_orientation(width: int, height: int) -> str:
    """Return 'landscape', 'portrait', or 'squared' based on dimensions."""
    ratio = width / height
    if ratio > 1:
        return "landscape"
    elif ratio < 1:
        return "portrait"
    return "squared"


def build_media_entry(
    file_path: str,
    stem: str,
    ext: str,
    dataset_path: str,
    existing_meta: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    """Build the per-file metadata dict for a multimedia entry.

    Checks for masks and masked derivatives. Preserves ``enabled``
    state from existing metadata.

    Returns:
        Metadata dict ready for ``media_metadata[rel_path]``.
    """
    ratio = round(width / height, 5)
    orientation = classify_orientation(width, height)

    # Check for mask
    mask_rel_path = f"masks/{stem}.png"
    mask_full_path = os.path.join(dataset_path, mask_rel_path)
    has_mask = os.path.exists(mask_full_path)

    # Check for masked image and caption (colocated in masked/)
    masked_img_rel = f"masked/{stem}.jpg"
    masked_cap_rel = f"masked/{stem}.txt"
    has_masked_img = os.path.exists(os.path.join(dataset_path, masked_img_rel))
    has_masked_cap = os.path.exists(os.path.join(dataset_path, masked_cap_rel))

    entry: dict[str, Any] = {
        "width": width,
        "height": height,
        "aspect_ratio": ratio,
        "orientation": orientation,
        "size_bytes": os.path.getsize(file_path),
        "has_mask": has_mask,
        "has_masked": has_masked_img,
        "has_masked_caption": has_masked_cap,
        "enabled": existing_meta.get("enabled", True),
    }

    # Add mask dimensions if mask exists
    if has_mask:
        try:
            with Image.open(mask_full_path) as m_img:
                m_w, m_h = m_img.size
            entry["mask_info"] = {
                "width": m_w,
                "height": m_h,
                "size_bytes": os.path.getsize(mask_full_path),
            }
        except Exception:
            pass

    return entry


# ── Aggregation ──────────────────────────────────────────────────────────


def compute_majority_ar(aspect_ratios: list[float]) -> float | None:
    """Return the most common aspect ratio, or None if list is empty."""
    if not aspect_ratios:
        return None
    counts = Counter(aspect_ratios)
    return counts.most_common(1)[0][0]


def compute_harmonization_score(
    media_metadata: dict[str, dict[str, Any]],
    calculate_target_fn: Any = None,
) -> tuple[float, dict[str, dict[str, Any]]]:
    """Compute harmonization score and annotate metadata with crop targets.

    Groups media by orientation, finds per-group majority AR, marks
    ``is_majority_ar``, and calculates per-image ``target_width`` /
    ``target_height``.

    Args:
        media_metadata: Mutable dict — entries are annotated in-place.
        calculate_target_fn: Callable ``(long_side, ar, orientation) -> (w, h)``.
            Defaults to ``geometry.calculate_target_dims``.

    Returns:
        (score, media_metadata) where score is in [0.0, 1.0].
    """
    if calculate_target_fn is None:
        calculate_target_fn = calculate_target_dims

    width_orientation_groups: dict[str, list[dict[str, Any]]] = {
        "landscape": [],
        "portrait": [],
        "squared": [],
    }

    for meta in media_metadata.values():
        ori = meta.get("orientation")
        if ori in width_orientation_groups:
            width_orientation_groups[ori].append(meta)

    total_matches = 0
    total_media = len(media_metadata)

    for ori, items in width_orientation_groups.items():
        if not items:
            continue

        # Local Majority AR
        local_ratios = [m["aspect_ratio"] for m in items if "aspect_ratio" in m]
        if not local_ratios:
            continue

        local_counts = Counter(local_ratios)
        local_majority_ar = local_counts.most_common(1)[0][0]

        # Count matches, flag, and calculate per-image crop targets
        for meta in items:
            is_match = False
            if "aspect_ratio" in meta:
                if abs(meta["aspect_ratio"] - local_majority_ar) < 0.01:
                    is_match = True
                    total_matches += 1
            meta["is_majority_ar"] = is_match

            # Calculate target crop dimensions from local majority AR
            w, h = meta.get("width", 0), meta.get("height", 0)
            if w > 0 and h > 0:
                long_side = max(w, h)
                t_w, t_h = calculate_target_fn(long_side, local_majority_ar, ori)
                # Shrink until targets fit within source
                while t_w > w or t_h > h:
                    long_side -= 32
                    if long_side <= 0:
                        t_w, t_h = w, h
                        break
                    t_w, t_h = calculate_target_fn(long_side, local_majority_ar, ori)
                meta["target_width"] = t_w
                meta["target_height"] = t_h

    score = total_matches / total_media if total_media > 0 else 0.0
    return (score, media_metadata)


def compute_caption_coverage(
    multimedia_stems: set[str],
    caption_stems: set[str],
    multimedia_count: int,
    caption_count: int,
) -> bool:
    """Determine whether all multimedia files have matching captions."""
    if multimedia_count > 0:
        return multimedia_stems.issubset(caption_stems)
    if multimedia_count == 0 and caption_count == 0:
        return False
    if multimedia_count == 0:
        return True
    return False
