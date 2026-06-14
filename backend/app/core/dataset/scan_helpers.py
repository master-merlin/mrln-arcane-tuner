"""Pure scan-related helpers — metadata extraction and aggregation.

No side-effects, no event broadcasting, no persistence. These functions
are called by ``DatasetManager.scan_dataset`` stages.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import structlog
from PIL import Image

from app.core.dataset.control_helpers import detect_control_slots
from app.core.dataset.media_types import VIDEO_EXTENSIONS, is_probeable_video

logger = structlog.get_logger(__name__)


# ── Per-File Extraction ──────────────────────────────────────────────────


def extract_media_dimensions(
    file_path: str, ext: str
) -> tuple[int, int]:
    """Read width/height from an image or video file.

    Returns:
        (width, height) tuple. (0, 0) on failure.
    """
    try:
        if ext in VIDEO_EXTENSIONS:
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

    Checks for masks, masked derivatives, and stem-matched control
    images (paired edit datasets). Preserves ``enabled`` state plus the
    logical ``role_order`` / ``target_edited_at`` pair metadata from
    existing metadata — both are user/edit state the disk can't rebuild.

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

    # Check for overlay (disk is source of truth)
    overlay_rel_path = f"overlays/{stem}.png"
    has_overlay = os.path.exists(os.path.join(dataset_path, overlay_rel_path))

    # Check for sibling caption (.txt or .caption) — disk is source of truth.
    # Without this, the per-image `has_caption` flag is never seeded by the
    # scan, and `dataset_manager.save_caption`'s false→true increment fails
    # to fire (or fires when it shouldn't), causing `caption_count` to drift
    # out of sync with the on-disk reality.
    has_caption = (
        os.path.exists(os.path.join(dataset_path, f"{stem}.txt"))
        or os.path.exists(os.path.join(dataset_path, f"{stem}.caption"))
    )

    entry: dict[str, Any] = {
        "width": width,
        "height": height,
        "aspect_ratio": ratio,
        "orientation": orientation,
        "size_bytes": os.path.getsize(file_path),
        "has_mask": has_mask,
        "has_masked": has_masked_img,
        "has_masked_caption": has_masked_cap,
        "has_overlay": has_overlay,
        "has_caption": has_caption,
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

    # Control slots (paired edit datasets): disk is source of truth for
    # the slot files; role_order/target_edited_at carry over from the
    # existing metadata (a rescan must never reset the logical ordering).
    control_slots = detect_control_slots(dataset_path, stem)
    control_info: dict[str, Any] = {"slots": control_slots} if control_slots else {}
    prev_control = existing_meta.get("control_info") or {}
    for preserved in ("role_order", "target_edited_at"):
        if prev_control.get(preserved) is not None:
            control_info[preserved] = prev_control[preserved]

    entry["control_count"] = len(control_slots)
    entry["control_info"] = control_info or None

    # Trainable-video probing (mp4/webm/mkv/avi — NOT animated .gif). Best
    # effort: a bad clip falls back to the dimensions already computed above
    # and must never fail the scan. clip_warnings is intentionally NOT
    # computed here (that's a later phase) — only the plumbing exists.
    if is_probeable_video(ext):
        entry["is_video"] = True
        _merge_video_probe(entry, file_path)

    return entry


def _merge_video_probe(entry: dict[str, Any], file_path: str) -> None:
    """Probe a trainable video and merge its metadata into *entry* in place.

    On any probe failure the existing best-effort metadata is kept and a
    warning is logged — a single unreadable clip never aborts the scan.
    """
    from app.core.video import VideoProbe, probe_video

    try:
        probe: VideoProbe = probe_video(Path(file_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("scan_video_probe_failed", path=file_path, error=str(exc))
        return

    if probe.width > 0 and probe.height > 0:
        entry["width"] = probe.width
        entry["height"] = probe.height
        entry["aspect_ratio"] = round(probe.width / probe.height, 5)
        entry["orientation"] = classify_orientation(probe.width, probe.height)

    entry["fps"] = probe.fps
    entry["duration_s"] = probe.duration_s
    entry["frame_count"] = probe.frame_count
    entry["frame_count_estimated"] = probe.frame_count_estimated
    entry["has_audio"] = probe.has_audio
    entry["video_codec"] = probe.video_codec


# ── Aggregation ──────────────────────────────────────────────────────────

# Standard aspect ratios for bucketing — listed as W/H (landscape form).
# Portrait equivalents are handled automatically (AR < 1 → 1/AR for lookup).
_STANDARD_RATIOS = [
    16 / 9,   # 1.7778
    3 / 2,    # 1.5
    4 / 3,    # 1.3333
    21 / 9,   # 2.3333
    5 / 4,    # 1.25
    7 / 5,    # 1.4
    5 / 3,    # 1.6667
    2 / 1,    # 2.0
    3 / 1,    # 3.0
    32 / 9,   # 3.5556
]


def _snap_ar(ar: float, tolerance: float = 0.03) -> float:
    """Snap a raw aspect ratio to the nearest standard ratio within tolerance.

    This prevents AR fragmentation where, e.g., 1920×1074 (AR=1.787) and
    1920×1080 (AR=1.778) are treated as different ratios even though both
    are effectively 16:9 for training purposes.

    Args:
        ar: Raw width/height ratio.
        tolerance: Maximum relative difference to snap (default 3%).

    Returns:
        Snapped standard ratio float, or ``round(ar, 2)`` for non-standard ARs.
    """
    for standard in _STANDARD_RATIOS:
        if abs(ar - standard) / standard < tolerance:
            return standard
    # Non-standard: round to 2 decimals to still bucket close values
    return round(ar, 2)


def compute_majority_ar(aspect_ratios: list[float]) -> float | None:
    """Return the most common aspect ratio (snapped to standards), or None."""
    if not aspect_ratios:
        return None
    snapped = [_snap_ar(ar) for ar in aspect_ratios]
    counts = Counter(snapped)
    return counts.most_common(1)[0][0]


def is_majority_match(ar: float, majority_ar: float, tolerance: float = 0.03) -> bool:
    """Check whether an aspect ratio matches the majority AR within tolerance.

    Uses relative tolerance (default 3%) to avoid fragmentation from
    32px-aligned dimensions that produce slightly different exact floats.
    """
    if majority_ar <= 0:
        return False
    return abs(ar - majority_ar) / majority_ar < tolerance


def _floor_32(v: float) -> int:
    """Round down to the nearest multiple of 32 (minimum 32)."""
    return max(32, int(v // 32) * 32)


def compute_crop_target(
    w: int, h: int, target_ar: float, orientation: str
) -> tuple[int, int]:
    """Find the largest 32-aligned rectangle within (w, h) matching target AR.

    Tries both width-anchored and height-anchored candidates, picks the
    one that fits within image bounds and best matches the target AR.

    Args:
        w: Image width.
        h: Image height.
        target_ar: Target aspect ratio (width / height).
        orientation: 'landscape', 'portrait', or 'squared'.

    Returns:
        (target_width, target_height) tuple, both multiples of 32.
    """
    if orientation == "portrait":
        # AR < 1 (W/H). Long side is height.
        h1 = _floor_32(h)
        w1 = _floor_32(h1 * target_ar)
        w2 = _floor_32(w)
        h2 = _floor_32(w2 / target_ar)
    else:
        # AR >= 1 (W/H). Long side is width.
        w1 = _floor_32(w)
        h1 = _floor_32(w1 / target_ar)
        h2 = _floor_32(h)
        w2 = _floor_32(h2 * target_ar)

    # Pick the candidate that fits within image bounds
    # and produces the AR closest to target
    candidates = []
    if w1 <= w and h1 <= h and h1 > 0:
        ar1 = w1 / h1
        candidates.append((w1, h1, abs(ar1 - target_ar), w1 * h1))
    if w2 <= w and h2 <= h and h2 > 0:
        ar2 = w2 / h2
        candidates.append((w2, h2, abs(ar2 - target_ar), w2 * h2))

    if candidates:
        # Prefer closest AR match; break ties by largest area
        candidates.sort(key=lambda c: (c[2], -c[3]))
        return candidates[0][0], candidates[0][1]
    return w, h


def compute_harmonization_score(
    media_metadata: dict[str, dict[str, Any]],
    calculate_target_fn: Any = None,
) -> tuple[float, dict[str, dict[str, Any]]]:
    """Compute harmonization score and annotate metadata with crop targets.

    Groups media by orientation, finds per-group majority AR (snapped to
    standard ratios), marks ``is_majority_ar``, and calculates per-image
    ``target_width`` / ``target_height``.

    Args:
        media_metadata: Mutable dict — entries are annotated in-place.
        calculate_target_fn: Unused, kept for API compatibility.

    Returns:
        (score, media_metadata) where score is in [0.0, 1.0].
    """
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

        # Local Majority AR — snap to standard ratios before counting
        local_ratios = [_snap_ar(m["aspect_ratio"]) for m in items if "aspect_ratio" in m]
        if not local_ratios:
            continue

        local_majority_ar = Counter(local_ratios).most_common(1)[0][0]

        for meta in items:
            if "aspect_ratio" in meta and is_majority_match(meta["aspect_ratio"], local_majority_ar):
                meta["is_majority_ar"] = True
                total_matches += 1
            else:
                meta["is_majority_ar"] = False

            w, h = meta.get("width", 0), meta.get("height", 0)
            if w > 0 and h > 0:
                t_w, t_h = compute_crop_target(w, h, local_majority_ar, ori)
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
