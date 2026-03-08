"""
Lightweight bucket assignment preview for dataset harmonization.

Reuses the training engine's BucketManager to show users which bucket
each image would be assigned to given a set of target resolutions and
bucketing mode — WITHOUT actually running training.
"""
from typing import Any

from app.engine.components.bucketing import BucketManager
from app.core.logger import get_logger

logger = get_logger(__name__)


def preview_buckets(
    images: list[dict[str, Any]],
    resolutions: list[int],
    bucketing_mode: str = "kohya",
) -> list[dict[str, Any]]:
    """Annotate each image record with its predicted bucket assignment(s).

    Args:
        images: List of dicts, each must have ``width`` and ``height`` keys.
        resolutions: Target base resolutions (e.g. ``[1024]``).
        bucketing_mode: ``"kohya"`` (single best) or ``"multi"`` (all qualifying).

    Returns:
        The same list, with an added ``"buckets"`` key per image containing
        a list of ``{"width": int, "height": int, "target_resolution": int}``.
    """
    if not resolutions:
        return images

    bm = BucketManager(base_resolutions=resolutions)

    for img in images:
        w, h = img.get("width", 0), img.get("height", 0)
        if w <= 0 or h <= 0:
            img["buckets"] = []
            continue

        if bucketing_mode == "multi":
            buckets = bm.get_buckets_for_all_resolutions(w, h)
        else:
            buckets = [bm.get_bucket(w, h)]

        img["buckets"] = buckets

    # Reset the distribution counters so we don't leak state into training
    bm.reset_distribution()

    logger.debug(
        "bucket_preview_complete",
        image_count=len(images),
        resolutions=resolutions,
        mode=bucketing_mode,
    )
    return images
