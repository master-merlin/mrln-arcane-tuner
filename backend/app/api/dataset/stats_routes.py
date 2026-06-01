"""Cross-dataset aggregate stats — KPI tiles, mini-charts, etc.

These endpoints compute one-shot aggregates over *every* loaded dataset's
``media_metadata`` so the frontend doesn't have to fetch each dataset
individually just to draw a histogram or a mean.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.dataset_manager import dataset_manager
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


# ── Pydantic Models ──────────────────────────────────────────────────────


class MpxBucket(BaseModel):
    """One histogram bin."""

    range_mp_min: float                     # bucket lower bound (megapixels)
    range_mp_max: float                     # bucket upper bound
    count: int                              # images in this bucket


class MpxDistribution(BaseModel):
    """Cross-dataset MPx + image-size aggregate."""

    total_images: int                       # total image count
    avg_size_bytes: float                   # mean file size, bytes
    avg_megapixels: float                   # mean megapixels (width * height / 1e6)
    median_megapixels: float                # 50th percentile MP
    buckets: list[MpxBucket]                # ordered, 10 buckets covering 0..max_MP


# Histogram is split into 10 equal-width bins from 0..max_observed_MP, with
# the upper edge clamped at ``_MAX_BUCKET_CAP`` so a stray 100MP scan can't
# crush all the typical (1–8MP) images into a single bucket.
_BUCKET_COUNT = 10
_MAX_BUCKET_CAP = 32.0


# ── Helpers ──────────────────────────────────────────────────────────────


def _compute_mpx_distribution() -> MpxDistribution:
    """Build the MPx distribution aggregate across every loaded dataset."""
    mps: list[float] = []
    sizes: list[int] = []

    for ds in dataset_manager.datasets.values():
        media_meta: dict[str, dict[str, Any]] = ds.media_metadata or {}
        for entry in media_meta.values():
            if not isinstance(entry, dict):
                continue
            width = entry.get("width")
            height = entry.get("height")
            # Defensive: skip entries lacking valid positive dimensions.
            if not width or not height:
                continue
            try:
                w = float(width)
                h = float(height)
            except (TypeError, ValueError):
                continue
            if w <= 0 or h <= 0:
                continue

            mps.append((w * h) / 1_000_000.0)
            size = entry.get("size_bytes")
            try:
                sizes.append(int(size) if size is not None else 0)
            except (TypeError, ValueError):
                sizes.append(0)

    total = len(mps)
    if total == 0:
        return MpxDistribution(
            total_images=0,
            avg_size_bytes=0.0,
            avg_megapixels=0.0,
            median_megapixels=0.0,
            buckets=[],
        )

    avg_size = sum(sizes) / total
    avg_mp = sum(mps) / total

    sorted_mps = sorted(mps)
    mid = total // 2
    if total % 2:
        median_mp = sorted_mps[mid]
    else:
        median_mp = (sorted_mps[mid - 1] + sorted_mps[mid]) / 2

    # Bucket boundaries: 10 equal-width bins from 0 to min(max_observed, cap).
    # If the dataset is all-zero MP (shouldn't happen given the filter above
    # but defensive), fall back to a 1MP span so we still emit 10 buckets.
    max_mp = min(sorted_mps[-1], _MAX_BUCKET_CAP)
    if max_mp <= 0:
        max_mp = 1.0
    bucket_width = max_mp / _BUCKET_COUNT

    buckets: list[MpxBucket] = []
    counts = [0] * _BUCKET_COUNT
    for mp in mps:
        # Clamp values above the cap into the last bucket.
        idx = int(mp / bucket_width) if bucket_width > 0 else 0
        if idx >= _BUCKET_COUNT:
            idx = _BUCKET_COUNT - 1
        counts[idx] += 1

    for i in range(_BUCKET_COUNT):
        buckets.append(MpxBucket(
            range_mp_min=round(i * bucket_width, 6),
            range_mp_max=round((i + 1) * bucket_width, 6),
            count=counts[i],
        ))

    return MpxDistribution(
        total_images=total,
        avg_size_bytes=avg_size,
        avg_megapixels=avg_mp,
        median_megapixels=median_mp,
        buckets=buckets,
    )


# ── Routes ───────────────────────────────────────────────────────────────


@router.get("/datasets/stats/mpx-distribution", response_model=MpxDistribution)
async def get_mpx_distribution() -> MpxDistribution:
    """Return a cross-dataset megapixel histogram + mean file size.

    Aggregates every image in every loaded dataset's ``media_metadata``
    into a 10-bucket equal-width histogram (capped at 32 MP), plus the
    mean file size, mean megapixels, and median megapixels.

    Entries missing ``width``/``height`` are silently skipped.
    """
    logger.info("computing_mpx_distribution")
    return await asyncio.to_thread(_compute_mpx_distribution)
