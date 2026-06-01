"""Tests for cross-dataset stats endpoints (mpx distribution, etc.)."""

from __future__ import annotations

import time
from unittest.mock import patch

from app.core.dataset_manager import Dataset


def _make_dataset(name: str, media_metadata: dict | None = None) -> Dataset:
    """Build a Dataset stub with arbitrary media_metadata for stats tests."""
    return Dataset(
        id=f"id-{name}",
        name=name,
        path=f"/tmp/{name}",
        description="",
        created_at=time.time(),
        file_count=len(media_metadata or {}),
        media_metadata=media_metadata or {},
    )


# ── /datasets/stats/mpx-distribution ──────────────────────────────────────


@patch("app.api.dataset.stats_routes.dataset_manager")
def test_mpx_distribution_empty(mock_manager, client):
    """No datasets → zero aggregates, empty buckets."""
    mock_manager.datasets = {}

    response = client.get("/api/datasets/stats/mpx-distribution")
    assert response.status_code == 200
    body = response.json()
    assert body["total_images"] == 0
    assert body["avg_size_bytes"] == 0.0
    assert body["avg_megapixels"] == 0.0
    assert body["median_megapixels"] == 0.0
    assert body["buckets"] == []


@patch("app.api.dataset.stats_routes.dataset_manager")
def test_mpx_distribution_aggregates_across_datasets(mock_manager, client):
    """Aggregates images across multiple datasets and bins them."""
    # ds_a: two 1MP images (1000x1000 = 1.0 MP), 100KB each
    # ds_b: one 4MP image (2000x2000 = 4.0 MP), 400KB
    ds_a = _make_dataset("ds_a", media_metadata={
        "img1.png": {"width": 1000, "height": 1000, "size_bytes": 100_000},
        "img2.png": {"width": 1000, "height": 1000, "size_bytes": 100_000},
    })
    ds_b = _make_dataset("ds_b", media_metadata={
        "img3.png": {"width": 2000, "height": 2000, "size_bytes": 400_000},
    })
    mock_manager.datasets = {"ds_a": ds_a, "ds_b": ds_b}

    response = client.get("/api/datasets/stats/mpx-distribution")
    assert response.status_code == 200
    body = response.json()

    assert body["total_images"] == 3
    # Mean file size: (100k + 100k + 400k) / 3 = 200_000
    assert body["avg_size_bytes"] == 200_000.0
    # Mean megapixels: (1 + 1 + 4) / 3 = 2.0
    assert body["avg_megapixels"] == 2.0
    # Median of [1, 1, 4] = 1.0
    assert body["median_megapixels"] == 1.0

    # 10 equal-width buckets, max_mp = 4.0 → bucket width = 0.4
    buckets = body["buckets"]
    assert len(buckets) == 10
    assert buckets[0]["range_mp_min"] == 0.0
    # Counts sum to total
    assert sum(b["count"] for b in buckets) == 3
    # Bucket ranges monotonically increase
    for i in range(len(buckets) - 1):
        assert buckets[i]["range_mp_max"] <= buckets[i + 1]["range_mp_max"]
        assert buckets[i]["range_mp_min"] < buckets[i]["range_mp_max"]


@patch("app.api.dataset.stats_routes.dataset_manager")
def test_mpx_distribution_skips_missing_dimensions(mock_manager, client):
    """Entries missing width or height are skipped, not crashed on."""
    ds = _make_dataset("ds_skip", media_metadata={
        "good.png": {"width": 1000, "height": 1000, "size_bytes": 100_000},
        "no_w.png": {"height": 1000, "size_bytes": 50_000},        # no width
        "no_h.png": {"width": 1000, "size_bytes": 50_000},          # no height
        "zero_w.png": {"width": 0, "height": 1000, "size_bytes": 50_000},  # zero width
        "no_size.png": {"width": 500, "height": 500},               # no size_bytes
    })
    mock_manager.datasets = {"ds_skip": ds}

    response = client.get("/api/datasets/stats/mpx-distribution")
    assert response.status_code == 200
    body = response.json()
    # Only good.png and no_size.png have valid w/h; no_size.png contributes 0 bytes.
    assert body["total_images"] == 2
    # avg_size_bytes: (100_000 + 0) / 2 = 50_000
    assert body["avg_size_bytes"] == 50_000.0
    # avg_megapixels: (1.0 + 0.25) / 2 = 0.625
    assert abs(body["avg_megapixels"] - 0.625) < 1e-6


@patch("app.api.dataset.stats_routes.dataset_manager")
def test_mpx_distribution_caps_at_32mp(mock_manager, client):
    """When max observed MP exceeds 32, the upper bound is capped at 32."""
    # A monster 100MP image alongside a normal 1MP image.
    ds = _make_dataset("ds_big", media_metadata={
        "huge.png": {"width": 10_000, "height": 10_000, "size_bytes": 5_000_000},
        "small.png": {"width": 1000, "height": 1000, "size_bytes": 100_000},
    })
    mock_manager.datasets = {"ds_big": ds}

    response = client.get("/api/datasets/stats/mpx-distribution")
    assert response.status_code == 200
    body = response.json()
    buckets = body["buckets"]
    assert len(buckets) == 10
    # Upper bound capped at 32 MP
    assert buckets[-1]["range_mp_max"] == 32.0
    # All images still accounted for (huge one lands in the top bucket)
    assert sum(b["count"] for b in buckets) == 2
