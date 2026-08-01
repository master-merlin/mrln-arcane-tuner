"""The dataset LIST response must not carry the per-file metadata map.

`media_metadata` is keyed by relative path and holds a dict per file. On a
single dataset that is bounded; on the list endpoint it dominates the response
and scales with the TOTAL file count across the library, while every other
field is O(1) per row — and it is read by nothing. Per-file data reaches the
client through `/datasets/{name}/pairs`; the cross-dataset MPx histogram is
computed server-side by `/datasets/stats/mpx-distribution` precisely so the
client never needs the raw map.

The exclusion is serialization-only, which is what makes it safe: `Dataset`
still holds the map, so the computed fields derived from it keep working. Both
are pinned below, because dropping them is the way this optimisation would
silently break the library grid.

Regression guard on the mechanism itself: FastAPI's `response_model_exclude`
takes `{"__all__": {...}}` for a list response model. The bare `{"field"}` form
that works for a single-object route silently does NOTHING here and ships the
field anyway — so this suite asserts on the real HTTP response, not on the
route's declared kwargs.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.dataset_manager import Dataset


def _dataset(name: str, files: int = 3) -> Dataset:
    return Dataset(
        id=f"id-{name}",
        name=name,
        path=f"/datasets/{name}",
        created_at=0.0,
        media_metadata={
            f"{name}-{i}.png": {
                "enabled": i != 0,          # exactly one disabled per dataset
                "quality_score": 0.5 + i / 10,
                "width": 1024,
                "height": 1024,
            }
            for i in range(files)
        },
    )


@pytest.fixture
def client(monkeypatch):
    """Mount the real router against a stubbed manager — no DB, no disk."""
    from app.api.dataset import crud_routes

    monkeypatch.setattr(
        crud_routes.dataset_manager,
        "list_datasets",
        lambda: [_dataset("alpha"), _dataset("beta")],
    )
    app = FastAPI()
    app.include_router(crud_routes.router, prefix="/api")
    return TestClient(app)


class TestListOmitsPerFileMetadata:
    def test_media_metadata_is_absent_from_every_row(self, client):
        rows = client.get("/api/datasets").json()
        assert rows, "fixture produced no datasets"
        for row in rows:
            assert "media_metadata" not in row, (
                "the list response is carrying the per-file map again — this is "
                "the 4 MB regression; check that response_model_exclude still "
                'uses the {"__all__": {...}} form.'
            )

    def test_the_rows_are_otherwise_intact(self, client):
        rows = client.get("/api/datasets").json()
        assert [r["name"] for r in rows] == ["alpha", "beta"]
        for row in rows:
            for key in ("id", "path", "created_at", "version", "kind"):
                assert key in row


class TestComputedFieldsSurviveTheExclusion:
    """Both are derived from `media_metadata`. Serialization-time exclusion
    keeps the model's own copy, so they must still be computed and returned —
    if they vanish or go null, the grid loses its excluded/quality signals."""

    def test_excluded_count_still_computed(self, client):
        rows = client.get("/api/datasets").json()
        for row in rows:
            assert row["excluded_count"] == 1

    def test_median_quality_score_still_computed(self, client):
        rows = client.get("/api/datasets").json()
        for row in rows:
            assert row["median_quality_score"] == pytest.approx(0.6)


class TestSingleDatasetStillCarriesIt:
    """The exclusion is scoped to the LIST route on purpose: one dataset is a
    bounded payload, and narrowing it would be a separate contract change."""

    def test_single_get_keeps_media_metadata(self, monkeypatch):
        from app.api.dataset import crud_routes

        row = _dataset("alpha")
        monkeypatch.setattr(crud_routes.dataset_manager, "get_dataset", lambda _n: row)
        app = FastAPI()
        app.include_router(crud_routes.router, prefix="/api")
        body = TestClient(app).get("/api/datasets/alpha").json()

        assert "media_metadata" in body
        assert len(body["media_metadata"]) == 3


class TestPayloadShapeIsActuallySmaller:
    def test_dropping_the_map_dominates_the_response_size(self, monkeypatch):
        """Guards the premise, not just the mechanism: if `media_metadata` ever
        stops being the bulk of a row, this optimisation is no longer the right
        one and the comment on the route is stale.

        Needs a fixture with enough files per dataset for the per-file map to
        dominate, which is the regime this exists for. The three-file fixture
        used above puts it well under half the payload and would assert nothing.
        """
        import json

        from app.api.dataset import crud_routes

        rows = [_dataset(f"ds{i}", files=40) for i in range(10)]
        monkeypatch.setattr(crud_routes.dataset_manager, "list_datasets", lambda: rows)
        app = FastAPI()
        app.include_router(crud_routes.router, prefix="/api")

        served = TestClient(app).get("/api/datasets").content
        full = json.dumps([json.loads(d.model_dump_json()) for d in rows]).encode()
        share = 1 - len(served) / len(full)

        assert share > 0.85, (
            f"served {len(served)}B vs full {len(full)}B — only {share:.0%} of "
            "the payload was the per-file map; it is no longer the dominant term"
        )
