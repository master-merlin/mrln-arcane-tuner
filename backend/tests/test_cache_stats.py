import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api import cache_routes
from app.core.tasks.task_manager import task_manager


@pytest.fixture(autouse=True)
def reset_cache():
    cache_routes._cache_stats_value = None
    cache_routes._cache_stats_at = 0.0
    task_manager.set_loop(None)
    yield


def test_get_caches_and_serves_without_recompute(monkeypatch):
    calls = {"n": 0}

    def fake_agg():
        calls["n"] += 1
        return {"total_bytes": 1, "latent_bytes": 0, "embedding_bytes": 0,
                "cached_datasets": 0, "dataset_root_bytes": 1}
    monkeypatch.setattr(cache_routes, "_aggregate_cache_stats", fake_agg)

    client = TestClient(app)
    r1 = client.get("/api/datasets/cache/stats")
    r2 = client.get("/api/datasets/cache/stats")
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["total_bytes"] == 1
    assert calls["n"] == 1            # second GET served from cache (TTL), no recompute


def test_warmup_worker_populates_cache(monkeypatch):
    monkeypatch.setattr(cache_routes, "_aggregate_cache_stats",
                        lambda: {"total_bytes": 7, "latent_bytes": 0, "embedding_bytes": 0,
                                 "cached_datasets": 0, "dataset_root_bytes": 7})
    t = task_manager.create(type="cache_stats_warmup", title="x", user_visible=False)
    cache_routes.run_cache_stats_refresh(t.id)
    assert task_manager.get(t.id).status.value == "completed"
    assert cache_routes._get_fresh_cache_stats()["total_bytes"] == 7
