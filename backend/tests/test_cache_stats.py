import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api import cache_routes
from app.core.tasks.task_manager import task_manager


@pytest.fixture(autouse=True)
def reset_cache():
    cache_routes._cache_stats_value = None
    cache_routes._cache_stats_at = 0.0
    cache_routes._cache_stats_refreshing = False
    task_manager.set_loop(None)
    yield
    cache_routes._cache_stats_refreshing = False


def _stats(total: int = 1) -> dict:
    return {"total_bytes": total, "latent_bytes": 0, "embedding_bytes": 0,
            "cached_datasets": 0, "dataset_root_bytes": total}


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


class TestStaleWhileRevalidate:
    """A stale value must never make the caller wait for the sweep.

    `_aggregate_cache_stats` walks every dataset root twice, so its cost scales
    with the whole library and is disk-latency bound. It used to run INLINE on
    the first request after the TTL expired, so any visit to the Datasets
    screen following an idle gap longer than the TTL paid the full walk. The
    startup warm-up disguised that as a once-per-boot cost.
    """

    def _client(self, monkeypatch, calls):
        def fake_agg():
            calls.append(1)
            return _stats(len(calls))
        monkeypatch.setattr(cache_routes, "_aggregate_cache_stats", fake_agg)

        scheduled = []
        monkeypatch.setattr(
            cache_routes, "_schedule_cache_stats_refresh", lambda: scheduled.append(1),
        )
        return TestClient(app), scheduled

    def test_cold_cache_computes_inline(self, monkeypatch):
        """Nothing to serve yet — blocking is the only correct answer."""
        calls: list[int] = []
        client, scheduled = self._client(monkeypatch, calls)

        assert client.get("/api/datasets/cache/stats").status_code == 200
        assert len(calls) == 1
        assert scheduled == []

    def test_stale_value_is_served_without_recomputing(self, monkeypatch):
        calls: list[int] = []
        client, scheduled = self._client(monkeypatch, calls)

        cache_routes._store_cache_stats(_stats(42))
        # Age it past the TTL.
        cache_routes._cache_stats_at -= cache_routes._CACHE_STATS_TTL_S + 1

        body = client.get("/api/datasets/cache/stats").json()

        assert body["total_bytes"] == 42, "the caller waited for a fresh sweep"
        assert calls == [], "the sweep ran inline on the request path"
        assert scheduled == [1], "no background refresh was queued"

    def test_fresh_value_schedules_nothing(self, monkeypatch):
        calls: list[int] = []
        client, scheduled = self._client(monkeypatch, calls)

        cache_routes._store_cache_stats(_stats(7))
        body = client.get("/api/datasets/cache/stats").json()

        assert body["total_bytes"] == 7
        assert calls == []
        assert scheduled == []


class TestRefreshIsSingleFlight:
    """Concurrent stale reads must not fan out into N library walks."""

    def test_second_schedule_is_a_no_op_while_one_is_live(self, monkeypatch):
        enqueued = []
        monkeypatch.setattr(task_manager, "enqueue",
                            lambda *a, **kw: enqueued.append(a))

        cache_routes._schedule_cache_stats_refresh()
        cache_routes._schedule_cache_stats_refresh()
        cache_routes._schedule_cache_stats_refresh()

        assert len(enqueued) == 1

    def test_the_worker_releases_the_guard(self, monkeypatch):
        monkeypatch.setattr(cache_routes, "_aggregate_cache_stats", lambda: _stats(3))
        enqueued = []
        monkeypatch.setattr(task_manager, "enqueue",
                            lambda *a, **kw: enqueued.append(a))

        cache_routes._schedule_cache_stats_refresh()
        assert len(enqueued) == 1

        # Run the worker the way the background lane would.
        t = task_manager.create(type="cache_stats_warmup", title="x", user_visible=False)
        cache_routes.run_cache_stats_refresh(t.id)

        cache_routes._schedule_cache_stats_refresh()
        assert len(enqueued) == 2, "the guard was never released; refreshes stop forever"

    def test_the_guard_is_released_even_when_the_sweep_raises(self, monkeypatch):
        def boom():
            raise OSError("disk gone")
        monkeypatch.setattr(cache_routes, "_aggregate_cache_stats", boom)
        enqueued = []
        monkeypatch.setattr(task_manager, "enqueue",
                            lambda *a, **kw: enqueued.append(a))

        cache_routes._schedule_cache_stats_refresh()
        t = task_manager.create(type="cache_stats_warmup", title="x", user_visible=False)
        cache_routes.run_cache_stats_refresh(t.id)
        assert task_manager.get(t.id).status.value == "failed"

        cache_routes._schedule_cache_stats_refresh()
        assert len(enqueued) == 2, "a failed sweep wedged the guard shut"
