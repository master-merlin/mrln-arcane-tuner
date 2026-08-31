from pathlib import Path

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
    """Within the TTL a warm value is served from memory and never recomputed.

    Rewritten for the LANE-52 contract: the first GET no longer computes the
    value inline (that hold was measured at 479.81 s on the live server), so the
    warm value comes from the sweep worker — the only producer — and both GETs
    then read it without touching the library.
    """
    calls = {"n": 0}

    def fake_agg():
        calls["n"] += 1
        return {"total_bytes": 1, "latent_bytes": 0, "embedding_bytes": 0,
                "cached_datasets": 0, "dataset_root_bytes": 1}
    monkeypatch.setattr(cache_routes, "_aggregate_cache_stats", fake_agg)

    warm = task_manager.create(type="cache_stats_warmup", title="x", user_visible=False)
    cache_routes.run_cache_stats_refresh(warm.id)

    client = TestClient(app)
    r1 = client.get("/api/datasets/cache/stats")
    r2 = client.get("/api/datasets/cache/stats")
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["total_bytes"] == 1
    assert r1.json()["ready"] is True
    assert calls["n"] == 1            # both GETs served from cache (TTL), no recompute


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


class TestColdCacheNeverHoldsTheRequest:
    """A cold `GET /datasets/cache/stats` must return at once.

    Measured on the live server (UAT round 4, LANE-52): the request was held
    open **479.81 s**. The handler ran the 96-root / 85 GB library walk inline
    because there was no value to serve, and the Datasets screen sat black
    behind it. "Every queue/buffer/wait bounded" (ARCHITECTURE D10) has no
    reading under which a multi-minute HTTP hold is acceptable, and the size of
    the wait is a property of the user's library, not of this code.

    The contract: an absent value is reported as an absent value
    (`ready: false`, zeros) and the sweep is queued. `ready` is appended to the
    response, never substituted for an existing key - public surfaces are
    append-only (ARCHITECTURE D2), so an old client reading `dataset_root_bytes`
    still parses the payload.
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

    def test_cold_get_returns_immediately_and_queues_the_sweep(self, monkeypatch):
        calls: list[int] = []
        client, scheduled = self._client(monkeypatch, calls)

        body = client.get("/api/datasets/cache/stats").json()

        assert calls == [], "the library walk ran on the request path"
        assert scheduled == [1], "the sweep was never queued, so no value ever arrives"
        assert body["ready"] is False
        assert body["dataset_root_bytes"] == 0

    def test_a_warm_value_is_reported_ready(self, monkeypatch):
        calls: list[int] = []
        client, _scheduled = self._client(monkeypatch, calls)
        cache_routes._store_cache_stats(_stats(9))

        body = client.get("/api/datasets/cache/stats").json()

        assert body["ready"] is True
        assert body["dataset_root_bytes"] == 9


class TestOneProducerForTheLibraryWalk:
    """RULE-21, one producer. The cold GET and the boot warm-up both walked the
    library, neither took `_begin_refresh`, and on the live server they ran the
    same 85 GB sweep TWICE IN PARALLEL and finished at the same instant."""

    def test_a_cold_get_does_not_walk_beside_a_live_warmup(self, monkeypatch):
        calls: list[int] = []

        def fake_agg():
            calls.append(1)
            return _stats(len(calls))
        monkeypatch.setattr(cache_routes, "_aggregate_cache_stats", fake_agg)
        enqueued: list[tuple] = []
        monkeypatch.setattr(task_manager, "enqueue", lambda *a, **kw: enqueued.append(a))

        cache_routes._schedule_cache_stats_refresh()   # the boot warm-up claims it
        assert len(enqueued) == 1

        client = TestClient(app)
        assert client.get("/api/datasets/cache/stats").status_code == 200

        assert calls == [], "the request walked the library alongside the warm-up"
        assert len(enqueued) == 1, "a second walk was queued while one was live"

    def test_the_startup_warmup_goes_through_the_guard(self):
        """The boot warm-up used to `create` + `enqueue` `run_cache_stats_refresh`
        by hand in main.py, so it never claimed `_cache_stats_refreshing` — and
        then RELEASED it in its finally, cancelling a guard it never held. There
        must be exactly one place in the app that enqueues the sweep."""
        import app
        app_root = Path(app.__file__).resolve().parent
        offenders = []
        for py in app_root.rglob("*.py"):
            if "tests" in py.parts:
                continue
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                if "run_cache_stats_refresh" in line and "enqueue" in line:
                    offenders.append(f"{py.relative_to(app_root)}:{i}")

        # Asserts a POSITIVE fact — exactly one producer, in a named file — so
        # it cannot go vacuously green the way an empty-offender-list scan can
        # (CONVENTIONS "Tests" 11).
        assert len(offenders) == 1, (
            f"the sweep must be enqueued from exactly one place, found: {offenders}"
        )
        assert offenders[0].replace("\\", "/").startswith("api/cache_routes.py:"), offenders
