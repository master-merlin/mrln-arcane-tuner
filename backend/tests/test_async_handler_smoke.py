"""Smoke test: async handlers do not block the event loop."""
import asyncio
import time
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_serialize():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        async def _hit():
            r = await ac.get("/api/jobs")
            return r.status_code

        start = time.perf_counter()
        results = await asyncio.gather(*[_hit() for _ in range(4)])
        elapsed = time.perf_counter() - start

    assert all(s == 200 for s in results)
    assert elapsed < 2.0, f"event loop appears blocked: 4 reqs took {elapsed:.2f}s"
