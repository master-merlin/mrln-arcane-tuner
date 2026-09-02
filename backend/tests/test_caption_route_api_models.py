"""Wiring tests: caption routes validate api-* configs and pick the lane."""
import asyncio

import pytest
from fastapi import HTTPException

from app.api import caption_routes
from app.core.llm import provider_settings


def test_batch_rejects_unconfigured_api_provider(monkeypatch):
    monkeypatch.setattr(
        provider_settings, "validate_caption_model",
        lambda mid, params=None: (_ for _ in ()).throw(
            ValueError("No API key configured")))
    req = caption_routes.BatchCaptionRequest(
        dataset_name="ds", image_rel_paths=["a.png"],
        model_id="api-openai", params={})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(caption_routes.batch_caption_api(req))
    assert exc.value.status_code == 400
    assert "API key" in exc.value.detail


def test_batch_enqueues_api_model_on_background_lane(monkeypatch):
    monkeypatch.setattr(provider_settings, "validate_caption_model",
                        lambda mid, params=None: None)
    # A lane-wiring test: configuration AND readiness (LANE-65) are stubbed as
    # "fine" at their seams; the readiness seam itself is exercised unstubbed,
    # against real sockets, in test_caption_generate_boundary_guard.py.
    from app.core.llm.refine_guard import RefineReadiness

    async def _ready(provider, model=None):
        return RefineReadiness(base_url="https://api.openai.com/v1", available=True)

    monkeypatch.setattr(caption_routes, "caption_provider_readiness", _ready)
    enq = {}

    class FakeTask:
        id = "t1"

    monkeypatch.setattr(caption_routes.task_manager, "create",
                        lambda **kw: FakeTask())
    monkeypatch.setattr(
        caption_routes.task_manager, "enqueue",
        lambda task_id, fn, *, lane: enq.update(lane=lane))

    req = caption_routes.BatchCaptionRequest(
        dataset_name="ds", image_rel_paths=["a.png"],
        model_id="api-openai", params={})
    resp = asyncio.run(caption_routes.batch_caption_api(req))
    assert resp == {"task_id": "t1"}
    assert enq["lane"] == "background"

    req2 = caption_routes.BatchCaptionRequest(
        dataset_name="ds", image_rel_paths=["a.png"],
        model_id="florence-2", params={})
    asyncio.run(caption_routes.batch_caption_api(req2))
    assert enq["lane"] == "gpu"
