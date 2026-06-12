"""The unload endpoint must not evict models while a caption batch runs.

The batch worker loads the model once and frees it in its own ``finally``
after the last image; the eager ``DELETE /captions/unload`` the frontend
fires on model/variant/tab changes must be a no-op while a caption batch
is pending or running (it would force a per-image reload — or crash a
generate in flight on the GPU lane).
"""
import asyncio
import time

import pytest

from app.api import caption_routes
from app.core.captioning.caption_service import CaptionService
from app.core.tasks.task import Task, TaskStatus


def _task(status: TaskStatus, type_: str = "caption_batch") -> Task:
    return Task(id=f"t-{status.value}-{type_}", type=type_, title="x",
                status=status, created_at=time.time())


@pytest.fixture
def unload_counter(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(
        CaptionService, "unload_models",
        classmethod(lambda cls: calls.__setitem__("n", calls["n"] + 1)))
    return calls


def test_unload_skipped_while_caption_batch_active(monkeypatch, unload_counter):
    for status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        monkeypatch.setattr(caption_routes.task_manager, "list",
                            lambda s=status: [_task(s)])
        resp = asyncio.run(caption_routes.unload_models_api())
        assert resp["status"] == "success"
        assert "in progress" in resp["message"]
    assert unload_counter["n"] == 0


def test_unload_proceeds_when_no_active_caption_batch(monkeypatch, unload_counter):
    monkeypatch.setattr(
        caption_routes.task_manager, "list",
        lambda: [_task(TaskStatus.COMPLETED),
                 _task(TaskStatus.RUNNING, type_="rescan")])
    resp = asyncio.run(caption_routes.unload_models_api())
    assert resp["status"] == "success"
    assert unload_counter["n"] == 1


def test_batch_keeps_model_loaded_until_final_image(monkeypatch):
    """Same model across a batch: one load, zero unloads between images."""
    calls = {"load": 0, "generate": 0, "unload": 0}

    class StubModel:
        model_id = "stub"

        def load(self, variant=None):
            calls["load"] += 1
            return None, None

        def generate(self, image, params):
            calls["generate"] += 1
            return "cap"

        def unload(self):
            calls["unload"] += 1

    monkeypatch.setattr(CaptionService, "_active_model_key", None)
    monkeypatch.setattr(
        CaptionService, "unload_models",
        classmethod(lambda cls: calls.__setitem__("unload", calls["unload"] + 1)))
    service = CaptionService()
    service.plugins["stub"] = StubModel()
    monkeypatch.setattr(service, "_load_image", lambda p: object())

    for i in range(3):
        assert service.generate_caption(f"img{i}.png", "stub", {}) == "cap"

    assert calls["load"] == 1
    assert calls["generate"] == 3
    assert calls["unload"] == 0
