"""PR9 — two-image (edit-instruction) captioning.

Covers the control+target VLM plumbing end to end at the seams that don't
need real model weights:
- CaptionService.supports_multi_image + extra-image resolution/validation
- chat_vision multi-image payload ordering (control first, target last)
- caption_batch include_control resolves + forwards control paths
- the route-level capability guard (400 for single-image models)
"""

from __future__ import annotations

import base64
import json
import os

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.core.captioning.caption_service import CaptionService
from app.core.llm import openai_compat


def _img(path: str, color="red"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)


# ── Capability + extra-image resolution ──────────────────────────────────


class TestSupportsMultiImage:
    def test_flags(self):
        svc = CaptionService()
        assert svc.supports_multi_image("qwen3-vl") is True
        assert svc.supports_multi_image("qwen3-vl-8B-Instruct") is True
        assert svc.supports_multi_image("api-openai") is True
        assert svc.supports_multi_image("api-custom") is True
        assert svc.supports_multi_image("florence-2") is False
        assert svc.supports_multi_image("youtu-vl") is False

    def test_resolve_extra_images_loads_for_multi_model(self, tmp_path):
        svc = CaptionService()
        ctrl = str(tmp_path / "ctrl.png")
        _img(ctrl, "blue")
        params: dict = {}
        svc._resolve_extra_images("qwen3-vl", [ctrl], params)
        assert len(params["extra_images"]) == 1
        assert params["extra_images"][0].size == (16, 16)

    def test_resolve_extra_images_rejected_for_single_model(self, tmp_path):
        svc = CaptionService()
        ctrl = str(tmp_path / "ctrl.png")
        _img(ctrl)
        with pytest.raises(ValueError, match="does not support multi-image"):
            svc._resolve_extra_images("florence-2", [ctrl], {})

    def test_resolve_extra_images_noop_without_extras(self):
        svc = CaptionService()
        params: dict = {}
        svc._resolve_extra_images("florence-2", None, params)
        assert "extra_images" not in params


# ── chat_vision multi-image payload ──────────────────────────────────────


def test_chat_vision_appends_extra_images_in_order():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "do x"}}]})

    openai_compat.chat_vision(
        base_url="https://api.test/v1", api_key="sk", model="gpt-4o",
        prompt="instruction", image_jpeg=b"TARGET",
        extra_images_jpeg=[b"CONTROL"],
        transport=httpx.MockTransport(handler),
    )
    parts = captured["body"]["messages"][0]["content"]
    # [text, control image, target image] — control first, target last.
    assert parts[0] == {"type": "text", "text": "instruction"}
    assert len(parts) == 3

    def _decode(part):
        return base64.b64decode(part["image_url"]["url"].split(",", 1)[1])

    assert _decode(parts[1]) == b"CONTROL"
    assert _decode(parts[2]) == b"TARGET"


# ── caption_batch include_control ────────────────────────────────────────


class _StubService:
    def __init__(self):
        self.calls: list[dict] = []

    def generate_caption(self, image_path, model_id, params, extra_image_paths=None):
        self.calls.append({"image_path": image_path, "extra": extra_image_paths})
        return f"caption {image_path}"

    @classmethod
    def unload_models(cls):
        pass


class TestCaptionBatchIncludeControl:
    def test_resolves_and_forwards_control_paths(self, monkeypatch):
        from app.core.captioning import caption_batch

        stub = _StubService()
        writes: list = []
        monkeypatch.setattr(caption_batch.CaptionService, "unload_models", lambda: None)
        monkeypatch.setattr(caption_batch, "_get_service", lambda: stub)
        monkeypatch.setattr(caption_batch, "_full_path", lambda ds, rel: f"/full/{rel}")
        monkeypatch.setattr(
            caption_batch, "_control_paths", lambda ds, rel: [f"/ctl/{rel}"],
        )
        monkeypatch.setattr(
            caption_batch, "_write_caption",
            lambda *a, **k: writes.append(a),
        )
        monkeypatch.setattr(caption_batch, "_emit_caption_written", lambda **k: None)

        from app.core.tasks.task_manager import task_manager
        task = task_manager.create(type="caption_batch", title="t", total=1)
        caption_batch.run_caption_batch(
            task.id, dataset_name="ds", image_rel_paths=["a.png"],
            model_id="qwen3-vl", params={}, system_prompt=None, target="original",
            include_control=True,
        )
        assert stub.calls == [{"image_path": "/full/a.png", "extra": ["/ctl/a.png"]}]

    def test_no_control_passes_none(self, monkeypatch):
        from app.core.captioning import caption_batch

        stub = _StubService()
        monkeypatch.setattr(caption_batch.CaptionService, "unload_models", lambda: None)
        monkeypatch.setattr(caption_batch, "_get_service", lambda: stub)
        monkeypatch.setattr(caption_batch, "_full_path", lambda ds, rel: f"/full/{rel}")
        monkeypatch.setattr(caption_batch, "_write_caption", lambda *a, **k: None)
        monkeypatch.setattr(caption_batch, "_emit_caption_written", lambda **k: None)

        from app.core.tasks.task_manager import task_manager
        task = task_manager.create(type="caption_batch", title="t", total=1)
        caption_batch.run_caption_batch(
            task.id, dataset_name="ds", image_rel_paths=["a.png"],
            model_id="qwen3-vl", params={}, system_prompt=None, target="original",
            include_control=False,
        )
        assert stub.calls[0]["extra"] is None


# ── Route capability guard ───────────────────────────────────────────────


class TestBatchRouteCapabilityGuard:
    def test_include_control_on_single_image_model_rejected(self, monkeypatch):
        from app.api import caption_routes

        # Isolate the include_control guard from provider validation.
        monkeypatch.setattr(
            caption_routes.provider_settings, "validate_caption_model",
            lambda model_id, params: None,
        )
        app = FastAPI()
        app.include_router(caption_routes.router, prefix="/captions")
        client = TestClient(app)

        res = client.post("/captions/batch", json={
            "dataset_name": "ds", "image_rel_paths": ["a.png"],
            "model_id": "florence-2", "params": {}, "target": "original",
            "include_control": True,
        })
        assert res.status_code == 400
        assert "multi-image" in res.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
