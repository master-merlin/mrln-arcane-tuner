"""Test: single-image /generate route applies structured format + normalization.

App wiring confirmed from backend/app/main.py:
  app.include_router(caption_router, prefix="/api/captions")
  → POST /api/captions/generate

Import pattern mirrors backend/tests/conftest.py (client fixture) and
backend/tests/api/test_models_download_emits.py (pytest.mark.asyncio + direct
module imports; no TestClient/httpx used in that test, but AsyncClient is used
here following the brief's pattern and confirmed against httpx being installed).
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_generate_normalizes_structured_caption(monkeypatch, tmp_path):
    """POST /api/captions/generate with definition_id="ideogram4-fp8" (ideogram4
    family) must parse the raw VLM output and return compact normalized JSON.

    The medium field "Painting." (capital P, trailing dot) must be normalized
    to the canonical token "painting" by Ideogram4Format.parse_and_normalize.
    """
    # Minimal dataset stub — just needs .path and .media_metadata
    ds = type("D", (), {"path": str(tmp_path), "media_metadata": {}})()
    (tmp_path / "a.png").write_bytes(b"\x89PNG\r\n")

    from app.core import dataset_manager as dm_mod

    monkeypatch.setattr(dm_mod.dataset_manager, "get_dataset", lambda name: ds)

    # Stub the registry so get_definition("ideogram4-fp8") returns a definition
    # with family="ideogram4", making get_caption_format_for_definition resolve
    # to Ideogram4Format (structured).
    import types

    fake_defn = types.SimpleNamespace(family="ideogram4")

    class _FakeRegistry:
        def get_definition(self, _id):
            return fake_defn

    fake_reg_mod = types.ModuleType("app.engine.models.registry")
    fake_reg_mod.registry = _FakeRegistry()
    monkeypatch.setitem(
        __import__("sys").modules, "app.engine.models.registry", fake_reg_mod
    )

    # Raw JSON as a VLM would return it — medium has wrong casing + trailing dot
    raw = json.dumps(
        {
            "high_level_description": "x",
            "style_description": {
                "aesthetics": "a",
                "lighting": "l",
                "photo": "p",
                "medium": "Painting.",
                "color_palette": ["#abc"],
            },
            "compositional_deconstruction": {"background": "bg", "elements": []},
        }
    )

    with patch(
        "app.core.captioning.caption_service.CaptionService.generate_caption",
        return_value=raw,
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as c:
            resp = await c.post(
                "/api/captions/generate",
                json={
                    "dataset_name": "ds",
                    "image_rel_path": "a.png",
                    "model_id": "qwen3-vl-8B-Instruct",
                    "params": {},
                    "definition_id": "ideogram4-fp8",
                },
            )

    assert resp.status_code == 200, resp.text
    out = json.loads(resp.json()["caption"])
    assert out["style_description"]["medium"] == "painting"


@pytest.mark.asyncio
async def test_generate_without_definition_id_returns_raw(monkeypatch, tmp_path):
    """Without definition_id the route must return the caption unchanged."""
    ds = type("D", (), {"path": str(tmp_path), "media_metadata": {}})()
    (tmp_path / "b.png").write_bytes(b"\x89PNG\r\n")

    from app.core import dataset_manager as dm_mod

    monkeypatch.setattr(dm_mod.dataset_manager, "get_dataset", lambda name: ds)

    raw = "plain text caption"

    with patch(
        "app.core.captioning.caption_service.CaptionService.generate_caption",
        return_value=raw,
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as c:
            resp = await c.post(
                "/api/captions/generate",
                json={
                    "dataset_name": "ds",
                    "image_rel_path": "b.png",
                    "model_id": "qwen3-vl-8B-Instruct",
                    "params": {},
                },
            )

    assert resp.status_code == 200, resp.text
    assert resp.json()["caption"] == raw
