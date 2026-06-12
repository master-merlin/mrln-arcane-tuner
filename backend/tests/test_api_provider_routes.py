"""Route tests for /api/captions/api-providers (status, save, models)."""
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import api_provider_routes
from app.core.llm import provider_settings


class FakeSettingsManager:
    def __init__(self):
        self.modules: dict[str, dict] = {}

    def get_module_settings(self, module):
        return self.modules.get(module, {})

    def update_module_settings(self, module, settings):
        self.modules.setdefault(module, {}).update(settings)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(provider_settings, "_manager",
                        lambda: FakeSettingsManager.instance)
    FakeSettingsManager.instance = FakeSettingsManager()
    app = FastAPI()
    app.include_router(api_provider_routes.router, prefix="/api/captions")
    return TestClient(app)


def test_status_lists_all_providers_unconfigured(client):
    resp = client.get("/api/captions/api-providers")
    assert resp.status_code == 200
    body = resp.json()
    assert {p["provider"] for p in body} == {
        "openai", "anthropic", "gemini", "openrouter", "custom"}
    assert all(p["configured"] is False for p in body)
    assert all(p["key_masked"] == "" for p in body)


def test_put_key_then_status_masked_never_raw(client):
    resp = client.put("/api/captions/api-providers/openai",
                      json={"api_key": "sk-abcdef123456"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["configured"] is True
    assert body["key_masked"] == "sk-…3456"
    assert "sk-abcdef123456" not in resp.text

    listed = client.get("/api/captions/api-providers").json()
    openai = next(p for p in listed if p["provider"] == "openai")
    assert openai["key_masked"] == "sk-…3456"


def test_put_custom_base_url_configures_without_key(client):
    resp = client.put("/api/captions/api-providers/custom",
                      json={"base_url": "http://localhost:11434/v1"})
    assert resp.json()["configured"] is True
    assert resp.json()["base_url"] == "http://localhost:11434/v1"


def test_put_unknown_provider_400(client):
    assert client.put("/api/captions/api-providers/nope",
                      json={"api_key": "k"}).status_code == 400


def test_models_passthrough(client, monkeypatch):
    client.put("/api/captions/api-providers/openai", json={"api_key": "sk-x"})
    monkeypatch.setattr(
        api_provider_routes, "list_models",
        lambda **kw: ["gpt-4o", "gpt-4o-mini"])
    resp = client.get("/api/captions/api-providers/openai/models")
    assert resp.status_code == 200
    assert resp.json()["models"] == ["gpt-4o", "gpt-4o-mini"]


def test_models_unconfigured_400(client):
    resp = client.get("/api/captions/api-providers/gemini/models")
    assert resp.status_code == 400
    assert "API key" in resp.json()["detail"]


def test_models_provider_error_502(client, monkeypatch):
    client.put("/api/captions/api-providers/openai", json={"api_key": "sk-bad"})

    def boom(**kw):
        raise httpx.HTTPStatusError(
            "401", request=httpx.Request("GET", "http://x"),
            response=httpx.Response(401))

    monkeypatch.setattr(api_provider_routes, "list_models", boom)
    resp = client.get("/api/captions/api-providers/openai/models")
    assert resp.status_code == 502


def test_generic_settings_route_refuses_api_captioning_module():
    from app.api import settings_routes

    app = FastAPI()
    app.include_router(settings_routes.router)
    c = TestClient(app)
    assert c.get("/api/settings/api_captioning").status_code == 403
    assert c.put("/api/settings/api_captioning", json={}).status_code == 403
