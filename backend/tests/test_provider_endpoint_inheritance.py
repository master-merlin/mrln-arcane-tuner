"""LANE-46: one resolver for a captioning provider's effective endpoint.

The defect (UAT-4.3): the Server screen wrote its LLM endpoint to ``llm_refine``
and the details view read ``api_captioning.providers.custom``, so a user who had
configured their Ollama server once was asked for it again with an empty field —
and the "configured" badge was computed from a third read of the same raw store,
which is how a badge and the request behind it are allowed to disagree.

These tests pin the resolver AND the agreement between the two readers (RULE-21).
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import api_provider_routes
from app.core.llm import provider_settings


class FakeSettingsManager:
    """In-memory stand-in for SettingsManager (module-dict semantics)."""

    def __init__(self):
        self.modules: dict[str, dict] = {}

    def get_module_settings(self, module: str) -> dict:
        return self.modules.get(module, {})

    def update_module_settings(self, module: str, settings: dict) -> None:
        self.modules.setdefault(module, {}).update(settings)


@pytest.fixture
def fake_settings(monkeypatch):
    fake = FakeSettingsManager()
    monkeypatch.setattr(provider_settings, "_manager", lambda: fake)
    return fake


@pytest.fixture
def client(fake_settings):
    app = FastAPI()
    app.include_router(api_provider_routes.router, prefix="/api/captions")
    return TestClient(app)


def _set_server_llm(fake, **values):
    """Write the Server screen's LLM endpoint module."""
    fake.update_module_settings(provider_settings.SERVER_SETTINGS_MODULE, values)


# --- the resolver ------------------------------------------------------------

def test_effective_base_url_source_precedence(fake_settings):
    # Nothing anywhere → "none" is a real string, never "" and never None.
    assert provider_settings.effective_base_url("custom") == \
        provider_settings.EffectiveBaseUrl("", "none")

    # Server screen only → inherited.
    _set_server_llm(fake_settings, base_url="http://localhost:11434",
                    model="qwen2.5:7b-instruct", provider="ollama")
    eff = provider_settings.effective_base_url("custom")
    assert (eff.base_url, eff.source) == ("http://localhost:11434", "server_settings")

    # The provider's own store wins over the inherited value.
    provider_settings.set_provider("custom", base_url="http://box:8000/v1")
    eff = provider_settings.effective_base_url("custom")
    assert (eff.base_url, eff.source) == ("http://box:8000/v1", "provider")

    # Cleared again → back to inherited; no stale "provider" latch.
    provider_settings.set_provider("custom", base_url="")
    assert provider_settings.effective_base_url("custom").source == "server_settings"


def test_hosted_providers_never_inherit_the_local_server_endpoint(fake_settings):
    _set_server_llm(fake_settings, base_url="http://localhost:11434")
    for provider in ("openai", "anthropic", "gemini", "openrouter"):
        eff = provider_settings.effective_base_url(provider)
        # Inheriting a LOCAL endpoint for a hosted provider would silently
        # redirect that provider's traffic — the inheritance is custom-only.
        assert eff.source == "builtin"
        assert eff.base_url == provider_settings.PROVIDER_BASE_URLS[provider]
    # …but an explicit per-provider override still wins over the preset.
    provider_settings.set_provider("openai", base_url="http://proxy.local/v1")
    eff = provider_settings.effective_base_url("openai")
    assert (eff.base_url, eff.source) == ("http://proxy.local/v1", "provider")


def test_effective_base_url_rejects_unknown_provider(fake_settings):
    with pytest.raises(ValueError, match="Unknown provider"):
        provider_settings.effective_base_url("nope")


def test_resolve_provider_inherits_the_server_base_url(fake_settings):
    """The endpoint configured once on the Server screen is usable for captioning."""
    _set_server_llm(fake_settings, base_url="http://localhost:11434")
    cfg = provider_settings.resolve_provider("custom")
    assert cfg.base_url == "http://localhost:11434"
    provider_settings.validate_caption_model("api-custom")  # no raise


def test_model_is_never_inherited_from_server_settings(fake_settings):
    """The Server model is a TEXT model for refining; captioning sends an IMAGE.

    Inheriting it would prefill a default that errors on first use, so the
    inheritance stops at the base URL (user's call, UAT-4.3). This pin exists
    because the asymmetry reads like an oversight to the next person.
    """
    _set_server_llm(fake_settings, base_url="http://localhost:11434",
                    model="qwen2.5:7b-instruct")
    cfg = provider_settings.resolve_provider("custom")
    assert not hasattr(cfg, "model")
    assert "qwen2.5" not in repr(cfg)
    # An empty provider-model selection still fails at enqueue time: the user
    # must pick a vision model themselves.
    with pytest.raises(ValueError, match="model"):
        provider_settings.validate_caption_model("api-custom", params={"model": ""})


def test_server_settings_model_is_not_offered_as_the_caption_model(client,
                                                                  fake_settings):
    """…and it does not leak through the status route either."""
    _set_server_llm(fake_settings, base_url="http://localhost:11434",
                    model="qwen2.5:7b-instruct")
    body = client.get("/api/captions/api-providers").json()
    assert "qwen2.5" not in str(body)


# --- the badge and the request must agree ------------------------------------

@pytest.mark.parametrize("provider", provider_settings.PROVIDERS)
@pytest.mark.parametrize("own", ["", "http://own.local/v1"])
@pytest.mark.parametrize("server", ["", "http://localhost:11434"])
def test_status_base_url_equals_what_the_request_will_use(
        client, fake_settings, provider, own, server):
    """Every combination of (provider store set/unset) x (llm_refine set/unset).

    This is the defect in its general form: the status route reporting one
    endpoint while ``resolve_provider`` hands the HTTP client another. Assert on
    the response body, not on the resolver both would share by construction.
    """
    if own:
        provider_settings.set_provider(provider, base_url=own)
    if server:
        _set_server_llm(fake_settings, base_url=server)
    if provider != "custom":
        # A key, so the only thing that can make resolve_provider raise here is
        # the endpoint — the question this test is about.
        provider_settings.set_provider(provider, api_key="sk-abcdef123456")

    status = next(p for p in client.get("/api/captions/api-providers").json()
                  if p["provider"] == provider)
    try:
        resolved = provider_settings.resolve_provider(provider).base_url
    except ValueError:
        # No endpoint at all → the badge may not claim one, and for custom
        # (where the endpoint IS the configuration) it may not claim configured.
        resolved = ""
        if provider == "custom":
            assert status["configured"] is False
    assert status["base_url"] == resolved, (
        f"{provider}: badge says {status['base_url']!r}, "
        f"request would use {resolved!r}")
    assert status["base_url_source"] in {
        "provider", "server_settings", "builtin", "none"}
    assert (status["base_url_source"] == "none") == (status["base_url"] == "")


def test_custom_is_configured_when_the_endpoint_is_inherited(client, fake_settings):
    """An inherited endpoint is genuinely usable, so the badge must say so."""
    before = next(p for p in client.get("/api/captions/api-providers").json()
                  if p["provider"] == "custom")
    assert (before["configured"], before["base_url_source"]) == (False, "none")

    _set_server_llm(fake_settings, base_url="http://localhost:11434")
    after = next(p for p in client.get("/api/captions/api-providers").json()
                 if p["provider"] == "custom")
    assert after["configured"] is True
    assert after["base_url"] == "http://localhost:11434"
    assert after["base_url_source"] == "server_settings"


def test_saving_a_base_url_here_overrides_the_inherited_one(client, fake_settings):
    """The user's Save wins on the next load — and says so via the source."""
    _set_server_llm(fake_settings, base_url="http://localhost:11434")
    resp = client.put("/api/captions/api-providers/custom",
                      json={"base_url": "http://box:8000/v1"})
    assert resp.status_code == 200
    assert resp.json()["base_url"] == "http://box:8000/v1"
    assert resp.json()["base_url_source"] == "provider"
    listed = next(p for p in client.get("/api/captions/api-providers").json()
                  if p["provider"] == "custom")
    assert listed["base_url_source"] == "provider"
    assert provider_settings.resolve_provider("custom").base_url == \
        "http://box:8000/v1"
