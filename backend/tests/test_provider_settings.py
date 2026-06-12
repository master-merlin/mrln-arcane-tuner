"""Tests for API-captioning provider settings (keys, masking, validation)."""
import pytest

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


def test_set_and_get_provider(fake_settings):
    provider_settings.set_provider("openai", api_key="sk-abc123xyz9")
    raw = provider_settings.get_provider_raw("openai")
    assert raw["api_key"] == "sk-abc123xyz9"


def test_set_provider_none_leaves_unchanged_empty_clears(fake_settings):
    provider_settings.set_provider("custom", api_key="k1", base_url="http://x/v1")
    provider_settings.set_provider("custom", api_key=None, base_url="http://y/v1")
    raw = provider_settings.get_provider_raw("custom")
    assert raw["api_key"] == "k1"          # None → unchanged
    assert raw["base_url"] == "http://y/v1"
    provider_settings.set_provider("custom", api_key="")
    assert provider_settings.get_provider_raw("custom")["api_key"] == ""  # "" → cleared


def test_set_provider_rejects_unknown_provider(fake_settings):
    with pytest.raises(ValueError, match="Unknown provider"):
        provider_settings.set_provider("nope", api_key="k")


def test_mask_key():
    assert provider_settings.mask_key("") == ""
    assert provider_settings.mask_key("short") == "•••"
    assert provider_settings.mask_key("sk-abcdef123456") == "sk-…3456"


def test_resolve_provider_preset_base_url(fake_settings):
    provider_settings.set_provider("anthropic", api_key="sk-ant-xyz")
    cfg = provider_settings.resolve_provider("anthropic")
    assert cfg.base_url == "https://api.anthropic.com/v1"
    assert cfg.api_key == "sk-ant-xyz"


def test_resolve_provider_requires_key_except_custom(fake_settings):
    with pytest.raises(ValueError, match="API key"):
        provider_settings.resolve_provider("openai")
    # custom without key is fine (Ollama) — but base_url is required
    with pytest.raises(ValueError, match="Base URL"):
        provider_settings.resolve_provider("custom")
    provider_settings.set_provider("custom", base_url="http://localhost:11434/v1")
    cfg = provider_settings.resolve_provider("custom")
    assert cfg.base_url == "http://localhost:11434/v1"
    assert cfg.api_key == ""


def test_validate_caption_model(fake_settings):
    provider_settings.set_provider("openai", api_key="sk-x")
    provider_settings.validate_caption_model("api-openai")  # no raise
    provider_settings.validate_caption_model("florence-2")  # non-api: no-op
    with pytest.raises(ValueError):
        provider_settings.validate_caption_model("api-gemini")  # unconfigured
    with pytest.raises(ValueError, match="Unknown"):
        provider_settings.validate_caption_model("api-bogus")
    # With params: an empty provider-model selection fails up front …
    with pytest.raises(ValueError, match="model"):
        provider_settings.validate_caption_model("api-openai", params={"model": ""})
    # … a real one passes, and params are ignored for local models.
    provider_settings.validate_caption_model("api-openai", params={"model": "gpt-4o"})
    provider_settings.validate_caption_model("florence-2", params={"model": ""})


def test_lane_for_model():
    assert provider_settings.lane_for_model("api-openai") == "background"
    assert provider_settings.lane_for_model("florence-2") == "gpu"
