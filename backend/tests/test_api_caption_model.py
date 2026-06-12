"""Tests for the external-API caption model plugin."""
import pytest
from PIL import Image

from app.core.captioning.models import api_model
from app.core.captioning.models.api_model import ApiCaptionModel, _encode_jpeg
from app.core.llm.provider_settings import ProviderConfig


@pytest.fixture
def fake_provider(monkeypatch):
    cfg = ProviderConfig(provider="openai", base_url="https://api.test/v1",
                         api_key="sk-x")
    monkeypatch.setattr(api_model, "resolve_provider", lambda p: cfg)
    return cfg


def test_model_id_and_noop_lifecycle():
    m = ApiCaptionModel(None, "openai")
    assert m.model_id == "api-openai"
    assert m.load() == (None, None)
    m.unload()  # must not raise


def test_generate_calls_chat_vision_with_resolved_config(monkeypatch, fake_provider):
    captured = {}

    def fake_chat_vision(**kwargs):
        captured.update(kwargs)
        return "a caption"

    monkeypatch.setattr(api_model, "chat_vision", fake_chat_vision)
    m = ApiCaptionModel(None, "openai")
    img = Image.new("RGB", (64, 64), "red")
    result = m.generate(img, {
        "model": "gpt-4o", "temperature": 0.5, "top_p": 0.9,
        "max_tokens": 256, "max_long_side": 512,
        "system_prompt": "Describe for a LoRA dataset.",
    })

    assert result == "a caption"
    assert captured["base_url"] == "https://api.test/v1"
    assert captured["api_key"] == "sk-x"
    assert captured["model"] == "gpt-4o"
    assert captured["prompt"] == "Describe for a LoRA dataset."
    assert captured["temperature"] == 0.5
    assert captured["max_tokens"] == 256
    assert captured["image_jpeg"][:2] == b"\xff\xd8"  # JPEG magic


def test_generate_requires_model_name(fake_provider):
    m = ApiCaptionModel(None, "openai")
    with pytest.raises(ValueError, match="model"):
        m.generate(Image.new("RGB", (8, 8)), {"model": "  "})


def test_encode_jpeg_resizes_long_side():
    img = Image.new("RGB", (2000, 1000), "blue")
    jpeg = _encode_jpeg(img, max_long_side=1024)
    out = Image.open(__import__("io").BytesIO(jpeg))
    assert max(out.size) == 1024
    # small images are not upscaled
    small = Image.new("RGB", (100, 50))
    out2 = Image.open(__import__("io").BytesIO(_encode_jpeg(small, 1024)))
    assert out2.size == (100, 50)


def test_caption_service_registers_api_plugins():
    from app.core.captioning.caption_service import CaptionService

    service = CaptionService()
    for provider in ("openai", "anthropic", "gemini", "openrouter", "custom"):
        assert f"api-{provider}" in service.plugins
        assert isinstance(service.plugins[f"api-{provider}"], ApiCaptionModel)


def test_generate_caption_api_model_never_touches_unload(monkeypatch):
    """API generate must bypass the shared load/unload bookkeeping (lane safety)."""
    from app.core.captioning.caption_service import CaptionService

    service = CaptionService()
    CaptionService._active_model_key = "florence-2"  # simulate GPU lane mid-run
    try:
        calls = {"unload": 0}
        monkeypatch.setattr(
            CaptionService, "unload_models",
            classmethod(
                lambda cls: calls.__setitem__("unload", calls["unload"] + 1)))
        captured = {}

        def fake_generate(self, image, params):
            captured["params"] = params
            return "cap"

        monkeypatch.setattr(
            "app.core.captioning.models.api_model.ApiCaptionModel.generate",
            fake_generate)
        monkeypatch.setattr(service, "_load_image",
                            lambda p: Image.new("RGB", (4, 4)))

        out = service.generate_caption("x.png", "api-openai", {"model": "gpt-4o"})
        assert out == "cap"
        assert captured["params"]["image_path"] == "x.png"
        assert calls["unload"] == 0
        assert CaptionService._active_model_key == "florence-2"  # untouched
    finally:
        CaptionService._active_model_key = None  # cleanup


def test_generate_caption_unknown_api_model_raises(monkeypatch):
    from app.core.captioning.caption_service import CaptionService

    service = CaptionService()
    with pytest.raises(ValueError, match="not supported"):
        service.generate_caption("x.png", "api-bogus", {"model": "gpt-4o"})
